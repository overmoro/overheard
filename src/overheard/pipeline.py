"""Orchestration: one recording in, one markdown transcript out.

Nothing here does any of the work. The order is the point:

    check signal -> split tracks -> transcribe each -> diarize each ->
    canonicalize labels -> merge and drop echo -> match voices -> name -> render

Each track is transcribed and diarized on its own, then merged. That is what
makes the local speaker known rather than inferred, and it is why the echo
suppression step exists: with the mic and the system recorded separately, a
remote voice coming back through the speakers arrives twice.
"""

import os
import sys

from overheard import config as cfg
from overheard.asr import transcribe_parakeet
from overheard.diarization import assign_speakers, canonicalize_turns, diarize
from overheard.render import write_markdown
from overheard.speakers import build_speaker_map
from overheard.tracks import check_audio_signal, drop_echo, split_tracks, warn_silent_tracks


def transcribe_audio(
    audio_path: str,
    output_path: str,
    status_callback=None,
    meeting_details=None,   # MeetingDetails | None
    mic_speaker: str | None = None,
    channels_info: dict | None = None,
) -> str:
    """Transcribe audio and write a diarized markdown file.

    Args:
        audio_path: Path to the WAV file to transcribe.
        output_path: Path for the output markdown file.
        status_callback: Optional callable for progress updates.
        meeting_details: Optional MeetingDetails for frontmatter and speaker labels.
        mic_speaker: Name for the local speaker, the one heard on the microphone
            track. Used when channels_info identifies a separate mic channel.
        channels_info: {"mic_channel": int, "system_channels": [int, ...]} from
            the recorder. When present the two sides are handled as separate
            tracks. None for mono recordings.

    Returns:
        The output_path on success.
    """
    # Pre-flight: check audio has meaningful signal before loading models
    check_audio_signal(audio_path)

    tracks = split_tracks(audio_path, channels_info)
    warn_silent_tracks(tracks)
    prefixes = {"mic": "MIC", "system": "SYS", "mixed": "SPEAKER"}
    results: dict[str, dict] = {}

    remember_speakers = cfg.get("speaker_memory")

    # The attendee list bounds how many distinct voices to expect. It's an
    # upper bound, not a count: being invited isn't the same as speaking.
    attendee_count = (
        len([a for a in meeting_details.attendees if a and a.strip()])
        if meeting_details is not None and meeting_details.attendees
        else 0
    )

    try:
        for track in tracks:
            label = track["label"]
            if status_callback and len(tracks) > 1:
                status_callback(f"Transcribing {label}...")

            segments = transcribe_parakeet(
                track["path"],
                status_callback=status_callback if len(tracks) == 1 else None,
            )

            turns = diarize(
                track["path"],
                status_callback=status_callback,
                max_speakers=attendee_count or None,
                with_embeddings=remember_speakers,
            )
            turns, order = canonicalize_turns(turns, prefix=prefixes[label])
            results[label] = {
                "segments": assign_speakers(segments, turns),
                "order": order,
                "turns": turns,
            }
    finally:
        for track in tracks:
            if track["temp"]:
                try:
                    os.unlink(track["path"])
                except OSError:
                    pass

    if status_callback:
        status_callback("Writing transcript...")

    if "mixed" in results:
        segments = results["mixed"]["segments"]
        local_labels: list[str] = []
        remote_labels = results["mixed"]["order"]
    else:
        mic = results["mic"]
        system = results["system"]
        # Discard the remote side bleeding back through the microphone before
        # merging, or every remote utterance appears twice.
        mic_segments = drop_echo(mic["segments"], system["segments"])
        segments = sorted(
            mic_segments + system["segments"], key=lambda s: s.get("start", 0.0)
        )
        surviving = {s.get("speaker") for s in mic_segments}
        local_labels = [label for label in mic["order"] if label in surviving]
        remote_labels = system["order"]

    attendees = list(meeting_details.attendees) if meeting_details is not None else []

    all_turns = [t for r in results.values() for t in r.get("turns", [])]
    label_embeddings = {}
    library = None
    known: dict[str, str] = {}
    if remember_speakers and all_turns:
        from overheard.speakers import SpeakerLibrary, mean_embeddings

        label_embeddings = mean_embeddings(all_turns)
        if label_embeddings:
            library = SpeakerLibrary()
            known = library.match_labels(
                label_embeddings,
                threshold=cfg.get("speaker_match_threshold"),
            )
            if known:
                print(f"[overheard] recognised by voice: {sorted(known.values())}",
                      file=sys.stderr)

    speaker_map = build_speaker_map(
        local_labels, remote_labels, attendees,
        mic_speaker=mic_speaker, known=known,
    )

    write_markdown(
        {"segments": segments},
        output_path,
        meeting_details=meeting_details,
        speaker_map=speaker_map,
    )

    _learn_confident_voices(
        library, label_embeddings, speaker_map, known,
        local_labels, remote_labels, attendees, mic_speaker,
    )

    return output_path


def _learn_confident_voices(
    library, label_embeddings, speaker_map, known,
    local_labels, remote_labels, attendees, mic_speaker,
) -> None:
    """Store voice embeddings, but only from sources that are actually knowledge.

    Names assigned by attendee order are a convention, not a fact. Storing one
    would launder a guess into evidence: a library match outranks every other
    signal, so a single wrong ordering would attach the wrong name to a voice
    and then reapply it confidently to every future meeting, with nothing to
    correct it. Only the local speaker, known from the mic track, and voices
    the library already recognised are safe to learn.
    """
    if library is None or not label_embeddings:
        return

    confident = set(known)
    if local_labels and mic_speaker:
        confident.add(local_labels[0])

    # A remote voice is safe to learn when the assignment could not have been
    # anything else: one unidentified remote speaker, one unclaimed attendee.
    # That covers the common two-person call. With two of each, the pairing came
    # from speech order and is exactly the guess this must not store.
    unknown_remote = [
        label for label in remote_labels
        if label not in known and speaker_map.get(label)
    ]
    used = {speaker_map[label].strip().casefold() for label in confident
            if speaker_map.get(label)}
    unclaimed = [a for a in attendees
                 if a.strip() and a.strip().casefold() not in used]
    if len(unknown_remote) == 1 and len(unclaimed) == 1:
        confident.add(unknown_remote[0])

    learned = False
    for label in confident:
        name = speaker_map.get(label)
        embedding = label_embeddings.get(label)
        if name and embedding:
            library.remember(name, embedding)
            learned = True
    if learned:
        library.save()
