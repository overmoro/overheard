"""Transcription with pyannote diarization.

Two engines are available, selected by the ``engine`` config key:

- ``parakeet`` (default): NVIDIA Parakeet TDT 0.6B v3 via MLX, running on the
  Apple Silicon GPU. Emits word-level timestamps natively, so no separate
  forced-alignment pass is needed.
- ``whisper``: the original WhisperX large-v3 path, kept for comparison. It
  runs on CPU because ctranslate2 has no MPS backend.

Both engines produce the same intermediate shape (a list of segment dicts with
word-level timing), which is then diarized and rendered to markdown.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

# torchcodec is incompatible with PyTorch 2.8, so suppress the wall of warnings at import
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

from overheard import config as cfg

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL = "large-v3"

# Parakeet handles long audio by chunking with overlap. Full attention over an
# hour-long meeting would exhaust memory, so chunk unless told otherwise.
CHUNK_DURATION = 300.0
OVERLAP_DURATION = 15.0

# Read output directory from config; fall back to ~/meeting-transcripts
OUTPUT_DIR = Path(cfg.get("output_dir", str(Path.home() / "meeting-transcripts")))


def _build_speaker_map(
    local_labels: list[str],
    remote_labels: list[str],
    attendees: list[str],
    mic_speaker: str | None = None,
) -> dict[str, str]:
    """Map diarized labels to real names, using which track each came from.

    Labels arrive already split by source: ``local_labels`` were heard on the
    microphone and ``remote_labels`` through the system audio. That split is
    knowledge rather than inference, so the first mic speaker is the person at
    this machine and gets ``mic_speaker`` outright.

    Everyone else is filled from the attendee list in first-speech order, which
    is a convention rather than a fact, but a deterministic and explainable one.
    Additional mic speakers occur when several people share the laptop in an
    in-person meeting.
    """
    mapping: dict[str, str] = {}
    remaining = [a for a in attendees if a]

    if local_labels and mic_speaker:
        mapping[local_labels[0]] = mic_speaker
        # Don't hand the local speaker's name out twice if they're also listed
        remaining = [
            a for a in remaining
            if a.strip().casefold() != mic_speaker.strip().casefold()
        ]

    # Remote speakers first: they are the ones the attendee list describes.
    for label in remote_labels + local_labels:
        if label in mapping:
            continue
        if not remaining:
            break
        mapping[label] = remaining.pop(0)

    return mapping


def _fallback_speaker_names(segments: list[dict], named: dict[str, str]) -> dict[str, str]:
    """Name any unmapped speakers 'Speaker 1', 'Speaker 2', ... by first appearance.

    Numbering is assigned across the merged transcript rather than per track, so
    the reader sees one consistent sequence regardless of which side each voice
    arrived on.
    """
    display: dict[str, str] = {}
    for segment in segments:
        label = segment.get("speaker")
        if not label or label in named or label in display:
            continue
        display[label] = f"Speaker {len(display) + 1}"
    return display


def _check_audio_signal(audio_path: str) -> None:
    """Raise if the recording is effectively silent.

    Catches the common misconfiguration where system output isn't routed
    through BlackHole, before we spend time loading models.
    """
    import numpy as np
    import soundfile as sf

    audio_check, _ = sf.read(audio_path)
    rms = float(np.sqrt(np.mean(audio_check ** 2)))
    if rms < 0.0001:
        raise RuntimeError(
            f"Audio appears silent (RMS={rms:.6f}). "
            "Set your Mac system output to 'Meeting Monitor' so audio routes through BlackHole."
        )


def _split_tracks(audio_path: str, channels_info: dict | None) -> list[dict]:
    """Split a recording into independent mono tracks.

    When the capture knows which channels are the microphone and which are the
    system, the two are written out separately rather than mixed. Keeping them
    apart matters for three reasons:

    - The local speaker is known by construction. There is no need to infer who
      the user is from relative channel energy.
    - Each side is diarized on its own, so the two halves of a conversation
      never have to be told apart by clustering.
    - Mixing a mic that is picking up the speakers against the system audio it
      is echoing produces a doubled, phase-shifted signal that measurably
      degrades transcription.

    Returns a list of {label, path, temp} dicts: either one "mixed" track or a
    "mic" and a "system" track.
    """
    import numpy as np
    import soundfile as sf
    import tempfile

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    n_ch = data.shape[1]

    mic_ch = (channels_info or {}).get("mic_channel")
    sys_chs = [c for c in ((channels_info or {}).get("system_channels") or [])
               if 0 <= c < n_ch]

    if mic_ch is None or not (0 <= mic_ch < n_ch) or not sys_chs:
        # Unknown layout: fall back to a single folded track.
        path, temp = _prepare_mono_audio(audio_path, channels_info)
        return [{"label": "mixed", "path": path, "temp": temp}]

    def _write(samples) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 1.0:
            samples = samples / peak
        sf.write(tmp.name, samples.astype(np.float32), int(sr))
        return tmp.name

    return [
        {"label": "mic", "path": _write(data[:, mic_ch]), "temp": True},
        {"label": "system", "path": _write(data[:, sys_chs].mean(axis=1)), "temp": True},
    ]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds of overlap between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _drop_echo(mic_segments: list[dict], system_segments: list[dict]) -> list[dict]:
    """Remove mic segments that are just the speakers bleeding into the mic.

    When the user listens on speakers rather than headphones, the microphone
    re-records the remote side, so the same words arrive on both tracks and
    would be transcribed twice.

    A mic segment is treated as echo when it overlaps system speech for most of
    its length *and* says substantially the same thing. Requiring both keeps
    genuine interruptions, where the user talks over the remote side and the
    words differ.
    """
    from difflib import SequenceMatcher

    if not system_segments:
        return mic_segments

    kept = []
    for segment in mic_segments:
        duration = max(1e-6, segment["end"] - segment["start"])
        covered = 0.0
        best_similarity = 0.0

        for other in system_segments:
            overlap = _overlap(segment["start"], segment["end"], other["start"], other["end"])
            if overlap <= 0:
                continue
            covered += overlap
            similarity = SequenceMatcher(
                None, segment["text"].lower(), other["text"].lower()
            ).ratio()
            best_similarity = max(best_similarity, similarity)

        if covered / duration > 0.5 and best_similarity > 0.6:
            continue
        kept.append(segment)

    return kept


def _prepare_mono_audio(audio_path: str, channels_info: dict | None) -> tuple[str, bool]:
    """Downmix a multichannel recording to mono, returning (path, is_temporary).

    Both engines ultimately load audio through ffmpeg's ``-ac 1``, and ffmpeg
    reads a 3-channel WAV as a 2.1 layout: channel 2 is treated as LFE and
    dropped entirely from the mono downmix. On the Meeting Capture aggregate
    channel 2 is the microphone, so the local speaker's voice was being
    discarded before transcription ever saw it.

    Doing the downmix here avoids that. When the channel layout is known, the
    microphone is mixed at equal weight against the combined system audio, so
    neither side of the conversation dominates. Otherwise all channels are
    averaged evenly.

    The result is written at the source sample rate; ffmpeg resamples mono
    audio correctly, it is only the channel folding that is wrong.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        return audio_path, False

    n_ch = data.shape[1]
    mic_ch = (channels_info or {}).get("mic_channel")
    sys_chs = [c for c in ((channels_info or {}).get("system_channels") or [])
               if 0 <= c < n_ch]

    if mic_ch is not None and 0 <= mic_ch < n_ch and sys_chs:
        mono = 0.5 * data[:, sys_chs].mean(axis=1) + 0.5 * data[:, mic_ch]
    else:
        mono = data.mean(axis=1)

    # Mixing can push peaks past full scale; scale back rather than clip
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 1.0:
        mono = mono / peak

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, mono.astype(np.float32), int(sr))
    return tmp.name, True


# ----------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------


def _transcribe_parakeet(audio_path: str, status_callback=None) -> list[dict]:
    """Transcribe with Parakeet TDT via MLX.

    Returns segments shaped as {start, end, text, words: [{start, end, word}]}.
    Parakeet emits subword tokens, so words are reconstructed by splitting on
    the leading space that marks a word boundary in its tokenizer.
    """
    from parakeet_mlx import from_pretrained

    if status_callback:
        status_callback("Loading model...")

    model = from_pretrained(PARAKEET_MODEL)

    def on_chunk(current, total):
        if status_callback and total:
            status_callback(f"Transcribing... {int(100 * current / total)}%")

    if status_callback:
        status_callback("Transcribing...")

    result = model.transcribe(
        audio_path,
        chunk_duration=CHUNK_DURATION,
        overlap_duration=OVERLAP_DURATION,
        chunk_callback=on_chunk,
    )

    segments = []
    for sentence in result.sentences:
        words = []
        for token in sentence.tokens:
            # A leading space marks the start of a new word; anything else is a
            # continuation of the word already in progress.
            if token.text.startswith(" ") or not words:
                words.append({
                    "start": token.start,
                    "end": token.end,
                    "word": token.text.strip(),
                })
            else:
                words[-1]["word"] += token.text
                words[-1]["end"] = token.end

        words = [w for w in words if w["word"]]
        text = sentence.text.strip()
        if not text:
            continue

        segments.append({
            "start": sentence.start,
            "end": sentence.end,
            "text": text,
            "words": words,
        })

    return segments


def _transcribe_whisper(
    audio_path: str,
    model_size: str = WHISPER_MODEL,
    language: str = "en",
    status_callback=None,
) -> list[dict]:
    """Transcribe with WhisperX, including the forced-alignment pass.

    ctranslate2 has no MPS backend, so this runs on CPU at int8.
    """
    import whisperx

    compute_device = "cpu"
    compute_type = "int8"  # int8 is fastest on CPU; float16 for GPU

    if status_callback:
        status_callback("Transcribing...")

    model = whisperx.load_model(
        model_size,
        compute_device,
        compute_type=compute_type,
        language=language,
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8)

    if status_callback:
        status_callback("Aligning...")

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=compute_device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, compute_device
    )

    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = [
            {"start": w["start"], "end": w["end"], "word": w.get("word", "")}
            for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": text,
            "words": words,
        })

    return segments


# ----------------------------------------------------------------------------
# Diarization
# ----------------------------------------------------------------------------


def _diarize(audio_path: str, status_callback=None) -> list[dict]:
    """Run pyannote speaker diarization.

    Returns a list of {start, end, speaker} turns, or an empty list if
    diarization is unavailable. A missing HF_TOKEN is not fatal: the transcript
    is still worth having without speaker labels.
    """
    import torch

    if status_callback:
        status_callback("Diarizing...")

    hf_token = os.environ.get("HF_TOKEN")

    try:
        from pyannote.audio import Pipeline
        import torchaudio

        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        )
        if diarize_pipeline is None:
            raise RuntimeError(
                "pyannote returned no pipeline; the model is gated and needs a "
                "valid HF_TOKEN on first download."
            )

        # pyannote's speaker embedding model runs happily on MPS
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        diarize_pipeline.to(torch.device(device))

        # Pre-load audio as a tensor to bypass torchcodec (broken on PyTorch 2.8+)
        waveform, sample_rate = torchaudio.load(audio_path)
        output = diarize_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    except Exception as e:
        print(f"[overheard] diarization unavailable, continuing without speakers: {e}")
        return []

    # DiarizeOutput is a named tuple, so extract the Annotation
    annotation = getattr(output, "speaker_diarization", output)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _canonicalize_turns(
    turns: list[dict], prefix: str = "SPEAKER"
) -> tuple[list[dict], list[str]]:
    """Relabel diarized speakers in order of first speech.

    The diarizer's own numbering reflects clustering order, which is arbitrary.
    Renumbering by first appearance makes the first label always the first voice
    heard, so downstream naming is reproducible across runs.

    ``prefix`` namespaces the labels per track, so the mic track's speakers can
    never collide with the system track's.

    Returns (relabelled turns, labels in first-speech order).
    """
    order: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t["start"]):
        if turn["speaker"] not in order:
            order[turn["speaker"]] = f"{prefix}_{len(order):02d}"

    relabelled = [{**t, "speaker": order[t["speaker"]]} for t in turns]
    return relabelled, list(order.values())


def _assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Attach speakers to words, then regroup into speaker-contiguous segments.

    A sentence can span a speaker change, so assignment happens per word and
    segments are rebuilt on the boundaries rather than inherited wholesale.
    Words that overlap no turn inherit the running speaker.
    """
    if not turns:
        return [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    def speaker_at(start: float, end: float) -> str | None:
        best, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap > best_overlap:
                best_overlap, best = overlap, turn["speaker"]
        return best

    regrouped: list[dict] = []
    current = turns[0]["speaker"]

    for seg in segments:
        words = seg.get("words") or []
        if not words:
            # No word timings to split on, so assign the segment as a whole.
            current = speaker_at(seg["start"], seg["end"]) or current
            regrouped.append({**seg, "speaker": current})
            continue

        run: list[dict] = []
        run_speaker = None
        for word in words:
            speaker = speaker_at(word["start"], word["end"]) or current
            if run and speaker != run_speaker:
                regrouped.append(_segment_from_words(run, run_speaker))
                run = []
            run_speaker = speaker
            current = speaker
            run.append(word)

        if run:
            regrouped.append(_segment_from_words(run, run_speaker))

    return regrouped


def _segment_from_words(words: list[dict], speaker: str) -> dict:
    """Build a segment dict from a run of words sharing one speaker."""
    text = " ".join(w["word"] for w in words if w["word"]).strip()
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
        "speaker": speaker,
    }


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def transcribe_audio(
    audio_path: str,
    output_path: str,
    model_size: str | None = None,
    language: str = "en",
    status_callback=None,
    meeting_details=None,   # MeetingDetails | None
    mic_speaker: str | None = None,
    engine: str | None = None,
    channels_info: dict | None = None,
) -> str:
    """Transcribe audio and write a diarized markdown file.

    Args:
        audio_path: Path to the WAV file to transcribe.
        output_path: Path for the output markdown file.
        model_size: Whisper model size; ignored by the parakeet engine.
        language: Language code, used by the whisper engine only.
        status_callback: Optional callable for progress updates.
        meeting_details: Optional MeetingDetails for frontmatter and speaker labels.
        mic_speaker: Name for the local speaker, the one heard on the microphone
            track. Used when channels_info identifies a separate mic channel.
        engine: "parakeet" or "whisper". Defaults to the ``engine`` config key.
        channels_info: {"mic_channel": int, "system_channels": [int, ...]} from
            the recorder. When present the two sides are handled as separate
            tracks. None for mono recordings.

    Returns:
        The output_path on success.
    """
    engine = (engine or cfg.get("engine", "parakeet")).lower()
    if engine not in ("parakeet", "whisper"):
        raise ValueError(f"Unknown transcription engine: {engine!r}")

    # Pre-flight: check audio has meaningful signal before loading models
    _check_audio_signal(audio_path)

    tracks = _split_tracks(audio_path, channels_info)
    prefixes = {"mic": "MIC", "system": "SYS", "mixed": "SPEAKER"}
    results: dict[str, dict] = {}

    try:
        for track in tracks:
            label = track["label"]
            if status_callback and len(tracks) > 1:
                status_callback(f"Transcribing {label}...")

            if engine == "whisper":
                segments = _transcribe_whisper(
                    track["path"],
                    model_size=model_size or WHISPER_MODEL,
                    language=language,
                    status_callback=status_callback if len(tracks) == 1 else None,
                )
            else:
                segments = _transcribe_parakeet(
                    track["path"],
                    status_callback=status_callback if len(tracks) == 1 else None,
                )

            turns = _diarize(track["path"], status_callback=status_callback)
            turns, order = _canonicalize_turns(turns, prefix=prefixes[label])
            results[label] = {
                "segments": _assign_speakers(segments, turns),
                "order": order,
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
        mic_segments = _drop_echo(mic["segments"], system["segments"])
        segments = sorted(
            mic_segments + system["segments"], key=lambda s: s.get("start", 0.0)
        )
        surviving = {s.get("speaker") for s in mic_segments}
        local_labels = [label for label in mic["order"] if label in surviving]
        remote_labels = system["order"]

    attendees = list(meeting_details.attendees) if meeting_details is not None else []
    speaker_map = _build_speaker_map(
        local_labels, remote_labels, attendees, mic_speaker=mic_speaker
    )

    _write_markdown(
        {"segments": segments},
        output_path,
        meeting_details=meeting_details,
        speaker_map=speaker_map,
    )
    return output_path


def _write_markdown(
    result: dict,
    output_path: str,
    meeting_details=None,   # MeetingDetails | None
    speaker_map: dict[str, str] | None = None,
) -> None:
    """Format a diarized result as a markdown transcript."""
    now = datetime.now()
    title = now.strftime("%-d %B %Y, %-I:%M%p").replace("AM", "am").replace("PM", "pm")

    lines = []

    # ---- YAML frontmatter --------------------------------------------------
    if meeting_details is not None:
        created = now.strftime("%Y-%m-%d")
        meeting_title = f"[[{meeting_details.name}]]" if meeting_details.name else "[[Meeting]]"
        attendee_lines = ""
        if meeting_details.attendees:
            formatted = [f'  - "[[{name}]]"' for name in meeting_details.attendees if name]
            if formatted:
                attendee_lines = "\n" + "\n".join(formatted)
        else:
            attendee_lines = ""

        location_val = meeting_details.location or ""
        source_val = meeting_details.source or "in-person"

        lines.append("---")
        lines.append("type: transcript")
        lines.append(f"created: {created}")
        lines.append(f"source: {source_val}")
        lines.append(f"location: {location_val}")
        lines.append(f'meeting: "{meeting_title}"')
        lines.append(f"attendees:{attendee_lines if attendee_lines else ' []'}")
        lines.append("status: inbox")
        lines.append("topics: []")
        lines.append("---")
        lines.append("")

    # ---- Heading -----------------------------------------------------------
    lines.append(f"# Meeting \u2014 {title}\n")

    # ---- Transcript body ---------------------------------------------------
    speaker_map = speaker_map or {}
    segments = result.get("segments", [])
    fallback_names = _fallback_speaker_names(segments, speaker_map)
    current_speaker = None
    current_ts = ""
    current_texts: list[str] = []

    def _flush_speaker():
        if current_speaker and current_texts:
            paragraph = " ".join(current_texts)
            lines.append(f"\n**{current_speaker}** {current_ts}\n{paragraph}")

    for seg in segments:
        raw_speaker = seg.get("speaker") or "SPEAKER_00"
        if raw_speaker in speaker_map:
            display_speaker = f"[[{speaker_map[raw_speaker]}]]"
        else:
            display_speaker = fallback_names.get(raw_speaker, raw_speaker)

        text = seg.get("text", "").strip()
        if not text:
            continue

        if display_speaker != current_speaker:
            _flush_speaker()
            start = seg.get("start", 0)
            current_ts = f"[{int(start)//3600:02d}:{(int(start)%3600)//60:02d}:{int(start)%60:02d}]"
            current_speaker = display_speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    _flush_speaker()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
