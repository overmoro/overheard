"""Who spoke when, and attaching that to the words.

Two halves. First diarization proper, which returns ``{start, end, speaker}``
turns from FluidAudio. Then the join back onto the transcript, which relabels
the diarizer's arbitrary cluster numbering into first-speech order and splits
transcript segments on speaker boundaries.

A failed diarization is not fatal anywhere in here. A transcript with no speaker
labels is still worth having; a transcript that never got written is not.
"""

import sys


def diarize(
    audio_path: str, status_callback=None, max_speakers: int | None = None,
    with_embeddings: bool = False,
) -> list[dict]:
    """Run speaker diarization, returning {start, end, speaker} turns.

    Diarization runs through FluidAudio on the Neural Engine: no Hugging Face
    account, no gated model, and measured 104x realtime.

    A pyannote fallback lived here until the single-engine cut. It could not
    fire for anyone running the bundled app, because it needed HF_TOKEN, a
    gated model accepted on the Hub, and torch installed as an optional extra.

    Returns an empty list when diarization is unavailable. That is deliberately
    not fatal: a transcript without speaker labels is still worth having.
    """
    turns = _diarize_fluidaudio(
        audio_path, status_callback=status_callback, max_speakers=max_speakers,
        with_embeddings=with_embeddings,
    )
    return turns if turns is not None else []


#: FluidAudio's speaker embeddings are fixed at this width, which is why the
#: --embeddings flag is described as 256 floats per segment where it is passed.
#: Taking the width from the first embedding that happened to arrive instead
#: made a malformed leading segment define what "correct" meant for the whole
#: run, and everything well-formed behind it was then discarded as the outlier.
_EMBEDDING_WIDTH = 256


def _usable_embedding(embedding) -> bool:
    """True for an embedding mean_embeddings can actually consume.

    Both halves matter and only the first was checked once already. Element
    type keeps a string out of np.asarray; width keeps two different shapes for
    one speaker out of the accumulate, which is where numpy raises "operands
    could not be broadcast together" rather than at the parse.
    """
    if not isinstance(embedding, (list, tuple)):
        return False
    if len(embedding) != _EMBEDDING_WIDTH:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in embedding
    )


def _diarize_fluidaudio(
    audio_path: str, status_callback=None, max_speakers: int | None = None,
    with_embeddings: bool = False,
) -> list[dict] | None:
    """Diarize through overheard-helper. Returns None if it could not run.

    None and [] mean different things here: None is "this backend is
    unavailable, try another", [] is "it ran and found no speech".
    """
    import json
    import subprocess

    from overheard.helper import helper_path

    binary = helper_path()
    if binary is None:
        return None

    if status_callback:
        status_callback("Diarizing...")

    argv = [str(binary), "diarize", audio_path]
    if max_speakers and max_speakers > 0:
        # An upper bound rather than an exact count: the attendee list says who
        # was invited, not how many of them actually spoke.
        argv += ["--max-speakers", str(int(max_speakers))]
    if with_embeddings:
        # 256 floats per segment, so only requested when identity matching is on
        argv.append("--embeddings")

    try:
        result = subprocess.run(argv, capture_output=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[overheard] helper diarize failed: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:200]
        print(f"[overheard] helper diarize exited {result.returncode}: {detail}",
              file=sys.stderr)
        return None

    try:
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[overheard] could not parse diarize output: {e}", file=sys.stderr)
        return None

    # Guarding the segment loop and not the payload above it was fixing the
    # instance and leaving the class. `{"segments": null}` is what a Swift
    # Encodable with an optional array produces, and .get's default only fires
    # when the key is absent, not when it is null; a bare `[]`, `null` or a
    # string payload has no .get at all. Each of those raised past diarize()
    # and out of transcribe_audio, whose caller then deletes the temp WAV, so a
    # fully transcribed meeting was destroyed by output the diarizer produced.
    if not isinstance(payload, dict):
        print(f"[overheard] diarize output was {type(payload).__name__}, not an object",
              file=sys.stderr)
        return None

    segments = payload.get("segments")
    if segments is None:
        segments = []
    if not isinstance(segments, (list, tuple)):
        print(f"[overheard] diarize segments were {type(segments).__name__}, not a list",
              file=sys.stderr)
        return None

    turns = []
    skipped = 0
    dropped_embeddings = 0
    for segment in segments:
        if not isinstance(segment, dict):
            skipped += 1
            continue
        # Well-formed JSON is not well-formed output. A missing key, a null
        # timestamp or a non-numeric one all raise here, and this loop sits
        # after transcription has already run: an exception escaping it takes
        # the whole meeting down at the last step, since the caller deletes the
        # temp WAV once transcribe_audio fails. One bad segment costs that
        # segment.
        #
        # speaker and embedding are validated for the same reason the
        # timestamps are, and they were left out of the first version of this
        # guard. Neither raises here, so both travel intact to a caller that has
        # no guard at all: a non-hashable speaker raises TypeError at
        # order[turn["speaker"]] in canonicalize_turns, and a malformed
        # embedding raises ValueError at np.asarray in speakers.mean_embeddings,
        # which pipeline.transcribe_audio calls after transcription and outside
        # every try. speaker_memory defaults to True, so that path is live.
        try:
            if segment.get("end", 0) <= segment.get("start", 0):
                continue
            speaker = segment["speaker"]
            if not isinstance(speaker, str):
                raise TypeError("speaker was not a string")
            turn = {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": speaker,
            }
            # A bad embedding costs the embedding, never the turn. Raising in
            # here instead discarded start, end and speaker along with it, and
            # with the width taken from whichever segment happened to arrive
            # first, one malformed leading segment threw away every well-formed
            # one behind it: assign_speakers then took its `if not turns` branch
            # and labelled the entire meeting SPEAKER_00. The surviving vector
            # was the malformed one, and _learn_confident_voices wrote it into
            # the permanent library, where remember() replaces a stored profile
            # outright on a shape mismatch. Twelve meetings of accumulated voice
            # data, gone, from one bad segment.
            embedding = segment.get("embedding") if with_embeddings else None
            if embedding and not _usable_embedding(embedding):
                dropped_embeddings += 1
                embedding = None
            if embedding:
                turn["embedding"] = list(embedding)
        except (KeyError, TypeError, ValueError, AttributeError):
            skipped += 1
            continue
        turns.append(turn)

    if skipped:
        print(f"[overheard] skipped {skipped} malformed diarizer segments",
              file=sys.stderr)
    if dropped_embeddings:
        # Reported separately because it costs something different: the turn is
        # intact and the meeting is still attributed, only cross-meeting voice
        # matching loses that sample. Folding it into the count above would
        # report a degradation as a data loss.
        print(f"[overheard] dropped {dropped_embeddings} unusable embeddings, "
              f"expected {_EMBEDDING_WIDTH} floats each",
              file=sys.stderr)
    return turns


def canonicalize_turns(
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


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
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
    current: str = turns[0]["speaker"]

    for seg in segments:
        words = seg.get("words") or []
        if not words:
            # No word timings to split on, so assign the segment as a whole.
            current = speaker_at(seg["start"], seg["end"]) or current
            regrouped.append({**seg, "speaker": current})
            continue

        run: list[dict] = []
        run_speaker: str = current
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
