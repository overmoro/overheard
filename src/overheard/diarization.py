"""Who spoke when, and attaching that to the words.

Two halves. First diarization proper, which returns ``{start, end, speaker}``
turns from one of two backends. Then the join back onto the transcript, which
relabels the diarizer's arbitrary cluster numbering into first-speech order and
splits transcript segments on speaker boundaries.

A failed diarization is not fatal anywhere in here. A transcript with no speaker
labels is still worth having; a transcript that never got written is not.
"""

import os
import sys

from overheard import config as cfg


def diarize(
    audio_path: str, status_callback=None, max_speakers: int | None = None,
    with_embeddings: bool = False,
) -> list[dict]:
    """Run speaker diarization, returning {start, end, speaker} turns.

    FluidAudio is the default: it runs on the Neural Engine, needs no Hugging
    Face account or gated model, and measured 104x realtime against pyannote's
    roughly 1x on the same audio. pyannote remains as a fallback, selectable
    with the ``diarizer`` config key, but its dependencies are now optional.

    Returns an empty list when diarization is unavailable. That is deliberately
    not fatal: a transcript without speaker labels is still worth having.
    """
    backend = (cfg.get("diarizer", "auto") or "auto").lower()

    if backend in ("auto", "fluidaudio"):
        turns = _diarize_fluidaudio(
            audio_path, status_callback=status_callback, max_speakers=max_speakers,
            with_embeddings=with_embeddings,
        )
        if turns is not None:
            return turns
        if backend == "fluidaudio":
            return []
        print("[overheard] FluidAudio diarization unavailable, trying pyannote",
              file=sys.stderr)

    return _diarize_pyannote(audio_path, status_callback=status_callback)


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

    turns = []
    for segment in payload.get("segments", []):
        if segment.get("end", 0) <= segment.get("start", 0):
            continue
        turn = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "speaker": segment["speaker"],
        }
        if with_embeddings and segment.get("embedding"):
            turn["embedding"] = segment["embedding"]
        turns.append(turn)
    return turns


def _diarize_pyannote(audio_path: str, status_callback=None) -> list[dict]:
    """Fallback diarization through pyannote.

    Kept for comparison and for machines where the helper cannot run. Needs
    HF_TOKEN in the environment and pulls in torch, so it is an optional extra
    rather than a default dependency.
    """
    if status_callback:
        status_callback("Diarizing...")

    hf_token = os.environ.get("HF_TOKEN")

    try:
        import torch
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
