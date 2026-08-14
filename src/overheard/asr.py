"""Speech to text, in two interchangeable engines.

Selected by the ``engine`` config key:

- ``parakeet`` (default): NVIDIA Parakeet TDT 0.6B v3 via MLX, running on the
  Apple Silicon GPU. Emits word-level timestamps natively, so no separate
  forced-alignment pass is needed.
- ``whisper``: the original WhisperX large-v3 path, kept for comparison. It
  runs on CPU because ctranslate2 has no MPS backend.

Both return the same shape, a list of
``{start, end, text, words: [{start, end, word}]}``, which is what lets the rest
of the pipeline stay engine-agnostic.
"""

import warnings

# torchcodec is incompatible with PyTorch 2.8, so suppress the wall of warnings
# the whisper path emits on import.
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL = "large-v3"

# WhisperX loads through faster-whisper, which caches under the CTranslate2
# conversion rather than under the bare model size above.
WHISPER_REPO = "Systran/faster-whisper-large-v3"

# Parakeet handles long audio by chunking with overlap. Full attention over an
# hour-long meeting would exhaust memory, so chunk unless told otherwise.
CHUNK_DURATION = 300.0
OVERLAP_DURATION = 15.0


def is_model_cached(repo_id: str = PARAKEET_MODEL) -> bool:
    """True when the Hugging Face cache already holds this model.

    Preferences used to display a fixed "Downloads on first use", which said
    the same thing whether or not the model was there. Reporting a state
    nothing has inspected is worse than reporting none, because the user has no
    way to tell it apart from a real answer.

    The cache layout is Hugging Face's own: ``<cache>/hub/models--org--name``
    with the real files under ``snapshots/<revision>/``.

    Presence of the weights is what is checked, not presence of any file.
    huggingface_hub materialises completed files smallest first, so a download
    cancelled after a few seconds leaves a perfectly good config.json and no
    weights at all. Counting that as installed would put the lie one layer
    below the label this replaced, and the user would only find out when a
    meeting failed to transcribe.
    """
    import os
    from pathlib import Path

    hf_home = os.environ.get("HF_HOME")
    hub = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    folder = hub / ("models--" + repo_id.replace("/", "--"))
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False

    def has_weights(revision: Path) -> bool:
        for pattern in ("*.safetensors", "*.bin", "*.npz"):
            for candidate in revision.glob(pattern):
                # Cache entries are symlinks into blobs/; a dangling one means
                # the blob was never finished or has been garbage collected.
                if candidate.exists() and candidate.stat().st_size > 0:
                    return True
        return False

    return any(has_weights(rev) for rev in snapshots.iterdir() if rev.is_dir())


def transcribe_parakeet(audio_path: str, status_callback=None) -> list[dict]:
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
        words: list[dict] = []
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


def transcribe_whisper(
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
