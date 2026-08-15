"""Speech to text, through NVIDIA Parakeet TDT 0.6B v3 via MLX.

Runs on the Apple Silicon GPU and emits word-level timestamps natively, so no
separate forced-alignment pass is needed. It covers 25 European languages
including English and detects the language itself, without configuration.

Returns a list of ``{start, end, text, words: [{start, end, word}]}``.

A second WhisperX engine lived here until the single-engine cut. It ran on CPU
because ctranslate2 has no MPS backend, needed a second model download for the
alignment pass Parakeet does natively, and disabled the live transcript. Its
only unique reach was the non-European languages, which no setting exposed.
"""

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

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
