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

# Parakeet handles long audio by chunking with overlap. Full attention over an
# hour-long meeting would exhaust memory, so chunk unless told otherwise.
CHUNK_DURATION = 300.0
OVERLAP_DURATION = 15.0


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
