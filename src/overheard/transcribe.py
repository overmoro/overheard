"""WhisperX transcription with pyannote diarization."""

import os
import warnings
from datetime import datetime
from pathlib import Path

# torchcodec is incompatible with PyTorch 2.8 — suppress the wall of warnings at import
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

from overheard import config as cfg

WHISPER_MODEL = "large-v3"

# Read output directory from config; fall back to ~/meeting-transcripts
OUTPUT_DIR = Path(cfg.get("output_dir", str(Path.home() / "meeting-transcripts")))


def transcribe_audio(
    audio_path: str,
    output_path: str,
    model_size: str = WHISPER_MODEL,
    language: str = "en",
    status_callback=None,
) -> str:
    """Transcribe audio with WhisperX and write a diarized markdown file.

    Args:
        audio_path: Path to the WAV file to transcribe.
        output_path: Path for the output markdown file.
        model_size: Whisper model size (default: large-v3).
        language: Language code for transcription.
        status_callback: Optional callable for progress updates.

    Returns:
        The output_path on success.
    """
    import whisperx
    import torch
    import numpy as np

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    # Pre-flight: check audio has meaningful signal before loading models
    import soundfile as sf
    audio_check, _ = sf.read(audio_path)
    rms = float(np.sqrt(np.mean(audio_check ** 2)))
    if rms < 0.001:
        raise RuntimeError(
            "Audio appears silent — check your recording device in Preferences. "
            "The aggregate device (Meeting Capture) must be selected."
        )

    # ctranslate2 (WhisperX backend) doesn't support MPS yet — use CPU
    # pyannote's speaker embedding model will use MPS if available
    compute_device = "cpu"
    compute_type = "int8"  # int8 is fastest on CPU; float16 for GPU

    # MPS available for diarization pipeline
    diarize_device = "mps" if torch.backends.mps.is_available() else "cpu"

    if status_callback:
        status_callback("Transcribing...")

    model = whisperx.load_model(
        model_size,
        compute_device,
        compute_type=compute_type,
        language=language,
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        batch_size=8,
        vad_parameters={"threshold": 0.3, "min_speech_duration_ms": 200},
    )

    if status_callback:
        status_callback("Aligning...")

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=compute_device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, compute_device
    )

    if status_callback:
        status_callback("Diarizing...")

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token, device=diarize_device
    )
    # Pre-load audio as tensor to bypass torchcodec (broken on PyTorch 2.8+)
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    diarize_segments = diarize_model({"waveform": waveform, "sample_rate": sample_rate})
    result = whisperx.assign_word_speakers(diarize_segments, result)

    if status_callback:
        status_callback("Writing transcript...")

    _write_markdown(result, output_path)
    return output_path


def _write_markdown(result: dict, output_path: str) -> None:
    """Format WhisperX result as a diarized markdown transcript."""
    now = datetime.now()
    title = now.strftime("%-d %B %Y, %-I:%M%p").replace("AM", "am").replace("PM", "pm")
    lines = [f"# Meeting \u2014 {title}\n"]

    current_speaker = None
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "Unknown")
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if not text:
            continue

        ts = f"[{int(start)//3600:02d}:{(int(start)%3600)//60:02d}:{int(start)%60:02d}]"

        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n**{speaker}:** {ts} {text}")
        else:
            lines.append(f"{ts} {text}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
