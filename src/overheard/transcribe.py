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


def _build_speaker_map(attendees: list[str]) -> dict[str, str]:
    """Build {SPEAKER_00: "Name", ...} from an ordered attendee list.

    Keys use whisperx's internal zero-padded format (SPEAKER_00, SPEAKER_01).
    Display conversion to 'Speaker 1', 'Speaker 2' happens at render time.
    """
    return {f"SPEAKER_{i:02d}": name for i, name in enumerate(attendees) if name}


def _format_speaker(raw: str) -> str:
    """Convert whisperx label to a readable name.

    SPEAKER_00 → Speaker 1, SPEAKER_01 → Speaker 2, etc.
    Leaves already-named speakers untouched.
    """
    import re
    m = re.match(r"SPEAKER_(\d+)$", raw)
    if m:
        return f"Speaker {int(m.group(1)) + 1}"
    return raw


def transcribe_audio(
    audio_path: str,
    output_path: str,
    model_size: str = WHISPER_MODEL,
    language: str = "en",
    status_callback=None,
    meeting_details=None,   # MeetingDetails | None
    mic_speaker: str | None = None,
) -> str:
    """Transcribe audio with WhisperX and write a diarized markdown file.

    Args:
        audio_path: Path to the WAV file to transcribe.
        output_path: Path for the output markdown file.
        model_size: Whisper model size (default: large-v3).
        language: Language code for transcription.
        status_callback: Optional callable for progress updates.
        meeting_details: Optional MeetingDetails for frontmatter and speaker labels.
        mic_speaker: Optional name for the mic channel speaker (plumbed through;
            channel-separation logic applied in audio pipeline).

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
    if rms < 0.0001:
        raise RuntimeError(
            f"Audio appears silent (RMS={rms:.6f}). "
            "Set your Mac system output to 'Meeting Monitor' so audio routes through BlackHole."
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
    result = model.transcribe(audio, batch_size=8)

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

    from pyannote.audio import Pipeline
    import torchaudio
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=hf_token
    )
    diarize_pipeline.to(torch.device(diarize_device))
    # Pre-load audio as tensor to bypass torchcodec (broken on PyTorch 2.8+)
    waveform, sample_rate = torchaudio.load(audio_path)
    diarize_output = diarize_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    # DiarizeOutput is a named tuple — extract the Annotation, then convert to DataFrame
    annotation = getattr(diarize_output, "speaker_diarization", diarize_output)
    import pandas as pd
    diarize_df = pd.DataFrame([
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ])
    result = whisperx.assign_word_speakers(diarize_df, result)

    if status_callback:
        status_callback("Writing transcript...")

    # Build speaker map from meeting details
    speaker_map: dict[str, str] = {}
    if meeting_details is not None and meeting_details.attendees:
        speaker_map = _build_speaker_map(meeting_details.attendees)

    _write_markdown(result, output_path, meeting_details=meeting_details, speaker_map=speaker_map)
    return output_path


def _write_markdown(
    result: dict,
    output_path: str,
    meeting_details=None,   # MeetingDetails | None
    speaker_map: dict[str, str] | None = None,
) -> None:
    """Format WhisperX result as a diarized markdown transcript."""
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
    current_speaker = None
    current_ts = ""
    current_texts: list[str] = []

    def _flush_speaker():
        if current_speaker and current_texts:
            paragraph = " ".join(current_texts)
            lines.append(f"\n**{current_speaker}** {current_ts}\n{paragraph}")

    for seg in result.get("segments", []):
        raw_speaker = seg.get("speaker", "SPEAKER_00")
        if raw_speaker in speaker_map:
            display_speaker = f"[[{speaker_map[raw_speaker]}]]"
        else:
            display_speaker = _format_speaker(raw_speaker)

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
