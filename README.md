# Overheard

Local macOS menu bar app that records meetings and produces diarized markdown transcripts. No cloud services — runs entirely on your machine using WhisperX and pyannote.

## What it does

- Sits in your menu bar with a start/stop toggle
- Records from a combined system audio + microphone device
- Transcribes with WhisperX (Whisper large-v3)
- Diarizes speakers with pyannote
- Outputs a timestamped markdown file to `~/meeting-transcripts/`

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.10+
- [Hugging Face](https://huggingface.co/) account with [pyannote model access](https://huggingface.co/pyannote/speaker-diarization-3.1) accepted
- BlackHole 2ch (virtual audio driver)

## Quick Start

```bash
# 1. Install system dependencies
./scripts/setup-macos.sh

# 2. Install the Python package
pip install -e .

# 3. Set your Hugging Face token
export HF_TOKEN=your_token_here  # add to ~/.zshrc

# 4. Run
overheard
```

## Audio Setup

You need two virtual devices configured in **Audio MIDI Setup** (search Spotlight):

### Aggregate Device (for recording)

This combines your mic and system audio into a single input:

1. Click **+** → **Create Aggregate Device**
2. Rename to **Meeting Capture**
3. Check **BlackHole 2ch** and **MacBook Pro Microphone**
4. Set **BlackHole 2ch** as the clock source

### Multi-Output Device (for monitoring)

This sends system audio to both your speakers and BlackHole:

1. Click **+** → **Create Multi-Output Device**
2. Check **MacBook Pro Speakers** and **BlackHole 2ch**
3. In **System Settings → Sound → Output**, select this Multi-Output Device before meetings

## Output Format

Files are saved to `~/meeting-transcripts/` as:

```
2026-04-02_1430_meeting.md
```

```markdown
# Meeting — 2 April 2026, 2:30pm

**SPEAKER_00:** [00:00:12] Thanks for joining...
**SPEAKER_01:** [00:00:15] Of course, good to be here...
```

## Configuration

| Environment Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Hugging Face access token for pyannote models |

The app looks for an audio device named **Meeting Capture** by default, falling back to **BlackHole** then **MacBook Pro Microphone**.

## Development

```bash
git clone <repo-url>
cd overheard
pip install -e .
overheard
```

## License

MIT
