# Overheard

Local macOS menu bar app that records meetings and produces diarized markdown
transcripts. Nothing leaves your machine, and it needs no accounts or API keys.

## What it does

- Sits in your menu bar with a start/stop toggle
- Captures system audio and your microphone as separate tracks, using Core Audio
  process taps. No virtual audio driver, and nothing to reconfigure before a call
- Transcribes with NVIDIA Parakeet TDT v3 on the Apple Silicon GPU
- Labels speakers with FluidAudio on the Neural Engine
- Shows a running transcript while the meeting is in progress
- Writes a timestamped markdown file, optionally straight into an Obsidian vault

## Requirements

- macOS 14.4 or later, Apple Silicon
- Python 3.10+

That is the whole list. Models download on first use.

## Quick start

```bash
pip install -e .
overheard
```

The first time you press Record, macOS asks for microphone access. Grant it: the
microphone is also the clock source for capture, so it is needed even when only
recording system audio.

## How capture works

Overheard uses Core Audio process taps, added in macOS 14.4. It records whatever
your Mac is playing, alongside your microphone, without routing your audio
through a virtual device first. There is nothing to install and nothing to switch
back afterwards.

The two sides are kept as separate tracks and transcribed independently. That
means the app knows which voice is yours rather than guessing, and it avoids
mixing your microphone against the system audio it is echoing when you use
speakers instead of headphones.

On macOS earlier than 14.4, Overheard falls back to recording from a normal input
device. For combined capture on those systems you still need BlackHole and an
aggregate device named **Meeting Capture**; `./scripts/setup-macos.sh` sets that
up.

## Output

Each transcript opens with YAML frontmatter (type, date, source, attendees,
status) and a dated heading, followed by the body:

```markdown
**[[Don Reddin]]** [00:00:12]
Thanks for joining...

**[[Sarah Kelly]]** [00:00:15]
Of course, good to be here...
```

Attendee names come from your calendar entry or the details panel shown after
recording. Speakers without a name are numbered in order of first speech.

## Configuration

Settings live in `~/.config/overheard/config.json` and the Preferences window.

| Key | Default | Description |
|---|---|---|
| `engine` | `parakeet` | `parakeet` (GPU) or `whisper` (CPU, 99 languages) |
| `capture_backend` | `auto` | `auto`, `tap`, or `device` |
| `diarizer` | `auto` | `auto`, `fluidaudio`, or `pyannote` |
| `live_preview` | `true` | Show the running transcript while recording |
| `output_dir` | `~/overheard/transcripts` | Where transcripts are written |
| `local_speaker_name` | `Don` | Name given to the voice on your microphone |
| `keep_recordings` | `false` | Keep the audio alongside the transcript |

No environment variables are required. `HF_TOKEN` is consulted only if you
deliberately select the `pyannote` diarizer.

### Optional extras

```bash
pip install -e '.[whisper]'    # WhisperX, for the CPU engine
pip install -e '.[pyannote]'   # pyannote fallback diarizer, pulls in torch
```

## Development

The native helper provides Core Audio capture and diarization. A prebuilt,
ad-hoc-signed binary is committed to `Resources/`, so this is only needed if you
change `helper/`:

```bash
./scripts/build-helper.sh
```

Building the menu bar app itself:

```bash
./build_app.sh
```

## License

MIT
