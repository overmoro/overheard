"""Shared fixtures.

Audio is generated rather than committed. Two reasons: the repo stays free of
binary blobs, and every fixture is deterministic, so a failure means the code
changed and not that a recording drifted.

Anything needing real speech (the ASR and diarization models cannot do useful
work on synthetic tones) lives in tests/integration and is generated with `say`
at test time.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Capability gates for integration tests
# ---------------------------------------------------------------------------


def helper_binary() -> Path | None:
    from overheard.helper import helper_path

    return helper_path()


requires_helper = pytest.mark.skipif(
    helper_binary() is None,
    reason="overheard-helper not built; run scripts/build-helper.sh",
)

requires_say = pytest.mark.skipif(
    shutil.which("say") is None or sys.platform != "darwin",
    reason="needs macOS `say` for speech fixtures",
)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config and the speaker library at a temporary directory.

    Without this, tests would read and write the developer's real settings and
    voice library.
    """
    from overheard import config as cfg
    from overheard import speakers

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(speakers, "LIBRARY_PATH", tmp_path / "speakers.json")
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------


def tone(seconds: float, freq: float, rate: int = SAMPLE_RATE, amplitude: float = 0.3):
    """A steady tone. Stands in for "some audio is present"."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, rate: int = SAMPLE_RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


@pytest.fixture
def dual_track_wav(tmp_path):
    """A 3-channel recording matching the Meeting Capture aggregate layout.

    Channels 0 and 1 carry system audio, channel 2 the microphone, which is the
    layout that exposed ffmpeg silently dropping channel 2 as LFE.
    """
    import soundfile as sf

    mic = np.concatenate([tone(2.0, 220.0), silence(2.0)])
    system = np.concatenate([silence(2.0), tone(2.0, 660.0)])
    data = np.stack([system, system, mic], axis=1)

    path = tmp_path / "dual.wav"
    sf.write(path, data, SAMPLE_RATE)
    return path


@pytest.fixture
def mono_wav(tmp_path):
    import soundfile as sf

    path = tmp_path / "mono.wav"
    sf.write(path, tone(2.0, 440.0), SAMPLE_RATE)
    return path


@pytest.fixture
def silent_mic_wav(tmp_path):
    """System audio present, microphone digitally silent.

    The case a whole-file signal check cannot catch: plenty of overall signal
    while everything the user said is missing.
    """
    import soundfile as sf

    system = tone(3.0, 440.0)
    data = np.stack([system, system, silence(3.0)], axis=1)
    path = tmp_path / "silent_mic.wav"
    sf.write(path, data, SAMPLE_RATE)
    return path


def segment(start: float, end: float, text: str, speaker: str | None = None) -> dict:
    """A transcript segment in the shape the pipeline passes around."""
    entry = {"start": start, "end": end, "text": text}
    if speaker is not None:
        entry["speaker"] = speaker
    return entry


def turn(start: float, end: float, speaker: str, embedding=None) -> dict:
    """A diarization turn."""
    entry = {"start": start, "end": end, "speaker": speaker}
    if embedding is not None:
        entry["embedding"] = embedding
    return entry
