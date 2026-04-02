"""Audio device discovery and recording."""

import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_DEVICE_NAME = "Meeting Capture"
FALLBACK_DEVICES = ["BlackHole", "MacBook Pro Microphone"]
SAMPLE_RATE = 16000
CHANNELS = 1


def find_device(name: str) -> int | None:
    """Find an audio input device by name (case-insensitive partial match)."""
    for d in sd.query_devices():
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return d["index"]
    return None


def find_recording_device(preferred: str = DEFAULT_DEVICE_NAME) -> int | None:
    """Find the best available recording device with fallback chain."""
    device_id = find_device(preferred)
    if device_id is not None:
        return device_id
    for name in FALLBACK_DEVICES:
        device_id = find_device(name)
        if device_id is not None:
            return device_id
    return None


def list_input_devices() -> list[dict]:
    """List all available input devices."""
    return [
        {"index": d["index"], "name": d["name"], "channels": d["max_input_channels"]}
        for d in sd.query_devices()
        if d["max_input_channels"] > 0
    ]


class Recorder:
    """Simple audio recorder using sounddevice."""

    def __init__(self, device_id: int, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._chunks = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio: {status}", file=sys.stderr)
            self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray | None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks, axis=0)
        self._chunks = []
        return audio

    def save(self, audio: np.ndarray, path: str) -> str:
        sf.write(path, audio, self.sample_rate)
        return path
