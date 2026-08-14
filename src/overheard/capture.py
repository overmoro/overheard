"""System audio capture via Core Audio process taps.

Wraps the `overheard-helper capture` binary, which streams interleaved 32-bit
float PCM on stdout and newline-delimited JSON status on stderr.

This replaces the BlackHole route entirely. No virtual audio driver, no
aggregate devices to create in Audio MIDI Setup, and no switching the system
output to a monitoring device before each meeting: the tap follows whatever the
user is already listening to.

TapRecorder deliberately mirrors audio.Recorder's interface (start/stop/pause/
resume/get_levels/set_tap) so the two are interchangeable, and audio.Recorder
remains the fallback on macOS older than 14.4 or when tap permission is refused.
"""

import json
import subprocess
import sys
import threading

import numpy as np

from overheard.helper import capture_available, helper_path

# Frames pulled from the pipe per read
_READ_FRAMES = 4096


def is_available() -> tuple[bool, str]:
    """Return (available, reason). Reason is empty when available."""
    return capture_available()


class TapRecorder:
    """Records system audio (and the mic) through the Core Audio tap helper.

    Produces a 2-channel float32 array, channel 0 the microphone and channel 1
    the system audio, matching the channels_info contract used by the
    transcription pipeline.
    """

    def __init__(self, capture_mic: bool = True, include_pids: list[int] | None = None):
        self.capture_mic = capture_mic
        self.include_pids = include_pids or []

        self.sample_rate = 48000       # replaced by the helper's ready message
        self.channels = 2 if capture_mic else 1
        self._layout: list[str] = ["mic", "system"] if capture_mic else ["system"]

        self._proc: subprocess.Popen | None = None
        self._chunks: list[np.ndarray] = []
        self._paused = False
        self._level_buf: np.ndarray | None = None
        self._stop_event: threading.Event | None = None
        self._reader: threading.Thread | None = None
        self._status_reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._tap = None

    # ------------------------------------------------------------------
    # Interface shared with audio.Recorder
    # ------------------------------------------------------------------

    @property
    def _is_multichannel(self) -> bool:
        """True when a separate mic track is present (drives the popover meters)."""
        return self.channels > 1

    @property
    def _channels_info(self) -> dict | None:
        if self.channels < 2:
            return None
        return {"mic_channel": 0, "system_channels": [1]}

    def set_tap(self, tap) -> None:
        self._tap = tap

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start(self, timeout: float = 10.0) -> None:
        """Launch the helper and block until it reports ready."""
        binary = helper_path()
        if binary is None:
            raise RuntimeError("overheard-helper not found; run scripts/build-helper.sh")

        argv = [str(binary), "capture"]
        if not self.capture_mic:
            argv.append("--no-mic")
        for pid in self.include_pids:
            argv += ["--include-pid", str(pid)]

        self._chunks = []
        self._paused = False
        self._level_buf = None
        self._error = None
        self._ready.clear()
        self._stop_event = threading.Event()

        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

        self._status_reader = threading.Thread(target=self._read_status, daemon=True)
        self._status_reader.start()

        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError(self._error or "overheard-helper did not start in time")
        if self._error:
            err = self._error
            self.stop()
            raise RuntimeError(err)

        self._reader = threading.Thread(target=self._read_audio, daemon=True)
        self._reader.start()

    def stop(self) -> tuple[np.ndarray | None, dict | None]:
        """Stop the helper and return (audio, channels_info)."""
        if self._stop_event:
            self._stop_event.set()

        proc = self._proc
        if proc is not None and proc.poll() is None:
            # SIGTERM so the helper can tear down the tap and aggregate device;
            # leaking those leaves stale objects registered with Core Audio.
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

        for thread in (self._reader, self._status_reader):
            if thread is not None:
                thread.join(timeout=2.0)
        self._reader = None
        self._status_reader = None
        self._proc = None

        if not self._chunks:
            return None, None
        audio = np.concatenate(self._chunks, axis=0)
        self._chunks = []
        self._level_buf = None
        return audio, self._channels_info

    def get_levels(self) -> tuple[float, float]:
        """Return (mic_rms, system_rms) from the most recent block."""
        buf = self._level_buf
        if buf is None or len(buf) == 0:
            return 0.0, 0.0
        if self.channels > 1:
            mic = float(np.sqrt(np.mean(buf[:, 0] ** 2)))
            system = float(np.sqrt(np.mean(buf[:, 1] ** 2)))
            return mic, system
        rms = float(np.sqrt(np.mean(buf[:, 0] ** 2)))
        return rms, rms

    def save(self, audio: np.ndarray, path: str) -> str:
        import soundfile as sf
        sf.write(path, audio, self.sample_rate)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_status(self) -> None:
        """Consume the helper's JSON status stream until it exits."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[overheard-helper] {line}", file=sys.stderr)
                continue

            kind = msg.get("type")
            if kind == "ready":
                self.sample_rate = int(msg.get("sampleRate") or self.sample_rate)
                self.channels = int(msg.get("channels") or self.channels)
                self._layout = msg.get("layout") or self._layout
                self._ready.set()
            elif kind == "fatal":
                self._error = msg.get("message") or "capture failed"
                self._ready.set()
            elif kind == "warning":
                print(f"[overheard-helper] {msg.get('message')}", file=sys.stderr)

    def _read_audio(self) -> None:
        """Pull PCM frames from stdout until the helper stops."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        frame_bytes = self.channels * 4
        want = _READ_FRAMES * frame_bytes

        buffer = b""
        while self._stop_event is not None and not self._stop_event.is_set():
            data = proc.stdout.read(want)
            if not data:
                break

            # A pipe read returns whatever bytes are available, which is often
            # a partial frame. The leftover must be carried into the next read:
            # discarding it would shift every following sample by part of a
            # frame, permanently swapping the mic and system channels for the
            # rest of the recording.
            buffer += data
            usable = len(buffer) - (len(buffer) % frame_bytes)
            if usable <= 0:
                continue
            block = np.frombuffer(buffer[:usable], dtype=np.float32).reshape(-1, self.channels)
            buffer = buffer[usable:]

            if self._paused:
                continue

            self._chunks.append(block)
            self._level_buf = block

            tap = self._tap
            if tap is not None:
                try:
                    tap(block)
                except Exception as e:
                    print(f"Capture: tap error: {e}", file=sys.stderr)
