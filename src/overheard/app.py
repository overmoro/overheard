"""Overheard — menu bar application."""

import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import rumps

from overheard import config as cfg
from overheard.audio import Recorder, find_recording_device, DEFAULT_DEVICE_NAME
from overheard.transcribe import transcribe_audio


def _output_dir() -> Path:
    return Path(cfg.get("output_dir", str(Path.home() / "overheard" / "transcripts")))


class TranscriberApp(rumps.App):
    def __init__(self):
        super().__init__("Overheard", icon=None, title="\U0001f3a4")
        self._state = "idle"
        self._recorder: Recorder | None = None
        self._transport = None   # lazy-init inside rumps run loop
        self._prefs_window = None

    # ------------------------------------------------------------------
    # Menu items
    # ------------------------------------------------------------------

    @rumps.clicked("Show Controls")
    def show_controls(self, _):
        self._ensure_transport()
        self._transport.show()

    @rumps.clicked("Open Transcripts")
    def open_transcripts(self, _):
        d = _output_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    @rumps.clicked("Preferences...")
    def open_preferences(self, _):
        from overheard.preferences import PreferencesWindow
        if self._prefs_window is None:
            self._prefs_window = PreferencesWindow()
        self._prefs_window.show()

    # ------------------------------------------------------------------
    # Transport callbacks
    # ------------------------------------------------------------------

    def _on_record(self):
        if self._state == "paused" and self._recorder:
            self._recorder.resume()
            self._set_state("recording", "Recording...")
            return

        device_id = find_recording_device()
        if device_id is None:
            rumps.notification(
                "Overheard", "No audio device found",
                f"Open Preferences to create an Aggregate Device named '{DEFAULT_DEVICE_NAME}'.",
            )
            return

        self._recorder = Recorder(device_id)
        self._recorder.start()
        self._set_state("recording", "Recording...")

    def _on_pause(self):
        if self._state == "recording" and self._recorder:
            self._recorder.pause()
            self._set_state("paused", "Paused")

    def _on_stop(self):
        if self._state not in ("recording", "paused") or not self._recorder:
            return

        audio = self._recorder.stop()
        self._recorder = None

        if audio is None or len(audio) == 0:
            self._set_state("idle", "Ready")
            rumps.notification("Overheard", "", "No audio captured.")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        from overheard.audio import SAMPLE_RATE
        import soundfile as sf
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp.close()

        output_dir = _output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H%M") + "_meeting.md"
        output_path = str(output_dir / filename)

        self._set_state("transcribing", "Transcribing...")

        def run():
            try:
                def on_status(msg):
                    self._set_state("transcribing", msg)
                transcribe_audio(tmp.name, output_path, status_callback=on_status)
                self._set_state("idle", "Done ✓")
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
                rumps.notification("Overheard", "Done", f"Saved: {filename}")
            except Exception as e:
                self._set_state("idle", "Error")
                rumps.notification("Overheard", "Error", str(e))
            finally:
                os.unlink(tmp.name)

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: str, status: str = "") -> None:
        self._state = state
        icons = {
            "idle":         "\U0001f3a4",
            "recording":    "\U0001f534",
            "paused":       "\u23f8",
            "transcribing": "\u23f3",
        }
        self.title = icons.get(state, "\U0001f3a4")
        if self._transport:
            from overheard.transport import IDLE, RECORDING, PAUSED, TRANSCRIBING
            state_map = {
                "idle": IDLE, "recording": RECORDING,
                "paused": PAUSED, "transcribing": TRANSCRIBING,
            }
            self._transport.set_state(state_map.get(state, IDLE), status)

    def _ensure_transport(self):
        if self._transport is None:
            from overheard.transport import TransportWindow
            self._transport = TransportWindow({
                "record": self._on_record,
                "pause":  self._on_pause,
                "stop":   self._on_stop,
            })


def main():
    _output_dir().mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_TOKEN"):
        stored = cfg.get("hf_token")
        if stored:
            os.environ["HF_TOKEN"] = stored
        else:
            print("Warning: HF_TOKEN not set. Open Preferences... to add it.", file=sys.stderr)

    app = TranscriberApp()

    # Auto-open controls on first launch
    def _open_on_start(_):
        app._ensure_transport()
        app._transport.show()

    rumps.Timer(_open_on_start, 0.5).start()

    app.run()


if __name__ == "__main__":
    main()
