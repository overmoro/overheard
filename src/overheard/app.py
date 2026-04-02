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
from overheard.transcribe import transcribe_audio, WHISPER_MODEL


def _output_dir() -> Path:
    """Return the current output directory from config (re-read each time so
    Preferences changes take effect without restarting the app)."""
    return Path(cfg.get("output_dir", str(Path.home() / "meeting-transcripts")))


class TranscriberApp(rumps.App):
    def __init__(self):
        super().__init__("Overheard", icon=None, title="\U0001f3a4")
        self.recording = False
        self.recorder: Recorder | None = None
        self._prefs_window = None  # lazy-init to avoid AppKit startup issues

    @rumps.clicked("Start Recording")
    def start_recording(self, sender):
        if self.recording:
            rumps.notification("Overheard", "", "Already recording.")
            return

        device_id = find_recording_device()
        if device_id is None:
            rumps.notification(
                "Transcriber",
                "No audio device found",
                f"Open Preferences to create an Aggregate Device named '{DEFAULT_DEVICE_NAME}'.",
            )
            return

        self.recorder = Recorder(device_id)
        self.recorder.start()
        self.recording = True
        self.title = "\U0001f534"  # Red circle
        rumps.notification("Overheard", "", "Recording started.")

    @rumps.clicked("Stop & Transcribe")
    def stop_recording(self, sender):
        if not self.recording or not self.recorder:
            rumps.notification("Overheard", "", "Not recording.")
            return

        self.recording = False
        self.title = "\u23f3"  # Hourglass

        audio = self.recorder.stop()
        if audio is None or len(audio) == 0:
            self.title = "\U0001f3a4"
            rumps.notification("Overheard", "", "No audio captured.")
            return

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.recorder.save(audio, tmp.name)
        tmp.close()
        self.recorder = None

        # Build output path — read from config at run time
        output_dir = _output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H%M") + "_meeting.md"
        output_path = str(output_dir / filename)

        def update_status(msg):
            self.title = f"\u23f3 {msg}"

        def run():
            try:
                transcribe_audio(tmp.name, output_path, status_callback=update_status)
                self.title = "\U0001f3a4"
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
                rumps.notification("Overheard", "Done", f"Saved: {filename}")
            except Exception as e:
                self.title = "\U0001f3a4"
                rumps.notification("Overheard", "Error", str(e))
            finally:
                os.unlink(tmp.name)

        threading.Thread(target=run, daemon=True).start()

    @rumps.clicked("Open Transcripts")
    def open_transcripts(self, _):
        d = _output_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    @rumps.clicked("Preferences...")
    def open_preferences(self, _):
        # Import here so AppKit is initialised inside the rumps run loop
        from overheard.preferences import PreferencesWindow
        if self._prefs_window is None:
            self._prefs_window = PreferencesWindow()
        self._prefs_window.show()


def main():
    _output_dir().mkdir(parents=True, exist_ok=True)

    # Load HF_TOKEN from config if not already in environment
    from overheard import config as cfg
    if not os.environ.get("HF_TOKEN"):
        stored = cfg.get("hf_token")
        if stored:
            os.environ["HF_TOKEN"] = stored
        else:
            print("Warning: HF_TOKEN not set. Open Preferences... to add it.", file=sys.stderr)

    app = TranscriberApp()
    app.run()


if __name__ == "__main__":
    main()
