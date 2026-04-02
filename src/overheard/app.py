"""Overheard — menu bar application."""

import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import rumps

from overheard.audio import Recorder, find_recording_device, DEFAULT_DEVICE_NAME
from overheard.transcribe import transcribe_audio, OUTPUT_DIR


class TranscriberApp(rumps.App):
    def __init__(self):
        super().__init__("Overheard", icon=None, title="\U0001f3a4")
        self.recording = False
        self.recorder: Recorder | None = None

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
                f"Create an Aggregate Device named '{DEFAULT_DEVICE_NAME}' in Audio MIDI Setup.",
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

        # Output path
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H%M") + "_meeting.md"
        output_path = str(OUTPUT_DIR / filename)

        def update_status(msg):
            self.title = f"\u23f3 {msg}"

        def run():
            try:
                transcribe_audio(tmp.name, output_path, status_callback=update_status)
                self.title = "\U0001f3a4"
                rumps.notification("Overheard", "Done", f"Saved: {filename}")
            except Exception as e:
                self.title = "\U0001f3a4"
                rumps.notification("Overheard", "Error", str(e))
            finally:
                os.unlink(tmp.name)

        threading.Thread(target=run, daemon=True).start()

    @rumps.clicked("Open Transcripts")
    def open_transcripts(self, _):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{OUTPUT_DIR}"')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set. Diarization will fail.", file=sys.stderr)
        print("Set it with: export HF_TOKEN=your_token_here", file=sys.stderr)

    app = TranscriberApp()
    app.run()


if __name__ == "__main__":
    main()
