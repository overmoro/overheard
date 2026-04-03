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
from overheard.audio import Recorder, find_recording_device, DEFAULT_DEVICE_NAME, SAMPLE_RATE
from overheard.transcribe import transcribe_audio



def _output_dir() -> Path:
    return Path(cfg.get("output_dir", str(Path.home() / "overheard" / "transcripts")))


def _resolve_output_path(filename: str) -> Path:
    """Determine where to write the transcript.

    If Obsidian integration is enabled and configured, write to vault/inbox/.
    Otherwise fall back to the configured output directory.
    """
    if cfg.get("obsidian_enabled", False):
        vault = cfg.get("obsidian_vault", "")
        inbox = cfg.get("obsidian_inbox", "01_Inbox")
        if vault:
            dest = Path(vault) / inbox
            dest.mkdir(parents=True, exist_ok=True)
            return dest / filename

    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out / filename


class TranscriberApp(rumps.App):
    def __init__(self):
        super().__init__("Overheard", icon=None, title="\U0001f3a4")
        self._state = "idle"
        self._recorder: Recorder | None = None
        self._transport = None   # lazy-init inside rumps run loop
        self._prefs_window = None
        self._details_panel = None
        self._level_timer: rumps.Timer | None = None

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
            self._start_level_timer()
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

        # Inform transport panel whether we're multichannel
        if self._transport:
            self._transport.configure_channels(self._recorder._is_multichannel)

        self._set_state("recording", "Recording...")
        self._start_level_timer()

    def _on_pause(self):
        if self._state == "recording" and self._recorder:
            self._recorder.pause()
            self._stop_level_timer()
            self._set_state("paused", "Paused")

    def _on_stop(self):
        if self._state not in ("recording", "paused") or not self._recorder:
            return

        self._stop_level_timer()

        # Step 1: stop recorder, get audio + channel info
        audio, channels_info = self._recorder.stop()
        self._recorder = None

        if audio is None or len(audio) == 0:
            self._set_state("idle", "Ready")
            rumps.notification("Overheard", "", "No audio captured.")
            return

        # Step 2: save temp WAV
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp.close()
        tmp_path = tmp.name

        self._pending_wav = tmp_path
        self._pending_channels_info = channels_info
        self._set_state("idle", "Gathering details...")

        # Steps 3-5: do slow work (AppleScript, pgrep) in background,
        # then schedule the panel show on the main thread via a timer.
        def _gather_and_show():
            print("DEBUG: _gather_and_show started", flush=True)

            # Calendar query runs in a daemon thread with a hard join timeout.
            meeting_info = [None]
            def _cal_query():
                print("DEBUG: calendar query starting", flush=True)
                try:
                    from overheard.calendar import get_current_meeting
                    meeting_info[0] = get_current_meeting()
                except Exception as e:
                    print(f"DEBUG: calendar exception: {e}", flush=True)
                print("DEBUG: calendar query done", flush=True)
            cal_thread = threading.Thread(target=_cal_query, daemon=True)
            cal_thread.start()
            cal_thread.join(timeout=3)
            print(f"DEBUG: calendar join complete, result={meeting_info[0]}", flush=True)
            meeting_info = meeting_info[0]

            print("DEBUG: running detect_source", flush=True)
            try:
                from overheard.meeting import detect_source, infer_location
                source = detect_source()
                location = infer_location(source)
            except Exception as e:
                print(f"DEBUG: detect_source exception: {e}", flush=True)
                source = "in-person"
                location = ""

            print(f"DEBUG: source={source}, scheduling panel", flush=True)
            cal_name = meeting_info.title if meeting_info else ""
            cal_attendees = meeting_info.attendees if meeting_info else []
            cal_location = (meeting_info.location
                            if (meeting_info and meeting_info.location) else location)

            self._pending_meeting_meta = (cal_name, source, cal_location, cal_attendees)

            print("DEBUG: starting rumps.Timer", flush=True)
            t = rumps.Timer(self._show_details_on_main, 0.05)
            t.start()
            print("DEBUG: rumps.Timer started", flush=True)

        threading.Thread(target=_gather_and_show, daemon=True).start()

    def _show_details_on_main(self, timer):
        print("DEBUG: _show_details_on_main fired", flush=True)
        timer.stop()
        meta = getattr(self, "_pending_meeting_meta", ("", "in-person", "", []))
        cal_name, source, cal_location, cal_attendees = meta
        self._ensure_details_panel()
        self._set_state("idle", "Fill in details...")
        self._details_panel.show(
            name=cal_name,
            source=source,
            location=cal_location,
            attendees=cal_attendees,
            speaker_count=2,
        )

    # ------------------------------------------------------------------
    # Details panel callback
    # ------------------------------------------------------------------

    def _on_details_confirmed(self, details):
        """Called from DetailsPanel after user fills in meeting metadata."""
        from overheard.details_panel import make_filename
        from overheard import config as cfg

        tmp_path = getattr(self, "_pending_wav", None)
        if not tmp_path:
            return

        filename = make_filename(details)
        output_path = str(_resolve_output_path(filename))

        # Mic speaker name for attribution (plumbed through, not yet active)
        mic_speaker = cfg.get("local_speaker_name") or None

        self._set_state("transcribing", "Transcribing...")

        def run():
            try:
                def on_status(msg):
                    self._set_state("transcribing", msg)

                transcribe_audio(
                    tmp_path,
                    output_path,
                    status_callback=on_status,
                    meeting_details=details,
                    mic_speaker=mic_speaker,
                )
                self._set_state("idle", "Done \u2713")
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
                rumps.notification("Overheard", "Done", f"Saved: {filename}")
            except Exception as e:
                msg = str(e)[:80]
                self._set_state("idle", f"\u2717 {msg}")
                rumps.notification("Overheard", "Error", str(e))
            finally:
                if cfg.get("keep_recordings", False):
                    audio_path = output_path.replace(".md", ".wav")
                    os.rename(tmp_path, audio_path)
                else:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                self._pending_wav = None
                self._pending_channels_info = None

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Level meter timer
    # ------------------------------------------------------------------

    def _start_level_timer(self):
        if self._level_timer is not None:
            return
        self._level_timer = rumps.Timer(self._update_levels, 0.1)
        self._level_timer.start()

    def _stop_level_timer(self):
        if self._level_timer is not None:
            self._level_timer.stop()
            self._level_timer = None
        if self._transport:
            self._transport.set_levels(0.0, 0.0)

    def _update_levels(self, timer):
        if self._recorder is None:
            self._stop_level_timer()
            return
        try:
            mic, sys_lvl = self._recorder.get_levels()
            if self._transport:
                self._transport.set_levels(mic, sys_lvl)
        except Exception:
            pass

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

    def _ensure_details_panel(self):
        if self._details_panel is None:
            from overheard.details_panel import DetailsPanel
            self._details_panel = DetailsPanel(callback=self._on_details_confirmed)


def main():
    _output_dir().mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_TOKEN"):
        stored = cfg.get("hf_token")
        if stored:
            os.environ["HF_TOKEN"] = stored
        else:
            print("Warning: HF_TOKEN not set. Open Preferences... to add it.", file=sys.stderr)

    app = TranscriberApp()

    # Auto-open controls once at startup
    def _open_on_start(timer):
        timer.stop()
        app._ensure_transport()
        app._transport.show()

    rumps.Timer(_open_on_start, 0.5).start()

    app.run()


if __name__ == "__main__":
    main()
