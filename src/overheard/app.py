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


def _resolve_icon(name: str) -> str | None:
    """Resolve an icon file from bundle Resources or repo icon/ directory."""
    for candidate in [
        Path(__file__).parent.parent.parent / "Resources" / name,
        Path(__file__).parent.parent.parent / "icon" / name,
    ]:
        if candidate.exists():
            return str(candidate)
    return None


class TranscriberApp(rumps.App):
    def __init__(self):
        icon = _resolve_icon("menubar.png")
        super().__init__("Overheard", icon=icon, template=True, title="")
        self._state = "idle"
        self._recorder: Recorder | None = None
        self._popover = None     # TransportPopover — built at startup
        self._prefs_window = None
        self._details_panel = None
        self._level_timer: rumps.Timer | None = None
        self._gather_poll_timer: rumps.Timer | None = None

    # ------------------------------------------------------------------
    # Menu items (minimal — main UI is the popover)
    # ------------------------------------------------------------------

    @rumps.clicked("Open Transcripts")
    def open_transcripts(self, _):
        d = _output_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    @rumps.clicked("Preferences...")
    def open_preferences(self, _):
        self._open_preferences_cb()

    # ------------------------------------------------------------------
    # Transport callbacks
    # ------------------------------------------------------------------

    def _on_record(self):
        if self._state == "paused" and self._recorder:
            self._recorder.resume()
            self._set_state("recording", "Recording...")
            self._start_level_timer()
            return

        if self._state != "idle":
            return

        device_id = find_recording_device()
        if device_id is None:
            rumps.notification(
                "Overheard", "No audio device found",
                f"Open Preferences to create an Aggregate Device named '{DEFAULT_DEVICE_NAME}'.",
            )
            return

        recorder = Recorder(device_id)
        self._recorder = recorder

        if self._popover:
            self._popover.configure_channels(recorder._is_multichannel)
        self._set_state("recording", "Recording...")
        self._start_level_timer()

        # Defer stream start to the next run-loop cycle so AppKit finishes
        # processing the current mouse event before CoreAudio begins firing
        # its realtime callbacks.  A race between CoreAudio's realtime thread
        # and AppKit's event unwinding caused intermittent C-level crashes when
        # recorder.start() was called synchronously inside mouseDown_.
        def _deferred_start(timer):
            timer.stop()
            try:
                recorder.start()
            except Exception:
                import traceback
                print(f"stream start failed:\n{traceback.format_exc()}", file=sys.stderr)
                self._recorder = None
                self._set_state("idle", "Audio error")
                self._stop_level_timer()

        rumps.Timer(_deferred_start, 0.05).start()

    def _on_pause(self):
        if self._state == "recording" and self._recorder:
            self._recorder.pause()
            self._stop_level_timer()
            self._set_state("paused", "Paused")

    def _on_stop(self):
        if self._state not in ("recording", "paused") or not self._recorder:
            return

        self._stop_level_timer()

        # Stop recorder and grab audio on the calling thread (fast).
        audio, channels_info = self._recorder.stop()
        self._recorder = None

        if audio is None or len(audio) == 0:
            self._set_state("idle", "Ready")
            rumps.notification("Overheard", "", "No audio captured.")
            return

        self._set_state("idle", "Gathering details...")

        # Reset the shared result slot so the poll timer knows nothing is ready yet.
        self._pending_meeting_meta = None
        self._pending_wav = None
        self._pending_channels_info = None

        # Do all slow work (WAV write, Calendar, pgrep) in a daemon thread.
        def _gather():
            import soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp.close()

            # Calendar — isolated sub-thread with hard join timeout so a
            # TCC dialog or slow iCloud sync can never block the gather thread.
            _mi = [None]
            def _cal():
                try:
                    from overheard.cal import get_current_meeting
                    _mi[0] = get_current_meeting()
                except Exception:
                    pass
            _ct = threading.Thread(target=_cal, daemon=True)
            _ct.start()
            _ct.join(timeout=3)
            meeting_info = _mi[0]

            try:
                from overheard.meeting import detect_source, infer_location
                source = detect_source()
                location = infer_location(source)
            except Exception:
                source = "in-person"
                location = ""

            cal_name = meeting_info.title if meeting_info else ""
            cal_attendees = meeting_info.attendees if meeting_info else []
            cal_location = (meeting_info.location
                            if (meeting_info and meeting_info.location) else location)

            # Write results — the main-thread poll timer will pick these up.
            self._pending_channels_info = channels_info
            self._pending_wav = tmp.name
            self._pending_meeting_meta = (cal_name, source, cal_location, cal_attendees)

        threading.Thread(target=_gather, daemon=True).start()

        # Poll every 200 ms from the main thread until _gather finishes.
        # This timer is created here, on the main thread, so it fires correctly.
        self._gather_poll_timer = rumps.Timer(self._poll_gather_done, 0.2)
        self._gather_poll_timer.start()

    def _poll_gather_done(self, timer):
        """Main-thread timer — fires until background gather completes."""
        if self._pending_meeting_meta is None or self._pending_wav is None:
            return  # not ready yet, wait for next tick
        timer.stop()
        self._gather_poll_timer = None
        meta = self._pending_meeting_meta
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
                import traceback
                traceback.print_exc()
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
        if self._popover:
            self._popover.set_levels(0.0, 0.0)

    def _update_levels(self, timer):
        if self._recorder is None:
            self._stop_level_timer()
            return
        try:
            mic, sys_lvl = self._recorder.get_levels()
            if self._popover:
                self._popover.set_levels(mic, sys_lvl)
        except Exception as e:
            print(f"levels error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: str, status: str = "") -> None:
        self._state = state
        titles = {
            "idle":         "",
            "recording":    " \U0001f534",
            "paused":       " \u23f8",
            "transcribing": " \u23f3",
        }
        self.title = titles.get(state, "")
        if self._popover:
            from overheard.transport import IDLE, RECORDING, PAUSED, TRANSCRIBING
            state_map = {
                "idle": IDLE, "recording": RECORDING,
                "paused": PAUSED, "transcribing": TRANSCRIBING,
            }
            self._popover.set_state(state_map.get(state, IDLE), status)

    def _build_popover(self):
        """Build the panel and hook it to the status bar button."""
        from overheard.popover import TransportPopover
        self._popover = TransportPopover({
            "record":           self._on_record,
            "pause":            self._on_pause,
            "stop":             self._on_stop,
            "open_transcripts": lambda: (
                _output_dir().mkdir(parents=True, exist_ok=True) or
                os.system(f'open "{_output_dir()}"')
            ),
            "preferences":      self._open_preferences_cb,
        })
        try:
            nsstatusitem = self._nsapp.nsstatusitem
            btn = nsstatusitem.button()
            self._popover.hook_status_item(btn)
            nsstatusitem.setMenu_(None)
        except Exception as e:
            print(f"Popover hook failed: {e}", flush=True)

    def _open_preferences_cb(self):
        from overheard.preferences import PreferencesWindow
        if self._prefs_window is None:
            self._prefs_window = PreferencesWindow()
        self._prefs_window.show()

    def _on_discard(self):
        """Discard the pending recording without transcribing."""
        tmp_path = getattr(self, "_pending_wav", None)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self._pending_wav = None
        self._pending_channels_info = None
        self._pending_meeting_meta = None
        self._set_state("idle", "Ready")

    def _ensure_details_panel(self):
        if self._details_panel is None:
            from overheard.details_panel import DetailsPanel
            self._details_panel = DetailsPanel(
                callback=self._on_details_confirmed,
                discard_callback=self._on_discard,
            )


def _ensure_homebrew_path() -> None:
    """Add Homebrew bin to PATH if not already present.

    Subprocesses (ffmpeg, swift, osascript wrappers) need /opt/homebrew/bin
    which may not be inherited when launching from a menu bar agent or
    outside a login shell.
    """
    homebrew_bin = "/opt/homebrew/bin"
    current = os.environ.get("PATH", "")
    if homebrew_bin not in current.split(":"):
        os.environ["PATH"] = homebrew_bin + ":" + current


def main():
    import faulthandler
    faulthandler.enable()   # print C-level backtraces to stderr on crash
    _ensure_homebrew_path()
    _output_dir().mkdir(parents=True, exist_ok=True)

    if not os.environ.get("HF_TOKEN"):
        stored = cfg.get("hf_token")
        if stored:
            os.environ["HF_TOKEN"] = stored
        else:
            print("Warning: HF_TOKEN not set. Open Preferences... to add it.", file=sys.stderr)

    app = TranscriberApp()

    # Build popover and hook it to the status bar button once the run loop starts
    def _init_popover(timer):
        timer.stop()
        app._build_popover()
        app._set_state("idle", "Ready")

    rumps.Timer(_init_popover, 0.5).start()

    app.run()


if __name__ == "__main__":
    main()
