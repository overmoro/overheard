"""Overheard: menu bar application."""

import os
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import rumps

from overheard import config as cfg
from overheard.audio import Recorder, find_recording_device, DEFAULT_DEVICE_NAME, SAMPLE_RATE
from overheard.protocols import AudioSource
from overheard.pipeline import transcribe_audio
from overheard.state import IDLE, PAUSED, RECORDING, TRANSCRIBING

if TYPE_CHECKING:
    from overheard.details_panel import DetailsPanel



def _output_dir() -> Path:
    return Path(cfg.get("output_dir"))


def _resolve_output_path(filename: str) -> Path:
    """Determine where to write the transcript.

    If Obsidian integration is enabled and configured, write to vault/inbox/.
    Otherwise fall back to the configured output directory.
    """
    if cfg.get("obsidian_enabled"):
        vault = cfg.get("obsidian_vault")
        inbox = cfg.get("obsidian_inbox")
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
        self._state = IDLE
        self._recorder: AudioSource | None = None
        self._popover = None     # TransportPopover, built at startup
        self._prefs_window = None
        # A real type, not Any. Using Any on a first-party class silently
        # suppresses checking, and it hid a misspelled method from the gate.
        self._details_panel: "DetailsPanel | None" = None
        self._level_timer: rumps.Timer | None = None
        self._gather_poll_timer: rumps.Timer | None = None
        self._live = None          # LiveTranscriber while recording
        self._live_panel = None    # LiveTranscriptPanel
        self._record_rate = SAMPLE_RATE   # native rate of the active recording
        self._recorder_poll_timer: rumps.Timer | None = None
        self._pending_recorder = None
        self._pending_recorder_error = None

    # ------------------------------------------------------------------
    # Menu items (minimal; main UI is the popover)
    # ------------------------------------------------------------------

    @rumps.clicked("Open Transcripts")
    def open_transcripts(self, _):
        d = _output_dir()
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    def _reveal_transcripts(self) -> None:
        """Open the transcripts folder, creating it if this is the first run."""
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
        if self._state == PAUSED and self._recorder:
            self._recorder.resume()
            self._set_state(RECORDING, "Recording...")
            self._start_level_timer()
            return

        if self._state != IDLE:
            return

        # Give immediate visual feedback, then defer ALL CoreAudio work.
        # find_recording_device() and Recorder() both call sd.query_devices()
        # which touches CoreAudio. Running those synchronously inside mouseDown_
        # caused C-level crashes in earlier builds. The timer fires on the main
        # run-loop 50 ms later, well after AppKit has finished unwinding the
        # click event, so UI calls inside the callback remain safe.
        self._set_state(RECORDING, "Starting...")

        def _deferred_start(timer):
            timer.stop()
            # Starting capture can block: the tap helper waits on the system
            # audio permission dialog the first time it runs. Do it off the main
            # thread and collect the result from a poll timer, matching the
            # pattern used by _on_stop.
            self._pending_recorder = None
            self._pending_recorder_error = None
            threading.Thread(target=self._start_recorder_bg, daemon=True).start()
            self._recorder_poll_timer = rumps.Timer(self._poll_recorder_started, 0.15)
            self._recorder_poll_timer.start()

        rumps.Timer(_deferred_start, 0.05).start()

    def _make_recorder(self) -> AudioSource:
        """Pick a capture backend.

        Prefers Core Audio process taps, which need no BlackHole install, no
        aggregate devices and no output rerouting. Falls back to the input
        device recorder when taps are unavailable.

        Either way the result satisfies AudioSource, which is the only thing
        the rest of this class may assume about it.
        """
        backend = cfg.get("capture_backend")

        if backend in ("auto", "tap"):
            from overheard.capture import TapRecorder, is_available
            available, reason = is_available()
            if available:
                return TapRecorder()
            if backend == "tap":
                raise RuntimeError(f"Tap capture unavailable: {reason}")
            print(f"Audio: tap capture unavailable ({reason}); using input device",
                  file=sys.stderr)

        device_id = find_recording_device()
        if device_id is None:
            raise RuntimeError(
                f"No audio device found. Open Preferences to create an Aggregate "
                f"Device named '{DEFAULT_DEVICE_NAME}'."
            )
        return Recorder(device_id)

    def _start_recorder_bg(self):
        """Build and start the recorder off the main thread."""
        try:
            recorder = self._make_recorder()
            recorder.start()
            self._pending_recorder = recorder
        except Exception as e:
            import traceback
            print(f"capture start failed:\n{traceback.format_exc()}", file=sys.stderr)
            self._pending_recorder_error = str(e)

    def _poll_recorder_started(self, timer):
        """Main-thread timer: finish wiring up once the recorder is running."""
        recorder = self._pending_recorder
        error = self._pending_recorder_error
        if recorder is None and error is None:
            return  # still starting

        timer.stop()
        self._recorder_poll_timer = None
        self._pending_recorder = None
        self._pending_recorder_error = None

        if recorder is None:
            detail = error or "capture did not start"
            rumps.notification("Overheard", "Could not start recording", detail)
            self._set_state(IDLE, detail[:60])
            return

        self._recorder = recorder
        self._record_rate = recorder.sample_rate

        if self._popover:
            self._popover.configure_channels(recorder.is_multichannel)
        self._set_state(RECORDING, "Recording...")
        self._start_level_timer()
        self._start_live(recorder)

    # ------------------------------------------------------------------
    # Live transcription
    # ------------------------------------------------------------------

    def _start_live(self, recorder):
        """Begin live transcription if enabled, and show the transcript panel.

        Failures here are non-fatal: live transcription is a convenience, and
        the authoritative transcript is still produced after the meeting.
        """
        if not cfg.get("live_preview"):
            return
        try:
            from overheard.live import LiveTranscriber, is_available
            if not is_available():
                print("Live transcription unavailable: parakeet-mlx not installed",
                      file=sys.stderr)
                return

            from overheard.live_panel import LiveTranscriptPanel
            if self._live_panel is None:
                self._live_panel = LiveTranscriptPanel()

            live = LiveTranscriber(recorder.sample_rate, recorder.channels_info)

            # Streaming speaker attribution, only useful when the mic and
            # system arrive on separate channels.
            if cfg.get("live_speakers") and recorder.channels_info:
                from overheard.live import LiveDiarizer
                diarizer = LiveDiarizer()
                if diarizer.start():
                    live.diarizer = diarizer

            live.start()
            recorder.set_tap(live.feed)
            self._live = live

            self._live_panel.bind(live)
            self._live_panel.show()
        except Exception:
            import traceback
            print(f"live transcription failed to start:\n{traceback.format_exc()}",
                  file=sys.stderr)
            self._live = None

    def _stop_live(self):
        """Stop live transcription, leaving the final text visible."""
        if self._live is not None:
            try:
                if getattr(self._live, "diarizer", None) is not None:
                    self._live.diarizer.stop()
            except Exception:
                pass
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        if self._live_panel is not None:
            try:
                self._live_panel.unbind()
            except Exception:
                pass

    def _on_pause(self):
        if self._state == RECORDING and self._recorder:
            self._recorder.pause()
            self._stop_level_timer()
            self._set_state(PAUSED, "Paused")

    def _on_stop(self):
        if self._state not in (RECORDING, PAUSED) or not self._recorder:
            return

        self._stop_level_timer()

        # Detach the live tap before stopping so no further blocks are fed.
        self._recorder.set_tap(None)

        # Stop recorder and grab audio on the calling thread (fast).
        #
        # This is fallible in two ways that both surface here: proc.wait after
        # a kill can time out, and np.concatenate over an hour of multichannel
        # chunks can raise MemoryError. The two teardown steps above have
        # already run and cannot be undone, so there is no recording to return
        # to. The objc_safe guard on the button keeps that from aborting the
        # process; without this, it would instead leave the app showing
        # RECORDING with a dead recorder behind it, and a second press of Stop
        # would call stop() on it again.
        try:
            audio, channels_info = self._recorder.stop()
            record_rate = self._recorder.sample_rate
        except Exception as e:
            print(f"[overheard] the recorder could not be stopped: {e}", file=sys.stderr)
            traceback.print_exc()
            self._recorder = None
            self._stop_live()
            self._set_state(IDLE, "Recording failed, see the log")
            rumps.notification(
                "Overheard", "Recording failed",
                "The recorder could not be stopped. Details are in the log.",
            )
            return

        self._recorder = None
        self._stop_live()

        if audio is None or len(audio) == 0:
            self._set_state(IDLE, "Ready")
            rumps.notification("Overheard", "", "No audio captured.")
            return

        self._set_state(IDLE, "Gathering details...")

        # Reset the shared result slot so the poll timer knows nothing is ready yet.
        self._pending_meeting_meta = None
        self._pending_wav = None
        self._pending_channels_info = None

        # Do all slow work (WAV write, Calendar, pgrep) in a daemon thread.
        def _gather():
            import soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # Write at the rate the audio was actually captured at. The Recorder
            # opens the device at its native rate (CoreAudio aggregates refuse
            # rate conversion), so stamping a fixed 16 kHz header here mislabelled
            # every recording and played it back at a third of its real speed.
            sf.write(tmp.name, audio, record_rate)
            tmp.close()

            # Calendar: isolated sub-thread with hard join timeout so a
            # TCC dialog or slow iCloud sync can never block the gather thread.
            _mi: list = [None]
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

            # Write results; the main-thread poll timer will pick these up.
            self._pending_channels_info = channels_info
            self._pending_wav = tmp.name
            self._pending_meeting_meta = (cal_name, source, cal_location, cal_attendees)

        threading.Thread(target=_gather, daemon=True).start()

        # Poll every 200 ms from the main thread until _gather finishes.
        # This timer is created here, on the main thread, so it fires correctly.
        self._gather_poll_timer = rumps.Timer(self._poll_gather_done, 0.2)
        self._gather_poll_timer.start()

    def _poll_gather_done(self, timer):
        """Main-thread timer, fires until background gather completes."""
        if self._pending_meeting_meta is None or self._pending_wav is None:
            return  # not ready yet, wait for next tick
        timer.stop()
        self._gather_poll_timer = None
        meta = self._pending_meeting_meta
        cal_name, source, cal_location, cal_attendees = meta
        # Not here to stop an abort: rumps wraps its own Timer callbacks in
        # try/except (rumps.py:730), so an exception escaping this one is
        # caught. It is caught silently, into a traceback nobody running a
        # bundled .app will ever see, and it leaves the status label sitting on
        # whatever it last said. The recording is already on disk at this
        # point, so the failure has to become a visible state and a message
        # the user can act on rather than a line in a log they do not have.
        try:
            panel = self._ensure_details_panel()
            self._set_state(IDLE, "Fill in details...")
            panel.show(
                name=cal_name,
                source=source,
                location=cal_location,
                attendees=cal_attendees,
                speaker_count=2,
            )
        except Exception as e:
            print(f"[overheard] could not open the details panel: {e}", file=sys.stderr)
            traceback.print_exc()
            self._set_state(IDLE, "Details panel failed, see the log")

    # ------------------------------------------------------------------
    # Details panel callback
    # ------------------------------------------------------------------

    def _on_details_confirmed(self, details):
        """Called from DetailsPanel after user fills in meeting metadata."""
        from overheard.details_panel import make_filename
        from overheard import config as cfg

        tmp_path = getattr(self, "_pending_wav", None)
        channels_info = getattr(self, "_pending_channels_info", None)
        if not tmp_path:
            return

        filename = make_filename(details)
        output_path = str(_resolve_output_path(filename))

        # Mic speaker name for attribution (plumbed through, not yet active)
        mic_speaker = cfg.get("local_speaker_name") or None

        self._set_state(TRANSCRIBING, "Transcribing...")

        def run():
            try:
                def on_status(msg):
                    self._set_state(TRANSCRIBING, msg)

                transcribe_audio(
                    tmp_path,
                    output_path,
                    status_callback=on_status,
                    meeting_details=details,
                    mic_speaker=mic_speaker,
                    channels_info=channels_info,
                )
                self._set_state(IDLE, "Done \u2713")
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
                rumps.notification("Overheard", "Done", f"Saved: {filename}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = str(e)[:80]
                self._set_state(IDLE, f"\u2717 {msg}")
                rumps.notification("Overheard", "Error", str(e))
            finally:
                if cfg.get("keep_recordings"):
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
            IDLE:         "",
            RECORDING:    " \U0001f534",
            PAUSED:       " \u23f8",
            TRANSCRIBING: " \u23f3",
        }
        self.title = titles.get(state, "")
        if self._popover:
            self._popover.set_state(state, status)

    def _build_popover(self):
        """Build the panel and hook it to the status bar button."""
        from overheard.popover import TransportPopover
        self._popover = TransportPopover({
            "record":           self._on_record,
            "pause":            self._on_pause,
            "stop":             self._on_stop,
            "open_transcripts": self._reveal_transcripts,
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
        """Open Preferences, reporting a failure rather than dying on one.

        This is called straight from the gear button's ObjC action, through
        _PopoverDelegate.openPreferences_. An exception escaping an action
        callback aborts the process below Python's handlers, with no traceback
        and no crash report, which is how three separate bugs hid behind "the
        app vanishes" earlier on this branch. A window that fails to build has
        to be a message, not an exit.
        """
        from overheard.preferences import PreferencesWindow
        try:
            if self._prefs_window is None:
                self._prefs_window = PreferencesWindow()
            self._prefs_window.show()
        except Exception as e:
            print(f"[overheard] could not open Preferences: {e}", file=sys.stderr)
            traceback.print_exc()

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
        self._set_state(IDLE, "Ready")

    def _ensure_details_panel(self) -> "DetailsPanel":
        """Build the panel on first use and return it.

        Returning it, rather than only assigning it, is what lets a caller use
        the panel without the type checker having to trust that this ran.
        """
        if self._details_panel is None:
            from overheard.details_panel import DetailsPanel
            self._details_panel = DetailsPanel(
                callback=self._on_details_confirmed,
                discard_callback=self._on_discard,
            )
        return self._details_panel


def _ensure_utf8() -> None:
    """Make text files decode as UTF-8 however the app was launched.

    A bundle started from Finder inherits no LANG, so Python's preferred
    encoding falls back to ASCII, and every dependency that opens a text file
    without naming an encoding inherits that. parakeet_mlx reads the model's
    config.json with a bare open(); the file has non-ASCII bytes, so loading
    the model raised UnicodeDecodeError, and parakeet_mlx's own fallback then
    reported the failure as a missing model. The model was there all along.

    LSEnvironment in the bundle's plist sets PYTHONUTF8 for the Finder launch,
    which is the real fix because it applies before the interpreter starts.
    This is the second layer, for the paths LaunchServices does not cover, such
    as running the binary straight from a shell that has no LANG.
    """
    import locale

    if sys.flags.utf8_mode:
        return
    try:
        if "utf-8" not in locale.getpreferredencoding(False).lower():
            locale.setlocale(locale.LC_CTYPE, "UTF-8")
    except (locale.Error, ValueError) as e:
        print(f"could not force a UTF-8 locale: {e}", file=sys.stderr)


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


#: Kept alive for the process lifetime: faulthandler writes to this handle from
#: a signal handler, and a closed file there would lose the very backtrace we
#: are trying to capture.
_diagnostics_log = None


def _log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "Overheard" / "overheard.log"


def _enable_crash_diagnostics() -> None:
    """Leave a trace when the app dies, because a bundled .app cannot.

    Launched from Finder there is nowhere for stderr to go, so a crash and a
    clean quit look identical from the outside: the menu bar icon disappears
    and nothing is written down.

    Two gaps are closed here. First, stderr is teed to a log file, so the two
    dozen diagnostic prints scattered through the app survive. Second,
    faulthandler is registered for SIGTRAP, which it does not install by
    default: ``faulthandler.enable`` covers SIGSEGV, SIGABRT, SIGBUS, SIGFPE
    and SIGILL only. SIGTRAP is the signal Core Audio's watchdog raises when a
    realtime thread is starved, and it is what killed the app silently, so the
    one failure we most needed to see was the one nothing was watching for.

    ``chain=True`` so the process still dies as it would have. This records
    what happened; it does not pretend to survive it.
    """
    import faulthandler
    import signal

    global _diagnostics_log

    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _diagnostics_log = open(path, "a", buffering=1, encoding="utf-8")
    except OSError:
        # No log is survivable; refusing to start over it is not.
        faulthandler.enable()
        return

    _diagnostics_log.write(
        f"\n=== Overheard started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
    )

    class _Tee:
        """Write to both the real stderr and the log.

        Both, not just the log: running from a terminal during development
        should still print where the developer is looking.
        """

        def __init__(self, *streams):
            self._streams = [s for s in streams if s is not None]

        def write(self, data):
            for stream in self._streams:
                try:
                    stream.write(data)
                except (OSError, ValueError):
                    pass
            return len(data)

        def flush(self):
            for stream in self._streams:
                try:
                    stream.flush()
                except (OSError, ValueError):
                    pass

    sys.stderr = _Tee(sys.stderr, _diagnostics_log)

    faulthandler.enable(file=_diagnostics_log, all_threads=True)
    if hasattr(signal, "SIGTRAP"):
        try:
            faulthandler.register(
                signal.SIGTRAP, file=_diagnostics_log, all_threads=True, chain=True
            )
        except (OSError, ValueError, RuntimeError):
            pass


def main():
    _enable_crash_diagnostics()
    _ensure_utf8()
    _ensure_homebrew_path()
    _output_dir().mkdir(parents=True, exist_ok=True)

    # Diarization runs through the bundled helper and needs no credentials.
    app = TranscriberApp()

    # Build popover and hook it to the status bar button once the run loop starts
    def _init_popover(timer):
        timer.stop()
        app._build_popover()
        app._set_state(IDLE, "Ready")

    rumps.Timer(_init_popover, 0.5).start()

    app.run()


if __name__ == "__main__":
    main()
