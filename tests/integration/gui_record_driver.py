"""Drive the real app through the record path and report what happened.

Run as a subprocess by test_gui_record_path.py, never imported by the suite.

Separate process for two reasons. The failure class here aborts rather than
raises, and in-process that kills pytest and reports nothing. And rumps wraps
every Timer callback in its own try/except (rumps.py:997), so an assertion
inside a timer is swallowed and the run finishes green regardless. Nothing here
asserts. It records what it saw, writes after every step, and lets the parent do
the judging.

What this deliberately does NOT claim: that AppKit drew anything. A Python-level
wrapper on drawRect_ intercepts a direct Python call but not AppKit's own
dispatch, which resolves the selector through the ObjC runtime, so there is no
honest way to observe a real draw from here. An earlier version walked the view
tree collecting class names and called that "drew", which the views satisfied by
merely existing: deleting the display() call left it green.
tests/unit/test_view_initialisers.py already proves every custom view can draw.
What this file adds is the real run loop, the real button and real Core Audio,
and a drawRect_ that fails under all three shows up as a fired guard.
"""

import json
import os
import sys
import tempfile
import time
import traceback

results = {
    "reached": [],
    "level_bars": {},
    "sys_row_hidden_at_idle": None,
    "record_button_enabled": None,
    "record_button_has_callback": None,
    "state_after_record": None,
    "recorder_class": None,
    "recorder_is_multichannel": None,
    "recorder_start_error": None,
    "terminal_state": None,
    "level_timer_running": False,
    "stop_button_enabled": None,
    "state_after_stop": None,
    "error": None,
}

RESULTS_PATH = None


def _write():
    """Atomically, and after every step.

    After each step rather than only at quit, because the abort this file exists
    to catch leaves whatever is on disk as the entire diagnosis. An earlier
    version wrote only at the end, so a run that died drawing reported "stages
    reached: []" and looked identical to one that never started.

    Atomic because an abort landing inside json.dump would otherwise leave
    truncated JSON, and the parent's json.loads then raises instead of
    reporting the crash it was called to report.
    """
    directory = os.path.dirname(RESULTS_PATH)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, RESULTS_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _level_bar_state(app):
    """Whether each level bar ran its own initialiser, reported by role.

    vars() and not hasattr: _LevelBar carries class-level _level and _active
    defaults as a second layer of defence, so hasattr is True even when no
    initialiser ran and the original SIGTRAP bug is fully present. Only the
    instance dict separates the two.

    Read BEFORE _set_state, and by role rather than as a count. _set_state
    reaches _set_meters_visible, which calls setActive_ on both bars and
    setLevel_ on the mic bar, and every one of those writes to the instance
    dict. An earlier version read after _set_state while claiming in a comment
    that only initWithFrame_ could have run: the mic bar then passed on the
    strength of those setters and only the sys bar still failed, so the check
    survived at half strength with its own comment denying it.
    """
    popover = app._popover
    return {
        role: sorted(
            key for key in ("_level", "_active") if key in vars(getattr(popover, attr))
        )
        for role, attr in (("mic", "_mic_bar"), ("sys", "_sys_bar"))
    }


class Driver:
    """One repeating timer, stepping on elapsed time.

    NOT one timer per step. ``rumps.Timer`` fires immediately on start and then
    repeats, so a quit timer set to 40 seconds quits the app at t=0. That has
    cost this project an hour once already. A single repeating timer that reads
    the clock itself cannot be fooled by it.
    """

    # Waiting for the recorder is condition-driven rather than a fixed sleep.
    # The first tap start on a machine can block on the system audio permission
    # dialog, and a fixed deadline turns that into "expected TapRecorder, got
    # None", which diagnoses nothing. This cap only bounds the hang.
    RECORDER_WAIT = 45.0

    def __init__(self, app):
        self.app = app
        self.started = None
        self.done = set()
        self.pressed_at = None

    def __call__(self, timer):
        import rumps

        now = time.monotonic()
        if self.started is None:
            self.started = now

        try:
            self.step(now - self.started, now)
        except Exception:
            results["error"] = traceback.format_exc()
            self._quit(rumps)

    def _quit(self, rumps):
        # Leave nothing running: an abandoned tap holds a Core Audio device and
        # an abandoned live transcriber holds a second helper process.
        try:
            if self.app._recorder is not None:
                self.app._recorder.stop()
        except Exception:
            pass
        _write()
        rumps.quit_application()

    def step(self, elapsed, now):
        import rumps

        from overheard import state

        app = self.app

        if "build" not in self.done:
            self.done.add("build")
            app._build_popover()
            # Before _set_state, which writes to both bars through its setters.
            results["level_bars"] = _level_bar_state(app)
            app._set_state(state.IDLE, "Ready")
            app._popover._show()
            results["sys_row_hidden_at_idle"] = bool(app._popover._sys_row.isHidden())
            button = app._popover._btn_record
            results["record_button_enabled"] = bool(button._enabled)
            results["record_button_has_callback"] = button._callback is not None
            results["reached"].append("built")
            _write()

        elif elapsed >= 1.0 and "record" not in self.done:
            self.done.add("record")
            self.pressed_at = now
            # The real control, not app._on_record(). Calling the handler
            # directly skips callbacks.get("record") in _build_popover and the
            # _enabled gate in mouseDown_, so a Record button wired to nothing
            # at all passed every assertion in this file.
            app._popover._btn_record.mouseDown_(None)
            results["reached"].append("pressed record")
            _write()

        elif "record" in self.done and "observe" not in self.done:
            waited = now - self.pressed_at
            started = app._recorder is not None
            failed = app._pending_recorder_error is not None
            if not (started or failed or waited >= self.RECORDER_WAIT):
                return

            self.done.add("observe")
            results["terminal_state"] = (
                "started" if started else "failed" if failed else "still pending"
            )
            results["recorder_start_error"] = app._pending_recorder_error
            results["state_after_record"] = app._state
            recorder = app._recorder
            results["recorder_class"] = type(recorder).__name__ if recorder else None
            if recorder is not None:
                results["recorder_is_multichannel"] = bool(recorder.is_multichannel)
            results["level_timer_running"] = app._level_timer is not None
            results["stop_button_enabled"] = bool(app._popover._btn_stop._enabled)
            results["reached"].append("observed")
            _write()

        elif "observe" in self.done and "stop" not in self.done:
            self.done.add("stop")
            # The real Stop control too, and for the same reason. This reaches
            # _on_stop, where the recorder is torn down.
            app._popover._btn_stop.mouseDown_(None)
            results["state_after_stop"] = app._state
            results["reached"].append("stopped")
            _write()

        elif "stop" in self.done and "quit" not in self.done:
            self.done.add("quit")
            results["reached"].append("quit")
            self._quit(rumps)


def main():
    global RESULTS_PATH

    RESULTS_PATH = sys.argv[1]
    src = sys.argv[2]
    config_dir = sys.argv[3]
    sys.path.insert(0, src)

    # Config isolation, before anything reads it. Without this the run is
    # decided by the developer's own settings: capture_backend chooses the
    # backend this test asserts on, and live_preview defaults True, which starts
    # a Parakeet load, spawns a second helper for diarize-stream and puts a live
    # transcript window on screen mid-test, none of it asserted and none of it
    # torn down.
    from pathlib import Path

    from overheard import settings

    settings.CONFIG_DIR = Path(config_dir)
    settings.CONFIG_PATH = Path(config_dir) / "config.json"
    settings.CONFIG_PATH.write_text(json.dumps({
        "capture_backend": "tap",
        "live_preview": False,
        "speaker_memory": False,
    }))

    import rumps

    from overheard.app import TranscriberApp

    _write()  # so an abort before the first step is distinguishable
    app = TranscriberApp()
    rumps.Timer(Driver(app), 0.25).start()
    app.run()


if __name__ == "__main__":
    main()
