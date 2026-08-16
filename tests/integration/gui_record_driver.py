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
    "panel_visible": None,
    "status_item_hooked": None,
    "state_transitions": [],
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
        self.states = []
        self._latch_state_changes()

    def _latch_state_changes(self):
        """Record every state transition, because production consumes the error.

        _poll_recorder_started reads _pending_recorder_error and immediately
        nulls it (app.py:186-187) on a 0.15s timer, while this driver polls on
        a 0.25s one. Both are NSTimers on the same run loop, so the faster one
        usually wins and the error is gone before it can be read: a machine
        whose tap start genuinely fails would spin for the full RECORDER_WAIT
        and report "still pending" with no error, and the environment skip
        built on that field would never fire.

        A failed start still transitions to IDLE with the reason as its status
        text, and nothing clears that, so it is latched here instead.
        """
        app = self.app
        original = app._set_state

        def recording_set_state(new_state, status=""):
            self.states.append({"state": new_state, "status": status})
            return original(new_state, status)

        app._set_state = recording_set_state

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
        # Leave nothing running and nothing behind. An abandoned tap holds a
        # Core Audio device; an abandoned helper holds a second process; and
        # _on_stop writes the recording to a NamedTemporaryFile(delete=False)
        # that only _on_details_confirmed or _on_discard removes, neither of
        # which this run reaches.
        #
        # _pending_recorder as well as _recorder: on the RECORDER_WAIT path the
        # start is still in flight, so _recorder is None while a helper
        # subprocess is already spawned and blocked on the permission dialog,
        # where it will never take EPIPE from a dead parent.
        app = self.app
        for attr in ("_recorder", "_pending_recorder"):
            recorder = getattr(app, attr, None)
            if recorder is not None:
                try:
                    recorder.stop()
                except Exception:
                    pass

        pending_wav = getattr(app, "_pending_wav", None)
        if pending_wav:
            try:
                os.unlink(pending_wav)
            except OSError:
                pass

        _write()
        rumps.quit_application()

    def _maybe_abort(self, step_name):
        """Abort the way a real crash does, when asked to, for one test.

        The per-step writes only pay off on an abort, so on any successful run
        deleting them changes nothing observable and no mutation can catch
        their loss. That is an untested safety net, which is the same shape as
        the defects this whole unit exists to stop shipping. os.abort() raises
        SIGABRT with no unwinding and no chance to flush, which is as close to
        the SIGTRAP case as can be arranged deliberately.
        """
        if os.environ.get("OVERHEARD_DRIVER_ABORT_AFTER") == step_name:
            os.abort()

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
            # _show() returns silently when _status_btn is None, and
            # _build_popover swallows a failing hook_status_item with a print.
            # Without these, a run where no UI ever appeared draws nothing, so
            # no guard can fire and every assertion here passes while a real
            # user sees an app with no window at all.
            results["panel_visible"] = bool(app._popover._panel.isVisible())
            results["status_item_hooked"] = app._popover._status_btn is not None
            button = app._popover._btn_record
            results["record_button_enabled"] = bool(button._enabled)
            results["record_button_has_callback"] = button._callback is not None
            results["reached"].append("built")
            _write()
            self._maybe_abort("built")

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
            self._maybe_abort("pressed record")

        elif "record" in self.done and "observe" not in self.done:
            waited = now - self.pressed_at
            started = app._recorder is not None
            # A failed start goes back to IDLE with the reason as its status.
            # Read from the latch rather than from _pending_recorder_error,
            # which production clears on a faster timer than this one.
            back_to_idle = [
                entry for entry in self.states[1:]
                if entry["state"] == state.IDLE
            ]
            failed = bool(back_to_idle) and not started
            if not (started or failed or waited >= self.RECORDER_WAIT):
                return

            self.done.add("observe")
            results["terminal_state"] = (
                "started" if started else "failed" if failed else "still pending"
            )
            results["recorder_start_error"] = (
                back_to_idle[-1]["status"] if back_to_idle else None
            )
            results["state_transitions"] = self.states
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
