"""Drive the real app through the record path and report what happened.

Run as a subprocess by test_gui_record_path.py, never imported by the suite.

It exists as a separate process for one reason: the failure this is written to
catch aborts the process. ``_LevelBar`` overrode ``init`` while ``_build``
constructed it with ``initWithFrame_``, so ``drawRect_`` raised AttributeError
inside AppKit's drawing machinery, where an unhandled Python exception is not an
exception at all but a SIGTRAP. Pressing Record unhid the system meter row, the
row drew for the first time, and Overheard vanished. In-process that would kill
pytest itself and report nothing useful; out here it is an exit code.

The second reason is subtler and cost this project several review rounds in
another form. rumps wraps every Timer callback in its own try/except
(rumps.py:997), so an assertion inside a timer is swallowed and the run finishes
green regardless. Nothing here asserts. It records what it observed, writes the
file, and lets the parent process do the judging.
"""

import json
import sys
import time
import traceback

RESULTS_PATH = sys.argv[1]
SRC = sys.argv[2]
sys.path.insert(0, SRC)

import rumps  # noqa: E402

from overheard import state  # noqa: E402
from overheard.app import TranscriberApp  # noqa: E402

results = {
    "reached": [],
    "popover_built": False,
    "state_after_record": None,
    "recorder_class": None,
    "recorder_is_multichannel": None,
    "sys_row_hidden_at_idle": None,
    "custom_views_drawn": [],
    "level_timer_running": False,
    "level_bars": None,
    "error": None,
}


def _write():
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def _draw_everything(app):
    """Force the views to draw, which is where the app used to die.

    Ordering the panel front does not guarantee a draw pass inside a short-lived
    test process, and "it did not crash" is worth nothing if nothing was ever
    rendered. displayIfNeeded drives the same drawRect_ path AppKit would.
    """
    from overheard import popover as popover_module

    panel = app._popover._panel
    panel.contentView().display()

    drawn = []

    def walk(view):
        if type(view).__module__ == popover_module.__name__:
            drawn.append(type(view).__name__)
        for sub in view.subviews() or []:
            walk(sub)

    walk(panel.contentView())
    return sorted(set(drawn))


def _initialisers_really_ran(app):
    """Did _LevelBar.initWithFrame_ actually run on every instance.

    vars() and not hasattr, and the distinction is the whole point. _LevelBar
    carries class-level _level and _active defaults as a second layer of
    defence, so hasattr is True even when no initialiser ran at all and the
    original bug is fully present. Only the instance dict tells them apart.

    This matters more than it looks. drawRect_ is wrapped in objc_safe now, so
    the original crash no longer aborts the process: it is caught, logged and
    turned into a no-op. That makes "the app survived" almost unfalsifiable
    here, and this is the observable that is left.
    """
    from overheard.popover import _LevelBar

    bars = []

    def walk(view):
        if isinstance(view, _LevelBar):
            bars.append(view)
        for sub in view.subviews() or []:
            walk(sub)

    walk(app._popover._panel.contentView())
    return {
        "count": len(bars),
        "with_instance_state": sum(
            1 for bar in bars if "_level" in vars(bar) and "_active" in vars(bar)
        ),
    }


class Driver:
    """One repeating timer, stepping on elapsed time.

    NOT one timer per step. ``rumps.Timer`` fires immediately on start and then
    repeats, so a timer set to quit the app after 40 seconds quits it at t=0.
    That trap has cost this project an hour once already. A single repeating
    timer that reads the clock itself cannot be fooled by it.
    """

    def __init__(self, app):
        self.app = app
        self.started = None
        self.done = set()

    def __call__(self, timer):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        elapsed = now - self.started

        try:
            self.step(elapsed)
        except Exception:
            results["error"] = traceback.format_exc()
            _write()
            rumps.quit_application()

    def step(self, elapsed):
        app = self.app

        if elapsed >= 0.0 and "build" not in self.done:
            self.done.add("build")
            # Exactly what main() does, including the order.
            app._build_popover()
            app._set_state(state.IDLE, "Ready")
            app._popover._show()
            results["popover_built"] = app._popover is not None
            results["sys_row_hidden_at_idle"] = bool(app._popover._sys_row.isHidden())
            # Read BEFORE Record is pressed. setLevel_ and setActive_ both
            # assign to the instance, and the level timer runs throughout a
            # recording, so after that point the instance dict is populated
            # whether or not the initialiser ever ran. Observed here, only
            # initWithFrame_ can have put anything there.
            results["level_bars"] = _initialisers_really_ran(app)
            results["reached"].append("built")

        elif elapsed >= 1.0 and "record" not in self.done:
            self.done.add("record")
            # The real entry point. _on_record defers the CoreAudio work to a
            # 0.05s timer and then polls a worker thread, so nothing is ready
            # yet when this returns; that is why the check waits.
            app._on_record()
            results["reached"].append("pressed record")

        elif elapsed >= 5.0 and "observe" not in self.done:
            self.done.add("observe")
            results["state_after_record"] = app._state
            recorder = app._recorder
            results["recorder_class"] = type(recorder).__name__ if recorder else None
            if recorder is not None:
                results["recorder_is_multichannel"] = bool(recorder.is_multichannel)
            results["level_timer_running"] = app._level_timer is not None
            results["custom_views_drawn"] = _draw_everything(app)
            results["reached"].append("observed")

        elif elapsed >= 7.0 and "stop" not in self.done:
            self.done.add("stop")
            if app._recorder is not None:
                app._recorder.stop()
            results["reached"].append("stopped")

        elif elapsed >= 8.5 and "quit" not in self.done:
            self.done.add("quit")
            results["reached"].append("quit")
            _write()
            rumps.quit_application()


def main():
    _write()  # so an abort is distinguishable from never having started
    app = TranscriberApp()
    rumps.Timer(Driver(app), 0.25).start()
    app.run()


main()
