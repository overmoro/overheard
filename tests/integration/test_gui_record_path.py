"""Something presses Record.

Nothing in this repo did, for the whole of Phases 1 and 2. That gap is not
incidental: it is the reason a single unit spent ten adversarial review rounds
finding the same class of defect over and over.

``popover.py``, ``preferences.py``, ``details_panel.py`` and ``live_panel.py``
are invisible to both mechanical checks. They subclass NSObject and NSView,
which PyObjC leaves untyped, so every attribute on them reads as valid however
it is spelled and mypy cannot object. They are also excluded from the coverage
denominator. Nothing contradicts a test that passes for the wrong reason there,
and hand-built fixtures kept producing exactly that: three tests that pinned a
regression by calling a method on a thread production never uses, a test named
for the live diarizer that substituted a stub whose body was one append, a
derived gate walking a hand-written list while claiming to derive it.

All three original crashes lived in those four modules, and every one of them
was found by running the app rather than by a test.

So this drives the real thing: the real TranscriberApp, the real popover, the
real ``_on_record``, real Core Audio. It is gated on hardware and the built
helper because it genuinely needs them, and it runs the app in a subprocess
because the failure it is looking for aborts the process.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ..conftest import requires_helper

pytestmark = [pytest.mark.integration, requires_helper]

DRIVER = Path(__file__).parent / "gui_record_driver.py"
SRC = str(Path(__file__).resolve().parents[2] / "src")

# Generous, and deliberately so. The driver's own script finishes at 8.5s, and
# the first tap start on a machine can block on the system audio permission
# dialog. A timeout here is a failure with a readable message rather than a
# hang, which is the only thing worse than a crash.
TIMEOUT = 90


def _capture_available():
    sys.path.insert(0, SRC)
    from overheard import capture

    return capture.is_available()


requires_capture = pytest.mark.skipif(
    not _capture_available()[0],
    reason=f"Core Audio taps unavailable: {_capture_available()[1]}",
)


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    """Drive the app once and share the outcome across the assertions below.

    One run, several assertions, because starting Core Audio taps six times
    costs six permission handshakes and about a minute. Module-scoped so each
    assertion still gets to name its own failure.
    """
    results_path = tmp_path_factory.mktemp("gui") / "results.json"
    completed = subprocess.run(
        [sys.executable, str(DRIVER), str(results_path), SRC],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    return completed, results


@requires_capture
class TestTheAppSurvivesPressingRecord:
    def test_the_process_did_not_abort(self, run_result):
        """The assertion the other three depend on, and the original bug.

        An unhandled Python exception inside an AppKit callback does not raise,
        it aborts, so a returncode of -5 (SIGTRAP) or -6 here is the crash this
        file exists to catch. stderr is included because the app tees its own
        diagnostics there and a bare exit code diagnoses nothing.
        """
        completed, results = run_result
        assert completed.returncode == 0, (
            f"the app died with returncode {completed.returncode}.\n"
            f"driver error: {results.get('error')}\n"
            f"stages reached: {results.get('reached')}\n"
            f"stderr:\n{completed.stderr[-3000:]}"
        )
        assert results.get("error") is None, results.get("error")

    def test_it_got_all_the_way_through(self, run_result):
        """Exit code 0 alone would also be satisfied by doing nothing at all."""
        _completed, results = run_result
        assert results.get("reached") == [
            "built", "pressed record", "observed", "stopped", "quit"
        ], f"the driver did not complete its script: {results.get('reached')}"

    def test_a_recorder_is_attached_and_the_app_is_recording(self, run_result):
        """What pressing Record is supposed to achieve.

        Checked after the deferred start and the poll timer have had time to
        run: _on_record hands the CoreAudio work to a 0.05s timer and a worker
        thread, so nothing is attached at the moment it returns.
        """
        _completed, results = run_result
        assert results.get("state_after_record") == "recording", (
            f"the app never reached RECORDING, it is in "
            f"{results.get('state_after_record')!r}"
        )
        assert results.get("recorder_class") == "TapRecorder", (
            f"expected Core Audio taps, got {results.get('recorder_class')!r}"
        )
        assert results.get("level_timer_running") is True, (
            "the level timer never started, so _poll_recorder_started did not "
            "reach its last two lines"
        )

    def test_every_custom_view_actually_drew(self, run_result):
        """The crash was in drawRect_, so a run that never drew proves nothing.

        _LevelBar is named explicitly rather than counted. The popover holds
        three _PillButtons, so a floor of four is satisfied with both level bars
        absent, and _LevelBar is the class the original crash lived in.
        """
        _completed, results = run_result
        drawn = results.get("custom_views_drawn") or []
        assert "_LevelBar" in drawn, (
            f"the level bars never drew, which is where the app used to die: {drawn}"
        )
        assert "_PillButton" in drawn, f"the buttons never drew: {drawn}"

    def test_the_level_bar_initialisers_really_ran(self, run_result):
        """The original crash, in the only form it can still take.

        drawRect_ is wrapped in objc_safe now, so an AttributeError there is
        caught, logged and returned as None instead of aborting. That is the
        right behaviour and it also means the process surviving proves very
        little: the guard delivers that unconditionally. Reintroducing the
        original bug, an init override on a view built with initWithFrame_,
        left every other assertion in this file green.

        _LevelBar also carries class-level defaults, so hasattr cannot tell an
        initialised instance from an uninitialised one. The instance dict can.
        """
        _completed, results = run_result
        bars = results.get("level_bars") or {}
        assert bars.get("count"), "no _LevelBar was found in the built popover"
        assert bars["with_instance_state"] == bars["count"], (
            f"only {bars['with_instance_state']} of {bars['count']} level bars "
            "ran their initialiser. The rest are relying on the class-level "
            "defaults, which is the original SIGTRAP bug with a net under it"
        )

    def test_no_objc_guard_fired_during_the_run(self, run_result):
        """A guard that fires is a bug that has been made invisible.

        objc_safety turns an abort into a logged no-op across 38 first-party
        methods, which is what makes the app survivable and also what makes a
        latent fault silent. The log line it prints is the only evidence, so
        this run treats any of them as a failure.

        Matched on the guard's own message rather than on a list of method
        names: a new guarded method is covered the day it is added.
        """
        completed, _results = run_result
        fired = [
            line for line in completed.stderr.splitlines()
            if line.startswith("[overheard]") and " failed: " in line
        ]
        assert not fired, (
            "an ObjC guard caught something during a clean record cycle, so a "
            "real fault is being swallowed:\n" + "\n".join(fired)
        )

    def test_the_system_meter_row_is_hidden_while_idle(self, run_result):
        """Verified in the real startup order, not a synthesised one.

        The unit tests for this call configure_channels first, which is the one
        ordering the app never has at launch. Here the popover is built and set
        to IDLE exactly as main() does it.
        """
        _completed, results = run_result
        assert results.get("sys_row_hidden_at_idle") is True, (
            "an idle popover showed a live system-audio meter row"
        )
