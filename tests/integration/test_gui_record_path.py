"""Something presses Record.

Nothing in this repo did, through all of Phases 1 and 2. That gap is why one
unit spent ten adversarial review rounds finding the same class of defect in a
new place each time.

popover.py, preferences.py, details_panel.py and live_panel.py are invisible to
both mechanical checks. They subclass NSObject and NSView, which PyObjC leaves
untyped, so every attribute reads as valid however it is spelled and mypy cannot
object, and they are excluded from the coverage denominator too. Nothing
contradicts a test that passes for the wrong reason there. All three original
crashes lived in those modules and every one was found by running the app.

Scope is deliberately narrow: only what a subprocess running the real app can
prove and a unit test cannot. Whether a view can draw, whether an initialiser
assigns its attributes, whether a selector exists, are all covered in
tests/unit/ and are not restated here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ..conftest import requires_helper

pytestmark = [pytest.mark.integration, requires_helper]

DRIVER = Path(__file__).parent / "gui_record_driver.py"
SRC = str(Path(__file__).resolve().parents[2] / "src")

# Generous. The driver waits up to 45s for the recorder alone, because a first
# tap start can block on the system audio permission dialog. A timeout here is
# a failure with a readable message rather than a hang.
TIMEOUT = 120

# Start failures that mean "this machine cannot run this test", as opposed to
# "the code is broken". Matched on substrings of the error the app itself
# reports, so the list is about the environment and not about spelling.
ENVIRONMENT_FAILURES = ("permission", "not authorized", "no input", "no device")


def _capture_available():
    from overheard import capture

    return capture.is_available()


# Evaluated once. Calling this twice, which is what happens when the same call
# is used for the condition and again inside the reason string, probes Core
# Audio twice on every collection of this file, skip or no skip.
_CAPTURE_OK, _CAPTURE_REASON = _capture_available()

requires_capture = pytest.mark.skipif(
    not _CAPTURE_OK,
    reason=f"Core Audio taps unavailable: {_CAPTURE_REASON}",
)


def _drive(work, env=None):
    """Run the driver once against an isolated config, returning (proc, results)."""
    results_path = work / "results.json"
    config_dir = work / "config"
    config_dir.mkdir(exist_ok=True)

    completed = subprocess.run(
        [sys.executable, str(DRIVER), str(results_path), SRC, str(config_dir)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, **(env or {})},
    )

    raw = results_path.read_text() if results_path.exists() else ""
    try:
        results = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        # Never mask a crash with a parse error. The driver writes atomically,
        # so this should not happen; if it does, the raw text is the evidence.
        results = {"error": f"results file was not valid JSON: {raw[:2000]!r}"}
    return completed, results


@requires_capture
def test_an_abort_still_reports_how_far_the_run_got(tmp_path):
    """The diagnostics this whole file leans on when everything else has failed.

    Writing results after every step only pays off on an abort, so on a healthy
    run the final write covers everything and deleting the intermediate ones
    changes nothing any assertion can see. That is an untested safety net, and
    shipping one of those is the mistake this unit exists to stop making.

    So one run is told to abort deliberately, right after pressing Record.
    os.abort() gives no unwinding and no flush, which is as close to the real
    SIGTRAP as can be arranged on purpose. What must survive is the file saying
    how far it got: a crash report of "stages reached: []" for a run that got
    two steps in is worse than useless when the next crash is a real one.
    """
    completed, results = _drive(
        tmp_path, env={"OVERHEARD_DRIVER_ABORT_AFTER": "pressed record"}
    )

    assert completed.returncode != 0, "the driver was asked to abort and did not"
    assert results.get("reached") == ["built", "pressed record"], (
        "the crash report lost the stages the run actually completed, which is "
        f"the only diagnosis an abort leaves: {results.get('reached')}"
    )
    assert results.get("record_button_has_callback") is True, (
        "observations recorded before the abort were lost with it"
    )


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    """Drive the app once and share the outcome.

    One run, several assertions, because starting Core Audio taps six times
    costs six handshakes and about a minute. Module-scoped so each assertion
    still names its own failure.
    """
    completed, results = _drive(tmp_path_factory.mktemp("gui"))

    if results.get("terminal_state") == "failed":
        detail = str(results.get("recorder_start_error") or "").lower()
        if any(needle in detail for needle in ENVIRONMENT_FAILURES):
            pytest.skip(f"this machine cannot start capture: {detail}")

    return completed, results


@requires_capture
class TestTheAppSurvivesPressingRecord:
    def test_the_process_did_not_abort(self, run_result):
        """An unhandled exception in an AppKit callback aborts rather than raises.

        A returncode of -5 (SIGTRAP) or -6 is the crash this file exists to
        catch. stderr is included because the app tees its own diagnostics
        there and a bare exit code diagnoses nothing.

        On its own this proves less than it looks: objc_safety guards 38
        first-party methods, so survival is close to unconditional. The guard
        assertion below is what carries the weight.
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
        """Exit code 0 is also satisfied by doing nothing at all."""
        _completed, results = run_result
        assert results.get("reached") == [
            "built", "pressed record", "observed", "stopped", "quit"
        ], f"the driver did not complete its script: {results.get('reached')}"

    def test_the_record_button_is_wired_to_something(self, run_result):
        """The seam this whole file exists to cover, and the one it used to skip.

        The driver called app._on_record() directly at first, which goes around
        callbacks.get("record") in _build_popover and around the _enabled gate
        in mouseDown_. Misspelling that key leaves a Record button that does
        nothing whatsoever for a real user, and every assertion here stayed
        green.
        """
        _completed, results = run_result
        assert results.get("record_button_has_callback") is True, (
            "the Record button has no callback, so pressing it does nothing"
        )
        assert results.get("record_button_enabled") is True, (
            "the Record button is disabled while the app is idle"
        )

    def test_pressing_it_starts_a_real_recorder(self, run_result):
        """What only this test can show: the button reaches Core Audio.

        Reached by pressing the control, so it covers the wiring, the _enabled
        gate, the deferred start and the poll timer as one path.
        """
        _completed, results = run_result
        assert results.get("terminal_state") == "started", (
            f"capture never started: {results.get('terminal_state')}, "
            f"error {results.get('recorder_start_error')!r}"
        )
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
        assert results.get("stop_button_enabled") is True, (
            "Stop is disabled while recording, so the user cannot stop"
        )

    def test_pressing_stop_returns_the_app_to_idle(self, run_result):
        """The other half of the transport, also through the real control."""
        _completed, results = run_result
        assert results.get("state_after_stop") != "recording", (
            "the app was still recording after Stop was pressed"
        )

    def test_no_objc_guard_fired_during_the_run(self, run_result):
        """A guard that fires is a bug that has been made invisible.

        objc_safety turns an abort into a logged no-op across 38 first-party
        methods, which is what makes the app survivable and also what makes a
        latent fault silent. Its log line is the only evidence, so any of them
        during a clean record cycle is a failure here.

        Matched on the guard's own message rather than a list of method names,
        so a newly guarded method is covered the day it is added. This is the
        assertion that catches the original _LevelBar crash: reintroducing it
        leaves the exit code at 0 and fails here.
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

    def test_the_level_bars_ran_their_own_initialisers(self, run_result):
        """The original crash, in the only form it can still take.

        drawRect_ is wrapped in objc_safe now, so an AttributeError there is
        caught and logged rather than aborting. _LevelBar also carries
        class-level defaults, so hasattr cannot tell an initialised instance
        from an uninitialised one. The instance dict can, and is read before
        _set_state, whose setters would otherwise populate it either way.
        """
        _completed, results = run_result
        bars = results.get("level_bars") or {}
        assert set(bars) == {"mic", "sys"}, f"expected both bars, got {bars}"
        for role, keys in bars.items():
            assert keys == ["_active", "_level"], (
                f"the {role} level bar did not run initWithFrame_: its instance "
                f"holds {keys}. It is falling back to the class-level defaults, "
                "which is the original SIGTRAP bug with a net under it"
            )

    def test_the_system_meter_row_is_hidden_while_idle(self, run_result):
        """In the real startup order, which the unit tests cannot use.

        The unit test for this calls configure_channels first, which is the one
        ordering the app never has at launch. Here the popover is built and set
        to IDLE exactly as main() does it.
        """
        _completed, results = run_result
        assert results.get("sys_row_hidden_at_idle") is True, (
            "an idle popover showed a live system-audio meter row"
        )
