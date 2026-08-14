"""Nothing on the audio tap path may import at call time.

The capture reader thread feeds every block straight into the live
transcriber. It has a hard realtime obligation: Core Audio watches whether
audio keeps moving, and kills the process outright when it stops.

A lazy import there is therefore not a style question. ``_resample`` imported
scipy.signal on first use, which pulls array_api_compat, which runs
``from numpy import *``, which drags in numpy.f2py. Several hundred
milliseconds holding the GIL, on the thread that must never stall, and the
watchdog fired SIGTRAP. Pressing Record made the whole app disappear with no
crash report and no output.

These tests fail on the mechanism rather than on the one import that caused
it, so a new lazy import anywhere on the hot path is caught the same way.
"""

import builtins
import sys

import numpy as np
import pytest

from overheard import live


class NoImportsAllowed:
    """Turns any import attempt into a failure for the duration of a block.

    Modules already in sys.modules still resolve, since re-importing those is
    a dict lookup and costs nothing. It is the first-time import that stalls.
    """

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch
        self.attempted = []

    def __enter__(self):
        real_import = builtins.__import__

        def guard(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if name not in sys.modules and root not in sys.modules:
                self.attempted.append(name)
                raise AssertionError(
                    f"{name!r} was imported on the audio thread path. "
                    "Import it at module scope instead: a first-time import "
                    "here stalls capture and Core Audio kills the process."
                )
            return real_import(name, globals, locals, fromlist, level)

        self._monkeypatch.setattr(builtins, "__import__", guard)
        return self

    def __exit__(self, *exc):
        return False


def test_resampling_imports_nothing(monkeypatch):
    """The exact call that killed the app, with imports made fatal."""
    audio = np.zeros(4800, dtype=np.float32)
    with NoImportsAllowed(monkeypatch):
        out = live._resample(audio, 48000)
    assert len(out) == 1600, "48k to 16k should be a third of the samples"


def test_resampling_at_the_target_rate_imports_nothing(monkeypatch):
    audio = np.zeros(1600, dtype=np.float32)
    with NoImportsAllowed(monkeypatch):
        assert live._resample(audio, live.TARGET_RATE) is audio


def test_downmixing_imports_nothing(monkeypatch):
    """_to_mono runs on the same thread, immediately before _resample."""
    block = np.zeros((512, 2), dtype=np.float32)
    with NoImportsAllowed(monkeypatch):
        live._to_mono(block, {"mic_channel": 0, "system_channels": [1]})


def test_the_resampler_is_bound_at_module_scope():
    """Pins the fix itself, so moving it back inside the function fails here."""
    assert hasattr(live, "resample_poly")
    assert "scipy.signal" in sys.modules, "scipy must load when live.py does"


@pytest.mark.parametrize("src_rate", [48000, 44100, 32000])
def test_resampling_still_produces_the_right_rate(src_rate):
    """The fix must not change what _resample actually does."""
    seconds = 0.5
    audio = np.zeros(int(src_rate * seconds), dtype=np.float32)
    out = live._resample(audio, src_rate)
    assert abs(len(out) - int(live.TARGET_RATE * seconds)) <= 2
