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

import importlib
import queue
import sys
import textwrap
import threading

import numpy as np
import pytest

from overheard import live


class NoImportsAllowed:
    """Turns any first-time import into a failure for the duration of a block.

    Installed as a ``sys.meta_path`` finder rather than as a hook over
    ``builtins.__import__``. A finder is consulted only when ``sys.modules``
    misses, which is precisely the expensive case: the import that has to hit
    the filesystem and can stall the audio thread. An import of something
    already loaded never reaches the finder at all, so it stays free.

    The earlier version hooked ``builtins.__import__`` and tested the ``name``
    argument, which saw only some spellings of an import. Two got through:

    - ``from scipy import interpolate`` calls ``__import__("scipy",
      fromlist=("interpolate",))``. The name is ``scipy``, already warm since
      ``live.py`` imports ``scipy.signal`` at module scope, so the guard waved
      it through and the real import then loaded the cold submodule.
    - ``importlib.import_module`` never routes through ``builtins.__import__``
      at all, and numpy and scipy both use importlib for their lazy submodules,
      which is the mechanism that caused the original stall.

    A finder sees every spelling, so neither hole survives the change.
    """

    def __init__(self):
        self.attempted = []

    def find_spec(self, fullname, path=None, target=None):
        self.attempted.append(fullname)
        raise AssertionError(
            f"{fullname!r} was imported on the audio thread path. "
            "Import it at module scope instead: a first-time import "
            "here stalls capture and Core Audio kills the process."
        )

    def __enter__(self):
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self)
        return False


@pytest.fixture
def warm_package(tmp_path, monkeypatch):
    """A real package, imported, with a submodule deliberately left cold.

    This is the shape the guard has to catch and the shape it used to miss: the
    parent is in sys.modules, so anything keyed on the parent reads as loaded
    while the submodule import still goes to disk.

    Built on tmp_path rather than borrowing a stdlib module, so the test cannot
    be turned red by whatever else the session happened to import.
    """
    (tmp_path / "coldpkg").mkdir()
    (tmp_path / "coldpkg" / "__init__.py").write_text("")
    (tmp_path / "coldpkg" / "sub.py").write_text(
        textwrap.dedent("""
            # Stands in for the several hundred milliseconds scipy.signal spent
            # dragging in array_api_compat and numpy.f2py.
            VALUE = 1
        """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module("coldpkg")
    assert "coldpkg" in sys.modules, "the parent must be warm"
    assert "coldpkg.sub" not in sys.modules, "the submodule must be cold"

    yield "coldpkg"

    for name in [n for n in sys.modules if n == "coldpkg" or n.startswith("coldpkg.")]:
        del sys.modules[name]


def test_resampling_imports_nothing():
    """The exact call that killed the app, with imports made fatal.

    Asserts on ``attempted`` as well as relying on the raise. An import wrapped
    in try/except swallows the AssertionError, so the call returns cleanly
    while the disk hit and the stall still happen in production. That shape is
    not contrived: it is how an optional accelerated backend is normally
    probed, and it is what numpy and scipy do internally for the very lazy
    submodules this file blames for the SIGTRAP.
    """
    audio = np.zeros(4800, dtype=np.float32)
    with NoImportsAllowed() as guard:
        out = live._resample(audio, 48000)
    assert guard.attempted == [], f"first-time imports on the audio path: {guard.attempted}"
    assert len(out) == 1600, "48k to 16k should be a third of the samples"


def test_resampling_at_the_target_rate_imports_nothing():
    audio = np.zeros(1600, dtype=np.float32)
    with NoImportsAllowed() as guard:
        assert live._resample(audio, live.TARGET_RATE) is audio
    assert guard.attempted == [], f"first-time imports on the audio path: {guard.attempted}"


def test_downmixing_imports_nothing():
    """_to_mono runs on the same thread, immediately before _resample."""
    block = np.zeros((512, 2), dtype=np.float32)
    with NoImportsAllowed() as guard:
        live._to_mono(block, {"mic_channel": 0, "system_channels": [1]})
    assert guard.attempted == [], f"first-time imports on the audio path: {guard.attempted}"


def test_the_registered_tap_callback_imports_nothing():
    """The function actually wired to the recorder, not just its helpers.

    app.py does recorder.set_tap(live.feed), so LiveTranscriber.feed is what
    runs on the capture thread for every block. Its whole body sits inside
    `except Exception`, which catches AssertionError, so a guard that signals
    only by raising is invisible here: the import would happen, the stall would
    happen, and the test would pass. Asserting on attempted is the only signal
    that survives.

    The other tests in this file exercise _resample and _to_mono, which feed
    calls. This exercises feed itself, so a lazy import added anywhere along
    that chain, including in _record_energy or _enqueue, is caught.
    """
    transcriber = live.LiveTranscriber.__new__(live.LiveTranscriber)
    transcriber.sample_rate = 48000
    transcriber.channels_info = {"mic_channel": 0, "system_channels": [1]}
    transcriber._stop_event = threading.Event()
    transcriber._queue = queue.Queue(maxsize=8)
    transcriber._lock = threading.Lock()
    transcriber._pending = np.zeros(0, dtype=np.float32)
    transcriber._chunk_frames = 16000
    transcriber.diarizer = None
    transcriber._energy = []
    transcriber._t0 = 0.0

    block = np.zeros((4800, 2), dtype=np.float32)
    with NoImportsAllowed() as guard:
        transcriber.feed(block)

    assert guard.attempted == [], (
        "first-time imports on the registered tap callback: "
        f"{guard.attempted}. feed() swallows AssertionError, so this "
        "assertion is the only thing that can see them."
    )


def test_a_swallowed_import_is_still_recorded(warm_package):
    """The raise alone is not enough, so the recording has to be trustworthy.

    A hot-path function that catches its own ImportError would sail through a
    guard that signals only by raising. Recording every attempt is what lets
    the tests above catch that, so this pins the recording itself.
    """
    with NoImportsAllowed() as guard:
        try:
            importlib.import_module("coldpkg.sub")
        except Exception:
            pass
    assert guard.attempted == ["coldpkg.sub"], (
        "a swallowed violation must still be recorded, or the hot-path tests "
        "above cannot see it"
    )


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


def test_the_guard_itself_actually_fires(warm_package):
    """Prove the harness can fail before trusting the tests that rely on it.

    The first version of this guard did not fire at all, so every test using it
    passed while a real first-time import sat on the audio path. A gate nobody
    has watched fail is not a gate.
    """
    with pytest.raises(AssertionError, match="audio thread path"):
        with NoImportsAllowed():
            importlib.import_module("coldpkg.sub")


def test_the_guard_catches_a_from_import_of_a_cold_submodule(warm_package):
    """The spelling the previous guard waved through.

    ``from coldpkg import sub`` calls ``__import__("coldpkg", fromlist=("sub",))``
    with coldpkg already in sys.modules, so a guard keyed on the __import__
    name argument saw a warm module and allowed the cold submodule load behind
    it. This is the dominant import spelling in this codebase, so the hole was
    open on the exact form the next lazy import would most likely take.
    """
    with pytest.raises(AssertionError, match="coldpkg.sub"):
        with NoImportsAllowed():
            from coldpkg import sub  # noqa: F401


def test_the_guard_catches_importlib_import_module(warm_package):
    """The other spelling that got through, and the one that caused the stall.

    importlib.import_module goes through _bootstrap._find_and_load, never
    builtins.__import__, so a hook over __import__ could not see it. numpy and
    scipy both resolve their lazy submodules this way.
    """
    with pytest.raises(AssertionError, match="coldpkg.sub"):
        with NoImportsAllowed():
            importlib.import_module("coldpkg.sub")


def test_the_guard_allows_already_imported_modules(warm_package):
    """A warm import never reaches the finder, so it stays free.

    This is the other half of the contract. A guard that failed on every import
    would make the tests above pass for the wrong reason.
    """
    with NoImportsAllowed() as guard:
        import sys as _sys           # noqa: F401  already in sys.modules
        from coldpkg import __name__ as _n   # noqa: F401  parent is warm too
    assert guard.attempted == [], f"the finder was consulted for {guard.attempted}"
