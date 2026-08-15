"""A panel that failed to build must say so, not die on an attribute of None.

Both lazily-built panels assign ``_window`` early in ``_build`` and their
delegate several lines later. Anything raising in between left a window in
place with a None delegate, and because the panel was keyed on the window, it
considered itself built forever after. Every later ``show()`` then dereferenced
None inside an ObjC action callback, where an unhandled Python exception aborts
the process with no traceback and no crash report.

These tests pin the guarantee rather than the wiring: if a build does not
produce the objects the panel needs, the caller gets a RuntimeError naming the
problem, at the point of failure.
"""

import pytest

from overheard.details_panel import DetailsPanel
from overheard.preferences import PreferencesWindow


def _build_that_produces_nothing(panel):
    """Stand in for _build raising after assigning _window.

    Patched over _build so the test never opens a real window: what matters is
    the state _build leaves behind, not how it got there.
    """
    panel._window = object()


class _Partial:
    """A _build that gets as far as the delegate and then dies.

    This is the shape that matters and the one the first version of these tests
    missed. Both _build methods assign _window and their delegate within the
    first handful of lines and then run for another hundred and more, so a
    failure anywhere in that long tail leaves the early sentinels set. Keying
    "is it built?" on one of them means the panel considers itself finished,
    permanently, and every later show() dies on a widget the build never
    reached.

    Patching _build to produce *nothing* never enters that window, which is why
    reverting the fix left the old tests green.
    """

    def __init__(self, panel, *, attrs):
        self.panel = panel
        self.attrs = attrs
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.panel._window = object()
        for name in self.attrs:
            # Assigned early by the real _build, long before it finishes.
            setattr(self.panel, name, type("HalfBuilt", (), {})())
        raise RuntimeError("AppKit failed partway through the build")


class TestPreferencesWindow:
    def test_a_build_that_produces_no_delegate_raises(self, monkeypatch):
        window = PreferencesWindow()
        monkeypatch.setattr(
            window, "_build", lambda: _build_that_produces_nothing(window)
        )
        with pytest.raises(RuntimeError, match="failed to build"):
            window._ensure_built()

    def test_the_window_alone_is_not_treated_as_built(self, monkeypatch):
        """Keying on _window is what made the broken state permanent.

        A window with no delegate has to count as not built, otherwise the
        panel never tries again and never reports why.
        """
        window = PreferencesWindow()
        window._window = object()
        rebuilt = []
        monkeypatch.setattr(window, "_build", lambda: rebuilt.append(True))
        with pytest.raises(RuntimeError):
            window._ensure_built()
        assert rebuilt, "a window with no delegate must still attempt a build"


class TestAPartialBuildIsNotABuild:
    """The long tail of _build, which the sentinel has to survive."""

    def test_preferences_retries_after_a_build_that_died_partway(self, monkeypatch):
        window = PreferencesWindow()
        partial = _Partial(window, attrs=["_delegate"])
        monkeypatch.setattr(window, "_build", partial)

        with pytest.raises(RuntimeError):
            window._ensure_built()
        with pytest.raises(RuntimeError):
            window._ensure_built()

        assert partial.calls == 2, (
            "a panel whose build died partway must try again, not hand back the "
            "half-populated delegate it managed to assign first"
        )

    def test_details_panel_retries_after_a_build_that_died_partway(self, monkeypatch):
        panel = DetailsPanel(callback=None)
        partial = _Partial(panel, attrs=["_delegate", "_data_source"])
        monkeypatch.setattr(panel, "_build", partial)

        with pytest.raises(RuntimeError):
            panel._ensure_built()
        with pytest.raises(RuntimeError):
            panel._ensure_built()

        assert partial.calls == 2, (
            "the second call handed back a delegate with none of the widgets "
            "show() reads, which is an AttributeError inside AppKit"
        )

    def test_a_finished_build_is_not_repeated(self, monkeypatch):
        """The sentinel has to mean built, or every open rebuilds the window."""
        window = PreferencesWindow()
        calls = []

        def complete_build():
            calls.append(True)
            window._window = object()
            window._delegate = object()
            window._built = True

        monkeypatch.setattr(window, "_build", complete_build)
        window._ensure_built()
        window._ensure_built()
        assert len(calls) == 1, "a completed build must not run again on reopen"


class TestDetailsPanel:
    def test_a_build_that_produces_nothing_raises(self, monkeypatch):
        panel = DetailsPanel(callback=None)
        monkeypatch.setattr(
            panel, "_build", lambda: _build_that_produces_nothing(panel)
        )
        with pytest.raises(RuntimeError, match="failed to build"):
            panel._ensure_built()

    def test_a_delegate_without_a_data_source_still_raises(self, monkeypatch):
        """Both halves are required, so half-assigned is still a failed build."""
        panel = DetailsPanel(callback=None)
        panel._delegate = object()
        panel._data_source = None
        monkeypatch.setattr(panel, "_build", lambda: None)
        with pytest.raises(RuntimeError, match="failed to build"):
            panel._ensure_built()
