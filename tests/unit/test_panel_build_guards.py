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
