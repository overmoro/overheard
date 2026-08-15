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

import ast
import inspect
import textwrap

import pytest
from AppKit import NSTabView

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


def _widgets_only(names, obj):
    """Drop names that resolve to methods on the class.

    ``_attrs_read_but_never_assigned`` returns the delegate's own helper methods
    alongside its widgets, and a method is satisfied by the class definition
    rather than by anything ``_build`` does. Of the 20 names it returns for the
    preferences delegate, 9 are its own methods, so counting them made the set
    look nearly twice as well covered as it is.
    """
    return {
        name for name in names
        if not callable(getattr(type(obj), name, None))
    }


def _attrs_read_but_never_assigned(cls):
    """Attributes a class reads off ``self`` and never assigns to itself.

    Both delegates are populated from outside, by the panel's ``_build``, so
    every name in this set is a promise ``_build`` has to keep. Deriving it from
    the source rather than listing it means a widget added tomorrow is covered
    without anyone remembering this file exists, and a widget renamed in only
    one of the two places fails here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    reads, writes = set(), set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            target = writes if isinstance(node.ctx, ast.Store) else reads
            target.add(node.attr)
    return {name for name in reads - writes if name.startswith("_")}


def _attrs_read_off(cls, *locals_):
    """Attributes read off the objects ``_ensure_built`` hands back.

    ``show()`` reaches through the delegate for widgets the delegate itself
    never mentions (``delegate._table_view.reloadData()``), so those are
    invisible to the helper above and need collecting from the caller's side.

    Public names count. Filtering to a leading underscore made this return the
    empty set for the data source, whose only use is ``data_source.setRows_``,
    and the assertion built on it could not fail for any object at all.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in locals_
    }


def _controls_with_actions(view, seen=None):
    """Every control in a built hierarchy that has a target/action wired.

    NSTabView keeps its pages off the content view's subview tree, so the tabs
    in Preferences are invisible to a plain walk and most of its buttons would
    go unchecked.
    """
    if view is None:
        return
    if isinstance(view, NSTabView):
        for item in view.tabViewItems():
            yield from _controls_with_actions(item.view())
    action = view.action() if hasattr(view, "action") else None
    if action:
        yield view, action
    for sub in view.subviews() or []:
        yield from _controls_with_actions(sub)


# Each returns (owner, root view). The owner has to be returned and held, not
# just built: an NSControl's target is a ZEROING WEAK reference, so letting the
# panel fall out of scope deallocates its delegate and every target() reads back
# as nil. A first version of these dropped the owner and the walk then reported
# every selector in both panels as unanswered.
def _built_preferences():
    panel = PreferencesWindow()
    panel._ensure_built()
    return panel, panel._window.contentView()


def _built_details():
    panel = DetailsPanel(callback=None)
    panel._ensure_built()
    return panel, panel._window.contentView()


def _built_popover():
    from overheard.popover import TransportPopover

    pop = TransportPopover({})
    return pop, pop._panel.contentView()


class TestTheRealBuildSucceeds:
    """The half every other test in this file leaves out.

    Everything above patches ``_build``, so between them they pin what the
    sentinel *means* four different ways and never once run the statement that
    sets it. Deleting ``self._built = True`` from the end of both real ``_build``
    methods left all 270 tests green while making both panels permanently
    unopenable: ``_ensure_built`` runs the whole hundred-line AppKit build, then
    raises anyway, on every single call. The gear button catches it and shows
    nothing; the details panel catches it and every finished recording becomes
    unreachable.

    A stand-in cannot catch that, because the statement lives in the code the
    stand-in replaces. These drive the real thing.

    Asserting only the sentinel was not enough either, and that was the next
    round's finding: the rest of both builds stayed mutable with the suite
    green. Deleting ``self._delegate._name_field = name_field``, or misspelling
    ``setAction_("onStartTranscription:")``, left 274 tests passing while making
    the details panel unopenable and the Start Transcription button inert. So
    these tests assert the build's *output*: every widget the code reaches for,
    and every selector it wires.

    Both contracts are derived rather than listed. A hand-written list of
    widgets would pass forever after the widget it names is renamed.

    ``isolated_config`` is not optional here: ``_build`` reads output_dir,
    obsidian_vault, obsidian_inbox and local_speaker_name, so without it these
    tests read the developer's real config and the local run stops being the
    same experiment as CI.
    """

    def test_preferences_really_builds(self, isolated_config):
        window = PreferencesWindow()
        delegate = window._ensure_built()
        assert delegate is not None
        assert window._built is True

    def test_details_panel_really_builds(self, isolated_config):
        panel = DetailsPanel(callback=None)
        delegate, data_source = panel._ensure_built()
        assert delegate is not None
        assert data_source is not None
        assert panel._built is True

    def test_the_preferences_build_keeps_every_promise_its_delegate_makes(
        self, isolated_config
    ):
        window = PreferencesWindow()
        delegate = window._ensure_built()
        # Asserted separately, not on the union. The union always contains
        # refresh_status, a method on the delegate class, so a non-emptiness
        # check on it passed no matter what the load-bearing derivation did.
        own = _widgets_only(_attrs_read_but_never_assigned(type(delegate)), delegate)
        via_show = _attrs_read_off(PreferencesWindow, "delegate")
        assert own, (
            "the widget derivation found nothing, so this test cannot fail. A "
            "rewrite reading widgets through a local alias would do that silently"
        )
        assert via_show, "the show() derivation found nothing"

        promised = own | via_show
        missing = sorted(name for name in promised if not hasattr(delegate, name))
        assert not missing, (
            f"_build never assigned these, and the delegate reads them: {missing}"
        )

    def test_the_details_build_keeps_every_promise_its_delegate_makes(
        self, isolated_config
    ):
        panel = DetailsPanel(callback=None)
        delegate, data_source = panel._ensure_built()
        own = _widgets_only(_attrs_read_but_never_assigned(type(delegate)), delegate)
        via_show = _attrs_read_off(DetailsPanel, "delegate")
        assert own, "the widget derivation found nothing to check, see above"
        assert via_show, "the show() derivation found nothing"

        promised = own | via_show
        missing = sorted(name for name in promised if not hasattr(delegate, name))
        assert not missing, (
            f"_build never assigned these, and show() reads them: {missing}"
        )

        promised_source = _attrs_read_off(DetailsPanel, "data_source")
        assert promised_source, (
            "the data-source derivation found nothing. It returned the empty "
            "set for a whole round, because setRows_ is a public name and the "
            "filter demanded a leading underscore"
        )
        missing_source = sorted(
            name for name in promised_source if not hasattr(data_source, name)
        )
        assert not missing_source, f"the data source is missing: {missing_source}"

    @pytest.mark.parametrize(
        "make_root",
        [
            pytest.param(_built_preferences, id="preferences"),
            pytest.param(_built_details, id="details"),
            # The popover is the only UI that is always on screen, and it was
            # left out of this walk for a round. It builds in __init__ rather
            # than through _ensure_built, and its window attribute is _panel.
            pytest.param(_built_popover, id="popover"),
        ],
    )
    def test_every_wired_control_targets_something_that_answers(
        self, make_root, isolated_config
    ):
        """A misspelled selector is caught by running the code and nothing else.

        These modules subclass NSObject and NSView, which PyObjC leaves untyped,
        so mypy accepts any spelling; they are excluded from the coverage
        denominator too. Both panels wire their buttons the same way, target then
        action, and a typo in the action string leaves a button that looks
        correct and does nothing forever.

        Walking the real hierarchy rather than the source means a control added
        later is covered on the day it is added.
        """
        owner, root = make_root()
        assert owner is not None  # held for the duration: see _built_* above
        wired = list(_controls_with_actions(root))
        assert wired, "found no wired controls at all, so this test proves nothing"

        dead = [
            str(action)
            for control, action in wired
            if control.target() is None
            or not control.target().respondsToSelector_(action)
        ]
        assert not dead, f"these controls are wired to a selector nobody answers: {dead}"

    def test_a_real_finished_build_is_not_repeated(self, monkeypatch, isolated_config):
        """The sentinel test above, against the real build rather than a stub.

        Its stubbed twin sets ``_built`` inside the stub, so it passes whether or
        not the production build ever sets it. This one cannot.
        """
        window = PreferencesWindow()
        window._ensure_built()
        rebuilt = []
        monkeypatch.setattr(window, "_build", lambda: rebuilt.append(True))
        window._ensure_built()
        assert not rebuilt, "a completed real build must not run again on reopen"
