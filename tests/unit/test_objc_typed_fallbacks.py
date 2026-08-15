"""A data source that fails must return the right kind of nothing.

AppKit asks these three for values, not for a favour. Returning None where it
asked for a row count is not a degradation, it is a different crash one frame
later, which is why they could not simply take the same guard as the actions.

Written after noticing the fallback had been added and verified with a
throwaway shell snippet and never pinned, which is the defect this whole branch
keeps circling: verified once, described as done, nothing to stop it going
back.
"""

import pytest

from overheard import details_panel
from overheard.objc_safety import objc_safe_returning


@pytest.fixture
def broken_source():
    """A data source whose backing rows have gone, as a failed build leaves it."""
    source = details_panel._AttendeeDataSource.alloc().init()
    del source._rows          # every method below reads this
    return source


class TestTheFallbacksAreTyped:
    def test_the_row_count_falls_back_to_an_int(self, broken_source, capsys):
        rows = broken_source.numberOfRowsInTableView_(None)
        assert rows == 0
        assert isinstance(rows, int), (
            f"AppKit asked for a row count and got {type(rows).__name__}; "
            "None here is a crash one frame later, not a degradation"
        )
        assert "failed" in capsys.readouterr().err

    def test_the_cell_value_falls_back_to_a_string(self, broken_source, capsys):
        value = broken_source.tableView_objectValueForTableColumn_row_(None, None, 0)
        assert value == ""
        assert isinstance(value, str), (
            f"the cell getter returned {type(value).__name__}, not a string"
        )
        assert "failed" in capsys.readouterr().err

    def test_the_setter_swallows_and_returns_none(self, broken_source, capsys):
        """This one returns nothing, so None is the honest fallback."""
        result = broken_source.tableView_setObjectValue_forTableColumn_row_(
            None, "Don", None, 0
        )
        assert result is None
        assert "failed" in capsys.readouterr().err


class TestTheOrdinaryPathIsUnchanged:
    """A guard that alters the working path is not a guard."""

    def test_a_healthy_source_still_reports_its_rows(self):
        source = details_panel._AttendeeDataSource.alloc().init()
        source.setRows_([["SPEAKER_00", "Don"], ["SPEAKER_01", "Kim"]])
        assert source.numberOfRowsInTableView_(None) == 2

    def test_a_healthy_source_still_returns_cell_values(self):
        source = details_panel._AttendeeDataSource.alloc().init()
        source.setRows_([["SPEAKER_00", "Don"]])

        class Column:
            def __init__(self, identity):
                self._identity = identity

            def identifier(self):
                return self._identity

        assert source.tableView_objectValueForTableColumn_row_(
            None, Column("speaker"), 0) == "SPEAKER_00"
        assert source.tableView_objectValueForTableColumn_row_(
            None, Column("name"), 0) == "Don"


class TestTheDecoratorItself:
    def test_the_selectors_survive_decoration(self):
        """PyObjC derives the selector from the argument count.

        The decorator spells each arity out rather than using *args for exactly
        this reason: a *args wrapper reports zero arguments and the class fails
        to build at import.
        """
        source = details_panel._AttendeeDataSource.alloc().init()
        for selector in (
            "numberOfRowsInTableView:",
            "tableView:objectValueForTableColumn:row:",
            "tableView:setObjectValue:forTableColumn:row:",
        ):
            assert source.respondsToSelector_(selector), f"lost {selector}"

    @pytest.mark.parametrize("args", [
        pytest.param(("a",), id="arity-2"),
        pytest.param(("a", "b"), id="arity-3"),
        pytest.param(("a", "b", "c"), id="arity-4"),
        pytest.param(("a", "b", "c", "d"), id="arity-5"),
    ])
    def test_every_supported_arity_guards(self, args, capsys):
        """The table covers the delegate shapes, not just the three in use.

        It was dispatched over exactly the shapes the NSTableView methods
        happen to have, so the next delegate with two arguments would have hit
        a TypeError at import for no reason.
        """
        params = ", ".join(f"a{i}" for i in range(len(args)))

        namespace = {"objc_safe_returning": objc_safe_returning}
        exec(
            f"@objc_safe_returning('fallback')\n"
            f"def method(self, {params}):\n"
            f"    raise RuntimeError('boom')\n",
            namespace,
        )
        assert namespace["method"](None, *args) == "fallback"
        assert "boom" in capsys.readouterr().err

    def test_an_unsupported_arity_is_refused_loudly(self):
        """Silently widening to *args would break the selector at import.

        Failing here, at decoration time, beats a class that will not build.
        """
        with pytest.raises(TypeError, match="no wrapper"):
            @objc_safe_returning(0)
            def six_arguments(self, a, b, c, d, e):
                return None


class TestTheNoArgGuard:
    """The third decorator, which shipped with no behavioural coverage at all.

    Its two siblings are covered: objc_safe by the mouseDown_ and drawRect_
    tests, objc_safe_returning by the class above. This one could be neutered
    entirely, by narrowing its except to something that never fires, with the
    whole suite green. Its only shipping use guards updateTrackingAreas, where
    it is the one thing between a degenerate rect during a live resize and a
    silent process abort.
    """

    def test_it_swallows_and_reports(self, capsys):
        from overheard.objc_safety import objc_safe_noarg

        class Thing:
            @objc_safe_noarg
            def refresh(self):
                raise RuntimeError("AppKit handed us a degenerate rect")

        assert Thing().refresh() is None
        err = capsys.readouterr().err
        assert "refresh failed" in err
        assert "RuntimeError" in err and "Traceback" in err, (
            "a bundled app has no terminal, so the log is the only diagnosis"
        )

    def test_it_does_not_change_the_working_path(self):
        from overheard.objc_safety import objc_safe_noarg

        class Thing:
            @objc_safe_noarg
            def refresh(self):
                return "done"

        assert Thing().refresh() == "done"

    def test_the_shipping_use_survives_a_failing_setup(self, capsys):
        """updateTrackingAreas must still call super even when our part fails.

        Skipping super would trade a crash for a view that silently stops
        receiving mouse events, which is why the guard is inside the method
        rather than around the whole of it.
        """
        from overheard import popover

        button = popover._PillButton.alloc().initWithIcon_label_color_callback_(
            "●", "Stop", None, lambda: None
        )
        button._setup_tracking = lambda: (_ for _ in ()).throw(
            RuntimeError("degenerate tracking rect")
        )
        button.updateTrackingAreas()          # must not raise
        assert "tracking area setup failed" in capsys.readouterr().err
