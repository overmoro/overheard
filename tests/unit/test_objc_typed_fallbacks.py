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

    def test_an_unsupported_arity_is_refused_loudly(self):
        """Silently widening to *args would break the selector at import.

        Failing here, at decoration time, beats a class that will not build.
        """
        with pytest.raises(TypeError, match="no wrapper"):
            @objc_safe_returning(0)
            def six_arguments(self, a, b, c, d, e):
                return None
