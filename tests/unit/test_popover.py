"""The status item's right-click menu.

This path had two bugs in a row. First ``NSApplication`` was never imported into
the method that used it, so right-clicking raised NameError. Fixing the import
revealed the second: the code sent ``popUpContextMenu:withEvent:forView:`` to
the status bar button, but that is a class method on NSMenu, so the menu was
built and then discarded one line later by an AttributeError.

Both bugs shipped because nothing exercised the path. It is the only way to quit
the app, since app.py clears the status item's default menu, so a broken context
menu leaves the user with no way out.

The menu is never actually popped here: ``popUpContextMenu`` starts a modal
tracking loop and would hang the suite. The spy stands in for exactly that one
call, which is the call that was wrong.
"""

import pytest

from overheard import popover


class MenuSpy:
    """Proxies the real NSMenu, capturing only the pop call.

    Everything else defers to AppKit so the menu under test is a genuine NSMenu
    with genuine items, and the assertions below are about real objects.
    """

    def __init__(self, real):
        self._real = real
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def popUpContextMenu_withEvent_forView_(self, menu, event, view):
        self.calls.append((menu, event, view))


@pytest.fixture
def spy(monkeypatch):
    spy = MenuSpy(popover.NSMenu)
    monkeypatch.setattr(popover, "NSMenu", spy)
    return spy


def _delegate():
    return popover._PopoverDelegate.alloc().initWithCallbacks_({})


def test_right_clicking_actually_pops_the_menu(spy):
    """The regression: this call used to go to the view and raise."""
    from AppKit import NSButton

    sender = NSButton.alloc().init()
    _delegate()._show_context_menu(sender)

    assert len(spy.calls) == 1, "the menu was built but never shown"
    menu, _event, view = spy.calls[0]
    assert view is sender


def test_the_menu_offers_a_way_to_quit(spy):
    """app.py clears the status item's default menu, so this is the only exit."""
    from AppKit import NSButton

    _delegate()._show_context_menu(NSButton.alloc().init())
    menu = spy.calls[0][0]

    titles = [menu.itemAtIndex_(i).title() for i in range(menu.numberOfItems())]
    assert "Quit Overheard" in titles
    assert "Show App" in titles


def test_popping_a_menu_is_a_class_method_on_nsmenu():
    """Why the call must target NSMenu, pinned so the old form cannot return.

    A view does not respond to this selector at all, which is what made the
    first fix look right and behave wrong.
    """
    from AppKit import NSButton, NSMenu

    assert hasattr(NSMenu, "popUpContextMenu_withEvent_forView_")
    assert not hasattr(NSButton.alloc().init(), "popUpContextMenu_withEvent_forView_")
