"""Custom AppKit views must actually run their initialiser.

This is where pressing Record killed the app.

``_LevelBar`` overrode ``init`` and set ``_level`` and ``_active`` there, but
``_build`` constructs it with ``alloc().initWithFrame_(...)``. ``initWithFrame:``
is NSView's designated initialiser and does not route through ``init``, so the
override never ran and neither attribute existed.

That did not fail quietly. ``drawRect_`` reads both, so the first time the view
was asked to draw it raised AttributeError from inside AppKit's drawing
machinery, where an unhandled Python exception aborts the process. Pressing
Record called ``configure_channels(True)``, which unhid the system meter row,
the row drew for the first time, and Overheard vanished with SIGTRAP and no
traceback anywhere.

The structural test is the one that matters: it catches the pattern rather than
this one instance, and it derives the view list instead of enumerating it.
"""

import pytest
from AppKit import NSImage, NSMakeRect, NSView

from overheard import popover


def custom_views():
    """Every NSView subclass defined in popover, found rather than listed."""
    for name, obj in vars(popover).items():
        if (isinstance(obj, type) and name.startswith("_")
                and obj is not NSView and issubclass(obj, NSView)):
            yield pytest.param(obj, id=name)


@pytest.mark.parametrize("view_class", list(custom_views()))
def test_an_init_override_is_not_silently_skipped(view_class):
    """A view overriding init but not initWithFrame_ is a trap.

    NSView's designated initialiser is initWithFrame:. Anything built that way
    skips an ``init`` override entirely, so whatever it assigned is missing and
    the first draw raises. Override initWithFrame_ instead.
    """
    own = view_class.__dict__
    if "init" in own:
        assert "initWithFrame_" in own, (
            f"{view_class.__name__} overrides init but not initWithFrame_. "
            "Anything built with alloc().initWithFrame_() will skip it."
        )


def test_the_level_bar_is_usable_when_built_with_initwithframe():
    """The exact construction _build uses."""
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    assert bar._level == 0.0
    assert bar._active is False


def _draw(view):
    """Draw a view offscreen, which is where the AttributeError landed.

    Drawing needs a focused context, so it goes into an image rather than a
    window. No window server interaction is required.
    """
    size = (max(1, view.bounds().size.width), max(1, view.bounds().size.height))
    image = NSImage.alloc().initWithSize_(size)
    image.lockFocus()
    try:
        view.drawRect_(view.bounds())
    finally:
        image.unlockFocus()


def test_an_undrawn_level_bar_still_draws():
    """The failing case: built, never told a level, then asked to draw.

    Calling setLevel_ or setActive_ first would create the very attributes the
    missing initialiser failed to assign, so a test that sets before drawing
    can never reach the bug. That is what the first version of this file did.
    """
    _draw(popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10)))


def test_a_level_bar_draws_after_being_given_a_level():
    """The ordinary path, drawn only after the undrawn case above has run."""
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    bar.setActive_(True)
    bar.setLevel_(0.75)
    _draw(bar)


def _all_subviews(view):
    for sub in view.subviews():
        yield sub
        yield from _all_subviews(sub)


def test_every_view_in_a_real_popover_can_draw():
    """The end-to-end guard, and the one that covers _PillButton.

    configure_channels(True) only unhides the row; it does not itself draw, so
    asserting it returns cleanly proves nothing. What killed the app was the
    draw that AppKit performed afterwards. This builds the real popover, shows
    the system meter row, and then draws every view in it.
    """
    pop = popover.TransportPopover({})
    pop.configure_channels(True)
    assert pop._is_multichannel is True

    drawn = 0
    for view in _all_subviews(pop._panel.contentView()):
        if type(view).__module__ == popover.__name__:
            _draw(view)
            drawn += 1
    assert drawn >= 4, f"expected the custom views to be reached, drew {drawn}"
