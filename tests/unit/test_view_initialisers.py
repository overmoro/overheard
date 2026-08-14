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


def test_the_level_bar_draws_without_raising():
    """Draw it for real, since drawRect_ is where the AttributeError landed.

    Drawing needs a focused context, so it happens into an offscreen image
    rather than a window. No window server interaction is required.
    """
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    bar.setActive_(True)
    bar.setLevel_(0.75)

    image = NSImage.alloc().initWithSize_((120, 10))
    image.lockFocus()
    try:
        bar.drawRect_(bar.bounds())
    finally:
        image.unlockFocus()


def test_an_undrawn_level_bar_still_draws():
    """The failing case: never told a level, then asked to draw."""
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    image = NSImage.alloc().initWithSize_((120, 10))
    image.lockFocus()
    try:
        bar.drawRect_(bar.bounds())
    finally:
        image.unlockFocus()


def test_showing_the_system_meter_row_is_survivable():
    """configure_channels(True) is the call that unhid the row and killed it."""
    pop = popover.TransportPopover({})
    pop.configure_channels(True)
    assert pop._is_multichannel is True
