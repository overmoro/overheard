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

There are two layers of defence now, and they catch different things:

- ``_LevelBar`` carries class-level ``_level`` and ``_active`` defaults, so a
  draw can no longer raise even if no initialiser ran at all.
- ``initWithFrame_`` assigns them on the instance, which is the correct fix.

The defaults are what make the draw tests below unable to reach the original
crash on their own: with them in place, deleting ``initWithFrame_`` outright
leaves every drawing test green. That is a good property of the shipping code
and a bad property of a test that claims to reproduce the bug, so the tests
here assert against the instance rather than against the value, which is the
one thing a class default cannot fake.
"""

import pytest
from AppKit import NSImage, NSMakeRect, NSView

from overheard import popover, state


def custom_view_classes():
    """Every NSView subclass defined in popover, found rather than listed."""
    for name, obj in vars(popover).items():
        if (isinstance(obj, type) and name.startswith("_")
                and obj is not NSView and issubclass(obj, NSView)):
            yield obj


def custom_views():
    return [pytest.param(cls, id=cls.__name__) for cls in custom_view_classes()]


@pytest.mark.parametrize("view_class", custom_views())
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


def test_the_level_bar_initialiser_populates_the_instance():
    """The construction _build uses, asserted where a class default cannot reach.

    Reading ``bar._level`` and finding 0.0 proves nothing on its own: the class
    carries ``_level = 0.0`` as a backstop, so that assertion passes with the
    initialiser deleted entirely. The instance dict is the discriminator. It is
    populated only by code that actually ran against this object.
    """
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    assert set(bar.__dict__) >= {"_level", "_active"}, (
        "initWithFrame_ did not assign the instance attributes; the values "
        f"being readable comes from the class defaults. Instance has: "
        f"{sorted(bar.__dict__)}"
    )
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


def test_a_level_bar_draws_before_being_given_a_level():
    """Drawing an untouched view, which is the order the crash happened in.

    This cannot fail on the original bug any more, because the class defaults
    absorb a missing initialiser. What it does guard is the next custom view
    whose drawRect_ reads an attribute that has neither a class default nor an
    assignment, which is the same defect wearing different clothes.
    """
    _draw(popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10)))


def test_a_level_bar_draws_after_being_given_a_level():
    """The ordinary path, drawn only after the untouched case above has run."""
    bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 120, 10))
    bar.setActive_(True)
    bar.setLevel_(0.75)
    _draw(bar)


def _all_subviews(view):
    for sub in view.subviews():
        yield sub
        yield from _all_subviews(sub)


def test_every_custom_view_in_a_real_popover_can_draw():
    """The end-to-end guard: build the real popover and draw everything in it.

    configure_channels(True) only unhides the row; it does not itself draw, so
    asserting it returns cleanly proves nothing. What killed the app was the
    draw AppKit performed afterwards.

    The assertion is on the set of classes reached, derived from the same
    helper the parametrised test uses. A count would not do: the popover holds
    three _PillButtons, so a floor of four is satisfied with both _LevelBars
    absent, and _LevelBar is the class this whole file exists for. Naming the
    classes means a view dropping out of the hierarchy fails by name, and a new
    custom view is covered without anyone remembering to update a number.
    """
    pop = popover.TransportPopover({})
    pop.configure_channels(True)
    assert pop._is_multichannel is True

    drawn = set()
    for view in _all_subviews(pop._panel.contentView()):
        if type(view).__module__ == popover.__name__:
            _draw(view)
            drawn.add(type(view))

    expected = set(custom_view_classes())
    missing = expected - drawn
    assert not missing, (
        "these custom views were never drawn: "
        f"{sorted(cls.__name__ for cls in missing)}. Drew: "
        f"{sorted(cls.__name__ for cls in drawn)}"
    )


def test_the_real_popover_survives_the_record_transition():
    """Pressing Record drives set_state then set_levels, and nothing tested it.

    This range deleted the ``if not self._mic_bar`` guards from both methods on
    the grounds that ``_build`` always assigns the bars. Nothing checked that
    claim: popover.py is outside the coverage denominator, and _LevelBar
    subclasses NSView, which PyObjC leaves untyped, so mypy accepts any
    attribute however spelled. Dropping ``self._sys_bar = sys_bar`` from _build
    and misspelling ``setActive_`` left all 270 tests green.

    In production that AttributeError lands inside _poll_recorder_started, a
    rumps.Timer callback, which rumps wraps in its own try/except. Be precise
    about what that costs, because an earlier version of this docstring was not:
    _set_state assigns app._state before delegating to the popover, and
    set_state sets all three buttons and the status label before it reaches
    _set_meters_visible, so the transport is left looking correct and _on_record
    still refuses a second press.

    What is actually lost is everything after the raise: _start_level_timer()
    and _start_live() never run (app.py:200-201). The app records with dead
    meters and no live transcript panel, and nothing on screen says why.

    Naming no attributes here is deliberate: the point is to run the real
    methods so every attribute name and selector spelling they depend on has to
    exist. An assertion listing them would need updating every time one moves,
    and would pin the list rather than the behaviour.
    """
    pop = popover.TransportPopover({})
    pop.configure_channels(True)

    pop.set_state(state.RECORDING, "Recording...")
    pop.set_levels(0.3, 0.3)
    pop.set_state(state.PAUSED, "Paused")
    pop.set_state(state.TRANSCRIBING, "Transcribing...")
    pop.set_state(state.IDLE, "")

    # Single channel is the other live configuration: _sys_bar is still built
    # and still driven, just deactivated, so it must survive the same path.
    mono = popover.TransportPopover({})
    mono.configure_channels(False)
    mono.set_state(state.RECORDING, "Recording...")
    mono.set_levels(0.3, 0.0)
