"""An exception must not leave a callback AppKit invoked.

PyObjC dispatches an action or a timer straight into Python. An exception that
unwinds out of one aborts the process below Python's own handlers, so there is
no traceback, no crash report, and nothing in the log: the app simply vanishes.
Three separate bugs hid behind that single symptom earlier on this branch.

This is the level the previous round got wrong. Making the panels raise a clear
RuntimeError instead of an AttributeError on None reads as a fix at the seam
where it was written, and changes nothing at all about the outcome, because
both unwind through the same unguarded callback. The failure has to be caught
where it crosses back into AppKit, and that is what these pin.
"""

from overheard import app as app_module


class Boom:
    """A panel whose show() fails, as a partially built one does."""

    def __init__(self):
        self.shown = 0

    def show(self, *a, **k):
        self.shown += 1
        raise RuntimeError("the preferences window failed to build")


class TestPreferencesGearButton:
    """popover gear -> _PopoverDelegate.openPreferences_ -> cb() -> here."""

    def test_a_failing_window_does_not_escape_the_callback(self, monkeypatch, capsys):
        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._prefs_window = Boom()

        instance._open_preferences_cb()   # must not raise

        assert instance._prefs_window.shown == 1
        assert "could not open Preferences" in capsys.readouterr().err

    def test_a_failing_constructor_does_not_escape_either(self, monkeypatch, capsys):
        """The window is constructed inside the callback on first open."""
        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._prefs_window = None

        import overheard.preferences as prefs

        def explode():
            raise RuntimeError("NSPanel allocation failed")

        monkeypatch.setattr(prefs, "PreferencesWindow", explode)

        instance._open_preferences_cb()   # must not raise

        assert "could not open Preferences" in capsys.readouterr().err

    def test_the_failure_is_logged_with_a_traceback(self, capsys):
        """A bundled app has no terminal, so stderr is the only diagnosis.

        Swallowing the exception silently would trade an invisible crash for an
        invisible no-op, which is not an improvement.
        """
        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._prefs_window = Boom()
        instance._open_preferences_cb()
        err = capsys.readouterr().err
        assert "RuntimeError" in err and "Traceback" in err


class TestDetailsPanelTimer:
    """rumps.Timer callback, reached after a recording stops."""

    def test_a_failing_panel_does_not_escape_the_timer(self, capsys):
        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._pending_meeting_meta = ("Standup", "zoom", "Zoom", ["Don"])
        instance._pending_wav = "/tmp/does-not-matter.wav"
        instance._gather_poll_timer = None
        instance._ensure_details_panel = lambda: Boom()

        states = []
        instance._set_state = lambda state, text="": states.append((state, text))

        class Timer:
            def stop(self):
                pass

        instance._poll_gather_done(Timer())   # must not raise

        err = capsys.readouterr().err
        assert "could not open the details panel" in err
        assert states, "the user must be told something went wrong"
        assert any("failed" in text for _, text in states), (
            f"no failure surfaced to the status label, got {states}"
        )

    def test_the_recording_is_not_discarded_when_the_panel_fails(self, capsys):
        """The audio is already on disk here, and losing it is the real cost."""
        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._pending_meeting_meta = ("Standup", "zoom", "Zoom", ["Don"])
        instance._pending_wav = "/tmp/recording.wav"
        instance._gather_poll_timer = None
        instance._ensure_details_panel = lambda: Boom()
        instance._set_state = lambda *a, **k: None

        class Timer:
            def stop(self):
                pass

        instance._poll_gather_done(Timer())
        assert instance._pending_wav == "/tmp/recording.wav", (
            "the pending recording must survive a panel failure"
        )


class TestFirstPartyObjCMethodsAreGuarded:
    """rumps guards its own callbacks; it does not guard ours.

    rumps wraps every menu item and Timer it dispatches in try/except
    (rumps.py:730 and :997), so those two surfaces were already safe. The
    first-party NSObject and NSView subclasses had nothing, and they are the
    larger surface: mouse events, actions and drawRect_ on _PillButton,
    _LevelBar, _DragHeader and four delegates.
    """

    def test_a_failing_transport_callback_does_not_escape_mousedown(self, capsys):
        """The worst case in the app: Stop, with the recording still in RAM.

        _PillButton.mouseDown_ calls the transport callback directly. That
        callback is _on_stop, which calls TapRecorder.stop, where proc.wait
        after a kill and np.concatenate over the accumulated chunks can both
        raise. The chunks ARE the meeting, and they have not been written to
        disk yet, so an abort here loses the recording outright.
        """
        from overheard import popover

        exploded = []

        def callback():
            exploded.append(True)
            raise MemoryError("np.concatenate over 2.7 GB of chunks")

        button = popover._PillButton.alloc().initWithIcon_label_color_callback_(
            "●", "Stop", None, callback
        )
        button.mouseDown_(None)          # must not raise

        assert exploded, "the callback should still have been attempted"
        assert "failed" in capsys.readouterr().err

    def test_a_failing_drawrect_does_not_escape(self, capsys):
        """drawRect_ is where the original SIGTRAP came from.

        An exception raised inside AppKit's drawing machinery aborts the
        process. The class-level defaults stop the known case, but the guard is
        what makes the next one survivable.
        """
        from AppKit import NSMakeRect
        from overheard import popover

        bar = popover._LevelBar.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        # Stand in for any attribute drawRect_ reads that is not there, which
        # is the shape of the original crash.
        bar._active = True
        bar._level = object()    # multiplying this raises inside the draw

        bar.drawRect_(bar.bounds())   # must not raise

        assert "failed" in capsys.readouterr().err

    def test_every_appkit_dispatched_delegate_action_is_guarded(self):
        """Derived, so a new action is covered without anyone remembering.

        Guarding by hand means guarding the ones somebody thought of, which is
        how the transport buttons stayed exposed while two callbacks that rumps
        already covered got guards.
        """
        import inspect
        from overheard import details_panel, live_panel, popover, preferences

        # Methods AppKit dispatches that carry a user action or an event.
        # Excluded by design: init* (they must return self), the table data
        # source methods (they must return typed values, so a guard returning
        # None would break the table), and our own setters.
        prefixes = ("on", "toggle", "browse", "save", "open", "quit", "create",
                    "download", "connect", "forget", "show", "mouse", "draw",
                    "refresh", "release", "select")
        skip = {"initWithFrame_", "initWithCallbacks_", "initWithWindow_",
                "initWithPanel_", "initWithIcon_label_color_callback_",
                "initWithCallback_discardCallback_", "setLevel_", "setActive_",
                "setEnabled_", "setRows_"}

        unguarded = []
        for module in (popover, preferences, details_panel, live_panel):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                for name, fn in vars(cls).items():
                    if not name.endswith("_") or name in skip:
                        continue
                    if not any(name.lower().startswith(p) for p in prefixes):
                        continue
                    # PyObjC replaces the function with a selector object and
                    # keeps the original on .callable, so the decorator's
                    # __wrapped__ marker lives there rather than on the
                    # attribute itself.
                    underlying = getattr(fn, "callable", fn)
                    if not callable(underlying):
                        continue
                    if getattr(underlying, "__wrapped__", None) is None:
                        unguarded.append(f"{module.__name__}.{cls.__name__}.{name}")

        assert not unguarded, (
            "these AppKit-dispatched methods have no objc_safe guard, so an "
            f"exception in one aborts the process with no traceback: {unguarded}"
        )
