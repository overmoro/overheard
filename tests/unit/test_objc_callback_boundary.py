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

    def test_every_method_on_a_first_party_objc_class_is_accounted_for(self):
        """Closed set with named exclusions, not an allow-list of prefixes.

        The first version filtered by name prefix, which is a negative
        assertion over an open set: it covered the vocabulary somebody thought
        of. Two plainly dispatched methods, windowWillClose_ and
        validateMenuItem_, matched no prefix and passed straight through it,
        and every zero-argument method was invisible because the filter also
        required a trailing underscore.

        This walks every method on every first-party NSObject and NSView
        subclass and requires each one to be either guarded or consciously
        excluded. Anything new fails until somebody classifies it, which is the
        only shape that closes an open set.
        """
        import importlib
        import inspect
        import pkgutil

        import overheard

        # Derived, not listed. A hand-written module tuple closes the set of
        # classes and leaves the set of modules open: a new file with an
        # AppKit delegate in it was invisible, which is the same open-set hole
        # this test was rewritten to remove, one level up.
        modules = []
        for info in pkgutil.iter_modules(overheard.__path__):
            modules.append(importlib.import_module(f"overheard.{info.name}"))

        # Excluded, each for a stated reason:
        #   init*        must return self, and a guard returning None on
        #                failure would be read by ObjC as "init failed", which
        #                is a different contract than the one we want.
        #   set*/names   called only by our own code, never dispatched.
        #   accepts*     must return a bool; None reads as False and would
        #                silently change responder behaviour. Bodies are a
        #                single `return True` and cannot raise.
        #   (The three NSTableView data source methods are NOT excluded. They
        #    are AppKit-dispatched and now carry objc_safe_returning, which
        #    hands back a typed fallback instead of None. An earlier version
        #    excluded them with the reason "guarded separately below", and
        #    there was nothing below: a false reason inside the one mechanism
        #    whose whole value is that its reasons are true.)
        EXCLUDED = {
            "initWithFrame_", "initWithCallbacks_", "initWithWindow_",
            "initWithPanel_", "initWithIcon_label_color_callback_",
            "initWithCallback_discardCallback_", "init",
            "setLevel_", "setActive_", "setEnabled_", "setRows_", "names",
            "acceptsFirstResponder", "acceptsFirstMouse_",
            # PyObjC injects this on every NSObject subclass. Not ours.
            "bundleForClass",
            # Not dispatched: reached only through refreshStatusOnMain_, which
            # is guarded, and through show(), which the app.py boundary
            # guards. It also guards each label independently already.
            "refresh_status",
        }

        unaccounted = []
        for module in modules:
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                if not any(base.__name__ in ("NSObject", "NSView")
                           for base in inspect.getmro(cls)[1:]):
                    continue
                for name, attr in vars(cls).items():
                    if name.startswith("__") or name in EXCLUDED:
                        continue
                    # Anything starting with a single underscore is ours to
                    # call, not something AppKit dispatches by selector.
                    if name.startswith("_"):
                        continue
                    # PyObjC replaces the function with a selector object and
                    # keeps the original on .callable, so the decorator's
                    # marker lives there.
                    underlying = getattr(attr, "callable", attr)
                    if not callable(underlying):
                        continue
                    if getattr(underlying, "__wrapped__", None) is None:
                        unaccounted.append(f"{cls.__name__}.{name}")

        assert not unaccounted, (
            "these methods on first-party ObjC classes are neither guarded nor "
            "listed in EXCLUDED, so an exception in one aborts the process "
            f"with no traceback: {sorted(unaccounted)}"
        )


class TestAFailedStopLandsInADefinedState:
    """Guarding the button stops the abort; it does not make the app coherent.

    _on_stop tears down the level timer and detaches the live tap before the
    fallible recorder.stop(), and neither can be undone. Without an explicit
    landing, the guard left the app showing RECORDING with a dead recorder
    behind it, the meters frozen and nothing on screen changed: the crash
    became a button that does nothing, which is not a failure reaching the
    user.
    """

    def _app_mid_recording(self, stop_raises):
        from overheard.state import RECORDING

        instance = app_module.TranscriberApp.__new__(app_module.TranscriberApp)
        instance._state = RECORDING

        class Recorder:
            sample_rate = 48000

            def set_tap(self, tap):
                pass

            def stop(self):
                if stop_raises:
                    raise MemoryError("np.concatenate over an hour of chunks")
                return None, None

        instance._recorder = Recorder()
        instance._stop_level_timer = lambda: None
        instance._stop_live = lambda: None
        instance.states = []
        instance._set_state = lambda state, text="": instance.states.append((state, text))
        return instance

    def test_a_failing_stop_leaves_the_app_idle_and_says_so(self, monkeypatch, capsys):
        from overheard.state import IDLE

        notes = []
        monkeypatch.setattr(
            app_module.rumps, "notification",
            lambda *a, **k: notes.append(a),
        )
        instance = self._app_mid_recording(stop_raises=True)

        instance._on_stop()          # guarded at the button; must land cleanly

        assert instance._recorder is None, "a dead recorder must not be kept"
        assert instance.states, "no state transition happened at all"
        final_state, final_text = instance.states[-1]
        assert final_state == IDLE, (
            f"the app stayed in {final_state} with a dead recorder, so the UI "
            "still claims to be recording"
        )
        assert "failed" in final_text.lower()
        assert notes, "the user was told nothing"
        assert "could not be stopped" in capsys.readouterr().err

    def test_a_successful_stop_is_unaffected(self, monkeypatch):
        """The guard must not change the ordinary path."""
        from overheard.state import IDLE

        monkeypatch.setattr(app_module.rumps, "notification", lambda *a, **k: None)
        instance = self._app_mid_recording(stop_raises=False)

        instance._on_stop()

        assert instance._recorder is None
        assert instance.states[-1][0] == IDLE
