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
