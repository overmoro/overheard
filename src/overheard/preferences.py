"""Preferences window for Overheard.

Opens as a standalone NSPanel so it doesn't block the rumps run loop.
Sections: General, Audio, Transcription, Output, Integrations.
"""

import os
import subprocess
import sys
import threading

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSSecureTextField,
    NSTabView,
    NSTabViewItem,
    NSTextField,
    NSOpenPanel,
    NSView,
)
from Foundation import NSObject, NSRunLoopCommonModes, NSThread

from overheard import config as cfg
from overheard.audio import create_aggregate_device, create_multi_output_device
from overheard.objc_safety import objc_safe
from typing import Any

# Window dimensions
WIN_W = 500
WIN_H = 460

_ENGINE_BLURB = (
    "Parakeet TDT 0.6B v3, on the Apple Silicon GPU. Detects its own language "
    "across 25 European languages."
)


# ---------------------------------------------------------------------------
# Detect correct NSWindowStyleMask constants across PyObjC versions
# ---------------------------------------------------------------------------
def _style_mask() -> int:
    try:
        from AppKit import (
            NSWindowStyleMaskTitled,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
        )
        return NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
    except ImportError:
        from AppKit import NSTitledWindowMask, NSClosableWindowMask, NSMiniaturizableWindowMask
        return NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _make_label(text: str, x: float, y: float, w: float, h: float, bold=False) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    if bold:
        field.setFont_(NSFont.boldSystemFontOfSize_(13))
    return field


def _make_status(x: float, y: float, w: float = 320) -> NSTextField:
    """Small muted status label for feedback next to a button."""
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, 20))
    field.setStringValue_("")
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(True)
    field.setFont_(NSFont.systemFontOfSize_(11))
    field.setTextColor_(NSColor.secondaryLabelColor())
    return field


def _make_button(title: str, x: float, y: float, w: float, h: float, action, target) -> NSButton:
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    btn.setTitle_(title)
    btn.setBezelStyle_(1)  # NSBezelStyleRounded
    btn.setTarget_(target)
    btn.setAction_(action)
    return btn


def _make_text_field(x: float, y: float, w: float, h: float,
                     placeholder: str = "", secure: bool = False) -> NSTextField:
    cls = NSSecureTextField if secure else NSTextField
    field = cls.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    field.setPlaceholderString_(placeholder)
    return field


# ---------------------------------------------------------------------------
# Delegate: NSObject subclass handles all button actions
# ---------------------------------------------------------------------------

def _set_label(label, text: str) -> None:
    """Write to an AppKit label from any thread.

    The download and device-creation workers run on daemon threads. Mutating an
    NSTextField from one is undefined behaviour, and this app has already died
    once from an exception crossing the ObjC boundary with no crash report, so
    a worker's write is handed to the main thread instead.

    On the main thread the write happens in place. Marshalling unconditionally
    would queue even the main thread's own writes to a later run-loop pass, and
    since _build no longer paints the status labels, show() would order the
    window front with all three of them blank and fill them in a frame or more
    later.
    """
    if NSThread.isMainThread():
        label.setStringValue_(text)
        return
    _perform_on_main(label, "setStringValue:", text)


def _perform_on_main(target, selector: str, argument) -> None:
    """Queue a selector on the main thread, in every run-loop mode.

    The plain asynchronous perform schedules in NSDefaultRunLoopMode only, so
    anything it delivers is withheld while the main thread sits in a modal or
    tracking mode. A user who clicks Download and then opens the Browse panel,
    or just holds the mouse on the popover header, would see the progress label
    freeze for the duration. A frozen label on a multi-gigabyte download is
    exactly the "button that appears to have done nothing" that makes people
    click again, which is what the download latch exists to survive.
    """
    target.performSelectorOnMainThread_withObject_waitUntilDone_modes_(
        selector, argument, False, [NSRunLoopCommonModes]
    )


class _PreferencesDelegate(NSObject):

    def initWithWindow_(self, window):
        self = objc.super(_PreferencesDelegate, self).init()
        if self is None:
            return None
        self._window = window
        # A download owns the dependency label while it runs. Declared here so
        # every path can read it without a getattr default standing in for an
        # invariant that is supposed to hold.
        self._downloading = False
        # Incremented per download, so a finishing worker can tell whether the
        # latch it is about to release is still its own.
        self._download_generation = 0
        return self

    # ---- General -----------------------------------------------------------

    @objc_safe
    def openTranscripts_(self, sender):
        from overheard import config as _cfg
        from pathlib import Path as _Path
        d = _Path(_cfg.get("output_dir"))
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    @objc_safe
    def quitApp_(self, sender):
        import rumps
        rumps.quit_application()

    # ---- Audio Setup -------------------------------------------------------

    @objc_safe
    def createRecordingDevice_(self, sender):
        self._aggregate_status.setStringValue_("Creating...")
        threading.Thread(target=self._do_create_aggregate, daemon=True).start()

    def _do_create_aggregate(self):
        ok, msg = create_aggregate_device()
        _set_label(self._aggregate_status, f"{'✓' if ok else '✗'} {msg}")

    @objc_safe
    def createMonitoringDevice_(self, sender):
        self._multiout_status.setStringValue_("Creating...")
        threading.Thread(target=self._do_create_multiout, daemon=True).start()

    def _do_create_multiout(self):
        ok, msg = create_multi_output_device()
        _set_label(self._multiout_status, f"{'✓' if ok else '✗'} {msg}")

    # ---- Engine ------------------------------------------------------------

    @objc_safe
    def refreshStatusOnMain_(self, _ignored):
        """ObjC entry point so a worker thread can request a refresh.

        refresh_status touches several widgets, not just text fields, so the
        whole method is marshalled rather than each write inside it.
        """
        self.refresh_status()

    def refresh_status(self) -> None:
        """Recompute the labels that read state from outside this window.

        Called on open and after a download, so the window never shows a
        reading taken earlier in the session.

        Not everything on the panes is recomputed, and which is which matters.
        The capture-backend label derives from sys.platform, the macOS version
        and whether the helper binary exists, none of which change while the
        process runs, so _build writes it once and it is correct forever. The
        three below genuinely move underneath us: models get downloaded,
        aggregate devices get created and deleted in Audio MIDI Setup, and
        voices are added to the library by every transcription. Each was
        written only during _build, so opening Preferences a second time
        reported the state as it was the first time.

        The body is guarded because this runs from the gear button's action
        callback, and _model_status walks the Hugging Face cache while
        _device_exists queries Core Audio. An OSError from either would escape
        into AppKit's dispatch and abort the process without a traceback,
        which is a worse outcome than a stale label.
        """
        # A running download owns the dependency label and publishes its own
        # progress, so it is skipped entirely: overwriting it would report
        # "Missing" over a live download and invite a second one. Skipped on
        # the failure path too, which a single shared handler got wrong by
        # putting an unrelated widget's error onto this label.
        if not self._downloading:
            self._recompute(
                self._deps_status, _model_status, "Model status unavailable"
            )

        self._recompute(
            self._aggregate_status,
            lambda: _device_exists("Meeting Capture"),
            "Device status unavailable",
        )
        self._recompute(
            self._multiout_status,
            lambda: _device_exists("Meeting Monitor"),
            "Device status unavailable",
        )
        self._recompute_speakers()

    @objc.python_method
    def _recompute(self, label, produce, on_failure: str) -> None:
        """Write one label, and let a failure cost only that label.

        One try around the whole refresh meant the first failure abandoned
        every recompute after it. Since _build no longer paints these, an
        abandoned label renders empty rather than stale, on that open and on
        every open after, because the same exception recurs.
        """
        try:
            _set_label(label, produce())
        except Exception as e:
            print(
                f"[overheard] could not refresh a Preferences label: {e}",
                file=sys.stderr,
            )
            try:
                _set_label(label, on_failure)
            except Exception:
                pass

    def _recompute_speakers(self) -> None:
        """The speaker popup is several widgets, so it gets its own guard."""
        try:
            self._refresh_speakers()
        except Exception as e:
            print(
                f"[overheard] could not reload the known voices: {e}",
                file=sys.stderr,
            )

    @objc_safe
    def toggleLivePreview_(self, sender):
        cfg.set_value("live_preview", bool(sender.state()))

    # ---- Speakers ----------------------------------------------------------

    @objc_safe
    def toggleSpeakerMemory_(self, sender):
        cfg.set_value("speaker_memory", bool(sender.state()))

    def _refresh_speakers(self):
        """Reload the known-voices popup from the library on disk."""
        from overheard.speakers import SpeakerLibrary

        known = SpeakerLibrary().describe()
        self._speakers_popup.removeAllItems()
        if known:
            self._speakers_popup.addItemsWithTitles_(
                [f"{name}  ({samples} recording{'s' if samples != 1 else ''})"
                 for name, samples, _ in known]
            )
            self._known_speaker_names = [name for name, _, _ in known]
        else:
            self._speakers_popup.addItemWithTitle_("No voices remembered yet")
            self._known_speaker_names = []
        self._speakers_popup.setEnabled_(bool(known))
        self._forget_btn.setEnabled_(bool(known))

    @objc_safe
    def forgetSpeaker_(self, sender):
        from overheard.speakers import SpeakerLibrary

        index = self._speakers_popup.indexOfSelectedItem()
        names = getattr(self, "_known_speaker_names", [])
        if not (0 <= index < len(names)):
            return
        SpeakerLibrary().forget(names[index])
        self._refresh_speakers()

    # ---- Dependencies ------------------------------------------------------

    @objc_safe
    def downloadModels_(self, sender):
        """Start a download, unless one is already running.

        Without the latch every click spawned another daemon thread, so a user
        who clicked twice ran two 30 minute subprocesses writing into the same
        Hugging Face cache. Clicking twice is the natural response to a button
        that appears to have done nothing, which is exactly what a multi
        gigabyte download looks like for its first minute.
        """
        if self._downloading:
            return
        self._downloading = True
        self._download_generation += 1
        self._deps_status.setStringValue_("Downloading... (this may take several minutes)")
        generation = self._download_generation
        threading.Thread(
            target=self._do_download_models, args=(generation,), daemon=True
        ).start()

    @objc_safe
    def releaseDownload_(self, generation):
        """Clear the latch, but only if this download still owns it.

        Runs on the main thread, after the label has been written, so the latch
        never clears while the pane still reads "Downloading...". Clearing it
        on the worker before queuing the refresh left a window in which the
        button was live and the label still said a download was running, which
        is exactly when a user clicks again.

        The generation check stops a finishing worker from releasing a latch
        that a later download has since taken.
        """
        if generation == self._download_generation:
            self._downloading = False

    @objc.python_method
    def _queue_refresh(self) -> None:
        """Ask the main thread to re-read everything, from a worker.

        No isMainThread() short-circuit. This is only ever reached from the
        worker downloadModels_ starts, so that branch was dead in the app and
        alive only for one test that called _do_download_models on the main
        thread. Keeping it meant the download failure path was verified on a
        branch production never takes, which is the defect class this unit
        spent seven rounds on.
        """
        _perform_on_main(self, "refreshStatusOnMain:", None)

    @objc.python_method
    def _finish_download(self, generation) -> None:
        """Release the latch on the main thread, ordered after the label.

        No isMainThread() short-circuit, for the reason given on _queue_refresh:
        production reaches this only from the download worker.
        """
        _perform_on_main(self, "releaseDownload:", generation)

    @objc.python_method
    def _do_download_models(self, generation=None):
        if generation is None:
            generation = self._download_generation
        succeeded = False
        try:
            _set_label(self._deps_status, "Downloading Parakeet TDT v3...")
            from overheard.asr import PARAKEET_MODEL
            code = ("from parakeet_mlx import from_pretrained; "
                    f"from_pretrained({PARAKEET_MODEL!r})")
            label = "Parakeet"

            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                _set_label(
                    self._deps_status, f"✗ {label} failed: {result.stderr[:120]}"
                )
                return

            # Warm the diarization models too. The helper fetches them on first
            # use with no credentials, so this only saves the wait later.
            from overheard.helper import helper_path
            binary = helper_path()
            if binary is not None:
                _set_label(self._deps_status, "Downloading speaker models...")
                result = subprocess.run(
                    [str(binary), "download"],
                    capture_output=True, text=True, timeout=1800,
                )
                if result.returncode != 0:
                    _set_label(
                        self._deps_status,
                        f"✓ {label} done  ✗ Speaker models: {result.stderr[:80]}",
                    )
                    return

            succeeded = True
        except subprocess.TimeoutExpired:
            _set_label(self._deps_status, "✗ Download timed out")
        except Exception as e:
            _set_label(self._deps_status, f"✗ {e}")
        finally:
            # The single release point, for every path through this method. An
            # earlier version released on the success path and again here,
            # because a return inside a try still runs its finally.
            #
            # It has to happen BEFORE the refresh, not after. refresh_status
            # skips the dependency label while the latch is set, so releasing
            # afterwards left the pane reading "Downloading..." forever with a
            # live button underneath, which is the double click the latch is
            # there to prevent. Releasing first also means the button is only
            # live once the label says what actually happened.
            self._finish_download(generation)
            if succeeded:
                self._queue_refresh()

    # ---- Output Folder -----------------------------------------------------

    @objc_safe
    def toggleKeepRecordings_(self, sender):
        cfg.set_value("keep_recordings", bool(sender.state()))

    @objc_safe
    def browseOutputFolder_(self, sender):
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Select Output Folder")

        current = self._output_field.stringValue()
        if current:
            from Foundation import NSURL
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(current))

        if panel.runModal() == 1:  # NSModalResponseOK
            path = panel.URL().path()
            self._output_field.setStringValue_(path)
            cfg.set_value("output_dir", path)
            self._output_status.setStringValue_("✓ Saved")

    # ---- Integrations, Obsidian -------------------------------------------

    @objc_safe
    def toggleObsidian_(self, sender):
        enabled = bool(sender.state())
        cfg.set_value("obsidian_enabled", enabled)
        self._obsidian_vault_field.setEnabled_(enabled)
        self._obsidian_inbox_field.setEnabled_(enabled)
        self._obsidian_browse_btn.setEnabled_(enabled)

    @objc_safe
    def browseObsidianVault_(self, sender):
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Select Obsidian Vault Folder")

        current = self._obsidian_vault_field.stringValue()
        if current:
            from Foundation import NSURL
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(current))

        if panel.runModal() == 1:
            path = panel.URL().path()
            self._obsidian_vault_field.setStringValue_(path)
            cfg.set_value("obsidian_vault", path)

    @objc_safe
    def saveObsidianInbox_(self, sender):
        val = self._obsidian_inbox_field.stringValue().strip()
        cfg.set_value("obsidian_inbox", val or "01_Inbox")

    @objc_safe
    def saveLocalSpeakerName_(self, sender):
        val = self._local_speaker_field.stringValue().strip()
        cfg.set_value("local_speaker_name", val or "Don")

    # ---- Integrations, Calendar -------------------------------------------

    @objc_safe
    def connectCalendar_(self, sender):
        """Trigger the macOS Calendar TCC permission prompt deliberately."""
        self._calendar_status.setStringValue_("Requesting access…")
        threading.Thread(target=self._do_connect_calendar, daemon=True).start()

    def _do_connect_calendar(self):
        try:
            import subprocess
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "Calendar" to return name of first calendar'],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                _set_label(self._calendar_status, "✓ Calendar access granted")
            else:
                err = (result.stderr or "Permission denied").strip()[:80]
                _set_label(self._calendar_status, f"✗ {err}")
        except subprocess.TimeoutExpired:
            _set_label(self._calendar_status, "✗ Timed out. Check System Settings → Privacy → Calendars")
        except Exception as e:
            _set_label(self._calendar_status, f"✗ {e}")


def _model_status() -> str:
    """Describe what is actually on disk, rather than assuming nothing is.

    This label used to read "Downloads on first use" unconditionally, which was
    indistinguishable from a real check and wrong for anyone who had already
    downloaded.
    """
    from overheard.asr import PARAKEET_MODEL, is_model_cached
    from overheard.helper import diarization_models_present

    have_asr = is_model_cached(PARAKEET_MODEL)
    have_diar = diarization_models_present()

    if have_asr and have_diar:
        return "\u2713 Installed"
    missing = []
    if not have_asr:
        missing.append("transcription")
    if not have_diar:
        missing.append("speaker")
    return "Missing: " + " and ".join(missing) + " models"


def _device_exists(name: str) -> str:
    """Return a status string for whether a named audio device is present."""
    try:
        import sounddevice as sd
        for d in sd.query_devices():
            if name.lower() in d["name"].lower():
                return "✓ Already exists"
    except Exception:
        pass
    return "Not found"


# ---------------------------------------------------------------------------
# Main preferences window
# ---------------------------------------------------------------------------

class PreferencesWindow:
    """Builds and manages the Preferences NSPanel.

    Call show() to bring it to the front; the first call builds it.
    """

    def __init__(self):
        self._window: Any = None
        self._delegate: "_PreferencesDelegate | None" = None
        # Set as the very last statement of _build, so it means "_build ran to
        # completion" and nothing weaker. Keying on _window or on _delegate
        # instead means keying on a value assigned in the first few lines of a
        # method that then runs for another hundred and forty: a failure past
        # that point leaves the panel believing it is built, forever, and
        # every later open dies on a widget the build never got to.
        self._built = False

    def _ensure_built(self) -> "_PreferencesDelegate":
        """Build on first use and hand back the delegate show() needs.

        Retries on every call until a build finishes, because a partially
        built panel is not a panel. Raising here rather than returning a
        half-populated delegate keeps the AttributeError-inside-AppKit failure
        off the table; the caller at the ObjC boundary is what turns this into
        a message instead of an abort.
        """
        if not self._built:
            self._build()
        delegate = self._delegate
        if not self._built or delegate is None:
            raise RuntimeError("the preferences window failed to build")
        return delegate

    def show(self) -> None:
        """Open the window, refreshing anything that can go stale while closed.

        The window is built once and reused, so status computed during _build
        would be a reading from the first time Preferences was ever opened.
        Models get downloaded long after that.
        """
        delegate = self._ensure_built()
        delegate.refresh_status()
        self._window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def _build(self) -> None:
        self._window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIN_W, WIN_H),
            _style_mask(),
            NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_("Overheard Preferences")
        self._window.center()
        self._window.setFloatingPanel_(True)

        self._delegate = _PreferencesDelegate.alloc().initWithWindow_(self._window)
        cv = self._window.contentView()

        # ---- Tab view fills the window ------------------------------------
        tab_view = NSTabView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_W, WIN_H))
        cv.addSubview_(tab_view)

        def _make_tab(label: str) -> NSView:
            item = NSTabViewItem.alloc().initWithIdentifier_(label)
            item.setLabel_(label)
            tab_view.addTabViewItem_(item)
            return item.view()

        # Inner pane dimensions (inside the tab chrome)
        PW = WIN_W - 40   # pane width available for content

        # ================================================================== #
        # Tab 1: General
        # ================================================================== #
        pane = _make_tab("General")
        y = 340

        pane.addSubview_(_make_label("Transcripts", 20, y, 300, 22, bold=True))
        y -= 36

        pane.addSubview_(_make_button(
            "Open Transcripts Folder  ↗", 20, y, 220, 28,
            "openTranscripts:", self._delegate,
        ))
        y -= 60

        pane.addSubview_(_make_label("Application", 20, y, 300, 22, bold=True))
        y -= 36

        quit_btn = _make_button("Quit Overheard", 20, y, 160, 28,
                                "quitApp:", self._delegate)
        pane.addSubview_(quit_btn)

        # ================================================================== #
        # Tab 2: Audio
        # ================================================================== #
        pane = _make_tab("Audio")
        y = 340

        pane.addSubview_(_make_label("Recording Devices", 20, y, 300, 22, bold=True))
        y -= 26

        from overheard.capture import is_available as _tap_available
        _tap_ok, _tap_why = _tap_available()
        self._delegate._capture_status = _make_label(
            "Capturing through Core Audio process taps. No extra devices needed."
            if _tap_ok else
            f"Process taps unavailable ({_tap_why}). The fallback below needs BlackHole.",
            20, y, PW, 16,
        )
        pane.addSubview_(self._delegate._capture_status)
        y -= 30

        btn_agg = _make_button("Create Recording Device", 20, y, 210, 28,
                               "createRecordingDevice:", self._delegate)
        pane.addSubview_(btn_agg)
        # Left empty: refresh_status is the single writer, and show() calls
        # it immediately after this, so nothing is ever painted blank.
        self._delegate._aggregate_status = _make_status(238, y + 5, PW - 220)
        pane.addSubview_(self._delegate._aggregate_status)
        y -= 36

        btn_mo = _make_button("Create Monitoring Device", 20, y, 210, 28,
                              "createMonitoringDevice:", self._delegate)
        pane.addSubview_(btn_mo)
        self._delegate._multiout_status = _make_status(238, y + 5, PW - 220)
        pane.addSubview_(self._delegate._multiout_status)
        y -= 50

        pane.addSubview_(_make_label(
            "Legacy fallback, for Macs where process taps are unavailable.\n"
            "Builds CoreAudio aggregates from BlackHole 2ch and your built-in\n"
            "microphone and speakers. Not needed when taps are working.",
            20, y, PW, 46,
        ))

        # ================================================================== #
        # Tab 3: Transcription
        # ================================================================== #
        pane = _make_tab("Transcription")
        y = 340
        from AppKit import NSPopUpButton, NSButton as _NSBtn

        pane.addSubview_(_make_label("Engine", 20, y, 300, 22, bold=True))
        y -= 32

        self._delegate._engine_status = _make_status(20, y, PW)
        self._delegate._engine_status.setStringValue_(_ENGINE_BLURB)
        pane.addSubview_(self._delegate._engine_status)
        y -= 28

        live_check = _NSBtn.alloc().initWithFrame_(NSMakeRect(20, y, PW, 20))
        live_check.setButtonType_(3)   # NSButtonTypeSwitch
        live_check.setTitle_("Show live transcript while recording")
        live_check.setState_(1 if cfg.get("live_preview") else 0)
        live_check.setTarget_(self._delegate)
        live_check.setAction_("toggleLivePreview:")
        pane.addSubview_(live_check)
        y -= 42

        pane.addSubview_(_make_label("Speakers", 20, y, 300, 22, bold=True))
        y -= 24
        pane.addSubview_(_make_label(
            "Runs on the Neural Engine. No account or token needed.",
            20, y, PW, 18,
        ))
        y -= 26

        remember_check = _NSBtn.alloc().initWithFrame_(NSMakeRect(20, y, PW, 20))
        remember_check.setButtonType_(3)
        remember_check.setTitle_("Recognise returning speakers by voice")
        remember_check.setState_(1 if cfg.get("speaker_memory") else 0)
        remember_check.setTarget_(self._delegate)
        remember_check.setAction_("toggleSpeakerMemory:")
        pane.addSubview_(remember_check)
        y -= 30

        speakers_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(20, y, PW - 110, 26), False
        )
        pane.addSubview_(speakers_popup)
        self._delegate._speakers_popup = speakers_popup
        forget_btn = _make_button("Forget", PW - 84, y + 1, 100, 24,
                                  "forgetSpeaker:", self._delegate)
        pane.addSubview_(forget_btn)
        self._delegate._forget_btn = forget_btn
        y -= 38

        pane.addSubview_(_make_label("AI Models", 20, y, 300, 22, bold=True))
        y -= 36

        pane.addSubview_(_make_button("Download Models", 20, y, 160, 28,
                                     "downloadModels:", self._delegate))
        self._delegate._deps_status = _make_status(190, y + 5, PW - 170)
        pane.addSubview_(self._delegate._deps_status)

        # ================================================================== #
        # Tab 4: Output
        # ================================================================== #
        pane = _make_tab("Output")
        y = 340
        from AppKit import NSButton as _NSButton

        pane.addSubview_(_make_label("Transcript Folder", 20, y, 300, 22, bold=True))
        y -= 36

        current_output = cfg.get("output_dir")
        self._delegate._output_field = _make_text_field(
            20, y, PW - 100, 24, placeholder="~/overheard/transcripts/",
        )
        self._delegate._output_field.setStringValue_(current_output)
        pane.addSubview_(self._delegate._output_field)
        pane.addSubview_(_make_button("Browse...", PW - 74, y, 90, 24,
                                     "browseOutputFolder:", self._delegate))
        y -= 28

        self._delegate._output_status = _make_status(20, y, PW)
        pane.addSubview_(self._delegate._output_status)
        y -= 40

        keep_btn = _NSButton.alloc().initWithFrame_(NSMakeRect(20, y, PW, 20))
        keep_btn.setButtonType_(3)
        keep_btn.setTitle_("Keep audio recordings after transcription")
        keep_btn.setState_(1 if cfg.get("keep_recordings") else 0)
        keep_btn.setTarget_(self._delegate)
        keep_btn.setAction_("toggleKeepRecordings:")
        pane.addSubview_(keep_btn)
        self._delegate._keep_recordings_btn = keep_btn

        # ================================================================== #
        # Tab 5: Integrations
        # ================================================================== #
        pane = _make_tab("Integrations")
        y = 340
        obsidian_enabled = cfg.get("obsidian_enabled")

        pane.addSubview_(_make_label("Obsidian", 20, y, 300, 22, bold=True))
        y -= 30

        obs_check = _NSButton.alloc().initWithFrame_(NSMakeRect(20, y, PW, 20))
        obs_check.setButtonType_(3)
        obs_check.setTitle_("Save transcripts to Obsidian vault")
        obs_check.setState_(1 if obsidian_enabled else 0)
        obs_check.setTarget_(self._delegate)
        obs_check.setAction_("toggleObsidian:")
        pane.addSubview_(obs_check)
        self._delegate._obsidian_check = obs_check
        y -= 32

        pane.addSubview_(_make_label("Vault:", 20, y, 60, 20))
        obs_vault_field = _make_text_field(84, y, PW - 160, 24,
                                           placeholder="/Users/you/Documents/MyVault")
        obs_vault_field.setStringValue_(cfg.get("obsidian_vault"))
        obs_vault_field.setEnabled_(obsidian_enabled)
        pane.addSubview_(obs_vault_field)
        self._delegate._obsidian_vault_field = obs_vault_field
        obs_browse_btn = _make_button("Browse...", PW - 70, y, 86, 24,
                                     "browseObsidianVault:", self._delegate)
        obs_browse_btn.setEnabled_(obsidian_enabled)
        pane.addSubview_(obs_browse_btn)
        self._delegate._obsidian_browse_btn = obs_browse_btn
        y -= 32

        pane.addSubview_(_make_label("Inbox:", 20, y, 60, 20))
        obs_inbox_field = _make_text_field(84, y, 180, 24, placeholder="01_Inbox")
        obs_inbox_field.setStringValue_(cfg.get("obsidian_inbox"))
        obs_inbox_field.setEnabled_(obsidian_enabled)
        obs_inbox_field.setTarget_(self._delegate)
        obs_inbox_field.setAction_("saveObsidianInbox:")
        pane.addSubview_(obs_inbox_field)
        self._delegate._obsidian_inbox_field = obs_inbox_field
        y -= 44

        pane.addSubview_(_make_label("Calendar", 20, y, 300, 22, bold=True))
        y -= 36

        pane.addSubview_(_make_button("Connect Calendar", 20, y, 160, 28,
                                     "connectCalendar:", self._delegate))
        self._delegate._calendar_status = _make_status(190, y + 5, PW - 170)
        self._delegate._calendar_status.setStringValue_(
            "Autofills meeting name and attendees"
        )
        pane.addSubview_(self._delegate._calendar_status)
        y -= 44

        pane.addSubview_(_make_label("Speaker name:", 20, y, 110, 20))
        local_speaker_field = _make_text_field(134, y, 160, 24, placeholder="Don")
        local_speaker_field.setStringValue_(cfg.get("local_speaker_name"))
        local_speaker_field.setTarget_(self._delegate)
        local_speaker_field.setAction_("saveLocalSpeakerName:")
        pane.addSubview_(local_speaker_field)
        self._delegate._local_speaker_field = local_speaker_field
        pane.addSubview_(_make_label("(mic attribution)", 300, y, 130, 20))

        # Last statement in the method, deliberately. Anything above can fail,
        # and until this runs the panel is not built.
        self._built = True
