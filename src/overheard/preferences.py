"""Preferences window for Overheard.

Opens as a standalone NSPanel so it doesn't block the rumps run loop.
Sections: General, Audio, Transcription, Output, Integrations.
"""

import os
import subprocess
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
from Foundation import NSObject, NSString

from overheard import config as cfg
from overheard.audio import create_aggregate_device, create_multi_output_device

# Window dimensions
WIN_W = 500
WIN_H = 460   # tall enough for the Transcription pane's engine picker

# Engine choice, in popup-menu order
_ENGINES = ["parakeet", "whisper"]
_ENGINE_TITLES = [
    "Parakeet TDT 0.6B v3 (fast, on GPU)",
    "Whisper large-v3 (slower, on CPU)",
]
_ENGINE_BLURB = {
    "parakeet": "Runs on the Apple Silicon GPU. Supports live transcription.",
    "whisper": "Runs on CPU, far slower, but handles 99 languages.",
}


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

class _PreferencesDelegate(NSObject):

    def initWithWindow_(self, window):
        self = objc.super(_PreferencesDelegate, self).init()
        if self is None:
            return None
        self._window = window
        return self

    # ---- General -----------------------------------------------------------

    def openTranscripts_(self, sender):
        from overheard import config as _cfg
        from pathlib import Path as _Path
        d = _Path(_cfg.get("output_dir"))
        d.mkdir(parents=True, exist_ok=True)
        os.system(f'open "{d}"')

    def quitApp_(self, sender):
        import rumps
        rumps.quit_application()

    # ---- Audio Setup -------------------------------------------------------

    def createRecordingDevice_(self, sender):
        self._aggregate_status.setStringValue_("Creating...")
        threading.Thread(target=self._do_create_aggregate, daemon=True).start()

    def _do_create_aggregate(self):
        ok, msg = create_aggregate_device()
        self._aggregate_status.setStringValue_(f"{'✓' if ok else '✗'} {msg}")

    def createMonitoringDevice_(self, sender):
        self._multiout_status.setStringValue_("Creating...")
        threading.Thread(target=self._do_create_multiout, daemon=True).start()

    def _do_create_multiout(self):
        ok, msg = create_multi_output_device()
        self._multiout_status.setStringValue_(f"{'✓' if ok else '✗'} {msg}")

    # ---- Engine ------------------------------------------------------------

    def selectEngine_(self, sender):
        """Radio-style engine selection driven by the popup button."""
        engine = "whisper" if sender.indexOfSelectedItem() == 1 else "parakeet"
        cfg.set_value("engine", engine)
        self._engine_status.setStringValue_(_ENGINE_BLURB[engine])
        # Live preview is streamed by Parakeet, so it's meaningless on Whisper
        self._live_check.setEnabled_(engine == "parakeet")

    def toggleLivePreview_(self, sender):
        cfg.set_value("live_preview", bool(sender.state()))

    # ---- Speakers ----------------------------------------------------------

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

    def forgetSpeaker_(self, sender):
        from overheard.speakers import SpeakerLibrary

        index = self._speakers_popup.indexOfSelectedItem()
        names = getattr(self, "_known_speaker_names", [])
        if not (0 <= index < len(names)):
            return
        SpeakerLibrary().forget(names[index])
        self._refresh_speakers()

    # ---- Dependencies ------------------------------------------------------

    def downloadModels_(self, sender):
        self._deps_status.setStringValue_("Downloading... (this may take several minutes)")
        threading.Thread(target=self._do_download_models, daemon=True).start()

    def _do_download_models(self):
        try:
            # Only fetch the model for the engine actually in use. Whisper
            # large-v3 is a 3 GB download nobody on Parakeet needs.
            engine = cfg.get("engine")
            if engine == "whisper":
                self._deps_status.setStringValue_("Downloading whisper large-v3...")
                code = ("import whisperx; "
                        "whisperx.load_model('large-v3', 'cpu', compute_type='int8')")
                label = "Whisper"
            else:
                self._deps_status.setStringValue_("Downloading Parakeet TDT v3...")
                from overheard.asr import PARAKEET_MODEL
                code = ("from parakeet_mlx import from_pretrained; "
                        f"from_pretrained({PARAKEET_MODEL!r})")
                label = "Parakeet"

            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                self._deps_status.setStringValue_(
                    f"✗ {label} failed: {result.stderr[:120]}"
                )
                return

            # Warm the diarization models too. The helper fetches them on first
            # use with no credentials, so this only saves the wait later.
            from overheard.helper import helper_path
            binary = helper_path()
            if binary is not None:
                self._deps_status.setStringValue_("Downloading speaker models...")
                result = subprocess.run(
                    [str(binary), "download"],
                    capture_output=True, text=True, timeout=1800,
                )
                if result.returncode != 0:
                    self._deps_status.setStringValue_(
                        f"✓ {label} done  ✗ Speaker models: {result.stderr[:80]}"
                    )
                    return

            self._deps_status.setStringValue_("✓ All models downloaded")
        except subprocess.TimeoutExpired:
            self._deps_status.setStringValue_("✗ Download timed out")
        except Exception as e:
            self._deps_status.setStringValue_(f"✗ {e}")

    # ---- Output Folder -----------------------------------------------------

    def toggleKeepRecordings_(self, sender):
        cfg.set_value("keep_recordings", bool(sender.state()))

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

    def toggleObsidian_(self, sender):
        enabled = bool(sender.state())
        cfg.set_value("obsidian_enabled", enabled)
        self._obsidian_vault_field.setEnabled_(enabled)
        self._obsidian_inbox_field.setEnabled_(enabled)
        self._obsidian_browse_btn.setEnabled_(enabled)

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

    def saveObsidianInbox_(self, sender):
        val = self._obsidian_inbox_field.stringValue().strip()
        cfg.set_value("obsidian_inbox", val or "01_Inbox")

    def saveLocalSpeakerName_(self, sender):
        val = self._local_speaker_field.stringValue().strip()
        cfg.set_value("local_speaker_name", val or "Don")

    # ---- Integrations, Calendar -------------------------------------------

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
                self._calendar_status.setStringValue_("✓ Calendar access granted")
            else:
                err = (result.stderr or "Permission denied").strip()[:80]
                self._calendar_status.setStringValue_(f"✗ {err}")
        except subprocess.TimeoutExpired:
            self._calendar_status.setStringValue_("✗ Timed out. Check System Settings → Privacy → Calendars")
        except Exception as e:
            self._calendar_status.setStringValue_(f"✗ {e}")


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
        self._window = None
        self._delegate = None

    def show(self) -> None:
        if self._window is None:
            self._build()
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
        BOTTOM = 20       # y baseline inside each pane

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
        y -= 36

        btn_agg = _make_button("Create Recording Device", 20, y, 210, 28,
                               "createRecordingDevice:", self._delegate)
        pane.addSubview_(btn_agg)
        self._delegate._aggregate_status = _make_status(238, y + 5, PW - 220)
        self._delegate._aggregate_status.setStringValue_(_device_exists("Meeting Capture"))
        pane.addSubview_(self._delegate._aggregate_status)
        y -= 36

        btn_mo = _make_button("Create Monitoring Device", 20, y, 210, 28,
                              "createMonitoringDevice:", self._delegate)
        pane.addSubview_(btn_mo)
        self._delegate._multiout_status = _make_status(238, y + 5, PW - 220)
        self._delegate._multiout_status.setStringValue_(_device_exists("Meeting Monitor"))
        pane.addSubview_(self._delegate._multiout_status)
        y -= 50

        pane.addSubview_(_make_label(
            "Creates CoreAudio aggregate devices combining BlackHole 2ch\n"
            "and your MacBook microphone/speakers for meeting capture.",
            20, y, PW, 32,
        ))

        # ================================================================== #
        # Tab 3: Transcription
        # ================================================================== #
        pane = _make_tab("Transcription")
        y = 340
        from AppKit import NSPopUpButton, NSButton as _NSBtn

        current_engine = cfg.get("engine")
        if current_engine not in _ENGINES:
            current_engine = "parakeet"

        pane.addSubview_(_make_label("Engine", 20, y, 300, 22, bold=True))
        y -= 32

        engine_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(20, y, PW - 20, 26), False
        )
        engine_popup.addItemsWithTitles_(_ENGINE_TITLES)
        engine_popup.selectItemAtIndex_(_ENGINES.index(current_engine))
        engine_popup.setTarget_(self._delegate)
        engine_popup.setAction_("selectEngine:")
        pane.addSubview_(engine_popup)
        self._delegate._engine_popup = engine_popup
        y -= 24

        self._delegate._engine_status = _make_status(20, y, PW)
        self._delegate._engine_status.setStringValue_(_ENGINE_BLURB[current_engine])
        pane.addSubview_(self._delegate._engine_status)
        y -= 28

        live_check = _NSBtn.alloc().initWithFrame_(NSMakeRect(20, y, PW, 20))
        live_check.setButtonType_(3)   # NSButtonTypeSwitch
        live_check.setTitle_("Show live transcript while recording")
        live_check.setState_(1 if cfg.get("live_preview") else 0)
        live_check.setTarget_(self._delegate)
        live_check.setAction_("toggleLivePreview:")
        live_check.setEnabled_(current_engine == "parakeet")
        pane.addSubview_(live_check)
        self._delegate._live_check = live_check
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
        self._delegate._refresh_speakers()
        y -= 38

        pane.addSubview_(_make_label("AI Models", 20, y, 300, 22, bold=True))
        y -= 36

        pane.addSubview_(_make_button("Download Models", 20, y, 160, 28,
                                     "downloadModels:", self._delegate))
        self._delegate._deps_status = _make_status(190, y + 5, PW - 170)
        self._delegate._deps_status.setStringValue_("Downloads on first use")
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
