"""Preferences window for Overheard.

Opens as a standalone NSPanel so it doesn't block the rumps run loop.
Sections: Audio Setup, Hugging Face Token, Dependencies, Output Folder.
"""

import os
import subprocess
import threading
from pathlib import Path

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
    NSTextField,
    NSOpenPanel,
)
from Foundation import NSObject

from overheard import config as cfg
from overheard.audio import create_aggregate_device, create_multi_output_device

# Window dimensions
WIN_W = 480
WIN_H = 720   # taller to accommodate Integrations section


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
# Delegate — NSObject subclass handles all button actions
# ---------------------------------------------------------------------------

class _PreferencesDelegate(NSObject):

    def initWithWindow_(self, window):
        self = objc.super(_PreferencesDelegate, self).init()
        if self is None:
            return None
        self._window = window
        return self

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

    # ---- HF Token ----------------------------------------------------------

    def saveToken_(self, sender):
        token = self._token_field.stringValue().strip()
        if not token:
            self._token_status.setStringValue_("✗ Token is empty")
            return
        _write_hf_token_to_zshrc(token)
        cfg.set_value("hf_token", token)
        os.environ["HF_TOKEN"] = token
        self._token_status.setStringValue_("✓ Saved")

    # ---- Dependencies ------------------------------------------------------

    def downloadModels_(self, sender):
        self._deps_status.setStringValue_("Downloading... (this may take several minutes)")
        threading.Thread(target=self._do_download_models, daemon=True).start()

    def _do_download_models(self):
        try:
            self._deps_status.setStringValue_("Downloading whisper large-v3...")
            result = subprocess.run(
                ["python3", "-c",
                 "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8')"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                self._deps_status.setStringValue_(
                    f"✗ Whisper failed: {result.stderr[:120]}"
                )
                return

            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                self._deps_status.setStringValue_("Downloading pyannote diarization model...")
                result = subprocess.run(
                    ["python3", "-c",
                     "from pyannote.audio import Pipeline; "
                     f"Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',"
                     f" use_auth_token='{hf_token}')"],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    self._deps_status.setStringValue_(
                        f"✓ Whisper done  ✗ Pyannote failed: {result.stderr[:80]}"
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

    # ---- Integrations — Obsidian -------------------------------------------

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


# ---------------------------------------------------------------------------
# HF Token helpers
# ---------------------------------------------------------------------------

def _write_hf_token_to_zshrc(token: str) -> None:
    """Write or update the HF_TOKEN export line in ~/.zshrc."""
    zshrc = Path.home() / ".zshrc"
    export_line = f'export HF_TOKEN="{token}"'
    lines = []
    replaced = False

    if zshrc.exists():
        with open(zshrc) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("export HF_TOKEN="):
                lines[i] = export_line + "\n"
                replaced = True
                break

    if not replaced:
        lines.append(export_line + "\n")

    with open(zshrc, "w") as f:
        f.writelines(lines)


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

        y = WIN_H - 40

        # ------------------------------------------------------------------ #
        # Section 1: Audio Setup
        # ------------------------------------------------------------------ #
        y -= 10
        cv.addSubview_(_make_label("Audio Setup", 20, y, 220, 22, bold=True))
        y -= 30

        btn_agg = _make_button(
            "Create Recording Device", 20, y, 200, 28,
            "createRecordingDevice:", self._delegate,
        )
        cv.addSubview_(btn_agg)
        self._delegate._aggregate_status = _make_status(228, y + 5)
        self._delegate._aggregate_status.setStringValue_(_device_exists("Meeting Capture"))
        cv.addSubview_(self._delegate._aggregate_status)
        y -= 36

        btn_mo = _make_button(
            "Create Monitoring Device", 20, y, 200, 28,
            "createMonitoringDevice:", self._delegate,
        )
        cv.addSubview_(btn_mo)
        self._delegate._multiout_status = _make_status(228, y + 5)
        self._delegate._multiout_status.setStringValue_(_device_exists("Meeting Monitor"))
        cv.addSubview_(self._delegate._multiout_status)
        y -= 44

        cv.addSubview_(_make_label("─" * 62, 20, y, WIN_W - 40, 16))
        y -= 22

        # ------------------------------------------------------------------ #
        # Section 2: Hugging Face Token
        # ------------------------------------------------------------------ #
        cv.addSubview_(_make_label("Hugging Face Token", 20, y, 300, 22, bold=True))
        y -= 30

        self._delegate._token_field = _make_text_field(
            20, y, 300, 24,
            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            secure=True,
        )
        if os.environ.get("HF_TOKEN"):
            self._delegate._token_field.setStringValue_(os.environ["HF_TOKEN"])
        cv.addSubview_(self._delegate._token_field)

        cv.addSubview_(_make_button(
            "Save", 330, y, 80, 24,
            "saveToken:", self._delegate,
        ))
        y -= 26

        self._delegate._token_status = _make_status(20, y)
        self._delegate._token_status.setStringValue_(
            "✓ Token is currently set" if os.environ.get("HF_TOKEN") else "No token set"
        )
        cv.addSubview_(self._delegate._token_status)
        y -= 40

        cv.addSubview_(_make_label("─" * 62, 20, y, WIN_W - 40, 16))
        y -= 22

        # ------------------------------------------------------------------ #
        # Section 3: Dependencies
        # ------------------------------------------------------------------ #
        cv.addSubview_(_make_label("Dependencies", 20, y, 300, 22, bold=True))
        y -= 30

        cv.addSubview_(_make_button(
            "Download Models", 20, y, 160, 28,
            "downloadModels:", self._delegate,
        ))
        self._delegate._deps_status = _make_status(190, y + 5)
        self._delegate._deps_status.setStringValue_("large-v3 + pyannote")
        cv.addSubview_(self._delegate._deps_status)
        y -= 44

        cv.addSubview_(_make_label("─" * 62, 20, y, WIN_W - 40, 16))
        y -= 22

        # ------------------------------------------------------------------ #
        # Section 4: Output Folder
        # ------------------------------------------------------------------ #
        cv.addSubview_(_make_label("Output Folder", 20, y, 300, 22, bold=True))
        y -= 30

        current_output = cfg.get("output_dir", str(Path.home() / "meeting-transcripts"))
        self._delegate._output_field = _make_text_field(
            20, y, 340, 24,
            placeholder="~/overheard/transcripts/",
        )
        self._delegate._output_field.setStringValue_(current_output)
        cv.addSubview_(self._delegate._output_field)

        cv.addSubview_(_make_button(
            "Browse...", 370, y, 90, 24,
            "browseOutputFolder:", self._delegate,
        ))
        y -= 26

        self._delegate._output_status = _make_status(20, y)
        cv.addSubview_(self._delegate._output_status)
        y -= 30

        from AppKit import NSButton as _NSButton
        keep_btn = _NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 300, 20))
        keep_btn.setButtonType_(3)  # NSSwitchButton / checkbox
        keep_btn.setTitle_("Keep audio recordings after transcription")
        keep_btn.setState_(1 if cfg.get("keep_recordings", False) else 0)
        keep_btn.setTarget_(self._delegate)
        keep_btn.setAction_("toggleKeepRecordings:")
        cv.addSubview_(keep_btn)
        self._delegate._keep_recordings_btn = keep_btn
        y -= 40

        cv.addSubview_(_make_label("─" * 62, 20, y, WIN_W - 40, 16))
        y -= 22

        # ------------------------------------------------------------------ #
        # Section 5: Integrations
        # ------------------------------------------------------------------ #
        cv.addSubview_(_make_label("Integrations", 20, y, 300, 22, bold=True))
        y -= 30

        # Obsidian enable checkbox
        obsidian_enabled = bool(cfg.get("obsidian_enabled", False))
        obs_check = _NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 300, 20))
        obs_check.setButtonType_(3)
        obs_check.setTitle_("Save transcripts to Obsidian vault")
        obs_check.setState_(1 if obsidian_enabled else 0)
        obs_check.setTarget_(self._delegate)
        obs_check.setAction_("toggleObsidian:")
        cv.addSubview_(obs_check)
        self._delegate._obsidian_check = obs_check
        y -= 30

        # Vault path
        cv.addSubview_(_make_label("Vault path:", 20, y, 90, 20))
        obs_vault_field = _make_text_field(
            115, y, WIN_W - 215, 24,
            placeholder="/Users/you/Documents/MyVault",
        )
        obs_vault_field.setStringValue_(cfg.get("obsidian_vault", ""))
        obs_vault_field.setEnabled_(obsidian_enabled)
        cv.addSubview_(obs_vault_field)
        self._delegate._obsidian_vault_field = obs_vault_field

        obs_browse_btn = _make_button(
            "Browse...", WIN_W - 94, y, 74, 24,
            "browseObsidianVault:", self._delegate,
        )
        obs_browse_btn.setEnabled_(obsidian_enabled)
        cv.addSubview_(obs_browse_btn)
        self._delegate._obsidian_browse_btn = obs_browse_btn
        y -= 32

        # Inbox folder
        cv.addSubview_(_make_label("Inbox folder:", 20, y, 90, 20))
        obs_inbox_field = _make_text_field(
            115, y, 200, 24,
            placeholder="01_Inbox",
        )
        obs_inbox_field.setStringValue_(cfg.get("obsidian_inbox", "01_Inbox"))
        obs_inbox_field.setEnabled_(obsidian_enabled)
        obs_inbox_field.setTarget_(self._delegate)
        obs_inbox_field.setAction_("saveObsidianInbox:")
        cv.addSubview_(obs_inbox_field)
        self._delegate._obsidian_inbox_field = obs_inbox_field
        y -= 32

        # Local speaker name (mic attribution)
        cv.addSubview_(_make_label("Your name:", 20, y, 90, 20))
        local_speaker_field = _make_text_field(
            115, y, 200, 24,
            placeholder="Don",
        )
        local_speaker_field.setStringValue_(cfg.get("local_speaker_name", "Don"))
        local_speaker_field.setTarget_(self._delegate)
        local_speaker_field.setAction_("saveLocalSpeakerName:")
        cv.addSubview_(local_speaker_field)
        self._delegate._local_speaker_field = local_speaker_field
        cv.addSubview_(_make_label("(used for mic channel attribution)", 320, y, 140, 20))
