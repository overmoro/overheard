"""Meeting Details panel — shown after recording stops, before transcription begins."""

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSPopUpButton,
    NSScrollView,
    NSTableColumn,
    NSTableView,
    NSTextField,
)
from Foundation import NSObject, NSMutableArray

try:
    from AppKit import (
        NSWindowStyleMaskTitled,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
    )
    _STYLE = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
except ImportError:
    from AppKit import NSTitledWindowMask, NSClosableWindowMask, NSMiniaturizableWindowMask
    _STYLE = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask

WIN_W = 480
WIN_H = 460

SOURCE_OPTIONS = [
    ("In-person",        "in-person"),
    ("Zoom",             "zoom"),
    ("Google Meet",      "meet"),
    ("Microsoft Teams",  "teams"),
    ("Other",            "other"),
]

# Display labels (for the popup)
_SOURCE_LABELS = [label for label, _ in SOURCE_OPTIONS]
_SOURCE_KEYS   = [key   for _, key   in SOURCE_OPTIONS]


@dataclass
class MeetingDetails:
    name: str
    source: str               # 'zoom', 'teams', 'meet', 'in-person', 'other'
    location: str
    attendees: list[str]      # ordered — index maps to SPEAKER_00, SPEAKER_01...
    date: datetime = field(default_factory=datetime.now)


def make_filename(details: MeetingDetails) -> str:
    """Generate a canonical filename from MeetingDetails."""
    date = details.date.strftime("%Y-%m-%d")
    source = details.source.replace(" ", "-").lower()
    name = re.sub(r"[^\w\s-]", "", details.name).strip().replace(" ", "-").lower()
    return f"{date}_{source}_{name}.md"


def _make_label(text: str, x: float, y: float, w: float, h: float, bold=False) -> NSTextField:
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    if bold:
        f.setFont_(NSFont.boldSystemFontOfSize_(13))
    return f


def _make_text_field(x: float, y: float, w: float, h: float, placeholder: str = "") -> NSTextField:
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setPlaceholderString_(placeholder)
    return f


# ---------------------------------------------------------------------------
# Table data source / delegate — handles attendee rows
# ---------------------------------------------------------------------------

class _AttendeeDataSource(NSObject):
    """NSTableView data source + delegate for the speaker/name table."""

    def init(self):
        self = objc.super(_AttendeeDataSource, self).init()
        if self is None:
            return None
        # List of [speaker_id, name_str]
        self._rows: list[list[str]] = []
        return self

    def setRows_(self, rows: list[list[str]]) -> None:
        self._rows = rows

    def numberOfRowsInTableView_(self, table_view) -> int:
        return len(self._rows)

    def tableView_objectValueForTableColumn_row_(self, table_view, column, row):
        if row >= len(self._rows):
            return ""
        col_id = column.identifier()
        if col_id == "speaker":
            return self._rows[row][0]
        if col_id == "name":
            return self._rows[row][1]
        return ""

    def tableView_setObjectValue_forTableColumn_row_(self, table_view, value, column, row):
        if row >= len(self._rows):
            return
        col_id = column.identifier()
        if col_id == "name":
            self._rows[row][1] = value or ""

    def names(self) -> list[str]:
        """Return attendee name list in speaker order."""
        return [row[1] for row in self._rows]


# ---------------------------------------------------------------------------
# Delegate — button actions
# ---------------------------------------------------------------------------

class _DetailsDelegate(NSObject):

    def initWithCallback_discardCallback_(self, callback, discard_callback):
        self = objc.super(_DetailsDelegate, self).init()
        if self is None:
            return None
        self._callback = callback
        self._discard_callback = discard_callback
        return self

    def onStartTranscription_(self, sender):
        name = self._name_field.stringValue().strip() or "meeting"
        source_idx = self._source_popup.indexOfSelectedItem()
        source = _SOURCE_KEYS[source_idx] if 0 <= source_idx < len(_SOURCE_KEYS) else "in-person"
        location = self._location_field.stringValue().strip()
        attendees = self._data_source.names()
        date = datetime.now()

        details = MeetingDetails(
            name=name,
            source=source,
            location=location,
            attendees=attendees,
            date=date,
        )

        self._window.orderOut_(None)
        if self._callback:
            threading.Thread(target=self._callback, args=(details,), daemon=True).start()

    def onSourceChanged_(self, sender):
        """Auto-fill location when source changes."""
        idx = sender.indexOfSelectedItem()
        source = _SOURCE_KEYS[idx] if 0 <= idx < len(_SOURCE_KEYS) else "in-person"
        from overheard.meeting import infer_location
        loc = infer_location(source)
        if loc:
            self._location_field.setStringValue_(loc)

    def onDiscard_(self, sender):
        """First click — reveal the red confirm button."""
        self._confirm_discard_btn.setHidden_(False)
        sender.setEnabled_(False)

    def onConfirmDiscard_(self, sender):
        """Second click — actually discard the recording."""
        self._window.orderOut_(None)
        # Reset discard button state for next time
        self._discard_btn.setEnabled_(True)
        self._confirm_discard_btn.setHidden_(True)
        if self._discard_callback:
            self._discard_callback()


# ---------------------------------------------------------------------------
# DetailsPanel
# ---------------------------------------------------------------------------

class DetailsPanel:
    """Floating panel to collect meeting metadata before transcription.

    Usage:
        panel = DetailsPanel(callback=my_func)
        panel.show(
            name="Weekly Standup",
            source="zoom",
            location="Zoom",
            attendees=["Don Reddin", "John Smith"],
            speaker_count=2,
        )

    callback receives a MeetingDetails instance.
    """

    def __init__(self, callback, discard_callback=None):
        self._callback = callback
        self._discard_callback = discard_callback
        self._window = None
        self._delegate = None
        self._data_source = None

    def show(
        self,
        name: str = "",
        source: str = "in-person",
        location: str = "",
        attendees: list[str] | None = None,
        speaker_count: int = 2,
    ) -> None:
        if self._window is None:
            self._build()

        # Pre-fill fields
        self._delegate._name_field.setStringValue_(name)
        self._delegate._location_field.setStringValue_(location)

        # Set source popup
        display_label = next(
            (lbl for lbl, key in SOURCE_OPTIONS if key == source),
            "In-person",
        )
        self._delegate._source_popup.selectItemWithTitle_(display_label)

        # Build attendee rows — pre-fill from calendar, pad to speaker_count
        rows = []
        attendees = attendees or []
        for i in range(max(speaker_count, len(attendees))):
            speaker_id = f"SPEAKER_{i:02d}"
            name_val = attendees[i] if i < len(attendees) else ""
            rows.append([speaker_id, name_val])
        self._data_source.setRows_(rows)
        self._delegate._table_view.reloadData()

        self._window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def _build(self) -> None:
        self._window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIN_W, WIN_H),
            _STYLE,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_("Meeting Details")
        self._window.center()
        self._window.setFloatingPanel_(True)

        self._delegate = _DetailsDelegate.alloc().initWithCallback_discardCallback_(self._callback, self._discard_callback)
        self._delegate._window = self._window
        self._data_source = _AttendeeDataSource.alloc().init()
        self._delegate._data_source = self._data_source

        cv = self._window.contentView()
        y = WIN_H - 40

        # ---- Heading --------------------------------------------------------
        cv.addSubview_(_make_label("Meeting Details", 20, y, 400, 22, bold=True))
        y -= 36

        # ---- Meeting Name ---------------------------------------------------
        cv.addSubview_(_make_label("Meeting Name", 20, y, 160, 18))
        y -= 26
        name_field = _make_text_field(20, y, WIN_W - 40, 24, placeholder="Weekly Standup")
        cv.addSubview_(name_field)
        self._delegate._name_field = name_field
        y -= 36

        # ---- Source ---------------------------------------------------------
        cv.addSubview_(_make_label("Source", 20, y, 160, 18))
        y -= 26
        popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(20, y, 200, 26), False
        )
        for label, _ in SOURCE_OPTIONS:
            popup.addItemWithTitle_(label)
        popup.setTarget_(self._delegate)
        popup.setAction_("onSourceChanged:")
        cv.addSubview_(popup)
        self._delegate._source_popup = popup
        y -= 36

        # ---- Location -------------------------------------------------------
        cv.addSubview_(_make_label("Location", 20, y, 160, 18))
        y -= 26
        loc_field = _make_text_field(20, y, WIN_W - 40, 24, placeholder="e.g. Zoom, Room 3A")
        cv.addSubview_(loc_field)
        self._delegate._location_field = loc_field
        y -= 44

        # ---- Attendees table ------------------------------------------------
        cv.addSubview_(_make_label("Attendees (Speaker → Name)", 20, y, 400, 18, bold=False))
        y -= 26

        table_h = 130
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, y - table_h, WIN_W - 40, table_h))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(2)  # NSBezelBorder

        table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_W - 40, table_h))

        col_speaker = NSTableColumn.alloc().initWithIdentifier_("speaker")
        col_speaker.setTitle_("Speaker")
        col_speaker.setWidth_(120)
        col_speaker.setEditable_(False)

        col_name = NSTableColumn.alloc().initWithIdentifier_("name")
        col_name.setTitle_("Name")
        col_name.setWidth_(WIN_W - 40 - 120 - 20)
        col_name.setEditable_(True)

        table.addTableColumn_(col_speaker)
        table.addTableColumn_(col_name)
        table.setDataSource_(self._data_source)
        table.setDelegate_(self._data_source)
        table.setUsesAlternatingRowBackgroundColors_(True)

        scroll.setDocumentView_(table)
        cv.addSubview_(scroll)
        self._delegate._table_view = table
        y -= (table_h + 20)

        # ---- Buttons row ----------------------------------------------------
        # Discard (left)
        discard_btn = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 100, 32))
        discard_btn.setTitle_("🗑 Discard")
        discard_btn.setBezelStyle_(1)
        discard_btn.setTarget_(self._delegate)
        discard_btn.setAction_("onDiscard:")
        cv.addSubview_(discard_btn)
        self._delegate._discard_btn = discard_btn

        # Confirm discard — hidden until first click, red destructive style
        confirm_btn = NSButton.alloc().initWithFrame_(NSMakeRect(128, y, 160, 32))
        confirm_btn.setTitle_("⚠️ Yes, delete it")
        confirm_btn.setBezelStyle_(1)
        confirm_btn.setTarget_(self._delegate)
        confirm_btn.setAction_("onConfirmDiscard:")
        confirm_btn.setContentTintColor_(NSColor.systemRedColor())
        confirm_btn.setHidden_(True)
        cv.addSubview_(confirm_btn)
        self._delegate._confirm_discard_btn = confirm_btn

        # Start Transcription (right)
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(WIN_W - 200, y, 180, 32))
        btn.setTitle_("Start Transcription")
        btn.setBezelStyle_(1)
        btn.setTarget_(self._delegate)
        btn.setAction_("onStartTranscription:")
        cv.addSubview_(btn)
