"""Overheard — menu bar popover transport UI (styled)."""

import math
import threading

import objc
from AppKit import (
    NSButton,
    NSColor,
    NSFont,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPopover,
    NSTextField,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSView,
    NSViewController,
    NSVisualEffectView,
)
from Foundation import NSObject

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
POP_W  = 300
POP_H  = 280

_HDR_H  = 52          # coloured header height
_BTN_Y  = POP_H - _HDR_H - 80   # transport button row y
_BTN_D  = 56          # button diameter
_STATUS_Y = _BTN_Y - 28
_SEP1_Y   = _STATUS_Y - 10
_M_Y      = _SEP1_Y - 26   # meter row y
_SEP2_Y   = _M_Y - 14
_FOOTER_Y = _SEP2_Y - 32

# NSPopoverBehaviorTransient = 1, edge: NSMinYEdge = 3 (below status bar)
_NSMinYEdge = 3

# Meter geometry
_M_SEGS  = 20
_M_H     = 8
_M_LBL_W = 22
_M_BAR_X = 20 + _M_LBL_W + 6
_M_BAR_W = POP_W - _M_BAR_X - 20


# ---------------------------------------------------------------------------
# Segmented level bar
# ---------------------------------------------------------------------------

class _LevelBar(NSView):
    def init(self):
        self = objc.super(_LevelBar, self).init()
        if self is None:
            return None
        self._level: float = 0.0
        self._active: bool = False
        return self

    def setLevel_(self, v: float) -> None:
        self._level = max(0.0, min(1.0, v))
        self.setNeedsDisplay_(True)

    def setActive_(self, v: bool) -> None:
        self._active = v
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect) -> None:
        from AppKit import NSBezierPath
        b    = self.bounds()
        sw   = (b.size.width - (_M_SEGS - 1)) / _M_SEGS
        n    = int(self._level * _M_SEGS) if self._active else 0
        for i in range(_M_SEGS):
            x = i * (sw + 1)
            if i < n:
                frac = i / _M_SEGS
                if   frac < 0.60: NSColor.systemGreenColor().colorWithAlphaComponent_(0.9).setFill()
                elif frac < 0.85: NSColor.systemYellowColor().colorWithAlphaComponent_(0.9).setFill()
                else:              NSColor.systemRedColor().colorWithAlphaComponent_(0.9).setFill()
            else:
                NSColor.tertiaryLabelColor().setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, 0, sw, b.size.height), 2, 2
            ).fill()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lbl(text, x, y, w, h, size=12, bold=False, color=None,
         align=NSTextAlignmentLeft) -> NSTextField:
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setLineBreakMode_(NSLineBreakByTruncatingTail)
    f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    if color: f.setTextColor_(color)
    f.setAlignment_(align)
    return f


def _sep(parent, y):
    from AppKit import NSBox
    box = NSBox.alloc().initWithFrame_(NSMakeRect(16, y, POP_W - 32, 1))
    box.setBoxType_(2)   # NSSeparator
    parent.addSubview_(box)


def _circle_btn(symbol, x, y, d, action, target,
                tint=None, bg=None) -> NSButton:
    """Circular button with large symbol glyph."""
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, d, d))
    btn.setTitle_(symbol)
    btn.setBezelStyle_(7)            # NSBezelStyleCircular
    btn.setFont_(NSFont.systemFontOfSize_(22))
    btn.setTarget_(target)
    btn.setAction_(action)
    if tint:
        btn.setContentTintColor_(tint)
    return btn


def _footer_btn(title, x, y, w, action, target) -> NSButton:
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 22))
    btn.setTitle_(title)
    btn.setBezelStyle_(14)           # NSBezelStyleInline
    btn.setFont_(NSFont.systemFontOfSize_(12))
    btn.setTarget_(target)
    btn.setAction_(action)
    btn.setContentTintColor_(NSColor.linkColor())
    return btn


# ---------------------------------------------------------------------------
# Popover delegate
# ---------------------------------------------------------------------------

class _PopoverDelegate(NSObject):

    def initWithCallbacks_(self, cbs: dict):
        self = objc.super(_PopoverDelegate, self).init()
        if self is None:
            return None
        self._cbs = cbs
        self._popover_ref = None
        return self

    def togglePopover_(self, sender):
        pop = self._popover_ref
        if pop is None: return
        if pop.isShown():
            pop.performClose_(sender)
        else:
            pop.showRelativeToRect_ofView_preferredEdge_(
                sender.bounds(), sender, _NSMinYEdge
            )

    def onRecord_(self, sender):
        cb = self._cbs.get("record")
        if cb: threading.Thread(target=cb, daemon=True).start()

    def onPause_(self, sender):
        cb = self._cbs.get("pause")
        if cb: cb()

    def onStop_(self, sender):
        cb = self._cbs.get("stop")
        if cb: cb()

    def openTranscripts_(self, sender):
        cb = self._cbs.get("open_transcripts")
        if cb: cb()

    def openPreferences_(self, sender):
        cb = self._cbs.get("preferences")
        if cb: cb()

    def quitApp_(self, sender):
        import rumps
        rumps.quit_application()


# ---------------------------------------------------------------------------
# TransportPopover
# ---------------------------------------------------------------------------

class TransportPopover:
    """Styled popover attached to the menu bar status item button."""

    def __init__(self, callbacks: dict):
        self._delegate = _PopoverDelegate.alloc().initWithCallbacks_(callbacks)
        self._popover: NSPopover | None = None
        self._btn_record = None
        self._btn_pause  = None
        self._btn_stop   = None
        self._status_lbl = None
        self._mic_bar:  _LevelBar | None = None
        self._sys_bar:  _LevelBar | None = None
        self._mic_row:  NSView | None = None
        self._sys_row:  NSView | None = None
        self._is_multichannel = False
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hook_status_item(self, status_btn) -> None:
        status_btn.setTarget_(self._delegate)
        status_btn.setAction_("togglePopover:")

    def set_state(self, state: str, status: str = "") -> None:
        from overheard.transport import IDLE, RECORDING, PAUSED, TRANSCRIBING
        enabled = {
            IDLE:         (True,  False, False),
            RECORDING:    (False, True,  True),
            PAUSED:       (True,  False, True),
            TRANSCRIBING: (False, False, False),
        }.get(state, (True, False, False))

        self._btn_record.setEnabled_(enabled[0])
        self._btn_pause.setEnabled_(enabled[1])
        self._btn_stop.setEnabled_(enabled[2])

        # Colour the record button red while recording
        if state == RECORDING:
            self._btn_record.setContentTintColor_(NSColor.systemRedColor())
        else:
            self._btn_record.setContentTintColor_(NSColor.labelColor())

        self._status_lbl.setStringValue_(status)

        show = state in (RECORDING, PAUSED)
        self._set_meters_visible(show)
        if not show:
            self.set_levels(0.0, 0.0)

    def set_levels(self, mic_rms: float, sys_rms: float) -> None:
        if not self._mic_bar: return

        def _to_level(rms):
            if rms <= 0: return 0.0
            db = 20 * math.log10(max(rms, 1e-9))
            return max(0.0, min(1.0, (db + 60) / 60))

        self._mic_bar.setLevel_(_to_level(mic_rms))
        if self._is_multichannel and self._sys_bar:
            self._sys_bar.setLevel_(_to_level(sys_rms))

    def configure_channels(self, is_multichannel: bool) -> None:
        self._is_multichannel = is_multichannel
        if self._sys_row:
            self._sys_row.setHidden_(not is_multichannel)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_meters_visible(self, v: bool) -> None:
        if self._mic_row: self._mic_row.setHidden_(not v)
        if self._sys_row: self._sys_row.setHidden_(not (v and self._is_multichannel))
        if self._mic_bar: self._mic_bar.setActive_(v)
        if self._sys_bar: self._sys_bar.setActive_(v and self._is_multichannel)

    def _build(self) -> None:
        d = self._delegate

        # ---- Root view (vibrancy) ----------------------------------------
        root = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, POP_W, POP_H)
        )
        root.setBlendingMode_(0)   # NSVisualEffectBlendingModeBehindWindow
        root.setState_(1)          # NSVisualEffectStateActive

        # ---- Header strip ------------------------------------------------
        hdr = NSView.alloc().initWithFrame_(NSMakeRect(0, POP_H - _HDR_H, POP_W, _HDR_H))
        hdr.setWantsLayer_(True)
        hdr.layer().setBackgroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.18, 1.0).CGColor()
        )

        hdr.addSubview_(_lbl(
            "Overheard", 16, 14, 160, 22,
            size=15, bold=True,
            color=NSColor.whiteColor(),
        ))

        gear = _footer_btn("⚙  Preferences", POP_W - 120, 12, 108,
                            "openPreferences:", d)
        gear.setContentTintColor_(NSColor.colorWithWhite_alpha_(0.75, 1.0))
        hdr.addSubview_(gear)

        root.addSubview_(hdr)

        # ---- Transport buttons -------------------------------------------
        total = 3 * _BTN_D + 2 * 16
        bx    = (POP_W - total) // 2

        self._btn_record = _circle_btn(
            "⏺", bx,           _BTN_Y, _BTN_D, "onRecord:", d,
            tint=NSColor.labelColor(),
        )
        self._btn_pause = _circle_btn(
            "⏸", bx + _BTN_D + 16, _BTN_Y, _BTN_D, "onPause:", d,
            tint=NSColor.secondaryLabelColor(),
        )
        self._btn_stop = _circle_btn(
            "⏹", bx + (_BTN_D + 16) * 2, _BTN_Y, _BTN_D, "onStop:", d,
            tint=NSColor.secondaryLabelColor(),
        )
        root.addSubview_(self._btn_record)
        root.addSubview_(self._btn_pause)
        root.addSubview_(self._btn_stop)

        # ---- Status label ------------------------------------------------
        self._status_lbl = _lbl(
            "Ready", 0, _STATUS_Y, POP_W, 18,
            size=12,
            color=NSColor.secondaryLabelColor(),
            align=NSTextAlignmentCenter,
        )
        root.addSubview_(self._status_lbl)

        _sep(root, _SEP1_Y)

        # ---- Mic meter row -----------------------------------------------
        mic_row = NSView.alloc().initWithFrame_(
            NSMakeRect(0, _M_Y, POP_W, 14)
        )
        mic_row.addSubview_(_lbl(
            "🎤", 20, 1, _M_LBL_W, 12, size=10,
        ))
        mic_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, 3, _M_BAR_W, _M_H)
        )
        mic_bar.setWantsLayer_(True)
        mic_row.addSubview_(mic_bar)
        mic_row.setHidden_(True)
        root.addSubview_(mic_row)
        self._mic_bar = mic_bar
        self._mic_row = mic_row

        # ---- Sys meter row -----------------------------------------------
        sys_row = NSView.alloc().initWithFrame_(
            NSMakeRect(0, _M_Y - 18, POP_W, 14)
        )
        sys_row.addSubview_(_lbl(
            "🔊", 20, 1, _M_LBL_W, 12, size=10,
        ))
        sys_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, 3, _M_BAR_W, _M_H)
        )
        sys_bar.setWantsLayer_(True)
        sys_row.addSubview_(sys_bar)
        sys_row.setHidden_(True)
        root.addSubview_(sys_row)
        self._sys_bar = sys_bar
        self._sys_row = sys_row

        _sep(root, _SEP2_Y)

        # ---- Footer links -----------------------------------------------
        root.addSubview_(_footer_btn(
            "Open Transcripts  ↗", 16, _FOOTER_Y, 180,
            "openTranscripts:", d,
        ))
        quit_btn = _footer_btn("Quit", POP_W - 60, _FOOTER_Y, 44, "quitApp:", d)
        quit_btn.setContentTintColor_(NSColor.secondaryLabelColor())
        root.addSubview_(quit_btn)

        # ---- Popover -----------------------------------------------------
        vc = NSViewController.alloc().init()
        vc.setView_(root)

        pop = NSPopover.alloc().init()
        pop.setContentViewController_(vc)
        pop.setContentSize_(root.frame().size)
        pop.setBehavior_(1)    # NSPopoverBehaviorTransient
        pop.setAnimates_(True)

        self._popover = pop
        self._delegate._popover_ref = pop
