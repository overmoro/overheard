"""Overheard — menu bar popover transport UI."""

import math
import threading

import objc
from AppKit import (
    NSApplication,
    NSButton,
    NSColor,
    NSFont,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPopover,
    NSTextField,
    NSTextAlignmentCenter,
    NSView,
    NSViewController,
)
from Foundation import NSObject

# Popover dimensions
POP_W = 280
POP_H = 256

# Edge: NSMaxYEdge = 1 (popover appears below status bar item)
_NSMinYEdge = 3

# Level meter geometry
_M_SEGS = 18
_M_H    = 7
_M_LABEL_W = 18
_M_X    = 22
_M_BAR_X = _M_X + _M_LABEL_W + 4
_M_BAR_W = POP_W - _M_BAR_X - 16


# ---------------------------------------------------------------------------
# Level bar view
# ---------------------------------------------------------------------------

class _LevelBar(NSView):
    """Segmented horizontal level meter."""

    def init(self):
        self = objc.super(_LevelBar, self).init()
        if self is None:
            return None
        self._level: float = 0.0
        self._active: bool = False
        return self

    def setLevel_(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        self.setNeedsDisplay_(True)

    def setActive_(self, active: bool) -> None:
        self._active = active
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect) -> None:
        from AppKit import NSBezierPath
        b = self.bounds()
        w = b.size.width
        h = b.size.height
        seg_w = (w - (_M_SEGS - 1)) / _M_SEGS
        active_n = int(self._level * _M_SEGS) if self._active else 0

        for i in range(_M_SEGS):
            x = i * (seg_w + 1)
            if i < active_n:
                frac = i / _M_SEGS
                if frac < 0.6:
                    NSColor.systemGreenColor().colorWithAlphaComponent_(0.85).setFill()
                elif frac < 0.85:
                    NSColor.systemYellowColor().colorWithAlphaComponent_(0.85).setFill()
                else:
                    NSColor.systemRedColor().colorWithAlphaComponent_(0.85).setFill()
            else:
                NSColor.separatorColor().setFill()
            bar_rect = NSMakeRect(x, 0, seg_w, h)
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar_rect, 2.0, 2.0
            ).fill()


# ---------------------------------------------------------------------------
# Popover action delegate
# ---------------------------------------------------------------------------

class _PopoverDelegate(NSObject):

    def initWithCallbacks_(self, callbacks: dict):
        self = objc.super(_PopoverDelegate, self).init()
        if self is None:
            return None
        self._callbacks = callbacks
        self._popover_ref = None   # set after popover is built
        return self

    # ------------------------------------------------------------------
    # Popover toggle (called by status bar button)
    # ------------------------------------------------------------------

    def togglePopover_(self, sender):
        pop = self._popover_ref
        if pop is None:
            return
        if pop.isShown():
            pop.performClose_(sender)
        else:
            pop.showRelativeToRect_ofView_preferredEdge_(
                sender.bounds(), sender, _NSMinYEdge
            )

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def onRecord_(self, sender):
        cb = self._callbacks.get("record")
        if cb:
            threading.Thread(target=cb, daemon=True).start()

    def onPause_(self, sender):
        cb = self._callbacks.get("pause")
        if cb:
            cb()

    def onStop_(self, sender):
        cb = self._callbacks.get("stop")
        if cb:
            cb()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def openTranscripts_(self, sender):
        cb = self._callbacks.get("open_transcripts")
        if cb:
            cb()

    def openPreferences_(self, sender):
        cb = self._callbacks.get("preferences")
        if cb:
            cb()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _label(text: str, x, y, w, h, size=12, bold=False, color=None,
           align=None) -> NSTextField:
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setLineBreakMode_(NSLineBreakByTruncatingTail)
    if bold:
        f.setFont_(NSFont.boldSystemFontOfSize_(size))
    else:
        f.setFont_(NSFont.systemFontOfSize_(size))
    if color:
        f.setTextColor_(color)
    if align is not None:
        f.setAlignment_(align)
    return f


def _sep(y: float) -> NSView:
    """Horizontal 1px separator."""
    from AppKit import NSBox
    box = NSBox.alloc().initWithFrame_(NSMakeRect(0, y, POP_W, 1))
    box.setBoxType_(2)        # NSSeparator
    return box


def _transport_btn(title: str, x: float, y: float, action: str,
                   target) -> NSButton:
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, 72, 44))
    btn.setTitle_(title)
    btn.setBezelStyle_(1)   # NSBezelStyleRounded
    btn.setFont_(NSFont.systemFontOfSize_(22))
    btn.setTarget_(target)
    btn.setAction_(action)
    return btn


def _link_btn(title: str, x: float, y: float, w: float, action: str,
              target) -> NSButton:
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 22))
    btn.setTitle_(title)
    btn.setBezelStyle_(14)   # NSBezelStyleInline
    btn.setFont_(NSFont.systemFontOfSize_(12))
    btn.setTarget_(target)
    btn.setAction_(action)
    btn.setContentTintColor_(NSColor.linkColor())
    return btn


# ---------------------------------------------------------------------------
# TransportPopover
# ---------------------------------------------------------------------------

class TransportPopover:
    """Menu bar popover with transport controls and level meters.

    Call hook_status_item(btn) once after the rumps run loop starts to
    attach toggle behaviour to the status bar button.
    """

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
    # Public API (mirrors TransportWindow)
    # ------------------------------------------------------------------

    def hook_status_item(self, status_btn) -> None:
        """Attach the popover toggle to the status bar button."""
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
        self._status_lbl.setStringValue_(status)

        show_meters = state in (RECORDING, PAUSED)
        self._set_meters_visible(show_meters)
        if not show_meters:
            self.set_levels(0.0, 0.0)

    def set_levels(self, mic_rms: float, system_rms: float) -> None:
        if self._mic_bar is None:
            return

        def _to_level(rms: float) -> float:
            if rms <= 0:
                return 0.0
            db = 20 * math.log10(max(rms, 1e-9))
            return max(0.0, min(1.0, (db + 60) / 60))

        self._mic_bar.setLevel_(_to_level(mic_rms))
        if self._is_multichannel and self._sys_bar:
            self._sys_bar.setLevel_(_to_level(system_rms))

    def configure_channels(self, is_multichannel: bool) -> None:
        self._is_multichannel = is_multichannel
        if self._sys_row:
            self._sys_row.setHidden_(not is_multichannel)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_meters_visible(self, visible: bool) -> None:
        if self._mic_row:
            self._mic_row.setHidden_(not visible)
        if self._sys_row:
            self._sys_row.setHidden_(not (visible and self._is_multichannel))
        if self._mic_bar:
            self._mic_bar.setActive_(visible)
        if self._sys_bar:
            self._sys_bar.setActive_(visible and self._is_multichannel)

    def _build(self) -> None:
        # ---- Content view -----------------------------------------------
        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, POP_W, POP_H))

        d = self._delegate

        # ---- Title row --------------------------------------------------
        content.addSubview_(_label(
            "Overheard", 16, POP_H - 36, 160, 22,
            size=14, bold=True,
        ))
        content.addSubview_(_link_btn(
            "⚙  Preferences", POP_W - 116, POP_H - 34, 104,
            "openPreferences:", d,
        ))

        content.addSubview_(_sep(POP_H - 44))

        # ---- Transport buttons ------------------------------------------
        # Three 72×44 buttons, centred
        total_btn_w = 3 * 72 + 2 * 10
        btn_start = (POP_W - total_btn_w) // 2

        self._btn_record = _transport_btn("⏺", btn_start,        POP_H - 100, "onRecord:", d)
        self._btn_pause  = _transport_btn("⏸", btn_start + 82,   POP_H - 100, "onPause:",  d)
        self._btn_stop   = _transport_btn("⏹", btn_start + 164,  POP_H - 100, "onStop:",   d)
        content.addSubview_(self._btn_record)
        content.addSubview_(self._btn_pause)
        content.addSubview_(self._btn_stop)

        # ---- Status label -----------------------------------------------
        self._status_lbl = _label(
            "Ready", 16, POP_H - 118, POP_W - 32, 16,
            size=11, color=NSColor.secondaryLabelColor(),
            align=NSTextAlignmentCenter,
        )
        content.addSubview_(self._status_lbl)

        content.addSubview_(_sep(POP_H - 126))

        # ---- Mic meter row ----------------------------------------------
        mic_row = NSView.alloc().initWithFrame_(
            NSMakeRect(0, POP_H - 148, POP_W, 16)
        )
        mic_row.addSubview_(_label(
            "\U0001f3a4", _M_X, 2, _M_LABEL_W, 12, size=10,
        ))
        mic_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, 4, _M_BAR_W, _M_H)
        )
        mic_bar.setWantsLayer_(True)
        mic_row.addSubview_(mic_bar)
        mic_row.setHidden_(True)
        content.addSubview_(mic_row)
        self._mic_bar = mic_bar
        self._mic_row = mic_row

        # ---- Sys meter row ----------------------------------------------
        sys_row = NSView.alloc().initWithFrame_(
            NSMakeRect(0, POP_H - 168, POP_W, 16)
        )
        sys_row.addSubview_(_label(
            "\U0001f50a", _M_X, 2, _M_LABEL_W, 12, size=10,
        ))
        sys_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, 4, _M_BAR_W, _M_H)
        )
        sys_bar.setWantsLayer_(True)
        sys_row.addSubview_(sys_bar)
        sys_row.setHidden_(True)
        content.addSubview_(sys_row)
        self._sys_bar = sys_bar
        self._sys_row = sys_row

        content.addSubview_(_sep(POP_H - 176))

        # ---- Footer links -----------------------------------------------
        content.addSubview_(_link_btn(
            "Open Transcripts  ↗", 16, POP_H - 200, 160,
            "openTranscripts:", d,
        ))

        # ---- Wire popover -----------------------------------------------
        vc = NSViewController.alloc().init()
        vc.setView_(content)

        popover = NSPopover.alloc().init()
        popover.setContentViewController_(vc)
        popover.setContentSize_(content.frame().size)
        popover.setBehavior_(1)   # NSPopoverBehaviorTransient
        popover.setAnimates_(True)

        self._popover = popover
        self._delegate._popover_ref = popover
