"""Overheard — menu bar popover transport UI."""

import math
import threading

import objc
from AppKit import (
    NSButton,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPopover,
    NSTextField,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSTrackingArea,
    NSView,
    NSViewController,
    NSVisualEffectView,
)
from Foundation import NSAttributedString, NSObject, NSTimer

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
POP_W   = 300
POP_H   = 270
_HDR_H  = 52

_BTN_D  = 50    # circle height / collapsed diameter
_BTN_EW = 90    # expanded width
_BTN_GAP = 8    # gap between button slots

# y positions from bottom
_BTN_Y    = 124
_STATUS_Y = 94
_SEP1_Y   = 83
_MIC_Y    = 61
_SYS_Y    = 43
_SEP2_Y   = 34
_FOOTER_Y = 10

# Level meter geometry
_M_SEGS  = 20
_M_H     = 8
_M_LBL_W = 22
_M_BAR_X = 16 + _M_LBL_W + 6
_M_BAR_W = POP_W - _M_BAR_X - 16

# NSTrackingArea: MouseEnteredAndExited(3) | ActiveAlways(0x80)
_TRACK_OPTS = 0x01 | 0x02 | 0x80

# NSMinYEdge — popover appears below status bar item
_NSMinYEdge = 3


# ---------------------------------------------------------------------------
# Segmented level bar
# ---------------------------------------------------------------------------

class _LevelBar(NSView):
    def init(self):
        self = objc.super(_LevelBar, self).init()
        if self is None:
            return None
        self._level = 0.0
        self._active = False
        return self

    def setLevel_(self, v):
        self._level = max(0.0, min(1.0, v))
        self.setNeedsDisplay_(True)

    def setActive_(self, v):
        self._active = bool(v)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        from AppKit import NSBezierPath
        b  = self.bounds()
        sw = (b.size.width - (_M_SEGS - 1)) / _M_SEGS
        n  = int(self._level * _M_SEGS) if self._active else 0
        for i in range(_M_SEGS):
            x = i * (sw + 1)
            if i < n:
                frac = i / _M_SEGS
                if   frac < 0.60: NSColor.systemGreenColor().colorWithAlphaComponent_(0.9).setFill()
                elif frac < 0.85: NSColor.systemYellowColor().colorWithAlphaComponent_(0.9).setFill()
                else:             NSColor.systemRedColor().colorWithAlphaComponent_(0.9).setFill()
            else:
                NSColor.tertiaryLabelColor().setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, 0, sw, b.size.height), 2, 2
            ).fill()


# ---------------------------------------------------------------------------
# Animated pill button (LS-style: circle → pill on hover)
# ---------------------------------------------------------------------------

class _PillButton(NSView):
    """Circular button that expands to pill + label on hover."""

    def initWithIcon_label_color_callback_(self, icon, label, color, callback):
        self = objc.super(_PillButton, self).initWithFrame_(
            NSMakeRect(0, 0, _BTN_EW, _BTN_D)
        )
        if self is None:
            return None
        self._icon     = icon
        self._label    = label
        self._color    = color
        self._callback = callback
        self._progress = 0.0      # 0 = collapsed circle, 1 = full pill
        self._expanding = False
        self._anim_timer = None
        self._enabled = True
        self._setup_tracking()
        return self

    # ---- Tracking area ------------------------------------------------

    def _setup_tracking(self):
        for a in list(self.trackingAreas()):
            self.removeTrackingArea_(a)
        self.addTrackingArea_(
            NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                self.bounds(), _TRACK_OPTS, self, None
            )
        )

    def updateTrackingAreas(self):
        self._setup_tracking()
        objc.super(_PillButton, self).updateTrackingAreas()

    # ---- State --------------------------------------------------------

    def setEnabled_(self, v):
        self._enabled = bool(v)
        self.setNeedsDisplay_(True)

    # ---- Mouse events -------------------------------------------------

    def mouseEntered_(self, event):
        self._expanding = True
        self._ensure_timer()

    def mouseExited_(self, event):
        self._expanding = False
        self._ensure_timer()

    def mouseDown_(self, event):
        if self._enabled and self._callback:
            self._callback()

    # ---- Animation timer (NSTimer at 60 fps) --------------------------

    def _ensure_timer(self):
        if self._anim_timer is None or not self._anim_timer.isValid():
            self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / 60.0, self, "onAnimTick:", None, True
            )

    def onAnimTick_(self, timer):
        step = (1.0 / 60.0) / 0.18   # 0.18s transition
        if self._expanding:
            self._progress = min(1.0, self._progress + step)
        else:
            self._progress = max(0.0, self._progress - step)
        self.setNeedsDisplay_(True)
        if self._progress <= 0.0 or self._progress >= 1.0:
            timer.invalidate()
            self._anim_timer = None

    # ---- Drawing ------------------------------------------------------

    def drawRect_(self, rect):
        from AppKit import NSBezierPath
        # Smoothstep ease
        t = self._progress
        t = t * t * (3.0 - 2.0 * t)

        # Current pill width — interpolates from circle (_BTN_D) to full (_BTN_EW)
        cw  = _BTN_D + (_BTN_EW - _BTN_D) * t
        bx  = (_BTN_EW - cw) / 2.0
        r   = _BTN_D / 2.0

        # Background pill
        color = self._color.colorWithAlphaComponent_(0.35 if not self._enabled else 1.0)
        color.setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(bx, 0, cw, _BTN_D), r, r
        ).fill()

        # Icon — centred in the left circle portion
        icon_alpha = 0.4 if not self._enabled else 1.0
        icon_attrs = {
            NSFontAttributeName: NSFont.systemFontOfSize_(20.0),
            NSForegroundColorAttributeName: NSColor.whiteColor().colorWithAlphaComponent_(icon_alpha),
        }
        icon_as = NSAttributedString.alloc().initWithString_attributes_(self._icon, icon_attrs)
        isz = icon_as.size()
        icon_as.drawAtPoint_((bx + r - isz.width / 2.0, r - isz.height / 2.0))

        # Label — fades in as pill opens
        if t > 0.3 and self._label:
            alpha = min(1.0, (t - 0.3) / 0.5) * icon_alpha
            lbl_attrs = {
                NSFontAttributeName: NSFont.boldSystemFontOfSize_(11),
                NSForegroundColorAttributeName: NSColor.whiteColor().colorWithAlphaComponent_(alpha),
            }
            lbl_as  = NSAttributedString.alloc().initWithString_attributes_(self._label, lbl_attrs)
            lsz     = lbl_as.size()
            label_x = bx + _BTN_D + 4.0
            lbl_as.drawAtPoint_((label_x, r - lsz.height / 2.0))


# ---------------------------------------------------------------------------
# Helper UI builders
# ---------------------------------------------------------------------------

def _lbl(text, x, y, w, h, size=12, bold=False, color=None,
         align=NSTextAlignmentLeft):
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setLineBreakMode_(NSLineBreakByTruncatingTail)
    f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    if color:
        f.setTextColor_(color)
    f.setAlignment_(align)
    return f


def _sep(parent, y):
    from AppKit import NSBox
    box = NSBox.alloc().initWithFrame_(NSMakeRect(16, y, POP_W - 32, 1))
    box.setBoxType_(2)   # NSSeparator
    parent.addSubview_(box)


def _footer_btn(title, x, y, w, action, target):
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, 22))
    btn.setTitle_(title)
    btn.setBezelStyle_(14)
    btn.setFont_(NSFont.systemFontOfSize_(12))
    btn.setTarget_(target)
    btn.setAction_(action)
    btn.setContentTintColor_(NSColor.linkColor())
    return btn


# ---------------------------------------------------------------------------
# Popover delegate (toggle + footer actions)
# ---------------------------------------------------------------------------

class _PopoverDelegate(NSObject):

    def initWithCallbacks_(self, cbs):
        self = objc.super(_PopoverDelegate, self).init()
        if self is None:
            return None
        self._cbs = cbs
        self._popover_ref = None
        return self

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

    def openTranscripts_(self, sender):
        cb = self._cbs.get("open_transcripts")
        if cb:
            cb()

    def openPreferences_(self, sender):
        cb = self._cbs.get("preferences")
        if cb:
            cb()

    def quitApp_(self, sender):
        import rumps
        rumps.quit_application()


# ---------------------------------------------------------------------------
# TransportPopover
# ---------------------------------------------------------------------------

class TransportPopover:

    def __init__(self, callbacks: dict):
        self._delegate = _PopoverDelegate.alloc().initWithCallbacks_(callbacks)
        self._popover   = None
        self._btn_record = None
        self._btn_pause  = None
        self._btn_stop   = None
        self._status_lbl = None
        self._mic_bar    = None
        self._sys_bar    = None
        self._mic_row    = None
        self._sys_row    = None
        self._is_multichannel = False
        self._build(callbacks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hook_status_item(self, btn):
        btn.setTarget_(self._delegate)
        btn.setAction_("togglePopover:")

    def set_state(self, state, status=""):
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

        # Record button turns red while recording
        self._btn_record._color = (
            NSColor.systemRedColor() if state == RECORDING
            else NSColor.systemRedColor()   # always red — dimmed when disabled
        )
        self._btn_record.setNeedsDisplay_(True)

        self._status_lbl.setStringValue_(status)

        show = state in (RECORDING, PAUSED)
        self._set_meters_visible(show)
        if not show:
            self.set_levels(0.0, 0.0)

    def set_levels(self, mic_rms, sys_rms):
        if not self._mic_bar:
            return

        def _db(rms):
            if rms <= 0:
                return 0.0
            db = 20 * math.log10(max(rms, 1e-9))
            return max(0.0, min(1.0, (db + 60) / 60))

        self._mic_bar.setLevel_(_db(mic_rms))
        if self._is_multichannel and self._sys_bar:
            self._sys_bar.setLevel_(_db(sys_rms))

    def configure_channels(self, is_multichannel):
        self._is_multichannel = is_multichannel
        if self._sys_row:
            self._sys_row.setHidden_(not is_multichannel)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_meters_visible(self, v):
        # Meters are always shown; active state controls whether bars light up
        if self._mic_bar: self._mic_bar.setActive_(v)
        if self._sys_bar: self._sys_bar.setActive_(v and self._is_multichannel)
        if self._sys_row: self._sys_row.setHidden_(not self._is_multichannel)

    def _build(self, callbacks):
        d = self._delegate

        # ---- Root (plain white body) ------------------------------------
        from AppKit import NSBox as _RootBox
        root_box = _RootBox.alloc().initWithFrame_(NSMakeRect(0, 0, POP_W, POP_H))
        root_box.setBoxType_(0)
        root_box.setBorderType_(0)
        root_box.setFillColor_(NSColor.whiteColor())
        root_box.setTitlePosition_(0)
        root = root_box.contentView()

        # ---- Header (soft grey, WARP-style) -----------------------------
        from AppKit import NSBox
        hdr = NSBox.alloc().initWithFrame_(NSMakeRect(0, POP_H - _HDR_H, POP_W, _HDR_H))
        hdr.setBoxType_(0)
        hdr.setBorderType_(0)
        hdr.setFillColor_(NSColor.colorWithWhite_alpha_(0.94, 1.0))
        hdr.setTitlePosition_(0)

        hdr_cv = hdr.contentView()
        hdr_cv.addSubview_(_lbl(
            "Overheard", 16, 22, 200, 18,
            size=13, bold=True, color=NSColor.labelColor(),
        ))
        hdr_cv.addSubview_(_lbl(
            "Meeting transcription", 16, 8, 200, 14,
            size=10, color=NSColor.secondaryLabelColor(),
        ))

        # Gear button
        gear = NSButton.alloc().initWithFrame_(NSMakeRect(POP_W - 44, 10, 28, 28))
        gear.setTitle_("⚙")
        gear.setFont_(NSFont.systemFontOfSize_(15))
        gear.setBordered_(False)
        gear.setBezelStyle_(14)
        gear.setTarget_(d)
        gear.setAction_("openPreferences:")
        gear.setContentTintColor_(NSColor.secondaryLabelColor())
        hdr_cv.addSubview_(gear)

        # Header bottom border
        border = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, POP_W, 1))
        border.setBoxType_(2)
        hdr_cv.addSubview_(border)

        root.addSubview_(hdr)

        # ---- Transport buttons (LS-style pill) --------------------------
        total_w = 3 * _BTN_EW + 2 * _BTN_GAP
        start_x = (POP_W - total_w) // 2

        self._btn_record = _PillButton.alloc().initWithIcon_label_color_callback_(
            "⏺", "Record",
            NSColor.systemRedColor(),
            callbacks.get("record") and
            (lambda: threading.Thread(target=callbacks["record"], daemon=True).start()),
        )
        self._btn_pause = _PillButton.alloc().initWithIcon_label_color_callback_(
            "⏸", "Pause",
            NSColor.systemBlueColor(),
            callbacks.get("pause"),
        )
        self._btn_stop = _PillButton.alloc().initWithIcon_label_color_callback_(
            "⏹", "Stop",
            NSColor.colorWithWhite_alpha_(0.22, 1.0),
            callbacks.get("stop"),
        )

        for i, btn in enumerate((self._btn_record, self._btn_pause, self._btn_stop)):
            x = start_x + i * (_BTN_EW + _BTN_GAP)
            frame = btn.frame()
            frame.origin.x = x
            frame.origin.y = _BTN_Y
            btn.setFrame_(frame)
            root.addSubview_(btn)

        # ---- Status label -----------------------------------------------
        self._status_lbl = _lbl(
            "Ready", 0, _STATUS_Y, POP_W, 18,
            size=12, color=NSColor.secondaryLabelColor(),
            align=NSTextAlignmentCenter,
        )
        root.addSubview_(self._status_lbl)

        _sep(root, _SEP1_Y)

        # ---- Mic meter --------------------------------------------------
        mic_row = NSView.alloc().initWithFrame_(NSMakeRect(0, _MIC_Y, POP_W, 14))
        mic_row.addSubview_(_lbl("🎤", 16, 1, _M_LBL_W, 12, size=10))
        mic_bar = _LevelBar.alloc().initWithFrame_(NSMakeRect(_M_BAR_X, 3, _M_BAR_W, _M_H))
        mic_bar.setWantsLayer_(True)
        mic_row.addSubview_(mic_bar)
        root.addSubview_(mic_row)
        self._mic_bar = mic_bar
        self._mic_row = mic_row

        # ---- Sys meter --------------------------------------------------
        sys_row = NSView.alloc().initWithFrame_(NSMakeRect(0, _SYS_Y, POP_W, 14))
        sys_row.addSubview_(_lbl("🔊", 16, 1, _M_LBL_W, 12, size=10))
        sys_bar = _LevelBar.alloc().initWithFrame_(NSMakeRect(_M_BAR_X, 3, _M_BAR_W, _M_H))
        sys_bar.setWantsLayer_(True)
        sys_row.addSubview_(sys_bar)
        root.addSubview_(sys_row)
        self._sys_bar = sys_bar
        self._sys_row = sys_row

        _sep(root, _SEP2_Y)

        # ---- Footer -----------------------------------------------------
        root.addSubview_(_footer_btn(
            "Open Transcripts  ↗", 16, _FOOTER_Y, 180, "openTranscripts:", d,
        ))
        quit_btn = _footer_btn("Quit", POP_W - 60, _FOOTER_Y, 44, "quitApp:", d)
        quit_btn.setContentTintColor_(NSColor.secondaryLabelColor())
        root.addSubview_(quit_btn)

        # ---- Wire popover -----------------------------------------------
        vc = NSViewController.alloc().init()
        vc.setView_(root_box)

        pop = NSPopover.alloc().init()
        pop.setContentViewController_(vc)
        pop.setContentSize_(root_box.frame().size)
        pop.setBehavior_(1)    # NSPopoverBehaviorTransient
        pop.setAnimates_(True)

        # Force light appearance so the white body stays white in dark mode
        from AppKit import NSAppearance
        pop.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameAqua"))

        self._popover = pop
        self._delegate._popover_ref = pop
