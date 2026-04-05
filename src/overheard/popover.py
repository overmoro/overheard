"""Overheard — menu bar popover transport UI."""

import math

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSLineBreakByTruncatingTail,
    NSMakeRect,
    NSPanel,
    NSStatusWindowLevel,
    NSTextField,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSTrackingArea,
    NSView,
)
from Foundation import NSAttributedString, NSObject, NSTimer

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
POP_W   = 300
POP_H   = 210
_HDR_H  = 52
_BODY_H = POP_H - _HDR_H   # 158 — available body height

_BTN_D   = 50    # button circle diameter (collapsed)
_BTN_EW  = 90    # button frame width (expanded pill)
_BTN_GAP = 8     # gap between button slots

# Level meter geometry
_M_SEGS    = 20
_M_H       = 8          # bar segment height
_M_PAD_L   = 16         # left padding inside meter row
_M_LBL_W   = 22         # emoji label width
_M_LBL_GAP = 6          # gap between emoji and bar
_M_PAD_R   = 16         # right padding inside meter row
_M_BAR_X   = _M_PAD_L + _M_LBL_W + _M_LBL_GAP   # = 44
_M_BAR_W   = POP_W - _M_BAR_X - _M_PAD_R          # = 240

# NSTrackingArea options: MouseEnteredAndExited | ActiveAlways
_TRACK_OPTS = 0x01 | 0x02 | 0x80

# ---------------------------------------------------------------------------
# Layout — computed top-down through the body.
#
# AppKit coordinate system: y=0 at bottom of view, increases upward.
# Every constant below is the BOTTOM EDGE y of that element in the body
# (root) coordinate space, which runs from y=0 (body bottom) to
# y=_BODY_H (header bottom).
#
# Read top-to-bottom: each row sits 10-12px below the one above it.
# ---------------------------------------------------------------------------
_ROW_H    = 14   # level-meter row height
_SEP_H    = 1    # separator height
_STATUS_H = 18   # status label height

# Body = 158px.  Bottom-up derivation (14px pad at bottom):
#                                              value   comment
_SYS_Y    = 14                          #  =  14   14px from body bottom
_MIC_Y    = _SYS_Y   +  _ROW_H +  5    #  =  33    5px above sys
_SEP1_Y   = _MIC_Y   +  _ROW_H + 10    #  =  57   10px above mic
_STATUS_Y = _SEP1_Y  +  _SEP_H + 10    #  =  68   10px above sep1
_BTN_Y    = _STATUS_Y + _STATUS_H + 10  #  =  96   10px above status
# PAD_TOP = _BODY_H - (_BTN_Y + _BTN_D) = 158 - 146 = 12px ✓

# Vertical centering within a meter row
_ROW_LBL_Y = (_ROW_H - 12) // 2    # = 1  (12pt emoji centred in 14px row)
_ROW_BAR_Y = (_ROW_H - _M_H) // 2  # = 3  (8px bar centred in 14px row)


# ---------------------------------------------------------------------------
# Segmented level bar
# ---------------------------------------------------------------------------

class _LevelBar(NSView):
    def init(self):
        self = objc.super(_LevelBar, self).init()
        if self is None:
            return None
        self._level  = 0.0
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
# Animated pill button (circle → pill on hover)
# ---------------------------------------------------------------------------

class _PillButton(NSView):
    """Circular button that expands to a pill + label on mouse-over."""

    def initWithIcon_label_color_callback_(self, icon, label, color, callback):
        self = objc.super(_PillButton, self).initWithFrame_(
            NSMakeRect(0, 0, _BTN_EW, _BTN_D)
        )
        if self is None:
            return None
        self._icon      = icon
        self._label     = label
        self._color     = color
        self._callback  = callback
        self._progress  = 0.0      # 0 = collapsed circle, 1 = full pill
        self._expanding = False
        self._anim_timer = None
        self._enabled   = True
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

    def acceptsFirstResponder(self):
        return True

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
        step = (1.0 / 60.0) / 0.18   # 0.18 s transition
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
        t  = self._progress
        t  = t * t * (3.0 - 2.0 * t)          # smoothstep easing

        cw  = _BTN_D + (_BTN_EW - _BTN_D) * t  # current pill width
        bx  = (_BTN_EW - cw) / 2.0
        r   = _BTN_D / 2.0

        color = self._color.colorWithAlphaComponent_(0.35 if not self._enabled else 1.0)
        color.setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(bx, 0, cw, _BTN_D), r, r
        ).fill()

        icon_alpha = 0.4 if not self._enabled else 1.0
        icon_attrs = {
            NSFontAttributeName: NSFont.systemFontOfSize_(20.0),
            NSForegroundColorAttributeName: NSColor.whiteColor().colorWithAlphaComponent_(icon_alpha),
        }
        icon_as = NSAttributedString.alloc().initWithString_attributes_(self._icon, icon_attrs)
        isz = icon_as.size()
        icon_as.drawAtPoint_((bx + r - isz.width / 2.0, r - isz.height / 2.0))

        if t > 0.3 and self._label:
            alpha   = min(1.0, (t - 0.3) / 0.5) * icon_alpha
            lbl_attrs = {
                NSFontAttributeName: NSFont.boldSystemFontOfSize_(11),
                NSForegroundColorAttributeName: NSColor.whiteColor().colorWithAlphaComponent_(alpha),
            }
            lbl_as  = NSAttributedString.alloc().initWithString_attributes_(self._label, lbl_attrs)
            lsz     = lbl_as.size()
            label_x = bx + _BTN_D + 4.0
            lbl_as.drawAtPoint_((label_x, r - lsz.height / 2.0))


# ---------------------------------------------------------------------------
# Draggable header view
# ---------------------------------------------------------------------------

class _DragHeader(NSView):
    """Header view that drags the parent NSPanel when clicked and dragged."""

    def init(self):
        self = objc.super(_DragHeader, self).init()
        if self is None:
            return None
        self._drag_event_loc = None
        return self

    def drawRect_(self, rect):
        NSColor.colorWithWhite_alpha_(0.94, 1.0).setFill()
        from AppKit import NSBezierPath
        NSBezierPath.fillRect_(self.bounds())

    def mouseDown_(self, event):
        # Record the panel origin at drag start in screen coords
        win = self.window()
        if win:
            self._drag_start = win.frame().origin
            self._drag_event_loc = event.locationInWindow()

    def mouseDragged_(self, event):
        win = self.window()
        if win is None or self._drag_start is None:
            return
        loc    = event.locationInWindow()
        dx     = loc.x - self._drag_event_loc.x
        dy     = loc.y - self._drag_event_loc.y
        origin = win.frame().origin
        win.setFrameOrigin_((origin.x + dx, origin.y + dy))

    def mouseUp_(self, event):
        self._drag_start = None

    # Allow subviews (gear button etc.) to receive their events normally
    def acceptsFirstMouse_(self, event):
        return True


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
    box = NSBox.alloc().initWithFrame_(NSMakeRect(16, y, POP_W - 32, _SEP_H))
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
        self._cbs       = cbs
        self._toggle_cb = None   # set by TransportPopover after build
        return self

    def togglePanel_(self, sender):
        from AppKit import NSApplication, NSEventTypeRightMouseDown
        event = NSApplication.sharedApplication().currentEvent()
        if event is not None and event.type() == NSEventTypeRightMouseDown:
            self._show_context_menu(sender)
        else:
            if self._toggle_cb:
                self._toggle_cb()

    def _show_context_menu(self, sender):
        from AppKit import NSMenu, NSMenuItem
        menu = NSMenu.alloc().init()

        show_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Show App", "showPanel:", ""
        )
        show_item.setTarget_(self)
        menu.addItem_(show_item)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Overheard", "quitApp:", ""
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        # Show at the status bar button position
        sender.popUpContextMenu_withEvent_forView_(
            menu,
            NSApplication.sharedApplication().currentEvent(),
            sender,
        )

    def showPanel_(self, sender):
        if self._toggle_cb:
            self._toggle_cb()

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
        self._delegate        = _PopoverDelegate.alloc().initWithCallbacks_(callbacks)
        self._panel           = None
        self._status_btn      = None   # status bar button (for positioning)
        self._btn_record      = None
        self._btn_pause       = None
        self._btn_stop        = None
        self._status_lbl      = None
        self._mic_bar         = None
        self._sys_bar         = None
        self._mic_row         = None
        self._sys_row         = None
        self._is_multichannel = False
        self._build(callbacks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hook_status_item(self, btn):
        self._status_btn = btn
        btn.setTarget_(self._delegate)
        btn.setAction_("togglePanel:")
        # Send action on both left-click and right-click
        btn.sendActionOn_((1 << 1) | (1 << 3))   # NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp

    def toggle(self):
        if self._panel and self._panel.isVisible():
            self._hide()
        else:
            self._show()

    def _show(self):
        btn = self._status_btn
        if btn is None or self._panel is None:
            return
        x, y = self._panel_origin(btn)
        self._panel.setFrameOrigin_((x, y))
        # LSUIElement apps (no dock icon) must be explicitly activated before
        # makeKeyAndOrderFront_ — otherwise windowDidResignKey_ fires immediately.
        from AppKit import NSApplication
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._panel.makeKeyAndOrderFront_(None)

    def _panel_origin(self, btn):
        """Return (x, y) screen origin so the panel sits flush below btn."""
        try:
            win = btn.window()
            if win is not None:
                btn_rect = btn.convertRect_toView_(btn.bounds(), None)
                screen_rect = win.convertRectToScreen_(btn_rect)
                x = screen_rect.origin.x + screen_rect.size.width / 2 - POP_W / 2
                y = screen_rect.origin.y - POP_H
                return x, y
        except Exception:
            pass
        # Fallback: top-right of main screen (reasonable for menu bar items)
        from AppKit import NSScreen
        sf = NSScreen.mainScreen().frame()
        mbar_h = NSScreen.mainScreen().visibleFrame().size.height
        y = sf.size.height - POP_H   # just below top of screen
        x = sf.size.width - POP_W - 8
        return x, y

    def _hide(self):
        if self._panel:
            self._panel.orderOut_(None)

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
        if self._mic_bar: self._mic_bar.setActive_(v)
        if self._sys_bar: self._sys_bar.setActive_(v and self._is_multichannel)
        if self._sys_row: self._sys_row.setHidden_(not self._is_multichannel)

    def _build(self, callbacks):
        d = self._delegate
        from AppKit import NSBox

        # ---- Root view (NSBox, content insets removed) ------------------
        root_box = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, POP_W, POP_H))
        root_box.setBoxType_(0)
        root_box.setBorderType_(0)
        root_box.setFillColor_(NSColor.windowBackgroundColor())
        root_box.setTitlePosition_(0)
        root_box.setContentViewMargins_((0, 0))   # no inset — coordinates match frame
        root = root_box.contentView()

        # ---- Header (soft grey, draggable) ----------------------------------
        hdr = _DragHeader.alloc().initWithFrame_(NSMakeRect(0, POP_H - _HDR_H, POP_W, _HDR_H))
        hdr_cv = hdr

        # Title + subtitle, left-aligned
        hdr_cv.addSubview_(_lbl(
            "Overheard", 16, 23, 180, 20,
            size=13, bold=True, color=NSColor.labelColor(),
        ))
        hdr_cv.addSubview_(_lbl(
            "Meeting transcription", 16, 8, 180, 14,
            size=10, color=NSColor.secondaryLabelColor(),
        ))

        # Gear button — right-aligned, vertically centred in header
        _GEAR_SZ = 38
        gear = NSButton.alloc().initWithFrame_(
            NSMakeRect(POP_W - _GEAR_SZ - 10, (_HDR_H - _GEAR_SZ) // 2, _GEAR_SZ, _GEAR_SZ)
        )
        gear.setTitle_("⚙")
        gear.setFont_(NSFont.systemFontOfSize_(26))
        gear.setBordered_(False)
        gear.setBezelStyle_(14)
        gear.setTarget_(d)
        gear.setAction_("openPreferences:")
        gear.setContentTintColor_(NSColor.secondaryLabelColor())
        hdr_cv.addSubview_(gear)

        # 1px separator at header bottom
        border = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, POP_W, 1))
        border.setBoxType_(2)
        hdr_cv.addSubview_(border)

        root.addSubview_(hdr)

        # ---- Transport buttons -------------------------------------------
        total_btn_w = 3 * _BTN_EW + 2 * _BTN_GAP
        start_x     = (POP_W - total_btn_w) // 2   # centres the button group

        self._btn_record = _PillButton.alloc().initWithIcon_label_color_callback_(
            "⏺", "Record",
            NSColor.systemRedColor(),
            callbacks.get("record"),
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
            btn.setFrame_(NSMakeRect(
                start_x + i * (_BTN_EW + _BTN_GAP), _BTN_Y,
                _BTN_EW, _BTN_D,
            ))
            root.addSubview_(btn)

        # ---- Status label (centred) -------------------------------------
        self._status_lbl = _lbl(
            "Ready", 0, _STATUS_Y, POP_W, _STATUS_H,
            size=12, color=NSColor.secondaryLabelColor(),
            align=NSTextAlignmentCenter,
        )
        root.addSubview_(self._status_lbl)

        # ---- Sep 1 -------------------------------------------------------
        _sep(root, _SEP1_Y)

        # ---- Mic meter row -----------------------------------------------
        mic_row = NSView.alloc().initWithFrame_(NSMakeRect(0, _MIC_Y, POP_W, _ROW_H))
        mic_row.addSubview_(_lbl("🎤", _M_PAD_L, _ROW_LBL_Y, _M_LBL_W, 12, size=10))
        mic_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, _ROW_BAR_Y, _M_BAR_W, _M_H)
        )
        mic_bar.setWantsLayer_(True)
        mic_row.addSubview_(mic_bar)
        root.addSubview_(mic_row)
        self._mic_bar = mic_bar
        self._mic_row = mic_row

        # ---- Sys meter row -----------------------------------------------
        sys_row = NSView.alloc().initWithFrame_(NSMakeRect(0, _SYS_Y, POP_W, _ROW_H))
        sys_row.addSubview_(_lbl("🔊", _M_PAD_L, _ROW_LBL_Y, _M_LBL_W, 12, size=10))
        sys_bar = _LevelBar.alloc().initWithFrame_(
            NSMakeRect(_M_BAR_X, _ROW_BAR_Y, _M_BAR_W, _M_H)
        )
        sys_bar.setWantsLayer_(True)
        sys_row.addSubview_(sys_bar)
        root.addSubview_(sys_row)
        self._sys_bar = sys_bar
        self._sys_row = sys_row

        # ---- Wire as borderless NSPanel (no arrow) --------------------------
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, POP_W, POP_H),
            0,                       # NSWindowStyleMaskBorderless
            NSBackingStoreBuffered,
            False,
        )
        # Add root_box as a subview of the panel's existing content view
        # rather than replacing it — avoids compositing issues with transparent panels.
        panel_cv = panel.contentView()
        panel_cv.addSubview_(root_box)
        panel.setHasShadow_(True)
        panel.setLevel_(NSStatusWindowLevel + 1)
        panel.setFloatingPanel_(True)

        self._panel = panel
        self._delegate._toggle_cb = self.toggle
