"""Overheard — floating transport controls panel."""

import math
import threading
from typing import Callable

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSTextField,
    NSView,
)
from Foundation import NSObject

try:
    from AppKit import (
        NSWindowStyleMaskTitled,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskNonactivatingPanel,
    )
    _STYLE = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskNonactivatingPanel
except ImportError:
    from AppKit import NSTitledWindowMask, NSClosableWindowMask
    _STYLE = NSTitledWindowMask | NSClosableWindowMask | (1 << 7)  # NonactivatingPanel

WIN_W = 400
WIN_H = 150   # extra height for level meters

# Transport states
IDLE = "idle"
RECORDING = "recording"
PAUSED = "paused"
TRANSCRIBING = "transcribing"

# Level meter geometry
_METER_LABEL_W = 18
_METER_BAR_X   = 36
_METER_BAR_W   = WIN_W - 46
_METER_H       = 10
_METER_SEGMENTS = 12


class _LevelMeterView(NSView):
    """Simple segmented level meter drawn with Core Graphics."""

    def init(self):
        self = objc.super(_LevelMeterView, self).init()
        if self is None:
            return None
        self._level: float = 0.0   # 0.0–1.0
        self._visible: bool = False
        return self

    def setLevel_(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        self.setNeedsDisplay_(True)

    def setMeterVisible_(self, visible: bool) -> None:
        self._visible = visible
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect) -> None:
        if not self._visible:
            return

        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        seg_w = (w - (_METER_SEGMENTS - 1)) / _METER_SEGMENTS
        active_segments = int(self._level * _METER_SEGMENTS)

        for i in range(_METER_SEGMENTS):
            x = i * (seg_w + 1)
            if i < active_segments:
                # Green for low, yellow for mid, red for high
                frac = i / _METER_SEGMENTS
                if frac < 0.6:
                    NSColor.systemGreenColor().setFill()
                elif frac < 0.85:
                    NSColor.systemYellowColor().setFill()
                else:
                    NSColor.systemRedColor().setFill()
            else:
                NSColor.tertiaryLabelColor().setFill()

            from AppKit import NSBezierPath
            bar_rect = NSMakeRect(x, 0, seg_w, h)
            NSBezierPath.fillRect_(bar_rect)


def _make_button(title: str, x: float, y: float, action: str, target) -> NSButton:
    btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, 68, 40))
    btn.setTitle_(title)
    btn.setBezelStyle_(1)
    btn.setTarget_(target)
    btn.setAction_(action)
    return btn


def _make_label(x, y, w, h) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    field.setStringValue_("")
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setAlignment_(1)  # centered
    field.setFont_(NSFont.systemFontOfSize_(11))
    field.setTextColor_(NSColor.secondaryLabelColor())
    return field


def _make_small_label(text: str, x, y, w, h) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(NSFont.systemFontOfSize_(10))
    field.setTextColor_(NSColor.tertiaryLabelColor())
    return field


class _TransportDelegate(NSObject):

    def initWithCallbacks_(self, callbacks: dict):
        self = objc.super(_TransportDelegate, self).init()
        if self is None:
            return None
        self._callbacks = callbacks
        return self

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


class TransportWindow:
    """Small floating panel with ⏺ ⏸ ⏹ transport buttons and level meters.

    Callbacks:
        record  — called when ⏺ is clicked (start or resume)
        pause   — called when ⏸ is clicked
        stop    — called when ⏹ is clicked (stop + transcribe)

    Level meters:
        Call set_levels(mic_rms, system_rms) from a timer while recording.
        set_meters_visible(False) hides them when idle.
    """

    def __init__(self, callbacks: dict):
        self._callbacks = callbacks
        self._window = None
        self._delegate = None
        self._btn_record = None
        self._btn_pause = None
        self._btn_stop = None
        self._status_label = None
        self._mic_meter: _LevelMeterView | None = None
        self._sys_meter: _LevelMeterView | None = None
        self._mic_label: NSTextField | None = None
        self._sys_label: NSTextField | None = None
        self._is_multichannel: bool = False

    def show(self) -> None:
        if self._window is None:
            self._build()
        self._window.orderFront_(None)

    def set_state(self, state: str, status: str = "") -> None:
        """Update button enabled states and status label for the given state."""
        if self._window is None:
            return
        enabled = {
            IDLE:         (True,  False, False),
            RECORDING:    (False, True,  True),
            PAUSED:       (True,  False, True),
            TRANSCRIBING: (False, False, False),
        }.get(state, (True, False, False))

        titles = {
            IDLE:         ("⏺", "⏸", "⏹"),
            RECORDING:    ("⏺", "⏸", "⏹"),
            PAUSED:       ("⏺", "⏸", "⏹"),
            TRANSCRIBING: ("⏺", "⏸", "⏹"),
        }.get(state, ("⏺", "⏸", "⏹"))

        self._btn_record.setEnabled_(enabled[0])
        self._btn_pause.setEnabled_(enabled[1])
        self._btn_stop.setEnabled_(enabled[2])
        self._btn_record.setTitle_(titles[0])
        self._status_label.setStringValue_(status)

        # Show/hide meters
        show_meters = state in (RECORDING, PAUSED)
        self.set_meters_visible(show_meters)
        if not show_meters:
            self.set_levels(0.0, 0.0)

    def set_levels(self, mic_rms: float, system_rms: float) -> None:
        """Update level meter displays. Call from a timer at ~100ms while recording.

        RMS values are typically in [0, 0.5] — we scale to fill the meter sensibly.
        """
        if self._mic_meter is None:
            return
        # Scale: -60dB to 0dB range mapped to 0–1
        def _rms_to_level(rms: float) -> float:
            if rms <= 0:
                return 0.0
            db = 20 * math.log10(max(rms, 1e-9))
            return max(0.0, min(1.0, (db + 60) / 60))

        level = _rms_to_level(mic_rms)
        print(f"DEBUG meter: mic_rms={mic_rms:.4f} level={level:.2f}", flush=True)
        self._mic_meter.setLevel_(level)
        if self._is_multichannel and self._sys_meter is not None:
            self._sys_meter.setLevel_(_rms_to_level(system_rms))

    def set_meters_visible(self, visible: bool) -> None:
        """Show or hide level meter section."""
        if self._mic_meter is None:
            return
        self._mic_meter.setMeterVisible_(visible)
        self._mic_label.setHidden_(not visible)
        if self._sys_meter is not None:
            self._sys_meter.setMeterVisible_(visible and self._is_multichannel)
        if self._sys_label is not None:
            self._sys_label.setHidden_(not (visible and self._is_multichannel))

    def configure_channels(self, is_multichannel: bool) -> None:
        """Call after device selection to configure meter layout."""
        self._is_multichannel = is_multichannel

    def _build(self) -> None:
        self._window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIN_W, WIN_H),
            _STYLE,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_("Overheard")
        self._window.center()
        self._window.setFloatingPanel_(True)

        self._delegate = _TransportDelegate.alloc().initWithCallbacks_(self._callbacks)
        cv = self._window.contentView()

        # Buttons row — centred in the window
        btn_y = 90
        btn_total_w = 3 * 68 + 2 * 16  # three buttons + two gaps
        btn_start_x = (WIN_W - btn_total_w) // 2
        self._btn_record = _make_button("⏺", btn_start_x,        btn_y, "onRecord:", self._delegate)
        self._btn_pause  = _make_button("⏸", btn_start_x + 84,   btn_y, "onPause:",  self._delegate)
        self._btn_stop   = _make_button("⏹", btn_start_x + 168,  btn_y, "onStop:",   self._delegate)
        cv.addSubview_(self._btn_record)
        cv.addSubview_(self._btn_pause)
        cv.addSubview_(self._btn_stop)

        # Status label
        self._status_label = _make_label(10, 60, WIN_W - 20, 20)
        cv.addSubview_(self._status_label)

        # Level meters (hidden by default)
        mic_y = 38
        sys_y = 18

        self._mic_label = _make_small_label("\U0001f3a4", 8, mic_y + 1, _METER_LABEL_W, _METER_H + 2)
        self._mic_label.setHidden_(True)
        cv.addSubview_(self._mic_label)

        self._mic_meter = _LevelMeterView.alloc().initWithFrame_(
            NSMakeRect(_METER_BAR_X, mic_y, _METER_BAR_W, _METER_H)
        )
        self._mic_meter.setWantsLayer_(True)
        self._mic_meter.setMeterVisible_(False)
        cv.addSubview_(self._mic_meter)

        self._sys_label = _make_small_label("\U0001f50a", 8, sys_y + 1, _METER_LABEL_W, _METER_H + 2)
        self._sys_label.setHidden_(True)
        cv.addSubview_(self._sys_label)

        self._sys_meter = _LevelMeterView.alloc().initWithFrame_(
            NSMakeRect(_METER_BAR_X, sys_y, _METER_BAR_W, _METER_H)
        )
        self._sys_meter.setWantsLayer_(True)
        self._sys_meter.setMeterVisible_(False)
        cv.addSubview_(self._sys_meter)

        self.set_state(IDLE, "Ready")
