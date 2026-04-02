"""Overheard — floating transport controls panel."""

import threading
from typing import Callable

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSTextField,
)
from Foundation import NSObject

try:
    from AppKit import NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable
    _STYLE = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
except ImportError:
    from AppKit import NSTitledWindowMask, NSClosableWindowMask, NSMiniaturizableWindowMask
    _STYLE = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask

WIN_W = 260
WIN_H = 110

# Transport states
IDLE = "idle"
RECORDING = "recording"
PAUSED = "paused"
TRANSCRIBING = "transcribing"


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
            threading.Thread(target=cb, daemon=True).start()


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


class TransportWindow:
    """Small floating panel with ⏺ ⏸ ⏹ transport buttons.

    Callbacks:
        record  — called when ⏺ is clicked (start or resume)
        pause   — called when ⏸ is clicked
        stop    — called when ⏹ is clicked (stop + transcribe)
    """

    def __init__(self, callbacks: dict):
        self._callbacks = callbacks
        self._window = None
        self._delegate = None
        self._btn_record = None
        self._btn_pause = None
        self._btn_stop = None
        self._status_label = None

    def show(self) -> None:
        if self._window is None:
            self._build()
        self._window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

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
            PAUSED:       ("▶", "⏸", "⏹"),
            TRANSCRIBING: ("⏺", "⏸", "⏹"),
        }.get(state, ("⏺", "⏸", "⏹"))

        self._btn_record.setEnabled_(enabled[0])
        self._btn_pause.setEnabled_(enabled[1])
        self._btn_stop.setEnabled_(enabled[2])
        self._btn_record.setTitle_(titles[0])
        self._status_label.setStringValue_(status)

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

        # Buttons row — centred in window
        btn_y = 50
        self._btn_record = _make_button("⏺", 18,  btn_y, "onRecord:", self._delegate)
        self._btn_pause  = _make_button("⏸", 96,  btn_y, "onPause:",  self._delegate)
        self._btn_stop   = _make_button("⏹", 174, btn_y, "onStop:",   self._delegate)
        cv.addSubview_(self._btn_record)
        cv.addSubview_(self._btn_pause)
        cv.addSubview_(self._btn_stop)

        # Status label
        self._status_label = _make_label(10, 20, WIN_W - 20, 20)
        cv.addSubview_(self._status_label)

        self.set_state(IDLE, "Ready")
