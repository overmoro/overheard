"""Live transcript window.

A separate floating panel rather than an expansion of the transport popover:
the popover's layout constants are derived bottom-up from a fixed height, and a
resizable scrolling text view does not belong inside that. Keeping it separate
also lets the transcript stay open and resized while the popover is dismissed.

The panel owns a main-thread NSTimer that polls the LiveTranscriber for new
text. Nothing here is ever called from the transcription worker thread, so all
AppKit access stays on the main thread.
"""

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSScrollView,
    NSTextView,
    NSMakeSize,
    NSMakeRange,
)
from Foundation import NSObject, NSTimer

PANEL_W = 380
PANEL_H = 420

# How often the panel pulls new text from the transcriber
POLL_INTERVAL = 0.3


def _style_mask() -> int:
    """Titled + closable + resizable, across PyObjC naming variants."""
    try:
        from AppKit import (
            NSWindowStyleMaskTitled,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskResizable,
        )
        return NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
    except ImportError:
        from AppKit import NSTitledWindowMask, NSClosableWindowMask, NSResizableWindowMask
        return NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask


class _PollTarget(NSObject):
    """NSTimer target that forwards ticks to the owning panel."""

    def initWithPanel_(self, panel):
        self = objc.super(_PollTarget, self).init()
        if self is None:
            return None
        self._panel = panel
        return self

    def onTick_(self, timer):
        try:
            self._panel._tick()
        except Exception:
            import traceback
            traceback.print_exc()


class LiveTranscriptPanel:
    """Floating window showing the running live transcript."""

    def __init__(self):
        self._panel = None
        self._text_view = None
        self._timer = None
        self._poll_target = None
        self._transcriber = None
        self._last_text = None
        self._last_status = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind(self, transcriber) -> None:
        """Attach a LiveTranscriber and begin polling it."""
        if self._panel is None:
            self._build()
        self._transcriber = transcriber
        self._last_text = None
        self._last_status = None
        self._set_text("")
        self._start_timer()

    def unbind(self) -> None:
        """Stop polling but leave the final text on screen."""
        self._stop_timer()
        self._transcriber = None

    def show(self) -> None:
        if self._panel is None:
            self._build()
        self._position()
        # orderFront_ rather than makeKeyAndOrderFront_: this is a read-only
        # view, and taking key status would steal focus from the meeting app.
        self._panel.orderFront_(None)

    def hide(self) -> None:
        if self._panel:
            self._panel.orderOut_(None)

    def is_visible(self) -> bool:
        return bool(self._panel and self._panel.isVisible())

    def toggle(self) -> None:
        if self.is_visible():
            self.hide()
        else:
            self.show()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        if self._timer is not None and self._timer.isValid():
            return
        self._poll_target = _PollTarget.alloc().initWithPanel_(self)
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            POLL_INTERVAL, self._poll_target, "onTick:", None, True
        )

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
        self._timer = None
        self._poll_target = None

    def _tick(self) -> None:
        """Main-thread poll: pull text from the transcriber if it changed."""
        t = self._transcriber
        if t is None or self._panel is None or not self._panel.isVisible():
            return

        status = t.status
        if status != self._last_status:
            self._last_status = status
            self._panel.setTitle_(f"Live Transcript  ({status})")

        text = self._render(t)
        if text != self._last_text:
            self._last_text = text
            self._set_text(text)

    def _render(self, transcriber) -> str:
        """Attribute each sentence to a speaker and lay it out for display.

        Two sources of attribution, used in order of how much they can be
        trusted. Which track carried the audio is a measurement, so it decides
        local versus remote. Among remote voices, the streaming diarizer's
        speaker index is the only signal available live.
        """
        sentences = transcriber.sentences
        if not sentences:
            return transcriber.text

        from overheard import config as cfg

        local_name = cfg.get("local_speaker_name") or cfg.DEFAULTS["local_speaker_name"]
        diarizer = getattr(transcriber, "diarizer", None)

        lines: list[str] = []
        previous: str | None = None
        # Sortformer's indices are positional and can start anywhere, so number
        # remote voices by when they first speak. The local speaker is named
        # separately, so a lone "Speaker 2" with no Speaker 1 would just puzzle.
        display_numbers: dict[int, int] = {}

        for sentence in sentences:
            share = transcriber.local_share(sentence["start"], sentence["end"])
            if share > 0.60:
                speaker = local_name
            elif diarizer is not None and diarizer.available:
                index = diarizer.speaker_at((sentence["start"] + sentence["end"]) / 2)
                if index is None:
                    speaker = "Speaker"
                else:
                    if index not in display_numbers:
                        display_numbers[index] = len(display_numbers) + 1
                    speaker = f"Speaker {display_numbers[index]}"
            else:
                speaker = "Remote"

            if speaker != previous:
                if lines:
                    lines.append("")
                lines.append(f"{speaker}:")
                previous = speaker
            lines.append(sentence["text"])

        return "\n".join(lines)

    def _set_text(self, text: str) -> None:
        if self._text_view is None:
            return
        self._text_view.setString_(text or "")
        # Keep the newest text in view
        length = len(self._text_view.string())
        self._text_view.scrollRangeToVisible_(NSMakeRange(length, 0))

    def _position(self) -> None:
        """Place the panel at the top right, clear of the transport popover."""
        from AppKit import NSScreen
        try:
            vf = NSScreen.mainScreen().visibleFrame()
            x = vf.origin.x + vf.size.width - PANEL_W - 20
            y = vf.origin.y + vf.size.height - PANEL_H - 40
            self._panel.setFrameOrigin_((x, y))
        except Exception:
            pass

    def _build(self) -> None:
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PANEL_W, PANEL_H),
            _style_mask(),
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Live Transcript")
        panel.setFloatingPanel_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setMinSize_(NSMakeSize(260, 200))

        cv = panel.contentView()
        bounds = cv.bounds()

        scroll = NSScrollView.alloc().initWithFrame_(bounds)
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(0)
        scroll.setDrawsBackground_(False)
        # Track the window as it is resized
        scroll.setAutoresizingMask_((1 << 1) | (1 << 4))  # WidthSizable | HeightSizable

        text_view = NSTextView.alloc().initWithFrame_(bounds)
        text_view.setEditable_(False)
        text_view.setSelectable_(True)
        text_view.setRichText_(False)
        text_view.setDrawsBackground_(False)
        text_view.setFont_(NSFont.systemFontOfSize_(13))
        text_view.setTextColor_(NSColor.labelColor())
        text_view.setTextContainerInset_(NSMakeSize(10, 10))
        text_view.setAutoresizingMask_(1 << 1)  # WidthSizable
        text_view.setVerticallyResizable_(True)
        text_view.setHorizontallyResizable_(False)
        container = text_view.textContainer()
        if container is not None:
            container.setWidthTracksTextView_(True)

        scroll.setDocumentView_(text_view)
        cv.addSubview_(scroll)

        self._panel = panel
        self._text_view = text_view
