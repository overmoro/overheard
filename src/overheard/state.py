"""The states the recorder can be in.

These four strings were the only living part of transport.py, a 308-line
floating-panel UI that popover.py superseded. A module named transport that
holds nothing but four strings misleads whoever opens it next, so they moved
here instead of the file staying behind to house them.

The values are the strings themselves rather than an enum: they cross into
AppKit callbacks and get compared against plain literals in more than one place,
and an enum would only add a ``.value`` at every boundary for no protection the
type checker cannot already give.
"""

IDLE = "idle"
RECORDING = "recording"
PAUSED = "paused"
TRANSCRIBING = "transcribing"

#: Every state, in the order a recording passes through them.
ALL = (IDLE, RECORDING, PAUSED, TRANSCRIBING)
