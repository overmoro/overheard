"""Keeping Python exceptions from crossing back into AppKit.

PyObjC dispatches actions, mouse events and delegate messages straight into
Python. An exception that unwinds out of one of those is not caught by anything:
it aborts the process below Python's own handlers, so there is no traceback, no
crash report, and nothing in the log. The app simply vanishes. Three separate
bugs hid behind that one symptom while this app was being built.

rumps already wraps the callbacks it owns, its menu items and its Timers, in
try/except (rumps.py:730 and :997). Nothing wraps ours. Every method on a
first-party NSObject or NSView subclass that AppKit can call is an unguarded
boundary, and there are enough of them that guarding by hand means guarding the
ones somebody thought of.

``objc_safe`` is the boundary. Put it on anything AppKit dispatches to.

A note on why the wrapper spells its arguments out. PyObjC builds a selector
from the method's name and its argument count, so a ``*args`` wrapper reports
zero arguments and the class fails to build at import. The wrapper therefore
takes exactly the (self, sender) shape that actions, mouse events and delegate
callbacks use.

Phase 3 replaces the print here with reporting.report_problem, so a user sees
these rather than only the log.
"""

import functools
import sys
import traceback


def objc_safe(method):
    """Stop an exception in an AppKit-dispatched method from killing the app.

    Logs with a traceback and returns None. Swallowing quietly would trade an
    invisible crash for an invisible no-op, which is not an improvement: the
    log is the only diagnosis a bundled .app has.
    """
    @functools.wraps(method)
    def wrapper(self, sender):
        try:
            return method(self, sender)
        except Exception as e:
            print(
                f"[overheard] {method.__qualname__} failed: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None

    return wrapper


def objc_safe_noarg(method):
    """objc_safe for the zero-argument shape, such as updateTrackingAreas."""
    @functools.wraps(method)
    def wrapper(self):
        try:
            return method(self)
        except Exception as e:
            print(
                f"[overheard] {method.__qualname__} failed: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return None

    return wrapper
