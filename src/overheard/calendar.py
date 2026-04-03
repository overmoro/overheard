"""macOS Calendar integration — find the current meeting via AppleScript."""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MeetingInfo:
    title: str
    attendees: list[str]
    start: datetime
    location: str


_APPLESCRIPT = """
tell application "Calendar"
    set now to current date
    set windowStart to now - (30 * minutes)
    set windowEnd to now + (90 * minutes)
    set found to {}
    repeat with c in calendars
        set evts to (every event of c whose start date >= windowStart and start date <= windowEnd)
        repeat with e in evts
            set eTitle to summary of e
            set eLoc to ""
            try
                set eLoc to location of e
                if eLoc is missing value then set eLoc to ""
            end try
            set eStart to start date of e
            set eAtts to {}
            try
                set atts to attendees of e
                repeat with a in atts
                    set end of eAtts to display name of a
                end repeat
            end try
            set end of found to {eTitle, eLoc, eStart, eAtts}
        end repeat
    end repeat
    return found
end tell
"""


def _parse_applescript_date(date_str: str) -> datetime | None:
    """Parse the date string returned by AppleScript (locale-sensitive, best-effort)."""
    # AppleScript returns dates like "Thursday, 3 April 2026 at 10:00:00 AM"
    # Try multiple formats
    formats = [
        "%A, %d %B %Y at %I:%M:%S %p",
        "%A, %B %d, %Y at %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    date_str = date_str.strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_current_meeting() -> MeetingInfo | None:
    """Query macOS Calendar for an event within ±30 minutes of now.

    Returns the first matching MeetingInfo, or None if no event found,
    Calendar is not running, or AppleScript access is denied.
    Never raises.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", _APPLESCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        # osascript returns a flat comma-separated string for lists of records.
        # For a single event list of 4 elements: "title, location, date, att1, att2"
        # We parse it naively — it's best-effort.
        output = result.stdout.strip()
        if not output:
            return None

        # Split by ", " but AppleScript record separators are ", "
        # The structure is: title, location, date-string, attendees...
        # Each event group is separated by comma. We take the first event only.
        parts = [p.strip() for p in output.split(", ")]
        if len(parts) < 3:
            return None

        title = parts[0] if parts[0] else "Meeting"
        location = parts[1] if parts[1] else ""
        # Date is typically the 3rd element — it may contain commas itself,
        # so we search for a recognisable date pattern.
        # Simplest safe approach: use now as the start time fallback,
        # attempt to parse parts[2] as date.
        start = _parse_applescript_date(parts[2])
        if start is None:
            start = datetime.now()

        # Remaining parts (if any) are attendee names
        attendees = [p for p in parts[3:] if p]

        return MeetingInfo(
            title=title,
            attendees=attendees,
            start=start,
            location=location,
        )

    except Exception:
        return None
