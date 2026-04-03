"""Meeting source detection — identify the active meeting platform."""

import subprocess


_PROCESS_MAP = [
    ("zoom", "zoom.us"),
    ("teams", "Microsoft Teams"),
    ("meet", "Google Chrome"),  # imperfect but acceptable fallback for Meet
]

_DISPLAY_MAP = {
    "zoom": "Zoom",
    "teams": "Microsoft Teams",
    "meet": "Google Meet",
    "in-person": "",
    "other": "",
}


def _process_running(name: str) -> bool:
    """Return True if a process with the given name is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_source() -> str:
    """Return 'zoom', 'teams', 'meet', or 'in-person' based on running processes."""
    if _process_running("zoom.us"):
        return "zoom"
    if _process_running("Microsoft Teams"):
        return "teams"
    if _process_running("Google Chrome"):
        # Chrome could be Meet — label it 'meet'; caller can override
        return "meet"
    return "in-person"


def infer_location(source: str) -> str:
    """Return a display location string for the given source identifier."""
    return _DISPLAY_MAP.get(source, "")
