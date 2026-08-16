"""Meeting source detection: identify the active meeting platform."""

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


# Bundle identifiers of apps that host meetings, mapped to our source keys.
# Browsers are lumped together as 'meet': the specific service isn't knowable
# without reading tab titles, which would need Screen Recording permission.
_BUNDLE_MAP = {
    "us.zoom.xos": "zoom",
    "com.microsoft.teams": "teams",
    "com.microsoft.teams2": "teams",
    "com.google.chrome": "meet",
    "com.apple.safari": "meet",
    "com.brave.browser": "meet",
    "org.mozilla.firefox": "meet",
    "com.microsoft.edgemac": "meet",
}


def _audio_processes() -> list[dict] | None:
    """Processes currently running audio, or None if the helper is unavailable."""
    import json

    try:
        from overheard.helper import helper_path

        binary = helper_path()
        if binary is None:
            return None
        result = subprocess.run(
            [str(binary), "processes"], capture_output=True, timeout=10
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.decode("utf-8", "replace"))
    except Exception:
        return None


def detect_source() -> str:
    """Return 'zoom', 'teams', 'meet', or 'in-person'.

    Asks Core Audio which processes are actually using the microphone, rather
    than which applications happen to be open. Checking for a running process
    meant a backgrounded Chrome reported a Google Meet on every recording, and
    that wrong value went straight into the transcript's frontmatter.

    Capturing input is the signal that distinguishes a call from a browser
    playing a video, which is output only.
    """
    processes = _audio_processes()
    if processes is not None:
        for process in processes:
            if not process.get("input"):
                continue
            bundle_id = (process.get("bundleID") or "").lower()
            source = _BUNDLE_MAP.get(bundle_id)
            if source:
                return source
        # Audio state was readable and nothing is in a call
        return "in-person"

    # Helper unavailable: fall back to asking which apps are running, accepting
    # that an open browser will be misreported.
    if _process_running("zoom.us"):
        return "zoom"
    if _process_running("Microsoft Teams"):
        return "teams"
    if _process_running("Google Chrome"):
        return "meet"
    return "in-person"


def infer_location(source: str) -> str:
    """Return a display location string for the given source identifier."""
    return _DISPLAY_MAP.get(source, "")
