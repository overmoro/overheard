"""Every setting Overheard has, declared once.

Before this module, a default could be written in two places: ``DEFAULTS`` in
config.py, and again as the second argument at each ``cfg.get(key, default)``
call site. There were fifteen of those, and two of them disagreed with the
dictionary they were supposedly restating. Nothing could detect that, because
both readings were defensible: whichever one the code path reached was, in
effect, the default.

So the default now lives in exactly one place, a field declaration, and ``get``
takes no default argument. A call site cannot restate a default because there is
no longer anywhere to put it.

Unknown keys are rejected rather than passed through. A typo in a key name used
to read as "absent, use the fallback", which is silence where an error belongs.
"""

import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "overheard"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _default_output_dir() -> str:
    return str(Path.home() / "overheard" / "transcripts")


@dataclass(frozen=True)
class Settings:
    """The whole configuration surface, with types and defaults.

    Frozen because a settings object is a reading of the file at a moment, not
    a mutable handle on it. Changing a setting goes through ``set_value``, which
    writes to disk; anything else would leave two answers to the same question.
    """

    #: Where transcripts are written when Obsidian output is off.
    output_dir: str = field(default_factory=_default_output_dir)

    #: Write transcripts into an Obsidian vault instead of output_dir.
    obsidian_enabled: bool = False
    obsidian_vault: str = ""
    obsidian_inbox: str = "01_Inbox"

    #: Name given to the voice on the microphone track. This is the local
    #: speaker in the finished transcript and in the live panel alike; the two
    #: used to disagree, showing "Don" in one and "You" in the other.
    local_speaker_name: str = "Don"

    #: Transcription engine: "parakeet" (MLX, GPU) or "whisper" (WhisperX, CPU).
    engine: str = "parakeet"

    #: Show a running transcript while recording (parakeet engine only).
    live_preview: bool = True

    #: Capture backend: "auto" (tap when available), "tap", or "device".
    capture_backend: str = "auto"

    #: Diarizer: "auto" (FluidAudio, falling back to pyannote), "fluidaudio",
    #: or "pyannote". pyannote needs HF_TOKEN and the optional extra installed.
    diarizer: str = "auto"

    #: Label speakers in the live transcript using streaming diarization.
    #: Needs separate mic and system tracks; at most four concurrent speakers.
    live_speakers: bool = True

    #: Remember voices between meetings so returning speakers are recognised
    #: without relying on the attendee list order.
    speaker_memory: bool = True

    #: Cosine similarity required to treat a voice as a known person. Raise it
    #: if wrong names appear; lower it if returning speakers go unrecognised.
    speaker_match_threshold: float = 0.70

    #: Keep the recorded WAV alongside the transcript instead of deleting it.
    keep_recordings: bool = False

    #: Hugging Face token, consulted only by the optional pyannote fallback.
    hf_token: str = ""


_FIELDS = {f.name: f for f in fields(Settings)}

#: Key to default value. Derived, never hand-written.
DEFAULTS: dict = {name: getattr(Settings(), name) for name in _FIELDS}


def keys() -> list[str]:
    return list(_FIELDS)


def _coerce(key: str, value, default):
    """Bring a stored value to the declared type, or fall back to the default.

    The config file is hand-editable, so a value can arrive as the wrong type.
    Coercing here means a call site never has to defend itself, and a value too
    mangled to coerce becomes the default rather than an exception halfway
    through a transcription.
    """
    declared = _FIELDS[key].type

    if declared is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if value in (0, 1):
            return bool(value)
    elif declared is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    elif declared is str:
        if isinstance(value, str):
            return value
    else:
        return value

    print(f"[overheard] config key {key!r} has an unusable value {value!r}; "
          f"using the default {default!r}", file=sys.stderr)
    return default


def read_raw() -> dict:
    """The config file as it is on disk, or {} if unreadable."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load() -> Settings:
    """Read settings from disk, coerced and completed with defaults."""
    stored = read_raw()
    values = {}
    for name, default in DEFAULTS.items():
        if name in stored:
            values[name] = _coerce(name, stored[name], default)
    return Settings(**values)


def load_dict() -> dict:
    """The same reading as ``load``, as a plain dict."""
    settings = load()
    return {name: getattr(settings, name) for name in _FIELDS}


def save(changes: dict) -> None:
    """Merge changes into the config file.

    Keys the schema does not know about are kept rather than dropped: they may
    belong to a newer version of the app that ran against the same file.
    """
    unknown = set(changes) - set(_FIELDS)
    if unknown:
        raise KeyError(f"unknown setting(s): {sorted(unknown)}")

    merged = {**read_raw(), **load_dict(), **changes}
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(merged, f, indent=2)
    except OSError as e:
        print(f"[overheard] could not write config: {e}", file=sys.stderr)


def get(key: str):
    """One setting, by name.

    There is deliberately no ``default`` argument. The default is the field
    declaration above; a second one written at the call site is how the two
    config bugs this module fixes came about.
    """
    if key not in _FIELDS:
        raise KeyError(f"unknown setting: {key!r}")
    return getattr(load(), key)


def set_value(key: str, value) -> None:
    """Set one setting and persist it."""
    save({key: value})
