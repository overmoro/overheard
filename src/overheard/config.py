"""Config read/write helper for overheard.

Config is stored at ~/.config/overheard/config.json.
"""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "overheard"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULTS = {
    "output_dir": str(Path.home() / "overheard" / "transcripts"),
    "obsidian_enabled": False,
    "obsidian_vault": "",
    "obsidian_inbox": "01_Inbox",
    "local_speaker_name": "Don",
}


def load() -> dict:
    """Load config from disk, merging with defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            return {**DEFAULTS, **data}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULTS)


def save(data: dict) -> None:
    """Write config to disk, merging with any existing values."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    current = load()
    current.update(data)
    with open(CONFIG_PATH, "w") as f:
        json.dump(current, f, indent=2)


def get(key: str, default=None):
    """Get a single config value."""
    return load().get(key, default)


def set_value(key: str, value) -> None:
    """Set a single config value and persist it."""
    save({key: value})
