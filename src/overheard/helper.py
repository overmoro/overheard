"""Locating overheard-helper, the native support binary.

One binary serves both jobs that need macOS APIs Python cannot reach: Core Audio
process taps for capture, and FluidAudio speaker diarization on the Neural
Engine. Both callers resolve it through here.

Built by scripts/build-helper.sh and committed to Resources/, so a working
install never needs a Swift toolchain.
"""

import os
import platform
import sys
from pathlib import Path

# Core Audio process taps landed in macOS 14.2, but the aggregate-device
# plumbing capture relies on is only dependable from 14.4.
MIN_MACOS = (14, 4)


def _candidates() -> list[Path]:
    """Locations to look for the helper, bundle first then repo checkout."""
    paths: list[Path] = []

    # Inside a standalone .app the sources live under Contents/Resources/lib/...,
    # so walking up from __file__ does not reach the bundle's Resources folder.
    # Derive it from the running executable instead.
    try:
        for parent in Path(sys.executable).resolve().parents:
            if parent.suffix == ".app":
                paths.append(parent / "Contents" / "Resources" / "overheard-helper")
                break
    except OSError:
        pass

    # Repo checkout, and py2app alias builds which run straight from the tree
    root = Path(__file__).parent.parent.parent
    paths += [
        root / "Resources" / "overheard-helper",
        root / "helper" / ".build" / "release" / "overheard-helper",
        root / "helper" / ".build" / "arm64-apple-macosx" / "release" / "overheard-helper",
    ]
    return paths


def helper_path() -> Path | None:
    """Path to the helper binary, or None if it hasn't been built."""
    for candidate in _candidates():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def macos_version() -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in platform.mac_ver()[0].split(".")[:2])
    except (ValueError, IndexError):
        return (0,)


def capture_available() -> tuple[bool, str]:
    """Return (available, reason) for tap-based capture. Reason is empty when available."""
    if sys.platform != "darwin":
        return False, "not macOS"
    if macos_version() < MIN_MACOS:
        have = ".".join(str(p) for p in macos_version())
        return False, f"macOS {have} predates process taps (needs 14.4)"
    if helper_path() is None:
        return False, "overheard-helper not built"
    return True, ""


#: Where the helper downloads FluidAudio's compiled models. Module level so a
#: test can point it somewhere hermetic instead of reading the real machine.
FLUIDAUDIO_MODELS = (
    Path.home() / "Library" / "Application Support" / "FluidAudio" / "Models"
)

#: The compiled CoreML bundles diarization cannot run without. FluidAudio ships
#: these as .mlmodelc directories rather than as single files.
_REQUIRED_DIARIZATION_MODELS = (
    "speaker-diarization/Segmentation.mlmodelc",
    "speaker-diarization/Embedding.mlmodelc",
)


def diarization_models_present() -> bool:
    """True when FluidAudio has the compiled models diarization needs.

    The helper downloads these on first use into Application Support. Checking
    for them is what lets Preferences say whether a download is actually
    pending, rather than asserting one always is.

    The previous test was "the Models directory exists and is not empty", which
    is the same lie is_model_cached was rewritten to stop telling: a download
    interrupted early leaves the directory created with a config.json in it and
    no weights, Preferences reports the models installed, and the user finds
    out at the end of a meeting. This names the bundles instead. They are
    directories, so an empty one is a download that did not finish.
    """
    def is_populated(relative: str) -> bool:
        bundle = FLUIDAUDIO_MODELS / relative
        return bundle.is_dir() and any(bundle.iterdir())

    return all(is_populated(name) for name in _REQUIRED_DIARIZATION_MODELS)
