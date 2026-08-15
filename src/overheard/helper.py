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

#: Copied from FluidAudio's ModelNames.OfflineDiarizer.requiredModels, which is
#: the set OfflineDiarizerModels.load() hard-fails without. The helper calls
#: exactly that (DiarizeCommand.swift), so anything short of the full list is a
#: download that cannot diarize. Naming only two of them reported "Installed"
#: with the other three missing.
_REQUIRED_DIARIZATION_MODELS = (
    "Segmentation.mlmodelc",
    "FBank.mlmodelc",
    "Embedding.mlmodelc",
    "PldaRho.mlmodelc",
    "plda-parameters.json",
)

#: The folder FluidAudio actually loads from: ModelHub resolves the diarizer to
#: exactly one path, built from Repo.diarizer.folderName. Checked first, so a
#: complete legacy sibling cannot vouch for a half-downloaded live one.
_DIARIZATION_SUBDIR = "speaker-diarization-coreml"

#: Consulted only when the folder above is absent entirely. FluidAudio probes
#: three names when loading the PLDA parameters, which is direct evidence this
#: path has moved before and will again, and a rename would otherwise leave the
#: label reading Missing while the helper finds everything and returns in under
#: a second: a Download button that completes instantly, changes nothing, and
#: never stops asking.
_DIARIZATION_SUBDIR_GLOB = "speaker-diarization*"


def _bundle_has_weights(bundle: Path) -> bool:
    """True when a compiled CoreML bundle holds actual weights.

    A .mlmodelc is a directory. "Not empty" is not enough: a partial download
    leaves analytics/coremldata.bin and metadata.json in place with no weights
    at all, which is the same directory-is-not-empty lie one level down. The
    weights live at weights/weight.bin, or model<N>/weights/<N>-weight.bin for
    multi-function models, so both shapes are matched.
    """
    return any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in bundle.glob("**/weights/*.bin")
    )


def diarization_models_present() -> bool:
    """True when FluidAudio has the compiled models diarization needs.

    The helper downloads these on first use into Application Support. Checking
    for them is what lets Preferences say whether a download is actually
    pending, rather than asserting one always is.

    The original test was "the Models directory exists and is not empty", which
    is the lie is_model_cached was rewritten to stop telling: a download
    interrupted early leaves the folder created with a config.json in it,
    Preferences reports the models installed, and the user finds out at the end
    of a meeting. This requires every artefact FluidAudio declares mandatory,
    and requires each compiled bundle to actually contain weights.
    """
    if not FLUIDAUDIO_MODELS.is_dir():
        return False

    def complete(parent: Path) -> bool:
        return all(_present(parent / name) for name in _REQUIRED_DIARIZATION_MODELS)

    # The folder the helper will really load from decides the answer whenever
    # it exists, complete or not. Falling through to a sibling here is how a
    # leftover legacy folder ends up reporting "Installed" for a live folder
    # that is still half downloaded.
    canonical = FLUIDAUDIO_MODELS / _DIARIZATION_SUBDIR
    if canonical.is_dir():
        return complete(canonical)

    return any(
        parent.is_dir() and complete(parent)
        for parent in sorted(FLUIDAUDIO_MODELS.glob(_DIARIZATION_SUBDIR_GLOB))
    )


def _present(artefact: Path) -> bool:
    """A compiled bundle with weights in it, or a non-empty plain file."""
    if artefact.name.endswith(".mlmodelc"):
        return artefact.is_dir() and _bundle_has_weights(artefact)
    return artefact.is_file() and artefact.stat().st_size > 0
