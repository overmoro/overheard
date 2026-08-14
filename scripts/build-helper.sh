#!/usr/bin/env bash
# build-helper.sh - Build overheard-helper and stage it into Resources/.
#
# The helper is the one native component: Core Audio process taps for capture,
# and FluidAudio speaker diarization on the Neural Engine. Neither is reachable
# from Python.
#
# The built binary is committed to Resources/ so end users never need a Swift
# toolchain. Rerun this after changing anything under helper/, or to pick up a
# newer FluidAudio.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER_DIR="$PROJECT_DIR/helper"
DEST_DIR="$PROJECT_DIR/Resources"

if ! command -v swift &>/dev/null; then
    echo "swift not found. Install Xcode Command Line Tools:" >&2
    echo "  xcode-select --install" >&2
    exit 1
fi

echo "==> Building overheard-helper (release)..."
cd "$HELPER_DIR"
swift build -c release

BINARY="$(swift build -c release --show-bin-path)/overheard-helper"
if [ ! -x "$BINARY" ]; then
    echo "Build reported success but $BINARY is missing." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp "$BINARY" "$DEST_DIR/overheard-helper"
echo "==> Staged: $DEST_DIR/overheard-helper"

# Ad-hoc signature. Enough for local use and for the audio-capture TCC prompt to
# attach to a stable identity. Distribution needs a real Developer ID signature
# and notarisation; see docs.
if command -v codesign &>/dev/null; then
    codesign --force --sign - "$DEST_DIR/overheard-helper" 2>/dev/null \
        && echo "==> Ad-hoc signed" \
        || echo "==> codesign failed (non-fatal for local use)" >&2
fi

echo ""
"$DEST_DIR/overheard-helper" --version
echo "Done."
