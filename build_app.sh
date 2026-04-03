#!/usr/bin/env bash
# build_app.sh — Build Overheard.app using py2app alias mode (development build)
#
# Alias mode:  sources are used directly from the project tree — no copy.
#              The .app is a thin wrapper; the source must stay in place.
#              This is fast, ideal for development.
#
# Full standalone (production):
#   Temporarily rename pyproject.toml (same trick as below), then run:
#     python setup.py py2app
#   Note: ML models (~4 GB) are NOT bundled. Users need the Python env in place.

set -euo pipefail

PYTHON="/opt/homebrew/opt/python@3.11/bin/python3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building Overheard.app (alias mode)..."
cd "$PROJECT_DIR"

# Clean previous build artifacts to avoid stale state
rm -rf build dist

# py2app 0.28+ forbids install_requires on the distribution object.
# setuptools auto-reads pyproject.toml and sets install_requires, which causes
# the build to fail. We temporarily rename pyproject.toml so setuptools skips it.
PYPROJECT="$PROJECT_DIR/pyproject.toml"
PYPROJECT_BAK="$PROJECT_DIR/pyproject.toml.py2app_bak"

_restore() {
    if [ -f "$PYPROJECT_BAK" ]; then
        mv "$PYPROJECT_BAK" "$PYPROJECT"
    fi
}
trap _restore EXIT

mv "$PYPROJECT" "$PYPROJECT_BAK"

"$PYTHON" setup.py py2app --alias 2>&1

# pyproject.toml restored by trap

APP_PATH="$PROJECT_DIR/dist/Overheard.app"

if [ -d "$APP_PATH" ]; then
    echo ""
    echo "Build succeeded."
    echo "App location: $APP_PATH"
    echo ""
    echo "To open the dist folder:"
    echo "  open \"$PROJECT_DIR/dist\""
    echo ""
    # Uncomment to auto-open dist/ after build:
    # open "$PROJECT_DIR/dist"
else
    echo "Build failed — dist/Overheard.app not found." >&2
    exit 1
fi
