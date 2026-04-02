#!/usr/bin/env bash
# setup-macos.sh — Install system dependencies for overheard on macOS.
# Run this once on a fresh machine. Requires sudo for BlackHole.
set -euo pipefail

echo "=== Overheard — macOS Setup ==="
echo ""

# --- Homebrew ---
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo ""
fi

# --- Python ---
if ! command -v python3 &>/dev/null; then
    echo "Installing Python..."
    brew install python@3.11
else
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python $PY_VERSION found."
fi

# --- ffmpeg ---
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "ffmpeg found."
fi

# --- BlackHole 2ch ---
if ! brew list --cask blackhole-2ch &>/dev/null 2>&1; then
    echo ""
    echo "Installing BlackHole 2ch (requires sudo)..."
    brew install blackhole-2ch
    echo ""
    echo "*** REBOOT REQUIRED after BlackHole installation ***"
    NEEDS_REBOOT=true
else
    echo "BlackHole 2ch found."
    NEEDS_REBOOT=false
fi

# --- Output directory ---
mkdir -p "$HOME/meeting-transcripts"

echo ""
echo "=== System dependencies installed ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Install the Python package:"
echo "     pip install -e ."
echo ""
echo "  2. Set your Hugging Face token:"
echo "     export HF_TOKEN=your_token_here"
echo "     (Add to ~/.zshrc for persistence)"
echo ""
echo "  3. Create the Aggregate Audio Device:"
echo "     - Open 'Audio MIDI Setup' (Spotlight → Audio MIDI Setup)"
echo "     - Click '+' in the bottom-left → 'Create Aggregate Device'"
echo "     - Rename it to 'Meeting Capture'"
echo "     - Check both 'BlackHole 2ch' and 'MacBook Pro Microphone'"
echo "     - Set BlackHole 2ch as the clock source"
echo ""
echo "  4. Route system audio through BlackHole:"
echo "     - Open 'Audio MIDI Setup'"
echo "     - Click '+' → 'Create Multi-Output Device'"
echo "     - Check both 'MacBook Pro Speakers' and 'BlackHole 2ch'"
echo "     - Set this Multi-Output as your system output in System Settings → Sound"
echo "     (This sends system audio to both your speakers AND BlackHole for capture)"
echo ""

if [ "${NEEDS_REBOOT:-false}" = true ]; then
    echo "  *** REBOOT YOUR MAC before proceeding — BlackHole needs it ***"
    echo ""
fi

echo "  5. Run the app:"
echo "     overheard"
