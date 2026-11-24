#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================"
echo " Aniimo Sanctum Finder - Linux Installer"
echo "============================================"
echo ""

chmod +x "$DIR/launch.sh" "$DIR/main.py"

if [ ! -x "$DIR/.venv/bin/python" ]; then
  echo "[1/2] Creating virtual environment in .venv..."
  python3 -m venv "$DIR/.venv"
else
  echo "[1/2] Virtual environment .venv already exists."
fi

echo "[2/2] Installing required packages..."
"$DIR/.venv/bin/python" -m pip install -q -r "$DIR/requirements.txt"

echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo " To bind a hotkey in KDE Plasma (Wayland / X11):"
echo "  1. Open System Settings -> Keyboard -> Shortcuts"
echo "  2. Click + Add New -> Command or Script"
echo "  3. Paste this exact path:"
echo "     $DIR/launch.sh"
echo "  4. Assign your shortcut (e.g. Meta+M or Ctrl+Shift+X) and click Apply."
echo ""
