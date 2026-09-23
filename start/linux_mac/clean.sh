#!/usr/bin/env bash
# =============================================================================
# CLEAN (Linux / macOS / Google Colab) — Auto Cut & Polishing Tool
# Location: start/linux_mac/clean.sh
# Program code is inside:  ../../program/
# =============================================================================

set -e

SCRIPT_PATH="$0"
while [ -L "$SCRIPT_PATH" ]; do
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
  SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
  case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH" ;;
  esac
done
DIR="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"

ROOT_DIR="$(cd "$DIR/../.." >/dev/null 2>&1 && pwd)"
PROGRAM_DIR="$ROOT_DIR/program"

if [ ! -d "$PROGRAM_DIR" ]; then
    echo "[ERROR] Cannot find program/ folder at: $PROGRAM_DIR"
    exit 1
fi

echo "================================================================"
echo "  CLEAN (Linux / macOS / Colab) — Auto Cut & Polishing Tool"
echo "================================================================"
echo ""
echo "This will clean:"
echo "  - Support folder (cache, temp, AI binaries)  [program/support/]"
echo "  - Python bytecode                            [__pycache__, *.pyc]"
echo ""
echo "Note: Output folder [output/] is ALWAYS kept safe and never touched."
echo ""

DEL_VENV=""
CONFIRM=""

for arg in "$@"; do
    case "$arg" in
        --venv|-v)
            DEL_VENV="Y"
            CONFIRM="Y"
            ;;
        -y|--yes)
            CONFIRM="Y"
            ;;
    esac
done

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    read -rp "Proceed with cleaning support folder and temporary files? (y/N): " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

if [[ -z "$DEL_VENV" ]]; then
    echo ""
    read -rp "Also delete Python virtual environment (program/venv)? (y/N): " DEL_VENV
fi

echo ""
echo "Cleaning in progress..."

if [ -d "$PROGRAM_DIR/support" ]; then
    rm -rf "$PROGRAM_DIR/support"
    echo "  [OK] Removed program/support/ folder completely."
fi

if [ -L "$HOME/.cache/voicefixer" ] || [ -d "$HOME/.cache/voicefixer" ]; then
    rm -rf "$HOME/.cache/voicefixer" 2>/dev/null || true
    echo "  [OK] Removed VoiceFixer cache symlink."
fi

find "$PROGRAM_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROGRAM_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  [OK] Bytecode cache cleared."

if [[ "$DEL_VENV" =~ ^[Yy]$ ]]; then
    if [ -d "$PROGRAM_DIR/venv" ]; then
        rm -rf "$PROGRAM_DIR/venv"
        echo "  [OK] Deleted program/venv"
    fi
fi

echo "  [OK] Output folder at $ROOT_DIR/output kept safe."
echo ""
echo "================================================================"
echo "  CLEANUP COMPLETE!"
echo "================================================================"
