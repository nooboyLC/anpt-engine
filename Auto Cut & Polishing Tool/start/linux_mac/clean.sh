#!/usr/bin/env bash
# =============================================================================
# CLEAN (Linux / macOS / Google Colab) — Auto Cut & Polishing Tool
# Location: start/linux_mac/clean.sh
# Program code is inside:  ../../program/
# =============================================================================

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
SUPPORT_DIR="$ROOT_DIR/support"

if [ ! -d "$PROGRAM_DIR" ]; then
    echo "[ERROR] Cannot find program/ folder at: $PROGRAM_DIR"
    exit 1
fi

echo "================================================================"
echo "  CLEAN (Linux / macOS / Colab) — Auto Cut & Polishing Tool"
echo "================================================================"
echo ""
echo "This will completely wipe:"
echo "  - Entire Support folder (venv, AI models, cache, temp, binaries) [support/]"
echo "  - Python bytecode                                                [__pycache__, *.pyc]"
echo "  - Run logs and CSV files"
echo ""
echo "Note: Output folder [output/] is ALWAYS kept safe."
echo ""

CONFIRM=""
for arg in "$@"; do
    case "$arg" in
        --all|-a|--yes|-y|--venv|-v)
            CONFIRM="y"
            ;;
    esac
done

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    if [ -t 0 ]; then
        read -rp "Proceed with completely wiping the support folder and caches? (y/N): " CONFIRM
    else
        echo "[INFO] Non-interactive environment, proceeding with cleanup."
        CONFIRM="y"
    fi
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

echo ""
echo "Cleaning in progress..."

# 1. Completely remove the entire support/ directory
if [ -d "$SUPPORT_DIR" ]; then
    rm -rf "$SUPPORT_DIR"
    echo "  [OK] Completely deleted support/ folder (venv, models, cache, temp)."
else
    echo "  [OK] support/ folder was not present."
fi

# 2. Clean user voicefixer cache if symlinked
if [ -L "$HOME/.cache/voicefixer" ] || [ -d "$HOME/.cache/voicefixer" ]; then
    rm -rf "$HOME/.cache/voicefixer" 2>/dev/null || true
    echo "  [OK] Removed VoiceFixer cache symlink."
fi

# 3. Clear Python bytecode
find "$PROGRAM_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROGRAM_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  [OK] Python bytecode cache cleared."

# 4. Clear temporary logs
rm -f "$PROGRAM_DIR"/*.log "$ROOT_DIR"/*.log "$ROOT_DIR"/hardware_log_*.csv 2>/dev/null || true
echo "  [OK] Cleaned temporary logs."

echo "  [OK] Output folder at $ROOT_DIR/output kept safe."
echo ""
echo "================================================================"
echo "  CLEANUP COMPLETE!"
echo "================================================================"