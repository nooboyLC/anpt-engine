#!/usr/bin/env bash
# =============================================================================
# Auto Cut & Polishing Tool — run.sh (Linux / macOS / Google Colab)
# Location: start/linux_mac/run.sh
# Program code is inside:  ../../program/
# =============================================================================

set -e

# Resolve symlinks to get the real script directory
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

# Root is two levels up from start/linux_mac/
ROOT_DIR="$(cd "$DIR/../.." >/dev/null 2>&1 && pwd)"
PROGRAM_DIR="$ROOT_DIR/program"
SUPPORT_DIR="$ROOT_DIR/support"

if [ ! -d "$PROGRAM_DIR" ]; then
    echo "[ERROR] Cannot find program/ folder. Expected at: $PROGRAM_DIR"
    exit 1
fi

cd "$PROGRAM_DIR"

mkdir -p "$SUPPORT_DIR/temp" "$SUPPORT_DIR/checkpoints" "$SUPPORT_DIR/cache" 2>/dev/null || true

export TMPDIR="$SUPPORT_DIR/temp"
export TEMP="$SUPPORT_DIR/temp"
export TMP="$SUPPORT_DIR/temp"
export PIP_CACHE_DIR="$SUPPORT_DIR/temp/pip_cache"
export TORCH_HOME="$SUPPORT_DIR/checkpoints"
export TORCH_EXTENSIONS_DIR="$SUPPORT_DIR/cache/torch_extensions"
export HF_HOME="$SUPPORT_DIR/checkpoints"
export TRANSFORMERS_CACHE="$SUPPORT_DIR/checkpoints/transformers"
export HF_DATASETS_CACHE="$SUPPORT_DIR/checkpoints/datasets"
export HUGGINGFACE_HUB_CACHE="$SUPPORT_DIR/checkpoints/hub"
export ASTEROID_CACHE="$SUPPORT_DIR/checkpoints/asteroid"
export SPEECHBRAIN_CACHE="$SUPPORT_DIR/checkpoints/speechbrain"
export SENTENCE_TRANSFORMERS_HOME="$SUPPORT_DIR/checkpoints/sentence_transformers"
export PYTHONUSERBASE="$SUPPORT_DIR/py_userbase"
export XDG_CACHE_HOME="$SUPPORT_DIR/cache"
export MPLCONFIGDIR="$SUPPORT_DIR/cache/matplotlib"
export VOICEFIXER_CACHE="$SUPPORT_DIR/checkpoints/voicefixer"
export VOICEFIXER_HOME="$SUPPORT_DIR/checkpoints/voicefixer"
export PATH="$SUPPORT_DIR/bin:$PATH"

# Find Python — prefer local venv
PY_CMD=""
if [ -f "$SUPPORT_DIR/venv/bin/python" ]; then
    PY_CMD="$SUPPORT_DIR/venv/bin/python"
elif [ -f "$SUPPORT_DIR/venv/bin/python3" ]; then
    PY_CMD="$SUPPORT_DIR/venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
fi

if [ -z "$PY_CMD" ]; then
    echo "[ERROR] Python not found. Run start/linux_mac/setup.sh first."
    exit 1
fi

exec "$PY_CMD" "$PROGRAM_DIR/main.py" "$@"
