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

if [ ! -d "$PROGRAM_DIR" ]; then
    echo "[ERROR] Cannot find program/ folder. Expected at: $PROGRAM_DIR"
    exit 1
fi

cd "$PROGRAM_DIR"

export TMPDIR="$PROGRAM_DIR/support/temp"
export TEMP="$PROGRAM_DIR/support/temp"
export TMP="$PROGRAM_DIR/support/temp"
export PIP_CACHE_DIR="$PROGRAM_DIR/support/temp/pip_cache"
export TORCH_HOME="$PROGRAM_DIR/support/checkpoints"
export TORCH_EXTENSIONS_DIR="$PROGRAM_DIR/support/cache/torch_extensions"
export HF_HOME="$PROGRAM_DIR/support/checkpoints"
export TRANSFORMERS_CACHE="$PROGRAM_DIR/support/checkpoints/transformers"
export HF_DATASETS_CACHE="$PROGRAM_DIR/support/checkpoints/datasets"
export HUGGINGFACE_HUB_CACHE="$PROGRAM_DIR/support/checkpoints/hub"
export ASTEROID_CACHE="$PROGRAM_DIR/support/checkpoints/asteroid"
export SPEECHBRAIN_CACHE="$PROGRAM_DIR/support/checkpoints/speechbrain"
export SENTENCE_TRANSFORMERS_HOME="$PROGRAM_DIR/support/checkpoints/sentence_transformers"
export PYTHONUSERBASE="$PROGRAM_DIR/support/py_userbase"
export XDG_CACHE_HOME="$PROGRAM_DIR/support/cache"
export MPLCONFIGDIR="$PROGRAM_DIR/support/cache/matplotlib"
export VOICEFIXER_CACHE="$PROGRAM_DIR/support/checkpoints/voicefixer"
export VOICEFIXER_HOME="$PROGRAM_DIR/support/checkpoints/voicefixer"

# Find Python — prefer local venv
PY_CMD=""
if [ -f "$PROGRAM_DIR/venv/bin/python" ]; then
    PY_CMD="$PROGRAM_DIR/venv/bin/python"
elif [ -f "$PROGRAM_DIR/venv/bin/python3" ]; then
    PY_CMD="$PROGRAM_DIR/venv/bin/python3"
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
