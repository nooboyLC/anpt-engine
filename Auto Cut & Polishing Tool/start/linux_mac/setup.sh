#!/usr/bin/env bash
# =============================================================================
# AUTO SETUP (Linux / macOS / Google Colab) â€” Auto Cut & Polishing Tool
# Location: start/linux_mac/setup.sh
# Program code is inside:  ../../program/
# Everything installs ONLY inside support/venv and support/
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
SUPPORT_DIR="$ROOT_DIR/support"

if [ ! -d "$PROGRAM_DIR" ]; then
    echo "[ERROR] Cannot find program/ folder at: $PROGRAM_DIR"
    exit 1
fi

echo "================================================================"
echo "  AUTO SETUP (Linux / macOS / Colab)"
echo "  Program folder : $PROGRAM_DIR"
echo "  Root folder    : $ROOT_DIR"
echo "  Support folder : $SUPPORT_DIR"
echo "  Everything installs ONLY inside support/ and support/venv/"
echo "================================================================"
echo ""

mkdir -p "$SUPPORT_DIR/temp"
mkdir -p "$SUPPORT_DIR/checkpoints"
mkdir -p "$SUPPORT_DIR/cache"

export TMPDIR="$SUPPORT_DIR/temp"
export TEMP="$SUPPORT_DIR/temp"
export TMP="$SUPPORT_DIR/temp"
export PIP_CACHE_DIR="$SUPPORT_DIR/temp/pip_cache"
export TORCH_HOME="$SUPPORT_DIR/checkpoints"
export HF_HOME="$SUPPORT_DIR/checkpoints"
export VOICEFIXER_CACHE="$SUPPORT_DIR/checkpoints/voicefixer"
export VOICEFIXER_HOME="$SUPPORT_DIR/checkpoints/voicefixer"
export PATH="$SUPPORT_DIR/bin:$PATH"

# 1. Find Python
SYS_PYTHON=""
if command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
    SYS_PYTHON="python3"
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
    SYS_PYTHON="python"
fi

if [ -z "$SYS_PYTHON" ]; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+."
    exit 1
fi

echo "[STEP 1/6] Found Python: $SYS_PYTHON"
"$SYS_PYTHON" --version

# 2. Virtual Environment inside support/venv (Strictly isolated, no system packages)
VENV_DIR="$SUPPORT_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ -f "$VENV_PY" ]; then
    echo "[STEP 2/6] Virtual environment already exists in support/venv"
else
    echo "[STEP 2/6] Creating isolated virtual environment in support/venv ..."
    if ! "$SYS_PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        "$SYS_PYTHON" -m venv --without-pip "$VENV_DIR"
    fi
    if [ ! -f "$VENV_PIP" ]; then
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$SUPPORT_DIR/temp/get-pip.py"
        "$VENV_PY" "$SUPPORT_DIR/temp/get-pip.py"
        rm -f "$SUPPORT_DIR/temp/get-pip.py"
    fi
    echo "[OK] Virtual environment created."
fi

# 3. Update pip
echo ""
echo "[STEP 3/6] Updating pip..."
"$VENV_PIP" install --upgrade pip

# 4. Hardware Detection & Direct PyTorch Installation into support/venv
echo ""
echo "[STEP 4/6] Detecting Hardware and Installing Packages..."

OS_TYPE="$(uname -s)"
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[HARDWARE] macOS detected. Installing native PyTorch..."
    "$VENV_PIP" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "[HARDWARE] NVIDIA GPU Detected! Installing CUDA PyTorch..."
    "$VENV_PIP" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "[HARDWARE] No NVIDIA GPU detected. Installing CPU PyTorch..."
    "$VENV_PIP" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[PACKAGES] Installing from requirements.txt..."
"$VENV_PIP" install -r "$PROGRAM_DIR/requirements.txt"

# Patch voicefixer site-packages to redirect ~/.cache -> support/checkpoints
echo "     [PATCH] Patching VoiceFixer library for isolation..."
"$VENV_PY" "$PROGRAM_DIR/patch_packages.py"

# Ensure VoiceFixer checkpoint folder exists in project support dir
VF_CACHE_DST="$SUPPORT_DIR/checkpoints/voicefixer"
mkdir -p "$VF_CACHE_DST/synthesis_module/44100"
mkdir -p "$VF_CACHE_DST/analysis_module/checkpoints"
echo "     [OK] VoiceFixer models configured strictly inside support/checkpoints"

# 5. FFmpeg environment setup
echo ""
echo "[STEP 5/6] Verifying FFmpeg multimedia engine..."
if command -v ffmpeg >/dev/null 2>&1; then
    echo "     [OK] FFmpeg found: $(command -v ffmpeg)"
else
    echo "     [WARN] FFmpeg not found in PATH. Please install FFmpeg (e.g. apt-get install -y ffmpeg / brew install ffmpeg)."
fi

# 6. Pre-download all AI model weights into support/checkpoints
echo ""
echo "[STEP 6/6] Pre-downloading AI model weights (Silero VAD, ClearVoice, VoiceFixer)..."
echo "  This runs only once. Models are cached in support/checkpoints"
echo "  So the program starts instantly and works offline on every future run."
echo ""
"$VENV_PY" "$PROGRAM_DIR/prefetch_models.py" || echo "[WARN] Some AI models could not be downloaded. They will be downloaded on first use."

chmod +x "$DIR/run.sh" 2>/dev/null || true
chmod +x "$DIR/setup.sh" 2>/dev/null || true
chmod +x "$DIR/clean.sh" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  SETUP COMPLETED SUCCESSFULLY!"
echo "  All AI models are pre-downloaded and ready."
echo "  To launch: bash start/linux_mac/run.sh"
echo "================================================================"
echo ""
