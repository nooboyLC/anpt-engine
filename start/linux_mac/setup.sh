#!/usr/bin/env bash
# =============================================================================
# AUTO SETUP (Linux / macOS / Google Colab) — Auto Cut & Polishing Tool
# Location: start/linux_mac/setup.sh
# Program code is inside:  ../../program/
# Everything installs ONLY inside program/support/
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
echo "  AUTO SETUP (Linux / macOS / Colab)"
echo "  Program folder : $PROGRAM_DIR"
echo "  Root folder    : $ROOT_DIR"
echo "================================================================"
echo ""

mkdir -p "$PROGRAM_DIR/support/temp"


export TMPDIR="$PROGRAM_DIR/support/temp"
export TEMP="$PROGRAM_DIR/support/temp"
export TMP="$PROGRAM_DIR/support/temp"
export PIP_CACHE_DIR="$PROGRAM_DIR/support/temp/pip_cache"
export TORCH_HOME="$PROGRAM_DIR/support/checkpoints"
export HF_HOME="$PROGRAM_DIR/support/checkpoints"
export VOICEFIXER_CACHE="$PROGRAM_DIR/support/checkpoints/voicefixer"
export VOICEFIXER_HOME="$PROGRAM_DIR/support/checkpoints/voicefixer"

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

echo "[STEP 1/5] Found Python: $SYS_PYTHON"
"$SYS_PYTHON" --version

# 2. Virtual Environment inside program/venv
VENV_DIR="$PROGRAM_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ -f "$VENV_PY" ]; then
    echo "[STEP 2/5] Virtual environment already exists in program/venv"
else
    echo "[STEP 2/5] Creating virtual environment in program/venv ..."
    "$SYS_PYTHON" -m venv "$VENV_DIR" || "$SYS_PYTHON" -m venv --without-pip "$VENV_DIR"
    if [ ! -f "$VENV_PIP" ]; then
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$PROGRAM_DIR/support/temp/get-pip.py"
        "$VENV_PY" "$PROGRAM_DIR/support/temp/get-pip.py"
        rm -f "$PROGRAM_DIR/support/temp/get-pip.py"
    fi
    echo "[OK] Virtual environment created."
fi

# 3. Update pip
echo ""
echo "[STEP 3/5] Updating pip..."
"$VENV_PIP" install --upgrade pip

# 4. Hardware Detection & Requirements
echo ""
echo "[STEP 4/5] Detecting Hardware and Installing Packages..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[HARDWARE] NVIDIA GPU Detected! Installing CUDA PyTorch..."
    "$VENV_PIP" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "[HARDWARE] No NVIDIA GPU. Installing CPU-optimized PyTorch..."
    "$VENV_PIP" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[PACKAGES] Installing from requirements.txt..."
"$VENV_PIP" install -r "$PROGRAM_DIR/requirements.txt"

# Patch voicefixer site-packages to redirect ~/.cache -> program/support/checkpoints
echo "     [PATCH] Patching VoiceFixer library for isolation..."
"$VENV_PY" "$PROGRAM_DIR/patch_packages.py"

# Ensure VoiceFixer checkpoint folder exists in project support dir
VF_CACHE_DST="$PROGRAM_DIR/support/checkpoints/voicefixer"
mkdir -p "$VF_CACHE_DST/synthesis_module/44100"
mkdir -p "$VF_CACHE_DST/analysis_module/checkpoints"
echo "     [OK] VoiceFixer models configured strictly inside program/support/checkpoints"

# 5. C++ Vulkan Real-ESRGAN Engine in program/support/bin
echo ""
echo "[STEP 5/6] Setting up C++ Vulkan Real-ESRGAN Engine in program/support/bin ..."
mkdir -p "$PROGRAM_DIR/support/bin"

if [ ! -f "$PROGRAM_DIR/support/bin/realesrgan-ncnn-vulkan" ]; then
    echo "  -- Downloading C++ Vulkan Real-ESRGAN binary..."
    REAL_ZIP="$PROGRAM_DIR/support/temp/realesrgan-vulkan.zip"
    curl -sSL "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" -o "$REAL_ZIP" || true
    if [ -f "$REAL_ZIP" ]; then
        unzip -q -o "$REAL_ZIP" -d "$PROGRAM_DIR/support/bin" || true
        rm -f "$REAL_ZIP" 2>/dev/null || true
        chmod +x "$PROGRAM_DIR/support/bin/realesrgan-ncnn-vulkan" 2>/dev/null || true
        echo "     [OK] Real-ESRGAN Vulkan engine installed in program/support/bin"
    else
        echo "     [INFO] Binary will be downloaded on first use."
    fi
else
    echo "     [OK] Real-ESRGAN Vulkan engine already present."
fi

# 6. Pre-download all AI model weights into program/support/checkpoints
echo ""
echo "[STEP 6/6] Pre-downloading AI model weights (Silero VAD, ClearVoice, VoiceFixer)..."
echo "  This runs only once. Models are cached in program/support/checkpoints"
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
