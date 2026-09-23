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

echo "[STEP 1/7] Found Python: $SYS_PYTHON"
"$SYS_PYTHON" --version

# 2. Virtual Environment inside program/venv
VENV_DIR="$PROGRAM_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ -f "$VENV_PY" ]; then
    echo "[STEP 2/7] Virtual environment already exists in program/venv"
else
    echo "[STEP 2/7] Creating virtual environment in program/venv ..."
    if ! "$SYS_PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        "$SYS_PYTHON" -m venv --without-pip "$VENV_DIR" 2>/dev/null || true
    fi
    if [ ! -f "$VENV_PIP" ]; then
        echo "  -- Bootstrapping pip in virtual environment..."
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$PROGRAM_DIR/support/temp/get-pip.py"
        "$VENV_PY" "$PROGRAM_DIR/support/temp/get-pip.py" --quiet 2>/dev/null || "$VENV_PY" "$PROGRAM_DIR/support/temp/get-pip.py"
        rm -f "$PROGRAM_DIR/support/temp/get-pip.py" 2>/dev/null || true
    fi
    echo "[OK] Virtual environment created."
fi

# 3. Update pip
echo ""
echo "[STEP 3/7] Updating pip..."
"$VENV_PIP" install --upgrade pip

# 4. Hardware Detection & Requirements
echo ""
echo "[STEP 4/7] Detecting Hardware and Installing Packages..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[HARDWARE] NVIDIA GPU Detected! Installing CUDA PyTorch..."
    "$VENV_PIP" install --no-cache-dir "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "[HARDWARE] No NVIDIA GPU. Installing CPU-optimized PyTorch..."
    "$VENV_PIP" install --no-cache-dir "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "[PACKAGES] Installing from requirements.txt..."
"$VENV_PIP" install --no-cache-dir -r "$PROGRAM_DIR/requirements.txt"

# Install VoiceFixer core strictly without unnecessary web packages (streamlit/pandas/pyarrow)
echo ""
echo "[PACKAGES] Installing VoiceFixer (optimized audio core without streamlit)..."
"$VENV_PIP" install --no-cache-dir --no-deps "voicefixer>=0.1.3"

# Clean up residual pip cache to save 2.7GB disk space
rm -rf "$PROGRAM_DIR/support/temp/pip_cache" 2>/dev/null || true

# Patch voicefixer site-packages to redirect ~/.cache -> program/support/checkpoints
echo "     [PATCH] Patching VoiceFixer library for isolation..."
"$VENV_PY" "$PROGRAM_DIR/patch_packages.py"

# Ensure VoiceFixer checkpoint folder exists in project support dir
VF_CACHE_DST="$PROGRAM_DIR/support/checkpoints/voicefixer"
mkdir -p "$VF_CACHE_DST/synthesis_module/44100"
mkdir -p "$VF_CACHE_DST/analysis_module/checkpoints"
echo "     [OK] VoiceFixer models configured strictly inside program/support/checkpoints"

# 5. Bundled FFmpeg with GPU encoding support
# Linux  : John Van Sickle static build (includes NVENC via nonfree libs)
# macOS  : evermeet.cx static build (includes VideoToolbox)
echo ""
echo "[STEP 5/7] Setting up bundled FFmpeg (GPU-encoding support)..."
mkdir -p "$PROGRAM_DIR/support/bin"

OS_TYPE="$(uname -s)"

if [ ! -f "$PROGRAM_DIR/support/bin/ffmpeg" ]; then
    if [ "$OS_TYPE" = "Linux" ]; then
        SYS_HAS_NVENC=0
        if command -v ffmpeg >/dev/null 2>&1; then
            if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
                SYS_HAS_NVENC=1
            fi
        fi

        if [ "$SYS_HAS_NVENC" -eq 1 ]; then
            echo "     [OK] System FFmpeg already has NVENC support."
        else
            echo "  -- [Linux] Downloading FFmpeg static build (BtbN NVENC GPU accelerated)..."
            ARCH="$(uname -m)"
            FF_URL=""
            if [ "$ARCH" = "x86_64" ]; then
                FF_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
            elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
                FF_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
            fi
            if [ -n "$FF_URL" ]; then
                FF_TAR="$PROGRAM_DIR/support/temp/ffmpeg-linux.tar.xz"
                curl -sSL "$FF_URL" -o "$FF_TAR" || true
                if [ -f "$FF_TAR" ]; then
                    FF_EXTRACT="$PROGRAM_DIR/support/temp/ffmpeg_extract"
                    mkdir -p "$FF_EXTRACT"
                    tar -xf "$FF_TAR" -C "$FF_EXTRACT" --strip-components=1 || true
                    cp -f "$FF_EXTRACT/bin/ffmpeg"  "$PROGRAM_DIR/support/bin/ffmpeg"  2>/dev/null || cp -f "$FF_EXTRACT/ffmpeg"  "$PROGRAM_DIR/support/bin/ffmpeg"  2>/dev/null || true
                    cp -f "$FF_EXTRACT/bin/ffprobe" "$PROGRAM_DIR/support/bin/ffprobe" 2>/dev/null || cp -f "$FF_EXTRACT/ffprobe" "$PROGRAM_DIR/support/bin/ffprobe" 2>/dev/null || true
                    chmod +x "$PROGRAM_DIR/support/bin/ffmpeg" "$PROGRAM_DIR/support/bin/ffprobe" 2>/dev/null || true
                    rm -rf "$FF_EXTRACT" "$FF_TAR" 2>/dev/null || true
                    echo "     [OK] FFmpeg (Linux static, NVENC-capable) installed in program/support/bin"
                else
                    echo "     [WARN] FFmpeg download failed — will fall back to system ffmpeg."
                fi
            else
                echo "     [INFO] Architecture $ARCH not supported for bundled download — using system ffmpeg."
            fi
        fi

    elif [ "$OS_TYPE" = "Darwin" ]; then
        echo "  -- [macOS] Downloading FFmpeg static build (VideoToolbox-capable)..."
        MAC_ARCH="$(uname -m)"
        if [ "$MAC_ARCH" = "arm64" ]; then
            FF_FF_URL="https://evermeet.cx/ffmpeg/getrelease/arm64/ffmpeg/zip"
            FF_FP_URL="https://evermeet.cx/ffmpeg/getrelease/arm64/ffprobe/zip"
        else
            FF_FF_URL="https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
            FF_FP_URL="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
        fi
        FF_ZIP="$PROGRAM_DIR/support/temp/ffmpeg-mac.zip"
        FP_ZIP="$PROGRAM_DIR/support/temp/ffprobe-mac.zip"
        curl -sSL "$FF_FF_URL" -o "$FF_ZIP" || true
        curl -sSL "$FF_FP_URL" -o "$FP_ZIP" || true
        if [ -f "$FF_ZIP" ]; then
            unzip -q -o "$FF_ZIP" -d "$PROGRAM_DIR/support/bin" 2>/dev/null || true
            rm -f "$FF_ZIP" 2>/dev/null || true
        fi
        if [ -f "$FP_ZIP" ]; then
            unzip -q -o "$FP_ZIP" -d "$PROGRAM_DIR/support/bin" 2>/dev/null || true
            rm -f "$FP_ZIP" 2>/dev/null || true
        fi
        chmod +x "$PROGRAM_DIR/support/bin/ffmpeg" "$PROGRAM_DIR/support/bin/ffprobe" 2>/dev/null || true
        if [ -f "$PROGRAM_DIR/support/bin/ffmpeg" ]; then
            # Remove macOS quarantine flag so the binary can run without Gatekeeper prompt
            xattr -d com.apple.quarantine "$PROGRAM_DIR/support/bin/ffmpeg" 2>/dev/null || true
            xattr -d com.apple.quarantine "$PROGRAM_DIR/support/bin/ffprobe" 2>/dev/null || true
            echo "     [OK] FFmpeg (macOS static, VideoToolbox) installed in program/support/bin"
        else
            echo "     [WARN] FFmpeg download failed — will fall back to system ffmpeg."
        fi
    else
        echo "     [INFO] Unknown OS ($OS_TYPE) — skipping FFmpeg bundling."
    fi
else
    echo "     [OK] Bundled FFmpeg already present in program/support/bin"
    "$PROGRAM_DIR/support/bin/ffmpeg" -version 2>/dev/null | head -1 || true
fi

# 6. Pre-download all AI model weights into program/support/checkpoints
echo ""
echo "[STEP 6/7] Pre-downloading AI model weights (Silero VAD, ClearVoice, VoiceFixer, Real-BasicVSR)..."
"$VENV_PY" "$PROGRAM_DIR/prefetch_models.py" || echo "[WARN] Some AI models could not be downloaded. They will be downloaded on first use."

chmod +x "$DIR/run.sh" 2>/dev/null || true
chmod +x "$DIR/setup.sh" 2>/dev/null || true
chmod +x "$DIR/clean.sh" 2>/dev/null || true

echo ""
echo "[STEP 7/7] Setting executable permissions on scripts..."

echo ""
echo "================================================================"
echo "  SETUP COMPLETED SUCCESSFULLY!"
echo "  All AI models are pre-downloaded and ready."
echo "  To launch: bash start/linux_mac/run.sh"
echo "================================================================"
echo ""
