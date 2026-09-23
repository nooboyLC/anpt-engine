@echo off
setlocal EnableExtensions

REM ============================================================
REM Auto Cut & Polishing Tool — setup.bat (Windows)
REM Location: start\windows\setup.bat
REM Program code is inside  ..\..\program\
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
set "PROGRAM_DIR=%ROOT_DIR%\program"

pushd "%PROGRAM_DIR%" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find program\ folder. Expected at: %PROGRAM_DIR%
    pause
    exit /b 1
)
set "PROGRAM_DIR=%CD%"
popd

pushd "%ROOT_DIR%"
set "ROOT_DIR=%CD%"
popd

echo ================================================================
echo   AUTO SETUP (Windows) — Auto Cut ^& Polishing Tool
echo   Program folder : "%PROGRAM_DIR%"
echo   Root folder    : "%ROOT_DIR%"
echo   Everything installs ONLY inside program\support\
echo ================================================================
echo.

set "NON_INTERACTIVE="
if /i "%~1"=="-y" set "NON_INTERACTIVE=1"
if /i "%~1"=="--yes" set "NON_INTERACTIVE=1"

if not exist "%PROGRAM_DIR%\support"            mkdir "%PROGRAM_DIR%\support"
if not exist "%PROGRAM_DIR%\support\temp"       mkdir "%PROGRAM_DIR%\support\temp"


set "TMPDIR=%PROGRAM_DIR%\support\temp"
set "TEMP=%PROGRAM_DIR%\support\temp"
set "TMP=%PROGRAM_DIR%\support\temp"
set "PIP_CACHE_DIR=%PROGRAM_DIR%\support\temp\pip_cache"
set "TORCH_HOME=%PROGRAM_DIR%\support\checkpoints"
set "HF_HOME=%PROGRAM_DIR%\support\checkpoints"
set "VOICEFIXER_CACHE=%PROGRAM_DIR%\support\checkpoints\voicefixer"
set "VOICEFIXER_HOME=%PROGRAM_DIR%\support\checkpoints\voicefixer"

REM 1. Find Python
set "SYS_PYTHON="
python --version >nul 2>&1
if not errorlevel 1 set "SYS_PYTHON=python"

if not defined SYS_PYTHON (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "SYS_PYTHON=py -3"
)

if not defined SYS_PYTHON (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" set "SYS_PYTHON=%%D\python.exe"
    )
)

if not defined SYS_PYTHON (
    echo [ERROR] Python 3 not found. Please install Python 3.10+ from python.org.
    pause
    exit /b 1
)

echo [STEP 1/6] Found Python: %SYS_PYTHON%
%SYS_PYTHON% --version

REM 2. Create venv inside program\
set "VENV_DIR=%PROGRAM_DIR%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

if exist "%VENV_PY%" (
    echo [STEP 2/6] Virtual environment already exists in program\venv
) else (
    echo [STEP 2/6] Creating virtual environment in program\venv ...
    %SYS_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM 3. Upgrade pip
echo.
echo [STEP 3/6] Updating pip...
"%VENV_PY%" -m pip install --upgrade pip

REM 4. Detect GPU and install PyTorch + requirements
echo.
echo [STEP 4/6] Detecting Hardware and Installing Packages...
taskkill /f /im python.exe >nul 2>&1
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [HARDWARE] No NVIDIA GPU detected. Installing CPU PyTorch...
    "%VENV_PIP%" install --no-cache-dir "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [HARDWARE] NVIDIA GPU Detected! Installing CUDA PyTorch...
    "%VENV_PIP%" install --no-cache-dir "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo [PACKAGES] Installing from requirements.txt...
taskkill /f /im python.exe >nul 2>&1
"%VENV_PIP%" install --no-cache-dir -r "%PROGRAM_DIR%\requirements.txt"
if errorlevel 1 (
    echo [RETRY] Resolving lock and retrying package installation...
    taskkill /f /im python.exe >nul 2>&1
    "%VENV_PIP%" install --no-cache-dir -r "%PROGRAM_DIR%\requirements.txt"
)

REM Install VoiceFixer core strictly without unnecessary web packages (streamlit/pandas/pyarrow)
echo.
echo [PACKAGES] Installing VoiceFixer (optimized audio core without streamlit)...
"%VENV_PIP%" install --no-cache-dir --no-deps "voicefixer>=0.1.3"

if exist "%PROGRAM_DIR%\src\core\compat_dlls\*.dll" (
    if exist "%VENV_DIR%\Lib\site-packages\cv2" (
        copy /y "%PROGRAM_DIR%\src\core\compat_dlls\*.dll" "%VENV_DIR%\Lib\site-packages\cv2\" >nul 2>&1
    )
)

REM Clean up residual pip cache to save 2.7GB disk space
if exist "%PROGRAM_DIR%\support\temp\pip_cache" (
    rd /s /q "%PROGRAM_DIR%\support\temp\pip_cache" >nul 2>&1
)

REM Ensure VoiceFixer checkpoint folder exists in program\support\checkpoints
set "VF_CACHE_DST=%PROGRAM_DIR%\support\checkpoints\voicefixer"
if not exist "%VF_CACHE_DST%" mkdir "%VF_CACHE_DST%"
echo      [OK] VoiceFixer models configured strictly inside program\support\checkpoints

REM Patch voicefixer site-packages to redirect ~/.cache -> program/support/checkpoints
echo      [PATCH] Patching VoiceFixer library for C: drive isolation...
"%VENV_PY%" "%PROGRAM_DIR%\patch_packages.py"

REM 5. Setup Standalone FFmpeg in program\support\bin
echo.
echo [STEP 5/6] Setting up Standalone FFmpeg in program\support\bin ...
if not exist "%PROGRAM_DIR%\support\bin" mkdir "%PROGRAM_DIR%\support\bin"
if not exist "%PROGRAM_DIR%\support\temp" mkdir "%PROGRAM_DIR%\support\temp"

if not exist "%PROGRAM_DIR%\support\bin\ffmpeg.exe" (
    echo   -- Downloading standalone compatible FFmpeg [NVENC GPU accelerated]...
    curl -sSL "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip" -o "%PROGRAM_DIR%\support\temp\ffmpeg.zip"
    if exist "%PROGRAM_DIR%\support\temp\ffmpeg.zip" (
        tar -xf "%PROGRAM_DIR%\support\temp\ffmpeg.zip" -C "%PROGRAM_DIR%\support\temp"
        for /d %%D in ("%PROGRAM_DIR%\support\temp\ffmpeg-7.1*") do (
            copy /y "%%D\bin\*.exe" "%PROGRAM_DIR%\support\bin\" >nul 2>&1
            rd /s /q "%%D" >nul 2>&1
        )
        del /f /q "%PROGRAM_DIR%\support\temp\ffmpeg.zip" >nul 2>&1
        echo      [OK] Standalone FFmpeg engine installed in program\support\bin
    ) else (
        echo      [INFO] Using system FFmpeg.
    )
) else (
    echo      [OK] Standalone FFmpeg engine already present in program\support\bin
)

REM 6. Pre-download all AI model weights into program\support\checkpoints
echo.
echo [STEP 6/6] Pre-downloading AI model weights (Silero VAD, ClearVoice, VoiceFixer, Real-BasicVSR)...
"%VENV_PY%" "%PROGRAM_DIR%\prefetch_models.py"
if errorlevel 1 (
    echo [WARN] Some AI models could not be downloaded. They will be downloaded on first use.
)

echo.
echo ================================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo   All AI models are pre-downloaded and ready.
echo   To launch the tool: double-click start\windows\run.bat
echo ================================================================
echo.
if not "%NON_INTERACTIVE%"=="1" pause
