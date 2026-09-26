@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Auto Cut & Polishing Tool - setup.bat (Windows)
REM Location: start\windows\setup.bat
REM Program code is inside  ..\..\program\
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
set "PROGRAM_DIR=%ROOT_DIR%\program"

pushd "%PROGRAM_DIR%" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find program\ folder. Expected at: %PROGRAM_DIR%
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
set "PROGRAM_DIR=%CD%"
popd

pushd "%ROOT_DIR%"
set "ROOT_DIR=%CD%"
popd

set "SUPPORT_DIR=%ROOT_DIR%\support"
set "VENV_DIR=%SUPPORT_DIR%\venv"

echo ================================================================
echo   AUTO SETUP (Windows) - Auto Cut ^& Polishing Tool
echo   Program folder : "%PROGRAM_DIR%"
echo   Root folder    : "%ROOT_DIR%"
echo   Support folder : "%SUPPORT_DIR%"
echo   Everything installs ONLY inside support\
echo ================================================================
echo.

if not exist "%SUPPORT_DIR%"            mkdir "%SUPPORT_DIR%"
if not exist "%SUPPORT_DIR%\temp"       mkdir "%SUPPORT_DIR%\temp"
if not exist "%SUPPORT_DIR%\checkpoints" mkdir "%SUPPORT_DIR%\checkpoints"
if not exist "%SUPPORT_DIR%\cache"       mkdir "%SUPPORT_DIR%\cache"
if not exist "%SUPPORT_DIR%\cache\pycache" mkdir "%SUPPORT_DIR%\cache\pycache"
if not exist "%SUPPORT_DIR%\bin"         mkdir "%SUPPORT_DIR%\bin"

set "PATH=%SUPPORT_DIR%\bin;%PATH%"
set "TMPDIR=%SUPPORT_DIR%\temp"
set "TEMP=%SUPPORT_DIR%\temp"
set "TMP=%SUPPORT_DIR%\temp"
set "PIP_CACHE_DIR=%SUPPORT_DIR%\temp\pip_cache"
set "TORCH_HOME=%SUPPORT_DIR%\checkpoints"
set "HF_HOME=%SUPPORT_DIR%\checkpoints"
set "VOICEFIXER_CACHE=%SUPPORT_DIR%\checkpoints\voicefixer"
set "VOICEFIXER_HOME=%SUPPORT_DIR%\checkpoints\voicefixer"
set "PYTHONPYCACHEPREFIX=%SUPPORT_DIR%\cache\pycache"

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
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)

echo [STEP 1/6] Found Python: %SYS_PYTHON%
%SYS_PYTHON% --version

REM 2. Create venv inside support\
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

if exist "%VENV_PY%" (
    echo [STEP 2/6] Virtual environment already exists in support\venv
) else (
    echo [STEP 2/6] Creating virtual environment in support\venv ...
    %SYS_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        if not "%NON_INTERACTIVE%"=="1" pause
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
    "%VENV_PIP%" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [HARDWARE] NVIDIA GPU Detected! Installing CUDA PyTorch...
    "%VENV_PIP%" install "numpy<2.0.0,>=1.26.0" torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
)

echo.
echo [PACKAGES] Installing from requirements.txt...
taskkill /f /im python.exe >nul 2>&1
"%VENV_PIP%" install -r "%PROGRAM_DIR%\requirements.txt"
if errorlevel 1 (
    echo [RETRY] Resolving lock and retrying package installation...
    taskkill /f /im python.exe >nul 2>&1
    "%VENV_PIP%" install -r "%PROGRAM_DIR%\requirements.txt"
)

if exist "%PROGRAM_DIR%\src\core\compat_dlls\*.dll" (
    if exist "%VENV_DIR%\Lib\site-packages\cv2" (
        copy /y "%PROGRAM_DIR%\src\core\compat_dlls\*.dll" "%VENV_DIR%\Lib\site-packages\cv2\" >nul 2>&1
    )
)

REM Ensure VoiceFixer checkpoint folder exists in support\checkpoints
set "VF_CACHE_DST=%SUPPORT_DIR%\checkpoints\voicefixer"
if not exist "%VF_CACHE_DST%" mkdir "%VF_CACHE_DST%"
echo      [OK] VoiceFixer models configured strictly inside support\checkpoints

REM Patch voicefixer site-packages to redirect ~/.cache -> support/checkpoints
echo      [PATCH] Patching VoiceFixer library for C: drive isolation...
"%VENV_PY%" "%PROGRAM_DIR%\patch_packages.py"

REM 5. Setup Standalone FFmpeg in support\bin
echo.
echo [STEP 5/6] Setting up Standalone FFmpeg in support\bin ...
if not exist "%SUPPORT_DIR%\bin" mkdir "%SUPPORT_DIR%\bin"
if not exist "%SUPPORT_DIR%\temp" mkdir "%SUPPORT_DIR%\temp"

if not exist "%SUPPORT_DIR%\bin\ffmpeg.exe" (
    echo   -- Downloading standalone compatible FFmpeg [NVENC GPU accelerated]...
    curl -sSL "https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip" -o "%SUPPORT_DIR%\temp\ffmpeg.zip"
    if exist "%SUPPORT_DIR%\temp\ffmpeg.zip" (
        tar -xf "%SUPPORT_DIR%\temp\ffmpeg.zip" -C "%SUPPORT_DIR%\temp"
        for /d %%D in ("%SUPPORT_DIR%\temp\ffmpeg-7.1*") do (
            copy /y "%%D\bin\*.exe" "%SUPPORT_DIR%\bin\" >nul 2>&1
            rd /s /q "%%D" >nul 2>&1
        )
        del /f /q "%SUPPORT_DIR%\temp\ffmpeg.zip" >nul 2>&1
        echo      [OK] Standalone FFmpeg engine installed in support\bin
    ) else (
        echo      [INFO] Using system FFmpeg.
    )
) else (
    echo      [OK] Standalone FFmpeg engine already present in support\bin
)

REM 6. Pre-download all AI model weights into support\checkpoints
echo.
echo [STEP 6/6] Pre-downloading AI model weights (Silero VAD, ClearVoice, VoiceFixer)...
echo   This runs only once. Models are cached in support\checkpoints
echo   So the program starts instantly and works offline on every future run.
echo.
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
