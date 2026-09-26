@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Auto Cut & Polishing Tool - run.bat (Windows Launcher)
REM Location: start\windows\run.bat
REM This script is inside  start\windows\
REM Program code is inside  ..\..\program\
REM ============================================================

REM Resolve the root of the project (two levels up from this script)
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\..\"
set "PROGRAM_DIR=%ROOT_DIR%program\"

REM Normalize paths
pushd "%PROGRAM_DIR%" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find program\ folder. Expected at: %PROGRAM_DIR%
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
set "PROGRAM_DIR=%CD%"
popd

pushd "%ROOT_DIR%" 2>nul
set "ROOT_DIR=%CD%"
popd

set "SUPPORT_DIR=%ROOT_DIR%\support"
set "VENV_DIR=%SUPPORT_DIR%\venv"

REM ============================================================
REM Runtime isolation - all runtime files go under support\
REM Prepend support\bin to PATH so local FFmpeg and tools are always found
REM ============================================================
set "PATH=%SUPPORT_DIR%\bin;%PATH%"

set "TMPDIR=%SUPPORT_DIR%\temp"
set "TEMP=%SUPPORT_DIR%\temp"
set "TMP=%SUPPORT_DIR%\temp"
set "TORCH_HOME=%SUPPORT_DIR%\checkpoints"
set "TORCH_EXTENSIONS_DIR=%SUPPORT_DIR%\cache\torch_extensions"
set "HF_HOME=%SUPPORT_DIR%\checkpoints"
set "TRANSFORMERS_CACHE=%SUPPORT_DIR%\checkpoints\transformers"
set "HF_DATASETS_CACHE=%SUPPORT_DIR%\checkpoints\datasets"
set "HUGGINGFACE_HUB_CACHE=%SUPPORT_DIR%\checkpoints\hub"
set "ASTEROID_CACHE=%SUPPORT_DIR%\checkpoints\asteroid"
set "SPEECHBRAIN_CACHE=%SUPPORT_DIR%\checkpoints\speechbrain"
set "SENTENCE_TRANSFORMERS_HOME=%SUPPORT_DIR%\checkpoints\sentence_transformers"
set "PIP_CACHE_DIR=%SUPPORT_DIR%\temp\pip_cache"
set "PYTHONUSERBASE=%SUPPORT_DIR%\py_userbase"
set "XDG_CACHE_HOME=%SUPPORT_DIR%\cache"
set "MPLCONFIGDIR=%SUPPORT_DIR%\cache\matplotlib"
set "VOICEFIXER_CACHE=%SUPPORT_DIR%\checkpoints\voicefixer"
set "VOICEFIXER_HOME=%SUPPORT_DIR%\checkpoints\voicefixer"

REM Redirect Python bytecode (__pycache__) to support\cache\pycache
REM so program\ source folders are never polluted
set "PYTHONPYCACHEPREFIX=%SUPPORT_DIR%\cache\pycache"

REM ============================================================
REM Find Python - prefer local venv created by setup.bat
REM ============================================================
if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PY_CMD=%VENV_DIR%\Scripts\python.exe"
    goto :run_app
)

python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :run_app
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :run_app
)

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "PY_CMD=%%D\python.exe"
        goto :run_app
    )
)

echo.
echo ================================================================
echo  [ERROR] Python / Virtual Environment not found!
echo  Please run "start\windows\setup.bat" first.
echo ================================================================
echo.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 1

:run_app
cd /d "%PROGRAM_DIR%"
mkdir "%SUPPORT_DIR%\cache\pycache" >nul 2>&1
"%PY_CMD%" "%PROGRAM_DIR%\main.py" %*

if errorlevel 1 (
    echo.
    echo [EXIT] Program exited with an error code.
    if not "%NON_INTERACTIVE%"=="1" pause
)
