@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Auto Cut & Polishing Tool — run.bat (Windows Launcher)
REM Location: start\windows\run.bat
REM This script is inside  start\windows\
REM Program code is inside  ..\..\program\
REM ============================================================

REM Resolve the root of the project (two levels up from this script)
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
set "PROGRAM_DIR=%ROOT_DIR%\program"

REM Normalize paths
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

REM ============================================================
REM Runtime isolation — all runtime files go under program\
REM Environment variables
set "TMPDIR=%PROGRAM_DIR%\support\temp"
set "TEMP=%PROGRAM_DIR%\support\temp"
set "TMP=%PROGRAM_DIR%\support\temp"
set "TORCH_HOME=%PROGRAM_DIR%\support\checkpoints"
set "TORCH_EXTENSIONS_DIR=%PROGRAM_DIR%\support\cache\torch_extensions"
set "HF_HOME=%PROGRAM_DIR%\support\checkpoints"
set "TRANSFORMERS_CACHE=%PROGRAM_DIR%\support\checkpoints\transformers"
set "HF_DATASETS_CACHE=%PROGRAM_DIR%\support\checkpoints\datasets"
set "HUGGINGFACE_HUB_CACHE=%PROGRAM_DIR%\support\checkpoints\hub"
set "ASTEROID_CACHE=%PROGRAM_DIR%\support\checkpoints\asteroid"
set "SPEECHBRAIN_CACHE=%PROGRAM_DIR%\support\checkpoints\speechbrain"
set "SENTENCE_TRANSFORMERS_HOME=%PROGRAM_DIR%\support\checkpoints\sentence_transformers"
set "PIP_CACHE_DIR=%PROGRAM_DIR%\support\temp\pip_cache"
set "PYTHONUSERBASE=%PROGRAM_DIR%\support\py_userbase"
set "XDG_CACHE_HOME=%PROGRAM_DIR%\support\cache"
set "MPLCONFIGDIR=%PROGRAM_DIR%\support\cache\matplotlib"
set "VOICEFIXER_CACHE=%PROGRAM_DIR%\support\checkpoints\voicefixer"
set "VOICEFIXER_HOME=%PROGRAM_DIR%\support\checkpoints\voicefixer"

REM ============================================================
REM Find Python — prefer local venv created by setup.bat
REM ============================================================
if exist "%PROGRAM_DIR%\venv\Scripts\python.exe" (
    set "PY_CMD=%PROGRAM_DIR%\venv\Scripts\python.exe"
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
pause
exit /b 1

:run_app
cd /d "%PROGRAM_DIR%"
"%PY_CMD%" "%PROGRAM_DIR%\main.py" %*

if errorlevel 1 (
    echo.
    echo [EXIT] Program exited with an error code.
    pause
)
