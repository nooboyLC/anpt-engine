@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Auto Cut & Polishing Tool - run.bat
REM Double-click OR run from terminal - both work the same.
REM ============================================================

REM %~dp0 = exact folder of THIS .bat file (always correct)
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
set "PROGRAM_DIR=%ROOT_DIR%\program"
set "SUPPORT_DIR=%ROOT_DIR%\support"
set "PY_EXE=%SUPPORT_DIR%\venv\Scripts\python.exe"

REM ---- Isolate all runtime files inside support\ ----
set "PATH=!SUPPORT_DIR!\bin;%PATH%"
set "TEMP=!SUPPORT_DIR!\temp"
set "TMP=!SUPPORT_DIR!\temp"
set "TORCH_HOME=!SUPPORT_DIR!\checkpoints"
set "TORCH_EXTENSIONS_DIR=!SUPPORT_DIR!\cache\torch_extensions"
set "HF_HOME=!SUPPORT_DIR!\checkpoints"
set "TRANSFORMERS_CACHE=!SUPPORT_DIR!\checkpoints\transformers"
set "HF_DATASETS_CACHE=!SUPPORT_DIR!\checkpoints\datasets"
set "HUGGINGFACE_HUB_CACHE=!SUPPORT_DIR!\checkpoints\hub"
set "ASTEROID_CACHE=!SUPPORT_DIR!\checkpoints\asteroid"
set "SPEECHBRAIN_CACHE=!SUPPORT_DIR!\checkpoints\speechbrain"
set "SENTENCE_TRANSFORMERS_HOME=!SUPPORT_DIR!\checkpoints\sentence_transformers"
set "PIP_CACHE_DIR=!SUPPORT_DIR!\temp\pip_cache"
set "PYTHONUSERBASE=!SUPPORT_DIR!\py_userbase"
set "XDG_CACHE_HOME=!SUPPORT_DIR!\cache"
set "MPLCONFIGDIR=!SUPPORT_DIR!\cache\matplotlib"
set "VOICEFIXER_CACHE=!SUPPORT_DIR!\checkpoints\voicefixer"
set "VOICEFIXER_HOME=!SUPPORT_DIR!\checkpoints\voicefixer"
set "PYTHONPYCACHEPREFIX=!SUPPORT_DIR!\cache\pycache"

REM ---- Check program\main.py exists ----
if not exist "!PROGRAM_DIR!\main.py" (
    echo.
    echo [ERROR] program\main.py not found.
    echo run.bat must remain in the start\windows\ folder.
    echo.
    pause
    exit /b 1
)

REM ---- Check setup was done ----
if not exist "!PY_EXE!" (
    echo.
    echo [ERROR] Python environment not found.
    echo Please run setup first: start\windows\setup.bat
    echo.
    pause
    exit /b 1
)

REM ---- Create required dirs ----
mkdir "!SUPPORT_DIR!\cache\pycache" >nul 2>&1
mkdir "!SUPPORT_DIR!\temp" >nul 2>&1

REM ---- Launch ----
cd /d "!PROGRAM_DIR!"
"!PY_EXE!" "!PROGRAM_DIR!\main.py" %*

if errorlevel 1 (
    echo.
    echo [EXIT] Program exited with an error.
    pause
)
