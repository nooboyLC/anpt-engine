@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Auto Cut & Polishing Tool — clean.bat (Windows)
REM Location: start\windows\clean.bat
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

set "SUPPORT_DIR=%ROOT_DIR%\support"
set "VENV_DIR=%SUPPORT_DIR%\venv"

echo ================================================================
echo   CLEAN (Windows) — Auto Cut ^& Polishing Tool
echo ================================================================
echo.
echo This will clean:
echo   - Support folder (cache, temp, AI binaries)  [support\]
echo   - Python bytecode                            [__pycache__, *.pyc]
echo.
echo Note: Output folder [output\] is ALWAYS kept safe and never touched.
echo.

set "DEL_VENV="
set "CONFIRM="
set "NON_INTERACTIVE="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--venv" (
    set "DEL_VENV=Y"
    set "CONFIRM=Y"
    set "NON_INTERACTIVE=1"
)
if /i "%~1"=="-v" (
    set "DEL_VENV=Y"
    set "CONFIRM=Y"
    set "NON_INTERACTIVE=1"
)
if /i "%~1"=="-y" (
    set "CONFIRM=Y"
    set "NON_INTERACTIVE=1"
)
if /i "%~1"=="--yes" (
    set "CONFIRM=Y"
    set "NON_INTERACTIVE=1"
)
shift
goto parse_args
:done_args

if "!CONFIRM!"=="Y" goto skip_confirm_prompt
set /p CONFIRM="Proceed with cleaning support folder and temporary files? (Y/N): "
if /i not "!CONFIRM:~0,1!"=="Y" (
    echo.
    echo Cleanup cancelled.
    pause
    exit /b 0
)
:skip_confirm_prompt

if "!DEL_VENV!"=="Y" goto skip_del_venv_prompt
echo.
set /p DEL_VENV="Also delete Python virtual environment (support\venv)? (y/N): "
:skip_del_venv_prompt

echo.
echo ================================================================
echo   CLEANING IN PROGRESS...
echo ================================================================

REM Force kill Python processes locking files
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1

REM 1. Delete the entire support\ directory completely
if exist "%SUPPORT_DIR%" (
    rmdir /s /q "%SUPPORT_DIR%" >nul 2>&1
    echo [OK] Removed support\ folder completely.
)

REM Remove VoiceFixer cache junction from user profile if it exists
if exist "%USERPROFILE%\.cache\voicefixer" (
    rmdir "%USERPROFILE%\.cache\voicefixer" >nul 2>&1
    echo [OK] Removed VoiceFixer cache junction.
)

REM 2. Clear Python bytecode cache
for /d /r "%PROGRAM_DIR%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" >nul 2>&1
)
del /s /q "%PROGRAM_DIR%\*.pyc" >nul 2>&1
echo [OK] Bytecode cache cleared.

REM 3. Optional: Delete venv if requested
if /i "%DEL_VENV:~0,1%"=="Y" (
    if exist "%VENV_DIR%" (
        rmdir /s /q "%VENV_DIR%" >nul 2>&1
        echo [OK] Deleted support\venv
    )
)

echo [OK] Output folder at "%ROOT_DIR%\output" kept safe.
echo.
echo ================================================================
echo   CLEANUP COMPLETE!
echo ================================================================
echo.
pause
