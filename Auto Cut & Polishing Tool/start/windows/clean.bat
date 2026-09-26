@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Auto Cut & Polishing Tool - clean.bat (Windows)
REM Location: start\windows\clean.bat
REM ============================================================

REM --- Path resolution (safe against & in folder names) ---
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
set "PROGRAM_DIR=%ROOT_DIR%\program"

pushd "%ROOT_DIR%" 2>nul
if errorlevel 1 ( echo [ERROR] Root folder not found. & pause & exit /b 1 )
set "ROOT_DIR=%CD%"
popd

pushd "%PROGRAM_DIR%" 2>nul
if errorlevel 1 ( echo [ERROR] program\ folder not found. & pause & exit /b 1 )
set "PROGRAM_DIR=%CD%"
popd

set "SUPPORT_DIR=%ROOT_DIR%\support"

REM --- Use subst to map paths with & to a safe temp drive letter ---
subst X: "%ROOT_DIR%" >nul 2>&1
subst Y: "%PROGRAM_DIR%" >nul 2>&1

echo ================================================================
echo   CLEAN (Windows) - Auto Cut ^& Polishing Tool
echo ================================================================
echo.
echo This will completely wipe:
echo   - Entire Support folder (venv, models, cache, temp, binaries)  [support\]
echo   - Python bytecode (__pycache__, *.pyc) from program\
echo   - Run logs and CSV files
echo.
echo Note: Output folder [output\] is ALWAYS kept safe.
echo.

set "CONFIRM="
set "NON_INTERACTIVE="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="-y"     ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
if /i "%~1"=="--yes"  ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
if /i "%~1"=="--all"  ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
if /i "%~1"=="-a"     ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
if /i "%~1"=="--venv" ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
if /i "%~1"=="-v"     ( set "CONFIRM=Y" & set "NON_INTERACTIVE=1" )
shift
goto parse_args
:done_args

if "%CONFIRM%"=="Y" goto skip_confirm_prompt
set /p USER_ANS="Proceed with completely wiping the support folder and caches? (Y/N): "
if /i not "%USER_ANS:~0,1%"=="Y" (
    echo.
    echo Cleanup cancelled.
    goto :cleanup_subst
)
:skip_confirm_prompt

echo.
echo ================================================================
echo   CLEANING IN PROGRESS...
echo ================================================================

REM 1. Kill lingering Python processes
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1

REM 2. Wipe entire support\ directory
if exist "X:\support" (
    rmdir /s /q "X:\support" >nul 2>&1
    echo [OK] Completely deleted support\ folder.
) else (
    echo [OK] support\ folder was not present.
)

REM 3. Remove VoiceFixer cache from user profile
if exist "%USERPROFILE%\.cache\voicefixer" (
    rmdir /s /q "%USERPROFILE%\.cache\voicefixer" >nul 2>&1
    echo [OK] Removed VoiceFixer cache.
)

REM 4. Clear Python bytecode (__pycache__, *.pyc) using safe Y: drive
for /d /r "Y:\" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" >nul 2>&1
)
del /s /q "Y:\*.pyc" >nul 2>&1
echo [OK] Python bytecode cache cleared.

REM 5. Clean temporary logs
del /q "Y:\*.log" >nul 2>&1
del /q "X:\*.log" >nul 2>&1
del /q "X:\hardware_log_*.csv" >nul 2>&1
echo [OK] Cleaned temporary logs.

echo [OK] Output folder at "%ROOT_DIR%\output" kept safe.
echo.
echo ================================================================
echo   CLEANUP COMPLETE!
echo ================================================================
echo.

:cleanup_subst
subst X: /d >nul 2>&1
subst Y: /d >nul 2>&1
if not "%NON_INTERACTIVE%"=="1" pause
