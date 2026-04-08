@echo off
REM ===============================================================
REM  autodl-sync.bat -- Upload self-contained script to AutoDL
REM
REM  Usage: scripts\autodl-sync.bat
REM
REM  Prereq: Edit scripts\autodl-config.bat with your SSH info
REM  Requires: scp (Windows 10+ has OpenSSH built-in)
REM ===============================================================
setlocal enabledelayedexpansion

REM Load config
call "%~dp0autodl-config.bat"

echo.
echo ============================================================
echo   AutoDL Code Sync
echo ============================================================
echo   Local:  %LOCAL_PROJECT_DIR%
echo   Remote: %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_PROJECT_DIR%
echo   Port:   %AUTODL_PORT%
echo ============================================================
echo.

REM Check scp available
where scp >nul 2>&1
if errorlevel 1 (
    echo [ERROR] scp not found!
    echo.
    echo Windows 10+ has OpenSSH built-in. Or install via:
    echo   Settings ^> Apps ^> Optional Features ^> OpenSSH Client
    exit /b 1
)

REM Create remote directory
echo [1/2] Creating remote directory...
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "mkdir -p %AUTODL_PROJECT_DIR%/colab"
if errorlevel 1 (
    echo [ERROR] SSH connection failed! Check:
    echo   1. Is your AutoDL instance running?
    echo   2. Is HOST/PORT correct in autodl-config.bat?
    exit /b 1
)

REM Upload self-contained script (no src/ needed)
echo [2/2] Uploading autodl_bootstrap.py (self-contained, no other files needed)...
scp -P %AUTODL_PORT% "%LOCAL_PROJECT_DIR%\colab\autodl_bootstrap.py" %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_PROJECT_DIR%/colab/

REM Optional: upload best_model.pth if it exists locally
if exist "%LOCAL_CKPT_DIR%\best_model.pth" (
    echo [Optional] Found local best_model.pth, uploading...
    ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "mkdir -p %AUTODL_CKPT_DIR%"
    scp -P %AUTODL_PORT% "%LOCAL_CKPT_DIR%\best_model.pth" %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%/
)

echo.
echo ============================================================
echo   Sync complete!
echo ============================================================
echo.
echo Next step: Start bootstrap:
echo   scripts\autodl-run.bat
echo.

endlocal
