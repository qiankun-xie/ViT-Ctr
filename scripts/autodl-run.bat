@echo off
REM ===============================================================
REM  autodl-run.bat -- Start bootstrap training on AutoDL
REM
REM  Usage:
REM    scripts\autodl-run.bat           -- Full run
REM    scripts\autodl-run.bat resume    -- Resume from checkpoint
REM    scripts\autodl-run.bat calibrate -- Calibration only
REM
REM  Runs in remote tmux, survives SSH disconnect.
REM  Use scripts\autodl-status.bat to check progress.
REM ===============================================================
setlocal enabledelayedexpansion

REM Load config
call "%~dp0autodl-config.bat"

REM Parse argument
set MODE=
set EXTRA_ARGS=
if /I "%~1"=="resume" (
    set MODE=resume
    set EXTRA_ARGS=--resume
    echo Mode: Resume from checkpoint
) else if /I "%~1"=="calibrate" (
    set MODE=calibrate
    set EXTRA_ARGS=--calibrate_only
    echo Mode: Calibration only
) else (
    set MODE=full
    echo Mode: Full run
)

echo.
echo ============================================================
echo   AutoDL Bootstrap Remote Launch
echo ============================================================
echo   Remote: %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_PORT%
echo   Mode:   %MODE%
echo ============================================================
echo.

REM Start bootstrap in tmux (kill existing session first)
echo [RUN] Starting bootstrap in remote tmux...
echo.

ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "tmux kill-session -t bootstrap 2>/dev/null; tmux new-session -d -s bootstrap 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && cd %AUTODL_PROJECT_DIR% && python colab/autodl_bootstrap.py %EXTRA_ARGS% 2>&1 | tee %AUTODL_CKPT_DIR%/bootstrap.log; echo DONE >> %AUTODL_CKPT_DIR%/bootstrap.log'"

if errorlevel 1 (
    echo [ERROR] Remote launch failed!
    exit /b 1
)

echo ============================================================
echo   Bootstrap started in remote tmux!
echo ============================================================
echo.
echo Live progress:
echo   ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "tmux attach -t bootstrap"
echo   (Ctrl+B, D to detach)
echo.
echo Progress summary:
echo   scripts\autodl-status.bat
echo.
echo Download results when done:
echo   scripts\autodl-download.bat
echo.

endlocal
