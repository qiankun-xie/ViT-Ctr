@echo off
REM ===============================================================
REM  autodl-status.bat -- Check bootstrap progress on AutoDL
REM
REM  Usage: scripts\autodl-status.bat
REM ===============================================================
setlocal enabledelayedexpansion

REM Load config
call "%~dp0autodl-config.bat"

echo.
echo ============================================================
echo   AutoDL Bootstrap Progress
echo ============================================================
echo.

REM Progress file
echo [Progress]
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "cat %AUTODL_CKPT_DIR%/bootstrap_progress.json 2>/dev/null || echo '  Not started or file not found'"
echo.

REM Last 10 lines of log
echo [Recent Log]
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "tail -10 %AUTODL_CKPT_DIR%/bootstrap.log 2>/dev/null || echo '  Log file not found'"
echo.

REM Check tmux session
echo [tmux Status]
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "tmux has-session -t bootstrap 2>/dev/null && echo '  bootstrap session running' || echo '  bootstrap session not found (may have finished)'"
echo.

REM Check output files
echo [Output Files]
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "ls -lh %AUTODL_CKPT_DIR%/bootstrap_heads.pth %AUTODL_CKPT_DIR%/calibration.json %AUTODL_CKPT_DIR%/bootstrap_results.tar.gz 2>/dev/null || echo '  Output files not yet generated'"
echo.

endlocal
