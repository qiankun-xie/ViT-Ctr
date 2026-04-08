@echo off
REM ===============================================================
REM  autodl-download.bat -- Download bootstrap results from AutoDL
REM
REM  Usage: scripts\autodl-download.bat
REM
REM  Downloads:
REM    - bootstrap_heads.pth    (needed for deployment)
REM    - calibration.json       (needed for deployment)
REM    - bootstrap_summary.json (run report)
REM    - bootstrap_results.tar.gz (packed version, optional)
REM ===============================================================
setlocal enabledelayedexpansion

REM Load config
call "%~dp0autodl-config.bat"

echo.
echo ============================================================
echo   AutoDL Results Download
echo ============================================================
echo   Remote: %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%
echo   Local:  %LOCAL_CKPT_DIR%
echo ============================================================
echo.

REM Create local directory
if not exist "%LOCAL_CKPT_DIR%" (
    mkdir "%LOCAL_CKPT_DIR%"
    echo [CREATE] Local dir: %LOCAL_CKPT_DIR%
)

REM Check remote files
echo [CHECK] Remote files...
ssh -p %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST% "ls -lh %AUTODL_CKPT_DIR%/bootstrap_heads.pth %AUTODL_CKPT_DIR%/calibration.json 2>&1"
echo.

REM Download core files
echo [1/4] Downloading bootstrap_heads.pth...
scp -P %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%/bootstrap_heads.pth "%LOCAL_CKPT_DIR%\"
if errorlevel 1 (
    echo [WARN] bootstrap_heads.pth download failed -- training may not be done
)

echo [2/4] Downloading calibration.json...
scp -P %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%/calibration.json "%LOCAL_CKPT_DIR%\"
if errorlevel 1 (
    echo [WARN] calibration.json download failed -- calibration may not be done
)

echo [3/4] Downloading bootstrap_summary.json...
scp -P %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%/bootstrap_summary.json "%LOCAL_CKPT_DIR%\" 2>nul
if errorlevel 1 (
    echo [SKIP] bootstrap_summary.json not found
)

echo [4/4] Downloading bootstrap_results.tar.gz (optional)...
scp -P %AUTODL_PORT% %AUTODL_USER%@%AUTODL_HOST%:%AUTODL_CKPT_DIR%/bootstrap_results.tar.gz "%LOCAL_CKPT_DIR%\" 2>nul
if errorlevel 1 (
    echo [SKIP] bootstrap_results.tar.gz not found
)

echo.
echo ============================================================
echo   Download complete!
echo ============================================================
echo.

REM Verify downloaded files
echo [VERIFY] Local files:
if exist "%LOCAL_CKPT_DIR%\bootstrap_heads.pth" (
    echo   [OK] bootstrap_heads.pth
) else (
    echo   [FAIL] bootstrap_heads.pth missing!
)

if exist "%LOCAL_CKPT_DIR%\calibration.json" (
    echo   [OK] calibration.json
    echo.
    echo [Calibration Result]
    type "%LOCAL_CKPT_DIR%\calibration.json"
    echo.
) else (
    echo   [FAIL] calibration.json missing!
)

if exist "%LOCAL_CKPT_DIR%\bootstrap_summary.json" (
    echo   [OK] bootstrap_summary.json
) else (
    echo   [--] bootstrap_summary.json not found
)

echo.
echo Files needed for deployment:
echo   - checkpoints\best_model.pth       (Phase 3 training)
echo   - checkpoints\bootstrap_heads.pth  (Bootstrap heads)
echo   - checkpoints\calibration.json     (Calibration factors)
echo.

endlocal
