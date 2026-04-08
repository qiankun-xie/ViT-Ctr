@echo off
REM ===============================================================
REM  AutoDL SSH Config -- Edit the 3 lines below
REM  Get these from AutoDL console > "SSH Connection"
REM ===============================================================

REM AutoDL SSH host (e.g. connect.westb.seetacloud.com)
set AUTODL_HOST=connect.westb.seetacloud.com

REM AutoDL SSH port
set AUTODL_PORT=37833

REM AutoDL SSH user (usually root)
set AUTODL_USER=root

REM ===============================================================
REM  Remote paths (usually no need to change)
REM ===============================================================

REM Remote project directory
set AUTODL_PROJECT_DIR=/root/autodl-tmp/ViT-Ctr

REM Remote checkpoint directory
set AUTODL_CKPT_DIR=/root/autodl-tmp/checkpoints

REM Remote data directory
set AUTODL_DATA_DIR=/root/autodl-tmp/data

REM ===============================================================
REM  Local paths (auto-detected, usually no need to change)
REM ===============================================================

REM Local project root (parent of scripts/)
set LOCAL_PROJECT_DIR=%~dp0..

REM Local checkpoint directory
set LOCAL_CKPT_DIR=%LOCAL_PROJECT_DIR%\checkpoints
