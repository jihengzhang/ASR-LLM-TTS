@echo off
chcp 65001 >nul
title VAD Recording System

echo ====================================
echo  VAD Recording System
echo ====================================

REM Activate environment
call conda activate ..\pyenv_3.10 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python environment not found. Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

echo [INFO] Starting VAD recording system...
echo --------------------------------
python FanASR_VAD_demo.py

echo --------------------------------
echo [INFO] VAD recording system stopped
echo Press any key to close window...
pause > nul
