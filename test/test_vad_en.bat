@echo off
chcp 65001 >nul
title VAD Function Test

echo ====================================
echo  VAD Function Test
echo ====================================

REM Activate environment
call conda activate ..\pyenv_3.10 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python environment not found. Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

REM Check if test_vad.py exists
if not exist "test_funasr_vad.py" (
    echo [ERROR] test_vad.py not found in current directory
    echo [INFO] Please ensure the VAD test script exists
    echo [INFO] Current directory: %CD%
    echo [INFO] You may need to create the test_vad.py file first
    echo --------------------------------
    echo [INFO] Available Python files in current directory:
    dir /b *.py 2>nul
    if %errorlevel% neq 0 (
        echo [INFO] No Python files found in current directory
    )
    pause
    exit /b 1
)

echo [INFO] Starting VAD test...
echo --------------------------------
python test_funasr_vad.py

if %errorlevel% neq 0 (
    echo --------------------------------
    echo [ERROR] VAD test failed with error code: %errorlevel%
) else (
    echo --------------------------------
    echo [INFO] VAD test completed successfully
)

echo Press any key to close window...
pause > nul
