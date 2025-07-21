@echo off
chcp 65001 >nul
title FunASR One-Click Installation and Testing
cls
echo.
echo  ====================================================
echo     FunASR One-Click Installation and Testing Tool
echo  ====================================================
echo.

@REM :: Check administrator privileges
@REM >nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
@REM if '%errorlevel%' NEQ '0' (
@REM     echo  [WARNING] Please run this script as administrator for best results
@REM     echo  [WARNING] Some operations may require administrator privileges
@REM     echo.
@REM     echo  Press any key to continue...
@REM     pause > nul
@REM )

:: Create recording directory
if not exist recordings mkdir recordings
echo  [INFO] Created recording directory: recordings

:: Create model directory
if not exist models mkdir models
echo  [INFO] Created model directory: models

:: Try to determine Python environment path
echo.
echo  [INFO] Activating Python environment...
call conda activate ..\pyenv_3.10 2>nul
if %errorlevel% EQU 0 (
    echo  [INFO] Activated environment: ..\pyenv_3.10
) else (
    echo  [WARNING] Cannot find or activate Python environment: ..\pyenv_3.10
    echo  [INFO] Will try to use system default Python
)

:: Display menu
:menu
cls
echo.
echo  ====================================================
echo     FunASR One-Click Installation and Testing Tool
echo  ====================================================
echo.
echo  Please select operation to execute:
echo.
echo  [1] Check Environment
echo  [2] Install Dependencies
echo  [3] Install FunASR
echo  [4] Download VAD Model
echo  [5] Test VAD Function
echo  [6] Run VAD Recording System
echo  [7] Execute All (1-6)
echo  [8] View Installation Guide
echo  [0] Exit
echo.
set /p choice="Please enter option (0-8): "

if "%choice%"=="1" goto check_env
if "%choice%"=="2" goto install_deps
if "%choice%"=="3" goto install_funasr
if "%choice%"=="4" goto download_model
if "%choice%"=="5" goto test_vad
if "%choice%"=="6" goto run_vad
if "%choice%"=="7" goto run_all
if "%choice%"=="8" goto show_guide
if "%choice%"=="0" goto end
goto menu

:check_env
echo.
echo  [INFO] Starting environment check...
python check_environment.py
echo.
echo  Press any key to return to menu...
pause > nul
goto menu

:install_deps
echo.
echo  [INFO] Installing basic dependencies...
echo  [INFO] Configuring pip for trusted hosts...
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org www.modelscope.cn"
pip install -r requirements.txt

echo.
echo  [INFO] Checking if ffmpeg is installed...
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo  [WARN] ffmpeg not installed, installing ffmpeg now...
    echo  [INFO] Use conda to install ffmpeg? (recommended)
    echo  1. Yes - Use conda install (recommended)
    echo  2. No - Use download method
    echo.
    set /p ffmpeg_choice="Please choose (1/2): "

    if "%ffmpeg_choice%"=="1" (
        echo.
        echo  [INFO] Installing ffmpeg using conda...
        conda install -y ffmpeg -c conda-forge
    ) else (
        echo.
        echo  [INFO] Please manually download ffmpeg and add it to PATH environment variable
        echo  [INFO] Download URL: https://ffmpeg.org/download.html
        echo  [INFO] Press any key to continue...
        pause > nul
    )
) else (
    echo  [INFO] ffmpeg is already installed
)

echo.
echo  [INFO] Use conda to install PyTorch? (recommended)
echo  1. Yes - Use conda install (recommended)
echo  2. No - Use pip install
echo.
set /p pytorch_choice="Please choose (1/2): "

if "%pytorch_choice%"=="1" (
    echo.
    echo  [INFO] Installing PyTorch using conda...
    conda install -y pytorch torchaudio cpuonly -c pytorch
) else (
    echo.
    echo  [INFO] Installing PyTorch using pip...
    pip install torch torchaudio
)

echo.
echo  Dependencies installation completed
echo  Press any key to return to menu...
pause > nul
goto menu

:install_funasr
echo.
echo  [INFO] Installing FunASR with SSL bypass...
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org www.modelscope.cn"
pip install -U funasr --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo.
echo  FunASR installation completed
echo  Press any key to return to menu...
pause > nul
goto menu

:download_model
echo.
echo  [INFO] Downloading FunASR VAD model to current project directory...
echo  [INFO] Setting model cache directory to: %CD%
echo  [INFO] Configuring SSL bypass for model download...

REM Set environment variables to bypass SSL verification
set PYTHONHTTPSVERIFY=0

echo  [INFO] Trying model download with SSL bypass...
python test_model_cache.py

if %errorlevel% neq 0 (
    echo  [WARNING] Model download failed, trying alternative method...
    echo  [INFO] Setting up alternative model download configuration...
    
    REM Try pip configuration for trusted hosts
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org www.modelscope.cn"
    
    echo  [INFO] Creating temporary SSL bypass script...
    echo import ssl > temp_ssl_bypass.py
    echo import os >> temp_ssl_bypass.py
    echo import urllib3 >> temp_ssl_bypass.py
    echo urllib3.disable_warnings() >> temp_ssl_bypass.py
    echo ssl._create_default_https_context = ssl._create_unverified_context >> temp_ssl_bypass.py
    echo os.environ['PYTHONHTTPSVERIFY'] = '0' >> temp_ssl_bypass.py
    echo os.environ.pop('CURL_CA_BUNDLE', None) >> temp_ssl_bypass.py
    echo os.environ.pop('REQUESTS_CA_BUNDLE', None) >> temp_ssl_bypass.py
    echo exec(open('test_model_cache.py', encoding='utf-8').read()) >> temp_ssl_bypass.py
    
    echo  [INFO] Trying direct model download with alternative settings...
    python temp_ssl_bypass.py
    
    if %errorlevel% neq 0 (
        echo  [ERROR] Model download still failed. Manual steps required:
        echo  [INFO] 1. Check internet connection
        echo  [INFO] 2. Try running: pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org funasr
        echo  [INFO] 3. Or download model manually from: https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
    )
    
    REM Clean up temporary file
    if exist temp_ssl_bypass.py del temp_ssl_bypass.py
)

REM Reset environment variables
set PYTHONHTTPSVERIFY=

echo.
echo  Press any key to return to menu...
pause > nul
goto menu

:test_vad
echo.
echo  [INFO] Testing VAD function...
call test_vad_en.bat
echo.
echo  Press any key to return to menu...
pause > nul
goto menu

:run_vad
echo.
echo  [INFO] Running VAD recording system...
call run_vad_en.bat
echo.
echo  Press any key to return to menu...
pause > nul
goto menu

:run_all
echo.
echo  [INFO] Executing all steps...
echo.
echo  [Step 1/6] Check Environment
python check_environment.py
echo.
echo  [Step 2/6] Install Dependencies
pip install -r requirements.txt

echo  [INFO] Checking ffmpeg...
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo  [INFO] Installing ffmpeg...
    conda install -y ffmpeg -c conda-forge
)

echo  [INFO] Installing PyTorch...
conda install -y pytorch torchaudio cpuonly -c pytorch

echo.
echo  [Step 3/6] Install FunASR
pip install -U funasr

echo.
echo  [Step 4/6] Download VAD model to current project directory
echo  [INFO] Setting model cache directory to: %CD%
echo  [INFO] Configuring SSL bypass for model download...

REM Set SSL bypass environment variables
set PYTHONHTTPSVERIFY=0

python test_model_cache.py

REM Reset environment variables
set PYTHONHTTPSVERIFY=

echo.
echo  [Step 5/6] Test VAD function
call test_vad_en.bat

echo.
echo  [Step 6/6] Run VAD recording system
call run_vad_en.bat

echo.
echo  All steps execution completed
echo  Press any key to return to menu...
pause > nul
goto menu

:show_guide
echo.
echo  [INFO] Opening installation guide...
start INSTALL_GUIDE.md
echo.
echo  Press any key to return to menu...
pause > nul
goto menu

:end
echo.
echo  Thank you for using FunASR One-Click Installation and Testing Tool
echo  Press any key to exit...
pause > nul
exit
@REM echo  Thank you for using FunASR One-Click Installation and Testing Tool
@REM echo  Press any key to exit...
@REM pause > nul
exit
