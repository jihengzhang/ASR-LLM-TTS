@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM FunASR VAD Keyword Activation Recording System - Hybrid Installation Script

echo "==================================="
echo " FunASR VAD Keyword Activation System"
echo "==================================="

REM Activate Miniconda environment
echo "Activating Miniconda environment..."
call conda activate ..\pyenv_3.10

echo.
echo "Step 1: Installing basic scientific computing packages with conda..."
conda install -y numpy scipy

echo.
echo "Step 2: Installing audio processing packages with pip..."
echo "Trying to install pyaudio..."
pip install pyaudio

echo "Trying to install webrtcvad..."
pip install webrtcvad
if !errorlevel! neq 0 (
    echo "webrtcvad installation failed, trying alternative methods..."
    echo "Option 1: Trying to install precompiled version"
    pip install --find-links https://download.pytorch.org/whl/torch_stable.html webrtcvad
    if !errorlevel! neq 0 (
        echo "Option 2: Using Silero VAD as alternative"
        pip install silero-vad
        echo "Note: Program will use Silero VAD instead of WebRTC VAD"
    )
)

echo.
echo "Step 3: Check if FunASR full functionality installation is needed..."
set /p install_funasr="Install FunASR full functionality? (y/n): "
if /i "!install_funasr!"=="y" (
    echo "Installing FunASR and PyTorch..."
    conda install -y pytorch torchaudio -c pytorch
    pip install funasr
)

echo.
echo "Dependencies installation completed!"
echo.
echo "Starting keyword activation recording system..."
python .\FanASR_VAD_demo.py

pause
