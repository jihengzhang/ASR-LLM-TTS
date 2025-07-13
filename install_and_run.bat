@echo off
chcp 65001 >nul
REM FanASR VAD 关键词激活录音系统 - 混合安装脚本

echo ===================================
echo  FanASR VAD 关键词激活录音系统
echo ===================================

REM 激活 Miniconda 环境
echo 正在激活 Miniconda 环境...
call conda activate ..\pyenv

echo.
echo 步骤 1: 使用 conda 安装基础科学计算包...
conda install -y numpy scipy

echo.
echo 步骤 2: 使用 pip 安装音频处理包...
echo 正在尝试安装 pyaudio...
pip install pyaudio

echo 正在尝试安装 webrtcvad...
pip install webrtcvad
if %errorlevel% neq 0 (
    echo webrtcvad 安装失败，尝试使用替代方案...
    echo 选项 1: 尝试安装预编译版本
    pip install --find-links https://download.pytorch.org/whl/torch_stable.html webrtcvad
    if %errorlevel% neq 0 (
        echo 选项 2: 使用 Silero VAD 作为替代
        pip install silero-vad
        echo 注意：程序将使用 Silero VAD 替代 WebRTC VAD
    )
)

echo.
echo 步骤 3: 检查是否需要安装 FunASR 完整功能...
set /p install_funasr="是否安装 FunASR 完整功能? (y/n): "
if /i "%install_funasr%"=="y" (
    echo 正在安装 FunASR 和 PyTorch...
    conda install -y pytorch torchaudio -c pytorch
    pip install funasr
)

echo.
echo 依赖安装完成！
echo.
echo 启动关键词激活录音系统...
python FanASR_VAD_demo.py

pause
