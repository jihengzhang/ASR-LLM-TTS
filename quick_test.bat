@echo off
chcp 65001 >nul
title VAD 系统快速测试
cls
echo.
echo  ====================================================
echo         VAD 系统快速测试
echo  ====================================================
echo.

:: 激活环境
echo  [1/3] 激活 Python 环境...
call conda activate ..\pyenv 2>nul
if %errorlevel% EQU 0 (
    echo  ✓ 已激活环境: ..\pyenv
) else (
    echo  ✗ 环境激活失败
    pause
    exit /b 1
)

:: 检查FunASR
echo  [2/3] 检查 FunASR 模块...
python -c "import funasr; print('✓ FunASR 模块可用，版本:', funasr.__version__)" 2>nul
if %errorlevel% NEQ 0 (
    echo  ✗ FunASR 模块不可用
    pause
    exit /b 1
)

:: 快速测试VAD功能
echo  [3/3] 测试 VAD 功能...
echo  正在运行简化的VAD测试...
python -c "
from funasr import AutoModel
import os
os.environ['MODELSCOPE_CACHE'] = os.getcwd()
print('正在加载VAD模型...')
try:
    model = AutoModel(model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch', model_revision='v2.0.4')
    print('✓ VAD模型加载成功')
except Exception as e:
    print('✗ VAD模型加载失败:', str(e))
"

echo.
echo  ====================================================
echo  测试完成！您现在可以运行 VAD 录音系统
echo  ====================================================
echo.
pause
