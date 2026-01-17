@echo off
chcp 65001 >nul
echo ===================================
echo  FunASR-style 高级 VAD 录音系统
echo ===================================

REM 激活环境并运行程序
call conda activate ..\pyenv

echo 启动高级 VAD 录音系统...
python FunASR_Advanced_VAD.py

pause
