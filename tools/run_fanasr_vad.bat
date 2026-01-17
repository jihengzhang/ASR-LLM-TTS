@echo off
chcp 65001 >nul
title FunASR VAD 语音激活录音系统
cls
echo.
echo  ====================================================
echo     FunASR VAD 语音激活录音系统
echo  ====================================================
echo.

:: 检查录音目录
if not exist recordings mkdir recordings
echo  [INFO] 录音将保存在 recordings 目录

:: 激活 Python 环境
echo.
echo  [INFO] 正在检测 Python 环境...
where python > nul 2>&1
if %errorlevel% EQU 0 (
    echo  [INFO] 找到系统 Python
    
    :: 尝试激活环境
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        echo  [INFO] 激活 Miniconda 环境...
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        echo  [INFO] 激活 Anaconda 环境...
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    ) else (
        echo  [INFO] 尝试激活 pyenv 环境...
        call conda activate ..\pyenv 2>nul
        if %errorlevel% NEQ 0 (
            echo  [INFO] 使用系统 Python 环境
        ) else (
            echo  [INFO] 已激活 pyenv 环境
        )
    )
    ) else (
        echo  [INFO] 使用系统 Python 环境
    )
) else (
    echo  [警告] 未找到 Python。请确保 Python 已安装并添加到 PATH 中。
    echo  按任意键退出...
    pause > nul
    exit /b 1
)

:: 运行程序
echo.
echo  [INFO] 启动 VAD 录音系统...
echo  ----------------------------------------------------
echo.
python FunASR_VAD_demo.py

echo.
echo  程序已退出
echo  按任意键关闭窗口...
pause > nul
