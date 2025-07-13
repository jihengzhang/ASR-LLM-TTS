@echo off
chcp 65001 >nul
title FunASR VAD 测试工具
cls
echo.
echo  ====================================================
echo     FunASR VAD 测试工具
echo  ====================================================
echo.

:: 激活 Python 环境
echo.
echo  [INFO] 正在激活 Python 环境...
call conda activate ..\pyenv 2>nul
if %errorlevel% EQU 0 (
    echo  [INFO] 已激活环境: ..\pyenv
) else (
    echo  [警告] 无法找到或激活 Python 环境: ..\pyenv
    echo  [INFO] 将尝试使用系统默认的 Python
    
    :: 检查系统Python
    where python > nul 2>&1
    if %errorlevel% NEQ 0 (
        echo  [错误] 未找到 Python。请确保 Python 已安装并添加到 PATH 中。
        echo  按任意键退出...
        pause > nul
        exit /b 1
    ) else (
        echo  [INFO] 找到系统 Python，将使用系统环境
    )
)

:: 运行测试程序
echo.
echo  [INFO] 启动 VAD 测试...
echo  ----------------------------------------------------
echo.
python test_funasr_vad.py

echo.
echo  测试已完成
echo  按任意键关闭窗口...
pause > nul
