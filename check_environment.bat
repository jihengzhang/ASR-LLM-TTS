@echo off
chcp 65001 >nul
title FunASR 环境检查
cls
echo.
echo  ====================================================
echo     FunASR 环境检查工具
echo  ====================================================
echo.

:: 激活 Python 环境
echo  [INFO] 正在激活 Python 环境...
:: 尝试使用 conda 激活环境
call conda activate ..\pyenv 2>nul
if %errorlevel% NEQ 0 (
    :: 如果 conda 激活失败，尝试其他方式
    if exist ..\venv\Scripts\activate.bat (
        call ..\venv\Scripts\activate.bat
    ) else if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else if exist ..\miniconda3\Scripts\activate.bat (
        call ..\miniconda3\Scripts\activate.bat
    ) else (
        echo  [警告] 无法找到 Python 环境激活脚本
        echo  [信息] 将尝试使用系统默认的 Python
    )
) else (
    echo  [INFO] 已成功激活 conda 环境: ..\pyenv
)

:: 运行环境检查
echo.
echo  [INFO] 开始检查环境...
echo  ----------------------------------------------------
echo.
python check_environment.py

echo.
echo  检查已完成
echo  按任意键关闭窗口...
pause > nul
