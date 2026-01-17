@echo off
chcp 65001 >nul
title 创建本地 Python 环境
cls
echo.
echo  ====================================================
echo     创建本地 Python 环境
echo  ====================================================
echo.

:: 检查Python是否已安装
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo  [错误] 未检测到 Python，请先安装 Python 3.8 或更高版本
    echo  [信息] 下载地址: https://www.python.org/downloads/
    echo.
    echo  按任意键退出...
    pause > nul
    exit /b 1
)

:: 显示 Python 版本
echo  [INFO] 检测到 Python:
python --version
echo.

:: 创建本地虚拟环境
echo  [INFO] 创建本地 Python 虚拟环境...
if exist venv (
    echo  [WARN] 已存在 venv 目录，是否删除并重新创建?
    echo  1. 是 - 删除并重新创建
    echo  2. 否 - 使用现有环境
    echo.
    set /p venv_choice="请选择 (1/2): "
    
    if "%venv_choice%"=="1" (
        echo  [INFO] 删除现有虚拟环境...
        rmdir /s /q venv
        python -m venv venv
        echo  [INFO] 已创建新的虚拟环境
    ) else (
        echo  [INFO] 将使用现有虚拟环境
    )
) else (
    python -m venv venv
    echo  [INFO] 已创建新的虚拟环境
)

:: 激活环境
echo.
echo  [INFO] 激活虚拟环境...
call venv\Scripts\activate.bat

:: 升级pip
echo.
echo  [INFO] 升级 pip...
python -m pip install --upgrade pip

echo.
echo  [成功] 本地 Python 环境已创建并激活
echo  [信息] 请运行 setup_funasr.bat 继续安装

echo.
echo  按任意键继续...
pause > nul
