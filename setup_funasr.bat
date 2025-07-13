@echo off
chcp 65001 >nul
title FunASR 一键安装和测试
cls
echo.
echo  ====================================================
echo     FunASR 一键安装和测试工具
echo  ====================================================
echo.

@REM :: 检查管理员权限
@REM >nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
@REM if '%errorlevel%' NEQ '0' (
@REM     echo  [警告] 请以管理员身份运行此脚本以获得最佳效果
@REM     echo  [警告] 某些操作可能需要管理员权限
@REM     echo.
@REM     echo  按任意键继续...
@REM     pause > nul
@REM )

:: 创建录音目录
if not exist recordings mkdir recordings
echo  [INFO] 创建录音目录: recordings

:: 创建模型目录
if not exist models mkdir models
echo  [INFO] 创建模型目录: models

:: 尝试确定 Python 环境路径
echo.
echo  [INFO] 正在激活 Python 环境...
call conda activate ..\pyenv 2>nul
if %errorlevel% EQU 0 (
    echo  [INFO] 已激活环境: ..\pyenv
) else (
    echo  [警告] 无法找到或激活 Python 环境: ..\pyenv
    echo  [INFO] 将尝试使用系统默认的 Python
)

:: 显示菜单
:menu
cls
echo.
echo  ====================================================
echo     FunASR 一键安装和测试工具
echo  ====================================================
echo.
echo  请选择要执行的操作:
echo.
echo  [1] 检查环境
echo  [2] 安装依赖
echo  [3] 安装 FunASR
echo  [4] 下载 VAD 模型
echo  [5] 测试 VAD 功能
echo  [6] 运行 VAD 录音系统
echo  [7] 全部执行 (1-6)
echo  [8] 查看安装指南
echo  [0] 退出
echo.
set /p choice="请输入选项 (0-8): "

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
echo  [INFO] 开始检查环境...
python check_environment.py
echo.
echo  按任意键返回菜单...
pause > nul
goto menu

:install_deps
echo.
echo  [INFO] 安装基本依赖...
pip install -r requirements.txt

echo.
echo  [INFO] 检查 ffmpeg 是否已安装...
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo  [WARN] ffmpeg 未安装，现在将安装 ffmpeg...
    echo  [INFO] 是否使用 conda 安装 ffmpeg? (推荐)
    echo  1. 是 - 使用 conda 安装 (推荐)
    echo  2. 否 - 使用下载方式
    echo.
    set /p ffmpeg_choice="请选择 (1/2): "

    if "%ffmpeg_choice%"=="1" (
        echo.
        echo  [INFO] 使用 conda 安装 ffmpeg...
        conda install -y ffmpeg -c conda-forge
    ) else (
        echo.
        echo  [INFO] 请手动下载 ffmpeg 并将其添加到 PATH 环境变量中
        echo  [INFO] 下载地址: https://ffmpeg.org/download.html
        echo  [INFO] 按任意键继续...
        pause > nul
    )
) else (
    echo  [INFO] ffmpeg 已安装
)

echo.
echo  [INFO] 是否使用 conda 安装 PyTorch? (推荐)
echo  1. 是 - 使用 conda 安装 (推荐)
echo  2. 否 - 使用 pip 安装
echo.
set /p pytorch_choice="请选择 (1/2): "

if "%pytorch_choice%"=="1" (
    echo.
    echo  [INFO] 使用 conda 安装 PyTorch...
    conda install -y pytorch torchaudio cpuonly -c pytorch
) else (
    echo.
    echo  [INFO] 使用 pip 安装 PyTorch...
    pip install torch torchaudio
)

echo.
echo  依赖安装完成
echo  按任意键返回菜单...
pause > nul
goto menu

:install_funasr
echo.
echo  [INFO] 安装 FunASR...
pip install -U funasr
echo.
echo  FunASR 安装完成
echo  按任意键返回菜单...
pause > nul
goto menu

:download_model
echo.
echo  [INFO] 下载 FunASR VAD 模型到当前项目目录...
echo  [INFO] 设置模型缓存目录为: %CD%
python test_model_cache.py
echo.
echo  按任意键返回菜单...
pause > nul
goto menu

:test_vad
echo.
echo  [INFO] 测试 VAD 功能...
call test_vad.bat
echo.
echo  按任意键返回菜单...
pause > nul
goto menu

:run_vad
echo.
echo  [INFO] 运行 VAD 录音系统...
call run_vad.bat
echo.
echo  按任意键返回菜单...
pause > nul
goto menu

:run_all
echo.
echo  [INFO] 执行所有步骤...
echo.
echo  [步骤 1/6] 检查环境
python check_environment.py
echo.
echo  [步骤 2/6] 安装依赖
pip install -r requirements.txt

echo  [INFO] 检查 ffmpeg...
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo  [INFO] 安装 ffmpeg...
    conda install -y ffmpeg -c conda-forge
)

echo  [INFO] 安装 PyTorch...
conda install -y pytorch torchaudio cpuonly -c pytorch

echo.
echo  [步骤 3/6] 安装 FunASR
pip install -U funasr

echo.
echo  [步骤 4/6] 下载 VAD 模型到当前项目目录
echo  [INFO] 设置模型缓存目录为: %CD%
python test_model_cache.py

echo.
echo  [步骤 5/6] 测试 VAD 功能
call test_vad.bat

echo.
echo  [步骤 6/6] 运行 VAD 录音系统
call run_vad.bat

echo.
echo  全部步骤执行完成
echo  按任意键返回菜单...
pause > nul
goto menu

:show_guide
echo.
echo  [INFO] 打开安装指南...
start INSTALL_GUIDE.md
echo.
echo  按任意键返回菜单...
pause > nul
goto menu

:end
@REM echo.
@REM echo  感谢使用 FunASR 一键安装和测试工具
@REM echo  按任意键退出...
@REM pause > nul
exit
