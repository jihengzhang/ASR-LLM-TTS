@echo off
chcp 65001 >nul
title FunASR 安装和配置
cls
echo.
echo  ====================================================
echo     FunASR 安装和配置助手
echo  ====================================================
echo.

:: 创建录音目录
if not exist recordings mkdir recordings
echo  [INFO] 创建录音目录: recordings

:: 激活 Python 环境
echo.
echo  [INFO] 正在激活 Python 环境...
:: 尝试使用 conda 激活环境
:: 使用 call conda activate 来确保脚本不会退出
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    call conda activate ..\pyenv
    if %errorlevel% EQU 0 (
        echo  [INFO] 已成功激活 conda 环境: ..\pyenv
    ) else (
        echo  [警告] 无法使用 conda activate 命令激活环境
        echo  [INFO] 尝试直接调用激活脚本...
        if exist ..\pyenv\Scripts\activate.bat (
            call ..\pyenv\Scripts\activate.bat
            echo  [INFO] 已激活环境: ..\pyenv
        )
    )
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    call conda activate ..\pyenv
    if %errorlevel% EQU 0 (
        echo  [INFO] 已成功激活 conda 环境: ..\pyenv
    ) else (
        echo  [警告] 无法使用 conda activate 命令激活环境
        echo  [INFO] 尝试直接调用激活脚本...
        if exist ..\pyenv\Scripts\activate.bat (
            call ..\pyenv\Scripts\activate.bat
            echo  [INFO] 已激活环境: ..\pyenv
        )
    )
) else (
    echo  [警告] 未找到 conda 初始化脚本
    echo  [INFO] 尝试直接调用环境激活脚本...
    if exist ..\pyenv\Scripts\activate.bat (
        call ..\pyenv\Scripts\activate.bat
        echo  [INFO] 已激活环境: ..\pyenv
    ) else if exist ..\venv\Scripts\activate.bat (
        call ..\venv\Scripts\activate.bat
        echo  [INFO] 已激活环境: ..\venv
    ) else if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
        echo  [INFO] 已激活环境: venv
    ) else (
        echo  [警告] 无法找到 Python 环境激活脚本
        echo  [信息] 将尝试使用系统默认的 Python
    )
)

:: 检查 Python 版本
echo.
echo  [INFO] 检查 Python 版本...
python -c "import sys; print(f'Python 版本: {sys.version}')"
python -c "import sys; is_compatible = sys.version_info >= (3, 7) and sys.version_info < (3, 11); print(f'Python 版本兼容性: {"兼容" if is_compatible else "可能不兼容，推荐 Python 3.7-3.10"}')"

:: 安装基本依赖
echo.
echo  [INFO] 安装基本依赖...
if exist requirements.txt (
    pip install -r requirements.txt
    if %errorlevel% NEQ 0 (
        echo  [错误] 安装依赖失败，请检查 requirements.txt 文件和网络连接
    ) else (
        echo  [成功] 基本依赖安装完成
    )
) else (
    echo  [警告] 未找到 requirements.txt 文件，跳过依赖安装
)

:: 检查并安装 ffmpeg
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

:: 安装 PyTorch (如果需要)
echo.
echo  [INFO] 是否使用 conda 安装 PyTorch? (推荐)
echo  1. 是 - 使用 conda 安装 (推荐)
echo  2. 否 - 使用 pip 安装
echo.
set /p choice="请选择 (1/2): "

if "%choice%"=="1" (
    echo.
    echo  [INFO] 使用 conda 安装 PyTorch...
    conda install -y pytorch torchaudio cpuonly -c pytorch
) else (
    echo.
    echo  [INFO] 跳过 conda 安装，将使用 pip 安装的 PyTorch...
)

:: 安装 FunASR
echo.
echo  [INFO] 安装 FunASR...
pip install -U funasr
if %errorlevel% NEQ 0 (
    echo  [错误] FunASR 安装失败，尝试降级安装...
    pip install funasr==0.2.5
    if %errorlevel% NEQ 0 (
        echo  [严重错误] FunASR 安装失败，请检查网络连接或手动安装
        echo  [提示] 尝试执行: pip install funasr -i https://mirrors.aliyun.com/pypi/simple/
    ) else (
        echo  [成功] FunASR 安装完成 (使用固定版本 0.2.5)
    )
) else (
    echo  [成功] FunASR 最新版本安装完成
)

:: 下载模型
echo.
echo  [INFO] 是否下载 FunASR 预训练模型? (可选)
echo  1. 是 - 下载模型
echo  2. 否 - 跳过
echo.
set /p model_choice="请选择 (1/2): "

if "%model_choice%"=="1" (
    echo.
    echo  [INFO] 下载 FunASR 预训练模型到项目文件夹...
    if not exist models mkdir models
    
    echo.
    echo  [INFO] 下载 VAD 模型到当前项目目录...
    echo  [INFO] 设置模型缓存目录为: %CD%
    python test_model_cache.py
    if %errorlevel% NEQ 0 (
        echo  [错误] VAD 模型下载失败，请检查网络连接
        echo  [提示] 请尝试手动运行: python test_model_cache.py
    ) else (
        echo  [成功] VAD 模型下载完成，保存在当前项目目录
        echo  [INFO] 项目路径: %CD%
    )
)

echo.
echo  [INFO] 安装完成！
echo  [INFO] 可以运行 run_vad.bat 启动语音激活录音系统
echo.
echo  按任意键继续...
pause > nul
