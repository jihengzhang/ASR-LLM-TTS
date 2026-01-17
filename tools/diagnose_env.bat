@echo off
chcp 65001 >nul
title 环境诊断工具
cls
echo.
echo  ====================================================
echo         环境诊断工具
echo  ====================================================
echo.

echo  [检查 1] 当前工作目录:
echo  %CD%
echo.

echo  [检查 2] Python 环境检测:
where python > nul 2>&1
if %errorlevel% EQU 0 (
    echo  ✓ 找到 Python
    python --version
) else (
    echo  ✗ 未找到 Python
)
echo.

echo  [检查 3] Conda 环境检测:
where conda > nul 2>&1
if %errorlevel% EQU 0 (
    echo  ✓ 找到 Conda
    conda --version
) else (
    echo  ✗ 未找到 Conda
)
echo.

echo  [检查 4] 目标环境路径检测:
if exist "..\pyenv" (
    echo  ✓ 找到环境目录: ..\pyenv
    echo    绝对路径: %CD%\..\pyenv
) else (
    echo  ✗ 未找到环境目录: ..\pyenv
    echo    预期路径: %CD%\..\pyenv
)
echo.

echo  [检查 5] 尝试激活环境:
call conda activate ..\pyenv 2>nul
if %errorlevel% EQU 0 (
    echo  ✓ 成功激活环境: ..\pyenv
    echo  当前环境信息:
    conda info --envs | findstr "*"
) else (
    echo  ✗ 无法激活环境: ..\pyenv
    echo  可用环境列表:
    conda info --envs 2>nul
)
echo.

echo  [检查 6] 环境变量检测:
echo  CONDA_DEFAULT_ENV: %CONDA_DEFAULT_ENV%
echo  CONDA_PREFIX: %CONDA_PREFIX%
echo  PATH前缀: 
echo  %PATH% | findstr /i conda
echo.

echo  [检查 7] FunASR 模块检测:
python -c "import funasr; print('✓ FunASR 模块可用')" 2>nul || echo  ✗ FunASR 模块不可用
echo.

echo  ====================================================
echo  诊断完成
echo  ====================================================
echo.
pause
