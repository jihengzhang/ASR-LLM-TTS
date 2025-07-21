# FunASR 安装指南和故障排除

本文档提供了安装 FunASR 语音识别系统的详细步骤和常见问题的解决方案。

## 安装前提

在安装 FunASR 之前，请确保您的系统满足以下要求：

1. **Python 3.8 或更高版本**
   - 推荐使用 Python 3.8-3.10，这些版本与 FunASR 兼容性最好

2. **C++ 编译器**
   - Windows: Visual C++ 14.0 或更高版本
   - Linux: GCC 5 或更高版本

3. **ffmpeg**
   - 必须安装并添加到系统 PATH 中
   - Windows 用户可以通过 conda 安装: `conda install -c conda-forge ffmpeg`

## 安装步骤

### 1. 自动安装（推荐）

最简单的方法是运行我们提供的安装脚本：

```
install_funasr.bat
```

这个脚本会：
- 安装所有必要的依赖
- 安装 FunASR
- 可选地安装 ffmpeg
- 下载预训练的 VAD 模型

### 2. 手动安装

如果自动安装失败，可以按照以下步骤手动安装：

1. **安装基础依赖**
   ```
   pip install numpy scipy pyaudio colorama
   ```

2. **安装 PyTorch**
   ```
   # 使用 conda (推荐)
   conda install pytorch torchaudio cpuonly -c pytorch

   # 或使用 pip
   pip install torch torchaudio
   ```

3. **安装 FunASR**
   ```
   pip install -U funasr
   ```

4. **安装 ffmpeg**
   - 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载
   - 将其添加到系统 PATH 中

5. **下载 VAD 模型**
   ```python
   from funasr import AutoModel
   model = AutoModel(model="paraformer-vad", model_revision="v2.0.4")
   ```

## 验证安装

安装完成后，运行以下脚本来验证安装是否成功：

```
check_environment.bat
```

或直接运行 VAD 测试：

```
test_vad.bat
```

## 常见问题及解决方案

### 1. ffmpeg 未安装或未找到

**症状**：
- 错误消息: "ffmpeg is not installed" 或 "ffmpeg command not found"

**解决方案**：
- 使用 conda 安装: `conda install -c conda-forge ffmpeg`
- 或从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，并添加到 PATH

### 2. PyAudio 安装失败

**症状**：
- 错误消息: "error: Microsoft Visual C++ 14.0 or greater is required"

**解决方案**：
- 安装 Visual C++ Build Tools
- 或使用预编译的 wheel: `pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl`

### 3. FunASR 安装失败

**症状**：
- 错误消息包含 "Failed building wheel for sentencepiece" 等

**解决方案**：
- 确保已安装 C++ 编译器
- 尝试降级 Python 版本到 3.9
- 使用 conda 环境: `conda create -n funasr python=3.9`

### 4. 模型下载失败

**症状**：
- 下载模型时连接超时或错误

**解决方案**：
- 检查网络连接
- 尝试使用 VPN
- 手动下载模型并放入 `models` 目录

## 卸载与重新安装

如果遇到问题需要重新安装：

1. 卸载现有的 FunASR:
   ```
   pip uninstall funasr
   ```

2. 清理 cache:
   ```
   pip cache purge
   ```

3. 重新安装:
   ```
   pip install -U funasr
   ```

## 更多资源

- [FunASR GitHub 仓库](https://github.com/alibaba-damo-academy/FunASR)
- [FunASR 官方文档](https://funasr.readthedocs.io/)
