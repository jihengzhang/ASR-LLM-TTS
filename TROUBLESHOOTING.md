# FunASR 问题修复指南

如果您在使用 FunASR VAD 系统时遇到问题，请按照以下步骤操作：

## 常见问题 1: Python 环境路径错误

**症状:**
```
'..\pyenv\Scripts\activate.bat' is not recognized as an internal or external command
```

**解决方案:**
1. 运行 `create_local_env.bat` 创建本地 Python 环境
2. 然后运行 `setup_funasr.bat` 继续安装

## 常见问题 2: FunASR 模型下载失败

**症状:**
```
Download: paraformer-vad failed!: Invalid repo_id: model
```

**解决方案:**
这是因为模型名称已更新。我们已经修复了脚本中的模型名称。请使用新的脚本运行。

## 常见问题 3: ffmpeg 未安装

**症状:**
```
ffmpeg is not installed
```

**解决方案:**
1. 运行 `setup_funasr.bat`
2. 选择选项 2 "安装依赖"
3. 按照提示安装 ffmpeg

## 常见问题 4: 无法安装 PyAudio

**症状:**
安装 PyAudio 时出现编译错误

**解决方案:**
1. 对于 Windows 用户:
   ```
   pip install pipwin
   pipwin install pyaudio
   ```

2. 或者下载预编译的 wheel 文件:
   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## 使用说明

1. 首先运行 `create_local_env.bat` 创建本地 Python 环境
2. 然后运行 `setup_funasr.bat` 安装所有依赖
3. 使用 `test_vad.bat` 测试 VAD 功能
4. 最后运行 `run_vad.bat` 启动 VAD 录音系统

## 联系支持

如果您仍然遇到问题，请联系我们获取更多支持。
