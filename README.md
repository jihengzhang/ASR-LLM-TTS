# FunASR VAD 语音激活录音系统

这是一个基于 FunASR 的语音激活检测(VAD)系统，可以自动检测语音并录音保存。系统支持两种工作模式：使用 FunASR 的高级 VAD 检测，以及基于简单振幅阈值的备用模式。

## 功能特点

- 实时语音活动检测 (VAD)
- 自动开始/停止录音
- 预缓冲录音 (捕获语音开始前的内容)
- 彩色终端输出
- 支持 FunASR 高级 VAD 模型
- 声音振幅可视化显示

## 安装说明

1. 运行 `install_funasr.bat` 安装所需依赖
   - 会安装基本依赖: PyAudio, NumPy, SciPy, colorama
   - 安装 FunASR 和相关依赖
   - 可选下载预训练 VAD 模型

2. 如果安装过程中遇到问题:
   - 对于 PyAudio 安装问题，可以尝试使用预编译的 wheel 文件
   - 对于 PyTorch 安装问题，建议使用 conda 安装
   - 如果 FunASR 安装失败，系统会自动回退到基本的振幅检测方式

## 使用方法

1. 运行 `run_vad.bat` 启动系统
2. 开始说话，系统会自动检测语音并开始录音
3. 停止说话 2 秒后，系统会自动停止录音并保存
4. 录音文件会保存在 `recordings` 目录中
5. 按 Ctrl+C 可以退出程序

## 录音文件

所有录音文件将保存在 `recordings` 文件夹下，文件命名格式为：
```
recording_YYYYMMDD_HHMMSS.wav
```

## 参数调整

可以在 `FunASR_VAD_demo.py` 文件中调整以下参数:

```python
recorder = KeywordActivatedRecorder(
    threshold=0.03,              # 音频激活阈值 (0.01-0.1)
    silence_duration=2.0,        # 停止录音前的静音持续时间 (秒)
    min_speech_duration=0.8,     # 最小语音持续时间 (秒)
    buffer_duration=2.0          # 预缓冲持续时间 (秒)
)
```

## 系统要求

- Python 3.8 或更高版本
- Windows 系统 (也支持 Linux/Mac，但需要修改 .bat 文件)
- 麦克风设备

## 技术说明

系统使用两种语音检测方法:
1. **FunASR VAD 模型** - 当可用时，使用高级语音活动检测
2. **振幅阈值检测** - 作为备用方法，基于音频电平检测语音

系统会优先使用 FunASR 的 VAD 模型，如果不可用则自动回退到基本检测。

## 故障排除

1. 如果安装 PyAudio 失败，Windows 用户可能需要安装 Visual C++ Build Tools
2. 确保麦克风正确连接且在系统中启用
3. 如果检测不到语音，尝试降低 `threshold` 参数值
4. 如果录音结束太快，尝试增加 `silence_duration` 参数值
5. 安装 FunASR 失败时，确保 Python 版本兼容 (推荐 3.8-3.10)
