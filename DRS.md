<!--
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
-->

# VAD_KWS_plot.py 设计文档

## 文档版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2025 | jiheng.zhang | 初始版本：基础VAD/KWS功能 |
| v2.0 | 2026-03-08 | jiheng.zhang | 降噪集成、Vosk KWS唤醒、状态机优化、可视化增强 |
| v2.0.1 | 2026-03-08 | jiheng.zhang | Facebook Denoiser集成（-24.5dB降噪）替换noisereduce |
| v2.0.2 | 2026-03-08 | jiheng.zhang | Silero VAD集成（torch.hub，无需C编译器） |
| v2.0.3 | 2026-03-08 | jiheng.zhang | Force Denoise All Frames复选框，解决VAD-降噪冲突 |
| v2.0.4 | 2026-03-08 | jiheng.zhang | Y轴调整至±0.2，适配高质量麦克风评估 |

---

## v1.0 需求概述（初始版本）

创建一个实时语音关键词检测系统，具备以下功能：

1. 通过PyAudio录制实时音频
2. 将音频实时显示在Matplotlib图表上
3. 使用线程增强实时性能
4. 在audio_callback中以可配置的间隔(默认0.25秒)调用VAD模型检测语音
5. 检测到有效语音时，调用KWS(关键词检测)进行识别
6. 识别到关键词后，开始录音并显示识别文本
7. 将VAD结果和音频幅度显示在图表上
8. 将关键词识别结果也显示在图表上
9. 在主循环里增加一个UI界面,包括Micphone mutu button, 开始语音识别button, matplotlib diagram, 文本脚本窗口,用于接收识别到的文本和时间 类似会议语音转录界面,可以选择输入设备

### v1.0 架构设计

#### 主要组件

1. **AudioRecorder类**：处理音频录制和回调
2. **VADProcessor类**：语音活动检测
3. **KWSDetector类**：关键词识别
4. **AudioVisualizer类**：实时音频可视化

#### 数据流

1. PyAudio通过callback捕获音频数据
2. 音频数据传递给VAD进行语音检测
3. 如检测到语音，数据传递给KWS进行关键词识别
4. 检测到关键词后，开始录音和记录文本
5. 所有处理结果实时显示在图表上

#### 线程设计

1. 主线程：UI和用户交互
2. 音频处理线程：处理音频数据
3. 可视化线程：更新图表显示
4. VAD线程：语音检测
5. KWS线程：关键词识别

### v1.0 关键技术要点

1. **PyAudio配置**：
   - 采样率：16kHz
   - 格式：16位整数
   - 缓冲大小：根据实时性要求调整

2. **VAD模型**：
   - 使用FunASR的VAD模型
   - 检测间隔可配置(默认0.25秒)
   - 结合平均幅度提高检测准确率

3. **KWS模型**：
   - 使用FunASR的ASR模型进行关键词识别
   - 支持中英文关键词

4. **可视化组件**：
   - 上半部分：音频波形图和VAD检测结果
   - 下半部分：频谱图和关键词检测结果
   - 使用实时更新机制确保显示流畅

5. **实时性优化**：
   - 使用线程池处理并行任务
   - 限制不必要的处理和计算
   - 采用缓冲机制平衡实时性和稳定性

### v1.0 用户界面设计

1. **实时音频波形图**：
   - 显示原始音频波形
   - VAD检测到的语音段用高亮区域标记

2. **语音活动指示器**：
   - 显示VAD检测结果
   - 结合音频平均幅度

3. **关键词检测结果区域**：
   - 显示最近检测到的关键词
   - 显示关键词触发状态

4. **控制面板**：
   - 开始/停止录音按钮
   - VAD敏感度调整
   - 关键词列表配置

### v1.0 实现步骤

1. 初始化PyAudio和音频流
2. 设置音频回调函数，收集数据
3. 创建VAD处理线程，定期检测语音
4. 实现KWS处理逻辑，识别关键词
5. 设计和实现可视化界面
6. 集成所有组件，确保实时性
7. 优化性能，减少延迟

### v1.0 性能考虑

1. VAD和KWS处理应在单独线程中进行，避免阻塞音频回调
2. 图表更新频率应适中，避免过度消耗CPU资源
3. 缓冲区大小需平衡实时性和处理效率
4. 考虑使用GPU加速模型推理(如果可用)

### v1.0 KWS处理逻辑

`kws_processing_thread` 按照以下逻辑工作:

1. 从音频队列读取数据，并将其连接到缓冲区
2. 定义语音起点：当平均音量值高于阈值时
3. 定义语音终点：当静音时间超过预设阈值(NO_VOICE_THRESHOLD = 2s)时
4. 使用活动窗口(起点到终点)捕获有效语音数据，发送给ASR模型
5. 将所有识别文本添加到`self.detected_keywords`
6. 当在识别文本中发现关键词时，设置`self.keyword_detected = True`
7. 当以下任一条件满足时，重置关键词检测状态(`self.keyword_detected = False`):
   - 静音时间超过5秒(END_CONV_THRESHOLD = 5)
   - 检测到终止关键词("bye", "再见", "ok")
8. 应用自适应窗口大小，在保证识别准确率的同时优化实时性能

### v1.0 已识别的改进方向

1. 支持自定义关键词列表
2. 添加关键词置信度指示
3. 实现录音文件保存功能
4. 支持更多可视化选项
5. 添加降噪预处理

---

## v2.0 优化方案（2026-03-08）

### 项目目标

优化现有VAD/KWS系统，用于**评估不同wireless麦克风的效果**，重点提升：
- 音频质量（通过降噪预处理）
- 唤醒机制（通过轻量级KWS避免持续ASR调用）
- 用户体验（可视化性能、可调参数、手动控制）
- 可扩展性（为声纹识别预留接口）

### v2.0 核心改进概览

| 改进项 | v1.0 | v2.0 |
|--------|------|------|
| 降噪处理 | 无 | webrtcvad + noisereduce |
| KWS机制 | FunASR ASR（重量级） | Vosk轻量级KWS + FunASR ASR |
| 状态管理 | 持续运行 | SLEEPING/AWAKE/LISTENING三状态机 |
| 暂停检测 | 固定1.5s | 可调500ms-2000ms |
| 可视化 | 3个subplot，~10fps | 4个subplot + blitting，30fps+ |
| 手动录音 | 空格键 | 保留并增强（状态指示） |
| 声纹识别 | 无 | 预留接口 |

### 1. 音频处理流水线重构

**新的数据流**：
```
麦克风 → PyAudio → 降噪处理 → 增强VAD → Vosk KWS唤醒 → FunASR ASR → 结果展示
                    ↓                                    ↓
               原始音频buffer                      降噪音频buffer
                    ↓                                    ↓
            可视化subplot 1                      可视化subplot 2
```

**关键变更**：
- 在`audio_callback()`中添加实时降噪处理
- 分离轻量级KWS（Vosk）和重量级ASR（FunASR）
- 同时保存原始和降噪后的音频用于对比展示

### 2. 降噪集成（webrtcvad + noisereduce）

**新增模块**：`audio_denoiser.py`

**核心功能**：
- **webrtcvad前置过滤**：静音帧跳过降噪处理（减少计算开销）
- **noisereduce降噪**：使用stationary模式处理实时音频流
- **三档强度控制**：
  - 弱档（prop_decrease=0.5）：保留更多原始音色，适合高质量麦克风
  - 中档（prop_decrease=0.8）：平衡，默认设置
  - 强档（prop_decrease=1.2）：最大降噪，适合嘈杂环境
- **滑动窗口**：维护500ms音频历史用于噪声估计
- **性能监控**：输出降噪前后RMS值，用于量化降噪效果

### 2.1 Facebook Denoiser深度学习降噪集成（2026-03-08更新）

**优化背景**：
传统spectral gating算法（noisereduce）存在本质局限：
- ❌ **问题**：通过频域噪声估计，均匀降低所有频率分量（包括语音）
- ❌ **表现**：降噪后语音和噪声同时衰减，SNR改善有限
- ❌ **用户反馈**："只是把声音赋值降低，我希望只针对语音以外的音频降低，不要降低人说话的声音"

**解决方案**：替换为Facebook Denoiser深度学习模型

**模型规格**：
- **架构**：Demucs U-Net（时域语音分离网络）
- **训练集**：Microsoft DNS Challenge（clean speech + noise pairs）
- **模型**：DNS64 checkpoint（128MB）
- **设备**：优先使用CUDA GPU，回退到CPU
- **输入格式**：16kHz单声道，float32张量[batch, channels, samples]

**性能对比**：

| 指标 | noisereduce (传统) | Facebook Denoiser (AI) |
|------|-------------------|------------------------|
| 降噪强度 | -13.1 dB | -24.5 dB |
| 语音保留 | ❌ 衰减（均匀降低） | ✅ 保留（智能分离） |
| 噪声抑制 | RMS 0.052 | RMS 0.014 |
| 处理延迟 | ~20ms | ~30ms (CPU) / ~5ms (GPU) |
| 计算资源 | CPU轻量 | GPU显存+100MB (可选) |
| SNR提升 | 有限 | 显著（真实语音/噪声比提升） |

**实现细节**：
```python
# audio_denoiser.py 关键代码
class AudioDenoiser:
    def __init__(self, denoiser_type='facebook'):
        if denoiser_type == 'facebook':
            self.facebook_model = pretrained.dns64()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.facebook_model.to(self.device)
            self.facebook_model.eval()
    
    def denoise_frame(self, audio_frame):
        # numpy [samples] → torch [1, 1, samples]
        audio_tensor = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            denoised = self.facebook_model(audio_tensor)
        
        # torch [1, 1, samples] → numpy [samples]
        return denoised.squeeze().cpu().numpy()
```

**可视化效果**：
- **ax1 (上图)**：灰色原始音频波形（保留语音+噪声）
- **ax2 (中图)**：蓝色降噪后波形（语音峰值保留，基线噪声降低） + 红色VAD检测线
- **ax3 (下图)**：关键词检测时间线
- **对比观察**：
  - 语音段：蓝色峰值幅度≈灰色峰值幅度（语音保留）
  - 静音段：蓝色基线<<灰色基线（噪声抑制）
  - Y轴范围：统一±0.2，刻度0.1，便于精确对比高质量麦克风

**依赖冲突解决**：
- **问题**：denoiser需要`hydra-core<1.0`，FunASR需要`hydra-core>=1.3.2`
- **解决**：强制升级到hydra-core 1.3.2（FunASR优先）
- **验证**：denoiser仍正常工作，hydra仅用于命令行参数解析（不影响核心推理）
- **风险**：denoiser的hydra功能（CLI）不可用，但Python API正常

**使用建议**：
- ✅ **推荐场景**：需要高SNR的语音识别前处理
- ✅ **推荐设备**：有NVIDIA GPU（RTX 2060+）
- ⚠️ **注意**：CPU模式下延迟30-50ms（可接受，但不如GPU实时）
- ⚠️ **备选方案**：如需极低延迟且无GPU，可降级回noisereduce

### 2.2 Silero VAD语音活动检测集成（2026-03-08更新）

**优化背景**：
webrtcvad在Windows上安装困难（需要C编译器），且准确率有限。

**解决方案**：采用Silero VAD（PyTorch-based深度学习模型）

**模型规格**：
- **架构**：TorchScript compiled模型（JIT优化）
- **模型大小**：~1.5MB（极轻量）
- **输入格式**：16kHz单声道，512 samples（32ms帧）
- **输出**：语音概率值 0.0-1.0（可灵活设置阈值）
- **设备**：CPU/GPU均支持，自动选择

**性能对比**：

| 指标 | webrtcvad | Silero VAD | FunASR VAD |
|------|-----------|------------|------------|
| 安装难度 | ❌ 需C编译器 | ✅ torch.hub自动 | ✅ pip安装 |
| 模型大小 | Built-in | ~1.5MB | ~200MB |
| 处理延迟 | <10ms | ~15-20ms (CPU) | ~50-100ms |
| 准确率 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 输出类型 | Boolean | Probability | Timestamps |
| 语言支持 | Language-free | Language-free | Chinese-optimized |
| 用途 | Pre-filter | Real-time VAD | Segmentation |

**实现细节**：
```python
# audio_denoiser.py 关键代码
import torch

# 加载Silero VAD（自动从torch.hub下载）
silero_model, silero_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

class AudioDenoiser:
    def __init__(self, vad_type='auto'):
        # auto模式：优先使用Silero，回退到webrtcvad
        if vad_type == 'auto' and SILERO_VAD_AVAILABLE:
            self.silero_vad = silero_model
            self.silero_threshold = 0.5  # 可调阈值
    
    def _is_speech(self, audio_frame):
        # Silero VAD需要512 samples @ 16kHz
        required_samples = 512
        audio_padded = np.pad(audio_frame, (0, required_samples - len(audio_frame)))
        
        audio_tensor = torch.from_numpy(audio_padded[:required_samples]).float()
        with torch.no_grad():
            speech_prob = self.silero_vad(audio_tensor, 16000).item()
        
        return speech_prob > self.silero_threshold  # True/False
```

**集成架构**：
```
Audio Frame (任意长度) → 
    ↓
Silero VAD (_is_speech) → 
    ├─ prob < 0.5: Skip denoising (silence) 
    └─ prob ≥ 0.5: Apply denoising (speech)
        ↓
    Facebook Denoiser → 
        ↓
    Enhanced Audio
```

**优势**：
- ✅ **易于安装**：无需C编译器，torch.hub自动下载
- ✅ **高准确率**：深度学习模型，优于传统webrtcvad
- ✅ **灵活阈值**：输出概率值，可根据场景调整（0.3-0.7）
- ✅ **低延迟**：CPU模式15-20ms，GPU模式<5ms
- ✅ **小模型**：1.5MB，不影响总体资源占用

**使用建议**：
- ✅ **默认选择**：`vad_type='auto'`自动使用Silero VAD
- ✅ **调整阈值**：安静环境0.3，嘈杂环境0.7
- ✅ **与FunASR协同**：Silero快速过滤 + FunASR精确分割
- ⚠️ **注意**：首次运行会从GitHub下载模型（~1.5MB，约3-5秒）

### 2.3 Force Denoise All Frames功能（2026-03-08更新）

**问题背景**：
VAD-降噪协同工作时发现的冲突：
- **初始实现**：降噪模块内部使用Silero VAD预过滤（`skip_vad=False`）
  - 逻辑：speech_prob < 0.5 → 跳过降噪，直接返回原始帧
  - 目的：节省计算资源（静音帧无需降噪）
- **问题表现**：用户截图显示蓝色（降噪）和灰色（原始）波形几乎完全重合
  - 原因：静音段被VAD过滤，未经降噪处理，导致静音噪声未被抑制
  - 影响：降噪效果不明显，麦克风评估不准确

**解决方案**：
添加UI控制选项"Force Denoise All Frames (强制所有帧降噪)"复选框

**实现细节**：

1. **UI控制**（Audio_test_UI.py）：
```python
# Lines 178-191: Force Denoise checkbox
self.force_denoise_checkbox = wx.CheckBox(
    self.panel,
    label="Force Denoise All Frames (强制所有帧降噪)"
)
self.force_denoise_checkbox.SetValue(True)  # 默认勾选
self.force_denoise_checkbox.Bind(wx.EVT_CHECKBOX, self.on_force_denoise_changed)

# 提示文本
✓ Checked: Denoise all frames (including silence) for maximum effect
✗ Unchecked: Only denoise frames detected as speech (save computation)
```

2. **后端参数**（FunASR_VAD_KWS_plot.py）：
```python
# Line 100-107: __init__ 签名
def __init__(self, force_denoise_all_frames=True, ...):
    self.force_denoise_all_frames = force_denoise_all_frames

# Lines 714-719: 动态控制skip_vad参数
audio_denoised, stats = self.audio_denoiser.denoise_frame(
    audio_data_norm,
    skip_vad=self.force_denoise_all_frames  # True=强制降噪所有帧
)

# Lines 298-321: 动态更新方法
def set_force_denoise_all_frames(self, force_denoise: bool):
    self.force_denoise_all_frames = force_denoise
```

3. **降噪逻辑**（audio_denoiser.py）：
```python
def denoise_frame(self, audio_frame, skip_vad=False):
    if not skip_vad:  # False时启用VAD过滤
        is_speech = self._is_speech(audio_frame)
        if not is_speech:
            return audio_frame, stats  # 静音帧跳过降噪
    
    # skip_vad=True时，所有帧都执行降噪
    denoised = self.facebook_model(audio_tensor)
    return denoised, stats
```

**行为对比**：

| 模式 | 语音段处理 | 静音段处理 | 计算开销 | 可视化效果 |
|------|-----------|-----------|---------|----------|
| ✓ Checked | 降噪 | 降噪 | 100% | 蓝色基线明显低于灰色 |
| ✗ Unchecked | 降噪 | 跳过（返回原始） | ~50-70% | 语音段降噪，静音段重合 |

**推荐设置**：
- ✅ **麦克风对比测试**：勾选（便于观察静音噪声差异）
- ✅ **演示降噪效果**：勾选（最大化可视化对比）
- ⚪ **实时语音识别**：不勾选（节省GPU资源，静音段无需降噪）
- ⚪ **低功耗场景**：不勾选（减少约30-50%计算量）

**技术意义**：
这个功能解决了"工具效果"vs"实际应用"的权衡：
- 评估麦克风时需要全帧降噪（包括静音噪声）
- 实际应用时只需语音段降噪（节省资源）

### 3. Vosk轻量级KWS唤醒

**设计目标**：
- 待机状态仅运行Vosk（计算量<FunASR的1/10）
- 检测到唤醒词后才启动FunASR深度识别
- 降低GPU占用，减少功耗

**新增模块**：`vosk_kws_engine.py`

**模型规格**：
- 模型：vosk-model-small-cn-0.22
- 大小：~42MB（相比FunASR SenseVoiceSmall的~500MB）
- 延迟：<100ms（相比FunASR的200-500ms）
- 准确率：grammar模式下关键词识别率>95%

### 4. 三状态机设计

**状态定义**：

| 状态 | 触发条件 | 工作模式 | 指示器颜色 |
|------|---------|---------|-----------|
| **SLEEPING** | 初始状态/超时无声音 | 仅运行Vosk KWS | 灰色 ⚫ |
| **AWAKE** | Vosk检测到关键词 | Vosk + FunASR ASR | 绿色 🟢 |
| **LISTENING** | 用户按住空格键 | Vosk + FunASR ASR + 强制录音 | 红色 🔴 |

**状态转换图**：
```
         启动
          ↓
    ┌──SLEEPING──┐
    │  (灰色)     │
    │  Vosk监听  │
    └─────────────┘
          ↓ 检测到关键词
          ↓
    ┌───AWAKE────┐
    │  (绿色)     │ ←──────┐
    │  FunASR识别│        │ 持续有声音
    └─────────────┘        │
          ↓ 静音2-5秒        │
          └────────────────┘
          ↓ 超时
    ┌──SLEEPING──┐
    
    任意状态 + 按下空格键 → LISTENING (红色)
    LISTENING + 释放空格键 → SLEEPING/AWAKE
```

### 5. 可调暂停检测

**需求背景**：不同说话风格需要不同的停顿检测阈值
- 快速对话：500ms停顿即触发识别
- 思考型说话：2000ms停顿才触发

**实现方案**：
- 原代码`PAUSE_VOICE_THRESHOLD = 1.5`（固定）
- 改为`self.pause_threshold`（可调）
- UI添加滑块控件：500ms - 2000ms

### 6. 可视化增强

**当前subplot布局**（3个子图）：
```
┌─────────────────────────────────────────────────┐
│ Subplot 1 (ax1): 原始音频波形                   │
│ └─ 灰色波形：原始麦克风输入                     │
│ └─ Y轴：±0.2，刻度0.1（精细观察）               │
├─────────────────────────────────────────────────┤
│ Subplot 2 (ax2): 降噪后音频波形 + VAD标记      │
│ └─ 蓝色波形：Facebook Denoiser降噪结果          │
│ └─ 红色线：FunASR VAD检测语音段                │
│ └─ Y轴：±0.2，刻度0.1（与ax1对齐）              │
├─────────────────────────────────────────────────┤
│ Subplot 3 (ax3): 关键词检测时间线               │
│ └─ 黄色高亮块：检测到的关键词及识别文本         │
│ └─ X轴：时间戳（秒）                            │
└─────────────────────────────────────────────────┘
```

**Y轴调整设计**（2026-03-08更新）：
- **v1.0**: Y轴范围±0.5，适合低质量麦克风（噪声大）
- **v2.0初期**: ±0.3，通用设置
- **v2.0当前**: ±0.2，适合高质量麦克风评估（放大细节）
- **刻度间隔**: 0.1（便于精确读数）
- **应用位置**: 
  - `ax1.set_ylim(-0.2, 0.2)` (Line 791)
  - `ax2.set_ylim(-0.2, 0.2)` (Line 797)
  - 初始化图表 (Lines 1008, 1067)
  - Y刻度数组：`np.arange(-0.2, 0.21, 0.1)` (Lines 1010, 1069)

**matplotlib blitting优化**：
- 使用`animated=True` + `blit=True`的FuncAnimation
- 仅更新数据，不重绘静态元素
- 预期性能提升：从~10fps提升到30-40fps

### 7. UI控制面板扩展

**新增控件**：
- **降噪强度**：RadioButton（⚫ 弱 ⚫ 中 ● 强）
  - 弱：prop_decrease=0.5，保留更多原始音色
  - 中：prop_decrease=0.8，默认平衡设置
  - 强：prop_decrease=1.2，最大降噪（嘈杂环境）
- **Force Denoise All Frames**：CheckBox（默认勾选）
  - ✓ 勾选：降噪所有帧（包括静音），便于麦克风对比
  - ✗ 不勾选：仅降噪语音帧，节省50-70%计算资源
- **暂停检测**：Slider（500-2000ms）
  - 快速对话：500ms短停顿触发识别
  - 思考型说话：2000ms长停顿才触发
- **状态显示**：StatusBar
  - 当前状态：SLEEPING/AWAKE/LISTENING
  - 降噪效果：RMS降低百分比
  - 处理延迟：实时延迟监控

### 8. 声纹识别接口预留

**新增模块**：`speaker_recognition.py`

**接口定义**：
- `enroll_speaker(audio, speaker_id)`: 注册说话人
- `verify_speaker(audio)`: 验证说话人身份
- `delete_speaker(speaker_id)`: 删除已注册的说话人

**推荐库**（未来实现）：
- resemblyzer：开源，基于GE2E
- pyannote.audio：更强大，支持speaker diarization
- speechbrain：端到端toolkit

### v2.0 技术栈更新

**新增依赖**（requirements.txt）：
```txt
# === v2.0新增 ===
noisereduce>=3.0.0      # 音频降噪（传统spectral gating，已废弃）
webrtcvad>=2.0.10       # VAD增强（可选，Windows安装困难，已被Silero VAD替代）
vosk>=0.3.45            # 轻量级KWS

# === v2.0.1新增（2026-03-08）===
denoiser>=0.1.5         # Facebook深度学习降噪（DNS64模型）
torch>=2.0.0            # PyTorch（Facebook Denoiser依赖，已有）
torchaudio>=2.0.0       # 音频处理（已有）
hydra-core>=1.3.2       # FunASR配置管理（强制升级以兼容）
omegaconf>=2.3.0        # 配置框架（已有）

# === v2.0.2新增（2026-03-08）===
# Silero VAD - 通过torch.hub自动加载，无需单独安装
# 优势：
# - 无需C编译器（纯PyTorch实现）
# - 准确率高于webrtcvad
# - 模型大小仅~1.5MB
# - 延迟~15-20ms
# - 输出概率值（0.0-1.0）可灵活调整阈值
# 使用：自动作为webrtcvad的替代方案

# === v2.0.3新增（2026-03-08）===
# Force Denoise All Frames功能
# UI: Audio_test_UI.py - CheckBox控件
# Backend: FunASR_VAD_KWS_plot.py - force_denoise_all_frames参数
# Logic: audio_denoiser.py - skip_vad参数控制
# 用途：解决VAD-降噪冲突，支持全帧降噪或按需降噪
```

**模型文件**：
```
models/
├── damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/  # 现有FunASR VAD
├── iic/SenseVoiceSmall/                            # 现有FunASR ASR
└── vosk_models/vosk-model-small-cn-0.22/           # 新增Vosk KWS
```

### v2.0 文件结构

```
vad KWS_ok/
├── Audio_test_UI.py                    # [修改] 主UI，添加降噪/暂停控制
├── FunASR_VAD_KWS_plot.py             # [修改] 核心引擎，集成降噪/Vosk/状态机
├── audio_denoiser.py                   # [新建] 降噪模块
├── vosk_kws_engine.py                  # [新建] Vosk KWS包装器
├── download_vosk_model.py              # [新建] 模型下载脚本
├── speaker_recognition.py              # [新建] 声纹识别接口（占位）
├── requirements.txt                    # [修改] 添加新依赖
├── DRS.md                              # [修改] 本文档
└── models/vosk_models/                 # [新建] Vosk模型目录
```

### v2.0 性能指标

| 指标 | v1.0 | v2.0 目标 |
|------|------|----------|
| 待机CPU占用 | 15-20%（持续FunASR） | 5-8%（仅Vosk） |
| 唤醒延迟 | N/A | <150ms（Vosk检测） |
| ASR处理延迟 | 200-500ms | 保持200-500ms |
| 可视化帧率 | ~10fps | 30fps+ |
| GPU显存占用 | ~800MB（FunASR常驻） | ~200MB（按需加载） |
| 降噪处理延迟 | N/A | <50ms（实时） |

### v2.0 实施计划

#### Phase 1: 环境准备 ✅
- [x] 更新requirements.txt
- [x] 下载Vosk模型
- [x] 验证依赖安装

#### Phase 2: 核心模块开发 ✅
- [x] 实现audio_denoiser.py（Facebook Denoiser + Silero VAD）
- [x] 实现vosk_kws_engine.py
- [x] 创建speaker_recognition.py占位

#### Phase 3: 集成修改 ✅
- [x] 修改FunASR_VAD_KWS_plot.py（降噪+状态机+Force Denoise参数）
- [x] 修改Audio_test_UI.py（UI控件+Force Denoise复选框）
- [x] Y轴范围优化（±0.2精细观察）

#### Phase 4: 测试验证 🔄
- [x] 降噪效果测试（Facebook Denoiser: -24.5dB）
- [x] Silero VAD准确率测试
- [x] Force Denoise All Frames功能测试
- [ ] 状态机切换测试
- [ ] 多麦克风对比测试

**当前状态**（2026-03-08）：
- ✅ Facebook Denoiser集成完成，降噪效果显著（-24.5 dB）
- ✅ Silero VAD替换webrtcvad完成，无需C编译器
- ✅ Force Denoise All Frames复选框实现，解决VAD-降噪冲突
- ✅ Y轴调整至±0.2，适合高质量麦克风评估
- ✅ 可视化系统已优化为3-subplot布局
- 🔄 持续优化中：状态机逻辑、多麦克风对比流程

### v2.0 已知限制

1. **Vosk模型语言限制**：当前使用中文模型，英文关键词识别率可能下降
2. **降噪实时性**：强档可能引入40-60ms延迟
3. **GPU显存**：FunASR仍需~600MB显存
4. **matplotlib性能瓶颈**：4个subplot仍有开销

### v2.0 后续扩展路线图

#### 短期（1-2个月）
- [ ] 声纹识别集成（resemblyzer）
- [ ] 多麦克风自动化测试脚本
- [ ] 导出性能对比报告（PDF/CSV）
- [ ] 支持双语KWS（中英文关键词并行）

#### 中期（3-6个月）
- [ ] 实时频谱图可视化
- [ ] 云端模型API支持
- [ ] 多用户管理（声纹库）
- [ ] 关键词自定义GUI

#### 长期（6-12个月）
- [ ] Web界面版本（FastAPI + React）
- [ ] 移动端支持
- [ ] 多麦克风阵列支持（beamforming）
- [ ] 实时翻译集成

---

## 附录

### A. 参考文献

1. **FunASR官方文档**: https://github.com/alibaba-damo-academy/FunASR
2. **Vosk API Documentation**: https://alphacephei.com/vosk/
3. **webrtcvad GitHub**: https://github.com/wiseman/py-webrtcvad
4. **noisereduce Documentation**: https://timsainburg.com/noise-reduction-python.html
5. **Matplotlib Blitting Tutorial**: https://matplotlib.org/stable/tutorials/advanced/blitting.html

### B. 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| VAD | Voice Activity Detection | 语音活动检测，区分语音和静音 |
| KWS | Keyword Spotting | 关键词唤醒/检测 |
| ASR | Automatic Speech Recognition | 自动语音识别 |
| WER | Word Error Rate | 词错误率，ASR准确率指标 |
| RMS | Root Mean Square | 均方根，音频能量指标 |
| FPS | Frames Per Second | 帧率，可视化性能指标 |
| Blitting | - | Matplotlib优化技术，仅重绘变化区域 |

### C. 联系方式

- **作者**: jiheng.zhang@gehealthcare.com
- **SSO**: 212597558
- **项目路径**: `c:\Users\212597558\AI_Tools\vad KWS_ok`
- **最后更新**: 2026-03-08

---

**文档结束**