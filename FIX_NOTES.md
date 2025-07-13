# FunASR VAD 错误修复说明

## 问题描述
用户遇到错误：`AutoModel.generate() missing 1 required positional argument: 'input'`

## 原因分析
1. **API变化**: FunASR 1.2.6版本的API有所更新，`generate()`方法的参数格式发生了变化
2. **参数格式错误**: 原代码使用了`audio_in`参数，但新版本需要使用`input`参数
3. **输入格式问题**: VAD模型期望特定的输入格式

## 修复内容

### 1. test_funasr_vad.py 修复
- ✅ 将`audio_in`参数改为`input`参数
- ✅ 移除了过时的参数：`audio_fs`, `mode`, `vad_info`
- ✅ 将音频数据保存为临时WAV文件，因为VAD模型期望文件路径输入
- ✅ 更新了结果解析格式，适配新的返回格式

### 2. FunASR_VAD_demo.py 修复
- ✅ 更新了实时VAD处理方式
- ✅ 使用流式VAD处理，支持chunk_size参数
- ✅ 添加了VAD缓存管理
- ✅ 修复了结果解析逻辑

### 3. 新增测试脚本
- ✅ `simple_vad_test.py`: 简化的API测试
- ✅ `minimal_test.py`: 最小化模型加载测试
- ✅ `debug_generate_input.py`: 针对input参数的调试测试
- ✅ `quick_vad_test.py`: 使用示例音频的快速测试

## 正确的API用法

### 基本VAD用法（文件输入）
```python
from funasr import AutoModel

model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
result = model.generate(input="audio_file.wav")
print(result)
```

### 流式VAD用法（实时音频）
```python
from funasr import AutoModel
import numpy as np

model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
cache = {}

# 处理音频块
audio_chunk = np.array([...], dtype=np.float32)
result = model.generate(
    input=audio_chunk, 
    cache=cache, 
    is_final=False,
    chunk_size=200  # 200ms
)
```

### 返回结果格式
```python
# VAD结果格式：
[
    {
        'value': [[start_ms, end_ms], [start_ms, end_ms], ...]
    }
]

# 其中 start_ms, end_ms 是以毫秒为单位的时间戳
```

## 环境配置
确保正确设置模型缓存目录：
```python
import os
current_dir = os.getcwd()
os.environ['MODELSCOPE_CACHE'] = current_dir
os.environ['HF_HOME'] = current_dir
os.environ['TRANSFORMERS_CACHE'] = current_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = current_dir
```

## 测试验证
运行以下命令验证修复：
```bash
python test_funasr_vad.py      # 完整的录音+VAD测试
python quick_vad_test.py       # 快速模型测试
python FunASR_VAD_demo.py      # 实时录音系统
```

## 注意事项
1. FunASR 1.2.6版本需要使用`input`参数而不是`audio_in`
2. VAD模型在非流式模式下期望文件路径作为输入
3. 流式模式需要使用cache来保持状态
4. 返回结果的格式已经更新，需要相应调整解析代码
5. 确保音频采样率为16kHz以获得最佳效果
