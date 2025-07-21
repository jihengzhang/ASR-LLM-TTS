#!/usr/bin/env python3
"""
专门针对 generate() 缺少 input 参数错误的测试
"""

import os
import numpy as np

# 设置环境变量
current_dir = os.getcwd()
os.environ['MODELSCOPE_CACHE'] = current_dir

print("正在导入 FunASR...")
from funasr import AutoModel

print("正在加载 VAD 模型...")
model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
print("✓ 模型加载成功")

# 创建测试音频数据
print("创建测试音频数据...")
sample_rate = 16000
duration = 2.0  # 2秒
samples = int(sample_rate * duration)

# 测试不同输入格式
print("\n测试 1: numpy float32 数组")
try:
    audio_float = np.random.normal(0, 0.1, samples).astype(np.float32)
    result = model.generate(input=audio_float)
    print(f"✓ 成功! 结果类型: {type(result)}")
    if isinstance(result, dict):
        print(f"  键: {list(result.keys())}")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试 2: numpy int16 数组")
try:
    audio_int16 = (np.random.normal(0, 0.1, samples) * 32767).astype(np.int16)
    result = model.generate(input=audio_int16)
    print(f"✓ 成功! 结果类型: {type(result)}")
    if isinstance(result, dict):
        print(f"  键: {list(result.keys())}")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试 3: 音频文件路径")
try:
    # 先创建一个测试WAV文件
    import wave
    test_wav = "test_audio.wav"
    with wave.open(test_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    result = model.generate(input=test_wav)
    print(f"✓ 成功! 结果类型: {type(result)}")
    if isinstance(result, dict):
        print(f"  键: {list(result.keys())}")
        
    # 清理
    os.remove(test_wav)
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试完成！")
