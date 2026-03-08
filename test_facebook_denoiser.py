#!/usr/bin/env python3
"""
测试 Facebook Denoiser 的效果
"""
import torch
import torchaudio
import numpy as np
from denoiser import pretrained
from denoiser.dsp import convert_audio

def test_denoiser():
    print("加载 Facebook Denoiser 模型...")
    try:
        # 加载预训练模型 (DNS64 - 最好的模型)
        model = pretrained.dns64()
        model.eval()
        print("✅ 模型加载成功！")
        print(f"   模型类型: {type(model)}")
        print(f"   设备: {next(model.parameters()).device}")
        
        # 测试音频处理
        print("\n测试音频处理...")
        # 创建测试音频 (16kHz, 1秒)
        sample_rate = 16000
        duration = 1.0
        test_audio = torch.randn(1, 1, int(sample_rate * duration))
        
        print(f"   输入形状: {test_audio.shape}")
        print(f"   输入范围: [{test_audio.min():.3f}, {test_audio.max():.3f}]")
        
        # 降噪
        with torch.no_grad():
            enhanced = model(test_audio)
        
        print(f"   输出形状: {enhanced.shape}")
        print(f"   输出范围: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
        print("✅ 音频处理成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_denoiser()
    if success:
        print("\n🎉 Facebook Denoiser 可以正常工作！")
        print("\n下一步:")
        print("1. 可以集成到 audio_denoiser.py")
        print("2. 模型会自动下载（~40MB）")
        print("3. 实时处理需要GPU（CPU也可以但较慢）")
    else:
        print("\n⚠️ 需要解决依赖冲突")
