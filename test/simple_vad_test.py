#!/usr/bin/env python3
"""
简化的 FunASR VAD 测试脚本
用于测试正确的 API 调用方式
"""

import os
import numpy as np

# 设置缓存目录到当前目录
current_dir = os.getcwd()
os.environ['MODELSCOPE_CACHE'] = current_dir
os.environ['HF_HOME'] = current_dir
os.environ['TRANSFORMERS_CACHE'] = current_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = current_dir

try:
    from funasr import AutoModel
    print("[成功] FunASR 模块已加载")
except ImportError:
    print("[错误] FunASR 模块未找到")
    exit(1)

def test_api():
    """测试不同的API调用方式"""
    print("\n正在加载 VAD 模型...")
    
    try:
        # 加载模型
        model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
        print("[成功] VAD 模型加载成功")
        
        # 创建测试音频数据 (1秒静音，16kHz)
        sample_rate = 16000
        duration = 1.0  # 1秒
        samples = int(sample_rate * duration)
        
        # 测试不同的输入格式
        test_cases = [
            ("numpy float32 array", np.zeros(samples, dtype=np.float32)),
            ("numpy int16 array", np.zeros(samples, dtype=np.int16)),
            ("bytes data", np.zeros(samples, dtype=np.int16).tobytes()),
        ]
        
        for name, test_data in test_cases:
            print(f"\n测试 {name}:")
            try:
                result = model.generate(input=test_data)
                print(f"  [成功] 返回类型: {type(result)}")
                if isinstance(result, dict):
                    print(f"  [成功] 返回键: {list(result.keys())}")
                    if 'vad_info' in result:
                        print(f"  [成功] VAD 信息: {result['vad_info']}")
                else:
                    print(f"  [信息] 返回值: {result}")
                break  # 如果成功，就使用这种格式
            except Exception as e:
                print(f"  [失败] 错误: {str(e)}")
        
        print("\n[总结] API 测试完成")
        
    except Exception as e:
        print(f"[错误] 模型加载失败: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("==================================================")
    print("    简化的 FunASR VAD API 测试")
    print("==================================================")
    
    test_api()
