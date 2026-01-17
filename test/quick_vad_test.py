#!/usr/bin/env python3
"""
快速VAD测试 - 使用示例音频文件
"""

import os

# 设置缓存目录
current_dir = os.getcwd()
os.environ['MODELSCOPE_CACHE'] = current_dir

print("正在导入 FunASR...")
from funasr import AutoModel

print("正在加载 VAD 模型...")
model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
print("✓ 模型加载成功")

# 检查模型是否有示例音频
model_path = getattr(model, 'model_path', None)
if model_path:
    print(f"模型路径: {model_path}")
    example_path = os.path.join(model_path, "example")
    if os.path.exists(example_path):
        print(f"示例目录: {example_path}")
        for file in os.listdir(example_path):
            if file.endswith('.wav'):
                wav_file = os.path.join(example_path, file)
                print(f"找到示例音频: {wav_file}")
                
                print("运行VAD测试...")
                try:
                    result = model.generate(input=wav_file)
                    print(f"✓ VAD成功! 结果: {result}")
                    break
                except Exception as e:
                    print(f"✗ VAD失败: {e}")
    else:
        print("未找到示例目录")
else:
    print("无法获取模型路径")

print("测试完成！")
