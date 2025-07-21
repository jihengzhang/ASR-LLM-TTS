#!/usr/bin/env python3
"""
最简单的 FunASR 测试
"""

import os

# 设置缓存目录
current_dir = os.getcwd()
os.environ['MODELSCOPE_CACHE'] = current_dir

print("正在导入 FunASR...")
try:
    from funasr import AutoModel
    print("✓ FunASR 导入成功")
except Exception as e:
    print(f"✗ FunASR 导入失败: {e}")
    exit(1)

print("正在加载模型...")
try:
    model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
    print("✓ 模型加载成功")
    print(f"模型类型: {type(model)}")
    
    # 检查模型的方法
    methods = [attr for attr in dir(model) if not attr.startswith('_')]
    print(f"可用方法: {methods[:10]}...")  # 只显示前10个
    
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
