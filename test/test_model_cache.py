#!/usr/bin/env python3
import os
import sys

# 设置缓存目录到当前目录
cache_dir = os.getcwd()
print(f"设置缓存目录为当前目录: {cache_dir}")

# 设置多个可能的环境变量
os.environ['MODELSCOPE_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

print("已设置环境变量:")
for var in ['MODELSCOPE_CACHE', 'HF_HOME', 'TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE']:
    print(f"  {var}: {os.environ.get(var, 'Not set')}")

try:
    print("\n开始下载模型...")
    from funasr import AutoModel
    model = AutoModel(model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch', model_revision='v2.0.4')
    print("✓ 模型下载/加载成功")
    
    # 检查当前目录下是否有模型文件
    print(f"\n当前目录内容:")
    for item in os.listdir(cache_dir):
        if os.path.isdir(item):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    # 查找可能的模型目录
    model_dirs = []
    for item in os.listdir(cache_dir):
        if os.path.isdir(item) and ('model' in item.lower() or 'hub' in item.lower() or 'cache' in item.lower()):
            model_dirs.append(item)
    
    if model_dirs:
        print(f"\n找到可能的模型目录:")
        for dir_name in model_dirs:
            print(f"  📁 {dir_name}/")
            dir_path = os.path.join(cache_dir, dir_name)
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    if root != dir_path:  # 只显示子目录
                        rel_path = os.path.relpath(root, dir_path)
                        print(f"    📁 {rel_path}/")
                    if len(dirs) > 3:  # 如果子目录太多，只显示前几个
                        break
    
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)
