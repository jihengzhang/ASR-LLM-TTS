"""
FunASR VAD 测试脚本
用于测试 FunASR 的 VAD 模型是否能够正常工作
"""

import os
import time
import numpy as np
import wave
import pyaudio
import subprocess
import sys
from datetime import datetime

# 检查 ffmpeg 是否安装
def check_ffmpeg():
    try:
        # 尝试运行 ffmpeg -version 命令
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("[成功] ffmpeg 已安装")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[错误] ffmpeg 未安装！FunASR 可能需要 ffmpeg 来处理音频")
        print("请运行 install_funasr.bat 安装 ffmpeg")
        return False

# 检查 ffmpeg
if not check_ffmpeg():
    print("按任意键退出...")
    input()
    sys.exit(1)

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print("[成功] FunASR 模块已加载")
except ImportError:
    FUNASR_AVAILABLE = False
    print("[警告] FunASR 模块未找到，请先安装 FunASR")
    print("提示: 运行 install_funasr.bat 安装")
    sys.exit(1)

# 配置参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

def test_vad_model():
    """测试 VAD 模型加载和推理"""
    print("\n正在测试 FunASR VAD 模型...")
    
    try:
        # 检查本地模型
        local_model_path = os.path.join(os.getcwd(), "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
        
        print("正在加载 VAD 模型...")
        if os.path.exists(local_model_path):
            print(f"[信息] 使用本地模型: {local_model_path}")
            vad_model = AutoModel(
                model=local_model_path,
                disable_update=True,
                device="cpu"
            )
        else:
            print("[信息] 未找到本地模型，尝试远程模型...")
            vad_model = AutoModel(
                model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
                model_revision="v2.0.4",
                disable_update=True,
                device="cpu"
            )
        
        print("[成功] VAD 模型加载成功")
        
        # 录制短音频进行测试
        print("\n录制 5 秒音频进行测试...")
        print("请说话...")
        
        # 初始化 PyAudio
        p = pyaudio.PyAudio()
        
        # 打开音频流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        # 录制音频
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            # 显示进度
            progress = (i + 1) / int(RATE / CHUNK * RECORD_SECONDS) * 100
            print(f"录音进度: {progress:.1f}%", end='\r')
        
        print("\n录音完成，正在处理...")
        
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 将帧组合成音频数据
        audio_data = b''.join(frames)
        
        # 保存测试音频到临时文件
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        temp_wav = os.path.join("temp", "temp_test.wav")
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16bit = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        # 执行 VAD 推理
        print("执行 VAD 推理...")
        result = vad_model.generate(input=temp_wav)
        
        # 打印结果
        if isinstance(result, list) and len(result) > 0:
            vad_info = result[0].get('value', [])
        else:
            vad_info = []
        
        if len(vad_info) > 0:
            print("\n[成功] VAD 检测结果:")
            for i, segment in enumerate(vad_info):
                start = segment[0] / 1000.0  # 转换为秒
                end = segment[1] / 1000.0    # 转换为秒
                duration = end - start
                print(f"  语音段 {i+1}: 开始={start:.2f}s, 结束={end:.2f}s, 持续={duration:.2f}s")
        else:
            print("\n[警告] 未检测到语音，请尝试在更安静的环境中测试")
        
        # 清理临时文件
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        # 保存测试音频
        if not os.path.exists("test"):
            os.makedirs("test")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("test", f"test_vad_{timestamp}.wav")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16bit = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        print(f"\n测试音频已保存到: {filename}")
        print("\n[总结] FunASR VAD 模型测试成功，系统已准备就绪")
        
    except Exception as e:
        print(f"\n[错误] VAD 测试失败: {str(e)}")
        print("请检查 FunASR 安装是否正确，或尝试重新安装")
        return False
    
    return True

if __name__ == "__main__":
    print("==================================================")
    print("      FunASR VAD 测试工具")
    print("==================================================")
    
    test_vad_model()
