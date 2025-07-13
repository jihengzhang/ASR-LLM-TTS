#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版 VAD 语音活动检测系统
实现基本的语音活动检测功能
"""

import os
import time
import wave
import numpy as np
import threading
from datetime import datetime
import colorama
from colorama import Fore, Style

# 初始化 colorama
colorama.init()

try:
    import pyaudio
    print(f"{Fore.GREEN}✓ PyAudio 模块已加载{Style.RESET_ALL}")
except ImportError:
    print(f"{Fore.RED}✗ PyAudio 未安装{Style.RESET_ALL}")
    exit(1)

# 配置参数
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# VAD 参数
ENERGY_THRESHOLD = 0.01
ZCR_THRESHOLD = 0.1
MIN_SPEECH_DURATION = 1.0
SILENCE_DURATION = 2.0

class SimpleVAD:
    """简单的 VAD 实现"""
    
    def __init__(self):
        self.energy_history = []
        self.background_energy = 0.001
        
    def calculate_energy(self, frame):
        """计算音频帧能量"""
        return np.mean(frame ** 2)
    
    def calculate_zcr(self, frame):
        """计算过零率"""
        return np.mean(0.5 * np.abs(np.diff(np.sign(frame))))
    
    def update_background(self, energy):
        """更新背景噪声估计"""
        self.energy_history.append(energy)
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)
        
        if len(self.energy_history) >= 10:
            self.background_energy = np.percentile(self.energy_history, 20)
    
    def is_speech(self, frame):
        """判断是否为语音"""
        # 归一化音频
        if len(frame) == 0:
            return False, {}
            
        frame = frame.astype(np.float32) / 32768.0
        
        # 计算特征
        energy = self.calculate_energy(frame)
        zcr = self.calculate_zcr(frame)
        
        # 更新背景噪声
        self.update_background(energy)
        
        # 自适应阈值
        adaptive_threshold = max(self.background_energy * 5, ENERGY_THRESHOLD)
        
        # 判断条件
        energy_check = energy > adaptive_threshold
        zcr_check = zcr > ZCR_THRESHOLD
        
        features = {
            'energy': energy,
            'zcr': zcr,
            'threshold': adaptive_threshold,
            'background': self.background_energy
        }
        
        # 语音判断：能量超过阈值且过零率合理
        is_speech = energy_check and zcr_check
        
        return is_speech, features

class VADRecorder:
    """VAD 录音器"""
    
    def __init__(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}      VAD 语音活动检测录音系统{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}\n")
        
        # 初始化音频
        self.audio = pyaudio.PyAudio()
        self.vad = SimpleVAD()
        
        # 检查音频设备
        self.check_audio_devices()
        
        # 录音状态
        self.is_recording = False
        self.frames = []
        self.stop_flag = False
        
        # 语音状态
        self.speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        
        # 输出目录
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"{Fore.GREEN}✓ 系统初始化完成{Style.RESET_ALL}\n")
    
    def check_audio_devices(self):
        """检查音频设备"""
        device_count = self.audio.get_device_count()
        print(f"{Fore.BLUE}音频设备列表:{Style.RESET_ALL}")
        
        input_devices = 0
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices += 1
                print(f"  [{i}] {info['name']} (输入通道: {info['maxInputChannels']})")
        
        if input_devices == 0:
            raise Exception("没有找到可用的音频输入设备")
        
        print(f"{Fore.GREEN}✓ 找到 {input_devices} 个输入设备{Style.RESET_ALL}")
    
    def display_status(self, is_speech, features):
        """显示实时状态"""
        status_color = Fore.GREEN if is_speech else Fore.BLUE
        speech_indicator = "🎤 语音" if is_speech else "🔇 静音"
        
        # 清除行并显示状态
        print(f"\r{status_color}{speech_indicator}{Style.RESET_ALL} | "
              f"能量: {features['energy']:.4f} | "
              f"过零率: {features['zcr']:.3f} | "
              f"阈值: {features['threshold']:.4f} | "
              f"背景: {features['background']:.4f}", end='')
    
    def start_stream(self):
        """开始音频流"""
        print(f"{Fore.YELLOW}开始语音监听... 按 Ctrl+C 停止{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            stream.start_stream()
            
            while not self.stop_flag:
                if stream.is_active():
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.process_audio(data)
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"\n{Fore.RED}音频流错误: {e}{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}用户停止程序{Style.RESET_ALL}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            self.audio.terminate()
            
            if self.is_recording:
                self.stop_recording()
    
    def process_audio(self, data):
        """处理音频数据"""
        # 转换为 numpy 数组
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # VAD 检测
        is_speech, features = self.vad.is_speech(audio_array)
        
        # 显示实时状态
        self.display_status(is_speech, features)
        
        current_time = time.time()
        
        # 语音状态机
        if is_speech:
            if not self.speech_active:
                # 语音开始
                self.speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"\n{Fore.YELLOW}🎤 检测到语音开始{Style.RESET_ALL}")
        else:
            if self.speech_active:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                # 检查静音时长
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= SILENCE_DURATION:
                    # 语音结束
                    speech_duration = current_time - self.speech_start_time
                    print(f"\n{Fore.CYAN}🔇 语音结束，持续 {speech_duration:.2f} 秒{Style.RESET_ALL}")
                    
                    if speech_duration >= MIN_SPEECH_DURATION:
                        if not self.is_recording:
                            print(f"{Fore.GREEN}✓ 开始录音 (语音时长: {speech_duration:.2f}s){Style.RESET_ALL}")
                            self.start_recording()
                        else:
                            print(f"{Fore.CYAN}✓ 停止录音{Style.RESET_ALL}")
                            self.stop_recording()
                    else:
                        print(f"{Fore.RED}✗ 语音太短，忽略 ({speech_duration:.2f}s < {MIN_SPEECH_DURATION}s){Style.RESET_ALL}")
                    
                    self.speech_active = False
                    self.speech_start_time = None
                    self.silence_start_time = None
        
        # 录音中则保存数据
        if self.is_recording:
            self.frames.append(data)
    
    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.frames = []
        print(f"{Fore.GREEN}🔴 开始录音...{Style.RESET_ALL}")
    
    def stop_recording(self):
        """停止录音"""
        if self.is_recording and len(self.frames) > 0:
            self.is_recording = False
            
            # 保存文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"vad_rec_{timestamp}.wav")
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            
            # 显示信息
            duration = len(self.frames) * CHUNK / RATE
            file_size = os.path.getsize(filename) / 1024
            print(f"{Fore.CYAN}📁 录音保存: {filename}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 时长: {duration:.2f}s, 大小: {file_size:.1f}KB{Style.RESET_ALL}")
            
            self.frames = []
    
    def stop(self):
        """停止系统"""
        self.stop_flag = True

def main():
    """主函数"""
    print(f"{Fore.CYAN}{Style.BRIGHT}简化版 VAD 语音活动检测系统{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}配置: 最小语音 {MIN_SPEECH_DURATION}s, 静音超时 {SILENCE_DURATION}s{Style.RESET_ALL}")
    
    try:
        recorder = VADRecorder()
        recorder.start_stream()
    except Exception as e:
        print(f"\n{Fore.RED}系统错误: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
