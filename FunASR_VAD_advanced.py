#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""
"""
FunASR VAD 语音活动检测录音系统
实现真正的语音活动检测功能
"""

import os
import time
import wave
import numpy as np
import threading
import queue
from datetime import datetime
import collections
import colorama
from colorama import Fore, Style
from scipy import signal
from scipy.fft import fft

# 初始化 colorama 以支持 Windows 控制台颜色
colorama.init()

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print(f"{Fore.RED}PyAudio 未安装，请先安装: pip install pyaudio{Style.RESET_ALL}")
    PYAUDIO_AVAILABLE = False

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"{Fore.GREEN}✓ FunASR 模块已加载{Style.RESET_ALL}")
except ImportError:
    FUNASR_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠ FunASR 模块未找到，将使用内置 VAD 算法{Style.RESET_ALL}")

if not PYAUDIO_AVAILABLE:
    print(f"{Fore.RED}请先安装必要的依赖: pip install pyaudio scipy{Style.RESET_ALL}")
    exit(1)

# 配置参数
RATE = 16000  # 采样率
CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 声道数

# VAD 参数
VAD_FRAME_MS = 30  # VAD 帧长度 (毫秒)
VAD_FRAME_SIZE = int(RATE * VAD_FRAME_MS / 1000)  # VAD 帧大小
ENERGY_THRESHOLD = 0.01  # 能量阈值
ZCR_THRESHOLD = 0.3  # 过零率阈值
SPECTRAL_CENTROID_THRESHOLD = 1000  # 频谱重心阈值
SILENCE_DURATION = 1.5  # 静音持续时间(秒)
MIN_SPEECH_DURATION = 0.5  # 最小语音持续时间(秒)

class VADProcessor:
    """语音活动检测处理器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_size = VAD_FRAME_SIZE
        self.energy_threshold = ENERGY_THRESHOLD
        self.zcr_threshold = ZCR_THRESHOLD
        self.spectral_threshold = SPECTRAL_CENTROID_THRESHOLD
        
        # 自适应阈值
        self.background_energy = 0.001
        self.energy_history = collections.deque(maxlen=100)
        
    def extract_features(self, frame):
        """提取音频特征"""
        # 1. 能量特征
        energy = np.sum(frame ** 2) / len(frame)
        
        # 2. 过零率 (Zero Crossing Rate)
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        # 3. 频谱重心 (Spectral Centroid)
        fft_result = fft(frame)
        magnitude = np.abs(fft_result[:len(frame)//2])
        freqs = np.fft.fftfreq(len(frame), 1/self.sample_rate)[:len(frame)//2]
        
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
            
        # 4. 频谱滚降 (Spectral Rolloff)
        cumsum = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        return {
            'energy': energy,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff
        }
    
    def update_background_noise(self, energy):
        """更新背景噪声估计"""
        self.energy_history.append(energy)
        if len(self.energy_history) >= 10:
            # 使用能量历史的下四分位数作为背景噪声估计
            self.background_energy = np.percentile(list(self.energy_history), 25)
    
    def is_speech(self, frame):
        """判断是否为语音"""
        features = self.extract_features(frame)
        
        # 更新背景噪声
        self.update_background_noise(features['energy'])
        
        # 自适应能量阈值
        adaptive_energy_threshold = max(self.background_energy * 3, self.energy_threshold)
        
        # 多特征判断
        energy_check = features['energy'] > adaptive_energy_threshold
        zcr_check = features['zcr'] > self.zcr_threshold
        spectral_check = features['spectral_centroid'] > self.spectral_threshold
        
        # 至少满足两个条件才认为是语音
        speech_indicators = sum([energy_check, zcr_check, spectral_check])
        
        return speech_indicators >= 2, features

class FunASRVADRecorder:
    """FunASR VAD 录音系统"""
    
    def __init__(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}      初始化 FunASR VAD 录音系统{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
        
        # 初始化音频
        self.audio = pyaudio.PyAudio()
        self.check_audio_devices()
        
        # 初始化 VAD 处理器
        self.vad_processor = VADProcessor(RATE)
        
        # 初始化 FunASR VAD (如果可用)
        self.funasr_vad = None
        if FUNASR_AVAILABLE:
            self.init_funasr_vad()
        
        # 录音状态
        self.is_recording = False
        self.frames = []
        self.stop_event = threading.Event()
        self.last_activity_time = 0
        
        # 语音检测状态
        self.speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        
        # 音频缓冲区
        self.audio_buffer = collections.deque(maxlen=int(RATE / CHUNK * 2))  # 2秒缓冲
        
        # 确保输出目录存在
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ VAD 录音系统初始化完成{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}配置: 最小语音时长 {MIN_SPEECH_DURATION}s, 静音超时 {SILENCE_DURATION}s{Style.RESET_ALL}\n")
    
    def init_funasr_vad(self):
        """初始化 FunASR VAD 模型"""
        try:
            print(f"{Fore.YELLOW}正在加载 FunASR VAD 模型...{Style.RESET_ALL}")
            
            # 设置模型缓存
            current_dir = os.getcwd()
            os.environ['MODELSCOPE_CACHE'] = current_dir
            
            self.funasr_vad = AutoModel(
                model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}✓ FunASR VAD 模型加载成功{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ FunASR VAD 模型加载失败: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}将使用内置 VAD 算法{Style.RESET_ALL}")
            self.funasr_vad = None
    
    def check_audio_devices(self):
        """检查音频设备"""
        device_count = self.audio.get_device_count()
        print(f"{Fore.BLUE}检测到 {device_count} 个音频设备:{Style.RESET_ALL}")
        
        input_devices = []
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info['name']))
                print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']}")
        
        if not input_devices:
            raise Exception("没有找到可用的音频输入设备")
        
        # 获取默认输入设备
        try:
            default_input = self.audio.get_default_input_device_info()
            print(f"{Fore.GREEN}✓ 使用默认输入设备: {default_input['name']}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ 获取默认设备失败: {e}{Style.RESET_ALL}")
    
    def funasr_vad_detect(self, audio_data):
        """使用 FunASR 进行 VAD 检测"""
        if self.funasr_vad is None:
            return None
        
        try:
            # 转换为 FunASR 所需的格式
            result = self.funasr_vad.generate(input=audio_data)
            
            if result and len(result) > 0:
                # 解析 VAD 结果
                vad_segments = result[0].get('value', [])
                return len(vad_segments) > 0  # 如果有语音段，返回 True
            
        except Exception as e:
            print(f"{Fore.RED}FunASR VAD 检测错误: {e}{Style.RESET_ALL}")
        
        return None
    
    def process_audio_chunk(self, audio_data):
        """处理音频块"""
        # 转换为 numpy 数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 添加到缓冲区
        self.audio_buffer.append(audio_array)
        
        # 使用内置 VAD
        is_speech_builtin, features = self.vad_processor.is_speech(audio_array)
        
        # 使用 FunASR VAD (如果可用)
        is_speech_funasr = None
        if self.funasr_vad and len(self.audio_buffer) >= 10:  # 积累足够的音频再检测
            buffer_audio = np.concatenate(list(self.audio_buffer))
            is_speech_funasr = self.funasr_vad_detect(buffer_audio)
        
        # 综合判断
        if is_speech_funasr is not None:
            is_speech = is_speech_funasr
            vad_method = "FunASR"
        else:
            is_speech = is_speech_builtin
            vad_method = "Built-in"
        
        return is_speech, vad_method, features
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        current_time = time.time()
        
        # 处理音频
        is_speech, vad_method, features = self.process_audio_chunk(in_data)
        
        # 语音活动状态机
        if is_speech:
            if not self.speech_active:
                # 语音开始
                self.speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"{Fore.YELLOW}🎤 检测到语音开始 ({vad_method}) - 能量: {features['energy']:.4f}{Style.RESET_ALL}")
            
            # 更新最后活动时间
            self.last_activity_time = current_time
            
        else:
            if self.speech_active:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                # 检查是否静音太久
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= SILENCE_DURATION:
                    # 语音结束
                    speech_duration = current_time - self.speech_start_time
                    print(f"{Fore.CYAN}🔇 语音结束 - 持续时长: {speech_duration:.2f}s{Style.RESET_ALL}")
                    
                    if speech_duration >= MIN_SPEECH_DURATION:
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            print(f"{Fore.GREEN}✓ 检测到有效语音段 ({speech_duration:.2f}s)，开始录音{Style.RESET_ALL}")
                            self.start_recording()
                    else:
                        print(f"{Fore.RED}✗ 语音太短 ({speech_duration:.2f}s < {MIN_SPEECH_DURATION}s)，忽略{Style.RESET_ALL}")
                    
                    self.speech_active = False
                    self.speech_start_time = None
                    self.silence_start_time = None
        
        # 如果正在录音，保存音频数据
        if self.is_recording:
            self.frames.append(in_data)
            
            # 检查录音时长，避免录音过长
            if len(self.frames) > RATE * 30:  # 最长30秒
                print(f"{Fore.YELLOW}⏰ 录音时长超过30秒，自动停止{Style.RESET_ALL}")
                self.stop_recording()
        
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """开始录音"""
        if not self.is_recording:
            print(f"{Fore.GREEN}🔴 开始录音...{Style.RESET_ALL}")
            self.is_recording = True
            self.frames = []
            
            # 添加缓冲区中的音频，保留语音开始前的一些内容
            for audio_chunk in list(self.audio_buffer)[-5:]:  # 最后5个块
                audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
                self.frames.append(audio_bytes)
    
    def stop_recording(self):
        """停止录音并保存"""
        if self.is_recording:
            self.is_recording = False
            
            if len(self.frames) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"vad_recording_{timestamp}.wav")
                self.save_audio(filename)
                
                # 显示录音信息
                duration = len(self.frames) * CHUNK / RATE
                file_size = os.path.getsize(filename) / 1024
                print(f"{Fore.CYAN}✓ 录音已保存: {filename}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}📊 时长: {duration:.2f}s, 大小: {file_size:.1f}KB{Style.RESET_ALL}")
            
            self.frames = []
            print(f"{Fore.YELLOW}🔍 继续监听语音活动...{Style.RESET_ALL}")
    
    def save_audio(self, filename):
        """保存音频文件"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
    
    def start_monitoring(self):
        """开始监听"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}             开始语音活动检测{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}正在监听语音活动... 按 Ctrl+C 停止{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback
            )
            
            # 持续监听
            while not self.stop_event.is_set():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"{Fore.RED}音频流错误: {e}{Style.RESET_ALL}")
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
    
    def stop(self):
        """停止系统"""
        self.stop_event.set()

def main():
    """主函数"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                    FunASR VAD 语音活动检测系统{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}功能: 实时语音活动检测和自动录音{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}支持: FunASR VAD + 内置多特征 VAD 算法{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    try:
        recorder = FunASRVADRecorder()
        recorder.start_monitoring()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}程序已停止{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}程序错误: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
