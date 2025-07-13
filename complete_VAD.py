#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整版 FunASR VAD 语音活动检测系统
集成 FunASR VAD 模型和内置算法
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

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"{Fore.GREEN}✓ FunASR 模块已加载{Style.RESET_ALL}")
except ImportError:
    FUNASR_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠ FunASR 模块未找到，将使用内置 VAD{Style.RESET_ALL}")

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

class HybridVAD:
    """混合 VAD 系统：FunASR + 内置算法"""
    
    def __init__(self):
        self.energy_history = []
        self.background_energy = 0.001
        self.funasr_vad = None
        self.audio_buffer = []
        self.buffer_size = RATE * 2  # 2秒缓冲
        
        # 初始化 FunASR VAD
        if FUNASR_AVAILABLE:
            self.init_funasr()
    
    def init_funasr(self):
        """初始化 FunASR VAD 模型"""
        try:
            print(f"{Fore.YELLOW}正在加载 FunASR VAD 模型...{Style.RESET_ALL}")
            
            # 设置模型缓存到当前目录
            current_dir = os.getcwd()
            os.environ['MODELSCOPE_CACHE'] = current_dir
            os.environ['HF_HOME'] = current_dir
            
            # 加载模型，但不在这里输出进度条
            self.funasr_vad = AutoModel(
                model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}✓ FunASR VAD 模型加载成功{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ FunASR VAD 加载失败: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}将使用内置 VAD 算法{Style.RESET_ALL}")
            self.funasr_vad = None
    
    def calculate_features(self, frame):
        """计算音频特征"""
        if len(frame) == 0:
            return {'energy': 0, 'zcr': 0}
            
        # 归一化
        frame = frame.astype(np.float32) / 32768.0
        
        # 能量
        energy = np.mean(frame ** 2)
        
        # 过零率
        zcr = np.mean(0.5 * np.abs(np.diff(np.sign(frame))))
        
        return {'energy': energy, 'zcr': zcr}
    
    def update_background(self, energy):
        """更新背景噪声"""
        self.energy_history.append(energy)
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)
        
        if len(self.energy_history) >= 10:
            self.background_energy = np.percentile(self.energy_history, 20)
    
    def builtin_vad(self, frame):
        """内置 VAD 算法"""
        features = self.calculate_features(frame)
        energy = features['energy']
        zcr = features['zcr']
        
        # 更新背景噪声
        self.update_background(energy)
        
        # 自适应阈值
        adaptive_threshold = max(self.background_energy * 5, ENERGY_THRESHOLD)
        
        # 判断语音
        energy_check = energy > adaptive_threshold
        zcr_check = zcr > ZCR_THRESHOLD
        
        features.update({
            'threshold': adaptive_threshold,
            'background': self.background_energy
        })
        
        return energy_check and zcr_check, features
    
    def funasr_vad_detect(self, audio_array):
        """FunASR VAD 检测"""
        if self.funasr_vad is None:
            return None
            
        try:
            # 确保音频长度足够
            if len(audio_array) < RATE * 0.5:  # 至少0.5秒
                return None
            
            # FunASR 需要的格式
            result = self.funasr_vad.generate(input=audio_array)
            
            if result and len(result) > 0:
                # 解析结果
                vad_result = result[0].get('value', [])
                # 如果有语音段落，认为检测到语音
                return len(vad_result) > 0
                
        except Exception as e:
            # 静默处理 FunASR 错误，避免干扰显示
            pass
        
        return None
    
    def detect(self, frame):
        """混合检测"""
        # 内置算法
        builtin_result, features = self.builtin_vad(frame)
        
        # 更新音频缓冲区
        audio_array = frame.astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_array)
        
        # 保持缓冲区大小
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # FunASR 检测（每隔一段时间检测一次）
        funasr_result = None
        if len(self.audio_buffer) >= RATE and len(self.audio_buffer) % (RATE // 2) == 0:
            # 每0.5秒检测一次
            funasr_result = self.funasr_vad_detect(np.array(self.audio_buffer))
        
        # 综合结果
        if funasr_result is not None:
            final_result = funasr_result
            method = "FunASR"
        else:
            final_result = builtin_result  
            method = "内置"
        
        features['method'] = method
        features['funasr'] = funasr_result
        features['builtin'] = builtin_result
        
        return final_result, features

class VADRecorder:
    """VAD 录音器"""
    
    def __init__(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}        完整版 FunASR VAD 语音活动检测系统{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
        
        # 初始化组件
        self.audio = pyaudio.PyAudio()
        self.vad = HybridVAD()
        
        # 检查设备
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
        
        print(f"{Fore.GREEN}✓ 系统初始化完成{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}VAD 模式: {'FunASR + 内置算法' if FUNASR_AVAILABLE and self.vad.funasr_vad else '内置算法'}{Style.RESET_ALL}\n")
    
    def check_audio_devices(self):
        """检查音频设备"""
        device_count = self.audio.get_device_count()
        print(f"{Fore.BLUE}音频设备检测:{Style.RESET_ALL}")
        
        input_count = 0
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_count += 1
                if input_count <= 3:  # 只显示前3个设备
                    print(f"  [{i}] {info['name']}")
        
        if input_count > 3:
            print(f"  ... 还有 {input_count - 3} 个设备")
        
        if input_count == 0:
            raise Exception("没有找到音频输入设备")
        
        print(f"{Fore.GREEN}✓ 找到 {input_count} 个输入设备{Style.RESET_ALL}")
    
    def display_status(self, is_speech, features):
        """显示实时状态"""
        # 状态指示器
        if is_speech:
            indicator = f"{Fore.GREEN}🎤 语音{Style.RESET_ALL}"
        else:
            indicator = f"{Fore.BLUE}🔇 静音{Style.RESET_ALL}"
        
        # 检测方法
        method = features.get('method', '未知')
        method_color = Fore.CYAN if method == "FunASR" else Fore.YELLOW
        
        # 状态行
        status_line = (
            f"\r{indicator} | "
            f"{method_color}{method}{Style.RESET_ALL} | "
            f"能量: {features.get('energy', 0):.4f} | "
            f"过零率: {features.get('zcr', 0):.3f} | "
            f"阈值: {features.get('threshold', 0):.4f}"
        )
        
        # 如果有 FunASR 结果，显示对比
        if features.get('funasr') is not None:
            builtin = "√" if features.get('builtin') else "×"
            funasr = "√" if features.get('funasr') else "×"
            status_line += f" | 内置:{builtin} FunASR:{funasr}"
        
        print(status_line + " " * 10, end='')
    
    def start_stream(self):
        """开始音频流"""
        print(f"{Fore.YELLOW}开始语音监听... 按 Ctrl+C 停止{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
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
        # 转换音频
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # VAD 检测
        is_speech, features = self.vad.detect(audio_array)
        
        # 显示状态
        self.display_status(is_speech, features)
        
        # 语音状态机
        current_time = time.time()
        
        if is_speech:
            if not self.speech_active:
                # 语音开始
                self.speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"\n{Fore.YELLOW}🎤 检测到语音开始 ({features.get('method', '未知')}){Style.RESET_ALL}")
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
                            print(f"{Fore.GREEN}✓ 开始录音{Style.RESET_ALL}")
                            self.start_recording()
                        else:
                            print(f"{Fore.CYAN}✓ 停止录音{Style.RESET_ALL}")
                            self.stop_recording()
                    else:
                        print(f"{Fore.RED}✗ 语音太短，忽略 ({speech_duration:.2f}s){Style.RESET_ALL}")
                    
                    self.speech_active = False
                    self.speech_start_time = None
                    self.silence_start_time = None
        
        # 录音中保存数据
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
            filename = os.path.join(self.output_dir, f"funasr_vad_{timestamp}.wav")
            
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

def main():
    """主函数"""
    print(f"{Fore.CYAN}{Style.BRIGHT}FunASR VAD 语音活动检测系统{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}功能: 实时语音检测 + 自动录音{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}配置: 最小语音 {MIN_SPEECH_DURATION}s, 静音超时 {SILENCE_DURATION}s{Style.RESET_ALL}")
    
    try:
        recorder = VADRecorder()
        recorder.start_stream()
    except Exception as e:
        print(f"\n{Fore.RED}系统错误: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
