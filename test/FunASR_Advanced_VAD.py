#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558

简化版 FunASR-style VAD 实现
使用 numpy 和基本音频处理实现语音活动检测
"""

import os
import time
import wave
import numpy as np
import threading
import queue
from datetime import datetime
import collections
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio 未安装，请先安装: pip install pyaudio")
    PYAUDIO_AVAILABLE = False

if not PYAUDIO_AVAILABLE:
    print("请先安装必要的依赖: pip install pyaudio")
    exit(1)

# 配置参数
RATE = 16000  # 采样率
CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 声道数
KEYWORD = "你好"  # 激活关键词

# 改进的 VAD 参数
SILENCE_THRESHOLD = 800  # 静音阈值
ACTIVITY_THRESHOLD = 1200  # 语音活动阈值
SILENCE_DURATION = 1.5  # 静音持续时间(秒)
RECORD_PADDING = 0.8  # 录音前后额外保留的秒数
MIN_SPEECH_DURATION = 0.3  # 最小语音持续时间(秒)

# 高级 VAD 参数
FRAME_WINDOW = 10  # 用于平滑的帧窗口大小
ENERGY_THRESHOLD = 0.1  # 能量阈值
ZERO_CROSSING_THRESHOLD = 0.3  # 过零率阈值

class AdvancedVAD:
    """改进的语音活动检测器"""
    
    def __init__(self):
        self.frame_buffer = collections.deque(maxlen=FRAME_WINDOW)
        self.energy_buffer = collections.deque(maxlen=FRAME_WINDOW)
        self.zcr_buffer = collections.deque(maxlen=FRAME_WINDOW)
        
    def calculate_energy(self, frame):
        """计算帧能量"""
        return np.sum(frame ** 2) / len(frame)
    
    def calculate_zero_crossing_rate(self, frame):
        """计算过零率"""
        signs = np.sign(frame)
        sign_changes = np.diff(signs)
        return np.sum(np.abs(sign_changes)) / (2 * len(frame))
    
    def is_speech(self, frame):
        """判断是否为语音"""
        # 计算基本特征
        volume = np.abs(frame).mean()
        energy = self.calculate_energy(frame)
        zcr = self.calculate_zero_crossing_rate(frame)
        
        # 添加到缓冲区
        self.frame_buffer.append(frame)
        self.energy_buffer.append(energy)
        self.zcr_buffer.append(zcr)
        
        # 基本音量检测
        volume_speech = volume > SILENCE_THRESHOLD
        
        # 如果缓冲区不够大，只使用音量检测
        if len(self.energy_buffer) < FRAME_WINDOW:
            return volume_speech, volume
        
        # 计算平均特征
        avg_energy = np.mean(self.energy_buffer)
        avg_zcr = np.mean(self.zcr_buffer)
        
        # 组合判断条件
        energy_speech = avg_energy > ENERGY_THRESHOLD * np.max(self.energy_buffer)
        zcr_speech = avg_zcr > ZERO_CROSSING_THRESHOLD
        
        # 综合判断（至少满足两个条件）
        speech_indicators = [volume_speech, energy_speech, zcr_speech]
        is_speech = sum(speech_indicators) >= 2
        
        return is_speech, volume

class FunASRStyleRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.vad = AdvancedVAD()
        self.is_recording = False
        self.frames = []
        self.stop_event = threading.Event()
        self.last_activity_time = 0
        self.ring_buffer = collections.deque(maxlen=int(RATE / CHUNK * RECORD_PADDING))
        
        # 关键词检测相关变量
        self.speech_start_time = None
        self.continuous_speech_frames = []
        self.is_speech_active = False
        
        # 确保输出目录存在
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"已初始化 FunASR-style VAD 录音系统")
        print(f"使用高级语音检测算法 - 需要持续语音 {MIN_SPEECH_DURATION}秒")
    
    def start_stream(self):
        """开始音频流处理"""
        print(f"正在监听语音活动 (模拟关键词 '{KEYWORD}')")
        print("使用: 音量 + 能量 + 过零率 综合检测")
        
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback
        )
        
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
                
                # 如果录音中但检测到长时间静音，停止录音
                if self.is_recording and (time.time() - self.last_activity_time) > SILENCE_DURATION:
                    print("检测到持续静音，停止录音")
                    self.stop_recording()
                    
        except KeyboardInterrupt:
            print("程序已停止")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            if self.is_recording:
                self.stop_recording()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        
        # 使用高级 VAD 检测
        is_speech, volume = self.vad.is_speech(data)
        is_strong_speech = volume > ACTIVITY_THRESHOLD
        
        # 始终将数据添加到环形缓冲区
        self.ring_buffer.append(in_data)
        
        # 关键词检测逻辑 (基于持续语音活动)
        if not self.is_recording:
            if is_strong_speech and not self.is_speech_active:
                # 检测到语音开始
                self.is_speech_active = True
                self.speech_start_time = time.time()
                self.continuous_speech_frames = [in_data]
                print(f"高级VAD检测到语音开始，音量: {volume:.0f}")
                
            elif is_speech and self.is_speech_active:
                # 持续的语音
                self.continuous_speech_frames.append(in_data)
                speech_duration = time.time() - self.speech_start_time
                
                # 如果语音持续时间超过阈值，视为"关键词"
                if speech_duration >= MIN_SPEECH_DURATION and not self.is_recording:
                    print(f"检测到持续语音 {speech_duration:.2f}秒，触发录音")
                    self.start_recording()
                    
            elif not is_speech and self.is_speech_active:
                # 语音中断
                self.is_speech_active = False
                speech_duration = time.time() - self.speech_start_time
                if speech_duration < MIN_SPEECH_DURATION:
                    print(f"语音太短 ({speech_duration:.2f}秒)，忽略")
                self.continuous_speech_frames = []
        
        # 如果正在录音，更新活动时间和保存帧
        if self.is_recording:
            if is_speech:
                self.last_activity_time = time.time()
            self.frames.append(in_data)
            
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """开始录音"""
        if not self.is_recording:
            print("🎤 开始录音...")
            self.is_recording = True
            self.frames = []
            # 添加之前的缓冲区音频
            for frame in self.ring_buffer:
                self.frames.append(frame)
            self.last_activity_time = time.time()
    
    def stop_recording(self):
        """停止录音并保存文件"""
        if self.is_recording:
            self.is_recording = False
            
            # 保存录音文件
            if len(self.frames) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"fanasr_recording_{timestamp}.wav")
                self.save_audio(filename)
                print(f"✅ 录音已保存到: {filename}")
                print("🔄 继续监听语音...")
            
            self.frames = []
    
    def save_audio(self, filename):
        """保存音频到文件"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            
        # 显示录音文件信息
        file_size = os.path.getsize(filename) / 1024  # KB
        duration = len(self.frames) * CHUNK / RATE
        print(f"📁 文件大小: {file_size:.2f} KB, 时长: {duration:.2f} 秒")
    
    def stop(self):
        """停止录音系统"""
        self.stop_event.set()
        if self.is_recording:
            self.stop_recording()

def main():
    print("=" * 60)
    print("🎯 FunASR-style 高级 VAD 语音激活录音系统")
    print("=" * 60)
    print(f"📋 配置:")
    print(f"   - 最小语音时长: {MIN_SPEECH_DURATION} 秒")
    print(f"   - 使用高级检测: 音量 + 能量 + 过零率")
    print(f"   - 静音终止时间: {SILENCE_DURATION} 秒")
    print("🎤 对着麦克风说话开始录音")
    print("⏹️  按 Ctrl+C 退出程序")
    print("-" * 60)
    
    recorder = FunASRStyleRecorder()
    
    try:
        recorder.start_stream()
    except KeyboardInterrupt:
        print("\n🛑 程序已停止")
    finally:
        recorder.stop()

if __name__ == "__main__":
    main()
