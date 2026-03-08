#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR_VAD_KWS_plot.py - Real-time Voice Activity Detection and Keyword Spotting with Visualization

Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

# 在所有其他导入之前设置PY_SSIZE_T_CLEAN宏
# 这是解决"PY_SSIZE_T_CLEAN macro must be defined for '#' formats"错误的最直接方法
import os
# 设置环境变量
os.environ['PY_SSIZE_T_CLEAN'] = '1'

# 直接在sys模块中设置属性
import sys
setattr(sys, 'PY_SSIZE_T_CLEAN', 1)

if not hasattr(sys, 'setdlopenflags'):
    import os
    if hasattr(os, 'add_dll_directory'):
        # Windows specific DLL loading
        os.add_dll_directory(os.getcwd())
print("Import libraries...")
import os, re
import time
import wave
import queue
import threading
import traceback
import argparse
from xml.parsers.expat import model
import numpy as np
import matplotlib
matplotlib.use('wxagg')  # 确保使用WXAgg后端，不使用Tkinter
import matplotlib.pyplot as plt
import pyaudio
from datetime import datetime
from matplotlib.animation import FuncAnimation
from scipy import signal
import collections

# v2.0 New imports: Audio denoiser, Vosk KWS, Speaker recognition
try:
    from audio_denoiser import AudioDenoiser
    DENOISER_AVAILABLE = True
except ImportError:
    print("Warning: audio_denoiser not available. Noise reduction disabled.")
    DENOISER_AVAILABLE = False

try:
    from vosk_kws_engine import VoskKWSEngine
    VOSK_AVAILABLE = True
except ImportError:
    print("Warning: vosk_kws_engine not available. Lightweight KWS disabled.")
    VOSK_AVAILABLE = False

try:
    from speaker_recognition import DummySpeakerRecognizer
    SPEAKER_RECOG_AVAILABLE = True
except ImportError:
    print("Warning: speaker_recognition not available.")
    SPEAKER_RECOG_AVAILABLE = False

# Check if FunASR is available
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    MODEL_PATH = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join("C:\\Users\\212597558\\.cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    model_vad_id = r"damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # VAD model name 1.6MB
    vad_model_path = os.path.join(os.path.expanduser("~"), ".cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")

    AUDIO_PATH = "test/test_vad_20250715_120521.wav"  # Input audio path
    AUDIO_PATH = r"test_2025-07-22-10-41-06.wav"
    AUDIO_PATH = r"test_music_开始声音测试.wav"
    # AUDIO_PATH = r"test.wav"
    OUTPUT_DIR = "output/segments"             # Output directory for saving speech segments

    # vad_model = AutoModel(model=MODEL_PATH, model_type="vad", device="cuda", disable_update=True) # works ok
    # vad_model = AutoModel(model=model_vad, model_type="vad", device="cuda", disable_update=True)
    kws_model_id =  r"iic\\SenseVoiceSmall"
    kws_model_path = os.path.join(os.path.expanduser("~"), ".cache\\models", kws_model_id)        
    kws_model = AutoModel(model=kws_model_path, model_type="kws", device="cuda:0", disable_update=True)

except ImportError:
    print("Warning: FunASR not available. Only basic amplitude detection will be used.")
    FUNASR_AVAILABLE = False

DEBUG = True

class VADKWSProcessor:

    """
    Real-time Voice Activity Detection and Keyword Spotting with visualization
    """
    def __init__(self, sample_rate=16000, chunk_size=1600, channels=1,
                 format=pyaudio.paInt16, threshold=0.01,
                 silence_duration=1.5, vad_interval=0.25, 
                 buffer_duration=3.0, keywords=None, stopwords="stop", time_to_end_conversation=10,
                 enable_denoiser=True, denoise_strength='medium',
                 enable_vosk_kws=True, vosk_model_path=None,
                 enable_speaker_recog=False, pause_threshold=1.0):
        """
        Initialize the processor
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            chunk_size: Size of audio chunks (default: 1600, 100ms at 16kHz)
            channels: Audio channels (default: 1)
            format: PyAudio format (default: paInt16)
            threshold: Amplitude threshold for backup detection (default: 0.01)
            silence_duration: Silence duration to stop recording (default: 2.0s)
            vad_interval: VAD check interval in seconds (default: 0.25s)
            buffer_duration: Pre-buffer duration in seconds (default: 3.0s)
            keywords: List of keywords to detect (default: None)
        """
        # Recording support variables
        self.audio_frame_callback = None
        self.recording_frames = []
        self.is_recording = False
        self.recording_done = False  # 标记录音是否已完成但尚未保存
        self.recording_start_time = None
        self.recordings_dir = "recordings"
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        self.last_recognized_text = ""  # Store most recent recognized text
        # Audio parameters
        self.isDebug = False
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.vad_interval = vad_interval
        self.time_to_end_conversation = time_to_end_conversation
        self.stopwords = stopwords
        # Calculate frames based on durations
        self.buffer_frames = int(buffer_duration * sample_rate / chunk_size)
        self.max_silence_frames = int(silence_duration * sample_rate / chunk_size)
        
        # Audio buffer and queues
        # self.buffer = queue.Queue(maxsize=self.buffer_frames)
        self.audio_data_queue = queue.Queue(maxsize=100)
        self.vad_result_queue = queue.Queue(maxsize=100)
        self.kws_result_queue = queue.Queue(maxsize=100)
        self.find_valid_start = False
        
        # Buffer for accumulating audio data
        # Status flags
        self.running = False
        self.recording = False
        self.speech_detected = False
        self.is_keyword_detected = False
        self.silence_counter = 0
        
        # Thread locks and events
        self.lock = threading.RLock()  # Changed to RLock for reentrant locking
        self.stop_event = threading.Event()
        
        # Audio objects
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []  # For recording after keyword detection
        self.input_device_index = None
        
        # Visualization data
        # self.audio_buffer = np.zeros(80000)  # Changed from 160000 to 80000 (5 seconds history at 16kHz)
        # self.vad_history = np.zeros(100)    # VAD result history
        # self.detected_keywords = collections.deque(maxlen=10)  # Use deque to limit memory usage
        # Change audio_buffer to audio_buffer_plot
        
        self.audio_buffer_original = collections.deque(maxlen=25 * sample_rate)  # 原始音频数据
        self.audio_buffer_speechonly = collections.deque(maxlen=25 * sample_rate)  # 只保存语音段
        self.raw_vad_history = collections.deque(maxlen=1000)
        self.vad_history = np.zeros(100)    
        self.detected_keywords = collections.deque(maxlen=20)  # [(text, timestamp), ...]
        self.detected_vad = collections.deque(maxlen=50)  # [(is_speech, timestamp), ...]
        self.keyword_status_history = collections.deque(maxlen=50)  # [(is_detected, timestamp), ...]
        self.speech_level = collections.deque(maxlen=100)  # [(is_detected, timestamp), ...]
        self.current_kws_results = []  # 当前最新的ASR结果，用于文件命名
        
        # VAD and KWS models
        self.vad_model = None   # 初始化为None
        self.asr_model = None
        self.last_vad_check = time.time()
        self.audio_buffer_for_vad = np.array([], dtype=np.float32)
        
        # Keywords to detect
        self.keywords = keywords or ["你好", "小艾", "开始", "小度", "小爱", "Hi Michael", "Hi Panda"]
        
        # v2.0: State machine for wake-word detection
        self.state = 'SLEEPING'  # SLEEPING | AWAKE | LISTENING
        self.awake_timeout = 5.0  # Seconds to stay awake after last speech
        self.last_speech_time = time.time()
        self.pause_threshold = pause_threshold  # Adjustable pause detection threshold
        
        # v2.0: Audio denoiser
        self.audio_denoiser = None
        self.denoised_audio_buffer = collections.deque(maxlen=25 * sample_rate)  # 降噪后音频
        if enable_denoiser and DENOISER_AVAILABLE:
            try:
                self.audio_denoiser = AudioDenoiser(
                    sample_rate=sample_rate,
                    strength=denoise_strength,
                    vad_mode=2,
                    noise_window_duration=0.5
                )
                print(f"✅ Audio denoiser enabled (strength: {denoise_strength})")
            except Exception as e:
                print(f"⚠️  Failed to initialize denoiser: {e}")
                self.audio_denoiser = None
        
        # v2.0: Vosk lightweight KWS
        self.vosk_kws = None
        if enable_vosk_kws and VOSK_AVAILABLE:
            try:
                if vosk_model_path is None:
                    vosk_model_path = os.path.join(os.path.dirname(__file__), 
                                                   'models', 'vosk_models', 
                                                   'vosk-model-small-cn-0.22')
                if os.path.exists(vosk_model_path):
                    self.vosk_kws = VoskKWSEngine(
                        model_path=vosk_model_path,
                        keywords=self.keywords,
                        sample_rate=sample_rate,
                        confidence_threshold=0.7,
                        use_grammar=True
                    )
                    print(f"✅ Vosk KWS engine enabled")
                else:
                    print(f"⚠️  Vosk model not found at: {vosk_model_path}")
                    print("Please run: python download_vosk_model.py cn-small")
            except Exception as e:
                print(f"⚠️  Failed to initialize Vosk KWS: {e}")
                self.vosk_kws = None
        
        # v2.0: Speaker recognition (placeholder)
        self.speaker_recognizer = None
        if enable_speaker_recog and SPEAKER_RECOG_AVAILABLE:
            try:
                self.speaker_recognizer = DummySpeakerRecognizer()
                print(f"✅ Speaker recognition enabled (placeholder)")
            except Exception as e:
                print(f"⚠️  Failed to initialize speaker recognition: {e}")
        
        # Thread pool
        self.threads = []
        
        # Check audio devices and initialize models
        self.check_audio_devices()
        if FUNASR_AVAILABLE:
            # self.init_models()   # Remove vad_model initialization
            try:
                print("Loading ASR model for keyword spotting...")
                self.asr_model = AutoModel(
                    model= kws_model_path,
                    model_revision="v2.0.4",
                    device="cuda" if self.check_cuda_available() else "cpu",
                    output_type="dict",
                    disable_update=True,
                    disable_log=True,
                    disable_progress_bar=True
                )
                print("ASR model loaded successfully")

                # add vad model
                self.vad_model = AutoModel(
                    model=vad_model_path,
                    model_revision="v2.0.4",
                    device="cuda" if self.check_cuda_available() else "cpu",
                    output_type="dict",
                    disable_update=True,
                    disable_log=True,
                    disable_progress_bar=True
                )
                print("VAD model loaded successfully")
            except Exception as e:
                print(f"Error loading ASR model: {e}")
                traceback.print_exc()
                print("Continuing without ASR model")
                self.asr_model = None
                self.vad_model = None
        else:
            print("FunASR not available, continuing without VAD/ASR models")
        
        # Setup visualization
        self.setup_visualization()  # add visualization later
    
    # v2.0: Setter methods for UI controls
    def set_denoise_strength(self, strength: str):
        """
        Change denoising strength
        
        Args:
            strength: 'weak', 'medium', or 'strong'
        """
        if self.audio_denoiser is not None:
            self.audio_denoiser.set_strength(strength)
            print(f"Denoising strength changed to: {strength}")
        else:
            print("Warning: Denoiser not available")
    
    def set_pause_threshold(self, threshold_seconds: float):
        """
        Set pause detection threshold
        
        Args:
            threshold_seconds: Pause duration in seconds (0.5-2.0 recommended)
        """
        if 0.5 <= threshold_seconds <= 5.0:
            self.pause_threshold = threshold_seconds
            print(f"Pause threshold changed to: {threshold_seconds:.2f}s")
        else:
            print(f"Warning: Pause threshold {threshold_seconds} out of range (0.5-5.0)")
    
    def set_state(self, new_state: str):
        """
        Change system state (SLEEPING/AWAKE/LISTENING)
        
        Args:
            new_state: Target state
        """
        if new_state in ['SLEEPING', 'AWAKE', 'LISTENING']:
            with self.lock:
                old_state = self.state
                self.state = new_state
                self.last_speech_time = time.time()
                print(f"State changed: {old_state} → {new_state}")
        else:
            print(f"Warning: Invalid state '{new_state}'")
    
    def get_denoise_stats(self):
        """
        Get denoising statistics for UI display
        
        Returns:
            tuple: (rms_before, rms_after, reduction_db)
        """
        if self.audio_denoiser is not None:
            stats = self.audio_denoiser.get_stats()
            return (
                stats['avg_rms_before'], 
                stats['avg_rms_after'], 
                stats['avg_reduction_db']
            )
        return (0.0, 0.0, 0.0)
    
    def check_audio_devices(self):
        """Check available audio devices and select input device"""
        try:
            print("Checking audio devices...")
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            print(f"Found {numdevices} audio devices:")
            input_devices = []
            default_input_index = None
            
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    if device_info.get('maxInputChannels') > 0:
                        input_devices.append((i, device_info.get('name')))
                        print(f"Input Device {i}: {device_info.get('name')}")
                        # Check if this is the default input device
                        if device_info.get('isDefaultInput', 0) == 1:
                            default_input_index = i
                    if device_info.get('maxOutputChannels') > 0:
                        print(f"Output Device {i}: {device_info.get('name')}")
                except Exception as e:
                    print(f"Error getting device {i} info: {e}")
            
            # Try to use default input device first
            if default_input_index is not None:
                self.input_device_index = default_input_index
                device_info = self.pyaudio.get_device_info_by_index(default_input_index)
                print(f"Using default input device {default_input_index}: {device_info.get('name')}")
            # Or first available input device
            elif input_devices:
                self.input_device_index = input_devices[0][0]
                print(f"Using input device {self.input_device_index}: {input_devices[0][1]}")
            else:
                print("Warning: No input device found!")
                
        except Exception as e:
            print(f"Error checking audio devices: {e}")
            traceback.print_exc()
    
    def check_cuda_available(self):
        """Check if CUDA is available for model inference"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def start_recording(self):
        """Start recording audio
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.running:
            print("Warning: Cannot start recording - processor not running")
            return False
        
        self.is_recording = True
        self.recording_frames = []  # 清空之前的录音
        self.current_kws_results = []  # 清空上一次的关键词结果，避免新录音使用旧关键词
        self.recording_start_time = time.time()
        
        if self.isDebug:
            print(f"Recording started at {datetime.fromtimestamp(self.recording_start_time).strftime('%H:%M:%S')}")
            print(f"已清空previous keywords，current_kws_results重置为空列表")
        
        return True
    
    def stop_recording(self):
        """Stop recording audio
        
        Returns:
            list: Recorded audio frames or None if no recording was in progress
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        self.recording_done = True  # 设置录音已完成标志
        
        if not self.recording_frames:
            print("警告: 没有录制到音频数据")
            # 即使没有录音数据，也不要重置recording_done标志
            # 这样可以确保在其他地方检查时不会误判
            return None
        
        # 打印更多调试信息
        duration = sum(len(frame) for frame in self.recording_frames) / self.sample_rate
        print(f"录音已停止, {len(self.recording_frames)} 帧, {duration:.2f} 秒")
        print(f"当前KWS结果: {self.current_kws_results}")
        
        # 当前录音结果已保存，下一次录音前应清空
        return_frames = self.recording_frames.copy()
        
        # 在这里不直接调用save_recording，而是通过标志让KWS线程来调用
        # 这样可以确保录音在ASR处理完成后保存，文件名能包含最新的识别结果
        print(f"录音停止完成，已设置recording_done标志为{self.recording_done}，等待KWS线程处理后保存")
        
        # 停止录音后，不要立即清空 current_kws_results，因为在保存录音时需要使用
        # self.current_kws_results = []
        
        return return_frames
    
    def save_recording(self, filename=None):
        """Save recorded audio to file
        
        Args:
            filename: Optional filename to save to. If None, a filename will be generated
                    based on the KWS processing state:
                    - If find_valid_start is True, VAD detection is active, and a keyword is detected,
                      use the keyword (first 10 chars) as filename prefix.
                    - If find_valid_start is False, use "quiet" as filename prefix.
        
        Returns:
            str: Path to the saved file or None if saving failed
        """
        if not self.recording_frames:
            print("No audio data to save")
            self.recording_done = False  # 重置录音完成标志
            return None
        
        try:
            import wave
            import numpy as np
            import re
            
            # 如果没有提供文件名，根据KWS处理状态生成文件名
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 默认文件名前缀
                filename_prefix = "recording"
                
                # 首先检查是否有有效的ASR结果
                if hasattr(self, 'current_kws_results') and self.current_kws_results:
                    # 使用当前ASR结果作为文件名
                    asr_text = ""
                    for item in self.current_kws_results:
                        if isinstance(item, dict) and 'text' in item and item['text']:
                            # 清理文本，去除特殊字符
                            clean_text = re.sub(r"<\|.*?\|>", "", item['text'])
                            clean_text = re.sub(r"[^\w\s]", "", clean_text)
                            if clean_text.strip():
                                asr_text += clean_text + "_"
                    
                    if asr_text:
                        # 去掉末尾的下划线
                        asr_text = asr_text.strip('_')
                        # 限制长度
                        if len(asr_text) > 50:
                            asr_text = asr_text[:50]
                        filename_prefix = re.sub(r'[\\/*?:"<>|]', '', asr_text)
                
                filename = os.path.join(self.recordings_dir, f"{filename_prefix}_{timestamp}.wav")
            
            # 将所有音频帧合并到一个数组中
            audio_data = np.concatenate(self.recording_frames)
            
            # 确保音频数据为int16格式 (已经是int16，无需转换)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # 创建WAV文件
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 2字节 = 16位
                wf.setframerate(self.sample_rate)  # 采样率
                wf.writeframes(audio_data.tobytes())
            
            if self.isDebug:
                print(f"Audio saved to {filename}")
            
            # 保存用于UI显示的文本，使用文件名前缀
            self.last_recognized_text = filename_prefix
            
            print(f"成功保存录音到文件: {filename}")
            
            # 重置录音完成标志
            self.recording_done = False
            
            # 清空录音帧，避免重复保存
            self.recording_frames = []
            
            return filename
        
        except Exception as e:
            print(f"Error saving audio: {e}")
            traceback.print_exc()
            return None
    def get_audio_devices(self):
        """Get list of all available audio input devices
        
        Returns:
            list: List of tuples (device_index, device_name)
            tuple: Default device as (device_index, device_name) or None
        """
        devices = []
        default_device = None
        
        try:
            # Recreate PyAudio instance to ensure we get fresh device info
            if hasattr(self, 'pyaudio'):
                try:
                    # Only recreate if not streaming
                    if not (hasattr(self, 'stream') and self.stream and self.stream.is_active()):
                        self.pyaudio.terminate()
                        self.pyaudio = pyaudio.PyAudio()
                        print("Refreshed PyAudio instance for device detection")
                except Exception as e:
                    print(f"Warning when refreshing PyAudio: {e}")
                    # Continue with existing pyaudio instance
            
            # Get device count
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            print(f"Found {numdevices} total audio devices")
            
            # Find all input devices
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_index(i)
                    # Only include input devices
                    if device_info.get('maxInputChannels') > 0:
                        name = device_info.get('name')
                        devices.append((i, name))
                        print(f"Found input device {i}: {name}")
                        
                        # Track default device
                        if device_info.get('isDefaultInput', 0) == 1:
                            default_device = (i, name)
                            print(f"Default input device: {name} (index {i})")
                except Exception as e:
                    print(f"Error getting device {i} info: {e}")
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            traceback.print_exc()
            
        print(f"Found {len(devices)} audio input devices")
        return devices, default_device

    def start(self):
        """Start the VAD/KWS processor"""
        print("Starting VAD/KWS processor...")
        
        with self.lock:
            if self.running:
                print("Processor is already running")
                return
            
            self.running = True
            self.stop_event.clear()
        
        try:
            # Open audio stream
            if not __name__ == "__main__":      
                # If we have an existing stream, close it
                if hasattr(self, 'stream') and self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception as e:
                        print(f"Error closing previous stream: {e}")
                        
                # Get device info for debug output
                device_name = "Default Device"
                if self.input_device_index is not None:
                    try:
                        device_info = self.pyaudio.get_device_info_by_index(self.input_device_index)
                        device_name = device_info.get('name', f"Device {self.input_device_index}")
                    except Exception:
                        pass
                        
                print(f"Opening audio stream on device: {device_name} (index: {self.input_device_index})")
                
                # Open new stream with current device
                self.stream = self.pyaudio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=self.input_device_index,
                    stream_callback=self.audio_callback
                )
                self.stream.start_stream()
                print(f"Voice activation system started, waiting for speech...")
            
            # Start processing threads (excluding deprecated audio_input_thread and plotting_thread)
            thread_names = ['KWSProcessing']
            thread_targets = [self.kws_processing_thread]
            
            for i, (name, target) in enumerate(zip(thread_names, thread_targets)):
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                self.threads.append(thread)
                print(f"Started {name} thread")
            
            pass  # Don't show plot here - will be shown in main thread
        except Exception as e:
            print(f"Error starting processor: {e}")
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Stop the processor"""
        try:
            self.stop_event.set()  # 设置停止事件
            print("Stopping VAD/KWS processor...")
            
            # 安全关闭音频流
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                
            # if hasattr(self, 'p') and self.p:
            #     self.p.terminate()
                
            # 正确关闭matplotlib图形，不使用plt.close()
            # if hasattr(self, 'fig'):
            #     try:
            #         import matplotlib.pyplot as plt
            #         plt.close(self.fig)  # 只关闭特定的figure
            #     except Exception as e:
            #         print(f"Error closing plot: {e}")
            with self.lock:
                if self.running:
                    print("Processor is already running")
                    self.running = False
        except Exception as e:
            print(f"Error stopping processor: {e}")
        finally:
            print("Processor stopped.")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback function"""
        try:
            # Convert bytes to numpy array and handle multi-channel input
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Reshape if multi-channel (samples, channels)
            if self.channels > 1:
                audio_data = audio_data.reshape(-1, self.channels)
                # Take first channel only
                audio_data = audio_data[:, 0]
            
            # Normalize to [-1, 1] for visualization
            audio_data_norm = audio_data.astype(np.float32) / 32768.0

            # Add to buffer collector with timestamp
            current_time = time.time()
            
            # 保存原始音频数据到显示缓冲区
            with self.lock:
                self.audio_buffer_original.append((audio_data_norm, current_time))
            
            # v2.0: Apply denoising if enabled
            audio_denoised = audio_data_norm
            if self.audio_denoiser is not None:
                try:
                    audio_denoised, denoise_stats = self.audio_denoiser.denoise_frame(
                        audio_data_norm, skip_vad=False
                    )
                    # Save denoised audio to buffer
                    with self.lock:
                        self.denoised_audio_buffer.append((audio_denoised, current_time))
                except Exception as e:
                    if self.isDebug:
                        print(f"Denoising error: {e}")
                    audio_denoised = audio_data_norm
            else:
                # No denoiser, just copy original
                with self.lock:
                    self.denoised_audio_buffer.append((audio_data_norm, current_time))
                
            # Call audio frame callback if set (for recording feature)
            if self.audio_frame_callback is not None:
                # Use denoised audio for recording
                self.audio_frame_callback(audio_denoised)
            
            # 如果正在录音，保存音频数据到录音缓冲区
            if self.is_recording:
                # 保存降噪后的数据用于WAV文件保存（转回int16格式）
                audio_denoised_int16 = (audio_denoised * 32768.0).astype(np.int16)
                self.recording_frames.append(audio_denoised_int16.copy())
                if self.isDebug:
                    if len(self.recording_frames) % 10 == 0:  # 每10帧输出一次，避免过多日志
                        duration = sum(len(frame) for frame in self.recording_frames) / self.sample_rate
                        print(f"Recording: {len(self.recording_frames)} frames, {duration:.2f} seconds")
                
            # Put denoised audio in queue for processing
            try:
                self.audio_data_queue.put_nowait((audio_denoised, current_time))
                if self.isDebug:
                    print(f"Audio data added to queue，length: {len(audio_denoised)}")
            except queue.Full:
                pass
        except Exception as e:
            print(f"Audio callback error: {e}")
            traceback.print_exc()

        return (None, pyaudio.paContinue)
    
    def setup_visualization(self):
        """Set up the visualization plots"""
        plt.rcParams['backend'] = 'TkAgg'
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
        # 设置中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # Create figure with subplots - now using 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 8), 
                                                      gridspec_kw={'height_ratios': [2, 2, 1], 'hspace': 0.5})
        # Adjust the layout to reduce left/right margins
        self.fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07)

        # Configure original audio waveform subplot
        self.ax1.set_title('Original Audio', pad=5)
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)

        # Configure detected speech subplot
        self.ax2.set_title('Mean filtered audio and VAD Detection', pad=5)
        self.ax2.set_ylim(-0.5, 0.5)
        self.ax2.set_ylabel('Amplitude')
        self.ax2.grid(True)

        # Configure keywords subplot
        self.ax3.set_title('Detected Keywords & Keyword Status', pad=5)
        self.ax3.set_xlabel('Time (s)', labelpad=8)
        self.ax3.set_ylabel('Keywords')
        self.ax3.grid(True)
        
        # Initialize empty plot lines for update_plot to work with
        self.waveform_line, = self.ax1.plot([], [], 'gray', linewidth=0.8, label='Original')
        self.speech_line, = self.ax2.plot([], [], 'b-', linewidth=1.0, label='Speech')
        self.vad_line, = self.ax1.plot([], [], 'r-', linewidth=1.5, label='Keyword Active')
        
        # Remove status text from figure - will be shown in UI instead
        self.status = 'Initializing'
        # Create a placeholder for status text that will be used by the UI
        self.status_text = None

        plt.ion()
        self.fig.canvas.draw()

        self.animation = FuncAnimation(
            self.fig, self.update_plot,
            interval=100,
            blit=False,
            repeat=True,
            cache_frame_data=False
        )

        self.fig.canvas.mpl_connect('close_event', self.on_plot_close)

    def on_plot_close(self, event):
        """Handle plot close event"""
        print("Plot window closed")
        # 在另一个线程中停止处理器，避免阻塞GUI线程
        threading.Thread(target=self._safe_stop, daemon=True).start()
    
    def _safe_stop(self):
        """Safely stop the processor in a separate thread"""
        try:
            self.stop()
        except Exception as e:
            print(f"Error during stop: {e}")
            traceback.print_exc()
        finally:
            # 使用sys.exit在主线程中退出
            import sys
            sys.exit(0)

    def update_plot(self, frame):
        """Update the plot with new data"""
        try:
            with self.lock:
                audio_buffer_speech = list(self.audio_buffer_speechonly)
                audio_buffer_original = list(self.audio_buffer_original)
                detected_keywords_copy = list(self.detected_keywords)
                detected_vad_copy = list(self.detected_vad)
                keyword_status_history_copy = list(self.keyword_status_history)
                keyword_detected = self.is_keyword_detected
                raw_vad_history_copy = list(self.raw_vad_history)
            
            # Skip update if no new data
            if not audio_buffer_original:
                return []  # Return empty list since we're not using blit=True

            # Use timestamps from audio buffer instead of current time
            if audio_buffer_original:
                last_chunk, last_timestamp = audio_buffer_original[-1]
                current_time_0 = last_timestamp + (len(last_chunk) / self.sample_rate)  # Add duration of last chunk
            else:
                current_time_0 = time.time()
            if self.running:
                current_time = time.time()
            else:
                current_time = current_time_0

            display_window = 20  # Show last 20 seconds
            start_time = current_time - display_window

            # 处理原始音频数据 in ax1
            all_original_audio = []
            all_original_timestamps = []
            
            # Filter buffer data within display window for original audio
            filtered_original = [
                (chunk, ts) for chunk, ts in audio_buffer_original 
                if ts + (len(chunk) / self.sample_rate) >= start_time
            ]
            
            for audio_chunk, chunk_start_time in filtered_original:
                # Generate timestamps for each sample in the chunk
                chunk_duration = len(audio_chunk) / self.sample_rate
                chunk_timestamps = np.linspace(
                    chunk_start_time, 
                    chunk_start_time + chunk_duration, 
                    len(audio_chunk)
                )
                all_original_audio.extend(audio_chunk)
                all_original_timestamps.extend(chunk_timestamps)
            
            # 处理语音段数据
            all_speech_audio = []
            all_speech_timestamps = []
            
            # Filter buffer data within display window for speech only
            filtered_speech = []
            for chunk, ts in audio_buffer_speech:
                if ts is None:
                    print(f"警告: audio_buffer_speech中发现None时间戳，已跳过")
                    continue
                if ts + (len(chunk) / self.sample_rate) >= start_time:
                    filtered_speech.append((chunk, ts))
            
            for audio_chunk, chunk_start_time in filtered_speech:
                # Generate timestamps for each sample in the chunk
                chunk_duration = len(audio_chunk) / self.sample_rate
                chunk_timestamps = np.linspace(
                    chunk_start_time, 
                    chunk_start_time + chunk_duration, 
                    len(audio_chunk)
                )
                all_speech_audio.extend(audio_chunk)
                all_speech_timestamps.extend(chunk_timestamps)


            # Process keyword detection status data
            keyword_status_values = []
            keyword_status_timestamps = []
            # 安全过滤keyword_status数据
            filtered_keyword_status = []
            for is_detected, ts in keyword_status_history_copy:
                if ts is None:
                    print(f"警告: 检测到keyword_status数据中有None时间戳，已跳过")
                    continue
                if ts >= start_time:
                    filtered_keyword_status.append((is_detected, ts))
            
            # Always ensure we have at least a baseline for the keyword status line
            if not filtered_keyword_status:
                # If no recent keyword status, create a flat line at 0
                keyword_status_timestamps = [start_time, current_time]
                keyword_status_values = [0.0, 0.0]
            else:
                # Create continuous line for keyword detection status
                last_status = 0
                last_time = start_time
                
                # Add initial point at start_time if needed
                keyword_status_timestamps.append(start_time)
                keyword_status_values.append(0.0)  # Default to 0 at start
                
                for is_detected, ts in filtered_keyword_status:
                    # Add point at previous status just before change
                    keyword_status_timestamps.append(ts - 0.001)
                    keyword_status_values.append(last_status)
                    
                    # Add point at new status
                    keyword_status_timestamps.append(ts)
                    # Use 0.5 instead of 0.8 for detected keywords as requested
                    keyword_status_values.append(0.5 if is_detected else 0.0)
                    
                    last_status = 0.5 if is_detected else 0.0
                    last_time = ts
                
                # Add final point to extend to current time
                keyword_status_timestamps.append(current_time)
                keyword_status_values.append(last_status)

            # 格式化时间轴函数
            def format_time(x, pos):
                return datetime.fromtimestamp(x).strftime('%H:%M:%S.%f')[:-4]
            
            # 更新原始音频子图 (ax1)
            self.ax1.clear()
            self.ax1.set_title('Original Audio')
            
            if all_original_timestamps:
                original_samples = np.array(all_original_audio)
                original_timestamps = np.array(all_original_timestamps)
                self.ax1.plot(original_timestamps, original_samples, 'gray', linewidth=0.8)

                # Plot speech level in ax1
                # 安全过滤speech_level数据
                filtered_speech_level = []
                for level, ts in self.speech_level:
                    if ts is None:
                        print(f"警告: 检测到speech_level数据中有None时间戳，已跳过")
                        continue
                    if ts >= start_time:
                        filtered_speech_level.append((level, ts))
                
                if filtered_speech_level:
                    level_times = []
                    level_values = []
                    last_level = 0
                    last_time = start_time
                    
                    # Add initial point
                    level_times.append(start_time)
                    level_values.append(0)
                    
                    for level, ts in filtered_speech_level:
                        # Add point just before state change
                        level_times.append(ts - 0.001)
                        level_values.append(last_level)
                        
                        # Add point at new state
                        level_times.append(ts)
                        level_values.append(level)  # level is already 0 or 1
                        
                        last_level = level
                        last_time = ts
                    
                    # Add final point
                    level_times.append(current_time)
                    level_values.append(last_level)
                    
                    # Plot speech level line in ax1
                    self.ax1.plot(level_times, level_values, 'b-', 
                                 linewidth=2.0, alpha=0.7, label='Speech Level')
            
            self.ax1.legend(loc='upper left')

            self.ax1.set_xlim(start_time, current_time)
            # self.ax1.set_xlim(time.time()-20, time.time())
            self.ax1.set_ylim(-1.0, 1.0)
            # 设置精确的y轴刻度间隔为0.1
            yticks = np.arange(-1.0, 1.1, 0.5)
            self.ax1.set_yticks(yticks)
            self.ax1.grid(True)
            self.ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
            
            # 更新语音检测子图 (ax2) - now showing VAD detection results
            self.ax2.clear()
            self.ax2.set_title('Mean filtered audio and VAD Detection')
            
            # Plot original speech waveform as background
            if all_speech_timestamps:
                speech_samples = np.array(all_speech_audio)
                speech_timestamps = np.array(all_speech_timestamps)
                self.ax2.plot(speech_timestamps, speech_samples, 'gray', 
                             linewidth=0.8, alpha=0.5, label='Audio')

            # Plot VAD detection results - 过滤掉任何None时间戳并添加调试日志
            filtered_vad = []
            for vad_level, ts in detected_vad_copy:
                if ts is None:
                    print(f"警告: 检测到VAD数据中有None时间戳，已跳过")
                    continue
                if ts >= start_time:
                    filtered_vad.append((vad_level, ts))
            
            if filtered_vad:
                vad_times = []
                vad_values = []
                last_level = 0
                last_time = start_time
                
                # Add initial point
                vad_times.append(start_time)
                vad_values.append(0)
                
                for level, ts in filtered_vad:
                    # Add point just before state change
                    vad_times.append(ts - 0.001)
                    vad_values.append(last_level)
                    
                    # Add point at new state
                    vad_times.append(ts)
                    vad_values.append(level)  # 直接使用level值(0或0.5)
                    
                    last_level = level
                    last_time = ts
                
                # Add final point
                vad_times.append(current_time)
                vad_values.append(last_level)
                
                # Plot VAD line
                self.ax2.plot(vad_times, vad_values, 'r-', 
                             linewidth=2.0, alpha=0.7, label='VAD')
                
                # Add colored background for speech periods
                # for i in range(len(vad_times)-1):
                #     if vad_values[i] > 0.1:  # If VAD is active
                #         self.ax2.axvspan(vad_times[i], vad_times[i+1], 
                #                        color='green', alpha=0.2)

            self.ax2.legend(loc='upper left')
            self.ax2.set_xlim(start_time, current_time)
            # self.ax2.set_xlim(time.time()-20, time.time())
            self.ax2.set_ylim(-1.0, 1.0)
            # 设置精确的y轴刻度间隔为0.1
            yticks = np.arange(-1.0, 1.1, 0.5)
            self.ax2.set_yticks(yticks)
            self.ax2.grid(True)
            self.ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

            # 更新关键词显示子图 (ax3) - now includes keyword status history
            self.ax3.clear()
            self.ax3.set_title('Detected Keywords & Keyword Status')
            self.ax3.set_xlim(start_time, current_time)
            # self.ax3.set_xlim(time.time()-20, time.time())
            self.ax3.set_ylim(-0.1, 1.1)
            # 设置精确的y轴刻度间隔为0.1
            yticks = np.arange(-0.1, 1.2, 0.5)
            self.ax3.set_yticks(yticks)
            self.ax3.grid(True)

            # Always plot the keyword status line, even if flat at zero
            self.ax3.plot(keyword_status_timestamps, keyword_status_values, 'r-', 
                          linewidth=2.0, alpha=0.6, label='Keyword Active')
            
            # Add shaded regions for active keyword periods
            last_val = 0
            start_shade = None
            for i, (ts, val) in enumerate(zip(keyword_status_timestamps, keyword_status_values)):
                if val > 0.1 and last_val < 0.1:  # Keyword became active
                    start_shade = ts
                elif val < 0.1 and last_val > 0.1 and start_shade is not None:  # Keyword became inactive
                    self.ax3.axvspan(start_shade, ts, alpha=0.2, color='red')
                    start_shade = None
                last_val = val
            
            # Handle case where we're still in active region at end
            if start_shade is not None:
                self.ax3.axvspan(start_shade, current_time, alpha=0.2, color='red')

            # Filter and display keywords within time window - 增强安全性检查
            visible_keywords = []
            for kw, ts in detected_keywords_copy:
                if ts is None:
                    print(f"警告: 检测到detected_keywords中有None时间戳，已跳过")
                    continue
                if start_time <= ts <= current_time:
                    visible_keywords.append((kw, ts))

            # 将所有关键词显示在同一行
            if visible_keywords:
                y_pos = 0.5  # 所有关键词在中间位置
                for i, (kw, ts) in enumerate(visible_keywords):
                    # 添加垂直指示线
                    self.ax3.axvline(ts, color='blue', linestyle='--', alpha=0.5)
                    
                    self.ax3.text(
                        ts, y_pos, f" {kw} ",
                        rotation=0,
                        color='black',
                        fontweight='bold',
                        bbox=dict(
                            facecolor='yellow', 
                            alpha=0.7, 
                            boxstyle='round,pad=0.5',
                            edgecolor='none'
                        ),
                        horizontalalignment='left',
                        verticalalignment='center'
                    )
            
            # Always show the legend in ax3
            self.ax3.legend(loc='upper left')

            self.ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
            
            # Update status text
            status = "Listening..." if keyword_detected else "Waiting for keyword"
            # Store status string for UI to display
            self.status = status
            
            # Adjust layout with consistent padding
            # Use subplots_adjust instead of tight_layout for better control
            self.fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07)

        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()

        return []  # Return empty list since we're using blit=False
    
    def kws_processing_thread(self):
        """Thread for processing Keyword Spotting with adaptive window"""
        print("KWS processing thread started")
        
        # 添加调试变量
        process_count = 0
        last_boundary = 0
        last_chunk_hash = None
        # Constants for voice detection
        AMPLITUDE_THRESHOLD = self.threshold
        SLIP_WINDOW_TIME = 0.1 # seconds
        SLIP_WINDOW_SIZE = int(SLIP_WINDOW_TIME * self.sample_rate)
        MIN_CHUNK = self.chunk_size
        MAX_CHUNK = 50 * MIN_CHUNK
        PAUSE_VOICE_THRESHOLD = self.pause_threshold  # v2.0: Use adjustable threshold
        END_CONV_THRESHOLD = self.time_to_end_conversation
        chunk_duration = SLIP_WINDOW_SIZE / self.sample_rate
        
        audio_buffer_kws = np.array([], dtype=np.int16)
        silence_timer = 0
        start_pos = 0
        speech_start_time = None  # 记录真实的语音开始时间
        search_start = start_pos + MIN_CHUNK
        # Add initial keyword status
        with self.lock:
            current_time = time.time()
            self.keyword_status_history.append((False, current_time))
        
        while not self.stop_event.is_set():
            try:
                #
                #  Get audio data from queue
                #
                
                # DEBUG: 打印当前状态
                if self.isDebug:
                    buffer_len = len(audio_buffer_kws)
                    print(f"DEBUG: Start read data from queue - buffer_len={buffer_len}, start_pos={start_pos}, find_valid_start={self.find_valid_start}")

                try:
                    audio_data, timestamp = self.audio_data_queue.get(timeout=0.1)
                    if self.isDebug:
                        print(f"DEBUG: Got audio chunk at {timestamp}, length={len(audio_data)}")
                except queue.Empty:
                    if self.isDebug:
                        print("No audio data in queue")
                        # break
                    continue
                
                # v2.0: State machine - lightweight KWS in SLEEPING state
                current_state = self.state
                if current_state == 'SLEEPING' and self.vosk_kws is not None:
                    # In SLEEPING state: only run Vosk for wake word detection
                    try:
                        result = self.vosk_kws.detect_keyword(audio_data, return_partial=False)
                        if result['detected']:
                            # Wake word detected! Switch to AWAKE state
                            with self.lock:
                                self.state = 'AWAKE'
                                self.last_speech_time = time.time()
                                self.keyword_status_history.append((True, timestamp))
                            print(f"🟢 WAKE WORD DETECTED: '{result['keyword']}' (confidence: {result['confidence']:.2f})")
                            print(f"State changed: SLEEPING → AWAKE")
                            # Add to detected keywords for visualization
                            with self.lock:
                                self.detected_keywords.append((result['keyword'], timestamp))
                    except Exception as e:
                        if self.isDebug:
                            print(f"Vosk KWS error: {e}")
                    # In SLEEPING state, skip FunASR processing
                    continue
                
                # Check for AWAKE timeout (auto sleep after no speech)
                if current_state == 'AWAKE':
                    if time.time() - self.last_speech_time > self.awake_timeout:
                        with self.lock:
                            self.state = 'SLEEPING'
                            print(f"⚫ Auto-sleep: AWAKE → SLEEPING (timeout: {self.awake_timeout}s)")
                        continue
                
                # Data is already normalized and single-channel from callback
                audio_buffer_kws = np.concatenate([audio_buffer_kws, audio_data])
                buffer_len = len(audio_buffer_kws)
                self.keyword_status_history.append((self.is_keyword_detected, timestamp))
                

                # 寻找语音开始点
                if not self.find_valid_start:
                    if self.isDebug:
                        print(f"DEBUG: Searching for speech start...")
                    # while start_pos + MIN_CHUNK <= buffer_len:
                    while start_pos + SLIP_WINDOW_SIZE <= buffer_len:
                        slip_window = audio_buffer_kws[start_pos:start_pos + SLIP_WINDOW_SIZE]
                        if slip_window.size < SLIP_WINDOW_SIZE:
                            break  # Not enough data for a window
                        
                        start_mean = np.mean(np.abs(slip_window))
                        if start_mean < AMPLITUDE_THRESHOLD:
                            timestamp = timestamp + SLIP_WINDOW_TIME  # Update timestamp for current position
                            start_pos += SLIP_WINDOW_SIZE
                            silence_timer += chunk_duration
                            with self.lock:
                                self.speech_level.append((0,timestamp))
                            # Update VAD status for this segment (no speech)
                            # with self.lock:
                            #     self.detected_vad.append((False, timestamp))
                            

                            if silence_timer >= END_CONV_THRESHOLD:
                                with self.lock:
                                    if self.is_keyword_detected:
                                        if self.isDebug:
                                            print(f"No voice detected for {END_CONV_THRESHOLD}s, resetting keyword detection")
                                        self.is_keyword_detected = False
                                        # Record keyword status change
                                        self.keyword_status_history.append((self.is_keyword_detected, timestamp))
                                silence_timer = 0  # Reset silence timer
                            continue  # continue to next WINDOW Voice CHECK
                        else:
                            self.find_valid_start = True
                            # 关键修复：计算真实的语音开始时间
                            speech_start_time = timestamp #Time stamp is end of speech
                            with self.lock:
                                self.speech_level.append((0.5,timestamp))
                            silence_timer = 0
 
                            # Find end point: search for silence within [MIN_CHUNK, MAX_CHUNK]
                            if self.isDebug:
                                print(f"DEBUG: Found speech start at start_pos={start_pos}, calculated start_time={speech_start_time}")
                            break
                    
                    #Trim buffer if no valid speech data
                    if self.isDebug:
                        print(f" len of audio_buffer is: {len(audio_buffer_kws)} before trimming")
                    audio_buffer_kws = audio_buffer_kws[start_pos:]
                    start_pos = 0
                    if len(audio_buffer_kws) == 0:
                        self.find_valid_start = False  
                        continue
                    else:
                        boundary = None                            
                        search_start = start_pos + MIN_CHUNK
                #     # trim buffer if not find valid start
                    if self.isDebug:
                        print(f" len of audio_buffer is: {len(audio_buffer_kws)} after trimming")
                #
                # Find end point: search for silence within [MIN_CHUNK, MAX_CHUNK]
                #
                if self.isDebug:
                    print(f"DEBUG: Searching for speech stop...")
                buffer_len = len(audio_buffer_kws)
                search_end = min(buffer_len, start_pos + MAX_CHUNK)
                for i in range(search_start, search_end, SLIP_WINDOW_SIZE):
                    window_end = min(i + SLIP_WINDOW_SIZE, search_end)
                    if speech_start_time is not None:
                        end_time = speech_start_time + (window_end - start_pos) / self.sample_rate
                    else:
                        # 如果 speech_start_time 为 None，使用当前时间作为合理的估计值
                        current_time = time.time()
                        end_time = current_time - (window_end - start_pos) / self.sample_rate
                        # 顺便更新speech_start_time以避免后续处理出现None
                        speech_start_time = current_time - (buffer_len / self.sample_rate)
                        print(f"修正: speech_start_time为None，已更新为估计值{speech_start_time}")
                    
                    window = audio_buffer_kws[i:window_end]
                    if window.size == 0:
                        continue
                        
                    mean_amp = np.mean(np.abs(window))
                    if mean_amp < AMPLITUDE_THRESHOLD:
                        # Found silence - this is our endpoint
                        silence_timer += SLIP_WINDOW_TIME                            
                        if silence_timer >= PAUSE_VOICE_THRESHOLD: # Speaker will pause to wait feedback or response
                            boundary = window_end
                            silence_timer = 0  # Reset silence timer
                            with self.lock:
                                self.speech_level.append((0,end_time))
                            break
                        else:
                            continue
                    else:
                        # Update VAD status (speech continues)
                        # with self.lock:
                        #     current_time = timestamp + (i - start_pos) / self.sample_rate
                        #     self.detected_vad.append((True, current_time))
                        with self.lock:
                            self.speech_level.append((0.5,end_time))
                        silence_timer = 0  # Reset silence timer
                # else:
                if boundary is None:
                    if buffer_len < MAX_CHUNK:
                        search_start = search_end # next time start from current search_end
                        continue
                    else:
                        # No silence found, use maximum chunk size
                        boundary = min(start_pos + MAX_CHUNK, buffer_len)
                
                # # Extract the active speech window
                chunk_data = audio_buffer_kws[start_pos:boundary]
                if len(chunk_data) < MIN_CHUNK:
                    boundary = None
                    self.find_valid_start = False
                    if self.isDebug:
                        print("Remaining data too short, waiting for more audio")
                    continue

                with self.lock:
                    # 确保speech_start_time不为None
                    if speech_start_time is None:
                        current_time = time.time()
                        speech_start_time = current_time - (len(chunk_data) / self.sample_rate)
                        print(f"警告: 添加到audio_buffer_speechonly前修正speech_start_time为: {speech_start_time}")
                    
                    self.audio_buffer_speechonly.append((chunk_data, speech_start_time))
                    if self.isDebug:
                        print(f"Appending {chunk_data.size} samples {chunk_data.size / self.sample_rate:.1f} seconds of speech to buffer")
                
                #
                # vad detection
                #
                if self.vad_model is not None:
                    vad_result = self.vad_model.generate(
                            audio_buffer_kws,
                            sampling_rate=self.sample_rate,
                            return_tensors="pt"
                        )
                    print(f"VAD result: {vad_result}")

                    # 处理VAD结果
                    if self.isDebug:
                        print(f"DEBUG: Processing VAD to speech ...")
                    if isinstance(vad_result, list) and len(vad_result) > 0:
                        vad_item = vad_result[0]  # 获取第一个结果
                        vad_value = vad_item.get('value', [])  # 获取value列表
                    
                        with self.lock:
                            if not vad_value:
                                # value为空列表时，标记为0 - 但确保时间戳不为None
                                if speech_start_time is not None:
                                    self.detected_vad.append((0, speech_start_time))
                                else:
                                    print("警告: speech_start_time为None，跳过添加VAD检测记录")
                            else:
                                # 对于每个检测到的语音区间
                                for segment in vad_value:
                                    start_sample, end_sample = segment
                                    # 将采样点索引转换为时间戳，但先检查speech_start_time是否为None
                                    if speech_start_time is not None:
                                        start_time = speech_start_time + start_sample / self.sample_rate
                                        end_time = speech_start_time + end_sample / self.sample_rate
                                        
                                        # 记录语音区间的起始和结束，使用0.5表示检测到语音
                                        self.detected_vad.append((0.5, start_time))
                                        self.detected_vad.append((0, end_time))
                                    else:
                                        print("警告: speech_start_time为None，无法计算VAD时间戳")
                else:
                    if self.isDebug:
                        print("VAD model not available, skipping VAD detection")

                # Process through ASR model - 仅当VAD检测的语音段足够长时才调用ASR
                if self.isDebug:
                    print(f"DEBUG: Processing speech ASR...")
                
                # 检查是否有有效的VAD结果并且语音段长度足够
                valid_speech_segment = False
                if self.vad_model is not None and 'vad_result' in locals() and isinstance(vad_result, list) and len(vad_result) > 0:
                    vad_item = vad_result[0]
                    vad_value = vad_item.get('value', [])
                    
                    # 检查是否有任何语音段长度超过500个样本点
                    for segment in vad_value:
                        start_sample, end_sample = segment
                        segment_length = end_sample - start_sample
                        if segment_length > 500:  # 只有当语音段长度超过500时才认为有效
                            valid_speech_segment = True
                            if self.isDebug:
                                print(f"Found valid speech segment with length {segment_length} samples")
                            break
                
                if self.asr_model is not None and (valid_speech_segment or self.vad_model is None):
                    try:
                        if self.isDebug:
                            print(f"Running ASR on chunk of {len(chunk_data)} samples")
                        asr_result = self.asr_model.generate(
                            input=chunk_data,
                            output_type="dict",
                            cache={},
                            language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech" "auto"
                            # language="zn" "en",  # "zn", "en", "yue", "ja", "ko", "nospeech" "auto"
                            use_itn=True,
                            batch_size_s=60,
                            merge_vad=True,  #
                            merge_length_s=15,
                            disable_log=True,
                            disable_progress_bar=True
                        )
                        
                        # Process ASR results
                        if isinstance(asr_result, list) and len(asr_result) > 0:
                            result_dict = asr_result[0]
                            if isinstance(result_dict, dict):
                                text = result_dict.get('text', '')
                            elif isinstance(result_dict, str):
                                text = result_dict
                            else:
                                text = str(result_dict)
                                
                            if text:
                                # Clean up text
                                text = re.sub(r"<\|.*?\|>", "", text)
                                text = re.sub(r"[^\w\s]", "", text)  # 去除所有标点符号
                                if self.isDebug:
                                    print(f"ASR result: {text}")
                                    if valid_speech_segment:
                                        print("Result from valid speech segment (length > 500 samples)")
                                    else:
                                        print("Result from VAD-less processing")
                                
                                # Save to detected keywords with timestamp
                                with self.lock:
                                    # kws_sample = chunk_start_sample + start_pos
                                    if self.isDebug:
                                        print(f"Appending {text} at {speech_start_time} to buffer")
                                    self.detected_keywords.append((text, speech_start_time)) #datetime.now())) timestamp is end time of detected keyword
                                    
                                    # 保存当前ASR结果，用于文件命名
                                    self.current_kws_results = [{'text': text}]
                                    if self.isDebug:
                                        print(f"Current KWS results updated: {self.current_kws_results}")

                                # Check if text contains any of our keywords
                                if not self.is_keyword_detected:
                                    for keyword in self.keywords:
                                        if keyword.lower() in text.lower():
                                            if self.isDebug:
                                                print(f"Keyword detected: {keyword}")
                                            with self.lock:
                                                self.is_keyword_detected = True
                                                # Record keyword status change with timestamp
                                                self.keyword_status_history.append((self.is_keyword_detected, speech_start_time))
                                            break
                                
                                # Check for conversation end keywords

                                end_keywords = self.stopwords if self.stopwords else ["bye", "再见", "goodbye", "结束", "停止"]
                                for end_kw in end_keywords:
                                    if end_kw.lower() in text.lower():                                        
                                        if self.isDebug:
                                            print(f"End keyword detected: {end_kw}")
                                        with self.lock:
                                            self.is_keyword_detected = False
                                            # Record keyword status change with timestamp
                                            self.keyword_status_history.append((self.is_keyword_detected, speech_start_time))
                                        break
                    except Exception as e:
                        if self.isDebug:
                            print(f"KWS inference error: {e}")
                        traceback.print_exc()
                self.find_valid_start = False # end of 1 processing

                # Move to next position
                # start_pos = boundary  # 关键：推进start_pos到boundary

                # Retain unprocessed tail
                if self.isDebug:
                    print(f" len of audio_buffer_kws: {len(audio_buffer_kws)} before trimming")
                if boundary < buffer_len:
                    if self.isDebug:
                        print(f"DEBUG: Trimming buffer from {len(audio_buffer_kws)} to {len(audio_buffer_kws) - boundary}")
                    audio_buffer_kws = audio_buffer_kws[boundary:]
                    start_pos = 0  # buffer已裁剪，start_pos归零
                else:
                    if self.isDebug:print(f"DEBUG: Clearing整个缓冲区")
                    audio_buffer_kws = np.array([], dtype=np.float32)
                start_pos = 0
                if self.isDebug:
                    print(f" len of audio_buffer_kws: {len(audio_buffer_kws)} after trimming")
            except Exception as e:
                if self.isDebug:
                    print(f"Error in KWS thread: {e}")
                traceback.print_exc()
                if self.isDebug:
                    break
            
            # 重要：检查是否需要保存录音文件（将此逻辑移出异常处理块）
            try:
                # print(f"DEBUG: 检查是否需要保存录音 - recording_done={self.recording_done}, frames={len(self.recording_frames) if self.recording_frames else 0}")
                with self.lock:
                    if self.recording_done and self.recording_frames:
                        # 如果录音已完成但尚未保存，且ASR处理可能已完成，保存录音
                        print(f"录音已完成，准备保存文件. recording_frames长度: {len(self.recording_frames)}")
                        saved_filename = self.save_recording()
                        if saved_filename:
                            print(f"KWS线程中成功保存录音到: {saved_filename}")
                        else:
                            print("KWS线程中保存录音失败")
            except Exception as e:
                print(f"Error saving recording in KWS thread: {e}")
                traceback.print_exc()

        if self.isDebug:
            print("KWS processing thread stopped")

def main():
    """Main function to run the VAD/KWS processor"""
    # Initialize processor with default parameters
    # processor = VADKWSProcessor(
    #     sample_rate=16000,
    #     chunk_size=8000,
    #     threshold=0.01, #(after normalization)
    #     silence_duration=2.0,
    #     buffer_duration=3.0,
    #     keywords=["hello", "hi panda", "hi Michael", "小爱小爱", "你好"]
    # )
    processor = VADKWSProcessor(
        sample_rate=16000,
        chunk_size=8000,
        threshold=0.005,
        channels=1,
        silence_duration=0.3,
        buffer_duration=5.0,
        keywords=["hello", "Hi panda", "hi siri"]
    )
    processor.isDebug = True
    processor_started = False

    try:
        import os
        test_wav_path = os.path.join(os.path.dirname(__file__), "Hi Panda.wav") #"test.wav")
        if os.path.exists(test_wav_path):
            import wave
            wf = wave.open(test_wav_path, 'rb')
            chunk = processor.chunk_size
            n_channels = wf.getnchannels()
            print(f"Simulating live input from {test_wav_path}... (channels={n_channels})")
            while True:  # read data into queue firstly
                data = wf.readframes(chunk)
                if not data:
                    break
                # Handle multi-channel: convert to mono by taking the first channel
                if n_channels > 1:
                    audio = np.frombuffer(data, dtype=np.int16)
                    audio = audio.reshape(-1, n_channels)
                    audio = audio[:, 0]  # take first channel
                    data = audio.astype(np.int16).tobytes()
                processor.audio_callback(data, chunk, time.time(), None)
                time.sleep(chunk / processor.sample_rate)
            wf.close()
            processor.isDebug = True
            processor.kws_processing_thread() # start thread
            # Use subplots_adjust instead of tight_layout for better control
            plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.07)
            # plt.show(block=False)
            plt.show(block=True)
            print(f"Finished simulating {test_wav_path}")
            print("Showing plot window. Close the window to exit.")

        else:
            # Start the processor for real mic input
            processor.start()
            processor_started = True
            print("Showing plot window. Close the window to exit.")
            plt.show()  # This runs the matplotlib event loop in the main thread
        # plt.show(block=True) #Not close the plot immediately, wait for user to close it
    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if processor_started:
            processor.stop()
        print("Processor stopped.")

if __name__ == "__main__":
     main()
