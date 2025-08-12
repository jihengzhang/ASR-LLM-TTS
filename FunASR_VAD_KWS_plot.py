#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR_VAD_KWS_plot.py - Real-time Voice Activity Detection and Keyword Spotting with Visualization
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
                 silence_duration=2.0, vad_interval=0.25, 
                 buffer_duration=3.0, keywords=None, stopwords="stop", time_to_end_conversation=10):
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
        self.raw_vad_history = collections.deque(maxlen=500)
        self.vad_history = np.zeros(100)    
        self.detected_keywords = collections.deque(maxlen=20)  # [(text, timestamp), ...]
        self.detected_vad = collections.deque(maxlen=50)  # [(is_speech, timestamp), ...]
        self.keyword_status_history = collections.deque(maxlen=50)  # [(is_detected, timestamp), ...]
        self.speechlevel = collections.deque(maxlen=100)  # [(is_detected, timestamp), ...]
        
        # VAD and KWS models
        # self.vad_model = None   # Remove vad_model
        self.asr_model = None
        self.last_vad_check = time.time()
        self.audio_buffer_for_vad = np.array([], dtype=np.float32)
        
        # Keywords to detect
        self.keywords = keywords or ["你好", "小艾", "开始", "小度", "小爱", "Hi Michael", "Hi Panda"]
        
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
        
        # Setup visualization
        self.setup_visualization()  # add visualization later
    
    def check_audio_devices(self):
        """Check available audio devices and select input device"""
        try:
            print("Checking audio devices...")
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            print(f"Found {numdevices} audio devices:")
            for i in range(numdevices):
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"Input Device {i}: {device_info.get('name')}")
                if device_info.get('maxOutputChannels') > 0:
                    print(f"Output Device {i}: {device_info.get('name')}")
            
            # Try to find a suitable input device
            for i in range(numdevices):
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    self.input_device_index = i
                    print(f"Using input device {i}: {device_info.get('name')}")
                    break
            
            if self.input_device_index is None:
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
                
            # Put in queue for processing
            try:
                self.audio_data_queue.put_nowait((audio_data_norm, current_time))
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
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8), 
                                                      gridspec_kw={'height_ratios': [2, 2, 1]})
        self.fig.tight_layout(pad=3.0)

        # Configure original audio waveform subplot
        self.ax1.set_title('Original Audio')
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)

        # Configure detected speech subplot
        self.ax2.set_title('Detected Speech')
        self.ax2.set_ylim(-0.5, 0.5)
        self.ax2.set_ylabel('Amplitude')
        self.ax2.grid(True)

        # Configure keywords subplot
        self.ax3.set_title('Detected Keywords')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Keywords')
        self.ax3.grid(True)
        
        # Initialize empty plot lines for update_plot to work with
        self.waveform_line, = self.ax1.plot([], [], 'gray', linewidth=0.8, label='Original')
        self.speech_line, = self.ax2.plot([], [], 'b-', linewidth=1.0, label='Speech')
        self.vad_line, = self.ax1.plot([], [], 'r-', linewidth=1.5, label='Keyword Active')
        
        # Add status indicator
        self.status_text = self.fig.text(
            0.01, 0.01, 'Status: Initializing',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )

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
                current_time = last_timestamp + (len(last_chunk) / self.sample_rate)  # Add duration of last chunk
            else:
                current_time = time.time()
            if self.running:
                current_time = time.time()

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
            filtered_speech = [
                (chunk, ts) for chunk, ts in audio_buffer_speech 
                if ts + (len(chunk) / self.sample_rate) >= start_time
            ]
            
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
            filtered_keyword_status = [
                (is_detected, ts) for is_detected, ts in keyword_status_history_copy
                if ts >= start_time
            ]
            
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
                filtered_speech_level = [
                    (level, ts) for level, ts in self.speechlevel
                    if ts >= start_time
                ]
                
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
            
            self.ax1.legend(loc='upper right')

            self.ax1.set_xlim(start_time, current_time)
            self.ax1.set_ylim(-1.0, 1.0)
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

            # Plot VAD detection results
            filtered_vad = [
                (vad_level, ts) for vad_level, ts in detected_vad_copy
                if ts >= start_time
            ]
            
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

            self.ax2.legend(loc='upper right')
            self.ax2.set_xlim(start_time, current_time)
            self.ax2.set_ylim(-1.0, 1.0)
            self.ax2.grid(True)
            self.ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

            # 更新关键词显示子图 (ax3) - now includes keyword status history
            self.ax3.clear()
            self.ax3.set_title('Detected Keywords & Keyword Status')
            self.ax3.set_xlim(start_time, current_time)
            self.ax3.set_ylim(-0.1, 1.1)
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

            # Filter and display keywords within time window
            visible_keywords = [
                (kw, ts) for kw, ts in detected_keywords_copy 
                if start_time <= ts <= current_time
            ]

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
            self.ax3.legend(loc='upper right')

            self.ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
            
            # Update status text
            status = "Listening..." if keyword_detected else "Waiting for keyword"
            self.status_text.set_text(f'Status: {status}')
            
            # Adjust layout
            self.fig.tight_layout()

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
        PAUSE_VOICE_THRESHOLD = self.silence_duration
        END_CONV_THRESHOLD = self.time_to_end_conversation
        
        audio_buffer_kws = np.array([], dtype=np.int16)
        silence_timer = 0
        start_pos = 0
        speech_start_time = None  # 记录真实的语音开始时间
        
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
                    print(f"DEBUG: Loop start - buffer_len={buffer_len}, start_pos={start_pos}, find_valid_start={self.find_valid_start}")
                
                try:
                    audio_data, timestamp = self.audio_data_queue.get(timeout=0.1)
                    if self.isDebug:
                        print(f"DEBUG: Got audio chunk at {timestamp}, length={len(audio_data)}")
                except queue.Empty:
                    if self.isDebug:
                        print("No audio data in queue")
                        break
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
                            chunk_duration = SLIP_WINDOW_SIZE / self.sample_rate
                            silence_timer += chunk_duration
                            with self.lock:
                                self.speechlevel.append((0,timestamp))
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
                                self.speechlevel.append((0.5,timestamp))
                            silence_timer = 0
                            

                            # Update VAD status (speech detected)
                            # with self.lock:
                            #     self.detected_vad.append((True, timestamp))
                                

                            # Find end point: search for silence within [MIN_CHUNK, MAX_CHUNK]
                            if self.isDebug:
                                print(f"DEBUG: Found speech start at start_pos={start_pos}, calculated start_time={speech_start_time}")
                            break
                    if not self.find_valid_start:
                        continue
                    else:
                        boundary = None                            
                        search_start = start_pos + MIN_CHUNK

                #
                # Find end point: search for silence within [MIN_CHUNK, MAX_CHUNK]
                #
                search_end = min(buffer_len, start_pos + MAX_CHUNK)
                for i in range(search_start, search_end, SLIP_WINDOW_SIZE):
                    window_end = min(i + SLIP_WINDOW_SIZE, search_end)
                    end_time = speech_start_time + (window_end - start_pos) / self.sample_rate
                    window = audio_buffer_kws[i:window_end]
                    if window.size == 0:
                        continue
                        
                    mean_amp = np.mean(np.abs(window))
                    if mean_amp < AMPLITUDE_THRESHOLD:
                        # Found silence - this is our endpoint
                        silence_timer += SLIP_WINDOW_TIME
                        
                        # Update VAD status (no speech)
                        # with self.lock:
                        #     current_time = timestamp + (i - start_pos) / self.sample_rate
                        #     self.detected_vad.append((False, current_time))
                            
                        if silence_timer >= PAUSE_VOICE_THRESHOLD: # Speaker will pause to wait feedback or response
                            boundary = window_end
                            silence_timer = 0  # Reset silence timer
                            with self.lock:
                                self.speechlevel.append((0,end_time))
                            break
                        else:
                            continue
                    else:
                        # Update VAD status (speech continues)
                        # with self.lock:
                        #     current_time = timestamp + (i - start_pos) / self.sample_rate
                        #     self.detected_vad.append((True, current_time))
                        with self.lock:
                            self.speechlevel.append((0.5,end_time))
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
                    if self.isDebug:
                        print("Remaining data too short, waiting for more audio")
                    continue

                with self.lock:
                    self.audio_buffer_speechonly.append((chunk_data, speech_start_time))
                    if self.isDebug:
                        print(f"Appending {chunk_data.size} samples {chunk_data.size / self.sample_rate:.1f} seconds of speech to buffer")
                
                #
                # vad detection
                #
                vad_result = self.vad_model.generate(
                        audio_buffer_kws,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    )
                print(f"VAD result: {vad_result}")

                # vad_result = self.vad_model.generate(audio_buffer_kws)

                # 处理VAD结果
                if isinstance(vad_result, list) and len(vad_result) > 0:
                    vad_item = vad_result[0]  # 获取第一个结果
                    vad_value = vad_item.get('value', [])  # 获取value列表
                
                    with self.lock:
                        if not vad_value:
                            # value为空列表时，标记为0
                            self.detected_vad.append((0, speech_start_time))
                        else:
                            # 对于每个检测到的语音区间
                            for segment in vad_value:
                                start_sample, end_sample = segment
                                # 将采样点索引转换为时间戳
                                start_time = speech_start_time + start_sample / self.sample_rate
                                end_time = speech_start_time + end_sample / self.sample_rate
                                
                                # 记录语音区间的起始和结束，使用0.5表示检测到语音
                                self.detected_vad.append((0.5, start_time))
                                self.detected_vad.append((0, end_time))

                # Process through ASR model
                if self.asr_model is not None:
                    try:
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
                                
                                # Save to detected keywords with timestamp
                                with self.lock:
                                    # kws_sample = chunk_start_sample + start_pos
                                    if self.isDebug:
                                        print(f"Appending {text} at {speech_start_time} to buffer")
                                    self.detected_keywords.append((text, speech_start_time)) #datetime.now())) timestamp is end time of detected keyword

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
                if boundary < buffer_len:
                    if self.isDebug:
                        print(f"DEBUG: Trimming buffer from {len(audio_buffer_kws)} to {len(audio_buffer_kws) - boundary}")
                    audio_buffer_kws = audio_buffer_kws[boundary:]
                    start_pos = 0  # buffer已裁剪，start_pos归零
                else:
                    if self.isDebug:print(f"DEBUG: Clearing整个缓冲区")
                    audio_buffer_kws = np.array([], dtype=np.float32)
                start_pos = 0
                
            except Exception as e:
                if self.isDebug:
                    print(f"Error in KWS thread: {e}")
                traceback.print_exc()
                if self.isDebug:
                    break

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
            plt.tight_layout()
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
