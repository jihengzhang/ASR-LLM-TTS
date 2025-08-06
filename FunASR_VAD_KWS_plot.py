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
import os
import time
import wave
import queue
import threading
import traceback
import argparse
from xml.parsers.expat import model
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 指定后端以避免兼容性问题
import matplotlib.pyplot as plt
import pyaudio
from datetime import datetime
from matplotlib.animation import FuncAnimation
from scipy import signal

# Check if FunASR is available
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    MODEL_PATH = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join("C:\\Users\\212597558\\.cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    model_vad_id = r"damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # VAD model name 1.6MB

    AUDIO_PATH = "test/test_vad_20250715_120521.wav"  # Input audio path
    AUDIO_PATH = r"test_2025-07-22-10-41-06.wav"
    AUDIO_PATH = r"test_music_开始声音测试.wav"
    # AUDIO_PATH = r"test.wav"
    OUTPUT_DIR = "output/segments"             # Output directory for saving speech segments

    vad_model = AutoModel(model=MODEL_PATH, model_type="vad", device="cuda", disable_update=True) # works ok
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
                 buffer_duration=3.0, keywords=None):
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
        
        # Calculate frames based on durations
        self.buffer_frames = int(buffer_duration * sample_rate / chunk_size)
        self.max_silence_frames = int(silence_duration * sample_rate / chunk_size)
        
        # Audio buffer and queues
        self.buffer = queue.Queue(maxsize=self.buffer_frames)
        self.audio_data_queue = queue.Queue(maxsize=100)
        self.vad_result_queue = queue.Queue(maxsize=100)
        self.kws_result_queue = queue.Queue(maxsize=20)
        
        # Status flags
        self.running = False
        self.recording = False
        self.speech_detected = False
        self.keyword_detected = False
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
        self.audio_buffer = np.zeros(80000)  # Changed from 160000 to 80000 (5 seconds history at 16kHz)
        self.vad_history = np.zeros(100)    # VAD result history
        self.detected_keywords = []         # List of detected keywords with timestamps
        
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
        """Stop the VAD/KWS processor"""
        print("Stopping VAD/KWS processor...")
        
        with self.lock:
            self.running = False
            self.stop_event.set()
        
        # Stop all threads
        for thread in self.threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            except SystemError:
                # Ignore SystemError exceptions during thread join
                pass
            except Exception as e:
                print(f"Error joining thread: {e}")
                traceback.print_exc()
        
        # Close audio stream safely
        try:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                except Exception:
                    pass  # Ignore if stream not open
                try:
                    self.stream.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error closing audio stream: {e}")
        
        # Close PyAudio
        if self.pyaudio:
            self.pyaudio.terminate()
        
        # Close plot
        try:
            plt.close('all')
        except Exception as e:
            print(f"Error closing plot: {e}")
        
        print("Processor stopped")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback function"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Normalize to [-1, 1] for visualization
            audio_data_norm = audio_data.astype(np.float32) / 32768.0

            # Add to buffer for visualization
            with self.lock:
                self.audio_buffer = np.append(self.audio_buffer, audio_data_norm)[-160000:]

            # Put in queue for processing (raw int16 for VAD/KWS)
            try:
                # self.audio_data_queue.put_nowait((audio_data, time.time()))
                self.audio_data_queue.put_nowait((in_data, time.time()))
            except queue.Full:
                pass  # Drop data if queue is full
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

        # Create figure with subplots - only 2 subplots now
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                                      gridspec_kw={'height_ratios': [3, 1]})
        self.fig.tight_layout(pad=3.0)

        # Configure audio waveform subplot
        self.ax1.set_title('Audio Waveform & VAD')
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)

        # Plot lines for audio data and VAD result
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1.0, label='Audio')
        self.vad_line, = self.ax1.plot([], [], 'r-', linewidth=2.0, label='VAD Activity')
        self.ax1.legend(loc='upper right')

        # Configure amplitude subplot
        self.ax2.set_title('Mean Amplitude & Keywords')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Mean Amplitude')
        self.ax2.grid(True)
        
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
                audio_buffer_copy = self.audio_buffer.copy() if self.audio_buffer.size > 0 else np.array([])
                vad_history_copy = self.vad_history.copy() if self.vad_history.size > 0 else np.array([])
                detected_keywords_copy = self.detected_keywords.copy()
                recording = self.recording
                keyword_detected = self.keyword_detected

            # 检测是否有活跃语音 - 使用平均振幅
            has_active_voice = False
            if audio_buffer_copy.size > 0:
                # 获取最近的音频数据
                recent_audio = audio_buffer_copy[-16000:]  # 最近1秒的数据
                mean_amplitude = np.mean(np.abs(recent_audio))
                has_active_voice = mean_amplitude > self.threshold

            # 只在以下情况刷新：1.有活跃语音 2.有新检测到的关键词 3.状态改变
            if not hasattr(self, '_last_buffer_len'):
                self._last_buffer_len = 0
            if not hasattr(self, '_last_keywords_len'):
                self._last_keywords_len = 0
            if not hasattr(self, '_last_state'):
                self._last_state = (False, False)
            
            current_state = (recording, keyword_detected)
            current_keywords_len = len(detected_keywords_copy)

            # 只有有活跃语音、新关键词或状态变化时才刷新
            if (not has_active_voice and 
                audio_buffer_copy.size == self._last_buffer_len and 
                current_keywords_len == self._last_keywords_len and
                current_state == self._last_state):
                # 没有变化，保持当前图表
                return self.waveform_line, self.vad_line, self.status_text
            
            # 更新跟踪变量
            self._last_buffer_len = audio_buffer_copy.size
            self._last_keywords_len = current_keywords_len
            self._last_state = current_state

            # 获取当前绝对时间作为参考点
            current_time = time.time()
            
            # 设置30秒的显示窗口
            display_time_window = 30.0  # 30秒显示窗口
            start_time = current_time - display_time_window
            
            # Use actual time as x-axis
            if audio_buffer_copy.size > 0:
                # 计算需要显示多少采样点（30秒的数据）
                samples_in_window = int(display_time_window * self.sample_rate)
                display_samples = audio_buffer_copy[-min(samples_in_window, len(audio_buffer_copy)):]

                # 调整开始时间，使其与实际显示的音频数据对齐
                start_time = current_time - len(display_samples) / self.sample_rate
                x_audio = np.linspace(start_time, current_time, len(display_samples))
                self.waveform_line.set_data(x_audio, display_samples)
                self.ax1.set_xlim(start_time, current_time)
                
                # 使用实际时间格式化x轴标签
                from datetime import datetime
                
                def format_time(x, pos):
                    return datetime.fromtimestamp(x).strftime('%H:%M:%S')
                
                self.ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

            # VAD history update
            if vad_history_copy.size > 0 and audio_buffer_copy.size > 0:
                # 调整VAD历史数据以匹配时间窗口
                vad_samples = min(len(vad_history_copy), int(display_time_window / (len(audio_buffer_copy) / self.sample_rate * len(vad_history_copy))))
                vad_display = vad_history_copy[-vad_samples:] if vad_samples > 0 else vad_history_copy
                x_vad = np.linspace(start_time, current_time, len(vad_display))
                self.vad_line.set_data(x_vad, vad_display * 0.3)

            # Calculate and plot mean amplitude
            if audio_buffer_copy.size > 0:
                window_size = int(self.sample_rate * 0.02)  # 20ms window
                stride = max(1, window_size // 2)
                mean_amplitudes = []
                time_points = []
                for i in range(0, len(display_samples) - window_size, stride):
                    window = display_samples[i:i+window_size]
                    mean_amplitudes.append(np.mean(np.abs(window)))
                    time_points.append(start_time + i / self.sample_rate)
                
                self.ax2.clear()
                self.ax2.set_title('Mean Amplitude & Keywords')
                self.ax2.set_xlim(start_time, current_time)  # 使下方图表时间与上方同步
                self.ax2.plot(time_points, mean_amplitudes, color='blue', linewidth=1)
                self.ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

                # Show all detected keywords on plot - 显示所有关键词
                if detected_keywords_copy:
                    max_amp = max(mean_amplitudes) if mean_amplitudes else 1.0
                    y_pos = max_amp * 0.7
                    
                    # 过滤出仅在当前时间窗口内的关键词
                    visible_keywords = []
                    for kw, ts in detected_keywords_copy:
                        ts_sec = ts.timestamp() if isinstance(ts, datetime) else float(ts)
                        if start_time <= ts_sec <= current_time:
                            visible_keywords.append((kw, ts_sec, y_pos))
                    
                    # 确保关键词不重叠显示
                    if visible_keywords:
                        # 简单的防重叠策略
                        min_time_gap = 1.0  # 最小1秒间隔
                        displayed = []
                        for kw, ts_sec, y in visible_keywords:
                            # 检查是否与已显示的关键词重叠
                            overlap = False
                            for d_kw, d_ts, d_y in displayed:
                                if abs(ts_sec - d_ts) < min_time_gap:
                                    overlap = True
                                    break
                            
                            # 添加垂直线标记关键词位置
                            self.ax2.axvline(ts_sec, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                            
                            # 如果重叠，调整y位置
                            y_offset = 0
                            if overlap:
                                y_offset = max_amp * 0.2  # 上移20%
                            
                            # 添加带背景的关键词文本，确保清晰可见
                            self.ax2.text(ts_sec, y + y_offset, f" {kw} ", 
                                         color='black', fontweight='bold',
                                         bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'),
                                         horizontalalignment='center',
                                         verticalalignment='center')
                            
                            displayed.append((kw, ts_sec, y + y_offset))

        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()

        return self.waveform_line, self.vad_line, self.status_text
    
    def kws_processing_thread(self):
        """Thread for processing Keyword Spotting (energy-based, no VAD)"""
        print("KWS processing thread started")
        window_size = self.chunk_size  # 使用初始化时的chunk_size
        amplitude_threshold = self.threshold
        
        # 添加无声音计时器
        silence_timer = 0
        max_silence_time = 50.0  # 5秒无声音阈值
        
        while not self.stop_event.is_set():
            try:
                # 从 audio_data_queue 获取原始音频数据
                try:
                    audio_data, timestamp = self.audio_data_queue.get(timeout=0.1)
                except queue.Empty:
                    if self.isDebug:
                        break
                    else:
                        continue

                # 跳过已检测到关键词或正在录音的情况
                with self.lock:
                    if self.keyword_detected or self.recording:
                        continue

                # 能量判据：只处理能量高于阈值的窗口
                audio_data_np = np.frombuffer(audio_data, dtype=np.int16) if isinstance(audio_data, (bytes, bytearray)) else audio_data
                audio_data_norm = audio_data_np.astype(np.float32) / 32768.0
                start_pos = 0
                found_voice = False
                while start_pos + window_size <= len(audio_data_norm):
                    window = audio_data_norm[start_pos:start_pos + window_size]
                    mean_amp = np.mean(np.abs(window))
                    if mean_amp > amplitude_threshold:
                        found_voice = True
                        silence_timer = 0  # 重置无声音计时器
                        break
                    start_pos += window_size // 2  # 可调整重叠

                if not found_voice:
                    # 更新无声音计时器
                    chunk_duration = len(audio_data_np) / self.sample_rate
                    silence_timer += chunk_duration
                    # 检查是否超过5秒无声音
                    if silence_timer >= max_silence_time:
                        with self.lock:
                            if self.keyword_detected:
                                print("No voice detected for 5 seconds, resetting keyword detection")
                                self.keyword_detected = False
                        silence_timer = 0  # 重置计时器
                    continue  # 本段无明显语音能量，跳过KWS

                # KWS/ASR模型推理
                if self.asr_model is not None:
                    try:
                        if hasattr(audio_data_np, 'tobytes'):
                            audio_data_bytes = audio_data_np.tobytes()
                        else:
                            audio_data_bytes = bytes(audio_data_np)
                        asr_result = self.asr_model.generate(
                            input=audio_data_bytes,
                            output_type="dict",
                            disable_log=True,
                            disable_progress_bar=True
                        )
                        if isinstance(asr_result, list) and len(asr_result) > 0:
                            result_dict = asr_result[0]
                            if isinstance(result_dict, dict):
                                text = result_dict.get('text', '')
                            elif isinstance(result_dict, str):
                                text = result_dict
                            else:
                                text = str(result_dict)
                            if text:
                                print(f"ASR result: {text}")
                                for keyword in self.keywords:
                                    with self.lock:
                                        self.detected_keywords.append((keyword, datetime.now()))
                                    if keyword.lower() in text.lower():
                                        print(f"Keyword detected: {keyword}")
                                        with self.lock:
                                            if not self.keyword_detected and not self.recording:
                                                self.keyword_detected = True
                                                threading.Thread(target=self.start_recording, daemon=True).start()
                                        break
                    except Exception as e:
                        print(f"KWS error: {e}")
                        traceback.print_exc()

            except Exception as e:
                print(f"Error in KWS thread: {e}")
                traceback.print_exc()

            # if self.isDebug:
            #     break  # Debug mode: exit after one iteration

def main():
    """Main function to run the VAD/KWS processor"""
    # Initialize processor with default parameters
    processor = VADKWSProcessor(
        sample_rate=16000,
        chunk_size=48000,
        threshold=0.01,
        silence_duration=2.0,
        buffer_duration=3.0,
        keywords=["hello", "computer", "system"]
    )
    processor.isDebug = True
    processor_started = False
    plt.tight_layout()
    plt.show(block=False)
    try:
        import os
        test_wav_path = os.path.join(os.path.dirname(__file__), "test.wav")
        if os.path.exists(test_wav_path):
            import wave
            wf = wave.open(test_wav_path, 'rb')
            chunk = processor.chunk_size
            print(f"Simulating live input from {test_wav_path}...")
            while True:
                data = wf.readframes(chunk)
                if not data:
                    break
                processor.audio_callback(data, chunk, None, None)
                processor.kws_processing_thread()
                time.sleep(chunk / processor.sample_rate)
            wf.close()
            print(f"Finished simulating {test_wav_path}")
            print("Showing plot window. Close the window to exit.")

        else:
            # Start the processor for real mic input
            processor.start()
            processor_started = True
            print("Showing plot window. Close the window to exit.")
            plt.show()  # This runs the matplotlib event loop in the main thread
        plt.show(block=True) #Not close the plot immediately, wait for user to close it
    
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
