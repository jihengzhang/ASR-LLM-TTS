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
        self.audio_buffer = np.zeros(160000)  # 10 seconds history for display (at 16kHz)
        self.vad_history = np.zeros(100)    # VAD result history
        self.detected_keywords = []         # List of detected keywords with timestamps
        
        # VAD and KWS models
        self.vad_model = None
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
            self.init_models()
        
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
            
            # Start processing threads
            thread_names = ['AudioInput', 'VADProcessing', 'KWSProcessing', 'Plotting']
            thread_targets = [self.audio_input_thread, self.vad_processing_thread, 
                            self.kws_processing_thread, self.plotting_thread]
            
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
                self.audio_data_queue.put_nowait((audio_data, time.time()))
            except queue.Full:
                pass  # Drop data if queue is full
        except Exception as e:
            print(f"Audio callback error: {e}")
            traceback.print_exc()

        return (None, pyaudio.paContinue)
    
    def audio_input_thread(self):
        """Thread for handling audio input"""
        print("Audio input thread started")
        
        while not self.stop_event.is_set():
            try:
                # Keep the stream alive
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in audio input thread: {e}")
                traceback.print_exc()
    
    def plotting_thread(self):
        """Thread for handling plot updates - deprecated due to matplotlib threading issues"""
        print("Plotting thread is deprecated. Use matplotlib animation instead.")
        return

    def init_models(self):
        """Initialize VAD and ASR models"""
        try:
            # Check if models directory exists locally
            vad_model_path = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            local_vad_path = os.path.join("models", vad_model_path)
            if os.path.exists(local_vad_path):
                vad_model_path = local_vad_path
                print(f"Using local VAD model: {local_vad_path}")
            else:
                print(f"Using remote VAD model: {vad_model_path}")
                
            print("Loading VAD model...")
            self.vad_model = AutoModel(
                model=vad_model_path,
                model_revision="v2.0.4",  # 添加模型版本以确保兼容性
                model_type="vad",
                device="cuda" if self.check_cuda_available() else "cpu",
                disable_update=True
            )
            
            print("Loading ASR model for keyword spotting...")
            self.asr_model = AutoModel(
                model="paraformer-zh",  # 使用更简单的模型名称
                model_revision="v2.0.4",  # 添加模型版本以确保兼容性
                device="cuda" if self.check_cuda_available() else "cpu",
                output_type="dict",
                disable_update=True,
                disable_log=True,  # 禁用日志以减少干扰
                disable_progress_bar=True  # 禁用进度条以减少干扰
            )
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()  # 打印详细错误信息
            print("Continuing without models - will use amplitude-based detection only")
            self.vad_model = None
            self.asr_model = None
    
    def setup_visualization(self):
        """Set up the visualization plots"""
        # 设置matplotlib后端参数以提高兼容性
        plt.rcParams['backend'] = 'TkAgg'
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
        
        # Create figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Configure audio waveform subplot
        self.ax1.set_title('Audio Waveform & VAD')
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)
        
        # Plot lines for audio data and VAD result
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1.0, label='Audio')
        self.vad_line, = self.ax1.plot([], [], 'r-', linewidth=2.0, label='VAD Activity')
        self.ax1.legend(loc='upper right')
        
        # Configure spectrogram subplot
        self.ax2.set_title('Spectrogram & Keywords')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Frequency (Hz)')
        
        # Initialize spectrogram
        self.spec_img = self.ax2.imshow(
            np.zeros((128, 100)), 
            aspect='auto',
            origin='lower',
            extent=[0, 10, 0, self.sample_rate/2],
            cmap='viridis'
        )
        
        # Add text for keyword display
        self.keyword_text = self.ax2.text(
            0.02, 0.95, 'No keywords detected', 
            transform=self.ax2.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Add status indicator
        self.status_text = self.fig.text(
            0.02, 0.01, 'Status: Initializing', 
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Make plot interactive
        plt.ion()
        self.fig.canvas.draw()
        
        # Create animation with更安全的参数
        self.animation = FuncAnimation(
            self.fig, self.update_plot, 
            interval=100,  # Update every 100ms
            blit=False,
            repeat=True,
            cache_frame_data=False  # 禁用帧缓存以避免内存问题
        )
        
        # Set up close event handler
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
            # Update audio waveform
            with self.lock:
                audio_buffer_copy = self.audio_buffer.copy() if self.audio_buffer.size > 0 else np.array([])
                vad_history_copy = self.vad_history.copy() if self.vad_history.size > 0 else np.array([])
                detected_keywords_copy = self.detected_keywords.copy()
                
                # 获取状态
                recording = self.recording
                keyword_detected = self.keyword_detected
            
            if audio_buffer_copy.size > 0:
                # Show last 10 seconds (160000 samples)
                display_samples = audio_buffer_copy[-160000:]
                x_audio = np.linspace(0, 10, len(display_samples))  # 10 seconds
                self.waveform_line.set_data(x_audio, display_samples)
                self.ax1.set_xlim(0, 10)
            
            # Update VAD history
            if vad_history_copy.size > 0 and audio_buffer_copy.size > 0:
                x_vad = np.linspace(0, len(audio_buffer_copy[-1000:]), len(vad_history_copy))
                self.vad_line.set_data(x_vad, vad_history_copy * 0.3)  # Scale to fit in the same plot
            
            # Update spectrogram
            if audio_buffer_copy.size > 1000:
                try:
                    # Calculate spectrogram
                    f, t, Sxx = signal.spectrogram(
                        audio_buffer_copy[-1000:],
                        fs=self.sample_rate,
                        nperseg=256,
                        noverlap=128
                    )
                    # Update spectrogram image
                    self.spec_img.set_array(10 * np.log10(Sxx + 1e-10))
                    self.spec_img.set_extent([0, t[-1], 0, f[-1]])
                except Exception as e:
                    print(f"Warning: spectrogram update failed: {e}")
            
            # Update keyword display
            if detected_keywords_copy:
                # Show last 3 detected keywords
                kw_text = "Detected Keywords:\n"
                for i, (kw, ts) in enumerate(detected_keywords_copy[-3:]):
                    kw_text += f"{ts.strftime('%H:%M:%S')}: {kw}\n"
                self.keyword_text.set_text(kw_text)
            else:
                self.keyword_text.set_text("No keywords detected")
            
            # Update status
            status = "Recording" if recording else "Listening"
            if keyword_detected:
                status += " (Keyword detected)"
            self.status_text.set_text(f"Status: {status}")
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()
        
        return self.waveform_line, self.vad_line, self.spec_img, self.keyword_text, self.status_text
    
    def vad_processing_thread(self):
        """Thread for processing VAD"""
        print("VAD processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from queue with timeout
                try:
                    audio_data, timestamp = self.audio_data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Skip if recording already started
                with self.lock:
                    if self.recording:
                        continue

                # Process with VAD model if available
                vad_segments = []
                is_speech = False

                if self.vad_model is not None and len(audio_data) >= 1600:
                    try:
                        # Ensure audio length is multiple of 400 samples (25ms)
                        valid_length = (len(audio_data) // 400) * 400
                        if valid_length > 0:
                            valid_audio = audio_data[:valid_length]
                            # Convert numpy array to bytes to avoid PY_SSIZE_T_CLEAN error
                            if hasattr(valid_audio, 'tobytes'):
                                valid_audio_bytes = valid_audio.tobytes()
                            else:
                                valid_audio_bytes = bytes(valid_audio)
                            # Process with VAD model
                            vad_result = self.vad_model.generate(
                                input=valid_audio_bytes,
                                model_type="vad",
                                output_type="dict"
                            )

                            # Parse results
                            if isinstance(vad_result, list) and len(vad_result) > 0:
                                result_dict = vad_result[0]
                                if isinstance(result_dict, dict) and 'value' in result_dict:
                                    vad_segments = result_dict['value']
                                    is_speech = len(vad_segments) > 0

                                    # Print segments info if speech detected
                                    if is_speech:
                                        print(f"Speech detected with {len(vad_segments)} segments")
                    except Exception as e:
                        print(f"VAD error: {e}")
                        traceback.print_exc()
                        # Fallback to amplitude-based detection
                        is_speech = np.abs(audio_data).mean() > self.threshold
                else:
                    # Fallback to amplitude-based detection
                    is_speech = np.abs(audio_data).mean() > self.threshold

                # 测试KWS（ASR）模型，直接在此处调用并打印输出
                if self.asr_model is not None:
                    try:
                        # Debug: print audio_data info
                        print(f"[ASR Debug] audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                        # Convert numpy int16 array to bytes
                        audio_data_bytes = audio_data.astype(np.int16).tobytes()
                        # Call ASR model with sample rate if needed
                        asr_result = self.asr_model.generate(
                            input=audio_data_bytes,
                            sample_rate=self.sample_rate,
                            output_type="dict",
                            disable_log=True,
                            disable_progress_bar=True
                        )
                        # Parse and print result
                        if isinstance(asr_result, list) and len(asr_result) > 0:
                            result_dict = asr_result[0]
                            if isinstance(result_dict, dict):
                                text = result_dict.get('text', '')
                            elif isinstance(result_dict, str):
                                text = result_dict
                            else:
                                text = str(result_dict)
                            print(f"[KWS Test] ASR result: {text}")
                    except Exception as e:
                        print(f"[KWS Test] ASR error: {e}")
                        traceback.print_exc()

                # Update VAD history for visualization
                with self.lock:
                    self.vad_history = np.append(self.vad_history[1:], float(is_speech))

                # If speech detected, process with KWS
                with self.lock:
                    keyword_not_detected = not self.keyword_detected

                if is_speech and keyword_not_detected:
                    try:
                        self.vad_result_queue.put_nowait((audio_data, timestamp, vad_segments))
                    except queue.Full:
                        pass

            except Exception as e:
                print(f"Error in VAD thread: {e}")
                traceback.print_exc()

    def kws_processing_thread(self):
        """Thread for processing Keyword Spotting"""
        print("KWS processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio data from VAD queue with timeout
                try:
                    audio_data, timestamp, vad_segments = self.vad_result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip if keyword already detected
                with self.lock:
                    if self.keyword_detected or self.recording:
                        continue
                
                # Process with ASR model if available
                if self.asr_model is not None:
                    try:
                        # Convert numpy array to bytes before passing to ASR model
                        if hasattr(audio_data, 'tobytes'):
                            audio_data_bytes = audio_data.tobytes()
                        else:
                            audio_data_bytes = bytes(audio_data)
                        # Process with ASR model
                        asr_result = self.asr_model.generate(
                            input=audio_data_bytes,
                            output_type="dict",
                            disable_log=True,
                            disable_progress_bar=True
                        )
                        
                        # Parse results
                        if isinstance(asr_result, list) and len(asr_result) > 0:
                            result_dict = asr_result[0]
                            
                            # Extract text based on result format
                            if isinstance(result_dict, dict):
                                text = result_dict.get('text', '')
                            elif isinstance(result_dict, str):
                                text = result_dict
                            else:
                                text = str(result_dict)
                            
                            if text:
                                print(f"ASR result: {text}")
                                
                                # Check for keywords
                                for keyword in self.keywords:
                                    if keyword.lower() in text.lower():
                                        print(f"Keyword detected: {keyword}")
                                        
                                        # Add to detected keywords list with timestamp
                                        with self.lock:
                                            self.detected_keywords.append((keyword, datetime.now()))
                                        
                                        # Set keyword detected flag and start recording
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

def main():
    """Main function to run the VAD/KWS processor"""
    # Initialize processor with default parameters
    processor = VADKWSProcessor(
        sample_rate=16000,
        chunk_size=1600,
        threshold=0.01,
        silence_duration=2.0,
        buffer_duration=3.0,
        keywords=["hello", "computer", "system"]
    )
    
    processor_started = False
    try:
        # Start the processor
        processor.start()
        processor_started = True
        
        # Show the plot window in the main thread (this is the key fix)
        print("Showing plot window. Close the window to exit.")
        plt.show()  # This runs the matplotlib event loop in the main thread
        
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
