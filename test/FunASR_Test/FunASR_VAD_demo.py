import os
import sys
import matplotlib
print(f"Original matplotlib backend: {matplotlib.get_backend()}")
# Set the backend to TkAgg for Windows compatibility
matplotlib.use('TkAgg')
print(f"Changed matplotlib backend to: {matplotlib.get_backend()}")

# Continue with the rest of the imports
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
import time
import wave
import threading
import queue
import numpy as np
import pyaudio
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for color output support
colorama.init()

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} FunASR module loaded")
except ImportError:
    FUNASR_AVAILABLE = False
    print(f"{Fore.RED}[WARNING]{Style.RESET_ALL} FunASR module not found, will use basic amplitude detection")

class KeywordActivatedRecorder:
    def __init__(self, sample_rate=16000, chunk_size=16000, channels=1, # chunk size 0.1s
                 format=pyaudio.paInt16, threshold=0.001,
                 silence_duration=2.0, min_speech_duration=1.0,
                 buffer_duration=5.0):
        """
        Initialize recorder
        
        Parameters:
            sample_rate: Sample rate (Hz)
            chunk_size: Audio chunk size for each processing
            channels: Number of channels
            format: Audio format
            threshold: Audio activation threshold (for backup detection)
            silence_duration: Silence duration before stopping recording (seconds)
            min_speech_duration: Minimum speech duration to be considered valid speech (seconds)
            buffer_duration: Pre-buffer duration (seconds) to capture audio before keywords
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        # Calculate pre-buffer size
        self.buffer_frames = int(buffer_duration * sample_rate / chunk_size)
        self.buffer = queue.Queue(maxsize=self.buffer_frames)
        
        # Create visualization data queue and control flags
        self.visualization_queue = queue.Queue(maxsize=100)
        self.visualization_running = False
        self.visualization_thread = None
        
        # Recording status
        self.recording = False
        self.speech_detected = False
        self.silence_counter = 0
        self.speech_counter = 0
        self.max_silence_frames = int(silence_duration * sample_rate / chunk_size)
        self.min_speech_frames = int(min_speech_duration * sample_rate / chunk_size)
        
        # Audio objects
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.input_device_index = None  # Will be set by check_audio_devices
        
        # Check audio devices
        self.check_audio_devices()
        
        # Thread lock
        self.lock = threading.Lock()
        
        # FunASR model initialization
        self.vad_model = None
        self.asr_model = None
        self.use_funasr = FUNASR_AVAILABLE
        
        if self.use_funasr:
            self.init_funasr_models()  #load models
        else:
            print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Using basic amplitude detection algorithm")
        
        # Keyword detection related variables
        self.keyword_buffer = queue.Queue(maxsize=int(4.0 * sample_rate / chunk_size))  # 4-second keyword buffer
        self.keyword_detected = False
        self.speech_segments = []  # 存储检测到的语音段
        self.keyword_candidates = ["你好", "小助手", "开始录音", "录音开始", "小爱", "小度"]
        self.current_keyword = "开始录音"  # 默认关键词
        
        # VAD related variables
        self.vad_cache = {}
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.last_vad_check = time.time()
        self.vad_check_interval = 1.0  # Reduced to 1 second for more responsive detection
        # Use 16000 samples (1 second) window, ensure it's a multiple of 400
        # self.vad_window_size = 16000  # 1 second window
        self.vad_window_size = 16000  # 0.5 second window
        self.audio_buffer_for_vad = np.array([], dtype=np.float32)
        self.min_vad_amplitude = 0.001  # Minimum amplitude to trigger VAD processing
        
        # Setup visualization
        self.setup_live_plot()
        
        # Start visualization thread
        self.start_visualization_thread()
        
        print(f"{Fore.YELLOW}[关键词]{Style.RESET_ALL} 当前激活关键词: '{self.current_keyword}'")
        print(f"{Fore.CYAN}[提示]{Style.RESET_ALL} 使用 FunASR VAD+ASR 进行关键词检测")
    
    def init_funasr_models(self):
        """Initialize FunASR VAD and ASR models"""
        try:
            # Define local model paths
            vad_model_path = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
            
            print(f"{Fore.CYAN}[INIT]{Style.RESET_ALL} Loading FunASR VAD model from local path...")
            # Simplest initialization, similar to debug file but with local model path
            self.vad_model = AutoModel(
                model=vad_model_path,
                disable_update=True,
                model_type="vad"
            )
            print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} FunASR VAD model loaded successfully from {vad_model_path}")
            
            print(f"{Fore.CYAN}[INIT]{Style.RESET_ALL} Loading FunASR ASR model...")
            # Load ASR model for keyword recognition - using same params as debug file
            self.asr_model = AutoModel(
                model="paraformer-zh",
                disable_update=True,
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} FunASR ASR model loaded successfully")
            
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} FunASR model loading failed: {e}")
            self.use_funasr = False
            self.vad_model = None
            self.asr_model = None

    def start(self):
        """Start listening to audio"""
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Starting audio monitoring...")
        
        # Test microphone input first
        print(f"{Fore.CYAN}[TEST]{Style.RESET_ALL} Testing microphone input...")
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print(f"{Fore.GREEN}[READY]{Style.RESET_ALL} Voice activation system started, waiting for speech...")
            
            # Test a few chunks to see if we're getting audio
            print(f"{Fore.CYAN}[MONITOR]{Style.RESET_ALL} Monitoring first few audio chunks...")
            
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to start audio stream: {e}")
            print(f"{Fore.YELLOW}[SUGGESTION]{Style.RESET_ALL} Try running Windows Sound settings to check microphone permissions")
        
    def stop(self):
        """Stop listening and close resources"""
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Shutting down audio system...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        
        # Stop visualization thread
        if hasattr(self, 'visualization_running'):
            self.visualization_running = False
            if self.visualization_thread and self.visualization_thread.is_alive():
                print(f"{Fore.CYAN}[VISUALIZATION]{Style.RESET_ALL} Waiting for visualization thread to terminate...")
                try:
                    self.visualization_thread.join(timeout=2.0)
                except Exception as e:
                    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Could not join visualization thread: {e}")
        
        # Close the visualization plot if it's open
        if hasattr(self, 'live_plot_enabled') and self.live_plot_enabled:
            try:
                plt.close(self.fig)
                print(f"{Fore.CYAN}[VISUALIZATION]{Style.RESET_ALL} Closed visualization window")
            except Exception as e:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Error closing visualization: {e}")
        
        print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Audio system closed")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function to process input audio data"""
        # Convert binary data to array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Debug: Check if we're getting actual audio data
        non_zero_samples = np.count_nonzero(audio_data)
        max_amplitude = np.max(np.abs(audio_data))
        
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 1
            
        # Print debug info only for the first few chunks to avoid console spam
        # if self.debug_counter <= 3:
        #     print(f"\n{Fore.CYAN}[AUDIO DEBUG]{Style.RESET_ALL} Chunk {self.debug_counter}: Non-zero samples: {non_zero_samples}/{len(audio_data)}, Max amplitude: {max_amplitude}")
        #     print(f"Sample data: {audio_data[:10]}...{audio_data[-10:]}")
        
        # Calculate audio amplitude (normalized)
        amplitude = np.abs(audio_data).mean() / 32767.0
        
        # Add to pre-buffer
        try:
            if self.buffer.full():
                self.buffer.get_nowait()
            self.buffer.put_nowait(in_data)
        except queue.Full:
            pass
        
        # VAD detection
        is_speech = amplitude > self.threshold  # Default use amplitude
        
        # If FunASR VAD is available, use it for detection
        if self.vad_model is not None:
            try:
                # Convert binary data to float32 and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Add to VAD specific buffer
                self.audio_buffer_for_vad = np.concatenate([self.audio_buffer_for_vad, audio_float])
                print(f"Length of audio buffer for VAD: {len(self.audio_buffer_for_vad)} samples")
                # Periodic VAD detection
                current_time = time.time()
                # Also trigger VAD if we have significant audio activity
                has_significant_audio = np.mean(np.abs(audio_float)) > self.min_vad_amplitude
                
                if (current_time - self.last_vad_check >= self.vad_check_interval and not self.recording) or \
                   (has_significant_audio and current_time - self.last_vad_check >= 0.5):  # More frequent checks when audio is present
                    self.last_vad_check = current_time
                    
                    # Check if there's enough audio data for VAD
                    if len(self.audio_buffer_for_vad) >= self.vad_window_size:
                        # Ensure window size is exact multiple of 400
                        window_samples = (self.vad_window_size // 400) * 400
                        vad_window = self.audio_buffer_for_vad[:window_samples]
                        
                        # Debug: Check the audio data going into VAD
                        # vad_audio_stats = {
                        #     'length': len(vad_window),
                        #     'max_val': np.max(np.abs(vad_window)),
                        #     'mean_val': np.mean(np.abs(vad_window)),
                        #     'non_zero_count': np.count_nonzero(vad_window),
                        #     'shape': vad_window.shape
                        # }
                        # print(f"\n{Fore.CYAN}[VAD DEBUG]{Style.RESET_ALL} Audio stats: {vad_audio_stats}")
                        
                        try:
                            # Use VAD model for speech detection with matching params to debug file
                            vad_result = self.vad_model.generate(
                                input=vad_window  #ok
                                # input=vad_window.reshape(1, -1),
                                # chunk_size=400   # Specify chunk size
                            )
                            print(f"\n{Fore.CYAN}[VAD RESULT]{Style.RESET_ALL} Raw VAD result: {vad_result}")
                            
                            # Check VAD result
                            if isinstance(vad_result, list) and len(vad_result) > 0:
                                result_dict = vad_result[0]
                                
                                if isinstance(result_dict, dict) and 'value' in result_dict:
                                    vad_segments = result_dict['value']
                                    print(f"\n{Fore.YELLOW}[VAD SEGMENTS]{Style.RESET_ALL} Found {len(vad_segments)} segments: {vad_segments}")
                                    
                                    # Add visualization of audio and VAD results
                                    if len(vad_segments) > 0:
                                        # Save audio sample with speech
                                        self.save_audio_sample(vad_window, "speech_detected")
                                        pass
                                        # Visualize in a separate thread to avoid blocking
                                        # threading.Thread(
                                        #     target=self.visualize_audio_vad,
                                        #     args=(vad_window, self.sample_rate, vad_segments),
                                        #     daemon=True
                                        # ).start()
                                    
                                    # If speech segments are detected, process them
                                    if vad_segments and not self.recording and not self.keyword_detected:
                                        print(f"\n{Fore.GREEN}[VAD SUCCESS]{Style.RESET_ALL} Processing {len(vad_segments)} speech segments")
                                        
                                        # Process each segment
                                        for i, segment in enumerate(vad_segments):
                                            if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                                                start_time, end_time = segment[0], segment[1]
                                                duration = end_time - start_time
                                                print(f"\n{Fore.CYAN}[SEGMENT {i+1}]{Style.RESET_ALL} Time: {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)")
                                                
                                                # Convert time units to sample points
                                                start_sample = int(start_time * self.sample_rate)
                                                end_sample = int(end_time * self.sample_rate)
                                                
                                                # Check index range
                                                if start_sample < 0:
                                                    start_sample = 0
                                                if end_sample > len(vad_window):
                                                    end_sample = len(vad_window)
                                                
                                                # Extract audio segment and perform keyword recognition
                                                if end_sample > start_sample:
                                                    segment_audio = vad_window[start_sample:end_sample]
                                                    if len(segment_audio) >= 1600:  # At least 100ms
                                                        print(f"\n{Fore.BLUE}[PROCESSING]{Style.RESET_ALL} Sending segment {i+1} to ASR (length: {len(segment_audio)} samples)")
                                                        # Use thread for asynchronous keyword recognition
                                                        # threading.Thread(
                                                        #     target=self.recognize_keyword,
                                                        #     args=(segment_audio,),
                                                        #     daemon=True
                                                        # ).start()
                                                        pass
                                                        
                                                        # Check amplitude, if high enough also consider as speech
                                                        segment_amplitude = np.abs(segment_audio).mean()
                                                        if segment_amplitude > self.threshold * 1.5:
                                                            is_speech = True
                                                    else:
                                                        print(f"\n{Fore.YELLOW}[SKIP]{Style.RESET_ALL} Segment {i+1} too short: {len(segment_audio)} samples")
                                            else:
                                                print(f"\n{Fore.YELLOW}[VAD]{Style.RESET_ALL} Invalid speech segment format: {segment}")
                                    elif not vad_segments:
                                        print(f"\n{Fore.BLUE}[VAD]{Style.RESET_ALL} No speech segments detected in this window")
                                else:
                                    print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} 'value' key not found in result_dict or result_dict is not a dict")
                            else:
                                print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} Invalid VAD result format")
                        except Exception as vad_error:
                            print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} {vad_error}")
                            # If VAD processing fails, fall back to amplitude-based detection
                            is_speech = amplitude > self.threshold
                            
                            # If amplitude is high enough, also try direct ASR
                            if amplitude > 0.03 and not self.recording and not self.keyword_detected:
                                # Try ASR keyword recognition with recent audio segment
                                recent_audio = self.audio_buffer_for_vad[-16000:] if len(self.audio_buffer_for_vad) > 16000 else self.audio_buffer_for_vad
                                if len(recent_audio) >= 8000:  # At least 0.5 seconds
                                    # threading.Thread(
                                    #     # target=self.recognize_keyword,
                                    #     args=(recent_audio,),
                                    #     daemon=True
                                    # ).start()
                                    pass
                        
                        # Keep part of audio as context for next detection
                        overlap_size = window_samples // 3  # Keep 33% overlap
                        self.audio_buffer_for_vad = self.audio_buffer_for_vad[-overlap_size:]
                    
                    # Limit buffer size to prevent excessive memory usage
                    max_buffer_size = self.vad_window_size * 2  # Keep at most 2 windows of data
                    if len(self.audio_buffer_for_vad) > max_buffer_size:
                        self.audio_buffer_for_vad = self.audio_buffer_for_vad[-max_buffer_size:]
                        
            except Exception as e:
                print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} {e}")
                # Fall back to amplitude detection
                is_speech = amplitude > self.threshold
        else:
            # If FunASR is not available, use amplitude detection
            is_speech = amplitude > self.threshold
        
        # Add data to visualization queue instead of directly updating
        try:
            # Use put_nowait to avoid blocking the audio callback
            pass
            # if self.visualization_running:
            #     self.visualization_queue.put_nowait((audio_data.copy(), is_speech))
        except queue.Full:
            # Queue is full, just skip this update
            pass
        
        # State logic processing
        with self.lock:
            if is_speech:
                self.speech_counter += 1
                self.silence_counter = 0
                
                # Start recording after keyword activation - enhanced activation logic
                if self.keyword_detected and not self.recording:
                    if self.speech_counter >= self.min_speech_frames:
                        print(f"\n{Fore.GREEN}[RECORDING]{Style.RESET_ALL} Keyword '{self.current_keyword}' activated - Start recording")
                        self.start_recording()
                    else:
                        print(f"\r{Fore.YELLOW}[WAITING]{Style.RESET_ALL} Waiting for sufficient speech duration: {self.speech_counter}/{self.min_speech_frames}", end='', flush=True)
            else:
                self.silence_counter += 1
                
                # If silence is long enough and recording, stop recording
                if self.silence_counter >= self.max_silence_frames and self.recording:
                    self.stop_recording()
                    
                # If speech is not long enough, reset speech counter
                if not self.recording and self.speech_counter < self.min_speech_frames:
                    self.speech_counter = 0
        
        return (in_data, pyaudio.paContinue)
    
    def process_vad(self, audio_data):
        """Process audio data with VAD model to detect speech segments
        
        Args:
            audio_data: Audio data as float32 numpy array, normalized to [-1.0, 1.0]
            
        Returns:
            tuple: (is_speech, vad_segments)
                is_speech: Boolean indicating if speech was detected
                vad_segments: List of speech segments detected by VAD, each as [start_time, end_time]
        """
        # Default to amplitude-based detection
        is_speech = np.mean(np.abs(audio_data)) > self.threshold
        vad_segments = []
        
        # If VAD model is not available, return amplitude-based result
        if self.vad_model is None:
            return is_speech, vad_segments
            
        try:
            # Debug: Check the audio data going into VAD
            # vad_audio_stats = {
            #     'length': len(audio_data),
            #     'max_val': np.max(np.abs(audio_data)),
            #     'mean_val': np.mean(np.abs(audio_data)),
            #     'non_zero_count': np.count_nonzero(audio_data),
            #     'shape': audio_data.shape
            # }
            # print(f"\n{Fore.CYAN}[VAD DEBUG]{Style.RESET_ALL} Audio stats: {vad_audio_stats}")
            
            # Ensure audio data is a multiple of 400 samples (required by FunASR VAD)
            if len(audio_data) >= 400:
                # Truncate to a multiple of 400
                valid_length = (len(audio_data) // 400) * 400
                valid_audio = audio_data[:valid_length]
                
                # Use VAD model for speech detection
                vad_result = self.vad_model.generate(
                    input=valid_audio.reshape(1, -1)
                    # chunk_size=400  # Specify chunk size
                )
                print(f"\n{Fore.CYAN}[VAD RESULT]{Style.RESET_ALL} Raw VAD result: {vad_result}")
            else:
                print(f"\n{Fore.YELLOW}[VAD WARNING]{Style.RESET_ALL} Audio data too short for VAD processing: {len(audio_data)} samples")
                return is_speech, vad_segments
            # Check VAD result
            if isinstance(vad_result, list) and len(vad_result) > 0:
                result_dict = vad_result[0]
                
                if isinstance(result_dict, dict) and 'value' in result_dict:
                    vad_segments = result_dict['value']
                    print(f"\n{Fore.YELLOW}[VAD SEGMENTS]{Style.RESET_ALL} Found {len(vad_segments)} segments: {vad_segments}")
                    
                    # Add visualization of audio and VAD results
                    if len(vad_segments) > 0:
                        threading.Thread(
                            target=self.visualize_audio_vad,
                            args=(audio_data.copy(), self.sample_rate, vad_segments),
                            daemon=True
                        ).start()
                    
                    # If any speech segments detected, consider this as speech
                    is_speech = len(vad_segments) > 0
                else:
                    print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} 'value' key not found in result_dict or result_dict is not a dict")
            else:
                print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} Invalid VAD result format")
        except Exception as vad_error:
            print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} {vad_error}")
            # Fall back to amplitude-based detection
            is_speech = np.mean(np.abs(audio_data)) > self.threshold
            
        return is_speech, vad_segments

    def process_vad_segments(self, vad_segments, audio_data):
        """Process VAD detected speech segments (simplified, mainly handled by recognize_keyword)"""
        # This method is now mainly used for debug information
        for segment in vad_segments:
            try:
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    start_time, end_time = segment[0], segment[1]
                    print(f"\n{Fore.CYAN}[VAD SEGMENT]{Style.RESET_ALL} Time: {start_time:.2f}s - {end_time:.2f}s")
            except Exception as e:
                print(f"\n{Fore.YELLOW}[VAD SEGMENT WARNING]{Style.RESET_ALL} Error processing segment info: {e}")

    def recognize_keyword(self, audio_segment):
        """Use ASR to recognize keywords in speech segments"""
        if self.asr_model is None or self.keyword_detected or self.recording:
            return
            
        try:
            # Ensure audio segment format is correct
            if len(audio_segment.shape) == 1:
                audio_input = audio_segment.reshape(1, -1)
            else:
                audio_input = audio_segment
            
            # Use ASR model for speech recognition
            asr_result = self.asr_model.generate(
                input=audio_input,
                cache={},
                language="zh"
            )
            
            # Parse recognition result
            if isinstance(asr_result, list) and len(asr_result) > 0:
                result_dict = asr_result[0]
                if isinstance(result_dict, dict):
                    recognized_text = result_dict.get('text', '').strip()
                    
                    if recognized_text:
                        print(f"\n{Fore.BLUE}[RECOGNITION]{Style.RESET_ALL} Speech recognition result: '{recognized_text}'")
                        
                        # Check if contains keywords
                        for keyword in self.keyword_candidates:
                            if keyword in recognized_text:
                                with self.lock:
                                    if not self.keyword_detected and not self.recording:
                                        print(f"\n{Fore.GREEN}[KEYWORD]{Style.RESET_ALL} Detected keyword '{keyword}' - Activated")
                                        self.keyword_detected = True
                                        self.current_keyword = keyword
                                        self.speech_counter = max(self.speech_counter, self.min_speech_frames // 2)  # Speed up activation
                                        return
                    else:
                        print(f"\n{Fore.YELLOW}[ASR]{Style.RESET_ALL} Recognition result is empty")
            
        except Exception as e:
            print(f"\n{Fore.RED}[ASR ERROR]{Style.RESET_ALL} Speech recognition failed: {e}")
    
    def start_recording(self):
        """Start recording"""
        with self.lock:
            if not self.recording:
                self.recording = True
                self.frames = []
                
                # Add all frames from pre-buffer
                while not self.buffer.empty():
                    self.frames.append(self.buffer.get())
                
                print(f"\n{Fore.GREEN}[RECORDING]{Style.RESET_ALL} Keyword activation successful - Start recording...")
                print(f"{Fore.CYAN}[STATUS]{Style.RESET_ALL} Please continue speaking, auto-stop after {self.silence_duration} seconds of silence")
                
                # Reset counters and status
                self.silence_counter = 0
                # Clear keyword buffer for next detection
                while not self.keyword_buffer.empty():
                    try:
                        self.keyword_buffer.get_nowait()
                    except queue.Empty:
                        break
    
    def stop_recording(self):
        """Stop recording and save file"""
        with self.lock:
            if self.recording:
                self.recording = False
                print(f"\n{Fore.YELLOW}[RECORDING]{Style.RESET_ALL} Silence detected - Stop recording")
                
                # If there are enough frames, save recording
                if len(self.frames) > 0:
                    threading.Thread(target=self.save_recording).start()
                
                # Reset counters and status
                self.speech_counter = 0
                self.speech_detected = False
                self.keyword_detected = False  # Reset keyword status for next detection
                
                print(f"{Fore.CYAN}[WAITING]{Style.RESET_ALL} Waiting for next keyword '{self.current_keyword}'...")
    
    def save_recording(self):
        """Save recording as WAV file"""
        if not self.frames:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No recording data to save")
            return
            
        # Ensure recordings directory exists
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("recordings", f"recording_{timestamp}.wav")
        
        print(f"{Fore.BLUE}[SAVING]{Style.RESET_ALL} Saving recording to: {filename}")
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
             
        duration = len(self.frames) * self.chunk_size / self.sample_rate
        file_size = os.path.getsize(filename) / 1024  # KB
        
        print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Recording saved! File: {filename}")
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Recording duration: {duration:.2f} seconds, File size: {file_size:.2f} KB")
        
        # Clear frames
        self.frames = []

    def process_chunk(self, in_data):
        """Process single audio chunk (for non-callback mode)"""
        if self.recording:
            self.frames.append(in_data)
    
    def check_audio_devices(self):
        """Check available audio devices"""
        device_count = self.pyaudio.get_device_count()
        print(f"{Fore.BLUE}[DEVICES]{Style.RESET_ALL} Detected {device_count} audio devices:")
        
        input_devices = []
        for i in range(device_count):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']} (Input channels: {info['maxInputChannels']}, Sample rate: {info['defaultSampleRate']})")
            except Exception as e:
                continue
        
        if not input_devices:
            raise Exception("No available audio input devices found")
        
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            print(f"{Fore.GREEN}[DEFAULT]{Style.RESET_ALL} Using default input device: {default_input['name']} (Index: {default_input['index']})")
            self.input_device_index = default_input['index']
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Cannot get default input device: {e}")
            # Use first available input device
            if input_devices:
                self.input_device_index = input_devices[0][0]
                print(f"{Fore.YELLOW}[FALLBACK]{Style.RESET_ALL} Using first available device: {input_devices[0][1]}")
            else:
                self.input_device_index = None
        
        print(f"{Fore.CYAN}[READY]{Style.RESET_ALL} Found {len(input_devices)} available input devices\n")
    
    def detect_keyword_in_audio(self, audio_data):
        """
        Backup keyword detection algorithm (used when FunASR is not available)
        """
        try:
            # Calculate audio features
            amplitude = np.abs(audio_data).mean()
            audio_length = len(audio_data) / self.sample_rate
            
            # Simplified heuristic detection
            if 0.1 <= audio_length <= 2.0 and amplitude > self.threshold * 2:
                mid_point = len(audio_data) // 2
                first_half_energy = np.sum(audio_data[:mid_point].astype(float) ** 2)
                second_half_energy = np.sum(audio_data[mid_point:].astype(float) ** 2)
                
                energy_ratio = min(first_half_energy, second_half_energy) / max(first_half_energy, second_half_energy)
                
                if energy_ratio > 0.3:
                    return True, f"Possible keyword detected (duration:{audio_length:.2f}s, energy ratio:{energy_ratio:.2f})"
            
            return False, f"Audio doesn't match keyword pattern (duration:{audio_length:.2f}s, amplitude:{amplitude:.3f})"
            
        except Exception as e:
            return False, f"Keyword detection error: {e}"

    def process_keyword_detection(self, audio_chunk):
        """Process keyword detection (backup method, used when FunASR is not available)"""
        if self.use_funasr:
            return False  # Don't use this method when using FunASR
            
        # Add audio chunk to keyword buffer
        try:
            if self.keyword_buffer.full():
                self.keyword_buffer.get_nowait()
            
            # Convert to numpy array
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_array = audio_chunk
                
            self.keyword_buffer.put_nowait(audio_array)
        except queue.Full:
            pass
        
        # Collect audio from buffer for keyword detection
        if not self.keyword_buffer.empty():
            buffer_data = []
            temp_queue = queue.Queue()
            
            while not self.keyword_buffer.empty():
                try:
                    chunk = self.keyword_buffer.get_nowait()
                    buffer_data.append(chunk)
                    temp_queue.put(chunk)
                except queue.Empty:
                    break
            
            # Restore queue state
            while not temp_queue.empty():
                try:
                    self.keyword_buffer.put_nowait(temp_queue.get_nowait())
                except queue.Full:
                    break
            
            # Combine audio data for keyword detection
            if len(buffer_data) > 5:
                combined_audio = np.concatenate(buffer_data)
                is_keyword, message = self.detect_keyword_in_audio(combined_audio)
                
                if is_keyword and not self.keyword_detected and not self.recording:
                    print(f"\n{Fore.GREEN}[KEYWORD]{Style.RESET_ALL} {message}")
                    print(f"{Fore.GREEN}[ACTIVATED]{Style.RESET_ALL} Detected keyword '{self.current_keyword}' - Ready to start recording")
                    self.keyword_detected = True
                    return True
        
        return False
    
    def visualize_audio_vad(self, audio_data, sample_rate, vad_segments):
        """
        Visualize audio waveform and VAD detection results
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            vad_segments: List of speech segments as [[start_time, end_time], ...]
        """
        try:
            # Create a new figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
            
            # Calculate time axis
            duration = len(audio_data) / sample_rate
            time_axis = np.linspace(0, duration, len(audio_data))
            
            # Plot audio waveform in the top subplot
            ax1.plot(time_axis, audio_data, color='blue', linewidth=0.5)
            ax1.set_title("Audio Waveform")
            ax1.set_xlim(0, duration)
            ax1.set_ylabel("Amplitude")
            
            # Highlight speech segments in the waveform
            if vad_segments and len(vad_segments) > 0:
                for start, end in vad_segments:
                    ax1.axvspan(start, end, color='green', alpha=0.2)
            else:
                # No segments detected
                ax1.text(duration/2, 0, "No speech detected", 
                         horizontalalignment='center', fontsize=12, color='red')
            
            # Plot VAD result in the bottom subplot
            ax2.set_title("VAD Result - Speech Activity Detection")
            ax2.set_xlim(0, duration)
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            ax2.set_xlabel("Time (seconds)")
            
            # Draw VAD segments
            if vad_segments and len(vad_segments) > 0:
                for start, end in vad_segments:
                    ax2.fill_between([start, end], 0, 1, color="green", alpha=0.6)
            else:
                # No segments detected
                ax2.text(duration/2, 0.5, "No speech segments", 
                         horizontalalignment='center', fontsize=12, color='red')
            
            plt.tight_layout()
            
            # Create folder for saving visualizations
            vis_dir = "visualizations"
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
                
            # Save figure with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(vis_dir, f"vad_visualization_{timestamp}.png")
            plt.savefig(fig_path)
            print(f"\n{Fore.GREEN}[VISUALIZATION]{Style.RESET_ALL} Saved VAD visualization to {fig_path}")
            
            # Show plot in a non-blocking way
            plt.show(block=False)
            plt.pause(0.1)  # Short pause to render the plot
            
        except Exception as e:
            print(f"\n{Fore.RED}[VISUALIZATION ERROR]{Style.RESET_ALL} Failed to visualize audio: {e}")
    
    def save_audio_sample(self, audio_data, prefix="audio_sample"):
        """Save audio data for testing and debugging
        
        Args:
            audio_data: Audio data as numpy array
            prefix: Prefix for the saved file name
        """
        try:
            # Create folder for samples
            samples_dir = "audio_samples"
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
                
            # Save as WAV file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(samples_dir, f"{prefix}_{timestamp}.wav")
            
            # Convert to int16 format if needed
            if audio_data.dtype != np.int16:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:  # Avoid division by zero
                    # Normalize and convert to int16
                    audio_data = (audio_data / max_val * 32767).astype(np.int16)
                else:
                    audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
                
            print(f"\n{Fore.GREEN}[AUDIO SAMPLE]{Style.RESET_ALL} Saved audio sample to {filename}")
            return filename
            
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to save audio sample: {e}")
            return None
        
    def setup_live_plot(self):
        """Set up a live plot for audio visualization"""
        try:
            print(f"{Fore.CYAN}[VISUALIZATION]{Style.RESET_ALL} Setting up live plot...")
            
            # Create a figure with 2 subplots - one for audio level, one for spectrogram
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
            plt.subplots_adjust(hspace=0.4)
            
            # Set up the first subplot for audio level monitoring
            self.ax1.set_title("Audio Level Monitor")
            self.ax1.set_ylim(0, 0.2)  # Amplitude range
            self.ax1.set_xlim(0, 100)  # Show last 100 frames
            self.ax1.set_ylabel("Amplitude")
            self.ax1.set_xlabel("Time (frames)")
            self.ax1.grid(True)
            
            # Create line for the audio level
            self.audio_line, = self.ax1.plot([], [], 'b-', linewidth=1.5, label="Audio Level")
            
            # Add threshold line
            self.threshold_line, = self.ax1.plot([], [], 'r--', linewidth=1, label="Threshold")
            
            # Set up the second subplot for spectrogram
            self.ax2.set_title("Audio Spectrogram")
            self.ax2.set_xlabel("Time (frames)")
            self.ax2.set_ylabel("Frequency")
            
            # Create a colorbar for the spectrogram
            self.spectrogram = self.ax2.imshow(
                np.zeros((64, 100)),  # Initial empty spectrogram
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, 100, 0, 8000]  # X start, X end, Y start, Y end
            )
            
            # Status indicator box in the corner
            self.status_text = self.fig.text(0.02, 0.02, "Status: Monitoring", 
                                       fontsize=10, ha="left", 
                                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Add a legend
            self.ax1.legend(loc="upper right")
            
            # Initialize data containers
            self.audio_levels = np.zeros(100)
            self.threshold_levels = np.ones(100) * self.threshold
            self.spectrogram_data = np.zeros((64, 100))
            self.time_axis = np.arange(100)
            
            # Make the plot interactive
            plt.ion()
            self.fig.canvas.draw()
            
            # Add event handler for window close
            self.fig.canvas.mpl_connect('close_event', self._handle_close)
            
            plt.show(block=False)
            
            # Set up plot update interval (ms)
            self.plot_update_interval = 100  # Update every 100ms
            self.last_plot_update = time.time()
            
            print(f"{Fore.GREEN}[VISUALIZATION]{Style.RESET_ALL} Live audio visualization initialized")
            self.live_plot_enabled = True
            
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to initialize visualization: {e}")
            import traceback
            traceback.print_exc()
            self.live_plot_enabled = False

    def update_live_plot(self, audio_data, is_speech=False):
        """Update the live plot with new audio data
        
        Args:
            audio_data: New audio data chunk
            is_speech: Whether VAD detected speech
        """
        if not hasattr(self, 'live_plot_enabled') or not self.live_plot_enabled:
            return
            
        try:
            # Check if it's time to update the plot (limit updates to avoid GUI freezing)
            current_time = time.time()
            if hasattr(self, 'last_plot_update') and current_time - self.last_plot_update < self.plot_update_interval / 1000:
                return
                
            self.last_plot_update = current_time
            
            # Calculate audio level
            if audio_data.dtype == np.int16:
                # Normalize int16 audio
                amplitude = np.abs(audio_data).mean() / 32767.0
            else:
                # Already normalized float32 audio
                amplitude = np.abs(audio_data).mean()
                
            # Shift data arrays to the left and add new value
            self.audio_levels = np.roll(self.audio_levels, -1)
            self.audio_levels[-1] = amplitude
            
            # Update audio level line
            self.audio_line.set_data(self.time_axis, self.audio_levels)
            
            # Update threshold line
            self.threshold_line.set_data(self.time_axis, self.threshold_levels)
            
            # Calculate spectrogram for the new data
            if len(audio_data) >= 512:  # Need enough samples for FFT
                # Take the most recent data for FFT
                fft_data = audio_data[-512:]
                # Calculate FFT (use window function for better results)
                fft_window = np.hanning(len(fft_data))
                fft_result = np.abs(np.fft.rfft(fft_data * fft_window))
                # Convert to dB scale, normalize and clip
                fft_db = 20 * np.log10(fft_result + 1e-10)  # Avoid log(0)
                fft_db = np.clip((fft_db + 80) / 80, 0, 1)  # Normalize
                
                # Resample to fit our display size (64 frequency bins)
                if len(fft_db) > 64:
                    # Downsample using max pooling for better visualization
                    ratio = len(fft_db) // 64
                    fft_resampled = np.zeros(64)
                    for i in range(64):
                        start = i * ratio
                        end = min(start + ratio, len(fft_db))
                        fft_resampled[i] = np.max(fft_db[start:end])
                else:
                    # If we have fewer points, use interpolation
                    fft_resampled = np.interp(
                        np.linspace(0, 1, 64),
                        np.linspace(0, 1, len(fft_db)),
                        fft_db
                    )
                
                # Update spectrogram data
                self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
                self.spectrogram_data[:, -1] = fft_resampled
                
                # Update spectrogram
                self.spectrogram.set_array(self.spectrogram_data)
            
            # Update status text
            status_color = 'green' if is_speech else 'blue'
            status_text = f"Status: {'🎤 Speech Detected' if is_speech else '🔇 Silence'}"
            if self.recording:
                status_text += " | 🔴 Recording"
            elif self.keyword_detected:
                status_text += f" | ⭐ Keyword: {self.current_keyword}"
                
            level_info = f" | Level: {amplitude:.3f}"
            self.status_text.set_text(status_text + level_info)
            self.status_text.set_bbox(dict(facecolor=status_color, alpha=0.2))
            
            # Redraw the plot
            # Use fig.canvas.draw_idle() for better performance in interactive mode
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            # Disable plot updates if an error occurs to prevent continuous errors
            if hasattr(self, 'live_plot_enabled'):
                self.live_plot_enabled = False
            print(f"\n{Fore.RED}[VISUALIZATION ERROR]{Style.RESET_ALL} Failed to update plot: {e}")
            import traceback
            traceback.print_exc()
    
    def voice_visualizer(self):
        """Background worker for processing and visualizing audio data"""
        print(f"{Fore.CYAN}[VISUALIZATION]{Style.RESET_ALL} Visualization thread started")
        while self.visualization_running:
            try:
                # Get data from queue with a timeout to allow for thread termination
                try:
                    audio_data, is_speech = self.visualization_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Update the visualization with the data
                # self.update_live_plot(audio_data, is_speech)
                
                # Process VAD on the audio data (for additional visualization if needed)
                # Only process if we have normalized float32 data or if we can convert int16 to float32
                if len(audio_data) > 0:
                    if audio_data.dtype == np.int16:
                        # Convert int16 to float32 for VAD processing
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data
                    
                    # Only run VAD processing occasionally to avoid overloading
                    current_time = time.time()
                    if current_time - getattr(self, 'last_vad_worker_check', 0) >= 2.0:  # Process every 2 seconds
                        self.last_vad_worker_check = current_time
                        
                        # Only process if we have enough data (at least 400 samples for VAD)
                        if len(audio_float) >= 400:
                            # Ensure the audio data length is a multiple of 400 for VAD
                            valid_length = (len(audio_float) // 400) * 400
                            if valid_length > 0:
                                # Process VAD on the audio data
                                valid_audio = audio_float[:valid_length]
                                is_speech_vad, vad_segments = self.process_vad(valid_audio)
                                
                                # If significant speech segments detected, visualize the audio and VAD results
                                if len(vad_segments) >= 2:  # Only visualize if we have multiple segments
                                    # Use a separate thread to avoid blocking this worker
                                    threading.Thread(
                                        target=self.visualize_audio_vad,
                                        args=(valid_audio, self.sample_rate, vad_segments),
                                        daemon=True
                                    ).start()
                
                # Mark the task as done
                self.visualization_queue.task_done()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"\n{Fore.RED}[VISUALIZATION WORKER ERROR]{Style.RESET_ALL} {e}")
                # Short sleep to prevent rapid error loops
                time.sleep(0.5)
    def start_visualization_thread(self):
        """Start a separate thread for visualization updates"""
        try:
            self.visualization_running = True
            self.visualization_thread = threading.Thread(
                target=self.voice_visualizer,
                daemon=True
            )
            self.visualization_thread.start()
            print(f"{Fore.GREEN}[VISUALIZATION]{Style.RESET_ALL} Started visualization thread")
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to start visualization thread: {e}")
            self.visualization_running = False
    
    def _handle_close(self, evt):
        """Handle plot window close event"""
        print(f"{Fore.CYAN}[VISUALIZATION]{Style.RESET_ALL} Plot window closed by user")
        self.live_plot_enabled = False
        self.visualization_running = False
        self.visualization_running = False
    
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}           FunASR VAD+ASR Keyword Activated Recording System{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Working Mode: FunASR VAD+ASR Keyword Activated Recording{Style.RESET_ALL}")
    print(f"  • Default keyword: {Fore.GREEN}'hello'{Style.RESET_ALL}")
    print(f"  • Supported keywords: hello, assistant, start recording, begin recording, record, listen")
    print(f"  • Detection method: VAD detects speech segments + ASR recognizes keywords")
    print(f"{Fore.YELLOW}System Configuration:{Style.RESET_ALL}")
    print(f"  • VAD check interval: 3.0 seconds")
    print(f"  • VAD window size: 1.0 second (16000 samples)")
    print(f"  • Silence timeout: 2.0 seconds")
    print(f"  • Minimum speech duration: 0.8 seconds") 
    print(f"  • Pre-buffer duration: 5.0 seconds")
    print(f"{Fore.RED}Important: System uses fixed 400 chunk size VAD detection!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Controls: Press Ctrl+C to exit program{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
    
    # Create and start recorder
    try:
        recorder = KeywordActivatedRecorder(
            threshold=0.001,             # Backup amplitude threshold
            silence_duration=2.0,        # Silence duration before stopping recording (seconds)
            min_speech_duration=0.5,     # Minimum speech duration to start recording (seconds)
            buffer_duration=5.0          # Pre-buffer duration (seconds)
        )
        
        # Ask the user if they want to test with a WAV file or use the microphone
        # mode = input(f"{Fore.YELLOW}[STARTUP]{Style.RESET_ALL} Choose mode: (1) Test with WAV file, (2) Live microphone: ").strip()
        mode = 1

        if mode == 1:
            # Find sample WAV files in the models directory
            sample_files = []
            model_dir = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch", "example")
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith(".wav"):
                        sample_files.append(os.path.join(model_dir, file))
            
            # Also check test directory
            test_dir = "test"
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.endswith(".wav"):
                        sample_files.append(os.path.join(test_dir, file))
            
            # User can also provide a custom WAV file
            if sample_files:
                print(f"{Fore.CYAN}[SAMPLES]{Style.RESET_ALL} Found {len(sample_files)} sample WAV files:")
                for i, file in enumerate(sample_files):
                    print(f"  {i+1}. {file}")

                choice = input(f"{Fore.YELLOW}[SELECT]{Style.RESET_ALL} Choose a sample file (1-{len(sample_files)}) or enter a custom path: ").strip()
                
                try:
                    # Check if the input is a number for sample selection
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(sample_files):
                        test_file = sample_files[choice_num-1]
                    else:
                        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Invalid selection")
                        test_file = input(f"{Fore.YELLOW}[INPUT]{Style.RESET_ALL} Enter path to a WAV file: ").strip()
                except ValueError:
                    # Input is not a number, treat as a custom path
                    test_file = choice
                
                # Validate the test file exists
                if not os.path.exists(test_file):
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} File not found: {test_file}")
                    test_file = input(f"{Fore.YELLOW}[RETRY]{Style.RESET_ALL} Enter path to a WAV file: ").strip()
                
                # Process the test file if it exists
                if os.path.exists(test_file):
                    print(f"{Fore.GREEN}[TEST]{Style.RESET_ALL} Loading WAV file: {test_file}")
                    
                    # Open the WAV file
                    with wave.open(test_file, 'rb') as wf:
                        # Check if the WAV file parameters match our expected parameters
                        if wf.getnchannels() != recorder.channels:
                            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} WAV file has {wf.getnchannels()} channels, expected {recorder.channels}")
                        
                        if wf.getsampwidth() != recorder.pyaudio.get_sample_size(recorder.format):
                            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} WAV file has {wf.getsampwidth()} bytes per sample, expected {recorder.pyaudio.get_sample_size(recorder.format)}")
                        
                        if wf.getframerate() != recorder.sample_rate:
                            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} WAV file has {wf.getframerate()} Hz sample rate, expected {recorder.sample_rate}")
                        
                        # Get total number of frames
                        n_frames = wf.getnframes()
                        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} WAV file contains {n_frames} frames ({n_frames/wf.getframerate():.2f} seconds)")
                        
                        # Read data in chunks matching recorder's chunk_size
                        chunk_size = recorder.chunk_size
                        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Processing in chunks of {chunk_size} frames")
                        
                        # Create fake time_info dict for audio_callback
                        time_info = {
                            'input_buffer_adc_time': 0,
                            'current_time': 0,
                            'output_buffer_dac_time': 0
                        }
                        
                        # Process the WAV file in chunks
                        frames_processed = 0
                        chunk_count = 0
                        
                        while frames_processed < n_frames:
                            # Read a chunk of data
                            data = wf.readframes(chunk_size)
                            
                            if not data:
                                break
                            
                            # Keep track of what we've processed
                            frames_read = len(data) // (wf.getsampwidth() * wf.getnchannels())
                            frames_processed += frames_read
                            chunk_count += 1
                            
                            # Update fake time_info
                            time_info['current_time'] = frames_processed / wf.getframerate()
                            
                            print(f"\n{Fore.CYAN}[PROCESSING]{Style.RESET_ALL} Chunk {chunk_count}: {frames_read} frames ({frames_read/wf.getframerate():.3f} sec)")
                            
                            # Call the audio_callback method with this chunk
                            print(f"{Fore.CYAN}[CALLBACK]{Style.RESET_ALL} Calling audio_callback with {len(data)} bytes")
                            try:
                                # Match the exact function signature: in_data, frame_count, time_info, status
                                result = recorder.audio_callback(data, frames_read, time_info, 0)
                                # print(f"{Fore.GREEN}[CALLBACK RESULT]{Style.RESET_ALL} Returned: {result}")
                            except Exception as e:
                                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Exception in audio_callback: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Add a short pause to allow for visualization updates
                            time.sleep(0.1)
                            
                            # Ask user if they want to continue to the next chunk
                            # if chunk_count % 5 == 0:  # Every 5 chunks
                            #     user_input = input(f"{Fore.YELLOW}[INTERACTIVE]{Style.RESET_ALL} Continue to next chunk? (Y/n): ").strip().lower()
                            #     if user_input == 'n':
                            #         print(f"{Fore.YELLOW}[ABORT]{Style.RESET_ALL} Test aborted by user")
                            #         break
                        
                        print(f"\n{Fore.GREEN}[COMPLETE]{Style.RESET_ALL} Processed {frames_processed} frames in {chunk_count} chunks")
                else:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} File not found, falling back to microphone mode")
                    recorder.start()
            else:
                # No sample files found
                test_file = input(f"{Fore.YELLOW}[INPUT]{Style.RESET_ALL} Enter path to a WAV file: ").strip()
                if os.path.exists(test_file):
                    # The processing code would be repeated here, but we'll skip it for brevity
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No sample implementation, falling back to microphone mode")
                    recorder.start()
                else:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} File not found, falling back to microphone mode")
                    recorder.start()
        else:
            # Start in normal microphone mode
            recorder.start()
            print(f"{Fore.GREEN}[STATUS]{Style.RESET_ALL} FunASR VAD+ASR keyword detection system started")
            print(f"{Fore.CYAN}[WAITING]{Style.RESET_ALL} Please say the keyword '{Fore.GREEN}你好{Style.RESET_ALL}' to activate recording")
            print(f"{Fore.BLUE}[TIP]{Style.RESET_ALL} System will show real-time VAD detection status and speech recognition results")
            print(f"{Fore.YELLOW}[TESTING]{Style.RESET_ALL} Try speaking clearly for 2-3 seconds to test VAD detection")
            print(f"{Fore.YELLOW}[KEYWORDS]{Style.RESET_ALL} Supported: 你好, 小助手, 开始录音, 录音开始, 小爱, 小度\n")
        
        # Keep program running - only for microphone mode
        if mode != "1":
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[SYSTEM]{Style.RESET_ALL} Keyboard interrupt detected, shutting down...")
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} System error: {e}")
    finally:
        try:
            recorder.stop()
        except:
            pass
        print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Program exited safely")
