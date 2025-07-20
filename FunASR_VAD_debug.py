import os
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
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 format=pyaudio.paInt16, threshold=0.03, 
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
        self.vad_check_interval = 3.0  # VAD check every 3 seconds
        # Use 16000 samples (1 second) window, ensure it's a multiple of 400
        self.vad_window_size = 16000  # 1 second window
        self.audio_buffer_for_vad = np.array([], dtype=np.float32)
        
        print(f"{Fore.YELLOW}[关键词]{Style.RESET_ALL} 当前激活关键词: '{self.current_keyword}'")
        print(f"{Fore.CYAN}[提示]{Style.RESET_ALL} 使用 FunASR VAD+ASR 进行关键词检测")
    
    def init_funasr_models(self):
        """Initialize FunASR VAD and ASR models"""
        try:
            print(f"{Fore.CYAN}[INIT]{Style.RESET_ALL} Loading FunASR VAD model...")
            # Load VAD model
            self.vad_model = AutoModel(
                model="fsmn-vad",
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} FunASR VAD model loaded successfully")
            
            print(f"{Fore.CYAN}[INIT]{Style.RESET_ALL} Loading FunASR ASR model...")
            # Load ASR model for keyword recognition
            self.asr_model = AutoModel(
                model="paraformer-zh",
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
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print(f"{Fore.GREEN}[READY]{Style.RESET_ALL} Voice activation system started, waiting for speech...")
        
    def stop(self):
        """Stop listening and close resources"""
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Shutting down audio system...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Audio system closed")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function to process input audio data"""
        # Convert binary data to array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
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
                if current_time - self.last_vad_check >= self.vad_check_interval and not self.recording:
                    self.last_vad_check = current_time
                    
                    # Check if there's enough audio data for VAD
                    if len(self.audio_buffer_for_vad) >= self.vad_window_size:
                        # Ensure window size is exact multiple of 400
                        window_samples = (self.vad_window_size // 400) * 400
                        vad_window = self.audio_buffer_for_vad[:window_samples]
                        
                        try:
                            # Use VAD model for speech detection
                            vad_result = self.vad_model.generate(
                                input=vad_window.reshape(1, -1),
                                chunk_size=400  # Specify chunk size
                            )
                            
                            # Check VAD result
                            if isinstance(vad_result, list) and len(vad_result) > 0:
                                result_dict = vad_result[0]
                                if isinstance(result_dict, dict) and 'value' in result_dict:
                                    vad_segments = result_dict['value']
                                    
                                    # If speech segments are detected, process them
                                    if vad_segments and not self.recording and not self.keyword_detected:
                                        print(f"\n{Fore.CYAN}[VAD]{Style.RESET_ALL} Detected {len(vad_segments)} speech segments")
                                        
                                        # Extract audio for each detected speech segment
                                        for segment in vad_segments:
                                            if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                                                start_time, end_time = segment[0], segment[1]
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
                                                        # Use thread for asynchronous keyword recognition
                                                        threading.Thread(
                                                            target=self.recognize_keyword,
                                                            args=(segment_audio,),
                                                            daemon=True
                                                        ).start()
                                                        
                                                        # Check amplitude, if high enough also consider as speech
                                                        segment_amplitude = np.abs(segment_audio).mean()
                                                        if segment_amplitude > self.threshold * 1.5:
                                                            is_speech = True
                                            else:
                                                print(f"\n{Fore.YELLOW}[VAD]{Style.RESET_ALL} Invalid speech segment format: {segment}")
                        except Exception as vad_error:
                            print(f"\n{Fore.RED}[VAD ERROR]{Style.RESET_ALL} {vad_error}")
                            # If VAD processing fails, fall back to amplitude-based detection
                            is_speech = amplitude > self.threshold
                            
                            # If amplitude is high enough, also try direct ASR
                            if amplitude > 0.03 and not self.recording and not self.keyword_detected:
                                # Try ASR keyword recognition with recent audio segment
                                recent_audio = self.audio_buffer_for_vad[-16000:] if len(self.audio_buffer_for_vad) > 16000 else self.audio_buffer_for_vad
                                if len(recent_audio) >= 8000:  # At least 0.5 seconds
                                    threading.Thread(
                                        target=self.recognize_keyword,
                                        args=(recent_audio,),
                                        daemon=True
                                    ).start()
                        
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
        
        # Display audio level and status
        vad_status = "🎤 Speech" if is_speech else "🔇 Silence"
        vad_color = Fore.GREEN if is_speech else Fore.BLUE
        
        if amplitude > 0.001:  # Only display when there's audio input
            level_bars = int(amplitude * 30)
            level_display = '█' * level_bars + '░' * (30 - level_bars)
            status_line = f"{vad_color}{vad_status}{Style.RESET_ALL} | Level: [{level_display}] {amplitude:.3f}"
            
            # Display recording status
            if self.recording:
                status_line += f" | {Fore.RED}● Recording{Style.RESET_ALL}"
            elif self.keyword_detected:
                status_line += f" | {Fore.YELLOW}● Keyword activated: {self.current_keyword}{Style.RESET_ALL}"
            
            print(f"\r{status_line}", end='', flush=True)
        
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
                    print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']} (Input channels: {info['maxInputChannels']})")
            except Exception as e:
                continue
        
        if not input_devices:
            raise Exception("No available audio input devices found")
        
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            print(f"{Fore.GREEN}[DEFAULT]{Style.RESET_ALL} Using default input device: {default_input['name']}")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Cannot get default input device: {e}")
        
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
            if 0.4 <= audio_length <= 2.0 and amplitude > self.threshold * 2:
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
            threshold=0.03,              # Backup amplitude threshold
            silence_duration=2.0,        # Silence duration before stopping recording (seconds)
            min_speech_duration=0.5,     # Minimum speech duration to start recording (seconds)
            buffer_duration=5.0          # Pre-buffer duration (seconds)
        )
        
        recorder.start()
        print(f"{Fore.GREEN}[STATUS]{Style.RESET_ALL} FunASR VAD+ASR keyword detection system started")
        print(f"{Fore.CYAN}[WAITING]{Style.RESET_ALL} Please say the keyword '{Fore.GREEN}hello{Style.RESET_ALL}' to activate recording")
        print(f"{Fore.BLUE}[TIP]{Style.RESET_ALL} System will show real-time VAD detection status and speech recognition results\n")
        
        # Keep program running
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
