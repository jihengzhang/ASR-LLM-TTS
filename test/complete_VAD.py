#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete FunASR VAD Voice Activity Detection System
Integrated FunASR VAD model and built-in algorithms
"""

import os
import time
import wave
import numpy as np
import threading
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

try:
    import pyaudio
    print(f"{Fore.GREEN}✓ PyAudio module loaded{Style.RESET_ALL}")
except ImportError:
    print(f"{Fore.RED}✗ PyAudio not installed{Style.RESET_ALL}")
    exit(1)

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"{Fore.GREEN}✓ FunASR module loaded{Style.RESET_ALL}")
except ImportError:
    FUNASR_AVAILABLE = False
    print(f"{Fore.YELLOW}⚠ FunASR module not found, will use built-in VAD{Style.RESET_ALL}")

# Configuration parameters
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# VAD parameters
ENERGY_THRESHOLD = 0.01
ZCR_THRESHOLD = 0.1
MIN_SPEECH_DURATION = 1.0
SILENCE_DURATION = 2.0

class HybridVAD:
    """Hybrid VAD system: FunASR + built-in algorithms"""
    
    def __init__(self):
        self.energy_history = []
        self.background_energy = 0.001
        self.funasr_vad = None
        self.audio_buffer = []
        self.buffer_size = RATE * 2  # 2 second buffer
        
        # Initialize FunASR VAD
        if FUNASR_AVAILABLE:
            self.init_funasr()
    
    def init_funasr(self):
        """Initialize FunASR VAD model"""
        try:
            print(f"{Fore.YELLOW}Loading FunASR VAD model...{Style.RESET_ALL}")
            
            # Try different local model path variations
            possible_paths = [
                os.path.join(os.getcwd(), "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                os.path.join(os.getcwd(), "models", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                os.path.join(os.getcwd(), "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            ]
            
            local_model_found = False
            for local_model_path in possible_paths:
                if os.path.exists(local_model_path):
                    print(f"{Fore.CYAN}Found local model: {local_model_path}{Style.RESET_ALL}")
                    try:
                        # Try loading with different methods
                        self.funasr_vad = AutoModel(
                            model=local_model_path,
                            disable_update=True,
                            device="cpu",
                            cache_dir=os.getcwd()
                        )
                        local_model_found = True
                        print(f"{Fore.GREEN}✓ Local model loaded successfully{Style.RESET_ALL}")
                        break
                    except Exception as local_e:
                        print(f"{Fore.YELLOW}Local model load attempt failed: {local_e}{Style.RESET_ALL}")
                        continue
            
            if not local_model_found:
                print(f"{Fore.YELLOW}No local model found or failed to load, trying remote model...{Style.RESET_ALL}")
                
                # Set environment variables for SSL bypass
                os.environ['PYTHONHTTPSVERIFY'] = '0'
                os.environ['MODELSCOPE_CACHE'] = os.getcwd()
                os.environ['HF_HOME'] = os.getcwd()
                
                # Try loading remote model with SSL bypass
                self.funasr_vad = AutoModel(
                    model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    disable_update=True,
                    device="cpu",
                    cache_dir=os.getcwd()
                )
                print(f"{Fore.GREEN}✓ Remote model loaded successfully{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}✗ FunASR VAD loading failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Will use built-in VAD algorithm{Style.RESET_ALL}")
            self.funasr_vad = None

    def calculate_features(self, frame):
        """Calculate audio features"""
        if len(frame) == 0:
            return {'energy': 0, 'zcr': 0}
            
        # Normalize
        frame = frame.astype(np.float32) / 32768.0
        
        # Energy
        energy = np.mean(frame ** 2)
        
        # Zero crossing rate
        zcr = np.mean(0.5 * np.abs(np.diff(np.sign(frame))))
        
        return {'energy': energy, 'zcr': zcr}
    
    def update_background(self, energy):
        """Update background noise"""
        self.energy_history.append(energy)
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)
        
        if len(self.energy_history) >= 10:
            self.background_energy = np.percentile(self.energy_history, 20)
    
    def builtin_vad(self, frame):
        """Built-in VAD algorithm"""
        features = self.calculate_features(frame)
        energy = features['energy']
        zcr = features['zcr']
        
        # Update background noise
        self.update_background(energy)
        
        # Adaptive threshold
        adaptive_threshold = max(self.background_energy * 5, ENERGY_THRESHOLD)
        
        # Voice detection
        energy_check = energy > adaptive_threshold
        zcr_check = zcr > ZCR_THRESHOLD
        
        features.update({
            'threshold': adaptive_threshold,
            'background': self.background_energy
        })
        
        return energy_check and zcr_check, features
    
    def funasr_vad_detect(self, audio_array):
        """FunASR VAD detection"""
        if self.funasr_vad is None:
            return None
            
        try:
            # Ensure sufficient audio length
            if len(audio_array) < RATE * 0.5:  # At least 0.5 seconds
                return None
            
            # Format required by FunASR
            result = self.funasr_vad.generate(input=audio_array)
            
            if result and len(result) > 0:
                # Parse results
                vad_result = result[0].get('value', [])
                # If there are speech segments, consider speech detected
                return len(vad_result) > 0
                
        except Exception as e:
            # Silently handle FunASR errors to avoid disrupting display
            pass
        
        return None
    
    def detect(self, frame):
        """Hybrid detection"""
        # Built-in algorithm
        builtin_result, features = self.builtin_vad(frame)
        
        # Update audio buffer
        audio_array = frame.astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_array)
        
        # Maintain buffer size
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
        
        # FunASR detection (detect periodically)
        funasr_result = None
        if len(self.audio_buffer) >= RATE and len(self.audio_buffer) % (RATE // 2) == 0:
            # Detect every 0.5 seconds
            funasr_result = self.funasr_vad_detect(np.array(self.audio_buffer))
        
        # Combine results
        if funasr_result is not None:
            final_result = funasr_result
            method = "FunASR"
        else:
            final_result = builtin_result  
            method = "Built-in"
        
        features['method'] = method
        features['funasr'] = funasr_result
        features['builtin'] = builtin_result
        
        return final_result, features

class VADRecorder:
    """VAD Recorder"""
    
    def __init__(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}        Complete FunASR VAD Voice Activity Detection System{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
        
        # Initialize components
        self.audio = pyaudio.PyAudio()
        self.vad = HybridVAD()
        
        # Check devices
        self.check_audio_devices()
        
        # Recording state
        self.is_recording = False
        self.frames = []
        self.stop_flag = False
        
        # Speech state
        self.speech_active = False
        self.speech_start_time = None
        self.silence_start_time = None
        
        # Output directory
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"{Fore.GREEN}✓ System initialization complete{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}VAD Mode: {'FunASR + Built-in Algorithm' if FUNASR_AVAILABLE and self.vad.funasr_vad else 'Built-in Algorithm'}{Style.RESET_ALL}\n")
    
    def check_audio_devices(self):
        """Check audio devices"""
        device_count = self.audio.get_device_count()
        print(f"{Fore.BLUE}Audio device detection:{Style.RESET_ALL}")
        
        input_count = 0
        for i in range(device_count):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_count += 1
                if input_count <= 3:  # Only show first 3 devices
                    print(f"  [{i}] {info['name']}")
        
        if input_count > 3:
            print(f"  ... and {input_count - 3} more devices")
        
        if input_count == 0:
            raise Exception("No audio input devices found")
        
        print(f"{Fore.GREEN}✓ Found {input_count} input devices{Style.RESET_ALL}")
    
    def display_status(self, is_speech, features):
        """Display real-time status"""
        # Status indicator
        if is_speech:
            indicator = f"{Fore.GREEN}🎤 Speech{Style.RESET_ALL}"
        else:
            indicator = f"{Fore.BLUE}🔇 Silence{Style.RESET_ALL}"
        
        # Detection method
        method = features.get('method', 'Unknown')
        method_color = Fore.CYAN if method == "FunASR" else Fore.YELLOW
        
        # Status line
        status_line = (
            f"\r{indicator} | "
            f"{method_color}{method}{Style.RESET_ALL} | "
            f"Energy: {features.get('energy', 0):.4f} | "
            f"ZCR: {features.get('zcr', 0):.3f} | "
            f"Threshold: {features.get('threshold', 0):.4f}"
        )
        
        # If FunASR results available, show comparison
        if features.get('funasr') is not None:
            builtin = "√" if features.get('builtin') else "×"
            funasr = "√" if features.get('funasr') else "×"
            status_line += f" | Built-in:{builtin} FunASR:{funasr}"
        
        print(status_line + " " * 10, end='')
    
    def start_stream(self):
        """Start audio stream"""
        print(f"{Fore.YELLOW}Starting voice monitoring... Press Ctrl+C to stop{Style.RESET_ALL}")
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
            print(f"\n{Fore.RED}Audio stream error: {e}{Style.RESET_ALL}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}User stopped program{Style.RESET_ALL}")
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
        """Process audio data"""
        # Convert audio
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # VAD detection
        is_speech, features = self.vad.detect(audio_array)
        
        # Display status
        self.display_status(is_speech, features)
        
        # Speech state machine
        current_time = time.time()
        
        if is_speech:
            if not self.speech_active:
                # Speech start
                self.speech_active = True
                self.speech_start_time = current_time
                self.silence_start_time = None
                print(f"\n{Fore.YELLOW}🎤 Speech detected ({features.get('method', 'Unknown')}){Style.RESET_ALL}")
        else:
            if self.speech_active:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                # Check silence duration
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= SILENCE_DURATION:
                    # Speech end
                    speech_duration = current_time - self.speech_start_time
                    print(f"\n{Fore.CYAN}🔇 Speech ended, duration {speech_duration:.2f} seconds{Style.RESET_ALL}")
                    
                    if speech_duration >= MIN_SPEECH_DURATION:
                        if not self.is_recording:
                            print(f"{Fore.GREEN}✓ Start recording{Style.RESET_ALL}")
                            self.start_recording()
                        else:
                            print(f"{Fore.CYAN}✓ Stop recording{Style.RESET_ALL}")
                            self.stop_recording()
                    else:
                        print(f"{Fore.RED}✗ Speech too short, ignored ({speech_duration:.2f}s){Style.RESET_ALL}")
                    
                    self.speech_active = False
                    self.speech_start_time = None
                    self.silence_start_time = None
        
        # Save data during recording
        if self.is_recording:
            self.frames.append(data)
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.frames = []
        print(f"{Fore.GREEN}🔴 Start recording...{Style.RESET_ALL}")
    
    def stop_recording(self):
        """Stop recording"""
        if self.is_recording and len(self.frames) > 0:
            self.is_recording = False
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"funasr_vad_{timestamp}.wav")
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            
            # Display info
            duration = len(self.frames) * CHUNK / RATE
            file_size = os.path.getsize(filename) / 1024
            print(f"{Fore.CYAN}📁 Recording saved: {filename}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 Duration: {duration:.2f}s, Size: {file_size:.1f}KB{Style.RESET_ALL}")
            
            self.frames = []

def main():
    """Main function"""
    print(f"{Fore.CYAN}{Style.BRIGHT}FunASR VAD Voice Activity Detection System{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Features: Real-time voice detection + Auto recording{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Config: Min speech {MIN_SPEECH_DURATION}s, Silence timeout {SILENCE_DURATION}s{Style.RESET_ALL}")
    
    try:
        recorder = VADRecorder()
        recorder.start_stream()
    except Exception as e:
        print(f"\n{Fore.RED}System error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
