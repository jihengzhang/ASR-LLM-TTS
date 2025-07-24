#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD_KWS_plot.py - Real-time Voice Activity Detection and Keyword Spotting with Visualization
"""

import os
import sys
import time
import wave
import queue
import threading
import traceback
import argparse
from xml.parsers.expat import model
import numpy as np
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
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Audio objects
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []  # For recording after keyword detection
        self.input_device_index = None
        
        # Visualization data
        self.audio_buffer = np.zeros(1000)  # 1000 samples history for display
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
        self.setup_visualization()
    
    def init_models(self):
        """Initialize VAD and ASR models"""
        try:
            # Check if models directory exists locally
            vad_model_path = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" #r"damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            local_vad_path = os.path.join("models", vad_model_path)
            if os.path.exists(local_vad_path):
                vad_model_path = local_vad_path
                print(f"Using local VAD model: {local_vad_path}")
            else:
                print(f"Using remote VAD model: {vad_model_path}")
                
            print("Loading VAD model...")
            self.vad_model = AutoModel(
                model=vad_model_path,
                model_type="vad",
                device="cuda" if self.check_cuda_available() else "cpu",
                disable_update=True
            )
            
            # Use a different ASR model that should be available
            # asr_model_path = r"damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            asr_model_path = r"damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            # local_asr_path = os.path.join("~\.cache\models", asr_model_path)
            # if os.path.exists(local_asr_path):
            #     asr_model_path = local_asr_path
            #     print(f"Using local ASR model: {local_asr_path}")
            # else:
            #     print(f"Using remote ASR model: {asr_model_path}")
            
            print("Loading ASR model for keyword spotting...")
            self.asr_model = AutoModel(model=asr_model_path, device="cuda:0", output_type="dict", output_format="text_with_speak_segments", disable_update=True,autoupdate=False)  # Use GPU if available, otherwise fallback to CPU
            # self.asr_model = AutoModel(
            #     model=asr_model_path,
            #     device="cuda" if self.check_cuda_available() else "cpu",
            #     disable_update=True,
            #     output_format="text"
            # )
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Continuing without models - will use amplitude-based detection only")
            self.vad_model = None
            self.asr_model = None
    
    def check_cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def check_audio_devices(self):
        """Check available audio devices and select input device"""
        device_count = self.pyaudio.get_device_count()
        print(f"Found {device_count} audio devices")
        
        input_devices = []
        for i in range(device_count):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  [{i}] {info['name']} (Input channels: {info['maxInputChannels']})")
            except:
                continue
        
        if not input_devices:
            raise Exception("No input devices found")
        
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            print(f"Using default input device: {default_input['name']}")
            self.input_device_index = default_input['index']
        except:
            # Use first available input device
            if input_devices:
                self.input_device_index = input_devices[0][0]
                print(f"Using first available device: {input_devices[0][1]}")
            else:
                self.input_device_index = None
    
    def setup_visualization(self):
        """Set up the visualization plots"""
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
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, self.update_plot, 
            interval=100,  # Update every 100ms
            blit=False
        )
        
        # Set up close event handler
        self.fig.canvas.mpl_connect('close_event', self.on_plot_close)
    
    def on_plot_close(self, event):
        """Handle plot window close event"""
        print("Plot window closed, stopping...")
        try:
            self.running = False
            self.stop_event.set()
            # Use a thread to avoid blocking the GUI thread
            threading.Thread(target=self.stop, daemon=True).start()
        except Exception as e:
            print(f"Error in plot close handler: {e}")
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        try:
            # Update audio waveform
            if self.audio_buffer.size > 0:
                # Get display samples (downsample if needed for performance)
                display_samples = self.audio_buffer[-1000:]
                x_audio = np.arange(len(display_samples))
                self.waveform_line.set_data(x_audio, display_samples)
                self.ax1.set_xlim(0, len(display_samples))
            
            # Update VAD history
            if self.vad_history.size > 0:
                x_vad = np.linspace(0, len(self.audio_buffer[-1000:]), len(self.vad_history))
                self.vad_line.set_data(x_vad, self.vad_history * 0.3)  # Scale to fit in the same plot
            
            # Update spectrogram
            if self.audio_buffer.size > 1000:
                # Calculate spectrogram
                f, t, Sxx = signal.spectrogram(
                    self.audio_buffer[-1000:],
                    fs=self.sample_rate,
                    nperseg=256,
                    noverlap=128
                )
                # Update spectrogram image
                self.spec_img.set_array(10 * np.log10(Sxx + 1e-10))
                self.spec_img.set_extent([0, t[-1], 0, f[-1]])
            
            # Update keyword display
            if self.detected_keywords:
                # Show last 3 detected keywords
                kw_text = "Detected Keywords:\n"
                for i, (kw, ts) in enumerate(self.detected_keywords[-3:]):
                    kw_text += f"{ts.strftime('%H:%M:%S')}: {kw}\n"
                self.keyword_text.set_text(kw_text)
            else:
                self.keyword_text.set_text("No keywords detected")
            
            # Update status
            status = "Recording" if self.recording else "Listening"
            if self.keyword_detected:
                status += " (Keyword detected)"
            self.status_text.set_text(f"Status: {status}")
            
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        return self.waveform_line, self.vad_line, self.spec_img, self.keyword_text, self.status_text
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function for processing audio input
        
        This runs in the audio thread and should be kept lightweight
        """
        if self.stop_event.is_set():
            return (None, pyaudio.paComplete)
        
        # Convert data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to audio buffer for visualization
        self.audio_buffer = np.append(self.audio_buffer, audio_data)[-10000:]  # Keep last 10k samples
        
        # Add to pre-recording buffer
        try:
            if self.buffer.full():
                self.buffer.get_nowait()
            self.buffer.put_nowait(in_data)
        except queue.Full:
            pass
        
        # Add to VAD processing buffer
        self.audio_buffer_for_vad = np.append(self.audio_buffer_for_vad, audio_data)
        
        # Check if it's time to run VAD
        current_time = time.time()
        if current_time - self.last_vad_check >= self.vad_interval:
            self.last_vad_check = current_time
            
            # Put in queue for VAD processing thread
            try:
                self.audio_data_queue.put_nowait((self.audio_buffer_for_vad.copy(), current_time))
                
                # Reset VAD buffer with overlap
                overlap = int(0.1 * self.sample_rate)  # 100ms overlap
                if len(self.audio_buffer_for_vad) > overlap:
                    self.audio_buffer_for_vad = self.audio_buffer_for_vad[-overlap:]
                else:
                    self.audio_buffer_for_vad = np.array([], dtype=np.float32)
            except queue.Full:
                pass
        
        # If recording, store the frame
        if self.recording:
            self.frames.append(in_data)
            
            # Check for silence to stop recording
            amplitude = np.abs(audio_data).mean()
            if amplitude < self.threshold:
                self.silence_counter += 1
                if self.silence_counter >= self.max_silence_frames:
                    # Use thread to avoid blocking audio callback
                    threading.Thread(target=self.stop_recording).start()
            else:
                self.silence_counter = 0
        
        return (in_data, pyaudio.paContinue)
    
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
                if self.recording:
                    continue
                
                # Process with VAD model if available
                vad_segments = []
                is_speech = False
                
                if self.vad_model is not None and len(audio_data) >= 1600:
                    try:
                        # Ensure audio length is multiple of 400 samples (25ms)
                        valid_length = (len(audio_data) // 400) * 400
                        valid_audio = audio_data[:valid_length]
                        
                        # Process with VAD model
                        vad_result = self.vad_model.generate(valid_audio)
                        
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
                        # Fallback to amplitude-based detection
                        is_speech = np.abs(audio_data).mean() > self.threshold
                else:
                    # Fallback to amplitude-based detection
                    is_speech = np.abs(audio_data).mean() > self.threshold
                
                # Update VAD history for visualization
                self.vad_history = np.append(self.vad_history[1:], float(is_speech))
                
                # If speech detected, process with KWS
                if is_speech and not self.keyword_detected:
                    try:
                        self.vad_result_queue.put_nowait((audio_data, timestamp, vad_segments))
                    except queue.Full:
                        pass
                
            except Exception as e:
                print(f"Error in VAD thread: {e}")
    
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
                if self.keyword_detected or self.recording:
                    continue
                
                # Process with ASR model if available
                if self.asr_model is not None:
                    try:
                        # Process with ASR model
                        asr_result = self.asr_model.generate(audio_data)
                        
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
                                        self.detected_keywords.append((keyword, datetime.now()))
                                        
                                        # Set keyword detected flag and start recording
                                        with self.lock:
                                            if not self.keyword_detected and not self.recording:
                                                self.keyword_detected = True
                                                threading.Thread(target=self.start_recording).start()
                                        
                                        break
                    except Exception as e:
                        print(f"KWS error: {e}")
            
            except Exception as e:
                print(f"Error in KWS thread: {e}")
    
    def start_recording(self):
        """Start recording after keyword detection"""
        with self.lock:
            if not self.recording:
                self.recording = True
                self.frames = []
                
                # Add pre-buffer data
                while not self.buffer.empty():
                    try:
                        self.frames.append(self.buffer.get_nowait())
                    except queue.Empty:
                        break
                
                print("Recording started after keyword detection")
                self.silence_counter = 0
    
    def stop_recording(self):
        """Stop recording and save the audio file"""
        with self.lock:
            if self.recording:
                self.recording = False
                self.keyword_detected = False
                
                print("Recording stopped, saving file...")
                
                # Save recording in a separate thread
                threading.Thread(target=self.save_recording).start()
    
    def save_recording(self):
        """Save the recorded audio to a file"""
        if not self.frames:
            print("No audio data to save")
            return
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("recordings", f"recording_{timestamp}.wav")
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        
        duration = len(self.frames) * self.chunk_size / self.sample_rate
        print(f"Recording saved: {filename} ({duration:.1f}s)")
    
    def start(self):
        """Start audio processing and visualization"""
        if self.running:
            print("Already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Start audio stream
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                # input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                # stream_callback=self.audio_callback
            )

            # self.stream = self.pyaudio.open(
            #     format=self.format,
            #     channels=self.channels,
            #     rate=self.sample_rate,
            #     input=True,
            #     frames_per_buffer=self.chunk_size
            # )

            self.stream.start_stream()
            print("Audio stream started")
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self.running = False
            return False
        
        # Start processing threads
        self.threads = []
        
        vad_thread = threading.Thread(target=self.vad_processing_thread)
        vad_thread.daemon = True
        vad_thread.start()
        self.threads.append(vad_thread)
        
        kws_thread = threading.Thread(target=self.kws_processing_thread)
        kws_thread.daemon = True
        kws_thread.start()
        self.threads.append(kws_thread)
        
        # Show plot (non-blocking)
        plt.show(block=False)
        
        return True
    
    def stop(self):
        """Stop audio processing and visualization"""
        print("Stopping...")
        
        # Set flags to stop processing
        self.running = False
        self.stop_event.set()
        
        # Stop audio stream first
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
        
        # Wait for threads to finish
        for thread in self.threads:
            try:
                thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error joining thread: {e}")
        
        # Terminate PyAudio
        try:
            self.pyaudio.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
        
        # Close plot safely
        try:
            if hasattr(self, 'animation') and self.animation:
                self.animation.event_source.stop()
            if hasattr(self, 'fig'):
                plt.close(self.fig)
        except Exception as e:
            print(f"Error closing plot: {e}")
        
        print("Stopped")
    
    def run(self):
        """Run the processor and handle keyboard interrupt"""
        try:
            if self.start():
                print("Press Ctrl+C to stop")
                
                # Keep main thread alive
                while self.running and not self.stop_event.is_set():
                    try:
                        time.sleep(0.1)
                    except (SystemError, KeyboardInterrupt):
                        break
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.stop()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Voice Activity Detection and Keyword Spotting with Visualization")
    parser.add_argument("--threshold", type=float, default=0.01, help="Amplitude threshold for backup detection")
    parser.add_argument("--vad_interval", type=float, default=0.25, help="VAD check interval in seconds")
    parser.add_argument("--keywords", type=str, nargs="+", default=None, help="List of keywords to detect")
    parser.add_argument("--no_model", action="store_true", help="Force disable model usage (use amplitude only)")
    args = parser.parse_args()
    
    print("Starting VAD-KWS with visualization...")
    
    # Check if FunASR is available
    if not FUNASR_AVAILABLE or args.no_model:
        print("Warning: Running in amplitude-only mode (no speech recognition)")
    
    # Create custom keywords list or use default
    keywords = args.keywords or [
        "你好", "小艾", "开始", "小度", "小爱", 
        "Hi Michael", "Hi Panda", "Hey Google", "Alexa"
    ]
    
    print(f"Listening for keywords: {', '.join(keywords)}")
    
    # Create and run the processor
    processor = VADKWSProcessor(
        sample_rate=16000,
        chunk_size=1600,  # 100ms at 16kHz
        threshold=args.threshold,
        vad_interval=args.vad_interval,  # Check VAD every 250ms
        keywords=keywords
    )
    
    processor.start()

    processor.run()


if __name__ == "__main__":
    main()
