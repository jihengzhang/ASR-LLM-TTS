#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""
"""
Simple Wave Recording Test Script
Based on FunASR_VAD_debug.py
Press and hold SPACE to record, release to stop and save
"""

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
import keyboard
import sys

# Initialize colorama for color output support
colorama.init()

class SimpleWaveRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 format=pyaudio.paInt16):
        """
        Initialize simple wave recorder
        
        Parameters:
            sample_rate: Sample rate (Hz)
            chunk_size: Audio chunk size for each processing
            channels: Number of channels
            format: Audio format
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        # Recording status
        self.recording = False
        self.frames = []
        
        # Audio objects
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Check and list audio devices
        self.list_audio_devices()
        
        # Thread lock
        self.lock = threading.Lock()
        
        print(f"{Fore.GREEN}[READY]{Style.RESET_ALL} Wave recorder initialized")
        print(f"{Fore.CYAN}[CONFIG]{Style.RESET_ALL} Sample rate: {sample_rate}Hz, Channels: {channels}, Format: 16-bit")
    
    def list_audio_devices(self):
        """List all available audio input devices"""
        device_count = self.pyaudio.get_device_count()
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}           Audio Input Device List{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}[DEVICES]{Style.RESET_ALL} Found {device_count} total audio devices:")
        
        input_devices = []
        for i in range(device_count):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  {Fore.GREEN}[{i:2d}]{Style.RESET_ALL} {info['name']}")
                    print(f"       └─ Input channels: {info['maxInputChannels']}, Sample rate: {int(info['defaultSampleRate'])}Hz")
            except Exception as e:
                print(f"  {Fore.RED}[{i:2d}]{Style.RESET_ALL} Error reading device info: {e}")
                continue
        
        if not input_devices:
            raise Exception("No available audio input devices found")
        
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            print(f"\n{Fore.GREEN}[DEFAULT]{Style.RESET_ALL} Default input device:")
            print(f"  └─ {default_input['name']} (Index: {default_input['index']})")
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Cannot get default input device: {e}")
        
        print(f"\n{Fore.CYAN}[SUMMARY]{Style.RESET_ALL} Total available input devices: {Fore.GREEN}{len(input_devices)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
        
        return input_devices

    def start_stream(self):
        """Start audio stream"""
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            print(f"{Fore.GREEN}[STREAM]{Style.RESET_ALL} Audio stream started successfully")
            return True
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to start audio stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print(f"{Fore.YELLOW}[STREAM]{Style.RESET_ALL} Audio stream stopped")
    
    def start_recording(self):
        """Start recording"""
        with self.lock:
            if not self.recording and self.stream:
                self.recording = True
                self.frames = []
                print(f"\n{Fore.RED}● [RECORDING]{Style.RESET_ALL} Recording started - keep holding SPACE")
                return True
        return False
    
    def stop_recording(self):
        """Stop recording and save file"""
        with self.lock:
            if self.recording:
                self.recording = False
                print(f"\n{Fore.YELLOW}[STOP]{Style.RESET_ALL} Recording stopped")
                
                # Save recording in a separate thread
                if len(self.frames) > 0:
                    threading.Thread(target=self.save_recording, daemon=True).start()
                else:
                    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No audio data recorded")
                
                return True
        return False
    
    def record_chunk(self):
        """Record one audio chunk"""
        if self.stream and self.recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Calculate amplitude for display
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.abs(audio_data).mean() / 32767.0
                
                # Display recording level
                level_bars = int(amplitude * 30)
                level_display = '█' * level_bars + '░' * (30 - level_bars)
                print(f"\r{Fore.RED}● Recording{Style.RESET_ALL} | Level: [{level_display}] {amplitude:.3f} | Frames: {len(self.frames)}", end='', flush=True)
                
                with self.lock:
                    if self.recording:
                        self.frames.append(data)
                
            except Exception as e:
                print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Error reading audio: {e}")
    
    def save_recording(self):
        """Save recording as WAV file"""
        if not self.frames:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No recording data to save")
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"test_{timestamp}.wav"
        
        print(f"\n{Fore.BLUE}[SAVING]{Style.RESET_ALL} Saving recording to: {filename}")
        
        try:
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                 
            duration = len(self.frames) * self.chunk_size / self.sample_rate
            file_size = os.path.getsize(filename) / 1024  # KB
            
            print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Recording saved successfully!")
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} File: {filename}")
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Duration: {duration:.2f} seconds")
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} File size: {file_size:.2f} KB")
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Sample rate: {self.sample_rate}Hz, Channels: {self.channels}")
            
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to save recording: {e}")
        
        # Clear frames
        self.frames = []
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_stream()
        self.pyaudio.terminate()
        print(f"{Fore.GREEN}[CLEANUP]{Style.RESET_ALL} Audio system closed")

def main():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}           Simple Wave Recording Test Script{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Recording Mode: Press and Hold SPACE Key{Style.RESET_ALL}")
    print(f"  • Press and hold {Fore.GREEN}SPACE{Style.RESET_ALL} to start recording")
    print(f"  • Release {Fore.GREEN}SPACE{Style.RESET_ALL} to stop recording and save")
    print(f"  • Press {Fore.RED}ESC{Style.RESET_ALL} or {Fore.RED}Ctrl+C{Style.RESET_ALL} to exit")
    print(f"{Fore.YELLOW}Output Format:{Style.RESET_ALL}")
    print(f"  • File format: WAV (16-bit, 16kHz, Mono)")
    print(f"  • File naming: test_YYYY-MM-DD-HH-MM-SS.wav")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
    
    # Create recorder instance
    try:
        recorder = SimpleWaveRecorder(
            sample_rate=16000,
            chunk_size=1024,
            channels=2,
            format=pyaudio.paInt16
        )
        
        # Start audio stream
        if not recorder.start_stream():
            print(f"{Fore.RED}[FATAL]{Style.RESET_ALL} Cannot start audio stream, exiting...")
            return
        
        print(f"{Fore.GREEN}[READY]{Style.RESET_ALL} Recording system ready!")
        print(f"{Fore.CYAN}[INSTRUCTION]{Style.RESET_ALL} Press and hold {Fore.GREEN}SPACE{Style.RESET_ALL} to start recording...")
        print(f"{Fore.BLUE}[TIP]{Style.RESET_ALL} You can record multiple files by pressing SPACE multiple times\n")
        
        space_pressed = False
        
        # Main loop
        while True:
            try:
                # Check keyboard events
                if keyboard.is_pressed('space'):
                    if not space_pressed:
                        space_pressed = True
                        recorder.start_recording()
                else:
                    if space_pressed:
                        space_pressed = False
                        recorder.stop_recording()
                        print(f"\n\n{Fore.CYAN}[WAITING]{Style.RESET_ALL} Press {Fore.GREEN}SPACE{Style.RESET_ALL} again to record another file, or {Fore.RED}ESC{Style.RESET_ALL} to exit...")
                
                # Check for exit
                if keyboard.is_pressed('esc'):
                    print(f"\n{Fore.YELLOW}[EXIT]{Style.RESET_ALL} ESC key pressed, exiting...")
                    break
                
                # Record audio chunk if recording
                if recorder.recording:
                    recorder.record_chunk()
                
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}[EXIT]{Style.RESET_ALL} Keyboard interrupt detected, exiting...")
                break
                
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} System error: {e}")
    finally:
        try:
            if 'recorder' in locals():
                recorder.cleanup()
        except:
            pass
        print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Program exited safely")

if __name__ == "__main__":
    # Check if required modules are available
    try:
        import keyboard
    except ImportError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Missing required module 'keyboard'")
        print(f"{Fore.YELLOW}[SOLUTION]{Style.RESET_ALL} Please install it with: pip install keyboard")
        sys.exit(1)
    
    try:
        import pyaudio
    except ImportError:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Missing required module 'pyaudio'")
        print(f"{Fore.YELLOW}[SOLUTION]{Style.RESET_ALL} Please install it with: pip install pyaudio")
        sys.exit(1)
    
    main()
