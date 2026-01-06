#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558

FunASR Keyword Spotting (KWS) Test Script

This script demonstrates how to use FunASR KWS model to detect keywords in audio files
and from microphone input in real-time. It includes comprehensive error handling
and setup instructions.

Requirements:
    - FunASR >= 0.8.0
    - PyAudio
    - NumPy
    - Torch
"""

import os
import sys
import time
import argparse
import wave
import numpy as np
import threading
import queue
from datetime import datetime

# Check for PyAudio first as it's often a common issue
try:
    import pyaudio
except ImportError:
    print("PyAudio is not installed. Please install it with: pip install pyaudio")
    print("On Windows, you might need to use: pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

# Check for FunASR installation
try:
    from funasr import AutoModel
except ImportError:
    print("FunASR is not installed. Installing required packages...")
    print("Run: pip install funasr>=0.8.0 torch torchaudio")
    print("If you continue to experience issues, please check the documentation at:")
    print("https://github.com/alibaba-damo-academy/FunASR")
    sys.exit(1)

class KWSTest:
    def __init__(self, model_name=None, device="cpu", keywords=None, threshold=0.5):
        """
        Initialize the KWS test module.

        Args:
            model_name (str): The FunASR KWS model to use
            device (str): Device to run the model on ('cpu' or 'cuda:0')
            keywords (list): List of keywords to detect
            threshold (float): Detection threshold (0.0-1.0)
        """
        self.model_name = model_name or "damo/speech_fsmn_kws_char_zh-cn-16k-common"
        self.device = device
        self.keywords = keywords or ["你好", "开始", "结束", "停止", "hello", "start", "stop"]
        self.threshold = threshold
        self.sample_rate = 16000
        self.model = None

        # Audio recording settings
        self.chunk_size = 1600  # 100ms at 16kHz
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.audio_buffer = queue.Queue()

        self.load_model()

    def load_model(self):
        """Load the KWS model and handle potential errors"""
        print(f"Loading KWS model: {self.model_name} on {self.device}...")
        try:
            self.model = AutoModel(
                model=self.model_name,
                device=self.device,
                model_type="kws",
                output_format="json",
                disable_update=True
            )
            print("KWS model loaded successfully!")
        except Exception as e:
            print(f"Error loading KWS model: {e}")
            print("\nPossible solutions:")
            print("1. Check your internet connection")
            print("2. Try specifying a different model")
            print("3. Make sure you have the latest version of FunASR")
            print("4. If using GPU, ensure CUDA is properly set up")
            sys.exit(1)

    def process_audio_file(self, audio_file):
        """
        Process an audio file and detect keywords

        Args:
            audio_file (str): Path to the audio file

        Returns:
            list: Detection results
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return []

        print(f"Processing audio file: {audio_file}")
        try:
            start_time = time.time()
            results = self.model(audio_file)
            elapsed = time.time() - start_time

            print(f"Processing completed in {elapsed:.2f} seconds")
            return self._format_results(results)
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return []

    def _format_results(self, results):
        """Format and filter the detection results"""
        formatted_results = []

        if not results or not isinstance(results, list):
            return formatted_results

        for result in results:
            if isinstance(result, dict):
                if 'text' in result and result.get('score', 0) >= self.threshold:
                    keyword = result.get('text', '')
                    score = result.get('score', 0)
                    start_time = result.get('start_time', 0)
                    end_time = result.get('end_time', 0)

                    formatted_results.append({
                        'keyword': keyword,
                        'score': score,
                        'start_time': start_time,
                        'end_time': end_time
                    })

                    print(f"Detected: '{keyword}' (confidence: {score:.2f}) at {start_time:.2f}s-{end_time:.2f}s")

        return formatted_results

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for real-time processing"""
        self.audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start_microphone_detection(self, duration=None):
        """
        Start real-time keyword detection from microphone

        Args:
            duration (int, optional): Duration in seconds to listen, or None for continuous
        """
        p = pyaudio.PyAudio()

        # Find right input device
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(num_devices):
            if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                print(f"Input Device {i}: {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

        # Open audio stream
        try:
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            print("\nListening for keywords. Press Ctrl+C to stop...")
            stream.start_stream()

            # Create a temp directory if it doesn't exist
            os.makedirs("", exist_ok=True)

            # Processing loop
            start_time = time.time()
            while stream.is_active():
                if duration and time.time() - start_time > duration:
                    break

                # Collect 1 second of audio (10 chunks of 100ms each)
                audio_data = b''
                for _ in range(10):
                    try:
                        chunk = self.audio_buffer.get(timeout=1.0)
                        audio_data += chunk
                    except queue.Empty:
                        continue

                if len(audio_data) > 0:
                    # Save to temporary file
                    temp_file = os.path.join("", f"kws_temp_{int(time.time())}.wav")
                    self._save_audio_to_file(audio_data, temp_file)

                    # Process with KWS model
                    threading.Thread(
                        target=self._process_chunk,
                        args=(temp_file,)
                    ).start()

                time.sleep(0.1)  # Small delay to reduce CPU usage

        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error during microphone detection: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
            p.terminate()
            print("Microphone detection stopped")

    def _process_chunk(self, audio_file):
        """Process a chunk of audio in a separate thread"""
        try:
            results = self.model(audio_file)
            self._format_results(results)
            # Clean up temp file
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception as e:
            print(f"Error processing audio chunk: {e}")

    def _save_audio_to_file(self, audio_data, filename):
        """Save raw audio data to a WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
        except Exception as e:
            print(f"Error saving audio to file: {e}")


def main():
    parser = argparse.ArgumentParser(description="FunASR Keyword Spotting Test")
    parser.add_argument("--model", type=str, default="damo/speech_fsmn_kws_char_zh-cn-16k-common",
                        help="KWS model name or path")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (cpu or cuda:0)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file for testing (WAV format)")
    parser.add_argument("--keywords", type=str, nargs="+",
                        default=["你好", "开始", "结束", "hello", "start", "stop"],
                        help="List of keywords to detect")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (0.0-1.0)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration to listen in seconds (default: continuous)")

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("FunASR Keyword Spotting (KWS) Test")
    print("=" * 60)

    # Initialize KWS system
    kws = KWSTest(
        model_name=args.model,
        device=args.device,
        keywords=args.keywords,
        threshold=args.threshold
    )

    # Either process a file or listen from microphone
    if args.audio:
        kws.process_audio_file(args.audio)
    else:
        kws.start_microphone_detection(duration=args.duration)


if __name__ == "__main__":
    main()
