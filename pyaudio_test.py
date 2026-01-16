#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import pyaudio
import numpy as np
import argparse

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    print(f"Audio callback: buffer length = {len(audio_data)}，frame count = {frame_count}， time info = {time_info}，status = {status}，{pyaudio.paContinue}")
    return (None, pyaudio.paContinue)

def list_devices(pa):
    print("Available audio input devices:")
    info = pa.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(numdevices):
        device_info = pa.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"  [{i}] {device_info.get('name')} (Channels: {device_info.get('maxInputChannels')})")

def main():
    # parser = argparse.ArgumentParser(description="PyAudio device test with callback")
    # parser.add_argument('--rate', type=int, default=16000, help='Sample rate')
    # parser.add_argument('--chunk', type=int, default=1600, help='Chunk size')
    # args = parser.parse_args()

    pa = pyaudio.PyAudio()
    print("\nUsing system default input device...")
    default_device_index = pa.get_default_input_device_info()['index']
    print(f"Default input device: [{default_device_index}] {pa.get_default_input_device_info()['name']}")

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1600,
            input_device_index=default_device_index,
            stream_callback=audio_callback
        )
        stream.start_stream()
        print(f"Started audio stream on default device {default_device_index}. Press Ctrl+C to exit.")
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()