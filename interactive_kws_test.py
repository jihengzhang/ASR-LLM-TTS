#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558

FunASR Keyword Spotting (KWS) Interactive Test Script
This script demonstrates how to use the FunASR KWS model for keyword recognition
with detailed step-by-step diagnostics.
"""

import os
import sys
import time
import traceback

# Try to import KWS with detailed error information
print("Attempting to import FunASR KWS module...")
try:
    from funasr import KWS
    print("✓ Successfully imported FunASR KWS module")
except ImportError as e:
    print(f"✗ Failed to import FunASR: {e}")
    print("\nDetailed import path information:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")

    # Suggest installation
    print("\nTo install FunASR, run one of the following commands:")
    print(f"  {sys.executable} -m pip install funasr")
    print("  or")
    print(f"  {sys.executable} -m pip install funasr==0.2.5")

    # Continue execution to show other diagnostics
    print("\nContinuing script execution for further diagnostics...\n")
    KWS = None

# Audio file diagnostics
print("\n--- Audio File Check ---")
default_audio = "test.wav"
if os.path.exists(default_audio):
    print(f"✓ Found test audio file: {default_audio}")
    audio_size = os.path.getsize(default_audio) / 1024  # Size in KB
    print(f"  Size: {audio_size:.2f} KB")
    print(f"  Path: {os.path.abspath(default_audio)}")
else:
    print(f"✗ Test audio file not found: {default_audio}")
    # Look for other WAV files
    print("  Searching for alternative audio files...")
    wav_files = [f for f in os.listdir(".") if f.endswith(".wav")]
    if wav_files:
        print(f"  Found {len(wav_files)} alternative WAV files:")
        for i, wav in enumerate(wav_files[:5], 1):  # Show up to 5 files
            print(f"  {i}. {wav}")
        if len(wav_files) > 5:
            print(f"  ... and {len(wav_files) - 5} more")
    else:
        print("  No WAV files found in the current directory")
        # Check audio_samples folder
        if os.path.exists("audio_samples"):
            sample_files = [f for f in os.listdir("audio_samples") if f.endswith(".wav")]
            if sample_files:
                print(f"  Found {len(sample_files)} WAV files in audio_samples/ folder")
                for i, wav in enumerate(sample_files[:3], 1):  # Show up to 3 files
                    print(f"  {i}. audio_samples/{wav}")
                if len(sample_files) > 3:
                    print(f"  ... and {len(sample_files) - 3} more")

# Define test keywords
print("\n--- Keyword Configuration ---")
keywords = ["hello", "open", "close", "start"]
print(f"Using keywords for detection: {', '.join(keywords)}")

# Load model only if KWS was imported successfully
if KWS is not None:
    print("\n--- Model Loading ---")
    try:
        print("Loading KWS model (this may take some time)...")
        start_time = time.time()
        model = KWS(model="kws-onnx", device="cpu")
        load_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Failed to load KWS model: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nPossible causes:")
        print("1. The model files may not be downloaded or accessible")
        print("2. There might be issues with the FunASR installation")
        print("3. Required dependencies (like PyTorch) might be missing")
        model = None

    # Run inference if model was loaded successfully
    if model is not None and os.path.exists(default_audio):
        print("\n--- Running Inference ---")
        try:
            print(f"Running keyword spotting on: {default_audio}")
            start_time = time.time()
            result = model(audio_file=default_audio, keywords=keywords)
            inference_time = time.time() - start_time
            print(f"✓ Processing completed in {inference_time:.2f} seconds")

            # Display results
            print("\n--- Results ---")
            print("Raw result from model:")
            print(result)

            if result and isinstance(result, dict):
                detected = result.get("detected_keywords", [])
                print("\nDetected Keywords:")
                if detected:
                    for kw in detected:
                        print(f"- {kw}")
                else:
                    print("No keywords detected in the audio.")

                if "timestamps" in result:
                    print("\nTimestamps:")
                    timestamps = result["timestamps"]
                    for kw, times in timestamps.items():
                        for t in times:
                            print(f"  '{kw}' at {t:.2f}s")
            else:
                print("No results or unexpected result format.")
        except Exception as e:
            print(f"✗ Error during keyword spotting: {e}")
            print("\nDetailed error information:")
            traceback.print_exc()

print("\n--- Diagnostics Complete ---")
print("If you experienced issues, try running the install_funasr.bat script")
print("or manually install FunASR in the Python environment used by PyCharm:")
print(f"{sys.executable} -m pip install funasr")
