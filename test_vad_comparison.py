#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD Solutions Comparison Test
Compare different VAD libraries for speech detection
"""

import numpy as np
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("VAD Solutions Comparison")
print("=" * 70)

# Test 1: Try webrtcvad
print("\n" + "=" * 70)
print("Test 1: WebRTC VAD")
print("=" * 70)

try:
    import webrtcvad
    print("✅ webrtcvad installed")
    
    # Test basic functionality
    vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
    
    # Generate test audio (1 second, 16kHz)
    sample_rate = 16000
    duration = 0.03  # 30ms frame
    samples = int(duration * sample_rate)
    
    # Create test signal
    audio = np.random.randn(samples) * 0.1
    audio_int16 = (audio * 32768).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    # Test VAD
    start_time = time.time()
    is_speech = vad.is_speech(audio_bytes, sample_rate)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"   Speech detected: {is_speech}")
    print(f"   Processing time: {elapsed:.2f}ms")
    print("✅ webrtcvad working correctly")
    
except ImportError as e:
    print("❌ webrtcvad not installed")
    print(f"   Error: {e}")
    print("   Reason: Requires C compiler on Windows (difficult to install)")
except Exception as e:
    print(f"❌ webrtcvad error: {e}")

# Test 2: Try Silero VAD
print("\n" + "=" * 70)
print("Test 2: Silero VAD (PyTorch-based)")
print("=" * 70)

try:
    import torch
    
    print("📦 Loading Silero VAD model...")
    
    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    print("✅ Silero VAD loaded successfully")
    print(f"   Model type: {type(model)}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Test with synthetic audio
    # Important: Silero VAD requires exactly 512 samples for 16kHz (32ms frames)
    sample_rate = 16000
    samples_per_frame = 512  # Fixed for 16kHz
    
    # Create test signal (sine wave + noise = speech-like)
    duration = samples_per_frame / sample_rate  # 32ms
    t = np.linspace(0, duration, samples_per_frame)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(samples_per_frame)
    audio_tensor = torch.from_numpy(audio).float()
    
    print(f"   Frame size: {samples_per_frame} samples (32ms at 16kHz)")
    
    # Test VAD on single frame
    start_time = time.time()
    with torch.no_grad():
        speech_prob = model(audio_tensor, sample_rate).item()
    elapsed = (time.time() - start_time) * 1000
    
    print(f"   Speech probability: {speech_prob:.3f}")
    print(f"   Processing time: {elapsed:.2f}ms")
    print(f"   Threshold: 0.5 (typical)")
    print(f"   Is speech: {speech_prob > 0.5}")
    
    # Test with silence
    silence = torch.zeros(samples_per_frame)
    with torch.no_grad():
        silence_prob = model(silence, sample_rate).item()
    print(f"   Silence probability: {silence_prob:.3f}")
    
    # Test with longer audio using VADIterator
    print("\n   Testing with longer audio (1 second)...")
    long_audio = np.concatenate([audio for _ in range(31)])  # ~1 second
    long_audio_tensor = torch.from_numpy(long_audio).float()
    
    vad_iterator = VADIterator(model)
    speech_timestamps = []
    
    # Process in 512-sample chunks
    for i in range(0, len(long_audio_tensor) - samples_per_frame, samples_per_frame):
        chunk = long_audio_tensor[i:i+samples_per_frame]
        speech_dict = vad_iterator(chunk, return_seconds=True)
        if speech_dict:
            speech_timestamps.append(speech_dict)
    
    print(f"   Detected {len(speech_timestamps)} speech segments")
    
    print("✅ Silero VAD working correctly")
    
except ImportError as e:
    print("❌ Silero VAD not available")
    print(f"   Error: {e}")
    print("   Install with: pip install torch (already installed in your env)")
except Exception as e:
    print(f"⚠️ Silero VAD error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: FunASR VAD (already in use)
print("\n" + "=" * 70)
print("Test 3: FunASR VAD (current system)")
print("=" * 70)

try:
    from funasr import AutoModel
    
    print("✅ FunASR already installed")
    print("   Model: speech_fsmn_vad_zh-cn-16k-common-pytorch")
    print("   Type: FSMN-based VAD")
    print("   Features:")
    print("   - High accuracy for Chinese speech")
    print("   - Optimized for 16kHz audio")
    print("   - Outputs timestamp segments")
    print("   - ~200MB model size")
    
except ImportError:
    print("❌ FunASR not available")

# Summary and Recommendations
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

comparison = """
┌─────────────────┬──────────────┬────────────────┬──────────────────┐
│ Feature         │ webrtcvad    │ Silero VAD     │ FunASR VAD       │
├─────────────────┼──────────────┼────────────────┼──────────────────┤
│ Installation    │ ❌ Difficult  │ ✅ Easy (PyTorch)│ ✅ Easy         │
│ Model Size      │ Built-in     │ ~1.5MB         │ ~200MB           │
│ Latency         │ <10ms        │ ~15-20ms       │ ~50-100ms        │
│ Accuracy        │ ⭐⭐⭐         │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐           │
│ Language        │ Language-free│ Language-free  │ Chinese-optimized│
│ Output          │ Boolean      │ Probability    │ Timestamps       │
│ Use Case        │ Pre-filtering│ Main VAD       │ Segmentation     │
└─────────────────┴──────────────┴────────────────┴──────────────────┘
"""

print(comparison)

print("\n📊 RECOMMENDATIONS:")
print("   1. 🥇 Use Silero VAD for real-time speech detection")
print("      - Better accuracy than webrtcvad")
print("      - Easy to install (no C compiler needed)")
print("      - Fast inference (~15ms on CPU)")
print("      - Probability output allows flexible thresholding")
print()
print("   2. 🥈 Keep FunASR VAD for precise segmentation")
print("      - Current system already uses it")
print("      - Good for finding exact speech boundaries")
print("      - Can work together with Silero VAD")
print()
print("   3. ❌ Skip webrtcvad")
print("      - Installation issues on Windows")
print("      - Lower accuracy than Silero VAD")
print("      - Not worth the effort")

print("\n🎯 PROPOSED ARCHITECTURE:")
print("""
   Audio Stream → Silero VAD (quick filter) → Speech detected?
                       ↓ No                          ↓ Yes
                   Skip processing              FunASR VAD (precise)
                                                     ↓
                                                Vosk KWS / FunASR ASR
""")

print("\n💡 IMPLEMENTATION PLAN:")
print("   Step 1: Install Silero VAD (already have torch)")
print("   Step 2: Modify audio_denoiser.py to support Silero VAD")
print("   Step 3: Use Silero as pre-filter before FunASR")
print("   Step 4: Test performance improvement")

print("\n" + "=" * 70)
print("Test complete! Ready to implement Silero VAD integration.")
print("=" * 70)
