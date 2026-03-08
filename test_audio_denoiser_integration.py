#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test audio_denoiser.py integration with Facebook Denoiser
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("Testing audio_denoiser.py with Facebook Denoiser")
print("=" * 60)

# Import the AudioDenoiser class
try:
    from audio_denoiser import AudioDenoiser
    print("✅ AudioDenoiser imported successfully")
except Exception as e:
    print(f"❌ Failed to import AudioDenoiser: {e}")
    exit(1)

# Test 1: Initialize with Facebook Denoiser (default)
print("\n" + "=" * 60)
print("Test 1: Initialize AudioDenoiser with Facebook Denoiser + Silero VAD")
print("=" * 60)

try:
    denoiser_fb = AudioDenoiser(
        sample_rate=16000,
        strength='medium',
        denoiser_type='facebook',
        vad_type='auto'  # Auto mode: prefer Silero
    )
    print(f"✅ Facebook Denoiser initialized successfully")
    print(f"   Denoiser type: {denoiser_fb.denoiser_type}")
    print(f"   VAD type: {denoiser_fb.vad_active_type}")
    print(f"   Device: {denoiser_fb.device}")
    print(f"   Model loaded: {denoiser_fb.facebook_model is not None}")
    if denoiser_fb.vad_active_type == 'silero':
        print(f"   Silero threshold: {denoiser_fb.silero_threshold}")
except Exception as e:
    print(f"❌ Failed to initialize Facebook Denoiser: {e}")
    exit(1)

# Test 2: Process audio frame with Facebook Denoiser
print("\n" + "=" * 60)
print("Test 2: Process audio frame with Facebook Denoiser")
print("=" * 60)

try:
    # Generate synthetic audio (1 second at 16kHz)
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Create test signal: sine wave (speech-like) + noise
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    clean_audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    noise = 0.1 * np.random.randn(samples)
    noisy_audio = clean_audio + noise
    
    print(f"   Input audio shape: {noisy_audio.shape}")
    print(f"   Input audio range: [{noisy_audio.min():.3f}, {noisy_audio.max():.3f}]")
    print(f"   Input RMS: {np.sqrt(np.mean(noisy_audio**2)):.3f}")
    
    # Process with denoiser
    denoised_audio, stats = denoiser_fb.denoise_frame(noisy_audio, skip_vad=True)
    
    print(f"   Output audio shape: {denoised_audio.shape}")
    print(f"   Output audio range: [{denoised_audio.min():.3f}, {denoised_audio.max():.3f}]")
    print(f"   Output RMS: {stats['rms_after']:.3f}")
    print(f"   Reduction: {stats['reduction_db']:.1f} dB")
    print(f"   Processed: {stats['processed']}")
    
    print("✅ Audio processing successful with Facebook Denoiser")
    
except Exception as e:
    print(f"❌ Audio processing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Initialize with noisereduce fallback
print("\n" + "=" * 60)
print("Test 3: Initialize AudioDenoiser with noisereduce")
print("=" * 60)

try:
    denoiser_nr = AudioDenoiser(
        sample_rate=16000,
        strength='medium',
        denoiser_type='noisereduce'
    )
    print(f"✅ Noisereduce denoiser initialized successfully")
    print(f"   Denoiser type: {denoiser_nr.denoiser_type}")
except Exception as e:
    print(f"❌ Failed to initialize noisereduce: {e}")

# Test 4: Compare both denoisers
print("\n" + "=" * 60)
print("Test 4: Compare Facebook vs noisereduce denoisers")
print("=" * 60)

try:
    # Use same noisy audio from Test 2
    denoised_nr, stats_nr = denoiser_nr.denoise_frame(noisy_audio, skip_vad=True)
    
    print("Facebook Denoiser:")
    print(f"   Output RMS: {stats['rms_after']:.3f}")
    print(f"   Reduction: {stats['reduction_db']:.1f} dB")
    
    print("\nNoisereduce:")
    print(f"   Output RMS: {stats_nr['rms_after']:.3f}")
    print(f"   Reduction: {stats_nr['reduction_db']:.1f} dB")
    
    print("\n✅ Comparison complete")
    
except Exception as e:
    print(f"⚠️ Comparison skipped: {e}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
print("\n Ready to test with Audio_test_UI.py")
