#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Denoiser Module
Provides real-time audio noise reduction using webrtcvad and noisereduce.

Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logging.warning("webrtcvad not available. Will try Silero VAD.")

try:
    import torch
    # Try to load Silero VAD model
    silero_model, silero_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        verbose=False
    )
    SILERO_VAD_AVAILABLE = True
    logging.info("Silero VAD loaded successfully (PyTorch-based, high accuracy)")
except Exception as e:
    SILERO_VAD_AVAILABLE = False
    silero_model = None
    silero_utils = None
    logging.warning(f"Silero VAD not available: {e}")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logging.warning("noisereduce not available. Noise reduction disabled.")

try:
    import torch
    import torchaudio
    from denoiser import pretrained
    FACEBOOK_DENOISER_AVAILABLE = True
except ImportError:
    FACEBOOK_DENOISER_AVAILABLE = False
    logging.warning("Facebook Denoiser not available. Will use noisereduce if available.")


class AudioDenoiser:
    """
    Real-time audio denoiser with adaptive noise estimation.
    
    Features:
    - WebRTC VAD for pre-filtering (skip silence frames)
    - Noisereduce spectral gating for noise reduction
    - Three strength levels: weak, medium, strong
    - Sliding window for noise profile estimation
    - RMS statistics for monitoring
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        strength: str = 'medium',
        vad_mode: int = 2,
        noise_window_duration: float = 0.5,
        denoiser_type: str = 'facebook',
        vad_type: str = 'auto'
    ):
        """
        Initialize AudioDenoiser
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            strength: Denoising strength - 'weak', 'medium', or 'strong'
            vad_mode: WebRTC VAD aggressiveness (0-3, higher = more aggressive) or Silero threshold (0.0-1.0)
            noise_window_duration: Duration of noise estimation window in seconds
            denoiser_type: Type of denoiser - 'facebook' (deep learning, default) or 'noisereduce' (spectral gating)
            vad_type: Type of VAD - 'auto' (prefer Silero > webrtc), 'silero', 'webrtc', or 'none'
        """
        self.sample_rate = sample_rate
        self.strength = strength
        self.vad_mode = vad_mode
        self.denoiser_type = denoiser_type
        self.vad_type = vad_type
        
        # Strength mapping to noisereduce prop_decrease parameter
        self.strength_map = {
            'weak': 0.5,    # Gentle noise reduction, preserves audio quality
            'medium': 0.8,  # Balanced reduction (default)
            'strong': 1.2   # Strong reduction for very noisy environments
        }
        
        if strength not in self.strength_map:
            logging.warning(f"Invalid strength '{strength}', using 'medium'")
            self.strength = 'medium'
        
        # Initialize VAD (prefer Silero > WebRTC)
        self.vad = None
        self.silero_vad = None
        self.silero_threshold = 0.5  # Default threshold for speech detection
        self.vad_active_type = 'none'  # Track which VAD is actually used
        
        if vad_type == 'auto':
            # Auto mode: prefer Silero, fallback to WebRTC
            if SILERO_VAD_AVAILABLE:
                self.silero_vad = silero_model
                self.vad_active_type = 'silero'
                self.silero_threshold = vad_mode if vad_mode <= 1.0 else 0.5
                logging.info(f"Silero VAD initialized (threshold={self.silero_threshold:.2f})")
            elif WEBRTCVAD_AVAILABLE:
                try:
                    self.vad = webrtcvad.Vad(int(vad_mode) if vad_mode > 1 else 2)
                    self.vad_active_type = 'webrtc'
                    logging.info(f"WebRTC VAD initialized (mode={int(vad_mode)})")
                except Exception as e:
                    logging.error(f"Failed to initialize WebRTC VAD: {e}")
        elif vad_type == 'silero' and SILERO_VAD_AVAILABLE:
            self.silero_vad = silero_model
            self.vad_active_type = 'silero'
            self.silero_threshold = vad_mode if vad_mode <= 1.0 else 0.5
            logging.info(f"Silero VAD initialized (threshold={self.silero_threshold:.2f})")
        elif vad_type == 'webrtc' and WEBRTCVAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(int(vad_mode) if vad_mode > 1 else 2)
                self.vad_active_type = 'webrtc'
                logging.info(f"WebRTC VAD initialized (mode={int(vad_mode)})")
            except Exception as e:
                logging.error(f"Failed to initialize WebRTC VAD: {e}")
        elif vad_type != 'none':
            logging.warning(f"Requested VAD type '{vad_type}' not available")
        
        # Noise estimation buffer (sliding window)
        noise_window_samples = int(noise_window_duration * sample_rate)
        self.noise_buffer = deque(maxlen=noise_window_samples)
        
        # Statistics tracking
        self.rms_before_history = deque(maxlen=100)
        self.rms_after_history = deque(maxlen=100)
        
        # Frame count for logging
        self.frame_count = 0
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        
        # Initialize Facebook Denoiser model if selected
        self.facebook_model = None
        self.device = None
        if denoiser_type == 'facebook' and FACEBOOK_DENOISER_AVAILABLE:
            try:
                logging.info("Loading Facebook Denoiser model (DNS64)...")
                self.facebook_model = pretrained.dns64()
                self.facebook_model.eval()
                
                # Use GPU if available, otherwise CPU
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.facebook_model.to(self.device)
                
                logging.info(f"Facebook Denoiser loaded successfully on {self.device}")
            except Exception as e:
                logging.error(f"Failed to load Facebook Denoiser: {e}")
                logging.info("Falling back to noisereduce")
                self.denoiser_type = 'noisereduce'
                self.facebook_model = None
        elif denoiser_type == 'facebook' and not FACEBOOK_DENOISER_AVAILABLE:
            logging.warning("Facebook Denoiser requested but not available. Using noisereduce instead.")
            self.denoiser_type = 'noisereduce'
        
        logging.info(f"AudioDenoiser initialized: "
                    f"denoiser_type={self.denoiser_type}, "
                    f"vad_type={self.vad_active_type}, "
                    f"strength={strength}, sample_rate={sample_rate}Hz")
    
    def set_strength(self, strength: str):
        """
        Change denoising strength
        
        Args:
            strength: 'weak', 'medium', or 'strong'
        """
        if strength in self.strength_map:
            self.strength = strength
            logging.info(f"Denoising strength changed to: {strength}")
        else:
            logging.warning(f"Invalid strength '{strength}', keeping current: {self.strength}")
    
    def _is_speech(self, audio_frame: np.ndarray) -> bool:
        """
        Check if audio frame contains speech using Silero VAD or WebRTC VAD
        
        Args:
            audio_frame: Audio data as float32 array [-1, 1]
        
        Returns:
            True if speech detected, False otherwise
        """
        # No VAD available, assume all frames contain speech
        if self.vad_active_type == 'none':
            return True
        
        # Silero VAD (preferred)
        if self.vad_active_type == 'silero' and self.silero_vad is not None:
            try:
                # Silero VAD requires exactly 512 samples for 16kHz (32ms)
                required_samples = 512 if self.sample_rate == 16000 else 256
                
                # For frames longer than required_samples, use sliding window
                if len(audio_frame) > required_samples:
                    # Check multiple windows and take maximum probability
                    max_prob = 0.0
                    stride = required_samples // 2  # 50% overlap
                    
                    for start in range(0, len(audio_frame) - required_samples + 1, stride):
                        window = audio_frame[start:start + required_samples]
                        audio_tensor = torch.from_numpy(window).float()
                        
                        with torch.no_grad():
                            speech_prob = self.silero_vad(audio_tensor, self.sample_rate).item()
                        
                        max_prob = max(max_prob, speech_prob)
                    
                    return max_prob > self.silero_threshold
                
                # For frames shorter than or equal to required_samples
                elif len(audio_frame) < required_samples:
                    audio_padded = np.pad(audio_frame, (0, required_samples - len(audio_frame)))
                else:
                    audio_padded = audio_frame
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_padded).float()
                
                # Get speech probability
                with torch.no_grad():
                    speech_prob = self.silero_vad(audio_tensor, self.sample_rate).item()
                
                return speech_prob > self.silero_threshold
                
            except Exception as e:
                logging.debug(f"Silero VAD error: {e}")
                return True  # Assume speech on error
        
        # WebRTC VAD (fallback)
        if self.vad_active_type == 'webrtc' and self.vad is not None:
            try:
                # Convert float32 [-1, 1] to int16
                audio_int16 = (audio_frame * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # WebRTC VAD requires frame size of 10, 20, or 30ms
                # For 16kHz: 160, 320, or 480 samples
                frame_duration_ms = len(audio_frame) * 1000 // self.sample_rate
                
                # Adjust to nearest valid duration
                if frame_duration_ms < 15:
                    frame_duration_ms = 10
                elif frame_duration_ms < 25:
                    frame_duration_ms = 20
                else:
                    frame_duration_ms = 30
                
                return self.vad.is_speech(audio_bytes, self.sample_rate)
            
            except Exception as e:
                logging.debug(f"WebRTC VAD error: {e}")
                return True  # Assume speech on error
        
        # No VAD matched, assume speech
        return True
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """
        Calculate Root Mean Square of audio signal
        
        Args:
            audio: Audio data array
        
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(audio ** 2))
    
    def denoise_frame(
        self, 
        audio_frame: np.ndarray,
        skip_vad: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply noise reduction to audio frame
        
        Args:
            audio_frame: Audio data as float32 array [-1, 1]
            skip_vad: If True, skip VAD check and always process
        
        Returns:
            Tuple of (denoised_audio, stats_dict)
            stats_dict contains: {'rms_before', 'rms_after', 'reduction_db', 
                                  'is_speech', 'processed'}
        """
        self.frame_count += 1
        
        # Initialize stats
        stats = {
            'rms_before': 0.0,
            'rms_after': 0.0,
            'reduction_db': 0.0,
            'is_speech': False,
            'processed': False
        }
        
        # Calculate input RMS
        rms_before = self._calculate_rms(audio_frame)
        stats['rms_before'] = rms_before
        self.rms_before_history.append(rms_before)
        
        # Check if noisereduce is available
        if not NOISEREDUCE_AVAILABLE:
            stats['rms_after'] = rms_before
            self.rms_after_history.append(rms_before)
            return audio_frame, stats
        
        # Always check VAD to get true speech status (for statistics and visualization)
        is_speech = self._is_speech(audio_frame)
        stats['is_speech'] = is_speech
        
        if is_speech:
            self.speech_frame_count += 1
        else:
            self.silence_frame_count += 1
        
        # Decide whether to process based on skip_vad flag
        if not skip_vad and not is_speech:
            # VAD-based filtering enabled and no speech detected: skip processing
            self.noise_buffer.extend(audio_frame)
            stats['rms_after'] = rms_before
            self.rms_after_history.append(rms_before)
            return audio_frame, stats
        
        # Update noise buffer
        self.noise_buffer.extend(audio_frame)
        
        # Use Facebook Denoiser if available and selected
        if self.denoiser_type == 'facebook' and self.facebook_model is not None:
            try:
                # Convert numpy array to torch tensor
                # Input shape: [samples] -> [batch=1, channels=1, samples]
                audio_tensor = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
                audio_tensor = audio_tensor.to(self.device)
                
                # Apply denoising with no gradient computation
                with torch.no_grad():
                    denoised_tensor = self.facebook_model(audio_tensor)
                
                # Convert back to numpy: [1, 1, samples] -> [samples]
                denoised = denoised_tensor.squeeze().cpu().numpy()
                
                # Calculate output RMS
                rms_after = self._calculate_rms(denoised)
                stats['rms_after'] = rms_after
                self.rms_after_history.append(rms_after)
                
                # Calculate noise reduction in dB
                if rms_before > 1e-10:
                    reduction_db = 20 * np.log10(rms_after / rms_before)
                    stats['reduction_db'] = reduction_db
                
                stats['processed'] = True
                
                # Periodic logging
                if self.frame_count % 100 == 0:
                    logging.debug(f"Facebook Denoiser stats: frames={self.frame_count}, "
                                f"speech={self.speech_frame_count}, "
                                f"silence={self.silence_frame_count}, "
                                f"avg_reduction={stats['reduction_db']:.1f}dB")
                
                return denoised, stats
                
            except Exception as e:
                logging.error(f"Facebook Denoiser failed: {e}, falling back to original audio")
                stats['rms_after'] = rms_before
                self.rms_after_history.append(rms_before)
                return audio_frame, stats
        
        # Fallback to noisereduce if Facebook Denoiser not available/selected
        try:
            # Apply noise reduction
            prop_decrease = self.strength_map[self.strength]
            
            denoised = nr.reduce_noise(
                y=audio_frame,
                sr=self.sample_rate,
                prop_decrease=prop_decrease,
                stationary=True,  # Assume stationary noise for real-time processing
                n_fft=512,        # Smaller FFT for lower latency
                hop_length=128,   # Smaller hop for better time resolution
                time_constant_s=2.0,  # Smoothing time constant
                freq_mask_smooth_hz=500,  # Frequency smoothing
                time_mask_smooth_ms=50    # Time smoothing
            )
            
            # Calculate output RMS
            rms_after = self._calculate_rms(denoised)
            stats['rms_after'] = rms_after
            self.rms_after_history.append(rms_after)
            
            # Calculate noise reduction in dB
            if rms_before > 1e-10:  # Avoid log(0)
                reduction_db = 20 * np.log10(rms_after / rms_before)
                stats['reduction_db'] = reduction_db
            
            stats['processed'] = True
            
            # Periodic logging
            if self.frame_count % 100 == 0:
                logging.debug(f"Denoiser stats: frames={self.frame_count}, "
                            f"speech={self.speech_frame_count}, "
                            f"silence={self.silence_frame_count}, "
                            f"avg_reduction={stats['reduction_db']:.1f}dB")
            
            return denoised, stats
        
        except Exception as e:
            logging.error(f"Noise reduction failed: {e}")
            stats['rms_after'] = rms_before
            self.rms_after_history.append(rms_before)
            return audio_frame, stats
    
    def get_average_reduction_db(self, last_n: int = 50) -> float:
        """
        Get average noise reduction over last N frames
        
        Args:
            last_n: Number of recent frames to average
        
        Returns:
            Average reduction in dB
        """
        if len(self.rms_before_history) < 2:
            return 0.0
        
        rms_before = list(self.rms_before_history)[-last_n:]
        rms_after = list(self.rms_after_history)[-last_n:]
        
        avg_before = np.mean(rms_before)
        avg_after = np.mean(rms_after)
        
        if avg_before > 1e-10:
            return 20 * np.log10(avg_after / avg_before)
        return 0.0
    
    def get_stats(self) -> dict:
        """
        Get comprehensive denoiser statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_frames': self.frame_count,
            'speech_frames': self.speech_frame_count,
            'silence_frames': self.silence_frame_count,
            'speech_ratio': self.speech_frame_count / max(self.frame_count, 1),
            'avg_rms_before': np.mean(list(self.rms_before_history)) if self.rms_before_history else 0.0,
            'avg_rms_after': np.mean(list(self.rms_after_history)) if self.rms_after_history else 0.0,
            'avg_reduction_db': self.get_average_reduction_db(),
            'strength': self.strength,
            'vad_enabled': self.vad is not None
        }
    
    def reset_stats(self):
        """Reset all statistics counters"""
        self.frame_count = 0
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.rms_before_history.clear()
        self.rms_after_history.clear()
        logging.info("Denoiser statistics reset")


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Audio Denoiser Module Test")
    print("="*60)
    
    # Test initialization
    denoiser = AudioDenoiser(sample_rate=16000, strength='medium')
    
    # Generate test signal (sine wave + noise)
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (1kHz sine wave)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(signal))
    noisy_signal = signal + noise
    
    print(f"\nTest signal generated: {len(noisy_signal)} samples")
    print(f"Original SNR: {10 * np.log10(np.var(signal) / np.var(noise)):.1f} dB")
    
    # Process in chunks (simulate real-time)
    chunk_size = 1600  # 100ms at 16kHz
    denoised_chunks = []
    
    for i in range(0, len(noisy_signal), chunk_size):
        chunk = noisy_signal[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break  # Skip incomplete last chunk
        
        denoised_chunk, stats = denoiser.denoise_frame(chunk)
        denoised_chunks.append(denoised_chunk)
    
    denoised_signal = np.concatenate(denoised_chunks)
    
    # Calculate results
    print("\nProcessing Results:")
    print(f"- Chunks processed: {len(denoised_chunks)}")
    print(f"- Denoised samples: {len(denoised_signal)}")
    
    stats = denoiser.get_stats()
    print(f"\nDenoiser Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.3f}")
        else:
            print(f"- {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)
