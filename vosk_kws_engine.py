#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vosk Keyword Spotting Engine
Lightweight keyword wake-word detection using Vosk offline speech recognition.

Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available. KWS functionality disabled.")


class VoskKWSEngine:
    """
    Vosk-based Keyword Spotting Engine
    
    Features:
    - Offline speech recognition (no internet required)
    - Grammar mode for fast keyword-only recognition
    - Low latency (<100ms typical)
    - Support for custom keyword lists
    - Confidence scoring
    """
    
    def __init__(
        self,
        model_path: str,
        keywords: List[str],
        sample_rate: int = 16000,
        confidence_threshold: float = 0.7,
        use_grammar: bool = True
    ):
        """
        Initialize Vosk KWS Engine
        
        Args:
            model_path: Path to Vosk model directory
            keywords: List of keywords to detect (e.g., ["你好", "小艾"])
            sample_rate: Audio sample rate in Hz
            confidence_threshold: Minimum confidence for keyword detection (0-1)
            use_grammar: Use grammar mode for faster recognition (recommended)
        """
        self.model_path = Path(model_path)
        self.keywords = [kw.lower() for kw in keywords]  # Normalize to lowercase
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.use_grammar = use_grammar
        
        self.model = None
        self.recognizer = None
        
        # Statistics
        self.total_detections = 0
        self.keyword_counts = {kw: 0 for kw in self.keywords}
        
        # Initialize Vosk
        if not VOSK_AVAILABLE:
            logging.error("Vosk is not installed. Please install: pip install vosk")
            raise RuntimeError("Vosk library not available")
        
        self._initialize_vosk()
    
    def _initialize_vosk(self):
        """Initialize Vosk model and recognizer"""
        try:
            # Check if model exists
            if not self.model_path.exists():
                logging.error(f"Vosk model not found at: {self.model_path}")
                logging.info("Please run: python download_vosk_model.py")
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load model
            logging.info(f"Loading Vosk model from: {self.model_path}")
            self.model = Model(str(self.model_path))
            
            # Create recognizer
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            # Set grammar if enabled
            if self.use_grammar:
                self._set_grammar()
            
            # Configure recognizer
            self.recognizer.SetMaxAlternatives(0)  # Don't need alternatives
            self.recognizer.SetWords(False)        # Don't need word-level timing
            
            logging.info(f"Vosk KWS Engine initialized successfully")
            logging.info(f"Keywords: {', '.join(self.keywords)}")
            logging.info(f"Confidence threshold: {self.confidence_threshold}")
        
        except Exception as e:
            logging.error(f"Failed to initialize Vosk: {e}")
            raise
    
    def _set_grammar(self):
        """
        Set grammar to restrict recognition to keywords only
        This significantly speeds up recognition
        """
        # Create grammar with keywords
        # Format: ["[unk]", "keyword1", "keyword2", ...]
        grammar_words = ["[unk]"] + self.keywords
        
        try:
            # Vosk grammar format
            grammar = json.dumps(grammar_words, ensure_ascii=False)
            self.recognizer.SetGrammar(grammar)
            logging.info(f"Grammar mode enabled with {len(self.keywords)} keywords")
        except Exception as e:
            logging.warning(f"Failed to set grammar: {e}")
            logging.warning("Falling back to full recognition mode")
    
    def detect_keyword(
        self, 
        audio_frame: np.ndarray,
        return_partial: bool = False
    ) -> Dict:
        """
        Process audio frame and detect keywords
        
        Args:
            audio_frame: Audio data as float32 array [-1, 1]
            return_partial: If True, also return partial results
        
        Returns:
            Dictionary with detection results:
            {
                'detected': bool,
                'keyword': str or None,
                'text': str (full recognized text),
                'confidence': float,
                'partial': str (partial result, if return_partial=True)
            }
        """
        if self.recognizer is None:
            return {'detected': False, 'keyword': None, 'text': '', 'confidence': 0.0}
        
        try:
            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_frame * 32768.0).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Feed audio to recognizer
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Final result available
                result_json = self.recognizer.Result()
                result = json.loads(result_json)
                
                text = result.get('text', '').strip().lower()
                confidence = result.get('confidence', 0.0)
                
                # Check if any keyword is in the recognized text
                detected_keyword = None
                for keyword in self.keywords:
                    if keyword in text:
                        detected_keyword = keyword
                        break
                
                # Update statistics
                if detected_keyword and confidence >= self.confidence_threshold:
                    self.total_detections += 1
                    self.keyword_counts[detected_keyword] += 1
                    
                    logging.info(f"✅ Keyword detected: '{detected_keyword}' "
                               f"(confidence: {confidence:.2f})")
                    
                    return {
                        'detected': True,
                        'keyword': detected_keyword,
                        'text': text,
                        'confidence': confidence
                    }
            
            # Check partial result if requested
            if return_partial:
                partial_json = self.recognizer.PartialResult()
                partial = json.loads(partial_json)
                partial_text = partial.get('partial', '').strip().lower()
                
                return {
                    'detected': False,
                    'keyword': None,
                    'text': '',
                    'confidence': 0.0,
                    'partial': partial_text
                }
            
            return {'detected': False, 'keyword': None, 'text': '', 'confidence': 0.0}
        
        except Exception as e:
            logging.error(f"Error in keyword detection: {e}")
            return {'detected': False, 'keyword': None, 'text': '', 'confidence': 0.0}
    
    def reset(self):
        """Reset recognizer state (call between utterances)"""
        if self.recognizer:
            try:
                # Get and discard any pending results
                _ = self.recognizer.FinalResult()
            except:
                pass
            
            # Recreate recognizer for clean state
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            if self.use_grammar:
                self._set_grammar()
            self.recognizer.SetMaxAlternatives(0)
            self.recognizer.SetWords(False)
    
    def add_keyword(self, keyword: str):
        """
        Add a new keyword to the detection list
        
        Args:
            keyword: New keyword to add
        """
        keyword_lower = keyword.lower()
        if keyword_lower not in self.keywords:
            self.keywords.append(keyword_lower)
            self.keyword_counts[keyword_lower] = 0
            
            # Update grammar if using grammar mode
            if self.use_grammar:
                self._set_grammar()
            
            logging.info(f"Keyword added: '{keyword}'")
    
    def remove_keyword(self, keyword: str):
        """
        Remove a keyword from the detection list
        
        Args:
            keyword: Keyword to remove
        """
        keyword_lower = keyword.lower()
        if keyword_lower in self.keywords:
            self.keywords.remove(keyword_lower)
            
            # Update grammar if using grammar mode
            if self.use_grammar:
                self._set_grammar()
            
            logging.info(f"Keyword removed: '{keyword}'")
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_detections': self.total_detections,
            'keyword_counts': self.keyword_counts.copy(),
            'keywords': self.keywords.copy(),
            'confidence_threshold': self.confidence_threshold,
            'use_grammar': self.use_grammar
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_detections = 0
        self.keyword_counts = {kw: 0 for kw in self.keywords}
        logging.info("Statistics reset")


# Example usage and testing
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Vosk KWS Engine Test")
    print("="*60)
    
    # Configuration
    model_path = Path(__file__).parent / 'models' / 'vosk_models' / 'vosk-model-small-cn-0.22'
    keywords = ["你好", "小艾", "开始", "小度", "小爱"]
    
    # Check if model exists
    if not model_path.exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please run: python download_vosk_model.py cn-small")
        print("="*60)
        exit(1)
    
    try:
        # Initialize engine
        print(f"\nInitializing KWS engine...")
        print(f"Model: {model_path}")
        print(f"Keywords: {keywords}")
        
        kws = VoskKWSEngine(
            model_path=str(model_path),
            keywords=keywords,
            sample_rate=16000,
            confidence_threshold=0.7,
            use_grammar=True
        )
        
        print("\n✅ KWS Engine initialized successfully!")
        
        # Generate test audio (you would replace this with actual mic input)
        print("\n📝 Note: This is a basic initialization test.")
        print("For full testing, integrate with audio input pipeline.")
        
        # Test with dummy audio
        sample_rate = 16000
        duration = 0.5  # 500ms
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        
        result = kws.detect_keyword(dummy_audio)
        print(f"\nTest detection result: {result}")
        
        # Display statistics
        stats = kws.get_statistics()
        print("\nEngine Statistics:")
        for key, value in stats.items():
            print(f"- {key}: {value}")
        
        print("\n" + "="*60)
        print("✅ Test completed!")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("="*60)
        exit(1)
