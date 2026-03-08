#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vosk Model Downloader
Downloads and extracts Vosk speech recognition models for keyword spotting.

Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


# Model configurations
VOSK_MODELS = {
    'cn-small': {
        'name': 'vosk-model-small-cn-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip',
        'size': '42 MB',
        'language': 'Chinese',
        'description': 'Lightweight Chinese model for KWS'
    },
    'en-small': {
        'name': 'vosk-model-small-en-us-0.15',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'size': '40 MB',
        'language': 'English (US)',
        'description': 'Lightweight English model for KWS'
    },
    'cn-large': {
        'name': 'vosk-model-cn-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip',
        'size': '1.3 GB',
        'language': 'Chinese',
        'description': 'Full Chinese model (higher accuracy, slower)'
    }
}


class DownloadProgressBar:
    """Simple progress bar for download tracking"""
    
    def __init__(self, total_size):
        self.total_size = total_size
        self.downloaded = 0
        self.bar_length = 50
    
    def update(self, chunk_size):
        self.downloaded += chunk_size
        if self.total_size > 0:
            progress = self.downloaded / self.total_size
            filled = int(self.bar_length * progress)
            bar = '=' * filled + '-' * (self.bar_length - filled)
            percent = progress * 100
            mb_downloaded = self.downloaded / (1024 * 1024)
            mb_total = self.total_size / (1024 * 1024)
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
            sys.stdout.flush()
    
    def finish(self):
        sys.stdout.write('\n')
        sys.stdout.flush()


def download_file(url, destination, show_progress=True):
    """
    Download file from URL with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        show_progress: Whether to show download progress
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    try:
        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            if show_progress:
                progress_bar = DownloadProgressBar(total_size)
            
            # Download in chunks
            chunk_size = 8192
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if show_progress:
                        progress_bar.update(len(chunk))
            
            if show_progress:
                progress_bar.finish()
        
        print("Download completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    Extract ZIP file to destination folder
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Destination folder
    """
    print(f"\nExtracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed to: {extract_to}")
        return True
    
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False


def download_vosk_model(model_key='cn-small', models_dir=None):
    """
    Download and extract Vosk model
    
    Args:
        model_key: Model identifier ('cn-small', 'en-small', 'cn-large')
        models_dir: Directory to save models (default: ./models/vosk_models/)
    
    Returns:
        Path to extracted model directory, or None if failed
    """
    if model_key not in VOSK_MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(VOSK_MODELS.keys())}")
        return None
    
    model_info = VOSK_MODELS[model_key]
    
    # Setup directories
    if models_dir is None:
        models_dir = Path(__file__).parent / 'models' / 'vosk_models'
    else:
        models_dir = Path(models_dir)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = model_info['name']
    zip_filename = f"{model_name}.zip"
    zip_path = models_dir / zip_filename
    model_path = models_dir / model_name
    
    # Check if model already exists
    if model_path.exists():
        print(f"Model '{model_name}' already exists at: {model_path}")
        user_input = input("Do you want to re-download? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Using existing model.")
            return str(model_path)
        else:
            print("Removing existing model...")
            shutil.rmtree(model_path)
    
    # Display model information
    print("\n" + "="*60)
    print(f"Model: {model_name}")
    print(f"Language: {model_info['language']}")
    print(f"Size: {model_info['size']}")
    print(f"Description: {model_info['description']}")
    print("="*60 + "\n")
    
    # Download model
    if not download_file(model_info['url'], str(zip_path)):
        return None
    
    # Extract model
    if not extract_zip(str(zip_path), str(models_dir)):
        return None
    
    # Clean up ZIP file
    print(f"\nCleaning up ZIP file: {zip_filename}")
    zip_path.unlink()
    
    print(f"\n✅ Model successfully installed at: {model_path}")
    return str(model_path)


def list_available_models():
    """Display all available Vosk models"""
    print("\nAvailable Vosk Models:")
    print("="*60)
    for key, info in VOSK_MODELS.items():
        print(f"\n[{key}]")
        print(f"  Name: {info['name']}")
        print(f"  Language: {info['language']}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
    print("="*60)


def main():
    """Main entry point for command-line usage"""
    print("="*60)
    print("Vosk Model Downloader")
    print("="*60)
    
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
    else:
        list_available_models()
        print("\nWhich model would you like to download?")
        print("Recommended for this project: cn-large (best KWS accuracy)")
        print("If resources are limited, use cn-small")
        model_key = input("\nEnter model key (or 'quit' to exit): ").strip().lower()
        
        if model_key == 'quit':
            print("Exiting...")
            return
    
    if model_key not in VOSK_MODELS:
        print(f"\n❌ Invalid model key: {model_key}")
        list_available_models()
        return
    
    # Download model
    model_path = download_vosk_model(model_key)
    
    if model_path:
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print(f"Model installed at: {model_path}")
        print("\nYou can now use this model in vosk_kws_engine.py")
        print("="*60)
    else:
        print("\n❌ FAILED to download and install model")
        sys.exit(1)


if __name__ == '__main__':
    main()
