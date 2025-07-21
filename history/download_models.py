"""
Python script to download models from hf-mirror.com
This script provides an alternative to the PowerShell script for downloading models.
"""

import os
import sys
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_model(model_name, output_dir):
    """Download a model from hf-mirror.com"""
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model name to hf-mirror.com format
    repo_path = model_name.replace("/", "--")
    base_url = f"https://hf-mirror.com/models--{repo_path}/snapshots/"
    
    # Try to get the latest snapshot ID
    try:
        response = requests.get(f"https://hf-mirror.com/{model_name}/tree/main")
        response.raise_for_status()
        
        import re
        snapshot_pattern = f"/models--{repo_path}/snapshots/([a-zA-Z0-9]+)"
        match = re.search(snapshot_pattern, response.text)
        snapshot_id = match.group(1) if match else "main"
        print(f"Found snapshot ID: {snapshot_id}")
    except Exception as e:
        print(f"Error getting snapshot ID: {e}")
        print("Using 'main' instead.")
        snapshot_id = "main"
    
    # Common model files to download
    files_to_download = [
        "config.json",
        "configuration_qwen.py",
        "generation_config.json",
        "modeling_qwen.py",
        "qwen_model.safetensors",
        "qwen_model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_qwen.py"
    ]
    
    # Additional files specific to different models
    if "Qwen1.5" in model_name:
        files_to_download.extend([
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json"
        ])
    
    # Download each file
    success_count = 0
    for file in files_to_download:
        file_url = f"https://hf-mirror.com/{model_name}/resolve/{snapshot_id}/{file}"
        output_path = output_dir / file
        
        print(f"Downloading {file} to {output_path}...")
        if download_file(file_url, output_path):
            success_count += 1
    
    print(f"Download process completed! Successfully downloaded {success_count}/{len(files_to_download)} files.")
    return success_count > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from hf-mirror.com")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="Model name in the format 'organization/model_name'")
    parser.add_argument("--output", type=str, default="./QWen/Qwen2.5-0.5B-Instruct",
                        help="Output directory to save the model files")
    
    args = parser.parse_args()
    download_model(args.model, args.output)
