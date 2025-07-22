#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import sys
from funasr import AutoModel
import os
from pathlib import Path

# Configuration Options
# ------------------------------------------------------------------------------
# Set USE_LOCAL_MODEL to True to use a local model, False to download from ModelScope
USE_LOCAL_MODEL = True  # Change this to switch between local and remote models

# Model Settings
model_id = "iic/SenseVoiceSmall"                 # Model ID on ModelScope
model_id = "iic/SenseVoiceSmall-onnx"                 # Model ID on ModelScope
cache_dir = r"C:\Users\212597558\.cache\models"  # Root cache directory for local models

# Input audio file to transcribe
input_file = "test_2025-07-21-06-27-17.wav"

# ------------------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------------------
try:
    if USE_LOCAL_MODEL:
        # Use local model with cache_dir
        print(f"Using local model from: {cache_dir}/{model_id}")
        model = AutoModel(
            model=model_id,          # Use the model ID, not the full path
            cache_dir=cache_dir,     # Specify the cache directory where models are stored
            trust_remote_code=True,
        )
    else:
        # Download and use model from ModelScope Hub
        print(f"Downloading model from ModelScope: {model_id}")
        model = AutoModel(
            model=model_id,          # Use the model ID directly
            model_revision="v1.0.0", # Specify version if needed
            trust_remote_code=True,
        )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Fallback to direct path approach
    model_path = os.path.join(cache_dir, "iic", "SenseVoiceSmall-onnx")
    print(f"Trying fallback approach using direct path: {model_path}")
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
    )

res = model.generate(
    input=input_file,
    cache={},
    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
)

print(res)
# import pdb; pdb.set_trace()
print(res[0]['text'].split(">")[-1])
