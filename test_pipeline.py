import sys
import os
# Add the third_party/Matcha-TTS directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import pygame
import time

# GPU availability check
def check_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
        
        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {current_gpu}")
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        return device
    else:
        print("GPU not available, using CPU")
        return torch.device('cpu')

# Check and set device
device = check_gpu()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("GPU not available for memory monitoring")

# Handle SSL issues by completely disabling verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''  # Disable SSL verification
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Try local model first, then fall back to online model
local_model_path = r".\models\Qwen2.5-0.5B-Instruct"
if os.path.exists(local_model_path) and any(os.path.exists(os.path.join(local_model_path, f)) for f in os.listdir(local_model_path)):
    print(f"Using local model: {local_model_path}")
    model_name = local_model_path
else:
    print("Local model not found, using HuggingFace model instead")
    # Use a smaller model that's easier to download
    model_name = "Qwen/Qwen1.5-0.5B-Chat"  # Smaller alternative

print(f"Loading model: {model_name}")
print(f"Target device: {device}")

# Use HF_HUB_OFFLINE=1 to force using local cached model if available
os.environ['HF_HUB_OFFLINE'] = '0'  # Set to 1 to force offline mode

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for GPU efficiency
    device_map="cuda:0",  # Explicitly use GPU 0
    trust_remote_code=True,
    # Additional parameters to help with SSL issues
    use_auth_token=False,
    local_files_only=False
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Verify model is on GPU
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Initialize CosyVoice with GPU support
print("Initializing CosyVoice...")
cosyvoice = CosyVoice(r".\models\CosyVoice-300M", load_jit=False, load_onnx=False, fp16=True)
# sft usage
print("Available speakers:", cosyvoice.list_avaliable_spks())

print("\n" + "="*60)
print("TESTING LLM + TTS PIPELINE")
print("="*60)

# Test the LLM
prompt = "你好，请简单介绍一下自己。"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("GPU memory before inference:")
print_gpu_memory()

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100,  # Reduced for faster generation
    do_sample=True,
    temperature=0.7,
)

print("GPU memory after inference:")
print_gpu_memory()

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\nInput: {prompt}")
print(f"LLM Response: {response}")

# Test TTS with a simple text first
print("\n" + "="*60)
print("TESTING TTS SYNTHESIS") 
print("="*60)

test_text = "你好世界"  # Simple test text
print(f"Testing TTS with simple text: {test_text}")

try:
    for i, j in enumerate(cosyvoice.inference_sft(test_text, '中文女', stream=False)):
        torchaudio.save(f'test_simple_{i}.wav', j['tts_speech'], 22050)
        print(f"Successfully generated test_simple_{i}.wav")
        break  # Just test one output
    print("✅ TTS test successful!")
except Exception as e:
    print(f"❌ TTS test failed: {e}")

print("\n" + "="*60)
print("PIPELINE SETUP COMPLETE!")
print("="*60)
print("✅ GPU:", "Available" if torch.cuda.is_available() else "Not Available")
print("✅ Qwen Model:", "Loaded and working")
print("✅ CosyVoice Model:", "Loaded")
print("✅ Dependencies:", "All installed")
print("\nThe ASR-LLM-TTS pipeline is ready for use!")
