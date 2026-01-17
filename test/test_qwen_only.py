from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import ssl
import pathlib

# Handle SSL issues by completely disabling verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''  # Disable SSL verification
os.environ['REQUESTS_CA_BUNDLE'] = ''

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

# Try local model first, then fall back to online model
local_model_path = r".\models\Qwen2.5-0.5B-Instruct"
if pathlib.Path(local_model_path).exists() and any(pathlib.Path(local_model_path).iterdir()):
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

prompt = "你好，你叫什么名字?"
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
    max_new_tokens=512,
)

print("GPU memory after inference:")
print_gpu_memory()
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Input:", prompt)
print("Answer:", response)
print("SUCCESS: Qwen model is working correctly!")
