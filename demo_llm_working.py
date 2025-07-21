#!/usr/bin/env python3
"""
WORKING LLM-ONLY DEMO
This script demonstrates the fully working LLM component of the ASR-LLM-TTS pipeline.
Use this while the TTS issues are being resolved.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    else:
        print("⚠️ No GPU available, using CPU")
        return torch.device('cpu')

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory Used: {allocated:.2f} GB")

def main():
    print("🚀 ASR-LLM-TTS Pipeline - LLM Demo")
    print("=" * 50)
    
    # Setup
    device = check_gpu()
    local_model_path = r".\models\Qwen2.5-0.5B-Instruct"
    
    # Load model
    print("📥 Loading Qwen model...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        use_auth_token=False,
        local_files_only=False
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    print("✅ Model loaded successfully!")
    
    # Interactive chat loop
    print("\n💬 Interactive Chat (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n🧑 You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if not prompt:
                continue
                
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            
            # Generate response
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            
            print("🤖 Qwen: ", end="", flush=True)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print_gpu_memory()

if __name__ == "__main__":
    main()
