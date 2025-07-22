import os
import time
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import funasr

# Example showing both ModelScope and FunASR approaches for speech-to-text

# Check for available GPUs
def check_gpu_availability():
    """Check available GPUs and their properties"""
    print("\n=== GPU Information ===")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return "cpu"
    
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    
    # Find the best GPU (preferring RTX if available)
    best_gpu = 0  # Default to first GPU
    found_rtx = False
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        
        # Look for RTX in the name
        if "RTX" in gpu_name and not found_rtx:
            best_gpu = i
            found_rtx = True
            print(f"  - RTX GPU detected! Will prefer this one.")
    
    if gpu_count > 0:
        selected_device = f"cuda:{best_gpu}"
        print(f"Selected GPU: {selected_device} ({torch.cuda.get_device_name(best_gpu)})")
        
        # Check GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(best_gpu).total_memory / (1024**3)  # GB
            reserved_memory = torch.cuda.memory_reserved(best_gpu) / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(best_gpu) / (1024**3)  # GB
            free_memory = total_memory - allocated_memory
            
            print(f"GPU Memory: Total {total_memory:.2f} GB, Free ~{free_memory:.2f} GB")
        except Exception as e:
            print(f"Could not query GPU memory: {e}")
        
        return selected_device
    else:
        return "cpu"

# Get the best available device
DEVICE = check_gpu_availability()

def modelscope_asr():
    """
    Transcribe audio using ModelScope pipeline approach
    """
    print("\n=== ModelScope Pipeline ASR ===")
    
    # Model Settings - Use a working model from ModelScope
    model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    
    # Input audio file to transcribe
    input_file = "test_2025-07-21-06-27-17.wav"
    
    print(f"Initializing ModelScope ASR pipeline with model: {model_id}")
    print(f"Using device: {DEVICE}")
    start_time = time.time()
    
    try:
        # Initialize the pipeline with error handling
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,  # Use the Tasks constant
            model=model_id,  # Use model_id directly
            disable_update=True,
            device=DEVICE  # Use the detected best GPU
        )
        
        print(f"Transcribing audio file: {input_file}")
        rec_result = inference_pipeline(input_file)
        
        print("\n--- ModelScope ASR Results ---")
        print(rec_result)
        
        # Extract and display text if available
        if isinstance(rec_result, dict) and 'text' in rec_result:
            print("\n--- Transcribed Text ---")
            print(rec_result['text'])
        
        elapsed_time = time.time() - start_time
        print(f"ModelScope inference completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nError during ModelScope ASR processing: {str(e)}")
        print("\nDebug information:")
        print(f"- Model ID: {model_id}")
        print(f"- Input file exists: {os.path.exists(input_file)}")


def funasr_asr():
    """
    Transcribe audio using FunASR approach
    """
    print("\n=== FunASR ASR ===")
    
    # Input audio file to transcribe
    input_file = "test_2025-07-21-06-27-17.wav"
    
    # Model configuration for FunASR - using the standard FunASR model
    model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    cache_dir = r"C:\Users\212597558\.cache\models"  # Root cache directory
    
    print(f"Initializing FunASR with model: {model_id}")
    print(f"Using device: {DEVICE}")
    start_time = time.time()
    
    # Extract device type and ID for FunASR
    device_type = "cpu" if DEVICE == "cpu" else "cuda"
    device_id = 0
    if DEVICE.startswith("cuda:"):
        try:
            device_id = int(DEVICE.split(":")[1])
        except:
            device_id = 0
    
    try:
        # Initialize FunASR recognizer
        recognizer = funasr.AutoModel(
            model=model_id,          # Use the model ID directly
            cache_dir=cache_dir,     # Specify the cache directory
            model_type="paraformer", # Use "paraformer" for this model
            device=device_type,      # Use detected device type
            device_id=device_id,     # Specify GPU ID if using CUDA
            trust_remote_code=True,
        )
        
        print(f"Transcribing audio file: {input_file}")
        result = recognizer.generate(input=input_file)
        
        print("\n--- FunASR Results ---")
        print(result)
        
        # Extract and display text from FunASR result
        if result and len(result) > 0:
            if isinstance(result[0], dict) and 'text' in result[0]:
                print("\n--- Transcribed Text ---")
                print(result[0]['text'])
        
        elapsed_time = time.time() - start_time
        print(f"FunASR inference completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nError during FunASR processing: {str(e)}")
        print("\nDebug information:")
        print(f"- Model ID: {model_id}")
        print(f"- Input file exists: {os.path.exists(input_file)}")


if __name__ == "__main__":
    print("Speech-to-Text Demo using ModelScope and FunASR")
    print("Input audio file: test_2025-07-21-06-27-17.wav")
    
    # Check if audio file exists
    if not os.path.exists("test_2025-07-21-06-27-17.wav"):
        print("Warning: test_2025-07-21-06-27-17.wav not found! Please ensure the audio file exists.")
        # Try to use one of the existing wav files in the workspace as a fallback
        sample_files = [f for f in os.listdir('.') if f.endswith('.wav')]
        if sample_files:
            print(f"Found alternative audio files: {sample_files}")
            print(f"Using {sample_files[0]} as fallback input")
            # Create a symbolic link or copy the file
            os.system(f'copy "{sample_files[0]}" test_2025-07-21-06-27-17.wav')
    
    # Run ModelScope pipeline approach
    try:
        modelscope_asr()
    except Exception as e:
        print(f"ModelScope pipeline failed with {DEVICE}: {str(e)}")
        if DEVICE != "cpu":
            print("Attempting to fall back to CPU for ModelScope...")
            try:
                # Save original device and temporarily override global DEVICE
                temp_device = "cpu"
                
                # Define a local function that uses the CPU
                def modelscope_asr_cpu():
                    """
                    Transcribe audio using ModelScope pipeline approach (CPU version)
                    """
                    print("\n=== ModelScope Pipeline ASR (CPU Fallback) ===")
                    
                    # Model Settings - Use a working model from ModelScope
                    model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                    
                    # Input audio file to transcribe
                    input_file = "test_2025-07-21-06-27-17.wav"
                    
                    print(f"Initializing ModelScope ASR pipeline with model: {model_id}")
                    print(f"Using device: cpu")
                    start_time = time.time()
                    
                    try:
                        # Initialize the pipeline with error handling
                        inference_pipeline = pipeline(
                            task=Tasks.auto_speech_recognition,
                            model=model_id,
                            disable_update=True,
                            device="cpu"
                        )
                        
                        print(f"Transcribing audio file: {input_file}")
                        rec_result = inference_pipeline(input_file)
                        
                        print("\n--- ModelScope ASR Results (CPU) ---")
                        print(rec_result)
                        
                        # Extract and display text if available
                        if isinstance(rec_result, dict) and 'text' in rec_result:
                            print("\n--- Transcribed Text ---")
                            print(rec_result['text'])
                        
                        elapsed_time = time.time() - start_time
                        print(f"ModelScope inference completed in {elapsed_time:.2f} seconds")
                        
                    except Exception as e:
                        print(f"\nError during ModelScope ASR CPU processing: {str(e)}")
                
                # Run the CPU version
                modelscope_asr_cpu()
            except Exception as e:
                print(f"ModelScope CPU fallback also failed: {str(e)}")
    
    # Run FunASR approach
    try:
        funasr_asr()
    except Exception as e:
        print(f"FunASR approach failed with {DEVICE}: {str(e)}")
        if DEVICE != "cpu":
            print("Attempting to fall back to CPU for FunASR...")
            try:
                # Define a local function that uses the CPU
                def funasr_asr_cpu():
                    """
                    Transcribe audio using FunASR approach (CPU version)
                    """
                    print("\n=== FunASR ASR (CPU Fallback) ===")
                    
                    # Input audio file to transcribe
                    input_file = "test_2025-07-21-06-27-17.wav"
                    
                    # Model configuration for FunASR
                    model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                    cache_dir = r"C:\Users\212597558\.cache\models"
                    
                    print(f"Initializing FunASR with model (CPU): {model_id}")
                    start_time = time.time()
                    
                    try:
                        # Initialize FunASR recognizer with CPU
                        recognizer = funasr.AutoModel(
                            model=model_id,
                            cache_dir=cache_dir,
                            model_type="paraformer",
                            device="cpu",
                            trust_remote_code=True,
                        )
                        
                        print(f"Transcribing audio file: {input_file}")
                        result = recognizer.generate(input=input_file)
                        
                        print("\n--- FunASR Results (CPU) ---")
                        print(result)
                        
                        # Extract and display text from FunASR result
                        if result and len(result) > 0:
                            if isinstance(result[0], dict) and 'text' in result[0]:
                                print("\n--- Transcribed Text ---")
                                print(result[0]['text'])
                        
                        elapsed_time = time.time() - start_time
                        print(f"FunASR inference completed in {elapsed_time:.2f} seconds")
                        
                    except Exception as e:
                        print(f"\nError during FunASR CPU processing: {str(e)}")
                
                # Run the CPU version
                funasr_asr_cpu()
            except Exception as e:
                print(f"FunASR CPU fallback also failed: {str(e)}")
    
    print("\nDemo completed.")
