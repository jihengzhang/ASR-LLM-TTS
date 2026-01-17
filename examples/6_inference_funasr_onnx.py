import os
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import funasr 
import time

# Model Settings
# model_id = "iic/SenseVoiceSmall-onnx"              # Model ID on ModelScope
model_id = "iic/SenseVoiceSmall"                 # Model ID on ModelScope
# model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
cache_dir = r"C:\Users\212597558\.cache\models"       # Root cache directory for local models
model_dir = os.path.join(cache_dir, model_id.replace("/", os.sep)) # Construct path to local model

# Input audio file to transcribe
input_file = "test_2025-07-21-06-27-17.wav"

print(f"\nInitializing ASR pipeline with model: {model_id}")

WITH_PIPELINE = False

if WITH_PIPELINE:
    try:
        # Initialize the pipeline with error handling
        inference_pipeline = pipeline(
            task = Tasks.auto_speech_recognition, #  Tasks.auto_speech_recognition, #  Tasks.automatic-speech-recognition,  # Use the built-in Tasks constant automatic-speech-recognition
            model = model_id,
            disable_update = True,
            device="cuda:0"                   # Use GPU for inference
        )

    # try:
        # model_id = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            # Initialize the pipeline with error handling
        # inference_pipeline = pipeline(
        #     task=Tasks.auto_speech_recognition,  # Use the Tasks constant
        #     model=model_id,  # Use model_id directly
        #     disable_update=True,
        #     device="cuda:0"                   # Use GPU for inference
        #     # device=DEVICE  # Use the detected best GPU
        # )   
        print(f"\nTranscribing audio file: {input_file}")
        rec_result = inference_pipeline(input_file)
        
        print("\n--- ASR Results ---")
        print(rec_result)
        
        # Extract and display text if available
        if isinstance(rec_result, dict) and 'text' in rec_result:
            print("\n--- Transcribed Text ---")
            print(rec_result['text'])
        
    except Exception as e:
        print(f"\nError during ASR processing: {str(e)}")
        print("\nDebug information:")
        print(f"- Model ID: {model_id}")
        print(f"- Cache directory: {cache_dir}")
        print(f"- Input file exists: {os.path.exists(input_file)}")
        raise  # Re-raise the exception for full traceback if needed
else:
    try:
        start_time = time.time()
        # Initialize FunASR recognizer
        recognizer = funasr.AutoModel(
            model=model_id,          # Use the model ID directly
            cache_dir=cache_dir,     # Specify the cache directory
            model_type="paraformer", # Use "paraformer" for this model
            device="cuda:0",      # Use detected device type
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