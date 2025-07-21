import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from funasr import AutoModel

# ================== Configuration Parameters ==================
# Local model path
MODEL_PATH = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")

AUDIO_PATH = "test/test_vad_20250715_120521.wav"  # Input audio path
OUTPUT_DIR = "output/segments"             # Output directory for saving speech segments
VISUALIZE = True                           # Whether to visualize speech activity intervals
SAVE_SEGMENTS = True                       # Whether to save detected speech segments
# ================================================================

def ensure_dir(path):
    """Ensure the output directory exists"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_audio_segment(audio, start_sample, end_sample, sample_rate, output_path):
    """Save audio segment as WAV file"""
    segment = audio[start_sample:end_sample]
    sf.write(output_path, segment, samplerate=sample_rate)
    print(f"Saved segment: {output_path}")

def load_audio(path):
    """Load audio file and convert to mono channel if needed"""
    audio_data, sample_rate = sf.read(path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Only use the first channel
    return audio_data, sample_rate

def plot_vad_result(audio_data, sample_rate, timestamps, total_duration):
    """Visualize speech activity intervals with audio waveform"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot audio waveform in the top subplot
    time_axis = np.linspace(0, total_duration, len(audio_data))
    ax1.plot(time_axis, audio_data, color='blue', linewidth=0.5)
    ax1.set_title("Audio Waveform")
    ax1.set_xlim(0, total_duration)
    ax1.set_ylabel("Amplitude")
    
    # Highlight speech segments in the waveform
    for start, end in timestamps:
        ax1.axvspan(start, end, color='green', alpha=0.2)
    
    # Plot VAD result in the bottom subplot
    ax2.set_title("VAD Result - Speech Activity Detection")
    ax2.set_xlim(0, total_duration)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (seconds)")

    for start, end in timestamps:
        ax2.fill_between([start, end], 0, 1, color="green", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def validate_audio(audio_data, sample_rate):
    """Validate audio data for VAD processing"""
    if len(audio_data) == 0:
        raise ValueError("Audio data is empty")
    
    # Check if audio has enough non-zero values
    non_zero_ratio = np.count_nonzero(audio_data) / len(audio_data)
    if non_zero_ratio < 0.01:  # Less than 1% non-zero
        print(f"WARNING: Audio contains mostly zeros ({non_zero_ratio:.4f} non-zero ratio)")
    
    # Check sample rate
    if sample_rate != 16000:
        print(f"WARNING: VAD model expects 16kHz audio, but input is {sample_rate}Hz")
        
    # Normalize audio if needed
    if np.max(np.abs(audio_data)) > 1.0:
        print("Normalizing audio data to [-1.0, 1.0] range")
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data

def main():
    try:
        # Load model and build VAD Inference Session
        print(f"Loading VAD model from: {MODEL_PATH}")
        vad_model = AutoModel(model=MODEL_PATH, model_type="vad", device="cpu", disable_update=True)
        
        # Read audio file
        print(f"Loading audio file: {AUDIO_PATH}")
        audio_data, sample_rate = load_audio(AUDIO_PATH)
        duration = len(audio_data) / sample_rate
        print(f"Audio loaded: {duration:.2f}s at {sample_rate}Hz")
        
        # Print audio statistics for debugging
        print(f"Audio statistics: min={audio_data.min():.4f}, max={audio_data.max():.4f}, mean={audio_data.mean():.4f}")
        print(f"Non-zero samples: {np.count_nonzero(audio_data)}/{len(audio_data)} ({np.count_nonzero(audio_data)/len(audio_data)*100:.2f}%)")

        # Validate audio data
        audio_data = validate_audio(audio_data, sample_rate)

        # Execute VAD inference using the generate method
        print("Running VAD inference...")
        result = vad_model.generate(audio_data)
        
        # Print raw result for debugging
        print("Raw VAD result:", result)
        
        # Extract timestamps from the result format
        timestamps = []
        if result and isinstance(result, list) and len(result) > 0:
            # Check if there's any detected segments
            if 'value' in result[0] and result[0]['value']:
                timestamps = result[0]['value']
            else:
                print("No speech segments detected in the result")
        
        print("Speech intervals:", timestamps)

        # Draw plot showing speech activity intervals
        if VISUALIZE:
            plot_vad_result(audio_data, sample_rate, timestamps, duration)

        # Save speech segments as separate files
        if SAVE_SEGMENTS and timestamps:
            ensure_dir(OUTPUT_DIR)
            for i, (start_sec, end_sec) in enumerate(timestamps):
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                seg_path = os.path.join(OUTPUT_DIR, f"segment_{i+1}_{start_sec:.2f}-{end_sec:.2f}.wav")
                save_audio_segment(audio_data, start_sample, end_sample, sample_rate, seg_path)
        elif not timestamps:
            print("No speech segments detected!")
            
    except Exception as e:
        print(f"Error in VAD processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()