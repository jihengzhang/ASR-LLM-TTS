#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from funasr import AutoModel
import pyaudio
import re
# ================== Configuration Parameters ==================
# Local model path

VISUALIZE = True                           # Whether to visualize speech activity intervals
SAVE_SEGMENTS = False                       # Whether to save detected speech segments
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

def plot_vad_result(audio_data, sample_rate, vad_timestamps, total_duration, kws_results=None):
    # 设置matplotlib支持中文
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 任选其一
    matplotlib.rcParams['axes.unicode_minus'] = False

    """Visualize speech activity intervals with audio waveform and KWS results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot audio waveform in the top subplot
    time_axis = np.linspace(0, total_duration, len(audio_data))
    if isinstance(audio_data, bytes):
        # audio_data = audio_data.astype(np.float32) #'bytes' object has no attribute 'astype'
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
    ax1.plot(time_axis, audio_data, color='blue', linewidth=0.5)
    ax1.set_title("Audio Waveform")
    ax1.set_xlim(0, total_duration)
    ax1.set_ylabel("Amplitude")
    
    # Highlight speech segments in the waveform
    for start, end in vad_timestamps:
        ax1.axvspan(start, end, color='green', alpha=0.2)
    # 显示KWS识别结果（上方不再显示）
    
    # Calculate mean amplitude using a sliding window
    window_size = int(sample_rate * 0.02)  # 20ms window
    stride = max(1, window_size // 2)  # 50% overlap
    mean_amplitudes = []
    time_points = []
    
    for i in range(0, len(audio_data) - window_size, stride):
        current_time = i / sample_rate
        window = audio_data[i:i+window_size]
        
        # Check if current window is within any timestamp period
        is_speech = False
        for start, end in vad_timestamps:
            if i >= start and i <= end:
                is_speech = True
                break
        
        # Only calculate amplitude if in speech segment, otherwise set to 0
        if is_speech:
            mean_amplitude = np.max(np.abs(window))
        else:
            mean_amplitude = 0
            
        mean_amplitudes.append(mean_amplitude)
        time_points.append(current_time)
    
    # Plot mean amplitude in the bottom subplot
    ax2.set_title("Mean Amplitude (Speech Segments Only)")
    ax2.set_xlim(0, total_duration)
    ax2.plot(time_points, mean_amplitudes, color='blue', linewidth=1)
    ax2.set_ylabel("Mean Amplitude")
    ax2.set_xlabel("Time (seconds)")
    
    # Add speech activity highlighting in the bottom subplot
    for start, end in vad_timestamps:
        ax2.axvspan(start, end, color='green', alpha=0.2)
    # 在下方显示KWS识别点，y坐标不要太靠下
    if kws_results:
        for kws_time, kws_word in kws_results:
            # 采样点转为秒，确保和波形对齐
            kws_time_sec = kws_time / sample_rate
            ax2.axvline(kws_time_sec, color='red', linestyle='--', alpha=0.7)
            y_base = max(mean_amplitudes) if mean_amplitudes else 1.0
            y_offset = y_base * 0.7  # 0.7倍最大值，避免太靠下
            # 关键字与竖线左对齐，略微右移避免重叠
            ax2.text(
                kws_time_sec + 0.01,  # 右移0.01秒
                y_offset,
                kws_word,
                color='red',
                rotation=0,
                horizontalalignment='left',  # 右对齐
                verticalalignment='bottom',
                fontsize=10,
                clip_on=True
            )
    
    plt.tight_layout()
    plt.show(block=True)

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
    """
    0: use soundfile to read audio file, return float64, normalized
    1: use wave to read audio file, convert to int16, NOT normalized
    2: use wave to read audio file, convert to float32, normalized
    """
    MODEL_PATH = os.path.join("models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join("C:\\Users\\212597558\\.cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    model_vad_id = r"damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # VAD model name 1.6MB

    AUDIO_PATH = "test/test_vad_20250715_120521.wav"  # Input audio path
    AUDIO_PATH = r"test_2025-07-22-10-41-06.wav"
    AUDIO_PATH = r"test_music_开始声音测试.wav"
    # AUDIO_PATH = r"test.wav"
    OUTPUT_DIR = "output/segments"             # Output directory for saving speech segments

    vad_model = AutoModel(model=MODEL_PATH, model_type="vad", device="cuda", disable_update=True) # works ok
    # vad_model = AutoModel(model=model_vad, model_type="vad", device="cuda", disable_update=True)
    kws_model_id =  r"iic\\SenseVoiceSmall"
    kws_model_path = os.path.join(os.path.expanduser("~"), ".cache\\models", kws_model_id)        
    kws_model = AutoModel(model=kws_model_path, model_type="kws", device="cuda:0", disable_update=True)

    method = 1
    method = "kws0"
    try:
        # Load model and build VAD Inference Session
        print(f"Loading VAD model from: {MODEL_PATH}")

        # Read audio file
        if method == 0: # Audio data as int 64 float, normalized
            print(f"Loading audio file: {AUDIO_PATH}")
            # audio_data, sample_rate = load_audio(AUDIO_PATH)
            audio_data, sample_rate = sf.read(AUDIO_PATH)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Only use the first channel
            duration = len(audio_data) / sample_rate
            print(f"Audio loaded: {duration:.2f}s at {sample_rate}Hz")
        
            # Print audio statistics for debugging
            print(f"Audio statistics: min={audio_data.min():.4f}, max={audio_data.max():.4f}, mean={audio_data.mean():.4f}")
            print(f"Non-zero samples: {np.count_nonzero(audio_data)}/{len(audio_data)} ({np.count_nonzero(audio_data)/len(audio_data)*100:.2f}%)")

            result = vad_model.generate(audio_data) 
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

        elif method == 1: # use wave to read audio file, int 16 or byte, Non-normalized
            # Simulate live input from audio file
            import wave
            wf = wave.open(AUDIO_PATH, 'rb')
            chunk = 6400
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            print(f"Simulating live input from {AUDIO_PATH}...")

            all_audio = []
            all_timestamps = []
            total_samples = 0

            while True:
                audio_data = wf.readframes(chunk)
                if not audio_data:
                    break
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    audio_data = audio_data[:, 0]
                # 累积音频数据
                all_audio.append(audio_data)
                duration = len(audio_data) / sample_rate / n_channels
                # VAD 推理
                result = vad_model.generate(audio_data)
                # 累积时间戳（修正为全局时间）
                if result and isinstance(result, list) and len(result) > 0:
                    if 'value' in result[0] and result[0]['value']:
                        # 将 chunk 内的时间戳转换为全局时间戳
                        for seg in result[0]['value']:
                            start, end = seg
                            global_start = start + total_samples
                            global_end = end + total_samples
                            all_timestamps.append((global_start, global_end))
                    else:
                        print("No speech segments detected in the result")
                total_samples += len(audio_data)
                print("Speech intervals:", result[0]['value'] if result and isinstance(result, list) and len(result) > 0 and 'value' in result[0] else [])

            # 合并所有音频数据
            all_audio_data = np.concatenate(all_audio) if all_audio else np.array([])
            total_duration = len(all_audio_data) / sample_rate / n_channels

            # merged_timestamps = merge_intervals(all_timestamps)

            if VISUALIZE and len(all_audio_data) > 0:
                plot_vad_result(all_audio_data, sample_rate, all_timestamps, total_duration)

        elif method == 2:  # Simulate live input from audio file, float32, normalized
            import wave
            wf = wave.open(AUDIO_PATH, 'rb')
            chunk = 1600
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            print(f"Simulating live input from {AUDIO_PATH}...")

            all_audio = []
            all_timestamps = []
            total_samples = 0

            while True:
                audio_data = wf.readframes(chunk)
                if not audio_data:
                    break
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    audio_data = audio_data[:, 0]
                audio_data_norm = audio_data.astype(np.float32) / 32768.0
                all_audio.append(audio_data_norm)
                duration = len(audio_data_norm) / sample_rate
                result = vad_model.generate(audio_data_norm)
                if result and isinstance(result, list) and len(result) > 0:
                    if 'value' in result[0] and result[0]['value']:
                        for seg in result[0]['value']:
                            start, end = seg
                            global_start = start + total_samples / sample_rate
                            global_end = end + total_samples / sample_rate
                            all_timestamps.append((global_start, global_end))
                    else:
                        print("No speech segments detected in the result")
                total_samples += len(audio_data_norm)
                print("Speech intervals:", result[0]['value'] if result and isinstance(result, list) and len(result) > 0 and 'value' in result[0] else [])

            all_audio_data = np.concatenate(all_audio) if all_audio else np.array([])
            total_duration = len(all_audio_data) / sample_rate

            def merge_intervals(intervals):
                if not intervals:
                    return []
                intervals.sort()
                merged = [intervals[0]]
                for current in intervals[1:]:
                    prev = merged[-1]
                    if current[0] <= prev[1]:
                        merged[-1] = (prev[0], max(prev[1], current[1]))
                    else:
                        merged.append(current)
                return merged
            merged_timestamps = merge_intervals(all_timestamps)

            if VISUALIZE and len(all_audio_data) > 0:
                plot_vad_result(all_audio_data, sample_rate, merged_timestamps, total_duration)

        elif method == "kws0": # use wave to read audio file, int 16 or byte, Non-normalized
            # Simulate live input from audio file

            import wave
            wf = wave.open(AUDIO_PATH, 'rb')
            sample_rate = wf.getframerate()
            chunk = 6 * sample_rate
            n_channels = wf.getnchannels()
            print(f"Simulating live input from {AUDIO_PATH}...")

            all_audio = []
            vad_timestamps = []
            kws_results = []
            total_samples = 0

            # 滑动窗口参数
            kws_chunk = chunk
            min_chunk = 1 * sample_rate  # 最小窗口长度
            max_chunk = 5 * sample_rate  # 最大窗口长度
            kws_buffer = np.array([], dtype=np.int16)
            kws_last_word_sample = {}   # word: last_sample_index
            amplitude_threshold = 100   # 静音阈值（可根据实际调整）
            window_size = 8000  # 窗口大小（可根据实际调整）

            while True:
                audio_data = wf.readframes(chunk)
                if not audio_data:
                    break
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    audio_data = audio_data[:, 0]
                # 累积音频数据
                all_audio.append(audio_data)
                duration = len(audio_data) / sample_rate / n_channels
                # VAD 推理

                # KWS 自适应边界滑动窗口推理
                kws_buffer = np.concatenate([kws_buffer, audio_data])
                buffer_len = len(kws_buffer)
                start_pos = 0
                # 记录本chunk的起始采样点
                chunk_start_sample = total_samples
                while start_pos + min_chunk <= buffer_len:
                    # 判断起始点电平
                    start_window = kws_buffer[start_pos : start_pos + window_size]
                    if start_window.size < window_size:
                        break  # 剩余数据不足一个窗口
                    start_mean = np.mean(np.abs(start_window))
                    if start_mean < amplitude_threshold:
                        # 电平低于阈值，跳过，向后滑动
                        start_pos += window_size
                        continue

                    # 在[min_chunk, max_chunk]范围内寻找幅值均值小于阈值的点（窗口为window_size）
                    search_start = start_pos + min_chunk
                    search_end = min(start_pos + max_chunk, buffer_len)
                    boundary = None
                    for i in range(search_start, search_end, window_size):
                        window_end = min(i + window_size, search_end)
                        window = kws_buffer[i:window_end]
                        if window.size == 0:
                            continue
                        mean_amp = np.mean(np.abs(window))
                        if mean_amp < amplitude_threshold:
                            boundary = window_end
                            break
                    if boundary is None:
                        # 没有找到静音点，强制用max_chunk
                        boundary = min(start_pos + max_chunk, buffer_len)
                    chunk_data = kws_buffer[start_pos:boundary]
                    if len(chunk_data) < min_chunk:
                        break  # 剩余数据太短，等待下次补齐
                    audio_data_float = chunk_data.astype(np.float32) / 32768.0
                    kws_result = kws_model.generate(audio_data_float)
                    vad_result = vad_model.generate(audio_data_float)
                    # 只保存VAD时间戳
                    if vad_result and isinstance(vad_result, list) and len(vad_result) > 0:
                        if 'value' in vad_result[0] and vad_result[0]['value']:
                            for seg in vad_result[0]['value']:
                                start, end = seg
                                global_start = start + total_samples
                                global_end = end + total_samples
                                vad_timestamps.append((global_start, global_end))
                        else:
                            print("No speech segments detected in the result")
                    if kws_result and isinstance(kws_result, list):
                        for item in kws_result:
                            if 'text' in item or 'timestamp' in item:
                                ts = item['timestamp'] if ('timestamp' in item and item['timestamp'] is not None) else 0
                                # 采样点为单位，需加上本chunk的起始采样点和窗口内偏移
                                kws_sample = int(ts + chunk_start_sample + start_pos)
                                kws_word = item['text'] if 'text' in item else ''
                                kws_word = re.sub(r"<\|.*?\|>", "", kws_word)
                                # 去除滑窗重复：同一关键词在相邻窗口只保留一次（如采样点差大于16000才保留，约1秒）
                                if len(kws_word) > 0:
                                    last_sample = kws_last_word_sample.get(kws_word, -99999999)
                                    if abs(kws_sample - last_sample) > 16000:
                                        kws_results.append((kws_sample, kws_word))
                                        kws_last_word_sample[kws_word] = kws_sample
                    start_pos = boundary
                # 保留未处理的尾部
                if start_pos < buffer_len:
                    kws_buffer = kws_buffer[start_pos:]
                else:
                    kws_buffer = np.array([], dtype=np.int16)
                total_samples += len(audio_data)
                if kws_result:
                    print("KWS result:", kws_result)

            all_audio_data = np.concatenate(all_audio) if all_audio else np.array([])
            total_duration = len(all_audio_data) / sample_rate / n_channels

            if VISUALIZE and len(all_audio_data) > 0:
                plot_vad_result(all_audio_data, sample_rate, vad_timestamps, total_duration, kws_results=kws_results)
        # Save speech segments as separate files
        if SAVE_SEGMENTS and timestamps:
            ensure_dir(OUTPUT_DIR)
            for i, (start_sec, end_sec) in enumerate(timestamps):
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                seg_path = os.path.join(OUTPUT_DIR, f"segment_{i+1}_{start_sec:.2f}-{end_sec:.2f}.wav")
                # save_audio_segment(audio_data, start_sample, end_sample, sample_rate, seg_path)   # not save segment now
        elif not timestamps:
            print("No speech segments detected!")

        pass

    except Exception as e:
        print(f"Error in VAD/KWS processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()