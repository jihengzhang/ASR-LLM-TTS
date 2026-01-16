#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio_test_UI_streamlit.py - Streamlit-based UI for FunASR VAD/KWS pipeline

Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558

This is a web-based version of the wxPython Audio_test_UI using Streamlit.
Features:
- Audio device selection with dropdown
- Start/Stop/Refresh controls
- Real-time VAD/KWS results display
- Real-time waveform visualization
- Recording indicator and status display
"""

import streamlit as st
import threading
import time
from datetime import datetime
import traceback
import os

# Import the processor
try:
    from FunASR_VAD_KWS_plot import VADKWSProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing VADKWSProcessor: {e}")
    PROCESSOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Audio Test UI - FunASR VAD/KWS",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-recording {
        background-color: #ffcccc;
        border-left: 4px solid #ff0000;
    }
    .status-idle {
        background-color: #ccffcc;
        border-left: 4px solid #00cc00;
    }
    .status-running {
        background-color: #ccccff;
        border-left: 4px solid #0000ff;
    }
    .results-container {
        background-color: #f0f0f0;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processor_started = False
    st.session_state.audio_devices = []
    st.session_state.selected_device = None
    st.session_state.results_history = []
    st.session_state.last_update = 0
    st.session_state.processor_thread = None
    st.session_state.device_changed = False

def initialize_processor():
    """Initialize the VAD/KWS processor"""
    try:
        if st.session_state.processor is None:
            st.session_state.processor = VADKWSProcessor(
                sample_rate=16000,
                chunk_size=8000,
                threshold=0.02,
                channels=1,
                silence_duration=0.3,
                buffer_duration=5.0,
                time_to_end_conversation=3,
                keywords=["hello", "Hi panda", "hi siri", "你好"],
                stopwords=["stop", "停止", "okay", "好了", "行了", "好的", "退出"]
            )
            st.session_state.processor.isDebug = False
            
            # Get audio devices
            refresh_audio_devices()
            
            st.session_state.results_history.append(
                f"{datetime.now().strftime('%H:%M:%S')}: Processor initialized"
            )
            
            return True
    except Exception as e:
        st.error(f"Error initializing processor: {e}")
        traceback.print_exc()
        return False

def refresh_audio_devices():
    """Refresh the list of available audio devices"""
    try:
        if st.session_state.processor:
            devices, default_device = st.session_state.processor.get_audio_devices()
            st.session_state.audio_devices = devices
            
            if default_device:
                st.session_state.selected_device = default_device[0]
            elif devices:
                st.session_state.selected_device = devices[0][0]
            
            st.session_state.results_history.append(
                f"{datetime.now().strftime('%H:%M:%S')}: Audio devices refreshed ({len(devices)} devices found)"
            )
            return True
    except Exception as e:
        st.error(f"Error refreshing audio devices: {e}")
        return False

def get_device_names():
    """Get list of device names for selection"""
    return [f"{name} (Device {device_id})" for device_id, name in st.session_state.audio_devices]

def start_processor():
    """Start the processor in a separate thread"""
    try:
        if st.session_state.processor and not st.session_state.processor_started:
            def run_processor():
                try:
                    st.session_state.processor.start()
                except Exception as e:
                    st.session_state.results_history.append(
                        f"{datetime.now().strftime('%H:%M:%S')}: ERROR starting processor: {e}"
                    )
                    st.session_state.processor_started = False
            
            st.session_state.processor_thread = threading.Thread(target=run_processor, daemon=True)
            st.session_state.processor_thread.start()
            st.session_state.processor_started = True
            st.session_state.results_history.append(
                f"{datetime.now().strftime('%H:%M:%S')}: Processor started"
            )
            return True
    except Exception as e:
        st.error(f"Error starting processor: {e}")
        return False

def stop_processor():
    """Stop the processor"""
    try:
        if st.session_state.processor and st.session_state.processor_started:
            st.session_state.processor.stop()
            st.session_state.processor_started = False
            st.session_state.results_history.append(
                f"{datetime.now().strftime('%H:%M:%S')}: Processor stopped"
            )
            return True
    except Exception as e:
        st.error(f"Error stopping processor: {e}")
        return False

def get_processor_results():
    """Get the latest results from processor"""
    results = []
    if st.session_state.processor and hasattr(st.session_state.processor, 'detected_keywords'):
        for kw, ts in list(st.session_state.processor.detected_keywords)[-10:]:
            if isinstance(ts, float):
                ts_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            else:
                ts_str = str(ts)
            results.append(f"{ts_str}: {kw}")
    return results

def get_processor_status():
    """Get the current status of the processor"""
    if st.session_state.processor and hasattr(st.session_state.processor, 'status'):
        return st.session_state.processor.status
    return "Idle"

def display_waveform():
    """Display the current waveform plot"""
    if st.session_state.processor and hasattr(st.session_state.processor, 'fig'):
        try:
            st.pyplot(st.session_state.processor.fig)
        except Exception as e:
            st.warning(f"Could not display waveform: {e}")

# Main UI
st.markdown("<h1 class='main-title'>🎤 Audio Test UI - FunASR VAD/KWS</h1>", unsafe_allow_html=True)

# Initialize processor on first load
if not PROCESSOR_AVAILABLE:
    st.error("VADKWSProcessor not available. Please check dependencies.")
    st.stop()

if st.session_state.processor is None:
    initialize_processor()

# Create two columns for top controls
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.markdown("#### Audio Input Device")
    if st.session_state.audio_devices:
        device_options = get_device_names()
        selected_idx = 0
        
        # Find the index of previously selected device
        for i, (device_id, _) in enumerate(st.session_state.audio_devices):
            if device_id == st.session_state.selected_device:
                selected_idx = i
                break
        
        selected_device_name = st.selectbox(
            "Select audio input device:",
            device_options,
            index=selected_idx,
            key="audio_device",
            label_visibility="collapsed"
        )
        
        # Update selected device
        selected_idx = device_options.index(selected_device_name)
        new_device_id = st.session_state.audio_devices[selected_idx][0]
        
        if new_device_id != st.session_state.selected_device:
            st.session_state.selected_device = new_device_id
            if st.session_state.processor:
                st.session_state.processor.input_device_index = new_device_id
                st.session_state.device_changed = True
            
            st.session_state.results_history.append(
                f"{datetime.now().strftime('%H:%M:%S')}: Changed device to {device_options[selected_idx]}"
            )
    else:
        st.warning("No audio devices found. Please refresh.")

with col2:
    st.markdown("#### Device Management")
    if st.button("🔄 Refresh Devices", use_container_width=True):
        if refresh_audio_devices():
            st.success("Audio devices refreshed")
            st.rerun()

with col3:
    st.markdown("#### Processor Control")
    col3a, col3b = st.columns(2)
    
    with col3a:
        if st.button("▶️ Start", use_container_width=True, 
                     disabled=st.session_state.processor_started):
            if start_processor():
                st.rerun()
    
    with col3b:
        if st.button("⏹️ Stop", use_container_width=True,
                     disabled=not st.session_state.processor_started):
            if stop_processor():
                st.rerun()

# Status indicator
st.markdown("---")
col_status1, col_status2 = st.columns([1, 3])

with col_status1:
    status = get_processor_status()
    if st.session_state.processor_started:
        st.markdown('<div class="status-box status-running">🟦 <b>RUNNING</b></div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-idle">🟩 <b>IDLE</b></div>', 
                   unsafe_allow_html=True)

with col_status2:
    st.markdown(f"**Status:** {status}")

st.markdown("---")

# Main content area with two columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 📊 Real-time Waveform")
    display_waveform()

with right_col:
    st.markdown("### 📋 Information")
    st.markdown("""
    **Processor Config:**
    - Sample Rate: 16000 Hz
    - Chunk Size: 8000 
    - VAD Threshold: 0.02
    - Channels: 1 (Mono)
    
    **Keywords:**
    - hello
    - Hi panda
    - hi siri
    - 你好
    """)

# Results section below
st.markdown("---")
st.markdown("### 🎯 KWS/ASR Detection Results")

# Create a container for results
col_result1, col_result2 = st.columns([2, 1])

with col_result1:
    # Display detection results
    current_results = get_processor_results()
    
    if current_results:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        for result in current_results:
            st.text(result)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.session_state.processor_started:
            st.info("🔍 Waiting for keyword detection...")
        else:
            st.info("ℹ️ Start the processor to begin detecting keywords")

with col_result2:
    st.markdown("### 📝 Status Log")
    if st.session_state.results_history:
        for msg in st.session_state.results_history[-8:]:
            st.caption(msg)

# Help section in sidebar
with st.sidebar:
    st.markdown("### 📖 Help & Instructions")
    st.markdown("""
    **Quick Start:**
    1. Select an audio input device
    2. Click the **Start** button
    3. Speak the keywords to test detection
    4. View results in real-time
    
    **Supported Keywords:**
    - hello
    - Hi panda
    - hi siri
    - 你好
    
    **Stop Words:**
    - stop
    - 停止
    - okay
    - 好了
    - 行了
    - 好的
    - 退出
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    with st.expander("Advanced Settings"):
        debug_mode = st.checkbox("Debug Mode", value=False, key="debug_mode")
        if st.session_state.processor:
            st.session_state.processor.isDebug = debug_mode
        
        threshold = st.slider("VAD Threshold", 0.0, 0.1, 0.02, step=0.01, key="vad_threshold")
        if st.session_state.processor:
            st.session_state.processor.threshold = threshold
    
    st.markdown("---")
    
    # Device info
    with st.expander("Device Information"):
        if st.session_state.audio_devices:
            st.write(f"**Total Devices:** {len(st.session_state.audio_devices)}")
            st.write("**Available Devices:**")
            for device_id, name in st.session_state.audio_devices:
                is_selected = "✓" if device_id == st.session_state.selected_device else " "
                st.write(f"  [{is_selected}] {name} (ID: {device_id})")
        else:
            st.warning("No audio devices detected")
    
    st.markdown("---")
    st.markdown("""
    **Copyright © 2026 GE Healthcare**
    
    Author: jiheng.zhang@gehealthcare.com
    
    SSO: 212597558
    """)

# Refresh mechanism for continuous updates - removed infinite loop
# The app will rerun based on user interactions
