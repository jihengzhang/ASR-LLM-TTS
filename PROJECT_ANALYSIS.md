# FunASR VAD/KWS Project Analysis

## Project Overview

**Project Name:** FunASR VAD/KWS Audio Testing UI  
**Purpose:** Real-time Voice Activity Detection (VAD) and Keyword Spotting (KWS) system with wxPython GUI  
**Main Entry Point:** `Audio_test_UI.py`  
**Architecture:** Multi-threaded audio processing with interactive UI

## System Architecture

### Core Components

1. **Audio_test_UI.py** (Main UI)
   - wxPython-based GUI application
   - Audio device selection and management
   - Real-time plot visualization
   - Keyboard-controlled recording (SPACE bar)
   - Result display and status monitoring

2. **FunASR_VAD_KWS_plot.py** (Audio Processing Engine)
   - Real-time audio capture using PyAudio
   - FunASR integration for VAD/KWS models
   - Matplotlib-based visualization with animation
   - Multi-threaded processing
   - Recording capabilities

### Dependencies Breakdown

#### Core Audio Processing (Essential)
```
PyAudio==0.2.14          # Audio I/O
numpy==2.0.1             # Numerical computing
scipy==1.15.3            # Signal processing
librosa==0.11.0          # Audio feature extraction
soundfile==0.13.1        # Audio file I/O
```

#### UI Framework (Essential)
```
wxPython==4.2.3          # GUI framework
matplotlib==3.10.5       # Data visualization
```

#### Deep Learning (Essential)
```
torch==2.5.1             # PyTorch core
torchaudio==2.5.1        # Audio processing for PyTorch
funasr==1.2.6            # FunASR framework for VAD/KWS
transformers==4.53.1     # Hugging Face transformers
modelscope==1.28.0       # Alibaba ModelScope for model management
huggingface-hub==0.34.4  # Hugging Face hub integration
```

#### Supporting Libraries (Optional)
```
sentencepiece==0.2.0     # Tokenization
safetensors==0.5.3       # Efficient tensor storage
editdistance==0.8.1      # Edit distance for text
omegaconf==2.3.0         # Configuration management
pydantic==2.11.7         # Data validation
pyyaml==6.0.2            # YAML configuration
requests==2.32.4         # HTTP library
tqdm==4.67.1             # Progress bars
filelock==3.19.1         # File locking
fsspec==2024.6.1         # Filesystem utilities
```

## Old Requirements Issues

### Problems Identified

1. **Bloated Size** (278 packages vs. ~35 needed)
   - Reduced by 87.4%

2. **Unrelated Categories** Removed:
   - **Embedded Systems**: esptool, pyocd, canopen, hidapi, libusb-package, etc.
   - **Development Tools**: pytest, mypy, pylint, isort, ruff, yamllint, vermin
   - **Build Tools**: setuptools-scm, poetry dependencies, cmake support
   - **Version Control**: gitpython, gitlint, gitdb
   - **API/Web**: fastapi, starlette, uvicorn, grpcio, websockets, gradio
   - **Database/Data**: sqlite, sqlalchemy, rdflib, xml tools
   - **Compression**: Brotli, binaryornot, patool
   - **Cryptography**: pycryptodome, PyNaCl, oscrypto
   - **Testing**: pytest, coverage, junit2html
   - **Documentation**: sphinx-lint, reuse, spdx-tools
   - **CLI Tools**: click, typer, rich, colorlog
   - **Terminal**: pyreadline3, windows-curses
   - **GPU/ML Extras**: pytorch-lightning, torchmetrics, torchvision, diffusers

3. **Conflicting Versions**:
   - Old: `torch==2.5.1+cu121` (CUDA 12.1 specific)
   - New: `torch==2.5.1` (Flexible, works with CPU or CUDA)
   - Users can specify CUDA variant if needed

4. **Redundant Dependencies**:
   - Many packages brought in as transitive dependencies
   - Now only explicit, necessary packages listed

## New Requirements Structure

### Organization
- **Core Dependencies**: Direct imports in main code
- **Optional/Supporting**: Required by FunASR/modelscope ecosystem

### Size Comparison
| Metric | Old | New | Reduction |
|--------|-----|-----|-----------|
| Packages | 278 | 35 | 87.4% |
| Core Focus | Unclear | Audio VAD/KWS | Clear |
| Maintainability | Low | High | Good |

## Installation Instructions

### For CPU-only System
```bash
pip install -r requirements.txt
```

### For CUDA 12.1 GPU
```bash
# First install CUDA-compatible PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements-gpu.txt
```

### For CUDA 11.8 GPU
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Key Features Enabled

1. ✅ Real-time audio capture and processing
2. ✅ FunASR VAD model integration
3. ✅ Keyword spotting (KWS)
4. ✅ wxPython GUI with live plotting
5. ✅ Audio device selection
6. ✅ Recording management
7. ✅ Model caching via modelscope

## Troubleshooting

### PyAudio Installation Issues
- On Windows: Use pre-compiled wheel or conda
- On Linux: Install `portaudio19-dev` first
- On macOS: Use Homebrew to install portaudio

### CUDA Issues
- Check `torch.cuda.is_available()` in Python
- Ensure NVIDIA drivers are installed
- May need to match CUDA version with system

### Model Download Issues
- Models auto-download to `~/.cache/models/`
- Network timeout issues can be resolved with `download_timeout` parameter
- Check disk space (models are several hundred MB)

## File Organization

```
vad KWS/
├── Audio_test_UI.py           # Main GUI application
├── FunASR_VAD_KWS_plot.py     # Core audio processing
├── requirements.txt            # Updated dependencies
├── PROJECT_ANALYSIS.md         # This file
├── models/                      # Pre-downloaded models
├── recordings/                  # Saved recordings
├── output/                      # Processing output
└── [other utility scripts]
```

## Future Improvements

1. **Dependency Management**
   - Create separate requirements files for dev/test
   - Consider using Poetry or Pipenv for better management

2. **Performance**
   - Profile audio processing bottlenecks
   - Consider ONNX export for inference optimization

3. **Documentation**
   - Add API documentation for VADKWSProcessor class
   - Create example notebooks for usage patterns

4. **Testing**
   - Add unit tests (currently missing)
   - Add integration tests for audio pipeline

5. **Containerization**
   - Create Dockerfile for reproducible environments
   - Consider multi-stage builds for optimization
