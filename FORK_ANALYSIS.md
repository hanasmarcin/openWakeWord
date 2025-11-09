# openWakeWord Fork Analysis

## Repository Information

**Fork Origin**: `hanasmarcin/openWakeWord` (https://github.com/hanasmarcin/openWakeWord.git)  
**Original Repository**: `dscripka/openWakeWord`  
**Current Version**: v0.6.0  
**Python Version Requirement**: >=3.10

## Overview

This is a fork of the openWakeWord library, an open-source wake word detection framework. The fork appears to be tracking the upstream repository closely, with recent merges from the original maintainer (dscripka). The codebase structure and functionality appear to be largely identical to the original.

## Architecture Analysis

### Core Components

1. **Model Architecture** (`openwakeword/model.py`)
   - Main `Model` class for wake word detection
   - Supports multiple wake word models simultaneously
   - Streaming audio processing with 80ms frames (1280 samples @ 16kHz)
   - Supports both ONNX and TFLite inference frameworks
   - Includes debounce and patience mechanisms for reducing false positives

2. **Audio Feature Extraction** (`openwakeword/utils.py`)
   - `AudioFeatures` class for melspectrogram computation
   - Google's speech_embedding model integration
   - Streaming feature extraction with buffering
   - Support for batch processing

3. **Voice Activity Detection** (`openwakeword/vad.py`)
   - Silero VAD model integration
   - Optional filtering to reduce false activations
   - Configurable threshold

4. **Custom Verifier Models** (`openwakeword/custom_verifier_model.py`)
   - Speaker-specific verification using logistic regression
   - Acts as a second-stage filter on predictions
   - Reduces false positives for known speakers

5. **Training Pipeline** (`openwakeword/train.py`)
   - PyTorch-based model training
   - Support for DNN and RNN architectures
   - Automated training sequences with validation
   - Synthetic data generation support
   - Model export to ONNX and TFLite formats

### Key Features

#### Inference Frameworks
- **TFLite** (default on Linux): Uses `ai-edge-litert` for efficient inference
- **ONNX**: Alternative framework, default on Windows
- Automatic fallback between frameworks

#### Pre-trained Models
- `alexa` - "alexa"
- `hey_mycroft` - "hey mycroft"
- `hey_jarvis` - "hey jarvis"
- `hey_rhasspy` - "hey rhasspy"
- `timer` - Timer-related phrases (multi-class)
- `weather` - Weather queries

#### Performance Optimizations
- Speex noise suppression (optional, Linux only)
- Voice Activity Detection (VAD) filtering
- Custom verifier models for speaker-specific detection
- Debounce and patience mechanisms
- Shared feature extraction backbone (efficient multi-model support)

### Data Processing Pipeline

1. **Audio Input**: 16-bit PCM, 16kHz, mono
2. **Preprocessing**: Optional Speex noise suppression
3. **Feature Extraction**: 
   - Melspectrogram computation (ONNX/TFLite)
   - Speech embedding extraction (Google's model)
4. **Prediction**: Wake word model inference
5. **Post-processing**: VAD filtering, custom verifier (optional)

## Dependencies

### Core Dependencies
- `onnxruntime>=1.10.0,<2`
- `ai-edge-litert>=2.0.2,<3` (Linux/Darwin)
- `speexdsp-ns>=0.1.2,<1` (Linux)
- `tqdm>=4.0,<5.0`
- `scipy>=1.3,<2`
- `scikit-learn>=1,<2`
- `requests>=2.0,<3`

### Training Dependencies (optional)
- PyTorch, torchaudio
- SpeechBrain
- AudioAugmentations
- TensorFlow (for ONNX→TFLite conversion)

## Code Quality Observations

### Strengths
1. **Well-structured**: Clear separation of concerns
2. **Comprehensive documentation**: README, docstrings, example scripts
3. **Flexible**: Supports multiple inference frameworks and model types
4. **Efficient**: Shared feature extraction, streaming processing
5. **Extensible**: Easy to add new wake word models

### Areas of Note
1. **Platform-specific dependencies**: Some features (Speex, TFLite) are Linux-only
2. **Model downloads**: Models must be downloaded separately (not in PyPI package)
3. **Training complexity**: Full training pipeline requires many dependencies

## Recent Changes (from git log)

- Python version bump to 3.10+
- Switch to `ai-edge-litert` for TFLite runtime
- Custom verifier type hints fix
- Bug fixes in training pipeline
- Debounce functionality improvements

## Usage Patterns

### Basic Usage
```python
import openwakeword
from openwakeword.model import Model

# Download models
openwakeword.utils.download_models()

# Initialize model
model = Model(wakeword_models=["hey_jarvis"])

# Predict on audio frame
prediction = model.predict(audio_frame)
```

### Advanced Features
- Custom verifier models for speaker-specific detection
- VAD threshold configuration
- Noise suppression
- Multi-model simultaneous detection
- Debounce and patience mechanisms

## Comparison with Original

Based on the git history and codebase:
- **Status**: Fork appears to be tracking upstream closely
- **Modifications**: No significant customizations detected
- **Sync Status**: Recent merges from upstream (PR #289)
- **Purpose**: Likely a personal fork for development/testing

## Recommendations

1. **If maintaining as a fork**: Consider documenting any planned customizations
2. **If contributing back**: The fork is well-positioned to contribute upstream
3. **For users**: This fork appears functionally identical to upstream; consider using upstream unless specific customizations are needed

## Testing

The repository includes:
- Test files in `tests/` directory
- Example scripts in `examples/`
- Benchmark utilities
- Performance evaluation tools

## License

- **Code**: Apache 2.0
- **Pre-trained Models**: CC BY-NC-SA 4.0 (due to training data licensing)

---

*Analysis generated on: 2024*
*Repository version: v0.6.0*

