# Verification Report: Android Model Creation Process

## Summary

After reviewing all library documentation, code, and training notebooks, I've verified the Android model creation process. This report documents the findings and any corrections needed.

## ✅ Verified Correct

### 1. Training Pipeline Flow
- **Status**: ✅ CORRECT
- The training script (`openwakeword/train.py`) correctly:
  1. Exports model to ONNX format
  2. Automatically converts ONNX to TFLite format
  3. Saves both formats in the output directory

**Evidence**: Lines 898-902 in `openwakeword/train.py`:
```python
oww.export_model(model=best_model, model_name=config["model_name"], output_dir=config["output_dir"])
convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                       os.path.join(config["output_dir"], config["model_name"] + ".tflite"))
```

### 2. Configuration File Format
- **Status**: ✅ CORRECT
- The YAML configuration format matches the expected structure
- All required fields are documented in `examples/custom_model.yml`

### 3. Helper Script Logic
- **Status**: ✅ CORRECT
- The `create_android_model.py` script correctly:
  - Calls the training script with proper flags
  - Handles the three-step process (generate, augment, train)
  - Verifies output files

### 4. TFLite Model Compatibility
- **Status**: ✅ CORRECT
- The library uses `ai-edge-litert` for TFLite inference
- Models exported are compatible with LiteRT on Android
- Input/output specifications are correct (16-bit PCM, 16kHz, 80ms frames)

## ⚠️ Issues Found & Corrections Needed

### 1. Piper Sample Generator Repository URL
- **Issue**: Documentation inconsistency
- **Found**: Notebook uses `rhasspy/piper-sample-generator`, but config example references `dscripka/piper-sample-generator`
- **Status**: ⚠️ NEEDS VERIFICATION
- **Action**: Need to verify which repository is the correct one to use

**Current references:**
- Notebook: `https://github.com/rhasspy/piper-sample-generator`
- My docs: `https://github.com/dscripka/piper-sample-generator`
- Config example: `https://github.com/dscripka/piper-sample-generator`

**Recommendation**: Check the actual repository that works with the training script.

### 2. Platform Limitations
- **Issue**: Training is Linux-only (not clearly documented in my guides)
- **Found**: Notebook states "Currently, automated model training is only supported on linux systems"
- **Status**: ⚠️ NEEDS DOCUMENTATION UPDATE
- **Action**: Add platform requirement warnings to all Android guides

**From notebook (Cell 3):**
> "Currently, automated model training is only supported on linux systems due to the requirements of the text to speech library used for synthetic sample generation (Piper). It may be possible to use Piper on Windows/Mac systems, but that has not (yet) been tested."

### 3. Missing Dependencies in Helper Script
- **Issue**: Helper script doesn't check for all required dependencies
- **Found**: Training requires additional packages beyond `openwakeword[full]`
- **Status**: ⚠️ NEEDS UPDATE
- **Action**: Update dependency checking in `create_android_model.py`

**Additional dependencies from notebook:**
- `piper-phonemize`
- `webrtcvad`
- Piper model file (`en_US-libritts_r-medium.pt`)

### 4. Export Model Function Documentation
- **Issue**: Function docstring is misleading
- **Found**: `export_model()` only exports ONNX, not TFLite (despite docstring saying "both")
- **Status**: ℹ️ INFORMATIONAL (not a bug - TFLite conversion happens separately)
- **Action**: This is actually correct - the function is designed to only export ONNX, and TFLite conversion is a separate step

## 📝 Required Corrections

### 1. Update Documentation Files

**Files to update:**
- `ANDROID_DEPLOYMENT_GUIDE.md` - Add platform requirements
- `QUICK_START_ANDROID.md` - Add platform requirements and verify Piper repo
- `create_android_model.py` - Improve dependency checking

### 2. Verify Piper Repository

Need to determine which repository is correct:
- Check if `dscripka/piper-sample-generator` is a fork
- Verify which one the training script actually uses
- Update all documentation to use the correct repository

### 3. Add Platform Warnings

Add clear warnings about Linux requirement for training (though inference on Android is fine).

## ✅ Verified Process Flow

The complete process flow is correct:

1. **Generate Clips** (`--generate_clips`)
   - Uses Piper TTS to generate synthetic speech
   - Creates positive and negative examples
   - ✅ Verified in code

2. **Augment Clips** (`--augment_clips`)
   - Applies data augmentation (RIR, background noise)
   - Computes openWakeWord features
   - ✅ Verified in code

3. **Train Model** (`--train_model`)
   - Trains PyTorch model
   - Exports to ONNX
   - Converts ONNX to TFLite
   - ✅ Verified in code

## 📋 Model Specifications (Verified)

- **Input Format**: 16-bit PCM, 16kHz, mono
- **Frame Size**: 80ms (1280 samples)
- **Model Input Shape**: `[1, 16, 96]` (16 frames of 96-dim embeddings)
- **Model Output**: `[1, 1]` (single float 0-1)
- **TFLite Format**: ✅ Compatible with LiteRT
- **Feature Extraction**: Requires separate melspectrogram and embedding models

## 🔍 Additional Findings

### Model Architecture
- Default `layer_size: 32` is appropriate for Android
- DNN architecture is faster than RNN on mobile
- Model size is reasonable for mobile deployment

### Training Data Requirements
- Minimum 20,000 samples recommended (my quick start uses 5,000 for testing - acceptable)
- Validation data from HuggingFace is recommended but optional
- RIR and background audio improve robustness

### Export Process
- ONNX export uses opset version 13
- TFLite conversion uses TensorFlow Lite converter
- Both formats are saved automatically

## ✅ Conclusion

The Android model creation process is **fundamentally correct** with the following minor issues to address:

1. ⚠️ Verify correct Piper repository URL
2. ⚠️ Add platform requirement warnings (Linux for training)
3. ⚠️ Improve dependency checking in helper script
4. ℹ️ Clarify that export_model only does ONNX (TFLite is separate step, but automatic)

The core process, configuration format, and model specifications are all correct and align with the library's implementation.

