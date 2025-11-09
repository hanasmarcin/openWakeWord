# Quick Start: Creating an Android Model

## Fastest Path (Minimal Setup)

### ⚠️ Platform Note
**Training requires Linux**. If you're on Windows/Mac, use a Linux VM, container, or Google Colab.

### 1. Install Dependencies

```bash
pip install openwakeword[full]
```

### 2. Clone Piper Sample Generator

```bash
git clone https://github.com/dscripka/piper-sample-generator
```

### 3. Create Model (Quick Test)

```bash
python create_android_model.py --wake-word "hey assistant"
```

This will:
- Create a minimal config automatically
- Generate 5,000 training samples (quick test)
- Train a small model
- Export to TFLite format

**Output**: `./android_model_output/android_wakeword.tflite`

### 4. For Production Model

Edit `android_model_config.yml` and run:

```bash
python create_android_model.py --config android_model_config.yml
```

## What You Need

### Required
- ✅ Python 3.10+ with openwakeword[full]
- ✅ Piper sample generator (cloned)
- ✅ Room Impulse Responses (download from https://www.openslr.org/28/)

### Optional (but recommended)
- Background audio clips for augmentation (see `BACKGROUND_NOISE_GUIDE.md`)
- Validation dataset (from HuggingFace)

## Model Output

After training, you'll get:
- `android_wakeword.tflite` - Your wake word model (use this in Android)
- `android_wakeword.onnx` - ONNX version (backup)

## Next Steps

1. Copy the `.tflite` file to your Android project
2. See `ANDROID_DEPLOYMENT_GUIDE.md` for integration instructions
3. Test the model with various audio samples

## Troubleshooting

**"Piper sample generator not found"**
```bash
git clone https://github.com/dscripka/piper-sample-generator
```

**"RIR directory not found"**
- Download from: https://www.openslr.org/28/
- Extract to `./mit_rirs/`

**"Background audio directory not found"**
- Create `./background_clips/` directory
- Add some audio files (optional, but recommended)

**Training takes too long?**
- Reduce `n_samples` in config (minimum 5,000 for testing)
- Reduce `steps` in config (minimum 10,000 for testing)

