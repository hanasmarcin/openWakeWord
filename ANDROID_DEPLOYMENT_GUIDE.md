# Android Deployment Guide for openWakeWord with LiteRT

This guide will help you create and deploy a wake word model for Android using LiteRT (ai-edge-litert).

## Overview

The openWakeWord library can create TFLite models that are compatible with LiteRT on Android. The process involves:

1. **Training a custom model** - Generate synthetic data and train a wake word model
2. **Export to TFLite** - Convert the trained model to TFLite format
3. **Android Integration** - Use LiteRT in your Android app to run inference

## Prerequisites

### ⚠️ Platform Requirements

**Important**: Model training is currently only supported on **Linux systems** due to the requirements of the Piper TTS library used for synthetic speech generation. 

- **Training**: Requires Linux (may work on Mac/Windows but not tested)
- **Inference on Android**: Works on any platform (Android devices)

If you're on Windows/Mac, consider:
- Using a Linux VM or container
- Using Google Colab (see the training notebook)
- Training on a Linux server and transferring the model

### Python Environment

Install the full openWakeWord package with training dependencies:

```bash
pip install openwakeword[full]
```

Or install dependencies individually:
```bash
pip install torch torchaudio numpy scipy scikit-learn onnx tensorflow onnx-tf pyyaml
```

### Required Data/Resources

1. **Piper Sample Generator** (for synthetic speech generation):
   ```bash
   git clone https://github.com/dscripka/piper-sample-generator
   ```

2. **Room Impulse Responses (RIRs)** (for data augmentation):
   - Download from: https://www.openslr.org/28/
   - Extract to a directory (e.g., `./mit_rirs`)

3. **Background Audio** (optional, for augmentation):
   - Create a directory with audio files (speech, music, noise)
   - Used to make the model more robust to background noise
   - **See `BACKGROUND_NOISE_GUIDE.md` for detailed information**

4. **Validation Data** (recommended):
   - Download from: https://huggingface.co/datasets/davidscripka/openwakeword_features
   - Used to evaluate false positive rates

## Quick Start

### Option 1: Using the Helper Script

The easiest way to create an Android model:

```bash
# Create a model with minimal configuration
python create_android_model.py --wake-word "hey assistant"

# Or use a full configuration file
python create_android_model.py --config android_model_config.yml
```

### Option 2: Manual Training

1. **Create a configuration file** (`android_model_config.yml`):
   ```yaml
   model_name: "android_wakeword"
   target_phrase:
     - "hey assistant"
   n_samples: 20000
   # ... (see android_model_config.yml for full example)
   ```

2. **Generate synthetic training data**:
   ```bash
   python -m openwakeword.train --training_config android_model_config.yml --generate_clips
   ```

3. **Augment data and compute features**:
   ```bash
   python -m openwakeword.train --training_config android_model_config.yml --augment_clips
   ```

4. **Train the model**:
   ```bash
   python -m openwakeword.train --training_config android_model_config.yml --train_model
   ```

The trained TFLite model will be saved in your output directory as `{model_name}.tflite`.

## Model Architecture for Android

For Android deployment, consider:

- **Model Size**: Smaller models (layer_size: 32) are better for mobile devices
- **Model Type**: DNN (fully connected) is typically faster than RNN on mobile
- **Input Format**: Models expect 16-bit PCM audio at 16kHz
- **Frame Size**: Process audio in 80ms frames (1280 samples)

## Android Integration

### 1. Add LiteRT to Your Android Project

In your `build.gradle` (app level):

```gradle
dependencies {
    implementation 'com.google.ai.edge.litert:litert:2.0.2'
    // Or use the latest version from Maven Central
}
```

### 2. Copy Model Files to Assets

1. Copy your trained `.tflite` model to `app/src/main/assets/`
2. Also copy the feature extraction models:
   - `melspectrogram.tflite`
   - `embedding_model.tflite`
   
   These can be downloaded from the openWakeWord releases or extracted from the Python package.

### 3. Android Code Example

```kotlin
import com.google.ai.edge.litert.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WakeWordDetector(context: Context) {
    private val interpreter: Interpreter
    private val melspectrogramInterpreter: Interpreter
    private val embeddingInterpreter: Interpreter
    
    init {
        // Load wake word model
        val modelBuffer = loadModelFile(context, "android_wakeword.tflite")
        interpreter = Interpreter(modelBuffer)
        
        // Load feature extraction models
        val melspecBuffer = loadModelFile(context, "melspectrogram.tflite")
        melspectrogramInterpreter = Interpreter(melspecBuffer)
        
        val embeddingBuffer = loadModelFile(context, "embedding_model.tflite")
        embeddingInterpreter = Interpreter(embeddingBuffer)
    }
    
    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        return buffer
    }
    
    fun predict(audioFrame: ShortArray): Float {
        // audioFrame should be 1280 samples (80ms @ 16kHz)
        // 1. Compute melspectrogram
        val melspec = computeMelspectrogram(audioFrame)
        
        // 2. Compute embeddings
        val embeddings = computeEmbeddings(melspec)
        
        // 3. Run wake word model
        val inputBuffer = prepareInput(embeddings)
        val outputBuffer = ByteBuffer.allocateDirect(4) // float output
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        return outputBuffer.getFloat(0)
    }
    
    private fun computeMelspectrogram(audio: ShortArray): FloatArray {
        // Implementation depends on your melspectrogram model
        // This is a placeholder - you'll need to implement the full pipeline
        val inputBuffer = prepareAudioInput(audio)
        val outputBuffer = ByteBuffer.allocateDirect(/* output size */)
        melspectrogramInterpreter.run(inputBuffer, outputBuffer)
        // Convert output buffer to float array
        return floatArrayOf() // placeholder
    }
    
    private fun computeEmbeddings(melspec: FloatArray): FloatArray {
        // Similar to melspectrogram computation
        val inputBuffer = prepareMelspecInput(melspec)
        val outputBuffer = ByteBuffer.allocateDirect(/* output size */)
        embeddingInterpreter.run(inputBuffer, outputBuffer)
        return floatArrayOf() // placeholder
    }
    
    private fun prepareInput(embeddings: FloatArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(embeddings.size * 4)
        buffer.order(ByteOrder.nativeOrder())
        for (value in embeddings) {
            buffer.putFloat(value)
        }
        return buffer
    }
}
```

### 4. Audio Processing

You'll need to:

1. **Capture audio** at 16kHz, 16-bit PCM
2. **Process in frames** of 1280 samples (80ms)
3. **Maintain state** for the feature extraction pipeline (melspectrogram and embedding buffers)
4. **Apply threshold** (typically 0.5) to determine activation

## Model Input/Output Specifications

### Wake Word Model
- **Input**: Shape `[1, 16, 96]` - 16 frames of 96-dimensional embeddings
- **Output**: Shape `[1, 1]` - Single float value between 0 and 1
- **Threshold**: Typically 0.5 for activation

### Melspectrogram Model
- **Input**: Shape `[1, 1280]` - 80ms of 16-bit PCM audio
- **Output**: Shape `[76, 32]` - Melspectrogram features

### Embedding Model
- **Input**: Shape `[batch, 76, 32, 1]` - Melspectrogram windows
- **Output**: Shape `[batch, 96]` - Embedding vectors

## Performance Optimization

1. **Use smaller models**: `layer_size: 32` instead of larger values
2. **Quantization**: Consider quantizing the TFLite model for even better performance
3. **Threading**: LiteRT supports multi-threading - configure based on your device
4. **Buffer management**: Reuse ByteBuffers to avoid allocations

## Testing Your Model

Before deploying, test your model:

1. **Test with various speakers**: Different voices, accents
2. **Test in noisy environments**: Background noise, music, speech
3. **Test false positives**: Long audio clips without the wake word
4. **Measure latency**: Ensure real-time performance on target devices

## Troubleshooting

### Model too large
- Reduce `layer_size` in training config
- Use quantization when converting to TFLite

### Poor accuracy
- Increase `n_samples` in training config
- Add more diverse background audio
- Adjust `target_false_positives_per_hour` threshold

### Slow inference
- Use smaller model architecture
- Optimize buffer management
- Consider model quantization

### Integration issues
- Verify model input/output shapes match
- Check that audio format matches (16kHz, 16-bit PCM)
- Ensure feature extraction pipeline matches Python implementation

## Additional Resources

- [LiteRT Documentation](https://github.com/google-ai-edge-media/ai-edge-litert)
- [openWakeWord Training Tutorial](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
- [TFLite Optimization Guide](https://www.tensorflow.org/lite/performance)

## Support

For issues specific to:
- **Model training**: Check openWakeWord repository issues
- **Android integration**: Check LiteRT documentation
- **Model accuracy**: Review training data and augmentation settings

