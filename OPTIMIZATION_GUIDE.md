# Model Optimization Guide - Getting the Best Model Possible

## Overview

This guide explains how to use **all available features** in the openWakeWord library to create the best possible Android model. The training script already includes background noise augmentation - here's how to maximize everything else.

## ✅ What's Already Included Automatically

The training script (`train.py`) **automatically includes**:

1. ✅ **Background Noise Augmentation** - Mixed with your clips during augmentation
2. ✅ **Room Impulse Responses (RIRs)** - Applied for acoustic simulation
3. ✅ **Multiple Augmentation Types** - Pitch shift, EQ, distortion, colored noise, etc.
4. ✅ **Model Checkpoint Averaging** - Best models are automatically averaged
5. ✅ **Early Stopping** - Based on validation metrics
6. ✅ **Adaptive Learning Rates** - Cosine decay with warmup
7. ✅ **High-Loss Example Focus** - Training focuses on difficult examples

## 🚀 Key Optimizations You Can Enable

### 1. Increase Training Data (Most Important!)

```yaml
# Minimum for decent results
n_samples: 20000

# Recommended for production
n_samples: 100000

# Best results
n_samples: 200000+
```

**Impact**: More data = better generalization, lower false positives

### 2. Multiple Augmentation Rounds (Free Data Multiplication!)

```yaml
# Default (1 round)
augmentation_rounds: 1

# Recommended (2-3 rounds)
augmentation_rounds: 3

# Maximum (4-5 rounds)
augmentation_rounds: 5
```

**Impact**: Each round creates unique augmented versions. 3 rounds = 3x effective dataset size!

**How it works**: The same synthetic clip is augmented multiple times with different:
- Background noise selections
- RIR selections
- Random augmentations (pitch, EQ, etc.)

### 3. Large Negative Dataset (Critical for False Positives)

```yaml
# Download from HuggingFace
feature_data_files:
  "ACAV100M_sample": "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
```

**Impact**: Trains model on 2000+ hours of real audio without your wake word. Dramatically reduces false positives.

**Where to get**: https://huggingface.co/datasets/davidscripka/openwakeword_features

### 4. Validation Dataset (For Model Selection)

```yaml
false_positive_validation_data_path: "./validation_set_features.npy"
```

**Impact**: Helps select the best model checkpoint and tune false positive rates.

**Where to get**: Same HuggingFace dataset (validation_set_features.npy)

### 5. Multiple Background Noise Types

```yaml
background_paths:
  - "./background_clips"      # General
  - "./background_speech"     # Conversations
  - "./background_music"       # Music
  - "./background_noise"       # Environmental

background_paths_duplication_rate:
  - 1    # General
  - 2    # Speech (more common)
  - 1    # Music
  - 1    # Noise
```

**Impact**: More diverse noise = better robustness to real environments

### 6. Optimize Model Architecture

```yaml
# For Android (fast, small)
layer_size: 32

# Balanced (recommended)
layer_size: 48

# Better accuracy (larger model)
layer_size: 64

# Maximum accuracy (largest)
layer_size: 128
```

**Trade-off**: Larger = better accuracy but slower inference and larger file size

### 7. Extended Training

```yaml
# Minimum
steps: 25000

# Recommended
steps: 50000

# Best results
steps: 100000
```

**Impact**: More training = better convergence, but diminishing returns after 50k-100k steps

### 8. Stricter False Positive Control

```yaml
# Standard
max_negative_weight: 1500
target_false_positives_per_hour: 0.2

# Stricter (fewer false activations)
max_negative_weight: 2000
target_false_positives_per_hour: 0.1

# Very strict
max_negative_weight: 3000
target_false_positives_per_hour: 0.05
```

**Trade-off**: Stricter = fewer false positives but may slightly reduce recall

## 📊 Optimization Priority (Best ROI)

### Tier 1: Highest Impact (Do These First!)

1. **Add Negative Dataset** (2000 hours)
   - **Impact**: ⭐⭐⭐⭐⭐ (Huge reduction in false positives)
   - **Effort**: Low (just download)
   - **Time**: +0 hours (no training time increase)

2. **Increase Augmentation Rounds to 3**
   - **Impact**: ⭐⭐⭐⭐⭐ (3x effective dataset)
   - **Effort**: Low (change one number)
   - **Time**: +20-30% training time

3. **Increase Training Samples to 100k**
   - **Impact**: ⭐⭐⭐⭐ (Better generalization)
   - **Effort**: Medium (more generation time)
   - **Time**: +2-4 hours generation, +20% training

### Tier 2: High Impact

4. **Add Validation Dataset**
   - **Impact**: ⭐⭐⭐⭐ (Better model selection)
   - **Effort**: Low (download)
   - **Time**: +0 hours

5. **Multiple Background Noise Types**
   - **Impact**: ⭐⭐⭐ (Better noise robustness)
   - **Effort**: Medium (collect/organize audio)
   - **Time**: +0 hours (no training time increase)

6. **Increase Training Steps to 50k**
   - **Impact**: ⭐⭐⭐ (Better convergence)
   - **Effort**: Low (change number)
   - **Time**: +50% training time

### Tier 3: Fine-Tuning

7. **Optimize Layer Size** (32 → 48)
   - **Impact**: ⭐⭐ (Slightly better accuracy)
   - **Effort**: Low
   - **Time**: +10% training time, slightly larger model

8. **Stricter False Positive Control**
   - **Impact**: ⭐⭐ (Fewer false activations)
   - **Effort**: Low
   - **Time**: +0 hours

## 🎯 Recommended Configurations

### Quick Test (Fast, Lower Quality)
```yaml
n_samples: 5000
augmentation_rounds: 1
steps: 10000
layer_size: 32
# No negative dataset
```

### Production (Balanced)
```yaml
n_samples: 100000
augmentation_rounds: 3
steps: 50000
layer_size: 48
feature_data_files: {ACAV100M: ...}  # 2000 hours
```

### Maximum Quality (Best Possible)
```yaml
n_samples: 200000
augmentation_rounds: 5
steps: 100000
layer_size: 64
feature_data_files: {ACAV100M: ...}  # 2000 hours
multiple_background_directories: true
max_negative_weight: 2000
```

## 📈 Expected Results

### With Minimal Config (20k samples, 1 round, no negative data)
- False Positive Rate: ~1-2 per hour
- False Reject Rate: ~5-10%
- Training Time: ~2-4 hours

### With Optimized Config (100k samples, 3 rounds, 2000h negative)
- False Positive Rate: ~0.2-0.5 per hour
- False Reject Rate: ~2-5%
- Training Time: ~8-16 hours

### With Maximum Config (200k samples, 5 rounds, 2000h negative)
- False Positive Rate: ~0.1-0.2 per hour
- False Reject Rate: ~1-3%
- Training Time: ~16-24 hours

## 🔍 What the Library Does Automatically

The training process includes sophisticated techniques:

1. **Checkpoint Averaging**: Best models are automatically averaged
2. **Early Stopping**: Training stops when validation metrics plateau
3. **Adaptive Learning**: Learning rate schedules with warmup and cosine decay
4. **High-Loss Focus**: Training focuses on examples the model struggles with
5. **Negative Weight Scheduling**: Gradually increases focus on negative examples
6. **Multiple Training Sequences**: 3 training sequences with decreasing learning rates

All of this is **automatic** - you just need to provide good data and configuration!

## 💡 Pro Tips

1. **Start with optimized config** - The time investment is worth it
2. **Download the negative dataset** - It's the single biggest improvement
3. **Use 3 augmentation rounds** - Best balance of quality vs time
4. **Test incrementally** - Start with 50k samples, increase if needed
5. **Monitor training logs** - Watch for validation metrics plateauing

## 📝 Quick Checklist

For the best possible model, ensure you have:

- [ ] 100,000+ training samples (`n_samples: 100000`)
- [ ] 3 augmentation rounds (`augmentation_rounds: 3`)
- [ ] Negative dataset (2000 hours from HuggingFace)
- [ ] Validation dataset (for model selection)
- [ ] Multiple background noise types (speech, music, noise)
- [ ] Room impulse responses (MIT RIRs)
- [ ] 50,000+ training steps (`steps: 50000`)
- [ ] Optimized layer size (`layer_size: 48`)
- [ ] Stricter false positive control (`max_negative_weight: 2000`)

## 🚀 Getting Started

Use the optimized config:

```bash
# Use the optimized configuration
python create_android_model.py --config android_model_config_OPTIMIZED.yml
```

Or modify your existing config with the optimizations above!

---

*Remember: The library already does a lot automatically. Your job is to provide good data and configuration!*

