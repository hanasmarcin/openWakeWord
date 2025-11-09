# Background Noise Augmentation Guide

## Overview

Background noise augmentation is a **critical component** of training robust wake word models. It helps your model work better in real-world environments with:
- Background music
- Other people talking
- Environmental noise (traffic, appliances, etc.)
- Various acoustic conditions

## How It Works

During the training process, background audio is **mixed** with your synthetic wake word clips to create more realistic training data. This happens during the **augmentation step** (`--augment_clips`).

### The Augmentation Process

1. **Synthetic Wake Word Clips** are generated (clean audio)
2. **Background Audio** is randomly selected and mixed at various signal-to-noise ratios (SNR)
3. **Room Impulse Responses (RIRs)** are applied to simulate different acoustic environments
4. **Additional augmentations** are applied (pitch shift, EQ, distortion, etc.)

## Configuration

### In Your Config File (`android_model_config.yml`)

```yaml
# The directories containing background audio files to mix with training data
# You can use any audio files (speech, music, noise) for background augmentation
background_paths:
  - "./background_clips"
  - "./music_clips"        # You can specify multiple directories
  - "./speech_clips"

# Duplication rate for background audio (1 = use each file once)
# Higher values = more variety from the same background files
background_paths_duplication_rate:
  - 1
  - 2    # Use music clips twice as often
  - 1
```

### Key Settings

- **`background_paths`**: List of directories containing background audio files
- **`background_paths_duplication_rate`**: How many times to use each directory's files (useful for oversampling important noise types)
- **`augmentation_rounds`**: How many times to augment each generated clip (default: 1)

## What Audio Files to Use

### Recommended Types

1. **Speech Audio**
   - Conversations, podcasts, radio
   - Different languages and accents
   - Helps model ignore other people talking

2. **Music**
   - Various genres and styles
   - Different volume levels
   - Helps model work with background music

3. **Environmental Noise**
   - Traffic, crowds, appliances
   - White/pink/brown noise
   - Helps model work in noisy environments

4. **TV/Media Audio**
   - News, shows, commercials
   - Realistic home/office environments

### File Requirements

- **Format**: Any audio format supported by `torchaudio` (WAV, MP3, FLAC, etc.)
- **Sample Rate**: Will be automatically resampled to 16kHz
- **Channels**: Mono or stereo (will be converted to mono)
- **Length**: Any length (will be trimmed/padded as needed)

## Where to Get Background Audio

### Option 1: Download Datasets

**AudioSet** (Recommended - Large, diverse):
```bash
# Download from HuggingFace
# See notebook for full instructions
# https://huggingface.co/datasets/agkphysics/AudioSet
```

**Free Music Archive (FMA)**:
```bash
# Available via HuggingFace datasets
# Good for music background
```

**MIT Environmental Impulse Responses**:
```bash
# For room acoustics simulation
# https://www.openslr.org/28/
```

### Option 2: Collect Your Own

Create directories and add your own audio files:

```bash
mkdir -p background_clips
# Add your audio files here
# - Recordings from your deployment environment
# - Music files
# - Noise samples
# - Speech recordings
```

### Option 3: Generate Synthetic Noise

You can also use generated noise (colored noise: white, pink, brown, blue, violet) - this is handled automatically by the augmentation process.

## Augmentation Details

### Signal-to-Noise Ratio (SNR)

The augmentation process mixes background audio at random SNR levels:
- **Default range**: -10 dB to 15 dB
- **Lower SNR** = more background noise (harder to detect wake word)
- **Higher SNR** = less background noise (easier to detect)

### Additional Augmentations Applied

When you use `augment_clips`, the following are also applied:

1. **AddBackgroundNoise** (75% probability)
   - Mixes your background audio files
   - Random SNR levels

2. **AddColoredNoise** (25% probability)
   - White, pink, blue, brown, or violet noise
   - Additional noise layer

3. **RIR (Room Impulse Response)** (50% probability)
   - Simulates different room acoustics
   - Makes audio sound like it's in different spaces

4. **PitchShift** (25% probability)
   - Slight pitch variations
   - More speaker diversity

5. **SevenBandParametricEQ** (25% probability)
   - Frequency response variations
   - Simulates different microphones/devices

6. **TanhDistortion** (25% probability)
   - Slight distortion
   - Simulates device limitations

7. **BandStopFilter** (25% probability)
   - Frequency filtering
   - More acoustic variety

8. **Gain** (100% probability)
   - Volume level variations
   - Realistic volume differences

## Best Practices

### 1. Diversity is Key
- Use **multiple types** of background audio
- Include **various environments** (home, office, car, etc.)
- Mix **different languages** if multilingual support needed

### 2. Match Your Deployment Environment
- If deploying in homes, use home-like background noise
- If deploying in cars, include car noise
- If deploying in offices, include office chatter

### 3. Quantity Matters
- **More background files = better robustness**
- Aim for **hours of background audio** (not just minutes)
- The augmentation process will reuse files automatically

### 4. Balance Training Data
- Don't make background too loud (model won't learn)
- Don't make background too quiet (not realistic)
- The default SNR range (-10 to 15 dB) is a good starting point

### 5. Use Duplication Rates Strategically
```yaml
background_paths:
  - "./home_noise"      # Common in your deployment
  - "./music"           # Less common
  - "./speech"          # Very common

background_paths_duplication_rate:
  - 3    # Use home noise 3x more often
  - 1    # Use music normally
  - 2    # Use speech 2x more often
```

## Example Setup

### Minimal Setup (Quick Test)

```yaml
background_paths:
  - "./background_clips"  # Just one directory

background_paths_duplication_rate:
  - 1

augmentation_rounds: 1
```

**What you need:**
- Create `./background_clips/` directory
- Add 10-20 audio files (any format, any length)
- That's it! The augmentation will handle the rest

### Production Setup (Best Results)

```yaml
background_paths:
  - "./background_speech"      # Conversations, podcasts
  - "./background_music"       # Various music genres
  - "./background_noise"       # Environmental noise
  - "./background_media"       # TV, radio, etc.

background_paths_duplication_rate:
  - 2    # Speech is very common
  - 1    # Music is common
  - 1    # Noise is common
  - 1    # Media is less common

augmentation_rounds: 2  # Augment each clip twice for more variety
```

**What you need:**
- Multiple directories with different audio types
- Hours of audio in each category
- Download from datasets (AudioSet, FMA, etc.) or collect your own

## Troubleshooting

### "Background audio directory not found"
- Create the directory: `mkdir -p background_clips`
- Add some audio files to it
- Or set `background_paths: []` to skip background augmentation (not recommended)

### Model performs poorly in noisy environments
- Add more diverse background audio
- Increase `augmentation_rounds` to 2 or 3
- Use lower SNR values (more noise)
- Add more background files

### Training takes too long
- Reduce `augmentation_rounds` to 1
- Use fewer background files
- Reduce `augmentation_batch_size` (but this may reduce quality)

### Model too sensitive to background noise
- The model should learn to ignore background - this is expected behavior
- If it's activating on background noise, you may need:
  - More negative training examples
  - Better threshold tuning
  - Custom verifier models

## Advanced: Custom Augmentation

You can modify the augmentation probabilities in the code if needed:

```python
augmentation_probabilities = {
    "AddBackgroundNoise": 0.75,  # 75% chance of adding background
    "AddColoredNoise": 0.25,      # 25% chance of colored noise
    "RIR": 0.5,                   # 50% chance of room reverb
    # ... etc
}
```

See `openwakeword/data.py` for the `augment_clips` function details.

## Summary

**Background noise augmentation is essential** for training robust wake word models. Even a small amount of background audio will significantly improve your model's performance in real-world conditions.

**Quick Start:**
1. Create `./background_clips/` directory
2. Add 10-20 audio files (any format)
3. Configure in your YAML file
4. Run training - augmentation happens automatically!

**For Best Results:**
- Use diverse audio types (speech, music, noise)
- Collect hours of background audio
- Match your deployment environment
- Use multiple directories with duplication rates

---

*For more details, see the training notebook and `openwakeword/data.py`*

