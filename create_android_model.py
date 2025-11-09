#!/usr/bin/env python3
"""
Script to create an Android-compatible wake word model using LiteRT/TFLite format.

This script simplifies the process of creating a wake word model for Android deployment.
It handles the full pipeline: data generation, augmentation, training, and TFLite export.

Usage:
    python create_android_model.py --config android_model_config.yml --wake-word "hey assistant"
"""

import argparse
import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'scipy', 'scikit-learn',
        'onnx', 'tensorflow', 'onnx_tf', 'yaml'
    ]
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logging.error(f"Missing required packages: {', '.join(missing)}")
        logging.info("Install with: pip install openwakeword[full]")
        return False
    
    # Check platform (training is Linux-only)
    import platform
    if platform.system() != 'Linux':
        logging.warning("⚠️  Model training is primarily supported on Linux systems.")
        logging.warning("   Training on other platforms may not work due to Piper TTS requirements.")
        logging.warning("   Consider using a Linux VM, container, or Google Colab.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def check_prerequisites(config):
    """Check if prerequisite directories and files exist."""
    issues = []
    
    # Check piper sample generator
    piper_path = config.get('piper_sample_generator_path', './piper-sample-generator')
    if not os.path.exists(piper_path):
        issues.append(f"Piper sample generator not found at: {piper_path}")
        issues.append("  Clone it with: git clone https://github.com/dscripka/piper-sample-generator")
    
    # Check RIR paths
    rir_paths = config.get('rir_paths', [])
    for rir_path in rir_paths:
        if not os.path.exists(rir_path):
            issues.append(f"RIR directory not found: {rir_path}")
            issues.append("  Download from: https://www.openslr.org/28/")
    
    # Check background paths
    background_paths = config.get('background_paths', [])
    for bg_path in background_paths:
        if not os.path.exists(bg_path):
            issues.append(f"Background audio directory not found: {bg_path}")
            issues.append("  Create this directory and add audio files for augmentation")
    
    # Check validation data (optional but recommended)
    val_data_path = config.get('false_positive_validation_data_path', '')
    if val_data_path and not os.path.exists(val_data_path):
        issues.append(f"Validation data not found: {val_data_path}")
        issues.append("  Download from: https://huggingface.co/datasets/davidscripka/openwakeword_features")
        issues.append("  Or leave empty to use a smaller validation set")
    
    if issues:
        logging.warning("Prerequisites check found issues:")
        for issue in issues:
            logging.warning(f"  - {issue}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def create_minimal_config(wake_word, output_dir='./android_model_output'):
    """Create a minimal configuration for quick testing."""
    config_content = f"""## Minimal configuration for Android model
model_name: "android_wakeword"
target_phrase:
  - "{wake_word}"
custom_negative_phrases: []
n_samples: 5000  # Reduced for quick testing
n_samples_val: 500
tts_batch_size: 50
augmentation_batch_size: 16
piper_sample_generator_path: "./piper-sample-generator"
output_dir: "{output_dir}"
rir_paths:
  - "./mit_rirs"
background_paths:
  - "./background_clips"
background_paths_duplication_rate:
  - 1
false_positive_validation_data_path: ""
augmentation_rounds: 1
feature_data_files: {{}}
batch_n_per_class:
  "adversarial_negative": 50
  "positive": 50
model_type: "dnn"
layer_size: 32
steps: 10000  # Reduced for quick testing
max_negative_weight: 1500
target_false_positives_per_hour: 0.2
"""
    return config_content


def run_training_pipeline(config_path, skip_generation=False, skip_augmentation=False):
    """Run the complete training pipeline."""
    train_script = os.path.join(os.path.dirname(__file__), 'openwakeword', 'train.py')
    
    commands = []
    
    # Step 1: Generate synthetic clips
    if not skip_generation:
        logging.info("=" * 60)
        logging.info("Step 1: Generating synthetic training clips")
        logging.info("=" * 60)
        cmd = [
            sys.executable, train_script,
            '--training_config', config_path,
            '--generate_clips'
        ]
        commands.append(('Generate clips', cmd))
    
    # Step 2: Augment clips and compute features
    if not skip_augmentation:
        logging.info("=" * 60)
        logging.info("Step 2: Augmenting clips and computing features")
        logging.info("=" * 60)
        cmd = [
            sys.executable, train_script,
            '--training_config', config_path,
            '--augment_clips'
        ]
        commands.append(('Augment clips', cmd))
    
    # Step 3: Train model
    logging.info("=" * 60)
    logging.info("Step 3: Training the model")
    logging.info("=" * 60)
    cmd = [
        sys.executable, train_script,
        '--training_config', config_path,
        '--train_model'
    ]
    commands.append(('Train model', cmd))
    
    # Execute commands
    for step_name, cmd in commands:
        logging.info(f"\nRunning: {step_name}")
        logging.info(f"Command: {' '.join(cmd)}\n")
        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error in {step_name}: {e}")
            return False
        except KeyboardInterrupt:
            logging.warning(f"\nInterrupted during {step_name}")
            return False
    
    return True


def verify_model_output(config):
    """Verify that the model was created successfully."""
    output_dir = config.get('output_dir', './android_model_output')
    model_name = config.get('model_name', 'android_wakeword')
    
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    if os.path.exists(tflite_path):
        file_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        logging.info(f"\n✓ TFLite model created successfully!")
        logging.info(f"  Location: {tflite_path}")
        logging.info(f"  Size: {file_size:.2f} MB")
        return True, tflite_path
    elif os.path.exists(onnx_path):
        logging.warning(f"\n⚠ ONNX model found but TFLite conversion may have failed")
        logging.warning(f"  ONNX location: {onnx_path}")
        logging.warning(f"  You can manually convert using the convert_onnx_to_tflite function")
        return False, onnx_path
    else:
        logging.error(f"\n✗ Model files not found in {output_dir}")
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description='Create an Android-compatible wake word model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create model with custom config
  python create_android_model.py --config android_model_config.yml
  
  # Quick start with minimal config
  python create_android_model.py --wake-word "hey assistant"
  
  # Skip data generation (if already done)
  python create_android_model.py --config android_model_config.yml --skip-generation
  
  # Only train (skip generation and augmentation)
  python create_android_model.py --config android_model_config.yml --skip-generation --skip-augmentation
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--wake-word',
        type=str,
        help='Wake word/phrase to detect (creates minimal config if --config not provided)'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip synthetic clip generation step'
    )
    parser.add_argument(
        '--skip-augmentation',
        action='store_true',
        help='Skip augmentation and feature computation step'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./android_model_output',
        help='Output directory for the model (used with --wake-word)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine config file
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config_path = args.config
    elif args.wake_word:
        # Create minimal config
        config_content = create_minimal_config(args.wake_word, args.output_dir)
        config_path = './android_model_config_minimal.yml'
        with open(config_path, 'w') as f:
            f.write(config_content)
        logging.info(f"Created minimal config: {config_path}")
        
        import yaml
        config = yaml.safe_load(config_content)
    else:
        parser.error("Either --config or --wake-word must be provided")
    
    # Check prerequisites
    if not check_prerequisites(config):
        sys.exit(1)
    
    # Run training pipeline
    success = run_training_pipeline(
        config_path,
        skip_generation=args.skip_generation,
        skip_augmentation=args.skip_augmentation
    )
    
    if success:
        # Verify output
        model_created, model_path = verify_model_output(config)
        if model_created:
            logging.info("\n" + "=" * 60)
            logging.info("Model ready for Android deployment!")
            logging.info("=" * 60)
            logging.info(f"\nNext steps:")
            logging.info(f"1. Copy the TFLite model to your Android project")
            logging.info(f"2. Use LiteRT (ai-edge-litert) in your Android app")
            logging.info(f"3. Load the model and process 16-bit PCM audio at 16kHz")
            logging.info(f"4. Process audio in 80ms frames (1280 samples)")
            logging.info(f"\nModel file: {model_path}")
        else:
            logging.warning("\nModel creation completed but verification failed")
            logging.warning("Check the output directory for generated files")
    else:
        logging.error("\nTraining pipeline failed. Check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

