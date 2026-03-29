"""
TensorFlow Lite Conversion Script - Indian Currency Recognition
================================================================
Converts trained Keras .h5 model to .tflite format for inference.

Pipeline:
  currency_model.h5 → convert_tflite.py → currency_model.tflite

Conversion method:
  TFLiteConverter.from_keras_model() with dynamic range quantization.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / 'models'
DATASET_ROOT = PROJECT_ROOT / 'dataset' / 'val'

IMG_SIZE = 224

# ── ImageNet normalization (must match training) ─────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model():
    """Load the trained Keras model directly."""
    print("[*] Loading trained model...")

    h5_path = MODELS_DIR / 'currency_model.h5'
    if not h5_path.exists():
        print(f"[!] Model not found: {h5_path}")
        print("    Run: python scripts/train_model.py")
        return None

    model = tf.keras.models.load_model(str(h5_path))

    print(f"    Loaded:       {h5_path.name}")
    print(f"    Input shape:  {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")
    print(f"    Parameters:   {model.count_params():,}")

    return model


def create_representative_dataset():
    """
    Create representative dataset generator for quantization calibration.
    Uses validation images with proper ImageNet normalization.
    """
    print("\n[*] Creating representative dataset for quantization...")

    def representative_data_gen():
        import cv2

        image_count = 0
        max_images = 100

        for class_dir in sorted(DATASET_ROOT.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob('*.jpg'))[:15]:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Match inference preprocessing exactly
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                normalized = resized.astype(np.float32) / 255.0

                # ImageNet normalization
                normalized[..., 0] = (normalized[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                normalized[..., 1] = (normalized[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                normalized[..., 2] = (normalized[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

                yield [np.expand_dims(normalized, axis=0)]

                image_count += 1
                if image_count >= max_images:
                    return

    return representative_data_gen


def convert_to_tflite(model):
    """
    Convert Keras model to TFLite with dynamic range quantization.
    This reduces model size while maintaining float32 inference.
    """
    print("\n[*] Converting to TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic range quantization (weights quantized, activations float)
    print("    Quantization: dynamic range")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    print("    Converting...")
    tflite_model = converter.convert()

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"    Done. Model size: {size_mb:.2f} MB")

    return tflite_model


def save_tflite_model(tflite_model):
    """Save TFLite model to disk."""
    output_path = MODELS_DIR / 'currency_model.tflite'

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n[*] Saved: {output_path}  ({size_mb:.2f} MB)")

    return output_path


def test_tflite_inference(tflite_path):
    """Run a quick inference test on the converted TFLite model."""
    print(f"\n[*] Testing TFLite inference...")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    Input:  shape={input_details[0]['shape']}  "
          f"dtype={input_details[0]['dtype']}")
    print(f"    Output: shape={output_details[0]['shape']}  "
          f"dtype={output_details[0]['dtype']}")

    # Create dummy input matching preprocessing
    dummy = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], dummy)

    # Time the inference
    t0 = time.time()
    interpreter.invoke()
    ms = (time.time() - t0) * 1000

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = int(np.argmax(output))
    top_conf = float(output[top_idx])

    print(f"    Inference time: {ms:.1f} ms")
    print(f"    Output sum:     {output.sum():.4f} (should be ~1.0)")
    print(f"    Top class idx:  {top_idx}  confidence: {top_conf:.4f}")

    # Load class mapping
    mapping_path = MODELS_DIR / 'class_mapping.json'
    if mapping_path.exists():
        with open(mapping_path) as f:
            class_map = json.load(f)
        label = class_map.get(str(top_idx), f'class_{top_idx}')
        print(f"    Top label:      {label}")
    else:
        print("    [!] class_mapping.json not found")

    # Test with a real validation image if available
    if DATASET_ROOT.exists():
        import cv2
        for class_dir in sorted(DATASET_ROOT.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob('*.jpg'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Preprocess exactly like inference.py
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                normalized = resized.astype(np.float32) / 255.0
                normalized[..., 0] = (normalized[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                normalized[..., 1] = (normalized[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                normalized[..., 2] = (normalized[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                tensor = np.expand_dims(normalized, axis=0)

                interpreter.set_tensor(input_details[0]['index'], tensor)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]

                top_idx = int(np.argmax(output))
                top_conf = float(output[top_idx])
                label = class_map.get(str(top_idx), f'class_{top_idx}') if mapping_path.exists() else f'class_{top_idx}'

                print(f"\n    Real image test:")
                print(f"      File:       {img_path.name}")
                print(f"      True class: {class_dir.name}")
                print(f"      Predicted:  {label} ({top_conf*100:.1f}%)")
                break
            break

    return True


def generate_model_info(tflite_path):
    """Generate model_info.json with all metadata for inference."""
    print("\n[*] Generating model info...")

    mapping_path = MODELS_DIR / 'class_mapping.json'
    class_map = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            class_map = json.load(f)

    model_info = {
        'model_name': 'indian_currency_model',
        'version': '2.0',
        'framework': 'TensorFlow Lite',
        'architecture': 'MobileNetV3-Small + custom head',
        'input': {
            'shape': [1, IMG_SIZE, IMG_SIZE, 3],
            'dtype': 'float32',
            'color_format': 'RGB',
            'normalization': 'ImageNet',
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD,
        },
        'output': {
            'shape': [1, len(class_map)],
            'dtype': 'float32',
            'activation': 'softmax',
        },
        'num_classes': len(class_map),
        'classes': class_map,
        'model_file': 'currency_model.tflite',
        'model_size_mb': round(tflite_path.stat().st_size / 1024 / 1024, 2),
        'quantization': 'dynamic_range',
        'inference_settings': {
            'num_threads': 4,
            'use_xnnpack': True,
        },
    }

    info_path = MODELS_DIR / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"    Saved: {info_path}")

    return model_info


def main():
    print("\n" + "=" * 60)
    print("  Indian Currency — TFLite Conversion")
    print("=" * 60)

    # Load Keras model
    model = load_model()
    if model is None:
        return

    # Convert
    tflite_model = convert_to_tflite(model)

    # Save
    tflite_path = save_tflite_model(tflite_model)

    # Test
    test_tflite_inference(tflite_path)

    # Generate info
    generate_model_info(tflite_path)

    print("\n" + "=" * 60)
    print("  Conversion Complete!")
    print("=" * 60)
    print(f"\n  TFLite model: {tflite_path}")
    print(f"  Model info:   {MODELS_DIR / 'model_info.json'}")
    print(f"\n  Next step: python src/inference.py\n")


if __name__ == '__main__':
    main()