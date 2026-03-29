"""
Test Inference Script
Tests the TFLite model on sample images
Verifies model is working correctly before deployment
"""

import json
import numpy as np
import cv2
from pathlib import Path
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATASET_ROOT = PROJECT_ROOT / 'dataset' / 'val'

IMG_SIZE = 224


def load_tflite_model():
    """Load TFLite model"""
    model_path = MODELS_DIR / 'currency_model.tflite'
    
    if not model_path.exists():
        print(f"[!] TFLite model not found: {model_path}")
        return None
    
    print(f"[*] Loading TFLite model: {model_path}")
    
    try:
        interpreter = tflite.Interpreter(model_path=str(model_path))
    except Exception as e:
        print(f"[!] Error loading with tflite_runtime: {e}")
        print("    Trying with TensorFlow Lite...")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    
    interpreter.allocate_tensors()
    print("[*] Model loaded successfully")
    
    return interpreter


def get_class_mapping():
    """Load class mapping"""
    mapping_path = MODELS_DIR / 'class_mapping.json'
    
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    return {int(k): v for k, v in class_mapping.items()}


def preprocess_image(image_path):
    """
    Load and preprocess image for inference
    """
    img = cv2.imread(str(image_path))
    
    if img is None:
        return None
    
    # Resize to model input size
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization (if used in training)
    # Uncomment if your training used ImageNet normalization
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img_normalized = (img_normalized - mean) / std
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch.astype(np.float32)


def run_inference(interpreter, image_batch):
    """Run inference on preprocessed image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], image_batch)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output[0], inference_time


def test_on_images():
    """Test inference on sample images from dataset"""
    print("\n[*] Testing inference on sample images...\n")
    
    # Load model
    interpreter = load_tflite_model()
    if interpreter is None:
        return
    
    # Get class mapping
    class_mapping = get_class_mapping()
    print(f"[*] Classes: {class_mapping}\n")
    
    # Find test images
    test_images = []
    for denom_dir in DATASET_ROOT.iterdir():
        if denom_dir.is_dir():
            images = list(denom_dir.glob('*.jpg')) + list(denom_dir.glob('*.png'))
            test_images.extend(images[:2])  # Take first 2 images from each class
    
    if not test_images:
        print("[!] No test images found")
        return
    
    print(f"[*] Found {len(test_images)} test images\n")
    print("Results:")
    print("-" * 100)
    print(f"{'Image':<40} {'Predicted':<20} {'Confidence':<15} {'Time (ms)':<12}")
    print("-" * 100)
    
    total_time = 0
    correct = 0
    
    for img_path in test_images[:10]:  # Test first 10
        # Get true label
        true_label = img_path.parent.name
        
        # Preprocess
        img_batch = preprocess_image(img_path)
        if img_batch is None:
            print(f"Error loading {img_path.name}")
            continue
        
        # Run inference
        output, inf_time = run_inference(interpreter, img_batch)
        total_time += inf_time
        
        # Get prediction
        pred_idx = np.argmax(output)
        pred_label = class_mapping[pred_idx]
        confidence = output[pred_idx] * 100
        
        # Check if correct
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {img_path.name:<38} {pred_label:<20} {confidence:<14.2f}% {inf_time:<11.2f}")
    
    print("-" * 100)
    avg_time = total_time / len(test_images[:10])
    accuracy = (correct / len(test_images[:10])) * 100
    
    print(f"\nAverage Inference Time: {avg_time:.2f} ms")
    print(f"Sample Accuracy: {accuracy:.1f}%")
    print(f"Expected FPS: {1000/avg_time:.1f} FPS")


def test_with_dummy_input():
    """Test with dummy input to verify model works"""
    print("\n[*] Testing with dummy input...")
    
    interpreter = load_tflite_model()
    if interpreter is None:
        return
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Create dummy input
    input_shape = input_details[0]['shape']
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    print("\nRunning dummy inference...")
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0]}")
    print(f"Sum of probabilities: {np.sum(output[0]):.4f}")
    
    print("\n[✓] Model inference working correctly!")


def main():
    print("\n" + "="*100)
    print("Currency Detection - TFLite Inference Test")
    print("="*100)
    
    # Test with dummy input first
    test_with_dummy_input()
    
    # Test with actual images
    test_on_images()
    
    print("\n" + "="*100)
    print("Testing Complete!")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
