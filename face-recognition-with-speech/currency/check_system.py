#!/usr/bin/env python3
"""
Quick Test Script - Verify system setup
Run this to check if all dependencies are installed correctly
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:<25} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:<25} NOT INSTALLED")
        return False


def check_files():
    """Check if required files exist"""
    project_root = Path(__file__).parent
    
    print("\n" + "="*60)
    print("Checking Files")
    print("="*60)
    
    required_files = {
        'models/currency_model.tflite': 'Trained TFLite model',
        'models/class_mapping.json': 'Class labels',
        'src/inference.py': 'Inference script',
        'src/detector.py': 'Detector library',
        'scripts/train_model.py': 'Training script',
    }
    
    all_found = True
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"✓ {file_path:<40} {size_str}")
        else:
            print(f"✗ {file_path:<40} NOT FOUND")
            all_found = False
    
    return all_found


def check_model():
    """Quick model inference test"""
    print("\n" + "="*60)
    print("Model Test")
    print("="*60)
    
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'currency_model.tflite'
    
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return False
    
    try:
        import numpy as np
        
        # Try tflite_runtime first
        try:
            import tflite_runtime.interpreter as tflite
            using_runtime = 'tflite_runtime'
        except ImportError:
            import tensorflow.lite as tflite
            using_runtime = 'tensorflow.lite'
        
        print(f"  Using: {using_runtime}")
        
        # Load model
        interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=4)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        # Test inference
        import time
        input_shape = input_details[0]['shape']
        test_input = np.random.rand(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        start = time.time()
        interpreter.invoke()
        inf_time = (time.time() - start) * 1000
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  Inference time: {inf_time:.2f} ms")
        print(f"  Output range: {output.min():.3f} - {output.max():.3f}")
        
        print("✓ Model inference working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("Currency Detection System - Health Check")
    print("="*60 + "\n")
    
    print("Checking Python Packages")
    print("="*60)
    
    packages = [
        ('numpy', 'numpy'),
        ('opencv', 'cv2'),
        ('pillow', 'PIL'),
        ('requests', 'requests'),
    ]
    
    # Check for TFLite
    try:
        import tflite_runtime
        packages.append(('tflite_runtime', 'tflite_runtime.interpreter'))
    except:
        try:
            import tensorflow
            packages.append(('tensorflow', 'tensorflow.lite'))
        except:
            pass
    
    all_packages = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_packages = False
    
    # Check files
    all_files = check_files()
    
    # Test model
    model_ok = check_model()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if all_packages and all_files and model_ok:
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. python src/inference.py --image test_image.jpg")
        print("  2. python src/inference.py (for camera)")
        return 0
    else:
        print("✗ Some checks failed. Please fix issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
