"""
Lightweight inference module for currency detection
Can be imported and used as a library in other Python projects
"""

import json
import numpy as np
import cv2
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class CurrencyDetectorLite:
    """Lightweight currency detector for use as library"""
    
    def __init__(self, model_dir=None):
        """
        Initialize detector
        Args:
            model_dir: Path to models directory (default: ../models relative to this file)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.img_size = 224
        self.interpreter = None
        self.class_mapping = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
        self._load_classes()
    
    def _load_model(self):
        """Load TFLite model"""
        model_path = self.model_dir / 'currency_model.tflite'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def _load_classes(self):
        """Load class mapping"""
        mapping_path = self.model_dir / 'class_mapping.json'
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found: {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        self.class_mapping = {int(k): v for k, v in mapping.items()}
    
    def detect(self, image_path=None, frame=None):
        """
        Detect currency in image
        Args:
            image_path: Path to image file OR
            frame: OpenCV frame (BGR format)
        
        Returns:
            {
                "currency": "INR",
                "value": 500,
                "confidence": 0.95,
                "raw_output": [probabilities]
            }
        """
        if image_path is not None:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Could not load image: {image_path}")
        elif frame is None:
            raise ValueError("Either image_path or frame must be provided")
        
        # Preprocess
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        # Infer
        self.interpreter.set_tensor(self.input_details[0]['index'], frame_batch.astype(np.float32))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Parse result
        pred_idx = np.argmax(output)
        confidence = float(output[pred_idx])
        denomination = self.class_mapping[pred_idx]
        
        # Extract currency and value
        parts = denomination.split('_')
        currency = parts[0] if len(parts) > 0 else "UNKNOWN"
        value = int(parts[1]) if len(parts) > 1 else None
        
        return {
            "currency": currency,
            "value": value,
            "confidence": confidence,
            "raw_output": output.tolist()
        }


def detect_currency(image_path=None, frame=None, model_dir=None):
    """
    Convenience function for one-off detection
    Args:
        image_path: Path to image file
        frame: OpenCV frame
        model_dir: Path to models directory
    
    Returns:
        Detection result dictionary
    """
    detector = CurrencyDetectorLite(model_dir=model_dir)
    return detector.detect(image_path=image_path, frame=frame)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Currency Detection Library')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model-dir', help='Path to models directory')
    
    args = parser.parse_args()
    
    result = detect_currency(image_path=args.image, model_dir=args.model_dir)
    print(json.dumps(result, indent=2))
