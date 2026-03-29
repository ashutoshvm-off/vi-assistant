#!/usr/bin/env python3
"""
Currency Detection System - Main Launcher
Provides easy access to all system functions
"""
import cv2
import numpy as np
import json
import tensorflow as tf
from pathlib import Path

# --- CONFIGURATION ---
# This ensures the script looks inside currency-detection/models/
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'currency_model.h5' 
MAPPING_PATH = PROJECT_ROOT / 'models' / 'class_mapping.json'
IMG_SIZE = 224

def run_accuracy_test():
    # 1. Load the Class Labels
    if not MAPPING_PATH.exists():
        print(f"Error: mapping file not found at {MAPPING_PATH}")
        print("Make sure you have run 'python scripts/train_model.py' first.")
        return
    
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    # Convert keys to integers for easy lookup
    class_mapping = {int(k): v for k, v in mapping.items()}
    
    # 2. Load the Keras Model (.h5)
    print(f"[*] Loading model for accuracy check: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("[✓] Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Start Webcam via OpenCV
    cap = cv2.VideoCapture(0) # 0 = Default Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[*] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- OPENCV PREPROCESSING ---
        # Resize to 224x224
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalize 
        img_normalized = img_rgb.astype(np.float32) / 255.0
        # Add batch dimension
        input_tensor = np.expand_dims(img_normalized, axis=0)

        # --- PREDICTION ---
        preds = model.predict(input_tensor, verbose=0)
        idx = np.argmax(preds[0])
        conf = preds[0][idx]
        label = class_mapping.get(idx, "Unknown")

        # --- UI OVERLAY ---
        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
        display_text = f"{label}: {conf*100:.1f}%"
        
        cv2.putText(frame, display_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Webcam Accuracy Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_accuracy_test()