"""
Example Usage Scripts for Currency Detection System
Copy and adapt these examples for your use case
"""

# ============================================================================
# Example 1: Real-time Camera Detection
# ============================================================================

"""
Basic real-time detection with camera display
"""

from src.inference import CurrencyDetector

detector = CurrencyDetector()
detector.capture_from_camera(use_display=True)


# ============================================================================
# Example 2: Single Image Detection
# ============================================================================

"""
Detect currency in a single image file
"""

from src.detector import CurrencyDetectorLite
import json

# Initialize detector
detector = CurrencyDetectorLite()

# Detect from image
result = detector.detect(image_path='path/to/currency.jpg')

print(json.dumps(result, indent=2))
# Output:
# {
#   "currency": "INR",
#   "value": 500,
#   "confidence": 0.956,
#   "raw_output": [0.001, 0.002, 0.003, ...]
# }


# ============================================================================
# Example 3: OpenCV Frame Detection
# ============================================================================

"""
Detect from live OpenCV video capture
"""

import cv2
from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect on frame
    result = detector.detect(frame=frame)
    
    # Draw result on frame
    cv2.putText(
        frame,
        f"{result['currency']} {result['value']} ({result['confidence']*100:.1f}%)",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        2
    )
    
    cv2.imshow('Currency Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================================
# Example 4: Batch Processing Images
# ============================================================================

"""
Process multiple images and save results to JSON
"""

import json
from pathlib import Path
from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()
image_dir = Path('images/')
results = []

for image_file in image_dir.glob('*.jpg'):
    try:
        result = detector.detect(image_path=str(image_file))
        result['image'] = image_file.name
        results.append(result)
        print(f"✓ {image_file.name}: {result['currency']} {result['value']}")
    except Exception as e:
        print(f"✗ {image_file.name}: {e}")

# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nProcessed {len(results)} images")


# ============================================================================
# Example 5: Confidence Filtering
# ============================================================================

"""
Only report detections above confidence threshold
"""

from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()
CONFIDENCE_THRESHOLD = 0.85

result = detector.detect(image_path='currency.jpg')

if result['confidence'] >= CONFIDENCE_THRESHOLD:
    print(f"High confidence detection: {result['currency']} {result['value']}")
    # Process detection
else:
    print(f"Low confidence ({result['confidence']:.2f}), rejecting")


# ============================================================================
# Example 6: Multi-frame Voting
# ============================================================================

"""
Average predictions across multiple frames for stability
"""

import cv2
from collections import Counter
from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()
WINDOW_SIZE = 5

predictions = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect(frame=frame)
    predictions.append(result['value'])
    
    # Keep only last N frames
    if len(predictions) > WINDOW_SIZE:
        predictions.pop(0)
    
    # Get consensus if we have enough frames
    if len(predictions) == WINDOW_SIZE:
        most_common = Counter(predictions).most_common(1)[0][0]
        avg_confidence = sum(r['confidence'] for r in [result]) / len([result])
        
        print(f"Consensus: {most_common} (confidence: {avg_confidence:.2f})")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ============================================================================
# Example 7: Save to CSV
# ============================================================================

"""
Log detections to CSV file
"""

import csv
from datetime import datetime
from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()

# Create CSV writer
csv_file = open('detections.csv', 'w', newline='')
writer = csv.DictWriter(csv_file, fieldnames=['timestamp', 'currency', 'value', 'confidence'])
writer.writeheader()

# Example detection
result = detector.detect(image_path='currency.jpg')

writer.writerow({
    'timestamp': datetime.now().isoformat(),
    'currency': result['currency'],
    'value': result['value'],
    'confidence': f"{result['confidence']:.3f}"
})

csv_file.close()


# ============================================================================
# Example 8: Real-time with Database
# ============================================================================

"""
Log detections to SQLite database
"""

import sqlite3
from datetime import datetime
from src.detector import CurrencyDetectorLite

# Create database
conn = sqlite3.connect('detections.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        currency TEXT,
        value INTEGER,
        confidence REAL
    )
''')

detector = CurrencyDetectorLite()
result = detector.detect(image_path='currency.jpg')

cursor.execute('''
    INSERT INTO detections (timestamp, currency, value, confidence)
    VALUES (?, ?, ?, ?)
''', (
    datetime.now().isoformat(),
    result['currency'],
    result['value'],
    result['confidence']
))

conn.commit()
conn.close()


# ============================================================================
# Example 9: REST API Wrapper
# ============================================================================

"""
Wrap detector in Flask REST API
"""

from flask import Flask, request, jsonify
from src.detector import CurrencyDetectorLite
import base64
import io
import cv2
import numpy as np

app = Flask(__name__)
detector = CurrencyDetectorLite()

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect
    Form data: image (file upload)
    Returns: {currency, value, confidence}
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Detect
    result = detector.detect(frame=frame)
    
    return jsonify(result)


@app.route('/detect-base64', methods=['POST'])
def detect_base64():
    """
    POST /detect-base64
    JSON: {image: "base64-encoded-image"}
    """
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # Decode image
    image_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect
    result = detector.detect(frame=frame)
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Usage:
# curl -X POST -F "image=@currency.jpg" http://localhost:5000/detect


# ============================================================================
# Example 10: Performance Benchmarking
# ============================================================================

"""
Benchmark model performance
"""

import time
import numpy as np
from src.detector import CurrencyDetectorLite

detector = CurrencyDetectorLite()

# Dummy frames
num_frames = 100
inference_times = []

print("Benchmarking...")
for i in range(num_frames):
    # Create random frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    start = time.time()
    result = detector.detect(frame=frame)
    elapsed = (time.time() - start) * 1000  # ms
    
    inference_times.append(elapsed)
    
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{num_frames}")

print(f"\nResults:")
print(f"  Mean: {np.mean(inference_times):.2f} ms")
print(f"  Median: {np.median(inference_times):.2f} ms")
print(f"  Min: {np.min(inference_times):.2f} ms")
print(f"  Max: {np.max(inference_times):.2f} ms")
print(f"  Std Dev: {np.std(inference_times):.2f} ms")
print(f"  FPS: {1000/np.mean(inference_times):.1f}")


# ============================================================================
# Example 11: Custom Model Path
# ============================================================================

"""
Use detector with custom model location
"""

from src.detector import CurrencyDetectorLite

# Load from custom path
detector = CurrencyDetectorLite(model_dir='/custom/model/path')
result = detector.detect(image_path='currency.jpg')
print(result)


# ============================================================================
# Example 12: Error Handling
# ============================================================================

"""
Robust error handling for production
"""

from src.detector import CurrencyDetectorLite
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    detector = CurrencyDetectorLite()
    result = detector.detect(image_path='currency.jpg')
    
    # Validate result
    if result['currency'] == 'UNKNOWN':
        logger.warning("No currency detected")
    elif result['confidence'] < 0.7:
        logger.warning(f"Low confidence: {result['confidence']:.2f}")
    else:
        logger.info(f"Detected: {result['currency']} {result['value']}")

except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
except Exception as e:
    logger.error(f"Detection failed: {e}")


# ============================================================================
# Example 13: Configuration Management
# ============================================================================

"""
Load configuration from JSON
"""

import json
from pathlib import Path
from src.detector import CurrencyDetectorLite

# Load config
config_path = Path('config.json')
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {
        'model_dir': './models',
        'confidence_threshold': 0.85,
        'num_threads': 4,
        'input_size': 224
    }

# Use config
detector = CurrencyDetectorLite(model_dir=config['model_dir'])
result = detector.detect(image_path='currency.jpg')

if result['confidence'] >= config['confidence_threshold']:
    print(f"Accepted: {result}")
else:
    print(f"Rejected (low confidence): {result}")


# ============================================================================
# Example 14: Multi-threading
# ============================================================================

"""
Process multiple images in parallel
"""

import threading
from pathlib import Path
from src.detector import CurrencyDetectorLite

def process_image(image_path, results, index):
    """Process single image in thread"""
    try:
        detector = CurrencyDetectorLite()
        result = detector.detect(image_path=str(image_path))
        results[index] = result
        print(f"✓ {image_path.name}")
    except Exception as e:
        results[index] = {'error': str(e)}
        print(f"✗ {image_path.name}: {e}")

# Process images
image_dir = Path('images/')
image_files = list(image_dir.glob('*.jpg'))
results = [None] * len(image_files)

threads = []
for i, image_file in enumerate(image_files):
    t = threading.Thread(target=process_image, args=(image_file, results, i))
    threads.append(t)
    t.start()

# Wait for completion
for t in threads:
    t.join()

print(f"Processed {len([r for r in results if r])} images")
