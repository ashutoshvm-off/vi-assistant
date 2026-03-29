"""Configuration for object detection + collision avoidance."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_PROTOTXT = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
DEFAULT_MODEL = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
DEFAULT_TFLITE_MODEL = MODELS_DIR / "unified_hazard_roads_homes_int8.tflite"
DEFAULT_TFLITE_LABELS = MODELS_DIR / "unified_hazard_labels.txt"

# MobileNet-SSD class labels (Caffe model trained on PASCAL VOC).
CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Objects that can become immediate mobility hazards.
HAZARD_LABELS = {
    "person",
    "bicycle",
    "motorcycle",
    "motorbike",
    "bus",
    "car",
    "truck",
    "chair",
    "diningtable",
    "table",
    "pottedplant",
    "sofa",
    "train",
    "stairs",
    "obstacle",
}

# Collision policy defaults.
DEFAULT_WARNING_CM = 120.0
DEFAULT_CRITICAL_CM = 60.0
ALERT_COOLDOWN_SECONDS = 2.0

# Camera defaults.
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_FPS = 20
