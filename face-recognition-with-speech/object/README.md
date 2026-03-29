# Object Detection + VL53L0X Collision Avoidance

This module adds real-time obstacle awareness for your Smart Vision Assistance System.

## What it does

- Detects objects from camera frames using MobileNet-SSD (OpenCV DNN)
- Reads distance from VL53L0X ToF sensor (I2C) or UART LiDAR modules
- Combines vision + distance to classify risk as clear/warning/critical
- Emits coordinator events:

COORD_EVENT|source=hazard|type=collision|priority=0|text=...

So your root coordinator can announce hazards with highest priority.

## Hardware wiring (Raspberry Pi CM4)

Use I2C for VL53L0X:

- VIN -> 3.3V
- GND -> GND
- SDA -> SDA (GPIO2)
- SCL -> SCL (GPIO3)

Enable I2C:

- sudo raspi-config
- Interface Options -> I2C -> Enable

## UART LiDAR support (for TF-Luna / TFmini-style modules)

If your sensor is a UART LiDAR module, connect TX/RX to a Pi serial port and run with:

python object/main.py --lidar-type uart --serial-port /dev/ttyS0 --serial-baud 115200

You can also use aliases:

- --lidar-type tfluna
- --lidar-type tfmini

## Setup

1. Install dependencies

pip install -r object/requirements.txt

1. Download detector model files

python object/scripts/download_models.py

1. Run module standalone

python object/main.py

Optional flags:

- --simulate-tof (use fake constant distance for dev machine)
- --no-display (headless)
- --no-speech
- --lidar-only (no camera model inference, only LiDAR collision logic)
- --detect-interval 2 (run detector every 2 frames to reduce lag)
- --lidar-type auto|vl53l0x|uart|tfluna|tfmini
- --serial-port /dev/ttyS0
- --serial-baud 115200
- --warning-cm 120 --critical-cm 60

## High-accuracy custom training

For best accuracy, train on your own labeled obstacle dataset.

Quick dataset bootstrap (open source only):

python object/scripts/get_india_road_home_datasets.py

This downloads COCO128 and prepares folders for India-road/home datasets that require manual terms acceptance.

1. Prepare YOLO-format dataset under object/dataset

1. Copy template and edit classes/paths:

cp object/dataset/data.template.yaml object/dataset/data.yaml

1. Install training package:

pip install ultralytics

1. Train:

python object/scripts/train_detector_yolo.py --data object/dataset/data.yaml --model yolov8n.pt --epochs 120 --imgsz 640

For higher accuracy (slower): use --model yolov8s.pt or yolov8m.pt.

## Coordinator mode

Once launcher is updated, run:

python main.py --module coordinated --target cm4 --components face voice currency object
