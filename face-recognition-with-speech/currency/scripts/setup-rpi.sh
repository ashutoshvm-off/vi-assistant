#!/bin/bash
# Installation script for Raspberry Pi 4 ARM64

echo "================================"
echo "Currency Detection - Raspberry Pi Setup"
echo "================================"

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libjasper-dev \
    libtiff-dev \
    libharfbuzz0b \
    libwebp6-0 \
    libjasper1 \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libwebp6

echo "[2/5] Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

echo "[3/5] Installing TensorFlow Lite Runtime (optimized for RPi)..."
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite-runtime

echo "[4/5] Installing required Python packages..."
pip3 install -r requirements-rpi.txt

echo "[5/5] Setting up camera and enabling interfaces..."
# Optional: Enable camera interface
# sudo raspi-config nonint do_camera 0

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Copy the trained model (currency_model.tflite) to the models/ directory"
echo "2. Run: python3 src/inference.py"
echo ""
