#!/bin/bash
# Installation script for local development (Linux/macOS)

echo "================================"
echo "Currency Detection - Local Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "[3/5] Installing development dependencies..."
pip install -r requirements-dev.txt

# Create dataset directories
echo "[4/5] Creating dataset directories..."
mkdir -p dataset/train/{INR_10,INR_20,INR_50,INR_100,INR_200,INR_500}
mkdir -p dataset/val/{INR_10,INR_20,INR_50,INR_100,INR_200,INR_500}
mkdir -p models
mkdir -p logs

echo "[5/5] Setup complete!"
echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Prepare dataset: python scripts/prepare_dataset.py"
echo "3. Train model: python scripts/train_model.py"
echo "4. Convert to TFLite: python scripts/convert_tflite.py"
echo "5. Test inference: python scripts/test_inference.py"
echo ""
