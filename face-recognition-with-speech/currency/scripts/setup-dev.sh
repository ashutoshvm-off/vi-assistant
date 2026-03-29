#!/bin/bash
# Local development setup

echo "================================"
echo "Currency Detection - Development Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "[3/4] Installing development dependencies..."
pip install -r requirements-dev.txt

# Create dataset directories
echo "[4/4] Creating dataset directories..."
mkdir -p dataset/train/{INR_10,INR_20,INR_50,INR_100,INR_200,INR_500}
mkdir -p dataset/val/{INR_10,INR_20,INR_50,INR_100,INR_200,INR_500}

echo ""
echo "================================"
echo "Development Setup Complete!"
echo "================================"
echo ""
echo "Activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/prepare_dataset.py"
echo "2. Run: python scripts/train_model.py"
echo "3. Run: python scripts/convert_tflite.py"
echo ""
