#!/usr/bin/env python3
"""
================================================================================
  CURRENCY DETECTION SYSTEM - FILE INDEX & QUICK REFERENCE
================================================================================

This file provides a quick reference to all files in the project.
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_STRUCTURE = """
currency-detection/
│
├── 📚 DOCUMENTATION
│   ├── MAIN_README.md              ← START HERE! Overview of everything
│   ├── QUICK_START.md              ← 30-second quickstart
│   ├── README.md                   ← Full feature documentation
│   ├── INSTALLATION.md             ← Step-by-step installation
│   ├── DEPLOYMENT_GUIDE.md         ← Raspberry Pi specific
│   ├── PROJECT_SUMMARY.md          ← Technical details
│   └── FILE_INDEX.md               ← This file!
│
├── 🐍 TRAINING SCRIPTS
│   └── scripts/
│       ├── setup-dev.sh            ← Setup development environment
│       ├── setup-rpi.sh            ← Setup Raspberry Pi
│       ├── setup-local.sh          ← Local setup (Linux/Mac)
│       ├── prepare_dataset.py      ← Download & prepare data
│       ├── train_model.py          ← Train MobileNetV3 model
│       ├── convert_tflite.py       ← Convert to TensorFlow Lite
│       └── test_inference.py       ← Test model accuracy
│
├── 🎯 INFERENCE SCRIPTS
│   └── src/
│       ├── inference.py            ← Main inference script (Pi)
│       └── detector.py             ← Detector library
│
├── 🔧 UTILITIES
│   └── check_system.py             ← Verify system setup
│
├── 📋 EXAMPLES & REFERENCE
│   ├── EXAMPLES.py                 ← 14 usage examples
│   ├── FILE_INDEX.md               ← This file
│   └── requirements*.txt           ← Dependencies
│
├── 📦 DATA DIRECTORIES (CREATED DURING SETUP)
│   ├── dataset/                    ← Training data
│   ├── models/                     ← Trained models
│   └── venv/                       ← Virtual environment
│
└── 📄 CONFIGURATION
    ├── requirements.txt            ← General dependencies
    ├── requirements-dev.txt        ← Development dependencies
    └── requirements-rpi.txt        ← Raspberry Pi dependencies
"""

# ============================================================================
# FILE DESCRIPTIONS
# ============================================================================

FILE_DESCRIPTIONS = {
    "DOCUMENTATION": {
        "MAIN_README.md": {
            "Purpose": "Main project overview and quick reference",
            "Content": "Features, quick start, commands, troubleshooting",
            "Read_First": True,
            "File_Size": "~15 KB",
            "Time": "5 minutes"
        },
        "QUICK_START.md": {
            "Purpose": "Minimal quickstart guide",
            "Content": "Copy-paste commands to get running",
            "Read_First": True,
            "File_Size": "~5 KB",
            "Time": "2 minutes"
        },
        "README.md": {
            "Purpose": "Complete feature documentation",
            "Content": "Detailed API, configuration, customization",
            "Read_First": False,
            "File_Size": "~30 KB",
            "Time": "15 minutes"
        },
        "INSTALLATION.md": {
            "Purpose": "Step-by-step installation guide",
            "Content": "Detailed setup, troubleshooting, optimization",
            "Read_First": False,
            "File_Size": "~17 KB",
            "Time": "20 minutes"
        },
        "DEPLOYMENT_GUIDE.md": {
            "Purpose": "Raspberry Pi deployment specific",
            "Content": "Pi setup, monitoring, performance tuning",
            "Read_First": False,
            "File_Size": "~12 KB",
            "Time": "15 minutes"
        },
        "PROJECT_SUMMARY.md": {
            "Purpose": "Technical project overview",
            "Content": "Architecture, files, workflow, specifications",
            "Read_First": False,
            "File_Size": "~12 KB",
            "Time": "10 minutes"
        }
    },
    
    "TRAINING_SCRIPTS": {
        "setup-dev.sh": {
            "Purpose": "Setup development environment",
            "Usage": "bash scripts/setup-dev.sh",
            "Does": "Create venv, install deps, create directories",
            "Output": "venv/ directory with all dependencies",
            "Time": "5-10 minutes"
        },
        "setup-rpi.sh": {
            "Purpose": "Setup Raspberry Pi",
            "Usage": "bash scripts/setup-rpi.sh",
            "Does": "Install system packages, Python deps, TFLite",
            "Output": "Pi ready for inference",
            "Time": "15-20 minutes"
        },
        "setup-local.sh": {
            "Purpose": "Setup local development (Linux/Mac)",
            "Usage": "bash scripts/setup-local.sh",
            "Does": "Same as setup-dev.sh",
            "Output": "venv/ with dependencies",
            "Time": "5-10 minutes"
        },
        "prepare_dataset.py": {
            "Purpose": "Download and prepare training data",
            "Usage": "python scripts/prepare_dataset.py",
            "Does": "Download INR images, augment data, split train/val",
            "Output": "dataset/ directory with augmented images",
            "Time": "2-5 minutes"
        },
        "train_model.py": {
            "Purpose": "Train MobileNetV3-Small on currency images",
            "Usage": "python scripts/train_model.py",
            "Does": "Fine-tune MobileNetV3, save best model, plot curves",
            "Output": "models/currency_model.h5 and best_model_*.h5",
            "Time": "15-30 minutes (CPU), 5-10 minutes (GPU)"
        },
        "convert_tflite.py": {
            "Purpose": "Convert Keras model to TensorFlow Lite",
            "Usage": "python scripts/convert_tflite.py",
            "Does": "Quantize, optimize, convert to .tflite",
            "Output": "models/currency_model.tflite (2.5-3.5 MB)",
            "Time": "2-3 minutes"
        },
        "test_inference.py": {
            "Purpose": "Test model inference on sample images",
            "Usage": "python scripts/test_inference.py",
            "Does": "Load model, run inference, check accuracy",
            "Output": "Accuracy metrics and inference timing",
            "Time": "1-2 minutes"
        }
    },
    
    "INFERENCE_SCRIPTS": {
        "src/inference.py": {
            "Purpose": "Main real-time inference script for Pi",
            "Usage": "python src/inference.py",
            "Features": "Camera capture, display, JSON output, FPS monitoring",
            "Modes": "Camera mode, headless, single image",
            "Args": "--image, --no-display, --output"
        },
        "src/detector.py": {
            "Purpose": "Lightweight detector library",
            "Usage": "from src.detector import CurrencyDetectorLite",
            "Features": "Image/frame detection, library interface",
            "Output": "JSON with currency, value, confidence"
        }
    },
    
    "UTILITIES": {
        "check_system.py": {
            "Purpose": "Verify system setup",
            "Usage": "python check_system.py",
            "Checks": "Packages, files, model inference",
            "Output": "✓/✗ for each check"
        }
    },
    
    "EXAMPLES": {
        "EXAMPLES.py": {
            "Purpose": "14 real-world usage examples",
            "Content": "Camera, image, batch processing, REST API, etc.",
            "Examples": [
                "Real-time camera detection",
                "Single image detection",
                "OpenCV frame detection",
                "Batch processing",
                "Confidence filtering",
                "Multi-frame voting",
                "CSV logging",
                "Database logging",
                "REST API wrapper",
                "Performance benchmarking",
                "Custom model path",
                "Error handling",
                "Configuration management",
                "Multi-threading"
            ]
        }
    },
    
    "DEPENDENCIES": {
        "requirements.txt": {
            "Purpose": "General Python dependencies",
            "Usage": "pip install -r requirements.txt",
            "Packages": "numpy, cv2, tflite-runtime, PIL, requests"
        },
        "requirements-dev.txt": {
            "Purpose": "Development dependencies (with TensorFlow)",
            "Usage": "pip install -r requirements-dev.txt",
            "Packages": "tensorflow, matplotlib, scikit-learn, etc."
        },
        "requirements-rpi.txt": {
            "Purpose": "Optimized for Raspberry Pi",
            "Usage": "pip install -r requirements-rpi.txt",
            "Packages": "tflite-runtime (optimized), numpy, cv2, PIL"
        }
    }
}

# ============================================================================
# QUICK COMMAND REFERENCE
# ============================================================================

COMMANDS = {
    "Development": [
        ("Setup environment", "bash scripts/setup-dev.sh && source venv/bin/activate"),
        ("Prepare data", "python scripts/prepare_dataset.py"),
        ("Train model", "python scripts/train_model.py"),
        ("Convert TFLite", "python scripts/convert_tflite.py"),
        ("Test model", "python scripts/test_inference.py"),
    ],
    
    "Raspberry Pi": [
        ("Setup Pi", "bash scripts/setup-rpi.sh"),
        ("Copy model", "scp models/currency_model.tflite pi@raspberry:~/currency-detection/models/"),
        ("Run realtime", "python src/inference.py"),
        ("Headless mode", "python src/inference.py --no-display"),
        ("Single image", "python src/inference.py --image path/to/image.jpg"),
    ],
    
    "Debugging": [
        ("Check system", "python check_system.py"),
        ("Verify camera", "vcgencmd get_camera"),
        ("Test camera", "raspistill -o test.jpg"),
    ]
}

# ============================================================================
# READING GUIDE
# ============================================================================

READING_GUIDE = """
IF YOU ARE...                           READ THIS
─────────────────────────────────────────────────────────────
New to the project                      → MAIN_README.md
In a hurry                              → QUICK_START.md
Setting up for first time               → INSTALLATION.md
Deploying to Raspberry Pi                → DEPLOYMENT_GUIDE.md
Looking for code examples                → EXAMPLES.py
Troubleshooting issues                   → INSTALLATION.md (Troubleshooting)
Understanding the architecture           → PROJECT_SUMMARY.md
Checking if setup is correct             → python check_system.py
Looking for API documentation            → README.md
Learning about customization             → README.md (Customization)
"""

# ============================================================================
# DIRECTORY PURPOSES
# ============================================================================

DIRECTORY_PURPOSES = {
    "scripts/": "Training and setup scripts",
    "src/": "Inference and detector code",
    "models/": "Trained models (created during training)",
    "dataset/": "Training data (created during setup)",
    "venv/": "Virtual environment (created during setup)"
}

# ============================================================================
# EXECUTION TIMELINE
# ============================================================================

TIMELINE = """
FIRST-TIME SETUP TIMELINE
──────────────────────────
1. Read MAIN_README.md                  (5 min)
2. bash scripts/setup-dev.sh            (5-10 min)
3. python scripts/prepare_dataset.py    (2-5 min)
4. python scripts/train_model.py        (15-30 min) ← Longest step
5. python scripts/convert_tflite.py     (2-3 min)
6. python scripts/test_inference.py     (1-2 min)
7. bash scripts/setup-rpi.sh (on Pi)    (10-15 min)
8. Transfer model to Pi                 (1-2 min)
9. python src/inference.py              (done!)
───────────────────────────────────────
TOTAL: ~1-2 hours

SUBSEQUENT RUNS
───────────────
1. Activate venv
2. python src/inference.py (on Pi)
───────────────────────────────────────
TIME: 30 seconds
"""

# ============================================================================
# KEY METRICS
# ============================================================================

METRICS = {
    "Model Size": "2.5-3.5 MB",
    "Inference Time": "80-150 ms/frame",
    "Expected FPS": "6-12 on Pi 4",
    "Memory Usage": "80-150 MB",
    "CPU Usage": "60-80%",
    "Training Time": "15-30 min (CPU), 5-10 min (GPU)",
    "Setup Time": "30-60 minutes (first time)",
    "Accuracy": ">90% on validation set"
}

# ============================================================================
# OUTPUT GUIDE
# ============================================================================

OUTPUT_FORMAT = {
    "JSON Detection": {
        "Example": '{"currency": "INR", "value": 500, "confidence": 0.956}',
        "Fields": {
            "currency": "Currency code (INR)",
            "value": "Denomination (10, 20, 50, 100, 200, 500)",
            "confidence": "Detection confidence (0.0-1.0)"
        }
    },
    
    "Files Generated": {
        "During Training": [
            "models/currency_model.h5 (Keras model)",
            "models/best_model_stage1.h5",
            "models/best_model_stage2.h5",
            "models/training_history.png"
        ],
        "During Conversion": [
            "models/currency_model.tflite (Main model)",
            "models/class_mapping.json",
            "models/model_info.json",
            "models/CONVERSION_REPORT.txt"
        ],
        "On Pi": [
            "detections.jsonl (if --output specified)",
            "/tmp/detections.jsonl (if running as service)"
        ]
    }
}

# ============================================================================
# MAIN EXECUTION GUIDE
# ============================================================================

def print_guide():
    """Print comprehensive file guide"""
    
    print("\n" + "="*80)
    print("CURRENCY DETECTION SYSTEM - COMPREHENSIVE FILE GUIDE")
    print("="*80)
    
    print("\n" + "─"*80)
    print("PROJECT STRUCTURE")
    print("─"*80)
    print(PROJECT_STRUCTURE)
    
    print("\n" + "─"*80)
    print("READING GUIDE - WHERE TO START")
    print("─"*80)
    print(READING_GUIDE)
    
    print("\n" + "─"*80)
    print("QUICK COMMANDS")
    print("─"*80)
    for category, cmds in COMMANDS.items():
        print(f"\n{category}:")
        for desc, cmd in cmds:
            print(f"  {desc:30} {cmd}")
    
    print("\n" + "─"*80)
    print("EXECUTION TIMELINE")
    print("─"*80)
    print(TIMELINE)
    
    print("\n" + "─"*80)
    print("KEY METRICS")
    print("─"*80)
    for metric, value in METRICS.items():
        print(f"  {metric:25} {value}")
    
    print("\n" + "="*80)
    print("START HERE: Read MAIN_README.md then run QUICK_START.md commands")
    print("="*80 + "\n")


if __name__ == '__main__':
    print_guide()
