"""
MobileNetV3-Small Training Script - Indian Currency Recognition
================================================================
Fine-tunes MobileNetV3-Small on Indian banknote classes.
Uses transfer learning with ImageNet weights.

The number of output classes is determined automatically from the
dataset directory structure — no hard-coded class count.

Pipeline:
  prepare_dataset.py → train_model.py → currency_model.h5

Architecture:
  MobileNetV3Small (frozen) → GlobalAveragePooling2D →
  Dense(256, relu) → Dropout(0.3) → Dense(128, relu) →
  Dropout(0.2) → Dense(num_classes, softmax)

Training:
  Stage 1: Frozen backbone, lr=1e-4, ~15 epochs
  Stage 2: Fine-tune last 30 layers, lr=1e-5, ~15 epochs
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from pathlib import Path
from datetime import datetime

# ── GPU Configuration ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[OK] GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"     GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"[!] GPU config error: {e}")
else:
    print("\n[!] No GPU detected — training will use CPU (slower)")

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-5

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'
TRAIN_DIR    = DATASET_ROOT / 'train'
VAL_DIR      = DATASET_ROOT / 'val'
MODELS_DIR   = PROJECT_ROOT / 'models'

MODELS_DIR.mkdir(exist_ok=True)

# ── ImageNet normalization constants ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def imagenet_preprocess(img):
    """
    Custom preprocessing function for ImageDataGenerator.
    Normalises to ImageNet mean/std after rescaling to [0,1].
    """
    img = img / 255.0
    img[..., 0] = (img[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    img[..., 1] = (img[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    img[..., 2] = (img[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    return img


# ── Dataset Validation ───────────────────────────────────────────────────────

def validate_dataset():
    """
    Validate dataset directory structure before training.

    Checks:
      1. Train folder exists
      2. Val folder exists
      3. Both contain the same set of class names
      4. Prints number of images per class

    Returns:
        (class_names, train_counts, val_counts) on success.
        Calls sys.exit(1) on failure.
    """
    print("\n[*] Validating dataset...")
    errors = []

    # 1. Check train folder
    if not TRAIN_DIR.exists():
        errors.append(f"Training folder not found: {TRAIN_DIR}")
    # 2. Check val folder
    if not VAL_DIR.exists():
        errors.append(f"Validation folder not found: {VAL_DIR}")

    if errors:
        for e in errors:
            print(f"    [FAIL] {e}")
        print("\n    Run: python scripts/prepare_dataset.py")
        sys.exit(1)

    # Discover class sub-folders
    train_classes = sorted([
        d.name for d in TRAIN_DIR.iterdir() if d.is_dir()
    ])
    val_classes = sorted([
        d.name for d in VAL_DIR.iterdir() if d.is_dir()
    ])

    if not train_classes:
        print(f"    [FAIL] No class sub-folders found in {TRAIN_DIR}")
        sys.exit(1)

    # 3. Check both splits have the same classes
    if train_classes != val_classes:
        print("    [FAIL] Train and val class folders do not match!")
        print(f"           Train: {train_classes}")
        print(f"           Val:   {val_classes}")
        sys.exit(1)

    # 4. Count images per class
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def count_images(folder):
        return sum(
            1 for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

    train_counts = {}
    val_counts = {}
    for cls in train_classes:
        train_counts[cls] = count_images(TRAIN_DIR / cls)
        val_counts[cls] = count_images(VAL_DIR / cls)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())

    print(f"    [OK] Dataset validated")
    print(f"    Classes: {len(train_classes)}")
    print(f"    Names:   {train_classes}")
    print()
    print(f"    {'Class':<16} {'Train':>8} {'Val':>8}")
    print(f"    {'─' * 16} {'─' * 8} {'─' * 8}")
    for cls in train_classes:
        print(f"    {cls:<16} {train_counts[cls]:>8} {val_counts[cls]:>8}")
    print(f"    {'─' * 16} {'─' * 8} {'─' * 8}")
    print(f"    {'TOTAL':<16} {total_train:>8} {total_val:>8}")

    if total_train == 0:
        print("\n    [FAIL] No training images found!")
        sys.exit(1)
    if total_val == 0:
        print("\n    [FAIL] No validation images found!")
        sys.exit(1)

    return train_classes, train_counts, val_counts


# ── Data Generators ──────────────────────────────────────────────────────────

def prepare_data_generators():
    """
    Create train and validation data generators.

    Training generator applies real-time augmentation:
      - Rotation, shift, shear, zoom
      - Horizontal flip
      - Brightness variation

    Both generators apply ImageNet normalization.
    """
    print("\n[*] Preparing data generators...")

    # Training: augmentation + ImageNet normalization
    train_datagen = ImageDataGenerator(
        preprocessing_function=imagenet_preprocess,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )

    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42,
    )

    # Validation: only normalization, no augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=imagenet_preprocess,
    )

    val_gen = val_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    num_classes = train_gen.num_classes

    print(f"    Training samples:   {train_gen.samples}")
    print(f"    Validation samples: {val_gen.samples}")
    print(f"    Classes detected:   {num_classes}")
    print(f"    Class indices:      {train_gen.class_indices}")

    return train_gen, val_gen, num_classes


# ── Model Creation ───────────────────────────────────────────────────────────

def create_model(num_classes):
    """
    Create MobileNetV3-Small with classification head.

    The output Dense layer size is set dynamically from `num_classes`,
    which is determined at runtime from the dataset — no hard-coding.

    Architecture:
      Input(224,224,3) → MobileNetV3Small(frozen, ImageNet) →
      GlobalAveragePooling2D → Dense(256,relu) → Dropout(0.3) →
      Dense(128,relu) → Dropout(0.2) → Dense(num_classes, softmax)
    """
    print(f"\n[*] Creating MobileNetV3-Small model ({num_classes} classes)...")

    # Load pre-trained backbone
    base_model = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0,
        minimalistic=False,
        include_preprocessing=False,  # We handle preprocessing ourselves
    )

    # Freeze all backbone layers for Stage 1
    base_model.trainable = False

    # Build classification head
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile with Stage 1 learning rate
    model.compile(
        optimizer=Adam(learning_rate=LR_STAGE1),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print(f"    Output classes: {num_classes}")
    print(f"    Parameters:     {model.count_params():,}")
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"    Trainable:      {trainable:,}")

    return model, base_model


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model, base_model, train_gen, val_gen):
    """
    Two-stage training:

    Stage 1 — Transfer Learning (frozen backbone):
      - Only classification head is trained
      - Learning rate: 1e-4
      - ~15 epochs with early stopping

    Stage 2 — Fine-tuning (partial unfreeze):
      - Unfreeze last 30 layers of backbone
      - Learning rate: 1e-5
      - ~15 epochs with early stopping
    """

    # ── Stage 1: Frozen backbone ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  STAGE 1: Transfer Learning (frozen backbone)")
    print("─" * 50)

    callbacks_s1 = [
        EarlyStopping(
            monitor='val_loss', patience=8,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'best_model_stage1.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    history_s1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks_s1,
        verbose=1,
    )

    # ── Stage 2: Fine-tuning ─────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  STAGE 2: Fine-tuning (unfreeze last 30 layers)")
    print("─" * 50)

    # Unfreeze backbone
    base_model.trainable = True

    # Keep early layers frozen (they learn generic features)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(
        1 for layer in base_model.layers if layer.trainable
    )
    print(f"    Unfrozen backbone layers: {trainable_count}")

    # Recompile with Stage 2 learning rate
    model.compile(
        optimizer=Adam(learning_rate=LR_STAGE2),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_s2 = [
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-8, verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'best_model_stage2.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    history_s2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks_s2,
        verbose=1,
    )

    return model, history_s1, history_s2


# ── Save ─────────────────────────────────────────────────────────────────────

def save_model(model, train_gen):
    """Save trained model and class mapping."""
    print("\n[*] Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save timestamped copy
    ts_path = MODELS_DIR / f'currency_model_{timestamp}.h5'
    model.save(str(ts_path))
    print(f"    Saved: {ts_path}")

    # Save as latest
    latest_path = MODELS_DIR / 'currency_model.h5'
    model.save(str(latest_path))
    print(f"    Saved: {latest_path}")

    # Save class mapping (index → label)
    class_indices = train_gen.class_indices          # {'INR_10': 0, ...}
    index_to_label = {str(v): k for k, v in class_indices.items()}

    mapping_path = MODELS_DIR / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(index_to_label, f, indent=2)
    print(f"    Saved: {mapping_path}")
    print(f"    Mapping: {index_to_label}")

    return latest_path


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_history(history_s1, history_s2):
    """Plot and save training curves for both stages."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Indian Currency Recognition — Training History',
                     fontsize=14, fontweight='bold')

        # Stage 1
        axes[0, 0].plot(history_s1.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history_s1.history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 0].set_title('Stage 1 — Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history_s1.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history_s1.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Stage 1 — Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Stage 2
        axes[1, 0].plot(history_s2.history['accuracy'], label='Train', linewidth=2)
        axes[1, 0].plot(history_s2.history['val_accuracy'], label='Val', linewidth=2)
        axes[1, 0].set_title('Stage 2 (Fine-tune) — Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(history_s2.history['loss'], label='Train', linewidth=2)
        axes[1, 1].plot(history_s2.history['val_loss'], label='Val', linewidth=2)
        axes[1, 1].set_title('Stage 2 (Fine-tune) — Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = MODELS_DIR / 'training_history.png'
        plt.savefig(str(plot_path), dpi=120)
        print(f"\n[*] Training plot saved: {plot_path}")
        plt.close()

    except ImportError:
        print("[!] matplotlib not available, skipping plot")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Indian Currency Recognition — Model Training")
    print("  Architecture: MobileNetV3-Small + Transfer Learning")
    print("=" * 60)

    # ── Step 1: Validate dataset ─────────────────────────────────────────
    class_names, train_counts, val_counts = validate_dataset()
    num_classes = len(class_names)

    # ── Step 2: Prepare data generators ──────────────────────────────────
    train_gen, val_gen, detected_classes = prepare_data_generators()

    # Sanity check: generator class count must match folder scan
    if detected_classes != num_classes:
        print(f"\n[FAIL] Generator found {detected_classes} classes but "
              f"folder scan found {num_classes}. Dataset may be corrupt.")
        sys.exit(1)

    # ── Logging summary ──────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  Training Configuration Summary")
    print("─" * 50)
    print(f"    Number of classes:      {num_classes}")
    print(f"    Class names:            {class_names}")
    print(f"    Training images:        {train_gen.samples}")
    print(f"    Validation images:      {val_gen.samples}")
    print(f"    Image size:             {IMG_SIZE}x{IMG_SIZE}")
    print(f"    Batch size:             {BATCH_SIZE}")
    print(f"    Stage 1 LR:             {LR_STAGE1}")
    print(f"    Stage 2 LR:             {LR_STAGE2}")
    print(f"    Stage 1 epochs (max):   {EPOCHS_STAGE1}")
    print(f"    Stage 2 epochs (max):   {EPOCHS_STAGE2}")

    # ── Step 3: Create model (dynamic class count) ───────────────────────
    model, base_model = create_model(num_classes)

    # ── Step 4: Train ────────────────────────────────────────────────────
    model, hist1, hist2 = train_model(model, base_model, train_gen, val_gen)

    # ── Step 5: Save ─────────────────────────────────────────────────────
    model_path = save_model(model, train_gen)

    # ── Step 6: Plot ─────────────────────────────────────────────────────
    plot_training_history(hist1, hist2)

    # ── Final evaluation ─────────────────────────────────────────────────
    print("\n[*] Final evaluation on validation set...")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"    Validation Loss:     {val_loss:.4f}")
    print(f"    Validation Accuracy: {val_acc * 100:.1f}%")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  Model:        {model_path}")
    print(f"  Class map:    {MODELS_DIR / 'class_mapping.json'}")
    print(f"  Val Accuracy: {val_acc * 100:.1f}%")
    print(f"\n  Next step: python scripts/convert_tflite.py\n")


if __name__ == '__main__':
    main()
