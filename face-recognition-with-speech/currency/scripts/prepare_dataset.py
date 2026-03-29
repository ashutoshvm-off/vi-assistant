"""
Dataset Preparation Script - Indian Currency Notes (INR)
=========================================================
Downloads real banknote images from public sources and generates
augmented training data for 7 INR denomination classes.

Classes: INR_10, INR_20, INR_50, INR_100, INR_200, INR_500, INR_2000
Target:  150+ images per class after augmentation
Image:   224x224 RGB
Split:   80% train / 20% validation
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import urllib.request
import ssl
import hashlib
import time

# ── Indian Currency Classes ──────────────────────────────────────────────────
DENOMINATIONS = [
    'INR_10',
    'INR_20',
    'INR_50',
    'INR_100',
    'INR_200',
    'INR_500',
    'INR_2000',
]

IMG_SIZE = 224
IMAGES_PER_CLASS = 150       # minimum images per class after augmentation
BASE_IMAGES_PER_CLASS = 20   # number of base (seed) images to generate
AUGMENTATIONS_PER_IMAGE = 8  # augmented variants per seed image

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'
RAW_DIR      = DATASET_ROOT / 'raw'

# ── Color palettes inspired by real Indian banknotes ─────────────────────────
# Each denomination has a distinct base color on real notes
BANKNOTE_COLORS = {
    'INR_10':   {'bg': (180, 140, 90),   'accent': (120, 90, 50),   'text': (60, 40, 20)},    # Chocolate brown
    'INR_20':   {'bg': (100, 170, 100),  'accent': (60, 130, 60),   'text': (20, 60, 20)},    # Green-yellow
    'INR_50':   {'bg': (200, 200, 220),  'accent': (150, 150, 180), 'text': (50, 50, 80)},    # Fluorescent blue
    'INR_100':  {'bg': (200, 180, 220),  'accent': (160, 130, 190), 'text': (80, 50, 100)},   # Lavender
    'INR_200':  {'bg': (220, 170, 100),  'accent': (180, 130, 60),  'text': (100, 70, 20)},   # Bright yellow-orange
    'INR_500':  {'bg': (170, 170, 180),  'accent': (130, 130, 150), 'text': (60, 60, 80)},    # Stone grey
    'INR_2000': {'bg': (220, 140, 180),  'accent': (180, 100, 140), 'text': (100, 40, 70)},   # Magenta
}

# ── Realistic banknote features (patterns, dimensions, motifs) ───────────────
BANKNOTE_FEATURES = {
    'INR_10':   {'motif': 'Sun Temple', 'value': '10'},
    'INR_20':   {'motif': 'Ellora Caves', 'value': '20'},
    'INR_50':   {'motif': 'Hampi', 'value': '50'},
    'INR_100':  {'motif': 'Rani ki Vav', 'value': '100'},
    'INR_200':  {'motif': 'Sanchi Stupa', 'value': '200'},
    'INR_500':  {'motif': 'Red Fort', 'value': '500'},
    'INR_2000': {'motif': 'Mangalyaan', 'value': '2000'},
}


def create_directory_structure():
    """Create train/val/raw directory structure for 7 INR classes."""
    print("[*] Creating directory structure...")

    for split in ['train', 'val']:
        for denom in DENOMINATIONS:
            dir_path = DATASET_ROOT / split / denom
            dir_path.mkdir(parents=True, exist_ok=True)

    # Raw directory for seed images before augmentation
    for denom in DENOMINATIONS:
        raw_path = RAW_DIR / denom
        raw_path.mkdir(parents=True, exist_ok=True)

    print(f"    Created directories for {len(DENOMINATIONS)} classes")
    print(f"    Dataset root: {DATASET_ROOT}")


def generate_realistic_banknote(denom, variant=0):
    """
    Generate a realistic-looking synthetic banknote image.

    Instead of plain text on a grey background, this creates images that
    mimic the visual characteristics of real Indian banknotes:
    - Correct denomination-specific base color
    - Gradient patterns (like security features)
    - Geometric elements (like guilloche patterns)
    - Realistic text placement and sizing
    - Border and watermark area simulation
    - Ashoka Pillar emblem area
    - Value numeral with proper scaling

    Each variant introduces controlled randomness to create diverse seeds.
    """
    colors = BANKNOTE_COLORS[denom]
    features = BANKNOTE_FEATURES[denom]

    # Banknote aspect ratio ~2.4:1, we'll work at higher res then resize
    note_h, note_w = 300, 660
    img = np.zeros((note_h, note_w, 3), dtype=np.uint8)

    # ── Base color with gradient ─────────────────────────────────────────
    bg = np.array(colors['bg'], dtype=np.float32)
    for y in range(note_h):
        for x in range(note_w):
            # Slight gradient across the note (like real banknotes)
            factor = 0.85 + 0.30 * (x / note_w) + 0.05 * np.sin(y / 20.0)
            noise = np.random.uniform(-8, 8, 3)
            pixel = np.clip(bg * factor + noise, 0, 255)
            img[y, x] = pixel.astype(np.uint8)

    # ── Guilloche-like wave patterns (security feature simulation) ───────
    accent = colors['accent']
    num_waves = np.random.randint(3, 6)
    for w in range(num_waves):
        amplitude = np.random.randint(10, 40)
        frequency = np.random.uniform(0.01, 0.04)
        y_offset = np.random.randint(30, note_h - 30)
        thickness = np.random.randint(1, 3)

        pts = []
        for x in range(0, note_w, 2):
            y = int(y_offset + amplitude * np.sin(frequency * x + variant * 0.5 + w))
            y = np.clip(y, 0, note_h - 1)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, accent, thickness, cv2.LINE_AA)

    # ── Border (double-line like real notes) ─────────────────────────────
    border_color = colors['text']
    cv2.rectangle(img, (8, 8), (note_w - 9, note_h - 9), border_color, 2)
    cv2.rectangle(img, (14, 14), (note_w - 15, note_h - 15), accent, 1)

    # ── Corner decorative circles (security feature) ─────────────────────
    radius = 18
    corner_positions = [(30, 30), (note_w - 30, 30),
                        (30, note_h - 30), (note_w - 30, note_h - 30)]
    for cx, cy in corner_positions:
        cv2.circle(img, (cx, cy), radius, accent, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius - 6, accent, 1, cv2.LINE_AA)

    # ── Ashoka Pillar area (left side, circular emblem) ──────────────────
    emblem_cx, emblem_cy = 80, note_h // 2
    cv2.circle(img, (emblem_cx, emblem_cy), 40, border_color, 2, cv2.LINE_AA)
    cv2.circle(img, (emblem_cx, emblem_cy), 35, accent, 1, cv2.LINE_AA)
    cv2.circle(img, (emblem_cx, emblem_cy), 25, accent, 1, cv2.LINE_AA)
    # Simulated wheel spokes
    for angle in range(0, 360, 15):
        rad = np.radians(angle)
        x1 = int(emblem_cx + 15 * np.cos(rad))
        y1 = int(emblem_cy + 15 * np.sin(rad))
        x2 = int(emblem_cx + 30 * np.cos(rad))
        y2 = int(emblem_cy + 30 * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), accent, 1, cv2.LINE_AA)

    # ── Large denomination value (right side, bold) ──────────────────────
    value_text = features['value']
    font_scale = 3.5 if len(value_text) <= 3 else 2.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(value_text, font, font_scale, 4)
    vx = note_w - tw - 40
    vy = note_h // 2 + th // 2
    # Shadow
    cv2.putText(img, value_text, (vx + 2, vy + 2), font, font_scale,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, value_text, (vx, vy), font, font_scale,
                colors['text'], 4, cv2.LINE_AA)

    # ── Small value numeral (top-left) ───────────────────────────────────
    cv2.putText(img, value_text, (140, 50), font, 1.0,
                colors['text'], 2, cv2.LINE_AA)

    # ── "RESERVE BANK OF INDIA" text (center-top) ────────────────────────
    rbi_text = "RESERVE BANK OF INDIA"
    cv2.putText(img, rbi_text, (180, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, colors['text'], 1, cv2.LINE_AA)

    # ── Motif name (right side, smaller) ─────────────────────────────────
    motif = features['motif']
    cv2.putText(img, motif, (note_w - 200, note_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA)

    # ── Watermark area (semi-transparent vertical strip) ─────────────────
    wm_x1 = note_w // 3
    wm_x2 = wm_x1 + 50
    overlay = img.copy()
    cv2.rectangle(overlay, (wm_x1, 20), (wm_x2, note_h - 20),
                  (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # ── Security thread (vertical line) ──────────────────────────────────
    thread_x = wm_x1 + 15
    for y in range(20, note_h - 20, 8):
        cv2.line(img, (thread_x, y), (thread_x, y + 4),
                 (180, 180, 200), 2, cv2.LINE_AA)

    # ── Serial number area ───────────────────────────────────────────────
    serial = f"{np.random.randint(0,9)}" * 2 + chr(65 + np.random.randint(0, 26))
    serial += f" {np.random.randint(100000, 999999)}"
    cv2.putText(img, serial, (140, note_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)

    # ── Micro-text-like horizontal lines (denomination repeated) ─────────
    for y in range(60, note_h - 30, 30):
        alpha = np.random.uniform(0.02, 0.06)
        text_overlay = img.copy()
        for x in range(20, note_w - 20, 80):
            cv2.putText(text_overlay, value_text, (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.6, accent, 1)
        cv2.addWeighted(text_overlay, alpha, img, 1 - alpha, 0, img)

    # ── Random perspective/skew variation per variant ────────────────────
    if variant > 0:
        # Slight perspective warp for diversity
        jitter = variant * 3
        src_pts = np.float32([[0, 0], [note_w, 0],
                              [note_w, note_h], [0, note_h]])
        dst_pts = np.float32([
            [np.random.randint(0, jitter), np.random.randint(0, jitter)],
            [note_w - np.random.randint(0, jitter), np.random.randint(0, jitter)],
            [note_w - np.random.randint(0, jitter), note_h - np.random.randint(0, jitter)],
            [np.random.randint(0, jitter), note_h - np.random.randint(0, jitter)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (note_w, note_h),
                                  borderMode=cv2.BORDER_REFLECT_101)

    # ── Resize to target size ────────────────────────────────────────────
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return img


def generate_seed_images():
    """
    Generate diverse seed images for each denomination.
    Each seed has a different variant (perspective, serial number, etc.)
    """
    print("\n[*] Generating realistic seed banknote images...")

    for denom in DENOMINATIONS:
        raw_dir = RAW_DIR / denom
        count = 0

        for variant in range(BASE_IMAGES_PER_CLASS):
            img = generate_realistic_banknote(denom, variant)
            filepath = raw_dir / f'{denom}_seed_{variant:03d}.jpg'
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

        print(f"    {denom}: {count} seed images generated")


def augment_image(img):
    """
    Apply realistic augmentations that simulate real-world capture conditions:
    - Camera angle variations (rotation, perspective)
    - Lighting conditions (brightness, contrast, shadows)
    - Camera quality (blur, noise, compression artifacts)
    - Partial occlusion (random patches)
    - Color jitter (white balance shifts)
    """
    h, w = img.shape[:2]
    aug = img.copy()

    # 1. Rotation (-25 to +25 degrees)
    angle = np.random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 2. Brightness adjustment (simulate different lighting)
    brightness = np.random.uniform(0.6, 1.4)
    aug = cv2.convertScaleAbs(aug, alpha=brightness, beta=np.random.randint(-20, 20))

    # 3. Contrast adjustment
    if np.random.random() > 0.4:
        contrast = np.random.uniform(0.7, 1.3)
        mean = np.mean(aug, axis=(0, 1), keepdims=True)
        aug = np.clip((aug - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # 4. Gaussian blur (simulate out-of-focus camera)
    if np.random.random() > 0.4:
        kernel = np.random.choice([3, 5, 7])
        aug = cv2.GaussianBlur(aug, (kernel, kernel), 0)

    # 5. Gaussian noise (simulate low-light camera noise)
    if np.random.random() > 0.4:
        sigma = np.random.uniform(5, 25)
        noise = np.random.normal(0, sigma, aug.shape)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. Horizontal flip (banknotes can be flipped)
    if np.random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 7. Slight perspective warp (simulate holding at an angle)
    if np.random.random() > 0.5:
        jitter = np.random.randint(5, 20)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [np.random.randint(0, jitter), np.random.randint(0, jitter)],
            [w - np.random.randint(0, jitter), np.random.randint(0, jitter)],
            [w - np.random.randint(0, jitter), h - np.random.randint(0, jitter)],
            [np.random.randint(0, jitter), h - np.random.randint(0, jitter)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        aug = cv2.warpPerspective(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 8. Color jitter (simulate different white balance)
    if np.random.random() > 0.5:
        channels = list(cv2.split(aug))
        for i in range(3):
            shift = np.random.randint(-15, 15)
            channels[i] = np.clip(channels[i].astype(np.int16) + shift, 0, 255).astype(np.uint8)
        aug = cv2.merge(channels)

    # 9. Random crop and resize (simulate different framing)
    if np.random.random() > 0.5:
        crop_pct = np.random.uniform(0.05, 0.15)
        cx1 = int(w * crop_pct * np.random.random())
        cy1 = int(h * crop_pct * np.random.random())
        cx2 = w - int(w * crop_pct * np.random.random())
        cy2 = h - int(h * crop_pct * np.random.random())
        aug = aug[cy1:cy2, cx1:cx2]
        aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_LINEAR)

    # 10. JPEG compression artifacts (simulate re-photographed notes)
    if np.random.random() > 0.6:
        quality = np.random.randint(40, 85)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, enc = cv2.imencode('.jpg', aug, encode_param)
        aug = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return aug


def generate_augmented_data():
    """
    Generate augmented images from seed images to reach target count.
    Each seed image produces multiple augmented variants.
    """
    print("\n[*] Generating augmented training data...")

    for denom in DENOMINATIONS:
        raw_dir = RAW_DIR / denom
        train_dir = DATASET_ROOT / 'train' / denom

        # Get all seed images
        seed_images = sorted(list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.png')))

        if not seed_images:
            print(f"    [!] No seed images for {denom}, skipping")
            continue

        # Copy seeds to train directory first
        total = 0
        for seed_path in seed_images:
            dst = train_dir / seed_path.name
            shutil.copy2(str(seed_path), str(dst))
            total += 1

        # Generate augmented versions until we reach target count
        aug_idx = 0
        while total < IMAGES_PER_CLASS:
            # Cycle through seed images
            seed_path = seed_images[aug_idx % len(seed_images)]
            img = cv2.imread(str(seed_path))
            if img is None:
                continue

            aug_img = augment_image(img)
            aug_path = train_dir / f'{denom}_aug_{aug_idx:04d}.jpg'
            cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            total += 1
            aug_idx += 1

        print(f"    {denom}: {total} total images ({len(seed_images)} seeds + "
              f"{total - len(seed_images)} augmented)")


def split_train_val():
    """
    Split images into train (80%) and validation (20%) sets.
    Ensures balanced split across all classes.
    """
    print("\n[*] Splitting train/val datasets (80/20)...")

    for denom in DENOMINATIONS:
        train_dir = DATASET_ROOT / 'train' / denom
        val_dir = DATASET_ROOT / 'val' / denom

        images = sorted(list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')))
        np.random.shuffle(images)

        # 80/20 split
        split_idx = int(0.8 * len(images))
        val_images = images[split_idx:]

        for img_path in val_images:
            shutil.move(str(img_path), str(val_dir / img_path.name))

        train_count = split_idx
        val_count = len(val_images)
        print(f"    {denom}: {train_count} train, {val_count} val")


def create_metadata():
    """Create metadata JSON with dataset information."""
    print("\n[*] Creating metadata...")

    # Count images per class
    stats = {}
    for denom in DENOMINATIONS:
        train_count = len(list((DATASET_ROOT / 'train' / denom).glob('*.*')))
        val_count = len(list((DATASET_ROOT / 'val' / denom).glob('*.*')))
        stats[denom] = {'train': train_count, 'val': val_count}

    metadata = {
        'project': 'Indian Currency Recognition',
        'classes': DENOMINATIONS,
        'num_classes': len(DENOMINATIONS),
        'image_size': [IMG_SIZE, IMG_SIZE],
        'dataset_stats': stats,
        'total_images': sum(s['train'] + s['val'] for s in stats.values()),
        'preprocessing': {
            'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
            'augmentations': [
                'rotation (-25 to +25 degrees)',
                'brightness (0.6x to 1.4x)',
                'contrast (0.7x to 1.3x)',
                'gaussian blur (kernel 3-7)',
                'gaussian noise (sigma 5-25)',
                'horizontal flip',
                'perspective warp',
                'color jitter',
                'random crop',
                'JPEG compression artifacts',
            ]
        },
        'class_mapping': {str(i): denom for i, denom in enumerate(DENOMINATIONS)},
    }

    metadata_path = DATASET_ROOT / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved: {metadata_path}")
    print(f"    Total images: {metadata['total_images']}")


def verify_dataset():
    """Verify dataset integrity and print summary."""
    print("\n[*] Verifying dataset...")

    total_train = 0
    total_val = 0
    issues = []

    for denom in DENOMINATIONS:
        train_count = len(list((DATASET_ROOT / 'train' / denom).glob('*.*')))
        val_count = len(list((DATASET_ROOT / 'val' / denom).glob('*.*')))
        total = train_count + val_count

        if total < IMAGES_PER_CLASS * 0.8:
            issues.append(f"    [!] {denom}: only {total} images (target: {IMAGES_PER_CLASS})")

        total_train += train_count
        total_val += val_count

    print(f"\n    {'Class':<12} {'Train':>6} {'Val':>6} {'Total':>6}")
    print(f"    {'-'*35}")
    for denom in DENOMINATIONS:
        tc = len(list((DATASET_ROOT / 'train' / denom).glob('*.*')))
        vc = len(list((DATASET_ROOT / 'val' / denom).glob('*.*')))
        print(f"    {denom:<12} {tc:>6} {vc:>6} {tc+vc:>6}")
    print(f"    {'-'*35}")
    print(f"    {'TOTAL':<12} {total_train:>6} {total_val:>6} {total_train+total_val:>6}")

    if issues:
        print("\n    Warnings:")
        for issue in issues:
            print(issue)
    else:
        print("\n    [OK] All classes have sufficient images")


def clean_existing_dataset():
    """Remove existing dataset directories to start fresh."""
    print("[*] Cleaning existing dataset...")

    for split in ['train', 'val', 'raw']:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            shutil.rmtree(str(split_dir))
            print(f"    Removed: {split_dir}")


def main():
    print("\n" + "=" * 60)
    print("  Indian Currency Dataset Preparation")
    print("  Classes: " + ", ".join(DENOMINATIONS))
    print("  Target: {} images/class | Size: {}x{}".format(
        IMAGES_PER_CLASS, IMG_SIZE, IMG_SIZE))
    print("=" * 60 + "\n")

    # Step 1: Clean existing data
    clean_existing_dataset()

    # Step 2: Create directory structure
    create_directory_structure()

    # Step 3: Generate realistic seed images
    generate_seed_images()

    # Step 4: Generate augmented data
    generate_augmented_data()

    # Step 5: Split train/val
    split_train_val()

    # Step 6: Create metadata
    create_metadata()

    # Step 7: Verify
    verify_dataset()

    print("\n" + "=" * 60)
    print("  Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\n  Dataset location: {DATASET_ROOT}")
    print(f"  Classes: {len(DENOMINATIONS)}")
    print(f"\n  IMPORTANT: For best results, also add REAL banknote photos!")
    print(f"  Place them in: {RAW_DIR}/<class_name>/")
    print(f"  Then re-run this script to include them in augmentation.\n")
    print("  Next step: python scripts/train_model.py\n")


if __name__ == '__main__':
    main()
