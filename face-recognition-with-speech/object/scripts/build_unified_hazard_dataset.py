#!/usr/bin/env python3
"""Build a unified YOLO dataset focused on mobility hazards.

Current source support:
- COCO128 (auto-downloaded bootstrap dataset)

Output:
- object/dataset_unified with YOLO train/val splits + data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# COCO (YOLO 0..79) -> unified hazard class names.
COCO_TO_HAZARD = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    56: "chair",
    57: "sofa",
    58: "pottedplant",
    60: "diningtable",
}

CLASS_NAMES = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "train",
    "chair",
    "sofa",
    "pottedplant",
    "diningtable",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified hazard dataset in YOLO format")
    p.add_argument("--coco-root", type=str, default="object/datasets/coco128")
    p.add_argument("--output", type=str, default="object/dataset_unified")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear-output", action="store_true")
    return p.parse_args()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_label_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _remap_label_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            old_cls = int(parts[0])
            bbox = parts[1:]
        except ValueError:
            continue

        mapped_name = COCO_TO_HAZARD.get(old_cls)
        if mapped_name is None:
            continue
        new_cls = CLASS_TO_INDEX[mapped_name]
        out.append(" ".join([str(new_cls), *bbox]))
    return out


def _collect_coco_pairs(coco_root: Path) -> list[tuple[Path, Path]]:
    images_dir = coco_root / "images" / "train2017"
    labels_dir = coco_root / "labels" / "train2017"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"COCO dataset folders not found under {coco_root}. Run bootstrap script first."
        )

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _write_data_yaml(out_root: Path) -> None:
    data_yaml = out_root / "data.yaml"
    names_blob = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    text = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names_blob}\n"
    )
    data_yaml.write_text(text, encoding="utf-8")


def _copy_sample(image_path: Path, label_lines: list[str], split: str, out_root: Path) -> bool:
    if not label_lines:
        return False

    out_img = out_root / "images" / split / image_path.name
    out_lbl = out_root / "labels" / split / f"{image_path.stem}.txt"

    shutil.copy2(image_path, out_img)
    out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    coco_root = Path(args.coco_root)
    out_root = Path(args.output)

    if args.clear_output and out_root.exists():
        shutil.rmtree(out_root)

    _safe_mkdir(out_root / "images" / "train")
    _safe_mkdir(out_root / "images" / "val")
    _safe_mkdir(out_root / "labels" / "train")
    _safe_mkdir(out_root / "labels" / "val")

    pairs = _collect_coco_pairs(coco_root)
    random.shuffle(pairs)

    val_count = int(len(pairs) * max(0.05, min(0.5, args.val_ratio)))
    val_set = set(pairs[:val_count])

    kept_train = 0
    kept_val = 0
    for image_path, label_path in pairs:
        lines = _read_label_file(label_path)
        mapped = _remap_label_lines(lines)
        split = "val" if (image_path, label_path) in val_set else "train"
        kept = _copy_sample(image_path, mapped, split, out_root)
        if not kept:
            continue
        if split == "train":
            kept_train += 1
        else:
            kept_val += 1

    _write_data_yaml(out_root)

    print(f"Unified dataset ready at: {out_root}")
    print(f"Train samples: {kept_train}")
    print(f"Val samples: {kept_val}")
    print(f"Classes: {CLASS_NAMES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
