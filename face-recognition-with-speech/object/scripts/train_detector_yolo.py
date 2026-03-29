#!/usr/bin/env python3
"""Train a high-accuracy YOLO detector for object module datasets.

Example:
  python object/scripts/train_detector_yolo.py \
    --data object/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 120 \
    --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO detector for Smart Vision object module")
    p.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, or CUDA index")
    p.add_argument("--project", type=str, default="object/training_runs")
    p.add_argument("--name", type=str, default="cm4_best")
    p.add_argument("--export-tflite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset config not found: {data_path}")

    try:
        import torch
        if args.device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
    except Exception:
        device = "cpu" if args.device == "auto" else args.device

    print(f"Using training device: {device}")

    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=30,
        workers=2,
        cache=True,
        cos_lr=True,
        plots=True,
    )

    best_path = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best checkpoint: {best_path}")

    if args.export_tflite:
        print("Exporting best model to TFLite (int8 if supported)...")
        YOLO(str(best_path)).export(format="tflite", int8=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
