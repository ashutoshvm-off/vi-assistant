#!/usr/bin/env python3
"""Download and organize object detection datasets for road + home scenes.

What this script does automatically:
- Downloads COCO128 sample dataset (quick baseline) using Ultralytics asset URL.
- Creates folder structure for additional datasets.

What still requires manual download (license/gated terms):
- IDD (Indian Driving Dataset)
- BDD100K
- Mapillary Vistas

After manual downloads, place files under object/datasets/raw and run your own
conversion pipeline to YOLO format.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RAW = DATASETS / "raw"
COCO128_ZIP = DATASETS / "coco128.zip"
COCO128_DIR = DATASETS / "coco128"

COCO128_URL = "https://ultralytics.com/assets/coco128.zip"


def _mkdirs() -> None:
    for p in [DATASETS, RAW, RAW / "idd", RAW / "bdd100k", RAW / "mapillary", RAW / "indoor"]:
        p.mkdir(parents=True, exist_ok=True)


def _download(url: str, out: Path) -> None:
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, out)
    print(f"[ok] {out}")


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> int:
    _mkdirs()

    if not COCO128_DIR.exists():
        _download(COCO128_URL, COCO128_ZIP)
        _extract(COCO128_ZIP, DATASETS)
    else:
        print("[skip] coco128 already present")

    # Keep workspace smaller after extraction.
    if COCO128_ZIP.exists():
        try:
            COCO128_ZIP.unlink()
        except OSError:
            pass

    print("\nDataset bootstrap complete.")
    print("Next: download IDD/BDD100K/Mapillary manually and place archives in object/datasets/raw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
