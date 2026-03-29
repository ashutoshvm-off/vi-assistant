#!/usr/bin/env python3
"""Download MobileNet-SSD Caffe model files used by object/main.py."""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "MobileNetSSD_deploy.prototxt": [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt",
    ],
    "MobileNetSSD_deploy.caffemodel": [
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel",
    ],
}


def _download_with_fallback(name: str, urls: list[str], out: Path) -> None:
    last_error = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, out)
            return
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            print(f"[warn] {name} failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {name}: {last_error}")


def main() -> int:
    for name, urls in FILES.items():
        out = MODELS_DIR / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {name} already exists")
            continue

        print(f"[download] {name}")
        _download_with_fallback(name, urls, out)
        print(f"[ok] saved to {out}")

    print("Model download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
