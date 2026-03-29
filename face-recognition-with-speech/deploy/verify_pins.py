#!/usr/bin/env python3
"""
Smart Vision — Hardware Pin Verifier for CM4.

Reads pins.json and tests that each device is reachable:
  • VL53L0X (I2C)  — scan I2C bus for addr 0x29
  • INMP441 (I2S)  — check ALSA audio capture devices
  • ArduCam  (CSI) — test camera with OpenCV

Usage:
  python deploy/verify_pins.py                # standard check
  python deploy/verify_pins.py --verbose      # detailed output
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS_FILE = ROOT / "pins.json"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

verbose = "--verbose" in sys.argv


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def load_pins() -> dict:
    if not PINS_FILE.exists():
        fail(f"pins.json not found at {PINS_FILE}")
        return {}
    with open(PINS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── VL53L0X (I2C) ──────────────────────────────────────────────────────────
def check_vl53l0x(pins: dict) -> bool:
    """Check VL53L0X is visible on I2C bus (default addr 0x29)."""
    vl = pins.get("vl53l0x", {})
    sda = vl.get("sda")
    scl = vl.get("scl")

    print(f"\n[VL53L0X] I2C LiDAR — SDA=GPIO{sda}, SCL=GPIO{scl}")

    # Try i2cdetect (standard Raspberry Pi tool)
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", "1"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()
        if verbose:
            print(result.stdout)

        # VL53L0X default I2C address is 0x29
        if "29" in output:
            ok("VL53L0X found at address 0x29 on I2C bus 1")
            return True
        else:
            fail("VL53L0X NOT found at address 0x29")
            warn("Check wiring: SDA→GPIO2, SCL→GPIO3, VCC→3.3V, GND→GND")
            return False
    except FileNotFoundError:
        warn("i2cdetect not installed — run: sudo apt install i2c-tools")
        return False
    except subprocess.TimeoutExpired:
        fail("i2cdetect timed out")
        return False
    except Exception as exc:
        fail(f"I2C check error: {exc}")
        return False


# ── INMP441 (I2S) ──────────────────────────────────────────────────────────
def check_inmp441(pins: dict) -> bool:
    """Check INMP441 I2S microphone is visible as an ALSA capture device."""
    mic = pins.get("inmp441", {})
    sd = mic.get("sd")
    sck = mic.get("sck")
    ws = mic.get("ws")
    lr = mic.get("lr")

    print(f"\n[INMP441] I2S Mic — SD=GPIO{sd}, SCK=GPIO{sck}, WS=GPIO{ws}, LR={lr}")

    # Check if any audio capture device is present
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
        if verbose:
            print(output)

        if "card" in output.lower():
            ok("Audio capture device(s) found:")
            # Print each card line
            for line in output.splitlines():
                if "card" in line.lower():
                    print(f"    {line.strip()}")
            return True
        else:
            fail("No audio capture devices found")
            warn("Check I2S overlay: dtoverlay=i2s-mmap in /boot/config.txt")
            warn("Check wiring: SD→GPIO20, SCK→GPIO18, WS→GPIO19, LR→GND")
            return False
    except FileNotFoundError:
        warn("arecord not installed — run: sudo apt install alsa-utils")
        return False
    except Exception as exc:
        fail(f"Audio check error: {exc}")
        return False


# ── ArduCam (CSI) ──────────────────────────────────────────────────────────
def check_arducam(pins: dict) -> bool:
    """Check CSI camera is accessible."""
    cam = pins.get("arducam", {})
    cam_type = cam.get("type", "CSI")
    cam_id = cam.get("camera_id", 0)

    print(f"\n[ArduCam] {cam_type} Camera — camera_id={cam_id}")

    # Method 1: Try libcamera-hello (modern Raspberry Pi OS)
    try:
        result = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout + result.stderr
        if verbose:
            print(output)

        if "available cameras" in output.lower() and "no cameras" not in output.lower():
            ok("CSI camera detected via libcamera")
            for line in output.splitlines():
                stripped = line.strip()
                if stripped and (":" in stripped or "camera" in stripped.lower()):
                    print(f"    {stripped}")
            return True
    except FileNotFoundError:
        pass  # libcamera not available, try OpenCV
    except Exception:
        pass

    # Method 2: Try OpenCV VideoCapture
    try:
        import cv2

        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                ok(f"Camera opened successfully on index {cam_id} via OpenCV")
                return True
            else:
                fail(f"Camera opened but returned no frames (index {cam_id})")
                return False
        else:
            cap.release()
            fail(f"Could not open camera on index {cam_id}")
            warn("Check CSI cable, camera_auto_detect=1 in /boot/config.txt")
            return False
    except ImportError:
        warn("OpenCV not installed — cannot verify camera")
        return False
    except Exception as exc:
        fail(f"Camera check error: {exc}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 60)
    print("  Smart Vision — Hardware Pin Verification")
    print("=" * 60)

    pins = load_pins()
    if not pins:
        return 1

    print(f"\n  Loaded pins.json: {list(pins.keys())}")

    results = {}

    if "vl53l0x" in pins:
        results["VL53L0X (LiDAR)"] = check_vl53l0x(pins)

    if "inmp441" in pins:
        results["INMP441 (Mic)"] = check_inmp441(pins)

    if "arducam" in pins:
        results["ArduCam (Camera)"] = check_arducam(pins)

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_ok = True
    for device, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {device:.<40} {status}")
        if not passed:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print(f"\n  {GREEN}All hardware checks passed!{RESET}\n")
        return 0
    else:
        print(f"\n  {YELLOW}Some checks failed — see details above.{RESET}")
        print(f"  The system will still start; failed sensors use fallback mode.\n")
        return 0  # non-fatal — system runs in fallback mode for missing sensors


if __name__ == "__main__":
    raise SystemExit(main())
