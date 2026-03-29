#!/usr/bin/env python3
"""
Shared camera server for Smart Vision on Raspberry Pi.

Starts rpicam-vid and continuously writes the latest JPEG frame
to /dev/shm/smartvision_frame.jpg so all modules can read it.

Run this BEFORE starting the main coordinator.
"""

import os
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

SHARED_FRAME_PATH = "/dev/shm/smartvision_frame.jpg"
PID_FILE = "/dev/shm/smartvision_camera.pid"


def cleanup(*_):
    for f in [SHARED_FRAME_PATH, SHARED_FRAME_PATH + ".tmp", PID_FILE]:
        try:
            os.unlink(f)
        except OSError:
            pass
    sys.exit(0)


def main():
    width = int(os.getenv("CAMERA_WIDTH", "640"))
    height = int(os.getenv("CAMERA_HEIGHT", "480"))
    fps = int(os.getenv("CAMERA_FPS", "30"))

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    print(f"[camera_server] Starting rpicam-vid ({width}x{height}@{fps})")

    proc = subprocess.Popen(
        [
            "rpicam-vid", "-t", "0",
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--codec", "mjpeg",
            "--inline", "-n",
            "-o", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    buf = bytearray()
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    tmp_path = SHARED_FRAME_PATH + ".tmp"
    frame_count = 0

    try:
        while proc.poll() is None:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf.extend(chunk)

            while True:
                soi_pos = buf.find(SOI)
                if soi_pos == -1:
                    break
                eoi_pos = buf.find(EOI, soi_pos + 2)
                if eoi_pos == -1:
                    break

                jpeg_data = bytes(buf[soi_pos : eoi_pos + 2])
                buf = buf[eoi_pos + 2 :]

                try:
                    with open(tmp_path, "wb") as f:
                        f.write(jpeg_data)
                    os.replace(tmp_path, SHARED_FRAME_PATH)
                    frame_count += 1
                    if frame_count == 1:
                        print(f"[camera_server] First frame written to {SHARED_FRAME_PATH}")
                except OSError as e:
                    print(f"[camera_server] Write error: {e}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        cleanup()


if __name__ == "__main__":
    main()
