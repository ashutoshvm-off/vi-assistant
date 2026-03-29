"""
Shared camera utility for Smart Vision.

Provides a unified camera interface that works on both desktop (OpenCV)
and Raspberry Pi (rpicam-vid subprocess, since libcamera Python bindings
are not available in pyenv-based venvs).

Usage:
    from camera_utils import open_camera
    cap = open_camera(index=0, width=640, height=480, fps=30)
    ret, frame = cap.read()
    cap.release()
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def _is_raspberry_pi() -> bool:
    """Detect if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def _rpicam_available() -> bool:
    """Check if rpicam-vid CLI is available."""
    return shutil.which("rpicam-vid") is not None


class PiCameraCapture:
    """
    OpenCV-compatible camera capture using rpicam-vid subprocess.

    Streams MJPEG from rpicam-vid and decodes frames with OpenCV.
    Provides the same interface as cv2.VideoCapture:
        - isOpened()
        - read() -> (bool, frame)
        - release()
        - set(prop, val)  (no-op, resolution set at init)
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._proc: Optional[subprocess.Popen] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        self._start()

    def _start(self) -> None:
        """Start rpicam-vid subprocess and frame reader thread."""
        cmd = [
            "rpicam-vid",
            "-t", "0",                  # run indefinitely
            "--width", str(self._width),
            "--height", str(self._height),
            "--framerate", str(self._fps),
            "--codec", "mjpeg",
            "--inline",
            "-n",                       # no preview window
            "-o", "-",                  # output to stdout
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_frames, daemon=True
            )
            self._reader_thread.start()
            # Give the camera a moment to warm up
            time.sleep(1.0)
        except Exception as exc:
            print(f"[camera_utils] rpicam-vid failed to start: {exc}")
            self._running = False

    def _read_frames(self) -> None:
        """Continuously read MJPEG frames from rpicam-vid stdout."""
        buf = bytearray()
        SOI = b"\xff\xd8"  # JPEG Start Of Image
        EOI = b"\xff\xd9"  # JPEG End Of Image

        while self._running and self._proc and self._proc.poll() is None:
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            buf.extend(chunk)

            # Extract complete JPEG frames
            while True:
                soi_pos = buf.find(SOI)
                if soi_pos == -1:
                    break
                eoi_pos = buf.find(EOI, soi_pos + 2)
                if eoi_pos == -1:
                    break

                # Extract the JPEG frame
                jpeg_data = bytes(buf[soi_pos : eoi_pos + 2])
                buf = buf[eoi_pos + 2 :]

                # Decode to numpy array
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame

    def isOpened(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def set(self, prop: int, val: float) -> bool:
        # No-op — resolution is set at init via rpicam-vid args
        return True

    def release(self) -> None:
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


def open_camera(
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Optional[object]:
    """
    Open a camera, auto-selecting the best backend.

    On Raspberry Pi with rpicam-vid available:
        Uses PiCameraCapture (rpicam-vid subprocess).
    On desktop / other:
        Uses cv2.VideoCapture with the given index.

    Returns an object with .isOpened(), .read(), .release(), .set() methods,
    compatible with cv2.VideoCapture interface.
    Returns None if no camera could be opened.
    """
    # Try rpicam-vid on Raspberry Pi
    if _is_raspberry_pi() and _rpicam_available():
        print(f"[camera_utils] Using rpicam-vid (width={width}, height={height}, fps={fps})")
        cap = PiCameraCapture(width=width, height=height, fps=fps)
        if cap.isOpened():
            return cap
        cap.release()
        print("[camera_utils] rpicam-vid failed, falling back to OpenCV")

    # Fall back to OpenCV VideoCapture
    for idx in dict.fromkeys([index, 0, 1, 2]):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[camera_utils] Using OpenCV VideoCapture index={idx}")
                return cap
        cap.release()

    return None
