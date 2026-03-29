"""
Shared camera utility for Smart Vision.

On Raspberry Pi: reads pre-captured frames from /dev/shm/smartvision_frame.jpg
written by camera_server.py (must be started before modules).
On desktop: uses cv2.VideoCapture directly.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Headless display patching ─────────────────────────────────────────────────
# On headless Pi (no DISPLAY), monkey-patch cv2.imshow/waitKey/destroyAllWindows
# so modules that call them don't crash.
if not os.environ.get("DISPLAY") and os.path.exists("/proc/device-tree/model"):
    cv2.imshow = lambda *a, **kw: None  # type: ignore[assignment]
    cv2.destroyAllWindows = lambda *a, **kw: None  # type: ignore[assignment]
    _orig_waitKey = getattr(cv2, "waitKey", None)
    cv2.waitKey = lambda delay=0: -1  # type: ignore[assignment]
    print("[camera_utils] Headless mode — cv2.imshow/waitKey patched.")

SHARED_FRAME_PATH = "/dev/shm/smartvision_frame.jpg"
PID_FILE = "/dev/shm/smartvision_camera.pid"


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def _camera_server_running() -> bool:
    """Check if camera_server.py is running."""
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


class SharedFrameReader:
    """
    Reads frames from /dev/shm/smartvision_frame.jpg written by camera_server.py.
    Same interface as cv2.VideoCapture.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self._width = width
        self._height = height
        self._running = False

        # Wait for camera server to start producing frames
        print("[camera_utils] Waiting for camera_server frames...")
        for i in range(100):  # 10 seconds max
            if os.path.exists(SHARED_FRAME_PATH) and os.path.getsize(SHARED_FRAME_PATH) > 0:
                self._running = True
                print("[camera_utils] Connected to shared camera.")
                break
            time.sleep(0.1)
        else:
            print("[camera_utils] Timeout waiting for camera_server frames.")

    def isOpened(self) -> bool:
        return self._running and _camera_server_running()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = cv2.imread(SHARED_FRAME_PATH)
            if frame is not None and frame.size > 0:
                return True, frame
        except Exception:
            pass
        return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def set(self, prop: int, val: float) -> bool:
        return True

    def release(self) -> None:
        self._running = False


def open_camera(
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Optional[object]:
    """
    Open a camera, auto-selecting the best backend.

    On Raspberry Pi: reads from shared frame file (/dev/shm/).
    On desktop: uses cv2.VideoCapture.
    Returns None if no camera could be opened.
    """
    if _is_raspberry_pi() and _camera_server_running():
        cap = SharedFrameReader(width=width, height=height)
        if cap.isOpened():
            return cap
        cap.release()

    # Fall back to OpenCV
    for idx in dict.fromkeys([index, 0, 1, 2]):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[camera_utils] Using OpenCV index={idx}")
                return cap
        cap.release()

    return None
