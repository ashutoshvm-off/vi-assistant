"""
Utility helpers: stability detection, speech gating, ROI geometry, drawing.
"""

import time
from typing import Optional, Tuple

import cv2

from config import (
    ROI_HEIGHT,
    ROI_WIDTH,
    STABLE_TIME,
    STABILITY_THRESHOLD,
)

# Type aliases for readability
Point  = Tuple[int, int]
Coords = Tuple[int, int, int, int]   # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Stability detection
# ---------------------------------------------------------------------------

class StabilityChecker:
    """
    Returns True once the tracked position has stayed within *threshold*
    pixels for at least *stable_time* seconds.
    """

    def __init__(
        self,
        threshold: int   = STABILITY_THRESHOLD,
        stable_time: float = STABLE_TIME,
    ) -> None:
        self.threshold   = threshold
        self.stable_time = stable_time
        self._last_pos: Optional[Point]  = None
        self._stable_since: Optional[float] = None

    def update(self, position: Point) -> bool:
        """
        Feed the latest position.
        Returns True if the position has been stable long enough.
        """
        now = time.time()

        if self._last_pos is not None:
            moved = (
                abs(position[0] - self._last_pos[0]) >= self.threshold
                or abs(position[1] - self._last_pos[1]) >= self.threshold
            )
            if moved:
                self._stable_since = None
            elif self._stable_since is None:
                self._stable_since = now
        else:
            self._stable_since = None

        self._last_pos = position
        return (
            self._stable_since is not None
            and (now - self._stable_since) >= self.stable_time
        )

    @property
    def is_moving(self) -> bool:
        """True when the position has left its last stable spot."""
        return self._stable_since is None

    def reset(self) -> None:
        self._last_pos     = None
        self._stable_since = None


# ---------------------------------------------------------------------------
# Speech gating
# ---------------------------------------------------------------------------

class SpeechCooldown:
    """
    Speech gating that matches the desired behaviour:

        Finger appears  → speak immediately (OCR result arrives)
        Finger held     → repeat speech every *cooldown_seconds*
        Finger removed  → reset; silence until next placement
    """

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown  = cooldown_seconds
        self._last_time = 0.0          # epoch of most-recent speak

    def can_speak(self, text: str) -> bool:
        """True when enough time has passed since the last TTS play."""
        return (time.time() - self._last_time) >= self._cooldown

    def mark_spoken(self, text: str) -> None:
        """Record that TTS just fired; starts the cooldown clock."""
        self._last_time = time.time()

    def reset(self) -> None:
        """Call when the finger is lifted to silence immediately."""
        self._last_time = 0.0


# ---------------------------------------------------------------------------
# ROI geometry
# ---------------------------------------------------------------------------

def get_roi_above_finger(
    finger_pos: Point,
    frame_shape: tuple,
) -> Optional[Coords]:
    """
    Return (x1, y1, x2, y2) for the ROI rectangle directly above the
    finger tip, clamped to the frame boundary.
    Returns None when the resulting ROI would have zero area.
    """
    x, y   = finger_pos
    h, w   = frame_shape[:2]

    x1 = max(0, x - ROI_WIDTH  // 2)
    y1 = max(0, y - ROI_HEIGHT)
    x2 = min(w, x + ROI_WIDTH  // 2)
    y2 = min(h, y)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_roi_rectangle(
    frame,
    roi_coords: Coords,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_text(
    frame,
    text: str,
    position: Point = (30, 40),
    color: tuple    = (0, 255, 0),
    font_scale: float = 1.0,
    thickness: int    = 2,
) -> None:
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
    )


def draw_status(frame, status: str, color: tuple = (0, 255, 255)) -> None:
    """Render *status* text in the bottom-left corner of *frame*."""
    h = frame.shape[0]
    cv2.putText(
        frame, status, (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )