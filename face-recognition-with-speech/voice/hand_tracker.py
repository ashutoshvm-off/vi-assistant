"""
Hand-tracking module using MediaPipe Hands.
"""

import cv2
import mediapipe as mp

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

# Index-finger tip landmark index (MediaPipe convention).
_INDEX_FINGER_TIP = 8


class HandTracker:
    """Detects hands and exposes landmark data for a single hand."""

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._hands    = self._mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Run MediaPipe hand detection on *frame* (BGR).
        Returns the raw MediaPipe results object.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb)

    def get_index_finger_tip(self, hand_landmarks, frame_width: int, frame_height: int):
        """
        Return the (x, y) pixel position of the index-finger tip.
        """
        tip = hand_landmarks.landmark[_INDEX_FINGER_TIP]
        return int(tip.x * frame_width), int(tip.y * frame_height)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame, hand_landmarks) -> None:
        """Overlay hand skeleton on *frame* in-place."""
        self._mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )

    def draw_finger_point(
        self,
        frame,
        position,
        color: tuple = (255, 0, 255),
        radius: int = 8,
    ) -> None:
        """Draw a filled circle at *position* (x, y)."""
        cv2.circle(frame, position, radius, color, -1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()