"""Collision risk assessment using TOF distance + detected objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from object_detector import Detection


@dataclass
class CollisionDecision:
    level: str
    message: Optional[str]


class CollisionAvoidanceEngine:
    def __init__(self, warning_cm: float, critical_cm: float):
        self.warning_cm = float(warning_cm)
        self.critical_cm = float(critical_cm)

    def evaluate(
        self,
        distance_cm: Optional[float],
        detections: Iterable[Detection],
        frame_width: int,
    ) -> CollisionDecision:
        hazard = self._nearest_hazard(detections)

        if distance_cm is None:
            if hazard is None:
                return CollisionDecision(level="clear", message=None)
            return CollisionDecision(level="warning", message=f"Caution: {hazard.label} ahead")

        if distance_cm <= self.critical_cm:
            if hazard is not None:
                side = self._relative_side(hazard, frame_width)
                return CollisionDecision(
                    level="critical",
                    message=f"Stop. {hazard.label} very close on {side}",
                )
            return CollisionDecision(level="critical", message="Stop. Obstacle very close")

        if distance_cm <= self.warning_cm:
            if hazard is not None:
                side = self._relative_side(hazard, frame_width)
                return CollisionDecision(
                    level="warning",
                    message=f"Warning. {hazard.label} ahead on {side}",
                )
            return CollisionDecision(level="warning", message="Obstacle ahead")

        if hazard is not None:
            return CollisionDecision(level="info", message=f"{hazard.label} detected")

        return CollisionDecision(level="clear", message=None)

    @staticmethod
    def _nearest_hazard(detections: Iterable[Detection]) -> Optional[Detection]:
        candidates = [d for d in detections if d.is_hazard]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    @staticmethod
    def _relative_side(det: Detection, frame_width: int) -> str:
        x1, _, x2, _ = det.bbox
        center_x = (x1 + x2) / 2.0
        left_boundary = frame_width * 0.4
        right_boundary = frame_width * 0.6
        if center_x < left_boundary:
            return "left"
        if center_x > right_boundary:
            return "right"
        return "center"
