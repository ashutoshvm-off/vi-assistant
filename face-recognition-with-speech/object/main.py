#!/usr/bin/env python3
"""Real-time object detection + VL53L0X-based collision avoidance."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import cv2

from collision_avoidance import CollisionAvoidanceEngine
from config import (
    ALERT_COOLDOWN_SECONDS,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CRITICAL_CM,
    DEFAULT_FPS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MODEL,
    DEFAULT_PROTOTXT,
    DEFAULT_TFLITE_LABELS,
    DEFAULT_TFLITE_MODEL,
    DEFAULT_WARNING_CM,
)
from object_detector import ObjectDetector, draw_detections
from tof_sensor import ToFSensor

try:
    import pyttsx3  # type: ignore[import-not-found]
except ImportError:
    pyttsx3 = None


class LocalSpeaker:
    def __init__(self, enabled: bool):
        self._engine = None
        self._enabled = enabled and pyttsx3 is not None and os.getenv("COORD_DISABLE_LOCAL_TTS", "0") != "1"
        if self._enabled:
            try:
                assert pyttsx3 is not None
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 165)
            except Exception as exc:
                print(f"[OBJECT] local TTS unavailable: {exc}")
                self._engine = None
                self._enabled = False

    def speak(self, text: str) -> None:
        if not self._enabled or self._engine is None:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass


def open_camera(index: int):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from camera_utils import open_camera as _shared_open_camera

    width = int(os.getenv("OBJECT_CAMERA_WIDTH", str(DEFAULT_FRAME_WIDTH)))
    height = int(os.getenv("OBJECT_CAMERA_HEIGHT", str(DEFAULT_FRAME_HEIGHT)))
    fps = int(os.getenv("OBJECT_CAMERA_FPS", str(DEFAULT_FPS)))
    return _shared_open_camera(index=index, width=width, height=height, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object detection and collision alerts")
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("OBJECT_CAMERA_INDEX", DEFAULT_CAMERA_INDEX)))
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument(
        "--detector-backend",
        choices=["tflite", "caffe"],
        default=os.getenv("OBJECT_DETECTOR_BACKEND", "tflite"),
        help="Object detector backend.",
    )
    parser.add_argument("--warning-cm", type=float, default=DEFAULT_WARNING_CM)
    parser.add_argument("--critical-cm", type=float, default=DEFAULT_CRITICAL_CM)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--simulate-tof", action="store_true")
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=int(os.getenv("OBJECT_DETECT_INTERVAL", "2")),
        help="Run object detection every N frames (>=1).",
    )
    parser.add_argument(
        "--lidar-only",
        action="store_true",
        help="Disable camera object model and use LiDAR-only collision alerts.",
    )
    parser.add_argument(
        "--lidar-type",
        choices=["auto", "vl53l0x", "uart", "tfluna", "tfmini"],
        default=os.getenv("OBJECT_LIDAR_TYPE", "auto"),
        help="Sensor backend to use.",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default=os.getenv("OBJECT_LIDAR_PORT", "/dev/ttyS0"),
        help="UART LiDAR serial port (for lidar-type uart/tfluna/tfmini).",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=int(os.getenv("OBJECT_LIDAR_BAUD", "115200")),
        help="UART LiDAR baud rate.",
    )
    parser.add_argument("--prototxt", type=str, default=str(DEFAULT_PROTOTXT))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--tflite-model", type=str, default=str(DEFAULT_TFLITE_MODEL))
    parser.add_argument("--tflite-labels", type=str, default=str(DEFAULT_TFLITE_LABELS))
    return parser.parse_args()


def _load_tflite_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        labels.append(cleaned)
    return labels


def main() -> int:
    args = parse_args()

    detector = None
    if not args.lidar_only:
        try:
            tflite_labels = _load_tflite_labels(Path(args.tflite_labels))
            detector = ObjectDetector(
                prototxt_path=Path(args.prototxt),
                model_path=Path(args.model),
                confidence_threshold=args.confidence,
                backend=args.detector_backend,
                tflite_model_path=Path(args.tflite_model),
                tflite_labels=tflite_labels,
            )
        except Exception as exc:
            print(f"[OBJECT] detector init failed: {exc}")
            return 1

    sensor = ToFSensor(
        simulate=args.simulate_tof,
        sensor_type=args.lidar_type,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
    )
    engine = CollisionAvoidanceEngine(warning_cm=args.warning_cm, critical_cm=args.critical_cm)
    speaker = LocalSpeaker(enabled=not args.no_speech)

    cap = open_camera(args.camera_index)
    if cap is None:
        print("[OBJECT] camera not available")
        return 1

    print("[OBJECT] Started. Press q to quit.")
    if not sensor.is_ready:
        print("[OBJECT] Running with camera-only fallback (no TOF distance).")
    if args.lidar_only:
        print("[OBJECT] LiDAR-only mode enabled. Camera model inference is disabled.")

    last_alert_text: Optional[str] = None
    last_alert_time = 0.0
    frame_index = 0
    detect_interval = max(1, int(args.detect_interval))
    cached_detections = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_index += 1
            if detector is None:
                detections = []
            elif frame_index % detect_interval == 0:
                cached_detections = detector.detect(frame)
                detections = cached_detections
            else:
                detections = cached_detections

            distance_cm = sensor.read_distance_cm()
            decision = engine.evaluate(distance_cm, detections, frame_width=frame.shape[1])

            now = time.time()
            if decision.message:
                should_emit = (
                    decision.message != last_alert_text
                    or (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS
                )
                if should_emit:
                    print(f"COORD_EVENT|source=hazard|type=collision|priority=0|text={decision.message}")
                    speaker.speak(decision.message)
                    last_alert_text = decision.message
                    last_alert_time = now

            if not args.no_display:
                draw_detections(frame, detections)
                distance_text = "TOF: n/a" if distance_cm is None else f"TOF: {distance_cm:.0f} cm"
                cv2.putText(frame, distance_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Risk: {decision.level}",
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if decision.level in {"warning", "critical"} else (0, 255, 0),
                    2,
                )
                cv2.imshow("Object Collision Avoidance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.02)

    finally:
        cap.release()
        sensor.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
