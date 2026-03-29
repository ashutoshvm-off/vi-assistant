"""OpenCV MobileNet-SSD object detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import CLASS_NAMES, HAZARD_LABELS


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def is_hazard(self) -> bool:
        return self.label in HAZARD_LABELS


class ObjectDetector:
    def __init__(
        self,
        prototxt_path: Optional[Path],
        model_path: Optional[Path],
        confidence_threshold: float = 0.5,
        backend: str = "caffe",
        tflite_model_path: Optional[Path] = None,
        tflite_labels: Optional[List[str]] = None,
    ):
        self.backend = backend
        self.confidence_threshold = float(confidence_threshold)

        self.net = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.tflite_labels = tflite_labels or []
        self.is_float_input = False
        self.tflite_width = 300
        self.tflite_height = 300

        if self.backend == "caffe":
            if prototxt_path is None or model_path is None or not prototxt_path.exists() or not model_path.exists():
                raise FileNotFoundError(
                    "Missing Caffe detector model files. "
                    "Run object/scripts/download_models.py or place model files in object/models/."
                )
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))

        elif self.backend == "tflite":
            if tflite_model_path is None or not tflite_model_path.exists():
                raise FileNotFoundError(
                    "Missing TFLite detector model file. "
                    "Expected a .tflite model in object/models/."
                )

            try:
                import tensorflow as tf
            except Exception as exc:  # pragma: no cover - dependency environment specific
                raise ImportError("TensorFlow is required for TFLite backend") from exc

            self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path), num_threads=2)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            in_shape = self.input_details[0]["shape"]
            self.tflite_height = int(in_shape[1])
            self.tflite_width = int(in_shape[2])
            self.is_float_input = self.input_details[0]["dtype"] == np.float32
        else:
            raise ValueError(f"Unsupported detector backend: {backend}")

    def detect(self, frame) -> List[Detection]:
        if self.backend == "tflite":
            return self._detect_tflite(frame)
        return self._detect_caffe(frame)

    def _detect_caffe(self, frame) -> List[Detection]:
        if self.net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        detections: List[Detection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            class_idx = int(raw[0, 0, i, 1])
            if class_idx < 0 or class_idx >= len(CLASS_NAMES):
                continue

            box = raw[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            detections.append(
                Detection(
                    label=CLASS_NAMES[class_idx],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    def _detect_tflite(self, frame) -> List[Detection]:
        if self.interpreter is None or self.input_details is None or self.output_details is None:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.tflite_width, self.tflite_height))

        if self.is_float_input:
            input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        else:
            input_tensor = np.expand_dims(resized.astype(np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        num = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        detections: List[Detection] = []
        for i in range(num):
            confidence = float(scores[i])
            if confidence < self.confidence_threshold:
                continue

            class_idx = int(classes[i])
            label = self._label_for_tflite_class(class_idx)

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0, min(w - 1, int(xmin * w)))
            y1 = max(0, min(h - 1, int(ymin * h)))
            x2 = max(0, min(w - 1, int(xmax * w)))
            y2 = max(0, min(h - 1, int(ymax * h)))

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    def _label_for_tflite_class(self, class_idx: int) -> str:
        # Some TF models are 1-based class ids; prefer exact match first, then shifted.
        if 0 <= class_idx < len(self.tflite_labels):
            return self.tflite_labels[class_idx]
        shifted = class_idx + 1
        if 0 <= shifted < len(self.tflite_labels):
            return self.tflite_labels[shifted]
        return f"class_{class_idx}"


def draw_detections(frame, detections: Sequence[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (0, 0, 255) if det.is_hazard else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.confidence * 100:.0f}%"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
