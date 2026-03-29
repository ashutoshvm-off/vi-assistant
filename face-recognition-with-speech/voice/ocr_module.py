"""
OCR module — OCR.space API back-end.
Free tier: 25,000 requests/month.
Get a key at: https://ocr.space/ocrapi/freekey
"""

import base64
import re
import threading
from typing import Optional, Tuple

import cv2
import requests

from config import (
    MIN_TEXT_LENGTH,
    OCR_ENGINE,
    OCR_JPEG_QUALITY,
    OCR_LANGUAGE,
    OCR_SCALE_FACTOR,
    OCR_SPACE_API_KEY,
    OCR_SPACE_API_URL,
    OCR_TIMEOUT,
)


class OCRSpaceOCR:
    """
    Thin wrapper around the OCR.space REST API.
    Supports both synchronous and fire-and-forget async calls.
    """

    def __init__(self) -> None:
        self._api_key = OCR_SPACE_API_KEY
        self._lock    = threading.Lock()
        self._result  = ""
        self._processing = False

        print(f"[ocr] OCR.space — engine {OCR_ENGINE}, lang={OCR_LANGUAGE}")
        if self._api_key == "helloworld":
            print("[ocr] ⚠️  Demo key active. "
                  "Get a free key at: https://ocr.space/ocrapi/freekey")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_base64(image) -> str:
        """Encode an OpenCV BGR image as a base64 JPEG string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, OCR_JPEG_QUALITY]
        _, buf = cv2.imencode(".jpg", image, encode_params)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def preprocess(roi):
        """
        Scale the ROI up for better OCR accuracy.
        Returns (gray_display_image, colour_api_image).
        """
        kwargs = dict(fx=OCR_SCALE_FACTOR, fy=OCR_SCALE_FACTOR,
                      interpolation=cv2.INTER_CUBIC)
        gray  = cv2.cvtColor(
            cv2.resize(roi, None, **kwargs), cv2.COLOR_BGR2GRAY
        )
        color = cv2.resize(roi, None, **kwargs)
        return gray, color

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_api(self, image) -> str:
        """
        POST *image* to OCR.space and return the raw extracted text.
        Returns an empty string on any error.
        """
        b64 = self._to_base64(image)
        payload = {
            "apikey":            self._api_key,
            "base64Image":       f"data:image/jpeg;base64,{b64}",
            "language":          OCR_LANGUAGE,
            "OCREngine":         OCR_ENGINE,
            "isOverlayRequired": False,
            "detectOrientation": True,
            "scale":             True,
            "isTable":           False,
        }
        try:
            resp = requests.post(
                OCR_SPACE_API_URL, data=payload, timeout=OCR_TIMEOUT
            )
        except requests.exceptions.Timeout:
            print("[ocr] Request timed out.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[ocr] Network error: {e}")
            return ""

        if resp.status_code != 200:
            print(f"[ocr] HTTP {resp.status_code}: {resp.text[:200]}")
            return ""

        data = resp.json()
        if data.get("IsErroredOnProcessing"):
            errors = data.get("ErrorMessage") or ["Unknown error"]
            print(f"[ocr] API error: {', '.join(errors)}")
            return ""

        parts = data.get("ParsedResults") or []
        return " ".join(
            r["ParsedText"].strip() for r in parts if r.get("ParsedText")
        ).strip()

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(raw: str) -> str:
        """Normalise whitespace and strip non-printable characters."""
        if not raw:
            return ""
        text = re.sub(r"[\r\n]+", " ", raw)
        text = re.sub(r"\s+",    " ", text).strip()
        text = re.sub(r"[^\x20-\x7E]", "", text)
        return text if len(text) >= MIN_TEXT_LENGTH else ""

    # ------------------------------------------------------------------
    # Async interface (used by AsyncOCRProcessor)
    # ------------------------------------------------------------------

    def submit_async(self, image) -> bool:
        """
        Start OCR in a background thread.
        Returns False immediately if a request is already in flight.
        """
        with self._lock:
            if self._processing:
                return False
            self._processing = True
            self._result = ""       # clear stale result

        def _run():
            try:
                raw = self._call_api(image)
                with self._lock:
                    self._result = self.clean_text(raw)
            finally:
                with self._lock:
                    self._processing = False

        threading.Thread(target=_run, daemon=True).start()
        return True

    def get_result(self) -> str:
        with self._lock:
            return self._result

    def is_processing(self) -> bool:
        with self._lock:
            return self._processing


# ---------------------------------------------------------------------------
# High-level non-blocking wrapper
# ---------------------------------------------------------------------------

class AsyncOCRProcessor:
    """
    Non-blocking OCR facade used by the main loop.

    Usage:
        submitted = processor.submit(roi)
        display, text, busy = processor.get_result()
    """

    def __init__(self) -> None:
        self._ocr: OCRSpaceOCR = OCRSpaceOCR()
        self._display_image    = None

    def submit(self, roi) -> bool:
        """
        Pre-process *roi* and kick off an async OCR request.
        The ROI is un-mirrored here because main.py flips the frame for
        display purposes; OCR needs the original (readable) orientation.
        Returns False when a request is already in flight.
        """
        if self._ocr.is_processing():
            return False

        roi_corrected = cv2.flip(roi, 1)   # restore original orientation
        gray, color   = self._ocr.preprocess(roi_corrected)
        self._display_image = gray

        return self._ocr.submit_async(color)

    def get_result(self) -> Tuple[Optional[object], str, bool]:
        """
        Returns (display_image, cleaned_text, is_processing).
        *display_image* is None until the first submission.
        """
        return (
            self._display_image,
            self._ocr.get_result(),
            self._ocr.is_processing(),
        )