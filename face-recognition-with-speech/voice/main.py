"""
Hand OCR Reader — Main Application.

Behaviour
─────────────────────────────────────────────────────
  Finger appears on text   → 1 OCR call → speak result
  Finger still there       → no more OCR  │  repeat speech (cooldown-gated)
  Finger removed           → reset        │  stop everything
  Finger on new text       → 1 OCR call → speak result
─────────────────────────────────────────────────────
Press Q to quit.
"""

import sys
import time
import os
from pathlib import Path

import cv2

# Add project root to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio import cleanup as audio_cleanup, speak_async
from camera_utils import open_camera as _shared_open_camera
from config import (
    CAMERA_INDEX,
    MIN_TEXT_LENGTH,
    OCR_INTERVAL,
    OCR_SPACE_API_KEY,
    SPEAK_COOLDOWN,
    WINDOW_NAME,
    OCR_DEBUG_WINDOW,
)
from hand_tracker import HandTracker
from ocr_module import AsyncOCRProcessor
from utils import (
    SpeechCooldown,
    StabilityChecker,
    draw_roi_rectangle,
    draw_status,
    draw_text,
    get_roi_above_finger,
)

_MAX_CAMERA_FAILURES = 10   # consecutive bad frames before giving up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_camera(preferred_index: int):
    """Open camera using shared utility (supports rpicam-vid on Pi)."""
    cam_w = int(os.getenv("VOICE_CAMERA_WIDTH", "640"))
    cam_h = int(os.getenv("VOICE_CAMERA_HEIGHT", "480"))
    cam_fps = int(os.getenv("VOICE_CAMERA_FPS", "20"))

    cap = _shared_open_camera(index=preferred_index, width=cam_w, height=cam_h, fps=cam_fps)
    if cap is not None:
        print(f"[main] Camera opened.")
    return cap


def _reset_session(stability: StabilityChecker, cooldown: SpeechCooldown) -> None:
    """Clear all per-placement state when the finger is lifted or moved."""
    stability.reset()
    cooldown.reset()
    print("[main] Finger lifted — ready for next text.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    if OCR_SPACE_API_KEY == "helloworld":
        print("=" * 50)
        print("⚠️  Using OCR.space demo key (limited rate).")
        print("Get your free key (25,000 req/month) at:")
        print("  https://ocr.space/ocrapi/freekey")
        print("Then set:  export OCR_SPACE_API_KEY=<your_key>")
        print("=" * 50)

    print("[main] Initialising Hand OCR Reader…")

    hand_tracker  = HandTracker()
    ocr_processor = AsyncOCRProcessor()
    stability     = StabilityChecker()
    cooldown      = SpeechCooldown(SPEAK_COOLDOWN)

    cap = _open_camera(CAMERA_INDEX)
    if cap is None:
        print("[main] Error: no camera found — exiting.")
        return

    print("\n" + "=" * 50)
    print("Hand OCR Reader started!")
    print("Point your index finger at text and hold steady.")
    print("Press Q to quit.")
    print("=" * 50 + "\n")

    # ── Per-placement state ─────────────────────────────────────────────────
    last_ocr_time    = 0.0
    hand_was_present = False
    ocr_done         = False   # True after the single OCR call fires
    camera_failures  = 0
    # ────────────────────────────────────────────────────────────────────────

    try:
        while True:
            ret, frame = cap.read()

            # Tolerate isolated dropped frames; bail on sustained loss
            if not ret:
                camera_failures += 1
                if camera_failures >= _MAX_CAMERA_FAILURES:
                    print("[main] Camera lost — exiting.")
                    break
                continue
            camera_failures = 0

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            results      = hand_tracker.process_frame(frame)
            hand_present = bool(results.multi_hand_landmarks)

            # ── Finger removed → full reset ─────────────────────────────────
            if hand_was_present and not hand_present:
                _reset_session(stability, cooldown)
                ocr_done = False

            hand_was_present = hand_present
            status = "Show hand to start"

            if not hand_present:
                draw_status(frame, status)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Hand is present ─────────────────────────────────────────────
            hand = results.multi_hand_landmarks[0]
            hand_tracker.draw_landmarks(frame, hand)

            finger_pos = hand_tracker.get_index_finger_tip(hand, w, h)
            hand_tracker.draw_finger_point(frame, finger_pos)

            is_stable = stability.update(finger_pos)

            # Noticeable mid-session movement → treat as a new placement
            if stability.is_moving and ocr_done:
                _reset_session(stability, cooldown)
                ocr_done = False

            roi_coords = get_roi_above_finger(finger_pos, frame.shape)

            if roi_coords is None:
                draw_status(frame, "Move finger away from edge")
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            x1, y1, x2, y2 = roi_coords
            roi = frame[y1:y2, x1:x2]

            roi_color = (0, 255, 0) if is_stable else (0, 255, 255)
            draw_roi_rectangle(frame, roi_coords, roi_color)

            display_img, current_text, is_processing = ocr_processor.get_result()

            if display_img is not None:
                cv2.imshow(OCR_DEBUG_WINDOW, display_img)

            # ── Status label ────────────────────────────────────────────────
            if is_processing:
                status = "Processing OCR…"
            elif ocr_done and current_text:
                status = "Reading — keep finger still to repeat"
            elif is_stable:
                status = "Stable — scanning…"
            else:
                status = "Hold finger steady…"

            # ── OCR: fire exactly ONCE per finger placement ─────────────────
            #    Guards: stable + not busy + not already done + interval elapsed
            now = time.time()
            if (
                is_stable
                and not is_processing
                and not ocr_done
                and (now - last_ocr_time) >= OCR_INTERVAL
            ):
                if ocr_processor.submit(roi):
                    last_ocr_time = now
                    ocr_done      = True
                    print("[main] OCR request submitted.")

            # ── Speech: speak on first result, then repeat every SPEAK_COOLDOWN
            #    No extra flag needed — SpeechCooldown is purely time-gated now.
            if current_text and len(current_text) >= MIN_TEXT_LENGTH:
                draw_text(frame, current_text)

                if cooldown.can_speak(current_text):
                    print(f"[main] Speaking: {current_text}")
                    print(f"COORD_EVENT|source=voice|type=ocr|priority=2|text={current_text}")
                    speak_async(current_text)
                    cooldown.mark_spoken(current_text)

            draw_status(frame, status)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("\n[main] Cleaning up…")
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.cleanup()
        audio_cleanup()
        print("[main] Goodbye!")


if __name__ == "__main__":
    main()