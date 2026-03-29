"""
Indian Currency Detection — Real-time Inference
=================================================
Webcam-based real-time detection of Indian banknotes using TFLite.

Usage:
  python src/inference.py                        # webcam live detection
  python src/inference.py --image foo.jpg        # single image
  python src/inference.py --no-display           # headless (Raspberry Pi)
  python src/inference.py --threshold 0.6        # custom confidence threshold
  python src/inference.py --no-speech            # disable text-to-speech
  python src/inference.py --cooldown 5           # repeat speech every 5 sec
  python src/inference.py --voice en-IN-NeerjaNeural  # custom voice

Features:
  - ROI box in center of frame
  - ImageNet normalization (matches training)
  - TFLite inference with 4 threads
  - 5-frame prediction smoothing (majority vote)
  - FPS counter
  - Snapshot saving (press 's')
  - Confidence bar overlay
  - 🔊 Text-to-speech via edge-tts (speaks detected denomination)
  - Press 'm' to mute / unmute speech on the fly

Expected output:
  Stable: INR_500
  Confidence: 92%
  [🔊] "500 rupees detected"

─── One-time Setup ───────────────────────────────────────────────────────────
  pip install edge-tts pygame
──────────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import time
import argparse
import asyncio
import threading
import tempfile
import os
from pathlib import Path
from collections import Counter

# ── Optional TTS imports ──────────────────────────────────────────────────────
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

TTS_AVAILABLE = EDGE_TTS_AVAILABLE and PYGAME_AVAILABLE
if not TTS_AVAILABLE:
    missing = []
    if not EDGE_TTS_AVAILABLE:
        missing.append("edge-tts")
    if not PYGAME_AVAILABLE:
        missing.append("pygame")
    print(f"[!] Missing TTS packages: {', '.join(missing)}")
    print(f"    Run: pip install {' '.join(missing)}")
    print("    Speech output will be disabled.\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / 'models'
TFLITE_PATH  = MODELS_DIR / 'currency_model.tflite'
MAPPING_PATH = MODELS_DIR / 'class_mapping.json'
SNAPSHOT_DIR = PROJECT_ROOT / 'snapshots'

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE          = 224
CONF_THRESHOLD    = 0.45
SMOOTH_FRAMES     = 5
SPEECH_COOLDOWN   = 3.0
DEFAULT_TTS_VOICE = "en-IN-NeerjaNeural"   # Indian English female

# ── ImageNet normalization (MUST match training) ─────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Denomination display names ────────────────────────────────────────────────
DISPLAY_NAMES = {
    'INR_10':   'Rs. 10',
    'INR_20':   'Rs. 20',
    'INR_50':   'Rs. 50',
    'INR_100':  'Rs. 100',
    'INR_200':  'Rs. 200',
    'INR_500':  'Rs. 500',
    'INR_2000': 'Rs. 2000',
}

# ── TTS speech phrases ────────────────────────────────────────────────────────
SPEECH_TEXT = {
    'INR_10':   '10 rupees detected',
    'INR_20':   '20 rupees detected',
    'INR_50':   '50 rupees detected',
    'INR_100':  '100 rupees detected',
    'INR_200':  '200 rupees detected',
    'INR_500':  '500 rupees detected',
    'INR_2000': '2000 rupees detected',
}

# ── Colors per denomination (BGR) ─────────────────────────────────────────────
DENOM_COLORS = {
    'INR_10':   (90,  140, 180),
    'INR_20':   (100, 170, 100),
    'INR_50':   (220, 200, 200),
    'INR_100':  (220, 180, 200),
    'INR_200':  (100, 170, 220),
    'INR_500':  (180, 170, 170),
    'INR_2000': (180, 140, 220),
}


# ─────────────────────────────────────────────────────────────────────────────
# Edge-TTS Speech Engine
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEngine:
    """
    Non-blocking TTS engine using edge-tts + pygame.

    How it works:
      1. edge-tts synthesises text → temp MP3 via Microsoft neural voices
      2. pygame plays the MP3 in a background daemon thread
      3. Temp MP3 is deleted after playback
      4. New speak() calls are silently dropped while audio is playing

    Requires internet connection (edge-tts uses Microsoft Azure voices).

    Available Indian English voices:
      en-IN-NeerjaNeural   (female, default)
      en-IN-PrabhatNeural  (male)

    Press 'm' during webcam mode to toggle mute/unmute at runtime.
    """

    def __init__(self, voice: str = DEFAULT_TTS_VOICE, enabled: bool = True):
        self.voice   = voice
        self.enabled = enabled and TTS_AVAILABLE
        self._lock   = threading.Lock()
        self._busy   = False

        if self.enabled:
            pygame.mixer.init()
            print(f"[*] Edge-TTS ready  voice={voice}")
        else:
            print("[*] TTS disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text in a background thread.
        Silently dropped if already speaking.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True

        threading.Thread(
            target=self._speak_sync,
            args=(text,),
            daemon=True,
        ).start()

    @property
    def is_speaking(self) -> bool:
        return self._busy

    # ── Internal ──────────────────────────────────────────────────────────────

    def _speak_sync(self, text: str):
        """Background thread: synthesise → play → cleanup."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name

            # edge-tts: text → MP3 file (async, run in new event loop)
            asyncio.run(self._synthesise(text, tmp_path))

            # pygame: play MP3
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

        except Exception as e:
            print(f"[!] TTS error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            with self._lock:
                self._busy = False

    async def _synthesise(self, text: str, out_path: str):
        """Async edge-tts synthesis → MP3 file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Announcement Manager
# ─────────────────────────────────────────────────────────────────────────────

class Announcer:
    """
    Controls when TTS fires to keep announcements useful, not spammy.

    Rules:
      - Speak immediately when the stable denomination changes.
      - Re-speak the same denomination only after cooldown seconds.
      - Only speak if stable confidence >= threshold.
    """

    def __init__(self, engine: SpeechEngine, cooldown: float = SPEECH_COOLDOWN):
        self.engine       = engine
        self.cooldown     = cooldown
        self._last_label  = None
        self._last_spoken = 0.0

    def update(self, stable_label, stable_conf: float, threshold: float):
        if stable_label is None or stable_conf < threshold:
            return

        now           = time.time()
        label_changed = stable_label != self._last_label
        cooldown_done = (now - self._last_spoken) >= self.cooldown

        if label_changed or cooldown_done:
            text = SPEECH_TEXT.get(stable_label, f"{stable_label} detected")
            print(f"  [🔊] \"{text}\"")
            print(f"COORD_EVENT|source=currency|type=denomination|priority=3|text={text}")
            self.engine.speak(text)
            self._last_label  = stable_label
            self._last_spoken = now


# ─────────────────────────────────────────────────────────────────────────────
# Core ML Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_labels() -> dict:
    """Load class index → label mapping from JSON."""
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {MAPPING_PATH}\n"
            "Run: python scripts/train_model.py"
        )
    with open(MAPPING_PATH) as f:
        class_map = json.load(f)

    labels = {}
    for k, v in class_map.items():
        if isinstance(v, int):
            labels[v] = k
        else:
            labels[int(k)] = v
    return labels


def load_interpreter():
    """Load TFLite interpreter and allocate tensors."""
    if not TFLITE_PATH.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {TFLITE_PATH}\n"
            "Run: python scripts/convert_tflite.py"
        )
    interp = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH),
        num_threads=4,
    )
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame → ImageNet-normalised (1, 224, 224, 3) float32 tensor."""
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized    = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(normalized, axis=0)


def predict(interp, in_det, out_det, frame):
    """Run TFLite inference. Returns (class_id, confidence, all_probs)."""
    tensor = preprocess(frame)
    interp.set_tensor(in_det[0]['index'], tensor)
    interp.invoke()
    probs = interp.get_tensor(out_det[0]['index'])[0]
    idx   = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


def open_camera():
    """Open camera using shared utility (supports rpicam-vid on Pi)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT.parent))
    from camera_utils import open_camera as _shared_open_camera

    preferred_idx_env = os.getenv("CURRENCY_CAMERA_INDEX")
    preferred_idx = int(preferred_idx_env) if preferred_idx_env is not None else 0

    cap = _shared_open_camera(index=preferred_idx, width=640, height=480, fps=30)
    if cap is not None:
        w, h = 640, 480
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(cap, 'get') else 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(cap, 'get') else 480
        except Exception:
            pass
        print(f"[*] Camera  {w}x{h}")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# Drawing / Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame, label, confidence, fps, above_thresh,
                 stable_label, speech_active: bool = False):
    """
    Draw full detection overlay on frame:
      • Center ROI rectangle (colored per denomination)
      • Label + confidence % banner above ROI
      • Confidence progress bar inside top of ROI
      • Stable (smoothed) label inside bottom of ROI
      • Speaking indicator when TTS is active
      • FPS + threshold info top-left
      • Key hints below ROI
    """
    h, w = frame.shape[:2]
    x1 = int(w * 0.1);  y1 = int(h * 0.1)
    x2 = int(w * 0.9);  y2 = int(h * 0.9)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if above_thresh and label in DENOM_COLORS:
        color = DENOM_COLORS[label]
    elif above_thresh:
        color = (0, 220, 80)
    else:
        color = (0, 165, 255)

    display_name = DISPLAY_NAMES.get(label, label) if above_thresh else '?'
    display_text = f"{display_name}  {confidence * 100:.1f}%"

    # ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Label banner (above ROI)
    (tw, th), _ = cv2.getTextSize(display_text, font, 0.95, 2)
    banner_h = th + 24
    cv2.rectangle(frame, (x1, y1 - banner_h), (x1 + tw + 24, y1), color, -1)
    cv2.putText(frame, display_text, (x1 + 12, y1 - 10),
                font, 0.95, (0, 0, 0), 2, cv2.LINE_AA)

    # Confidence bar
    bar_w  = x2 - x1
    filled = int(bar_w * min(confidence, 1.0))
    cv2.rectangle(frame, (x1, y1 + 4), (x2, y1 + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1 + 4), (x1 + filled, y1 + 18), color, -1)

    # Stable label (bottom of ROI)
    if stable_label:
        stable_name = DISPLAY_NAMES.get(stable_label, stable_label)
        cv2.putText(frame, f"Stable: {stable_name}",
                    (x1 + 8, y2 - 38), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Speaking indicator
    if speech_active:
        cv2.putText(frame, "Speaking...",
                    (x1 + 8, y2 - 12), font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    # FPS + threshold (top-left)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Threshold: {CONF_THRESHOLD * 100:.0f}%",
                (12, 56), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Instructions (below ROI)
    cv2.putText(frame, "Place Indian banknote inside box",
                (x1, y2 + 24), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "q=quit  s=snapshot  m=mute",
                (x1, y2 + 48), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return frame, (x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
# Run Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(interp, in_det, out_det, labels,
               display: bool = True,
               speech_engine: SpeechEngine = None,
               cooldown: float = SPEECH_COOLDOWN):
    """
    Real-time webcam detection loop with edge-tts announcements.

    Keyboard controls:
      q  — quit
      s  — save snapshot
      m  — mute / unmute speech
    """
    cap = open_camera()
    if cap is None:
        print("[!] No camera found. Close other apps using the camera.")
        return

    print("[*] Running...  q=quit  s=snapshot  m=mute\n")
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    announcer      = Announcer(speech_engine, cooldown) if speech_engine else None
    history        = []
    conf_history   = []
    snapshot_n     = 0
    prev_time      = time.time()
    last_print_sec = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.mean() < 1.0:
            continue

        # Extract ROI (center 80% of frame)
        h, w = frame.shape[:2]
        roi  = frame[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

        # Predict
        class_id, confidence, probs = predict(interp, in_det, out_det, roi)
        label = labels.get(class_id, f"class_{class_id}")

        # Smoothing — majority vote over last SMOOTH_FRAMES frames
        history.append(label)
        conf_history.append(confidence)
        if len(history) > SMOOTH_FRAMES:
            history.pop(0)
            conf_history.pop(0)

        stable_label = None
        stable_conf  = 0.0
        if len(history) == SMOOTH_FRAMES:
            counter = Counter(history)
            stable_label, _ = counter.most_common(1)[0]
            stable_conf = float(np.mean([
                c for l, c in zip(history, conf_history)
                if l == stable_label
            ]))

        # FPS
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        above     = confidence >= CONF_THRESHOLD

        # TTS announcement
        if announcer and stable_label:
            announcer.update(stable_label, stable_conf, CONF_THRESHOLD)

        speaking = (
            speech_engine is not None
            and speech_engine.enabled
            and speech_engine.is_speaking
        )

        # Display
        if display:
            frame, _ = draw_overlay(
                frame, label, confidence, fps, above,
                stable_label, speech_active=speaking,
            )
            cv2.imshow(
                "Indian Currency Detection  [q=quit  s=snapshot  m=mute]",
                frame,
            )

        # Console output (once per second)
        current_sec = int(now)
        if current_sec != last_print_sec:
            last_print_sec = current_sec
            status = (
                f"{DISPLAY_NAMES.get(label, label)} ({confidence * 100:.1f}%)"
                if above else
                f"? (best: {label} {confidence * 100:.1f}%)"
            )
            stable_str = (
                f"Stable: {DISPLAY_NAMES.get(stable_label, stable_label)} "
                f"({stable_conf * 100:.0f}%)"
                if stable_label else "Stable: ..."
            )
            print(f"  {status:<30}  {stable_str:<35}  FPS={fps:.1f}")

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and display:
            snapshot_n += 1
            fname = SNAPSHOT_DIR / (
                f"snapshot_{snapshot_n:03d}_{label}_{confidence * 100:.0f}pct.jpg"
            )
            cv2.imwrite(str(fname), frame)
            print(f"[*] Snapshot saved: {fname}")
        elif key == ord('m') and speech_engine:
            speech_engine.enabled = not speech_engine.enabled
            print(f"[*] Speech {'ON' if speech_engine.enabled else 'OFF'}")

    cap.release()
    if display:
        cv2.destroyAllWindows()
    print("\n[*] Done.")


def run_image(image_path: str, interp, in_det, out_det, labels,
              speech_engine: SpeechEngine = None):
    """Run inference on a single image file and optionally speak the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[!] Cannot read image: {image_path}")
        return

    class_id, confidence, probs = predict(interp, in_det, out_det, frame)
    label        = labels.get(class_id, f"class_{class_id}")
    display_name = DISPLAY_NAMES.get(label, label)

    print(f"\n{'=' * 50}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {display_name}")
    print(f"  Class      : {label}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print(f"{'=' * 50}")

    print("\n  Top predictions:")
    top5 = np.argsort(probs)[::-1][:5]
    for rank, i in enumerate(top5, 1):
        lbl   = labels.get(i, f'class_{i}')
        dname = DISPLAY_NAMES.get(lbl, lbl)
        bar   = '#' * int(probs[i] * 30)
        print(f"    {rank}. {dname:<12} {probs[i] * 100:6.2f}%  {bar}")

    # Speak result
    if speech_engine and speech_engine.enabled and confidence >= CONF_THRESHOLD:
        text = SPEECH_TEXT.get(label, f"{label} detected")
        print(f"\n  [🔊] Speaking: \"{text}\"")
        speech_engine.speak(text)
        time.sleep(4)

    # Save annotated image
    above = confidence >= CONF_THRESHOLD
    color = (0, 220, 80) if above else (0, 165, 255)
    cv2.putText(frame, f"{display_name}  {confidence * 100:.1f}%",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    out_path = Path(image_path).stem + '_result.jpg'
    cv2.imwrite(out_path, frame)
    print(f"\n[*] Annotated image saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global CONF_THRESHOLD

    parser = argparse.ArgumentParser(
        description='Indian Currency Detection — Real-time Inference (edge-tts)'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to image file for single-image detection',
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Headless mode, no OpenCV window (for Raspberry Pi)',
    )
    parser.add_argument(
        '--threshold', type=float, default=CONF_THRESHOLD,
        help=f'Confidence threshold (default: {CONF_THRESHOLD})',
    )
    parser.add_argument(
        '--no-speech', action='store_true',
        help='Disable text-to-speech output',
    )
    parser.add_argument(
        '--voice', type=str, default=DEFAULT_TTS_VOICE,
        help=f'Edge-TTS voice name (default: {DEFAULT_TTS_VOICE})\n'
             f'Other option: en-IN-PrabhatNeural (male)',
    )
    parser.add_argument(
        '--cooldown', type=float, default=SPEECH_COOLDOWN,
        help=f'Seconds before repeating same denomination (default: {SPEECH_COOLDOWN})',
    )
    args           = parser.parse_args()
    CONF_THRESHOLD = args.threshold

    # Banner
    print("\n" + "=" * 60)
    print("  Indian Currency Detection — Real-time Inference")
    print("=" * 60)
    print(f"  TFLite model : {TFLITE_PATH.name}")
    print(f"  Threshold    : {CONF_THRESHOLD * 100:.0f}%")
    print(f"  Smoothing    : {SMOOTH_FRAMES} frames")
    print(f"  Normalize    : ImageNet (mean/std)")
    print(f"  TTS engine   : {'disabled' if args.no_speech else 'edge-tts (online)'}")
    if not args.no_speech:
        print(f"  TTS voice    : {args.voice}")
        print(f"  Cooldown     : {args.cooldown}s")
    print("=" * 60 + "\n")

    # Load labels & TFLite model
    labels     = load_labels()
    label_list = [labels[i] for i in sorted(labels.keys())]
    print(f"[*] Loaded {len(labels)} classes: {label_list}")

    interp, in_det, out_det = load_interpreter()
    print(f"[*] TFLite ready  input={in_det[0]['shape']}  "
          f"output={out_det[0]['shape']}")

    # Init edge-tts
    speech_engine = SpeechEngine(
        voice   = args.voice,
        enabled = not args.no_speech,
    )

    # Run
    if args.image:
        run_image(
            args.image, interp, in_det, out_det, labels,
            speech_engine=speech_engine,
        )
    else:
        run_webcam(
            interp, in_det, out_det, labels,
            display       = not args.no_display,
            speech_engine = speech_engine,
            cooldown      = args.cooldown,
        )


if __name__ == '__main__':
    main()