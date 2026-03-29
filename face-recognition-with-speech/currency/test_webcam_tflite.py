"""
Webcam Currency Detection - TFLite
Press 'q' to quit, 's' to save snapshot
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import time

MODEL_PATH     = "models/currency_model.tflite"
MAPPING_PATH   = "models/class_mapping.json"
IMG_SIZE       = 224
CONF_THRESHOLD = 0.5

# ── Load labels ──────────────────────────────────────────────────────────────
with open(MAPPING_PATH) as f:
    class_map = json.load(f)

if all(isinstance(v, int) for v in class_map.values()):
    labels = {v: k for k, v in class_map.items()}
else:
    labels = {int(k): v for k, v in class_map.items()}

# ── Load TFLite model ────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Find a working camera ─────────────────────────────────────────────────────
# From camera scan: index=1 DirectShow and index=0 MSMF both work
# Try them in order until we get a real frame
def open_camera():
    candidates = [
        (1, cv2.CAP_DSHOW),   # index=1 DirectShow — confirmed working
        (0, cv2.CAP_MSMF),    # index=0 MSMF       — confirmed working
        (0, cv2.CAP_DSHOW),
        (0, None),
        (1, cv2.CAP_MSMF),
        (1, None),
    ]
    for idx, backend in candidates:
        try:
            cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            # Read a few frames to confirm it gives real data
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 1.0:
                    bname = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF"}.get(backend, "Default")
                    print(f"[*] Camera opened: index={idx} backend={bname}  {int(cap.get(3))}x{int(cap.get(4))}")
                    return cap
            cap.release()
        except Exception as e:
            continue
    return None

cap = open_camera()
if cap is None:
    print("[!] No working camera found. Make sure no other app is using it.")
    exit(1)

print("[*] Currency Detection running.  q = quit   s = snapshot\n")

snapshot_n = 0
prev_time  = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[!] Failed to grab frame.")
        break

    # Skip blank/corrupted frames
    if frame.mean() < 1.0:
        continue

    h, w = frame.shape[:2]

    # ── ROI box ───────────────────────────────────────────────────────────────
    x1, y1 = int(w * 0.1), int(h * 0.1)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    roi = frame[y1:y2, x1:x2]

    # ── Preprocess ────────────────────────────────────────────────────────────
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # ── Inference ─────────────────────────────────────────────────────────────
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds      = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id   = int(np.argmax(preds))
    confidence = float(preds[class_id])
    label      = labels.get(class_id, f"class_{class_id}")

    # ── FPS ───────────────────────────────────────────────────────────────────
    now       = time.time()
    fps       = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    # ── Colours ───────────────────────────────────────────────────────────────
    if confidence >= CONF_THRESHOLD:
        box_color = (0, 220, 80)
        display   = f"{label}  {confidence*100:.1f}%"
    else:
        box_color = (0, 165, 255)
        display   = f"?  {confidence*100:.1f}%"

    # ── Draw ──────────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x1, y1 - th - 20), (x1 + tw + 20, y1), box_color, -1)
    cv2.putText(frame, display, (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    bar_w  = x2 - x1
    filled = int(bar_w * confidence)
    cv2.rectangle(frame, (x1, y1 + 4), (x2, y1 + 14), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y1 + 4), (x1 + filled, y1 + 14), box_color, -1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Currency Detection  [q=quit  s=snapshot]", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snapshot_n += 1
        fname = f"snapshot_{snapshot_n:03d}_{label.replace(' ', '_')}_{confidence*100:.0f}pct.jpg"
        cv2.imwrite(fname, frame)
        print(f"[*] Snapshot saved: {fname}")

cap.release()
cv2.destroyAllWindows()
print("Done.")