"""
Camera finder - run this first to find your working camera index and backend
"""
import cv2

backends = [
    (cv2.CAP_MSMF,  "MSMF (Windows Media Foundation)"),
    (None,           "Default (no backend)"),
    (cv2.CAP_DSHOW, "DirectShow"),
]

print("Testing camera backends...\n")

found = False
for index in range(3):
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"SUCCESS: Camera index={index}  backend={name}")
                    print(f"         Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
                    found = True
                cap.release()
            else:
                print(f"  FAIL:  index={index}  backend={name}")
        except Exception as e:
            print(f"  ERROR: index={index}  backend={name}  -> {e}")

if not found:
    print("\nNo working camera found.")
    print("Make sure your webcam is plugged in and not used by another app (Teams, Zoom, etc.)")