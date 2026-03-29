# CM4 Integration Guide

This project now supports a coordinator runtime profile for Raspberry Pi CM4.

## 1) Run coordinated mode on CM4

```bash
python3 main.py --module coordinated --target cm4 --components face voice currency
```

Recommended if face stack is too heavy:

```bash
python3 main.py --module coordinated --target cm4 --components voice currency object
```

## 2) What CM4 target changes

- Enables headless-friendly behavior for currency (`--no-display`)
- Centralizes speech in one queue (no overlapping audio)
- Uses machine-readable events emitted by all modules
- Prints preflight diagnostics for target and dependencies

## 3) Install dependencies (Linux/CM4)

Create a virtual environment and install module dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r voice/requirements.txt
pip install -r currency/requirements.txt
pip install -r face/requirements.txt
```

If `face` dependencies fail or are too heavy on CM4, run with `--components voice currency`.

## 4) Camera setup notes

- Ensure each module can access the camera it needs.
- If multiple modules require video input simultaneously, use separate streams or dedicated cameras.
- For DroidCam / USB cams, verify device index mapping before boot-time deployment.

## 5) Systemd autostart

A service template is provided at `deploy/smartvision-cm4.service`.

Recommended install (auto-detects user/path and enables boot autostart):

```bash
chmod +x deploy/install_cm4_service.sh
./deploy/install_cm4_service.sh
```

Install with selected components:

```bash
./deploy/install_cm4_service.sh "voice currency"
```

Manual install (if you prefer):

```bash
sudo cp deploy/smartvision-cm4.service /etc/systemd/system/smartvision.service
sudo systemctl daemon-reload
sudo systemctl enable smartvision.service
sudo systemctl start smartvision.service
sudo systemctl status smartvision.service
```

Logs:

```bash
tail -f /var/log/smartvision.log
```

Power-loss recovery check:

```bash
sudo systemctl is-enabled smartvision.service
sudo systemctl status smartvision.service
```

If enabled, the system starts automatically whenever power returns.

Uninstall service:

```bash
chmod +x deploy/uninstall_cm4_service.sh
./deploy/uninstall_cm4_service.sh
```

## 6) Audio and speech behavior

- Face and voice local TTS are disabled only in coordinated mode.
- Currency local TTS is disabled in coordinated mode.
- Object local TTS is disabled in coordinated mode.
- One global queue speaks with priority:
  - hazard (highest)
  - face
  - OCR text
  - currency

## 7) Environment variables

Set before running for OCR API quality and limits:

```bash
export OCR_SPACE_API_KEY=<your_key>
```

Optional local control variables used by coordinator:

- `COORD_DISABLE_LOCAL_TTS=1` (set automatically by launcher for face/voice children)

Object LiDAR defaults in CM4 coordinated profile:

- `OBJECT_LIDAR_TYPE=uart`
- `OBJECT_LIDAR_PORT=/dev/ttyS0`
- `OBJECT_LIDAR_BAUD=115200`
- `OBJECT_DETECTOR_BACKEND=tflite`

## 8) Validation checklist

- Coordinator starts and all selected components launch
- Camera feed opens for each enabled component
- Speech is single-stream (no overlap)
- OCR, face, and currency events are spoken with expected priority
- Service restarts automatically after failure/reboot
