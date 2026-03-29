#!/usr/bin/env python3
"""
Unified launcher for the Smart Vision Assistance System.

This file lets you run the existing modules from one entrypoint:
- face: automated face recognition
- voice: hand-guided OCR + speech
- currency: currency detection inference
- object: object detection + collision avoidance

Examples:
  python main.py --module face
  python main.py --module voice
  python main.py --module currency -- --threshold 0.6
    python main.py --module object
  python main.py --module all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; env vars can be set manually

try:
    import pyttsx3  # type: ignore[import-not-found]
except ImportError:
    pyttsx3 = None

ROOT = Path(__file__).resolve().parent
PINS_PATH = ROOT / "pins.json"
CM4_PROFILE_PATH = ROOT / "cm4_runtime_profile.json"

MODULES: Dict[str, dict] = {
    "face": {
        "title": "Face Recognition",
        "script": ROOT / "face" / "automated_system.py",
        "cwd": ROOT / "face",
    },
    "voice": {
        "title": "Voice OCR",
        "script": ROOT / "voice" / "main.py",
        "cwd": ROOT / "voice",
    },
    "currency": {
        "title": "Currency Detection",
        "script": ROOT / "currency" / "src" / "inference.py",
        "cwd": ROOT / "currency",
    },
    "object": {
        "title": "Object Collision Avoidance",
        "script": ROOT / "object" / "main.py",
        "cwd": ROOT / "object",
    },
}

SOURCE_ORDER = {"hazard": 0, "face": 1, "voice": 2, "currency": 3, "object": 4}
COORD_EVENT_RE = re.compile(
    r"^COORD_EVENT\|source=(?P<source>[^|]+)\|type=(?P<etype>[^|]+)\|priority=(?P<priority>\d+)\|text=(?P<text>.+)$"
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_preflight(target: str, components: List[str]) -> None:
    print("=" * 70)
    print(f"Preflight checks for target: {target}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    required = ["cv2", "numpy"]
    if "voice" in components:
        required.extend(["mediapipe", "requests"])
    if "currency" in components:
        required.append("tensorflow")
    if "face" in components:
        required.extend(["torch", "facenet_pytorch", "supabase"])
    if "object" in components:
        required.append("adafruit_vl53l0x")

    missing = [name for name in required if not _module_available(name)]
    if missing:
        print(f"[!] Missing packages: {', '.join(sorted(set(missing)))}")
    else:
        print("[+] Dependency check passed for selected components.")

    if target == "cm4":
        print("[CM4] Headless-friendly mode will be applied where possible.")
        if "face" in components:
            print("[CM4] Note: face module requires PyTorch/facenet and may be resource-heavy.")
    print("=" * 70)


def _probe_cameras(max_index: int = 4) -> List[int]:
    """Best-effort camera probe that does not keep devices open."""
    if not _module_available("cv2"):
        return []

    import cv2

    found: List[int] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break
        cap.release()
        if ok:
            found.append(idx)
    return found


def _load_pins() -> dict:
    """Read hardware pin mapping from pins.json (if it exists)."""
    if not PINS_PATH.exists():
        return {}
    try:
        with open(PINS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load pins.json: {exc}")
        return {}


def _build_runtime_profile(target: str) -> dict:
    """Create a runtime profile with camera mapping, fps settings, and pin config."""
    cameras = _probe_cameras(4)
    pins = _load_pins()

    # CM4 defaults trade a little fidelity for stability and thermal headroom.
    fps = 20 if target == "cm4" else 30
    width = 640
    height = 480

    # Use camera_id from pins.json ArduCam definition as the preferred index.
    arducam_id = pins.get("arducam", {}).get("camera_id")
    if arducam_id is not None:
        preferred_idx = int(arducam_id)
    else:
        preferred_idx = cameras[0] if cameras else 0

    face_idx = preferred_idx
    voice_idx = cameras[1] if len(cameras) > 1 else face_idx
    currency_idx = cameras[2] if len(cameras) > 2 else face_idx
    object_idx = cameras[3] if len(cameras) > 3 else currency_idx

    # LiDAR configuration from pins.json (VL53L0X I2C pins).
    vl53_pins = pins.get("vl53l0x", {})
    lidar_sda = vl53_pins.get("sda")
    lidar_scl = vl53_pins.get("scl")

    # Default LiDAR profile: CM4 usually uses UART modules for obstacle ranging.
    object_lidar_type = "uart" if target == "cm4" else "auto"
    object_lidar_port = "/dev/ttyS0"
    object_lidar_baud = 115200
    # TFLite is the default — Caffe model files are not shipped in the repo.
    object_detector_backend = "tflite"

    # Microphone pin info (INMP441 I2S) — stored for reference / future audio input.
    mic_pins = pins.get("inmp441", {})

    profile = {
        "target": target,
        "detected_cameras": cameras,
        "pins": pins,
        "face_camera_index": face_idx,
        "voice_camera_index": voice_idx,
        "currency_camera_index": currency_idx,
        "face_camera_width": width,
        "face_camera_height": height,
        "face_camera_fps": fps,
        "voice_camera_width": width,
        "voice_camera_height": height,
        "voice_camera_fps": fps,
        "object_camera_index": object_idx,
        "object_camera_width": width,
        "object_camera_height": height,
        "object_camera_fps": fps,
        "object_lidar_type": object_lidar_type,
        "object_lidar_port": object_lidar_port,
        "object_lidar_baud": object_lidar_baud,
        "object_lidar_sda": lidar_sda,
        "object_lidar_scl": lidar_scl,
        "object_detector_backend": object_detector_backend,
        "mic_sd": mic_pins.get("sd"),
        "mic_sck": mic_pins.get("sck"),
        "mic_ws": mic_pins.get("ws"),
        "mic_lr": mic_pins.get("lr"),
    }
    return profile


def _save_runtime_profile(profile: dict) -> None:
    with open(CM4_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def _load_runtime_profile() -> Optional[dict]:
    if not CM4_PROFILE_PATH.exists():
        return None
    try:
        with open(CM4_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load runtime profile: {exc}")
        return None


class SpeechCoordinator:
    """Single speech queue with priority ordering and dedup cooldown."""

    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._last_spoken_text: Dict[str, float] = {}
        self._engine = None

        if pyttsx3 is not None:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 165)
                self._engine.setProperty("volume", 1.0)
            except Exception as exc:
                print(f"[COORD] pyttsx3 init failed: {exc}")
                self._engine = None

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def available(self) -> bool:
        return self._engine is not None

    def enqueue(self, source: str, text: str, priority: int) -> None:
        text = (text or "").strip()
        if not text:
            return

        now = time.time()
        last_time = self._last_spoken_text.get(text)
        if last_time is not None and (now - last_time) < self.cooldown_seconds:
            return

        with self._counter_lock:
            self._counter += 1
            ordinal = self._counter

        order = SOURCE_ORDER.get(source, 9)
        self._queue.put((priority, order, ordinal, source, text))

    def stop(self) -> None:
        self._stop.set()
        self._queue.put((999, 9, 0, "system", ""))
        self._worker.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                _, _, _, source, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._stop.is_set() and not text:
                break
            if not text:
                continue

            self._last_spoken_text[text] = time.time()
            print(f"[COORD-SPEAK][{source.upper()}] {text}")

            if self._engine is not None:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as exc:
                    print(f"[COORD] TTS error: {exc}")


def _extract_event(source: str, line: str) -> Optional[tuple]:
    """
    Map module stdout lines to coordinator speech events.

    Priority meaning:
      0 = hazard (highest)
      1 = face recognition
      2 = OCR text
      3 = currency
    """
    text = line.strip()
    if not text:
        return None

    # Preferred path: machine-readable module event.
    m = COORD_EVENT_RE.match(text)
    if m:
        ev_source = m.group("source").strip().lower()
        ev_text = m.group("text").strip()
        ev_priority = int(m.group("priority"))
        return (ev_source, ev_text, ev_priority)

    lower = text.lower()

    for kw in ("fire", "hazard", "stairs", "vehicle", "obstacle"):
        if kw in lower:
            return ("hazard", f"Warning: {text}", 0)

    if source == "face":
        m = re.search(r"RECOGNIZED:\s*([A-Za-z0-9_\- ]+)", text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().replace("_", " ")
            return ("face", f"{name} recognized", 1)

        m = re.search(r"NEW FACE REGISTERED:\s*([A-Za-z0-9_\- ]+)", text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().replace("_", " ")
            return ("face", f"{name} registered", 1)

    if source == "voice":
        m = re.search(r"\[main\]\s*Speaking:\s*(.+)$", text)
        if m:
            return ("voice", m.group(1).strip(), 2)

    if source == "currency":
        m = re.search(r"\[🔊\]\s*\"(.+?)\"", text)
        if m:
            return ("currency", m.group(1).strip(), 3)

    return None


def _stream_module_output(
    source: str,
    proc: subprocess.Popen,
    coordinator: SpeechCoordinator,
) -> None:
    if proc.stdout is None:
        return

    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        print(f"[{source.upper()}] {line}")

        event = _extract_event(source, line)
        if event is not None:
            ev_source, ev_text, ev_priority = event
            coordinator.enqueue(ev_source, ev_text, ev_priority)


def _check_scripts_exist() -> None:
    missing = [name for name, cfg in MODULES.items() if not cfg["script"].exists()]
    if missing:
        print("Missing module scripts:")
        for name in missing:
            print(f"  - {name}: {MODULES[name]['script']}")
        sys.exit(1)


def _run_self_test(target: str, components: List[str]) -> int:
    """Run non-invasive checks suitable for desktop and CM4 deployment validation."""
    print("=" * 70)
    print("Running Smart Vision self-test")
    print(f"Target: {target}")
    print(f"Components: {', '.join(components)}")
    print("=" * 70)

    failures: List[str] = []

    for module_name in components:
        script = MODULES[module_name]["script"]
        if not script.exists():
            failures.append(f"Missing script for {module_name}: {script}")

    if "currency" in components:
        model = ROOT / "currency" / "models" / "currency_model.tflite"
        mapping = ROOT / "currency" / "models" / "class_mapping.json"
        if not model.exists():
            failures.append(f"Missing currency model: {model}")
        if not mapping.exists():
            failures.append(f"Missing class mapping: {mapping}")

    if "face" in components:
        env_file = ROOT / ".env"
        face_env_file = ROOT / "face" / ".env"
        if not env_file.exists() and not face_env_file.exists():
            print("[WARN] No .env found (root or face/). Face cloud integrations may fail.")

    if "object" in components:
        obj_tflite = ROOT / "object" / "models" / "unified_hazard_roads_homes_int8.tflite"
        obj_labels = ROOT / "object" / "models" / "unified_hazard_labels.txt"
        if not obj_tflite.exists():
            failures.append(f"Missing object TFLite model: {obj_tflite}")
        if not obj_labels.exists():
            failures.append(f"Missing object labels: {obj_labels}")

    _print_preflight(target, components)

    coordinator = SpeechCoordinator(cooldown_seconds=1.0)
    if coordinator.available:
        print("[+] Coordinator TTS engine is available.")
    else:
        print("[WARN] Coordinator TTS unavailable. Coordinated mode will fall back to module TTS.")
    coordinator.stop()

    profile = _build_runtime_profile(target)
    _save_runtime_profile(profile)
    print(f"[+] Runtime profile saved: {CM4_PROFILE_PATH}")
    print(f"[+] Detected camera indices: {profile['detected_cameras']}")

    if len(profile["detected_cameras"]) < len([c for c in components if c in ("face", "voice", "currency", "object")]):
        print("[WARN] Fewer cameras than active components. Some modules may contend for one camera.")

    if failures:
        print("\nSelf-test failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n[+] Self-test passed.")
    return 0


def _run_single(module_name: str, passthrough_args: List[str]) -> int:
    cfg = MODULES[module_name]
    cmd = [sys.executable, str(cfg["script"]), *passthrough_args]

    print("=" * 70)
    print(f"Starting {cfg['title']}")
    print(f"Command: {' '.join(cmd)}")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    try:
        completed = subprocess.run(cmd, cwd=str(cfg["cwd"]))
        return int(completed.returncode)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


def _run_all_parallel() -> int:
    print("=" * 70)
    print("Starting all modules in parallel")
    print("Note: if modules share the same camera index, camera conflicts may happen.")
    print("Stop all modules with Ctrl+C.")
    print("=" * 70)

    processes: List[subprocess.Popen] = []
    for module_name in ["face", "voice", "currency", "object"]:
        cfg = MODULES[module_name]
        cmd = [sys.executable, str(cfg["script"])]
        print(f"Launching {cfg['title']}...")
        proc = subprocess.Popen(cmd, cwd=str(cfg["cwd"]))
        processes.append(proc)
        time.sleep(0.5)

    try:
        while True:
            alive = [p for p in processes if p.poll() is None]
            if not alive:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all modules...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return 130

    codes = [p.returncode for p in processes]
    print(f"Modules exited with codes: {codes}")
    return 0 if all(code == 0 for code in codes) else 1


def _run_coordinated(target: str, components: List[str]) -> int:
    print("=" * 70)
    print("Starting coordinator mode (single combined speech stream)")
    print("Priority: hazard > face > OCR > currency")
    print(f"Target: {target}")
    print(f"Components: {', '.join(components)}")
    print("Stop all modules with Ctrl+C.")
    print("=" * 70)

    _print_preflight(target, components)

    coordinator = SpeechCoordinator(cooldown_seconds=2.0)
    use_central_tts = coordinator.available

    if not use_central_tts:
        print("[WARN] pyttsx3 is unavailable. Falling back to module-local TTS.")

    module_cmds = {
        "face": [sys.executable, str(MODULES["face"]["script"])],
        "voice": [sys.executable, str(MODULES["voice"]["script"])],
        "currency": [sys.executable, str(MODULES["currency"]["script"])],
        "object": [sys.executable, str(MODULES["object"]["script"])],
    }

    if use_central_tts:
        module_cmds["currency"].append("--no-speech")
        module_cmds["object"].append("--no-speech")

    profile = _load_runtime_profile()
    if profile is None:
        profile = _build_runtime_profile(target)
        _save_runtime_profile(profile)
        print(f"[INFO] Generated runtime profile: {CM4_PROFILE_PATH}")

    print(
        "[INFO] Camera map "
        f"face={profile.get('face_camera_index')} "
        f"voice={profile.get('voice_camera_index')} "
        f"currency={profile.get('currency_camera_index')} "
        f"object={profile.get('object_camera_index', profile.get('currency_camera_index'))}"
    )

    processes: Dict[str, subprocess.Popen] = {}
    threads: List[threading.Thread] = []

    try:
        for module_name in components:
            cfg = MODULES[module_name]
            env = os.environ.copy()

            if use_central_tts and module_name in ("face", "voice"):
                env["COORD_DISABLE_LOCAL_TTS"] = "1"

            if use_central_tts and module_name == "object":
                env["COORD_DISABLE_LOCAL_TTS"] = "1"

            if module_name == "face":
                env["FACE_CAMERA_INDEX"] = str(profile.get("face_camera_index", 0))
                env["FACE_CAMERA_WIDTH"] = str(profile.get("face_camera_width", 640))
                env["FACE_CAMERA_HEIGHT"] = str(profile.get("face_camera_height", 480))
                env["FACE_CAMERA_FPS"] = str(profile.get("face_camera_fps", 20))

            if module_name == "voice":
                env["VOICE_CAMERA_INDEX"] = str(profile.get("voice_camera_index", 1))
                env["VOICE_CAMERA_WIDTH"] = str(profile.get("voice_camera_width", 640))
                env["VOICE_CAMERA_HEIGHT"] = str(profile.get("voice_camera_height", 480))
                env["VOICE_CAMERA_FPS"] = str(profile.get("voice_camera_fps", 20))

            if module_name == "currency":
                env["CURRENCY_CAMERA_INDEX"] = str(profile.get("currency_camera_index", 0))

            if module_name == "object":
                env["OBJECT_CAMERA_INDEX"] = str(profile.get("object_camera_index", profile.get("currency_camera_index", 0)))
                env["OBJECT_CAMERA_WIDTH"] = str(profile.get("voice_camera_width", 640))
                env["OBJECT_CAMERA_HEIGHT"] = str(profile.get("voice_camera_height", 480))
                env["OBJECT_CAMERA_FPS"] = str(profile.get("voice_camera_fps", 20))
                env["OBJECT_LIDAR_TYPE"] = str(profile.get("object_lidar_type", "auto"))
                env["OBJECT_LIDAR_PORT"] = str(profile.get("object_lidar_port", "/dev/ttyS0"))
                env["OBJECT_LIDAR_BAUD"] = str(profile.get("object_lidar_baud", 115200))
                env["OBJECT_DETECTOR_BACKEND"] = str(profile.get("object_detector_backend", "tflite"))

            if target == "cm4":
                env["PYTHONUNBUFFERED"] = "1"

                # Keep OpenCV from trying X/GUI plugins when not available.
                if module_name == "currency":
                    if "--no-display" not in module_cmds[module_name]:
                        module_cmds[module_name].append("--no-display")
                if module_name == "object":
                    if "--no-display" not in module_cmds[module_name]:
                        module_cmds[module_name].append("--no-display")

            print(f"Launching {cfg['title']} in coordinator mode...")
            proc = subprocess.Popen(
                module_cmds[module_name],
                cwd=str(cfg["cwd"]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            processes[module_name] = proc

            thread = threading.Thread(
                target=_stream_module_output,
                args=(module_name, proc, coordinator),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
            time.sleep(0.5)

        while True:
            alive = [name for name, proc in processes.items() if proc.poll() is None]
            if not alive:
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping coordinator mode...")
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return 130
    finally:
        coordinator.stop()

    codes = [proc.returncode for proc in processes.values()]
    print(f"Coordinator session exit codes: {codes}")
    return 0 if all(code == 0 for code in codes) else 1


def _interactive_menu() -> str:
    print("\n" + "=" * 70)
    print("SMART VISION ASSISTANCE SYSTEM")
    print("=" * 70)
    print("1. Face Recognition")
    print("2. Voice OCR")
    print("3. Currency Detection")
    print("4. Object Collision Avoidance")
    print("5. Run All Modules")
    print("6. Coordinated Mode (single speech stream)")
    print("7. Exit")

    choices = {
        "1": "face",
        "2": "voice",
        "3": "currency",
        "4": "object",
        "5": "all",
        "6": "coordinated",
        "7": "exit",
    }

    while True:
        pick = input("Select option (1-7): ").strip()
        if pick in choices:
            return choices[pick]
        print("Invalid choice. Please select 1-7.")


def main() -> int:
    _check_scripts_exist()

    parser = argparse.ArgumentParser(
        description="Unified launcher for face, voice, currency, and object modules"
    )
    parser.add_argument(
        "--module",
        choices=["face", "voice", "currency", "object", "all", "coordinated"],
        help="Module to run. If omitted, interactive menu is shown.",
    )
    parser.add_argument(
        "--target",
        choices=["desktop", "cm4"],
        default="desktop",
        help="Runtime target profile.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["face", "voice", "currency", "object"],
        default=["face", "voice", "currency", "object"],
        help="Components to run in coordinated mode.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deployment checks and exit.",
    )

    args, passthrough = parser.parse_known_args()

    if args.self_test:
        return _run_self_test(target=args.target, components=args.components)

    module = args.module
    if module is None:
        module = _interactive_menu()

    if module == "exit":
        print("Goodbye.")
        return 0

    if module == "all":
        if passthrough:
            print("Ignoring passthrough args in all mode.")
        return _run_all_parallel()

    if module == "coordinated":
        if passthrough:
            print("Ignoring passthrough args in coordinated mode.")
        return _run_coordinated(target=args.target, components=args.components)

    return _run_single(module, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
