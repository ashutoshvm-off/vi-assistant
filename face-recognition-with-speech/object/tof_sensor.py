"""LiDAR/ToF sensor wrapper with fallback behavior for development machines."""

from __future__ import annotations

import os
from collections import deque
from statistics import median
from typing import Deque, Optional


class _UartLidar:
    """Read distance from UART LiDAR modules using 0x59 0x59 frame format.

    This works for many TF-Luna / TFmini-style modules and clones.
    """

    def __init__(self, port: str, baudrate: int):
        import serial  # type: ignore

        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.2)

    def read_distance_cm(self) -> Optional[float]:
        # Frame: 9 bytes, header 0x59 0x59
        # Dist low/high are bytes 2 and 3, in centimeters.
        try:
            while self._serial.in_waiting >= 9:
                b = self._serial.read(1)
                if not b or b[0] != 0x59:
                    continue

                b2 = self._serial.read(1)
                if not b2 or b2[0] != 0x59:
                    continue

                payload = self._serial.read(7)
                if len(payload) != 7:
                    return None

                dist_cm = payload[0] + (payload[1] << 8)
                if dist_cm <= 0:
                    return None
                return float(dist_cm)
        except Exception:
            return None
        return None

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass


class ToFSensor:
    """Read distance in centimeters with median smoothing.

    Supported backends:
      - vl53l0x: I2C VL53L0X modules
      - uart: TF-Luna / TFmini-style serial LiDAR (0x59 frame)
      - auto: try VL53L0X first, then UART
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        simulate: bool = False,
        sensor_type: str = "auto",
        serial_port: str = "/dev/ttyS0",
        serial_baud: int = 115200,
    ):
        self._simulate = simulate or os.getenv("TOF_SIMULATE", "0") == "1"
        self._buffer: Deque[float] = deque(maxlen=max(1, smoothing_window))
        self._sensor = None
        self._vl53_sensor = None
        self._uart_sensor: Optional[_UartLidar] = None
        self._ready = False
        self._sensor_type = (sensor_type or "auto").strip().lower()
        self._serial_port = serial_port
        self._serial_baud = int(serial_baud)
        self._driver = None

        if self._simulate:
            self._ready = True
            return

        if self._sensor_type in {"auto", "vl53l0x"} and self._init_vl53l0x():
            return

        if self._sensor_type in {"auto", "uart", "tfluna", "tfmini"} and self._init_uart_lidar():
            return

        print("[OBJECT][TOF] No supported LiDAR backend could be initialized.")
        print("[OBJECT][TOF] Distance-based collision checks will be limited.")

    def _init_vl53l0x(self) -> bool:
        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_vl53l0x  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            self._vl53_sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self._sensor = self._vl53_sensor
            self._driver = "vl53l0x"
            self._ready = True
            print("[OBJECT][TOF] Using VL53L0X (I2C).")
            return True
        except Exception as exc:
            print(f"[OBJECT][TOF] VL53L0X init failed: {exc}")
            return False

    def _init_uart_lidar(self) -> bool:
        try:
            self._uart_sensor = _UartLidar(port=self._serial_port, baudrate=self._serial_baud)
            self._sensor = self._uart_sensor
            self._driver = "uart"
            self._ready = True
            print(f"[OBJECT][TOF] Using UART LiDAR on {self._serial_port} @ {self._serial_baud}.")
            return True
        except Exception as exc:
            print(f"[OBJECT][TOF] UART LiDAR init failed: {exc}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def read_distance_cm(self) -> Optional[float]:
        """Return smoothed distance in cm, or None when unavailable."""
        if not self._ready:
            return None

        if self._simulate:
            # Keep this deterministic-ish for local testing if no sensor is attached.
            value_cm = 200.0
        else:
            if self._sensor is None:
                return None

            if self._driver == "vl53l0x":
                if self._vl53_sensor is None:
                    return None
                try:
                    mm = float(self._vl53_sensor.range)
                except Exception:
                    return None

                if mm <= 0:
                    return None
                value_cm = mm / 10.0
            elif self._driver == "uart":
                if self._uart_sensor is None:
                    return None
                value_cm = self._uart_sensor.read_distance_cm()
                if value_cm is None:
                    return None
            else:
                return None

        self._buffer.append(value_cm)
        return float(median(self._buffer))

    def close(self) -> None:
        if self._driver == "uart" and self._sensor is not None:
            try:
                if self._uart_sensor is not None:
                    self._uart_sensor.close()
            except Exception:
                pass
