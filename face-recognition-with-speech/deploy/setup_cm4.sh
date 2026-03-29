#!/usr/bin/env bash
# ============================================================
# Smart Vision — One-time CM4 Setup
# Run once after flashing Raspberry Pi OS:
#   chmod +x deploy/setup_cm4.sh
#   sudo deploy/setup_cm4.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PINS_FILE="${PROJECT_DIR}/pins.json"
RUN_USER="${SUDO_USER:-pi}"
VENV_DIR="${PROJECT_DIR}/.venv"
CONFIG_TXT="/boot/firmware/config.txt"
NEEDS_REBOOT=false

# Fallback for older Pi OS where config.txt is at /boot/config.txt
if [[ ! -f "${CONFIG_TXT}" ]]; then
  CONFIG_TXT="/boot/config.txt"
fi

echo ""
echo "============================================================"
echo "  Smart Vision — CM4 Setup"
echo "============================================================"
echo "  Project : ${PROJECT_DIR}"
echo "  User    : ${RUN_USER}"
echo "  Config  : ${CONFIG_TXT}"
echo "============================================================"
echo ""

# ── 1. System packages ──────────────────────────────────────
echo "[1/7] Installing system packages…"
apt-get update -qq
apt-get install -y -qq \
  python3-venv python3-dev python3-pip \
  libcap-dev libatlas-base-dev \
  libportaudio2 portaudio19-dev \
  espeak espeak-ng \
  i2c-tools libi2c-dev \
  libcamera-apps libcamera-dev \
  libjpeg-dev libopenjp2-7 \
  git curl jq > /dev/null
echo "  ✓ System packages installed"

# ── 2. Enable I2C (VL53L0X LiDAR — SDA=2, SCL=3) ───────────
echo "[2/7] Enabling I2C for VL53L0X LiDAR…"
if ! grep -q "^dtparam=i2c_arm=on" "${CONFIG_TXT}" 2>/dev/null; then
  echo "dtparam=i2c_arm=on" >> "${CONFIG_TXT}"
  NEEDS_REBOOT=true
  echo "  ✓ I2C enabled in ${CONFIG_TXT}"
else
  echo "  ✓ I2C already enabled"
fi
# Load the module now so we can test without rebooting
modprobe i2c-dev 2>/dev/null || true
if ! grep -q "^i2c-dev" /etc/modules 2>/dev/null; then
  echo "i2c-dev" >> /etc/modules
fi

# ── 3. Enable I2S (INMP441 Mic — SD=20, SCK=18, WS=19) ─────
echo "[3/7] Enabling I2S for INMP441 microphone…"
if ! grep -q "^dtoverlay=i2s-mmap" "${CONFIG_TXT}" 2>/dev/null; then
  echo "dtoverlay=i2s-mmap" >> "${CONFIG_TXT}"
  NEEDS_REBOOT=true
  echo "  ✓ I2S overlay added to ${CONFIG_TXT}"
else
  echo "  ✓ I2S overlay already enabled"
fi
# snd-bcm2835 handles I2S on CM4
modprobe snd-bcm2835 2>/dev/null || true

# ── 4. Enable CSI camera (ArduCam) ──────────────────────────
echo "[4/7] Enabling CSI camera for ArduCam…"
# camera_auto_detect works for most ArduCam variants
if ! grep -q "^camera_auto_detect=1" "${CONFIG_TXT}" 2>/dev/null; then
  # Remove legacy camera_auto_detect=0 if present
  sed -i 's/^camera_auto_detect=0/camera_auto_detect=1/' "${CONFIG_TXT}" 2>/dev/null || true
  if ! grep -q "^camera_auto_detect=1" "${CONFIG_TXT}" 2>/dev/null; then
    echo "camera_auto_detect=1" >> "${CONFIG_TXT}"
  fi
  NEEDS_REBOOT=true
  echo "  ✓ CSI camera enabled in ${CONFIG_TXT}"
else
  echo "  ✓ CSI camera already enabled"
fi

# ── 5. Python venv + dependencies ───────────────────────────
echo "[5/7] Setting up Python virtual environment…"
if [[ ! -d "${VENV_DIR}" ]]; then
  sudo -u "${RUN_USER}" python3 -m venv "${VENV_DIR}"
  echo "  ✓ Created venv at ${VENV_DIR}"
else
  echo "  ✓ Venv already exists"
fi

echo "  Installing Python packages (this may take a while)…"
sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip wheel setuptools -q
sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt" -q
# CM4-specific packages
sudo -u "${RUN_USER}" "${VENV_DIR}/bin/pip" install \
  adafruit-blinka adafruit-circuitpython-vl53l0x pyserial -q
echo "  ✓ Python dependencies installed"

# ── 6. Verify hardware pins ─────────────────────────────────
echo "[6/7] Verifying hardware pins from pins.json…"
sudo -u "${RUN_USER}" "${VENV_DIR}/bin/python3" "${PROJECT_DIR}/deploy/verify_pins.py" || true

# ── 7. Install systemd service ──────────────────────────────
echo "[7/7] Installing systemd service…"
bash "${SCRIPT_DIR}/install_cm4_service.sh" "voice currency object"

# ── Log rotation ────────────────────────────────────────────
cat > /etc/logrotate.d/smartvision <<'LOGROTATE'
/var/log/smartvision.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    maxsize 50M
}
LOGROTATE
echo "  ✓ Log rotation configured (7 days, max 50MB)"

# ── Done ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✓ Setup complete!"
echo "============================================================"
echo ""
echo "  Service status:  sudo systemctl status smartvision"
echo "  Live logs:       tail -f /var/log/smartvision.log"
echo "  Self-test:       ${VENV_DIR}/bin/python3 main.py --self-test --target cm4"
echo ""

if [[ "${NEEDS_REBOOT}" == "true" ]]; then
  echo "  ⚠  REBOOT REQUIRED for I2C/I2S/Camera changes."
  echo ""
  read -p "  Reboot now? [y/N] " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Rebooting…"
    reboot
  else
    echo "  Please reboot manually: sudo reboot"
  fi
else
  echo "  No reboot needed. All hardware was already enabled."
fi
