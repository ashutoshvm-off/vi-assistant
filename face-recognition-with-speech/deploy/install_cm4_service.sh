#!/usr/bin/env bash
# ============================================================
# Smart Vision — Install systemd service for CM4
#
# Usage:
#   sudo ./deploy/install_cm4_service.sh                     # default: all components
#   sudo ./deploy/install_cm4_service.sh "voice currency"    # selected components
# ============================================================
set -euo pipefail

SERVICE_NAME="smartvision.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"
ENV_FILE="${PROJECT_DIR}/.env"

# Prefer venv Python, fall back to system Python
if [[ -x "${PROJECT_DIR}/.venv/bin/python3" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python3"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python3 not found"
  exit 1
fi

COMPONENTS="${1:-face voice currency object}"
OCR_KEY="${OCR_SPACE_API_KEY:-helloworld}"

# Read additional env vars from .env file  (key=value lines, no export)
ENV_LINES=""
if [[ -f "${ENV_FILE}" ]]; then
  while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ -z "${key}" || "${key}" =~ ^# ]] && continue
    # Strip surrounding whitespace
    key="$(echo "${key}" | xargs)"
    value="$(echo "${value}" | xargs)"
    ENV_LINES="${ENV_LINES}Environment=${key}=${value}\n"
  done < "${ENV_FILE}"
fi

echo "[INFO] Installing ${SERVICE_NAME}"
echo "[INFO] Project : ${PROJECT_DIR}"
echo "[INFO] User    : ${RUN_USER}"
echo "[INFO] Python  : ${PYTHON_BIN}"
echo "[INFO] Comps   : ${COMPONENTS}"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=Smart Vision Assistance System (CM4 — 24/7)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}

# Core environment
Environment=PYTHONUNBUFFERED=1
Environment=OCR_SPACE_API_KEY=${OCR_KEY}
$(echo -e "${ENV_LINES}")

# Main process
ExecStartPre=-${PYTHON_BIN} ${PROJECT_DIR}/deploy/verify_pins.py
ExecStart=${PYTHON_BIN} ${PROJECT_DIR}/main.py --module coordinated --target cm4 --components ${COMPONENTS}

# Restart policy — always restart, with escalating backoff
Restart=always
RestartSec=5

# Watchdog — if no activity for 120s, systemd restarts the service
WatchdogSec=120

# Graceful shutdown
KillSignal=SIGINT
TimeoutStopSec=30

# Logging
StandardOutput=append:/var/log/smartvision.log
StandardError=append:/var/log/smartvision.log

[Install]
WantedBy=multi-user.target
EOF

# Create log file with correct ownership
touch /var/log/smartvision.log
chown "${RUN_USER}:${RUN_USER}" /var/log/smartvision.log

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "[OK] Service installed, enabled, and started."
echo "[OK] Auto-starts on boot / power loss / crash."
echo ""
echo "  Status  : sudo systemctl status ${SERVICE_NAME}"
echo "  Logs    : tail -f /var/log/smartvision.log"
echo "  Stop    : sudo systemctl stop ${SERVICE_NAME}"
echo "  Restart : sudo systemctl restart ${SERVICE_NAME}"
echo ""
