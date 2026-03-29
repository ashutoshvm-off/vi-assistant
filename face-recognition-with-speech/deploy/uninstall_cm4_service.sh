#!/usr/bin/env bash
# ============================================================
# Smart Vision — Uninstall systemd service
# Usage: sudo ./deploy/uninstall_cm4_service.sh
# ============================================================
set -euo pipefail

SERVICE_NAME="smartvision.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

echo "[INFO] Stopping and disabling ${SERVICE_NAME}"
sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true

if [[ -f "${SERVICE_PATH}" ]]; then
  sudo rm -f "${SERVICE_PATH}"
  echo "[OK] Service file removed"
fi

# Remove log rotation config
if [[ -f /etc/logrotate.d/smartvision ]]; then
  sudo rm -f /etc/logrotate.d/smartvision
  echo "[OK] Log rotation config removed"
fi

sudo systemctl daemon-reload

echo "[OK] Service uninstalled."
echo "[INFO] Logs are still at /var/log/smartvision.log (delete manually if desired)"
