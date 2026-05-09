#!/usr/bin/env bash
# One-time VPS preparation for systemd + venv deploy path used by GitHub Actions.
# Run as root on Ubuntu 22.04/24.04 LTS.

set -euo pipefail

APP_USER="${APP_USER:-cse354}"
APP_DIR="${APP_DIR:-/opt/cse354-api}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (sudo)."
  exit 1
fi

apt-get update -qq
apt-get install -y python3 python3-venv python3-pip nginx redis-server git rsync

if ! id -u "$APP_USER" &>/dev/null; then
  useradd --system --home "$APP_DIR" --create-home "$APP_USER"
fi

mkdir -p "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "Copy systemd unit:"
echo "  cp deploy/systemd/cse354-master-api.service /etc/systemd/system/"
echo "  systemctl daemon-reload && systemctl enable --now cse354-master-api"
echo ""
echo "Place .env at $APP_DIR/.env with REDIS_URL=..."
echo "First deploy will create venv via CI or: sudo -u $APP_USER python3 -m venv $APP_DIR/.venv"
