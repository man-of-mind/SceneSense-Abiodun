#!/usr/bin/env bash
# Stop the OAI 5G core network.
set -euo pipefail
source "$(dirname "$0")/config.env"

cd "${OAI_CN_DIR}"
echo "[cn_stop] sudo docker compose down"
sudo docker compose down
