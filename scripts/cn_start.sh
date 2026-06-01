#!/usr/bin/env bash
# Start the OAI 5G core network (docker compose).
set -euo pipefail
source "$(dirname "$0")/config.env"

echo "[cn_start] cd ${OAI_CN_DIR}"
cd "${OAI_CN_DIR}"
echo "[cn_start] sudo docker compose up -d"
sudo docker compose up -d

echo "[cn_start] containers:"
sudo docker compose ps
