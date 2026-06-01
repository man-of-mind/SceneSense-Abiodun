#!/usr/bin/env bash
# Stop the oai-perception-rx container.
set -euo pipefail
source "$(dirname "$0")/config.env"

RX_DIR="$(dirname "$0")/../receiver_container"
cd "${RX_DIR}"

echo "[rx_down] docker compose down"
sudo docker compose down

# Tighten X back up; harmless if it was already tight.
echo "[rx_down] xhost -local:"
xhost -local: >/dev/null || true
