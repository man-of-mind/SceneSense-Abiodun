#!/usr/bin/env bash
# Build (if needed) and start the oai-perception-rx container.
# Prerequisite: OAI core network must be UP (./cn_start.sh) so the
# oai-cn5g-public-net network exists.
set -euo pipefail
source "$(dirname "$0")/config.env"

RX_DIR="$(dirname "$0")/../receiver_container"

# Allow container apps to draw on the host X display.
# Scoped to local connections so it doesn't open the display to the network.
echo "[rx_up] xhost +local:"
xhost +local: >/dev/null

# Verify the OAI public_net exists (cn_start.sh must have been run first).
if ! sudo docker network inspect oai-cn5g-public-net >/dev/null 2>&1; then
    echo "[rx_up] ERROR: oai-cn5g-public-net not found. Run cn_start.sh first."
    exit 1
fi

cd "${RX_DIR}"
echo "[rx_up] docker compose up -d --build (in ${RX_DIR})"
sudo DISPLAY="${DISPLAY:-:0}" UDP_PORT="${UDP_PORT:-65000}" \
    docker compose up -d --build

echo "[rx_up] container state:"
sudo docker ps --filter "name=oai-perception-rx" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "[rx_up] tail logs with:   sudo docker logs -f oai-perception-rx"
