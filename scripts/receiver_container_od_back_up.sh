#!/usr/bin/env bash
# Build (if needed) and start oai-perception-rx as the OD split-inference
# back-half with GPU access. Prerequisites:
#   1. OAI core network is up (./cn_start.sh)
#   2. Docker has NVIDIA GPU support (nvidia-container-toolkit configured)
set -euo pipefail
source "$(dirname "$0")/config.env"

RX_DIR="$(dirname "$0")/../receiver_container"

if ! sudo docker network inspect oai-cn5g-public-net >/dev/null 2>&1; then
    echo "[od_back_up] ERROR: oai-cn5g-public-net not found. Run cn_start.sh first."
    exit 1
fi

if ! sudo docker info 2>/dev/null | grep -qi "nvidia"; then
    echo "[od_back_up] ERROR: Docker does not report an NVIDIA runtime/CDI setup."
    echo "[od_back_up] Install and configure nvidia-container-toolkit before starting Phase 2."
    exit 1
fi

mkdir -p "$(dirname "$0")/../torch_cache"

export OD_BACK_BIND_HOST="${OD_BACK_BIND_HOST:-0.0.0.0}"
export OD_BACK_REMOTE_HOST="${OD_BACK_REMOTE_HOST:-${OAI_UE_IP}}"
export OD_BACK_DEVICE="${OD_BACK_DEVICE:-cuda}"
export OD_BACK_EXTRA_ARGS="${OD_BACK_EXTRA_ARGS:-}"

cd "${RX_DIR}"
echo "[od_back_up] docker compose up -d --build (OD back-half, GPU)"
echo "[od_back_up] remote UE IP: ${OD_BACK_REMOTE_HOST}"
sudo OD_BACK_BIND_HOST="${OD_BACK_BIND_HOST}" \
    OD_BACK_REMOTE_HOST="${OD_BACK_REMOTE_HOST}" \
    OD_BACK_DEVICE="${OD_BACK_DEVICE}" \
    OD_BACK_EXTRA_ARGS="${OD_BACK_EXTRA_ARGS}" \
    docker compose -f docker-compose.yaml -f docker-compose.od-back.yaml up -d --build

echo "[od_back_up] container state:"
sudo docker ps --filter "name=oai-perception-rx" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "[od_back_up] tail logs with:   sudo docker logs -f oai-perception-rx"
