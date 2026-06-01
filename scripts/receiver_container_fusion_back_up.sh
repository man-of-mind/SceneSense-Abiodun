#!/usr/bin/env bash
# Build (if needed) and start oai-perception-rx as the RGB+radar fusion
# split-inference back-half with GPU access. Prerequisites:
#   1. OAI core network is up (./cn_start.sh)
#   2. Docker has NVIDIA GPU support (nvidia-container-toolkit configured)
set -euo pipefail
source "$(dirname "$0")/config.env"

RX_DIR="$(dirname "$0")/../receiver_container"

if ! sudo docker network inspect oai-cn5g-public-net >/dev/null 2>&1; then
    echo "[fusion_back_up] ERROR: oai-cn5g-public-net not found. Run cn_start.sh first."
    exit 1
fi

if ! sudo docker info 2>/dev/null | grep -qi "nvidia"; then
    echo "[fusion_back_up] ERROR: Docker does not report an NVIDIA runtime/CDI setup."
    echo "[fusion_back_up] Install and configure nvidia-container-toolkit first."
    exit 1
fi

mkdir -p "$(dirname "$0")/../torch_cache"

export FUSION_BACK_BIND_HOST="${FUSION_BACK_BIND_HOST:-0.0.0.0}"
export FUSION_BACK_REMOTE_HOST="${FUSION_BACK_REMOTE_HOST:-${OAI_UE_IP}}"
export FUSION_BACK_DUAL="${FUSION_BACK_DUAL:-1}"
export FUSION_BACK_REMOTE_HOST_1="${FUSION_BACK_REMOTE_HOST_1:-${FUSION_BACK_REMOTE_HOST}}"
if [ -z "${FUSION_BACK_REMOTE_HOST_2:-}" ]; then
    if [ "${FUSION_BACK_DUAL}" = "1" ] && ip -br addr show "${OAI_UE2_IFACE}" >/dev/null 2>&1; then
        export FUSION_BACK_REMOTE_HOST_2="${OAI_UE2_IP}"
    else
        export FUSION_BACK_REMOTE_HOST_2="${FUSION_BACK_REMOTE_HOST}"
    fi
fi
export FUSION_BACK_DEVICE="${FUSION_BACK_DEVICE:-cuda}"
export FUSION_BACK_CHECKPOINT="${FUSION_BACK_CHECKPOINT:-/work/abiodun/checkpoints/fusion_object_best.pt}"
export FUSION_QUANTIZATION_MODE="${FUSION_QUANTIZATION_MODE:-per_channel_uint8}"
export FUSION_ENTROPY_CODER="${FUSION_ENTROPY_CODER:-zlib}"
export FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY:-30}"
export FUSION_BACK_EXTRA_ARGS="${FUSION_BACK_EXTRA_ARGS:-}"

cd "${RX_DIR}"
echo "[fusion_back_up] docker compose up -d --build (RGB+radar fusion back-half, GPU)"
echo "[fusion_back_up] remote UE IP worker 1: ${FUSION_BACK_REMOTE_HOST_1}"
echo "[fusion_back_up] remote UE IP worker 2: ${FUSION_BACK_REMOTE_HOST_2}"
echo "[fusion_back_up] dual workers: ${FUSION_BACK_DUAL}"
echo "[fusion_back_up] back log every: ${FUSION_BACK_LOG_EVERY}"
sudo FUSION_BACK_BIND_HOST="${FUSION_BACK_BIND_HOST}" \
    FUSION_BACK_REMOTE_HOST="${FUSION_BACK_REMOTE_HOST}" \
    FUSION_BACK_REMOTE_HOST_1="${FUSION_BACK_REMOTE_HOST_1}" \
    FUSION_BACK_REMOTE_HOST_2="${FUSION_BACK_REMOTE_HOST_2}" \
    FUSION_BACK_DEVICE="${FUSION_BACK_DEVICE}" \
    FUSION_BACK_CHECKPOINT="${FUSION_BACK_CHECKPOINT}" \
    FUSION_QUANTIZATION_MODE="${FUSION_QUANTIZATION_MODE}" \
    FUSION_ENTROPY_CODER="${FUSION_ENTROPY_CODER}" \
    FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY}" \
    FUSION_BACK_DUAL="${FUSION_BACK_DUAL}" \
    FUSION_BACK_EXTRA_ARGS="${FUSION_BACK_EXTRA_ARGS}" \
    docker compose -f docker-compose.yaml -f docker-compose.fusion-back.yaml up -d --build --force-recreate

echo "[fusion_back_up] container state:"
sudo docker ps --filter "name=oai-perception-rx" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "[fusion_back_up] tail logs with:   sudo docker logs -f oai-perception-rx"
