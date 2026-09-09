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

# Structured, retried NVIDIA-runtime probe.
#
# This was `sudo docker info 2>/dev/null | grep -qi "nvidia"`: one shot, a
# substring match against free-text output, stderr discarded. On 2026-09-07 it
# killed the 288-cell campaign at cell a61 while the runtime was registered the
# whole time -- dockerd logged an internal plugin-refcount error at that instant,
# `docker info` returned nothing, and the discarded stderr left no evidence.
# The probe now reads the exact Runtimes map key, retries a transient daemon
# fault with a per-attempt timeout and bounded backoff, and prints every
# attempt's combined output so a real failure is diagnosable from the launcher
# log. It grants no capability it did not previously require.
NVIDIA_RUNTIME_PROBE_ATTEMPTS="${NVIDIA_RUNTIME_PROBE_ATTEMPTS:-5}"
NVIDIA_RUNTIME_PROBE_TIMEOUT_S="${NVIDIA_RUNTIME_PROBE_TIMEOUT_S:-20}"
NVIDIA_RUNTIME_PROBE_NAME="nvidia"
nvidia_runtime_present=0
nvidia_probe_backoff_s=1
for nvidia_probe_attempt in $(seq 1 "${NVIDIA_RUNTIME_PROBE_ATTEMPTS}"); do
    nvidia_probe_rc=0
    nvidia_probe_output="$(timeout "${NVIDIA_RUNTIME_PROBE_TIMEOUT_S}" \
        sudo docker info --format '{{range $name, $runtime := .Runtimes}}{{$name}}
{{end}}' 2>&1)" || nvidia_probe_rc=$?
    if [ "${nvidia_probe_rc}" -eq 0 ] \
        && printf '%s\n' "${nvidia_probe_output}" | grep -Fxq "${NVIDIA_RUNTIME_PROBE_NAME}"; then
        nvidia_runtime_present=1
        echo "[fusion_back_up] nvidia runtime probe: attempt ${nvidia_probe_attempt} ok;" \
            "runtimes=$(printf '%s' "${nvidia_probe_output}" | tr '\n' ' ')"
        break
    fi
    echo "[fusion_back_up] nvidia runtime probe: attempt ${nvidia_probe_attempt}/${NVIDIA_RUNTIME_PROBE_ATTEMPTS}" \
        "failed rc=${nvidia_probe_rc}; output=<<${nvidia_probe_output}>>"
    if [ "${nvidia_probe_attempt}" -lt "${NVIDIA_RUNTIME_PROBE_ATTEMPTS}" ]; then
        sleep "${nvidia_probe_backoff_s}"
        nvidia_probe_backoff_s=$((nvidia_probe_backoff_s * 2))
        if [ "${nvidia_probe_backoff_s}" -gt 8 ]; then
            nvidia_probe_backoff_s=8
        fi
    fi
done
if [ "${nvidia_runtime_present}" -ne 1 ]; then
    echo "[fusion_back_up] ERROR: Docker does not report the '${NVIDIA_RUNTIME_PROBE_NAME}' runtime" \
        "after ${NVIDIA_RUNTIME_PROBE_ATTEMPTS} probes."
    echo "[fusion_back_up] Install and configure nvidia-container-toolkit first."
    exit 1
fi

if [ -n "${SPLITFUSION_EDGE_STATE_ROOT:-}" ]; then
    if [ ! -d "${SPLITFUSION_EDGE_STATE_ROOT}" ] || [ ! -w "${SPLITFUSION_EDGE_STATE_ROOT}" ]; then
        echo "[fusion_back_up] ERROR: supplied edge state root must be an existing writable directory."
        exit 1
    fi
    SPLITFUSION_EDGE_STATE_ROOT="$(realpath -e "${SPLITFUSION_EDGE_STATE_ROOT}")"
else
    SPLITFUSION_EDGE_STATE_ROOT="$(dirname "$0")/../torch_cache"
    mkdir -p "${SPLITFUSION_EDGE_STATE_ROOT}"
    SPLITFUSION_EDGE_STATE_ROOT="$(realpath -e "${SPLITFUSION_EDGE_STATE_ROOT}")"
fi
export SPLITFUSION_EDGE_STATE_ROOT

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
export FUSION_BACK_SCRIPT="${FUSION_BACK_SCRIPT:-/work/abiodun/carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py}"
export FUSION_BACK_CHECKPOINT="${FUSION_BACK_CHECKPOINT:-/work/abiodun/checkpoints/fusion_object_best.pt}"
export FUSION_QUANTIZATION_MODE="${FUSION_QUANTIZATION_MODE:-per_channel_uint8}"
export FUSION_ENTROPY_CODER="${FUSION_ENTROPY_CODER:-zstd}"  # must match the front codec; zstd is deployed (2026-07-22)
export FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY:-30}"
export FUSION_BACK_EXTRA_ARGS="${FUSION_BACK_EXTRA_ARGS:-}"
export FUSION_REMOTE_PORT_1="${FUSION_REMOTE_PORT_1:-51002}"
export FUSION_REMOTE_SOURCE_PORT_1="${FUSION_REMOTE_SOURCE_PORT_1:-51003}"
export FUSION_CAMERA_RESULT_PORT_1="${FUSION_CAMERA_RESULT_PORT_1:-51004}"
export FUSION_REMOTE_PORT_2="${FUSION_REMOTE_PORT_2:-51102}"
export FUSION_REMOTE_SOURCE_PORT_2="${FUSION_REMOTE_SOURCE_PORT_2:-51103}"
export FUSION_CAMERA_RESULT_PORT_2="${FUSION_CAMERA_RESULT_PORT_2:-51104}"

cd "${RX_DIR}"
echo "[fusion_back_up] docker compose up -d --build (RGB+radar fusion back-half, GPU)"
echo "[fusion_back_up] remote UE IP worker 1: ${FUSION_BACK_REMOTE_HOST_1}"
echo "[fusion_back_up] remote UE IP worker 2: ${FUSION_BACK_REMOTE_HOST_2}"
echo "[fusion_back_up] dual workers: ${FUSION_BACK_DUAL}"
echo "[fusion_back_up] script: ${FUSION_BACK_SCRIPT}"
echo "[fusion_back_up] checkpoint: ${FUSION_BACK_CHECKPOINT}"
echo "[fusion_back_up] edge state root: ${SPLITFUSION_EDGE_STATE_ROOT}"
echo "[fusion_back_up] back log every: ${FUSION_BACK_LOG_EVERY}"
echo "[fusion_back_up] worker 1 ports: recv ${FUSION_REMOTE_PORT_1}, send ${FUSION_REMOTE_SOURCE_PORT_1}->${FUSION_CAMERA_RESULT_PORT_1}"
echo "[fusion_back_up] worker 2 ports: recv ${FUSION_REMOTE_PORT_2}, send ${FUSION_REMOTE_SOURCE_PORT_2}->${FUSION_CAMERA_RESULT_PORT_2}"
sudo FUSION_BACK_BIND_HOST="${FUSION_BACK_BIND_HOST}" \
    FUSION_BACK_REMOTE_HOST="${FUSION_BACK_REMOTE_HOST}" \
    FUSION_BACK_REMOTE_HOST_1="${FUSION_BACK_REMOTE_HOST_1}" \
    FUSION_BACK_REMOTE_HOST_2="${FUSION_BACK_REMOTE_HOST_2}" \
    FUSION_BACK_DEVICE="${FUSION_BACK_DEVICE}" \
    FUSION_BACK_SCRIPT="${FUSION_BACK_SCRIPT}" \
    FUSION_BACK_CHECKPOINT="${FUSION_BACK_CHECKPOINT}" \
    FUSION_QUANTIZATION_MODE="${FUSION_QUANTIZATION_MODE}" \
    FUSION_ENTROPY_CODER="${FUSION_ENTROPY_CODER}" \
    FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY}" \
    FUSION_BACK_DUAL="${FUSION_BACK_DUAL}" \
    FUSION_BACK_EXTRA_ARGS="${FUSION_BACK_EXTRA_ARGS}" \
    FUSION_REMOTE_PORT_1="${FUSION_REMOTE_PORT_1}" \
    FUSION_REMOTE_SOURCE_PORT_1="${FUSION_REMOTE_SOURCE_PORT_1}" \
    FUSION_CAMERA_RESULT_PORT_1="${FUSION_CAMERA_RESULT_PORT_1}" \
    FUSION_REMOTE_PORT_2="${FUSION_REMOTE_PORT_2}" \
    FUSION_REMOTE_SOURCE_PORT_2="${FUSION_REMOTE_SOURCE_PORT_2}" \
    FUSION_CAMERA_RESULT_PORT_2="${FUSION_CAMERA_RESULT_PORT_2}" \
    SPLITFUSION_EDGE_STATE_ROOT="${SPLITFUSION_EDGE_STATE_ROOT}" \
    docker compose -f docker-compose.yaml -f docker-compose.fusion-back.yaml up -d --build --force-recreate

echo "[fusion_back_up] container state:"
sudo docker ps --filter "name=oai-perception-rx" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "[fusion_back_up] tail logs with:   sudo docker logs -f oai-perception-rx"
