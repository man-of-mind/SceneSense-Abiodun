#!/usr/bin/env bash
# Start oai-perception-rx as the Step-1 downlink/FPS study back-half.
#
# This intentionally uses staleness/carla_fusion_staleness_scenario.py rather
# than the older pole-stream back script, because Step 1 needs the newer
# tail-result/downlink timing fields in the returned payload.
set -euo pipefail

source "$(dirname "$0")/config.env"

export FUSION_BACK_DUAL="${FUSION_BACK_DUAL:-0}"
export FUSION_BACK_SCRIPT="${FUSION_BACK_SCRIPT:-/work/abiodun/staleness/carla_fusion_staleness_scenario.py}"
export FUSION_BACK_CHECKPOINT="${FUSION_BACK_CHECKPOINT:-/work/abiodun/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt}"
export FUSION_QUANTIZATION_MODE="${FUSION_QUANTIZATION_MODE:-per_channel_uint8}"
export FUSION_ENTROPY_CODER="${FUSION_ENTROPY_CODER:-zlib}"
export FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY:-100}"
export FUSION_BACK_REMOTE_HOST="${FUSION_BACK_REMOTE_HOST:-${OAI_UE_IP}}"
export FUSION_BACK_REMOTE_HOST_1="${FUSION_BACK_REMOTE_HOST_1:-${FUSION_BACK_REMOTE_HOST}}"
export FUSION_REMOTE_PORT_1="${FUSION_REMOTE_PORT_1:-51002}"
export FUSION_REMOTE_SOURCE_PORT_1="${FUSION_REMOTE_SOURCE_PORT_1:-51003}"
export FUSION_CAMERA_RESULT_PORT_1="${FUSION_CAMERA_RESULT_PORT_1:-51004}"

"$(dirname "$0")/receiver_container_fusion_back_up.sh"
