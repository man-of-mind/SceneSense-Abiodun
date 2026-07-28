#!/usr/bin/env bash
# Compare vanilla OAI BLER/OLLA against the SceneSense hold-few-samples logic
# under the same RFsim AWGN channelmod condition.
#
# This uses the validated 273PRB CARLA/T-tracer runner with:
#   - no-AE baseline model
#   - zstd entropy coding
#   - ROI 0.0 / per-channel uint8 (~1 MB feature payload)
#   - 10 FPS one-loop CARLA frontend
#
# The same patched nr-softmodem binary is used for both runs. Vanilla behavior
# is selected by HOLD_MCS_FEW_SAMPLES=0; patched behavior is selected by =1.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2

BASE_BATCH_ID="${BASE_BATCH_ID:-awgn273_compare_$(date +%Y%m%d_%H%M%S)}"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1800}"
UE_TRACE_PROFILE="${TTRACER_UE_PROFILE:-all}"
GNB_TRACE_PROFILE="${TTRACER_GNB_PROFILE:-latency}"

COMMON_ENV=(
  RFSIM_CHANMOD=1
  GNB_CONF_273=gnb.sa.band78.fr1.273PRB.scenesense_rfsim.awgn.conf
  UE_CONF_273=ue.awgn.conf
  FRONT_DURATION_S="${FRONT_DURATION_S}"
  TTRACER_DURATION_S="${TTRACER_DURATION_S}"
  TTRACER_UE_PROFILE="${UE_TRACE_PROFILE}"
  TTRACER_GNB_PROFILE="${GNB_TRACE_PROFILE}"
  RECORD_GNB=1
  FORCE_UL_MCS=
  QUANTIZATION_MODE=per_channel_uint8
  ROI_THRESHOLD=0.0
  ENTROPY_CODER=zstd
  ZSTD_LEVEL=3
)

run_one() {
  local label="$1"
  local hold="$2"
  local condition="oai_bw273_awgn_${label}"
  local batch="${BASE_BATCH_ID}_${label}"
  echo "[awgn-compare] ===== ${label}: HOLD_MCS_FEW_SAMPLES=${hold}, condition=${condition}, batch=${batch} ====="
  env "${COMMON_ENV[@]}" \
    CONDITION="${condition}" \
    BATCH_ID="${batch}" \
    HOLD_MCS_FEW_SAMPLES="${hold}" \
    bash downlink_latency_fps/run_oai_bw273_ttracer_10fps.sh

  local run_group="downlink_${condition}_fps10_${batch}"
  local log_dir="${AB}/metrics_logs/carla_oai_ttracer/${run_group}/logs"
  mkdir -p "${log_dir}"
  echo "[awgn-compare] analyzing layer latency for ${run_group}"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${run_group}" \
    > "${log_dir}/analyze_uplink_layer_latency_stdout.log" 2>&1 || {
      echo "[awgn-compare] WARN: layer latency analyzer failed for ${run_group}; see ${log_dir}/analyze_uplink_layer_latency_stdout.log"
    }
}

run_one vanilla 0
run_one hold 1

echo "[awgn-compare] done. Base batch: ${BASE_BATCH_ID}"
