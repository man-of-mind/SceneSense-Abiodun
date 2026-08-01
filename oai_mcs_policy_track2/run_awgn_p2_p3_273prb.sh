#!/usr/bin/env bash
# Diagnostic-only Track 2 bad-channel sanity: compare MCS policies under the
# existing RFsim AWGN 273PRB channelmod setup.
#
# Do not use this as the official Track 2 bad-channel comparison against the
# 106PRB clear-channel baseline. For official results, use:
#
#   abiodun/oai_mcs_policy_track2/run_awgn_106prb_policies.sh
#
# Purpose:
#   - P2 should keep MCS high and may incur retransmission pressure.
#   - P3/P4-style AIMD should keep the good-channel gain when clean, but
#     back off when BLER/retransmission evidence appears.
#
# Default is a short diagnostic: 300 requested frames at 10 FPS.  The CARLA
# frontend is closed-loop, so wall-clock runtime is longer than 30 seconds.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2

BASE_BATCH_ID="${BASE_BATCH_ID:-track2_awgn273_$(date +%Y%m%d_%H%M%S)}"
FRONT_DURATION_S="${FRONT_DURATION_S:-30}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1200}"
POLICIES="${POLICIES:-vanilla hold aimd aimd_cap}"
UE_TRACE_PROFILE="${TTRACER_UE_PROFILE:-all}"
GNB_TRACE_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
AIMD_CAP_DROP="${AIMD_CAP_DROP:-3}"

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
  RADAR_RASTERIZER=fast
  QUANTIZATION_MODE=per_channel_uint8
  ROI_THRESHOLD=0.0
  ENTROPY_CODER=zstd
  ZSTD_LEVEL=3
)

run_one() {
  local label="$1"
  local hold="0"
  local mcs_policy=""
  local aimd_max_drop=""

  case "${label}" in
    vanilla)
      hold="0"
      mcs_policy="vanilla"
      ;;
    hold)
      hold="1"
      mcs_policy=""
      ;;
    aimd)
      hold="0"
      mcs_policy="aimd"
      ;;
    aimd_cap)
      hold="0"
      mcs_policy="aimd"
      aimd_max_drop="${AIMD_CAP_DROP}"
      ;;
    *)
      echo "[track2-awgn] ERROR: unknown policy '${label}' (use vanilla, hold, aimd, aimd_cap)" >&2
      return 2
      ;;
  esac

  local condition="oai_bw273_awgn_track2_${label}"
  local batch="${BASE_BATCH_ID}_${label}"
  local run_group="downlink_${condition}_fps10_${batch}"
  local log_dir="${AB}/metrics_logs/carla_oai_ttracer/${run_group}/logs"

  echo "[track2-awgn] ===== ${label}: HOLD_MCS_FEW_SAMPLES=${hold}, MCS_POLICY=${mcs_policy:-legacy}, AIMD_MAX_DROP=${aimd_max_drop:-uncapped}, condition=${condition}, batch=${batch} ====="
  env "${COMMON_ENV[@]}" \
    CONDITION="${condition}" \
    BATCH_ID="${batch}" \
    HOLD_MCS_FEW_SAMPLES="${hold}" \
    MCS_POLICY="${mcs_policy}" \
    AIMD_MAX_DROP="${aimd_max_drop}" \
    bash downlink_latency_fps/run_oai_bw273_ttracer_10fps.sh

  mkdir -p "${log_dir}"
  echo "[track2-awgn] analyzing uplink layer latency for ${run_group}"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${run_group}" \
    > "${log_dir}/analyze_uplink_layer_latency_stdout.log" 2>&1 || {
      echo "[track2-awgn] WARN: layer latency analyzer failed for ${run_group}; see ${log_dir}/analyze_uplink_layer_latency_stdout.log"
    }
}

echo "[track2-awgn] Base batch: ${BASE_BATCH_ID}"
echo "[track2-awgn] Policies: ${POLICIES}"
echo "[track2-awgn] AIMD cap drop: ${AIMD_CAP_DROP}"
echo "[track2-awgn] Requested frames per run: $((FRONT_DURATION_S * 10))"

for policy in ${POLICIES}; do
  run_one "${policy}" || exit $?
done

echo "[track2-awgn] done. Base batch: ${BASE_BATCH_ID}"
