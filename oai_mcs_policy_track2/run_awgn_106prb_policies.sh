#!/usr/bin/env bash
# Track 2 official bad-channel policy gate on the default 106PRB OAI config.
#
# This is the fair counterpart to the 106PRB clear-channel Track 2 runs:
# keep PRB/TDD/UE launch/model/payload fixed and vary only:
#   - RFsim channel condition: AWGN enabled here
#   - MCS policy: vanilla, hold-few, uncapped AIMD, capped AIMD, SINR-driven
#
# Default is a short diagnostic: 300 requested frames at 10 FPS. The CARLA
# frontend is closed-loop, so wall-clock runtime is longer than 30 seconds.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${AB}" || exit 2

BASE_BATCH_ID="${BASE_BATCH_ID:-track2_awgn106_$(date +%Y%m%d_%H%M%S)}"
FRONT_DURATION_S="${FRONT_DURATION_S:-30}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1200}"
POLICIES="${POLICIES:-vanilla hold aimd aimd_cap}"
UE_TRACE_PROFILE="${TTRACER_UE_PROFILE:-all}"
GNB_TRACE_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
AIMD_CAP_DROP="${AIMD_CAP_DROP:-3}"
AWGN_PROFILE="${AWGN_PROFILE:-mild}"
DRY_RUN="${DRY_RUN:-0}"

case "${AWGN_PROFILE}" in
  mild)
    AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:--10}"
    AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-0}"
    ;;
  medium)
    AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:--5}"
    AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-0}"
    ;;
  strong)
    AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:--4}"
    AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-0}"
    ;;
  harsh)
    AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:-0}"
    AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-0}"
    ;;
  edge)
    AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:-5}"
    AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-0}"
    ;;
  *)
    echo "[track2-awgn106] ERROR: unknown AWGN_PROFILE='${AWGN_PROFILE}' (use mild, medium, strong, harsh, edge)" >&2
    exit 2
    ;;
esac

GNB_CONF_DEFAULT="${GNB_CONF_DEFAULT:-gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_${AWGN_PROFILE}.conf}"
UE_CONF_DEFAULT="${UE_CONF_DEFAULT:-ue.awgn_${AWGN_PROFILE}.conf}"
CONDITION_PREFIX="${CONDITION_PREFIX:-oai_default106_awgn_${AWGN_PROFILE}_track2}"
TRANSPORT_LABEL="${TRANSPORT_LABEL:-oai_default106_awgn_${AWGN_PROFILE}_noae_ttracer}"

COMMON_ENV=(
  RFSIM_CHANMOD=1
  GNB_CONF_DEFAULT="${GNB_CONF_DEFAULT}"
  UE_CONF_DEFAULT="${UE_CONF_DEFAULT}"
  AWGN_PROFILE="${AWGN_PROFILE}"
  AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB}"
  AWGN_PLOSS_DB="${AWGN_PLOSS_DB}"
  UE_PRB=106
  UE_DL_FREQ=3619200000
  FRONT_DURATION_S="${FRONT_DURATION_S}"
  TTRACER_DURATION_S="${TTRACER_DURATION_S}"
  TTRACER_UE_PROFILE="${UE_TRACE_PROFILE}"
  TTRACER_GNB_PROFILE="${GNB_TRACE_PROFILE}"
  RECORD_GNB=1
  FORCE_UL_MCS=
  RADAR_RASTERIZER=fast
  QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}"
  ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}"
  ENTROPY_CODER="${ENTROPY_CODER:-zstd}"
  ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
  TRANSPORT_LABEL="${TRANSPORT_LABEL}"
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
    sinr)
      hold="0"
      mcs_policy="sinr"
      ;;
    *)
      echo "[track2-awgn106] ERROR: unknown policy '${label}' (use vanilla, hold, aimd, aimd_cap, sinr)" >&2
      return 2
      ;;
  esac

  local condition="${CONDITION_PREFIX}_${label}"
  local batch="${BASE_BATCH_ID}_${label}"

  echo "[track2-awgn106] ===== ${label}: HOLD_MCS_FEW_SAMPLES=${hold}, MCS_POLICY=${mcs_policy:-legacy}, AIMD_MAX_DROP=${aimd_max_drop:-uncapped}, condition=${condition}, batch=${batch} ====="
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  env "${COMMON_ENV[@]}" \
    CONDITION="${condition}" \
    BATCH_ID="${batch}" \
    HOLD_MCS_FEW_SAMPLES="${hold}" \
    MCS_POLICY="${mcs_policy}" \
    AIMD_MAX_DROP="${aimd_max_drop}" \
    bash downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
}

echo "[track2-awgn106] Base batch: ${BASE_BATCH_ID}"
echo "[track2-awgn106] AWGN profile: ${AWGN_PROFILE} (noise_power_dB=${AWGN_NOISE_POWER_DB}, ploss_dB=${AWGN_PLOSS_DB})"
echo "[track2-awgn106] gNB config: ${GNB_CONF_DEFAULT}"
echo "[track2-awgn106] UE config: ${UE_CONF_DEFAULT}"
echo "[track2-awgn106] Policies: ${POLICIES}"
echo "[track2-awgn106] AIMD cap drop: ${AIMD_CAP_DROP}"
echo "[track2-awgn106] Requested frames per run: $((FRONT_DURATION_S * 10))"

for policy in ${POLICIES}; do
  run_one "${policy}" || exit $?
done

echo "[track2-awgn106] done. Base batch: ${BASE_BATCH_ID}"
