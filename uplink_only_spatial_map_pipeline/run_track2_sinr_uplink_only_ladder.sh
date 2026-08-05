#!/usr/bin/env bash
# Run the uplink-only spatial-map CARLA pipeline over 106PRB OAI using the
# SINR-driven UL MCS policy across clear and AWGN channel profiles.
#
# Fairness constraints:
#   - same uplink-only application path for every run
#   - same default 106PRB / 7DL-2UL OAI config family
#   - same no-AE, ROI 0, per-channel uint8, zstd payload knobs
#   - same fast radar rasterizer / optimized sensor-prep path
#   - same T-tracer profiles: UE=all, gNB=latency
#   - only channel profile varies
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${AB}" || exit 2
source scripts/config.env

BASE_BATCH_ID="${BASE_BATCH_ID:-track2_sinr_uplink_only_$(date +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-clear_sinr mild_sinr mid15_sinr strong_sinr}"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-900}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-120}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ENV=(
  UE_PRB=106
  UE_DL_FREQ=3619200000
  TARGET_FPS=10
  FRONT_DURATION_S="${FRONT_DURATION_S}"
  TTRACER_DURATION_S="${TTRACER_DURATION_S}"
  WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES}"
  TTRACER_UE_PROFILE=all
  TTRACER_GNB_PROFILE=latency
  RECORD_GNB=1
  FORCE_UL_MCS=
  HOLD_MCS_FEW_SAMPLES=0
  MCS_POLICY=sinr
  AIMD_MAX_DROP=
  RADAR_RASTERIZER=fast
  QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}"
  ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}"
  ENTROPY_CODER="${ENTROPY_CODER:-zstd}"
  ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
  AE_CHECKPOINT="${AE_CHECKPOINT:-}"
  AE_CHECKPOINT_CONTAINER="${AE_CHECKPOINT_CONTAINER:-}"
  CAPTURE_PIPELINE=1
  CAPTURE_PIPELINE_QUEUE_SIZE=2
  CAPTURE_PIPELINE_DROP_OLDEST=0
  SENSOR_EVERY_TICK=0
)

show_preflight() {
  echo "[uplink-sinr-ladder] Base batch: ${BASE_BATCH_ID}"
  echo "[uplink-sinr-ladder] Runs: ${RUNS}"
  echo "[uplink-sinr-ladder] Front duration per run: ${FRONT_DURATION_S}s"
  echo "[uplink-sinr-ladder] T-tracer duration per run: ${TTRACER_DURATION_S}s"
  echo "[uplink-sinr-ladder] OAI build dir: ${OAI_RAN_BUILD}"
  if [[ -x "${OAI_RAN_BUILD}/nr-softmodem" ]]; then
    stat -c "[uplink-sinr-ladder] nr-softmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-softmodem"
    strings "${OAI_RAN_BUILD}/nr-softmodem" | grep -q 'SCENESENSE_MCS_POLICY' \
      && echo "[uplink-sinr-ladder] nr-softmodem contains SCENESENSE_MCS_POLICY" \
      || echo "[uplink-sinr-ladder] WARN: could not find SCENESENSE_MCS_POLICY string in nr-softmodem"
  else
    echo "[uplink-sinr-ladder] ERROR: missing executable ${OAI_RAN_BUILD}/nr-softmodem" >&2
    return 1
  fi
  if [[ -x "${OAI_RAN_BUILD}/nr-uesoftmodem" ]]; then
    stat -c "[uplink-sinr-ladder] nr-uesoftmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-uesoftmodem"
  else
    echo "[uplink-sinr-ladder] ERROR: missing executable ${OAI_RAN_BUILD}/nr-uesoftmodem" >&2
    return 1
  fi
}

run_one() {
  local label="$1"
  local channel=""
  local gnb_conf=""
  local ue_conf=""
  local rfsim_chanmod="0"
  local awgn_profile="clear"
  local awgn_noise=""
  local awgn_ploss=""

  case "${label}" in
    clear_sinr)
      channel="clear"
      gnb_conf="gnb.sa.band78.fr1.106PRB.usrpb210.conf"
      ue_conf="ue.conf"
      rfsim_chanmod="0"; awgn_profile="clear"
      ;;
    mild_sinr)
      channel="mild_awgn"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"
      ue_conf="ue.awgn_mild.conf"
      rfsim_chanmod="1"; awgn_profile="mild"; awgn_noise="-10"; awgn_ploss="0"
      ;;
    mid15_sinr)
      channel="mid15_awgn"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mid15.conf"
      ue_conf="ue.awgn_mid15.conf"
      rfsim_chanmod="1"; awgn_profile="mid15"; awgn_noise="-8"; awgn_ploss="0"
      ;;
    strong_sinr)
      channel="strong_awgn"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_strong.conf"
      ue_conf="ue.awgn_strong.conf"
      rfsim_chanmod="1"; awgn_profile="strong"; awgn_noise="-4"; awgn_ploss="0"
      ;;
    *)
      echo "[uplink-sinr-ladder] ERROR: unknown run label '${label}'" >&2
      return 2
      ;;
  esac

  local condition="track2_uplink_only_sinr_${channel}"
  local batch="${BASE_BATCH_ID}_${label}"

  echo "[uplink-sinr-ladder] ===== ${label}: channel=${channel}, condition=${condition}, batch=${batch} ====="
  echo "[uplink-sinr-ladder] gNB=${gnb_conf}, UE=${ue_conf}, rfsim_chanmod=${rfsim_chanmod}, awgn_profile=${awgn_profile}, awgn_noise=${awgn_noise:-none}, policy=sinr"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  env "${COMMON_ENV[@]}" \
    CONDITION="${condition}" \
    BATCH_ID="${batch}" \
    GNB_CONF_DEFAULT="${gnb_conf}" \
    UE_CONF_DEFAULT="${ue_conf}" \
    RFSIM_CHANMOD="${rfsim_chanmod}" \
    AWGN_PROFILE="${awgn_profile}" \
    AWGN_NOISE_POWER_DB="${awgn_noise}" \
    AWGN_PLOSS_DB="${awgn_ploss}" \
    bash uplink_only_spatial_map_pipeline/run_track1_oai_default106_ttracer_10fps.sh
}

show_preflight || exit $?

for label in ${RUNS}; do
  run_one "${label}" || exit $?
done

echo "[uplink-sinr-ladder] done."
echo "Suggested summary:"
echo "  python3 abiodun/uplink_only_spatial_map_pipeline/summarize_track2_sinr_uplink_only_ladder.py --base-batch ${BASE_BATCH_ID}"
