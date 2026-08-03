#!/usr/bin/env bash
# Fair Track-2 rerun for the MCS/TBS/grant-rate/RLC-drain question.
#
# Goal:
#   Re-run clear, mild-AWGN, and medium-AWGN 106PRB closed-loop CARLA traffic
#   with identical application/model/tracing settings so we can explain why
#   high MCS sometimes still coincides with high latency.
#
# Fairness constraints enforced here:
#   - same default 106PRB / 7DL-2UL OAI path
#   - same closed-loop CARLA frontend
#   - same model/payload knobs: no-AE, ROI 0, per-channel uint8, zstd
#   - same fast radar rasterizer
#   - same tracing profile: UE=all, gNB=latency
#   - only vary channel profile and MCS policy
#
# Default first-pass matrix:
#   clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap
#
# SINR-driven diagnostic labels are also supported:
#   clear_sinr mild_sinr medium_sinr
#
# Medium AWGN is intentionally explicit:
#   RUNS="medium_vanilla medium_aimd_cap"
#
# Dry-run example:
#   DRY_RUN=1 BASE_BATCH_ID=track2_fair_grant_20260801 \
#     bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
#
# Execute example:
#   BASE_BATCH_ID=track2_fair_grant_20260801 FRONT_DURATION_S=30 \
#     bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${AB}" || exit 2
source scripts/config.env

BASE_BATCH_ID="${BASE_BATCH_ID:-track2_fair_grant_$(date +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap}"
FRONT_DURATION_S="${FRONT_DURATION_S:-30}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1200}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-120}"
AIMD_CAP_DROP="${AIMD_CAP_DROP:-3}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ENV=(
  UE_PRB=106
  UE_DL_FREQ=3619200000
  FRONT_DURATION_S="${FRONT_DURATION_S}"
  TTRACER_DURATION_S="${TTRACER_DURATION_S}"
  WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES}"
  TTRACER_UE_PROFILE=all
  TTRACER_GNB_PROFILE=latency
  RECORD_GNB=1
  FORCE_UL_MCS=
  HOLD_MCS_FEW_SAMPLES=0
  RADAR_RASTERIZER=fast
  QUANTIZATION_MODE=per_channel_uint8
  ROI_THRESHOLD=0.0
  ENTROPY_CODER=zstd
  ZSTD_LEVEL=3
  AE_CHECKPOINT=
)

show_preflight() {
  echo "[fair-mcs-grant] Base batch: ${BASE_BATCH_ID}"
  echo "[fair-mcs-grant] Runs: ${RUNS}"
  echo "[fair-mcs-grant] Front duration per run: ${FRONT_DURATION_S}s"
  echo "[fair-mcs-grant] T-tracer duration per run: ${TTRACER_DURATION_S}s"
  echo "[fair-mcs-grant] UE profile: all; gNB profile: latency"
  echo "[fair-mcs-grant] Model/payload: no-AE, ROI=0.0, per_channel_uint8, zstd level 3"
  echo "[fair-mcs-grant] OAI build dir: ${OAI_RAN_BUILD}"
  if [[ -x "${OAI_RAN_BUILD}/nr-softmodem" ]]; then
    stat -c "[fair-mcs-grant] nr-softmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-softmodem"
  else
    echo "[fair-mcs-grant] ERROR: missing executable ${OAI_RAN_BUILD}/nr-softmodem" >&2
    return 1
  fi
  if [[ -x "${OAI_RAN_BUILD}/nr-uesoftmodem" ]]; then
    stat -c "[fair-mcs-grant] nr-uesoftmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-uesoftmodem"
  else
    echo "[fair-mcs-grant] ERROR: missing executable ${OAI_RAN_BUILD}/nr-uesoftmodem" >&2
    return 1
  fi
  sha256sum \
    "${OAI_RAN_CONF}/gnb.sa.band78.fr1.106PRB.usrpb210.conf" \
    "${OAI_RAN_CONF}/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf" \
    "${OAI_RAN_CONF}/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_medium.conf" \
    "${OAI_RAN_CONF}/ue.conf" \
    "${OAI_RAN_CONF}/ue.awgn_mild.conf" \
    "${OAI_RAN_CONF}/ue.awgn_medium.conf" \
    "${OAI_ROOT}/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c" \
    2>/dev/null | sed 's/^/[fair-mcs-grant] sha256 /' || true
}

run_one() {
  local label="$1"
  local channel=""
  local policy=""
  local condition=""
  local gnb_conf=""
  local ue_conf=""
  local rfsim_chanmod="0"
  local awgn_profile="clear"
  local awgn_noise=""
  local awgn_ploss=""
  local mcs_policy=""
  local aimd_max_drop=""

  case "${label}" in
    clear_vanilla)
      channel="clear"; policy="vanilla"
      gnb_conf="gnb.sa.band78.fr1.106PRB.usrpb210.conf"; ue_conf="ue.conf"
      rfsim_chanmod="0"; awgn_profile="clear"; mcs_policy="vanilla"
      ;;
    clear_aimd_cap)
      channel="clear"; policy="aimd_cap"
      gnb_conf="gnb.sa.band78.fr1.106PRB.usrpb210.conf"; ue_conf="ue.conf"
      rfsim_chanmod="0"; awgn_profile="clear"; mcs_policy="aimd"; aimd_max_drop="${AIMD_CAP_DROP}"
      ;;
    clear_sinr)
      channel="clear"; policy="sinr"
      gnb_conf="gnb.sa.band78.fr1.106PRB.usrpb210.conf"; ue_conf="ue.conf"
      rfsim_chanmod="0"; awgn_profile="clear"; mcs_policy="sinr"
      ;;
    mild_vanilla)
      channel="mild_awgn"; policy="vanilla"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"; ue_conf="ue.awgn_mild.conf"
      rfsim_chanmod="1"; awgn_profile="mild"; awgn_noise="-10"; awgn_ploss="0"; mcs_policy="vanilla"
      ;;
    mild_aimd_cap)
      channel="mild_awgn"; policy="aimd_cap"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"; ue_conf="ue.awgn_mild.conf"
      rfsim_chanmod="1"; awgn_profile="mild"; awgn_noise="-10"; awgn_ploss="0"; mcs_policy="aimd"; aimd_max_drop="${AIMD_CAP_DROP}"
      ;;
    mild_sinr)
      channel="mild_awgn"; policy="sinr"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"; ue_conf="ue.awgn_mild.conf"
      rfsim_chanmod="1"; awgn_profile="mild"; awgn_noise="-10"; awgn_ploss="0"; mcs_policy="sinr"
      ;;
    medium_vanilla)
      channel="medium_awgn"; policy="vanilla"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_medium.conf"; ue_conf="ue.awgn_medium.conf"
      rfsim_chanmod="1"; awgn_profile="medium"; awgn_noise="-5"; awgn_ploss="0"; mcs_policy="vanilla"
      ;;
    medium_aimd_cap)
      channel="medium_awgn"; policy="aimd_cap"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_medium.conf"; ue_conf="ue.awgn_medium.conf"
      rfsim_chanmod="1"; awgn_profile="medium"; awgn_noise="-5"; awgn_ploss="0"; mcs_policy="aimd"; aimd_max_drop="${AIMD_CAP_DROP}"
      ;;
    medium_sinr)
      channel="medium_awgn"; policy="sinr"
      gnb_conf="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_medium.conf"; ue_conf="ue.awgn_medium.conf"
      rfsim_chanmod="1"; awgn_profile="medium"; awgn_noise="-5"; awgn_ploss="0"; mcs_policy="sinr"
      ;;
    *)
      echo "[fair-mcs-grant] ERROR: unknown run label '${label}'" >&2
      return 2
      ;;
  esac

  condition="oai_default106_fair_${channel}_${policy}"
  local batch="${BASE_BATCH_ID}_${label}"

  echo "[fair-mcs-grant] ===== ${label}: channel=${channel}, policy=${policy}, condition=${condition}, batch=${batch} ====="
  echo "[fair-mcs-grant] gNB=${gnb_conf}, UE=${ue_conf}, rfsim_chanmod=${rfsim_chanmod}, awgn_profile=${awgn_profile}, awgn_noise=${awgn_noise:-none}, mcs_policy=${mcs_policy}, aimd_max_drop=${aimd_max_drop:-none}"

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
    MCS_POLICY="${mcs_policy}" \
    AIMD_MAX_DROP="${aimd_max_drop}" \
    TRANSPORT_LABEL="${condition}_noae_zstd_ttracer" \
    bash downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
}

show_preflight || exit $?

for label in ${RUNS}; do
  run_one "${label}" || exit $?
done

echo "[fair-mcs-grant] done. Summarize with:"
echo "  python3 abiodun/oai_mcs_policy_track2/summarize_fair_mcs_grant_rerun.py --base-batch ${BASE_BATCH_ID}"
