#!/usr/bin/env bash
# Track 2 AWGN ladder on the default 106PRB OAI path.
#
# Purpose:
#   Test whether stronger AWGN makes BLER persistent enough that vanilla OAI
#   and hold-few-samples lower MCS, and compare that against capped AIMD.
#
# Default matrix:
#   profiles: mild medium strong
#   policies: vanilla hold aimd_cap
#
# Optional:
#   PROFILES="mild medium strong harsh edge" to include boundary profiles.
#   POLICIES="vanilla hold aimd aimd_cap" to include uncapped AIMD.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${AB}" || exit 2

BASE_BATCH_ID="${BASE_BATCH_ID:-track2_awgn_ladder_$(date +%Y%m%d_%H%M%S)}"
PROFILES="${PROFILES:-mild medium strong}"
POLICIES="${POLICIES:-vanilla hold aimd_cap}"
FRONT_DURATION_S="${FRONT_DURATION_S:-30}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1200}"
AIMD_CAP_DROP="${AIMD_CAP_DROP:-3}"

echo "[track2-awgn-ladder] Base batch: ${BASE_BATCH_ID}"
echo "[track2-awgn-ladder] Profiles: ${PROFILES}"
echo "[track2-awgn-ladder] Policies: ${POLICIES}"
echo "[track2-awgn-ladder] Front duration per run: ${FRONT_DURATION_S}s"

for profile in ${PROFILES}; do
  case "${profile}" in
    mild|medium|strong|harsh|edge) ;;
    *)
      echo "[track2-awgn-ladder] ERROR: unknown profile '${profile}' (use mild, medium, strong, harsh, edge)" >&2
      exit 2
      ;;
  esac

  echo "[track2-awgn-ladder] ===== profile=${profile} ====="
  AWGN_PROFILE="${profile}" \
  BASE_BATCH_ID="${BASE_BATCH_ID}_${profile}" \
  FRONT_DURATION_S="${FRONT_DURATION_S}" \
  TTRACER_DURATION_S="${TTRACER_DURATION_S}" \
  POLICIES="${POLICIES}" \
  AIMD_CAP_DROP="${AIMD_CAP_DROP}" \
    bash oai_mcs_policy_track2/run_awgn_106prb_policies.sh || exit $?
done

echo "[track2-awgn-ladder] done. Summarize with:"
echo "  python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py --base-batch ${BASE_BATCH_ID} --profiles \"${PROFILES}\" --policies \"${POLICIES}\""
