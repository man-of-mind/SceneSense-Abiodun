#!/usr/bin/env bash
# Run the live CARLA Step-1 frontend over default OAI 106PRB with a selected
# lossless entropy coder (`CODEC=zlib` or `CODEC=zstd`).
#
# This uses downlink_latency_fps/run_common.sh for the frontend scene, so it
# inherits the current drivable deployment recipe:
#   seed 31, 28 vehicles, 35 pedestrians, ignore-lights 50%,
#   fixed waypoint loop 80,85,91,94,99,80.
set -euo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

CODEC="${CODEC:-zstd}"
if [[ "${CODEC}" != "zlib" && "${CODEC}" != "zstd" ]]; then
  echo "CODEC must be zlib or zstd; got ${CODEC}" >&2
  exit 2
fi

CONDITION="${CONDITION:-oai_default_drivable_${CODEC}}"
TRANSPORT_LABEL="${TRANSPORT_LABEL:-oai_default_noae_${CODEC}_drivable}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="downlink_${CONDITION}_fps10_${BATCH_ID}"
CAP_ROOT="${AB}/metrics_logs/carla_oai_codec/${RUN_GROUP}"
LOG_ROOT="${CAP_ROOT}/logs"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"

mkdir -p "${LOG_ROOT}"

say() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${CAP_ROOT}/run.log"
}

stop_ran() {
  say "stopping existing nr-softmodem/nr-uesoftmodem processes"
  sudo pkill -x nr-uesoftmodem 2>/dev/null || true
  sudo pkill -x nr-softmodem 2>/dev/null || true
  sleep 4
}

restart_core() {
  say "restarting AMF/SMF/UPF for clean default-106PRB UE attach"
  (
    cd "${OAI_CN_DIR}" &&
      sudo docker compose restart oai-amf oai-smf oai-upf >/dev/null 2>&1
  )
  for _ in $(seq 1 45); do
    local healthy
    healthy="$(sudo docker ps --format '{{.Names}} {{.Status}}' | grep -E 'oai-(amf|smf|upf)' | grep -c healthy || true)"
    if [[ "${healthy}" -ge 3 ]]; then
      sleep 3
      return 0
    fi
    sleep 2
  done
  say "WARN: core health check did not see all AMF/SMF/UPF containers healthy"
}

start_gnb_106() {
  say "starting default gNB: ${GNB_CONF}, PRB=${UE_PRB}, min_rxtxtime=${GNB_MIN_RXTXTIME}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-softmodem \
        -O "${OAI_RAN_CONF}/${GNB_CONF}" \
        --gNBs.[0].min_rxtxtime "${GNB_MIN_RXTXTIME}" \
        --rfsim \
        > "${LOG_ROOT}/gnb_106_${CODEC}_stdout.log" 2>&1 &
  )
}

start_ue_106() {
  say "starting default UE: PRB=${UE_PRB}, numerology=${UE_NUMEROLOGY}, band=${UE_BAND}, freq=${UE_DL_FREQ}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-uesoftmodem \
        --rfsim \
        --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER}" \
        -r "${UE_PRB}" \
        --numerology "${UE_NUMEROLOGY}" \
        --band "${UE_BAND}" \
        -C "${UE_DL_FREQ}" \
        -O "${OAI_RAN_CONF}/${UE_CONF}" \
        > "${LOG_ROOT}/ue_106_${CODEC}_stdout.log" 2>&1 &
  )
}

wait_tunnel() {
  say "waiting for ${OAI_UE_IFACE}=${OAI_UE_IP}"
  for _ in $(seq 1 "${WAIT_TUNNEL_TRIES}"); do
    if ip -4 addr show "${OAI_UE_IFACE}" 2>/dev/null | grep -q "${OAI_UE_IP}"; then
      ip -br addr show "${OAI_UE_IFACE}" | tee -a "${CAP_ROOT}/run.log"
      sleep 3
      return 0
    fi
    sleep 2
  done
  say "ERROR: ${OAI_UE_IFACE} did not attach as ${OAI_UE_IP}"
  return 1
}

start_back_half_codec() {
  say "starting/recreating Step-1 back-half container with codec=${CODEC}, result return host=${OAI_UE_IP}"
  FUSION_BACK_REMOTE_HOST="${OAI_UE_IP}" \
  FUSION_BACK_REMOTE_HOST_1="${OAI_UE_IP}" \
  FUSION_BACK_DUAL=0 \
  FUSION_ENTROPY_CODER="${CODEC}" \
    scripts/receiver_container_downlink_fps_back_up.sh \
    > "${LOG_ROOT}/receiver_container_downlink_fps_back_up.log" 2>&1
}

verify_back_half_codec() {
  say "verifying ${CODEC} back-half container stayed up"
  sleep 8
  local running
  running="$(sudo docker inspect -f '{{.State.Running}}' oai-perception-rx 2>/dev/null || echo false)"
  if [[ "${running}" != "true" ]]; then
    say "ERROR: oai-perception-rx is not running after ${CODEC} startup"
    sudo docker logs --tail 160 oai-perception-rx 2>&1 | tee -a "${CAP_ROOT}/run.log" || true
    return 1
  fi
  sudo docker logs --tail 60 oai-perception-rx 2>&1 | tee -a "${CAP_ROOT}/back_half_startup_tail.log" >/dev/null || true
  if ! grep -q "entropy=${CODEC}" "${CAP_ROOT}/back_half_startup_tail.log"; then
    say "WARN: did not see entropy=${CODEC} in back-half startup tail"
  fi
}

run_front_codec() {
  say "running CARLA frontend 10FPS one-loop codec=${CODEC} run_group=${RUN_GROUP}"
  FPS_LIST=10 \
  DURATION_S="${FRONT_DURATION_S}" \
  CONDITION="${CONDITION}" \
  TRANSPORT_LABEL="${TRANSPORT_LABEL}" \
  FRONT_BIND_HOST="${OAI_UE_IP}" \
  BACK_REMOTE_HOST="${OAI_RX_IP}" \
  START_LOCAL_BACK=0 \
  ENTROPY_CODER="${CODEC}" \
  BATCH_ID="${BATCH_ID}" \
    bash downlink_latency_fps/run_common.sh \
    > "${LOG_ROOT}/front_run_common_stdout.log" 2>&1
}

postprocess() {
  say "summarizing ${CODEC} run metrics"
  "${PY}" downlink_latency_fps/analyze_downlink_fps.py \
    downlink_latency_fps/runs \
    --contains "${BATCH_ID}" \
    > "${CAP_ROOT}/summary_stdout.md" 2>&1
}

say "===== START default 106PRB OAI ${CODEC} CARLA run ${RUN_GROUP} ====="
say "config: gNB=${GNB_CONF}, UE_PRB=${UE_PRB}, UE_FREQ=${UE_DL_FREQ}, codec=${CODEC}, duration=${FRONT_DURATION_S}s, scene=drivable_28veh_35ped_seed31"

stop_ran
restart_core
start_gnb_106
sleep 22
start_ue_106
if ! wait_tunnel; then
  say "gNB log: ${LOG_ROOT}/gnb_106_${CODEC}_stdout.log"
  say "UE log: ${LOG_ROOT}/ue_106_${CODEC}_stdout.log"
  exit 1
fi

start_back_half_codec
verify_back_half_codec

FRONT_RC=0
run_front_codec || FRONT_RC=$?
say "front completed rc=${FRONT_RC}"

postprocess
say "===== DONE default 106PRB OAI ${CODEC} CARLA run ${RUN_GROUP} rc=${FRONT_RC} ====="
say "artifacts: ${CAP_ROOT}"
exit "${FRONT_RC}"
