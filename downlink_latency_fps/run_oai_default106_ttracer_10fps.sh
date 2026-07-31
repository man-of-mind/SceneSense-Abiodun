#!/usr/bin/env bash
# Run the corrected live CARLA Step-1 frontend over the default OAI 106PRB
# 7DL/2UL TDD config with UE/gNB T-tracer enabled.
#
# Intended for payload/codec/model-knob probes where we want the default
# 106PRB adaptive-MCS baseline, not the UL-heavy TDD variant.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

CONDITION="${CONDITION:-oai_default106_ttracer}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="downlink_${CONDITION}_fps10_${BATCH_ID}"
CAP_ROOT="${AB}/metrics_logs/carla_oai_ttracer/${RUN_GROUP}"
LOG_ROOT="${CAP_ROOT}/logs"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1800}"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
GNB_CONF_DEFAULT="${GNB_CONF_DEFAULT:-${GNB_CONF}}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"
ENABLE_SOFTMODEM_TTRACER="${ENABLE_SOFTMODEM_TTRACER:-1}"
ATTACH_ONLY="${ATTACH_ONLY:-0}"
RECORD_GNB="${RECORD_GNB:-1}"
TTRACER_UE_PROFILE="${TTRACER_UE_PROFILE:-latency}"
TTRACER_GNB_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
FORCE_UL_MCS="${FORCE_UL_MCS:-}"
RADAR_RASTERIZER="${RADAR_RASTERIZER:-fast}"

mkdir -p "${LOG_ROOT}"

SAMPLER_PID=""
UE_RECORD_PID=""
GNB_RECORD_PID=""

say() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${CAP_ROOT}/run.log"
}

kill_pid() {
  local pid="$1"
  local name="$2"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    say "stopping ${name} pid=${pid}"
    kill -INT "${pid}" 2>/dev/null || true
    sleep 2
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  kill_pid "${SAMPLER_PID}" "network sampler"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
}
trap cleanup EXIT

stop_ran() {
  say "stopping existing nr-softmodem/nr-uesoftmodem processes"
  sudo pkill -x nr-uesoftmodem 2>/dev/null || true
  sudo pkill -x nr-softmodem 2>/dev/null || true
  sleep 4
}

restart_core() {
  say "restarting AMF/SMF/UPF so UE tunnel gets ${OAI_UE_IP}"
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

start_gnb_106_default() {
  local t_args=()
  local sudo_env=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_GNB_T_PORT:-2021}")
  fi
  if [[ -n "${FORCE_UL_MCS}" ]]; then
    sudo_env=(env SCENESENSE_FORCE_UL_MCS="${FORCE_UL_MCS}")
  fi
  say "starting default gNB: ${GNB_CONF_DEFAULT}, min_rxtxtime=${GNB_MIN_RXTXTIME}, ttracer=${ENABLE_SOFTMODEM_TTRACER}, force_ul_mcs=${FORCE_UL_MCS:-adaptive}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo "${sudo_env[@]}" ./nr-softmodem \
        -O "${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}" \
        --gNBs.[0].min_rxtxtime "${GNB_MIN_RXTXTIME}" \
        --rfsim \
        "${t_args[@]}" \
        > "${LOG_ROOT}/gnb_106_default_ttracer_stdout.log" 2>&1 &
  )
}

start_ue_106() {
  local t_args=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_UE_T_PORT:-2023}")
  fi
  say "starting single-UE softmodem: PRB=${UE_PRB}, conf=${UE_CONF}, freq=${UE_DL_FREQ}, ttracer=${ENABLE_SOFTMODEM_TTRACER}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-uesoftmodem \
        --rfsim \
        --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER:-127.0.0.1}" \
        -r "${UE_PRB}" \
        --numerology "${UE_NUMEROLOGY}" \
        --band "${UE_BAND}" \
        -C "${UE_DL_FREQ}" \
        -O "${OAI_RAN_CONF}/${UE_CONF}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/ue_106_default_ttracer_stdout.log" 2>&1 &
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

start_back_half() {
  say "starting/recreating Step-1 back-half container, result return host=${OAI_UE_IP}"
  FUSION_BACK_REMOTE_HOST="${OAI_UE_IP}" \
  FUSION_BACK_REMOTE_HOST_1="${OAI_UE_IP}" \
  FUSION_BACK_DUAL=0 \
  FUSION_QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}" \
  FUSION_ENTROPY_CODER="${ENTROPY_CODER:-zstd}" \
  ZSTD_LEVEL="${ZSTD_LEVEL:-3}" \
  ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}" \
  AE_CHECKPOINT="${AE_CHECKPOINT:-}" \
  AE_CHECKPOINT_CONTAINER="${AE_CHECKPOINT_CONTAINER:-}" \
    scripts/receiver_container_downlink_fps_back_up.sh \
    > "${LOG_ROOT}/receiver_container_downlink_fps_back_up.log" 2>&1
}

verify_back_half() {
  say "verifying Step-1 back-half container stayed up"
  sleep 8
  local running
  running="$(sudo docker inspect -f '{{.State.Running}}' oai-perception-rx 2>/dev/null || echo false)"
  if [[ "${running}" != "true" ]]; then
    say "ERROR: oai-perception-rx is not running after startup"
    sudo docker logs --tail 160 oai-perception-rx 2>&1 | tee -a "${CAP_ROOT}/run.log" || true
    return 1
  fi
  sudo docker logs --tail 80 oai-perception-rx 2>&1 | tee -a "${CAP_ROOT}/back_half_startup_tail.log" >/dev/null || true
  if [[ -n "${AE_CHECKPOINT_CONTAINER:-${AE_CHECKPOINT:-}}" ]] && ! grep -q "Loaded feature-AE" "${CAP_ROOT}/back_half_startup_tail.log"; then
    say "WARN: did not see feature-AE load confirmation in back-half startup tail"
  fi
}

start_network_sampler() {
  say "starting UE tunnel/network sampler"
  "${PY}" scripts/sample_oai_network_metrics.py \
    --run-group "${RUN_GROUP}" \
    --ping-host "${OAI_EXT_DN_IP}" \
    > "${LOG_ROOT}/network_sampler_stdout.log" 2>&1 &
  SAMPLER_PID=$!
}

start_ttracer_recorders() {
  say "starting UE ${TTRACER_UE_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
  scripts/ttracer_record_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source ue \
    --profile "${TTRACER_UE_PROFILE}" \
    --duration-s "${TTRACER_DURATION_S}" \
    > "${LOG_ROOT}/ttracer_record_ue_stdout.log" 2>&1 &
  UE_RECORD_PID=$!

  if [[ "${RECORD_GNB}" == "1" ]]; then
    say "starting optional gNB ${TTRACER_GNB_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
    scripts/ttracer_record_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --duration-s "${TTRACER_DURATION_S}" \
      > "${LOG_ROOT}/ttracer_record_gnb_stdout.log" 2>&1 &
    GNB_RECORD_PID=$!
  fi
}

run_front() {
  say "running CARLA frontend 10FPS one-loop run_group=${RUN_GROUP}"
  FPS_LIST=10 \
  DURATION_S="${FRONT_DURATION_S}" \
  CONDITION="${CONDITION}" \
  TRANSPORT_LABEL="${TRANSPORT_LABEL:-oai_default106_ttracer}" \
  FRONT_BIND_HOST="${OAI_UE_IP}" \
  BACK_REMOTE_HOST="${OAI_RX_IP}" \
  START_LOCAL_BACK=0 \
  QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}" \
  ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}" \
  ENTROPY_CODER="${ENTROPY_CODER:-zstd}" \
  ZSTD_LEVEL="${ZSTD_LEVEL:-3}" \
  AE_CHECKPOINT="${AE_CHECKPOINT:-}" \
  BATCH_ID="${BATCH_ID}" \
    bash downlink_latency_fps/run_common.sh \
    > "${LOG_ROOT}/front_run_common_stdout.log" 2>&1
}

postprocess() {
  say "stopping recorders before extraction"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  UE_RECORD_PID=""
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
  GNB_RECORD_PID=""
  kill_pid "${SAMPLER_PID}" "network sampler"
  SAMPLER_PID=""

  say "extracting UE T-tracer CSV"
  scripts/ttracer_extract_csv_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source ue \
    --profile "${TTRACER_UE_PROFILE}" \
    --clean-output \
    > "${LOG_ROOT}/ttracer_extract_ue_stdout.log" 2>&1

  if [[ "${RECORD_GNB}" == "1" ]]; then
    say "extracting gNB T-tracer CSV"
    scripts/ttracer_extract_csv_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --clean-output \
      > "${LOG_ROOT}/ttracer_extract_gnb_stdout.log" 2>&1
  fi

  say "analyzing UE grant windows"
  "${PY}" scripts/analyze_nrue_grant_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s 1.0 \
    > "${LOG_ROOT}/analyze_nrue_grant_metrics_stdout.log" 2>&1

  say "preparing compact plot artifacts"
  printf "Validated default 106PRB CARLA/T-tracer run.\n\ngNB config: %s\nUE launch: -r %s -C %s\nRun group: %s\n" \
    "${GNB_CONF_DEFAULT}" "${UE_PRB}" "${UE_DL_FREQ}" "${RUN_GROUP}" \
    > "${CAP_ROOT}/VALIDATED_DEFAULT106_TTRACER.ok"

  "${PY}" downlink_latency_fps/prepare_ttracer_grant_artifacts.py \
    --run-group "${RUN_GROUP}" \
    > "${LOG_ROOT}/prepare_ttracer_grant_artifacts_stdout.log" 2>&1

  say "summarizing frontend metrics"
  "${PY}" downlink_latency_fps/analyze_downlink_fps.py \
    downlink_latency_fps/runs \
    --contains "${BATCH_ID}" \
    > "${CAP_ROOT}/summary_stdout.md" 2>&1

  say "analyzing uplink layer latency"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${RUN_GROUP}" \
    > "${CAP_ROOT}/layer_latency_stdout.md" 2>&1
}

say "===== START default 106PRB CARLA/T-tracer run ${RUN_GROUP} ====="
say "config: gNB=${GNB_CONF_DEFAULT}, UE_PRB=${UE_PRB}, UE_FREQ=${UE_DL_FREQ}, min_rxtxtime=${GNB_MIN_RXTXTIME}, softmodem_ttracer=${ENABLE_SOFTMODEM_TTRACER}, UE_profile=${TTRACER_UE_PROFILE}, gNB_profile=${TTRACER_GNB_PROFILE}, record_gNB=${RECORD_GNB}, force_ul_mcs=${FORCE_UL_MCS:-adaptive}, quant=${QUANTIZATION_MODE:-per_channel_uint8}, roi=${ROI_THRESHOLD:-0.0}, entropy=${ENTROPY_CODER:-zstd}, ae=${AE_CHECKPOINT:-none}, T_ports gNB=${OAI_GNB_T_PORT:-2021}, UE=${OAI_UE_T_PORT:-2023}"

if [[ ! -f "${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}" ]]; then
  say "ERROR: missing gNB config ${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}"
  exit 1
fi

stop_ran
restart_core
start_gnb_106_default
sleep 22
start_ue_106
if ! wait_tunnel; then
  say "gNB log: ${LOG_ROOT}/gnb_106_default_ttracer_stdout.log"
  say "UE log: ${LOG_ROOT}/ue_106_default_ttracer_stdout.log"
  exit 1
fi

if [[ "${ATTACH_ONLY}" == "1" ]]; then
  say "ATTACH_ONLY=1; attach succeeded, skipping back-half/CARLA/postprocess"
  exit 0
fi

start_back_half
verify_back_half || exit 1
start_network_sampler
start_ttracer_recorders
sleep 6

FRONT_RC=0
run_front || FRONT_RC=$?
say "front completed rc=${FRONT_RC}"
postprocess

say "===== DONE default 106PRB CARLA/T-tracer run ${RUN_GROUP} rc=${FRONT_RC} ====="
say "artifacts: ${CAP_ROOT}"
exit "${FRONT_RC}"
