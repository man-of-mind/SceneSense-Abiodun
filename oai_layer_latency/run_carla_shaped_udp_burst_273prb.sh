#!/usr/bin/env bash
# Experiment 1: CARLA-shaped UDP burst over OAI 273PRB, with UE/gNB latency
# T-tracer profiles. This isolates traffic shape/backlog from CARLA/model compute.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

CONDITION="${CONDITION:-carla_shape_udp_bw273}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="${CONDITION}_${BATCH_ID}"
CAP_ROOT="${AB}/metrics_logs/carla_shape_udp/${RUN_GROUP}"
LOG_ROOT="${CAP_ROOT}/logs"

GNB_CONF_273="${GNB_CONF_273:-gnb.sa.band78.fr1.273PRB.scenesense_rfsim.conf}"
UE_CONF_273="${UE_CONF_273:-ue.conf}"
UE_DL_FREQ_273="${UE_DL_FREQ_273:-3649260000}"
UE_SSB_273="${UE_SSB_273:-516}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"
ENABLE_SOFTMODEM_TTRACER="${ENABLE_SOFTMODEM_TTRACER:-1}"
TTRACER_UE_PROFILE="${TTRACER_UE_PROFILE:-latency}"
TTRACER_GNB_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-260}"

BURST_FPS="${BURST_FPS:-10}"
BURST_FRAMES="${BURST_FRAMES:-1300}"
BURST_FRAME_BYTES="${BURST_FRAME_BYTES:-1079400}"
BURST_CHUNK_BYTES="${BURST_CHUNK_BYTES:-60000}"
BURST_INTER_CHUNK_GAP_US="${BURST_INTER_CHUNK_GAP_US:-0}"
BURST_PORT="${BURST_PORT:-5001}"

mkdir -p "${LOG_ROOT}"

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

start_gnb_273() {
  local t_args=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_GNB_T_PORT:-2021}")
  fi
  say "starting gNB 273PRB: ${GNB_CONF_273}, min_rxtxtime=${GNB_MIN_RXTXTIME}, ttracer=${ENABLE_SOFTMODEM_TTRACER}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-softmodem \
        -O "${OAI_RAN_CONF}/${GNB_CONF_273}" \
        --gNBs.[0].min_rxtxtime "${GNB_MIN_RXTXTIME}" \
        --rfsim \
        "${t_args[@]}" \
        > "${LOG_ROOT}/gnb_273_stdout.log" 2>&1 &
  )
}

start_ue_273() {
  local t_args=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_UE_T_PORT:-2023}")
  fi
  say "starting UE 273PRB: freq=${UE_DL_FREQ_273}, ssb=${UE_SSB_273}, ttracer=${ENABLE_SOFTMODEM_TTRACER}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-uesoftmodem \
        --rfsim \
        --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER:-127.0.0.1}" \
        -r 273 \
        --numerology "${UE_NUMEROLOGY}" \
        --band "${UE_BAND}" \
        -C "${UE_DL_FREQ_273}" \
        --ssb "${UE_SSB_273}" \
        -O "${OAI_RAN_CONF}/${UE_CONF_273}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/ue_273_stdout.log" 2>&1 &
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

start_udp_sink() {
  say "starting UDP sink on ext-DN ${OAI_EXT_DN_IP}:${BURST_PORT}"
  sudo docker exec -d oai-ext-dn sh -c "pkill -f 'iperf -s' 2>/dev/null; iperf -s -u -B ${OAI_EXT_DN_IP} -p ${BURST_PORT} -i 1 >/tmp/scenesense_carla_shape_udp_iperf.log 2>&1"
  sleep 1
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

  say "starting gNB ${TTRACER_GNB_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
  scripts/ttracer_record_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source gnb \
    --profile "${TTRACER_GNB_PROFILE}" \
    --duration-s "${TTRACER_DURATION_S}" \
    > "${LOG_ROOT}/ttracer_record_gnb_stdout.log" 2>&1 &
  GNB_RECORD_PID=$!
}

run_burst_sender() {
  say "running CARLA-shaped UDP burst: fps=${BURST_FPS}, frames=${BURST_FRAMES}, frame_bytes=${BURST_FRAME_BYTES}, chunk_bytes=${BURST_CHUNK_BYTES}"
  "${PY}" oai_layer_latency/carla_shaped_udp_burst_sender.py \
    --bind-host "${OAI_UE_IP}" \
    --remote-host "${OAI_EXT_DN_IP}" \
    --remote-port "${BURST_PORT}" \
    --fps "${BURST_FPS}" \
    --frames "${BURST_FRAMES}" \
    --frame-bytes "${BURST_FRAME_BYTES}" \
    --chunk-bytes "${BURST_CHUNK_BYTES}" \
    --inter-chunk-gap-us "${BURST_INTER_CHUNK_GAP_US}" \
    --log-csv "${CAP_ROOT}/burst_sender_log.csv" \
    > "${LOG_ROOT}/burst_sender_stdout.log" 2>&1
}

postprocess() {
  say "stopping recorders before extraction"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  UE_RECORD_PID=""
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
  GNB_RECORD_PID=""

  say "copying ext-DN UDP sink tail"
  sudo docker exec oai-ext-dn sh -c "tail -120 /tmp/scenesense_carla_shape_udp_iperf.log 2>/dev/null || true" \
    > "${LOG_ROOT}/ext_dn_udp_sink_tail.log" 2>&1 || true

  say "extracting UE T-tracer CSV"
  scripts/ttracer_extract_csv_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source ue \
    --profile "${TTRACER_UE_PROFILE}" \
    --timeout-s 120 \
    --clean-output \
    > "${LOG_ROOT}/ttracer_extract_ue_stdout.log" 2>&1

  say "extracting gNB T-tracer CSV"
  scripts/ttracer_extract_csv_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source gnb \
    --profile "${TTRACER_GNB_PROFILE}" \
    --timeout-s 120 \
    --clean-output \
    > "${LOG_ROOT}/ttracer_extract_gnb_stdout.log" 2>&1

  say "analyzing UE grant windows"
  "${PY}" scripts/analyze_nrue_grant_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s 1.0 \
    > "${LOG_ROOT}/analyze_nrue_grant_metrics_stdout.log" 2>&1

  say "analyzing uplink layer latency"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${RUN_GROUP}" \
    --slots-per-frame 20 \
    > "${LOG_ROOT}/analyze_uplink_layer_latency_stdout.log" 2>&1

  printf "CARLA-shaped UDP burst over OAI 273PRB completed.\\nRun group: %s\\nFrame bytes: %s\\nChunk bytes: %s\\nFPS: %s\\nFrames: %s\\n" \
    "${RUN_GROUP}" "${BURST_FRAME_BYTES}" "${BURST_CHUNK_BYTES}" "${BURST_FPS}" "${BURST_FRAMES}" \
    > "${CAP_ROOT}/EXPERIMENT1_CARLA_SHAPED_UDP.ok"
}

say "===== START Experiment 1 CARLA-shaped UDP burst ${RUN_GROUP} ====="
say "config: gNB=${GNB_CONF_273}, UE -r 273 -C ${UE_DL_FREQ_273} --ssb ${UE_SSB_273}, ttracer UE=${TTRACER_UE_PROFILE}, gNB=${TTRACER_GNB_PROFILE}"
say "traffic: ${BURST_FRAMES} frames @ ${BURST_FPS} FPS, ${BURST_FRAME_BYTES} B/frame, ${BURST_CHUNK_BYTES} B/chunk"

if [[ ! -f "${OAI_RAN_CONF}/${GNB_CONF_273}" ]]; then
  say "ERROR: missing gNB config ${OAI_RAN_CONF}/${GNB_CONF_273}"
  exit 1
fi

stop_ran
restart_core
start_gnb_273
sleep 22
start_ue_273
if ! wait_tunnel; then
  say "gNB log: ${LOG_ROOT}/gnb_273_stdout.log"
  say "UE log: ${LOG_ROOT}/ue_273_stdout.log"
  exit 1
fi

start_udp_sink
start_ttracer_recorders
sleep 6

RUN_RC=0
run_burst_sender || RUN_RC=$?
say "burst sender completed rc=${RUN_RC}"
postprocess

say "===== DONE Experiment 1 ${RUN_GROUP} rc=${RUN_RC} ====="
say "artifacts: ${CAP_ROOT}"
exit "${RUN_RC}"
