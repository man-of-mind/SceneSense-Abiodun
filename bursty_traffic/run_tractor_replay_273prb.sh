#!/usr/bin/env bash
# Replay a TRACTOR raw packet trace over the OAI 273PRB RFsim path.
#
# Purpose: compare real bursty application traces against CARLA split-feature
# bursts under the same ideal/default OAI channel. This runner intentionally
# does not start CARLA and does not enable RFsim channelmod/AWGN.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

TRACE_CSV="${TRACE_CSV:-bursty_traffic/TRACTOR/raw/urllc_03_03.csv}"
TRACE_NAME="$(basename "${TRACE_CSV}" .csv)"
CONDITION="${CONDITION:-tractor_replay_bw273}"
BATCH_ID="${BATCH_ID:-${TRACE_NAME}_$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="${CONDITION}_${BATCH_ID}"
CAP_ROOT="${AB}/metrics_logs/tractor_replay/${RUN_GROUP}"
LOG_ROOT="${CAP_ROOT}/logs"

GNB_CONF_273="${GNB_CONF_273:-gnb.sa.band78.fr1.273PRB.scenesense_rfsim.conf}"
UE_CONF_273="${UE_CONF_273:-ue.conf}"
UE_DL_FREQ_273="${UE_DL_FREQ_273:-3649260000}"
UE_SSB_273="${UE_SSB_273:-516}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"
ENABLE_SOFTMODEM_TTRACER="${ENABLE_SOFTMODEM_TTRACER:-1}"
TTRACER_UE_PROFILE="${TTRACER_UE_PROFILE:-all}"
TTRACER_GNB_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-420}"
HOLD_MCS_FEW_SAMPLES="${HOLD_MCS_FEW_SAMPLES:-0}"
FORCE_UL_MCS="${FORCE_UL_MCS:-}"

REPLAY_DIRECTION="${REPLAY_DIRECTION:-uplink}"
REPLAY_PORT="${REPLAY_PORT:-55000}"
REPLAY_MAX_PAYLOAD="${REPLAY_MAX_PAYLOAD:-1400}"
REPLAY_LARGE_PACKET_MODE="${REPLAY_LARGE_PACKET_MODE:-split}"
REPLAY_START_OFFSET_S="${REPLAY_START_OFFSET_S:-0}"
REPLAY_MAX_DURATION_S="${REPLAY_MAX_DURATION_S:-120}"
REPLAY_TIME_SCALE="${REPLAY_TIME_SCALE:-1.0}"
REPLAY_LENGTH_SCALE="${REPLAY_LENGTH_SCALE:-1.0}"
SINK_TIMEOUT_S="${SINK_TIMEOUT_S:-20}"

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
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true" >/dev/null 2>&1 || true
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
  local sudo_env=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_GNB_T_PORT:-2021}")
  fi
  sudo_env=(env SCENESENSE_HOLD_MCS_FEW_SAMPLES="${HOLD_MCS_FEW_SAMPLES}")
  if [[ -n "${FORCE_UL_MCS}" ]]; then
    sudo_env+=(SCENESENSE_FORCE_UL_MCS="${FORCE_UL_MCS}")
  fi
  say "starting gNB 273PRB: ${GNB_CONF_273}, hold_mcs=${HOLD_MCS_FEW_SAMPLES}, force_mcs=${FORCE_UL_MCS:-adaptive}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo "${sudo_env[@]}" ./nr-softmodem \
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
  say "starting UE 273PRB: conf=${UE_CONF_273}, freq=${UE_DL_FREQ_273}, ssb=${UE_SSB_273}"
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
  say "starting tcpdump UDP sink capture in ext-DN ${OAI_EXT_DN_IP}:${REPLAY_PORT}"
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true; rm -f /tmp/scenesense_tractor_udp_sink.pcap /tmp/scenesense_tractor_udp_sink.log /tmp/scenesense_tractor_udp_sink_packets.txt"
  sudo docker exec -d oai-ext-dn sh -c "tcpdump -i any -n -s 0 'udp and dst port ${REPLAY_PORT}' -w /tmp/scenesense_tractor_udp_sink.pcap >/tmp/scenesense_tractor_udp_sink.log 2>&1"
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

run_replay() {
  say "replaying TRACTOR trace=${TRACE_CSV}, direction=${REPLAY_DIRECTION}, start_offset=${REPLAY_START_OFFSET_S}s, max_duration=${REPLAY_MAX_DURATION_S}s, time_scale=${REPLAY_TIME_SCALE}, length_scale=${REPLAY_LENGTH_SCALE}, max_payload=${REPLAY_MAX_PAYLOAD}, large_packet_mode=${REPLAY_LARGE_PACKET_MODE}"
  local duration_args=()
  if [[ -n "${REPLAY_MAX_DURATION_S}" ]]; then
    duration_args=(--max-duration-s "${REPLAY_MAX_DURATION_S}")
  fi
  "${PY}" bursty_traffic/udp_trace_replay.py "${TRACE_CSV}" \
    --dst "${OAI_EXT_DN_IP}" \
    --port "${REPLAY_PORT}" \
    --bind-ip "${OAI_UE_IP}" \
    --direction "${REPLAY_DIRECTION}" \
    --start-offset-s "${REPLAY_START_OFFSET_S}" \
    --time-scale "${REPLAY_TIME_SCALE}" \
    --length-scale "${REPLAY_LENGTH_SCALE}" \
    --max-payload "${REPLAY_MAX_PAYLOAD}" \
    --large-packet-mode "${REPLAY_LARGE_PACKET_MODE}" \
    "${duration_args[@]}" \
    > "${CAP_ROOT}/replay_sender.log" 2>&1
}

postprocess() {
  say "stopping recorders before extraction"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  UE_RECORD_PID=""
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
  GNB_RECORD_PID=""

  say "copying ext-DN UDP sink logs"
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true" >/dev/null 2>&1 || true
  sleep 2
  sudo docker exec oai-ext-dn sh -c "tcpdump -tt -n -r /tmp/scenesense_tractor_udp_sink.pcap 2>/tmp/scenesense_tractor_udp_sink_read.log > /tmp/scenesense_tractor_udp_sink_packets.txt || true" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_tractor_udp_sink.pcap "${CAP_ROOT}/udp_sink.pcap" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_tractor_udp_sink.log "${LOG_ROOT}/udp_sink_tcpdump.log" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_tractor_udp_sink_read.log "${LOG_ROOT}/udp_sink_tcpdump_read.log" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_tractor_udp_sink_packets.txt "${CAP_ROOT}/udp_sink_packets.txt" >/dev/null 2>&1 || true

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
    > "${LOG_ROOT}/analyze_uplink_layer_latency_stdout.log" 2>&1 || true

  say "summarizing TRACTOR OAI replay"
  "${PY}" bursty_traffic/analyze_tractor_oai_run.py \
    --run-group "${RUN_GROUP}" \
    > "${LOG_ROOT}/analyze_tractor_oai_run_stdout.log" 2>&1 || true

  printf "TRACTOR UDP replay over OAI 273PRB completed.\\nRun group: %s\\nTrace: %s\\nDirection: %s\\nStart offset: %s\\nMax duration: %s\\nTime scale: %s\\nLength scale: %s\\nMax payload: %s\\nLarge packet mode: %s\\nHold MCS few samples: %s\\n" \
    "${RUN_GROUP}" "${TRACE_CSV}" "${REPLAY_DIRECTION}" "${REPLAY_START_OFFSET_S}" "${REPLAY_MAX_DURATION_S:-full}" "${REPLAY_TIME_SCALE}" "${REPLAY_LENGTH_SCALE}" "${REPLAY_MAX_PAYLOAD}" "${REPLAY_LARGE_PACKET_MODE}" "${HOLD_MCS_FEW_SAMPLES}" \
    > "${CAP_ROOT}/TRACTOR_REPLAY_273PRB.ok"
}

say "===== START TRACTOR replay ${RUN_GROUP} ====="
say "config: gNB=${GNB_CONF_273}, UE=${UE_CONF_273}, ttracer UE=${TTRACER_UE_PROFILE}, gNB=${TTRACER_GNB_PROFILE}, hold_mcs=${HOLD_MCS_FEW_SAMPLES}, force_mcs=${FORCE_UL_MCS:-adaptive}"
say "traffic: trace=${TRACE_CSV}, direction=${REPLAY_DIRECTION}, start_offset=${REPLAY_START_OFFSET_S}, max_duration=${REPLAY_MAX_DURATION_S:-full}, time_scale=${REPLAY_TIME_SCALE}, length_scale=${REPLAY_LENGTH_SCALE}, max_payload=${REPLAY_MAX_PAYLOAD}, large_packet_mode=${REPLAY_LARGE_PACKET_MODE}"

if [[ ! -f "${TRACE_CSV}" ]]; then
  say "ERROR: missing trace ${TRACE_CSV}"
  exit 1
fi

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
run_replay || RUN_RC=$?
say "replay completed rc=${RUN_RC}"
sleep "${SINK_TIMEOUT_S}"
postprocess

say "===== DONE TRACTOR replay ${RUN_GROUP} rc=${RUN_RC} ====="
say "artifacts: ${CAP_ROOT}"
exit "${RUN_RC}"
