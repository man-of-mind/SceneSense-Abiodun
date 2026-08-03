#!/usr/bin/env bash
# Non-CARLA vanilla OAI 106PRB traffic diagnostic.
#
# Purpose:
#   Isolate whether the mild-AWGN "high MCS but low useful drain / high queue"
#   behavior appears for ordinary uplink traffic too, without CARLA, model
#   compute, result timeout, or edge-tail processing in the loop.
#
# Fairness constraints:
#   - 106PRB default 7DL/2UL path only
#   - vanilla OAI MCS policy only
#   - same UE/gNB T-tracer profiles as the CARLA fair rerun
#   - only vary traffic type and RFsim channel profile
#
# Default matrix:
#   iperf_clear iperf_mild tractor_clear tractor_mild
#
# Dry-run:
#   DRY_RUN=1 BASE_BATCH_ID=noncarla_awgn_20260803 \
#     bash abiodun/oai_mcs_policy_track2/run_noncarla_vanilla_awgn_106prb.sh
#
# Execute:
#   BASE_BATCH_ID=noncarla_awgn_20260803 \
#     bash abiodun/oai_mcs_policy_track2/run_noncarla_vanilla_awgn_106prb.sh
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

BASE_BATCH_ID="${BASE_BATCH_ID:-noncarla_awgn_$(date +%Y%m%d_%H%M%S)}"
RUNS="${RUNS:-iperf_clear iperf_mild tractor_clear tractor_mild}"
TRAFFIC_DURATION_S="${TRAFFIC_DURATION_S:-60}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-300}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-120}"
ENABLE_SOFTMODEM_TTRACER="${ENABLE_SOFTMODEM_TTRACER:-1}"
TTRACER_UE_PROFILE="${TTRACER_UE_PROFILE:-all}"
TTRACER_GNB_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
RECORD_GNB="${RECORD_GNB:-1}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
DRY_RUN="${DRY_RUN:-0}"

# Constant UDP baseline. 18 Mbps is close to the active offered load observed
# in the CARLA fair-rerun family while avoiding CARLA's closed-loop wait.
IPERF_BITRATE="${IPERF_BITRATE:-18M}"

# Tractor baseline. Default window matches the earlier clean 273PRB tractor
# diagnostic, then scales byte volume to be comparable with CARLA's active load.
TRACE_CSV="${TRACE_CSV:-bursty_traffic/TRACTOR/raw/embb_03_03a.csv}"
TRACE_NAME="$(basename "${TRACE_CSV}" .csv)"
REPLAY_PORT="${REPLAY_PORT:-55000}"
REPLAY_START_OFFSET_S="${REPLAY_START_OFFSET_S:-100}"
REPLAY_MAX_DURATION_S="${REPLAY_MAX_DURATION_S:-${TRAFFIC_DURATION_S}}"
REPLAY_TIME_SCALE="${REPLAY_TIME_SCALE:-1.0}"
REPLAY_LENGTH_SCALE="${REPLAY_LENGTH_SCALE:-3.0}"
REPLAY_MAX_PAYLOAD="${REPLAY_MAX_PAYLOAD:-1400}"
REPLAY_LARGE_PACKET_MODE="${REPLAY_LARGE_PACKET_MODE:-split}"

RUN_GROUP=""
CAP_ROOT=""
LOG_ROOT=""
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
  if [[ -n "${CAP_ROOT}" && -d "${CAP_ROOT}" ]]; then
    kill_pid "${SAMPLER_PID}" "network sampler"
    kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
    kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
  fi
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true; pkill -INT iperf3 2>/dev/null || true" >/dev/null 2>&1 || true
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

start_gnb_106() {
  local gnb_conf="$1"
  local rfsim_chanmod="$2"
  local awgn_profile="$3"
  local awgn_noise="$4"
  local t_args=()
  local chanmod_args=()
  local sudo_env=(env SCENESENSE_MCS_POLICY=vanilla SCENESENSE_HOLD_MCS_FEW_SAMPLES=0)

  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_GNB_T_PORT:-2021}")
  fi
  if [[ "${rfsim_chanmod}" == "1" ]]; then
    chanmod_args=(--rfsimulator.[0].options chanmod)
  fi

  say "starting gNB 106PRB vanilla: conf=${gnb_conf}, min_rxtxtime=${GNB_MIN_RXTXTIME}, rfsim_chanmod=${rfsim_chanmod}, awgn_profile=${awgn_profile}, awgn_noise_power_db=${awgn_noise:-config}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo "${sudo_env[@]}" ./nr-softmodem \
        -O "${OAI_RAN_CONF}/${gnb_conf}" \
        --gNBs.[0].min_rxtxtime "${GNB_MIN_RXTXTIME}" \
        --rfsim \
        "${chanmod_args[@]}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/gnb_106_stdout.log" 2>&1 &
  )
}

start_ue_106() {
  local ue_conf="$1"
  local rfsim_chanmod="$2"
  local awgn_profile="$3"
  local awgn_noise="$4"
  local t_args=()
  local chanmod_args=()

  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_UE_T_PORT:-2023}")
  fi
  if [[ "${rfsim_chanmod}" == "1" ]]; then
    chanmod_args=(--rfsimulator.[0].options chanmod)
  fi

  say "starting UE 106PRB: conf=${ue_conf}, prb=${UE_PRB}, freq=${UE_DL_FREQ}, rfsim_chanmod=${rfsim_chanmod}, awgn_profile=${awgn_profile}, awgn_noise_power_db=${awgn_noise:-config}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-uesoftmodem \
        --rfsim \
        --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER:-127.0.0.1}" \
        "${chanmod_args[@]}" \
        -r "${UE_PRB}" \
        --numerology "${UE_NUMEROLOGY}" \
        --band "${UE_BAND}" \
        -C "${UE_DL_FREQ}" \
        -O "${OAI_RAN_CONF}/${ue_conf}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/ue_106_stdout.log" 2>&1 &
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
    say "starting gNB ${TTRACER_GNB_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
    scripts/ttracer_record_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --duration-s "${TTRACER_DURATION_S}" \
      > "${LOG_ROOT}/ttracer_record_gnb_stdout.log" 2>&1 &
    GNB_RECORD_PID=$!
  fi
}

record_traffic_start() {
  local traffic="$1"
  local label="$2"
  local start_epoch start_hms
  start_epoch="$(date +%s.%N)"
  start_hms="$(date +%H:%M:%S.%N)"
  cat > "${CAP_ROOT}/traffic_interval.json" <<EOF
{
  "traffic": "${traffic}",
  "label": "${label}",
  "start_epoch": ${start_epoch},
  "start_hms": "${start_hms}",
  "duration_target_s": ${TRAFFIC_DURATION_S}
}
EOF
}

record_traffic_end() {
  local end_epoch end_hms tmp
  end_epoch="$(date +%s.%N)"
  end_hms="$(date +%H:%M:%S.%N)"
  tmp="${CAP_ROOT}/traffic_interval.tmp"
  "${PY}" - "${CAP_ROOT}/traffic_interval.json" "${tmp}" "${end_epoch}" "${end_hms}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
end_epoch = float(sys.argv[3])
end_hms = sys.argv[4]
data = json.loads(src.read_text())
data["end_epoch"] = end_epoch
data["end_hms"] = end_hms
data["elapsed_s"] = max(0.0, end_epoch - float(data["start_epoch"]))
dst.write_text(json.dumps(data, indent=2) + "\n")
PY
  mv "${tmp}" "${CAP_ROOT}/traffic_interval.json"
}

start_iperf_server() {
  say "starting iperf3 server in oai-ext-dn at ${OAI_EXT_DN_IP}"
  sudo docker exec oai-ext-dn sh -c "pkill -INT iperf3 2>/dev/null || true; rm -f /tmp/scenesense_iperf3_server.json /tmp/scenesense_iperf3_server.log" >/dev/null 2>&1 || true
  sudo docker exec -d oai-ext-dn sh -c "iperf3 -s -1 -B ${OAI_EXT_DN_IP} -J >/tmp/scenesense_iperf3_server.json 2>/tmp/scenesense_iperf3_server.log"
  sleep 2
}

run_iperf_client() {
  say "traffic start: iperf UDP uplink bitrate=${IPERF_BITRATE}, duration=${TRAFFIC_DURATION_S}s, bind=${OAI_UE_IP}, dst=${OAI_EXT_DN_IP}"
  record_traffic_start "iperf" "$1"
  iperf3 -c "${OAI_EXT_DN_IP}" \
    -u \
    -b "${IPERF_BITRATE}" \
    -t "${TRAFFIC_DURATION_S}" \
    -B "${OAI_UE_IP}" \
    -J \
    > "${CAP_ROOT}/iperf3_client.json" \
    2> "${LOG_ROOT}/iperf3_client.stderr"
  local rc=$?
  record_traffic_end
  say "traffic done: iperf rc=${rc}"
  sleep 3
  sudo docker cp oai-ext-dn:/tmp/scenesense_iperf3_server.json "${CAP_ROOT}/iperf3_server.json" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_iperf3_server.log "${LOG_ROOT}/iperf3_server.log" >/dev/null 2>&1 || true
  return "${rc}"
}

start_udp_tcpdump_sink() {
  say "starting tcpdump UDP sink capture in ext-DN ${OAI_EXT_DN_IP}:${REPLAY_PORT}"
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true; rm -f /tmp/scenesense_noncarla_udp_sink.pcap /tmp/scenesense_noncarla_udp_sink.log /tmp/scenesense_noncarla_udp_sink_packets.txt" >/dev/null 2>&1 || true
  sudo docker exec -d oai-ext-dn sh -c "tcpdump -i any -n -s 0 'udp and dst port ${REPLAY_PORT}' -w /tmp/scenesense_noncarla_udp_sink.pcap >/tmp/scenesense_noncarla_udp_sink.log 2>&1"
  sleep 1
}

run_tractor_replay() {
  say "traffic start: tractor replay trace=${TRACE_CSV}, offset=${REPLAY_START_OFFSET_S}s, duration=${REPLAY_MAX_DURATION_S}s, time_scale=${REPLAY_TIME_SCALE}, length_scale=${REPLAY_LENGTH_SCALE}, max_payload=${REPLAY_MAX_PAYLOAD}, mode=${REPLAY_LARGE_PACKET_MODE}"
  record_traffic_start "tractor" "$1"
  "${PY}" bursty_traffic/udp_trace_replay.py "${TRACE_CSV}" \
    --dst "${OAI_EXT_DN_IP}" \
    --port "${REPLAY_PORT}" \
    --bind-ip "${OAI_UE_IP}" \
    --direction uplink \
    --start-offset-s "${REPLAY_START_OFFSET_S}" \
    --max-duration-s "${REPLAY_MAX_DURATION_S}" \
    --time-scale "${REPLAY_TIME_SCALE}" \
    --length-scale "${REPLAY_LENGTH_SCALE}" \
    --max-payload "${REPLAY_MAX_PAYLOAD}" \
    --large-packet-mode "${REPLAY_LARGE_PACKET_MODE}" \
    > "${CAP_ROOT}/replay_sender.log" \
    2> "${LOG_ROOT}/replay_sender.stderr"
  local rc=$?
  record_traffic_end
  say "traffic done: tractor rc=${rc}"
  sudo docker exec oai-ext-dn sh -c "pkill -INT tcpdump 2>/dev/null || true" >/dev/null 2>&1 || true
  sleep 2
  sudo docker exec oai-ext-dn sh -c "tcpdump -tt -n -r /tmp/scenesense_noncarla_udp_sink.pcap 2>/tmp/scenesense_noncarla_udp_sink_read.log > /tmp/scenesense_noncarla_udp_sink_packets.txt || true" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_noncarla_udp_sink.pcap "${CAP_ROOT}/udp_sink.pcap" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_noncarla_udp_sink.log "${LOG_ROOT}/udp_sink_tcpdump.log" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_noncarla_udp_sink_read.log "${LOG_ROOT}/udp_sink_tcpdump_read.log" >/dev/null 2>&1 || true
  sudo docker cp oai-ext-dn:/tmp/scenesense_noncarla_udp_sink_packets.txt "${CAP_ROOT}/udp_sink_packets.txt" >/dev/null 2>&1 || true
  return "${rc}"
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
    > "${LOG_ROOT}/ttracer_extract_ue_stdout.log" 2>&1 || true

  if [[ "${RECORD_GNB}" == "1" ]]; then
    say "extracting gNB T-tracer CSV"
    scripts/ttracer_extract_csv_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --clean-output \
      > "${LOG_ROOT}/ttracer_extract_gnb_stdout.log" 2>&1 || true
  fi

  say "analyzing UE grant windows"
  "${PY}" scripts/analyze_nrue_grant_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s 1.0 \
    > "${LOG_ROOT}/analyze_nrue_grant_metrics_stdout.log" 2>&1 || true

  say "analyzing uplink layer latency where packet-layer traces permit it"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${RUN_GROUP}" \
    > "${LOG_ROOT}/analyze_uplink_layer_latency_stdout.log" 2>&1 || true

  cat > "${CAP_ROOT}/NONCARLA_VANILLA_AWGN_106PRB.ok" <<EOF
Validated non-CARLA vanilla OAI 106PRB traffic run.

Run group: ${RUN_GROUP}
Condition: ${CONDITION}
Batch ID: ${BATCH_ID}
Traffic: ${TRAFFIC}
Channel: ${CHANNEL}
gNB config: ${GNB_CONF_RUN}
UE config: ${UE_CONF_RUN}
UE launch: -r ${UE_PRB} -C ${UE_DL_FREQ}
RFsim chanmod: ${RFSIM_CHANMOD_RUN}
AWGN profile: ${AWGN_PROFILE_RUN}
AWGN noise_power_dB: ${AWGN_NOISE_POWER_DB_RUN:-config}
MCS policy: vanilla
T-tracer UE profile: ${TTRACER_UE_PROFILE}
T-tracer gNB profile: ${TTRACER_GNB_PROFILE}
iperf bitrate: ${IPERF_BITRATE}
tractor trace: ${TRACE_CSV}
tractor trace name: ${TRACE_NAME}
tractor offset s: ${REPLAY_START_OFFSET_S}
tractor max duration s: ${REPLAY_MAX_DURATION_S}
tractor time scale: ${REPLAY_TIME_SCALE}
tractor length scale: ${REPLAY_LENGTH_SCALE}
EOF
}

show_preflight() {
  echo "[noncarla-awgn106] Base batch: ${BASE_BATCH_ID}"
  echo "[noncarla-awgn106] Runs: ${RUNS}"
  echo "[noncarla-awgn106] Traffic duration: ${TRAFFIC_DURATION_S}s"
  echo "[noncarla-awgn106] iperf UDP bitrate: ${IPERF_BITRATE}"
  echo "[noncarla-awgn106] tractor: trace=${TRACE_CSV}, offset=${REPLAY_START_OFFSET_S}s, max_duration=${REPLAY_MAX_DURATION_S}s, length_scale=${REPLAY_LENGTH_SCALE}, time_scale=${REPLAY_TIME_SCALE}"
  echo "[noncarla-awgn106] OAI: 106PRB default 7DL/2UL, vanilla MCS only"
  echo "[noncarla-awgn106] T-tracer: UE=${TTRACER_UE_PROFILE}, gNB=${TTRACER_GNB_PROFILE}"
  if [[ -x "${OAI_RAN_BUILD}/nr-softmodem" ]]; then
    stat -c "[noncarla-awgn106] nr-softmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-softmodem"
  else
    echo "[noncarla-awgn106] ERROR: missing executable ${OAI_RAN_BUILD}/nr-softmodem" >&2
    return 1
  fi
  if [[ -x "${OAI_RAN_BUILD}/nr-uesoftmodem" ]]; then
    stat -c "[noncarla-awgn106] nr-uesoftmodem: %n mtime=%y size=%s" "${OAI_RAN_BUILD}/nr-uesoftmodem"
  else
    echo "[noncarla-awgn106] ERROR: missing executable ${OAI_RAN_BUILD}/nr-uesoftmodem" >&2
    return 1
  fi
  sha256sum \
    "${OAI_RAN_CONF}/gnb.sa.band78.fr1.106PRB.usrpb210.conf" \
    "${OAI_RAN_CONF}/gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf" \
    "${OAI_RAN_CONF}/ue.conf" \
    "${OAI_RAN_CONF}/ue.awgn_mild.conf" \
    "${OAI_ROOT}/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c" \
    2>/dev/null | sed 's/^/[noncarla-awgn106] sha256 /' || true
}

run_one() {
  local label="$1"
  TRAFFIC=""
  CHANNEL=""
  CONDITION=""
  BATCH_ID="${BASE_BATCH_ID}_${label}"
  GNB_CONF_RUN=""
  UE_CONF_RUN=""
  RFSIM_CHANMOD_RUN="0"
  AWGN_PROFILE_RUN="clear"
  AWGN_NOISE_POWER_DB_RUN=""

  case "${label}" in
    iperf_clear)
      TRAFFIC="iperf"; CHANNEL="clear"
      CONDITION="noncarla_iperf_clear_vanilla"
      GNB_CONF_RUN="gnb.sa.band78.fr1.106PRB.usrpb210.conf"
      UE_CONF_RUN="ue.conf"
      RFSIM_CHANMOD_RUN="0"; AWGN_PROFILE_RUN="clear"; AWGN_NOISE_POWER_DB_RUN=""
      ;;
    iperf_mild)
      TRAFFIC="iperf"; CHANNEL="mild_awgn"
      CONDITION="noncarla_iperf_mild_awgn_vanilla"
      GNB_CONF_RUN="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"
      UE_CONF_RUN="ue.awgn_mild.conf"
      RFSIM_CHANMOD_RUN="1"; AWGN_PROFILE_RUN="mild"; AWGN_NOISE_POWER_DB_RUN="-10"
      ;;
    tractor_clear)
      TRAFFIC="tractor"; CHANNEL="clear"
      CONDITION="noncarla_tractor_clear_vanilla"
      GNB_CONF_RUN="gnb.sa.band78.fr1.106PRB.usrpb210.conf"
      UE_CONF_RUN="ue.conf"
      RFSIM_CHANMOD_RUN="0"; AWGN_PROFILE_RUN="clear"; AWGN_NOISE_POWER_DB_RUN=""
      ;;
    tractor_mild)
      TRAFFIC="tractor"; CHANNEL="mild_awgn"
      CONDITION="noncarla_tractor_mild_awgn_vanilla"
      GNB_CONF_RUN="gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_mild.conf"
      UE_CONF_RUN="ue.awgn_mild.conf"
      RFSIM_CHANMOD_RUN="1"; AWGN_PROFILE_RUN="mild"; AWGN_NOISE_POWER_DB_RUN="-10"
      ;;
    *)
      echo "[noncarla-awgn106] ERROR: unknown run label '${label}'" >&2
      return 2
      ;;
  esac

  RUN_GROUP="${CONDITION}_${BATCH_ID}"
  CAP_ROOT="${AB}/metrics_logs/noncarla_awgn/${RUN_GROUP}"
  LOG_ROOT="${CAP_ROOT}/logs"
  mkdir -p "${LOG_ROOT}"

  say "===== START label=${label} run_group=${RUN_GROUP} ====="
  say "config: traffic=${TRAFFIC}, channel=${CHANNEL}, condition=${CONDITION}, batch=${BATCH_ID}, gNB=${GNB_CONF_RUN}, UE=${UE_CONF_RUN}, UE_PRB=${UE_PRB}, UE_FREQ=${UE_DL_FREQ}, min_rxtxtime=${GNB_MIN_RXTXTIME}, mcs_policy=vanilla, rfsim_chanmod=${RFSIM_CHANMOD_RUN}, awgn_profile=${AWGN_PROFILE_RUN}, awgn_noise_power_db=${AWGN_NOISE_POWER_DB_RUN:-config}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    say "DRY_RUN=1; skipping execution"
    return 0
  fi

  stop_ran
  restart_core
  start_gnb_106 "${GNB_CONF_RUN}" "${RFSIM_CHANMOD_RUN}" "${AWGN_PROFILE_RUN}" "${AWGN_NOISE_POWER_DB_RUN}"
  sleep 22
  start_ue_106 "${UE_CONF_RUN}" "${RFSIM_CHANMOD_RUN}" "${AWGN_PROFILE_RUN}" "${AWGN_NOISE_POWER_DB_RUN}"
  if ! wait_tunnel; then
    say "ERROR: UE tunnel attach failed"
    say "gNB log: ${LOG_ROOT}/gnb_106_stdout.log"
    say "UE log: ${LOG_ROOT}/ue_106_stdout.log"
    return 1
  fi

  start_network_sampler
  start_ttracer_recorders
  sleep 6

  local traffic_rc=0
  if [[ "${TRAFFIC}" == "iperf" ]]; then
    start_iperf_server
    run_iperf_client "${label}" || traffic_rc=$?
  else
    start_udp_tcpdump_sink
    run_tractor_replay "${label}" || traffic_rc=$?
  fi
  postprocess

  say "===== DONE label=${label} run_group=${RUN_GROUP} traffic_rc=${traffic_rc} ====="
  return "${traffic_rc}"
}

show_preflight || exit $?

for label in ${RUNS}; do
  run_one "${label}" || exit $?
done

echo "[noncarla-awgn106] done. Summarize with:"
echo "  ${PY} oai_mcs_policy_track2/summarize_noncarla_vanilla_awgn_106prb.py --base-batch ${BASE_BATCH_ID}"
