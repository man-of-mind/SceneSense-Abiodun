#!/usr/bin/env bash
# Hash-bound n78 100-MHz/273-PRB/4D5U one-UE OAI attachment for SplitFusion.
set -euo pipefail

ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PYTHON="/usr/bin/python3"
RUNNER="${ROOT}/rl_agent/splitfusion_phase14a_100mhz_calibration_v1.py"
CONFIG="${ROOT}/rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json"
TOKEN="SPLITFUSION_OAI_100MHZ_4D5U_ATTACH"
RAN_BUILD="${ROOT}/OAI/openairinterface5g/cmake_targets/ran_build/build"
CN_DIR="${ROOT}/OAI/oai-cn5g"
UE_INTERFACE="oaitun_ue1"
UE_IP="10.0.0.2"
EXT_DN_IP="192.168.70.135"

if [[ "${SPLITFUSION_BINDING_ONLY:-0}" == "1" || "${1:-}" == "--binding-only" ]]; then
  exec "${PYTHON}" "${RUNNER}" --config "${CONFIG}" --reconcile-only
fi

if [[ "$#" -ne 4 || "$1" != "--execute" || "$2" != "${TOKEN}" || "$3" != "--output" || -z "$4" ]]; then
  echo "usage: $0 --execute ${TOKEN} --output experiments/splitfusion_oai_100mhz_4d5u_v1/RUN_ID" >&2
  exit 2
fi
OUTPUT="$4"

# These names belong to legacy 40-MHz launchers. Refuse rather than override
# them, even if a value happens to describe the selected radio.
for legacy_override in \
  GNB_CONF GNB_CONF_DEFAULT UE_CONF UE_CONF_DEFAULT UE_PRB UE_NUMEROLOGY \
  UE_BAND UE_DL_FREQ UE_SSB AWGN_PROFILE AWGN_NOISE_POWER_DB; do
  if [[ -n "${!legacy_override:-}" ]]; then
    echo "ERROR: legacy radio override ${legacy_override} is forbidden" >&2
    exit 2
  fi
done

for required in \
  "${RAN_BUILD}/nr-softmodem" \
  "${RAN_BUILD}/nr-uesoftmodem" \
  "${RAN_BUILD}/libtelnetsrv.so" \
  "${CN_DIR}/docker-compose.yaml"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required OAI input is missing: ${required}" >&2
    exit 2
  fi
done

if pgrep -x nr-softmodem >/dev/null || pgrep -x nr-uesoftmodem >/dev/null; then
  echo "ERROR: launcher requires a cold RAN; an OAI softmodem is already active" >&2
  exit 2
fi
if ip link show "${UE_INTERFACE}" >/dev/null 2>&1; then
  echo "ERROR: launcher requires no stale ${UE_INTERFACE}" >&2
  exit 2
fi
if ! sudo -n true; then
  echo "ERROR: noninteractive sudo is unavailable" >&2
  exit 2
fi

"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" \
  --materialize-radio-config \
  --output "${OUTPUT}"

STATE_DIR="$("${PYTHON}" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); print((p if p.is_absolute() else Path(sys.argv[2])/p).resolve(strict=True))' "${OUTPUT}" "${ROOT}")"
MATERIALIZATION="${STATE_DIR}/radio_materialization.json"
GNB_CONFIG="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["effective_gnb_path"])' "${MATERIALIZATION}")"
UE_CONFIG="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["effective_ue_path"])' "${MATERIALIZATION}")"
GNB_LOG="${STATE_DIR}/gnb.log"
UE_LOG="${STATE_DIR}/ue.log"
GNB_PID=""
UE_PID=""
LAUNCH_COMPLETE=0

cleanup_failed_launch() {
  local rc=$?
  if [[ "${LAUNCH_COMPLETE}" != "1" ]]; then
    for pid in "${UE_PID}" "${GNB_PID}"; do
      if [[ -n "${pid}" ]] && sudo -n kill -0 "${pid}" 2>/dev/null; then
        sudo -n kill -INT -- "-${pid}" 2>/dev/null || sudo -n kill -INT "${pid}" 2>/dev/null || true
      fi
    done
  fi
  exit "${rc}"
}
trap cleanup_failed_launch EXIT

(
  cd "${CN_DIR}"
  sudo -n docker compose up -d --force-recreate --remove-orphans
)

setsid nohup sudo -n env \
  -u SCENESENSE_FORCE_UL_MCS \
  -u SCENESENSE_HOLD_MCS_FEW_SAMPLES \
  -u SCENESENSE_AIMD_MAX_DROP \
  SCENESENSE_MCS_POLICY=sinr \
  "${RAN_BUILD}/nr-softmodem" \
  -O "${GNB_CONFIG}" \
  --gNBs.[0].min_rxtxtime 6 \
  --rfsim \
  --rfsimulator.[0].options chanmod \
  --telnetsrv \
  --telnetsrv.listenaddr 127.0.0.1 \
  --telnetsrv.listenport 9090 \
  --T_stdout 2 \
  --T_nowait \
  --T_port 2021 \
  >"${GNB_LOG}" 2>&1 </dev/null &
GNB_PID=$!

sleep 5
if ! sudo -n kill -0 "${GNB_PID}" 2>/dev/null; then
  echo "ERROR: 100-MHz gNB exited during startup" >&2
  exit 1
fi

setsid nohup sudo -n \
  "${RAN_BUILD}/nr-uesoftmodem" \
  --rfsim \
  --rfsimulator.[0].serveraddr 127.0.0.1 \
  --rfsimulator.[0].options chanmod \
  -r 273 \
  --numerology 1 \
  --band 78 \
  -C 3649260000 \
  --ssb 516 \
  -O "${UE_CONFIG}" \
  --T_stdout 2 \
  --T_nowait \
  --T_port 2023 \
  >"${UE_LOG}" 2>&1 </dev/null &
UE_PID=$!

attached=0
for _ in $(seq 1 120); do
  if ! sudo -n kill -0 "${GNB_PID}" 2>/dev/null || ! sudo -n kill -0 "${UE_PID}" 2>/dev/null; then
    echo "ERROR: 100-MHz gNB or UE exited before attachment" >&2
    exit 1
  fi
  if ip -j -4 addr show dev "${UE_INTERFACE}" 2>/dev/null |
    "${PYTHON}" -c 'import json,sys; d=json.load(sys.stdin); a=[x["local"] for r in d for x in r.get("addr_info",[]) if x.get("family")=="inet"]; raise SystemExit(0 if a==["10.0.0.2"] else 1)'; then
    if ping -I "${UE_INTERFACE}" -c 3 -W 2 "${EXT_DN_IP}" >/dev/null; then
      attached=1
      break
    fi
  fi
  sleep 1
done
if [[ "${attached}" != "1" ]]; then
  echo "ERROR: ${UE_INTERFACE} did not attach as ${UE_IP} and reach ${EXT_DN_IP}" >&2
  exit 1
fi

"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" \
  --record-attached-radio \
  --radio-state "${STATE_DIR}"

LAUNCH_COMPLETE=1
trap - EXIT
echo "SPLITFUSION_OAI_100MHZ_4D5U_ATTACHED ${STATE_DIR}"
