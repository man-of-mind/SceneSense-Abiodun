#!/usr/bin/env bash
# Record a short OAI T-tracer raw file from the gNB or UE softmodem.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.env"

RUN_GROUP="$(date +%Y%m%d_%H%M%S)_ttracer_smoke"
SOURCE=""
DURATION_S=30
IP="127.0.0.1"
PORT=""
OUTPUT_ROOT="${ABIODUN_DIR}/metrics_logs/scenesense_ttracer"
PROFILE=""
EXTRA_TRACES=()

usage() {
  cat <<'EOF'
Usage:
  ttracer_record_smoke.sh --source gnb|ue [options]

Options:
  --run-group LABEL       Label shared with app/network metrics.
  --source gnb|ue         T source to record from.
  --duration-s SECONDS    Recording duration; default: 30.
  --ip HOST               Trace source IP; default: 127.0.0.1.
  --port PORT             Override tracer port.
  --output-root DIR       Output root; default: metrics_logs/scenesense_ttracer.
  --profile NAME          Trace profile. UE: clean, payload, legacy/full.
                          gNB: clean/full. Defaults to clean for UE, full for gNB.
  --trace EVENT_ID        Add an extra event to the default smoke profile.
  -h, --help              Show this help.

Expected softmodem launch flags:
  --T_stdout 2 --T_nowait --T_port <port>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-group)
      RUN_GROUP="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --duration-s)
      DURATION_S="$2"
      shift 2
      ;;
    --ip)
      IP="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --trace)
      EXTRA_TRACES+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ttracer_record_smoke] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${SOURCE}" != "gnb" && "${SOURCE}" != "ue" ]]; then
  echo "[ttracer_record_smoke] --source must be gnb or ue" >&2
  usage >&2
  exit 2
fi

if [[ ! "${DURATION_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ttracer_record_smoke] --duration-s must be numeric" >&2
  exit 2
fi

if [[ -z "${PORT}" ]]; then
  if [[ "${SOURCE}" == "gnb" ]]; then
    PORT="${OAI_GNB_T_PORT:-2021}"
  else
    PORT="${OAI_UE_T_PORT:-2023}"
  fi
fi

if [[ ! -x "${OAI_T_TRACER_DIR}/record" ]]; then
  echo "[ttracer_record_smoke] missing ${OAI_T_TRACER_DIR}/record" >&2
  echo "[ttracer_record_smoke] run: ${SCRIPT_DIR}/ttracer_build_tools.sh" >&2
  exit 1
fi

if [[ ! -f "${OAI_T_MESSAGES}" ]]; then
  echo "[ttracer_record_smoke] missing T database: ${OAI_T_MESSAGES}" >&2
  exit 1
fi

if [[ -z "${PROFILE}" ]]; then
  if [[ "${SOURCE}" == "gnb" ]]; then
    PROFILE="full"
  else
    PROFILE="clean"
  fi
fi

if [[ "${SOURCE}" == "gnb" ]]; then
  case "${PROFILE}" in
    clean|full)
      ;;
    *)
      echo "[ttracer_record_smoke] unsupported gNB profile: ${PROFILE}" >&2
      echo "[ttracer_record_smoke] supported gNB profiles: clean, full" >&2
      exit 2
      ;;
  esac
  DEFAULT_TRACES=(
    GNB_MAC_UL
    GNB_MAC_DL
    GNB_MAC_LCID_UL
    GNB_MAC_LCID_DL
    GNB_MAC_PUSCH_POWER_CONTROL
    GNB_MAC_PUCCH_POWER_CONTROL
    ENB_RLC_UL
    ENB_RLC_DL
    ENB_RLC_MAC_UL
    ENB_RLC_MAC_DL
    ENB_PDCP_UL
    ENB_PDCP_DL
    GNB_PHY_UL_PAYLOAD_RX_BITS
    GNB_PHY_UL_TICK
  )
else
  case "${PROFILE}" in
    clean)
      DEFAULT_TRACES=(
        NRUE_MAC_DCI_GRANT
      )
      ;;
    payload)
      DEFAULT_TRACES=(
        NRUE_MAC_DCI_GRANT
        UE_PHY_UL_PAYLOAD_TX_BITS
      )
      ;;
    legacy|full)
      DEFAULT_TRACES=(
        NRUE_MAC_DCI_GRANT
        UE_PHY_MEAS
        UE_PHY_ULSCH_UE_DCI
        UE_PHY_DLSCH_UE_DCI
        UE_PHY_UL_PAYLOAD_TX_BITS
        UE_PHY_UL_TICK
      )
      ;;
    *)
      echo "[ttracer_record_smoke] unsupported UE profile: ${PROFILE}" >&2
      echo "[ttracer_record_smoke] supported UE profiles: clean, payload, legacy, full" >&2
      exit 2
      ;;
  esac
fi

TRACE_ARGS=(-OFF)
for trace in "${DEFAULT_TRACES[@]}" "${EXTRA_TRACES[@]}"; do
  TRACE_ARGS+=(-on "${trace}")
done

OUT_DIR="${OUTPUT_ROOT}/${RUN_GROUP}/${SOURCE}"
RAW_PATH="${OUT_DIR}/${SOURCE}.raw"
LOG_PATH="${OUT_DIR}/${SOURCE}_record.log"
MANIFEST_PATH="${OUT_DIR}/${SOURCE}_record_manifest.json"
mkdir -p "${OUT_DIR}"

printf '{\n' > "${MANIFEST_PATH}"
printf '  "created_at": "%s",\n' "$(date -Is)" >> "${MANIFEST_PATH}"
printf '  "run_group": "%s",\n' "${RUN_GROUP}" >> "${MANIFEST_PATH}"
printf '  "source": "%s",\n' "${SOURCE}" >> "${MANIFEST_PATH}"
printf '  "ip": "%s",\n' "${IP}" >> "${MANIFEST_PATH}"
printf '  "port": %s,\n' "${PORT}" >> "${MANIFEST_PATH}"
printf '  "duration_s": %s,\n' "${DURATION_S}" >> "${MANIFEST_PATH}"
printf '  "raw_path": "%s",\n' "${RAW_PATH}" >> "${MANIFEST_PATH}"
printf '  "profile": "%s",\n' "${PROFILE}" >> "${MANIFEST_PATH}"
printf '  "trace_profile": "%s"\n' "${DEFAULT_TRACES[*]} ${EXTRA_TRACES[*]}" >> "${MANIFEST_PATH}"
printf '}\n' >> "${MANIFEST_PATH}"

echo "[ttracer_record_smoke] source=${SOURCE} ${IP}:${PORT}"
echo "[ttracer_record_smoke] run_group=${RUN_GROUP}"
echo "[ttracer_record_smoke] duration=${DURATION_S}s"
echo "[ttracer_record_smoke] raw=${RAW_PATH}"
echo "[ttracer_record_smoke] profile=${PROFILE}"
echo "[ttracer_record_smoke] traces=${DEFAULT_TRACES[*]} ${EXTRA_TRACES[*]}"

set +e
timeout --foreground --signal=INT "${DURATION_S}s" \
  "${OAI_T_TRACER_DIR}/record" \
  -d "${OAI_T_MESSAGES}" \
  -ip "${IP}" \
  -p "${PORT}" \
  -o "${RAW_PATH}" \
  "${TRACE_ARGS[@]}" \
  > "${LOG_PATH}" 2>&1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 && "${STATUS}" -ne 124 && "${STATUS}" -ne 130 ]]; then
  echo "[ttracer_record_smoke] record failed with status ${STATUS}; see ${LOG_PATH}" >&2
  exit "${STATUS}"
fi

if [[ ! -s "${RAW_PATH}" ]]; then
  echo "[ttracer_record_smoke] raw file is missing or empty: ${RAW_PATH}" >&2
  echo "[ttracer_record_smoke] check that the ${SOURCE} softmodem is running with T-tracer enabled." >&2
  echo "[ttracer_record_smoke] log: ${LOG_PATH}" >&2
  exit 1
fi

echo "[ttracer_record_smoke] wrote ${RAW_PATH}"
echo "[ttracer_record_smoke] log ${LOG_PATH}"
