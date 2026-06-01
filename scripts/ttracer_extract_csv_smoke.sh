#!/usr/bin/env bash
# Extract a small CSV panel from a recorded OAI T-tracer raw file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/config.env"

RUN_GROUP=""
SOURCE=""
RAW_PATH=""
OUTPUT_ROOT="${ABIODUN_DIR}/metrics_logs/scenesense_ttracer"
REPLAY_PORT=""
REPLAY_TIMEOUT_S=20
PROFILE=""
CLEAN_OUTPUT=0
EVENTS=()

usage() {
  cat <<'EOF'
Usage:
  ttracer_extract_csv_smoke.sh --run-group LABEL --source gnb|ue [options]
  ttracer_extract_csv_smoke.sh --raw PATH --source gnb|ue [options]

Options:
  --run-group LABEL       Locate raw file under metrics_logs/scenesense_ttracer.
  --source gnb|ue         Source profile to extract.
  --raw PATH              Explicit raw file path.
  --output-root DIR       Output root; default: metrics_logs/scenesense_ttracer.
  --replay-port PORT      Local replay port; defaults to 2201 for gNB, 2203 for UE.
  --timeout-s SECONDS     Per-event CSV extraction timeout; default: 20.
  --profile NAME          Extraction profile. UE: clean, payload, legacy/full.
                          gNB: clean/full. Defaults to clean for UE, full for gNB.
  --clean-output          Remove existing CSVs in the output csv/ folder first.
  --event EVENT_ID        Extract only this event; can be repeated.
  -h, --help              Show this help.

The script writes one CSV per event under:
  metrics_logs/scenesense_ttracer/<run_group>/<source>/csv/
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
    --raw)
      RAW_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --replay-port)
      REPLAY_PORT="$2"
      shift 2
      ;;
    --timeout-s)
      REPLAY_TIMEOUT_S="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --clean-output)
      CLEAN_OUTPUT=1
      shift
      ;;
    --event)
      EVENTS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ttracer_extract_csv_smoke] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${SOURCE}" != "gnb" && "${SOURCE}" != "ue" ]]; then
  echo "[ttracer_extract_csv_smoke] --source must be gnb or ue" >&2
  usage >&2
  exit 2
fi

if [[ -z "${RAW_PATH}" ]]; then
  if [[ -z "${RUN_GROUP}" ]]; then
    echo "[ttracer_extract_csv_smoke] provide --run-group or --raw" >&2
    usage >&2
    exit 2
  fi
  RAW_PATH="${OUTPUT_ROOT}/${RUN_GROUP}/${SOURCE}/${SOURCE}.raw"
fi

if [[ -z "${RUN_GROUP}" ]]; then
  RUN_GROUP="$(basename "$(dirname "$(dirname "${RAW_PATH}")")")"
fi

if [[ -z "${REPLAY_PORT}" ]]; then
  if [[ "${SOURCE}" == "gnb" ]]; then
    REPLAY_PORT=2201
  else
    REPLAY_PORT=2203
  fi
fi

for tool in replay csv extract_config; do
  if [[ ! -x "${OAI_T_TRACER_DIR}/${tool}" ]]; then
    echo "[ttracer_extract_csv_smoke] missing ${OAI_T_TRACER_DIR}/${tool}" >&2
    echo "[ttracer_extract_csv_smoke] run: ${SCRIPT_DIR}/ttracer_build_tools.sh" >&2
    exit 1
  fi
done

if [[ ! -s "${RAW_PATH}" ]]; then
  echo "[ttracer_extract_csv_smoke] raw file is missing or empty: ${RAW_PATH}" >&2
  exit 1
fi

if [[ -z "${PROFILE}" ]]; then
  if [[ "${SOURCE}" == "gnb" ]]; then
    PROFILE="full"
  else
    PROFILE="clean"
  fi
fi

if [[ "${#EVENTS[@]}" -eq 0 ]]; then
  if [[ "${SOURCE}" == "gnb" ]]; then
    case "${PROFILE}" in
      clean|full)
        EVENTS=(
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
        )
        ;;
      *)
        echo "[ttracer_extract_csv_smoke] unsupported gNB profile: ${PROFILE}" >&2
        echo "[ttracer_extract_csv_smoke] supported gNB profiles: clean, full" >&2
        exit 2
        ;;
    esac
  else
    case "${PROFILE}" in
      clean)
        EVENTS=(
          NRUE_MAC_DCI_GRANT
        )
        ;;
      payload)
        EVENTS=(
          NRUE_MAC_DCI_GRANT
          UE_PHY_UL_PAYLOAD_TX_BITS
        )
        ;;
      legacy|full)
        EVENTS=(
          NRUE_MAC_DCI_GRANT
          UE_PHY_MEAS
          UE_PHY_ULSCH_UE_DCI
          UE_PHY_DLSCH_UE_DCI
          UE_PHY_UL_PAYLOAD_TX_BITS
        )
        ;;
      *)
        echo "[ttracer_extract_csv_smoke] unsupported UE profile: ${PROFILE}" >&2
        echo "[ttracer_extract_csv_smoke] supported UE profiles: clean, payload, legacy, full" >&2
        exit 2
        ;;
    esac
  fi
fi

fields_for_event() {
  case "$1" in
    GNB_MAC_UL|GNB_MAC_DL)
      echo "time rnti frame slot mcs tbs"
      ;;
    GNB_MAC_LCID_UL)
      echo "time rnti frame slot lcid data_size"
      ;;
    GNB_MAC_LCID_DL)
      echo "time rnti frame slot lcid data_size tx_list_occupancy"
      ;;
    GNB_MAC_PUSCH_POWER_CONTROL)
      echo "time rnti frame slot snrx10 phr tpc tb_size txpower_calc rbSize mcs rssi"
      ;;
    GNB_MAC_PUCCH_POWER_CONTROL)
      echo "time rnti frame slot snrx10 tpc rssi"
      ;;
    ENB_RLC_UL|ENB_RLC_DL|ENB_RLC_MAC_UL|ENB_RLC_MAC_DL|ENB_PDCP_UL|ENB_PDCP_DL)
      echo "time eNB_ID rnti rb_id length"
      ;;
    GNB_PHY_UL_PAYLOAD_RX_BITS)
      echo "time frame slot rnti rb_size rb_start qam_mod_order mcs_index number_of_bits"
      ;;
    NRUE_MAC_DCI_GRANT)
      echo "time direction dci_format rnti_type rnti dci_frame dci_slot sched_frame sched_slot mcs mcs_table rb_start rb_size start_symbol nr_symbols tbs harq_pid ndi rv round qam_mod_order target_code_rate tpc n_cce N_cce"
      ;;
    UE_PHY_MEAS)
      echo "time eNB_ID frame subframe rsrp rssi snr rx_power noise_power w_cqi freq_offset"
      ;;
    UE_PHY_ULSCH_UE_DCI)
      echo "time eNB_ID frame subframe rnti harq_pid mcs round first_rb nb_rb TBS"
      ;;
    UE_PHY_DLSCH_UE_DCI)
      echo "time eNB_ID frame subframe rnti dci_format harq_pid mcs TBS"
      ;;
    UE_PHY_UL_PAYLOAD_TX_BITS)
      echo "time frame slot rnti rb_size rb_start qam_mod_order mcs_index number_of_bits"
      ;;
    *)
      return 1
      ;;
  esac
}

OUT_DIR="${OUTPUT_ROOT}/${RUN_GROUP}/${SOURCE}"
CSV_DIR="${OUT_DIR}/csv"
mkdir -p "${CSV_DIR}"
if [[ "${CLEAN_OUTPUT}" -eq 1 ]]; then
  find "${CSV_DIR}" -maxdepth 1 -type f -name '*.csv' -delete
fi

EXTRACTED_DB="${OUT_DIR}/${SOURCE}_extracted_T_messages.txt"
EXTRACT_LOG="${OUT_DIR}/${SOURCE}_extract_config.log"
set +e
"${OAI_T_TRACER_DIR}/extract_config" -i "${RAW_PATH}" > "${EXTRACTED_DB}" 2> "${EXTRACT_LOG}"
EXTRACT_STATUS=$?
set -e

if [[ "${EXTRACT_STATUS}" -eq 0 && -s "${EXTRACTED_DB}" ]]; then
  T_DB="${EXTRACTED_DB}"
else
  T_DB="${OAI_T_MESSAGES}"
fi

echo "[ttracer_extract_csv_smoke] raw=${RAW_PATH}"
echo "[ttracer_extract_csv_smoke] db=${T_DB}"
echo "[ttracer_extract_csv_smoke] csv_dir=${CSV_DIR}"
echo "[ttracer_extract_csv_smoke] profile=${PROFILE}"

for event in "${EVENTS[@]}"; do
  if ! FIELD_STRING="$(fields_for_event "${event}")"; then
    echo "[ttracer_extract_csv_smoke] no default field list for ${event}" >&2
    exit 2
  fi
  read -r -a FIELDS <<< "${FIELD_STRING}"

  CSV_PATH="${CSV_DIR}/${event}.csv"
  REPLAY_LOG="${OUT_DIR}/${event}_replay.log"
  CSV_LOG="${OUT_DIR}/${event}_csv.log"

  echo "[ttracer_extract_csv_smoke] extracting ${event} -> ${CSV_PATH}"

  "${OAI_T_TRACER_DIR}/replay" -i "${RAW_PATH}" -p "${REPLAY_PORT}" \
    > "${REPLAY_LOG}" 2>&1 &
  REPLAY_PID=$!
  sleep 0.5

  set +e
  timeout --foreground --signal=INT "${REPLAY_TIMEOUT_S}s" \
    "${OAI_T_TRACER_DIR}/csv" \
    -d "${T_DB}" \
    -ip 127.0.0.1 \
    -p "${REPLAY_PORT}" \
    -t time \
    "${event}" \
    "${FIELDS[@]}" \
    > "${CSV_PATH}" 2> "${CSV_LOG}"
  CSV_STATUS=$?
  set -e

  if kill -0 "${REPLAY_PID}" 2>/dev/null; then
    kill "${REPLAY_PID}" 2>/dev/null || true
  fi
  wait "${REPLAY_PID}" 2>/dev/null || true

  if [[ "${CSV_STATUS}" -ne 0 && "${CSV_STATUS}" -ne 124 && "${CSV_STATUS}" -ne 130 ]]; then
    echo "[ttracer_extract_csv_smoke] csv extraction failed for ${event}; see ${CSV_LOG}" >&2
    exit "${CSV_STATUS}"
  fi

  if [[ ! -s "${CSV_PATH}" ]]; then
    echo "[ttracer_extract_csv_smoke] warning: ${event} produced an empty CSV" >&2
  fi
done

echo "[ttracer_extract_csv_smoke] done"
