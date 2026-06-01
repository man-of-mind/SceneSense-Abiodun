#!/usr/bin/env bash
# Run the SceneSense logging-validation analysis suite for one run_group.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_GROUP=""
WINDOW_S="1.0"
NO_PLOTS=0

usage() {
  cat <<'EOF'
Usage:
  run_logging_validation_analysis.sh --run-group LABEL [options]

Options:
  --run-group LABEL     Shared run_group used by app, network, and T-tracer logs.
  --window-s SECONDS    Window size for UE grant aggregation; default: 1.0.
  --no-plots            Pass --no-plots to application/network analysis.
  -h, --help            Show this help.

This helper runs the available post-processing steps:
  - application + UE tunnel summary
  - UE NR grant window summary
  - UE grant-vs-payload validation
  - UE-vs-gNB grant comparison, when both sides are populated
  - gNB stdout MAC parser, when gNB stdout was captured
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-group)
      RUN_GROUP="$2"
      shift 2
      ;;
    --window-s)
      WINDOW_S="$2"
      shift 2
      ;;
    --no-plots)
      NO_PLOTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[logging-analysis] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_GROUP}" ]]; then
  echo "[logging-analysis] --run-group is required" >&2
  usage >&2
  exit 2
fi

TTRACER_DIR="${ABIODUN_DIR}/metrics_logs/scenesense_ttracer/${RUN_GROUP}"
UE_CSV_DIR="${TTRACER_DIR}/ue/csv"
GNB_CSV_DIR="${TTRACER_DIR}/gnb/csv"
GNB_STDOUT="${TTRACER_DIR}/gnb/stdout/gnb_stdout.log"
MANIFEST="${TTRACER_DIR}/analysis/logging_validation_manifest.txt"
mkdir -p "$(dirname "${MANIFEST}")"

line_count() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    wc -l < "${path}" | tr -d ' '
  else
    echo 0
  fi
}

run_step() {
  local name="$1"
  shift
  echo "[logging-analysis] ${name}"
  "$@"
}

{
  echo "run_group=${RUN_GROUP}"
  echo "created_at=$(date -Is)"
  echo "window_s=${WINDOW_S}"
} > "${MANIFEST}"

cd "${ABIODUN_DIR}"

APP_MATCHES=0
if [[ -d "${ABIODUN_DIR}/metrics_logs/scenesense_runs" ]]; then
  if rg -l --glob '*_metrics.csv' "${RUN_GROUP}" "${ABIODUN_DIR}/metrics_logs/scenesense_runs" >/dev/null 2>&1; then
    APP_MATCHES=1
  fi
fi
if [[ "${APP_MATCHES}" -eq 1 ]]; then
  APP_ARGS=(--run-group "${RUN_GROUP}")
  if [[ "${NO_PLOTS}" -eq 1 ]]; then
    APP_ARGS+=(--no-plots)
  fi
  run_step "application + tunnel metrics" python3 scripts/analyze_scenesense_app_metrics.py "${APP_ARGS[@]}"
  echo "application_analysis=done" >> "${MANIFEST}"
else
  echo "[logging-analysis] skip application analysis; no app metrics found for ${RUN_GROUP}"
  echo "application_analysis=skipped" >> "${MANIFEST}"
fi

UE_GRANT="${UE_CSV_DIR}/NRUE_MAC_DCI_GRANT.csv"
UE_PAYLOAD="${UE_CSV_DIR}/UE_PHY_UL_PAYLOAD_TX_BITS.csv"
UE_GRANT_LINES="$(line_count "${UE_GRANT}")"
UE_PAYLOAD_LINES="$(line_count "${UE_PAYLOAD}")"
echo "ue_grant_lines=${UE_GRANT_LINES}" >> "${MANIFEST}"
echo "ue_payload_lines=${UE_PAYLOAD_LINES}" >> "${MANIFEST}"

if [[ "${UE_GRANT_LINES}" -gt 1 ]]; then
  run_step "UE NR grant windows" python3 scripts/analyze_nrue_grant_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s "${WINDOW_S}"
  echo "ue_grant_analysis=done" >> "${MANIFEST}"
else
  echo "[logging-analysis] skip UE grant analysis; missing populated ${UE_GRANT}"
  echo "ue_grant_analysis=skipped" >> "${MANIFEST}"
fi

if [[ "${UE_GRANT_LINES}" -gt 1 && "${UE_PAYLOAD_LINES}" -gt 1 ]]; then
  run_step "UE grant-vs-payload validation" python3 scripts/validate_nrue_grant_payload.py \
    --run-group "${RUN_GROUP}"
  echo "ue_payload_validation=done" >> "${MANIFEST}"
else
  echo "[logging-analysis] skip UE grant-vs-payload validation; payload profile was not captured"
  echo "ue_payload_validation=skipped" >> "${MANIFEST}"
fi

GNB_UL="${GNB_CSV_DIR}/GNB_MAC_UL.csv"
GNB_DL="${GNB_CSV_DIR}/GNB_MAC_DL.csv"
GNB_PHY_UL="${GNB_CSV_DIR}/GNB_PHY_UL_PAYLOAD_RX_BITS.csv"
GNB_UL_LINES="$(line_count "${GNB_UL}")"
GNB_DL_LINES="$(line_count "${GNB_DL}")"
GNB_PHY_LINES="$(line_count "${GNB_PHY_UL}")"
echo "gnb_mac_ul_lines=${GNB_UL_LINES}" >> "${MANIFEST}"
echo "gnb_mac_dl_lines=${GNB_DL_LINES}" >> "${MANIFEST}"
echo "gnb_phy_ul_lines=${GNB_PHY_LINES}" >> "${MANIFEST}"

if [[ "${UE_GRANT_LINES}" -gt 1 && "${GNB_UL_LINES}" -gt 1 && "${GNB_DL_LINES}" -gt 1 && "${GNB_PHY_LINES}" -gt 1 ]]; then
  run_step "UE-vs-gNB grant comparison" python3 scripts/compare_nrue_gnb_grants.py \
    --run-group "${RUN_GROUP}"
  echo "ue_gnb_comparison=done" >> "${MANIFEST}"
else
  echo "[logging-analysis] skip UE-vs-gNB comparison; both UE and gNB CSV panels must be populated"
  echo "ue_gnb_comparison=skipped" >> "${MANIFEST}"
fi

if [[ -s "${GNB_STDOUT}" ]]; then
  run_step "gNB stdout MAC parser" python3 scripts/parse_oai_gnb_mac_stats.py \
    --input "${GNB_STDOUT}" \
    --output-dir "${TTRACER_DIR}/gnb/stdout_parsed"
  echo "gnb_stdout_parse=done" >> "${MANIFEST}"
else
  echo "[logging-analysis] skip gNB stdout parser; missing ${GNB_STDOUT}"
  echo "gnb_stdout_parse=skipped" >> "${MANIFEST}"
fi

echo "[logging-analysis] manifest ${MANIFEST}"
