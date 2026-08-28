#!/usr/bin/env bash
# One sequential, fail-closed chain: train -> evaluate. Writes a completion sentinel.
# No polling loops; the sentinel and the per-phase logs are the progress record.
set -euo pipefail

ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PKG="${ROOT}/pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
EXP="${ROOT}/$(cat "${PKG}/EXPERIMENT_DIR.txt")"

cd "${ROOT}"
echo $$ > "${EXP}/pipeline.pid"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${EXP}/pipeline.log"; }

fail() {
  local phase="$1"
  log "PIPELINE_FAILED phase=${phase}"
  if [ ! -f "${EXP}/TERMINAL_VERDICT.txt" ]; then
    echo "LRASPP_NATIVE_GRID_RUNTIME_FAILURE" > "${EXP}/TERMINAL_VERDICT.txt"
  fi
  echo "FAILED ${phase}" > "${EXP}/PIPELINE_SENTINEL"
  exit 1
}

log "PIPELINE_START exp=${EXP}"

log "PHASE_TRAIN_START"
if ! python3 "${PKG}/train_native_v1.py" \
      --experiment "${EXP}" \
      --config "${PKG}/configs/route_b_v3_1_native_grid_v1.yaml" \
      --trial "${PKG}/configs/native_grid_training_v1.json" \
      >> "${EXP}/training.log" 2>&1; then
  fail train
fi
log "PHASE_TRAIN_DONE"

log "PHASE_EVAL_START"
if ! python3 "${PKG}/evaluate_v1.py" \
      --experiment "${EXP}" \
      --contract "${PKG}/configs/selection_contract_v1.json" \
      >> "${EXP}/evaluation.log" 2>&1; then
  fail evaluate
fi
log "PHASE_EVAL_DONE"

VERDICT="$(cat "${EXP}/TERMINAL_VERDICT.txt")"
log "PIPELINE_DONE verdict=${VERDICT}"
echo "DONE ${VERDICT}" > "${EXP}/PIPELINE_SENTINEL"

# Desktop notification if one is available; never fatal.
command -v notify-send >/dev/null 2>&1 && \
  notify-send "native-grid pilot" "${VERDICT}" || true
