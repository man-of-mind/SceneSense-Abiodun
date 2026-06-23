#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)_radar_ablation}"
ABLATION_CONFIGS="${ABLATION_CONFIGS:-5000:bbox 5000:radius 12000:bbox 12000:radius}"
LOOPS_PER_DENSITY="${LOOPS_PER_DENSITY:-2}"
MIN_SAMPLES_PER_DENSITY="${MIN_SAMPLES_PER_DENSITY:-1200}"
MAX_SAMPLES_PER_DENSITY="${MAX_SAMPLES_PER_DENSITY:-2200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BUDGET_HOURS="${TRAIN_BUDGET_HOURS:-3.0}"
RUN_COLLECTION="${RUN_COLLECTION:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-1}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

check_cuda() {
  if [[ "$REQUIRE_CUDA" != "1" ]]; then
    log "Skipping CUDA guard because REQUIRE_CUDA=$REQUIRE_CUDA."
    return
  fi
  python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run ablation on CPU.")
print("CUDA OK:", torch.cuda.get_device_name(0))
PY
}

stop_carla_server() {
  if [[ "$STOP_CARLA_BEFORE_TRAINING" != "1" ]]; then
    log "Leaving CARLA running because STOP_CARLA_BEFORE_TRAINING=$STOP_CARLA_BEFORE_TRAINING."
    return
  fi
  local patterns=("CarlaUnreal.sh" "CarlaUnreal-Linux-Shipping" "CarlaUE4-Linux-Shipping" "CarlaUE4")
  local found=0
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      found=1
      log "Sending SIGINT to CARLA process pattern: $pattern"
      pkill -INT -f "$pattern" || true
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    log "No CARLA server process found."
    return
  fi
  sleep "$CARLA_STOP_GRACE_S"
  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      log "CARLA still running for pattern $pattern; sending SIGTERM."
      pkill -TERM -f "$pattern" || true
    fi
  done
  sleep "$CARLA_TERM_GRACE_S"
}

run_one_config() {
  local radar_pps="$1"
  local support_mode="$2"
  local phase="$3"

  local run_collection=0
  local run_train=0
  local run_eval=0
  case "$phase" in
    collect)
      run_collection=1
      ;;
    train_eval)
      run_train="$RUN_TRAIN"
      run_eval="$RUN_EVAL"
      ;;
    *)
      die "Unknown ablation phase: $phase"
      ;;
  esac

  log "Ablation config: radar_pps=$radar_pps support_mode=$support_mode phase=$phase"
  RADAR_PPS="$radar_pps" \
  RADAR_PERSON_SUPPORT_MODE="$support_mode" \
  DATE_TAG="$DATE_TAG" \
  LOOPS_PER_DENSITY="$LOOPS_PER_DENSITY" \
  MIN_SAMPLES_PER_DENSITY="$MIN_SAMPLES_PER_DENSITY" \
  MAX_SAMPLES_PER_DENSITY="$MAX_SAMPLES_PER_DENSITY" \
  TRAIN_EPOCHS="$TRAIN_EPOCHS" \
  TRAIN_BUDGET_HOURS="$TRAIN_BUDGET_HOURS" \
  RUN_COLLECTION="$run_collection" \
  RUN_ANALYZE="$run_collection" \
  RUN_TRAIN="$run_train" \
  RUN_EVAL="$run_eval" \
  FORCE_TRAIN="$FORCE_TRAIN" \
  STOP_CARLA_BEFORE_TRAINING=0 \
  REQUIRE_CUDA_EVAL="$REQUIRE_CUDA" \
  bash scripts/run_moving_radar12k_pilot_training_pipeline.sh
}

log "SceneSense moving radar model ablation"
log "DATE_TAG=$DATE_TAG"
log "ABLATION_CONFIGS=$ABLATION_CONFIGS"
log "loops=$LOOPS_PER_DENSITY min_rows=$MIN_SAMPLES_PER_DENSITY max_rows=$MAX_SAMPLES_PER_DENSITY"
log "epochs=$TRAIN_EPOCHS budget=${TRAIN_BUDGET_HOURS}h"

if [[ "$RUN_TRAIN" == "1" || "$RUN_EVAL" == "1" ]]; then
  check_cuda
fi

if [[ "$RUN_COLLECTION" == "1" ]]; then
  log "Collection phase: keep CARLA running until all configs are collected."
  for cfg in $ABLATION_CONFIGS; do
    IFS=: read -r radar_pps support_mode <<<"$cfg"
    run_one_config "$radar_pps" "$support_mode" collect
  done
else
  log "Skipping collection phase because RUN_COLLECTION=$RUN_COLLECTION."
fi

if [[ "$RUN_TRAIN" == "1" || "$RUN_EVAL" == "1" ]]; then
  stop_carla_server
  log "Training/evaluation phase."
  for cfg in $ABLATION_CONFIGS; do
    IFS=: read -r radar_pps support_mode <<<"$cfg"
    run_one_config "$radar_pps" "$support_mode" train_eval
  done
else
  log "Skipping training/evaluation phase because RUN_TRAIN=$RUN_TRAIN RUN_EVAL=$RUN_EVAL."
fi

log "Radar model ablation complete."
