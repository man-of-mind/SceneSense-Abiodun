#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260622}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"

DATASET="${DATASET:-$ROOT_DIR/fusion_training_data/moving_ego_tl16_spawn80_fixedroute_speed60_merged_8loops_cap6000_stride2}"
BASE_EXP="${BASE_EXP:-$ROOT_DIR/experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617}"
BASE_TRIAL="${BASE_TRIAL:-moving_fixedroute_8loops_cap6000_768x432_lr1e-4_bs2}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/checkpoints/$BASE_TRIAL/best.pt}"
EXP="${EXP:-$ROOT_DIR/experiments/moving_ego_tl16_spawn80_fixedroute_speed60_seg_tuning_${DATE_TAG}}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
TRAIN_BUDGET_HOURS="${TRAIN_BUDGET_HOURS:-3.0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.00005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.03}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-0}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run() {
  log "RUN: $*"
  "$@"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file is missing: $path"
}

manifest_rows() {
  local manifest="$1/manifest.csv"
  if [[ ! -f "$manifest" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "$manifest")"
  if [[ "$lines" -le 0 ]]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

metrics_last_epoch() {
  local metrics_csv="$1"
  if [[ ! -f "$metrics_csv" ]]; then
    echo -1
    return
  fi
  python3 - "$metrics_csv" <<'PY'
import csv
import sys

path = sys.argv[1]
last = -1
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            last = max(last, int(row.get("epoch", -1)))
        except (TypeError, ValueError):
            pass
print(last)
PY
}

stop_carla_server() {
  if [[ "$STOP_CARLA_BEFORE_TRAINING" != "1" ]]; then
    log "Skipping CARLA shutdown because STOP_CARLA_BEFORE_TRAINING=$STOP_CARLA_BEFORE_TRAINING."
    return
  fi

  local patterns=(
    "CarlaUnreal.sh"
    "CarlaUnreal-Linux-Shipping"
    "CarlaUE4-Linux-Shipping"
    "CarlaUE4"
  )
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

prepare_experiment() {
  local exp_dir="$1"
  local dataset_dir="$2"
  mkdir -p "$exp_dir"
  if [[ -e "$exp_dir/dataset" || -L "$exp_dir/dataset" ]]; then
    local existing
    local desired
    existing="$(readlink -f "$exp_dir/dataset")"
    desired="$(readlink -f "$dataset_dir")"
    [[ "$existing" == "$desired" ]] || die "$exp_dir/dataset points to $existing, expected $desired."
    log "Experiment dataset link already exists: $exp_dir/dataset"
  else
    run ln -s "$dataset_dir" "$exp_dir/dataset"
  fi
}

train_trial() {
  local name="$1"
  local class_weights="$2"
  local object_total="$3"
  local selection_mode="$4"
  local trial_dir="$EXP/checkpoints/$name"
  local metrics_csv="$EXP/metrics/${name}_metrics.csv"
  local target_last_epoch=$((EPOCHS - 1))
  local last_epoch
  last_epoch="$(metrics_last_epoch "$metrics_csv")"
  if [[ -f "$trial_dir/best.pt" && "$last_epoch" -ge "$target_last_epoch" ]]; then
    log "Skipping training for $name; checkpoint exists and metrics reached epoch $last_epoch."
    return
  fi
  if [[ -f "$trial_dir/best.pt" ]]; then
    log "Continuing training for $name; checkpoint exists but metrics only reached epoch $last_epoch of $target_last_epoch."
  fi

  local trial_json
  trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":%s,"weight_decay":%s,"augment_strength":"strong","input_size":[768,432],"batch_size":%s,"epochs":%s,"init_rgb_checkpoint":"%s","selection_score_mode":"%s","class_loss_weights":%s,"loss_weights":{"object_total":%s}}' \
    "$name" "$LR" "$WEIGHT_DECAY" "$BATCH_SIZE" "$EPOCHS" "$BASE_CKPT" "$selection_mode" "$class_weights" "$object_total")"

  run env PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$EXP" \
    --trial-json "$trial_json" \
    --training-budget-hours "$TRAIN_BUDGET_HOURS"

  if [[ -f "$metrics_csv" ]]; then
    run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
      python3 scripts/plot_fusion_training_curves.py "$metrics_csv" --prefix "$name"
  fi
}

eval_trial() {
  local name="$1"
  local ckpt="$EXP/checkpoints/$name/best.pt"
  [[ -f "$ckpt" ]] || die "Missing checkpoint for eval: $ckpt"

  local eval_dir="$EXP/eval_${name}_overall"
  prepare_experiment "$eval_dir" "$DATASET"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$ckpt" \
      --split test \
      --device cuda \
      --require-cuda \
      --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
      --match-distance-m "$MATCH_DISTANCE_M"

  local density
  for density in low medium crowded; do
    eval_dir="$EXP/eval_${name}_${density}"
    prepare_experiment "$eval_dir" "$DATASET"
    run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
      python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
        --config "$CONFIG_PATH" \
        --experiment-dir "$eval_dir" \
        --checkpoint "$ckpt" \
        --split test \
        --sample-id-contains "_${density}_" \
        --device cuda \
        --require-cuda \
        --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
        --match-distance-m "$MATCH_DISTANCE_M"
  done
}

log "SceneSense moving-fusion segmentation tuning"
log "Dataset: $DATASET ($(manifest_rows "$DATASET") rows)"
log "Base checkpoint: $BASE_CKPT"
log "Experiment: $EXP"

require_file "$CONFIG_PATH"
require_file "$BASE_CKPT"
require_file "$DATASET/manifest.csv"
prepare_experiment "$EXP" "$DATASET"

if [[ "$RUN_TRAIN" == "1" ]]; then
  stop_carla_server
  # Current baseline weights are [0.5, 1.0, 4.0] and object_total=1.0.
  # These trials ask whether vehicle segmentation improves when checkpoint
  # selection and the loss are aligned with the vehicle-IoU target.
  train_trial "vehicle_weighted_obj025_vehicle_iou" "[0.35,2.50,3.00]" "0.25" "vehicle_iou"
  train_trial "vehicle_miou_obj025_vehicle_miou" "[0.35,2.00,3.00]" "0.25" "vehicle_miou"
  train_trial "vehicle_weighted_obj010_vehicle_iou" "[0.35,2.50,2.50]" "0.10" "vehicle_iou"
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_trial "vehicle_weighted_obj025_vehicle_iou"
  eval_trial "vehicle_miou_obj025_vehicle_miou"
  eval_trial "vehicle_weighted_obj010_vehicle_iou"
else
  log "Skipping evaluation because RUN_EVAL=$RUN_EVAL."
fi

log "Tuning pipeline complete."
