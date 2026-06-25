#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260624_segonly}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"

DATASET="${DATASET:-$ROOT_DIR/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
BASE_EXP="${BASE_EXP:-$ROOT_DIR/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_fusion_train_20260624_r100k_r4_tw2_pilot}"
BASE_TRIAL="${BASE_TRIAL:-moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_768x432_lr1e-4_bs2}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/checkpoints/$BASE_TRIAL/best.pt}"
EXP="${EXP:-$ROOT_DIR/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_segonly_ablation_${DATE_TAG}}"
TRIAL_NAME="${TRIAL_NAME:-segonly_stronggeo_bnfreeze_bs${BATCH_SIZE:-8}_miou}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
TRAIN_BUDGET_HOURS="${TRAIN_BUDGET_HOURS:-3.0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-10}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
LR="${LR:-0.00005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-[0.5,1.0,4.0]}"
LOVASZ_WEIGHT="${LOVASZ_WEIGHT:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
POLY_POWER="${POLY_POWER:-0.9}"
SELECTION_SCORE_MODE="${SELECTION_SCORE_MODE:-miou}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
GEOMETRIC_AUGMENT="${GEOMETRIC_AUGMENT:-true}"
FREEZE_BN="${FREEZE_BN:-true}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.03}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"

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

metrics_last_epoch() {
  local metrics_csv="$1"
  if [[ ! -f "$metrics_csv" ]]; then
    echo -1
    return
  fi
  python3 - "$metrics_csv" <<'PY'
import csv
import sys

last = -1
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            last = max(last, int(row.get("epoch", -1)))
        except (TypeError, ValueError):
            pass
print(last)
PY
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
  else
    run ln -s "$dataset_dir" "$exp_dir/dataset"
  fi
}

train_trial() {
  local metrics_csv="$EXP/metrics/${TRIAL_NAME}_metrics.csv"
  local target_last_epoch=$((EPOCHS - 1))
  local last_epoch
  last_epoch="$(metrics_last_epoch "$metrics_csv")"
  if [[ -f "$EXP/checkpoints/$TRIAL_NAME/best.pt" && "$last_epoch" -ge "$target_last_epoch" ]]; then
    log "Skipping training; checkpoint exists and metrics reached epoch $last_epoch."
    return
  fi

  local trial_json
  trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":%s,"weight_decay":%s,"augment_strength":"strong","geometric_augment":%s,"freeze_bn":%s,"input_size":[768,432],"batch_size":%s,"num_workers":%s,"prefetch_factor":%s,"persistent_workers":%s,"epochs":%s,"init_rgb_checkpoint":"%s","selection_score_mode":"%s","class_loss_weights":%s,"lovasz_weight":%s,"lr_scheduler":"%s","lr_warmup_epochs":%s,"min_lr_ratio":%s,"poly_power":%s,"early_stop_patience":%s,"loss_weights":{"object_total":0.0}}' \
    "$TRIAL_NAME" "$LR" "$WEIGHT_DECAY" "$GEOMETRIC_AUGMENT" "$FREEZE_BN" "$BATCH_SIZE" "$NUM_WORKERS" "$PREFETCH_FACTOR" "$PERSISTENT_WORKERS" "$EPOCHS" "$BASE_CKPT" "$SELECTION_SCORE_MODE" "$CLASS_WEIGHTS" "$LOVASZ_WEIGHT" "$LR_SCHEDULER" "$LR_WARMUP_EPOCHS" "$MIN_LR_RATIO" "$POLY_POWER" "$EARLY_STOP_PATIENCE")"

  run env PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$EXP" \
    --trial-json "$trial_json" \
    --training-budget-hours "$TRAIN_BUDGET_HOURS"

  if [[ -f "$metrics_csv" ]]; then
    run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
      python3 scripts/plot_fusion_training_curves.py "$metrics_csv" --prefix "$TRIAL_NAME"
  fi
}

eval_density() {
  local density="$1"
  local eval_dir="$EXP/eval_${TRIAL_NAME}_${density}"
  local ckpt="$EXP/checkpoints/$TRIAL_NAME/best.pt"
  local sample_args=()
  [[ -f "$ckpt" ]] || die "Missing checkpoint: $ckpt"
  if [[ "$density" != "overall" ]]; then
    sample_args=(--sample-id-contains "_${density}_")
  fi
  prepare_experiment "$eval_dir" "$DATASET"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$ckpt" \
      --split test \
      "${sample_args[@]}" \
      --device cuda \
      --require-cuda \
      --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
      --match-distance-m "$MATCH_DISTANCE_M"
}

[[ -f "$CONFIG_PATH" ]] || die "Missing config: $CONFIG_PATH"
[[ -f "$BASE_CKPT" ]] || die "Missing base checkpoint: $BASE_CKPT"
[[ -f "$DATASET/manifest.csv" ]] || die "Missing dataset manifest: $DATASET/manifest.csv"
prepare_experiment "$EXP" "$DATASET"

log "SceneSense r100k/r4/tw2 segmentation-only ablation"
log "Dataset: $DATASET"
log "Base checkpoint: $BASE_CKPT"
log "Experiment: $EXP"
log "Trial: $TRIAL_NAME"
log "Settings: batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS prefetch_factor=$PREFETCH_FACTOR object_total=0 geometric_augment=$GEOMETRIC_AUGMENT freeze_bn=$FREEZE_BN lovasz_weight=$LOVASZ_WEIGHT lr_scheduler=$LR_SCHEDULER warmup=$LR_WARMUP_EPOCHS selection=$SELECTION_SCORE_MODE patience=$EARLY_STOP_PATIENCE"

if [[ "$RUN_TRAIN" == "1" ]]; then
  train_trial
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_density overall
  eval_density low
  eval_density medium
  eval_density crowded
else
  log "Skipping evaluation because RUN_EVAL=$RUN_EVAL."
fi

log "Segmentation-only ablation complete."
