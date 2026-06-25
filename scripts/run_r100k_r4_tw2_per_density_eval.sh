#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"

EXP="${EXP:-$ROOT_DIR/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_fusion_train_20260624_r100k_r4_tw2_pilot}"
DATASET="${DATASET:-$ROOT_DIR/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
TRIAL="${TRIAL:-moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_768x432_lr1e-4_bs2}"
CKPT="${CKPT:-$EXP/checkpoints/$TRIAL/best.pt}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.03}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

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

prepare_eval_dir() {
  local eval_dir="$1"
  mkdir -p "$eval_dir"
  if [[ -e "$eval_dir/dataset" || -L "$eval_dir/dataset" ]]; then
    local existing
    local desired
    existing="$(readlink -f "$eval_dir/dataset")"
    desired="$(readlink -f "$DATASET")"
    [[ "$existing" == "$desired" ]] || die "$eval_dir/dataset points to $existing, expected $desired."
  else
    run ln -s "$DATASET" "$eval_dir/dataset"
  fi
}

eval_density() {
  local density="$1"
  local eval_dir
  local sample_args=()
  if [[ "$density" == "overall" ]]; then
    eval_dir="$EXP/eval_pilot_on_pilot_test"
  else
    eval_dir="$EXP/eval_pilot_on_${density}_test"
    sample_args=(--sample-id-contains "_${density}_")
  fi
  prepare_eval_dir "$eval_dir"
  if [[ "$FORCE_EVAL" != "1" && -f "$eval_dir/metrics/test_fusion_evaluation_metrics.json" ]]; then
    log "Skipping $density eval; metrics already exist. Set FORCE_EVAL=1 to rerun."
    return
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$CKPT" \
      --split test \
      "${sample_args[@]}" \
      --device cuda \
      --require-cuda \
      --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
      --match-distance-m "$MATCH_DISTANCE_M"
}

[[ -f "$CONFIG_PATH" ]] || die "Missing config: $CONFIG_PATH"
[[ -f "$CKPT" ]] || die "Missing checkpoint: $CKPT"
[[ -f "$DATASET/manifest.csv" ]] || die "Missing dataset manifest: $DATASET/manifest.csv"

log "Evaluating r100k/r4/tw2 checkpoint by density"
log "Experiment: $EXP"
log "Dataset: $DATASET"
log "Checkpoint: $CKPT"

eval_density overall
eval_density low
eval_density medium
eval_density crowded

log "Per-density eval complete."
