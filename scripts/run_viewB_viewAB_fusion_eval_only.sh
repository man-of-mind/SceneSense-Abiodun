#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260612}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
EVAL_STRICT="${EVAL_STRICT:-0}"
RUN_CROSS_EVAL="${RUN_CROSS_EVAL:-1}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.03}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"

PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"

A_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2"
A_EXP="$ROOT_DIR/experiments/parked_ego_tl16_right7_fusion_train_20260612"
A_TRIAL="parked_right7_lowmedcrowd_768x432_lr1e-4_bs2"
A_CKPT="$A_EXP/checkpoints/$A_TRIAL/best.pt"

B_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_spawn80_right8_fwd16_merged_12000_stride2"
B_EXP="$ROOT_DIR/experiments/parked_ego_tl16_viewB_fusion_train_${DATE_TAG}"
B_TRIAL="parked_viewB_12000_768x432_lr1e-4_bs2"
B_CKPT="$B_EXP/checkpoints/$B_TRIAL/best.pt"

AB_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_viewA_viewB_merged_24000_stride2"
AB_EXP="$ROOT_DIR/experiments/parked_ego_tl16_viewAB_fusion_train_${DATE_TAG}"
AB_TRIAL="parked_viewAB_24000_768x432_lr1e-4_bs2"
AB_CKPT="$AB_EXP/checkpoints/$AB_TRIAL/best.pt"

SUMMARY_CSV="$ROOT_DIR/analysis_outputs/parked_ego_fusion_viewB_viewAB_eval_summary_${DATE_TAG}.csv"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || die "Missing required path: $path"
}

prepare_experiment() {
  local exp_dir="$1"
  local dataset_dir="$2"
  mkdir -p "$exp_dir"
  if [[ -e "$exp_dir/dataset" || -L "$exp_dir/dataset" ]]; then
    local existing
    existing="$(readlink -f "$exp_dir/dataset")"
    local desired
    desired="$(readlink -f "$dataset_dir")"
    if [[ "$existing" != "$desired" ]]; then
      die "$exp_dir/dataset points to $existing, expected $desired."
    fi
    log "Experiment dataset link already exists: $exp_dir/dataset"
  else
    log "RUN: ln -s $dataset_dir $exp_dir/dataset"
    ln -s "$dataset_dir" "$exp_dir/dataset"
  fi
}

eval_checkpoint_on_dataset() {
  local label="$1"
  local checkpoint="$2"
  local dataset_dir="$3"
  local eval_dir="$4"

  require_path "$checkpoint"
  require_path "$dataset_dir/manifest.csv"
  require_path "$dataset_dir/object_boxes.csv"
  prepare_experiment "$eval_dir" "$dataset_dir"

  log "Evaluating $label"
  if env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$checkpoint" \
      --split test \
      --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
      --match-distance-m "$MATCH_DISTANCE_M"; then
    log "Evaluation completed: $label"
  else
    local rc="$?"
    log "Warning: evaluation failed for '$label' with exit code $rc."
    if [[ "$EVAL_STRICT" == "1" ]]; then
      die "Stopping because EVAL_STRICT=1."
    fi
  fi
}

summarize_evals() {
  mkdir -p "$(dirname "$SUMMARY_CSV")"
  python3 - "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
rows = [
    ("A_on_A", "experiments/parked_ego_tl16_right7_fusion_train_20260612/eval_A_model_on_viewA/metrics/test_fusion_evaluation_metrics.json"),
    ("A_on_B", "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_A_model_on_viewB/metrics/test_fusion_evaluation_metrics.json"),
    ("B_on_A", "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_B_model_on_viewA/metrics/test_fusion_evaluation_metrics.json"),
    ("B_on_B", "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_B_model_on_viewB/metrics/test_fusion_evaluation_metrics.json"),
    ("AB_on_A", "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewA/metrics/test_fusion_evaluation_metrics.json"),
    ("AB_on_B", "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewB/metrics/test_fusion_evaluation_metrics.json"),
    ("AB_on_AB", "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewAB/metrics/test_fusion_evaluation_metrics.json"),
]
fields = [
    "eval",
    "metrics_path",
    "samples",
    "miou",
    "vehicle_iou",
    "person_iou",
    "pixel_accuracy",
    "learned_object_precision",
    "learned_object_recall",
    "learned_object_f1",
    "learned_global_xy_mae_m",
    "learned_vehicle_global_xy_mae_m",
    "learned_person_global_xy_mae_m",
    "learned_vehicle_object_precision",
    "learned_vehicle_object_recall",
    "learned_person_object_precision",
    "learned_person_object_recall",
]
with summary_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for label, rel_path in rows:
        path = Path(rel_path)
        row = {"eval": label, "metrics_path": str(path)}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in fields[2:]:
                row[key] = data.get(key, "")
        writer.writerow(row)
print(summary_path)
PY
}

log "SceneSense View-B / View-A+B eval-only pipeline"
log "Root: $ROOT_DIR"
log "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
log "Object score threshold: $OBJECT_SCORE_THRESHOLD; match distance: ${MATCH_DISTANCE_M}m"

require_path "$A_CKPT"
require_path "$B_CKPT"
require_path "$AB_CKPT"
require_path "$A_DATASET/manifest.csv"
require_path "$B_DATASET/manifest.csv"
require_path "$AB_DATASET/manifest.csv"

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "A model on View A test" "$A_CKPT" "$A_DATASET" "$A_EXP/eval_A_model_on_viewA"
  eval_checkpoint_on_dataset "A model on View B test" "$A_CKPT" "$B_DATASET" "$B_EXP/eval_A_model_on_viewB"
  eval_checkpoint_on_dataset "B model on View A test" "$B_CKPT" "$A_DATASET" "$B_EXP/eval_B_model_on_viewA"
fi

eval_checkpoint_on_dataset "B model on View B test" "$B_CKPT" "$B_DATASET" "$B_EXP/eval_B_model_on_viewB"

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "A+B model on View A test" "$AB_CKPT" "$A_DATASET" "$AB_EXP/eval_AB_model_on_viewA"
  eval_checkpoint_on_dataset "A+B model on View B test" "$AB_CKPT" "$B_DATASET" "$AB_EXP/eval_AB_model_on_viewB"
fi

eval_checkpoint_on_dataset "A+B model on combined View A+B test" "$AB_CKPT" "$AB_DATASET" "$AB_EXP/eval_AB_model_on_viewAB"
summarize_evals

log "Eval-only pipeline complete."
log "Summary CSV: $SUMMARY_CSV"
