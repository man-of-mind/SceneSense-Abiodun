#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260623_salvaged}"
RADAR_PPS="${RADAR_PPS:-12000}"
SUPPORT_TAG="${SUPPORT_TAG:-bboxsupport}"
SOURCE_LOOPS_TAG="${SOURCE_LOOPS_TAG:-8loops_cap6000}"
SALVAGE_TAG="${SALVAGE_TAG:-8loops_cap6000_salvaged}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"

MIN_ROWS_LOW="${MIN_ROWS_LOW:-3500}"
MIN_ROWS_MEDIUM="${MIN_ROWS_MEDIUM:-3500}"
MIN_ROWS_CROWDED="${MIN_ROWS_CROWDED:-3500}"
MIN_LOOPS_LOW="${MIN_LOOPS_LOW:-8}"
MIN_LOOPS_MEDIUM="${MIN_LOOPS_MEDIUM:-8}"
MIN_LOOPS_CROWDED="${MIN_LOOPS_CROWDED:-6}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-40}"
TRAIN_BUDGET_HOURS="${TRAIN_BUDGET_HOURS:-8.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0002}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-1}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"
REQUIRE_CUDA_EVAL="${REQUIRE_CUDA_EVAL:-1}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.03}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"

PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

SOURCE_PREFIX="moving_ego_radarpps${RADAR_PPS}_${SUPPORT_TAG}_${SOURCE_LOOPS_TAG}"
LOW="$ROOT_DIR/fusion_training_data/${SOURCE_PREFIX}_low_stride${SAMPLE_STRIDE}"
MEDIUM="$ROOT_DIR/fusion_training_data/${SOURCE_PREFIX}_medium_stride${SAMPLE_STRIDE}"
CROWDED="$ROOT_DIR/fusion_training_data/${SOURCE_PREFIX}_crowded_stride${SAMPLE_STRIDE}"

SALVAGE_PREFIX="moving_ego_radarpps${RADAR_PPS}_${SUPPORT_TAG}_${SALVAGE_TAG}"
DATASET="$ROOT_DIR/fusion_training_data/${SALVAGE_PREFIX}_merged_stride${SAMPLE_STRIDE}"
EXP="$ROOT_DIR/experiments/${SALVAGE_PREFIX}_fusion_train_${DATE_TAG}"
TRIAL="${SALVAGE_PREFIX}_768x432_lr1e-4_bs2"
CKPT="$EXP/checkpoints/$TRIAL/best.pt"
MIN_TOTAL=$((MIN_ROWS_LOW + MIN_ROWS_MEDIUM + MIN_ROWS_CROWDED))

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
  local dataset_dir="$1"
  local manifest="$dataset_dir/manifest.csv"
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

route_summary_value() {
  local dataset_dir="$1"
  local key="$2"
  local summary="$dataset_dir/route_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo ""
    return
  fi
  python3 - "$summary" "$key" <<'PY' 2>/dev/null || true
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data.get(key)
print("" if value is None else value)
PY
}

route_summary_loops() {
  local loops
  loops="$(route_summary_value "$1" "loop_count")"
  if [[ -z "$loops" ]]; then
    echo 0
  else
    python3 - "$loops" <<'PY'
import sys
try:
    print(int(float(sys.argv[1])))
except Exception:
    print(0)
PY
  fi
}

require_dataset_minimum() {
  local label="$1"
  local dataset_dir="$2"
  local min_rows="$3"
  local min_loops="$4"
  local rows loops stop_reason
  rows="$(manifest_rows "$dataset_dir")"
  loops="$(route_summary_loops "$dataset_dir")"
  stop_reason="$(route_summary_value "$dataset_dir" "stop_reason")"
  log "$label dataset rows=$rows loops=$loops stop_reason=${stop_reason:-unknown}"
  [[ "$rows" -ge "$min_rows" ]] || die "$label dataset has $rows rows; expected at least $min_rows: $dataset_dir"
  [[ "$loops" -ge "$min_loops" ]] || die "$label dataset has $loops loops; expected at least $min_loops: $dataset_dir"
}

config_value() {
  local key="$1"
  python3 - "$CONFIG_PATH" "$key" <<'PY' | tail -n 1
import sys

config_path, key = sys.argv[1], sys.argv[2]
prefix = key + ":"
with open(config_path, "r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if line.startswith(prefix):
            print(line[len(prefix):].strip().strip("\"'"))
PY
}

require_config_file() {
  local key="$1"
  local path
  path="$(config_value "$key")"
  if [[ -n "$path" && "$path" != "null" && "$path" != "None" ]]; then
    require_file "$path"
  fi
}

preflight() {
  require_file "scripts/merge_fusion_training_datasets.py"
  require_file "scripts/validate_fusion_training_dataset.py"
  require_file "scripts/dry_run_fusion_training_targets.py"
  require_file "scripts/analyze_radar_class_aware_support.py"
  require_file "scripts/plot_fusion_training_curves.py"
  require_file "$CONFIG_PATH"
  require_config_file "init_rgb_checkpoint"
  require_config_file "baseline_rgb_checkpoint"
}

stop_carla_server() {
  if [[ "$STOP_CARLA_BEFORE_TRAINING" != "1" ]]; then
    log "Skipping CARLA shutdown because STOP_CARLA_BEFORE_TRAINING=$STOP_CARLA_BEFORE_TRAINING."
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
    log "No CARLA server process found; continuing."
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

analyze_density() {
  local dataset_dir="$1"
  local label="$2"
  if [[ "$RUN_ANALYZE" != "1" ]]; then
    return
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
    python3 scripts/analyze_radar_class_aware_support.py \
      "$dataset_dir" \
      --output-dir "analysis_outputs/radar_class_aware_support/${SALVAGE_PREFIX}_${label}" \
      --person-mode bbox
}

merge_dataset() {
  local rows
  rows="$(manifest_rows "$DATASET")"
  if [[ "$rows" -ge "$MIN_TOTAL" ]]; then
    log "Skipping merge; $DATASET already has $rows rows."
    return
  fi
  if [[ -e "$DATASET" ]]; then
    die "Merge output exists but is incomplete: $DATASET ($rows rows). Move/delete it before rerunning."
  fi
  run python3 scripts/merge_fusion_training_datasets.py "$DATASET" "$LOW" "$MEDIUM" "$CROWDED" --link-mode symlink
  rows="$(manifest_rows "$DATASET")"
  [[ "$rows" -ge "$MIN_TOTAL" ]] || die "Merged dataset has $rows rows; expected at least $MIN_TOTAL."
}

validate_dataset() {
  if [[ "$RUN_VALIDATE" != "1" ]]; then
    return
  fi
  run python3 scripts/validate_fusion_training_dataset.py "$DATASET" --max-samples 120 --write-summary
  run python3 scripts/dry_run_fusion_training_targets.py "$DATASET" \
    --object-classes vehicle,person \
    --max-samples 180 \
    --require-positive-target \
    --write-summary
}

prepare_experiment() {
  local exp_dir="$1"
  local dataset_dir="$2"
  mkdir -p "$exp_dir"
  if [[ -e "$exp_dir/dataset" || -L "$exp_dir/dataset" ]]; then
    local existing desired
    existing="$(readlink -f "$exp_dir/dataset")"
    desired="$(readlink -f "$dataset_dir")"
    [[ "$existing" == "$desired" ]] || die "$exp_dir/dataset points to $existing, expected $desired."
  else
    run ln -s "$dataset_dir" "$exp_dir/dataset"
  fi
}

train_model() {
  prepare_experiment "$EXP" "$DATASET"
  if [[ -f "$CKPT" && "$FORCE_TRAIN" != "1" ]]; then
    log "Skipping training because checkpoint already exists: $CKPT"
    return
  fi
  local trial_json
  trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":%s,"weight_decay":%s,"augment_strength":"strong","input_size":[768,432],"batch_size":%s,"epochs":%s}' \
    "$TRIAL" "$LR" "$WEIGHT_DECAY" "$BATCH_SIZE" "$TRAIN_EPOCHS")"
  run env PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$EXP" \
    --trial-json "$trial_json" \
    --training-budget-hours "$TRAIN_BUDGET_HOURS"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
    python3 scripts/plot_fusion_training_curves.py "$EXP/metrics/${TRIAL}_metrics.csv" --prefix "$TRIAL"
}

eval_one() {
  local label="$1"
  local sample_filter="$2"
  local eval_dir="$EXP/eval_${label}"
  local sample_args=()
  local require_cuda_args=()
  if [[ -n "$sample_filter" ]]; then
    sample_args=(--sample-id-contains "$sample_filter")
  fi
  if [[ "$REQUIRE_CUDA_EVAL" == "1" ]]; then
    require_cuda_args=(--require-cuda)
  fi
  prepare_experiment "$eval_dir" "$DATASET"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$CKPT" \
      --split test \
      "${sample_args[@]}" \
      --device cuda \
      "${require_cuda_args[@]}" \
      --object-score-threshold "$OBJECT_SCORE_THRESHOLD" \
      --match-distance-m "$MATCH_DISTANCE_M"
}

eval_model() {
  if [[ "$RUN_EVAL" != "1" ]]; then
    log "Skipping evaluation because RUN_EVAL=$RUN_EVAL."
    return
  fi
  [[ -f "$CKPT" ]] || die "Checkpoint missing: $CKPT"
  eval_one "salvaged_overall" ""
  eval_one "salvaged_low" "_low_"
  eval_one "salvaged_medium" "_medium_"
  eval_one "salvaged_crowded" "_crowded_"
}

log "Moving radar-12k bbox salvage pipeline"
log "Source low=$LOW"
log "Source medium=$MEDIUM"
log "Source crowded=$CROWDED"
log "Merged dataset=$DATASET"
log "Experiment=$EXP"
log "Minimum rows: low=$MIN_ROWS_LOW medium=$MIN_ROWS_MEDIUM crowded=$MIN_ROWS_CROWDED"
log "Minimum loops: low=$MIN_LOOPS_LOW medium=$MIN_LOOPS_MEDIUM crowded=$MIN_LOOPS_CROWDED"

preflight
require_dataset_minimum "low" "$LOW" "$MIN_ROWS_LOW" "$MIN_LOOPS_LOW"
require_dataset_minimum "medium" "$MEDIUM" "$MIN_ROWS_MEDIUM" "$MIN_LOOPS_MEDIUM"
require_dataset_minimum "crowded" "$CROWDED" "$MIN_ROWS_CROWDED" "$MIN_LOOPS_CROWDED"

analyze_density "$LOW" "low"
analyze_density "$MEDIUM" "medium"
analyze_density "$CROWDED" "crowded"

if [[ "$RUN_MERGE" == "1" ]]; then
  merge_dataset
else
  log "Skipping merge because RUN_MERGE=$RUN_MERGE."
fi
validate_dataset

if [[ "$RUN_TRAIN" == "1" ]]; then
  stop_carla_server
  train_model
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

eval_model

log "Salvage pipeline complete."
log "Dataset: $DATASET"
log "Experiment: $EXP"
