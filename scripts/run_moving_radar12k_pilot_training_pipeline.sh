#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260622}"
RADAR_PPS="${RADAR_PPS:-12000}"
RADAR_RASTER_RADIUS_PX="${RADAR_RASTER_RADIUS_PX:-2}"
RADAR_TEMPORAL_WINDOW_FRAMES="${RADAR_TEMPORAL_WINDOW_FRAMES:-1}"
LOOPS_PER_DENSITY="${LOOPS_PER_DENSITY:-2}"
MIN_SAMPLES_PER_DENSITY="${MIN_SAMPLES_PER_DENSITY:-1200}"
MAX_SAMPLES_PER_DENSITY="${MAX_SAMPLES_PER_DENSITY:-2200}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"
EGO_SPEED_DIFF="${EGO_SPEED_DIFF:-60}"
EGO_FOLLOW_DISTANCE_M="${EGO_FOLLOW_DISTANCE_M:-28.0}"
LOW_NPC_VEHICLES="${LOW_NPC_VEHICLES:-8}"
LOW_NPC_PEDESTRIANS="${LOW_NPC_PEDESTRIANS:-10}"
MEDIUM_NPC_VEHICLES="${MEDIUM_NPC_VEHICLES:-20}"
MEDIUM_NPC_PEDESTRIANS="${MEDIUM_NPC_PEDESTRIANS:-25}"
CROWDED_NPC_VEHICLES="${CROWDED_NPC_VEHICLES:-28}"
CROWDED_NPC_PEDESTRIANS="${CROWDED_NPC_PEDESTRIANS:-35}"
ROUTE_SPAWN_INDICES="${ROUTE_SPAWN_INDICES:-80,85,91,94,99,110,137,80}"
ROUTE_POINT_SPACING_M="${ROUTE_POINT_SPACING_M:-3.0}"
LOOP_RETURN_RADIUS_M="${LOOP_RETURN_RADIUS_M:-2.0}"
LOOP_MIN_DISTANCE_M="${LOOP_MIN_DISTANCE_M:-200.0}"
PERSON_RADIUS_M="${PERSON_RADIUS_M:-2.0}"
PERSON_Z_DOWN_M="${PERSON_Z_DOWN_M:-0.5}"
PERSON_Z_UP_M="${PERSON_Z_UP_M:-2.0}"
RADAR_PERSON_SUPPORT_MODE="${RADAR_PERSON_SUPPORT_MODE:-radius}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BUDGET_HOURS="${TRAIN_BUDGET_HOURS:-3.0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RUN_COLLECTION="${RUN_COLLECTION:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-1}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"
REQUIRE_CUDA_EVAL="${REQUIRE_CUDA_EVAL:-0}"

PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

if [[ "$RADAR_PERSON_SUPPORT_MODE" == "radius" ]]; then
  SUPPORT_TAG="${SUPPORT_TAG:-classaware}"
elif [[ "$RADAR_PERSON_SUPPORT_MODE" == "bbox" ]]; then
  SUPPORT_TAG="${SUPPORT_TAG:-bboxsupport}"
else
  die "RADAR_PERSON_SUPPORT_MODE must be radius or bbox, got: $RADAR_PERSON_SUPPORT_MODE"
fi

DATA_TAG="radarpps${RADAR_PPS}_${SUPPORT_TAG}_r${RADAR_RASTER_RADIUS_PX}_tw${RADAR_TEMPORAL_WINDOW_FRAMES}_${LOOPS_PER_DENSITY}loops_cap${MAX_SAMPLES_PER_DENSITY}"
PREFIX="moving_ego_${DATA_TAG}"
LOW="$ROOT_DIR/fusion_training_data/${PREFIX}_low_stride${SAMPLE_STRIDE}"
MEDIUM="$ROOT_DIR/fusion_training_data/${PREFIX}_medium_stride${SAMPLE_STRIDE}"
CROWDED="$ROOT_DIR/fusion_training_data/${PREFIX}_crowded_stride${SAMPLE_STRIDE}"
DATASET="$ROOT_DIR/fusion_training_data/${PREFIX}_merged_stride${SAMPLE_STRIDE}"
EXP="$ROOT_DIR/experiments/${PREFIX}_fusion_train_${DATE_TAG}"
TRIAL="${PREFIX}_768x432_lr1e-4_bs2"
CKPT="$EXP/checkpoints/$TRIAL/best.pt"
MIN_TOTAL=$((MIN_SAMPLES_PER_DENSITY * 3))

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

route_summary_loops() {
  local dataset_dir="$1"
  local summary="$dataset_dir/route_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo 0
    return
  fi
  python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("loop_count") or 0))' "$summary" 2>/dev/null || echo 0
}

require_complete_dataset() {
  local dataset_dir="$1"
  local min_rows="$2"
  local rows
  rows="$(manifest_rows "$dataset_dir")"
  [[ "$rows" -ge "$min_rows" ]] || die "Dataset $dataset_dir has $rows rows; expected at least $min_rows."
}

config_value() {
  local key="$1"
  python3 -c '
import sys
config_path, key = sys.argv[1], sys.argv[2]
prefix = key + ":"
with open(config_path, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if line.startswith(prefix):
            print(line[len(prefix):].strip().strip("\"'\''"))
' "$CONFIG_PATH" "$key" | tail -n 1
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
  require_file "carla_collect_moving_ego_fusion_training_data.py"
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

collect_density() {
  local label="$1"
  local dataset_dir="$2"
  local npc_vehicles="$3"
  local npc_pedestrians="$4"
  local seed="$5"
  local rows loops
  rows="$(manifest_rows "$dataset_dir")"
  loops="$(route_summary_loops "$dataset_dir")"
  if [[ "$rows" -ge "$MIN_SAMPLES_PER_DENSITY" && "$loops" -ge "$LOOPS_PER_DENSITY" ]]; then
    log "Skipping $label collection; $dataset_dir already has $rows rows and $loops loops."
    return
  fi
  if [[ -e "$dataset_dir" ]]; then
    die "Partial/stale dataset exists at $dataset_dir with $rows rows and $loops loops. Move/delete it before rerunning."
  fi

  set +e
  run python3 carla_collect_moving_ego_fusion_training_data.py \
    --experiment-id "$(basename "$dataset_dir")" \
    --seed "$seed" \
    --no-ego-freeze \
    --ego-autopilot-speed-difference-pct "$EGO_SPEED_DIFF" \
    --ego-follow-distance-m "$EGO_FOLLOW_DISTANCE_M" \
    --ego-ignore-lights-pct 0 \
    --ego-fixed-path-spawn-indices "$ROUTE_SPAWN_INDICES" \
    --ego-fixed-path-loop \
    --ego-fixed-path-min-spacing-m "$ROUTE_POINT_SPACING_M" \
    --ego-disable-lane-change \
    --route-progress-every-s 2.0 \
    --loop-return-radius-m "$LOOP_RETURN_RADIUS_M" \
    --loop-min-distance-m "$LOOP_MIN_DISTANCE_M" \
    --loop-min-elapsed-s 30.0 \
    --stop-after-loops "$LOOPS_PER_DENSITY" \
    --stop-on-stuck \
    --stuck-ignore-traffic-light-waits \
    --stuck-speed-threshold-mps 0.20 \
    --stuck-timeout-s 60.0 \
    --stuck-min-elapsed-s 30.0 \
    --max-samples "$MAX_SAMPLES_PER_DENSITY" \
    --sample-stride "$SAMPLE_STRIDE" \
    --warmup-ticks 30 \
    --fps 10 \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fov 120 \
    --model-input-width 768 \
    --model-input-height 432 \
    --ego-spawn-index 80 \
    --ego-spawn-forward-offset-m 0.0 \
    --ego-spawn-right-offset-m 0.0 \
    --ego-spawn-yaw-offset-deg 0.0 \
    --ego-camera-x 1.8 \
    --ego-camera-y 0.0 \
    --ego-camera-z 1.55 \
    --ego-camera-pitch -4.0 \
    --ego-camera-yaw 0.0 \
    --ego-radar-yaw 0.0 \
    --radar-hfov 120 \
    --radar-vfov 30 \
    --radar-range 120 \
    --radar-points-per-second "$RADAR_PPS" \
    --radar-raster-radius-px "$RADAR_RASTER_RADIUS_PX" \
    --radar-temporal-window-frames "$RADAR_TEMPORAL_WINDOW_FRAMES" \
    --radar-person-support-mode "$RADAR_PERSON_SUPPORT_MODE" \
    --radar-person-support-radius-m "$PERSON_RADIUS_M" \
    --radar-person-support-z-down-m "$PERSON_Z_DOWN_M" \
    --radar-person-support-z-up-m "$PERSON_Z_UP_M" \
    --npc-vehicles "$npc_vehicles" \
    --npc-pedestrians "$npc_pedestrians" \
    --npc-vehicle-speed-difference-pct 10 \
    --npc-pedestrian-max-speed-mps 0.9 \
    --npc-pedestrian-cross-factor 0.5 \
    --spawn-radius 80 \
    --gt-max-distance-m 140 \
    --include-pedestrians
  local collect_rc="$?"
  set -e

  rows="$(manifest_rows "$dataset_dir")"
  loops="$(route_summary_loops "$dataset_dir")"
  if [[ "$collect_rc" -ne 0 ]]; then
    log "Warning: $label collector exited with code $collect_rc; checking whether dataset is complete."
  fi
  require_complete_dataset "$dataset_dir" "$MIN_SAMPLES_PER_DENSITY"
  [[ "$loops" -ge "$LOOPS_PER_DENSITY" ]] || die "Dataset $dataset_dir collected $loops loops; expected $LOOPS_PER_DENSITY."
}

analyze_density() {
  local dataset_dir="$1"
  local label="$2"
  if [[ "$RUN_ANALYZE" != "1" ]]; then
    return
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" python3 scripts/analyze_radar_class_aware_support.py \
    "$dataset_dir" \
    --output-dir "analysis_outputs/radar_class_aware_support/${PREFIX}_${label}" \
    --person-mode "$RADAR_PERSON_SUPPORT_MODE" \
    --person-radius-m "$PERSON_RADIUS_M" \
    --person-z-down-m "$PERSON_Z_DOWN_M" \
    --person-z-up-m "$PERSON_Z_UP_M"
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
  require_complete_dataset "$DATASET" "$MIN_TOTAL"
}

validate_dataset() {
  run python3 scripts/validate_fusion_training_dataset.py "$DATASET" --max-samples 80
  run python3 scripts/dry_run_fusion_training_targets.py "$DATASET" --object-classes vehicle,person --max-samples 160 --require-positive-target
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

train_pilot() {
  prepare_experiment "$EXP" "$DATASET"
  if [[ -f "$CKPT" && "$FORCE_TRAIN" != "1" ]]; then
    log "Skipping training because checkpoint already exists: $CKPT"
    return
  fi
  local trial_json
  trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":0.0001,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2,"epochs":%s}' "$TRIAL" "$TRAIN_EPOCHS")"
  run env PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$EXP" \
    --trial-json "$trial_json" \
    --training-budget-hours "$TRAIN_BUDGET_HOURS"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
    python3 scripts/plot_fusion_training_curves.py "$EXP/metrics/${TRIAL}_metrics.csv" --prefix "$TRIAL"
}

eval_pilot() {
  if [[ "$RUN_EVAL" != "1" ]]; then
    log "Skipping evaluation because RUN_EVAL=$RUN_EVAL."
    return 0
  fi
  [[ -f "$CKPT" ]] || die "Pilot checkpoint missing: $CKPT"
  local eval_dir="$EXP/eval_pilot_on_pilot_test"
  prepare_experiment "$eval_dir" "$DATASET"
  local require_cuda_args=()
  if [[ "$REQUIRE_CUDA_EVAL" == "1" ]]; then
    require_cuda_args=(--require-cuda)
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" \
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
      --config "$CONFIG_PATH" \
      --experiment-dir "$eval_dir" \
      --checkpoint "$CKPT" \
      --split test \
      --object-score-threshold 0.03 \
      --match-distance-m 3.0 \
      "${require_cuda_args[@]}"
}

log "Moving radar fusion pilot pipeline"
log "Datasets: low=$LOW medium=$MEDIUM crowded=$CROWDED"
log "RADAR_PPS=$RADAR_PPS raster_radius=$RADAR_RASTER_RADIUS_PX temporal_window=$RADAR_TEMPORAL_WINDOW_FRAMES"
log "loops=$LOOPS_PER_DENSITY min_rows=$MIN_SAMPLES_PER_DENSITY max_rows=$MAX_SAMPLES_PER_DENSITY sample_stride=$SAMPLE_STRIDE"
log "Person radar support: mode=${RADAR_PERSON_SUPPORT_MODE} radius=${PERSON_RADIUS_M}m z_down=${PERSON_Z_DOWN_M}m z_up=${PERSON_Z_UP_M}m"
log "Train epochs=$TRAIN_EPOCHS budget=${TRAIN_BUDGET_HOURS}h"

preflight

if [[ "$RUN_COLLECTION" == "1" ]]; then
  collect_density "pilot low" "$LOW" "$LOW_NPC_VEHICLES" "$LOW_NPC_PEDESTRIANS" 31
  collect_density "pilot medium" "$MEDIUM" "$MEDIUM_NPC_VEHICLES" "$MEDIUM_NPC_PEDESTRIANS" 41
  collect_density "pilot crowded" "$CROWDED" "$CROWDED_NPC_VEHICLES" "$CROWDED_NPC_PEDESTRIANS" 51
  analyze_density "$LOW" "low"
  analyze_density "$MEDIUM" "medium"
  analyze_density "$CROWDED" "crowded"
  merge_dataset
  validate_dataset
else
  require_complete_dataset "$DATASET" "$MIN_TOTAL"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  stop_carla_server
  train_pilot
else
  log "Skipping training because RUN_TRAIN=$RUN_TRAIN."
fi

eval_pilot

log "Pilot pipeline complete."
log "Dataset: $DATASET"
log "Experiment: $EXP"
