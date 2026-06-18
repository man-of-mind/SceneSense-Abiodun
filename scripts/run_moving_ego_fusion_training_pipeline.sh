#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260617}"
SAMPLES_PER_DENSITY="${SAMPLES_PER_DENSITY:-4000}"
COLLECT_BY_LOOPS="${COLLECT_BY_LOOPS:-1}"
LOOPS_PER_DENSITY="${LOOPS_PER_DENSITY:-8}"
MIN_SAMPLES_PER_DENSITY="${MIN_SAMPLES_PER_DENSITY:-3500}"
MAX_SAMPLES_PER_DENSITY="${MAX_SAMPLES_PER_DENSITY:-6000}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"
EGO_SPEED_DIFF="${EGO_SPEED_DIFF:-60}"
EGO_FOLLOW_DISTANCE_M="${EGO_FOLLOW_DISTANCE_M:-28.0}"
ROUTE_SPAWN_INDICES="${ROUTE_SPAWN_INDICES:-80,85,91,94,99,80}"
ROUTE_POINT_SPACING_M="${ROUTE_POINT_SPACING_M:-3.0}"
LOOP_RETURN_RADIUS_M="${LOOP_RETURN_RADIUS_M:-2.0}"
LOOP_MIN_DISTANCE_M="${LOOP_MIN_DISTANCE_M:-200.0}"
TRAIN_STAGE1_EPOCHS="${TRAIN_STAGE1_EPOCHS:-40}"
TRAIN_STAGE2_EPOCHS="${TRAIN_STAGE2_EPOCHS:-80}"
TRAIN_STAGE1_BUDGET_HOURS="${TRAIN_STAGE1_BUDGET_HOURS:-6.0}"
TRAIN_STAGE2_BUDGET_HOURS="${TRAIN_STAGE2_BUDGET_HOURS:-6.0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CROSS_EVAL="${RUN_CROSS_EVAL:-1}"
EVAL_STRICT="${EVAL_STRICT:-0}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-1}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"

PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

PREFIX="moving_ego_tl16_spawn80_fixedroute_speed${EGO_SPEED_DIFF}"
if [[ "$COLLECT_BY_LOOPS" == "1" ]]; then
  COLLECTION_TAG="${LOOPS_PER_DENSITY}loops_cap${MAX_SAMPLES_PER_DENSITY}"
  MIN_TOTAL=$((MIN_SAMPLES_PER_DENSITY * 3))
else
  COLLECTION_TAG="${SAMPLES_PER_DENSITY}"
  MIN_TOTAL=$((SAMPLES_PER_DENSITY * 3))
fi
LOW="$ROOT_DIR/fusion_training_data/${PREFIX}_low_${COLLECTION_TAG}_stride${SAMPLE_STRIDE}"
MEDIUM="$ROOT_DIR/fusion_training_data/${PREFIX}_medium_${COLLECTION_TAG}_stride${SAMPLE_STRIDE}"
CROWDED="$ROOT_DIR/fusion_training_data/${PREFIX}_crowded_${COLLECTION_TAG}_stride${SAMPLE_STRIDE}"
DATASET="$ROOT_DIR/fusion_training_data/${PREFIX}_merged_${COLLECTION_TAG}_stride${SAMPLE_STRIDE}"
EXP="$ROOT_DIR/experiments/${PREFIX}_fusion_train_${DATE_TAG}"
TRIAL="moving_fixedroute_${COLLECTION_TAG}_768x432_lr1e-4_bs2"
CKPT="$EXP/checkpoints/$TRIAL/best.pt"

AB_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_viewA_viewB_merged_24000_stride2"
AB_EXP="$ROOT_DIR/experiments/parked_ego_tl16_viewAB_fusion_train_20260612"
AB_TRIAL="parked_viewAB_24000_768x432_lr1e-4_bs2"
AB_CKPT="$AB_EXP/checkpoints/$AB_TRIAL/best.pt"

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
  if [[ "$rows" -lt "$min_rows" ]]; then
    die "Dataset $dataset_dir has $rows rows; expected at least $min_rows."
  fi
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
    log "No CARLA server process found; continuing to training/evaluation."
    return
  fi

  log "Waiting ${CARLA_STOP_GRACE_S}s for CARLA to exit cleanly."
  sleep "$CARLA_STOP_GRACE_S"

  local still_running=0
  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      still_running=1
      log "CARLA still running for pattern $pattern; sending SIGTERM."
      pkill -TERM -f "$pattern" || true
    fi
  done

  if [[ "$still_running" -eq 1 ]]; then
    log "Waiting ${CARLA_TERM_GRACE_S}s after SIGTERM."
    sleep "$CARLA_TERM_GRACE_S"
  fi
}

collect_density() {
  local label="$1"
  local dataset_dir="$2"
  local npc_vehicles="$3"
  local npc_pedestrians="$4"
  local seed="$5"
  local rows
  local loops
  rows="$(manifest_rows "$dataset_dir")"
  loops="$(route_summary_loops "$dataset_dir")"
  if [[ "$COLLECT_BY_LOOPS" == "1" ]]; then
    if [[ "$rows" -ge "$MIN_SAMPLES_PER_DENSITY" && "$loops" -ge "$LOOPS_PER_DENSITY" ]]; then
      log "Skipping $label collection; $dataset_dir already has $rows rows and $loops loops."
      return
    fi
  else
    if [[ "$rows" -ge "$SAMPLES_PER_DENSITY" ]]; then
      log "Skipping $label collection; $dataset_dir already has $rows rows."
      return
    fi
  fi
  if [[ -e "$dataset_dir" ]]; then
    die "Partial or stale dataset exists at $dataset_dir with $rows rows and $loops loops. Move/delete it before rerunning."
  fi

  local stop_after_loops
  local max_samples
  local min_required_rows
  if [[ "$COLLECT_BY_LOOPS" == "1" ]]; then
    stop_after_loops="$LOOPS_PER_DENSITY"
    max_samples="$MAX_SAMPLES_PER_DENSITY"
    min_required_rows="$MIN_SAMPLES_PER_DENSITY"
  else
    stop_after_loops=0
    max_samples="$SAMPLES_PER_DENSITY"
    min_required_rows="$SAMPLES_PER_DENSITY"
  fi

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
    --stop-after-loops "$stop_after_loops" \
    --stop-on-stuck \
    --stuck-ignore-traffic-light-waits \
    --stuck-speed-threshold-mps 0.20 \
    --stuck-timeout-s 60.0 \
    --stuck-min-elapsed-s 30.0 \
    --max-samples "$max_samples" \
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
    --radar-points-per-second 5000 \
    --radar-raster-radius-px 2 \
    --npc-vehicles "$npc_vehicles" \
    --npc-pedestrians "$npc_pedestrians" \
    --npc-vehicle-speed-difference-pct 10 \
    --npc-pedestrian-max-speed-mps 0.9 \
    --npc-pedestrian-cross-factor 0.5 \
    --spawn-radius 80 \
    --gt-max-distance-m 140 \
    --include-pedestrians

  rows="$(manifest_rows "$dataset_dir")"
  loops="$(route_summary_loops "$dataset_dir")"
  require_complete_dataset "$dataset_dir" "$min_required_rows"
  if [[ "$COLLECT_BY_LOOPS" == "1" && "$loops" -lt "$LOOPS_PER_DENSITY" ]]; then
    die "Dataset $dataset_dir collected $loops loops; expected at least $LOOPS_PER_DENSITY. Rows=$rows, max_samples=$max_samples."
  fi
}

merge_dataset() {
  local output_dir="$1"
  local expected_rows="$2"
  shift 2
  local rows
  rows="$(manifest_rows "$output_dir")"
  if [[ "$rows" -ge "$expected_rows" ]]; then
    log "Skipping merge; $output_dir already has $rows rows."
    return
  fi
  if [[ -e "$output_dir" ]]; then
    die "Merge output exists but is incomplete: $output_dir ($rows rows). Move/delete it before rerunning."
  fi
  run python3 scripts/merge_fusion_training_datasets.py "$output_dir" "$@" --link-mode symlink
  require_complete_dataset "$output_dir" "$expected_rows"
}

validate_dataset() {
  local dataset_dir="$1"
  run python3 scripts/validate_fusion_training_dataset.py "$dataset_dir" --max-samples 80
  run python3 scripts/dry_run_fusion_training_targets.py "$dataset_dir" --object-classes vehicle,person --max-samples 160 --require-positive-target
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
    run ln -s "$dataset_dir" "$exp_dir/dataset"
  fi
}

max_metric_epoch() {
  local csv_path="$1"
  if [[ ! -f "$csv_path" ]]; then
    echo -1
    return
  fi
  awk -F, '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "epoch") epoch_col = i
      }
      next
    }
    epoch_col {
      value = $epoch_col + 0
      if (value > max_epoch) max_epoch = value
      seen = 1
    }
    END {
      if (seen) print max_epoch
      else print -1
    }
  ' "$csv_path"
}

train_stage() {
  local exp_dir="$1"
  local dataset_dir="$2"
  local trial="$3"
  local target_epochs="$4"
  local budget_hours="$5"
  local resume_lr="${6:-}"
  local metrics_csv="$exp_dir/metrics/${trial}_metrics.csv"
  local target_last_epoch=$((target_epochs - 1))
  local current_epoch
  current_epoch="$(max_metric_epoch "$metrics_csv")"
  if [[ "$current_epoch" -ge "$target_last_epoch" ]]; then
    log "Skipping training stage for $trial; metrics already reached epoch $current_epoch."
    return
  fi

  prepare_experiment "$exp_dir" "$dataset_dir"

  local trial_json
  if [[ -n "$resume_lr" ]]; then
    trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":0.0001,"resume_lr":%s,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2,"epochs":%s}' "$trial" "$resume_lr" "$target_epochs")"
  else
    trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":0.0001,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2,"epochs":%s}' "$trial" "$target_epochs")"
  fi

  run env PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$exp_dir" \
    --trial-json "$trial_json" \
    --training-budget-hours "$budget_hours"
}

train_two_stage() {
  local exp_dir="$1"
  local dataset_dir="$2"
  local trial="$3"
  train_stage "$exp_dir" "$dataset_dir" "$trial" "$TRAIN_STAGE1_EPOCHS" "$TRAIN_STAGE1_BUDGET_HOURS"
  train_stage "$exp_dir" "$dataset_dir" "$trial" "$TRAIN_STAGE2_EPOCHS" "$TRAIN_STAGE2_BUDGET_HOURS" "0.00005"
}

plot_curves() {
  local exp_dir="$1"
  local trial="$2"
  local metrics_csv="$exp_dir/metrics/${trial}_metrics.csv"
  if [[ ! -f "$metrics_csv" ]]; then
    log "Skipping curves; missing metrics CSV: $metrics_csv"
    return
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
    python3 scripts/plot_fusion_training_curves.py "$metrics_csv" --prefix "$trial"
}

eval_checkpoint_on_dataset() {
  local label="$1"
  local checkpoint="$2"
  local dataset_dir="$3"
  local eval_dir="$4"
  if [[ "$RUN_EVAL" != "1" ]]; then
    log "Skipping eval $label because RUN_EVAL=$RUN_EVAL."
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    log "Skipping eval $label; checkpoint not found: $checkpoint"
    return
  fi
  if [[ ! -f "$dataset_dir/manifest.csv" ]]; then
    log "Skipping eval $label; dataset not found: $dataset_dir"
    return
  fi
  prepare_experiment "$eval_dir" "$dataset_dir"
  log "Evaluating $label"
  if env MPLCONFIGDIR="$MPLCONFIGDIR" QT_QPA_PLATFORM="$QT_QPA_PLATFORM" PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$eval_dir" \
    --checkpoint "$checkpoint" \
    --split test \
    --object-score-threshold 0.03 \
    --match-distance-m 3.0; then
    log "Evaluation completed: $label"
  else
    local rc="$?"
    log "Warning: evaluation failed for '$label' with exit code $rc."
    if [[ "$EVAL_STRICT" == "1" ]]; then
      die "Stopping because EVAL_STRICT=1."
    fi
  fi
}

log "SceneSense moving-ego RGB+radar fusion training pipeline"
log "Route spawn indices: $ROUTE_SPAWN_INDICES"
log "Route point spacing: ${ROUTE_POINT_SPACING_M}m"
log "Ego speed difference: $EGO_SPEED_DIFF%; follow distance: ${EGO_FOLLOW_DISTANCE_M}m"
log "Loop return radius: ${LOOP_RETURN_RADIUS_M}m"
log "Loop min distance: ${LOOP_MIN_DISTANCE_M}m"
if [[ "$COLLECT_BY_LOOPS" == "1" ]]; then
  log "Collection mode: loops; loops per density: $LOOPS_PER_DENSITY; max samples cap: $MAX_SAMPLES_PER_DENSITY; min samples: $MIN_SAMPLES_PER_DENSITY"
else
  log "Collection mode: samples; samples per density: $SAMPLES_PER_DENSITY"
fi

collect_density "moving low" "$LOW" 8 10 31
collect_density "moving medium" "$MEDIUM" 20 25 41
collect_density "moving crowded" "$CROWDED" 35 45 51

merge_dataset "$DATASET" "$MIN_TOTAL" "$LOW" "$MEDIUM" "$CROWDED"
validate_dataset "$DATASET"

stop_carla_server

train_two_stage "$EXP" "$DATASET" "$TRIAL"
plot_curves "$EXP" "$TRIAL"

eval_checkpoint_on_dataset "moving model on moving test" "$CKPT" "$DATASET" "$EXP/eval_moving_model_on_moving"

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "parked A+B model on moving test" "$AB_CKPT" "$DATASET" "$EXP/eval_parked_AB_model_on_moving"
  eval_checkpoint_on_dataset "moving model on parked A+B test" "$CKPT" "$AB_DATASET" "$EXP/eval_moving_model_on_parked_AB"
fi

log "Pipeline complete."
log "Moving dataset: $DATASET"
log "Moving experiment: $EXP"
