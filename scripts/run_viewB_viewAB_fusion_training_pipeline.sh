#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-20260612}"
SAMPLES_PER_DENSITY="${SAMPLES_PER_DENSITY:-4000}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"
TRAIN_STAGE1_EPOCHS="${TRAIN_STAGE1_EPOCHS:-40}"
TRAIN_STAGE2_EPOCHS="${TRAIN_STAGE2_EPOCHS:-80}"
TRAIN_STAGE1_BUDGET_HOURS="${TRAIN_STAGE1_BUDGET_HOURS:-6.0}"
TRAIN_STAGE2_BUDGET_HOURS="${TRAIN_STAGE2_BUDGET_HOURS:-6.0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CROSS_EVAL="${RUN_CROSS_EVAL:-1}"
STOP_CARLA_BEFORE_TRAINING="${STOP_CARLA_BEFORE_TRAINING:-1}"
CARLA_STOP_GRACE_S="${CARLA_STOP_GRACE_S:-15}"
CARLA_TERM_GRACE_S="${CARLA_TERM_GRACE_S:-8}"

PYTHONPATH_BASE="$ROOT_DIR/pole_lraspp_multimodal_fusion:$ROOT_DIR"
CONFIG_PATH="$ROOT_DIR/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

A_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2"
A_EXP="$ROOT_DIR/experiments/parked_ego_tl16_right7_fusion_train_20260612"
A_TRIAL="parked_right7_lowmedcrowd_768x432_lr1e-4_bs2"
A_CKPT="$A_EXP/checkpoints/$A_TRIAL/best.pt"

B_PREFIX="parked_ego_tl16_spawn80_right8_fwd16"
B_LOW="$ROOT_DIR/fusion_training_data/${B_PREFIX}_low_${SAMPLES_PER_DENSITY}_stride${SAMPLE_STRIDE}"
B_MEDIUM="$ROOT_DIR/fusion_training_data/${B_PREFIX}_medium_${SAMPLES_PER_DENSITY}_stride${SAMPLE_STRIDE}"
B_CROWDED="$ROOT_DIR/fusion_training_data/${B_PREFIX}_crowded_${SAMPLES_PER_DENSITY}_stride${SAMPLE_STRIDE}"
B_TOTAL=$((SAMPLES_PER_DENSITY * 3))
B_DATASET="$ROOT_DIR/fusion_training_data/${B_PREFIX}_merged_${B_TOTAL}_stride${SAMPLE_STRIDE}"
B_EXP="$ROOT_DIR/experiments/parked_ego_tl16_viewB_fusion_train_${DATE_TAG}"
B_TRIAL="parked_viewB_${B_TOTAL}_768x432_lr1e-4_bs2"
B_CKPT="$B_EXP/checkpoints/$B_TRIAL/best.pt"

AB_TOTAL=$((12000 + B_TOTAL))
AB_DATASET="$ROOT_DIR/fusion_training_data/parked_ego_tl16_viewA_viewB_merged_${AB_TOTAL}_stride${SAMPLE_STRIDE}"
AB_EXP="$ROOT_DIR/experiments/parked_ego_tl16_viewAB_fusion_train_${DATE_TAG}"
AB_TRIAL="parked_viewAB_${AB_TOTAL}_768x432_lr1e-4_bs2"
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

  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      log "Warning: CARLA process still appears to be running for pattern: $pattern"
      return
    fi
  done
  log "CARLA server is stopped."
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

require_complete_dataset() {
  local dataset_dir="$1"
  local min_rows="$2"
  local rows
  rows="$(manifest_rows "$dataset_dir")"
  if [[ "$rows" -lt "$min_rows" ]]; then
    die "Dataset $dataset_dir has $rows rows; expected at least $min_rows."
  fi
}

collect_density() {
  local label="$1"
  local dataset_dir="$2"
  local npc_vehicles="$3"
  local npc_pedestrians="$4"
  local seed="$5"
  local rows
  rows="$(manifest_rows "$dataset_dir")"
  if [[ "$rows" -ge "$SAMPLES_PER_DENSITY" ]]; then
    log "Skipping $label collection; $dataset_dir already has $rows rows."
    return
  fi
  if [[ -e "$dataset_dir" ]]; then
    die "Partial or stale dataset exists at $dataset_dir with $rows rows. Move/delete it before rerunning."
  fi

  run python3 carla_collect_parked_ego_fusion_training_data.py \
    --experiment-id "$(basename "$dataset_dir")" \
    --max-samples "$SAMPLES_PER_DENSITY" \
    --sample-stride "$SAMPLE_STRIDE" \
    --fps 10 \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fov 120 \
    --model-input-width 768 \
    --model-input-height 432 \
    --ego-spawn-index 80 \
    --ego-spawn-forward-offset-m 16.0 \
    --ego-spawn-right-offset-m 8.0 \
    --ego-spawn-yaw-offset-deg -28.414 \
    --ego-camera-x 1.8 \
    --ego-camera-y 0.0 \
    --ego-camera-z 1.55 \
    --ego-camera-pitch -4.0 \
    --ego-camera-yaw 0.0 \
    --ego-radar-yaw 0.0 \
    --radar-hfov 120 \
    --radar-vfov 30 \
    --radar-range 120 \
    --npc-vehicles "$npc_vehicles" \
    --npc-pedestrians "$npc_pedestrians" \
    --spawn-radius 95 \
    --seed "$seed" \
    --include-pedestrians

  require_complete_dataset "$dataset_dir" "$SAMPLES_PER_DENSITY"
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
  run python3 scripts/validate_fusion_training_dataset.py "$dataset_dir" --max-samples 50
  run python3 scripts/dry_run_fusion_training_targets.py "$dataset_dir" --object-classes vehicle,person --max-samples 50
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
    die "Cannot plot; missing metrics CSV: $metrics_csv"
  fi
  run env MPLCONFIGDIR="$MPLCONFIGDIR" python3 scripts/plot_fusion_training_curves.py "$metrics_csv" --prefix "$trial"
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
  prepare_experiment "$eval_dir" "$dataset_dir"
  log "Evaluating $label"
  run env MPLCONFIGDIR="$MPLCONFIGDIR" PYTHONPATH="$PYTHONPATH_BASE" python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
    --config "$CONFIG_PATH" \
    --experiment-dir "$eval_dir" \
    --checkpoint "$checkpoint" \
    --split test \
    --object-score-threshold 0.03 \
    --match-distance-m 3.0
}

log "SceneSense View-B / View-A+B fusion training pipeline"
log "Root: $ROOT_DIR"
log "Samples per density: $SAMPLES_PER_DENSITY (View B total: $B_TOTAL; A+B total: $AB_TOTAL)"
log "If you intended 6000 per density, rerun with SAMPLES_PER_DENSITY=6000; A+B will then be 30000 rows, not 24000."

require_complete_dataset "$A_DATASET" 12000

collect_density "View B low" "$B_LOW" 5 10 61
collect_density "View B medium" "$B_MEDIUM" 20 25 71
collect_density "View B crowded" "$B_CROWDED" 35 45 81

merge_dataset "$B_DATASET" "$B_TOTAL" "$B_LOW" "$B_MEDIUM" "$B_CROWDED"
validate_dataset "$B_DATASET"

stop_carla_server

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "A model on View B test" "$A_CKPT" "$B_DATASET" "$B_EXP/eval_A_model_on_viewB"
fi

train_two_stage "$B_EXP" "$B_DATASET" "$B_TRIAL"
plot_curves "$B_EXP" "$B_TRIAL"

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "B model on View A test" "$B_CKPT" "$A_DATASET" "$B_EXP/eval_B_model_on_viewA"
fi
eval_checkpoint_on_dataset "B model on View B test" "$B_CKPT" "$B_DATASET" "$B_EXP/eval_B_model_on_viewB"

merge_dataset "$AB_DATASET" "$AB_TOTAL" "$A_DATASET" "$B_DATASET"
validate_dataset "$AB_DATASET"

train_two_stage "$AB_EXP" "$AB_DATASET" "$AB_TRIAL"
plot_curves "$AB_EXP" "$AB_TRIAL"

if [[ "$RUN_CROSS_EVAL" == "1" ]]; then
  eval_checkpoint_on_dataset "A+B model on View A test" "$AB_CKPT" "$A_DATASET" "$AB_EXP/eval_AB_model_on_viewA"
  eval_checkpoint_on_dataset "A+B model on View B test" "$AB_CKPT" "$B_DATASET" "$AB_EXP/eval_AB_model_on_viewB"
fi
eval_checkpoint_on_dataset "A+B model on combined View A+B test" "$AB_CKPT" "$AB_DATASET" "$AB_EXP/eval_AB_model_on_viewAB"

log "Pipeline complete."
log "View B experiment: $B_EXP"
log "A+B experiment: $AB_EXP"
