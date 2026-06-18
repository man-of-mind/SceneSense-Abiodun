#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DENSITY="${1:-medium}"
EGO_SPEED_DIFF="${EGO_SPEED_DIFF:-60}"
EGO_FOLLOW_DISTANCE_M="${EGO_FOLLOW_DISTANCE_M:-28.0}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-3600}"
STOP_AFTER_LOOPS="${STOP_AFTER_LOOPS:-2}"
LOOP_RETURN_RADIUS_M="${LOOP_RETURN_RADIUS_M:-2.0}"
LOOP_MIN_DISTANCE_M="${LOOP_MIN_DISTANCE_M:-200.0}"
DATE_TAG="${DATE_TAG:-20260617}"
ROUTE_SPAWN_INDICES="${ROUTE_SPAWN_INDICES:-80,85,91,94,99,80}"
ROUTE_POINT_SPACING_M="${ROUTE_POINT_SPACING_M:-3.0}"

case "$DENSITY" in
  low)
    NPC_VEHICLES="${NPC_VEHICLES:-8}"
    NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-10}"
    SEED="${SEED:-31}"
    ;;
  medium)
    NPC_VEHICLES="${NPC_VEHICLES:-20}"
    NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-25}"
    SEED="${SEED:-41}"
    ;;
  crowded)
    NPC_VEHICLES="${NPC_VEHICLES:-35}"
    NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-45}"
    SEED="${SEED:-51}"
    ;;
  *)
    echo "Usage: $0 {low|medium|crowded}" >&2
    exit 2
    ;;
esac

EXPERIMENT_ID="${EXPERIMENT_ID:-moving_ego_tl16_spawn80_fixedroute_speed${EGO_SPEED_DIFF}_${DENSITY}_visual_${STOP_AFTER_LOOPS}loops_${DATE_TAG}_stride${SAMPLE_STRIDE}}"

python3 carla_collect_moving_ego_fusion_training_data.py \
  --experiment-id "$EXPERIMENT_ID" \
  --seed "$SEED" \
  --preview \
  --preview-width 1440 \
  --preview-height 810 \
  --no-ego-freeze \
  --ego-autopilot-speed-difference-pct "$EGO_SPEED_DIFF" \
  --ego-follow-distance-m "$EGO_FOLLOW_DISTANCE_M" \
  --ego-ignore-lights-pct 0 \
  --ego-fixed-path-spawn-indices "$ROUTE_SPAWN_INDICES" \
  --ego-fixed-path-loop \
  --ego-fixed-path-min-spacing-m "$ROUTE_POINT_SPACING_M" \
  --ego-disable-lane-change \
  --route-progress-every-s 1.0 \
  --loop-return-radius-m "$LOOP_RETURN_RADIUS_M" \
  --loop-min-distance-m "$LOOP_MIN_DISTANCE_M" \
  --loop-min-elapsed-s 30.0 \
  --stop-after-loops "$STOP_AFTER_LOOPS" \
  --stop-on-stuck \
  --stuck-ignore-traffic-light-waits \
  --stuck-speed-threshold-mps 0.20 \
  --stuck-timeout-s 60.0 \
  --stuck-min-elapsed-s 30.0 \
  --max-samples "$MAX_SAMPLES" \
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
  --npc-vehicles "$NPC_VEHICLES" \
  --npc-pedestrians "$NPC_PEDESTRIANS" \
  --npc-vehicle-speed-difference-pct 10 \
  --npc-pedestrian-max-speed-mps 0.9 \
  --npc-pedestrian-cross-factor 0.5 \
  --spawn-radius 80 \
  --gt-max-distance-m 140 \
  --include-pedestrians
