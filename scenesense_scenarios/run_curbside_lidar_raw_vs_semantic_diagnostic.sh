#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

TARGET_START_LAT="${TARGET_START_LAT:-5.5}"
TARGET_FORWARD="${TARGET_FORWARD:--6.5}"
TARGET_END_LAT="${TARGET_END_LAT:-2.6}"
TARGET_SPEED="${TARGET_SPEED:-26.5}"
ROUTE_LEAD="${ROUTE_LEAD:-24.0}"
EGO_TARGET_SPEED="${EGO_TARGET_SPEED:-15.2}"
EGO_THROTTLE="${EGO_THROTTLE:-0.45}"
CONFLICT_DISTANCE="${CONFLICT_DISTANCE:-31.0}"
OCCLUDER_LAT="${OCCLUDER_LAT:-2.8}"
OCCLUDER_COUNT="${OCCLUDER_COUNT:-1}"
SLOT1_FORWARD="${SLOT1_FORWARD:--7.5}"
OCCLUDER_BP="${OCCLUDER_BP:-vehicle.sprinter.mercedes}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
PREVIEW_WIDTH="${PREVIEW_WIDTH:-1440}"
PREVIEW_HEIGHT="${PREVIEW_HEIGHT:-810}"
DURATION_S="${DURATION_S:-25}"
FPS="${FPS:-10}"
LIDAR_PPS="${LIDAR_PPS:-600000}"
EXPERIMENT_ID="${EXPERIMENT_ID:-curbside_raw_vs_semantic_lidar_clean_crossing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-lidar_diagnostic_runs}"
EGO_MOTION="${EGO_MOTION:-stationary}"

args=(
  python3 "carla_curbside_lidar_raw_vs_semantic_diagnostic.py"
  --load-town
  --town Town10HD_Opt
  --seed 7
  --experiment-id "${EXPERIMENT_ID}"
  --output-root "${OUTPUT_ROOT}"
  --duration-s "${DURATION_S}"
  --fps "${FPS}"
  --preview
  --preview-width "${PREVIEW_WIDTH}"
  --preview-height "${PREVIEW_HEIGHT}"
  --camera-width "${CAMERA_WIDTH}"
  --camera-height "${CAMERA_HEIGHT}"
  --camera-fov 120
  --anchor-spawn-index 152
  --ego-spawn-index 152
  --ego-motion "${EGO_MOTION}"
  --ego-target-speed "${EGO_TARGET_SPEED}"
  --ego-drive-throttle "${EGO_THROTTLE}"
  --ego-route-lookahead-m 24.0
  --target-crossing-delay-s 3.0
  --target-crossing-speed "${TARGET_SPEED}"
  --target-crossing-control-speed "${TARGET_SPEED}"
  --target-crossing-trigger-route-lead-m "${ROUTE_LEAD}"
  --curbside-conflict-distance-m "${CONFLICT_DISTANCE}"
  --curbside-target-forward-offset-m "${TARGET_FORWARD}"
  --curbside-target-start-lateral-offset-m "${TARGET_START_LAT}"
  --curbside-target-end-lateral-offset-m "${TARGET_END_LAT}"
  --curbside-occluder-lateral-offset-m "${OCCLUDER_LAT}"
  --curbside-occluder-count "${OCCLUDER_COUNT}"
  --curbside-slot-1-forward-m "${SLOT1_FORWARD}"
  --curbside-occluder-blueprint "${OCCLUDER_BP}"
  --sensor-x 1.8
  --sensor-y 0.0
  --sensor-z 1.55
  --sensor-pitch -4.0
  --sensor-yaw 0.0
  --sensor-roll 0.0
  --lidar-range 120
  --lidar-upper-fov 15
  --lidar-lower-fov -15
  --lidar-channels 64
  --lidar-rotation-frequency 20
  --lidar-pps "${LIDAR_PPS}"
  --gt-max-distance-m 140
  --person-association-mode radius
  --person-association-radius-m 1.1
  --person-association-z-down-m 0.4
  --person-association-z-up-m 5.0
  --min-person-points 2
  --debug-every 20
)

printf 'Running curbside raw-vs-semantic LiDAR diagnostic with EXPERIMENT_ID=%s EGO_MOTION=%s TARGET_START_LAT=%s ROUTE_LEAD=%s TARGET_SPEED=%s LIDAR_PPS=%s CAMERA=%sx%s PREVIEW=%sx%s\n' \
  "${EXPERIMENT_ID}" "${EGO_MOTION}" "${TARGET_START_LAT}" "${ROUTE_LEAD}" "${TARGET_SPEED}" "${LIDAR_PPS}" \
  "${CAMERA_WIDTH}" "${CAMERA_HEIGHT}" "${PREVIEW_WIDTH}" "${PREVIEW_HEIGHT}"
"${args[@]}"
