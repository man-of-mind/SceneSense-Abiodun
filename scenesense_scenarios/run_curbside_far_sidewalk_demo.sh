#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

TARGET_START_LAT="${TARGET_START_LAT:-5.2}"
TARGET_FORWARD="${TARGET_FORWARD:--6.5}"
TARGET_END_LAT="${TARGET_END_LAT:-2.6}"
TARGET_SPEED="${TARGET_SPEED:-21.0}"
ROUTE_LEAD="${ROUTE_LEAD:-24.0}"
EGO_TARGET_SPEED="${EGO_TARGET_SPEED:-6.0}"
EGO_THROTTLE="${EGO_THROTTLE:-0.45}"
CONFLICT_DISTANCE="${CONFLICT_DISTANCE:-31.0}"
OCCLUDER_LAT="${OCCLUDER_LAT:-2.8}"
OCCLUDER_COUNT="${OCCLUDER_COUNT:-1}"
SLOT1_FORWARD="${SLOT1_FORWARD:--7.5}"
OCCLUDER_BP="${OCCLUDER_BP:-vehicle.sprinter.mercedes}"
HELPER_SPEED="${HELPER_SPEED:-5.5}"
HELPER_SPAWN_FORWARD="${HELPER_SPAWN_FORWARD:-25.0}"
HELPER_TARGET_FORWARD="${HELPER_TARGET_FORWARD:--30.0}"

args=(
  python3 "${SCRIPT_DIR}/scenesense_scenario_harness.py"
  --scenario curbside_parked_vehicle_pedestrian_occlusion
  --load-town
  --town Town10HD_Opt
  --anchor-source spawn_point
  --anchor-spawn-index 152
  --ego-spawn-index 152
  --seed 7
  --duration-s 25
  --ego-sensors
  --ego-camera-preview
  --scripted-ego-drive
  --ego-drive-mode waypoint
  --ego-route-choice straight
  --ego-target-speed "${EGO_TARGET_SPEED}"
  --ego-drive-throttle "${EGO_THROTTLE}"
  --stop-on-target-collision
  --post-target-collision-hold-s 3.0
  --target-crossing
  --target-crossing-delay-s 0.0
  --target-crossing-speed "${TARGET_SPEED}"
  --target-crossing-control-speed "${TARGET_SPEED}"
  --target-crossing-motion-mode walker_control
  --target-crossing-trigger-distance-m 0.0
  --target-crossing-trigger-ttc-s 0.0
  --target-crossing-trigger-route-lead-m "${ROUTE_LEAD}"
  --curbside-conflict-distance-m "${CONFLICT_DISTANCE}"
  --curbside-target-forward-offset-m "${TARGET_FORWARD}"
  --curbside-target-start-lateral-offset-m "${TARGET_START_LAT}"
  --curbside-target-end-lateral-offset-m "${TARGET_END_LAT}"
  --curbside-occluder-lateral-offset-m "${OCCLUDER_LAT}"
  --curbside-occluder-count "${OCCLUDER_COUNT}"
  --curbside-slot-1-forward-m "${SLOT1_FORWARD}"
  --curbside-occluder-z-offset-m 0.0
  --curbside-ego-start-forward-m 0.0
  --helper-vehicle
  --helper-drive
  --helper-target-speed "${HELPER_SPEED}"
  --helper-stop-distance-to-conflict-m 1.0
  --curbside-helper-spawn-forward-m "${HELPER_SPAWN_FORWARD}"
  --curbside-helper-target-forward-m "${HELPER_TARGET_FORWARD}"
  --helper-camera-preview
  --evidence-pack
  --evidence-camera-buffer-size 300
  --spectator-focus conflict
)

if [[ -n "${OCCLUDER_BP}" ]]; then
  args+=(--curbside-occluder-blueprint "${OCCLUDER_BP}")
fi

printf 'Running curbside demo with TARGET_START_LAT=%s ROUTE_LEAD=%s TARGET_SPEED=%s EGO_THROTTLE=%s OCCLUDER_BP=%s HELPER_SPEED=%s\n' \
  "${TARGET_START_LAT}" "${ROUTE_LEAD}" "${TARGET_SPEED}" "${EGO_THROTTLE}" "${OCCLUDER_BP}" "${HELPER_SPEED}"
"${args[@]}"
