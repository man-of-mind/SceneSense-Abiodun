#!/usr/bin/env bash
# Shared runner for the downlink/result-return latency × FPS study.
#
# Source this from condition-specific scripts after setting:
#   CONDITION, TRANSPORT_LABEL, FRONT_BIND_HOST, BACK_REMOTE_HOST
#   START_LOCAL_BACK=1 for loopback, 0 for OAI-front-only.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "$AB" || exit 2

PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"

SCEN="staleness/carla_fusion_staleness_scenario.py"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
ROUTE_PROGRESS_CSV="${ROUTE_PROGRESS_CSV:-fusion_training_data/moving_ego_pps200000_crowded_8loops_stride2/route_progress.csv}"

CONDITION="${CONDITION:?CONDITION is required}"
TRANSPORT_LABEL="${TRANSPORT_LABEL:-$CONDITION}"
FRONT_BIND_HOST="${FRONT_BIND_HOST:-127.0.0.1}"
BACK_BIND_HOST="${BACK_BIND_HOST:-127.0.0.1}"
BACK_REMOTE_HOST="${BACK_REMOTE_HOST:-127.0.0.1}"
BACK_RESULT_REMOTE_HOST="${BACK_RESULT_REMOTE_HOST:-$FRONT_BIND_HOST}"
START_LOCAL_BACK="${START_LOCAL_BACK:-1}"

FPS_LIST="${FPS_LIST:-5 10 20 30}"
DURATION_S="${DURATION_S:-130}"
RESULT_TIMEOUT="${RESULT_TIMEOUT:-1.5}"
SEED="${SEED:-31}"
ENTROPY_CODER="${ENTROPY_CODER:-zstd}"  # 2026-07-22: zstd is the deployed codec (lossless, ~4x faster than zlib, +delivery)
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}"
ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}"
AE_CHECKPOINT="${AE_CHECKPOINT:-}"
RADAR_RASTERIZER="${RADAR_RASTERIZER:-legacy}"
QUEUE_PROBE_MODE="${QUEUE_PROBE_MODE:-0}"
QUEUE_PROBE_IDLE_BEFORE_S="${QUEUE_PROBE_IDLE_BEFORE_S:-10}"
QUEUE_PROBE_COOLDOWN_S="${QUEUE_PROBE_COOLDOWN_S:-120}"

NPC_VEHICLES="${NPC_VEHICLES:-28}"
NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-35}"
EGO_IGNORE_LIGHTS_PCT="${EGO_IGNORE_LIGHTS_PCT:-50}"
EGO_SPAWN_INDICES="${EGO_SPAWN_INDICES:-80,85,91,94,99,80}"
SPAWN_RADIUS="${SPAWN_RADIUS:-80}"
NPC_SPEED_DIFFERENCE_PCT="${NPC_SPEED_DIFFERENCE_PCT:-10}"
EGO_SPEED_DIFFERENCE_PCT="${EGO_SPEED_DIFFERENCE_PCT:-60}"
EGO_FOLLOW_DISTANCE_M="${EGO_FOLLOW_DISTANCE_M:-28.0}"

CAMERA_SOURCE_PORT="${CAMERA_SOURCE_PORT:-51001}"
REMOTE_PORT="${REMOTE_PORT:-51002}"
FRONT_SOURCE_PORT="${FRONT_SOURCE_PORT:-51003}"
BACK_SOURCE_PORT="${BACK_SOURCE_PORT:-51013}"
CAMERA_RESULT_PORT="${CAMERA_RESULT_PORT:-51004}"

RUN_ROOT="${RUN_ROOT:-$AB/downlink_latency_fps/runs}"
LOG_ROOT="${LOG_ROOT:-$AB/downlink_latency_fps/logs}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/$CONDITION" "$LOG_ROOT/$CONDITION"

say() {
  echo "[$(date +%H:%M:%S)] $*"
}

calc_frames() {
  local fps="$1"
  echo $(( DURATION_S * fps ))
}

wait_udp_port() {
  local port="$1"
  local tries="${2:-30}"
  for _ in $(seq 1 "$tries"); do
    if ss -lunp 2>/dev/null | grep -q ":$port"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

udp_port_in_use() {
  local port="$1"
  ss -lunp 2>/dev/null | grep -q ":$port"
}

start_local_back() {
  local fps="$1"
  local back_log="$LOG_ROOT/$CONDITION/back_fps${fps}_${BATCH_ID}.log"
  if udp_port_in_use "$REMOTE_PORT"; then
    say "ERROR: UDP :$REMOTE_PORT is already in use before starting local back-half; refusing to mask a stale process"
    ss -lunp 2>/dev/null | grep ":$REMOTE_PORT" || true
    return 1
  fi
  say "starting local back-half for fps=$fps on ${BACK_BIND_HOST}:${REMOTE_PORT}"
  local ae_args=()
  if [[ -n "$AE_CHECKPOINT" ]]; then
    ae_args=(--ae-checkpoint "$AE_CHECKPOINT")
  fi
  "$PY" "$SCEN" \
    --role back \
    --bind-host "$BACK_BIND_HOST" \
    --remote-host "$BACK_RESULT_REMOTE_HOST" \
    --fusion-checkpoint "$CKPT" \
    --quantization-mode "$QUANTIZATION_MODE" \
    --entropy-coder "$ENTROPY_CODER" \
    --zstd-level "$ZSTD_LEVEL" \
    --roi-threshold "$ROI_THRESHOLD" \
    "${ae_args[@]}" \
    --remote-port "$REMOTE_PORT" \
    --remote-source-port "$BACK_SOURCE_PORT" \
    --camera-result-port "$CAMERA_RESULT_PORT" \
    --front-device cuda \
    --back-device cuda \
    --back-log-every 100 \
    > "$back_log" 2>&1 &
  BACK_PID=$!
  if ! wait_udp_port "$REMOTE_PORT" 30; then
    say "ERROR: local back-half did not bind UDP :$REMOTE_PORT; see $back_log"
    return 1
  fi
  say "local back-half ready pid=$BACK_PID"
  return 0
}

stop_local_back() {
  if [[ -n "${BACK_PID:-}" ]]; then
    kill "$BACK_PID" 2>/dev/null
    sleep 2
    kill -9 "$BACK_PID" 2>/dev/null
    BACK_PID=""
  fi
}

run_front_point() {
  local fps="$1"
  local frames
  frames="$(calc_frames "$fps")"
  local run_group="downlink_${CONDITION}_fps${fps}_${BATCH_ID}"
  local run_dir="$RUN_ROOT/$CONDITION/fps_${fps}_${BATCH_ID}"
  local front_log="$LOG_ROOT/$CONDITION/front_fps${fps}_${BATCH_ID}.log"
  local queue_probe_args=()
  local ae_args=()
  if [[ "$QUEUE_PROBE_MODE" == "1" ]]; then
    queue_probe_args=(
      --queue-probe-mode
      --queue-probe-idle-before-s "$QUEUE_PROBE_IDLE_BEFORE_S"
      --queue-probe-cooldown-s "$QUEUE_PROBE_COOLDOWN_S"
    )
  fi
  if [[ -n "$AE_CHECKPOINT" ]]; then
    ae_args=(--ae-checkpoint "$AE_CHECKPOINT")
  fi
  mkdir -p "$run_dir"

  say "front run: condition=$CONDITION fps=$fps frames=$frames run_dir=$run_dir"
  "$PY" "$SCEN" \
    --role front \
    --bind-host "$FRONT_BIND_HOST" \
    --remote-host "$BACK_REMOTE_HOST" \
    --sync-world \
    --fps "$fps" \
    --seed "$SEED" \
    --sensor-platform ego_vehicle \
    --no-ego-freeze \
    --ego-ignore-lights-pct "$EGO_IGNORE_LIGHTS_PCT" \
    --ego-disable-lane-change \
    --ego-fixed-path-spawn-indices "$EGO_SPAWN_INDICES" \
    --ego-fixed-path-loop \
    --ego-spawn-index 80 \
    --ego-spawn-forward-offset-m 0.0 \
    --ego-spawn-right-offset-m 0.0 \
    --ego-spawn-z-offset-m 0.15 \
    --camera-resolution custom \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fov 120 \
    --model-input-width 768 \
    --model-input-height 432 \
    --ego-camera-x 1.8 \
    --ego-camera-y 0.0 \
    --ego-camera-z 1.55 \
    --ego-camera-pitch -4.0 \
    --ego-camera-yaw 0.0 \
    --ego-radar-yaw 0.0 \
    --radar-hfov 120 \
    --radar-vfov 30 \
    --radar-range 120 \
    --radar-points-per-second 200000 \
    --radar-raster-radius-px 4 \
    --radar-rasterizer "$RADAR_RASTERIZER" \
    --radar-temporal-window-frames 2 \
    --npc-vehicles "$NPC_VEHICLES" \
    --npc-pedestrians "$NPC_PEDESTRIANS" \
    --spawn-radius "$SPAWN_RADIUS" \
    --npc-speed-difference-pct "$NPC_SPEED_DIFFERENCE_PCT" \
    --fusion-checkpoint "$CKPT" \
    --quantization-mode "$QUANTIZATION_MODE" \
    --entropy-coder "$ENTROPY_CODER" \
    --zstd-level "$ZSTD_LEVEL" \
    --roi-threshold "$ROI_THRESHOLD" \
    "${ae_args[@]}" \
    --no-spatial-map-stream \
    --headless \
    --max-frames "$frames" \
    --result-timeout "$RESULT_TIMEOUT" \
    "${queue_probe_args[@]}" \
    --run-group "$run_group" \
    --run-id "$run_group" \
    --transport-label "$TRANSPORT_LABEL" \
    --metrics-run-dir "$run_dir" \
    --spatial-map-stream-id "$run_group" \
    --camera-source-port "$CAMERA_SOURCE_PORT" \
    --remote-port "$REMOTE_PORT" \
    --remote-source-port "$FRONT_SOURCE_PORT" \
    --camera-result-port "$CAMERA_RESULT_PORT" \
    --front-device cuda \
    > "$front_log" 2>&1
  local rc=$?
  say "front done: condition=$CONDITION fps=$fps rc=$rc log=$front_log"
  return "$rc"
}

run_sweep() {
  say "===== START condition=$CONDITION transport=$TRANSPORT_LABEL batch=$BATCH_ID fps_list=[$FPS_LIST] duration_s=$DURATION_S ====="
  local rc_all=0
  for fps in $FPS_LIST; do
    BACK_PID=""
    if [[ "$START_LOCAL_BACK" == "1" ]]; then
      if ! start_local_back "$fps"; then
        stop_local_back
        rc_all=1
        continue
      fi
    else
      say "using existing remote back-half at ${BACK_REMOTE_HOST}:${REMOTE_PORT}"
    fi

    if ! run_front_point "$fps"; then
      rc_all=1
    fi
    stop_local_back
    sleep 3
  done
  say "===== DONE condition=$CONDITION batch=$BATCH_ID rc=$rc_all ====="
  return "$rc_all"
}

cleanup_common() {
  stop_local_back
  if declare -F after_run_common >/dev/null 2>&1; then
    after_run_common
  fi
}

trap cleanup_common EXIT
run_sweep
