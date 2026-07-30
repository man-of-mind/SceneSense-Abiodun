#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$AB_DIR/.." && pwd)"

PY="${PY:-/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_FRAMES="${MAX_FRAMES:-120}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/track1_ideal_loopback_matrix_${STAMP}}"
RADAR_RASTERIZER="${RADAR_RASTERIZER:-legacy}"
FPS_LIST="${FPS_LIST:-10 20}"
MAP_DELAYS_MS="${MAP_DELAYS_MS:-0 40}"
BASE_UDP_PORT="${BASE_UDP_PORT:-39310}"
BASE_API_PORT="${BASE_API_PORT:-5210}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
MODEL_INPUT_WIDTH="${MODEL_INPUT_WIDTH:-768}"
MODEL_INPUT_HEIGHT="${MODEL_INPUT_HEIGHT:-432}"
CAPTURE_PIPELINE="${CAPTURE_PIPELINE:-0}"
CAPTURE_PIPELINE_QUEUE_SIZE="${CAPTURE_PIPELINE_QUEUE_SIZE:-2}"
CAPTURE_PIPELINE_DROP_OLDEST="${CAPTURE_PIPELINE_DROP_OLDEST:-0}"
SENSOR_EVERY_TICK="${SENSOR_EVERY_TICK:-0}"
NPC_VEHICLES="${NPC_VEHICLES:-28}"
NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-35}"

CLIENT="$SCRIPT_DIR/carla_fusion_staleness_scenario_uplink_only.py"
MAP_SERVER="$SCRIPT_DIR/spatial_map_server_moving_ego_uplink_only_baseline.py"
CHECKPOINT="$AB_DIR/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"

mkdir -p "$RUN_ROOT"

rmem="$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)"
wmem="$(sysctl -n net.core.wmem_max 2>/dev/null || echo 0)"
if [ "$rmem" -lt 8000000 ] || [ "$wmem" -lt 8000000 ]; then
  echo "FATAL: ideal loopback buffers not active: rmem_max=$rmem wmem_max=$wmem"
  echo "Run: sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608"
  exit 3
fi

"$PY" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
s.close()
print(f"SO_RCVBUF actual after requesting 8 MiB = {actual}")
if actual < 8 * 1024 * 1024:
    raise SystemExit("FATAL: actual UDP receive buffer is below 8 MiB")
PY

printf "label\tfps\tmap_delay_ms\tradar_rasterizer\trun_dir\n" > "$RUN_ROOT/run_index.tsv"

run_one() {
  local label="$1"
  local fps="$2"
  local map_delay_ms="$3"
  local udp_port="$4"
  local api_port="$5"
  local run_dir="$RUN_ROOT/$label"
  local stream_id="$label"
  local capture_pipeline_args=()
  if [ "$CAPTURE_PIPELINE" = "1" ]; then
    capture_pipeline_args+=(--capture-pipeline)
    capture_pipeline_args+=(--capture-pipeline-queue-size "$CAPTURE_PIPELINE_QUEUE_SIZE")
    if [ "$CAPTURE_PIPELINE_DROP_OLDEST" = "1" ]; then
      capture_pipeline_args+=(--capture-pipeline-drop-oldest)
    else
      capture_pipeline_args+=(--no-capture-pipeline-drop-oldest)
    fi
  fi
  local sensor_tick_args=()
  if [ "$SENSOR_EVERY_TICK" = "1" ]; then
    sensor_tick_args+=(--sensor-every-tick)
  else
    sensor_tick_args+=(--no-sensor-every-tick)
  fi

  mkdir -p "$run_dir"
  if [ "${SKIP_DONE:-1}" = "1" ] && [ -s "$run_dir/edge_uplink_metrics.summary.json" ] && [ -s "$run_dir/map_ingest_metrics.csv" ]; then
    echo "===== SKIP $label; existing artifacts found ====="
    return 0
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$label" "$fps" "$map_delay_ms" "$RADAR_RASTERIZER" "$run_dir" >> "$RUN_ROOT/run_index.tsv"

  echo "===== START $label fps=$fps map_delay_ms=$map_delay_ms max_frames=$MAX_FRAMES radar_rasterizer=$RADAR_RASTERIZER camera=${CAMERA_WIDTH}x${CAMERA_HEIGHT} model=${MODEL_INPUT_WIDTH}x${MODEL_INPUT_HEIGHT} capture_pipeline=$CAPTURE_PIPELINE queue=$CAPTURE_PIPELINE_QUEUE_SIZE drop_oldest=$CAPTURE_PIPELINE_DROP_OLDEST sensor_every_tick=$SENSOR_EVERY_TICK npc=${NPC_VEHICLES}/${NPC_PEDESTRIANS} ====="
  env MPLCONFIGDIR=/tmp/matplotlib "$PY" "$MAP_SERVER" \
    --udp-host 127.0.0.1 \
    --udp-port "$udp_port" \
    --api-host 127.0.0.1 \
    --api-port "$api_port" \
    --render-hz 2 \
    --focus-follow-stream-id "$stream_id" \
    --focus-radius-m 80 \
    --focus-follow-forward-bias 0.35 \
    --ingest-metrics-csv "$run_dir/map_ingest_metrics.csv" \
    --map-update-delay-ms "$map_delay_ms" \
    > "$run_dir/map_server.log" 2>&1 &
  local server_pid="$!"
  sleep 2
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "FATAL: map server failed for $label; see $run_dir/map_server.log"
    return 4
  fi

  set +e
  "$PY" "$CLIENT" \
    --role loopback \
    --uplink-only-spatial-map \
    --edge-result-mode none \
    --sync-world \
    --fps "$fps" \
    --seed 31 \
    --sensor-platform ego_vehicle \
    --no-ego-freeze \
    --ego-ignore-lights-pct 50 \
    --ego-disable-lane-change \
    --ego-fixed-path-spawn-indices 80,85,91,94,99,80 \
    --ego-fixed-path-loop \
    --ego-spawn-index 80 \
    --ego-spawn-z-offset-m 0.15 \
    --camera-resolution custom \
    --camera-width "$CAMERA_WIDTH" \
    --camera-height "$CAMERA_HEIGHT" \
    --camera-fov 120 \
    "${sensor_tick_args[@]}" \
    --model-input-width "$MODEL_INPUT_WIDTH" \
    --model-input-height "$MODEL_INPUT_HEIGHT" \
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
    "${capture_pipeline_args[@]}" \
    --radar-temporal-window-frames 2 \
    --npc-vehicles "$NPC_VEHICLES" \
    --npc-pedestrians "$NPC_PEDESTRIANS" \
    --spawn-radius 80 \
    --npc-speed-difference-pct 10 \
    --fusion-checkpoint "$CHECKPOINT" \
    --quantization-mode per_channel_uint8 \
    --entropy-coder zstd \
    --zstd-level 3 \
    --roi-threshold 0.0 \
    --spatial-map-stream \
    --spatial-map-host 127.0.0.1 \
    --spatial-map-port "$udp_port" \
    --spatial-map-stream-id "$stream_id" \
    --edge-metrics-csv "$run_dir/edge_uplink_metrics.csv" \
    --enable-run-logging \
    --metrics-run-dir "$run_dir/front_metrics" \
    --transport-label "$label" \
    --run-group "$(basename "$RUN_ROOT")" \
    --run-id "$label" \
    --headless \
    --max-frames "$MAX_FRAMES" \
    --uplink-drain-grace-s 8 \
    --front-device cuda \
    --back-device cuda \
    --back-log-every 20 \
    > "$run_dir/client.log" 2>&1
  local client_rc="$?"
  set -e

  kill -TERM "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true

  if [ "$client_rc" -ne 0 ]; then
    echo "FATAL: client failed for $label rc=$client_rc; see $run_dir/client.log"
    return "$client_rc"
  fi
  echo "===== DONE $label ====="
}

idx=0
for map_delay_ms in $MAP_DELAYS_MS; do
  for fps in $FPS_LIST; do
    fps_label="${fps//./p}"
    label="ideal_none_fps${fps_label}_map${map_delay_ms}_${RADAR_RASTERIZER}"
    if [ "$CAPTURE_PIPELINE" = "1" ]; then
      label="${label}_pipeq${CAPTURE_PIPELINE_QUEUE_SIZE}"
    fi
    if [ "$SENSOR_EVERY_TICK" = "1" ]; then
      label="${label}_sensoreverytick"
    fi
    if [ "$NPC_VEHICLES" != "28" ] || [ "$NPC_PEDESTRIANS" != "35" ]; then
      label="${label}_npc${NPC_VEHICLES}_ped${NPC_PEDESTRIANS}"
    fi
    run_one "$label" "$fps" "$map_delay_ms" "$((BASE_UDP_PORT + idx))" "$((BASE_API_PORT + idx))"
    idx=$((idx + 1))
  done
done

echo "Track-1 ideal loopback matrix complete: $RUN_ROOT"
