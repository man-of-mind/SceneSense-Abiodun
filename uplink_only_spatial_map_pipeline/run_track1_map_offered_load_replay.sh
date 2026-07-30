#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/track1_map_offered_load_replay_${STAMP}}"
FPS_LIST="${FPS_LIST:-10 20 30}"
MAP_DELAY_LIST="${MAP_DELAY_LIST:-0 40 60}"
FRAMES="${FRAMES:-180}"
DRAIN_S="${DRAIN_S:-10}"

MAP_SERVER="$SCRIPT_DIR/spatial_map_server_moving_ego_uplink_only_baseline.py"
REPLAY="$SCRIPT_DIR/replay_spatial_map_offered_load.py"

mkdir -p "$RUN_ROOT"
printf "label\tfps\tmap_delay_ms\tframes\trun_dir\n" > "$RUN_ROOT/run_index.tsv"

stop_server() {
  local pid="$1"
  if [ -z "$pid" ]; then
    return 0
  fi
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

run_one() {
  local fps="$1"
  local delay="$2"
  local udp_port="$3"
  local api_port="$4"
  local label="replay_fps${fps}_map${delay}"
  local run_dir="$RUN_ROOT/$label"
  mkdir -p "$run_dir"
  printf "%s\t%s\t%s\t%s\t%s\n" "$label" "$fps" "$delay" "$FRAMES" "$run_dir" >> "$RUN_ROOT/run_index.tsv"

  echo "===== START $label frames=$FRAMES drain_s=$DRAIN_S ====="
  env MPLCONFIGDIR=/tmp/matplotlib "$PY" "$MAP_SERVER" \
    --udp-host 127.0.0.1 \
    --udp-port "$udp_port" \
    --api-host 127.0.0.1 \
    --api-port "$api_port" \
    --render-hz 2 \
    --focus-follow-stream-id "$label" \
    --focus-radius-m 80 \
    --focus-follow-forward-bias 0.35 \
    --ingest-metrics-csv "$run_dir/map_ingest_metrics.csv" \
    --map-update-delay-ms "$delay" \
    > "$run_dir/map_server.log" 2>&1 &
  local server_pid="$!"
  sleep 1
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "FATAL: map server failed for $label; see $run_dir/map_server.log"
    return 4
  fi

  set +e
  "$PY" "$REPLAY" \
    --host 127.0.0.1 \
    --port "$udp_port" \
    --fps "$fps" \
    --frames "$FRAMES" \
    --stream-id "$label" \
    --output-csv "$run_dir/replay_send_metrics.csv" \
    > "$run_dir/replay.log" 2>&1
  local replay_rc="$?"
  set -e

  sleep "$DRAIN_S"
  stop_server "$server_pid"

  if [ "$replay_rc" -ne 0 ]; then
    echo "FATAL: replay failed for $label rc=$replay_rc; see $run_dir/replay.log"
    return "$replay_rc"
  fi
  echo "===== DONE $label ====="
}

port_offset=0
for delay in $MAP_DELAY_LIST; do
  for fps in $FPS_LIST; do
    udp_port=$((39400 + port_offset))
    api_port=$((5300 + port_offset))
    run_one "$fps" "$delay" "$udp_port" "$api_port"
    port_offset=$((port_offset + 1))
  done
done

echo "Track-1 map offered-load replay complete: $RUN_ROOT"
