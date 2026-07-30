#!/usr/bin/env bash
# Fresh confirmation run for the uplink-only latency-budget staleness analysis (PLAN.md step A/B refinement).
#
# Two conditions, deliberately separated because ONE run cannot honestly measure both:
#
#   A) "L" conditions  -- true uplink-only (--edge-result-mode none), spatial-map server attached,
#      fast radar rasterizer. This is the ONLY configuration in which capture_to_map_update_done_ms
#      is the genuine uplink-only age: the EDGE publishes to the map, so no downlink is in the chain.
#      It logs NO front-side predictions/GT (the no-wait loop skips that block) -- accepted.
#
#   B) "ACC" conditions -- same sensor/model/codec/rasterizer recipe, result-return enabled and the
#      spatial-map stream OFF, so object predictions + actor-origin GT are logged front-side. This is
#      the accuracy/object-motion dataset. Its returned result is NOT part of L and is never added to it.
#
# Both use the deployed recipe: no-AE u8, zstd, 200k radar PPS, radius 4, temporal window 2, fast
# rasterizer, in-domain car-height ego camera (z=1.55, pitch -4, x=1.8, FOV 120).
# Traffic regimes mirror the original opportunity-window speed sweep so object speeds span walk -> ~32 mph.
#
# Reuses the ALREADY-RUNNING CARLA server on rpc-port 2000. Does not start or kill CARLA.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
# NOTE: deliberately do NOT export PYTHONPATH here. The client bootstraps its own sys.path with
#   for _path in (neu_collab, abiodun): if _path not in sys.path: sys.path.insert(0, _path)
# so if abiodun is ALREADY on PYTHONPATH it is not re-inserted, neu_collab ends up ahead of it, and the
# stale top-level copy of carla_split_inference_udp_data_collect.py wins -> UDPMessageSocket TypeError
# on remote_host. The Track-1 runner sets no PYTHONPATH for the same reason.
unset PYTHONPATH
export MPLCONFIGDIR=/tmp/matplotlib-cache

HERE="$AB/staleness/uplink_only_latency_budget"
CLIENT="$AB/uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py"
MAP_SERVER="$AB/uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py"
CKPT="$AB/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$HERE/fresh_run_${STAMP}}"
L_FRAMES="${L_FRAMES:-200}"
ACC_FRAMES="${ACC_FRAMES:-400}"
mkdir -p "$RUN_ROOT"

# ---- ideal-loopback buffer precondition (hard gate, same as the Track-1 matrix) ----
rmem="$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)"
wmem="$(sysctl -n net.core.wmem_max 2>/dev/null || echo 0)"
if [ "$rmem" -lt 8000000 ] || [ "$wmem" -lt 8000000 ]; then
  echo "ideal loopback buffers not active (rmem=$rmem wmem=$wmem); raising them"
  sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608 || {
    echo "FATAL: could not raise UDP buffers"; exit 3; }
fi
"$PY" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
s.close()
print(f"SO_RCVBUF actual after requesting 8 MiB = {actual}")
if actual < 8 * 1024 * 1024:
    raise SystemExit("FATAL: actual UDP receive buffer below 8 MiB")
PY

# ---- CARLA must already be up; we reuse it ----
if ! pgrep -f "CarlaUnreal-Linux-Shipping.*carla-rpc-port=2000" > /dev/null; then
  echo "FATAL: no CARLA server on rpc-port 2000. This script reuses an existing server; it will not start one."
  exit 3
fi
echo "Reusing existing CARLA server on rpc-port 2000."

COMMON_SENSOR=(
  --sync-world --fps 10 --seed 31
  --sensor-platform ego_vehicle --no-ego-freeze
  --ego-ignore-lights-pct 100
  --ego-camera-x 1.8 --ego-camera-y 0.0 --ego-camera-z 1.55
  --ego-camera-pitch -4.0 --ego-camera-yaw 0.0 --camera-fov 120
  --ego-radar-yaw 0.0 --radar-hfov 120 --radar-vfov 30 --radar-range 120
  --radar-points-per-second 200000 --radar-raster-radius-px 4
  --radar-temporal-window-frames 2
  --radar-rasterizer fast
  --fusion-checkpoint "$CKPT"
  --quantization-mode per_channel_uint8 --entropy-coder zstd --zstd-level 3
  --roi-threshold 0.0
  --front-device cuda --back-device cuda
  --headless
)

# regime name : NPC speed-difference-% : npc count   (negative pct = faster than limit)
# Override for a smoke test with e.g. REGIME_SPEC="smoke:-45:40"
if [ -n "${REGIME_SPEC:-}" ]; then
  read -r -a REGIMES <<< "$REGIME_SPEC"
else
  REGIMES=(
    "normal:0:70"
    "fast:-45:70"
    "veryfast:-88:50"
  )
fi

# =================================================================================================
# A) uplink-only L conditions
# =================================================================================================
run_L() {
  local name="$1" spd="$2" npc="$3" idx="$4"
  local run_dir="$RUN_ROOT/L_$name"
  mkdir -p "$run_dir"
  local udp_port=$((39510 + idx)) api_port=$((5410 + idx))
  echo "===== [A/L] regime=$name npc_speed_diff=${spd}% npc=${npc} frames=$L_FRAMES ====="

  "$PY" "$MAP_SERVER" \
    --udp-host 127.0.0.1 --udp-port "$udp_port" \
    --api-host 127.0.0.1 --api-port "$api_port" \
    --render-hz 2 \
    --focus-follow-stream-id "Lsweep_$name" \
    --focus-radius-m 80 --focus-follow-forward-bias 0.35 \
    --ingest-metrics-csv "$run_dir/map_ingest_metrics.csv" \
    --map-update-delay-ms 0 \
    > "$run_dir/map_server.log" 2>&1 &
  local server_pid="$!"
  sleep 3
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "FATAL: map server failed for L_$name; see $run_dir/map_server.log"; return 4
  fi

  "$PY" "$CLIENT" \
    --role loopback --bind-host 127.0.0.1 --remote-host 127.0.0.1 \
    --uplink-only-spatial-map --edge-result-mode none \
    "${COMMON_SENSOR[@]}" \
    --npc-vehicles "$npc" --npc-pedestrians 20 --spawn-radius 150 \
    --npc-speed-difference-pct "$spd" --npc-ignore-lights-pct 100 \
    --ego-autopilot-speed-difference-pct -20 \
    --spatial-map-stream --spatial-map-host 127.0.0.1 \
    --spatial-map-port "$udp_port" --spatial-map-stream-id "Lsweep_$name" \
    --edge-metrics-csv "$run_dir/edge_uplink_metrics.csv" \
    --enable-run-logging --metrics-run-dir "$run_dir/front_metrics" \
    --transport-label "L_$name" --run-group "uplinkonly_Lsweep" --run-id "L_$name" \
    --max-frames "$L_FRAMES" --uplink-drain-grace-s 8 \
    --camera-source-port $((51301 + idx * 10)) --remote-port $((51302 + idx * 10)) \
    --remote-source-port $((51303 + idx * 10)) --camera-result-port $((51304 + idx * 10)) \
    > "$run_dir/client.log" 2>&1
  local rc="$?"

  kill -TERM "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$server_pid" 2>/dev/null || break; sleep 0.2; done
  kill -KILL "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  echo "  [A/L] $name rc=$rc  map_rows=$(wc -l < "$run_dir/map_ingest_metrics.csv" 2>/dev/null || echo 0)"
  return 0
}

# =================================================================================================
# B) accuracy / object-motion conditions (result-return ON purely so preds+GT log; NOT part of L)
# =================================================================================================
run_ACC() {
  local name="$1" spd="$2" npc="$3" idx="$4"
  local run_dir="$RUN_ROOT/ACC_$name"
  mkdir -p "$run_dir"
  echo "===== [B/ACC] regime=$name npc_speed_diff=${spd}% npc=${npc} frames=$ACC_FRAMES ====="

  "$PY" "$CLIENT" \
    --role loopback --bind-host 127.0.0.1 --remote-host 127.0.0.1 \
    --edge-result-mode full --no-spatial-map-stream \
    "${COMMON_SENSOR[@]}" \
    --npc-vehicles "$npc" --npc-pedestrians 20 --spawn-radius 150 \
    --npc-speed-difference-pct "$spd" --npc-ignore-lights-pct 100 \
    --ego-autopilot-speed-difference-pct -20 \
    --result-timeout 1.5 \
    --enable-run-logging --metrics-run-dir "$run_dir/front_metrics" \
    --transport-label "ACC_$name" --run-group "speedsweep_fresh_$name" --run-id "ACC_$name" \
    --max-frames "$ACC_FRAMES" \
    --camera-source-port $((51401 + idx * 10)) --remote-port $((51402 + idx * 10)) \
    --remote-source-port $((51403 + idx * 10)) --camera-result-port $((51404 + idx * 10)) \
    > "$run_dir/client.log" 2>&1
  local rc="$?"
  local gtf predf
  gtf="$(ls "$run_dir"/front_metrics/streams/*object_ground_truth.csv 2>/dev/null | head -1)"
  predf="$(ls "$run_dir"/front_metrics/streams/*object_predictions.csv 2>/dev/null | head -1)"
  echo "  [B/ACC] $name rc=$rc gt_rows=$(wc -l < "${gtf:-/dev/null}" 2>/dev/null || echo 0) pred_rows=$(wc -l < "${predf:-/dev/null}" 2>/dev/null || echo 0)"
  return 0
}

idx=0
for spec in "${REGIMES[@]}"; do
  IFS=: read -r name spd npc <<< "$spec"
  run_L "$name" "$spd" "$npc" "$idx"
  idx=$((idx + 1))
done

idx=0
for spec in "${REGIMES[@]}"; do
  IFS=: read -r name spd npc <<< "$spec"
  run_ACC "$name" "$spd" "$npc" "$idx"
  idx=$((idx + 1))
done

echo "FRESH_RUN_DONE $RUN_ROOT"
