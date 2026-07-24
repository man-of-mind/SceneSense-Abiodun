#!/usr/bin/env bash
# Self-contained loopback live deployment (reuses the run_common front/back invocations) WITH overlay
# frame capture, so the route/scene can be eyeballed and loc/latency recomputed. Reuses existing CARLA.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
SCEN="staleness/carla_fusion_staleness_scenario.py"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
ROUTE="fusion_training_data/moving_ego_pps200000_crowded_8loops_stride2/route_progress.csv"
RUN="downlink_latency_fps/runs/loopback_inspect/fps10_inspect"
OVR="downlink_latency_fps/overlay_inspect"
LOG="downlink_latency_fps/logs/loopback_inspect"
mkdir -p "$RUN" "$OVR" "$LOG"
FRAMES="${FRAMES:-600}"

echo "[inspect] starting local back-half on 127.0.0.1:51002"
"$PY" "$SCEN" --role back --bind-host 127.0.0.1 --remote-host 127.0.0.1 \
  --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zstd --zstd-level 3 \
  --roi-threshold 0.0 --remote-port 51002 --remote-source-port 51013 --camera-result-port 51004 \
  --front-device cuda --back-device cuda --back-log-every 100 > "$LOG/back.log" 2>&1 &
BACK_PID=$!
for i in $(seq 1 30); do ss -lunp 2>/dev/null | grep -q ":51002" && break; sleep 2; done
ss -lunp 2>/dev/null | grep -q ":51002" || { echo "[inspect] back-half failed to bind; see $LOG/back.log"; kill $BACK_PID 2>/dev/null; exit 2; }
echo "[inspect] back ready pid=$BACK_PID; starting front ($FRAMES frames, overlay every 10)"

# Drivable config from spatial_map_coop/rl_agent: waypoint loop (not dense CSV), ignore-lights 50 so the ego
# does not get trapped behind stopped traffic, 28 NPCs (not 60), seed 31. This actually drives the trained loop.
"$PY" "$SCEN" --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps 10 --seed 31 \
  --sensor-platform ego_vehicle --no-ego-freeze \
  --ego-spawn-index 80 --ego-spawn-forward-offset-m 0 --ego-spawn-right-offset-m 0 --ego-spawn-z-offset-m 0.15 \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop \
  --ego-disable-lane-change --ego-ignore-lights-pct 50 \
  --camera-resolution custom --camera-width 1280 --camera-height 720 --camera-fov 120 \
  --model-input-width 768 --model-input-height 432 \
  --ego-camera-x 1.8 --ego-camera-y 0.0 --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-yaw 0.0 \
  --ego-radar-yaw 0.0 --radar-hfov 120 --radar-vfov 30 --radar-range 120 \
  --radar-points-per-second 200000 --radar-raster-radius-px 4 --radar-temporal-window-frames 2 \
  --npc-vehicles 28 --npc-pedestrians 35 --spawn-radius 80 --npc-speed-difference-pct 10 \
  --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zstd --zstd-level 3 \
  --roi-threshold 0.0 --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 \
  --overlay-save-dir "$OVR" --overlay-save-every 10 \
  --run-group loopback_inspect --run-id loopback_inspect --transport-label loopback_inspect \
  --metrics-run-dir "$RUN" \
  --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
  --front-device cuda > "$LOG/front.log" 2>&1
RC=$?
echo "[inspect] front rc=$RC"
kill $BACK_PID 2>/dev/null; sleep 2
echo "[inspect] overlays: $(ls "$OVR" | wc -l) frames in $OVR"
echo "INSPECT_DONE rc=$RC"
