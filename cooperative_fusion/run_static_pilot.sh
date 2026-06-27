#!/usr/bin/env bash
set -uo pipefail
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
ABS="$CARLA_ROOT/PythonAPI/neu_collab/abiodun"
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export PYTHONPATH="$ABS/pole_lraspp_multimodal_fusion:$ABS:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
CLOG="$ABS/cooperative_fusion/phase0b/carla_pilot.log"
cleanup(){ echo "[pilot] teardown CARLA"; pkill -f CarlaUnreal 2>/dev/null; pkill -f "Carla.*Shipping" 2>/dev/null; sleep 4; }
trap cleanup EXIT
echo "[pilot] launching CARLA..."
( cd "$CARLA_ROOT" && ./CarlaUnreal.sh -RenderOffScreen > "$CLOG" 2>&1 ) &
for i in $(seq 1 90); do python3 -c "import socket;s=socket.socket();s.settimeout(1);import sys;sys.exit(0 if s.connect_ex(('127.0.0.1',2000))==0 else 1)" && break; sleep 2; done
python3 -c "import socket;s=socket.socket();s.settimeout(1);import sys;sys.exit(0 if s.connect_ex(('127.0.0.1',2000))==0 else 1)" || { echo "[pilot] CARLA failed"; exit 2; }
echo "[pilot] CARLA up; collecting..."; sleep 6
cd "$ABS"
timeout 1500 python3 carla_collect_parked_ego_fusion_training_data.py \
  --experiment-id coop_static_pilot_20260626 \
  --max-samples 1500 --sample-stride 2 --fps 10 \
  --camera-width 1280 --camera-height 720 --camera-fov 120 \
  --model-input-width 768 --model-input-height 432 \
  --ego-spawn-index 80 --ego-spawn-forward-offset-m 4.0 --ego-spawn-right-offset-m 7.0 --ego-spawn-yaw-offset-deg -28.414 \
  --ego-camera-x 1.8 --ego-camera-z 1.55 --ego-camera-pitch -4.0 \
  --radar-hfov 120 --radar-vfov 30 --radar-range 120 \
  --radar-points-per-second 100000 --radar-raster-radius-px 4 --radar-temporal-window-frames 2 \
  --npc-vehicles 20 --npc-pedestrians 30 --spawn-radius 95 --seed 21 --include-pedestrians
echo "[pilot] collector exit=$?"
