#!/usr/bin/env bash
# Deploy the final fusion model and capture seg + 2D-box snapshots at 10/20/30 m.
# Run this once CARLA is back up (it launches CARLA itself if not running).
set -e
ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
AB="$ROOT/PythonAPI/neu_collab/abiodun"
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen

cd "$AB"
if ! python3 -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" 2>/dev/null; then
  echo "Starting CARLA..."
  ( cd "$ROOT" && ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_deploy.log 2>&1 & )
  for i in $(seq 1 40); do
    python3 -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world();print('UP')" 2>/dev/null | grep -q UP && break
    sleep 3
  done
fi
python3 -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(5);c.get_world()" 2>/dev/null \
  || { echo "CARLA still not reachable — restart it (machine-level) and retry."; exit 1; }

echo "Running snapshot capture (car+person at 10/20/30 m)..."
python3 cooperative_fusion/deploy_snapshots.py
echo "Snapshots: $AB/cooperative_fusion/deploy_snapshots/snapshot_{10,20,30}m.png"
