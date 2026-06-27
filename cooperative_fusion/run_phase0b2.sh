#!/usr/bin/env bash
set -uo pipefail
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
ABS="$CARLA_ROOT/PythonAPI/neu_collab/abiodun"
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export PYTHONPATH="$ABS/pole_lraspp_multimodal_fusion:$ABS:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
CLOG="$ABS/cooperative_fusion/phase0b/carla.log"
mkdir -p "$ABS/cooperative_fusion/phase0b"
cleanup(){ echo "[run] tearing down CARLA"; pkill -f CarlaUnreal 2>/dev/null; pkill -f "Carla.*Shipping" 2>/dev/null; sleep 4; }
trap cleanup EXIT
echo "[run] launching CARLA headless..."
( cd "$CARLA_ROOT" && ./CarlaUnreal.sh -RenderOffScreen > "$CLOG" 2>&1 ) &
for i in $(seq 1 90); do
  python3 -c "import socket;s=socket.socket();s.settimeout(1);import sys;sys.exit(0 if s.connect_ex(('127.0.0.1',2000))==0 else 1)" && break
  sleep 2
done
python3 -c "import socket;s=socket.socket();s.settimeout(1);import sys;sys.exit(0 if s.connect_ex(('127.0.0.1',2000))==0 else 1)" || { echo "[run] CARLA failed to start"; tail -15 "$CLOG"; exit 2; }
echo "[run] CARLA up; running scene-health check..."; sleep 6
cd "$ABS"
timeout 240 python3 cooperative_fusion/phase0b2_single_view_infer.py
echo "[run] done exit=$?"
