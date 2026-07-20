#!/usr/bin/env bash
# FULL zlib latency sweep: all 36 AE x quant x ROI action profiles the agent will act on, deployed codec
# zlib, ideal 8 MB loopback. Replaces the 34 interpolated (~) latencies with measured values. Reuses an
# existing CARLA; writes loopback_latency_zlib.json (36 profiles, entropy-keyed).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
LOG="$AB/rl_agent/IDEAL_LOOPBACK_ZLIB_FULL_LOG.md"
LB="$AB/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib_full"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
carla_up(){ "$PY" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }
start_carla(){
  for a in 1 2 3; do
    log "CARLA launch attempt $a"
    ( cd "$CARLA_ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_zlib_full.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { sleep 12; carla_up && { log "CARLA up"; return 0; }; }; sleep 4; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done
  return 1
}
log "===== FULL ZLIB LATENCY SWEEP START (36 profiles, rmem=8MB) ====="
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM" -lt 8000000 ]; then log "  FATAL rmem_max=$RMEM < 8MB — raise buffers first"; exit 3; fi
log "rmem_max=$RMEM OK"
CARLA_WAS_UP=0
if carla_up; then CARLA_WAS_UP=1; log "reusing existing CARLA (will NOT kill it)"; fi
if [ "$CARLA_WAS_UP" = "1" ] || start_carla; then
  "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_zlib_full.json >> "$LOG" 2>&1 || log "  WARN sweep rc=$?"
  if [ "$CARLA_WAS_UP" = "0" ]; then pkill -9 -f CarlaUnreal 2>/dev/null; sleep 5; fi
  "$PY" rl_agent/agg_loopback.py "$LB" rl_agent/LOOPBACK_LATENCY_ZLIB.md rl_agent/loopback_latency_zlib.json >> "$LOG" 2>&1 || log "  WARN agg rc=$?"
  N=$("$PY" -c "import json;print(len(json.load(open('rl_agent/loopback_latency_zlib.json'))))" 2>/dev/null || echo 0)
  log "aggregated $N zlib profiles -> loopback_latency_zlib.json"
else
  log "  WARN CARLA did not start -> no full sweep"
fi
log "===== FULL ZLIB LATENCY SWEEP END ====="
echo "ZLIB_FULL_DONE" >> "$LOG"
