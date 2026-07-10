#!/usr/bin/env bash
# Ideal-transport (8MB buffers) loopback re-run: clean latency for quant + ROI + AE actions on M',
# then aggregate + refresh COMPLETE_KNOB_MATRIX with MEASURED latency. Delivery ~100% by design.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
LOG="$AB/rl_agent/IDEAL_LOOPBACK_LOG.md"
LB="$AB/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
carla_up(){ "$PY" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }
start_carla(){
  for a in 1 2 3; do
    log "CARLA launch attempt $a"
    ( cd "$CARLA_ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_ideal.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { sleep 12; carla_up && { log "CARLA up"; return 0; }; }; sleep 4; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done
  return 1
}
log "===== IDEAL LOOPBACK re-run START (rmem=8MB) ====="
if start_carla; then
  "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_quantroi.json >> "$LOG" 2>&1 || log "  WARN quantroi rc=$?"
  "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_ae.json      >> "$LOG" 2>&1 || log "  WARN ae rc=$?"
  pkill -9 -f CarlaUnreal 2>/dev/null; sleep 5
  "$PY" rl_agent/agg_loopback.py "$LB" rl_agent/LOOPBACK_LATENCY.md rl_agent/loopback_latency.json >> "$LOG" 2>&1 || log "  WARN agg rc=$?"
  "$PY" rl_agent/build_knob_matrix.py "$AB/experiments/mprime_dropaware_20260708/sweeps" rl_agent/COMPLETE_KNOB_MATRIX.md rl_agent/loopback_latency.json 2835 >> "$LOG" 2>&1 || log "  WARN matrix rc=$?"
  log "matrix refreshed with measured ideal-transport latency"
else
  log "  WARN CARLA did not start -> matrix keeps interpolated latency"
  pkill -9 -f CarlaUnreal 2>/dev/null
fi
log "===== IDEAL LOOPBACK re-run END ====="
echo "IDEAL_LOOPBACK_DONE" >> "$LOG"
