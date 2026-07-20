#!/usr/bin/env bash
# ZLIB latency counterpart of run_ideal_loopback.sh. Measures the DEPLOYED-codec (zlib) latency
# column for quant + ROI + AE actions on M' under ideal transport (8MB buffers). Writes a SEPARATE
# json (loopback_latency_zlib.json) + table (LOOPBACK_LATENCY_ZLIB.md) so it cannot collide with the
# zstd aggregation. Matrix build (PERMODEL_KNOB_MATRIX_ZLIB.md) is a separate manual step.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
LOG="$AB/rl_agent/IDEAL_LOOPBACK_ZLIB_LOG.md"
LB="$AB/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zlib"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
carla_up(){ "$PY" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }
start_carla(){
  for a in 1 2 3; do
    log "CARLA launch attempt $a"
    ( cd "$CARLA_ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_ideal_zlib.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { sleep 12; carla_up && { log "CARLA up"; return 0; }; }; sleep 4; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done
  return 1
}
log "===== IDEAL LOOPBACK ZLIB re-run START (rmem=8MB, deployed codec) ====="
# fail-fast: buffers must be raised, else this measures a delivery cliff, not latency
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM" -lt 8000000 ]; then log "  FATAL rmem_max=$RMEM < 8MB — raise buffers first (sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608)"; exit 3; fi
log "rmem_max=$RMEM OK"
# Reuse an existing CARLA if one is already up (don't launch a duplicate or kill someone else's).
CARLA_WAS_UP=0
if carla_up; then CARLA_WAS_UP=1; log "reusing existing CARLA on :2000 (will NOT kill it on exit)"; fi
if [ "$CARLA_WAS_UP" = "1" ] || start_carla; then
  "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_quantroi_zlib.json >> "$LOG" 2>&1 || log "  WARN quantroi_zlib rc=$?"
  "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_ae_zlib.json      >> "$LOG" 2>&1 || log "  WARN ae_zlib rc=$?"
  if [ "$CARLA_WAS_UP" = "0" ]; then pkill -9 -f CarlaUnreal 2>/dev/null; sleep 5; fi
  "$PY" rl_agent/agg_loopback.py "$LB" rl_agent/LOOPBACK_LATENCY_ZLIB.md rl_agent/loopback_latency_zlib.json >> "$LOG" 2>&1 || log "  WARN agg rc=$?"
  log "zlib latency aggregated -> loopback_latency_zlib.json"
else
  log "  WARN CARLA did not start -> no zlib latency measured"
fi
log "===== IDEAL LOOPBACK ZLIB re-run END ====="
echo "IDEAL_LOOPBACK_ZLIB_DONE" >> "$LOG"
