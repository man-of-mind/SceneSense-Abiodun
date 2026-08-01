#!/usr/bin/env bash
# MEASURED high-ROI (q=0.7/0.9/0.98) loopback latency sweep, zstd, ideal 8MB loopback. Fills the one
# gap the density run left DERIVED: front/back/transport ms above q=0.5. u4 x 4 AE x 3 high-ROI = 12
# profiles. Reuses an existing CARLA (never kills it); writes loopback_latency_zstd_highroi.json and
# merges it into loopback_latency_zstd.json (backing up the original first).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
LOG="$AB/rl_agent/IDEAL_LOOPBACK_ZSTD_HIGHROI_LOG.md"
LB="$AB/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_highroi"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
carla_up(){ "$PY" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }

log "===== HIGH-ROI ZSTD LATENCY SWEEP START (12 profiles, rmem=8MB) ====="
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM" -lt 8000000 ]; then log "  FATAL rmem_max=$RMEM < 8MB — raise buffers first"; exit 3; fi
log "rmem_max=$RMEM OK"
if ! carla_up; then log "  FATAL CARLA not up on :2000 — start it first (this script never launches/kills CARLA)"; exit 4; fi
log "reusing existing CARLA (will NOT kill it)"

"$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_zstd_highroi.json >> "$LOG" 2>&1 || log "  WARN sweep rc=$?"
"$PY" rl_agent/agg_loopback.py "$LB" rl_agent/LOOPBACK_LATENCY_ZSTD_HIGHROI.md \
      rl_agent/loopback_latency_zstd_highroi.json >> "$LOG" 2>&1 || log "  WARN agg rc=$?"
N=$("$PY" -c "import json;print(len(json.load(open('rl_agent/loopback_latency_zstd_highroi.json'))))" 2>/dev/null || echo 0)
log "aggregated $N high-ROI profiles -> loopback_latency_zstd_highroi.json"

# merge into the main json (backup first), so analyze_density_knob.py's transport fit uses MEASURED points
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json, shutil, pathlib
base = pathlib.Path("rl_agent/loopback_latency_zstd.json")
add  = pathlib.Path("rl_agent/loopback_latency_zstd_highroi.json")
shutil.copy(base, base.with_suffix(".json.bak_prehighroi"))
d = json.load(base.open()); a = json.load(add.open())
before = len(d); d.update(a)
json.dump(d, base.open("w"), indent=1)
print(f"[merge] loopback_latency_zstd.json {before} -> {len(d)} profiles (added {len(d)-before})")
PYEOF
log "merged into loopback_latency_zstd.json"
log "===== HIGH-ROI ZSTD LATENCY SWEEP END ====="
echo "ZSTD_HIGHROI_DONE" >> "$LOG"
