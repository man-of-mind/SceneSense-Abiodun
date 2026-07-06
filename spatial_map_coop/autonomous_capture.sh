#!/usr/bin/env bash
# Autonomous capture+render daemon (NO LLM — deterministic, cheap, safe to leave running ~2 days).
# Every INTERVAL: if the live spatial-map pipeline is up, record a short trace and render offline PNGs
# into autonomous_run/. If the pipeline is down, it just logs and waits. Falls back to nothing harmful.
#
# Launch (persists across your SSH session):
#   cd .../abiodun/spatial_map_coop
#   setsid nohup bash autonomous_capture.sh > autonomous_run/daemon.out 2>&1 &
# Stop:
#   pkill -f autonomous_capture.sh
set -uo pipefail
CD=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/spatial_map_coop
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export MPLCONFIGDIR=/tmp/matplotlib-cache
cd "$CD"
API=${API:-http://127.0.0.1:35011}
INTERVAL=${INTERVAL:-1800}   # seconds between checks (default 30 min)
DUR=${DUR:-90}               # seconds to record when the pipeline is up
mkdir -p recordings autonomous_run/figs
log(){ echo "[$(date '+%F %T')] $*" >> autonomous_run/PROGRESS.md; }

log "capture daemon started (interval=${INTERVAL}s, capture=${DUR}s, api=${API})"
while true; do
  if curl -s --max-time 4 "$API/healthz" | grep -q '"ok"'; then
    ts=$(date '+%Y%m%d_%H%M%S')
    out="recordings/auto_${ts}.jsonl"
    python3 record_trace.py --out "$out" --hz 5 --duration-s "$DUR" >/dev/null 2>&1 || true
    nlines=$(wc -l < "$out" 2>/dev/null || echo 0)
    streams=$(python3 -c "import json;r=[json.loads(l) for l in open('$out')];s=(r[-1]['snap'].get('active_streams') or []) if r else [];print(','.join(str(x.get('stream_id')) for x in s))" 2>/dev/null || echo "?")
    if [ "${nlines:-0}" -gt 0 ]; then
      python3 replay_trace.py --trace "$out" --last --outdir "autonomous_run/figs" >/dev/null 2>&1 || true
      # also a sampled contact sheet if the trace is long enough
      [ "${nlines:-0}" -gt 60 ] && python3 replay_trace.py --trace "$out" --every 30 --outdir "autonomous_run/figs" >/dev/null 2>&1 || true
      log "pipeline UP: captured ${nlines} snapshots (streams=[${streams}]) -> rendered PNGs in autonomous_run/figs"
    else
      log "pipeline UP but no snapshots captured (clients streaming? map server publishing?)"
    fi
    # keep disk tidy: retain only the 12 most recent traces
    ls -1t recordings/auto_*.jsonl 2>/dev/null | tail -n +13 | xargs -r rm -f
  else
    log "pipeline DOWN — waiting"
  fi
  sleep "$INTERVAL"
done
