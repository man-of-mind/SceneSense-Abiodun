#!/usr/bin/env bash
# Split-inference deployment measurement for the 5 pps models (loopback role): 2 crowded loops each,
# recording front_ms, back_ms, round_trip_ms, transport estimate, and feature payload (compressed +
# uncompressed). Same seed/route so scenes match across models. Aggregates -> PPS_DEPLOY_RESULTS.md.
#
# Real-5G RTT/transport is NOT measured here (needs the OAI stack up: gNB/core/UE + a `--role back`
# receiver). Loopback gives payload (the pps-dependent driver) + front/back compute + a transport
# estimate; the 5G RTT for each payload is a follow-on over the live link.
set -uo pipefail
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
cd "$AB"; source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:${PYTHONPATH:-}"; export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
CLIENT=carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py
FRAMES="${FRAMES:-400}"; SEED=31   # ~2 loops of the 5-wp route; 1300 was a 6x over-estimate
RESULT_TIMEOUT="${RESULT_TIMEOUT:-0.15}"  # big (~1MB) feature payloads fragment over UDP -> most loopback
                                          # results drop; a short wait stops each dropped frame stalling 0.6s.
                                          # Arriving results (RTT~44ms) still land; payload+front_ms are per-frame.
declare -A CKPT=(
 [100000]="$AB/experiments/autonomous_arch_runs_20260625/det_stage2c_centerw4/checkpoints/det_stage2c_centerw4/best.pt"
 [150000]="$AB/experiments/autonomous_arch_runs_20260625/det_pps150000_v2/checkpoints/det_pps150000_v2/best.pt"
 [200000]="$AB/experiments/autonomous_arch_runs_20260625/det_pps200000_v2/checkpoints/det_pps200000_v2/best.pt"
 [250000]="$AB/experiments/autonomous_arch_runs_20260625/det_pps250000_v2/checkpoints/det_pps250000_v2/best.pt"
 [300000]="$AB/experiments/autonomous_arch_runs_20260625/det_pps300000_v2/checkpoints/det_pps300000_v2/best.pt")
say(){ echo "[$(date '+%F %T')] $*"; }
carla_up(){ python3 -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" 2>/dev/null; }
ensure_carla(){
  carla_up && return 0
  for a in 1 2 3 4 5; do
    say "launch CARLA attempt $a"; ( cd "$ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_deploy.log 2>&1 & )
    for i in $(seq 1 20); do carla_up && { sleep 12; carla_up && { say "CARLA up+stable"; return 0; }; }; sleep 3; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done; say "ERROR CARLA won't start"; return 1
}
ensure_carla || { say "aborting (no CARLA)"; exit 1; }
for pps in 100000 150000 200000 250000 300000; do
  ck="${CKPT[$pps]}"; [[ -f "$ck" ]] || { say "SKIP pps=$pps (no ckpt)"; continue; }
  rd="$AB/metrics_logs/pps_deploy/pps${pps}"; rm -rf "$rd"; mkdir -p "$rd"
  say "DEPLOY loopback pps=$pps (2 crowded loops)"
  python3 "$CLIENT" --role loopback --headless --fusion-checkpoint "$ck" --host 127.0.0.1 --seed "$SEED" \
    --npc-vehicles 28 --npc-pedestrians 35 --spawn-radius 80 \
    --ego-fixed-path-spawn-indices "80,85,91,94,99,80" --ego-fixed-path-loop \
    --result-timeout "$RESULT_TIMEOUT" \
    --max-frames "$FRAMES" --transport-label "pps${pps}" --metrics-run-dir "$rd" \
    > "$AB/logs/deploy_pps${pps}.log" 2>&1 && say "deploy pps=$pps OK" || say "WARN deploy pps=$pps failed"
done
pkill -f CarlaUnreal 2>/dev/null; sleep 3
# ---- aggregate ----
python3 - <<'PY'
import glob,csv,math,statistics as st
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
def num(x):
    try:
        v=float(x); return v if math.isfinite(v) else None  # drop 'nan'/'inf' (dropped-result rows)
    except: return None
rows=[]
for pps in (100000,150000,200000,250000,300000):
    fs=glob.glob(f"{AB}/metrics_logs/pps_deploy/pps{pps}/**/*_metrics.csv",recursive=True)+glob.glob(f"{AB}/metrics_logs/pps_deploy/pps{pps}/*_metrics.csv")
    if not fs: continue
    R=list(csv.DictReader(open(fs[0])))
    def col(k): return [v for v in (num(r.get(k)) for r in R) if v is not None]
    def m(k):
        v=col(k); return (st.mean(v) if v else 0)
    def p95(k):
        v=sorted(col(k)); return (v[int(0.95*len(v))-1] if v else 0)
    n_res=sum(1 for r in R if str(r.get('result_received')).lower()=='true')
    rows.append((pps,len(R),n_res,m('front_ms'),m('back_ms'),m('round_trip_ms'),m('transport_round_trip_ms_estimate'),
                 m('feature_payload_bytes'),m('feature_payload_bytes_uncompressed'),p95('round_trip_ms')))
out=f"{AB}/PPS_DEPLOY_RESULTS.md"
with open(out,'w') as f:
    f.write("# PPS split-inference deployment (loopback, 2 crowded loops, seed 31)\n\n")
    f.write("| pps | frames | results_n | front_ms | back_ms | RTT_ms(loopback) | transport_est_ms | payload_KB(comp) | payload_KB(uncomp) | RTT_p95_ms |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for (pps,n,nr,fr,bk,rtt,tr,pc,pu,p95v) in rows:
        f.write(f"| {pps} | {n} | {nr} | {fr:.1f} | {bk:.1f} | {rtt:.1f} | {tr:.1f} | {pc/1024:.1f} | {pu/1024:.1f} | {p95v:.1f} |\n")
print(open(out).read())
PY
echo "DEPLOY_MEASUREMENT_COMPLETE"
