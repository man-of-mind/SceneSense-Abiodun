#!/usr/bin/env bash
# Crowded-flow probe: launch CARLA headless, run a short crowded collection with the
# deadlock-fix config + hard safety caps, measure (a) does the ego flow / loop complete,
# (b) is crowded data near-field-rich. Tears CARLA down on exit. Bounded run time.
set -uo pipefail

CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
ABS="$CARLA_ROOT/PythonAPI/neu_collab/abiodun"
VENV="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate"
SCRATCH="/tmp/claude-200171/-home-shr-aisvcs-workarea-carla-0-10-env-Carla-0-10-0-Linux-Shipping-PythonAPI-neu-collab/2c781335-b7e8-42c9-8114-5c12ec5681d6/scratchpad"
PROBE_ID="PROBE_crowded_flowfix"
PROBE_DIR="$ABS/fusion_training_data/$PROBE_ID"
CARLA_LOG="$SCRATCH/carla_probe.log"
COLLECT_LOG="$SCRATCH/probe_collect.log"

source "$VENV" 2>/dev/null
export PYTHONPATH="$ABS/pole_lraspp_multimodal_fusion:$ABS:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
mkdir -p "$SCRATCH"
rm -rf "$PROBE_DIR"

cleanup() {
  echo "[probe] tearing down CARLA..."
  pkill -f "CarlaUnreal" 2>/dev/null
  pkill -f "Carla.*Shipping" 2>/dev/null
  sleep 5
}
trap cleanup EXIT

port_open() { python3 -c "import socket;s=socket.socket();s.settimeout(1);import sys;sys.exit(0 if s.connect_ex(('127.0.0.1',2000))==0 else 1)"; }

echo "[probe] launching CARLA headless..."
( cd "$CARLA_ROOT" && ./CarlaUnreal.sh -RenderOffScreen > "$CARLA_LOG" 2>&1 ) &
echo "[probe] waiting for port 2000 (up to ~180s)..."
UP=0
for i in $(seq 1 90); do
  if port_open; then UP=1; break; fi
  sleep 2
done
if [[ "$UP" != "1" ]]; then
  echo "[probe] RESULT: CARLA failed to start within timeout -> NO-GO (lock in). CARLA log tail:"
  tail -15 "$CARLA_LOG" 2>/dev/null
  exit 2
fi
echo "[probe] CARLA up. Starting short crowded collection (hard caps)..."
sleep 5

cd "$ABS"
timeout 900 python3 carla_collect_moving_ego_fusion_training_data.py \
  --experiment-id "$PROBE_ID" --seed 51 --no-ego-freeze \
  --ego-autopilot-speed-difference-pct 25 --ego-follow-distance-m 6.0 \
  --no-ego-disable-lane-change \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop --ego-fixed-path-min-spacing-m 3.0 \
  --route-progress-every-s 2.0 \
  --loop-return-radius-m 2.0 --loop-min-distance-m 200 --loop-min-elapsed-s 30 \
  --stop-after-loops 1 --stop-on-stuck --stuck-ignore-traffic-light-waits \
  --stuck-speed-threshold-mps 0.20 --stuck-timeout-s 45 --stuck-min-elapsed-s 30 \
  --skip-save-when-stuck --max-wall-clock-s 600 \
  --max-samples 1500 --sample-stride 2 --warmup-ticks 30 --fps 10 \
  --camera-width 1280 --camera-height 720 --camera-fov 120 \
  --model-input-width 768 --model-input-height 432 \
  --ego-spawn-index 80 --ego-camera-z 1.55 --ego-camera-pitch -4.0 \
  --radar-range 120 --radar-points-per-second 5000 --radar-raster-radius-px 2 \
  --npc-vehicles 24 --npc-pedestrians 35 --npc-vehicle-speed-difference-pct 10 \
  --npc-pedestrian-max-speed-mps 0.9 --npc-pedestrian-cross-factor 0.5 \
  --spawn-radius 80 --gt-max-distance-m 140 --include-pedestrians \
  > "$COLLECT_LOG" 2>&1
CE=$?
echo "[probe] collector exit=$CE. tail:"
tail -8 "$COLLECT_LOG"

echo ""
echo "=== PROBE ANALYSIS ==="
python3 - "$PROBE_DIR" <<'PY'
import csv, math, os, sys, glob, json
d=sys.argv[1]
ob=os.path.join(d,"object_boxes.csv")
if not os.path.exists(ob):
    print("RESULT: no object_boxes.csv produced -> NO-GO (collection did not yield data)."); sys.exit(0)
near=tot=0; perframe={}
with open(ob) as f:
    for r in csv.DictReader(f):
        if r.get("gt_source")!="actor": continue
        try: dist=float(r.get("gt_distance_m") or 0.0)
        except: dist=0.0
        if dist<=0: continue
        tot+=1
        if dist<20: near+=1
        perframe[r.get("sample_id","")]=perframe.get(r.get("sample_id",""),0)+1
# loop completion from summary
summ=glob.glob(os.path.join(d,"*summary*.json"))+glob.glob(os.path.join(d,"**","*summary*.json"),recursive=True)
loops="?"
for s in summ:
    try:
        j=json.load(open(s));
        if "loop_count" in j: loops=j["loop_count"]; break
    except: pass
nframes=len([p for p in perframe if p])
print(f"objects(actor)={tot}  near<20m={near} ({100*near/max(1,tot):.1f}%)  frames_with_objs={nframes}  mean_obj/frame={tot/max(1,nframes):.1f}  loop_count={loops}")
print(f"(baseline near-field fraction was ~3.6%; GO if flow OK AND near% >= ~8-10% or mean_obj/frame notably higher)")
PY
echo "=== END PROBE ==="
