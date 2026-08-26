#!/usr/bin/env bash
# Post-training driver for focused_noae_v1. Create-only; skips existing eval tags.
# Runs decodes in N lanes, each with its OWN experiment dir, because the evaluator
# writes a fixed intermediate path (metrics/val_*.json) that would race if two
# decodes shared one dir. Results are collected back into the run's eval/.
set -u
HERE=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion/object_head_pilot_v1
ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/focused_noae_v1
EXP=$ROOT/20260826_hybridq
BASE=$ROOT/baseline_epoch13
SRC=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_noae_precision_full_v1/20260825_195301
CFG=$HERE/configs/route_b_noae_precision_pilot_v1.yaml
TRIAL=focused_noae_v1
LANES=3
cd "$HERE"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

mklane(){ # $1 = lane index
  L=$EXP/lane$1
  if [ ! -e "$L" ]; then
    mkdir -p "$L"
    ln -s "$SRC/dataset" "$L/dataset"
    ln -s "$SRC/provenance" "$L/provenance"
  fi
  echo "$L"
}

decode(){ # $1=lane_dir $2=epoch $3=q
  L=$1; EP=$2; Q=$3
  TAG=$(printf "focused_ep%03d_q%03d" "$EP" "$(python3 -c "print(int(round($Q*100)))")")
  [ -d "$EXP/eval/$TAG" ] && { echo "skip $TAG"; return 0; }
  CK=$EXP/checkpoints/$TRIAL/$(printf "epoch_%03d.pt" "$EP")
  [ -f "$CK" ] || { echo "MISSING $CK"; return 1; }
  T0=$(date +%s)
  python3 evaluate_route_b_checkpoint_v1.py --experiment-dir "$L" --checkpoint "$CK" \
      --tag "$TAG" --config "$CFG" --split val --feature-drop-fraction "$Q" \
      > "$L/${TAG}.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    mkdir -p "$EXP/eval"
    mv "$L/eval/$TAG" "$EXP/eval/$TAG"
    echo "ok $TAG $(( $(date +%s) - T0 ))s"
  else
    echo "FAIL $TAG rc=$RC"; tail -5 "$L/${TAG}.log"
  fi
  return $RC
}
export -f decode log
export EXP CFG TRIAL

run_grid(){ # stdin: "epoch q" pairs -> distributes across lanes
  mapfile -t JOBS
  for i in $(seq 0 $((LANES-1))); do
    L=$(mklane "$i")
    ( for j in $(seq $i $LANES $(( ${#JOBS[@]} - 1 )) ); do
        set -- ${JOBS[$j]}
        decode "$L" "$1" "$2"
      done ) &
  done
  wait
}

# ---------- Phase 1: wait for training ----------
log "waiting for training to finish"
while pgrep -f "run_object_head_pilot_v1.py .*focused_noae_v1.json" > /dev/null; do sleep 60; done
log "training finished; epochs on disk: $(ls "$EXP/checkpoints/$TRIAL"/epoch_*.pt 2>/dev/null | wc -l)"

# ---------- Phase 2: clean q=0.00 decode of every epoch ----------
LAST=$(( $(ls "$EXP/checkpoints/$TRIAL"/epoch_*.pt 2>/dev/null | wc -l) - 1 ))
log "phase 2: clean decode of epochs 0..$LAST (no epochs excluded)"
for ep in $(seq 0 "$LAST"); do echo "$ep 0.0"; done | run_grid
log "phase 2 done"

# ---------- Phase 3: shortlist on decoded clean metrics ----------
log "phase 3: shortlist"
python3 focused_noae_select_v1.py --experiment-dir "$EXP" --baseline-dir "$BASE" \
    --stage shortlist --shortlist-size 3 > "$EXP/shortlist_stdout.log" 2>&1
SHORT=$(python3 -c "
import json;print(' '.join(str(e) for e in json.load(open('$EXP/focused_noae_selection_shortlist.json'))['shortlist_epochs']))")
log "shortlist epochs: $SHORT"

# ---------- Phase 4: shortlist across 6 anchors + 5 midpoints ----------
log "phase 4: shortlist at 6 anchors + 5 interval midpoints"
for ep in $SHORT; do
  for q in 0.30 0.50 0.70 0.90 0.98 0.15 0.40 0.60 0.80 0.94; do echo "$ep $q"; done
done | run_grid
log "phase 4 done"

# ---------- Phase 5: final selection ----------
log "phase 5: final selection"
python3 focused_noae_select_v1.py --experiment-dir "$EXP" --baseline-dir "$BASE" \
    --stage final > "$EXP/final_selection_stdout.log" 2>&1
log "DRIVER_DONE verdict=$(python3 -c "
import json;print(json.load(open('$EXP/focused_noae_selection_final.json'))['verdict'])" 2>/dev/null || echo UNKNOWN)"
