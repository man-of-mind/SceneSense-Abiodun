#!/usr/bin/env bash
# AE64 -> AE32 -> AE128, one family trained at a time, cost-bounded evaluation
# per REGISTERED_GATE_AE_FAMILIES.md. Create-only; skips existing eval tags.
set -u
HERE=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion/object_head_pilot_v1
ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_ae_adapt_v2
SRC=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_noae_precision_full_v1/20260825_195301
AEINT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/ae_integrated_20260710
CFG=$HERE/configs/route_b_noae_precision_pilot_v1.yaml
LANES=4
SPARSE="5 9 13 17 20 23"
cd "$HERE"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

mklane(){ L=$1/lane$2; [ -e "$L" ] || { mkdir -p "$L"; ln -s "$SRC/dataset" "$L/dataset"; ln -s "$SRC/provenance" "$L/provenance"; }; echo "$L"; }

# decode <collect_dir> <lane_dir> <tag> <checkpoint> <q>
decode(){
  CD=$1; L=$2; TAG=$3; CK=$4; Q=$5
  [ -d "$CD/eval/$TAG" ] && { echo "skip $TAG"; return 0; }
  [ -f "$CK" ] || { echo "MISSING_CKPT $TAG $CK"; return 1; }
  T0=$(date +%s)
  python3 evaluate_route_b_checkpoint_v1.py --experiment-dir "$L" --checkpoint "$CK" \
      --tag "$TAG" --config "$CFG" --split val --feature-drop-fraction "$Q" > "$L/${TAG}.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then mkdir -p "$CD/eval"; mv "$L/eval/$TAG" "$CD/eval/$TAG"; echo "ok $TAG $(( $(date +%s) - T0 ))s"
  else echo "FAIL $TAG rc=$RC"; tail -4 "$L/${TAG}.log"; fi
  return $RC
}
export -f decode
export CFG

# run_grid <collect_dir> <lane_root>   (stdin: "tag ckpt q" lines)
run_grid(){
  CD=$1; LR=$2; mapfile -t JOBS
  [ ${#JOBS[@]} -eq 0 ] && return 0
  for i in $(seq 0 $((LANES-1))); do
    L=$(mklane "$LR" "$i")
    ( for j in $(seq $i $LANES $(( ${#JOBS[@]} - 1 )) ); do
        set -- ${JOBS[$j]}; decode "$CD" "$L" "$1" "$2" "$3"
      done ) &
  done
  wait
}

qtag(){ python3 -c "print('q%03d'%round($1*100))"; }

# ---------- Phase 0: family baselines at six anchors ----------
log "PHASE 0: family baselines at six anchors"
{ for f in ae64 ae32 ae128; do
    for q in 0.0 0.3 0.5 0.7 0.9 0.98; do
      echo "base_${f}_$(qtag $q) $AEINT/$f/checkpoints/${f}_integrated/best.pt $q"
    done
  done; } | run_grid "$ROOT/baselines" "$ROOT/baselines"
log "PHASE 0 done: $(ls "$ROOT/baselines/eval" 2>/dev/null | wc -l) baseline cells"

# ---------- Per family ----------
for FAM in ae64 ae32 ae128; do
  E=$ROOT/$FAM
  TRIAL=${FAM}_adapt_v2
  log "=== FAMILY $FAM: training ==="
  if [ ! -f "$E/checkpoints/$TRIAL/epoch_023.pt" ]; then
    python3 run_object_head_pilot_v1.py --config "$CFG" \
      --trial-json "$HERE/configs/${TRIAL}.json" --experiment-dir "$E" \
      --training-budget-hours 2.0 > "$E/train_stdout.log" 2>&1
    log "$FAM training rc=$? epochs=$(ls "$E/checkpoints/$TRIAL"/epoch_*.pt 2>/dev/null | wc -l)"
  else
    log "$FAM training already complete, skipping"
  fi

  log "$FAM: sparse decode {$SPARSE} at q=0.00 and q=0.98"
  { for ep in $SPARSE; do
      CK=$E/checkpoints/$TRIAL/$(printf "epoch_%03d.pt" $ep)
      for q in 0.0 0.98; do echo "$(printf "%s_ep%03d" "$FAM" $ep)_$(qtag $q) $CK $q"; done
    done; } | run_grid "$E" "$E"

  python3 ae_family_select_v1.py --family "$FAM" --experiment-dir "$E" \
     --baseline-dir "$ROOT/baselines" --baseline-prefix "base_${FAM}_" \
     --prefix "${FAM}_ep" --stage shortlist > "$E/shortlist_stdout.log" 2>&1
  SHORT=$(python3 -c "import json;print(' '.join(str(x) for x in json.load(open('$E/${FAM}_shortlist.json'))['shortlist_epochs']))")
  log "$FAM shortlist: $SHORT"

  log "$FAM: shortlist at remaining four anchors"
  { for ep in $SHORT; do
      CK=$E/checkpoints/$TRIAL/$(printf "epoch_%03d.pt" $ep)
      for q in 0.3 0.5 0.7 0.9; do echo "$(printf "%s_ep%03d" "$FAM" $ep)_$(qtag $q) $CK $q"; done
    done; } | run_grid "$E" "$E"

  python3 ae_family_select_v1.py --family "$FAM" --experiment-dir "$E" \
     --baseline-dir "$ROOT/baselines" --baseline-prefix "base_${FAM}_" \
     --prefix "${FAM}_ep" --stage final > "$E/final_stdout.log" 2>&1
  V=$(python3 -c "import json;print(json.load(open('$E/${FAM}_final.json'))['verdict'])")
  log "$FAM VERDICT=$V"

  if [ "$V" = "PASS" ]; then
    SEP=$(python3 -c "import json;print(json.load(open('$E/${FAM}_final.json'))['selected']['epoch'])")
    log "$FAM: midpoints for selected epoch $SEP (+ matched baseline midpoints)"
    CK=$E/checkpoints/$TRIAL/$(printf "epoch_%03d.pt" $SEP)
    { for q in 0.15 0.40 0.60 0.80 0.94; do echo "$(printf "%s_ep%03d" "$FAM" $SEP)_$(qtag $q) $CK $q"; done; } | run_grid "$E" "$E"
    { for q in 0.15 0.40 0.60 0.80 0.94; do
        echo "base_${FAM}_$(qtag $q) $AEINT/$FAM/checkpoints/${FAM}_integrated/best.pt $q"; done; } | run_grid "$ROOT/baselines" "$ROOT/baselines"
    python3 ae_family_select_v1.py --family "$FAM" --experiment-dir "$E" \
       --baseline-dir "$ROOT/baselines" --baseline-prefix "base_${FAM}_" \
       --prefix "${FAM}_ep" --stage final > "$E/final_stdout.log" 2>&1
  fi
  log "=== FAMILY $FAM complete: $V ==="
done
log "AE_FAMILIES_DRIVER_DONE"
