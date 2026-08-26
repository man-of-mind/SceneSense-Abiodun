#!/usr/bin/env bash
# Automatic phase chain for the LR-ASPP/CenterFusion hybrid noAE pilot.
#
#   C2  decode the warm start + the baseline recall ceiling, then the parity gate
#   D   six warm-started clean-q epochs
#   E   decode epoch 6 at score 0.20 and 0.02, then the early continuation gate
#   F   only if the gate passes: continue the SAME run to epoch 24, decode
#       epochs 10/14/18/22/24, then final selection
#
# Create-only: every decode tag and every checkpoint refuses to overwrite.
# Decodes run in three lanes because the production evaluator writes a fixed
# intermediate path under the experiment dir that would race if shared.
set -u

ABIODUN=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
PKG=$ABIODUN/pole_lraspp_multimodal_fusion
HERE=$PKG/object_head_pilot_v1/hybrid_centerfusion_v1
EXP=$(cat "$HERE/EXP_DIR.txt")
BASE_DIR=$ABIODUN/experiments/route_b_noae_precision_full_v1/20260825_195301
BASE_CKPT=$BASE_DIR/checkpoints/curriculum_stage2_joint_v1/epoch_013.pt
CFG=$PKG/object_head_pilot_v1/configs/route_b_noae_precision_pilot_v1.yaml
TRIAL_JSON=$HERE/configs/hybrid_centerfusion_v1.json
TRIAL=hybrid_centerfusion_v1
PY=/usr/bin/python3
LANES=3
CKDIR=$EXP/checkpoints/$TRIAL

cd "$PKG"
LOG=$EXP/chain.log
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

notify(){ # $1 = summary line
  command -v notify-send >/dev/null 2>&1 && notify-send "hybrid noAE pilot" "$1" || true
  log "NOTIFY: $1"
}

finish(){ # $1 = terminal verdict
  echo "$1" > "$EXP/TERMINAL_VERDICT.txt"
  notify "$1"
  log "CHAIN_DONE verdict=$1"
  exit 0
}

mklane(){ # $1 = lane index
  local lane="$EXP/lane$1"
  if [ ! -e "$lane" ]; then
    mkdir -p "$lane"
    ln -s "$EXP/dataset" "$lane/dataset"
    ln -s "$EXP/provenance" "$lane/provenance"
  fi
  echo "$lane"
}

decode(){ # $1=lane_dir $2=checkpoint $3=tag $4=score_threshold
  local lane=$1 ckpt=$2 tag=$3 score=$4
  [ -d "$EXP/eval/$tag" ] && { log "skip $tag (already decoded)"; return 0; }
  [ -f "$ckpt" ] || { log "MISSING checkpoint $ckpt"; return 1; }
  local t0=$(date +%s)
  "$PY" "$HERE/evaluate_hybrid_route_b_v1.py" \
      --experiment-dir "$lane" --checkpoint "$ckpt" --tag "$tag" --config "$CFG" \
      --split val --feature-drop-fraction 0.0 --object-score-threshold "$score" \
      > "$lane/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    mkdir -p "$EXP/eval"
    mv "$lane/eval/$tag" "$EXP/eval/$tag"
    log "decoded $tag in $(( $(date +%s) - t0 ))s"
  else
    log "DECODE_FAILED $tag rc=$rc"; tail -20 "$lane/${tag}.log" | tee -a "$LOG"
  fi
  return $rc
}

run_grid(){ # stdin: "checkpoint<TAB>tag<TAB>score" lines, distributed over lanes
  mapfile -t JOBS
  local i j lane
  for i in $(seq 0 $((LANES-1))); do
    lane=$(mklane "$i")
    (
      for j in $(seq "$i" "$LANES" $(( ${#JOBS[@]} - 1 )) ); do
        IFS=$'\t' read -r ck tag score <<< "${JOBS[$j]}"
        decode "$lane" "$ck" "$tag" "$score"
      done
    ) &
  done
  wait
}

train_to(){ # $1 = run_until_epoch
  local until=$1
  local trial="$EXP/trial_run_until_${until}.json"
  "$PY" - "$TRIAL_JSON" "$trial" "$until" <<'PYEOF'
import json, sys
src, dst, until = sys.argv[1], sys.argv[2], int(sys.argv[3])
trial = json.loads(open(src).read())
trial["run_until_epoch"] = until
json.dump(trial, open(dst, "w"), indent=2)
PYEOF
  "$PY" "$HERE/train_entry_v1.py" --config "$CFG" --trial-json "$trial" \
      --experiment-dir "$EXP" --training-budget-hours 0 \
      >> "$EXP/train_stdout.log" 2>&1
  return $?
}

# ---------------------------------------------------------------- Phase C2
log "phase C2: decode the warm start and the baseline recall ceiling"
printf '%s\t%s\t%s\n' \
  "$CKDIR/warm_start.pt" "warm_start_s020" "0.20" \
  "$CKDIR/warm_start.pt" "warm_start_s002" "0.02" \
  "$BASE_CKPT" "baseline_epoch013_s002" "0.02" | run_grid

log "phase C2: warm-start parity gate"
if ! "$PY" "$HERE/gate_and_select_v1.py" --experiment-dir "$EXP" --baseline-dir "$BASE_DIR" \
        --stage parity >> "$EXP/gate_parity_stdout.log" 2>&1; then
  finish WARM_START_PARITY_FAILED
fi
log "phase C2: parity PASS"

# ----------------------------------------------------------------- Phase D
log "phase D: six warm-started clean-q epochs"
if ! train_to 6; then log "training returned non-zero"; finish IMPLEMENTATION_BLOCKED; fi
[ -f "$CKDIR/epoch_006.pt" ] || { log "epoch_006.pt missing after phase D"; finish IMPLEMENTATION_BLOCKED; }

# ----------------------------------------------------------------- Phase E
log "phase E: decode epoch 6 at score 0.20 and 0.02"
printf '%s\t%s\t%s\n' \
  "$CKDIR/epoch_006.pt" "hybrid_ep006_s020" "0.20" \
  "$CKDIR/epoch_006.pt" "hybrid_ep006_s002" "0.02" | run_grid

log "phase E: early continuation gate"
if ! "$PY" "$HERE/gate_and_select_v1.py" --experiment-dir "$EXP" --baseline-dir "$BASE_DIR" \
        --stage early >> "$EXP/gate_early_stdout.log" 2>&1; then
  log "early continuation gate FAILED - stopping, no second architecture, no gate change"
  finish HYBRID_NOAE_PILOT_NO_GAIN
fi
log "phase E: early gate PASS"

# ----------------------------------------------------------------- Phase F
log "phase F: continue the same run to epoch 24"
if ! train_to 24; then log "continuation training returned non-zero"; finish IMPLEMENTATION_BLOCKED; fi

log "phase F: decode epochs 10 14 18 22 24"
{
  for ep in 10 14 18 22 24; do
    printf '%s\t%s\t%s\n' "$CKDIR/$(printf 'epoch_%03d.pt' $ep)" "$(printf 'hybrid_ep%03d_s020' $ep)" "0.20"
    printf '%s\t%s\t%s\n' "$CKDIR/$(printf 'epoch_%03d.pt' $ep)" "$(printf 'hybrid_ep%03d_s002' $ep)" "0.02"
  done
} | run_grid

log "phase F: final selection"
"$PY" "$HERE/gate_and_select_v1.py" --experiment-dir "$EXP" --baseline-dir "$BASE_DIR" \
    --stage final --epochs 6,10,14,18,22,24 >> "$EXP/gate_final_stdout.log" 2>&1
VERDICT=$("$PY" -c "import json;print(json.load(open('$EXP/gate_final_v1.json'))['verdict'])" 2>/dev/null || echo IMPLEMENTATION_BLOCKED)
finish "$VERDICT"
