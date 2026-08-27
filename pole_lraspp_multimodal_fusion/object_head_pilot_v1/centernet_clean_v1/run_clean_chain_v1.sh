#!/usr/bin/env bash
# One automatic, bounded clean Route B CenterNet chain.
set -uo pipefail

ABIODUN=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
PKG=$ABIODUN/pole_lraspp_multimodal_fusion
HERE=$PKG/object_head_pilot_v1/centernet_clean_v1
SOURCE_VIEW=$ABIODUN/experiments/route_b_noae_precision_full_v1/20260825_195301
CONFIG=$HERE/configs/route_b_centernet_clean_v1.yaml
TRIAL_SOURCE=$HERE/configs/resnet34_fpn_centerfusion_v1.json
PY=/usr/bin/python3
EXP=${1:?usage: run_clean_chain_v1.sh EXPERIMENT_DIR}
TRIAL_NAME=resnet34_fpn_centerfusion_v1
CKDIR=$EXP/checkpoints/$TRIAL_NAME
LOG=$EXP/chain.log

mkdir -p "$EXP" "$EXP/decision" "$EXP/eval" "$EXP/metrics" "$EXP/figures"
date +%s > "$EXP/chain_started_unix.txt"

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"
}

notify() {
  echo "$1" > "$EXP/NOTIFICATION.txt"
  command -v notify-send >/dev/null 2>&1 && notify-send "Route B clean CenterNet" "$1" || true
  log "desktop notification: $1"
}

blocked() {
  echo "$1" > "$EXP/BLOCKED_REASON.txt"
  echo IMPLEMENTATION_BLOCKED > "$EXP/TERMINAL_VERDICT.txt"
  write_report
  notify IMPLEMENTATION_BLOCKED
  log "CHAIN_DONE verdict=IMPLEMENTATION_BLOCKED reason=$1"
  exit 0
}

write_report() {
  "$PY" "$HERE/report_v1.py" \
    --experiment-dir "$EXP" \
    --output-json "$EXP/FINAL_REPORT.json" \
    --output-md "$EXP/FINAL_REPORT.md" >> "$LOG" 2>&1 || log "final report generation failed"
}

finish() {
  local verdict=$1
  echo "$verdict" > "$EXP/TERMINAL_VERDICT.txt"
  write_report
  notify "$verdict"
  log "CHAIN_DONE verdict=$verdict"
  exit 0
}

if [ -e "$EXP/dataset" ] || [ -e "$EXP/provenance" ]; then
  blocked "experiment dataset/provenance path already exists; create-only run refused"
fi
ln -s "$SOURCE_VIEW/dataset" "$EXP/dataset"
ln -s "$SOURCE_VIEW/provenance" "$EXP/provenance"

log "phase 1a: py_compile and config parsing"
if ! "$PY" -m py_compile "$HERE"/*.py; then
  blocked "py_compile failed"
fi
if ! "$PY" -c "import json,yaml; json.load(open('$TRIAL_SOURCE')); yaml.safe_load(open('$CONFIG')); print('config parse PASS')" >> "$LOG" 2>&1; then
  blocked "YAML/JSON parsing failed"
fi

log "phase 1b: real q=0 AMP launch batch, trying batch 24"
if "$PY" "$HERE/launch_check_v1.py" \
    --config "$CONFIG" --trial-json "$TRIAL_SOURCE" --experiment-dir "$EXP" \
    --batch-size 24 --output "$EXP/launch_gate.json" > "$EXP/launch_batch24.log" 2>&1; then
  BATCH=24
  log "launch gate PASS at batch 24"
else
  if rg -qi "out of memory|cuda error: memory|CUBLAS_STATUS_ALLOC_FAILED" "$EXP/launch_batch24.log"; then
    log "batch 24 did not fit; retaining concise log and trying fixed fallback batch 16"
    if ! "$PY" "$HERE/launch_check_v1.py" \
        --config "$CONFIG" --trial-json "$TRIAL_SOURCE" --experiment-dir "$EXP" \
        --batch-size 16 --output "$EXP/launch_gate.json" > "$EXP/launch_batch16.log" 2>&1; then
      blocked "batch 16 launch gate failed"
    fi
    BATCH=16
    log "launch gate PASS at batch 16"
  else
    blocked "batch 24 launch gate failed for a non-memory reason"
  fi
fi

TRIAL4=$EXP/resolved_trial_to_epoch4.json
if ! "$PY" "$HERE/make_trial_v1.py" --source "$TRIAL_SOURCE" --output "$TRIAL4" \
    --batch-size "$BATCH" --run-until-epoch 4; then
  blocked "could not create the epoch-4 resolved trial"
fi

log "phase 2: train the four-epoch clean pilot"
if ! "$PY" "$HERE/train_entry_v1.py" --config "$CONFIG" --trial-json "$TRIAL4" \
    --experiment-dir "$EXP" --training-budget-hours 0 >> "$EXP/train.log" 2>&1; then
  blocked "four-epoch training failed"
fi
if [ ! -f "$CKDIR/epoch_004.pt" ]; then
  blocked "four-epoch checkpoint is missing"
fi

make_lane() {
  local lane=$1
  mkdir -p "$lane"
  [ -e "$lane/dataset" ] || ln -s "$EXP/dataset" "$lane/dataset"
  [ -e "$lane/provenance" ] || ln -s "$EXP/provenance" "$lane/provenance"
}

decode_one() {
  local lane=$1 checkpoint=$2 tag=$3 score=$4
  "$PY" "$HERE/evaluate_checkpoint_v1.py" \
    --experiment-dir "$lane" --checkpoint "$checkpoint" --tag "$tag" \
    --config "$CONFIG" --split val --object-score-threshold "$score" \
    > "$lane/$tag.log" 2>&1
}

decode_epoch() {
  local epoch=$1
  local checkpoint
  checkpoint=$(printf '%s/epoch_%03d.pt' "$CKDIR" "$epoch")
  local lane20=$EXP/lane_s020 lane02=$EXP/lane_s002
  local tag20 tag02
  tag20=$(printf 'centernet_ep%03d_s020' "$epoch")
  tag02=$(printf 'centernet_ep%03d_s002' "$epoch")
  make_lane "$lane20"
  make_lane "$lane02"
  decode_one "$lane20" "$checkpoint" "$tag20" 0.20 &
  local pid20=$!
  decode_one "$lane02" "$checkpoint" "$tag02" 0.02 &
  local pid02=$!
  local rc=0
  wait "$pid20" || rc=1
  wait "$pid02" || rc=1
  if [ "$rc" -ne 0 ]; then
    return 1
  fi
  if [ -e "$EXP/eval/$tag20" ] || [ -e "$EXP/eval/$tag02" ]; then
    return 1
  fi
  mv "$lane20/eval/$tag20" "$EXP/eval/$tag20"
  mv "$lane02/eval/$tag02" "$EXP/eval/$tag02"
  log "decoded epoch $epoch at fixed scores 0.20 and 0.02"
}

log "phase 2: fixed validation decode of epoch 4"
if ! decode_epoch 4; then
  blocked "epoch-4 fixed validation decode failed"
fi
if ! "$PY" "$HERE/gate_and_select_v1.py" --experiment-dir "$EXP" --stage pilot \
    --output "$EXP/decision/pilot_gate_v1.json" >> "$LOG" 2>&1; then
  finish CENTERNET_BASE_PILOT_FAILED
fi
log "four-epoch continuation gate PASS"

TRIAL24=$EXP/resolved_trial_to_epoch24.json
if ! "$PY" "$HERE/make_trial_v1.py" --source "$TRIAL_SOURCE" --output "$TRIAL24" \
    --batch-size "$BATCH" --run-until-epoch 24; then
  blocked "could not create the epoch-24 resolved trial"
fi

log "phase 3: continue the same run to 24 total epochs"
if ! "$PY" "$HERE/train_entry_v1.py" --config "$CONFIG" --trial-json "$TRIAL24" \
    --experiment-dir "$EXP" --training-budget-hours 0 >> "$EXP/train.log" 2>&1; then
  blocked "continuation training failed"
fi

for epoch in 8 12 16 20 24; do
  log "phase 3: fixed validation decode of epoch $epoch"
  if ! decode_epoch "$epoch"; then
    blocked "epoch-$epoch fixed validation decode failed"
  fi
done

if ! "$PY" "$HERE/gate_and_select_v1.py" --experiment-dir "$EXP" --stage final \
    --epochs 4,8,12,16,20,24 --output "$EXP/decision/final_selection_v1.json" >> "$LOG" 2>&1; then
  blocked "final clean checkpoint selection failed"
fi
VERDICT=$($PY -c "import json; print(json.load(open('$EXP/decision/final_selection_v1.json'))['verdict'])")
finish "$VERDICT"
