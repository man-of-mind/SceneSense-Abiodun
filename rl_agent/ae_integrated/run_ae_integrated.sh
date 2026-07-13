#!/usr/bin/env bash
# AUTONOMOUS: train the 3 integrated-AE models (bottleneck 64 validated first, then 32 & 128), warm-started
# from M', AE end-to-end, ROI drop-aware, entropy=zlib. Each GATED on seg + localization + ped-recall>=0.80.
# AE-64 is the make-or-break: if localization does NOT recover, STOP (don't train 32/128 on a broken idea).
# Full-model trainings are heavy (~12GB) -> run SEQUENTIALLY on the single GPU (parallel would OOM).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
PARENT="$AB/experiments/ae_integrated_20260710"
D="$AB/rl_agent/ae_integrated"
LOG="$D/AE_INTEGRATED_LOG.md"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
mkdir -p "$PARENT" /tmp/matplotlib-cache
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
link_ds(){ local d="$1"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"; }

"$PY" "$D/gen_trials.py" >> "$LOG" 2>&1

train_and_gate(){          # $1=bottleneck ; echoes gate rc: 0=recovered,1=missed,2=no-checkpoint
  local BN="$1" EXP="$PARENT/ae$1"
  link_ds "$EXP"
  log "=== TRAIN ae${BN}_integrated (warm-start M', AE end-to-end, drop-aware) ==="
  "$PY" -m pole_lraspp_multimodal_fusion.train_fusion --config "$CFG" --experiment-dir "$EXP" \
    --trial-json "$(cat "$D/ae${BN}_integrated.json")" --training-budget-hours 3.0 >> "$EXP/train.log" 2>&1
  local BEST="$EXP/checkpoints/ae${BN}_integrated/best.pt"
  if [[ ! -f "$BEST" ]]; then log "ae${BN}: NO checkpoint produced"; return 2; fi
  local GD="$EXP/gate_eval"; link_ds "$GD"
  log "=== GATE-eval ae${BN} (clean; AE runs inside forward) ==="
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$GD" \
    --checkpoint "$BEST" --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
    --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda >> "$GD/eval.log" 2>&1
  "$PY" "$D/gate_ae_check.py" "$GD/metrics/test_fusion_evaluation_metrics.json" >> "$LOG" 2>&1
  return $?
}

log "===== INTEGRATED-AE AUTONOMOUS RUN START ====="
train_and_gate 64; rc=$?
if [[ $rc -eq 2 ]]; then log "ABORT: AE-64 produced no checkpoint (see train.log)."; echo "AE_INTEGRATED_DONE FAIL_NOCKPT" >> "$LOG"; exit 1; fi
if [[ $rc -ne 0 ]]; then
  log "AE-64 HYPOTHESIS FAILED: localization did not recover. Per the plan, NOT spending compute on 32/128 for a"
  log "  broken approach. This is the evidence that the integrated-AE idea also does not fix loc -> for review."
  echo "AE_INTEGRATED_DONE HYPOTHESIS_FAILED" >> "$LOG"; exit 0
fi
log "AE-64 RECOVERED localization -> proceeding to AE-32 and AE-128"
train_and_gate 32  || log "  ae32 gate: review (see gate output above)"
train_and_gate 128 || log "  ae128 gate: review (see gate output above)"

# ---- final summary of all trained models ----
log "--- SUMMARY (clean accuracy of integrated-AE models) ---"
"$PY" - >> "$LOG" 2>&1 <<PYEOF
import json, glob, os
for bn in (64,32,128):
    p=f"{os.environ['PWD'] if False else '$PARENT'}/ae{bn}/gate_eval/metrics/test_fusion_evaluation_metrics.json"
    try:
        m=json.load(open(p))
        print(f"  ae{bn}: mIoU {m['miou']:.3f} veh {m['vehicle_iou']:.3f} ped-rec {m['learned_person_object_recall']:.3f} "
              f"obj-rec {m['learned_object_recall']:.3f} loc {m['learned_global_xy_mae_m']:.2f}m")
    except Exception as e:
        print(f"  ae{bn}: (no metrics: {e})")
PYEOF
log "===== INTEGRATED-AE RUN END ====="
echo "AE_INTEGRATED_DONE OK" >> "$LOG"
