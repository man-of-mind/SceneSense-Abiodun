#!/usr/bin/env bash
# AUTONOMOUS: AE-from-phase-1 (from-scratch) ablation. Rebuilds the full M' pipeline
# (stage1 seg -> stage2 obj -> phase3 joint) with the AE integrated from phase 1, for a
# given bottleneck. AE carried across phases via extract_ae.py. AE-32 runs FIRST and is
# GATED against the current warm-started AE-32: only if CLEARLY BETTER do we spend compute
# on AE-64 and AE-128. Sequential (heavy trainings, single GPU).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
PARENT="$AB/experiments/ae_integrated_fs_20260713"
WARM="$AB/experiments/ae_integrated_20260710"     # current warm-started AE models (comparison)
AEI="$AB/rl_agent/ae_integrated"
LOG="$AEI/FS_AE_LOG.md"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
mkdir -p "$PARENT" /tmp/matplotlib-cache
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
link_ds(){ local d="$1"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"; }

# train one phase; skips if its checkpoint already exists (idempotent restart)
train_phase(){          # $1=trial_name  $2=experiment_dir  $3=budget_hours
  local trial="$1" expdir="$2" hours="$3"
  local ck="$expdir/checkpoints/$trial/best.pt"
  if [[ -f "$ck" ]]; then log "  skip $trial (checkpoint exists)"; return 0; fi
  link_ds "$expdir"
  log "  TRAIN $trial (budget ${hours}h)"
  "$PY" -m pole_lraspp_multimodal_fusion.train_fusion --config "$CFG" --experiment-dir "$expdir" \
    --trial-json "$(cat "$AEI/${trial}.json")" --training-budget-hours "$hours" >> "$expdir/train.log" 2>&1
  [[ -f "$ck" ]] || { log "  ERROR $trial produced no checkpoint"; return 2; }
  return 0
}

# full stage1->2->3 build for one bottleneck; echoes phase3 gate-eval metrics path via GATE_JSON
run_bottleneck(){       # $1=bottleneck
  local bn="$1" base="$PARENT/ae$1"
  local s1="$base/stage1" s2="$base/stage2" p3="$base/phase3"
  local s1ck="$s1/checkpoints/fs_stage1_ae${bn}/best.pt"
  local s2ck="$s2/checkpoints/fs_stage2_ae${bn}/best.pt"
  local p3ck="$p3/checkpoints/fs_phase3_ae${bn}/best.pt"
  log "===== AE-${bn} FROM-PHASE-1 BUILD ====="
  "$PY" "$AEI/gen_fs_trials.py" "$bn" "$PARENT" >> "$LOG" 2>&1

  train_phase "fs_stage1_ae${bn}" "$s1" 3.0 || return 2
  "$PY" "$AEI/extract_ae.py" "$s1ck" "$s1/ae_extracted.pt" >> "$LOG" 2>&1 || { log "  ERROR extract stage1 AE"; return 2; }

  train_phase "fs_stage2_ae${bn}" "$s2" 3.0 || return 2
  "$PY" "$AEI/extract_ae.py" "$s2ck" "$s2/ae_extracted.pt" >> "$LOG" 2>&1 || { log "  ERROR extract stage2 AE"; return 2; }

  train_phase "fs_phase3_ae${bn}" "$p3" 3.0 || return 2

  local gd="$p3/gate_eval"; link_ds "$gd"
  log "  GATE-eval fs_phase3_ae${bn} (clean; AE in forward)"
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$gd" \
    --checkpoint "$p3ck" --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
    --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda >> "$gd/eval.log" 2>&1
  GATE_JSON="$gd/metrics/test_fusion_evaluation_metrics.json"
  "$PY" -c "import json;m=json.load(open('$GATE_JSON'));print(f\"  fs_ae${bn}: mIoU {m['miou']:.3f} veh {m['vehicle_iou']:.3f} ped-rec {m['learned_person_object_recall']:.3f} obj-rec {m['learned_object_recall']:.3f} loc {m['learned_global_xy_mae_m']:.2f}m\")" >> "$LOG" 2>&1 || true
  return 0
}

log "===== AE-FROM-PHASE-1 AUTONOMOUS RUN START ====="
run_bottleneck 32; rc=$?
if [[ $rc -ne 0 ]]; then log "AE-32 from-phase-1 build FAILED (rc=$rc) -- see logs."; echo "FS_AE_DONE FAIL_AE32" >> "$LOG"; exit 1; fi

log "--- COMPARE fs AE-32 vs current warm-started AE-32 ---"
CUR32="$WARM/ae32/gate_eval/metrics/test_fusion_evaluation_metrics.json"
FS32="$PARENT/ae32/phase3/gate_eval/metrics/test_fusion_evaluation_metrics.json"
"$PY" "$AEI/gate_fs_compare.py" "$FS32" "$CUR32" >> "$LOG" 2>&1; cmp=$?
if [[ $cmp -eq 0 ]]; then
  log "AE-32 from-phase-1 is CLEARLY BETTER -> building AE-64 and AE-128 from phase 1"
  run_bottleneck 64  || log "  ae64 from-phase-1: review (see log)"
  run_bottleneck 128 || log "  ae128 from-phase-1: review (see log)"
else
  log "AE-32 from-phase-1 NOT clearly better (cmp rc=$cmp) -> per plan, NOT training 64/128."
  log "  Conclusion: AE-from-phase-1 adds no advantage; the warm-started (phase-3) models stand."
fi
log "===== AE-FROM-PHASE-1 RUN END ====="
echo "FS_AE_DONE OK" >> "$LOG"
