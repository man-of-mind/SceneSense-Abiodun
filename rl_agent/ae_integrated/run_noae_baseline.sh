#!/usr/bin/env bash
# Fair-control baseline: retrain the model with the SAME joint recipe as the integrated-AE models but NO AE
# (warm-start M', joint seg+object, all trainable, drop-aware). Gives a directly comparable no-AE baseline.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
EXP="$AB/experiments/ae_integrated_20260710/noae_baseline"
D="$AB/rl_agent/ae_integrated"
LOG="$D/NOAE_BASELINE_LOG.md"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
link_ds(){ local d="$1"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"; }
log "===== NO-AE BASELINE (joint recipe, fair control) START ====="
link_ds "$EXP"
"$PY" -m pole_lraspp_multimodal_fusion.train_fusion --config "$CFG" --experiment-dir "$EXP" \
  --trial-json "$(cat "$D/mprime_joint_noae.json")" --training-budget-hours 3.0 >> "$EXP/train.log" 2>&1
BEST="$EXP/checkpoints/mprime_joint_noae/best.pt"
if [[ ! -f "$BEST" ]]; then log "NO checkpoint produced"; echo "NOAE_BASELINE_DONE FAIL" >> "$LOG"; exit 1; fi
GD="$EXP/gate_eval"; link_ds "$GD"
log "=== GATE-eval no-AE baseline (clean) ==="
"$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$GD" \
  --checkpoint "$BEST" --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
  --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda >> "$GD/eval.log" 2>&1
"$PY" "$D/gate_ae_check.py" "$GD/metrics/test_fusion_evaluation_metrics.json" >> "$LOG" 2>&1 || true
"$PY" -c "import json;m=json.load(open('$GD/metrics/test_fusion_evaluation_metrics.json'));print(f\"  noae_baseline: mIoU {m['miou']:.3f} veh {m['vehicle_iou']:.3f} ped-rec {m['learned_person_object_recall']:.3f} obj-rec {m['learned_object_recall']:.3f} loc {m['learned_global_xy_mae_m']:.2f}m\")" >> "$LOG" 2>&1
log "===== NO-AE BASELINE END ====="
echo "NOAE_BASELINE_DONE OK" >> "$LOG"
