#!/usr/bin/env bash
# Per-model quant x ROI sweep over the 4 comparable models (no-AE baseline + AE-32/64/128), full test set,
# entropy=zlib. AE models: evaluate_fusion auto-attaches the integrated AE + uses it as the split codec
# (bottleneck quant + ROI + payload). No-AE baseline: standard backbone-feature quant + ROI. -> combined matrix.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
S="$AB/experiments/ae_integrated_20260710/sweeps_permodel"
LOG="$AB/rl_agent/ae_integrated/PERMODEL_SWEEP_LOG.md"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
mkdir -p "$S"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
link_ds(){ local d="$1"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"; }
# wait for the no-AE baseline to finish (frees the GPU + gives the 4th model)
for i in $(seq 1 120); do grep -q NOAE_BASELINE_DONE "$AB/rl_agent/ae_integrated/NOAE_BASELINE_LOG.md" 2>/dev/null && break; sleep 60; done

declare -A CKPT=(
  [noae]="$AB/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
  [ae32]="$AB/experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt"
  [ae64]="$AB/experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt"
  [ae128]="$AB/experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt"
)
ev(){ # $1=name  $2=checkpoint  rest=eval args
  local name="$1"; shift; local d="$S/$name"   # after shift: $1=checkpoint, ${@:2}=extra eval args
  [[ -f "$d/metrics/test_fusion_evaluation_metrics.json" ]] && { log "skip $name (done)"; return 0; }
  link_ds "$d"; log "eval $name: $*"
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$d" --checkpoint "$1" \
    --split test --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
    --max-gt-distance-m 40 --device cuda --entropy-coder zlib "${@:2}" >> "$d/eval.log" 2>&1 || log "  WARN $name rc=$?"
}
log "===== PER-MODEL SWEEP START (4 models x quant{8,6,4} x ROI{0,0.3,0.5}) ====="
for M in noae ae32 ae64 ae128; do
  CK="${CKPT[$M]}"
  [[ -f "$CK" ]] || { log "MISSING checkpoint for $M ($CK) -- skipping"; continue; }
  # clean (no compression) reference for this model
  ev "${M}__clean" "$CK"
  for Q in per_channel_uint8 per_channel_uint6 per_channel_uint4; do
    for R in 0.0 0.3 0.5; do
      ev "${M}__${Q#per_channel_}__roi${R}" "$CK" --quantization-mode "$Q" --roi-threshold "$R"
    done
  done
done
log "--- aggregate PERMODEL_KNOB_MATRIX ---"
"$PY" rl_agent/build_knob_matrix.py "$S" rl_agent/PERMODEL_KNOB_MATRIX.md rl_agent/loopback_latency.json 2835 2216 >> "$LOG" 2>&1 || log "  WARN matrix"
log "===== PER-MODEL SWEEP END ====="
echo "PERMODEL_SWEEP_DONE OK" >> "$LOG"
