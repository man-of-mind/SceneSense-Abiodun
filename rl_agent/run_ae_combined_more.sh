#!/usr/bin/env bash
# More AE-combined profiles for balance: AE x quant (harder bottleneck quant) and AE x ROI across bottlenecks.
# Offline accuracy+payload; latency joins/interpolates. Rebuilds matrix at the end. GPU (no CARLA).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
S="$AB/experiments/mprime_dropaware_20260708/sweeps"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
LOG="$AB/rl_agent/AE_COMBINED_MORE_LOG.md"
CKD=rl_agent/feature_ae/checkpoints
echo "[$(date '+%F %T')] more AE-combined START" | tee -a "$LOG"
run(){ local name="$1"; shift; local d="$S/$name"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"
  echo "[$(date '+%F %T')] $name: $*" | tee -a "$LOG"
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$d" --checkpoint "$MP" \
    --split test --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
    --max-gt-distance-m 40 --device cuda --entropy-coder zstd "$@" >> "$d/eval.log" 2>&1 || echo "  WARN $name rc=$?" | tee -a "$LOG"; }
# AE x quant (quantize the bottleneck harder)
run comb_ae64_u4     --quantization-mode per_channel_uint4 --ae-checkpoint $CKD/ae_b64_v2clean.pt
run comb_ae32_u4     --quantization-mode per_channel_uint4 --ae-checkpoint $CKD/ae_b32_v2clean.pt
# AE x ROI across bottlenecks
run comb_ae128_roi0.3 --quantization-mode per_channel_uint8 --roi-threshold 0.3 --ae-checkpoint $CKD/ae_b128_v2clean.pt
run comb_ae64_roi0.5  --quantization-mode per_channel_uint8 --roi-threshold 0.5 --ae-checkpoint $CKD/ae_b64_v2clean.pt
run comb_ae32_roi0.3  --quantization-mode per_channel_uint8 --roi-threshold 0.3 --ae-checkpoint $CKD/ae_b32_v2clean.pt
"$PY" rl_agent/build_knob_matrix.py "$S" rl_agent/COMPLETE_KNOB_MATRIX.md rl_agent/loopback_latency.json 2835 >> "$LOG" 2>&1 || echo "  WARN matrix rc=$?" | tee -a "$LOG"
echo "AE_COMBINED_MORE_DONE" >> "$LOG"
echo "[$(date '+%F %T')] more AE-combined END" | tee -a "$LOG"
