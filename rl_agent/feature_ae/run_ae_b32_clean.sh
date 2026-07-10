#!/usr/bin/env bash
# v2-clean AE at b32 (most aggressive bottleneck). Waits for the ideal-loopback latency run to finish
# first (so training load doesn't pollute those latency measurements), then trains + full-set accuracy
# eval + refreshes the matrix. Latency for b32 (AE compute ~ arch-constant) folded in via a later loopback.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
S="$AB/experiments/mprime_dropaware_20260708/sweeps"
LOG="$AB/rl_agent/feature_ae/ae_b32clean_log.md"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
# 1) wait for the ideal-loopback latency run to finish (keep its latency clean)
for i in $(seq 1 90); do grep -q IDEAL_LOOPBACK_DONE "$AB/rl_agent/IDEAL_LOOPBACK_LOG.md" 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] b32 clean-v2 train START" | tee -a "$LOG"
"$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck 32 --arch v2 \
  --epochs 40 --batch-size 8 --lr 3e-4 --drop-max 0.0 --num-workers 4 \
  --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _v2clean >> "$LOG" 2>&1 || echo "  WARN train rc=$?" | tee -a "$LOG"
# 2) full-set accuracy eval -> matrix sweep dir
d="$S/ae_v2clean_b32"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"
"$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$d" --checkpoint "$MP" \
  --split test --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
  --max-gt-distance-m 40 --device cuda --quantization-mode per_channel_uint8 --entropy-coder zstd \
  --ae-checkpoint rl_agent/feature_ae/checkpoints/ae_b32_v2clean.pt >> "$LOG" 2>&1 || echo "  WARN eval rc=$?" | tee -a "$LOG"
# 3) refresh matrix (b32 accuracy+payload in; latency interpolated until a b32 loopback runs)
"$PY" rl_agent/build_knob_matrix.py "$S" rl_agent/COMPLETE_KNOB_MATRIX.md rl_agent/loopback_latency.json 2835 >> "$LOG" 2>&1 || echo "  WARN matrix rc=$?" | tee -a "$LOG"
echo "AE_B32CLEAN_DONE" >> "$LOG"
echo "[$(date '+%F %T')] b32 clean-v2 END" | tee -a "$LOG"
