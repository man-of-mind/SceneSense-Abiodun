#!/usr/bin/env bash
# Combined-action probes (offline accuracy + payload) to test how knobs COMPOSE and enrich the RL
# action space: quant x ROI and AE x ROI. Waits for b32 first (GPU), then rebuilds the matrix -> 21 profiles.
# Latency for these joins by (quant,roi,ae); un-measured combos interpolate from the loopback curve.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
S="$AB/experiments/mprime_dropaware_20260708/sweeps"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
LOG="$AB/rl_agent/COMBINED_PROBES_LOG.md"
AE64="rl_agent/feature_ae/checkpoints/ae_b64_v2clean.pt"
# wait for b32 (shares GPU)
for i in $(seq 1 90); do grep -q AE_B32CLEAN_DONE "$AB/rl_agent/feature_ae/ae_b32clean_log.md" 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] combined probes START" | tee -a "$LOG"
run(){ # $1=name  rest=extra eval args
  local name="$1"; shift; local d="$S/$name"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"
  echo "[$(date '+%F %T')] eval $name: $*" | tee -a "$LOG"
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$d" --checkpoint "$MP" \
    --split test --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
    --max-gt-distance-m 40 --device cuda --entropy-coder zstd "$@" >> "$d/eval.log" 2>&1 || echo "  WARN $name rc=$?" | tee -a "$LOG"
}
run comb_u4_roi0.3   --quantization-mode per_channel_uint4 --roi-threshold 0.3
run comb_u4_roi0.5   --quantization-mode per_channel_uint4 --roi-threshold 0.5
run comb_ae64_roi0.3 --quantization-mode per_channel_uint8 --roi-threshold 0.3 --ae-checkpoint "$AE64"
"$PY" rl_agent/build_knob_matrix.py "$S" rl_agent/COMPLETE_KNOB_MATRIX.md rl_agent/loopback_latency.json 2835 >> "$LOG" 2>&1 || echo "  WARN matrix rc=$?" | tee -a "$LOG"
echo "COMBINED_PROBES_DONE" >> "$LOG"
echo "[$(date '+%F %T')] combined probes END" | tee -a "$LOG"
