#!/usr/bin/env bash
# Offline accuracy-vs-compression sweep (deterministic; no CARLA). Runs evaluate_fusion through the
# split-inference codec round-trip at each quant profile + an uncompressed baseline, on the 200k model's
# test split. Reuses ALL of evaluate_fusion's recall/loc-error/IoU code (only the features are quantized).
#
#   VALIDATE first:  bash run_accuracy_sweep.sh baseline q_pchan_u8_zlib   # baseline~=ref, uint8~=baseline
#   FULL:            bash run_accuracy_sweep.sh
set -uo pipefail
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
cd "$AB"; source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
CKPT="$AB/experiments/autonomous_arch_runs_20260625/det_pps200000_v2/checkpoints/det_pps200000_v2/best.pt"
DATASET="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
CONFIG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
OUT="$AB/experiments/rl_accuracy_sweep"
THR="--object-score-threshold 0.10 --object-nms-radius-px 6 --topk-objects 120 --max-gt-distance-m 40"
# name|quant|entropy   (baseline has empty quant = uncompressed, routed through model() directly)
PROFILES=(
  "baseline||"
  "q_pchan_u8_zlib|per_channel_uint8|zlib"
  "q_pchan_u6_zlib|per_channel_uint6|zlib"
  "q_pchan_u6_none|per_channel_uint6|none"
  "q_pchan_u4_zlib|per_channel_uint4|zlib"
  "q_ptensor_u8_zlib|per_tensor_uint8|zlib"
  "q_pchan_u8_none|per_channel_uint8|none"
  "q_pchan_u4_none|per_channel_uint4|none"
  "q_ptensor_u8_none|per_tensor_uint8|none"
)
WANT="${*:-}"
for p in "${PROFILES[@]}"; do
  IFS='|' read -r name quant ent <<< "$p"
  [ -n "$WANT" ] && [[ " $WANT " != *" $name "* ]] && continue
  ed="$OUT/$name"; mkdir -p "$ed"
  [ -L "$ed/dataset" ] && unlink "$ed/dataset"; ln -s "$DATASET" "$ed/dataset"
  qarg=""; [ -n "$quant" ] && qarg="--quantization-mode $quant --entropy-coder $ent"
  echo "[$(date '+%F %T')] accuracy eval: $name  ($quant ${ent})"
  if python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CONFIG" --experiment-dir "$ed" \
       --checkpoint "$CKPT" --split test $THR $qarg --device auto >"$ed/eval.log" 2>&1; then
    echo "  OK"
  else
    echo "  FAIL -> $ed/eval.log"; tail -3 "$ed/eval.log"
  fi
done
python3 "$AB/rl_agent/accuracy_aggregate.py"
