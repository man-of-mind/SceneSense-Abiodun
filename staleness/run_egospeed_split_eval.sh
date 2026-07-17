#!/usr/bin/env bash
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR=/tmp/fusion_mpl
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
CK="$AB/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
for SPLIT in test val; do   # test=stopped-ego, val=moving-ego (relabeled)
  echo "=== eval split=$SPLIT ==="
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$AB/staleness/egospeed_eval" --checkpoint "$CK" \
    --split "$SPLIT" --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
    --max-gt-distance-m 40 --device cuda --entropy-coder zlib --quantization-mode per_channel_uint8 --roi-threshold 0.0 \
    > "$AB/staleness/egospeed_eval/ev_$SPLIT.log" 2>&1 && echo "  $SPLIT OK" || echo "  $SPLIT rc=$?"
done
echo EGO3_DONE
