#!/usr/bin/env bash
# Train the feature-AE at bottlenecks 128/64/32 on M' (frozen), objectness-drop q~U(0,0.8) in loop.
# Run AFTER GATE A passes (needs M' = stage2_obj_drop/best.pt). GPU-sequenced (one bottleneck at a time).
set -uo pipefail
ABIODUN="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${ABIODUN}"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="${ABIODUN}/pole_lraspp_multimodal_fusion:${ABIODUN}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="/tmp/matplotlib-cache"; export QT_QPA_PLATFORM="offscreen"

MPRIME="${MPRIME:-${ABIODUN}/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt}"
[[ -f "${MPRIME}" ]] || { echo "M' not found: ${MPRIME} (has GATE A passed?)"; exit 1; }
echo "[$(date '+%F %T')] AE training on M'=${MPRIME}"
for BN in 128 64 32; do
  echo "[$(date '+%F %T')] === AE bottleneck ${BN} ==="
  "${PYTHON}" rl_agent/feature_ae/train_ae.py \
    --model-checkpoint "${MPRIME}" --bottleneck "${BN}" \
    --epochs 15 --batch-size 8 --lr 1e-3 --drop-max 0.8 --num-workers 4
done
echo "[$(date '+%F %T')] AE training complete: rl_agent/feature_ae/checkpoints/ae_b{128,64,32}.pt"
