#!/usr/bin/env bash
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
D="$AB/rl_agent/feature_ae"
echo "[$(date '+%F %T')] v2 PARALLEL AE sweep START {256,128,64,32}" >> "$D/ae_v2_log.md"
for BN in 256 128 64 32; do
  (
    "$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck "$BN" --arch v2 \
      --epochs 18 --batch-size 8 --lr 8e-4 --drop-max 0.8 --num-workers 3 \
      --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _v2obj > "$D/ae_v2_b${BN}.log" 2>&1
    echo "V2_B${BN}_DONE" >> "$D/ae_v2_b${BN}.log"
  ) &
done
wait
echo "AE_V2_DONE" >> "$D/ae_v2_log.md"
echo "[$(date '+%F %T')] v2 PARALLEL AE sweep END" >> "$D/ae_v2_log.md"
