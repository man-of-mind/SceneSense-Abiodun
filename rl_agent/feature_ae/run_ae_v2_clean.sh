#!/usr/bin/env bash
# Definitive fair test of the v2 (nonlinear+spatial) AE: NO ROI-drop in the loop (isolate the AE's raw
# ability to preserve detection) + proper schedule (40 epochs, lr 3e-4). b128 + b64 in parallel.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
D="$AB/rl_agent/feature_ae"
echo "[$(date '+%F %T')] v2 CLEAN test START (drop_max=0, 40ep, lr3e-4) b128,b64" >> "$D/ae_v2clean_log.md"
for BN in 128 64; do
  (
    "$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck "$BN" --arch v2 \
      --epochs 40 --batch-size 8 --lr 3e-4 --drop-max 0.0 --num-workers 4 \
      --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _v2clean > "$D/ae_v2clean_b${BN}.log" 2>&1
    echo "V2CLEAN_B${BN}_DONE" >> "$D/ae_v2clean_b${BN}.log"
  ) &
done
wait
echo "AE_V2CLEAN_DONE" >> "$D/ae_v2clean_log.md"
echo "[$(date '+%F %T')] v2 CLEAN test END" >> "$D/ae_v2clean_log.md"
