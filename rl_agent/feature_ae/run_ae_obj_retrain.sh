#!/usr/bin/env bash
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
LOG="$AB/rl_agent/feature_ae/ae_obj_retrain_log.md"
echo "[$(date '+%F %T')] object-weighted AE retrain START (seg0.3 heat8 reg5 recon0.05)" | tee -a "$LOG"
for BN in 128 64 32; do
  echo "[$(date '+%F %T')] AE_obj b$BN" | tee -a "$LOG"
  "$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck $BN \
    --epochs 20 --batch-size 8 --lr 1e-3 --drop-max 0.8 --num-workers 4 \
    --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _obj >> "$LOG" 2>&1 || echo "  WARN b$BN rc=$?" | tee -a "$LOG"
done
echo "AE_OBJ_RETRAIN_DONE" >> "$LOG"
echo "[$(date '+%F %T')] object-weighted AE retrain END" | tee -a "$LOG"
