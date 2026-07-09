#!/usr/bin/env bash
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
LOG="$AB/rl_agent/feature_ae/ae_b256_log.md"
echo "[$(date '+%F %T')] b256 object-weighted AE START (heat8 reg5 seg0.3)" | tee -a "$LOG"
"$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck 256 \
  --epochs 18 --batch-size 8 --lr 1e-3 --drop-max 0.8 --num-workers 4 \
  --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _obj >> "$LOG" 2>&1 || echo "  WARN rc=$?" | tee -a "$LOG"
echo "AE_B256_DONE" >> "$LOG"
echo "[$(date '+%F %T')] b256 END" | tee -a "$LOG"
