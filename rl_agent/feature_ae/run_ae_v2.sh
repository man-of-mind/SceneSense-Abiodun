#!/usr/bin/env bash
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
MP="$AB/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
LOG="$AB/rl_agent/feature_ae/ae_v2_log.md"
# GPU-sequence: wait for the b256(v1) run to release the GPU
for i in $(seq 1 120); do grep -q AE_B256_DONE "$AB/rl_agent/feature_ae/ae_b256_log.md" 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] v2 (nonlinear+spatial) object-weighted AE START @ aggressive bottlenecks" | tee -a "$LOG"
for BN in 128 64; do
  echo "[$(date '+%F %T')] v2 b$BN" | tee -a "$LOG"
  "$PY" rl_agent/feature_ae/train_ae.py --model-checkpoint "$MP" --bottleneck $BN --arch v2 \
    --epochs 18 --batch-size 8 --lr 8e-4 --drop-max 0.8 --num-workers 4 \
    --seg-w 0.3 --heat-w 8 --reg-w 5 --recon-w 0.05 --tag _v2obj >> "$LOG" 2>&1 || echo "  WARN b$BN rc=$?" | tee -a "$LOG"
done
echo "AE_V2_DONE" >> "$LOG"
echo "[$(date '+%F %T')] v2 END" | tee -a "$LOG"
