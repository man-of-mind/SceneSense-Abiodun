#!/usr/bin/env bash
# Rebalance re-run of Stage-2 ONLY (reuse M'_seg from Stage-1) with maximin selection
# (best at worst of clean q=0 and q~0.4) to recover clean object accuracy while keeping robustness.
# Then GATE A eval + authoritative gate_a_check.py. Writes a DONE marker for the watcher.
set -uo pipefail
ABIODUN="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${ABIODUN}"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CONFIG="${ABIODUN}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DATASET="${ABIODUN}/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
PARENT="${ABIODUN}/experiments/mprime_dropaware_20260708"
LOG="${ABIODUN}/rl_agent/PIPELINE_LOG.md"
export PYTHONPATH="${ABIODUN}/pole_lraspp_multimodal_fusion:${ABIODUN}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"; export MPLCONFIGDIR="/tmp/matplotlib-cache"; export QT_QPA_PLATFORM="offscreen"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
link_dataset(){ local d="$1"; mkdir -p "${d}"; [[ -L "${d}/dataset" ]] && unlink "${d}/dataset"; ln -s "${DATASET}" "${d}/dataset"; }

log "===== Stage-2 REBALANCE (maximin selection) START ====="
EXP2="${PARENT}/stage2_obj_drop"
link_dataset "${EXP2}"
"${PYTHON}" -m pole_lraspp_multimodal_fusion.train_fusion \
  --config "${CONFIG}" --experiment-dir "${EXP2}" \
  --trial-json "$(cat "${ABIODUN}/rl_agent/m_prime/stage2_obj_drop.json")" --training-budget-hours 5.0 \
  >> "${EXP2}/train.log" 2>&1
MPRIME="${EXP2}/checkpoints/mprime_stage2_obj_drop/best.pt"
[[ -f "${MPRIME}" ]] || { log "REBALANCE FAIL: no M' at ${MPRIME}"; echo "STAGE2_REBALANCE_DONE FAIL" >> "${LOG}"; exit 1; }
log "M' (rebalanced) READY: ${MPRIME}"

GDIR="${EXP2}/gateA_rebalance_best_thr020"; link_dataset "${GDIR}"
log "GATE A eval (rebalanced) -> ${GDIR}"
"${PYTHON}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
  --config "${CONFIG}" --experiment-dir "${GDIR}" --checkpoint "${MPRIME}" \
  --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
  --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda >> "${GDIR}/eval.log" 2>&1
"${PYTHON}" rl_agent/gate_a_check.py "${GDIR}/metrics/test_fusion_evaluation_metrics.json" >> "${LOG}" 2>&1
log "===== Stage-2 REBALANCE END ====="
echo "STAGE2_REBALANCE_DONE OK" >> "${LOG}"
