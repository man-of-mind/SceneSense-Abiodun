#!/usr/bin/env bash
# OVERNIGHT Month-2 static-knob completion. Fully offline (no CARLA).
# Waits for the rebalanced M', then: AE train {128,64,32} -> quant/entropy + ROI + AE sweeps
# (accuracy + payload BYTES) -> COMPLETE_KNOB_MATRIX.md -> Month-2 summary.
# Resumable: each eval skips if its metrics json already exists. proceed-with-flag on GATE A.
set -uo pipefail
ABIODUN="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${ABIODUN}"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CFG="${ABIODUN}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="${ABIODUN}/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
MPRIME="${ABIODUN}/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
SWEEP="${ABIODUN}/experiments/mprime_dropaware_20260708/sweeps"
AEDIR="${ABIODUN}/rl_agent/feature_ae/checkpoints"
LOG="${ABIODUN}/rl_agent/MONTH2_LOG.md"
export PYTHONPATH="${ABIODUN}/pole_lraspp_multimodal_fusion:${ABIODUN}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"; export MPLCONFIGDIR="/tmp/matplotlib-cache"; export QT_QPA_PLATFORM="offscreen"
mkdir -p "${SWEEP}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---- 0. wait for the rebalanced M' (poll up to ~3h) ----
log "===== OVERNIGHT MONTH-2 START ====="
for i in $(seq 1 180); do
  grep -q "STAGE2_REBALANCE_DONE" "${ABIODUN}/rl_agent/PIPELINE_LOG.md" 2>/dev/null && [[ -f "${MPRIME}" ]] && break
  sleep 60
done
[[ -f "${MPRIME}" ]] || { log "FATAL: rebalanced M' never appeared. Abort."; echo "MONTH2_DONE FAIL" >> "${LOG}"; exit 1; }
log "M' ready: ${MPRIME}"
grep -A10 "GATE A" "${ABIODUN}/rl_agent/PIPELINE_LOG.md" | tail -12 | while read -r l; do log "  gateA| $l"; done || true

# ---- helper: one offline eval (skips if done) ----
NROWS_DONE=0
eval_cfg(){  # $1=name  rest=extra args
  local name="$1"; shift
  local d="${SWEEP}/${name}"
  local mj="${d}/metrics/test_fusion_evaluation_metrics.json"
  if [[ -f "${mj}" ]]; then log "skip ${name} (exists)"; return 0; fi
  mkdir -p "${d}"; [[ -L "${d}/dataset" ]] && unlink "${d}/dataset"; ln -s "${DS}" "${d}/dataset"
  log "eval ${name}: $*"
  "${PY}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
    --config "${CFG}" --experiment-dir "${d}" --checkpoint "${MPRIME}" \
    --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
    --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda \
    "$@" >> "${d}/eval.log" 2>&1 || log "  WARN ${name} eval rc=$?"
}

# ---- 1. AE training on M' (3 bottlenecks), then keep going even if one fails ----
log "--- AE training on M' (128,64,32) ---"
for BN in 128 64 32; do
  if [[ -f "${AEDIR}/ae_b${BN}.pt" ]]; then log "skip AE b${BN} (exists)"; continue; fi
  log "AE train b${BN}"
  "${PY}" rl_agent/feature_ae/train_ae.py --model-checkpoint "${MPRIME}" --bottleneck "${BN}" \
     --epochs 15 --batch-size 8 --lr 1e-3 --drop-max 0.8 --num-workers 4 >> "${ABIODUN}/rl_agent/feature_ae/ae_train_log.md" 2>&1 || log "  WARN AE b${BN} rc=$?"
done

# ---- 2. sweeps (all offline; accuracy + payload) ----
# 2a. clean accuracy reference (no split codec)
eval_cfg "clean_noquant"
# 2b. quant x entropy (isolate compression), ROI=0, no AE
for Q in per_channel_uint8 per_channel_uint6 per_channel_uint4; do
  for E in zlib zstd none; do
    eval_cfg "quant_${Q#per_channel_}_${E}" --quantization-mode "${Q}" --entropy-coder "${E}"
  done
done
# 2c. ROI fraction sweep at u8/zlib
for R in 0.1 0.3 0.5 0.7; do
  eval_cfg "roi_${R}" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold "${R}"
done
# 2d. AE bottleneck sweep at u8/zlib (if trained)
for BN in 128 64 32; do
  [[ -f "${AEDIR}/ae_b${BN}.pt" ]] && eval_cfg "ae_b${BN}" --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint "${AEDIR}/ae_b${BN}.pt"
done
# 2e. a few combined operating points the agent might pick (AE + ROI)
for BN in 64 32; do
  [[ -f "${AEDIR}/ae_b${BN}.pt" ]] && eval_cfg "ae_b${BN}_roi0.3" --quantization-mode per_channel_uint8 --entropy-coder zlib --ae-checkpoint "${AEDIR}/ae_b${BN}.pt" --roi-threshold 0.3
done

# ---- 2.5 loopback latency/reliability sweep (needs CARLA; runs LAST so a CARLA hiccup can't
#          cost the offline matrix). Establishes the payload->{latency,reliability} curve. ----
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
LBDIR="${ABIODUN}/experiments/mprime_dropaware_20260708/sweeps_loopback"
LB_JSON="${ABIODUN}/rl_agent/loopback_latency.json"
carla_up(){ "${PY}" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }
start_carla(){
  for a in 1 2 3; do
    log "CARLA launch attempt ${a}"
    ( cd "${CARLA_ROOT}" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_overnight.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { sleep 12; carla_up && { log "CARLA up+stable"; return 0; }; }; sleep 4; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done
  return 1
}
log "--- loopback latency/reliability sweep (starting CARLA) ---"
if start_carla; then
  "${PY}" rl_agent/sweep_runner.py rl_agent/configs/loopback_sweep_mprime.json >> "${LOG}" 2>&1 || log "  WARN loopback sweep rc=$?"
  pkill -9 -f CarlaUnreal 2>/dev/null; sleep 5
  "${PY}" rl_agent/agg_loopback.py "${LBDIR}" "${ABIODUN}/rl_agent/LOOPBACK_LATENCY.md" "${LB_JSON}" >> "${LOG}" 2>&1 || log "  WARN agg_loopback failed"
else
  log "  WARN CARLA did not start after retries -> skipping loopback (offline matrix stands). Flagged for morning."
  pkill -9 -f CarlaUnreal 2>/dev/null
fi

# ---- 3. aggregate the matrix (+ join loopback latency/reliability if present) ----
log "--- aggregate COMPLETE_KNOB_MATRIX ---"
"${PY}" rl_agent/build_knob_matrix.py "${SWEEP}" "${ABIODUN}/rl_agent/COMPLETE_KNOB_MATRIX.md" "${LB_JSON}" >> "${LOG}" 2>&1 || log "  WARN aggregate failed"
log "===== OVERNIGHT MONTH-2 END ====="
echo "MONTH2_DONE OK" >> "${LOG}"
