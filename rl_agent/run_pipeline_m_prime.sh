#!/usr/bin/env bash
# Drop-aware M' pipeline: Stage-1 (seg, backbone+seg trainable, object head = frozen
# objectness oracle) -> Stage-2 (object head, backbone+seg frozen) -> GATE A eval at q=0.
# Both stages add objectness feature-dropout q~U(0,0.8) via trial "feature_drop_max".
# Launched via setsid so it survives the session. All steps logged to PIPELINE_LOG.md.
set -uo pipefail

ABIODUN="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
cd "${ABIODUN}"

PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CONFIG="${ABIODUN}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DATASET="${ABIODUN}/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
PARENT="${ABIODUN}/experiments/mprime_dropaware_20260708"
MP="${ABIODUN}/rl_agent/m_prime"
LOG="${ABIODUN}/rl_agent/PIPELINE_LOG.md"

export PYTHONPATH="${ABIODUN}/pole_lraspp_multimodal_fusion:${ABIODUN}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="/tmp/matplotlib-cache"
export QT_QPA_PLATFORM="offscreen"

mkdir -p "${PARENT}" "/tmp/matplotlib-cache"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

link_dataset() {  # $1 = dir
  local d="$1"; mkdir -p "${d}"
  if [[ -L "${d}/dataset" ]]; then unlink "${d}/dataset"; fi
  [[ -e "${d}/dataset" ]] && { log "refuse: ${d}/dataset not a symlink"; exit 1; }
  ln -s "${DATASET}" "${d}/dataset"
}

run_stage() {  # $1=exp_dir $2=trial_json $3=trial_name $4=budget_hours
  local edir="$1" tjson="$2" tname="$3" budget="$4"
  link_dataset "${edir}"
  log "STAGE start: ${tname}  (budget ${budget}h)  -> ${edir}"
  "${PYTHON}" -m pole_lraspp_multimodal_fusion.train_fusion \
    --config "${CONFIG}" --experiment-dir "${edir}" \
    --trial-json "$(cat "${tjson}")" --training-budget-hours "${budget}" \
    >> "${edir}/train.log" 2>&1
  local rc=$?
  local best="${edir}/checkpoints/${tname}/best.pt"
  if [[ ${rc} -ne 0 ]]; then log "STAGE ${tname} exited rc=${rc} (see ${edir}/train.log)"; fi
  if [[ ! -f "${best}" ]]; then
    log "GATE FAIL: no checkpoint at ${best}. Halting pipeline."
    return 1
  fi
  log "STAGE done: ${tname}  best=${best}"
  return 0
}

log "===== M' drop-aware pipeline START (pid $$) ====="

# ---- Stage 1: drop-aware seg (backbone+seg train; object head frozen oracle) ----
EXP1="${PARENT}/stage1_seg_drop"
run_stage "${EXP1}" "${MP}/stage1_seg_drop.json" "mprime_stage1_seg_drop" 6.0 || exit 1

# ---- Stage 2: drop-aware object head (backbone+seg frozen) => M' ----
EXP2="${PARENT}/stage2_obj_drop"
run_stage "${EXP2}" "${MP}/stage2_obj_drop.json" "mprime_stage2_obj_drop" 5.0 || exit 1

MPRIME="${EXP2}/checkpoints/mprime_stage2_obj_drop/best.pt"
log "M' READY: ${MPRIME}"

# ---- GATE A: eval M' at q=0 (clean, no split codec) on test split @ thr 0.20 ----
GDIR="${EXP2}/gateA_eval_best_thr020"
link_dataset "${GDIR}"
log "GATE A eval start -> ${GDIR}"
"${PYTHON}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
  --config "${CONFIG}" --experiment-dir "${GDIR}" --checkpoint "${MPRIME}" \
  --split test --object-score-threshold 0.20 --object-nms-radius-px 2 \
  --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda \
  >> "${GDIR}/eval.log" 2>&1
log "GATE A eval done. Metrics: ${GDIR}/metrics/test_fusion_evaluation_metrics.json"

# ---- GATE A automated compare vs 200k targets ----
"${PYTHON}" - "$GDIR/metrics/test_fusion_evaluation_metrics.json" >> "${LOG}" 2>&1 <<'PYEOF'
import json, sys
p = sys.argv[1]
try:
    m = json.load(open(p))
except Exception as e:
    print(f"[GATE A] could not read metrics: {e}"); sys.exit(0)
def find(keys):
    for k in keys:
        if k in m: return m[k]
    # shallow nested search
    for v in m.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v: return v[k]
    return None
targets = {
  "mIoU":        (find(["mean_iou","miou","mIoU"]),                 0.837, ">="),
  "vehicle_IoU": (find(["vehicle_iou","car_iou","iou_vehicle"]),    0.934, ">="),
  "obj_recall":  (find(["object_recall","recall","obj_recall"]),    0.775, ">="),
  "ped_loc_m":   (find(["pedestrian_xy_mae","person_xy_mae","ped_loc_error_m","xy_mae"]), 1.38, "<="),
}
print("[GATE A] q=0 acceptance check vs 200k targets:")
ok = True
for name,(val,thr,op) in targets.items():
    if val is None:
        print(f"   {name}: (metric key not found - inspect json)"); ok=False; continue
    passed = (val >= thr) if op==">=" else (val <= thr)
    ok = ok and passed
    print(f"   {name}: {val:.4f} {op} {thr}  -> {'PASS' if passed else 'FAIL'}")
print(f"[GATE A] RESULT: {'PASS - proceed to sweeps/AE on M-prime' if ok else 'REVIEW - some metric off; inspect before building on M-prime'}")
PYEOF

log "===== M' drop-aware pipeline END ====="
