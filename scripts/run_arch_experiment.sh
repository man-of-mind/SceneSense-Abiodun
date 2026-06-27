#!/usr/bin/env bash
# Autonomous architecture experiment driver:
#   train (stage-2, frozen seg backbone) -> eval sweep on best.pt AND last.pt -> append to RESULTS.md
# Parameterized entirely by env vars so the orchestrator can launch variants.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

# ---- fixed assets ----
CONFIG="${ABIODUN_DIR}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DATASET="${DATASET:-${ABIODUN_DIR}/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
SEG_CHECKPOINT="${SEG_CHECKPOINT:-${ABIODUN_DIR}/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_segonly_ablation_20260624_segonly_lovasz05_personselect_pat20_bs24/checkpoints/segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20/best.pt}"
PARENT="${ABIODUN_DIR}/experiments/autonomous_arch_runs_20260625"
RESULTS="${PARENT}/RESULTS.md"

# ---- per-run params (env-overridable) ----
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
HEAD_ARCH="${HEAD_ARCH:-decoupled}"
USE_COORDCONV="${USE_COORDCONV:-true}"
USE_GROUNDPLANE_PRIOR="${USE_GROUNDPLANE_PRIOR:-false}"
PREDICT_BBOX2D="${PREDICT_BBOX2D:-false}"
ADAPTIVE_RADIUS="${ADAPTIVE_RADIUS:-false}"
HEAD_DEPTH="${HEAD_DEPTH:-3}"
HEATMAP_RADIUS_PX="${HEATMAP_RADIUS_PX:-4}"
MAX_GT_DISTANCE_M="${MAX_GT_DISTANCE_M:-0}"   # 0 = no range gate; >0 = operating-range gate (m)
EPOCHS="${EPOCHS:-60}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0002}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
SELECTION_SCORE_MODE="${SELECTION_SCORE_MODE:-loc_dim_loss}"
TRAINING_BUDGET_HOURS="${TRAINING_BUDGET_HOURS:-4.0}"
# Backbone-adaptation knobs (defaults reproduce the frozen-backbone stage-2 recipe).
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
FREEZE_CLASSIFIER="${FREEZE_CLASSIFIER:-true}"
FREEZE_BN="${FREEZE_BN:-false}"
SEG_LOSS_WEIGHT="${SEG_LOSS_WEIGHT:-0.0}"
INIT_OBJECT_CHECKPOINT="${INIT_OBJECT_CHECKPOINT:-}"
UNFREEZE_BACKBONE_LAST_N="${UNFREEZE_BACKBONE_LAST_N:-0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
DISTILL_TEMP="${DISTILL_TEMP:-2.0}"
DISTILL_TEACHER_CHECKPOINT="${DISTILL_TEACHER_CHECKPOINT:-${SEG_CHECKPOINT}}"
# Optional per-component object-loss override (JSON). Empty -> use config defaults.
OBJECT_LOSS_JSON="${OBJECT_LOSS_JSON:-}"
PYTHON="${PYTHON:-python3}"

export MPLCONFIGDIR="/tmp/matplotlib-cache"
export QT_QPA_PLATFORM="offscreen"
export PYTHONPATH="${ABIODUN_DIR}/pole_lraspp_multimodal_fusion:${ABIODUN_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EXPERIMENT_DIR="${PARENT}/${RUN_NAME}"
mkdir -p "${PARENT}" "${EXPERIMENT_DIR}"

# Assemble loss_weights, optionally injecting the per-component object override.
if [[ -n "${OBJECT_LOSS_JSON}" ]]; then
  LOSS_WEIGHTS_JSON="{\"segmentation\":${SEG_LOSS_WEIGHT},\"object_total\":1.0,\"object\":${OBJECT_LOSS_JSON}}"
else
  LOSS_WEIGHTS_JSON="{\"segmentation\":${SEG_LOSS_WEIGHT},\"object_total\":1.0}"
fi

link_dataset() {  # $1 = dir
  local d="$1"
  if [[ -L "${d}/dataset" ]]; then unlink "${d}/dataset"; fi
  [[ -e "${d}/dataset" ]] && { echo "refuse: ${d}/dataset not a symlink"; exit 1; }
  ln -s "${DATASET}" "${d}/dataset"
}
link_dataset "${EXPERIMENT_DIR}"

trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":%s,"weight_decay":%s,"augment_strength":"strong","input_size":[768,432],"batch_size":%s,"num_workers":%s,"prefetch_factor":%s,"persistent_workers":%s,"epochs":%s,"init_rgb_checkpoint":"%s","init_object_checkpoint":"%s","freeze_backbone":%s,"freeze_classifier":%s,"freeze_bn":%s,"freeze_object_head":false,"unfreeze_backbone_last_n":%s,"distill_weight":%s,"distill_temp":%s,"distill_teacher_checkpoint":"%s","selection_score_mode":"%s","early_stop_patience":%s,"loss_weights":%s,"object_heads":{"heatmap_radius_px":%s,"fuse_low_feature":true,"head_arch":"%s","use_coordconv":%s,"head_depth":%s,"use_groundplane_prior":%s,"predict_bbox2d":%s,"adaptive_heatmap_radius":%s,"max_gt_distance_m":%s}}' \
  "${RUN_NAME}" "${LR}" "${WEIGHT_DECAY}" "${BATCH_SIZE}" "${NUM_WORKERS}" "${PREFETCH_FACTOR}" "${PERSISTENT_WORKERS}" "${EPOCHS}" "${SEG_CHECKPOINT}" "${INIT_OBJECT_CHECKPOINT}" "${FREEZE_BACKBONE}" "${FREEZE_CLASSIFIER}" "${FREEZE_BN}" "${UNFREEZE_BACKBONE_LAST_N}" "${DISTILL_WEIGHT}" "${DISTILL_TEMP}" "${DISTILL_TEACHER_CHECKPOINT}" "${SELECTION_SCORE_MODE}" "${EARLY_STOP_PATIENCE}" "${LOSS_WEIGHTS_JSON}" "${HEATMAP_RADIUS_PX}" "${HEAD_ARCH}" "${USE_COORDCONV}" "${HEAD_DEPTH}" "${USE_GROUNDPLANE_PRIOR}" "${PREDICT_BBOX2D}" "${ADAPTIVE_RADIUS}" "${MAX_GT_DISTANCE_M}")"

echo "=== TRAIN ${RUN_NAME} ==="
echo "  arch=${HEAD_ARCH} coordconv=${USE_COORDCONV} depth=${HEAD_DEPTH} radius=${HEATMAP_RADIUS_PX}"
echo "  trial: ${trial_json}"
"${PYTHON}" -m pole_lraspp_multimodal_fusion.train_fusion \
  --config "${CONFIG}" --experiment-dir "${EXPERIMENT_DIR}" \
  --trial-json "${trial_json}" --training-budget-hours "${TRAINING_BUDGET_HOURS}"

# ---- eval best.pt and last.pt at threshold sweep ----
CKPT_DIR="${EXPERIMENT_DIR}/checkpoints/${RUN_NAME}"
eval_one() {  # $1 = which (best/last), $2 = threshold
  local which="$1" thr="$2"
  local ckpt="${CKPT_DIR}/${which}.pt"
  [[ -f "${ckpt}" ]] || { echo "missing ${ckpt}"; return 0; }
  local edir="${EXPERIMENT_DIR}/eval_${which}_thr${thr//./}"
  mkdir -p "${edir}"; link_dataset "${edir}"
  local range_gate=""
  [[ "${MAX_GT_DISTANCE_M}" != "0" ]] && range_gate="--max-gt-distance-m ${MAX_GT_DISTANCE_M}"
  "${PYTHON}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
    --config "${CONFIG}" --experiment-dir "${edir}" --checkpoint "${ckpt}" \
    --split test --object-score-threshold "${thr}" --object-nms-radius-px 2 \
    --topk-objects 120 --match-distance-m 5.0 ${range_gate} --device cuda > "${edir}/eval.log" 2>&1 || true
}
for which in best last; do
  for thr in 0.10 0.20 0.30; do eval_one "${which}" "${thr}"; done
done

# ---- summarize into RESULTS.md ----
"${PYTHON}" - "${EXPERIMENT_DIR}" "${RUN_NAME}" "${RESULTS}" "${HEAD_ARCH}" "${USE_COORDCONV}" "${HEAD_DEPTH}" "${HEATMAP_RADIUS_PX}" <<'PY'
import json, sys, glob, os
exp, run, results, arch, coord, depth, radius = sys.argv[1:8]
rows=[]
for which in ("best","last"):
    for thr in ("010","020","030"):
        f=os.path.join(exp, f"eval_{which}_thr{thr}", "metrics", "test_fusion_evaluation_metrics.json")
        if not os.path.exists(f): continue
        d=json.load(open(f))
        rows.append((which, f"0.{thr[1:]}" if thr[0]=="0" else thr,
                     d.get("learned_object_f1",0), d.get("learned_object_recall",0), d.get("learned_object_precision",0),
                     d.get("learned_person_object_f1",0), d.get("learned_person_object_recall",0),
                     d.get("learned_vehicle_object_f1",0), d.get("learned_vehicle_object_recall",0),
                     d.get("learned_global_xy_mae_m",0), d.get("learned_person_global_xy_mae_m",0),
                     d.get("vehicle_iou",0), d.get("person_iou",0)))
best = max(rows, key=lambda r: r[2]) if rows else None
header = "" if os.path.exists(results) else "# Autonomous architecture runs — RESULTS\n\nBaseline (ep57 shared/no-coord/depth2): overall F1 0.345, person F1 0.36-0.37, vehicle F1 0.32-0.33, recall ~0.35, XY MAE ~2.1m, seg 0.916/0.758.\n\n"
with open(results, "a") as fh:
    if header: fh.write(header)
    fh.write(f"\n## {run}\n")
    fh.write(f"arch={arch} coordconv={coord} depth={depth} radius={radius}\n\n")
    fh.write("| ckpt | thr | F1 | recall | prec | person F1 | person rec | veh F1 | veh rec | xy MAE | person MAE | seg veh/person |\n")
    fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f | %.3f/%.3f |\n" %
                 (r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12]))
    if best:
        fh.write(f"\n**Best F1 = {best[2]:.3f}** ({best[0]} @ thr {best[1]}); person F1 {best[5]:.3f}, vehicle F1 {best[7]:.3f}, person XY MAE {best[10]:.2f}m.\n")
print("BEST_F1", f"{best[2]:.4f}" if best else "NA", "RUN", run)
PY

echo "=== DONE ${RUN_NAME} ==="
