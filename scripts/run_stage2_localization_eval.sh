#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

CONFIG="${CONFIG:-${ABIODUN_DIR}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml}"
DATASET="${DATASET:-${ABIODUN_DIR}/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
CHECKPOINT="${CHECKPOINT:-${ABIODUN_DIR}/experiments/moving_ego_r100k_r4_tw2_stage2_localization_from_locked_seg_20260624/checkpoints/stage2_lockedseg_objhead_radius4_lowfuse_bs24_locdim/best.pt}"
EVAL_DIR="${EVAL_DIR:-${ABIODUN_DIR}/experiments/moving_ego_r100k_r4_tw2_stage2_localization_from_locked_seg_20260624/eval_stage2_best_thr010_nms2_top120}"
OBJECT_SCORE_THRESHOLD="${OBJECT_SCORE_THRESHOLD:-0.10}"
OBJECT_NMS_RADIUS_PX="${OBJECT_NMS_RADIUS_PX:-2}"
TOPK_OBJECTS="${TOPK_OBJECTS:-120}"
MATCH_DISTANCE_M="${MATCH_DISTANCE_M:-3.0}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python3}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export PYTHONPATH="${ABIODUN_DIR}/pole_lraspp_multimodal_fusion:${ABIODUN_DIR}:${PYTHONPATH:-}"

if [[ ! -d "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"
if [[ -L "${EVAL_DIR}/dataset" ]]; then
  unlink "${EVAL_DIR}/dataset"
elif [[ -e "${EVAL_DIR}/dataset" ]]; then
  echo "${EVAL_DIR}/dataset exists and is not a symlink; refusing to replace it." >&2
  exit 1
fi
ln -s "${DATASET}" "${EVAL_DIR}/dataset"

echo "Evaluating Stage-2 localization checkpoint:"
echo "  checkpoint: ${CHECKPOINT}"
echo "  threshold=${OBJECT_SCORE_THRESHOLD} nms=${OBJECT_NMS_RADIUS_PX} topk=${TOPK_OBJECTS} match=${MATCH_DISTANCE_M}m"

"${PYTHON}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
  --config "${CONFIG}" \
  --experiment-dir "${EVAL_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --object-score-threshold "${OBJECT_SCORE_THRESHOLD}" \
  --object-nms-radius-px "${OBJECT_NMS_RADIUS_PX}" \
  --topk-objects "${TOPK_OBJECTS}" \
  --match-distance-m "${MATCH_DISTANCE_M}" \
  --device "${DEVICE}" \
  --require-cuda

echo "Done. Metrics:"
echo "${EVAL_DIR}/metrics/test_fusion_evaluation_metrics.json"
echo "${EVAL_DIR}/metrics/test_learned_object_metrics.csv"
