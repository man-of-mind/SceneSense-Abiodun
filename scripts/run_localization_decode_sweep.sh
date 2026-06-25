#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

CONFIG="${CONFIG:-${ABIODUN_DIR}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml}"
DATASET="${DATASET:-${ABIODUN_DIR}/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
CHECKPOINT="${CHECKPOINT:-${ABIODUN_DIR}/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_fusion_train_20260624_r100k_r4_tw2_pilot/checkpoints/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_768x432_lr1e-4_bs2/best.pt}"
SWEEP_DIR="${SWEEP_DIR:-${ABIODUN_DIR}/experiments/localization_decode_sweep_20260624}"
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

mkdir -p "${SWEEP_DIR}"

settings=(
  "cfg_default,0.03,4,80"
  "thr001_nms4_top80,0.01,4,80"
  "thr001_nms4_top120,0.01,4,120"
  "thr003_nms2_top120,0.03,2,120"
  "thr001_nms2_top120,0.01,2,120"
  "thr005_nms2_top120,0.05,2,120"
  "thr010_nms2_top120,0.10,2,120"
)

for setting in "${settings[@]}"; do
  IFS=',' read -r name threshold nms_radius topk <<< "${setting}"
  exp_dir="${SWEEP_DIR}/${name}"
  mkdir -p "${exp_dir}"
  if [[ -L "${exp_dir}/dataset" ]]; then
    unlink "${exp_dir}/dataset"
  elif [[ -e "${exp_dir}/dataset" ]]; then
    echo "${exp_dir}/dataset exists and is not a symlink; refusing to replace it." >&2
    exit 1
  fi
  ln -s "${DATASET}" "${exp_dir}/dataset"
  "${PYTHON}" - <<PY
import json
from pathlib import Path
Path("${exp_dir}/decode_config.json").write_text(json.dumps({
  "object_score_threshold": float("${threshold}"),
  "object_nms_radius_px": int("${nms_radius}"),
  "topk_objects": int("${topk}"),
  "match_distance_m": float("${MATCH_DISTANCE_M}"),
  "checkpoint": "${CHECKPOINT}",
  "dataset": "${DATASET}",
}, indent=2) + "\\n", encoding="utf-8")
PY
  echo "Evaluating ${name}: threshold=${threshold} nms=${nms_radius} topk=${topk}"
  "${PYTHON}" -m pole_lraspp_multimodal_fusion.evaluate_fusion \
    --config "${CONFIG}" \
    --experiment-dir "${exp_dir}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --object-score-threshold "${threshold}" \
    --object-nms-radius-px "${nms_radius}" \
    --topk-objects "${topk}" \
    --match-distance-m "${MATCH_DISTANCE_M}" \
    --device "${DEVICE}" \
    --require-cuda
done

"${PYTHON}" scripts/summarize_localization_decode_sweep.py "${SWEEP_DIR}" --out "${SWEEP_DIR}/decode_sweep_summary.csv"
echo "Done. Summary: ${SWEEP_DIR}/decode_sweep_summary.csv"
