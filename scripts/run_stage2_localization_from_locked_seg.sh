#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ABIODUN_DIR}"

CONFIG="${CONFIG:-${ABIODUN_DIR}/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml}"
DATASET="${DATASET:-${ABIODUN_DIR}/fusion_training_data/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_merged_stride1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ABIODUN_DIR}/experiments/moving_ego_r100k_r4_tw2_stage2_localization_from_locked_seg_20260624}"
SEG_CHECKPOINT="${SEG_CHECKPOINT:-${ABIODUN_DIR}/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_segonly_ablation_20260624_segonly_lovasz05_personselect_pat20_bs24/checkpoints/segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20/best.pt}"
OBJECT_INIT_CHECKPOINT="${OBJECT_INIT_CHECKPOINT:-${ABIODUN_DIR}/experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_fusion_train_20260624_r100k_r4_tw2_pilot/checkpoints/moving_ego_radarpps100000_bboxsupport_r4_tw2_2loops_cap3200_768x432_lr1e-4_bs2/best.pt}"

TRIAL_NAME="${TRIAL_NAME:-stage2_lockedseg_objhead_radius4_lowfuse_bs24_locdim}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0002}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
TRAINING_BUDGET_HOURS="${TRAINING_BUDGET_HOURS:-5.0}"
HEATMAP_RADIUS_PX="${HEATMAP_RADIUS_PX:-4}"
SELECTION_SCORE_MODE="${SELECTION_SCORE_MODE:-loc_dim_loss}"
PYTHON="${PYTHON:-python3}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export PYTHONPATH="${ABIODUN_DIR}/pole_lraspp_multimodal_fusion:${ABIODUN_DIR}:${PYTHONPATH:-}"

if [[ ! -d "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "${SEG_CHECKPOINT}" ]]; then
  echo "SEG checkpoint not found: ${SEG_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_INIT_CHECKPOINT}" ]]; then
  echo "Object init checkpoint not found: ${OBJECT_INIT_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_DIR}"
if [[ -L "${EXPERIMENT_DIR}/dataset" ]]; then
  unlink "${EXPERIMENT_DIR}/dataset"
elif [[ -e "${EXPERIMENT_DIR}/dataset" ]]; then
  echo "${EXPERIMENT_DIR}/dataset exists and is not a symlink; refusing to replace it." >&2
  exit 1
fi
ln -s "${DATASET}" "${EXPERIMENT_DIR}/dataset"

trial_json="$(printf '{"name":"%s","optimizer":"adamw","lr":%s,"weight_decay":%s,"augment_strength":"strong","input_size":[768,432],"batch_size":%s,"num_workers":%s,"prefetch_factor":%s,"persistent_workers":%s,"epochs":%s,"init_rgb_checkpoint":"%s","init_object_checkpoint":"%s","freeze_backbone":true,"freeze_classifier":true,"freeze_object_head":false,"selection_score_mode":"%s","early_stop_patience":%s,"loss_weights":{"segmentation":0.0,"object_total":1.0},"object_heads":{"heatmap_radius_px":%s,"fuse_low_feature":true}}' \
  "${TRIAL_NAME}" "${LR}" "${WEIGHT_DECAY}" "${BATCH_SIZE}" "${NUM_WORKERS}" "${PREFETCH_FACTOR}" "${PERSISTENT_WORKERS}" "${EPOCHS}" "${SEG_CHECKPOINT}" "${OBJECT_INIT_CHECKPOINT}" "${SELECTION_SCORE_MODE}" "${EARLY_STOP_PATIENCE}" "${HEATMAP_RADIUS_PX}")"

echo "Starting stage-2 localization:"
echo "  experiment: ${EXPERIMENT_DIR}"
echo "  seg checkpoint: ${SEG_CHECKPOINT}"
echo "  object init checkpoint: ${OBJECT_INIT_CHECKPOINT}"
echo "  trial: ${trial_json}"

"${PYTHON}" -m pole_lraspp_multimodal_fusion.train_fusion \
  --config "${CONFIG}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --trial-json "${trial_json}" \
  --training-budget-hours "${TRAINING_BUDGET_HOURS}"

echo "Stage-2 training complete. Checkpoint:"
echo "${EXPERIMENT_DIR}/checkpoints/${TRIAL_NAME}/best.pt"
