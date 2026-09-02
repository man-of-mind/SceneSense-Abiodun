from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "splitfusion_fcos_hybrid_q_transport_v1"
FORMAT_VERSION = 1
LOCKED_CONFIG_FILENAME = "locked_config.json"

# Frozen perception binding. Hybrid-q is transport-only downstream of this lock.
PERCEPTION_LOCK_RELPATH = (
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/"
    "PERCEPTION_FORWARD_LOCK_P025_V1.json"
)
PERCEPTION_LOCK_SHA256 = (
    "86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1"
)
FROZEN_CHECKPOINT_SHA256 = (
    "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"
)

# Transported fused C2 tensor. These are the only accepted production dimensions.
SPLIT_CHANNELS = 256
SPLIT_HEIGHT = 112
SPLIT_WIDTH = 192
SPLIT_SHAPE = (SPLIT_CHANNELS, SPLIT_HEIGHT, SPLIT_WIDTH)
SPLIT_SPATIAL_SHAPE = (SPLIT_HEIGHT, SPLIT_WIDTH)
SPLIT_CELLS = SPLIT_HEIGHT * SPLIT_WIDTH  # 21504

# Payload accounting. The raw FP32 tensor size is a *reference* only; the primary
# hybrid-q compression denominator is the framed q=0 payload.
RAW_FP32_REFERENCE_BYTES = 22020096
FRAMED_Q0_PAYLOAD_BYTES = 22020140
HEADER_OVERHEAD_BYTES = FRAMED_Q0_PAYLOAD_BYTES - RAW_FP32_REFERENCE_BYTES  # 44

# Registered spatial drop fractions.
REGISTERED_Q_VALUES = (0.00, 0.30, 0.50, 0.70, 0.90, 0.98)
PARITY_Q = 0.00
Q_AWARE_TRAINING_CYCLE = (0.30, 0.50, 0.70)
EVALUATION_STRESS_Q_VALUES = (0.90, 0.98)

# Ranker shape and initialization. The final 1x1 conv has no bias: a global scalar
# score offset cannot change cell ranking, so it is unidentifiable under the mask.
RANKER_HIDDEN_CHANNELS = 8
RANKER_PARAMETER_COUNT = 2144
RANKER_INIT_SEED = 20260829

# Locked teacher supervision: exactly the registered frozen-model loss groups.
TEACHER_GROUPS = ("D", "G", "S", "A")
TEACHER_GROUP_DEFINITIONS = {
    "D": "FCOS classification, box regression and centerness",
    "G": "registered geometry loss",
    "S": "registered semantic loss",
    "A": "registered dense-depth and radar-consistency loss",
}
TEACHER_NORMALIZATION = "l1"
TEACHER_GROUP_COMBINATION = "equal_weight"

# Locked distillation stage.
DISTILLATION_TEMPERATURE = 1.0
DISTILLATION_EPOCHS = 4
DISTILLATION_LOSS = "listwise_soft_cross_entropy"

# Locked q-aware stage.
Q_AWARE_EPOCHS = 8
Q_AWARE_DISTILLATION_WEIGHT = 0.1
Q_AWARE_OBJECTIVE = (
    "mean(valid masked task loss / frozen train-reference median for that task) "
    "+ 0.1 * distillation loss"
)
REFERENCE_MEDIAN_SOURCE = "fit_train"

# Locked train fit/holdout split (Phase 4). The partition is by episode over the
# registered `split == "train"` manifest rows, taken in manifest order. The two
# holdout episodes are reserved for checkpoint selection: they must never
# contribute to the frozen reference medians or to any optimizer step. Phase 3
# touched some holdout frames in a disposable qualification; that use is not
# carried forward.
TRAIN_MANIFEST_SHA256 = (
    "5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2"
)
TRAIN_TOTAL_FRAMES = 16827
TRAIN_FIT_FRAMES = 13543
TRAIN_HOLDOUT_FRAMES = 3284
TRAIN_HOLDOUT_EPISODES = (
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
)
TRAIN_FIT_EPISODES = (
    "canonical_v3_01_train_30_30_s501_tm1501",
    "canonical_v3_02_train_50_50_s502_tm1502",
    "extra_v3_09_train_30_30_s801_tm1801",
    "extra_v3_10_train_50_50_s802_tm1802",
    "extra_v3_11_train_30_30_s803_tm1803",
    "extra_v3_12_train_50_50_s804_tm1804",
    "extra_v3_13_train_30_30_s805_tm1805",
    "extra_v3_14_train_50_50_s806_tm1806",
)
TRAIN_FIT_SAMPLE_ID_SHA256 = (
    "3e20cceeec48718b7df763f95bf873f3febbb435256c36104996631f86fa252e"
)
TRAIN_HOLDOUT_SAMPLE_ID_SHA256 = (
    "8c7c4cc617626b31df7bcb68f89bf72649d05e1934e316002b2dd8a8d5754f0a"
)
SPLIT_LABELS = ("fit", "holdout")

# Locked teacher-cache generation parameters. The cache batch size is part of the
# contract because batch construction determines the per-batch task-loss values
# that the frozen reference medians are taken over.
TEACHER_CACHE_BATCH_SIZE = 16
TEACHER_CACHE_SHARD_FRAMES = 256
TEACHER_CACHE_MIN_GPU_TOTAL_GIB = 30.0
TEACHER_CACHE_MAP_DTYPE = "float32"
TEACHER_CACHE_COMPRESSION = "none"

# Locked optimization.
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = "constant"
GRAD_CLIP_GLOBAL_NORM = 5.0
CHECKPOINT_EPOCHS = (4, 8, 12)
AUGMENTATION_ENABLED = False

# Locked straight-through surrogate.
STRAIGHT_THROUGH_TEMPERATURE = 1.0
STRAIGHT_THROUGH_BOUNDARY = "midpoint_lowest_retained_highest_dropped"


def _q_to_e4(q: float) -> int:
    """Exact integer encoding of q in ten-thousandths, used in the wire header."""
    scaled = q * 10000.0
    rounded = int(math.floor(scaled + 0.5))
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(f"q={q!r} is not representable in ten-thousandths")
    return rounded


def is_registered_q(q: float) -> bool:
    try:
        e4 = _q_to_e4(float(q))
    except (TypeError, ValueError):
        return False
    return e4 in {_q_to_e4(value) for value in REGISTERED_Q_VALUES}


def drop_count(q: float, cells: int = SPLIT_CELLS) -> int:
    """drop_count = floor(q * N + 0.5)."""
    return int(math.floor(float(q) * int(cells) + 0.5))


def keep_count(q: float, cells: int = SPLIT_CELLS) -> int:
    """keep_count = N - drop_count."""
    return int(cells) - drop_count(q, cells)


def mask_byte_count(cells: int = SPLIT_CELLS) -> int:
    """Fixed-order bitmask length: one bit per spatial cell, MSB-first per byte."""
    return (int(cells) + 7) // 8


def ranker_mac_count(height: int = SPLIT_HEIGHT, width: int = SPLIT_WIDTH) -> int:
    """Multiply-accumulate count of the three ranker convolutions (bias excluded)."""
    cells = int(height) * int(width)
    hidden = RANKER_HIDDEN_CHANNELS
    pointwise_in = SPLIT_CHANNELS * hidden * cells
    depthwise = hidden * 3 * 3 * cells
    pointwise_out = hidden * 1 * cells
    return pointwise_in + depthwise + pointwise_out


def framed_payload_ratio(payload_bytes: int) -> float:
    """Primary hybrid-q compression ratio: framed q payload / framed q=0 payload."""
    return int(payload_bytes) / FRAMED_Q0_PAYLOAD_BYTES


def raw_fp32_ratio(payload_bytes: int) -> float:
    """Secondary diagnostic against the unframed raw FP32 tensor size."""
    return int(payload_bytes) / RAW_FP32_REFERENCE_BYTES


def repository_root() -> Path:
    """abiodun/ working root, derived from this file's location."""
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    return Path(__file__).resolve().parent


def perception_lock_path() -> Path:
    return repository_root() / PERCEPTION_LOCK_RELPATH


def locked_config_path() -> Path:
    return package_root() / LOCKED_CONFIG_FILENAME


def load_perception_lock() -> dict[str, Any]:
    """Read the frozen p025 forward lock and confirm the bound split contract.

    This reads JSON metadata only; it never loads the frozen checkpoint.
    """
    path = perception_lock_path()
    with path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    architecture = lock["architecture"]
    if tuple(architecture["split_shape"]) != SPLIT_SHAPE:
        raise ValueError("perception lock split_shape does not match hybrid-q contract")
    if int(architecture["split_payload_fp32_bytes"]) != RAW_FP32_REFERENCE_BYTES:
        raise ValueError("perception lock split payload bytes do not match hybrid-q contract")
    if lock["base_checkpoint"]["sha256"] != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("perception lock checkpoint sha256 does not match hybrid-q contract")
    return lock


def registered_keep_table(cells: int = SPLIT_CELLS) -> dict[float, int]:
    return {q: keep_count(q, cells) for q in REGISTERED_Q_VALUES}


def sample_id_digest(sample_ids: Any) -> str:
    """Order-sensitive identity digest of a frame list: sha256 of newline-terminated ids."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def split_for_episode(episode_id: str) -> str:
    """Locked fit/holdout label of a registered training episode."""
    if episode_id in TRAIN_HOLDOUT_EPISODES:
        return "holdout"
    if episode_id in TRAIN_FIT_EPISODES:
        return "fit"
    raise ValueError(f"episode {episode_id!r} is not a registered training episode")


def load_locked_config() -> dict[str, Any]:
    """Read locked_config.json and fail closed on any drift from these constants."""
    with locked_config_path().open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if config["schema"] != CONTRACT_SCHEMA:
        raise ValueError("locked_config schema mismatch")

    binding = config["perception_binding"]
    if binding["perception_forward_lock_path"] != PERCEPTION_LOCK_RELPATH:
        raise ValueError("locked_config perception lock path drift")
    if binding["perception_forward_lock_sha256"] != PERCEPTION_LOCK_SHA256:
        raise ValueError("locked_config perception lock sha256 drift")
    if binding["checkpoint_sha256"] != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("locked_config checkpoint sha256 drift")

    c2 = config["c2_contract"]
    if tuple(c2["shape"]) != SPLIT_SHAPE or int(c2["cells"]) != SPLIT_CELLS:
        raise ValueError("locked_config C2 contract drift")
    if c2["dtype"] != "float32" or c2["spatial_index_order"] != "row_major":
        raise ValueError("locked_config C2 dtype/index-order drift")
    if int(c2["raw_fp32_reference_bytes"]) != RAW_FP32_REFERENCE_BYTES:
        raise ValueError("locked_config raw FP32 reference drift")

    wire = config["wire_format"]
    if int(wire["version"]) != FORMAT_VERSION:
        raise ValueError("locked_config wire version drift")
    if int(wire["header_bytes"]) != HEADER_OVERHEAD_BYTES:
        raise ValueError("locked_config header size drift")
    if int(wire["framed_q0_payload_bytes"]) != FRAMED_Q0_PAYLOAD_BYTES:
        raise ValueError("locked_config framed q=0 payload drift")

    q_block = config["q_contract"]
    table = {float(entry["q"]): int(entry["keep_count"]) for entry in q_block["values"]}
    if table != registered_keep_table():
        raise ValueError("locked_config q keep-count table drift")

    ranker = config["ranker"]
    if int(ranker["parameter_count"]) != RANKER_PARAMETER_COUNT:
        raise ValueError("locked_config ranker parameter count drift")
    if int(ranker["init_seed"]) != RANKER_INIT_SEED:
        raise ValueError("locked_config ranker seed drift")
    if int(ranker["mac_count_112x192"]) != ranker_mac_count():
        raise ValueError("locked_config ranker MAC count drift")

    training = config["training"]
    if tuple(training["teacher"]["groups"]) != TEACHER_GROUPS:
        raise ValueError("locked_config teacher group drift")
    if training["teacher"]["normalization"] != TEACHER_NORMALIZATION:
        raise ValueError("locked_config teacher normalization drift")
    if float(training["distillation"]["temperature"]) != DISTILLATION_TEMPERATURE:
        raise ValueError("locked_config distillation temperature drift")
    if int(training["distillation"]["epochs"]) != DISTILLATION_EPOCHS:
        raise ValueError("locked_config distillation epoch drift")
    if int(training["q_aware"]["epochs"]) != Q_AWARE_EPOCHS:
        raise ValueError("locked_config q-aware epoch drift")
    if tuple(training["q_aware"]["cycle"]) != Q_AWARE_TRAINING_CYCLE:
        raise ValueError("locked_config q-aware cycle drift")
    if float(training["q_aware"]["distillation_weight"]) != Q_AWARE_DISTILLATION_WEIGHT:
        raise ValueError("locked_config q-aware distillation weight drift")
    if training["q_aware"]["reference_median_source"] != REFERENCE_MEDIAN_SOURCE:
        raise ValueError("locked_config reference median source drift")

    optimization = training["optimization"]
    if optimization["optimizer"] != OPTIMIZER:
        raise ValueError("locked_config optimizer drift")
    if float(optimization["learning_rate"]) != LEARNING_RATE:
        raise ValueError("locked_config learning rate drift")
    if float(optimization["weight_decay"]) != WEIGHT_DECAY:
        raise ValueError("locked_config weight decay drift")
    if optimization["lr_schedule"] != LR_SCHEDULE:
        raise ValueError("locked_config lr schedule drift")
    if float(optimization["grad_clip_global_norm"]) != GRAD_CLIP_GLOBAL_NORM:
        raise ValueError("locked_config gradient clip drift")
    if tuple(optimization["checkpoint_epochs"]) != CHECKPOINT_EPOCHS:
        raise ValueError("locked_config checkpoint epoch drift")
    if bool(optimization["augmentation"]) != AUGMENTATION_ENABLED:
        raise ValueError("locked_config augmentation drift")

    surrogate = training["straight_through"]
    if float(surrogate["temperature"]) != STRAIGHT_THROUGH_TEMPERATURE:
        raise ValueError("locked_config straight-through temperature drift")
    if surrogate["boundary"] != STRAIGHT_THROUGH_BOUNDARY:
        raise ValueError("locked_config straight-through boundary drift")

    split = config["train_split"]
    if split["manifest_sha256"] != TRAIN_MANIFEST_SHA256:
        raise ValueError("locked_config train manifest sha256 drift")
    if int(split["total_frames"]) != TRAIN_TOTAL_FRAMES:
        raise ValueError("locked_config train frame count drift")
    if int(split["fit_frames"]) != TRAIN_FIT_FRAMES:
        raise ValueError("locked_config fit frame count drift")
    if int(split["holdout_frames"]) != TRAIN_HOLDOUT_FRAMES:
        raise ValueError("locked_config holdout frame count drift")
    if int(split["fit_frames"]) + int(split["holdout_frames"]) != TRAIN_TOTAL_FRAMES:
        raise ValueError("locked_config split does not partition the train set")
    if tuple(split["holdout_episodes"]) != TRAIN_HOLDOUT_EPISODES:
        raise ValueError("locked_config holdout episode drift")
    if tuple(split["fit_episodes"]) != TRAIN_FIT_EPISODES:
        raise ValueError("locked_config fit episode drift")
    if set(TRAIN_FIT_EPISODES) & set(TRAIN_HOLDOUT_EPISODES):
        raise ValueError("locked_config fit and holdout episodes overlap")
    if split["fit_sample_id_sha256"] != TRAIN_FIT_SAMPLE_ID_SHA256:
        raise ValueError("locked_config fit frame identity drift")
    if split["holdout_sample_id_sha256"] != TRAIN_HOLDOUT_SAMPLE_ID_SHA256:
        raise ValueError("locked_config holdout frame identity drift")

    cache = config["teacher_cache"]
    if int(cache["batch_size"]) != TEACHER_CACHE_BATCH_SIZE:
        raise ValueError("locked_config teacher-cache batch size drift")
    if int(cache["shard_frames"]) != TEACHER_CACHE_SHARD_FRAMES:
        raise ValueError("locked_config teacher-cache shard size drift")
    if cache["map_dtype"] != TEACHER_CACHE_MAP_DTYPE:
        raise ValueError("locked_config teacher-cache map dtype drift")
    if cache["compression"] != TEACHER_CACHE_COMPRESSION:
        raise ValueError("locked_config teacher-cache compression drift")
    if bool(cache["augmentation"]) != AUGMENTATION_ENABLED:
        raise ValueError("locked_config teacher-cache augmentation drift")

    if int(config["seed"]) != RANKER_INIT_SEED:
        raise ValueError("locked_config seed drift")
    return config


# ---------------------------------------------------------------------------
# Phase 5: ranker training and train-holdout checkpoint selection
# ---------------------------------------------------------------------------

# Bound Phase-4 artifact. The manifest hash binds all 66 shard hashes, so verifying
# the manifest and then every shard against it is one closed identity chain.
TEACHER_CACHE_RELPATH = (
    "experiments/splitfusion_fcos_hybrid_q_v1/20260901_180439_phase4_teacher_cache"
)
TEACHER_CACHE_MANIFEST_SHA256 = (
    "e1ef600eb83c1924a52c37dbdbf0b435990eaaae4f1db1b0e07d9b1acd7fc273"
)
FIT_REFERENCE_MEDIANS_SHA256 = (
    "f91570c7e9b7b895b9c02ae5831c382b68f5a4f6b7a20fa9aed6c5bcf552729e"
)
LOCKED_CONFIG_SHA256 = (
    "b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6"
)
TEACHER_CACHE_SHARD_COUNT = 66

# The frozen Phase-4 fit-partition reference medians the q-aware objective divides by.
# Held here as an independent check on fit_reference_medians.json, not as its source.
FROZEN_FIT_REFERENCE_MEDIANS = {
    "D": 0.7242346405982971,
    "G": 0.10442040860652924,
    "S": 0.09630495309829712,
    "A": 0.013652884401381016,
}

# Locked Phase-5 training schedule.
TRAINING_EPOCHS = DISTILLATION_EPOCHS + Q_AWARE_EPOCHS  # 12
TRAINING_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
DROP_LAST_TRAINING_BATCH = False
EPOCH_SHUFFLE_SEED_BASE = RANKER_INIT_SEED  # generator seed = base + epoch


def epoch_shuffle_seed(epoch: int) -> int:
    """Deterministic epoch-specific shuffle seed: 20260829 + epoch, epochs 1..12."""
    value = int(epoch)
    if not 1 <= value <= TRAINING_EPOCHS:
        raise ValueError(f"epoch {epoch!r} is outside 1..{TRAINING_EPOCHS}")
    return EPOCH_SHUFFLE_SEED_BASE + value


def stage_for_epoch(epoch: int) -> str:
    """Epochs 1-4 are distillation (stage A); 5-12 are q-aware preservation (stage B)."""
    value = int(epoch)
    if not 1 <= value <= TRAINING_EPOCHS:
        raise ValueError(f"epoch {epoch!r} is outside 1..{TRAINING_EPOCHS}")
    return "distillation" if value <= DISTILLATION_EPOCHS else "q_aware"


# Locked Phase-5 holdout evaluation. q=0.90 and q=0.98 are NOT evaluated in Phase 5.
HOLDOUT_BASELINE_Q = 0.00
HOLDOUT_EVALUATION_Q_VALUES = Q_AWARE_TRAINING_CYCLE  # 0.30, 0.50, 0.70
HOLDOUT_CANDIDATE_EPOCHS = CHECKPOINT_EPOCHS  # 4, 8, 12

# Existing frozen scoring settings, restated so drift fails closed. None of these is
# a Phase-5 choice: the vehicle score point is the canonical 0.20 applied to the
# calibrated service score (equivalently base >= 0.5224518340619145), and the person
# service threshold is the locked p025 output filter.
PRIMARY_CONTRACT = "v010"
VEHICLE_SCORE_THRESHOLD = 0.20
PERSON_SERVICE_SCORE_THRESHOLD = 0.25
PERSON_AVO_THRESHOLD = 0.65
PERSON_LONG_RANGE_BINS = ("20_30m", "30_40m")
HOLDOUT_AVO_TABLE_RELPATH = (
    "experiments/splitfusion_fcos_person_p025_calibration_v1/"
    "holdout_actor_volume_observability_table.csv"
)
HOLDOUT_AVO_QUALIFIED_ACTOR_FRAMES = 4703
HOLDOUT_AVO_OBSERVABLE_ACTOR_FRAMES = 2556
HOLDOUT_STRUCTURAL_ACTOR_FRAMES = 21054
HOLDOUT_RAW_PERSON_ACTOR_FRAMES = 25757

# Registered holdout preservation gates, relative to the exact q=0 holdout baseline.
# "loss" gates are baseline - candidate; "increase" gates are candidate - baseline.
HOLDOUT_PRESERVATION_GATES = (
    ("vehicle_precision", "loss", 0.01),
    ("vehicle_recall", "loss", 0.01),
    ("vehicle_f1", "loss", 0.01),
    ("person_avo_precision", "loss", 0.015),
    ("person_avo_recall", "loss", 0.015),
    ("person_avo_f1", "loss", 0.015),
    ("vehicle_xy_mae_m", "increase", 0.05),
    ("person_avo_xy_mae_m", "increase", 0.05),
    ("vehicle_iou", "loss", 0.01),
    ("person_box_mask_iou", "loss", 0.01),
    ("foreground_miou", "loss", 0.01),
    ("person_avo_recall_20_40m", "loss", 0.03),
)
PROTECTED_METRICS = tuple(name for name, _direction, _bound in HOLDOUT_PRESERVATION_GATES)

PHASE5_TERMINAL_SELECTED = "HYBRID_Q_PHASE5_CHECKPOINT_SELECTED"
PHASE5_TERMINAL_NOT_SAFE = "ROI_DROP_NOT_SAFE_ON_TRAIN_HOLDOUT"
PHASE5_TERMINAL_FAILED = "HYBRID_Q_PHASE5_FAILED"


def gate_degradation(metric: str, baseline: float, candidate: float) -> float:
    """Signed degradation of one protected metric: positive is worse than baseline."""
    for name, direction, _bound in HOLDOUT_PRESERVATION_GATES:
        if name == metric:
            return (
                float(baseline) - float(candidate)
                if direction == "loss"
                else float(candidate) - float(baseline)
            )
    raise ValueError(f"{metric!r} is not a registered protected metric")


# ---------------------------------------------------------------------------
# Phase 6: fixed-validation accuracy-payload curve measurement
# ---------------------------------------------------------------------------
#
# Phase 6 is a *measurement* phase. It does not train, tune, recalibrate, select a
# checkpoint or change any threshold. It measures the complete validation
# accuracy-payload curve of the one stable Phase-5 checkpoint over the whole
# registered q ladder, so that each q becomes a characterized transport action.

VALIDATION_FRAMES = 3345
VALIDATION_EPISODES = (
    "canonical_v3_05_val_30_30_s601_tm1601",
    "canonical_v3_06_val_50_50_s602_tm1602",
)
VALIDATION_BASELINE_Q = 0.00

# The Phase-5 holdout runner evaluated only the q-aware training cycle. Phase 6
# measures the whole registered ladder, including the two evaluation-stress rungs
# that Phase 5 deliberately left unmeasured. Nothing else about q semantics moves:
# 0.90 and 0.98 were already registered q values, so `guards.require_valid_q`,
# `selection` and `codec` accept them unchanged.
VALIDATION_EVALUATION_Q_VALUES = Q_AWARE_TRAINING_CYCLE + EVALUATION_STRESS_Q_VALUES

# The single stable Phase-5 checkpoint: end of the distillation stage, before the
# q-aware stage. Epochs 8 and 12 are excluded because the q-aware stage
# demonstrably diverged on the reserved train-holdout (worst absolute protected
# degradation 0.2402 -> 0.8235 versus 0.0379 at epoch 4). Phase 6 must not load
# or evaluate them.
VALIDATION_RANKER_EPOCH = 4
VALIDATION_RANKER_RELPATH = (
    "experiments/splitfusion_fcos_hybrid_q_v1/"
    "20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt"
)
VALIDATION_RANKER_SHA256 = (
    "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5"
)
VALIDATION_EXCLUDED_RANKER_EPOCHS = (8, 12)
VALIDATION_EXCLUDED_RANKER_REASON = (
    "the Phase-5 q-aware stage diverged; those checkpoints are not reopened and "
    "the training failure is unchanged by this measurement phase"
)
VALIDATION_RANKER_STAGE = "distillation_only"

# Frozen p025 q=0 validation result. Phase 6 reuses it verbatim and never reruns
# q=0 inference. The existing prediction set is re-scored by the identical Phase-6
# scoring functions purely to prove the q>0 rows are comparable; that is scoring,
# not inference, and the published values below must be reproduced exactly.
FROZEN_Q0_PREDICTION_ROOT = (
    "experiments/splitfusion_fcos_service_candidate_v1/predictions"
)
FROZEN_Q0_DETECTIONS_SHA256 = (
    "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260"
)
FROZEN_Q0_INFERENCE_MANIFEST_SHA256 = (
    "3b930b9ad4bd6e4f0b93d269fdcb599facc145fd988bb67159581230bef38153"
)
FROZEN_Q0_SEGMENTATION_MANIFEST_SHA256 = (
    "52d6131e674adbfc75d8ea449a4f3be54a51fd993d3cb818cca23e8d759078f7"
)
FROZEN_Q0_EVALUATION_SHA256 = (
    "81fb31b4c3423a3a381946afdf1012d435116bbba7fc75d7cb5553225a616773"
)
VALIDATION_AVO_TABLE_RELPATH = (
    "experiments/actor_volume_observability_model_comparison_v1/"
    "20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv"
)
VALIDATION_AVO_TABLE_SHA256 = (
    "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386"
)
P025_VALIDATION_CONFIRMATION_RELPATH = (
    "experiments/splitfusion_fcos_person_p025_calibration_v1/"
    "validation_confirmation.json"
)
P025_VALIDATION_CONFIRMATION_SHA256 = (
    "ce1bc88736064d8dba59a3bb578ab47db2d0400857f704af2c591701f4dd403b"
)

# Published frozen p025 q=0 validation values, restated here so any scoring-path
# drift fails closed instead of silently rebasing the curve. Vehicle and
# segmentation come from the frozen q=0 evaluation (the p025 person filter
# provably touches no vehicle field); person comes from the p025 confirmation.
FROZEN_Q0_VALIDATION_METRICS = {
    "vehicle_precision": 0.9315917644454283,
    "vehicle_recall": 0.8684346300691363,
    "vehicle_f1": 0.8989052069425901,
    "vehicle_xy_mae_m": 0.4786754207718958,
    "vehicle_iou": 0.8990128473391599,
    "person_box_mask_iou": 0.5278940800907954,
    "foreground_miou": 0.7134534637149776,
    "person_avo_precision": 0.7041866849691146,
    "person_avo_recall": 0.7132429614181439,
    "person_avo_f1": 0.708685891901226,
    "person_avo_xy_mae_m": 0.8121813463526099,
    "person_canonical_precision": 0.7966862271315154,
    "person_canonical_recall": 0.5960743801652892,
    "person_canonical_f1": 0.6819323386024523,
    "person_canonical_xy_mae_m": 0.8395157289651327,
}
# 20-40 m person recall is the exact union of the two published long-range bins:
# (731 + 277) / (1004 + 741).
FROZEN_Q0_PERSON_RECALL_20_40M_TP = 1008
FROZEN_Q0_PERSON_RECALL_20_40M_GT = 1745

# The nine original absolute service targets, verbatim from the registered
# evaluator `splitfusion_fcos_r50_fpn_p2_p7_v1/evaluate.py:service`. These are
# absolute deployment targets and are *not* a Phase-6 choice.
ABSOLUTE_SERVICE_TARGETS = (
    ("vehicle_precision", 0.80, "higher"),
    ("vehicle_recall", 0.85, "higher"),
    ("person_precision", 0.80, "higher"),
    ("person_recall", 0.80, "higher"),
    ("vehicle_xy_mae_m", 1.00, "lower"),
    ("person_xy_mae_m", 1.20, "lower"),
    ("vehicle_iou", 0.85, "higher"),
    ("person_box_mask_iou", 0.50, "higher"),
    ("foreground_miou", 0.675, "higher"),
)
FROZEN_Q0_SERVICE_PASS_COUNT = 7
FROZEN_Q0_FAILED_SERVICE_GATES = ("person_precision", "person_recall")

# Descriptive action-profile classification. Registered *before* the measurement so
# the labels describe fixed results instead of being fitted to them. Evaluated as a
# priority cascade: the first matching rule wins. Every bound is a degradation
# relative to the exact frozen q=0 validation row; none of these is a pass/fail
# acceptance gate, and no q is discarded for landing in a lower band.
VALIDATION_PROFILE_CASCADE = (
    (
        "unusable",
        "person AVO F1 or vehicle F1 collapses by more than 0.20 absolute, or at "
        "most three of the nine absolute service targets survive",
    ),
    (
        "accuracy-first",
        "every registered near-lossless preservation gate passes",
    ),
    (
        "balanced",
        "person AVO F1 loss <= 0.05, vehicle F1 loss <= 0.02 and foreground mIoU "
        "loss <= 0.02",
    ),
    (
        "localization-preserving/segmentation-reduced",
        "both XY MAE increases stay within 0.10 m while foreground mIoU loss "
        "exceeds 0.02",
    ),
    (
        "emergency-bandwidth",
        "still finite and scientifically usable, but detection or segmentation "
        "quality is materially reduced",
    ),
)
PROFILE_UNUSABLE_F1_COLLAPSE = 0.20
PROFILE_UNUSABLE_MAX_SERVICE_PASS = 3
PROFILE_BALANCED_PERSON_F1_LOSS = 0.05
PROFILE_BALANCED_VEHICLE_F1_LOSS = 0.02
PROFILE_BALANCED_SEGMENTATION_LOSS = 0.02
PROFILE_LOCALIZATION_XY_INCREASE = 0.10

PHASE6_TERMINAL = "HYBRID_Q_PHASE6_VALIDATION_CURVE_COMPLETE"
PHASE6_SCHEMA = "splitfusion_fcos_hybrid_q_phase6_validation_curve_v1"


def continuous_keep_count(q: float, cells: int = SPLIT_CELLS) -> int:
    """Readiness-only keep count for an arbitrary future q: K(q) = round((1-q)*N).

    This is the deterministic keep-count convention a future continuous-q agent
    would use, bounded to the supported range 1..N. It is documentation and a
    constructibility diagnostic only: the production discrete contract still runs
    through `keep_count`/`drop_count`, and this function is never used to serve a
    transport payload. Rounding is half-up, matching `drop_count`.
    """
    value = float(q)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"q must satisfy 0 <= q < 1, got {q!r}")
    total = int(cells)
    keep = int(math.floor((1.0 - value) * total + 0.5))
    return max(1, min(total, keep))


def continuous_keep_count_agrees_with_registered(cells: int = SPLIT_CELLS) -> bool:
    """True when K(q) = round((1-q)*N) reproduces every registered keep count."""
    return all(
        continuous_keep_count(q, cells) == keep_count(q, cells)
        for q in REGISTERED_Q_VALUES
        if q > 0.0
    )


def snap_continuous_q(q: float) -> float:
    """Snap a requested continuous q down to the nearest validated, less-aggressive q.

    Until a denser validation sweep exists, an unmeasured q must not be served on
    the assumption that its accuracy interpolates between neighbours. The
    conservative choice is the largest registered q that is <= the request, which
    drops no more cells than requested and whose accuracy has been measured.
    """
    value = float(q)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"q must satisfy 0 <= q < 1, got {q!r}")
    return max(item for item in REGISTERED_Q_VALUES if item <= value)


def absolute_service_gates(metrics: Any) -> dict[str, Any]:
    """The nine original absolute service targets, evaluated verbatim.

    Mirrors `splitfusion_fcos_r50_fpn_p2_p7_v1/evaluate.py:service`, including its
    attainment-ratio convention and its treatment of a non-finite metric as a
    failure with an undefined ratio.
    """
    rows: dict[str, Any] = {}
    ratios: list[Any] = []
    for name, target, direction in ABSOLUTE_SERVICE_TARGETS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        finite = math.isfinite(value)
        passed = finite and (
            value >= target if direction == "higher" else value <= target
        )
        ratio = (
            (value / target if direction == "higher" else target / max(value, 1e-12))
            if finite
            else None
        )
        rows[name] = {
            "value": value if finite else None,
            "target": target,
            "direction": direction,
            "passed": passed,
            "attainment_ratio": ratio,
        }
        ratios.append(ratio)
    return {
        "targets": rows,
        "pass_count": sum(row["passed"] for row in rows.values()),
        "all_pass": all(row["passed"] for row in rows.values()),
        "failed": sorted(name for name, row in rows.items() if not row["passed"]),
        "minimum_attainment_ratio": (
            min(ratios) if all(value is not None for value in ratios) else None
        ),
    }
