from __future__ import annotations

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

# Ranker shape and initialization.
RANKER_HIDDEN_CHANNELS = 8
RANKER_PARAMETER_COUNT = 2145
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

    if int(config["seed"]) != RANKER_INIT_SEED:
        raise ValueError("locked_config seed drift")
    return config
