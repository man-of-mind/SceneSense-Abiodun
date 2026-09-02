from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "splitfusion_fcos_hybrid_q_transport_v1"
FORMAT_VERSION = 1

# Frozen perception binding. Hybrid-q is transport-only downstream of this lock.
PERCEPTION_LOCK_RELPATH = (
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/"
    "PERCEPTION_FORWARD_LOCK_P025_V1.json"
)
FROZEN_CHECKPOINT_SHA256 = (
    "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"
)

# Transported fused C2 tensor.
SPLIT_CHANNELS = 256
SPLIT_HEIGHT = 112
SPLIT_WIDTH = 192
SPLIT_SHAPE = (SPLIT_CHANNELS, SPLIT_HEIGHT, SPLIT_WIDTH)
SPLIT_CELLS = SPLIT_HEIGHT * SPLIT_WIDTH  # 21504
SPLIT_PAYLOAD_FP32_BYTES = 22020096

# Registered spatial drop fractions.
REGISTERED_Q_VALUES = (0.00, 0.30, 0.50, 0.70, 0.90, 0.98)

# Ranker shape constants.
RANKER_HIDDEN_CHANNELS = 8
RANKER_PARAMETER_COUNT = 2145


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


def repository_root() -> Path:
    """abiodun/ working root, derived from this file's location."""
    return Path(__file__).resolve().parents[3]


def perception_lock_path() -> Path:
    return repository_root() / PERCEPTION_LOCK_RELPATH


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
    if int(architecture["split_payload_fp32_bytes"]) != SPLIT_PAYLOAD_FP32_BYTES:
        raise ValueError("perception lock split payload bytes do not match hybrid-q contract")
    if lock["base_checkpoint"]["sha256"] != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("perception lock checkpoint sha256 does not match hybrid-q contract")
    return lock


def registered_keep_table(cells: int = SPLIT_CELLS) -> dict[float, int]:
    return {q: keep_count(q, cells) for q in REGISTERED_Q_VALUES}
