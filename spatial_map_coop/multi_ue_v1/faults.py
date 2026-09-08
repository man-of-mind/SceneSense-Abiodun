"""Deterministic shadow faults for the live two-UE map demonstration.

Faults are applied only to a separate shadow service.  The nominal map always
receives the original, finite SplitFusion outputs.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


FAULT_CASES = (
    "DUPLICATE_IDENTICAL",
    "IDENTITY_CONFLICT",
    "NONFINITE",
    "STALE",
    "XY_BIAS",
)
STALE_OFFSET_NS = 1_000_000_000
XY_BIAS_M = 8.0


def deterministic_fault_case(pair_index: int) -> str:
    """Return the registered case for a zero-based processed pair."""

    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError("pair_index must be a nonnegative integer")
    return FAULT_CASES[pair_index % len(FAULT_CASES)]


def mutate_edge_result(
    payload: Mapping[str, Any],
    case: str,
) -> dict[str, Any]:
    """Return a deep-copied bad-UE payload for one registered fault case."""

    if case not in FAULT_CASES:
        raise ValueError(f"unknown fault case: {case}")
    value = copy.deepcopy(dict(payload))
    update = value["object_map_update"]
    records = update["records"]
    if case == "NONFINITE":
        if not records:
            raise ValueError("NONFINITE requires at least one object record")
        records[0]["world_x"] = float("nan")
    elif case == "STALE":
        capture_ns = int(value["capture_timestamp_ns"]) - STALE_OFFSET_NS
        if capture_ns < 0:
            raise ValueError("STALE mutation would produce a negative timestamp")
        value["capture_timestamp_ns"] = capture_ns
        update["capture_timestamp_ns"] = capture_ns
        for record in records:
            record["capture_timestamp_ns"] = capture_ns
    elif case == "XY_BIAS":
        for record in records:
            record["world_x"] = float(record["world_x"]) + XY_BIAS_M
    elif case == "IDENTITY_CONFLICT":
        if not records:
            raise ValueError("IDENTITY_CONFLICT requires at least one object record")
        records[0]["world_x"] = float(records[0]["world_x"]) + XY_BIAS_M
    # DUPLICATE_IDENTICAL is intentionally byte-for-byte equivalent at the
    # data-model boundary; the caller submits it twice.
    return value
