from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


FAMILIES = ("gradient_norm", "momentum_norm", "proposed_sgd_update_norm")


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError("healthy envelope requires nonempty finite nonnegative observations")
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return {"count": len(values), "min": float(array.min()), "median": float(np.percentile(array, 50)),
            "p99": float(np.percentile(array, 99)), "max": maximum, "ceiling": 10.0 * maximum}


def build_healthy_envelope(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Accept only labeled epoch4-9 telemetry or epoch10 replay updates1-446."""
    observed: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in FAMILIES}
    relative: list[float] = []
    sources = []
    for record in records:
        epoch, update = int(record["epoch"]), int(record["update_in_epoch"])
        source = record.get("source")
        allowed = (source == "ORIGINAL_HEALTHY_TELEMETRY" and 4 <= epoch <= 9) or (
            source == "EPOCH10_EXPLICIT_REPLAY" and epoch == 10 and 1 <= update <= 446)
        if not allowed or not record.get("finite", False):
            raise RuntimeError(f"disallowed healthy-envelope observation: epoch={epoch} update={update} source={source}")
        metrics = record["metrics"]
        for family in FAMILIES:
            for group, value in metrics[family].items():
                observed[family][group].append(float(value))
        relative.append(float(metrics["max_parameter_relative_update"]))
        sources.append({"epoch": epoch, "update_in_epoch": update, "source": source})
    statistics = {family: {group: _summary(values) for group, values in groups.items()}
                  for family, groups in observed.items()}
    relative_statistics = _summary(relative)
    required_groups = {"pretrained_backbone", "pretrained_fpn_heads", "new"}
    if set(statistics["gradient_norm"]) != required_groups | {"global"}:
        raise RuntimeError("healthy envelope lacks exact gradient groups/global")
    for family in ("momentum_norm", "proposed_sgd_update_norm"):
        if set(statistics[family]) != required_groups:
            raise RuntimeError(f"healthy envelope lacks exact {family} groups")
    ceilings = {family: {group: row["ceiling"] for group, row in groups.items()}
                for family, groups in statistics.items()}
    ceilings["max_parameter_relative_update"] = relative_statistics["ceiling"]
    return {"schema": "splitfusion_fcos_healthy_numerical_envelope_v1",
            "sources_allowed": ["healthy epoch4-9 telemetry", "explicit epoch10 replay updates1-446"],
            "threshold_formula": "10 * maximum_healthy_value", "statistics": statistics,
            "parameter_relative_statistics": relative_statistics, "ceilings": ceilings,
            "observation_count": len(sources), "source_records": sources}
