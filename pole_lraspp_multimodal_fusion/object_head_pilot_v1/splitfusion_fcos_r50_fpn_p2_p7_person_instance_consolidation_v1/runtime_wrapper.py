from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .core import apply_rule_to_detections, retained_detection_indices, validate_configuration
from .runtime import FROZEN_CHECKPOINT_SHA256


def load_selected_rule(result_path: Path) -> dict[str, float | int | None]:
    result = json.loads(Path(result_path).resolve(strict=True).read_text(encoding="utf-8"))
    selected = result.get("selected_fit")
    if (result.get("schema") != "splitfusion_fcos_person_instance_consolidation_result_v1"
            or result.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or result.get("status") != "holdout_feasible"
            or int(result.get("holdout_evaluations", -1)) != 1
            or result.get("validation_or_test_accessed") is not False
            or not isinstance(selected, dict)):
        raise RuntimeError("person consolidation result is not deployment-feasible")
    rule = {name: selected[name] for name in (
        "grid_index", "semantic_support_threshold", "group_box_iou_threshold",
    )}
    validate_configuration(rule)
    return rule


def apply_selected_rule(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor], rule: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return apply_rule_to_detections(outputs, detections, rule)


def canonical_records(
    base: Any,
    row: dict[str, str],
    outputs: Mapping[str, Any],
    detections: Mapping[str, torch.Tensor],
    rule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Emit selected records using their unchanged original prediction indices."""
    keep = retained_detection_indices(outputs, detections, rule)
    return [base.infer.record(dict(detections), row, int(index)) for index in keep]
