"""Exact quality scorers imported from the qualified Route-B evaluator."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import (
    greedy_match_predictions,
)
from rl_agent.ue_route_b_split_cell_adapter_v1 import (
    mean_or_nan,
    oriented_footprint_iou,
    segmentation_quality_columns,
)


class QualityScoringError(RuntimeError):
    """The exact evaluator received invalid or mutable evidence."""


def _number(value: Any) -> float | int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    result = float(value)
    return result if math.isfinite(result) else None


def immutable_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Own a JSON-canonical copy so inference can safely advance."""

    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, list):
        raise QualityScoringError("object evidence is not a list")
    return tuple(dict(value) for value in decoded)


def immutable_mask(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8).copy()
    if result.ndim != 2:
        raise QualityScoringError(f"semantic mask must be HxW, got {result.shape}")
    result.flags.writeable = False
    return result


def score_segmentation(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> dict[str, Any]:
    """Call the existing qualified segmentation formula, without a rewrite."""

    columns = segmentation_quality_columns(predicted, ground_truth)
    return {str(key): _number(value) for key, value in columns.items()}


def _class_localization(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    class_name: str,
    match_distance_m: float,
) -> dict[str, Any]:
    preds = [row for row in predictions if row.get("class_name") == class_name]
    truth = [row for row in ground_truth if row.get("class_name") == class_name]
    matches = greedy_match_predictions(
        preds, truth, max_distance_m=float(match_distance_m), class_aware=True
    )
    errors = [distance for _pred, _truth, distance in matches]
    dimensions: list[float] = []
    footprints: list[float] = []
    for prediction_index, truth_index, _distance in matches:
        prediction, target = preds[prediction_index], truth[truth_index]
        dimensions.append(
            float(
                np.mean(
                    np.abs(
                        np.asarray(
                            [
                                prediction["size_x"],
                                prediction["size_y"],
                                prediction["size_z"],
                            ]
                        )
                        - np.asarray(
                            [target["size_x"], target["size_y"], target["size_z"]]
                        )
                    )
                )
            )
        )
        footprints.append(oriented_footprint_iou(prediction, target))
    tp, fp, fn = len(matches), len(preds) - len(matches), len(truth) - len(matches)
    precision = tp / len(preds) if preds else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else (1.0 if not preds else 0.0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "valid_empty": int(not preds and not truth),
        "source_time_world_xy_error_m": _number(mean_or_nan(errors)),
        "dimension_error_m": _number(mean_or_nan(dimensions)),
        "footprint_iou": _number(mean_or_nan(footprints)),
    }


def score_localization(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    match_distance_m: float,
) -> dict[str, Any]:
    """Use the exact existing greedy matcher and footprint formula."""

    return {
        class_name: _class_localization(
            predictions,
            ground_truth,
            class_name=class_name,
            match_distance_m=match_distance_m,
        )
        for class_name in ("vehicle", "person")
    }


@dataclass(frozen=True)
class QualityInputs:
    predicted_mask: np.ndarray
    ground_truth_mask: np.ndarray
    predictions: tuple[dict[str, Any], ...]
    ground_truth_objects: tuple[dict[str, Any], ...]
    match_distance_m: float

    @classmethod
    def own(
        cls,
        *,
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
        predictions: Sequence[Mapping[str, Any]],
        ground_truth_objects: Sequence[Mapping[str, Any]],
        match_distance_m: float,
    ) -> "QualityInputs":
        return cls(
            predicted_mask=immutable_mask(predicted_mask),
            ground_truth_mask=immutable_mask(ground_truth_mask),
            predictions=immutable_rows(predictions),
            ground_truth_objects=immutable_rows(ground_truth_objects),
            match_distance_m=float(match_distance_m),
        )


def score_serial(inputs: QualityInputs) -> dict[str, Any]:
    return {
        "segmentation": score_segmentation(
            inputs.predicted_mask, inputs.ground_truth_mask
        ),
        "localization": score_localization(
            inputs.predictions,
            inputs.ground_truth_objects,
            match_distance_m=inputs.match_distance_m,
        ),
    }


def score_concurrent(inputs: QualityInputs) -> dict[str, Any]:
    """Parallelize only independent scoring over immutable final evidence."""

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="quality-score") as pool:
        segmentation = pool.submit(
            score_segmentation, inputs.predicted_mask, inputs.ground_truth_mask
        )
        localization = pool.submit(
            score_localization,
            inputs.predictions,
            inputs.ground_truth_objects,
            match_distance_m=inputs.match_distance_m,
        )
        return {
            "segmentation": segmentation.result(),
            "localization": localization.result(),
        }


def require_exact_parity(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    left = json.dumps(reference, sort_keys=True, separators=(",", ":"), allow_nan=False)
    right = json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if left != right:
        raise QualityScoringError("serial/concurrent exact quality mismatch")
