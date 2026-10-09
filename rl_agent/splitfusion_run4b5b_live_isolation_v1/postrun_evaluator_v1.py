"""Exact post-run policy evaluation over separately retained evidence.

This module is deliberately outside the live policy path.  It joins an edge
prediction bundle to a CARLA ground-truth bundle by the complete
``FrameActionIdentityV1``, reuses the qualified quality-feedback scorers, and
delegates ``Q_perc`` to the existing protocol-v2 implementation via
``offline_quality_grid.evaluate_exact_quality``.  It never sends ground truth,
quality, or reward back to the actor.

Timeout latency is censored, not fabricated.  A timeout row therefore has an
empty ``operational_latency_ms`` and a 170-ms lower-bound field.  Plotting code
may draw its marker at the censoring boundary only when it labels it as such.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import (
    greedy_match_predictions,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality import (
    evaluate_exact_quality,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    RewardSpecV1,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.scoring import (
    QualityInputs,
    score_serial,
)

from .branch_evidence_v1 import (
    PredictionEvidenceRecordV1,
    PredictionEvidenceStoreV1,
)
from .operational_ack_v1 import (
    ACK_DEADLINE_NS,
    FrameActionIdentityV1,
    OperationalOutcomeV1,
    OperationalTerminal,
)
from .postrun_artifact_v1 import (
    GROUND_TRUTH_KIND,
    PREDICTION_KIND,
    GroundTruthEvidenceRecordV1,
    GroundTruthEvidenceStoreV1,
    decode_bundle,
)


FRAME_METRICS_SCHEMA = "scenesense.splitfusion.run4b5b.frame_metrics.v1"
OBJECT_METRICS_SCHEMA = "scenesense.splitfusion.run4b5b.object_metrics.v1"
SUMMARY_SCHEMA = "scenesense.splitfusion.run4b5b.postrun_summary.v1"
MANIFEST_SCHEMA = "scenesense.splitfusion.run4b5b.postrun_manifest.v1"
CLAIM_SCOPE = "OFFLINE_CARLA_RESEARCH_EVALUATION_NOT_LIVE_POLICY_FEEDBACK"


class PostRunEvaluationError(RuntimeError):
    """The offline evidence cannot support an exact evaluation."""


class PostRunJoinError(PostRunEvaluationError):
    """Prediction, ground truth, outcome, or payload identities differ."""


class PostRunCreateOnlyError(PostRunEvaluationError):
    """The create-only evaluation output already exists."""


def _require(condition: bool, message: str,
             error: type[PostRunEvaluationError] = PostRunEvaluationError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PostRunEvaluationError("result is not canonicalizable") from exc


def _identity_map(values: Sequence[Any], *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        identity = value.identity
        _require(type(identity) is FrameActionIdentityV1,
                 f"{label} contains a foreign identity", PostRunJoinError)
        digest = identity.exact_sha256()
        _require(digest not in result, f"duplicate {label} identity {digest}",
                 PostRunJoinError)
        result[digest] = value
    return result


def _require_equal_sets(left: set[str], right: set[str], *, label: str) -> None:
    if left != right:
        missing = sorted(left - right)
        foreign = sorted(right - left)
        raise PostRunJoinError(
            f"{label} identity set differs: missing={missing[:3]}, "
            f"foreign={foreign[:3]}, counts={len(left)}/{len(right)}"
        )


def _mask_counts(predicted: np.ndarray, truth: np.ndarray,
                 class_id: int) -> dict[str, int]:
    pred = predicted == class_id
    gt = truth == class_id
    return {
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
        "intersection_pixels": int(np.logical_and(pred, gt).sum()),
        "union_pixels": int(np.logical_or(pred, gt).sum()),
    }


def _object_id(row: Mapping[str, Any], fallback: int) -> str:
    for field in ("actor_id", "track_id", "object_id", "detection_id", "id"):
        if row.get(field) not in (None, ""):
            return str(row[field])
    return f"row:{fallback}"


@dataclass(frozen=True, slots=True)
class _ClassMatch:
    class_name: str
    predictions: tuple[dict[str, Any], ...]
    truth: tuple[dict[str, Any], ...]
    matches: tuple[tuple[int, int, float], ...]

    @property
    def errors(self) -> tuple[float, ...]:
        return tuple(float(item[2]) for item in self.matches)


def _match_class(predictions: Sequence[Mapping[str, Any]],
                 truth: Sequence[Mapping[str, Any]], *, class_name: str,
                 match_distance_m: float) -> _ClassMatch:
    preds = tuple(dict(row) for row in predictions
                  if row.get("class_name") == class_name)
    targets = tuple(dict(row) for row in truth
                    if row.get("class_name") == class_name)
    matches = tuple(
        (int(pred), int(gt), float(distance))
        for pred, gt, distance in greedy_match_predictions(
            preds, targets, max_distance_m=float(match_distance_m),
            class_aware=True,
        )
    )
    return _ClassMatch(class_name, preds, targets, matches)


def _measurement(frame_id: int, predicted_mask: np.ndarray,
                 truth_mask: np.ndarray,
                 matches: Mapping[str, _ClassMatch]) -> dict[str, Any]:
    if predicted_mask.shape != truth_mask.shape:
        import cv2

        truth_mask = cv2.resize(
            truth_mask,
            (predicted_mask.shape[1], predicted_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    value: dict[str, Any] = {"frame_id": int(frame_id)}
    for name, class_id in (("vehicle", 1), ("person", 2)):
        for field, count in _mask_counts(predicted_mask, truth_mask, class_id).items():
            value[f"seg_{name}_{field}"] = count
        match = matches[name]
        value[f"loc_{name}_eligible_gt"] = len(match.truth)
        value[f"loc_{name}_tp"] = len(match.matches)
        value[f"loc_{name}_fn"] = len(match.truth) - len(match.matches)
        value[f"loc_{name}_matched_xy_errors_m"] = list(match.errors)
    return value


def _finite_or_none(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return value
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return None if not selected else float(statistics.fmean(selected))


def _percentile(values: Sequence[float | None], quantile: float) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return None if not selected else float(np.percentile(selected, quantile))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: "" if row[name] is None else row[name]
                         for name in fields})
    return output.getvalue().encode("utf-8")


FRAME_FIELDS = (
    "schema", "frame_order", "identity_sha256", "run_id", "cell_id",
    "stream_id", "session_uuid", "frame_id", "capture_timestamp_ns",
    "elapsed_capture_ms", "decision_seq", "ticket_seq", "tensor_seq",
    "mode_id", "q_e4", "q_exec", "keep_count", "anchor_action_id",
    "profile_id", "payload_bytes", "operational_terminal",
    "operational_success", "operational_latency_ms",
    "latency_censored", "latency_censor_lower_bound_ms", "deadline_ms",
    "prediction_present", "ground_truth_present",
    "segmentation_binary_iou", "segmentation_miou_3class",
    "segmentation_vehicle_iou", "segmentation_person_iou",
    "vehicle_tp", "vehicle_fp", "vehicle_fn", "vehicle_recall",
    "vehicle_xy_error_mean_m", "vehicle_xy_error_median_m",
    "vehicle_footprint_iou", "person_tp", "person_fp", "person_fn",
    "person_recall", "person_xy_error_mean_m", "person_xy_error_median_m",
    "person_footprint_iou", "quality_defined", "q_seg", "q_loc",
    "q_perc", "quality_exclusion_reason", "evaluation_reward",
    "evaluation_status",
)

OBJECT_FIELDS = (
    "schema", "frame_order", "identity_sha256", "frame_id", "class_name",
    "match_status", "prediction_row_index", "prediction_object_id",
    "ground_truth_row_index", "ground_truth_object_id",
    "localization_error_m",
)


@dataclass(frozen=True, slots=True)
class PostRunEvaluationResultV1:
    output_root: Path
    frame_metrics_path: Path
    object_metrics_path: Path
    summary_path: Path
    manifest_path: Path
    frame_count: int
    quality_defined_count: int


class PostRunEvaluatorV1:
    """One-shot, create-only evaluator for a completed validation run."""

    def __init__(self, *, reward_spec: RewardSpecV1,
                 match_distance_m: float = 3.0) -> None:
        _require(type(reward_spec) is RewardSpecV1,
                 "reward_spec must be exactly RewardSpecV1")
        _require(isinstance(match_distance_m, (int, float))
                 and math.isfinite(float(match_distance_m))
                 and float(match_distance_m) > 0.0,
                 "match_distance_m must be finite and positive")
        self.reward_spec = reward_spec
        self.match_distance_m = float(match_distance_m)

    @staticmethod
    def _prediction_bundle(root: Path,
                           record: PredictionEvidenceRecordV1):
        payload = (root / record.artifact_relative_path).read_bytes()
        return decode_bundle(payload, expected_kind=PREDICTION_KIND,
                             expected_identity=record.identity)

    @staticmethod
    def _gt_bundle(root: Path, record: GroundTruthEvidenceRecordV1):
        payload = (root / record.artifact_relative_path).read_bytes()
        return decode_bundle(payload, expected_kind=GROUND_TRUTH_KIND,
                             expected_identity=record.identity)

    @staticmethod
    def _object_rows(*, frame_order: int, identity_sha: str,
                     frame_id: int, matches: Mapping[str, _ClassMatch],
                     ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in ("vehicle", "person"):
            value = matches[name]
            matched_pred = {item[0] for item in value.matches}
            matched_gt = {item[1] for item in value.matches}
            for pred_index, gt_index, distance in value.matches:
                rows.append({
                    "schema": OBJECT_METRICS_SCHEMA,
                    "frame_order": frame_order,
                    "identity_sha256": identity_sha,
                    "frame_id": frame_id,
                    "class_name": name,
                    "match_status": "TP",
                    "prediction_row_index": pred_index,
                    "prediction_object_id": _object_id(
                        value.predictions[pred_index], pred_index),
                    "ground_truth_row_index": gt_index,
                    "ground_truth_object_id": _object_id(
                        value.truth[gt_index], gt_index),
                    "localization_error_m": float(distance),
                })
            for index, row in enumerate(value.predictions):
                if index not in matched_pred:
                    rows.append({
                        "schema": OBJECT_METRICS_SCHEMA,
                        "frame_order": frame_order,
                        "identity_sha256": identity_sha,
                        "frame_id": frame_id,
                        "class_name": name,
                        "match_status": "FP",
                        "prediction_row_index": index,
                        "prediction_object_id": _object_id(row, index),
                        "ground_truth_row_index": None,
                        "ground_truth_object_id": None,
                        "localization_error_m": None,
                    })
            for index, row in enumerate(value.truth):
                if index not in matched_gt:
                    rows.append({
                        "schema": OBJECT_METRICS_SCHEMA,
                        "frame_order": frame_order,
                        "identity_sha256": identity_sha,
                        "frame_id": frame_id,
                        "class_name": name,
                        "match_status": "FN",
                        "prediction_row_index": None,
                        "prediction_object_id": None,
                        "ground_truth_row_index": index,
                        "ground_truth_object_id": _object_id(row, index),
                        "localization_error_m": None,
                    })
        return rows

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise PostRunCreateOnlyError(
                f"create-only evaluation output exists: {path}"
            ) from exc

    def evaluate(
        self, *, prediction_root: Path, ground_truth_root: Path,
        outcomes: Sequence[OperationalOutcomeV1],
        payload_bytes_by_identity: Mapping[str, int], output_root: Path,
    ) -> PostRunEvaluationResultV1:
        """Verify, join, score, and write immutable CSV/JSON evidence."""

        from .postrun_operational_population_v1 import (
            evaluate_operational_population,
        )

        return evaluate_operational_population(
            self, prediction_root=prediction_root,
            ground_truth_root=ground_truth_root, outcomes=outcomes,
            payload_bytes_by_identity=payload_bytes_by_identity,
            output_root=output_root,
        )



__all__ = [
    "FRAME_METRICS_SCHEMA", "OBJECT_METRICS_SCHEMA", "SUMMARY_SCHEMA",
    "MANIFEST_SCHEMA", "CLAIM_SCOPE", "PostRunEvaluationError",
    "PostRunJoinError", "PostRunCreateOnlyError", "FRAME_FIELDS",
    "OBJECT_FIELDS", "PostRunEvaluationResultV1", "PostRunEvaluatorV1",
]
