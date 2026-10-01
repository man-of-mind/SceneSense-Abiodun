"""Operational-ledger-primary implementation of the post-run evaluator.

Kept separate from the public facade to make the population rule unmistakable:
every durable operational outcome produces one frame row.  Prediction and GT
evidence enrich that row when present; they never define or shrink the frame
population.
"""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality import (
    evaluate_exact_quality,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.scoring import (
    QualityInputs,
    score_serial,
)

from .branch_evidence_v1 import PredictionEvidenceStoreV1
from .operational_ack_v1 import (
    ACK_DEADLINE_NS,
    OperationalOutcomeV1,
    OperationalTerminal,
)
from .operational_trace_v1 import OperationalTraceStoreV1
from .postrun_artifact_v1 import GroundTruthEvidenceStoreV1


def _index(values: Sequence[Any], label: str) -> dict[str, Any]:
    # Imported lazily to avoid a module-import cycle.
    from . import postrun_evaluator_v1 as public

    return public._identity_map(values, label=label)  # noqa: SLF001


def _empty_quality_fields() -> dict[str, Any]:
    return {
        "segmentation_binary_iou": None,
        "segmentation_miou_3class": None,
        "segmentation_vehicle_iou": None,
        "segmentation_person_iou": None,
        "vehicle_tp": None,
        "vehicle_fp": None,
        "vehicle_fn": None,
        "vehicle_recall": None,
        "vehicle_xy_error_mean_m": None,
        "vehicle_xy_error_median_m": None,
        "vehicle_footprint_iou": None,
        "person_tp": None,
        "person_fp": None,
        "person_fn": None,
        "person_recall": None,
        "person_xy_error_mean_m": None,
        "person_xy_error_median_m": None,
        "person_footprint_iou": None,
        "quality_defined": 0,
        "q_seg": None,
        "q_loc": None,
        "q_perc": None,
    }


def evaluate_operational_population(
    evaluator: Any, *, prediction_root: Path, ground_truth_root: Path,
    outcomes: Sequence[OperationalOutcomeV1],
    payload_bytes_by_identity: Mapping[str, int], output_root: Path,
):
    """Evaluate every operational row, including timeout/no-output rows."""

    from . import postrun_evaluator_v1 as public

    predictions = PredictionEvidenceStoreV1.open_existing(
        Path(prediction_root)).verify_all()
    truths = GroundTruthEvidenceStoreV1.open_existing(
        Path(ground_truth_root)).verify_all()
    pred_by_id = _index(predictions, "prediction")
    gt_by_id = _index(truths, "ground truth")
    outcome_by_id = _index(outcomes, "operational outcome")
    operational_ids = set(outcome_by_id)
    public._require(bool(operational_ids), "operational trace is empty",  # noqa: SLF001
                    public.PostRunJoinError)
    foreign_predictions = set(pred_by_id) - operational_ids
    foreign_truth = set(gt_by_id) - operational_ids
    public._require(not foreign_predictions,
                    f"prediction lacks an operational row: "
                    f"{sorted(foreign_predictions)[:3]}",
                    public.PostRunJoinError)
    public._require(not foreign_truth,
                    f"ground truth lacks an operational row: "
                    f"{sorted(foreign_truth)[:3]}",
                    public.PostRunJoinError)
    public._require_equal_sets(  # noqa: SLF001
        operational_ids, set(payload_bytes_by_identity), label="payload byte map")
    for digest, size in payload_bytes_by_identity.items():
        public._require(type(size) is int and size > 0,
                        f"payload size for {digest} is not a positive integer",
                        public.PostRunJoinError)

    ordered = sorted(
        outcomes,
        key=lambda value: (
            value.identity.capture_timestamp_ns, value.identity.stream_id,
            value.identity.frame_id, value.identity.tensor_seq,
        ),
    )
    first_capture = min(value.identity.capture_timestamp_ns for value in ordered)
    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []

    for frame_order, outcome in enumerate(ordered):
        identity = outcome.identity
        digest = identity.exact_sha256()
        prediction_record = pred_by_id.get(digest)
        gt_record = gt_by_id.get(digest)
        prediction_present = prediction_record is not None
        gt_present = gt_record is not None
        success = outcome.terminal is OperationalTerminal.SUCCESS

        if success:
            public._require(  # noqa: SLF001
                prediction_present,
                "successful ACK row has no retained prediction",
                public.PostRunJoinError,
            )
            public._require(
                outcome.observed_latency_ns is not None
                and outcome.observed_latency_ns <= ACK_DEADLINE_NS,
                "successful outcome lacks an on-time observed latency",
            )
            operational_latency_ms: float | None = (
                float(outcome.observed_latency_ns) / 1_000_000.0
            )
            censor_lower_bound_ms: float | None = None
        else:
            public._require(
                outcome.terminal is OperationalTerminal.TIMEOUT
                and outcome.observed_latency_ns is None,
                "timeout outcome carries an observed on-time latency",
            )
            operational_latency_ms = None
            censor_lower_bound_ms = ACK_DEADLINE_NS / 1_000_000.0

        quality = None
        matches = None
        quality_fields = _empty_quality_fields()
        if prediction_present and gt_present:
            public._require(  # noqa: SLF001
                prediction_record.identity == identity
                and gt_record.identity == identity,
                "joined exact identities differ", public.PostRunJoinError,
            )
            public._require(
                dict(gt_record.postrun_join_key)
                == dict(prediction_record.postrun_join_key),
                "joined post-run keys differ", public.PostRunJoinError,
            )
            prediction = evaluator._prediction_bundle(
                Path(prediction_root), prediction_record)
            truth = evaluator._gt_bundle(Path(ground_truth_root), gt_record)
            inputs = QualityInputs.own(
                predicted_mask=prediction.semantic_mask,
                ground_truth_mask=truth.semantic_mask,
                predictions=prediction.objects,
                ground_truth_objects=truth.objects,
                match_distance_m=evaluator.match_distance_m,
            )
            score = score_serial(inputs)
            matches = {
                name: public._match_class(  # noqa: SLF001
                    prediction.objects, truth.objects, class_name=name,
                    match_distance_m=evaluator.match_distance_m,
                )
                for name in ("vehicle", "person")
            }
            for name in ("vehicle", "person"):
                loc = score["localization"][name]
                public._require(  # noqa: SLF001
                    int(loc["tp"]) == len(matches[name].matches)
                    and int(loc["fn"]) == len(matches[name].truth)
                    - len(matches[name].matches),
                    "per-object matches differ from qualified scorer",
                )
            measurement = public._measurement(  # noqa: SLF001
                identity.frame_id, prediction.semantic_mask,
                truth.semantic_mask, matches,
            )
            for name in ("vehicle", "person"):
                score_iou = score["segmentation"][f"miou_{name}_iou"]
                union = measurement[f"seg_{name}_union_pixels"]
                measured_iou = (
                    None if union == 0 else
                    measurement[f"seg_{name}_intersection_pixels"] / union
                )
                public._require(  # noqa: SLF001
                    (score_iou is None and measured_iou is None)
                    or (score_iou is not None and measured_iou is not None
                        and float(score_iou) == float(measured_iou)),
                    "segmentation counts differ from qualified scorer",
                )
            quality = evaluate_exact_quality(evaluator.reward_spec, measurement)
            segmentation = score["segmentation"]
            localization = score["localization"]

            def loc_field(name: str, key: str) -> Any:
                return public._finite_or_none(localization[name][key])  # noqa: SLF001

            def median_error(name: str) -> float | None:
                errors = matches[name].errors
                return None if not errors else float(statistics.median(errors))

            quality_fields = {
                "segmentation_binary_iou": public._finite_or_none(  # noqa: SLF001
                    segmentation["miou_binary"]),
                "segmentation_miou_3class": public._finite_or_none(
                    segmentation["miou_3class_macro"]),
                "segmentation_vehicle_iou": public._finite_or_none(
                    segmentation["miou_vehicle_iou"]),
                "segmentation_person_iou": public._finite_or_none(
                    segmentation["miou_person_iou"]),
                "vehicle_tp": loc_field("vehicle", "tp"),
                "vehicle_fp": loc_field("vehicle", "fp"),
                "vehicle_fn": loc_field("vehicle", "fn"),
                "vehicle_recall": loc_field("vehicle", "recall"),
                "vehicle_xy_error_mean_m": loc_field(
                    "vehicle", "source_time_world_xy_error_m"),
                "vehicle_xy_error_median_m": median_error("vehicle"),
                "vehicle_footprint_iou": loc_field("vehicle", "footprint_iou"),
                "person_tp": loc_field("person", "tp"),
                "person_fp": loc_field("person", "fp"),
                "person_fn": loc_field("person", "fn"),
                "person_recall": loc_field("person", "recall"),
                "person_xy_error_mean_m": loc_field(
                    "person", "source_time_world_xy_error_m"),
                "person_xy_error_median_m": median_error("person"),
                "person_footprint_iou": loc_field("person", "footprint_iou"),
                "quality_defined": int(quality is not None),
                "q_seg": None if quality is None else quality.q_seg,
                "q_loc": None if quality is None else quality.q_loc,
                "q_perc": None if quality is None else quality.q_perc,
            }
            quality_exclusion_reason = (
                "" if quality is not None else "NO_ELIGIBLE_GROUND_TRUTH"
            )
            object_rows.extend(evaluator._object_rows(
                frame_order=frame_order, identity_sha=digest,
                frame_id=identity.frame_id, matches=matches,
            ))
        elif prediction_present:
            quality_exclusion_reason = "GROUND_TRUTH_NOT_RETAINED"
        elif gt_present:
            public._require(
                not success,
                "successful row has GT but no retained prediction",
                public.PostRunJoinError,
            )
            quality_exclusion_reason = "TAIL_OUTPUT_NOT_PRODUCED"
        else:
            public._require(
                not success,
                "successful row has neither prediction nor GT",
                public.PostRunJoinError,
            )
            quality_exclusion_reason = "TAIL_OUTPUT_AND_GT_NOT_RETAINED"

        if not success:
            evaluation_reward: float | None = float(
                run4_contract.REGISTERED_FAILURE_REWARD)
            evaluation_status = (
                "TIMEOUT_EVALUATED_QUALITY" if quality is not None
                else "TIMEOUT_QUALITY_UNAVAILABLE"
            )
        elif quality is None:
            evaluation_reward = None
            evaluation_status = "EXCLUDED_QUALITY_UNAVAILABLE"
        else:
            evaluation_reward = float(quality.q_perc) - float(
                run4_contract.REWARD_LATENCY_WEIGHT
            ) * operational_latency_ms / float(run4_contract.REWARD_DEADLINE_MS)
            evaluation_status = "EVALUATED_SUCCESS"

        frame_rows.append({
            "schema": public.FRAME_METRICS_SCHEMA,
            "frame_order": frame_order,
            "identity_sha256": digest,
            "run_id": identity.run_id,
            "cell_id": identity.cell_id,
            "stream_id": identity.stream_id,
            "session_uuid": identity.session_uuid,
            "frame_id": identity.frame_id,
            "capture_timestamp_ns": identity.capture_timestamp_ns,
            "elapsed_capture_ms": (
                identity.capture_timestamp_ns - first_capture) / 1_000_000.0,
            "decision_seq": identity.decision_seq,
            "ticket_seq": identity.ticket_seq,
            "tensor_seq": identity.tensor_seq,
            "mode_id": identity.mode_id,
            "q_e4": identity.q_e4,
            "q_exec": identity.q_e4 / 10_000.0,
            "keep_count": identity.keep_count,
            "anchor_action_id": identity.anchor_action_id,
            "profile_id": identity.profile_id,
            "payload_bytes": payload_bytes_by_identity[digest],
            "operational_terminal": outcome.terminal.value,
            "operational_success": int(success),
            "operational_latency_ms": operational_latency_ms,
            "latency_censored": int(not success),
            "latency_censor_lower_bound_ms": censor_lower_bound_ms,
            "deadline_ms": run4_contract.REWARD_DEADLINE_MS,
            "prediction_present": int(prediction_present),
            "ground_truth_present": int(gt_present),
            **quality_fields,
            "quality_exclusion_reason": quality_exclusion_reason,
            "evaluation_reward": evaluation_reward,
            "evaluation_status": evaluation_status,
        })

    root = Path(output_root)
    try:
        root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise public.PostRunCreateOnlyError(
            f"evaluation output root already exists: {root}"
        ) from exc
    frame_path = root / "FRAME_METRICS.csv"
    object_path = root / "OBJECT_LOCALIZATION_METRICS.csv"
    summary_path = root / "SUMMARY.json"
    manifest_path = root / "MANIFEST.json"
    frame_bytes = public._csv_bytes(frame_rows, public.FRAME_FIELDS)  # noqa: SLF001
    object_bytes = public._csv_bytes(object_rows, public.OBJECT_FIELDS)  # noqa: SLF001
    evaluator._write_exclusive(frame_path, frame_bytes)
    evaluator._write_exclusive(object_path, object_bytes)

    q_values = [row["q_perc"] for row in frame_rows]
    rewards = [row["evaluation_reward"] for row in frame_rows]
    latencies = [row["operational_latency_ms"] for row in frame_rows]
    exclusion_histogram: dict[str, int] = {}
    for row in frame_rows:
        reason = str(row["quality_exclusion_reason"])
        if reason:
            exclusion_histogram[reason] = exclusion_histogram.get(reason, 0) + 1
    summary = {
        "schema": public.SUMMARY_SCHEMA,
        "claim_scope": public.CLAIM_SCOPE,
        "live_policy_feedback": False,
        "ground_truth_transmitted_to_edge": False,
        "primary_population": "DURABLE_OPERATIONAL_OUTCOMES",
        "q_perc_formula_source": (
            "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid."
            "quality.evaluate_exact_quality"
        ),
        "qualified_raw_scorer_source": (
            "rl_agent.splitfusion_quality_feedback_probe_v1.scoring.score_serial"
        ),
        "reward_spec_sha256": evaluator.reward_spec.canonical_sha256(),
        "frame_count": len(frame_rows),
        "prediction_present_count": sum(int(row["prediction_present"])
                                        for row in frame_rows),
        "ground_truth_present_count": sum(int(row["ground_truth_present"])
                                          for row in frame_rows),
        "object_metric_row_count": len(object_rows),
        "success_count": sum(int(row["operational_success"])
                             for row in frame_rows),
        "timeout_count": sum(row["operational_terminal"] == "TIMEOUT"
                             for row in frame_rows),
        "quality_defined_count": sum(int(row["quality_defined"])
                                     for row in frame_rows),
        "quality_excluded_count": sum(not int(row["quality_defined"])
                                      for row in frame_rows),
        "quality_exclusion_histogram": exclusion_histogram,
        "mean_q_perc_defined": public._mean(q_values),  # noqa: SLF001
        "mean_evaluation_reward_defined": public._mean(rewards),  # noqa: SLF001
        "operational_latency_success_ms": {
            "mean": public._mean(latencies),  # noqa: SLF001
            "p50": public._percentile(latencies, 50),  # noqa: SLF001
            "p95": public._percentile(latencies, 95),  # noqa: SLF001
            "maximum": None if not any(value is not None for value in latencies)
            else max(float(value) for value in latencies if value is not None),
        },
        "deadline_ms": run4_contract.REWARD_DEADLINE_MS,
        "timeout_latency_semantics": (
            "RIGHT_CENSORED_STRICTLY_ABOVE_DEADLINE; never reported as an "
            "observed latency"
        ),
    }
    summary_bytes = public._canonical(summary) + b"\n"  # noqa: SLF001
    evaluator._write_exclusive(summary_path, summary_bytes)
    manifest = {
        "schema": public.MANIFEST_SCHEMA,
        "claim_scope": public.CLAIM_SCOPE,
        "primary_population": "DURABLE_OPERATIONAL_OUTCOMES",
        "prediction_record_count": len(predictions),
        "ground_truth_record_count": len(truths),
        "outcome_count": len(outcomes),
        "outputs": {
            frame_path.name: hashlib.sha256(frame_bytes).hexdigest(),
            object_path.name: hashlib.sha256(object_bytes).hexdigest(),
            summary_path.name: hashlib.sha256(summary_bytes).hexdigest(),
        },
        "reward_spec_sha256": evaluator.reward_spec.canonical_sha256(),
    }
    evaluator._write_exclusive(manifest_path, public._canonical(manifest) + b"\n")  # noqa: SLF001
    return public.PostRunEvaluationResultV1(
        output_root=root,
        frame_metrics_path=frame_path,
        object_metrics_path=object_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        frame_count=len(frame_rows),
        quality_defined_count=int(summary["quality_defined_count"]),
    )


def evaluate_from_operational_trace(
    evaluator: Any, *, operational_trace_root: Path,
    prediction_root: Path, ground_truth_root: Path, output_root: Path,
):
    """Load the durable superset ledger, then run the public evaluator."""

    records = OperationalTraceStoreV1.open_existing(
        Path(operational_trace_root)).verify_all()
    return evaluator.evaluate(
        prediction_root=prediction_root,
        ground_truth_root=ground_truth_root,
        outcomes=[record.outcome for record in records],
        payload_bytes_by_identity={
            record.identity_sha256: record.payload_bytes for record in records
        },
        output_root=output_root,
    )


__all__ = ["evaluate_operational_population", "evaluate_from_operational_trace"]
