"""Phase-6 fixed-validation accuracy-payload curve for the stable epoch-4 ranker.

Measures the complete validation accuracy-payload curve of the one stable Phase-5
checkpoint (`ranker_epoch_04.pt`, end of the distillation stage) over the whole
registered q ladder: 0.30, 0.50, 0.70, 0.90 and 0.98. The frozen p025 q=0
validation result is reused verbatim as the reference row; q=0 inference is never
rerun.

This is a measurement phase. It does not train, fine-tune, recalibrate, move a
threshold, modify the ranker, select a checkpoint or reopen the diverged epoch-8
and epoch-12 q-aware checkpoints. Every q that is scientifically usable stays
available as an agent action, and a q is never discarded for failing the earlier
near-lossless preservation gates -- those gates are reported as one column of the
characterization, not as an acceptance test.

The locked test split is not opened. Quantization, zstd and INT8 stay deferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1 import (
    run_comparison as avo,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from . import codec, contract, guards
from .gpu_qualification import load_frozen_perception, sha256_file
from .phase5_common import bind_inputs, load_frozen_scorers, source_delta
from .ranker import build_ranker
from .selection import _select_cells, apply_selection, select_cells

EXECUTE_TOKEN = "HYBRID_Q_PHASE6_VALIDATION_CURVE"
DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8
# Framed encode/decode is deterministic and was proved bit-exact in Phase 3; a
# bounded per-pass re-verification keeps the check without a 22 MB device-to-host
# copy of every frame purely for comparison.
ROUNDTRIP_VERIFY_FRAMES = 8
# Frames used for the q-independent-ordering and mask-nesting diagnostics.
ORDERING_DIAGNOSTIC_FRAMES = 16
# Readiness-only cutoff used to show an unregistered q is constructible from the
# same ordering. It is never encoded, transported or scored.
CONSTRUCTIBILITY_PROBE_Q = 0.55


def _collate(items: Sequence[tuple]) -> tuple[torch.Tensor, list, list]:
    return (
        torch.stack([item[0] for item in items]),
        [dict(item[1]) for item in items],
        [item[2] for item in items],
    )


# ---------------------------------------------------------------------------
# Bound-input verification
# ---------------------------------------------------------------------------


def bind_phase6_inputs() -> dict[str, Any]:
    """Verify every bound Phase-6 input by exact hash and fail closed on drift.

    Covers the four hashes the phase is bound to (stable ranker, frozen
    perception checkpoint, p025 forward lock, hybrid-q locked config) plus the
    frozen q=0 validation artifacts that are reused instead of recomputed.
    """
    root = contract.repository_root()
    binding = bind_inputs()

    ranker_path = (root / contract.VALIDATION_RANKER_RELPATH).resolve(strict=True)
    ranker_hash = sha256_file(ranker_path)
    if ranker_hash != contract.VALIDATION_RANKER_SHA256:
        raise guards.HybridQConfigError("stable epoch-4 ranker sha256 drift")

    frozen: dict[str, str] = {}
    for relative, expected in (
        (
            f"{contract.FROZEN_Q0_PREDICTION_ROOT}/detections.csv",
            contract.FROZEN_Q0_DETECTIONS_SHA256,
        ),
        (
            f"{contract.FROZEN_Q0_PREDICTION_ROOT}/inference_manifest.json",
            contract.FROZEN_Q0_INFERENCE_MANIFEST_SHA256,
        ),
        (
            f"{contract.FROZEN_Q0_PREDICTION_ROOT}/segmentation_manifest.csv",
            contract.FROZEN_Q0_SEGMENTATION_MANIFEST_SHA256,
        ),
        (
            f"{contract.FROZEN_Q0_PREDICTION_ROOT}/evaluation_v010.json",
            contract.FROZEN_Q0_EVALUATION_SHA256,
        ),
        (contract.VALIDATION_AVO_TABLE_RELPATH, contract.VALIDATION_AVO_TABLE_SHA256),
        (
            contract.P025_VALIDATION_CONFIRMATION_RELPATH,
            contract.P025_VALIDATION_CONFIRMATION_SHA256,
        ),
    ):
        observed = sha256_file((root / relative).resolve(strict=True))
        if observed != expected:
            raise guards.HybridQConfigError(f"{relative} sha256 drift")
        frozen[relative] = observed

    manifest = json.loads(
        (root / contract.FROZEN_Q0_PREDICTION_ROOT / "inference_manifest.json")
        .read_text(encoding="utf-8")
    )
    if int(manifest["validation_frames"]) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("frozen q=0 validation frame count drift")
    if manifest["detections_sha256"] != contract.FROZEN_Q0_DETECTIONS_SHA256:
        raise guards.HybridQConfigError("frozen q=0 manifest/detections binding drift")
    if int(manifest["inference_pass_count"]) != 1:
        raise guards.HybridQConfigError("frozen q=0 inference pass count drift")

    confirmation = json.loads(
        (root / contract.P025_VALIDATION_CONFIRMATION_RELPATH).read_text(encoding="utf-8")
    )
    if confirmation["terminal"] != (
        "PERSON_P025_TRAIN_HOLDOUT_QUALIFIED_VALIDATION_CONFIRMED"
    ):
        raise guards.HybridQConfigError("p025 validation confirmation terminal drift")
    if bool(confirmation["model_inference_run"]):
        raise guards.HybridQConfigError("p025 confirmation unexpectedly ran inference")

    return {
        **binding,
        "stable_ranker": {
            "path": contract.VALIDATION_RANKER_RELPATH,
            "sha256": ranker_hash,
            "epoch": contract.VALIDATION_RANKER_EPOCH,
            "stage": contract.VALIDATION_RANKER_STAGE,
            "excluded_epochs": list(contract.VALIDATION_EXCLUDED_RANKER_EPOCHS),
            "excluded_reason": contract.VALIDATION_EXCLUDED_RANKER_REASON,
        },
        "frozen_q0_validation_artifacts": frozen,
    }


# ---------------------------------------------------------------------------
# Validation-split person ground truth, taken from the frozen validation path
# ---------------------------------------------------------------------------


def load_validation_person_truth() -> dict[str, Any]:
    """The frozen validation AVO ground truth, loaded exactly as p025 loaded it."""
    root = contract.repository_root()
    raw = avo.load_raw_sources()
    table = avo.read_csv_pandas(
        root / contract.VALIDATION_AVO_TABLE_RELPATH, dtype={"gt_actor_id": str}
    )
    table_keys = {(str(row["sample_id"]), str(row["gt_actor_id"])) for row in table}
    raw_keys = {
        (str(row["sample_id"]), str(row["gt_actor_id"])) for row in raw["qualified"]
    }
    if len(table_keys) != len(table) or table_keys != raw_keys:
        raise guards.HybridQConfigError(
            "frozen AVO table does not exactly cover qualified validation GT"
        )
    frame_ids = [str(value) for value in raw["frame_ids"]]
    if len(frame_ids) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation frame count drift")
    if len(set(frame_ids)) != len(frame_ids):
        raise guards.HybridQConfigError("validation frame ids are not unique")
    if set(avo.EPISODES) != set(contract.VALIDATION_EPISODES):
        raise guards.HybridQConfigError("validation episode identity drift")
    return {
        "frame_ids": frame_ids,
        "qualified_gt": avo.gt_from_table(table),
        "structural_gt": avo.structural_gt(raw),
        "episode_by_sample": {
            sample_id: str(meta["experiment_id"])
            for sample_id, meta in raw["manifest_by_sample"].items()
        },
        "input_hashes": raw["input_hashes"],
        "avo_table_rows": len(table),
    }


# ---------------------------------------------------------------------------
# One transport configuration over the fixed validation split
# ---------------------------------------------------------------------------


def run_validation_pass(
    *, model: torch.nn.Module, base: Any, ranker: torch.nn.Module, q: float,
    dataset: Any, positions: Sequence[int], frame_ids: Sequence[str],
    device: torch.device, output: Path, workers: int, limit: int | None = None,
) -> dict[str, Any]:
    """Encode, transport, decode and serve one q over the validation split.

    Identical in structure to the Phase-5 holdout pass: the same ranker, the same
    exact-cardinality selection, the same framed codec and the same frozen p025
    service policy. Only the split and the accepted q ladder differ.
    """
    value = guards.require_valid_q(q)
    if value == contract.VALIDATION_BASELINE_Q:
        raise guards.HybridQConfigError(
            "q=0 is reused from the frozen p025 validation result and must not be rerun"
        )
    if ranker is None:
        raise guards.HybridQConfigError("a nonzero q requires the stable ranker")

    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    manifest_path = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=INFERENCE_BATCH, shuffle=False, num_workers=workers,
        collate_fn=_collate, drop_last=False, pin_memory=False,
    )
    expected_keep = contract.keep_count(value)
    payload_bytes: set[int] = set()
    keep_counts: set[int] = set()
    roundtrip_exact = True
    roundtrip_verified = 0
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    detection_count = 0
    person_count = 0
    vehicle_count = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base.infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, (fused, rows, calibrations) in enumerate(loader):
                if limit is not None and batch_index >= limit:
                    break
                inputs = fused.to(device, non_blocking=True)
                c2 = model.encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen validation C2")

                transported = []
                for index in range(c2.shape[0]):
                    frame = c2[index]
                    scores = ranker.score_cells(frame)
                    selection = select_cells(scores, value)
                    payload = codec.encode(
                        apply_selection(frame, selection), value, selection
                    )
                    decoded, decoded_q = codec.decode(payload)
                    if decoded_q != value:
                        raise guards.HybridQPayloadError("decoded q drift")
                    keep_counts.add(int(selection.keep_count))
                    payload_bytes.add(int(payload.total_bytes))
                    if roundtrip_verified < ROUNDTRIP_VERIFY_FRAMES:
                        cpu_frame = frame.detach().cpu()
                        mask = selection.keep_mask.unsqueeze(0).expand_as(cpu_frame).cpu()
                        roundtrip_exact &= bool(torch.equal(decoded[mask], cpu_frame[mask]))
                        roundtrip_exact &= bool((decoded[~mask] == 0).all())
                        roundtrip_verified += 1
                        del cpu_frame
                    transported.append(decoded.to(device))

                hybrid = torch.stack(transported)
                outputs = model.decode_tail(hybrid, dense=False)
                calibration_gpu = [
                    {name: tensor.to(device) for name, tensor in calibration.items()}
                    for calibration in calibrations
                ]
                detections = model.postprocess(outputs, calibration_gpu)
                for index, row in enumerate(rows):
                    frame_view = {"semantic_logits": outputs["semantic_logits"][index:index + 1]}
                    served, original_indices = apply_p025_service_policy(
                        frame_view, detections[index]
                    )
                    records = combined_records(base, row, served, original_indices)
                    for record in records:
                        writer.writerow(record)
                        if record["class_name"] == "person":
                            person_count += 1
                        else:
                            vehicle_count += 1
                    detection_count += len(records)
                    observed_ids.append(str(row["sample_id"]))

                    source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                    labels = F.interpolate(
                        outputs["semantic_logits"][index:index + 1].float(),
                        size=source_hw, mode="bilinear", align_corners=False,
                    ).argmax(1)[0]
                    array = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), array):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append({
                        "sample_id": row["sample_id"], "prediction_path": str(relative),
                        "width": array.shape[1], "height": array.shape[0],
                    })
                del c2, hybrid, outputs, detections, transported

    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        manifest_writer.writeheader()
        manifest_writer.writerows(segmentation_rows)

    del loader
    if limit is None:
        if observed_ids != list(frame_ids):
            raise guards.HybridQConfigError("validation inference order drift")
        if len(set(observed_ids)) != contract.VALIDATION_FRAMES:
            raise guards.HybridQConfigError("validation frame coverage drift")
    if keep_counts != {expected_keep}:
        raise guards.HybridQPayloadError(f"observed keep counts {sorted(keep_counts)}")
    if not roundtrip_exact:
        raise guards.HybridQPayloadError("framed encode/decode was not exact")
    observed_payload = sorted(payload_bytes)
    if len(observed_payload) != 1:
        raise guards.HybridQPayloadError(f"non-constant framed payload {observed_payload}")

    return {
        "q": value,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": expected_keep,
        "dropped_cells": contract.drop_count(value),
        "framed_payload_bytes": observed_payload[0],
        "framed_payload_ratio": contract.framed_payload_ratio(observed_payload[0]),
        "raw_fp32_ratio": contract.raw_fp32_ratio(observed_payload[0]),
        "framed_encode_decode_exact": True,
        "framed_encode_decode_frames_verified": roundtrip_verified,
        "inference_run_here": True,
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


def frozen_q0_pass() -> dict[str, Any]:
    """The reused frozen p025 q=0 validation row. No inference is performed."""
    root = contract.repository_root()
    prediction_root = root / contract.FROZEN_Q0_PREDICTION_ROOT
    detections = prediction_root / "detections.csv"
    people = 0
    vehicles = 0
    total = 0
    with detections.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            if row["class_name"] == "person":
                # The frozen prediction set is the p020 base; the locked p025
                # filter is applied by the scorer at the registered 0.25 point.
                if float(row["score"]) >= contract.PERSON_SERVICE_SCORE_THRESHOLD:
                    people += 1
            else:
                vehicles += 1
    return {
        "q": contract.VALIDATION_BASELINE_Q,
        "frames": contract.VALIDATION_FRAMES,
        "prediction_root": str(prediction_root),
        "detections_csv_sha256": contract.FROZEN_Q0_DETECTIONS_SHA256,
        "detections": total,
        "person_service_outputs": people,
        "vehicle_service_outputs": vehicles,
        "retained_cells": contract.SPLIT_CELLS,
        "dropped_cells": 0,
        "framed_payload_bytes": contract.FRAMED_Q0_PAYLOAD_BYTES,
        "framed_payload_ratio": 1.0,
        "raw_fp32_ratio": contract.raw_fp32_ratio(contract.FRAMED_Q0_PAYLOAD_BYTES),
        "framed_encode_decode_exact": True,
        "framed_encode_decode_frames_verified": 0,
        "inference_run_here": False,
        "reused_frozen_result": True,
        "reuse_note": (
            "the frozen p025 q=0 validation prediction set is reused verbatim and "
            "re-scored by the identical Phase-6 scoring functions; q=0 inference "
            "was not rerun"
        ),
    }


# ---------------------------------------------------------------------------
# Scoring: existing frozen vehicle, segmentation and p025 person views
# ---------------------------------------------------------------------------


def _person_only(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    return {
        sample_id: [dict(row) for row in rows if str(row["class_name"]) == "person"]
        for sample_id, rows in grouped.items()
    }


def score_validation_pass(
    *, result: Mapping[str, Any], scorers: Any, truth: Mapping[str, Any],
    experiment: Path, frame_ids: Sequence[str], gt: Mapping[str, Any],
    person_gt: Mapping[str, Any], ignore_cache: dict[str, Any],
) -> dict[str, Any]:
    """Existing frozen vehicle and segmentation scoring plus the frozen p025 views.

    Vehicle is `audit_v1.score_arm` at the canonical 0.20 service point; the
    canonical person view is the same scorer restricted to person rows at the
    locked p025 0.25 output threshold; segmentation is
    `score_contract_v1.score_segmentation`; and the person AVO>=0.65 view is the
    frozen validation `run_comparison.score_person_view`. All four are the same
    functions that produced the frozen q=0 validation numbers.
    """
    prediction_root = Path(result["prediction_root"])
    detections_path = prediction_root / "detections.csv"
    predictions, missing = scorers.load_predictions(detections_path)
    if missing:
        raise guards.HybridQNumericalError(
            f"missing/nonfinite prediction fields: {missing[:5]}"
        )

    arm = scorers.score_arm(
        experiment=experiment, contract=contract.PRIMARY_CONTRACT,
        frame_ids=list(frame_ids), predictions=predictions, gt=gt,
        threshold=contract.VEHICLE_SCORE_THRESHOLD, ignore_cache=ignore_cache,
    )
    canonical_person = scorers.score_arm(
        experiment=experiment, contract=contract.PRIMARY_CONTRACT,
        frame_ids=list(frame_ids), predictions=_person_only(predictions),
        gt=person_gt, threshold=contract.PERSON_SERVICE_SCORE_THRESHOLD,
        ignore_cache=ignore_cache,
    )["classes"]["person"]

    segmentation = scorers.score_segmentation(
        experiment, contract.PRIMARY_CONTRACT, list(frame_ids), prediction_root,
        prediction_root / "segmentation_manifest.csv",
    )

    people, prediction_rows = avo.load_person_predictions(detections_path)
    person = avo.score_person_view(
        frame_ids=list(frame_ids), predictions=people,
        qualified_gt=truth["qualified_gt"],
        structural_ignored_gt=truth["structural_gt"],
        episode_by_sample=truth["episode_by_sample"],
        avo_threshold=contract.PERSON_AVO_THRESHOLD,
        detection_threshold=contract.PERSON_SERVICE_SCORE_THRESHOLD,
    )
    overall = person["overall"]
    bins = person["distance_bins"]
    # The frozen validation scorer names the per-bin denominator `eligible_gt`;
    # it is the AVO-observable GT of that bin, so the 20-40 m recall is a strict
    # partition of the same TP/FN accounting as the aggregate.
    long_tp = sum(int(bins[name]["tp"]) for name in contract.PERSON_LONG_RANGE_BINS)
    long_gt = sum(
        int(bins[name]["eligible_gt"]) for name in contract.PERSON_LONG_RANGE_BINS
    )
    if long_tp + sum(
        int(bins[name]["fn"]) for name in contract.PERSON_LONG_RANGE_BINS
    ) != long_gt:
        raise guards.HybridQConfigError("person 20-40 m denominator failure")
    if sum(int(bucket["eligible_gt"]) for bucket in bins.values()) != int(
        overall["observable_gt"]
    ):
        raise guards.HybridQConfigError(
            "distance-bin partition does not cover observable GT"
        )

    vehicle = arm["classes"]["vehicle"]

    def scalar(value: Any) -> float:
        return float("nan") if value is None else float(value)

    metrics = {
        "vehicle_precision": scalar(vehicle["precision"]),
        "vehicle_recall": scalar(vehicle["recall"]),
        "vehicle_f1": scalar(vehicle["f1"]),
        "vehicle_xy_mae_m": scalar(vehicle["xy_mae_m"]),
        "person_avo_precision": scalar(overall["precision"]),
        "person_avo_recall": scalar(overall["recall"]),
        "person_avo_f1": scalar(overall["f1"]),
        "person_avo_xy_mae_m": scalar(overall["xy_mae_m"]),
        "vehicle_iou": float(segmentation["vehicle_iou"]),
        "person_box_mask_iou": float(segmentation["person_box_mask_iou"]),
        "foreground_miou": float(segmentation["foreground_miou"]),
        "person_avo_recall_20_40m": (long_tp / long_gt) if long_gt else 0.0,
    }
    if set(metrics) != set(contract.PROTECTED_METRICS):
        raise guards.HybridQConfigError("protected metric set drift")

    # The nine absolute service targets are defined on the canonical v010 person
    # view, so they are evaluated on exactly the metric names the registered
    # evaluator uses.
    service_metrics = {
        "vehicle_precision": metrics["vehicle_precision"],
        "vehicle_recall": metrics["vehicle_recall"],
        "vehicle_xy_mae_m": metrics["vehicle_xy_mae_m"],
        "vehicle_iou": metrics["vehicle_iou"],
        "person_box_mask_iou": metrics["person_box_mask_iou"],
        "foreground_miou": metrics["foreground_miou"],
        "person_precision": scalar(canonical_person["precision"]),
        "person_recall": scalar(canonical_person["recall"]),
        "person_xy_mae_m": scalar(canonical_person["xy_mae_m"]),
    }

    return {
        **dict(result),
        "metrics": metrics,
        "canonical_person_metrics": {
            "person_precision": service_metrics["person_precision"],
            "person_recall": service_metrics["person_recall"],
            "person_f1": scalar(canonical_person["f1"]),
            "person_xy_mae_m": service_metrics["person_xy_mae_m"],
        },
        "absolute_service_gates": contract.absolute_service_gates(service_metrics),
        "vehicle_detail": vehicle,
        "person_canonical_v010_detail": canonical_person,
        "person_avo_detail": {
            "overall": overall,
            "episodes": person["episodes"],
            "distance_bins": bins,
            "observable_gt_20_40m": long_gt,
            "tp_20_40m": long_tp,
            "prediction_rows_all_classes": prediction_rows,
        },
        "segmentation_detail": {
            name: segmentation[name] for name in
            ("vehicle_iou", "person_box_mask_iou", "foreground_miou",
             "background_iou", "background_iou_role", "ignored_pixels",
             "confusion_matrix")
        },
    }


def require_frozen_q0_reproduced(scored: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the Phase-6 scoring path reproduces the published q=0 row.

    The q=0 row is the frozen p025 validation result, so the comparison is exact:
    if the Phase-6 scoring path did not reproduce it bit-for-bit, the q>0 rows
    would not be comparable and the whole curve would be invalid.
    """
    published = dict(contract.FROZEN_Q0_VALIDATION_METRICS)
    observed = dict(scored["metrics"])
    observed.update({
        f"person_canonical_{name}": value
        for name, value in (
            ("precision", scored["canonical_person_metrics"]["person_precision"]),
            ("recall", scored["canonical_person_metrics"]["person_recall"]),
            ("f1", scored["canonical_person_metrics"]["person_f1"]),
            ("xy_mae_m", scored["canonical_person_metrics"]["person_xy_mae_m"]),
        )
    })
    mismatched = {
        name: {"published": value, "observed": observed.get(name)}
        for name, value in published.items()
        if observed.get(name) != value
    }
    if mismatched:
        raise guards.HybridQConfigError(
            f"Phase-6 scoring path does not reproduce the frozen q=0 result: {mismatched}"
        )
    expected_recall = (
        contract.FROZEN_Q0_PERSON_RECALL_20_40M_TP
        / contract.FROZEN_Q0_PERSON_RECALL_20_40M_GT
    )
    if scored["metrics"]["person_avo_recall_20_40m"] != expected_recall:
        raise guards.HybridQConfigError("frozen q=0 20-40 m person recall drift")
    if int(scored["person_avo_detail"]["tp_20_40m"]) != (
        contract.FROZEN_Q0_PERSON_RECALL_20_40M_TP
    ):
        raise guards.HybridQConfigError("frozen q=0 20-40 m TP drift")
    gates = scored["absolute_service_gates"]
    if int(gates["pass_count"]) != contract.FROZEN_Q0_SERVICE_PASS_COUNT:
        raise guards.HybridQConfigError("frozen q=0 absolute service pass count drift")
    if tuple(gates["failed"]) != contract.FROZEN_Q0_FAILED_SERVICE_GATES:
        raise guards.HybridQConfigError("frozen q=0 failed service gate identity drift")
    return {
        "published_metrics_reproduced_exactly": True,
        "compared_metrics": sorted(published),
        "absolute_service_pass_count": int(gates["pass_count"]),
        "failed_absolute_service_gates": list(gates["failed"]),
    }


# ---------------------------------------------------------------------------
# Preservation gates and descriptive action-profile classification
# ---------------------------------------------------------------------------


def evaluate_preservation_gates(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, Any]:
    """The registered near-lossless preservation gates, reported not enforced.

    Phase 6 records how many of these a q retains as one column of its profile.
    A q is never excluded from the measured action set for failing them.
    """
    rows = {}
    for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES:
        degradation = contract.gate_degradation(name, baseline[name], candidate[name])
        rows[name] = {
            "direction": direction,
            "bound": bound,
            "baseline": baseline[name],
            "candidate": candidate[name],
            "degradation": degradation,
            "passed": degradation <= bound,
        }
    degradations = [row["degradation"] for row in rows.values()]
    return {
        "gates": rows,
        "pass_count": sum(row["passed"] for row in rows.values()),
        "gate_count": len(rows),
        "all_passed": all(row["passed"] for row in rows.values()),
        "failed": sorted(name for name, row in rows.items() if not row["passed"]),
        "worst_degradation": max(degradations),
        "worst_absolute_degradation": max(abs(value) for value in degradations),
        "role": "reported characterization column, not a Phase-6 acceptance gate",
    }


def classify_profile(
    *, baseline: Mapping[str, float], candidate: Mapping[str, float],
    preservation: Mapping[str, Any], service_pass_count: int,
) -> dict[str, Any]:
    """Descriptive action-profile label from the registered priority cascade.

    The cascade and every bound were registered in `contract.py` before the
    measurement. This function only reads fixed results; it fits nothing.
    """
    def loss(name: str) -> float:
        return float(baseline[name]) - float(candidate[name])

    def increase(name: str) -> float:
        return float(candidate[name]) - float(baseline[name])

    person_f1_loss = loss("person_avo_f1")
    vehicle_f1_loss = loss("vehicle_f1")
    segmentation_loss = loss("foreground_miou")
    vehicle_xy_increase = increase("vehicle_xy_mae_m")
    person_xy_increase = increase("person_avo_xy_mae_m")

    finite = all(
        math.isfinite(float(value)) for value in candidate.values()
    )
    if (
        not finite
        or person_f1_loss > contract.PROFILE_UNUSABLE_F1_COLLAPSE
        or vehicle_f1_loss > contract.PROFILE_UNUSABLE_F1_COLLAPSE
        or int(service_pass_count) <= contract.PROFILE_UNUSABLE_MAX_SERVICE_PASS
    ):
        label = "unusable"
    elif bool(preservation["all_passed"]):
        label = "accuracy-first"
    elif (
        person_f1_loss <= contract.PROFILE_BALANCED_PERSON_F1_LOSS
        and vehicle_f1_loss <= contract.PROFILE_BALANCED_VEHICLE_F1_LOSS
        and segmentation_loss <= contract.PROFILE_BALANCED_SEGMENTATION_LOSS
    ):
        label = "balanced"
    elif (
        vehicle_xy_increase <= contract.PROFILE_LOCALIZATION_XY_INCREASE
        and person_xy_increase <= contract.PROFILE_LOCALIZATION_XY_INCREASE
        and segmentation_loss > contract.PROFILE_BALANCED_SEGMENTATION_LOSS
    ):
        label = "localization-preserving/segmentation-reduced"
    else:
        label = "emergency-bandwidth"

    return {
        "classification": label,
        "scientifically_usable": label != "unusable",
        "available_as_agent_action": label != "unusable",
        "inputs": {
            "person_avo_f1_loss": person_f1_loss,
            "vehicle_f1_loss": vehicle_f1_loss,
            "foreground_miou_loss": segmentation_loss,
            "vehicle_xy_mae_increase_m": vehicle_xy_increase,
            "person_avo_xy_mae_increase_m": person_xy_increase,
            "absolute_service_pass_count": int(service_pass_count),
            "all_metrics_finite": finite,
        },
        "basis": "registered pre-measurement cascade in contract.VALIDATION_PROFILE_CASCADE",
    }


# ---------------------------------------------------------------------------
# Continuous-q readiness diagnostics
# ---------------------------------------------------------------------------


def ordering_diagnostics(
    *, model: torch.nn.Module, ranker: torch.nn.Module, dataset: Any,
    positions: Sequence[int], device: torch.device, frames: int,
) -> dict[str, Any]:
    """Confirm one q-independent ordering, nested registered masks and K(q) validity.

    The ranker consumes only the detached C2 tensor and never sees q, so a single
    score map fixes one spatial ordering per frame and every registered q is a
    prefix of it. This measures that directly rather than asserting it, and also
    shows an unregistered cutoff is constructible from the same ordering.
    """
    ladder = (contract.VALIDATION_BASELINE_Q,) + tuple(
        contract.VALIDATION_EVALUATION_Q_VALUES
    )
    nonzero = tuple(contract.VALIDATION_EVALUATION_Q_VALUES)
    checked = 0
    nested = True
    prefix_of_single_ordering = True
    probe_nested = True
    probe_keep = contract.continuous_keep_count(CONSTRUCTIBILITY_PROBE_Q)

    with torch.inference_mode():
        for position in list(positions)[:frames]:
            fused, _row, _calibration = dataset[position]
            c2 = model.encode_front(fused.unsqueeze(0).to(device)).float()[0]
            scores = ranker.score_cells(c2)
            # One ordering, computed once from the q-independent score map.
            order = torch.argsort(
                scores.reshape(-1).detach().to(torch.float32),
                descending=True, stable=True,
            )
            selections = {q: select_cells(scores, q) for q in nonzero}
            for q, selection in selections.items():
                keep = contract.keep_count(q)
                expected = torch.sort(order[:keep]).values.to(torch.int64)
                prefix_of_single_ordering &= bool(
                    torch.equal(selection.keep_indices.cpu(), expected.cpu())
                )
            for larger, smaller in zip(nonzero, nonzero[1:]):
                outer = set(selections[larger].keep_indices.cpu().tolist())
                inner = set(selections[smaller].keep_indices.cpu().tolist())
                nested &= inner.issubset(outer)
            # An unregistered cutoff is just a different prefix length of the
            # same ordering. Diagnostic only: never encoded or transported.
            probe = _select_cells(scores, CONSTRUCTIBILITY_PROBE_Q, registered_only=False)
            probe_indices = set(probe.keep_indices.cpu().tolist())
            probe_nested &= int(probe.keep_count) == probe_keep
            probe_nested &= probe_indices.issubset(
                set(selections[0.50].keep_indices.cpu().tolist())
            )
            probe_nested &= set(
                selections[0.70].keep_indices.cpu().tolist()
            ).issubset(probe_indices)
            checked += 1
            del c2, scores, order, selections

    return {
        "frames_checked": checked,
        "ranker_sees_q": False,
        "ranker_runtime_inputs": "detached fused C2 only",
        "single_q_independent_ordering": prefix_of_single_ordering,
        "registered_masks_nested": nested,
        "nesting_direction": "keep-set at a larger q is a subset of every smaller q",
        "registered_keep_counts": {
            f"{q:.2f}": contract.keep_count(q) for q in ladder
        },
        "continuous_keep_count_convention": "K(q) = round((1 - q) * 21504)",
        "continuous_keep_count_agrees_with_registered_table": (
            contract.continuous_keep_count_agrees_with_registered()
        ),
        "constructibility_probe": {
            "q": CONSTRUCTIBILITY_PROBE_Q,
            "keep_count": probe_keep,
            "constructible_from_same_ordering": probe_nested,
            "nested_between": ["q=0.50", "q=0.70"],
            "encoded_or_transported": False,
            "accuracy_measured": False,
        },
        "unmeasured_q_accuracy_is_not_interpolated_or_validated": True,
        "recommendation": (
            "snap a requested continuous q down to the nearest validated, "
            "less-aggressive q (contract.snap_continuous_q) until a denser "
            "validation sweep is completed"
        ),
        "production_contract_changed": False,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid-q Phase-6 fixed-validation accuracy-payload curve"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    # Inference-plumbing check only. The frozen validation person scorer asserts
    # its AVO eligibility partition over the whole registered split, so a subset
    # cannot be scored without editing a frozen scorer. Smoke mode therefore runs
    # transport and inference and stops before scoring; it never emits a terminal.
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()

    output = args.output
    if output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-6 validation measurement requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    started = time.time()

    binding = bind_phase6_inputs()
    delta = source_delta(binding)
    model, base, perception = load_frozen_perception(device)
    frozen_snapshot = guards.snapshot_module_state(model)

    payload = torch.load(
        contract.repository_root() / contract.VALIDATION_RANKER_RELPATH,
        map_location="cpu", weights_only=False,
    )
    if int(payload["epoch"]) != contract.VALIDATION_RANKER_EPOCH:
        raise guards.HybridQConfigError("stable ranker epoch drift")
    if int(payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError("stable ranker parameter count drift")
    ranker = build_ranker()
    ranker.load_state_dict(payload["ranker"])
    ranker = ranker.to(device).eval()
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    ranker_snapshot = guards.snapshot_module_state(ranker)

    root = contract.repository_root()
    config = json.loads(
        (root / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
         "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json").read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    truth = load_validation_person_truth()
    frame_ids = list(truth["frame_ids"])

    inference = base.data.InferenceDataset(dataset_root, "val")
    position_by_id = {row["sample_id"]: index for index, row in enumerate(inference.rows)}
    if len(position_by_id) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("inference dataset validation frame count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]

    output.mkdir(parents=True, exist_ok=False)
    scorers = load_frozen_scorers()
    gt, gt_states = scorers.load_gt(dataset_root, contract.PRIMARY_CONTRACT)
    validation_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    person_gt = _person_only(validation_gt)
    ignore_cache: dict[str, Any] = {}

    limit = args.smoke_batches or None
    scored_ids = frame_ids
    print(f"[phase6] {len(frame_ids)} validation frames; "
          f"{sum(1 for rows in validation_gt.values() for row in rows if row['class_name'] == 'vehicle')}"
          f" v010 vehicle GT rows; ladder "
          f"{[contract.VALIDATION_BASELINE_Q] + list(contract.VALIDATION_EVALUATION_Q_VALUES)}",
          flush=True)

    if limit is not None:
        for q in contract.VALIDATION_EVALUATION_Q_VALUES:
            probe = run_validation_pass(
                model=model, base=base, ranker=ranker, q=q, dataset=inference,
                positions=positions, frame_ids=frame_ids, device=device,
                output=output / "predictions" / f"q{int(round(q * 100)):02d}",
                workers=int(args.workers), limit=limit,
            )
            guards.require_module_state_unchanged(model, frozen_snapshot)
            guards.require_module_state_unchanged(ranker, ranker_snapshot)
            print(json.dumps({
                "smoke_pass": f"q{q:.2f}",
                "frames": probe["frames"],
                "retained_cells": probe["retained_cells"],
                "framed_payload_bytes": probe["framed_payload_bytes"],
                "framed_payload_ratio": probe["framed_payload_ratio"],
                "framed_encode_decode_exact": probe["framed_encode_decode_exact"],
            }), flush=True)
        print("[phase6] inference-plumbing smoke check only; no scoring, no terminal",
              flush=True)
        return 0

    # --- q = 0: reuse the frozen p025 validation result, scored by this path ---
    baseline = score_validation_pass(
        result=frozen_q0_pass(), scorers=scorers, truth=truth, experiment=dataset_root,
        frame_ids=frame_ids, gt=validation_gt, person_gt=person_gt,
        ignore_cache=ignore_cache,
    )
    baseline["configuration"] = "q0_frozen_p025_validation_reused"
    reproduction = require_frozen_q0_reproduced(baseline)
    baseline["frozen_q0_reproduction"] = reproduction
    baseline["preservation_gates"] = None
    baseline["profile"] = {
        "classification": "accuracy-first",
        "scientifically_usable": True,
        "available_as_agent_action": True,
        "basis": "exact frozen reference row; zero degradation by construction",
    }
    print(json.dumps({"pass": baseline["configuration"],
                      "metrics": baseline["metrics"],
                      "frozen_q0_reproduced_exactly": True}, indent=2), flush=True)

    # --- q > 0: one inference and one evaluation pass each ---
    measured: list[dict[str, Any]] = []
    for q in contract.VALIDATION_EVALUATION_Q_VALUES:
        raw = run_validation_pass(
            model=model, base=base, ranker=ranker, q=q, dataset=inference,
            positions=positions, frame_ids=frame_ids, device=device,
            output=output / "predictions" / f"q{int(round(q * 100)):02d}",
            workers=int(args.workers),
        )
        guards.require_module_state_unchanged(model, frozen_snapshot)
        guards.require_module_state_unchanged(ranker, ranker_snapshot)
        scored = score_validation_pass(
            result=raw, scorers=scorers, truth=truth, experiment=dataset_root,
            frame_ids=scored_ids, gt=validation_gt, person_gt=person_gt,
            ignore_cache=ignore_cache,
        )
        scored["configuration"] = f"q{q:.2f}"
        scored["preservation_gates"] = evaluate_preservation_gates(
            baseline["metrics"], scored["metrics"]
        )
        scored["absolute_change_from_q0"] = {
            name: float(scored["metrics"][name]) - float(baseline["metrics"][name])
            for name in contract.PROTECTED_METRICS
        }
        scored["profile"] = classify_profile(
            baseline=baseline["metrics"], candidate=scored["metrics"],
            preservation=scored["preservation_gates"],
            service_pass_count=scored["absolute_service_gates"]["pass_count"],
        )
        measured.append(scored)
        print(json.dumps({
            "pass": scored["configuration"],
            "retained_cells": scored["retained_cells"],
            "framed_payload_bytes": scored["framed_payload_bytes"],
            "framed_payload_ratio": scored["framed_payload_ratio"],
            "absolute_service_pass_count": scored["absolute_service_gates"]["pass_count"],
            "preservation_gates_passed": scored["preservation_gates"]["pass_count"],
            "classification": scored["profile"]["classification"],
        }, indent=2), flush=True)

    readiness = ordering_diagnostics(
        model=model, ranker=ranker, dataset=inference, positions=positions,
        device=device, frames=ORDERING_DIAGNOSTIC_FRAMES,
    )
    guards.require_module_state_unchanged(model, frozen_snapshot)
    guards.require_module_state_unchanged(ranker, ranker_snapshot)

    passes = [baseline] + measured
    if not args.keep_segmentation:
        for entry in measured:
            directory = Path(entry["prediction_root"]) / "segmentation"
            if directory.is_dir():
                shutil.rmtree(directory)
            entry["segmentation_masks_removed_after_scoring"] = True

    curve = [
        {
            "q": entry["q"],
            "retained_cells": entry["retained_cells"],
            "framed_payload_bytes": entry["framed_payload_bytes"],
            "framed_payload_ratio": entry["framed_payload_ratio"],
            "metrics": entry["metrics"],
            "canonical_person_metrics": entry["canonical_person_metrics"],
            "absolute_change_from_q0": entry.get("absolute_change_from_q0"),
            "absolute_service_pass_count": entry["absolute_service_gates"]["pass_count"],
            "failed_absolute_service_gates": entry["absolute_service_gates"]["failed"],
            "preservation_gates_passed": (
                None if entry["preservation_gates"] is None
                else entry["preservation_gates"]["pass_count"]
            ),
            "preservation_gate_count": len(contract.HOLDOUT_PRESERVATION_GATES),
            "classification": entry["profile"]["classification"],
            "scientifically_usable": entry["profile"]["scientifically_usable"],
        }
        for entry in passes
    ]

    report = {
        "schema": contract.PHASE6_SCHEMA,
        "terminal": contract.PHASE6_TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_role": (
            "measurement of the validation accuracy-payload curve; not checkpoint "
            "selection and not model development"
        ),
        "scope": {
            "evaluated_split": "registered fixed validation split",
            "validation_episodes": list(contract.VALIDATION_EPISODES),
            "validation_frames": contract.VALIDATION_FRAMES,
            "test_accessed": False,
            "baseline_q": contract.VALIDATION_BASELINE_Q,
            "baseline_source": "frozen p025 q=0 validation result, reused verbatim",
            "q0_inference_rerun": False,
            "measured_q_values": list(contract.VALIDATION_EVALUATION_Q_VALUES),
            "inference_passes_run": len(measured),
            "training_run": False,
            "tuning_or_recalibration": False,
            "thresholds_changed": False,
            "ranker_modified": False,
            "teacher_maps_recomputed": False,
            "new_cache_created": False,
            "zstd_or_int8_measured": False,
            "carla_launched": False,
            "excluded_ranker_epochs": list(contract.VALIDATION_EXCLUDED_RANKER_EPOCHS),
            "excluded_ranker_reason": contract.VALIDATION_EXCLUDED_RANKER_REASON,
        },
        "stable_checkpoint_statement": (
            "ranker_epoch_04.pt is the stable distillation-only checkpoint: it is "
            "taken at the end of the four distillation epochs, before the q-aware "
            "stage. The Phase-5 q-aware training failure is unchanged by this "
            "measurement phase; epochs 8 and 12 were neither loaded nor evaluated."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "inference_precision": "fp32 inference_mode, no autocast",
        },
        "binding": {k: v for k, v in binding.items() if k != "teacher_cache_shards"},
        "source_delta": delta,
        "perception_binding": perception,
        "service_pipeline": {
            "policy": (
                "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1."
                "apply_p025_service_policy"
            ),
            "person_output_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "person_avo_threshold": contract.PERSON_AVO_THRESHOLD,
            "vehicle_score_point": contract.VEHICLE_SCORE_THRESHOLD,
            "thresholds_or_postprocessing_changed": False,
        },
        "scoring": {
            "vehicle_and_canonical_person": "frozen audit_v1.score_arm",
            "segmentation": "frozen score_contract_v1.score_segmentation",
            "person_avo": (
                "frozen validation run_comparison.score_person_view, the same "
                "function that produced the frozen q=0 p025 validation numbers"
            ),
            "frozen_scorer_sha256": scorers.sha256,
            "contract": contract.PRIMARY_CONTRACT,
            "gt_contract_states": gt_states,
            "validation_person_truth": {
                "avo_table": contract.VALIDATION_AVO_TABLE_RELPATH,
                "avo_table_sha256": contract.VALIDATION_AVO_TABLE_SHA256,
                "avo_table_rows": truth["avo_table_rows"],
                "raw_input_hashes": truth["input_hashes"],
            },
        },
        "absolute_service_targets": [
            {"metric": name, "target": target, "direction": direction}
            for name, target, direction in contract.ABSOLUTE_SERVICE_TARGETS
        ],
        "preservation_gates": [
            {"metric": name, "direction": direction, "bound": bound}
            for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
        ],
        "preservation_gate_role": (
            "reported as one characterization column; a q is never rejected here "
            "for failing the earlier near-lossless gates"
        ),
        "profile_cascade": [
            {"classification": name, "rule": rule}
            for name, rule in contract.VALIDATION_PROFILE_CASCADE
        ],
        "curve": curve,
        "q0_baseline": baseline,
        "measured_q_passes": measured,
        "continuous_q_readiness": readiness,
        "available_agent_actions": [
            row["q"] for row in curve if row["scientifically_usable"]
        ],
        "wall_seconds": time.time() - started,
        "frozen_state_unchanged_at_end": True,
    }
    guards.require_module_state_unchanged(model, frozen_snapshot)
    guards.require_module_state_unchanged(ranker, ranker_snapshot)

    (output / "validation_curve.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with (output / "validation_curve.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = (
            "q", "retained_cells", "framed_payload_bytes", "framed_payload_ratio",
            "vehicle_precision", "vehicle_recall", "vehicle_f1",
            "person_avo_precision", "person_avo_recall", "person_avo_f1",
            "vehicle_xy_mae_m", "person_avo_xy_mae_m", "person_avo_recall_20_40m",
            "vehicle_iou", "person_box_mask_iou", "foreground_miou",
            "absolute_service_pass_count", "preservation_gates_passed",
            "classification",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in curve:
            writer.writerow({
                "q": f"{row['q']:.2f}",
                "retained_cells": row["retained_cells"],
                "framed_payload_bytes": row["framed_payload_bytes"],
                "framed_payload_ratio": f"{row['framed_payload_ratio']:.6f}",
                **{
                    name: f"{float(row['metrics'][name]):.6f}"
                    for name in contract.PROTECTED_METRICS
                },
                "absolute_service_pass_count": row["absolute_service_pass_count"],
                "preservation_gates_passed": (
                    "" if row["preservation_gates_passed"] is None
                    else row["preservation_gates_passed"]
                ),
                "classification": row["classification"],
            })
    (output / contract.PHASE6_TERMINAL).write_text(
        f"{contract.PHASE6_TERMINAL} {report['generated_utc']}\n", encoding="utf-8"
    )
    print(json.dumps({
        "terminal": contract.PHASE6_TERMINAL,
        "output": str(output),
        "curve": [
            {
                "q": row["q"],
                "retained_cells": row["retained_cells"],
                "framed_payload_ratio": row["framed_payload_ratio"],
                "classification": row["classification"],
            }
            for row in curve
        ],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - runner entry point
    raise SystemExit(main())
