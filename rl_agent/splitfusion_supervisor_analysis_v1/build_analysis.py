#!/usr/bin/env python3
"""Build the post-direct-map supervisor analysis from immutable evidence."""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288 as source
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288_final_v3 as final_v3
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288_direct_map_v1 as direct
from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageConfig,
    simulate_two_stage,
)
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_288 as prior
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_timing_boundaries as timing


ROOT = Path(__file__).resolve().parents[2]
DIRECT_ROOT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/"
    "20260914_direct_map_288_counterfactual"
)
LIVE_DIRECT_ROOT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/"
    "20260914_live_validation_retry1"
)
LATEST_EDGE_VALIDATION_ROOT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/"
    "20260914_edge_optimization_validation_v1_retry1"
)
SENSOR_BASELINE_ROOT = ROOT / (
    "experiments/splitfusion_sensor_preparation_live_v1/"
    "20260914_v2_action50_favorable_baseline"
)
SENSOR_OPTIMIZED_ROOT = ROOT / (
    "experiments/splitfusion_sensor_preparation_live_v1/"
    "20260914_v2_action50_favorable_optimized_retry1"
)
SENSOR_PRESENTATION_ROOT = ROOT / (
    "experiments/splitfusion_sensor_preparation_live_v1/"
    "20260914_v2_optimization_presentation"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_supervisor_analysis_v1/"
    "20260914_optimized_pipeline_policy_analysis_v2"
)
SCHEMA = "scenesense.splitfusion.supervisor_analysis.optimized_pipeline.v2"
TERMINAL = "SPLITFUSION_OPTIMIZED_PIPELINE_POLICY_ANALYSIS_COMPLETE"
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
PROFILE_LABEL = {
    "FAVORABLE_STABLE": "Favorable Stable",
    "MID_VARIABLE": "Mid Variable",
    "ADVERSE_STABLE": "Adverse Stable",
    "FADE_RECOVERY": "Fade Recovery",
}
FAMILY_COLOR = {
    "noAE": "#4C78A8",
    "AE128": "#F58518",
    "AE64": "#54A24B",
    "AE32": "#E45756",
}
STAGES = {
    "pure_front": "Model front backbone",
    "ue_action": "7-channel-concat-start-to-UDP-send path",
    "sensor_compute": "Optimized sensor compute before concatenation",
    "network": "Feature uplink",
    "edge_queue": "Tail-busy wait",
    "model_tail": "FCOS model tail",
    "tail_support": "Other tail processing",
    "map_install": "Map service",
    "edge_map": "Complete edge-to-map service",
    "total": "Action-start-to-map total",
}
SCATTER_STAGES = ("ue_action", "network", "edge_map", "total")
QUALITY_VIEWS = (
    {
        "letter": "a",
        "key": "val_segmentation_miou",
        "slug": "semantic_segmentation_miou",
        "label": "Semantic segmentation mIoU (%)",
        "title": "Semantic segmentation mIoU",
        "scale": 100.0,
    },
    {
        "letter": "b",
        "key": "val_vehicle_iou",
        "slug": "vehicle_overlap_iou",
        "label": "Vehicle overlap IoU (%)",
        "title": "Vehicle overlap IoU",
        "scale": 100.0,
    },
    {
        "letter": "c",
        "key": "val_person_box_mask_iou",
        "slug": "person_box_mask_iou",
        "label": "Person box-mask IoU (%)",
        "title": "Person box-mask IoU",
        "scale": 100.0,
    },
    {
        "letter": "d",
        "key": "val_vehicle_xy_mae_m",
        "slug": "vehicle_xy_error",
        "label": "Vehicle centroid XY MAE (m; lower is better)",
        "title": "Vehicle localization error",
        "scale": 1.0,
    },
    {
        "letter": "e",
        "key": "val_canonical_person_xy_mae_m",
        "slug": "person_xy_error",
        "label": "Person centroid XY MAE (m; lower is better)",
        "title": "Person localization error",
        "scale": 1.0,
    },
    {
        "letter": "f",
        "key": "combined_quality",
        "slug": "joint_model_quality",
        "label": "Joint segmentation-localization quality (%)",
        "title": "Joint model quality",
        "scale": 100.0,
    },
)
PERCENTILES = (0.50, 0.95, 0.99)

# Newest family-wide repaired-v3 edge-compute medians.  The corresponding
# publication ledger did not survive teardown, so the earlier hash-verified
# family distributions retain their shape and are rescaled to these live
# medians.  The source report publishes these values at 0.1 ms precision.
LATEST_EDGE_COMPUTE_P50_MS = {
    "noAE": 67.0,
    "AE128": 66.8,
    "AE64": 53.1,
    "AE32": 46.5,
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"column drift: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def stats(values: Sequence[float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_p50_ms": percentile(values, 0.50),
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_p99_ms": percentile(values, 0.99),
    }


LOCALIZATION_DISTANCE_REFERENCE_M = 1.0


def quality_score(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    segmentation = float(row["val_segmentation_miou"])
    vehicle_iou = float(row["val_vehicle_iou"])
    person_iou = float(row["val_person_box_mask_iou"])
    vehicle_xy_mae_m = float(row["val_vehicle_xy_mae_m"])
    person_xy_mae_m = float(row["val_canonical_person_xy_mae_m"])
    overlap = math.sqrt(max(0.0, vehicle_iou) * max(0.0, person_iou))
    centroid_rms_m = math.sqrt(
        (max(0.0, vehicle_xy_mae_m) ** 2 + max(0.0, person_xy_mae_m) ** 2)
        / 2.0
    )
    centroid_score = math.exp(-centroid_rms_m / LOCALIZATION_DISTANCE_REFERENCE_M)
    localization = math.sqrt(overlap * centroid_score)
    combined = math.sqrt(max(0.0, segmentation) * localization)
    return overlap, centroid_rms_m, centroid_score, localization, combined


def _verified_attempt_csv(root: Path, relative: str) -> Path:
    """Resolve one hash-bound file from a single-cell attempt manifest."""

    ledger = json.loads((root / "campaign_ledger.json").read_text(encoding="utf-8"))
    attempts = list(ledger.get("attempts") or [])
    require(
        len(attempts) == 1 and attempts[0].get("status") == "PASSED",
        f"{root}: expected exactly one passed attempt",
    )
    attempt = root / str(attempts[0]["attempt_dir"])
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = {str(row["path"]): str(row["sha256"]) for row in manifest["files"]}
    require(relative in registered, f"{root}: {relative} is not attempt-manifest bound")
    path = attempt / relative
    require(path.is_file(), f"{root}: missing {relative}")
    require(sha256(path) == registered[relative], f"{root}: hash drift for {relative}")
    return path


def load_sensor_optimization() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the live optimized sensor distribution and its contemporaneous control.

    Source 288-cell values are transformed by equal-percentile mapping from the
    live baseline to the live optimized distribution.  This applies a measured
    distributional change instead of subtracting one constant from every frame.
    """

    presentation_manifest_path = SENSOR_PRESENTATION_ROOT / "artifact_manifest.json"
    presentation_manifest = json.loads(
        presentation_manifest_path.read_text(encoding="utf-8")
    )
    result_path = SENSOR_PRESENTATION_ROOT / "SENSOR_OPTIMIZATION_V2_RESULT.json"
    require(
        presentation_manifest["artifacts"][result_path.name] == sha256(result_path),
        "sensor-optimization presentation result hash drift",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(
        result.get("verdict") == "SENSOR_PREPARATION_OPTIMIZATION_VALIDATED",
        "sensor optimization is not live validated",
    )

    distributions: dict[str, list[float]] = {}
    source_paths: dict[str, Any] = {}
    for label, root in (
        ("baseline", SENSOR_BASELINE_ROOT),
        ("optimized", SENSOR_OPTIMIZED_ROOT),
    ):
        path = _verified_attempt_csv(root, "per_frame_metrics.csv")
        values = []
        production = []
        concatenation = []
        for row in read_csv(path):
            if row.get("prepare_status") != "SENT":
                continue
            prefront = finite(row.get("pre_front_compute_ms"))
            snapshot = finite(row.get("scene_snapshot_ms"))
            total = finite(row.get("profile_sensor_compute_production_estimate_ms"))
            concat = finite(row.get("profile_seven_channel_concatenate_ms"))
            if prefront is None or snapshot is None or total is None or concat is None:
                continue
            values.append(max(0.0, prefront - snapshot))
            production.append(total)
            concatenation.append(concat)
        require(len(values) >= 500, f"{label}: insufficient live sensor samples")
        distributions[f"{label}_pre_action_ms"] = sorted(values)
        distributions[f"{label}_production_ms"] = sorted(production)
        distributions[f"{label}_concatenation_ms"] = sorted(concatenation)
        source_paths[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "samples": len(values),
        }
    return distributions, {
        "method": (
            "equal-percentile mapping from contemporaneous live baseline to "
            "live optimized sensor-preparation distributions"
        ),
        "action_anchor": 50,
        "network_profile_anchor": "FAVORABLE_STABLE",
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256(result_path),
        "presentation_manifest_sha256": sha256(presentation_manifest_path),
        "sources": source_paths,
        "full_population_p50_ms": {
            "baseline": result["total_production_sensor_compute"]["full_sent_population"]["baseline"]["p50_ms"],
            "optimized": result["total_production_sensor_compute"]["full_sent_population"]["optimized"]["p50_ms"],
        },
        "limitations": (
            "one live action/profile anchors the action-independent sensor path; "
            "the replay keeps the 288-cell sent population and radio outcomes fixed"
        ),
    }


def equal_percentile_map(value: float, baseline: Sequence[float], optimized: Sequence[float]) -> float:
    require(bool(baseline) and bool(optimized), "empty sensor calibration distribution")
    rank = bisect.bisect_right(baseline, float(value))
    probability = (rank - 0.5) / len(baseline)
    probability = min(1.0, max(0.0, probability))
    index = int(round(probability * (len(optimized) - 1)))
    return float(optimized[index])


def load_latest_map_service_samples() -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Renderer-off direct edge-to-map publication-to-install samples."""

    ledger = json.loads(
        (SENSOR_OPTIMIZED_ROOT / "campaign_ledger.json").read_text(encoding="utf-8")
    )
    attempt = SENSOR_OPTIMIZED_ROOT / str(ledger["attempts"][0]["attempt_dir"])
    path = attempt / "direct_edge_map/direct_map_ingest.csv"
    require(path.is_file(), "optimized renderer-off direct-map ingest ledger is absent")
    values = [
        float(row["install_latency_from_publish_ms"])
        for row in read_csv(path)
        if row.get("outcome") == "RESULT_INSTALLED"
        and finite(row.get("install_latency_from_publish_ms")) is not None
    ]
    require(len(values) >= 400, "insufficient renderer-off map-service samples")
    pools = {family: list(values) for family in LATEST_EDGE_COMPUTE_P50_MS}
    return pools, {
        "definition": "first direct-map datagram send to authoritative map install",
        "renderer": "off",
        "family_pooling": "single action-independent renderer-off pool",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "samples": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def shift_arrivals_for_sensor_optimization(
    frames: Sequence[Any],
    sent: Sequence[Mapping[str, str]],
    calibration: Mapping[str, Sequence[float]],
    bridge_ns: int,
) -> tuple[list[Any], list[float], list[float], list[float], list[float], int]:
    """Replace sensor compute while preserving measured transport outcomes.

    Imputed capture-to-arrival samples are floored at the shifted UE send
    completion so the scheduler never receives a feature before it was sent.
    """

    baseline = calibration["baseline_pre_action_ms"]
    optimized = calibration["optimized_pre_action_ms"]
    rebuilt = []
    optimized_pre_action: list[float] = []
    optimized_production: list[float] = []
    optimized_concatenation: list[float] = []
    deltas: list[float] = []
    arrival_causal_floor_frames = 0
    require(len(frames) == len(sent), "sensor transform frame/row length drift")
    for frame, row in zip(frames, sent):
        old = max(
            0.0,
            float(row["pre_front_compute_ms"]) - float(row.get("scene_snapshot_ms") or 0.0),
        )
        new = equal_percentile_map(old, baseline, optimized)
        new_production = equal_percentile_map(
            old,
            baseline,
            calibration["optimized_production_ms"],
        )
        new_concatenation = equal_percentile_map(
            old,
            baseline,
            calibration["optimized_concatenation_ms"],
        )
        require(
            new_concatenation <= new_production + 1e-9,
            "seven-channel concatenation exceeds total production sensor compute",
        )
        delta_ns = int(round((new - old) * 1e6))
        arrival_ns = frame.arrival_ns
        if arrival_ns is not None:
            shifted_send_finished_wall_ns = (
                int(row["send_finished_ns"]) + int(bridge_ns) + delta_ns
            )
            shifted_arrival_ns = int(arrival_ns) + delta_ns
            arrival_ns = max(
                int(frame.capture_ns),
                shifted_send_finished_wall_ns,
                shifted_arrival_ns,
            )
            if arrival_ns != shifted_arrival_ns:
                arrival_causal_floor_frames += 1
        rebuilt.append(dataclasses.replace(frame, arrival_ns=arrival_ns))
        optimized_pre_action.append(new)
        optimized_production.append(new_production)
        optimized_concatenation.append(new_concatenation)
        # This is the sensor-timeline shift, not the possibly larger causal
        # floor applied to an imputed scheduler arrival above.
        deltas.append(new - old)
    return (
        rebuilt,
        optimized_pre_action,
        optimized_production,
        optimized_concatenation,
        deltas,
        arrival_causal_floor_frames,
    )


def rescale_edge_compute(
    frames: Sequence[Any], family: str
) -> tuple[list[Any], float | None]:
    """Retain service shape but bind total edge processing to repaired-v3 data."""

    current = [
        float(frame.compute_ns + frame.publication_ns) / 1e6
        for frame in frames
        if frame.arrival_ns is not None
    ]
    # A measured zero-admission cell has no service duration to transform.
    # Preserve it as zero delivery rather than fabricating an edge sample.
    if not current:
        return list(frames), None
    current_p50 = percentile(current, 0.50)
    require(current_p50 is not None and current_p50 > 0, f"{family}: invalid compute median")
    factor = LATEST_EDGE_COMPUTE_P50_MS[family] / current_p50
    rebuilt = []
    for frame in frames:
        if frame.arrival_ns is None:
            rebuilt.append(frame)
            continue
        scaled_total_ns = max(
            frame.publication_ns + 1,
            int(round((frame.compute_ns + frame.publication_ns) * factor)),
        )
        rebuilt.append(
            dataclasses.replace(
                frame,
                compute_ns=scaled_total_ns - frame.publication_ns,
            )
        )
    return rebuilt, factor


def load_tail_stage_samples() -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    """Load pure-tail and non-model service from live optimization anchors.

    These are kept as independent measured marginals. They are not forced
    through the counterfactual service distribution because doing so would
    distort the measured pure-tail CUDA duration.
    """

    pools: dict[str, dict[str, list[float]]] = {}
    provenance: dict[str, Any] = {}
    for family, (action_id, target_root, variant) in final_v3.TARGETS.items():
        files = sorted((target_root / "per_frame").glob(f"action_{action_id}_*.csv"))
        require(len(files) == 1, f"{family}: expected one optimized per-frame file")
        support_ms: list[float] = []
        pure_tail_ms: list[float] = []
        for row in read_csv(files[0]):
            total_ms = finite(row.get("edge_total_edge_processing_ms"))
            tail_ms = finite(row.get("decode_tail_cuda_ms"))
            if total_ms is None or tail_ms is None or total_ms <= 0 or tail_ms < 0:
                continue
            require(tail_ms <= total_ms + 1e-9, f"{family}: pure tail exceeds edge service")
            pure_tail_ms.append(tail_ms)
            support_ms.append(max(0.0, total_ms - tail_ms))
        require(bool(pure_tail_ms), f"{family}: no pure-tail calibration samples")
        source_total_p50 = percentile(
            [tail + support for tail, support in zip(pure_tail_ms, support_ms)],
            0.50,
        )
        target_total_p50 = LATEST_EDGE_COMPUTE_P50_MS[family]
        require(
            source_total_p50 is not None and source_total_p50 > 0,
            f"{family}: invalid source total edge median",
        )
        # Preserve the measured pure FCOS CUDA duration.  Scale only the
        # non-model support until the paired total reaches the newest live
        # repaired-v3 median.  Binary search is deterministic and avoids the
        # false assumption that medians are additive.
        low, high = 0.0, 8.0
        for _ in range(80):
            factor = (low + high) / 2.0
            candidate = percentile(
                [
                    tail + support * factor
                    for tail, support in zip(pure_tail_ms, support_ms)
                ],
                0.50,
            )
            if candidate is not None and candidate < target_total_p50:
                low = factor
            else:
                high = factor
        support_factor = (low + high) / 2.0
        support_ms = [value * support_factor for value in support_ms]
        calibrated_total_p50 = percentile(
            [tail + support for tail, support in zip(pure_tail_ms, support_ms)],
            0.50,
        )
        require(
            calibrated_total_p50 is not None
            and abs(calibrated_total_p50 - target_total_p50) < 0.01,
            f"{family}: newest edge median calibration failed",
        )
        pools[family] = {
            "model_tail_ms": pure_tail_ms,
            "tail_support_ms": support_ms,
        }
        provenance[family] = {
            "anchor_action_id": action_id,
            "variant": variant,
            "per_frame_path": str(files[0].relative_to(ROOT)),
            "per_frame_sha256": sha256(files[0]),
            "samples": len(pure_tail_ms),
            "pure_model_tail_p50_ms": percentile(pure_tail_ms, 0.50),
            "pure_model_tail_p95_ms": percentile(pure_tail_ms, 0.95),
            "tail_support_p50_ms": percentile(support_ms, 0.50),
            "tail_support_p95_ms": percentile(support_ms, 0.95),
            "source_total_edge_p50_ms": source_total_p50,
            "latest_total_edge_p50_ms": target_total_p50,
            "calibrated_total_edge_p50_ms": calibrated_total_p50,
            "tail_support_scale_factor": support_factor,
            "decomposition": (
                "pure tail is decode_tail_cuda_ms; tail support is measured "
                "optimized total edge processing minus pure tail, rescaled "
                "to the newest repaired-v3 family median"
            ),
        }
    report_path = (
        ROOT
        / "rl_agent/splitfusion_direct_edge_map_v1/"
        "EDGE_OPTIMIZATION_VALIDATION_RESULTS.md"
    )
    provenance["newest_live_validation"] = {
        "root": str(LATEST_EDGE_VALIDATION_ROOT.relative_to(ROOT)),
        "artifact_manifest_sha256": sha256(
            LATEST_EDGE_VALIDATION_ROOT / "artifact_manifest.json"
        ),
        "report_path": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256(report_path),
        "published_precision_ms": 0.1,
    }
    return pools, provenance


def load_direct_map_stage_samples() -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Load measured direct publication-to-install durations by family."""

    action_to_family = {
        int(action_id): family
        for family, (action_id, _target_root, _variant) in final_v3.TARGETS.items()
    }
    pools: dict[str, list[float]] = defaultdict(list)
    files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(LIVE_DIRECT_ROOT.rglob("direct_map_ingest.csv")):
        rows = read_csv(path)
        action_ids = {int(row["action_id"]) for row in rows if row.get("action_id")}
        require(len(action_ids) == 1, f"mixed direct-map action identities: {path}")
        family = action_to_family[next(iter(action_ids))]
        accepted = 0
        for row in rows:
            publish_ms = finite(row.get("install_latency_from_publish_ms"))
            tail_ms = finite(row.get("install_latency_from_tail_ms"))
            if publish_ms is None or tail_ms is None or tail_ms <= 0:
                continue
            require(0 <= publish_ms <= tail_ms + 1e-9, f"invalid direct-map substage: {path}")
            pools[family].append(publish_ms)
            accepted += 1
        require(accepted > 0, f"no direct-map substage samples: {path}")
        files[family].append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "samples": accepted,
            }
        )
    require(set(pools) == set(final_v3.TARGETS), "direct-map substage family coverage drift")
    return dict(pools), {
        "definition": (
            "measured install_latency_from_publish_ms on the live direct "
            "container-to-host map path"
        ),
        "per_family_files": dict(files),
    }


def useful_outcomes(result: Any) -> list[Any]:
    installed = sorted(
        (item for item in result.outcomes if item.install_ns is not None),
        key=lambda item: (int(item.install_ns), int(item.frame.sequence_id)),
    )
    useful: list[Any] = []
    newest_capture = -1
    for item in installed:
        capture = int(item.frame.capture_ns)
        if capture > newest_capture and int(item.install_ns) < int(result.observation_end_ns):
            useful.append(item)
            newest_capture = capture
    return useful


def sensor_components(rows: Sequence[Mapping[str, str]], bridge_ns: int) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        capture_wall_ns = timing.wall_seconds_to_ns(row["capture_wall_s"])
        action_start_ns = int(row["capture_started_ns"]) + bridge_ns
        capture_to_action = (action_start_ns - capture_wall_ns) / 1e6
        prefront = float(row["pre_front_compute_ms"])
        callback_to_worker = capture_to_action - prefront
        radar_window = float(row["radar_window_ms"])
        radar_prepare = float(row["radar_prepare_ms"])
        rgb_convert = float(row["rgb_convert_ms"])
        scene_snapshot = float(row["scene_snapshot_ms"])
        residual = prefront - radar_window - radar_prepare - rgb_convert - scene_snapshot
        require(capture_to_action >= -0.01, "capture-to-action interval is negative")
        require(callback_to_worker >= -0.01, "callback-to-worker interval is negative")
        require(residual >= -0.01, "pre-front residual is negative")
        values["capture_to_action_ms"].append(max(0.0, capture_to_action))
        values["callback_to_worker_ms"].append(max(0.0, callback_to_worker))
        values["radar_window_ms"].append(radar_window)
        values["radar_prepare_ms"].append(radar_prepare)
        values["rgb_convert_ms"].append(rgb_convert)
        values["evaluation_snapshot_ms"].append(scene_snapshot)
        values["prefront_residual_ms"].append(max(0.0, residual))
        # Operational wait is diagnostic, not additive to capture-based age:
        # its start can precede the RGB capture event used as time zero.
        values["sensor_wait_nonadditive_ms"].append(float(row["sensor_wait_ms"]))
    return values


def pareto_ids(rows: Sequence[Mapping[str, Any]], x_key: str) -> set[int]:
    points = [
        (float(row[x_key]), float(row["combined_quality"]), int(row["action_id"]))
        for row in rows
        if row.get(x_key) not in (None, "")
    ]
    frontier: set[int] = set()
    best_quality = -math.inf
    for latency, quality, action_id in sorted(points):
        if quality > best_quality + 1e-12:
            frontier.add(action_id)
            best_quality = quality
    return frontier


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_quality_latency(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    names: list[str] = []
    for number, stage in enumerate(SCATTER_STAGES, start=1):
        label = STAGES[stage]
        key = f"{stage}_p50_ms"
        for view in QUALITY_VIEWS:
            quality_key = str(view["key"])
            scale = float(view["scale"])
            if stage == "ue_action":
                # The UE action is not caused by the radio profile. Pool its
                # four independent live-cell medians per action so Figure 01
                # does not present host/scene jitter as a network effect.
                pooled: list[dict[str, Any]] = []
                for action_id in range(72):
                    action_rows = [
                        row for row in rows if int(row["action_id"]) == action_id
                    ]
                    require(len(action_rows) == 4, f"action {action_id}: profile drift")
                    record = dict(action_rows[0])
                    values = [
                        float(row[key]) for row in action_rows if row[key] != ""
                    ]
                    require(len(values) == 4, f"action {action_id}: UE timing absent")
                    record[key] = statistics.median(values)
                    pooled.append(record)
                fig, ax = plt.subplots(figsize=(10.5, 7.5))
                for family, color in FAMILY_COLOR.items():
                    group = [row for row in pooled if row["family"] == family]
                    ax.scatter(
                        [float(row[key]) for row in group],
                        [scale * float(row[quality_key]) for row in group],
                        s=76,
                        alpha=0.94,
                        c=color,
                        edgecolors="#202020",
                        linewidths=0.95,
                        label=family,
                    )
                ax.set_xlabel(f"{label} P50 (ms)")
                ax.set_ylabel(str(view["label"]))
                ax.set_title(
                    f"{view['title']} vs 7-channel-concat-start-to-UDP-send latency"
                )
                ax.grid(alpha=0.25)
                ax.legend(ncol=4, loc="upper center", frameon=False)
                ax.tick_params(axis="both", labelsize=9, width=1.2)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontweight("bold")
                fig.tight_layout()
                name = f"{number:02d}{view['letter']}_{view['slug']}_vs_{stage}_p50"
                save_figure(fig, output / name)
                names.extend([name + ".png", name + ".pdf"])
                continue
            fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=True)
            for ax, profile in zip(axes.flat, PROFILE_ORDER):
                selected = [row for row in rows if row["network_profile"] == profile]
                for family, color in FAMILY_COLOR.items():
                    group = [
                        row
                        for row in selected
                        if row["family"] == family
                        and row[key] != ""
                        and row[quality_key] != ""
                    ]
                    ax.scatter(
                        [float(row[key]) for row in group],
                        [scale * float(row[quality_key]) for row in group],
                        s=68,
                        alpha=0.94,
                        c=color,
                        edgecolors="#202020",
                        linewidths=0.9,
                        label=family,
                    )
                ax.set_title(PROFILE_LABEL[profile])
                ax.set_xlabel(f"{label} P50 (ms)")
                ax.set_ylabel(str(view["label"]))
                ax.grid(alpha=0.25)
                ax.tick_params(axis="both", labelsize=9, width=1.2)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontweight("bold")
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            sentence_label = label[0].lower() + label[1:]
            fig.suptitle(
                f"{view['title']} vs {sentence_label} by network profile",
                y=0.995,
            )
            fig.legend(
                handles,
                legend_labels,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            name = f"{number:02d}{view['letter']}_{view['slug']}_vs_{stage}_p50"
            save_figure(fig, output / name)
            names.extend([name + ".png", name + ".pdf"])
    return names


def action_balanced_latency(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    network_common_actions = set(range(72))
    for profile in PROFILE_ORDER:
        represented = {
            int(row["action_id"])
            for row in rows
            if row["network_profile"] == profile and row["network_p50_ms"] != ""
        }
        network_common_actions &= represented
    require(
        bool(network_common_actions),
        "no common observed feature-uplink action support",
    )
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["network_profile"] == profile]
        for stage, label in STAGES.items():
            stage_rows = selected
            support_rule = "ALL_AVAILABLE_ACTIONS"
            if stage == "network":
                stage_rows = [
                    row
                    for row in selected
                    if int(row["action_id"]) in network_common_actions
                ]
                support_rule = "COMMON_ACTION_SUPPORT_ACROSS_ALL_PROFILES"
            record: dict[str, Any] = {
                "network_profile": profile,
                "stage": stage,
                "stage_label": label,
                "actions_total": len(selected),
                "support_rule": support_rule,
                "support_actions": len(stage_rows),
            }
            for probability in PERCENTILES:
                suffix = int(probability * 100)
                values = [
                    float(row[f"{stage}_p{suffix}_ms"])
                    for row in stage_rows
                    if row[f"{stage}_p{suffix}_ms"] != ""
                ]
                record[f"action_balanced_p{suffix}_ms"] = percentile(values, 0.50)
                record[f"actions_with_p{suffix}"] = len(values)
            output.append(record)
    return output


def plot_latency_bars(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    network_supports = {
        int(row["support_actions"])
        for row in rows
        if row["stage"] == "network"
    }
    require(len(network_supports) == 1, "network common-support count drift")
    network_support = next(iter(network_supports))
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 10.5), sharey=False)
    components = (
        "sensor_compute",
        "ue_action",
        "network",
        "edge_queue",
        "model_tail",
        "tail_support",
        "map_install",
    )
    colors = ("#4C78A8", "#F58518", "#54A24B")
    x = np.arange(len(components))
    width = 0.23
    for ax, profile in zip(axes.flat, PROFILE_ORDER):
        selected = {row["stage"]: row for row in rows if row["network_profile"] == profile}
        for offset, suffix in enumerate((50, 95, 99)):
            heights = [float(selected[stage][f"action_balanced_p{suffix}_ms"]) for stage in components]
            bars = ax.bar(x + (offset - 1) * width, heights, width, color=colors[offset], label=f"P{suffix}")
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2, fontweight="bold")
        ax.set_xticks(
            x,
            (
                "Sensor compute\nbefore concat",
                "7-channel concat\nto UDP send",
                "Feature\nuplink",
                "Tail-busy\nwait",
                "FCOS model\ntail",
                "Other tail\nprocessing",
                "Map\nservice",
            ),
        )
        ax.set_ylabel("Latency (ms)")
        ax.set_title(PROFILE_LABEL[profile])
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Action-balanced latency percentiles by causal stage", y=0.995)
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        f"Marginal percentiles are not additive. Feature uplink uses the same {network_support} observed actions in every profile.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    name = "05_latency_percentiles_by_profile"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def plot_feature_delivery(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> list[str]:
    """Weighted complete application-message delivery for each radio profile."""

    configure_plot()
    percentages: list[float] = []
    counts: list[tuple[int, int]] = []
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["network_profile"] == profile]
        sent = sum(int(row["frames_sent"]) for row in selected)
        reassembled = sum(
            int(row["measured_complete_reassemblies"]) for row in selected
        )
        require(0 <= reassembled <= sent, f"{profile}: invalid delivery accounting")
        percentages.append(100.0 * reassembled / sent)
        counts.append((reassembled, sent))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    x = np.arange(len(PROFILE_ORDER))
    colors = ("#4C78A8", "#F58518", "#E45756", "#54A24B")
    bars = ax.bar(x, percentages, width=0.62, color=colors, edgecolor="#202020")
    ax.bar_label(
        bars,
        labels=[
            f"{value:.1f}%\n{reassembled:,}/{sent:,}"
            for value, (reassembled, sent) in zip(percentages, counts)
        ],
        padding=4,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(x, [PROFILE_LABEL[profile] for profile in PROFILE_ORDER])
    ax.set_ylabel("Complete feature delivery (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Measured complete feature delivery by network profile")
    ax.grid(axis="y", alpha=0.25)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.text(
        0.5,
        0.012,
        "Complete application reassemblies / frames sent, weighted over all 72 actions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    name = "07_feature_delivery_percentage_by_profile"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def sensor_optimization_stage_rows() -> list[dict[str, Any]]:
    result_path = SENSOR_PRESENTATION_ROOT / "SENSOR_OPTIMIZATION_V2_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in result["stages_full_sent_population"]:
        record: dict[str, Any] = {
            "stage": item["stage"],
            "field": item["field"],
            "label": item["label"],
            "changed_by_this_work": bool(item["changed_by_this_work"]),
        }
        for version in ("baseline", "optimized"):
            for suffix in ("count", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                record[f"{version}_{suffix}"] = item[version][suffix]
        rows.append(record)
    return rows


def plot_optimized_sensor_breakdown(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> list[str]:
    """Figure 06: per-function live sensor timing before and after optimization."""

    configure_plot()
    radar = [row for row in rows if 7 <= int(str(row["stage"])[1:]) <= 13]
    camera = [row for row in rows if 14 <= int(str(row["stage"])[1:]) <= 25]
    fig, axes = plt.subplots(2, 3, figsize=(22, 11.5), sharey=False)
    colors = {"baseline": "#9ECAE1", "optimized": "#2C7FB8"}
    for row_index, (group, group_name) in enumerate(
        ((radar, "Radar preparation"), (camera, "RGB and tensor preparation"))
    ):
        labels = [f"{row['stage']}\n{row['label']}" for row in group]
        x = np.arange(len(group))
        for column, suffix in enumerate((50, 95, 99)):
            ax = axes[row_index, column]
            width = 0.36
            for offset, version in enumerate(("baseline", "optimized")):
                heights = [float(row[f"{version}_p{suffix}_ms"]) for row in group]
                ax.bar(
                    x + (offset - 0.5) * width,
                    heights,
                    width,
                    color=colors[version],
                    label=version.capitalize(),
                )
            ax.set_xticks(x, labels, rotation=31, ha="right")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"{group_name}: P{suffix}")
            ax.grid(axis="y", alpha=0.25)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Live sensor-preparation function timing before and after optimization",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
    )
    fig.text(
        0.5,
        0.008,
        "Component percentiles are marginal and are not additive; all rows use the full sent-frame population.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    name = "06_sensor_preparation_optimized_breakdown"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def sensor_profile_rows(cell_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stages = (
        "callback_to_worker",
        "radar_window",
        "radar_prepare",
        "rgb_convert",
        "evaluation_snapshot",
        "prefront_residual",
        "capture_to_action",
        "sensor_wait_nonadditive",
    )
    result: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        selected = [row for row in cell_rows if row["network_profile"] == profile]
        for stage in stages:
            record: dict[str, Any] = {
                "network_profile": profile,
                "stage": stage,
                "actions_total": len(selected),
            }
            for suffix in (50, 95, 99):
                values = [float(row[f"{stage}_p{suffix}_ms"]) for row in selected]
                record[f"action_balanced_p{suffix}_ms"] = percentile(values, 0.50)
            result.append(record)
    return result


def plot_sensor_breakdown(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), sharey=True)
    stages = (
        "callback_to_worker",
        "radar_window",
        "radar_prepare",
        "rgb_convert",
        "evaluation_snapshot",
        "prefront_residual",
    )
    labels = ("Callback→worker", "Radar window", "Radar prepare", "RGB convert", "Eval snapshot", "Residual")
    x = np.arange(len(stages))
    width = 0.23
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for ax, profile in zip(axes.flat, PROFILE_ORDER):
        selected = {row["stage"]: row for row in rows if row["network_profile"] == profile}
        for offset, suffix in enumerate((50, 95, 99)):
            heights = [float(selected[stage][f"action_balanced_p{suffix}_ms"]) for stage in stages]
            ax.bar(x + (offset - 1) * width, heights, width, color=colors[offset], label=f"P{suffix}")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(PROFILE_LABEL[profile])
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Retained pre-action timing breakdown", y=0.995)
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    name = "06_sensor_preparation_retained_breakdown"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def make_report(
    rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    sensor_rows: Sequence[Mapping[str, Any]],
    bridge_audit: Mapping[str, Any],
) -> str:
    by_stage = {
        (row["network_profile"], row["stage"]): row for row in latency_rows
    }
    network_supports = {
        int(row["support_actions"])
        for row in latency_rows
        if row["stage"] == "network"
    }
    require(len(network_supports) == 1, "network common-support count drift")
    network_support = next(iter(network_supports))
    arrival_causal_floor_frames = sum(
        int(row["scheduler_arrival_causal_floor_frames"]) for row in rows
    )
    lines = [
        "# SplitFusion optimized-pipeline latency and quality analysis",
        "",
        "This is an offline causal replay of the immutable 288-cell sent-frame",
        "population. It applies the subsequently live-validated sensor preparation,",
        "repaired-v3 edge service, renderer-off direct edge-to-map service, and",
        "latest-only scheduling. It is not a second 288-cell live campaign.",
        "",
        "## What changed and what did not",
        "",
        "- The measured radio reassembly/admission outcomes and observed per-frame transport delays are held fixed. No missing uplink frame is fabricated.",
        "- Scheduler-only imputed capture-to-arrival samples are causally floored at shifted UE send completion; observed uplink timing is never imputed into the network plots.",
        "- Sensor timing is changed by equal-percentile mapping from the contemporaneous live baseline distribution to the live optimized distribution; this is not a constant subtraction.",
        "- The measured family service shapes are retained and rescaled to the newest repaired-v3 live edge-compute medians.",
        "- Direct map service is sampled from the renderer-off live action-50 run. It is independent of split action and does not traverse the radio.",
        "- Validation quality is action-dependent and unchanged by runtime optimization. Network profile changes delivery, latency, and freshness—not the offline quality anchor.",
        "- A point is absent when its conditional stage has no trustworthy sample or no useful map installation; absence is never encoded as zero latency.",
        "- Physical map age ends at authoritative map installation. The later compact UE feedback is excluded.",
        "",
        "## Action-balanced stage percentiles",
        "",
        "| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        for stage in (
            "sensor_compute",
            "pure_front",
            "ue_action",
            "network",
            "edge_queue",
            "model_tail",
            "tail_support",
            "map_install",
            "edge_map",
            "total",
        ):
            row = by_stage[(profile, stage)]
            lines.append(
                f"| {PROFILE_LABEL[profile]} | {STAGES[stage]} | "
                f"{float(row['action_balanced_p50_ms']):.1f} | "
                f"{float(row['action_balanced_p95_ms']):.1f} | "
                f"{float(row['action_balanced_p99_ms']):.1f} | "
                f"{int(row['actions_with_p50'])}/72 |"
            )
    lines.extend(
        [
            "",
            "## Complete feature delivery",
            "",
            "| Profile | Complete reassemblies | Frames sent | Delivery |",
            "|---|---:|---:|---:|",
        ]
    )
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["network_profile"] == profile]
        sent = sum(int(row["frames_sent"]) for row in selected)
        reassembled = sum(
            int(row["measured_complete_reassemblies"]) for row in selected
        )
        lines.append(
            f"| {PROFILE_LABEL[profile]} | {reassembled:,} | {sent:,} | "
            f"{100.0 * reassembled / sent:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.",
            "The stage percentiles are marginals with different conditional denominators and must not be added to reconstruct an end-to-end percentile. Figure 04 and the `total_*` columns provide the causally simulated seven-channel-concatenation-start-to-map distribution.",
            "",
            "## Scheduling and causal boundaries",
            "",
            "- Optimized sensor computation excludes waiting for CARLA to produce a synchronized sample and ends when seven-channel concatenation starts.",
            "- The UE action path begins at seven-channel concatenation and continues through front inference, ranker/selection, compression/packing, serialization, and the UDP send loop. The 288 cells timestamp the boundary immediately after concatenation; the live optimized P23 distribution is therefore added explicitly.",
            "- Feature uplink runs from UE send completion to complete edge reassembly. Only retained same-clock observed receipts are plotted; imputed arrivals used by the scheduler are excluded from this metric.",
            "- Latest-only scheduling removes a multi-frame FIFO but cannot preempt a running CUDA/tail call. At most one newest frame waits; older pending frames receive explicit `SUPERSEDED_PENDING` outcomes.",
            "- Edge-to-map includes tail-busy waiting, edge processing/publication, and renderer-off map service. Action-start-to-map excludes sensor preparation, as requested. Capture-to-map remains in `capture_total_*` for physical freshness accounting but is not used in Figures 01 or 04.",
            "",
            "## Sensor optimization",
            "",
            "Figure 06 uses the full sent-frame populations from the live action-50 FAVORABLE_STABLE baseline and optimized cells. It reports production stages P07–P23 plus P25 at their measured function boundaries; P24 is evaluation-only and P26 is diagnostic synchronization, so neither is included. Camera and radar callbacks are distinct, but the numerical preparation for one selected frame remains sequential in the front worker. Component percentiles are marginal and cannot be summed.",
            "",
            "## Clock and denominator integrity",
            "",
            f"The same-host clock bridge used {int(bridge_audit['anchor_count']):,} anchors; its absolute error P99 was {float(bridge_audit['absolute_deviation_ms_p99']):.6f} ms.",
            f"Every stage carries its own count. Sensor, pure-front, and UE-action timing use all sent frames. Feature-uplink timing uses the same {network_support}-action observed support in every profile. Edge and total timing use useful direct-map installations.",
            f"The scheduler causality floor affected {arrival_causal_floor_frames:,} imputed arrivals. It prevents a replay-only feature arrival from preceding that frame's shifted send completion; it does not change measured reassembly/admission counts or enter the observed-uplink plots.",
            "",
            "## Quality definition",
            "",
            "$$",
            "Q_{\\mathrm{overlap}}=\\sqrt{\\mathrm{IoU}_{\\mathrm{vehicle}}\\,\\mathrm{IoU}_{\\mathrm{person}}},",
            "\\qquad",
            "e_{xy}=\\sqrt{\\frac{e_{\\mathrm{vehicle}}^2+e_{\\mathrm{person}}^2}{2}},",
            "\\qquad Q_{xy}=\\exp(-e_{xy}/1\\,\\mathrm{m}),",
            "$$",
            "",
            "$$",
            "Q_{\\mathrm{loc}}=\\sqrt{Q_{\\mathrm{overlap}}Q_{xy}},",
            "\\qquad",
            "Q_{\\mathrm{joint}}=\\sqrt{mIoU_{\\mathrm{seg}}Q_{\\mathrm{loc}}}.",
            "$$",
            "",
            "The overlap terms measure spatial box/footprint agreement; they are not class-specific semantic-segmentation IoUs. Centroid XY MAE is explicit through a smooth one-metre reference scale. The one-metre value normalizes the presentation coordinate and is not a correctness gate. Geometric means are conservative: one strong dimension cannot hide a weak one. `Q_joint` is a presentation coordinate, not a calibrated probability and not yet the PPO reward.",
            "",
            "## Figure guide",
            "",
            "Figures 01a–01f show quality against the cross-profile-pooled seven-channel-concatenation-start-to-UDP-send latency. Figures 02a–02f use observed feature-uplink latency. Figures 03a–03f use optimized edge-reassembly-to-map latency. Figures 04a–04f use optimized seven-channel-concatenation-start-to-map latency. Views a–e keep semantic mIoU, vehicle overlap, person box-mask overlap, vehicle centroid error, and person centroid error separate; view f restores joint model quality. Figure 05 gives action-balanced P50/P95/P99 causal-stage marginals using common action support for uplink. Figure 06 shows the measured live sensor function breakdown before and after optimization. Figure 07 reports weighted complete feature delivery for each network profile.",
            "",
            "## Limitations",
            "",
            "- Sensor optimization is anchored by one live action/profile because the sensor path is action-independent; run-to-run host variation remains possible.",
            "- Newest edge medians are live for one action per family. The older hash-verified family distributions provide the residual shape because the newest publication ledger did not survive that validation run.",
            "- Renderer-off map service is a single action-50 live pool and is intentionally treated as action-independent.",
            "- The replay can estimate changed installation and freshness behavior under these measured transformations, but it is not a substitute for a new 288-cell live campaign.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output exists: {output}")
    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    bridge_ns, bridge_audit, bridge_hashes = timing.collect_clock_bridge(cells)
    final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    sensor_calibration, sensor_provenance = load_sensor_optimization()
    tail_stage_samples, tail_stage_provenance = load_tail_stage_samples()
    map_service_samples, map_service_provenance = load_latest_map_service_samples()
    quality = source._quality_by_action()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        pool_hashes,
    ) = source._build_empirical_pools(cells)
    require(pool_hashes == bridge_hashes, "per-frame source hash verification drift")

    rows: list[dict[str, Any]] = []
    observed_network_intervals = 0
    network_boundary_inversions_excluded = 0
    scheduler_arrival_causal_floor_frames = 0
    sensor_delta_samples: list[float] = []
    edge_rescale_factors: dict[str, list[float]] = defaultdict(list)
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = sorted(source._sent_rows(attempt), key=lambda row: float(row["capture_wall_s"]))
        family = cell["family"]
        frames, counters = source._candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=calibration[family],
            publication_samples=publication_samples[family],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get((int(cell["action_id"]), cell["network_profile"]), ()),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        (
            frames,
            optimized_pre_action_ms,
            optimized_sensor_production_ms,
            optimized_concatenation_ms,
            sensor_deltas_ms,
            cell_arrival_causal_floor_frames,
        ) = shift_arrivals_for_sensor_optimization(
            frames,
            sent,
            sensor_calibration,
            bridge_ns,
        )
        scheduler_arrival_causal_floor_frames += cell_arrival_causal_floor_frames
        optimized_sensor_compute_ms = [
            production - concatenation
            for production, concatenation in zip(
                optimized_sensor_production_ms,
                optimized_concatenation_ms,
            )
        ]
        sensor_delta_samples.extend(sensor_deltas_ms)
        frames, edge_rescale_factor = rescale_edge_compute(frames, family)
        if edge_rescale_factor is not None:
            edge_rescale_factors[family].append(edge_rescale_factor)
        delay_pool = [
            max(0, int(round(value * 1e6)))
            for value in map_service_samples[family]
        ]
        frames, _ = direct._direct_frames(frames, cell_id=cell["cell_id"], delay_pool=delay_pool)
        publication_ns = int(round(calibration[family]["optimized_publication_ms_median"] * 1e6))
        total_ns = int(round(LATEST_EDGE_COMPUTE_P50_MS[family] * 1e6))
        result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=None,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
                predicted_compute_ns=max(1, total_ns - publication_ns),
                predicted_publication_ns=publication_ns,
                predicted_post_publication_install_ns=int(statistics.median(delay_pool)),
            ),
        )
        summary = result.summary()
        require(
            sum(int(value) for value in summary["reason_counts"].values())
            == len(sent),
            f"{cell['cell_id']}: simulated terminal accounting drift",
        )
        require(
            int(summary["ack_installed_frames"])
            <= int(counters["measured_edge_admissions"]),
            f"{cell['cell_id']}: installs exceed measured edge admissions",
        )

        ue_action_ms: list[float] = []
        pure_front_ms: list[float] = []
        for row, concatenation_ms in zip(sent, optimized_concatenation_ms):
            action_start_ns = int(row["capture_started_ns"])
            send_finish_ns = int(row["send_finished_ns"])
            ue_action_ms.append(
                concatenation_ms + (send_finish_ns - action_start_ns) / 1e6
            )
            parsed = ast.literal_eval(row["front_timing_ns"])
            if finite(parsed.get("front_backbone")) is not None:
                pure_front_ms.append(float(parsed["front_backbone"]) / 1e6)
        network_ms: list[float] = []
        for frame, row, delta_ms in zip(frames, sent, sensor_deltas_ms):
            # The published replay imputes enough arrivals to reproduce the
            # measured aggregate admission counts. Those synthetic arrivals
            # are necessary for scheduling, but are not network measurements.
            # Plot only the retained, same-clock-bridge edge receipt here.
            if frame.arrival_ns is None or not row.get("edge_receipt_wall_s"):
                continue
            start_ns = (
                int(row["send_finished_ns"])
                + bridge_ns
                + int(round(delta_ms * 1e6))
            )
            value = (int(frame.arrival_ns) - start_ns) / 1e6
            # The receiver can timestamp the final datagram before the
            # sender thread returns from its final sendto() call.  Such a row
            # does not define a causal send-finished-to-reassembly interval;
            # exclude it rather than clamping or changing the start boundary.
            if value < -0.001:
                network_boundary_inversions_excluded += 1
                continue
            network_ms.append(max(0.0, value))
        observed_network_intervals += len(network_ms)
        edge_map_ms: list[float] = []
        edge_queue_ms: list[float] = []
        model_tail_ms = tail_stage_samples[family]["model_tail_ms"]
        tail_support_ms = tail_stage_samples[family]["tail_support_ms"]
        map_install_ms = map_service_samples[family]
        total_ms: list[float] = []
        capture_total_ms: list[float] = []
        for item in useful_outcomes(result):
            edge_map_ms.append((int(item.install_ns) - int(item.frame.arrival_ns)) / 1e6)
            require(
                item.compute_start_ns is not None
                and item.compute_finish_ns is not None
                and item.publication_start_ns is not None
                and item.publication_finish_ns is not None,
                f"{cell['cell_id']}: useful install lacks stage timing",
            )
            compute_wait_ns = int(item.compute_start_ns) - int(item.frame.arrival_ns)
            publication_wait_ns = int(item.publication_start_ns) - int(item.compute_finish_ns)
            require(compute_wait_ns >= 0, f"{cell['cell_id']}: negative compute wait")
            require(publication_wait_ns >= 0, f"{cell['cell_id']}: negative publication wait")
            edge_queue_ms.append((compute_wait_ns + publication_wait_ns) / 1e6)
            sequence = int(item.frame.sequence_id)
            row = sent[sequence]
            optimized_action_start_wall_ns = (
                int(row["capture_started_ns"])
                + bridge_ns
                + int(round(sensor_deltas_ms[sequence] * 1e6))
                - int(round(optimized_concatenation_ms[sequence] * 1e6))
            )
            action_total = (
                int(item.install_ns) - optimized_action_start_wall_ns
            ) / 1e6
            require(
                action_total >= -0.001,
                (
                    f"{cell['cell_id']}: action-to-map interval is negative "
                    f"for sequence {sequence}: {action_total:.6f} ms "
                    f"(install={int(item.install_ns)}, "
                    f"action_start={optimized_action_start_wall_ns}, "
                    f"arrival={int(item.frame.arrival_ns)}, "
                    f"sensor_delta_ms={sensor_deltas_ms[sequence]:.6f}, "
                    f"concat_ms={optimized_concatenation_ms[sequence]:.6f})"
                ),
            )
            total_ms.append(max(0.0, action_total))
            capture_total_ms.append(
                (int(item.install_ns) - int(item.frame.capture_ns)) / 1e6
            )

        qoverlap, centroid_rms_m, qxy, qloc, qjoint = quality_score(
            quality[int(cell["action_id"])]
        )
        record: dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "action_id": int(cell["action_id"]),
            "profile_id": cell["profile_id"],
            "network_profile": cell["network_profile"],
            "family": family,
            "quantizer": cell["quantizer"],
            "q": float(cell["q"]),
            "frames_sent": len(sent),
            "measured_complete_reassemblies": int(counters["measured_reassemblies"]),
            "measured_edge_admissions": int(counters["measured_edge_admissions"]),
            "simulated_map_installs": int(summary["ack_installed_frames"]),
            "simulated_useful_newer_map_installs": int(summary["useful_newer_map_installations"]),
            "median_payload_bytes": percentile([float(row["payload_bytes"]) for row in sent], 0.50),
            **quality[int(cell["action_id"])],
            "localization_overlap": qoverlap,
            "localization_centroid_rms_m": centroid_rms_m,
            "localization_centroid_score": qxy,
            "localization_quality": qloc,
            "combined_quality": qjoint,
            "rate_reassembled_per_sent": int(counters["measured_reassemblies"]) / len(sent),
            "rate_admitted_per_sent": int(counters["measured_edge_admissions"]) / len(sent),
            "rate_installed_per_sent": int(summary["ack_installed_frames"]) / len(sent),
            "rate_useful_installations_per_sent": int(summary["useful_newer_map_installations"]) / len(sent),
            "network_latency_observed_only": True,
            "scheduler_arrival_causal_floor_frames": cell_arrival_causal_floor_frames,
            **direct._fresh_fractions(result, (150, 200, 250, 300)),
            **stats(ue_action_ms, "ue_action"),
            **stats(optimized_sensor_compute_ms, "sensor_compute"),
            **stats(optimized_sensor_production_ms, "sensor_compute_including_concat"),
            **stats(optimized_concatenation_ms, "seven_channel_concat"),
            **stats(pure_front_ms, "pure_front"),
            **stats(network_ms, "network"),
            **stats(edge_queue_ms, "edge_queue"),
            **stats(model_tail_ms, "model_tail"),
            **stats(tail_support_ms, "tail_support"),
            **stats(map_install_ms, "map_install"),
            **stats(edge_map_ms, "edge_map"),
            **stats(total_ms, "total"),
            **stats(capture_total_ms, "capture_total"),
        }
        for reason, count in sorted(summary["reason_counts"].items()):
            record[f"terminal_{reason.lower()}"] = int(count)
        # CSV uses an empty string, rather than a fabricated zero, for an absent stage.
        record = {key: "" if value is None else value for key, value in record.items()}
        rows.append(record)
        if number % 24 == 0:
            print(f"supervisor analysis: {number}/288 cells", flush=True)

    require(len(rows) == 288, "analysis inventory drift")
    aggregate_counts = {
        "frames_sent": sum(int(row["frames_sent"]) for row in rows),
        "measured_complete_reassemblies": sum(
            int(row["measured_complete_reassemblies"]) for row in rows
        ),
        "measured_edge_admissions": sum(
            int(row["measured_edge_admissions"]) for row in rows
        ),
        "simulated_map_installs": sum(
            int(row["simulated_map_installs"]) for row in rows
        ),
        "simulated_useful_newer_map_installs": sum(
            int(row["simulated_useful_newer_map_installs"]) for row in rows
        ),
    }
    require(
        aggregate_counts["simulated_map_installs"]
        <= aggregate_counts["measured_edge_admissions"]
        <= aggregate_counts["measured_complete_reassemblies"]
        <= aggregate_counts["frames_sent"],
        "aggregate causal count ordering failed",
    )
    latency_rows = action_balanced_latency(rows)
    sensor_rows = sensor_optimization_stage_rows()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "action_profile_quality_latency.csv", rows)
    write_csv(output / "latency_percentiles_by_profile.csv", latency_rows)
    write_csv(output / "sensor_optimization_stage_percentiles.csv", sensor_rows)
    figure_names = []
    figure_names.extend(plot_quality_latency(rows, output))
    figure_names.extend(plot_latency_bars(latency_rows, output))
    figure_names.extend(plot_optimized_sensor_breakdown(sensor_rows, output))
    figure_names.extend(plot_feature_delivery(rows, output))
    atomic_json(
        output / "analysis_summary.json",
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "scientific_status": "OFFLINE_COUNTERFACTUAL_NOT_LIVE_REMEASUREMENT",
            "inventory": {"cells": 288, "actions": 72, "profiles": 4},
            "aggregate_counts": aggregate_counts,
            "complete_feature_delivery_by_profile": {
                profile: {
                    "complete_reassemblies": sum(
                        int(row["measured_complete_reassemblies"])
                        for row in rows
                        if row["network_profile"] == profile
                    ),
                    "frames_sent": sum(
                        int(row["frames_sent"])
                        for row in rows
                        if row["network_profile"] == profile
                    ),
                    "definition": "complete application reassemblies / frames sent",
                }
                for profile in PROFILE_ORDER
            },
            "source_bindings": {
                "builder_sha256": sha256(Path(__file__).resolve()),
                "campaign_cell_table_sha256": sha256(source.CONSOLIDATION / "campaign_288_cell_table.csv"),
                "latest_edge_validation_manifest_sha256": sha256(LATEST_EDGE_VALIDATION_ROOT / "artifact_manifest.json"),
                "sensor_optimization_manifest_sha256": sha256(SENSOR_PRESENTATION_ROOT / "artifact_manifest.json"),
            },
            "component_boundaries": {
                "sensor_compute": "live optimized production sensor computation before P23 seven-channel concatenation; CARLA wait excluded",
                "seven_channel_concat": "live optimized P23 seven-channel concatenation",
                "sensor_compute_including_concat": "live optimized production sensor computation through P23 concatenation",
                "ue_action": "P23 seven-channel concatenation start to send_finished_ns; P23 is distributionally assigned from the live optimized calibration",
                "pure_front": "front_timing_ns.front_backbone only",
                "network": "send_finished_ns to complete edge reassembly, observed receipts only",
                "edge_queue": "complete edge reassembly to compute start plus compute finish to publication start",
                "model_tail": "pure decode_tail CUDA duration from final live family anchor",
                "tail_support": "non-model edge work rescaled to newest repaired-v3 total edge median",
                "map_install": "renderer-off direct publication-to-spatial-map install",
                "edge_map": "complete edge reassembly to direct spatial-map install",
                "total": "P23 seven-channel concatenation start to direct spatial-map install",
                "capture_total": "RGB capture_wall_s to direct spatial-map install, retained for physical freshness but excluded from Figures 01 and 04",
                "physical_map_aoi_excludes_controller_feedback": True,
            },
            "quality": {
                "overlap": "sqrt(vehicle_iou * person_box_mask_iou)",
                "centroid_rms_m": "sqrt((vehicle_xy_mae_m^2 + person_xy_mae_m^2) / 2)",
                "centroid_score": "exp(-centroid_rms_m / 1 metre)",
                "localization_quality": "sqrt(overlap * centroid_score)",
                "combined_quality": "sqrt(segmentation_miou * localization_quality)",
                "distance_reference_m": LOCALIZATION_DISTANCE_REFERENCE_M,
                "role": "provisional presentation coordinate, not PPO reward",
            },
            "denominators": {
                "sensor_compute": "all sent frames, live optimized production sensor compute with P23 concatenation excluded",
                "ue_action": "all sent frames",
                "seven_channel_concat": "all sent frames, distributionally mapped to the live optimized P23 distribution",
                "sensor_compute_including_concat": "all sent frames, distributionally mapped to live optimized production sensor compute",
                "pure_front": "all sent frames",
                "network": "frames with retained observed complete edge receipt",
                "edge_queue": "useful direct-map installations",
                "model_tail": "final live family-anchor samples; repeated as a marginal for each family action/profile",
                "tail_support": "final live family-anchor samples; repeated as a marginal for each family action/profile",
                "map_install": "live renderer-off direct-map samples; repeated as an action-independent marginal",
                "edge_map": "useful direct-map installations",
                "total": "useful direct-map installations",
                "capture_total": "useful direct-map installations",
            },
            "clock_bridge": bridge_audit,
            "observed_network_intervals": observed_network_intervals,
            "network_boundary_inversions_excluded": network_boundary_inversions_excluded,
            "scheduler_arrival_causal_floor_frames": scheduler_arrival_causal_floor_frames,
            "scheduler_arrival_causal_floor_reason": (
                "capture-to-arrival imputations that predated the shifted UE send completion "
                "were moved to send completion before latest-only scheduling"
            ),
            "imputed_arrivals_excluded_from_network_latency": True,
            "measured_radio_outcomes_held_fixed": True,
            "sensor_optimization": sensor_provenance,
            "sensor_delta_ms": {
                "count": len(sensor_delta_samples),
                "p50": percentile(sensor_delta_samples, 0.50),
                "p95": percentile(sensor_delta_samples, 0.95),
                "p99": percentile(sensor_delta_samples, 0.99),
            },
            "edge_rescale_factors": {
                family: {
                    "minimum": min(values),
                    "maximum": max(values),
                    "median": statistics.median(values),
                }
                for family, values in edge_rescale_factors.items()
            },
            "tail_stage_calibration": tail_stage_provenance,
            "direct_map_stage_calibration": map_service_provenance,
            "scheduler": {
                "policy": "non-preemptive compute with one latest pending slot",
                "queue_wait_budget_ms": None,
                "superseded_pending_is_explicit_terminal": True,
            },
            "latency_percentile_warning": (
                "stage P50/P95/P99 values are marginal percentiles with stage-specific "
                "denominators and must not be added to reconstruct a total percentile"
            ),
            "sensor_threading_audit": {
                "carla_callbacks": "camera and radar callbacks are distinct",
                "numerical_preparation": "radar and RGB preparation are sequential in one route-b-split-front worker",
                "evaluation": "separate bounded evaluation worker after immutable snapshot capture",
                "production_stages": "P07-P23 plus P25 measured in the live sensor optimization; P24 evaluation-only and P26 diagnostic synchronization excluded",
            },
        },
    )
    atomic_text(output / "REPORT.md", make_report(rows, latency_rows, sensor_rows, bridge_audit))
    primary = (
        "action_profile_quality_latency.csv",
        "latency_percentiles_by_profile.csv",
        "sensor_optimization_stage_percentiles.csv",
        "analysis_summary.json",
        "REPORT.md",
        *figure_names,
    )
    atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": {name: sha256(output / name) for name in primary}},
    )
    atomic_text(output / TERMINAL, TERMINAL + "\n")
    return {"output": str(output), "figures": figure_names}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv).output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
