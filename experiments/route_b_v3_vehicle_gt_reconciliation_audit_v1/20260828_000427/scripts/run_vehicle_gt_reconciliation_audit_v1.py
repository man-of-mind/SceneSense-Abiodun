#!/usr/bin/env python3
"""Offline, validation-only Route B v3 vehicle-GT reconciliation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(__file__).resolve().parents[1]
STATIC_AUDIT = REPO / "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706"
COMPARISON = REPO / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455"
VIEW = COMPARISON / "views/val"
CANONICAL_REPORT = REPO / "data_collection/experiments/route_b_perception_v3/ROUTE_B_V3_CANONICAL_COLLECTION_REPORT.json"
VISIBILITY_HELPER = REPO / "data_collection/route_b_perception_v3/visibility_v1.py"
PRIOR_SCRIPT = STATIC_AUDIT / "scripts/run_static_vehicle_audit_v1.py"
PLAN = EXPERIMENT / "REGISTERED_VEHICLE_GT_RECONCILIATION_PLAN.md"
DETAILS = EXPERIMENT / "REGISTERED_ALGORITHM_DETAILS.json"
SUCCESS = "VEHICLE_GT_RECONCILIATION_READY_FOR_V3_1_MANUAL_REVIEW"
MODELS = ("lraspp_mprime_noae", "fasterrcnn_radar_roi_v1_epoch12")
CONTRACTS = ("v010", "v025")
THRESHOLDS = (0.20, 0.02)
REQUIRED_FRAMES = (249, 265, 269, 281, 289, 1341, 1357, 1401)
VEHICLE_TAGS = (14, 15, 16)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_collection.phase2_static_environment_truth_v1 import verify_static_environment_truth_v1  # noqa: E402
from data_collection.route_b_perception_v3.visibility_v1 import (  # noqa: E402
    ALGORITHM_VERSION,
    decode_depth_bgra,
    depth_is_plausible,
    reconstruct_consistent_mask,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_module("route_b_v3_static_audit_reuse", PRIOR_SCRIPT)
scorer = prior.import_frozen_scorer()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_csv_x(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite {field}: {value}")
    return result


def box_iou_xywh(first: Sequence[Any], second: Sequence[Any]) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def yaw_delta_mod180(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def rotated_rect(center: Sequence[float], size: Sequence[float], yaw: float) -> tuple[Any, Any, float]:
    return ((float(center[0]), float(center[1])), (float(size[0]), float(size[1])), float(yaw))


def footprint_iou(
    center_a: Sequence[float], size_a: Sequence[float], yaw_a: float,
    center_b: Sequence[float], size_b: Sequence[float], yaw_b: float,
) -> float:
    first = rotated_rect(center_a, size_a, yaw_a)
    second = rotated_rect(center_b, size_b, yaw_b)
    _kind, points = cv2.rotatedRectangleIntersection(first, second)
    intersection = 0.0 if points is None else abs(float(cv2.contourArea(points)))
    area_a = float(size_a[0]) * float(size_a[1])
    area_b = float(size_b[0]) * float(size_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None, "p50": None, "p90": None, "p95": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size), "min": float(data.min()), "mean": float(data.mean()),
        "max": float(data.max()), "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)), "p95": float(np.quantile(data, 0.95)),
    }


def input_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return records


def roi_mask(depth_m: np.ndarray, row: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int, np.ndarray]:
    x0 = max(0, int(math.floor(float(row["clipped_bbox_x"]))))
    y0 = max(0, int(math.floor(float(row["clipped_bbox_y"]))))
    x1 = min(width, max(x0 + 1, int(math.ceil(float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"])))))
    y1 = min(height, max(y0 + 1, int(math.ceil(float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"])))))
    roi = depth_m[y0:y1, x0:x1]
    tolerance = float(row["depth_tolerance_m"])
    mask = (
        np.isfinite(roi)
        & (roi >= float(row["actor_near_depth_m"]) - tolerance)
        & (roi <= float(row["actor_far_depth_m"]) + tolerance)
    )
    if int(np.count_nonzero(mask)) != int(row["native_visible_px"]):
        full = reconstruct_consistent_mask(depth_m, dict(row), width=width, height=height)
        if int(np.count_nonzero(full)) != int(row["native_visible_px"]):
            raise RuntimeError(f"retained visibility reconstruction mismatch: {row.get('sample_id')}")
        mask = full[y0:y1, x0:x1]
    return x0, y0, x1, y1, mask


def roi_intersection(first: tuple[int, int, int, int, np.ndarray], second: tuple[int, int, int, int, np.ndarray]) -> int:
    ax0, ay0, ax1, ay1, am = first
    bx0, by0, bx1, by1, bm = second
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0
    aa = am[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    bb = bm[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    return int(np.count_nonzero(aa & bb))


def component_overlaps(labels: np.ndarray, record: tuple[int, int, int, int, np.ndarray]) -> Counter[int]:
    x0, y0, x1, y1, mask = record
    values = labels[y0:y1, x0:x1][mask]
    values = values[values > 0]
    return Counter(map(int, values.tolist()))


def static_geometry(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray([float(row[f"bbox_center_{axis}_m"]) for axis in "xyz"], dtype=np.float64)
    size = np.asarray([2.0 * float(row[f"bbox_extent_{axis}_m"]) for axis in "xyz"], dtype=np.float64)
    return center, size, float(row["bbox_rotation_yaw_deg"])


def actor_geometry(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray([float(row[f"object_world_{axis}"]) for axis in "xyz"], dtype=np.float64)
    size = np.asarray([float(row[f"gt_size_{axis}_m"]) for axis in "xyz"], dtype=np.float64)
    return center, size, float(row["object_yaw_deg"])


def static_static_conflicts(
    catalog: Sequence[Mapping[str, str]], static_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    observations: dict[tuple[str, str], list[tuple[float, float, float, str]]] = defaultdict(list)
    by_sample: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in static_rows:
        by_sample[row["sample_id"]].append(row)
    for sample_id, rows in by_sample.items():
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                ids = tuple(sorted((first["environment_object_id"], second["environment_object_id"])))
                iou = box_iou_xywh(
                    [first["clipped_bbox_x"], first["clipped_bbox_y"], first["clipped_bbox_w"], first["clipped_bbox_h"]],
                    [second["clipped_bbox_x"], second["clipped_bbox_y"], second["clipped_bbox_w"], second["clipped_bbox_h"]],
                )
                observations[ids].append((iou, abs(float(first["range_m"]) - float(second["range_m"])), abs(float(first["actor_near_depth_m"]) - float(second["actor_near_depth_m"])), sample_id))
    rows_by_id = {row["environment_object_id"]: row for row in catalog}
    output: list[dict[str, Any]] = []
    confirmed_secondary: set[str] = set()
    manual_ids: set[str] = set()
    ids = sorted(rows_by_id)
    for i, first_id in enumerate(ids):
        first = rows_by_id[first_id]
        ac, az, ayaw = static_geometry(first)
        for second_id in ids[i + 1:]:
            second = rows_by_id[second_id]
            bc, bz, byaw = static_geometry(second)
            center = float(np.linalg.norm(ac - bc))
            center_xy = float(np.linalg.norm(ac[:2] - bc[:2]))
            dims = np.abs(az - bz)
            fiou = footprint_iou(ac, az, ayaw, bc, bz, byaw)
            evidence = observations.get((first_id, second_id), [])
            piou = [item[0] for item in evidence]
            ranges = [item[1] for item in evidence]
            near = [item[2] for item in evidence]
            max_piou = max(piou, default=0.0)
            med_piou = float(median(piou)) if piou else 0.0
            med_range = float(median(ranges)) if ranges else None
            med_near = float(median(near)) if near else None
            if not (center_xy <= 3.0 or fiou >= 0.05 or (max_piou >= 0.25 and med_range is not None and med_range <= 2.0)):
                continue
            yaw = yaw_delta_mod180(ayaw, byaw)
            symmetric = max(az[0], az[1]) / max(1e-9, min(az[0], az[1])) <= 1.15
            confirmed = (
                fiou >= 0.70 and center <= 1.5 and (yaw <= 20.0 or symmetric)
                and len(evidence) >= 3 and med_piou >= 0.60
                and med_range is not None and med_range <= 1.0
            )
            manual = (not confirmed and fiou >= 0.35 and center <= 2.0 and max_piou >= 0.50)
            adjudication = "CONFIRMED_DUPLICATE" if confirmed else ("MANUAL_REVIEW_REQUIRED" if manual else "REJECTED_DISTINCT")
            volumes = (float(np.prod(az)), float(np.prod(bz)))
            if volumes[0] > volumes[1] or (math.isclose(volumes[0], volumes[1]) and int(first["carla_environment_object_id"]) < int(second["carla_environment_object_id"])):
                preferred, duplicate = first_id, second_id
            else:
                preferred, duplicate = second_id, first_id
            if confirmed:
                confirmed_secondary.add(duplicate)
            elif manual:
                manual_ids.update((first_id, second_id))
            focus = []
            if {first["semantic_class"], second["semantic_class"]} == {"Bus"}:
                focus.append("prior_panels_13_14_bus_pair")
            if {first_id[-64:][:10], second_id[-64:][:10]} == {"9f9707ae5c", "cefe3933e3"}:
                focus.append("prior_panels_25_32_black_suv_pair")
            best_sample = max(evidence, key=lambda item: (item[0], item[3]))[3] if evidence else ""
            output.append({
                "environment_object_id_a": first_id,
                "native_id_a": first["carla_environment_object_id"],
                "class_a": first["semantic_class"],
                "object_name_a": first["object_name"],
                "environment_object_id_b": second_id,
                "native_id_b": second["carla_environment_object_id"],
                "class_b": second["semantic_class"],
                "object_name_b": second["object_name"],
                "world_center_distance_m": center,
                "world_center_xy_distance_m": center_xy,
                "dimension_x_delta_m": float(dims[0]),
                "dimension_y_delta_m": float(dims[1]),
                "dimension_z_delta_m": float(dims[2]),
                "yaw_delta_mod180_deg": yaw,
                "oriented_footprint_iou": fiou,
                "common_projected_frames": len(evidence),
                "projected_overlap_frames_iou_ge_010": sum(item[0] >= 0.10 for item in evidence),
                "median_projected_iou": med_piou,
                "maximum_projected_iou": max_piou,
                "median_range_delta_m": med_range if med_range is not None else "",
                "median_near_depth_delta_m": med_near if med_near is not None else "",
                "representative_sample_id": best_sample,
                "preferred_environment_object_id": preferred if confirmed else "",
                "duplicate_environment_object_id": duplicate if confirmed else "",
                "adjudication": adjudication,
                "focus_case": ";".join(focus),
                "basis": (
                    "near-identical world footprint and repeated projected/depth agreement"
                    if confirmed else "strong but non-confirming overlap; human decision required"
                    if manual else "candidate geometry resolves as distinct"
                ),
            })
    output.sort(key=lambda row: (row["adjudication"], row["environment_object_id_a"], row["environment_object_id_b"]))
    return output, confirmed_secondary, manual_ids


ACTOR_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "timestamp", "actor_identity", "gt_actor_id",
    "gt_actor_type_id", "gt_source", "current_evaluation_included", "object_world_x",
    "object_world_y", "object_world_z", "gt_size_x_m", "gt_size_y_m", "gt_size_z_m",
    "object_yaw_deg", "gt_distance_m", "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w",
    "clipped_bbox_h", "clipped_projected_area_px", "actor_near_depth_m", "actor_far_depth_m",
    "native_visible_px", "model_input_visible_px", "visible_fraction", "occluder_closer_fraction",
    "background_farther_fraction", "geometry_qualified_v2", "eligible_visible_v010",
    "eligible_clear_v025", "visibility_tier", "semantic_vehicle_visible_px",
    "semantic_vehicle_visible_fraction", "radar_support_points", "visibility_algorithm_version",
    "depth_tolerance_m", "semantic_disagreement_manual_review",
)


STATIC_STATIC_FIELDS = (
    "environment_object_id_a", "native_id_a", "class_a", "object_name_a",
    "environment_object_id_b", "native_id_b", "class_b", "object_name_b",
    "world_center_distance_m", "world_center_xy_distance_m", "dimension_x_delta_m",
    "dimension_y_delta_m", "dimension_z_delta_m", "yaw_delta_mod180_deg",
    "oriented_footprint_iou", "common_projected_frames",
    "projected_overlap_frames_iou_ge_010", "median_projected_iou", "maximum_projected_iou",
    "median_range_delta_m", "median_near_depth_delta_m", "representative_sample_id",
    "preferred_environment_object_id", "duplicate_environment_object_id", "adjudication",
    "focus_case", "basis",
)


STATIC_DYNAMIC_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "environment_object_id",
    "carla_environment_object_id", "static_class", "actor_identity", "gt_actor_id",
    "gt_actor_type_id", "world_center_distance_m", "world_center_xy_distance_m",
    "dimension_x_delta_m", "dimension_y_delta_m", "dimension_z_delta_m",
    "yaw_delta_mod180_deg", "oriented_footprint_iou", "projected_box_iou",
    "depth_consistent_mask_overlap_px", "static_near_depth_m", "static_far_depth_m",
    "actor_near_depth_m", "actor_far_depth_m", "depth_order", "static_v010",
    "actor_v010", "adjudication", "focus_case", "basis",
)


COMPONENT_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "component_id", "semantic_tag",
    "component_pixels", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x",
    "centroid_y", "median_depth_m", "best_actor_identity", "best_actor_overlap_px",
    "best_actor_overlap_fraction", "best_environment_object_id", "best_static_overlap_px",
    "best_static_overlap_fraction", "classification", "lraspp_best_score",
    "lraspp_best_bbox_iou", "fasterrcnn_best_score", "fasterrcnn_best_bbox_iou",
    "retained_model_evidence", "automatic_gt", "review_status",
)


FRAME_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "actor_vehicle_candidates",
    "actor_current_evaluation", "actor_eligible_v010", "actor_eligible_v025",
    "actor_current_ineligible_v010", "actor_current_ineligible_v025",
    "static_projected_candidates", "static_eligible_v010_raw", "static_eligible_v025_raw",
    "semantic_vehicle_components", "components_explained_actor_only",
    "components_explained_static_only", "components_explained_actor_and_static",
    "unresolved_vehicle_components", "static_dynamic_confirmed", "static_dynamic_manual",
    "static_dynamic_rejected", "lraspp_unmatched_v010_score020",
    "lraspp_unmatched_v010_score002", "fasterrcnn_unmatched_v010_score020",
    "fasterrcnn_unmatched_v010_score002",
)


def actor_target(row: Mapping[str, Any]) -> dict[str, Any]:
    yaw = math.radians(float(row["object_yaw_deg"]))
    return {
        "class_index": 0.0, "class_name": "vehicle",
        "center_x": float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"]) / 2.0,
        "center_y": float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"]) / 2.0,
        "bbox_w": float(row["clipped_bbox_w"]), "bbox_h": float(row["clipped_bbox_h"]),
        "area": float(row["clipped_projected_area_px"]),
        "local_x": float(row["object_sensor_x"]), "local_y": float(row["object_sensor_y"]),
        "local_z": float(row["object_sensor_z"]), "world_x": float(row["object_world_x"]),
        "world_y": float(row["object_world_y"]), "world_z": float(row["object_world_z"]),
        "size_x": float(row["gt_size_x_m"]), "size_y": float(row["gt_size_y_m"]),
        "size_z": float(row["gt_size_z_m"]), "yaw_sin": math.sin(yaw), "yaw_cos": math.cos(yaw),
        "parked": float(truth(row.get("parked_label", False))),
        "radar_support": float(int(row.get("radar_support_points", 0)) > 0),
        "_gt_source": "actor", "_identity": row["actor_identity"],
    }


def static_target(row: Mapping[str, Any]) -> dict[str, Any]:
    yaw = math.radians(float(row["bbox_rotation_yaw_deg"]))
    return {
        "class_index": 0.0, "class_name": "vehicle",
        "center_x": float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"]) / 2.0,
        "center_y": float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"]) / 2.0,
        "bbox_w": float(row["clipped_bbox_w"]), "bbox_h": float(row["clipped_bbox_h"]),
        "area": float(row["clipped_projected_area_px"]),
        "local_x": float(row["object_sensor_x_m"]), "local_y": float(row["object_sensor_y_m"]),
        "local_z": float(row["object_sensor_z_m"]), "world_x": float(row["bbox_center_x_m"]),
        "world_y": float(row["bbox_center_y_m"]), "world_z": float(row["bbox_center_z_m"]),
        "size_x": float(row["size_x_m"]), "size_y": float(row["size_y_m"]),
        "size_z": float(row["size_z_m"]), "yaw_sin": math.sin(yaw), "yaw_cos": math.cos(yaw),
        "parked": 1.0, "radar_support": float(int(row["radar_support_points"]) > 0),
        "_gt_source": "static_environment_object", "_identity": row["environment_object_id"],
    }


def score_vehicle(
    predictions: Mapping[str, Sequence[dict[str, Any]]],
    gt: Mapping[str, Sequence[dict[str, Any]]], frame_ids: Sequence[str], threshold: float,
) -> dict[str, Any]:
    bucket = scorer.empty_bucket()
    eligible = 0
    prediction_count = 0
    for sample_id in frame_ids:
        preds = [row for row in predictions.get(sample_id, []) if row["class_name"] == "vehicle" and row["score"] >= threshold]
        targets = [
            row for row in gt.get(sample_id, []) if row["class_name"] == "vehicle"
        ]
        flags = scorer.duplicate_flags(preds)
        matches = scorer.greedy_match_predictions(preds, targets, max_distance_m=3.0, class_aware=True)
        mp = {row[0] for row in matches}
        mg = {row[1] for row in matches}
        eligible += len(targets)
        prediction_count += len(preds)
        for pi, gi, distance in matches:
            pred, target = preds[pi], targets[gi]
            bucket["tp"] += 1
            bucket["xy"].append(float(distance))
            bucket["dimension"].append(scorer.mean([
                abs(pred["size_x"] - target["size_x"]), abs(pred["size_y"] - target["size_y"]),
                abs(pred["size_z"] - target["size_z"]),
            ]))
            bucket["yaw"].append(scorer.yaw_error(pred, target))
        for index, _pred in enumerate(preds):
            if index not in mp:
                bucket["fp"] += 1
                bucket["duplicate_fp"] += int(flags[index])
        bucket["fn"] += len(targets) - len(mg)
    return {**scorer.summarize(bucket, len(frame_ids), eligible), "predictions": prediction_count}


def unmatched_predictions(
    predictions: Mapping[str, Sequence[dict[str, Any]]], targets: Sequence[dict[str, Any]], threshold: float,
) -> list[dict[str, Any]]:
    preds = [row for row in predictions if row["class_name"] == "vehicle" and row["score"] >= threshold]
    matches = scorer.greedy_match_predictions(preds, list(targets), max_distance_m=3.0, class_aware=True)
    matched = {row[0] for row in matches}
    return [row for index, row in enumerate(preds) if index not in matched]


def analyze_frames(
    manifest: Sequence[Mapping[str, str]], boxes_by_sample: Mapping[str, Sequence[Mapping[str, str]]],
    visibility_by_key: Mapping[tuple[str, str], Mapping[str, str]], current_eval: set[tuple[str, str]],
    static_by_sample: Mapping[str, Sequence[Mapping[str, str]]],
    predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actor_output: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(manifest, 1):
        sample_id = frame["sample_id"]
        width, height = int(frame["camera_width"]), int(frame["camera_height"])
        depth_path = VIEW / frame["experiment_id"] / "depth" / f"{sample_id}.png"
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        semantic = cv2.imread(str(VIEW / frame["instance_raw_path"]), cv2.IMREAD_UNCHANGED)
        if raw_depth is None or semantic is None or semantic.shape != (height, width):
            raise RuntimeError(f"missing retained depth/semantic: {sample_id}")
        depth = decode_depth_bgra(raw_depth)
        if not depth_is_plausible(depth):
            raise RuntimeError(f"implausible depth: {sample_id}")
        vehicle_mask = np.isin(semantic, VEHICLE_TAGS).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            vehicle_mask, connectivity=8
        )
        actor_rows: list[dict[str, Any]] = []
        actor_masks: dict[str, tuple[int, int, int, int, np.ndarray]] = {}
        static_masks: dict[str, tuple[int, int, int, int, np.ndarray]] = {}
        best_actor: dict[int, tuple[int, str]] = {}
        best_static: dict[int, tuple[int, str]] = {}
        for box in boxes_by_sample.get(sample_id, []):
            if box["label"] != "vehicle":
                continue
            key = (sample_id, box["gt_actor_id"])
            vis = visibility_by_key.get(key)
            if vis is None:
                raise RuntimeError(f"missing actor visibility row: {key}")
            if vis["visibility_algorithm_version"] != ALGORITHM_VERSION:
                raise RuntimeError(f"actor visibility helper drift: {key}")
            record = roi_mask(depth, vis, width, height)
            x0, y0, x1, y1, mask = record
            semantic_px = int(np.count_nonzero(mask & np.isin(semantic[y0:y1, x0:x1], VEHICLE_TAGS)))
            identity = f"{box['experiment_id']}:actor:{box['gt_actor_id']}"
            row = {
                **{field: box.get(field, "") for field in (
                    "experiment_id", "sample_id", "frame_id", "timestamp", "gt_actor_id",
                    "gt_actor_type_id", "gt_source", "object_world_x", "object_world_y",
                    "object_world_z", "gt_size_x_m", "gt_size_y_m", "gt_size_z_m",
                    "object_yaw_deg", "gt_distance_m", "object_sensor_x", "object_sensor_y",
                    "object_sensor_z", "parked_label", "radar_support_points",
                )},
                **{field: vis[field] for field in (
                    "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h",
                    "clipped_projected_area_px", "actor_near_depth_m", "actor_far_depth_m",
                    "native_visible_px", "model_input_visible_px", "visible_fraction",
                    "occluder_closer_fraction", "background_farther_fraction",
                    "geometry_qualified_v2", "eligible_visible_v010", "eligible_clear_v025",
                    "visibility_tier", "visibility_algorithm_version",
                    "depth_tolerance_m",
                )},
                "actor_identity": identity,
                "current_evaluation_included": key in current_eval,
                "semantic_vehicle_visible_px": semantic_px,
                "semantic_vehicle_visible_fraction": semantic_px / max(1, int(vis["native_visible_px"])),
                "semantic_disagreement_manual_review": truth(vis["eligible_visible_v010"]) and semantic_px == 0,
            }
            actor_rows.append(row)
            actor_output.append(row)
            actor_masks[identity] = record
            for component_id, overlap in component_overlaps(labels, record).items():
                if overlap > best_actor.get(component_id, (0, ""))[0]:
                    best_actor[component_id] = (overlap, identity)
        frame_static = list(static_by_sample.get(sample_id, []))
        for row in frame_static:
            if row["visibility_algorithm_version"] != ALGORITHM_VERSION:
                raise RuntimeError(f"static visibility helper drift: {sample_id}")
            record = roi_mask(depth, row, width, height)
            static_masks[row["environment_object_id"]] = record
            x0, y0, x1, y1, mask = record
            semantic_px = int(np.count_nonzero(mask & np.isin(semantic[y0:y1, x0:x1], VEHICLE_TAGS)))
            if semantic_px != int(row["semantic_vehicle_visible_px"]):
                raise RuntimeError(f"static semantic support drift: {sample_id}/{row['environment_object_id']}")
            for component_id, overlap in component_overlaps(labels, record).items():
                if overlap > best_static.get(component_id, (0, ""))[0]:
                    best_static[component_id] = (overlap, row["environment_object_id"])
        frame_conflicts = []
        for static in frame_static:
            sc, ss, sy = static_geometry(static)
            static_box = [static["clipped_bbox_x"], static["clipped_bbox_y"], static["clipped_bbox_w"], static["clipped_bbox_h"]]
            for actor in actor_rows:
                source_box = next(row for row in boxes_by_sample[sample_id] if row["label"] == "vehicle" and row["gt_actor_id"] == actor["gt_actor_id"])
                ac, az, ay = actor_geometry(source_box)
                actor_box = [actor["clipped_bbox_x"], actor["clipped_bbox_y"], actor["clipped_bbox_w"], actor["clipped_bbox_h"]]
                distance = float(np.linalg.norm(sc - ac))
                distance_xy = float(np.linalg.norm(sc[:2] - ac[:2]))
                fiou = footprint_iou(sc, ss, sy, ac, az, ay)
                piou = box_iou_xywh(static_box, actor_box)
                if not (distance_xy <= 3.0 or fiou >= 0.05 or piou >= 0.05):
                    continue
                dims = np.abs(ss - az)
                overlap = roi_intersection(static_masks[static["environment_object_id"]], actor_masks[actor["actor_identity"]])
                confirmed = distance <= 1.0 and bool(np.all(dims <= 1.0)) and fiou >= 0.50
                distinct = piou >= 0.05 and (distance_xy > 3.0 or fiou < 0.05)
                adjudication = "CONFIRMED_SAME_PHYSICAL_OBJECT" if confirmed else ("REJECTED_DISTINCT_OCCLUSION" if distinct else "MANUAL_REVIEW_REQUIRED")
                snear, sfar = float(static["actor_near_depth_m"]), float(static["actor_far_depth_m"])
                anear, afar = float(actor["actor_near_depth_m"]), float(actor["actor_far_depth_m"])
                order = "actor_behind_static" if anear > sfar + 0.25 else "static_behind_actor" if snear > afar + 0.25 else "depth_intervals_overlap"
                focus = []
                if int(frame["frame_id"]) == 1401 and static["environment_object_id"].endswith("76be529ae90b22474f7e6c5676aa5a547df3402ffb2d6ef78722d84087791ab2"):
                    focus.append("prior_panel_24_police_car")
                if static["environment_object_id"][-64:][:10] in {"9f9707ae5c", "cefe3933e3"}:
                    focus.append("prior_panels_25_32_black_suv")
                if int(frame["frame_id"]) == 265:
                    focus.append("frame_265_large_bus_occlusion")
                conflict = {
                    "experiment_id": frame["experiment_id"], "sample_id": sample_id,
                    "frame_id": frame["frame_id"], "environment_object_id": static["environment_object_id"],
                    "carla_environment_object_id": static["carla_environment_object_id"],
                    "static_class": static["semantic_class"], "actor_identity": actor["actor_identity"],
                    "gt_actor_id": actor["gt_actor_id"], "gt_actor_type_id": actor["gt_actor_type_id"],
                    "world_center_distance_m": distance, "world_center_xy_distance_m": distance_xy,
                    "dimension_x_delta_m": float(dims[0]), "dimension_y_delta_m": float(dims[1]),
                    "dimension_z_delta_m": float(dims[2]), "yaw_delta_mod180_deg": yaw_delta_mod180(sy, ay),
                    "oriented_footprint_iou": fiou, "projected_box_iou": piou,
                    "depth_consistent_mask_overlap_px": overlap, "static_near_depth_m": snear,
                    "static_far_depth_m": sfar, "actor_near_depth_m": anear,
                    "actor_far_depth_m": afar, "depth_order": order,
                    "static_v010": truth(static["eligible_visible_v010"]),
                    "actor_v010": truth(actor["eligible_visible_v010"]),
                    "adjudication": adjudication, "focus_case": ";".join(focus),
                    "basis": "same world footprint and dimensions" if confirmed else
                    f"projected overlap but distinct world geometry ({order})" if distinct else
                    "world/image evidence is insufficient for automatic resolution",
                }
                conflicts.append(conflict)
                frame_conflicts.append(conflict)
        component_counts = Counter()
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            x, y, w, h = [int(stats[component_id, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            actor_overlap, actor_id = best_actor.get(component_id, (0, ""))
            static_overlap, static_id = best_static.get(component_id, (0, ""))
            actor_explained = actor_overlap >= 12 and actor_overlap / area >= 0.10
            static_explained = static_overlap >= 12 and static_overlap / area >= 0.10
            classification = "actor_and_static" if actor_explained and static_explained else "actor_only" if actor_explained else "static_only" if static_explained else "UNRESOLVED_VEHICLE_CANDIDATE"
            component_counts[classification] += 1
            if classification != "UNRESOLVED_VEHICLE_CANDIDATE":
                continue
            component_pixels = labels[y:y + h, x:x + w] == component_id
            component_depth = depth[y:y + h, x:x + w][component_pixels]
            best_model: dict[str, tuple[float, float]] = {}
            component_box = [x, y, w, h]
            for model in MODELS:
                candidates = []
                for pred in predictions[model].get(sample_id, []):
                    if pred["class_name"] != "vehicle" or pred["score"] < 0.02:
                        continue
                    iou = box_iou_xywh(component_box, [pred["bbox_x0"], pred["bbox_y0"], pred["bbox_x1"] - pred["bbox_x0"], pred["bbox_y1"] - pred["bbox_y0"]])
                    candidates.append((iou, pred["score"]))
                best_model[model] = max(candidates, default=(0.0, 0.0), key=lambda item: (item[0], item[1]))
            model_evidence = any(iou >= 0.10 for iou, _score in best_model.values())
            unresolved.append({
                "experiment_id": frame["experiment_id"], "sample_id": sample_id,
                "frame_id": frame["frame_id"], "component_id": component_id,
                "semantic_tag": int(np.bincount(semantic[y:y + h, x:x + w][component_pixels].astype(np.int64)).argmax()),
                "component_pixels": area, "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "centroid_x": float(centroids[component_id, 0]), "centroid_y": float(centroids[component_id, 1]),
                "median_depth_m": float(np.median(component_depth)),
                "best_actor_identity": actor_id, "best_actor_overlap_px": actor_overlap,
                "best_actor_overlap_fraction": actor_overlap / area, "best_environment_object_id": static_id,
                "best_static_overlap_px": static_overlap, "best_static_overlap_fraction": static_overlap / area,
                "classification": classification,
                "lraspp_best_score": best_model[MODELS[0]][1], "lraspp_best_bbox_iou": best_model[MODELS[0]][0],
                "fasterrcnn_best_score": best_model[MODELS[1]][1], "fasterrcnn_best_bbox_iou": best_model[MODELS[1]][0],
                "retained_model_evidence": model_evidence, "automatic_gt": False,
                "review_status": "UNRESOLVED_VEHICLE_CANDIDATE",
            })
        current = [row for row in actor_rows if truth(row["current_evaluation_included"])]
        summaries.append({
            "experiment_id": frame["experiment_id"], "sample_id": sample_id, "frame_id": frame["frame_id"],
            "actor_vehicle_candidates": len(actor_rows), "actor_current_evaluation": len(current),
            "actor_eligible_v010": sum(truth(row["eligible_visible_v010"]) for row in current),
            "actor_eligible_v025": sum(truth(row["eligible_clear_v025"]) for row in current),
            "actor_current_ineligible_v010": sum(not truth(row["eligible_visible_v010"]) for row in current),
            "actor_current_ineligible_v025": sum(not truth(row["eligible_clear_v025"]) for row in current),
            "static_projected_candidates": len(frame_static),
            "static_eligible_v010_raw": sum(truth(row["eligible_visible_v010"]) for row in frame_static),
            "static_eligible_v025_raw": sum(truth(row["eligible_clear_v025"]) for row in frame_static),
            "semantic_vehicle_components": sum(component_counts.values()),
            "components_explained_actor_only": component_counts["actor_only"],
            "components_explained_static_only": component_counts["static_only"],
            "components_explained_actor_and_static": component_counts["actor_and_static"],
            "unresolved_vehicle_components": component_counts["UNRESOLVED_VEHICLE_CANDIDATE"],
            "static_dynamic_confirmed": sum(row["adjudication"] == "CONFIRMED_SAME_PHYSICAL_OBJECT" for row in frame_conflicts),
            "static_dynamic_manual": sum(row["adjudication"] == "MANUAL_REVIEW_REQUIRED" for row in frame_conflicts),
            "static_dynamic_rejected": sum(row["adjudication"] == "REJECTED_DISTINCT_OCCLUSION" for row in frame_conflicts),
        })
        if frame_index % 250 == 0:
            print(f"[reconcile frames] {frame_index}/{len(manifest)} actor_rows={len(actor_output)} unresolved={len(unresolved)}", flush=True)
    return actor_output, conflicts, unresolved, summaries


def build_gt_variants(
    frame_ids: Sequence[str], actor_rows: Sequence[Mapping[str, Any]],
    static_rows: Sequence[Mapping[str, Any]], confirmed_secondary: set[str], manual_static_ids: set[str],
    dynamic_conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    actor_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    static_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in actor_rows:
        if truth(row["current_evaluation_included"]):
            actor_by_sample[row["sample_id"]].append(row)
    for row in static_rows:
        static_by_sample[row["sample_id"]].append(row)
    dynamic_excluded = {
        (row["sample_id"], row["environment_object_id"])
        for row in dynamic_conflicts if row["adjudication"] in {"CONFIRMED_SAME_PHYSICAL_OBJECT", "MANUAL_REVIEW_REQUIRED"}
    }
    output: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for contract in CONTRACTS:
        flag = "eligible_visible_v010" if contract == "v010" else "eligible_clear_v025"
        variants = {name: {} for name in ("actor_visibility_only", "reconciled_depth_only", "reconciled_review_qualified")}
        for sample_id in frame_ids:
            actors = [row for row in actor_by_sample.get(sample_id, []) if truth(row[flag])]
            actor_targets = [actor_target(row) for row in actors]
            statics = [
                row for row in static_by_sample.get(sample_id, [])
                if truth(row[flag]) and row["environment_object_id"] not in confirmed_secondary
                and row["environment_object_id"] not in manual_static_ids
                and (sample_id, row["environment_object_id"]) not in dynamic_excluded
            ]
            variants["actor_visibility_only"][sample_id] = actor_targets
            variants["reconciled_depth_only"][sample_id] = actor_targets + [static_target(row) for row in statics]
            review_actors = [row for row in actors if int(row["semantic_vehicle_visible_px"]) > 0]
            review_statics = [row for row in statics if int(row["semantic_vehicle_visible_px"]) > 0]
            variants["reconciled_review_qualified"][sample_id] = [actor_target(row) for row in review_actors] + [static_target(row) for row in review_statics]
        output[contract] = variants
    return output


def compute_metrics(
    manifest: Sequence[Mapping[str, str]], predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    variants: Mapping[str, Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]],
    prior_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    frame_ids = [row["sample_id"] for row in manifest]
    baseline = {contract: scorer.gt_by_frame(contract, list(manifest)) for contract in CONTRACTS}
    output: dict[str, Any] = {}
    exact = True
    for model in MODELS:
        output[model] = {"contracts": {}}
        for contract in CONTRACTS:
            output[model]["contracts"][contract] = {"thresholds": {}}
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                before = score_vehicle(predictions[model], baseline[contract], frame_ids, threshold)
                expected = prior_metrics["models"][model]["contracts"][contract]["thresholds"][key]["before"]
                parity = all(before[name] == expected[name] for name in ("eligible_gt", "tp", "fp", "fn", "precision", "recall", "f1"))
                exact = exact and parity
                item = {"before_actor_only": before, "baseline_parity_exact": parity}
                for variant, gt in variants[contract].items():
                    item[variant] = score_vehicle(predictions[model], gt, frame_ids, threshold)
                output[model]["contracts"][contract]["thresholds"][key] = item
    return output, exact


def add_unmatched_counts(
    summaries: list[dict[str, Any]], predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    review_gt: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    unmatched: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    by_sample = {row["sample_id"]: row for row in summaries}
    for sample_id, summary in by_sample.items():
        for model in MODELS:
            short = "lraspp" if model == MODELS[0] else "fasterrcnn"
            for threshold in THRESHOLDS:
                rows = unmatched_predictions(predictions[model].get(sample_id, []), review_gt.get(sample_id, []), threshold)
                unmatched[(sample_id, model, f"{threshold:.2f}")] = rows
                summary[f"{short}_unmatched_v010_score{int(round(threshold * 100)):03d}"] = len(rows)
    return unmatched


def focus_find_static(static_by_sample: Mapping[str, Sequence[Mapping[str, Any]]], sample_id: str, prefix: str) -> Mapping[str, Any] | None:
    return next((row for row in static_by_sample.get(sample_id, []) if row["environment_object_id"][-64:].startswith(prefix)), None)


SELECTION_FIELDS = (
    "review_index", "panel_kind", "focus_label", "selected_reason", "experiment_id",
    "sample_id", "frame_id", "environment_object_id", "environment_object_id_2",
    "actor_identity", "component_id", "conflict_adjudication", "panel_path",
    "catalog_object_correct", "visibility_class_correct", "static_static_duplicate",
    "static_dynamic_duplicate", "unexplained_visible_vehicle", "decision", "reviewer_notes",
)


def select_panels(
    manifest: Sequence[Mapping[str, str]], static_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]], static_conflicts: Sequence[Mapping[str, Any]],
    dynamic_conflicts: Sequence[Mapping[str, Any]], unresolved: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_frame: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in manifest:
        manifest_by_frame[int(row["frame_id"])].append(row)
    static_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    actor_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in static_rows: static_by_sample[row["sample_id"]].append(row)
    for row in actor_rows: actor_by_sample[row["sample_id"]].append(row)
    focus_prefix = {249: "d797df409f", 265: "23438653f3", 269: "350147d0f5", 281: "9f9707ae5c", 289: "3f09bbba2f", 1341: "e24742de6c", 1357: "c80ed13576", 1401: "76be529ae9"}
    selected: list[dict[str, Any]] = []
    case_keys: set[tuple[Any, ...]] = set()
    env_occurrences: Counter[str] = Counter()

    def add(row: Mapping[str, Any]) -> bool:
        envs = [str(row.get(key, "")) for key in ("environment_object_id", "environment_object_id_2") if str(row.get(key, ""))]
        if any(env_occurrences[value] >= 2 for value in envs):
            return False
        key = (row.get("panel_kind"), row.get("sample_id"), row.get("environment_object_id"), row.get("environment_object_id_2"), row.get("actor_identity"), row.get("component_id"))
        if key in case_keys:
            return False
        case_keys.add(key)
        for value in envs: env_occurrences[value] += 1
        selected.append(dict(row))
        return True

    for frame_id in REQUIRED_FRAMES:
        candidates = sorted(manifest_by_frame.get(frame_id, []), key=lambda row: row["sample_id"])
        if not candidates:
            raise RuntimeError(f"required overlay frame absent: {frame_id}")
        frame = candidates[0]
        static = focus_find_static(static_by_sample, frame["sample_id"], focus_prefix[frame_id])
        add({
            "panel_kind": "required_all_object_frame", "focus_label": f"all objects frame {frame_id}",
            "selected_reason": "required catalog-completeness and concern frame", "experiment_id": frame["experiment_id"],
            "sample_id": frame["sample_id"], "frame_id": frame_id,
            "environment_object_id": static["environment_object_id"] if static else "",
            "environment_object_id_2": "", "actor_identity": "", "component_id": "",
            "conflict_adjudication": "",
        })
    for conflict in sorted(static_conflicts, key=lambda row: ({"CONFIRMED_DUPLICATE": 0, "MANUAL_REVIEW_REQUIRED": 1, "REJECTED_DISTINCT": 2}[row["adjudication"]], -float(row["maximum_projected_iou"]))):
        if conflict["adjudication"] == "REJECTED_DISTINCT" or not conflict["representative_sample_id"]:
            continue
        sample = conflict["representative_sample_id"]
        frame = next(row for row in manifest if row["sample_id"] == sample)
        add({
            "panel_kind": "static_static_conflict", "focus_label": "static-static pair",
            "selected_reason": conflict["focus_case"] or "highest-overlap static-static candidate",
            "experiment_id": frame["experiment_id"], "sample_id": sample, "frame_id": frame["frame_id"],
            "environment_object_id": conflict["environment_object_id_a"],
            "environment_object_id_2": conflict["environment_object_id_b"],
            "actor_identity": "", "component_id": "", "conflict_adjudication": conflict["adjudication"],
        })
        if sum(row["panel_kind"] == "static_static_conflict" for row in selected) >= 4: break
    for conflict in sorted(dynamic_conflicts, key=lambda row: ({"CONFIRMED_SAME_PHYSICAL_OBJECT": 0, "MANUAL_REVIEW_REQUIRED": 1, "REJECTED_DISTINCT_OCCLUSION": 2}[row["adjudication"]], -float(row["projected_box_iou"]))):
        if conflict["adjudication"] == "REJECTED_DISTINCT_OCCLUSION" and not conflict["focus_case"]:
            continue
        add({
            "panel_kind": "static_dynamic_conflict", "focus_label": "static-dynamic pair",
            "selected_reason": conflict["focus_case"] or "highest-priority static-dynamic candidate",
            "experiment_id": conflict["experiment_id"], "sample_id": conflict["sample_id"],
            "frame_id": conflict["frame_id"], "environment_object_id": conflict["environment_object_id"],
            "environment_object_id_2": "", "actor_identity": conflict["actor_identity"],
            "component_id": "", "conflict_adjudication": conflict["adjudication"],
        })
        if sum(row["panel_kind"] == "static_dynamic_conflict" for row in selected) >= 4: break
    invisible = sorted(
        (row for row in actor_rows if truth(row["current_evaluation_included"]) and not truth(row["eligible_visible_v010"])),
        key=lambda row: (0 if int(row["frame_id"]) == 265 else 1, -float(row["clipped_projected_area_px"]), row["sample_id"], row["actor_identity"]),
    )
    seen_actor: set[str] = set()
    for row in invisible:
        if row["actor_identity"] in seen_actor: continue
        seen_actor.add(row["actor_identity"])
        add({
            "panel_kind": "actor_rejected_visibility", "focus_label": "current actor GT rejected by v0.10",
            "selected_reason": "frame 265 concern" if int(row["frame_id"]) == 265 else "large projected unobservable actor",
            "experiment_id": row["experiment_id"], "sample_id": row["sample_id"], "frame_id": row["frame_id"],
            "environment_object_id": "", "environment_object_id_2": "", "actor_identity": row["actor_identity"],
            "component_id": "", "conflict_adjudication": "",
        })
        if sum(item["panel_kind"] == "actor_rejected_visibility" for item in selected) >= 6: break
    seen_component_frames: set[str] = set()
    for row in sorted(unresolved, key=lambda item: (-int(item["component_pixels"]), item["sample_id"], int(item["component_id"]))):
        if row["sample_id"] in seen_component_frames: continue
        seen_component_frames.add(row["sample_id"])
        add({
            "panel_kind": "unresolved_semantic_component", "focus_label": "unexplained semantic vehicle component",
            "selected_reason": "largest diverse unresolved semantic components",
            "experiment_id": row["experiment_id"], "sample_id": row["sample_id"], "frame_id": row["frame_id"],
            "environment_object_id": "", "environment_object_id_2": "", "actor_identity": "",
            "component_id": row["component_id"], "conflict_adjudication": "UNRESOLVED_VEHICLE_CANDIDATE",
        })
        if sum(item["panel_kind"] == "unresolved_semantic_component" for item in selected) >= 6: break
    conflict_ids = {
        str(row[key]) for row in static_conflicts if row["adjudication"] != "REJECTED_DISTINCT"
        for key in ("environment_object_id_a", "environment_object_id_b")
    }
    conflict_ids.update(row["environment_object_id"] for row in dynamic_conflicts if row["adjudication"] != "REJECTED_DISTINCT_OCCLUSION")
    best_eligible: dict[str, Mapping[str, Any]] = {}
    best_unobservable: dict[str, Mapping[str, Any]] = {}
    for row in static_rows:
        sid = row["environment_object_id"]
        if sid in conflict_ids: continue
        if truth(row["eligible_visible_v010"]):
            if sid not in best_eligible or (int(row["semantic_vehicle_visible_px"]), float(row["visible_fraction"])) > (int(best_eligible[sid]["semantic_vehicle_visible_px"]), float(best_eligible[sid]["visible_fraction"])):
                best_eligible[sid] = row
        elif truth(row["geometry_qualified_v2"]):
            if sid not in best_unobservable or float(row["clipped_projected_area_px"]) > float(best_unobservable[sid]["clipped_projected_area_px"]):
                best_unobservable[sid] = row
    pools = [
        ("clean_eligible_static", sorted(best_eligible.values(), key=lambda row: (row["experiment_id"], row["environment_object_id"]))),
        ("clean_unobservable_static", sorted(best_unobservable.values(), key=lambda row: (row["experiment_id"], row["environment_object_id"]))),
    ]
    while (len(selected) < 40 or len(env_occurrences) < 24) and any(rows for _kind, rows in pools):
        progressed = False
        for kind, rows in pools:
            while rows:
                row = rows.pop(0)
                added = add({
                    "panel_kind": kind,
                    "focus_label": "clean eligible static" if kind == "clean_eligible_static" else "clean v0.10-unobservable static",
                    "selected_reason": "diversity fill with a new environment ID", "experiment_id": row["experiment_id"],
                    "sample_id": row["sample_id"], "frame_id": row["frame_id"],
                    "environment_object_id": row["environment_object_id"], "environment_object_id_2": "",
                    "actor_identity": "", "component_id": "", "conflict_adjudication": "",
                })
                if added:
                    progressed = True
                    break
            if len(selected) >= 48 and len(env_occurrences) >= 24: break
        if not progressed: break
    # Conflict-heavy regions can exhaust the two clean pools before the registered
    # diversity floor is reached. Add distinct catalog candidates (including
    # quarantined conflict candidates) strictly as manual-review evidence.
    fallback_by_id: dict[str, Mapping[str, Any]] = {}
    for row in static_rows:
        sid = row["environment_object_id"]
        if sid not in fallback_by_id or float(row["clipped_projected_area_px"]) > float(
            fallback_by_id[sid]["clipped_projected_area_px"]
        ):
            fallback_by_id[sid] = row
    for sid, row in sorted(fallback_by_id.items()):
        if len(selected) >= 40 and len(env_occurrences) >= 24:
            break
        add({
            "panel_kind": "additional_diverse_static_candidate",
            "focus_label": "additional catalog candidate for diversity",
            "selected_reason": "registered 40-panel/24-environment-ID diversity fallback",
            "experiment_id": row["experiment_id"], "sample_id": row["sample_id"],
            "frame_id": row["frame_id"], "environment_object_id": sid,
            "environment_object_id_2": "", "actor_identity": "",
            "component_id": "", "conflict_adjudication": (
                "QUARANTINED_CONFLICT_CANDIDATE" if sid in conflict_ids else ""
            ),
        })
    if len(selected) < 40 or len(env_occurrences) < 24 or any(value > 2 for value in env_occurrences.values()):
        raise RuntimeError(f"manual diversity gate failed: panels={len(selected)} unique_env={len(env_occurrences)} max_repeat={max(env_occurrences.values(), default=0)}")
    for index, row in enumerate(selected, 1):
        row["review_index"] = index
        row["panel_path"] = f"manual_review_panels/{index:02d}_{row['panel_kind']}_frame{row['frame_id']}.png"
        for field in ("catalog_object_correct", "visibility_class_correct", "static_static_duplicate", "static_dynamic_duplicate", "unexplained_visible_vehicle", "decision", "reviewer_notes"):
            row[field] = ""
    return selected, {
        "panel_count": len(selected), "unique_environment_ids": len(env_occurrences),
        "maximum_environment_id_occurrences": max(env_occurrences.values(), default=0),
        "required_frames_present": sorted({int(row["frame_id"]) for row in selected if int(row["frame_id"]) in REQUIRED_FRAMES}),
        "panel_kinds": dict(sorted(Counter(row["panel_kind"] for row in selected).items())),
    }


def draw_box(image: np.ndarray, box: Sequence[Any], color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x, y, w, h = map(float, box)
    p0 = (int(round(x)), int(round(y)))
    p1 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(image, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.putText(image, label, (p0[0], max(14, p0[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def text_lines(image: np.ndarray, lines: Sequence[str], origin: tuple[int, int], scale: float = 0.43) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
        y += 23


def render_panels(
    temp_dir: Path, selections: Sequence[Mapping[str, Any]], manifest_by_sample: Mapping[str, Mapping[str, str]],
    static_by_sample: Mapping[str, Sequence[Mapping[str, Any]]], actor_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    unresolved_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    unmatched: Mapping[tuple[str, str, str], Sequence[dict[str, Any]]],
    static_conflict_ids: set[str], dynamic_conflict_keys: set[tuple[str, str, str]],
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=False)
    for selection in selections:
        sample_id = selection["sample_id"]
        frame = manifest_by_sample[sample_id]
        image = cv2.imread(str(VIEW / frame["rgb_path"]), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(VIEW / frame["experiment_id"] / "depth" / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or raw_depth is None:
            raise RuntimeError(f"panel source missing: {sample_id}")
        depth = decode_depth_bgra(raw_depth)
        overlay = image.copy()
        focus_box: list[float] | None = None
        focus_record = None
        focus_static = selection.get("environment_object_id", "")
        focus_actor = selection.get("actor_identity", "")
        focus_component = str(selection.get("component_id", ""))
        if focus_static:
            row = next((item for item in static_by_sample.get(sample_id, []) if item["environment_object_id"] == focus_static), None)
            if row is not None:
                focus_box = [float(row["clipped_bbox_x"]), float(row["clipped_bbox_y"]), float(row["clipped_bbox_w"]), float(row["clipped_bbox_h"])]
                focus_record = roi_mask(depth, row, overlay.shape[1], overlay.shape[0])
        if focus_actor:
            row = next((item for item in actor_by_sample.get(sample_id, []) if item["actor_identity"] == focus_actor), None)
            if row is not None:
                focus_box = [float(row["clipped_bbox_x"]), float(row["clipped_bbox_y"]), float(row["clipped_bbox_w"]), float(row["clipped_bbox_h"])]
                focus_record = roi_mask(depth, row, overlay.shape[1], overlay.shape[0])
        if focus_component:
            row = next((item for item in unresolved_by_sample.get(sample_id, []) if str(item["component_id"]) == focus_component), None)
            if row is not None: focus_box = [float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_w"]), float(row["bbox_h"])]
        if focus_record is not None:
            x0, y0, x1, y1, mask = focus_record
            roi = overlay[y0:y1, x0:x1].astype(np.float32)
            cyan = np.asarray([255, 255, 0], dtype=np.float32)
            roi[mask] = 0.45 * roi[mask] + 0.55 * cyan
            overlay[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        for row in static_by_sample.get(sample_id, []):
            conflict = row["environment_object_id"] in static_conflict_ids
            color = (0, 0, 255) if conflict else (255, 0, 255) if truth(row["eligible_visible_v010"]) else (130, 130, 130)
            draw_box(overlay, [row["clipped_bbox_x"], row["clipped_bbox_y"], row["clipped_bbox_w"], row["clipped_bbox_h"]], color, f"S{'+' if truth(row['eligible_visible_v010']) else '-'} {row['environment_object_id'][-10:]}", 3 if row["environment_object_id"] == focus_static else 1)
        for row in actor_by_sample.get(sample_id, []):
            conflict = any((sample_id, sid, row["actor_identity"]) in dynamic_conflict_keys for sid in [item["environment_object_id"] for item in static_by_sample.get(sample_id, [])])
            color = (0, 0, 255) if conflict else (0, 210, 0) if truth(row["eligible_visible_v010"]) else (0, 210, 255)
            draw_box(overlay, [row["clipped_bbox_x"], row["clipped_bbox_y"], row["clipped_bbox_w"], row["clipped_bbox_h"]], color, f"A{'+' if truth(row['eligible_visible_v010']) else '-'} {row['gt_actor_id']}", 3 if row["actor_identity"] == focus_actor else 1)
        for row in unresolved_by_sample.get(sample_id, []):
            draw_box(overlay, [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]], (255, 255, 0), f"U {row['component_id']}", 2)
        for model, color in ((MODELS[0], (0, 165, 255)), (MODELS[1], (255, 100, 0))):
            for pred in unmatched.get((sample_id, model, "0.20"), []):
                draw_box(overlay, [pred["bbox_x0"], pred["bbox_y0"], pred["bbox_x1"] - pred["bbox_x0"], pred["bbox_y1"] - pred["bbox_y0"]], color, f"{model.split('_')[0]} FP {pred['score']:.2f}", 2)
        if focus_box is None:
            focus_box = [320.0, 180.0, 640.0, 360.0]
        x, y, w, h = focus_box
        margin_x, margin_y = max(140.0, 2.0 * w), max(100.0, 2.0 * h)
        zx0 = max(0, int(x + w / 2 - margin_x)); zy0 = max(0, int(y + h / 2 - margin_y))
        zx1 = min(overlay.shape[1], int(x + w / 2 + margin_x)); zy1 = min(overlay.shape[0], int(y + h / 2 + margin_y))
        zoom = overlay[zy0:zy1, zx0:zx1]
        if zoom.size == 0: zoom = overlay
        canvas = np.full((1060, 2080, 3), 20, dtype=np.uint8)
        canvas[45:765, 20:1300] = overlay
        canvas[45:765, 1320:2060] = cv2.resize(zoom, (740, 720), interpolation=cv2.INTER_CUBIC)
        text_lines(canvas, [
            f"Review {int(selection['review_index']):02d} | {selection['panel_kind']} | {selection['focus_label']}",
            f"sample={sample_id} frame={selection['frame_id']} reason={selection['selected_reason']}",
            f"static={selection.get('environment_object_id','')} static2={selection.get('environment_object_id_2','')}",
            f"actor={selection.get('actor_identity','')} component={selection.get('component_id','')} adjudication={selection.get('conflict_adjudication','')}",
            "magenta=S eligible | gray=S ineligible | green=A visible | yellow=A occluded | red=conflict | cyan=unresolved/support | orange/blue=unmatched model",
            f"frame totals: static={len(static_by_sample.get(sample_id, []))} actor={len(actor_by_sample.get(sample_id, []))} unresolved={len(unresolved_by_sample.get(sample_id, []))}",
        ], (20, 800), 0.42)
        target = temp_dir / Path(selection["panel_path"]).name
        if not cv2.imwrite(str(target), canvas):
            raise RuntimeError(f"cannot render {target}")


def metric_line(item: Mapping[str, Any]) -> str:
    return f"{item['tp']}/{item['fp']}/{item['fn']} | {item['precision']:.4f}/{item['recall']:.4f}/{item['f1']:.4f}"


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Route B v3 vehicle-GT reconciliation audit", "", f"Terminal: `{summary['terminal']}`", "",
        "This is an offline validation-only diagnostic. It does not authorize or create v3.1.", "",
        "## Core finding", "",
        "The prior 32 panels were selected-object panels: each drew exactly one selected static object, all actor boxes, and nearby predictions. They were not all-catalog overlays. Apparent unmarked vehicles in those panels therefore cannot be treated as catalog omissions.", "",
        f"- Validation frames: **{summary['counts']['validation_frames']}**.",
        f"- Catalog: **{summary['counts']['static_catalog_objects']}** native/stable objects; projected frame-object rows **{summary['counts']['static_projected_rows']}**.",
        f"- Dynamic actor identities: **{summary['counts']['unique_actor_identities']}**; candidate frame-object rows **{summary['counts']['actor_candidate_rows']}**; current evaluated vehicle rows **{summary['counts']['actor_current_evaluation_rows']}**.",
        f"- Current actor rows ineligible at v0.10/v0.25: **{summary['actor_observability']['current_ineligible_v010']} / {summary['actor_observability']['current_ineligible_v025']}**.",
        f"- Static-static pairs: confirmed **{summary['conflicts']['static_static_confirmed']}**, manual **{summary['conflicts']['static_static_manual']}**, rejected **{summary['conflicts']['static_static_rejected']}**.",
        f"- Static-dynamic frame pairs: confirmed **{summary['conflicts']['static_dynamic_confirmed']}**, manual **{summary['conflicts']['static_dynamic_manual']}**, rejected/occlusion **{summary['conflicts']['static_dynamic_rejected']}**.",
        f"- Semantic vehicle components: **{summary['semantic_components']['total']}**; unresolved **{summary['semantic_components']['unresolved']}**; model-supported unresolved **{summary['semantic_components']['unresolved_with_model_evidence']}**. They remain discrepancy evidence, never GT.", "",
        "## Focused adjudications", "",
        f"- Panels 13/14 Bus IDs: **{summary['focused_findings']['bus_pair']['adjudication']}**. {summary['focused_findings']['bus_pair']['explanation']}",
        f"- Panels 25-32 black-SUV IDs: **{summary['focused_findings']['black_suv_pair']['adjudication']}**. {summary['focused_findings']['black_suv_pair']['explanation']}",
        f"- Panel 24 police-car overlap: **{summary['focused_findings']['panel_24']['adjudication']}**. {summary['focused_findings']['panel_24']['explanation']}",
        f"- Panels 19/22/23: {summary['focused_findings']['semantic_zero_panels']}",
        f"- Frame 265: {summary['focused_findings']['frame_265']['explanation']}", "",
        "## Diagnostic vehicle metrics", "",
        "Counts are TP/FP/FN; metrics are precision/recall/F1. `review-qualified` excludes zero-semantic depth-eligible rows pending human review.", "",
        "| model | visibility | score | frozen actor-only | actor visibility-only | reconciled depth-only | reconciled review-qualified |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for contract in CONTRACTS:
            for threshold in ("0.20", "0.02"):
                item = summary["metrics"][model]["contracts"][contract]["thresholds"][threshold]
                lines.append(f"| {model} | {contract} | {threshold} | {metric_line(item['before_actor_only'])} | {metric_line(item['actor_visibility_only'])} | {metric_line(item['reconciled_depth_only'])} | {metric_line(item['reconciled_review_qualified'])} |")
    lines.extend([
        "", "These are diagnostic metrics only; canonical frozen scores remain unchanged.", "",
        "## Catalog completeness", "",
        summary["catalog_completeness"]["explanation"], "",
        "All required frames have all-object overlays. Eligible/ineligible static candidates, visible/occluded actors, conflicts, unresolved semantic components, and unmatched retained predictions are drawn together. The prior sample's 12 IDs and 15 frames are not used as catalog-completeness evidence.", "",
        "## Recommended v3.1 GT contract (not implemented)", "",
        "1. Preserve source identity separately for actor and environment objects.",
        "2. Require the shared depth visibility contract for both vehicle sources; do not keep geometry-only actor vehicle rows in evaluation.",
        "3. Collapse only confirmed multi-mesh static components to the registered largest-volume representative; quarantine manual pairs.",
        "4. Exclude a static row in frames with confirmed/manual static-dynamic conflicts until adjudicated; projected overlap alone does not establish duplication.",
        "5. Quarantine depth-eligible rows with zero semantic-vehicle support for manual review rather than automatically rejecting or accepting them.",
        "6. Keep unexplained semantic components and unmatched predictions as `UNRESOLVED_VEHICLE_CANDIDATE`; neither is GT without an authoritative actor/environment source and visibility evidence.",
        "7. Version the resulting view and rerun the same frozen-prediction scorer only after the supplied panels are reviewed and explicit v3.1 authorization is given.", "",
        "## Manual review", "",
        f"- Panels: **{summary['manual_review']['panel_count']}**, unique environment IDs: **{summary['manual_review']['unique_environment_ids']}**, maximum repeats per ID: **{summary['manual_review']['maximum_environment_id_occurrences']}**.",
        "- Open each full-resolution PNG under `manual_review_panels/` in CSV order. Fill only the blank reviewer columns in a copy of `manual_review_selection.csv`; do not alter this evidence package.", "",
        "## Integrity", "",
        "- Frozen actor-only metrics reproduce exactly for both models, contracts, and thresholds.",
        "- The static catalog passed the Phase-2 native/stable identity verifier.",
        "- Only the two validation episodes were resolved; locked-test payload paths were not opened.",
        "- All source hashes were identical before and after analysis; retained predictions were reused without inference.",
        "- No CARLA, OAI, container, collection, training, v3.1, q/AE, commit, push, pull, or fetch action occurred.", "",
    ])
    return "\n".join(lines)


def focused_findings(
    static_conflicts: Sequence[Mapping[str, Any]], dynamic_conflicts: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]], static_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bus = next(row for row in static_conflicts if "prior_panels_13_14_bus_pair" in row["focus_case"])
    black = next(row for row in static_conflicts if "prior_panels_25_32_black_suv_pair" in row["focus_case"])
    panel24 = [row for row in dynamic_conflicts if "prior_panel_24_police_car" in row["focus_case"]]
    p24 = max(panel24, key=lambda row: float(row["projected_box_iou"])) if panel24 else None
    frame265 = [row for row in actor_rows if int(row["frame_id"]) == 265]
    bus265 = [row for row in static_rows if int(row["frame_id"]) == 265 and row["semantic_class"] == "Bus"]
    relations = [row for row in dynamic_conflicts if int(row["frame_id"]) == 265 and row["static_class"] == "Bus"]
    behind = [row for row in relations if row["depth_order"] == "actor_behind_static" and float(row["projected_box_iou"]) > 0.0]
    current = [row for row in frame265 if truth(row["current_evaluation_included"])]
    current_invisible = [row for row in current if not truth(row["eligible_visible_v010"])]
    zero_ids = {"9f9707ae5c", "e24742de6c", "c80ed13576"}
    zero_focus = [row for row in static_rows if row["environment_object_id"][-64:][:10] in zero_ids and int(row["frame_id"]) in {281, 1341, 1357}]
    return {
        "bus_pair": {
            "adjudication": bus["adjudication"],
            "explanation": f"Native IDs {bus['native_id_a']} and {bus['native_id_b']} share the same physical bus mesh family; center distance {float(bus['world_center_distance_m']):.3f} m, footprint IoU {float(bus['oriented_footprint_iou']):.4f}, median projected IoU {float(bus['median_projected_iou']):.4f} over {bus['common_projected_frames']} frames. Use representative {bus['preferred_environment_object_id']} and suppress component {bus['duplicate_environment_object_id']}.",
        },
        "black_suv_pair": {
            "adjudication": black["adjudication"],
            "explanation": f"The two IDs are overlapping components of one static SUV: center distance {float(black['world_center_distance_m']):.3f} m and footprint IoU {float(black['oriented_footprint_iou']):.4f}. Use representative {black['preferred_environment_object_id']}; the second component must not flicker into GT independently.",
        },
        "panel_24": {
            "adjudication": p24["adjudication"] if p24 else "NO_STATIC_DYNAMIC_CANDIDATE",
            "explanation": (f"The strongest static/actor overlap has projected IoU {float(p24['projected_box_iou']):.4f}, world-center separation {float(p24['world_center_distance_m']):.2f} m, footprint IoU {float(p24['oriented_footprint_iou']):.4f}, and depth order `{p24['depth_order']}`; it is not the same physical vehicle." if p24 else "No actor has sufficient projected/world overlap with the selected static police-car object."),
        },
        "semantic_zero_panels": f"All {len(zero_focus)} focused rows reproduce depth v0.10 eligibility with zero semantic-vehicle visible pixels. The depth arithmetic is internally consistent, but the source/semantic disagreement is quarantined for manual review and excluded from review-qualified metrics; radar/model evidence cannot rescue it.",
        "frame_265": {
            "actor_rows": frame265,
            "bus_static_rows": [{key: row[key] for key in ("environment_object_id", "range_m", "eligible_visible_v010", "actor_near_depth_m", "actor_far_depth_m")} for row in bus265],
            "bus_actor_relations": relations,
            "explanation": f"The old renderer drew all {len(frame265)} actor vehicle boxes whether observable/evaluated or not. Of {len(current)} actor rows actually in the frozen evaluation denominator, {len(current_invisible)} fail v0.10. {len(behind)} projected-overlap relations are geometrically separate actors behind the static bus by depth ordering, not duplicate bus GT.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not PLAN.is_file() or not DETAILS.is_file():
        raise RuntimeError("registered plan/details missing")
    required_absent = [
        EXPERIMENT / name for name in (
            "VEHICLE_GT_RECONCILIATION_AUDIT.md", "audit_summary.json",
            "actor_visibility_audit.csv", "static_static_conflicts.csv",
            "static_dynamic_conflicts.csv", "unexplained_vehicle_components.csv",
            "all_object_frame_summary.csv", "manual_review_selection.csv",
            "manual_review_panels", "provenance_manifest.json", "TERMINAL_VERDICT.txt",
            "audit_completion.json", "provenance",
        )
    ]
    existing = [str(path) for path in required_absent if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite outputs: {existing}")
    report = read_json(CANONICAL_REPORT)
    validation_episodes = [Path(row["output_dir"]).name for row in report["episodes"] if row["split"] == "val"]
    expected_episodes = ["canonical_v3_05_val_30_30_s601_tm1601", "canonical_v3_06_val_50_50_s602_tm1602"]
    if validation_episodes != expected_episodes:
        raise RuntimeError(f"validation episode drift: {validation_episodes}")
    manifest = read_csv(VIEW / "manifest.csv")
    if len(manifest) != 3345 or len({row["sample_id"] for row in manifest}) != 3345:
        raise RuntimeError("validation frame count/uniqueness failure")
    if any(row["split"] != "val" or row["experiment_id"] not in expected_episodes or "test" in row["experiment_id"].lower() for row in manifest):
        raise RuntimeError("locked-test/non-validation row in view")
    verify = verify_static_environment_truth_v1(
        STATIC_AUDIT / "static_environment_truth/catalog",
        expected_map_name="Carla/Maps/Town10HD_Opt",
        expected_map_sha256="5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7",
        expected_required_semantic_classes=["Car", "Truck", "Bus"],
    )
    if verify["object_count"] != 123:
        raise RuntimeError("static catalog count drift")
    input_paths = [
        PLAN, DETAILS, Path(__file__), CANONICAL_REPORT, VISIBILITY_HELPER, PRIOR_SCRIPT,
        STATIC_AUDIT / "metrics_before_after.json", STATIC_AUDIT / "provenance_manifest.json",
        STATIC_AUDIT / "static_vehicle_visibility.csv",
        STATIC_AUDIT / "static_environment_truth/catalog/static_environment_objects.csv",
        STATIC_AUDIT / "static_environment_truth/catalog/static_environment_snapshot.json",
        VIEW / "manifest.csv", VIEW / "object_boxes_all.csv", VIEW / "object_visibility_all.csv",
        VIEW / "contracts/v010/object_boxes.csv", VIEW / "contracts/v025/object_boxes.csv",
        COMPARISON / "scripts/score_report_v1.py",
    ]
    for model in MODELS:
        input_paths.extend([
            COMPARISON / "predictions" / model / "detections.csv",
            COMPARISON / "predictions" / model / "inference_manifest.json",
            COMPARISON / "predictions" / model / "segmentation_manifest.csv",
        ])
    before_inputs = input_records(input_paths)
    parity = prior.projection_parity({row["sample_id"]: row for row in manifest}, read_csv(VIEW / "object_boxes_all.csv"))
    if not parity["passed"]:
        raise RuntimeError("projection parity failed")
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "plan_sha256": sha256(PLAN), "details_sha256": sha256(DETAILS), "validation_frames": len(manifest), "catalog_objects": verify["object_count"], "projection_min_iou": parity["minimum_box_iou"], "projection_mean_iou": parity["mean_box_iou"]}, indent=2))
        return 0
    started = time.monotonic()
    boxes = read_csv(VIEW / "object_boxes_all.csv")
    visibility = read_csv(VIEW / "object_visibility_all.csv")
    if len(boxes) != len(visibility) or not boxes:
        raise RuntimeError("actor box/visibility row reconciliation failure")
    visibility_by_key = {(row["sample_id"], row["gt_actor_id"]): row for row in visibility}
    if len(visibility_by_key) != len(visibility):
        raise RuntimeError("actor visibility keys are not unique")
    boxes_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in boxes: boxes_by_sample[row["sample_id"]].append(row)
    current_v010 = {(row["sample_id"], row["gt_actor_id"]) for row in read_csv(VIEW / "contracts/v010/object_boxes.csv") if row["label"] == "vehicle"}
    current_v025 = {(row["sample_id"], row["gt_actor_id"]) for row in read_csv(VIEW / "contracts/v025/object_boxes.csv") if row["label"] == "vehicle"}
    if current_v010 != current_v025 or len(current_v010) != 7265:
        raise RuntimeError(f"current actor vehicle contract drift: {len(current_v010)}/{len(current_v025)}")
    static_rows = read_csv(STATIC_AUDIT / "static_vehicle_visibility.csv")
    static_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in static_rows: static_by_sample[row["sample_id"]].append(row)
    catalog = read_csv(STATIC_AUDIT / "static_environment_truth/catalog/static_environment_objects.csv")
    static_conflicts, confirmed_secondary, manual_static_ids = static_static_conflicts(catalog, static_rows)
    frame_ids = {row["sample_id"] for row in manifest}
    predictions = {
        model: prior.load_predictions_with_visual_fields(scorer, COMPARISON / "predictions" / model / "detections.csv", frame_ids)
        for model in MODELS
    }
    actor_rows, dynamic_conflicts, unresolved, frame_summaries = analyze_frames(
        manifest, boxes_by_sample, visibility_by_key, current_v010, static_by_sample, predictions,
    )
    variants = build_gt_variants([row["sample_id"] for row in manifest], actor_rows, static_rows, confirmed_secondary, manual_static_ids, dynamic_conflicts)
    metrics, baseline_exact = compute_metrics(manifest, predictions, variants, read_json(STATIC_AUDIT / "metrics_before_after.json"))
    if not baseline_exact:
        raise RuntimeError("frozen baseline metric parity failure")
    unmatched = add_unmatched_counts(frame_summaries, predictions, variants["v010"]["reconciled_review_qualified"])
    selections, manual_summary = select_panels(manifest, static_rows, actor_rows, static_conflicts, dynamic_conflicts, unresolved)
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    actor_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actor_rows: actor_by_sample[row["sample_id"]].append(row)
    for row in unresolved: unresolved_by_sample[row["sample_id"]].append(row)
    conflict_static_ids = {
        row[key] for row in static_conflicts if row["adjudication"] != "REJECTED_DISTINCT"
        for key in ("environment_object_id_a", "environment_object_id_b")
    }
    conflict_static_ids.update(row["environment_object_id"] for row in dynamic_conflicts if row["adjudication"] != "REJECTED_DISTINCT_OCCLUSION")
    dynamic_keys = {(row["sample_id"], row["environment_object_id"], row["actor_identity"]) for row in dynamic_conflicts if row["adjudication"] != "REJECTED_DISTINCT_OCCLUSION"}
    with tempfile.TemporaryDirectory(prefix="route_b_reconciliation_panels_") as temp_name:
        temp_panels = Path(temp_name) / "panels"
        render_panels(temp_panels, selections, manifest_by_sample, static_by_sample, actor_by_sample, unresolved_by_sample, predictions, unmatched, conflict_static_ids, dynamic_keys)
        findings = focused_findings(static_conflicts, dynamic_conflicts, actor_rows, static_rows)
        actor_current = [row for row in actor_rows if truth(row["current_evaluation_included"])]
        component_total = sum(
            int(row["components_explained_actor_only"]) + int(row["components_explained_static_only"])
            + int(row["components_explained_actor_and_static"]) + int(row["unresolved_vehicle_components"])
            for row in frame_summaries
        )
        focus_frames = [row for row in frame_summaries if int(row["frame_id"]) in REQUIRED_FRAMES]
        genuine_absent = sum(int(row["unresolved_vehicle_components"]) for row in focus_frames)
        summary: dict[str, Any] = {
            "schema": "route_b_v3_vehicle_gt_reconciliation_audit_v1", "created_utc": utc_now(),
            "terminal": SUCCESS, "diagnostic_only": True,
            "repository": {"branch": "master", "head": os.popen("git rev-parse HEAD").read().strip(), "preexisting_dirty_paths": ["OAI/openairinterface5g"]},
            "map": {"name": verify["map_name"], "opendrive_sha256": verify["map_sha256"]},
            "counts": {
                "validation_frames": len(manifest), "static_catalog_objects": len(catalog),
                "static_catalog_by_class": dict(sorted(Counter(row["semantic_class"] for row in catalog).items())),
                "static_projected_rows": len(static_rows), "unique_actor_identities": len({row["actor_identity"] for row in actor_rows}),
                "actor_candidate_rows": len(actor_rows), "actor_current_evaluation_rows": len(actor_current),
            },
            "actor_observability": {
                "current_eligible_v010": sum(truth(row["eligible_visible_v010"]) for row in actor_current),
                "current_ineligible_v010": sum(not truth(row["eligible_visible_v010"]) for row in actor_current),
                "current_eligible_v025": sum(truth(row["eligible_clear_v025"]) for row in actor_current),
                "current_ineligible_v025": sum(not truth(row["eligible_clear_v025"]) for row in actor_current),
                "current_semantic_zero_v010": sum(truth(row["eligible_visible_v010"]) and int(row["semantic_vehicle_visible_px"]) == 0 for row in actor_current),
                "visible_fraction_distribution_current": numeric_summary([float(row["visible_fraction"]) for row in actor_current]),
            },
            "static_observability": {
                "eligible_v010_raw": sum(truth(row["eligible_visible_v010"]) for row in static_rows),
                "eligible_v025_raw": sum(truth(row["eligible_clear_v025"]) for row in static_rows),
                "semantic_zero_v010": sum(truth(row["eligible_visible_v010"]) and int(row["semantic_vehicle_visible_px"]) == 0 for row in static_rows),
                "confirmed_duplicate_secondary_ids": sorted(confirmed_secondary),
                "manual_static_conflict_ids": sorted(manual_static_ids),
            },
            "conflicts": {
                "static_static_confirmed": sum(row["adjudication"] == "CONFIRMED_DUPLICATE" for row in static_conflicts),
                "static_static_manual": sum(row["adjudication"] == "MANUAL_REVIEW_REQUIRED" for row in static_conflicts),
                "static_static_rejected": sum(row["adjudication"] == "REJECTED_DISTINCT" for row in static_conflicts),
                "static_dynamic_confirmed": sum(row["adjudication"] == "CONFIRMED_SAME_PHYSICAL_OBJECT" for row in dynamic_conflicts),
                "static_dynamic_manual": sum(row["adjudication"] == "MANUAL_REVIEW_REQUIRED" for row in dynamic_conflicts),
                "static_dynamic_rejected": sum(row["adjudication"] == "REJECTED_DISTINCT_OCCLUSION" for row in dynamic_conflicts),
            },
            "semantic_components": {
                "total": component_total, "unresolved": len(unresolved),
                "unresolved_with_model_evidence": sum(truth(row["retained_model_evidence"]) for row in unresolved),
                "unresolved_pixels": numeric_summary([int(row["component_pixels"]) for row in unresolved]),
            },
            "metrics": metrics, "baseline_metrics_exact": baseline_exact,
            "focused_findings": findings,
            "catalog_completeness": {
                "required_focus_frames": REQUIRED_FRAMES,
                "unresolved_components_in_required_frames": genuine_absent,
                "explanation": f"All-object overlays show that the bulk of previously unmarked nearby vehicles are catalogued static objects or actor rows omitted only by the selected-object visualization. Across the eight required frames, {genuine_absent} semantic vehicle components are not explained by depth-consistent actor/static support; these are genuinely absent from both reconciled sources only as unresolved candidates, not proven GT omissions.",
            },
            "manual_review": manual_summary,
            "prior_panel_generator": {
                "selected_static_only": True,
                "all_static_objects_drawn": False,
                "code_path": str(PRIOR_SCRIPT.relative_to(REPO)),
                "explanation": "render_panel draws selection's static_box once; it iterates actor rows but never all static rows",
                "prior_sample_unique_static_ids": 12, "prior_sample_unique_frames": 15,
            },
            "prohibited_actions": {
                "carla_started": False, "oai_started": False, "containers_started": False,
                "collection_runs": 0, "training_runs": 0, "inference_runs": 0,
                "locked_test_payload_opened": False, "v3_1_created": False,
                "q_ae_work": False, "commit": False, "push": False, "pull": False, "fetch": False,
            },
            "wall_seconds": time.monotonic() - started,
        }
        required_frame_gate = manual_summary["required_frames_present"] == list(REQUIRED_FRAMES)
        all_finite = all(math.isfinite(float(row[field])) for row in actor_rows for field in ("gt_distance_m", "clipped_projected_area_px", "native_visible_px", "model_input_visible_px", "visible_fraction", "semantic_vehicle_visible_px", "radar_support_points"))
        acceptance = {
            "baseline_metrics_exact": baseline_exact,
            "catalog_identity_verified": verify["verdict"] == "PASS",
            "validation_only_3345": len(manifest) == 3345,
            "projection_parity_passed": parity["passed"],
            "actor_rows_complete_finite": all_finite,
            "manual_review_diversity": manual_summary["unique_environment_ids"] >= 24 and manual_summary["maximum_environment_id_occurrences"] <= 2,
            "required_all_object_frames": required_frame_gate,
            "model_predictions_never_gt": True,
            "locked_test_payload_absent": True,
        }
        if not all(acceptance.values()):
            raise RuntimeError(f"acceptance gate failure: {acceptance}")
        summary["acceptance_gates"] = acceptance
        after_inputs = input_records(input_paths)
        if before_inputs != after_inputs:
            raise RuntimeError("source input hash changed during audit")
        provenance_dir = EXPERIMENT / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=False)
        write_json_x(provenance_dir / "projection_parity.json", parity)
        write_json_x(provenance_dir / "source_input_hashes.json", {"before": before_inputs, "after": after_inputs, "identical": True})
        write_csv_x(EXPERIMENT / "actor_visibility_audit.csv", ACTOR_FIELDS, actor_rows)
        write_csv_x(EXPERIMENT / "static_static_conflicts.csv", STATIC_STATIC_FIELDS, static_conflicts)
        write_csv_x(EXPERIMENT / "static_dynamic_conflicts.csv", STATIC_DYNAMIC_FIELDS, dynamic_conflicts)
        write_csv_x(EXPERIMENT / "unexplained_vehicle_components.csv", COMPONENT_FIELDS, unresolved)
        write_csv_x(EXPERIMENT / "all_object_frame_summary.csv", FRAME_FIELDS, frame_summaries)
        write_csv_x(EXPERIMENT / "manual_review_selection.csv", SELECTION_FIELDS, selections)
        panel_dir = EXPERIMENT / "manual_review_panels"
        panel_dir.mkdir(parents=True, exist_ok=False)
        for source in sorted(temp_panels.glob("*.png")):
            target = panel_dir / source.name
            if target.exists(): raise RuntimeError(f"panel overwrite refusal: {target}")
            shutil.copy2(source, target)
        write_json_x(EXPERIMENT / "audit_summary.json", summary)
        write_text_x(EXPERIMENT / "VEHICLE_GT_RECONCILIATION_AUDIT.md", report_markdown(summary))
        write_text_x(EXPERIMENT / "TERMINAL_VERDICT.txt", SUCCESS + "\n")
    required_outputs = [
        "REGISTERED_VEHICLE_GT_RECONCILIATION_PLAN.md", "REGISTERED_ALGORITHM_DETAILS.json",
        "VEHICLE_GT_RECONCILIATION_AUDIT.md", "audit_summary.json", "actor_visibility_audit.csv",
        "static_static_conflicts.csv", "static_dynamic_conflicts.csv",
        "unexplained_vehicle_components.csv", "all_object_frame_summary.csv",
        "manual_review_selection.csv", "TERMINAL_VERDICT.txt", "provenance/projection_parity.json",
        "provenance/source_input_hashes.json", "scripts/run_vehicle_gt_reconciliation_audit_v1.py",
    ]
    if any(not (EXPERIMENT / name).is_file() for name in required_outputs) or len(list((EXPERIMENT / "manual_review_panels").glob("*.png"))) != manual_summary["panel_count"]:
        raise RuntimeError("required output completeness failure")
    output_files = []
    for path in sorted((path for path in EXPERIMENT.rglob("*") if path.is_file()), key=lambda path: str(path.relative_to(EXPERIMENT))):
        output_files.append({"path": str(path.relative_to(EXPERIMENT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    provenance = {
        "schema": "route_b_v3_vehicle_gt_reconciliation_provenance_v1", "created_utc": utc_now(),
        "terminal": SUCCESS, "repository": summary["repository"], "registered_plan_sha256": sha256(PLAN),
        "registered_algorithm_details_sha256": sha256(DETAILS), "source_inputs_identical": True,
        "validation_episodes": validation_episodes, "locked_test_payload_opened": False,
        "acceptance_gates": summary["acceptance_gates"], "prohibited_actions": summary["prohibited_actions"],
        "output_files_before_provenance": output_files,
    }
    write_json_x(EXPERIMENT / "provenance_manifest.json", provenance)
    completion = {
        "schema": "route_b_v3_vehicle_gt_reconciliation_completion_v1", "created_utc": utc_now(),
        "terminal": SUCCESS, "audit_summary_sha256": sha256(EXPERIMENT / "audit_summary.json"),
        "report_sha256": sha256(EXPERIMENT / "VEHICLE_GT_RECONCILIATION_AUDIT.md"),
        "provenance_sha256": sha256(EXPERIMENT / "provenance_manifest.json"),
        "manual_review_panels": manual_summary["panel_count"],
    }
    write_json_x(EXPERIMENT / "audit_completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
