#!/usr/bin/env python3
"""Frozen-pair depth/ray oracles and inherited dense-localizer compositions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
AUDIT = PACKAGE.parent / "route_b_v3_1_person_contract_audit_v1"
EXPANDED = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2"
for path in (str(PACKAGE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common_v1 import read_csv, sha256, utc_now, write_csv_x, write_json_x, write_text_x  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matching = load_module("localizer_counterfactual_final_matching_v1", AUDIT / "matching_v1.py")
scoring = load_module("localizer_counterfactual_final_scoring_v2", EXPANDED / "scoring_v2.py")
native = scoring.native_evaluator()


def parse_matrix(value: str) -> np.ndarray:
    matrix = np.asarray(json.loads(value), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid camera matrix")
    return matrix


def transform_point(matrix: np.ndarray, local: Sequence[float]) -> np.ndarray:
    homogeneous = np.asarray([float(local[0]), float(local[1]), float(local[2]), 1.0])
    return np.asarray(matrix, dtype=np.float64).dot(homogeneous)[:3]

DETECTION_FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1",
)

COMPACT_FIELDS = (
    "arm", "threshold", "stable_row", "sample_id", "source_identity", "score",
    "sample_method", "cell_x", "cell_y", "local_x", "local_y", "local_z",
    "world_x", "world_y", "world_z", "size_x", "size_y", "size_z", "yaw_sin",
    "yaw_cos",
)


def raw_detections(path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    rows = read_csv(path)
    return rows, {index: row for index, row in enumerate(rows)}


def dense_samples(path: Path) -> tuple[dict[int, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    by_candidate: dict[int, dict[str, str]] = {}
    by_gt: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        if row["sample_kind"] == "candidate_predicted_full_box_cell":
            by_candidate[int(row["candidate_stable_row"])] = row
        else:
            by_gt[(row["sample_id"], row["source_identity"])] = row
    return by_candidate, by_gt


def numeric_local(row: Mapping[str, Any], prefix: str = "") -> np.ndarray:
    return np.asarray([float(row[f"{prefix}local_x"]), float(row[f"{prefix}local_y"]),
                       float(row[f"{prefix}local_z"])], dtype=np.float64)


def numeric_world(row: Mapping[str, Any], prefix: str = "") -> np.ndarray:
    return np.asarray([float(row[f"{prefix}world_x"]), float(row[f"{prefix}world_y"]),
                       float(row[f"{prefix}world_z"])], dtype=np.float64)


def field_values(sample: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = numeric_local(sample, "base_")
    world = numeric_world(sample, "base_")
    dims = np.asarray([float(sample[f"base_size_{axis}"]) for axis in "xyz"], dtype=np.float64)
    yaw = np.asarray([float(sample["base_yaw_sin"]), float(sample["base_yaw_cos"])])
    return local, world, dims, yaw


def replace_geometry(row: Mapping[str, str], *, local: np.ndarray, world: np.ndarray,
                     dims: np.ndarray | None = None, yaw: np.ndarray | None = None) -> dict[str, Any]:
    value: dict[str, Any] = dict(row)
    value.update({
        "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
        "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
    })
    if dims is not None:
        value.update({"size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2])})
    if yaw is not None:
        value.update({"yaw_sin": float(yaw[0]), "yaw_cos": float(yaw[1])})
    return value


def frame_calibration(dataset: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_csv(dataset / "dataset/manifest.csv"):
        if row["split"] != "val":
            continue
        sx, sy = 768.0 / float(row["camera_width"]), 432.0 / float(row["camera_height"])
        result[row["sample_id"]] = {
            "matrix": np.asarray(parse_matrix(row["camera_matrix_json"]), dtype=np.float64),
            "fx": float(row["camera_fx"]) * sx, "fy": float(row["camera_fy"]) * sy,
            "cx": float(row["camera_cx"]) * sx, "cy": float(row["camera_cy"]) * sy,
        }
    return result


def visibility_map(dataset: Path) -> dict[tuple[str, str], float]:
    path = dataset / "derived_targets/visible_anchor_targets_v010.csv"
    return {(row["sample_id"], row["source_identity"]): float(row["visible_fraction"])
            for row in read_csv(path) if row["split"] == "val"}


def fixed_pairs(dataset: Path, detections: Path) -> tuple[
    list[str], dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, str]],
    dict[float, dict[str, Any]], dict[str, list[dict[str, Any]]],
]:
    frames = matching.load_frame_ids(dataset)
    gt, metadata, _clear = matching.load_person_gt(dataset)
    predictions = matching.load_predictions(detections)
    matching.annotate_neutral_predictions(predictions, dataset, frames)
    results = {threshold: matching.image_match(
        frames, gt, predictions, threshold, "FULL_BOX_IOU_050",
    ) for threshold in (0.02, 0.20)}
    return frames, gt, metadata, results, predictions


def oracle_geometry(raw: Mapping[str, str], gt_row: Mapping[str, str],
                    calibration: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predicted = numeric_local(raw)
    target = np.asarray([float(gt_row["object_sensor_x"]), float(gt_row["object_sensor_y"]),
                         float(gt_row["object_sensor_z"])], dtype=np.float64)
    if predicted[0] <= 0.0 or target[0] <= 0.0:
        raise RuntimeError("nonpositive camera-forward depth in oracle pair")
    pred_ratio_y, pred_ratio_z = predicted[1] / predicted[0], predicted[2] / predicted[0]
    gt_ratio_y, gt_ratio_z = target[1] / target[0], target[2] / target[0]
    locals_by_arm = {
        "predicted_ray_predicted_depth": predicted,
        "predicted_ray_gt_depth": np.asarray(
            [target[0], pred_ratio_y * target[0], pred_ratio_z * target[0]], dtype=np.float64,
        ),
        "gt_ray_predicted_depth": np.asarray(
            [predicted[0], gt_ratio_y * predicted[0], gt_ratio_z * predicted[0]], dtype=np.float64,
        ),
        "gt_ray_gt_depth": target,
    }
    return {name: (local, np.asarray(transform_point(calibration["matrix"], local), dtype=np.float64))
            for name, local in locals_by_arm.items()}


def pair_error(world: np.ndarray, target: Mapping[str, Any]) -> float:
    return float(math.hypot(float(world[0]) - float(target["world_x"]),
                            float(world[1]) - float(target["world_y"])))


def percentile(values: Sequence[float], q: float) -> float | str:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else ""


def summarize_errors(values: Sequence[float]) -> dict[str, Any]:
    values = list(values)
    return {
        "count": len(values), "within_1m_fraction": sum(v <= 1.0 for v in values) / max(1, len(values)),
        "within_2m_fraction": sum(v <= 2.0 for v in values) / max(1, len(values)),
        "within_3m_fraction": sum(v <= 3.0 for v in values) / max(1, len(values)),
        "within_5m_fraction": sum(v <= 5.0 for v in values) / max(1, len(values)),
        "mean_m": statistics.fmean(values) if values else "", "median_m": percentile(values, 50),
        "p75_m": percentile(values, 75), "p90_m": percentile(values, 90),
        "outside_3m_count": sum(v > 3.0 for v in values),
    }


def build_oracle_pairs(results: Mapping[float, Mapping[str, Any]],
                       raw_by_stable: Mapping[int, Mapping[str, str]],
                       metadata: Mapping[tuple[str, str], Mapping[str, str]],
                       calibrations: Mapping[str, Mapping[str, Any]],
                       visible: Mapping[tuple[str, str], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (0.02, 0.20):
        for pair in results[threshold]["matches"]:
            prediction, target = pair["prediction"], pair["gt"]
            stable = int(prediction["stable_row"])
            key = (target["sample_id"], target["source_identity"])
            gt_row = metadata[key]
            raw = raw_by_stable[stable]
            geometry = oracle_geometry(raw, gt_row, calibrations[target["sample_id"]])
            predicted_local = geometry["predicted_ray_predicted_depth"][0]
            gt_local = geometry["gt_ray_gt_depth"][0]
            delta = predicted_local - gt_local
            ray = gt_local / max(1e-12, float(np.linalg.norm(gt_local)))
            along = float(np.dot(delta, ray))
            perpendicular = float(np.linalg.norm(delta - along * ray))
            cosine = float(np.dot(predicted_local, gt_local) / max(
                1e-12, np.linalg.norm(predicted_local) * np.linalg.norm(gt_local),
            ))
            value: dict[str, Any] = {
                "threshold": threshold, "sample_id": target["sample_id"],
                "source_identity": target["source_identity"], "gt_stable_row": target["stable_row"],
                "candidate_stable_row": stable, "score": prediction["score"],
                "distance_m": target["distance_m"], "area_px": target["area_px"],
                "radar_supported": int(target["radar_supported"]),
                "clear_v025": int(target["clear_v025"]),
                "visible_fraction": visible[key], "predicted_depth_m": predicted_local[0],
                "gt_depth_m": gt_local[0], "signed_forward_depth_error_m": delta[0],
                "absolute_forward_depth_error_m": abs(float(delta[0])),
                "local_along_gt_ray_error_m": along,
                "local_ray_perpendicular_error_m": perpendicular,
                "ray_angular_error_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            }
            for arm, (_local, world) in geometry.items():
                value[f"{arm}_world_xy_error_m"] = pair_error(world, target)
            rows.append(value)
    return rows


def write_standard_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DETECTION_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def vehicle_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    fields = [field for field in DETECTION_FIELDS if field != "prediction_index"]
    return [tuple(row[field] for field in fields) for row in rows if row["class_name"] == "vehicle"]


def score_rows(dataset: Path, rows: Sequence[Mapping[str, Any]], frames: Sequence[str],
               gt_native: Mapping[str, Any], threshold: float,
               *, taxonomy: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    with tempfile.TemporaryDirectory(prefix="lraspp_counterfactual_", dir="/tmp") as directory:
        path = Path(directory) / "detections.csv"
        write_standard_csv(path, rows)
        native_predictions, missing = native.load_predictions(path)
        if missing:
            raise RuntimeError(f"counterfactual native prediction fields missing: {missing[:3]}")
        primary = native.score_arm(
            experiment=dataset, contract="v010", frame_ids=frames, predictions=native_predictions,
            gt=gt_native, threshold=threshold, ignore_cache={},
        )
        match_predictions = matching.load_predictions(path)
        matching.annotate_neutral_predictions(match_predictions, dataset, frames)
        person_gt, _metadata, _clear = matching.load_person_gt(dataset)
        image = matching.image_match(
            frames, person_gt, match_predictions, threshold, "FULL_BOX_IOU_050",
        )
        diagnostic = matching.summarize_conditional(
            "counterfactual", threshold, "FULL_BOX_IOU_050", image,
        )[0]
        taxonomy_result = (native.run_taxonomy(
            dataset, frames, native_predictions, gt_native, {},
        ) if taxonomy else None)
    return primary, diagnostic, taxonomy_result


def dense_geometry(sample: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(sample["cell_valid"]) != 1:
        raise RuntimeError("registered dense sample is invalid")
    return field_values(sample)


def compact_record(arm: str, threshold: float | str, stable: int,
                   source_identity: str, raw: Mapping[str, Any], method: str,
                   sample: Mapping[str, Any] | None, local: np.ndarray,
                   world: np.ndarray, dims: np.ndarray, yaw: np.ndarray) -> dict[str, Any]:
    return {
        "arm": arm, "threshold": threshold, "stable_row": stable,
        "sample_id": raw["sample_id"], "source_identity": source_identity,
        "score": raw["score"], "sample_method": method,
        "cell_x": sample["cell_x"] if sample is not None else "",
        "cell_y": sample["cell_y"] if sample is not None else "",
        "local_x": local[0], "local_y": local[1], "local_z": local[2],
        "world_x": world[0], "world_y": world[1], "world_z": world[2],
        "size_x": dims[0], "size_y": dims[1], "size_z": dims[2],
        "yaw_sin": yaw[0], "yaw_cos": yaw[1],
    }


def make_deployable(candidate_rows: Sequence[Mapping[str, str]],
                    candidate_samples: Mapping[int, Mapping[str, str]],
                    compact: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stable, raw in enumerate(candidate_rows):
        if raw["class_name"] != "person":
            output.append(dict(raw)); continue
        sample = candidate_samples[stable]
        local, world, dims, yaw = dense_geometry(sample)
        output.append(replace_geometry(raw, local=local, world=world, dims=dims, yaw=yaw))
        compact.append(compact_record(
            "inherited_at_candidate_cell", "both", stable, "", raw,
            "hard_floor_candidate_predicted_full_box_cell", sample, local, world, dims, yaw,
        ))
    return output


def pair_map(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(pair["prediction"]["stable_row"]): pair for pair in result["matches"]}


def make_gt_cell(candidate_rows: Sequence[Mapping[str, str]], pairs: Mapping[int, Mapping[str, Any]],
                 gt_samples: Mapping[tuple[str, str], Mapping[str, str]], threshold: float,
                 compact: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stable, raw in enumerate(candidate_rows):
        pair = pairs.get(stable)
        if raw["class_name"] != "person" or pair is None:
            output.append(dict(raw)); continue
        target = pair["gt"]
        key = (target["sample_id"], target["source_identity"])
        sample = gt_samples[key]
        local, world, dims, yaw = dense_geometry(sample)
        output.append(replace_geometry(raw, local=local, world=world, dims=dims, yaw=yaw))
        compact.append(compact_record(
            "inherited_at_gt_cell_oracle", threshold, stable, target["source_identity"], raw,
            "diagnostic_GT_full_box_center_cell", sample, local, world, dims, yaw,
        ))
    return output


def make_oracle(candidate_rows: Sequence[Mapping[str, str]], pairs: Mapping[int, Mapping[str, Any]],
                metadata: Mapping[tuple[str, str], Mapping[str, str]],
                calibrations: Mapping[str, Mapping[str, Any]], arm: str, threshold: float,
                compact: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stable, raw in enumerate(candidate_rows):
        pair = pairs.get(stable)
        if raw["class_name"] != "person" or pair is None:
            output.append(dict(raw)); continue
        target = pair["gt"]
        key = (target["sample_id"], target["source_identity"])
        geometry = oracle_geometry(raw, metadata[key], calibrations[target["sample_id"]])
        local, world = geometry[arm]
        dims = np.asarray([float(raw[f"size_{axis}"]) for axis in "xyz"])
        yaw = np.asarray([float(raw["yaw_sin"]), float(raw["yaw_cos"])])
        output.append(replace_geometry(raw, local=local, world=world))
        compact.append(compact_record(
            arm, threshold, stable, target["source_identity"], raw,
            "fixed_iou50_GT_component_substitution", None, local, world, dims, yaw,
        ))
    return output


def visible_band(value: float, edges: Sequence[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right:
            return f"[{left:g},{right:g})"
    return "outside_registered_bins"


def slice_rows(pair_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
               error_fields: Mapping[str, str]) -> list[dict[str, Any]]:
    definitions: list[tuple[str, str, Callable[[Mapping[str, Any]], bool]]] = [("overall", "all", lambda _r: True)]
    edges = config["strata"]["distance_edges_m"]
    for left, right in zip(edges[:-1], edges[1:]):
        definitions.append(("distance_m", f"[{left:g},{right:g})",
                            lambda row, l=left, r=right: l <= float(row["distance_m"]) < r))
    definitions += [
        ("radar_support", "supported", lambda row: bool(int(row["radar_supported"]))),
        ("radar_support", "unsupported", lambda row: not bool(int(row["radar_supported"]))),
        ("visibility_contract", "clear_v025", lambda row: bool(int(row["clear_v025"]))),
        ("visibility_contract", "primary_v010_only", lambda row: not bool(int(row["clear_v025"]))),
    ]
    vedges = config["strata"]["visible_fraction_edges"]
    for left, right in zip(vedges[:-1], vedges[1:]):
        definitions.append(("visible_fraction", f"[{left:g},{right:g})",
                            lambda row, l=left, r=right: l <= float(row["visible_fraction"]) < r))
    rows: list[dict[str, Any]] = []
    thresholds = sorted({float(row["threshold"]) for row in pair_rows})
    for threshold in thresholds:
        population = [row for row in pair_rows if float(row["threshold"]) == threshold]
        for arm, field in error_fields.items():
            for kind, label, predicate in definitions:
                values = [float(row[field]) for row in population if predicate(row)]
                rows.append({"threshold": threshold, "arm": arm, "subset_kind": kind,
                             "subset_label": label, **summarize_errors(values)})
    return rows


def composition_gates(metrics: Mapping[str, Any], candidate_expected: Mapping[str, Any],
                      config: Mapping[str, Any], *, diagnostic_gt: bool) -> dict[str, bool]:
    gate = config["composition_success"]
    checks = {
        "retain_candidate_iou50_f1_020_exact": math.isclose(
            metrics["iou50_f1_020"], candidate_expected["iou50_f1_020"], rel_tol=0.0, abs_tol=1e-12,
        ),
        "retain_candidate_iou50_recall_002_exact": math.isclose(
            metrics["iou50_recall_002"], candidate_expected["iou50_recall_002"], rel_tol=0.0, abs_tol=1e-12,
        ),
        "person_recall_020": metrics["person_recall_020"] >= gate["person_recall_020_min"],
        "person_f1_020": metrics["person_f1_020"] >= gate["person_f1_020_min"],
        "person_xy_mae_m_020": metrics["person_xy_mae_m_020"] <= gate["person_xy_mae_m_020_max"],
        "additional_world_wrong_reduction": (
            metrics["center_present_world_wrong_reduction"] >= gate["additional_world_wrong_reduction_min"]
        ),
        "conditional_within_3m_002": (
            metrics["iou50_conditional_within_3m_002"] >= gate["iou50_conditional_within_3m_002_min"]
        ),
        "vehicle_and_segmentation_bit_identical": metrics["vehicle_and_segmentation_bit_identical"],
    }
    if not diagnostic_gt:
        checks["no_gt_or_depth_image_at_inference"] = True
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve(strict=True)
    if not (experiment / "DENSE_TRAVERSAL_COMPLETE").is_file():
        raise RuntimeError("analysis requires the registered dense traversal")
    config = json.loads((experiment / "RESOLVED_CONFIG.json").read_text())
    traversal = json.loads((experiment / "DENSE_TRAVERSAL.json").read_text())
    if (traversal["base_traversal_count"] != 1 or traversal["candidate_traversal_count"] != 0
            or traversal["invalid_candidate_cells"] != 0 or traversal["invalid_gt_cells"] != 0
            or traversal["samples_sha256"] != sha256(experiment / "DENSE_FIELD_SAMPLES.csv")):
        raise RuntimeError("dense traversal contract invalid")
    dataset = (ROOT / config["dataset_root"]).resolve(strict=True)
    candidate_path = (ROOT / config["candidate_predictions"] / "detections.csv").resolve(strict=True)
    candidate_rows, raw_by_stable = raw_detections(candidate_path)
    frames, gt, metadata, pair_results, _candidate_predictions = fixed_pairs(dataset, candidate_path)
    gt_native, _states = native.load_gt(dataset, "v010")
    calibrations = frame_calibration(dataset)
    visible = visibility_map(dataset)
    candidate_samples, gt_samples = dense_samples(experiment / "DENSE_FIELD_SAMPLES.csv")
    if len(candidate_samples) != sum(row["class_name"] == "person" for row in candidate_rows):
        raise RuntimeError("candidate dense sample coverage mismatch")

    oracle_rows = build_oracle_pairs(pair_results, raw_by_stable, metadata, calibrations, visible)
    for row in oracle_rows:
        stable = int(row["candidate_stable_row"])
        key = (row["sample_id"], row["source_identity"])
        gt_row = metadata[key]
        target_xy = np.asarray(
            [float(gt_row["object_world_x"]), float(gt_row["object_world_y"])], dtype=np.float64,
        )
        predicted_cell_world = numeric_world(candidate_samples[stable], "base_")
        gt_cell_world = numeric_world(gt_samples[key], "base_")
        row["inherited_candidate_cell_world_xy_error_m"] = float(
            np.linalg.norm(predicted_cell_world[:2] - target_xy)
        )
        row["inherited_gt_cell_world_xy_error_m"] = float(
            np.linalg.norm(gt_cell_world[:2] - target_xy)
        )
    write_csv_x(experiment / "ORACLE_PAIR_RESULTS.csv", oracle_rows)
    component_oracle_arms = (
        "predicted_ray_predicted_depth", "predicted_ray_gt_depth",
        "gt_ray_predicted_depth", "gt_ray_gt_depth",
    )
    oracle_error_fields = {arm: f"{arm}_world_xy_error_m" for arm in component_oracle_arms}
    oracle_error_fields.update({
        "inherited_candidate_cell": "inherited_candidate_cell_world_xy_error_m",
        "inherited_gt_cell": "inherited_gt_cell_world_xy_error_m",
    })
    slice_metrics = slice_rows(oracle_rows, config, oracle_error_fields)

    compact: list[dict[str, Any]] = []
    candidate_vehicle = vehicle_rows(candidate_rows)
    deployable_rows = make_deployable(candidate_rows, candidate_samples, compact)
    if vehicle_rows(deployable_rows) != candidate_vehicle:
        raise RuntimeError("deployable arm changed vehicle rows")
    pair_maps = {threshold: pair_map(pair_results[threshold]) for threshold in (0.02, 0.20)}
    gt_cell_rows = {threshold: make_gt_cell(
        candidate_rows, pair_maps[threshold], gt_samples, threshold, compact,
    ) for threshold in (0.02, 0.20)}

    deploy_primary: dict[float, Any] = {}
    deploy_diag: dict[float, Any] = {}
    deploy_taxonomy = None
    for threshold in (0.02, 0.20):
        primary, diagnostic, taxonomy = score_rows(
            dataset, deployable_rows, frames, gt_native, threshold, taxonomy=(threshold == 0.02),
        )
        deploy_primary[threshold], deploy_diag[threshold] = primary, diagnostic
        if taxonomy is not None:
            deploy_taxonomy = taxonomy
    gt_primary: dict[float, Any] = {}
    gt_diag: dict[float, Any] = {}
    gt_taxonomy = None
    for threshold in (0.02, 0.20):
        primary, diagnostic, taxonomy = score_rows(
            dataset, gt_cell_rows[threshold], frames, gt_native, threshold,
            taxonomy=(threshold == 0.02),
        )
        gt_primary[threshold], gt_diag[threshold] = primary, diagnostic
        if taxonomy is not None:
            gt_taxonomy = taxonomy

    oracle_end_to_end: dict[str, dict[float, Any]] = defaultdict(dict)
    oracle_diagnostics: dict[str, dict[float, Any]] = defaultdict(dict)
    for threshold in (0.02, 0.20):
        for arm in component_oracle_arms:
            rows = make_oracle(
                candidate_rows, pair_maps[threshold], metadata, calibrations, arm, threshold, compact,
            )
            primary, diagnostic, _taxonomy = score_rows(
                dataset, rows, frames, gt_native, threshold, taxonomy=False,
            )
            oracle_end_to_end[arm][threshold] = primary
            oracle_diagnostics[arm][threshold] = diagnostic

    write_csv_x(experiment / "COUNTERFACTUAL_DETECTIONS.csv", compact, COMPACT_FIELDS)

    candidate_expected = config["reconciliation"]["candidate"]
    candidate_preservation = json.loads((ROOT / config["published_evaluation"]).read_text())["preservation"]
    preserved = bool(candidate_preservation["all_preserved"] and vehicle_rows(deployable_rows) == candidate_vehicle)

    def composition_metrics(primary: Mapping[float, Any], diagnostic: Mapping[float, Any],
                            taxonomy: Mapping[str, Any]) -> dict[str, Any]:
        at020 = primary[0.20]["classes"]["person"]
        at002 = primary[0.02]["classes"]["person"]
        wrong = taxonomy["person_fn_at_0_02"]["counts"]["CENTER_PRESENT_WORLD_WRONG"]
        return {
            "person_precision_020": at020["precision"], "person_recall_020": at020["recall"],
            "person_f1_020": at020["f1"], "person_recall_002": at002["recall"],
            "person_xy_mae_m_020": at020["xy_mae_m"],
            "iou50_f1_020": candidate_expected["iou50_f1_020"],
            "iou50_recall_002": candidate_expected["iou50_recall_002"],
            "iou50_conditional_within_3m_002": diagnostic[0.02]["within_3m_fraction"],
            "center_present_world_wrong_002": wrong,
            "center_present_world_wrong_reduction": candidate_expected["center_present_world_wrong_002"] - wrong,
            "vehicle_and_segmentation_bit_identical": preserved,
        }

    deploy_metrics = composition_metrics(deploy_primary, deploy_diag, deploy_taxonomy)
    gt_metrics = composition_metrics(gt_primary, gt_diag, gt_taxonomy)
    deploy_gates = composition_gates(deploy_metrics, candidate_expected, config, diagnostic_gt=False)
    gt_gates = composition_gates(gt_metrics, candidate_expected, config, diagnostic_gt=True)
    deploy_pass, gt_pass = all(deploy_gates.values()), all(gt_gates.values())

    # Cell-location and inherited-field quality on the frozen score-0.02 IoU50 cohort.
    cell_rows: list[dict[str, Any]] = []
    for pair in pair_results[0.02]["matches"]:
        stable = int(pair["prediction"]["stable_row"])
        target = pair["gt"]
        predicted_sample = candidate_samples[stable]
        gt_sample = gt_samples[(target["sample_id"], target["source_identity"])]
        dx = int(predicted_sample["cell_x"]) - int(gt_sample["cell_x"])
        dy = int(predicted_sample["cell_y"]) - int(gt_sample["cell_y"])
        pred_field_error = pair_error(numeric_world(predicted_sample, "base_"), target)
        gt_field_error = pair_error(numeric_world(gt_sample, "base_"), target)
        cell_rows.append({"candidate_stable_row": stable, "gt_stable_row": target["stable_row"],
                          "cell_dx": dx, "cell_dy": dy, "chebyshev_cell_distance": max(abs(dx), abs(dy)),
                          "predicted_cell_epoch40_world_error_m": pred_field_error,
                          "gt_cell_epoch40_world_error_m": gt_field_error})
    cell_audit = {
        "schema": "route_b_v3_1_localizer_counterfactual_cell_sampling_audit_v1",
        "created_utc": utc_now(), "pairs": len(cell_rows),
        "same_native_cell": sum(row["chebyshev_cell_distance"] == 0 for row in cell_rows),
        "adjacent_native_cell": sum(row["chebyshev_cell_distance"] == 1 for row in cell_rows),
        "farther_native_cell": sum(row["chebyshev_cell_distance"] > 1 for row in cell_rows),
        "same_native_cell_fraction": sum(row["chebyshev_cell_distance"] == 0 for row in cell_rows) / len(cell_rows),
        "within_one_cell_fraction": sum(row["chebyshev_cell_distance"] <= 1 for row in cell_rows) / len(cell_rows),
        "predicted_cell_field": summarize_errors(
            [row["predicted_cell_epoch40_world_error_m"] for row in cell_rows]
        ),
        "gt_cell_field": summarize_errors([row["gt_cell_epoch40_world_error_m"] for row in cell_rows]),
        "sampler": config["counterfactual_contract"]["dense_sampler"],
    }
    write_json_x(experiment / "CELL_SAMPLING_AUDIT.json", cell_audit)

    # Registered depth/ray dominance on original score-0.02 pairwise failures.
    rows002 = [row for row in oracle_rows if float(row["threshold"]) == 0.02]
    failures = [row for row in rows002 if row["predicted_ray_predicted_depth_world_xy_error_m"] > 3.0]
    depth_recovered = sum(row["predicted_ray_gt_depth_world_xy_error_m"] <= 3.0 for row in failures)
    ray_recovered = sum(row["gt_ray_predicted_depth_world_xy_error_m"] <= 3.0 for row in failures)
    ratio = float(config["attribution"]["dominance_ratio"])
    minimum = float(config["attribution"]["minimum_original_failure_recovery_fraction"]) * len(failures)
    if depth_recovered >= ratio * max(1, ray_recovered) and depth_recovered >= minimum:
        attribution = "LRASPP_DEPTH_ERROR_DOMINANT"
    elif ray_recovered >= ratio * max(1, depth_recovered) and ray_recovered >= minimum:
        attribution = "LRASPP_RAY_ERROR_DOMINANT"
    else:
        attribution = "LRASPP_DEPTH_AND_RAY_ERROR_MIXED"
    gt_both_max_error = max(row["gt_ray_gt_depth_world_xy_error_m"] for row in oracle_rows)
    # object_sensor_* and object_world_* are independently serialized CSV fields;
    # allow sub-millimetre decimal round-trip noise while still making any geometric
    # or calibration mismatch fail decisively.
    gt_both_sanity_tolerance_m = 1e-4
    if gt_both_max_error > gt_both_sanity_tolerance_m:
        raise RuntimeError(f"GT/GT oracle sanity failure: {gt_both_max_error}")
    oracle_summary = {
        "schema": "route_b_v3_1_localizer_counterfactual_oracle_summary_v1",
        "created_utc": utc_now(), "fixed_pair_counts": {
            "0.02": len(rows002), "0.20": sum(float(row["threshold"]) == 0.20 for row in oracle_rows),
        },
        "pairwise": {threshold: {arm: summarize_errors([
            float(row[field]) for row in oracle_rows if float(row["threshold"]) == threshold
        ]) for arm, field in oracle_error_fields.items()} for threshold in (0.02, 0.20)},
        "original_failed_pairs_002": len(failures),
        "depth_only_recovered_002": depth_recovered,
        "ray_only_recovered_002": ray_recovered,
        "both_gt_recovered_002": sum(row["gt_ray_gt_depth_world_xy_error_m"] <= 3.0 for row in failures),
        "gt_gt_max_world_xy_error_m": gt_both_max_error,
        "gt_gt_sanity_tolerance_m": gt_both_sanity_tolerance_m,
        "gt_gt_sanity_pass": gt_both_max_error <= gt_both_sanity_tolerance_m,
        "secondary_attribution": attribution,
        "end_to_end": {arm: {
            "person_precision_020": values[0.20]["classes"]["person"]["precision"],
            "person_recall_020": values[0.20]["classes"]["person"]["recall"],
            "person_f1_020": values[0.20]["classes"]["person"]["f1"],
            "person_xy_mae_m_020": values[0.20]["classes"]["person"]["xy_mae_m"],
            "person_recall_002": values[0.02]["classes"]["person"]["recall"],
            "canonical_tp_020": values[0.20]["classes"]["person"]["tp"],
            "canonical_tp_002": values[0.02]["classes"]["person"]["tp"],
            "iou50_conditional_within_3m_002": oracle_diagnostics[arm][0.02]["within_3m_fraction"],
        } for arm, values in oracle_end_to_end.items()},
    }
    write_json_x(experiment / "ORACLE_SUMMARY.json", oracle_summary)
    write_csv_x(experiment / "SLICE_METRICS.csv", slice_metrics)

    if deploy_pass:
        terminal = "LRASPP_INHERITED_LOCALIZER_RECOVERS_VISIBLE_ANCHOR_GAINS"
        recommendation = ("License exactly one hybrid qualification run using the corrected detector "
                          "and hard-cell epoch-40 localization field; no residual unless separately justified.")
    elif gt_pass:
        terminal = "LRASPP_INHERITED_LOCALIZER_SAMPLING_LIMITED"
        recommendation = ("Stop without training. The inherited field is sufficient only at the GT cell; "
                          "request review of box-centre-to-cell sampling before any further LR-ASPP work.")
    else:
        terminal = "LRASPP_INHERITED_LOCALIZER_DOES_NOT_RECOVER"
        recommendation = ("Close further LR-ASPP person-head work under the frozen low/high transport "
                          "contract and move the next person-accuracy effort to a different architecture.")
    decision = {
        "schema": "route_b_v3_1_localizer_counterfactual_decision_v1",
        "created_utc": utc_now(), "primary_terminal": terminal,
        "secondary_attribution": attribution, "deployable_gates": deploy_gates,
        "deployable_pass": deploy_pass, "gt_cell_diagnostic_gates": gt_gates,
        "gt_cell_diagnostic_pass": gt_pass, "recommendation": recommendation,
        "new_inference_counts": {"candidate": 0, "base": 1, "segmentation": 0},
        "training_runs": 0, "optimizer_steps": 0,
    }
    counterfactual_metrics = {
        "schema": "route_b_v3_1_localizer_counterfactual_metrics_v1",
        "created_utc": utc_now(), "candidate": candidate_expected,
        "base": config["reconciliation"]["base"],
        "deployable_inherited_at_candidate_cell": deploy_metrics,
        "diagnostic_inherited_at_gt_cell": gt_metrics,
        "deployable_primary": deploy_primary, "gt_cell_primary": gt_primary,
        "deployable_conditional": deploy_diag, "gt_cell_conditional": gt_diag,
        "vehicle_and_segmentation_reused_bit_identical": preserved,
    }
    write_json_x(experiment / "COUNTERFACTUAL_METRICS.json", counterfactual_metrics)
    write_json_x(experiment / "DECISION.json", decision)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "ANALYSIS_COMPLETE", terminal + "\n")
    write_json_x(experiment / "ANALYSIS_RUNTIME.json", {
        "created_utc": utc_now(), "wall_seconds": time.monotonic() - started,
        "candidate_inference_runs": 0, "base_inference_runs": 0,
        "counterfactuals_derived_offline": True,
    })
    print(json.dumps({"decision": decision, "deployable": deploy_metrics,
                      "gt_cell": gt_metrics, "oracle": oracle_summary,
                      "cell_audit": cell_audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
