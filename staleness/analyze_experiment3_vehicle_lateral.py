#!/usr/bin/env python3
"""Analyze the tagged vehicle in the parked-ego Experiment-3 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TARGET_ROLE = "scenesense_experiment3_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--tight-gate-m", type=float, default=2.0)
    parser.add_argument("--diagnostic-gate-m", type=float, default=5.0)
    parser.add_argument("--expected-centered-frames", type=int, default=60)
    parser.add_argument("--center-mean-max-m", type=float, default=1.30)
    parser.add_argument("--center-median-max-m", type=float, default=1.20)
    parser.add_argument("--center-tight-availability-min", type=float, default=0.80)
    parser.add_argument("--expected-forward-m", type=float, default=15.0)
    parser.add_argument("--expected-lateral-m", type=float, default=0.0)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_file(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"Expected one {pattern!r} below {root}, found {len(paths)}")
    return paths[0]


def number(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def profile_metrics(errors: Sequence[float], gate_m: float, opportunities: int) -> Dict[str, float]:
    matched = [error for error in errors if math.isfinite(error) and error <= float(gate_m)]
    return {
        "gate_m": float(gate_m),
        "matches": len(matched),
        "availability": len(matched) / opportunities if opportunities else 0.0,
        "mean_error_m": mean(matched),
        "median_error_m": median(matched),
        "p90_error_m": percentile(matched, 0.90),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metrics_path = one_file(run_dir, "streams/*_metrics.csv")
    predictions_path = one_file(run_dir, "streams/*_object_predictions.csv")
    ground_truth_path = one_file(run_dir, "streams/*_object_ground_truth.csv")
    manifest_path = one_file(run_dir, "manifests/*_manifest.json")
    config_path = one_file(run_dir, "manifests/*_resolved_config.json")

    metrics = read_csv(metrics_path)
    predictions = read_csv(predictions_path)
    ground_truth = read_csv(ground_truth_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    target_manifest = manifest.get("experiment3_target") or {}
    target_role = str(target_manifest.get("role_name") or TARGET_ROLE)
    target_actor_id = int(target_manifest.get("actor_id", -1))
    target_gt_rows = [
        row
        for row in ground_truth
        if str(row.get("role_name", "")) == target_role
        or int(float(row.get("actor_id", -1) or -1)) == target_actor_id
    ]
    gt_by_frame = {int(float(row["frame_id"])): row for row in target_gt_rows}
    metric_by_frame = {int(float(row["frame_id"])): row for row in metrics}

    predictions_by_frame: Dict[int, List[Dict[str, str]]] = {}
    for row in predictions:
        if str(row.get("class_name", "")).lower() != "vehicle":
            continue
        if number(row, "score", -1.0) < float(args.score_threshold):
            continue
        frame_id = int(float(row["frame_id"]))
        predictions_by_frame.setdefault(frame_id, []).append(row)

    yaw_deg = float(
        ((manifest.get("anchor") or {}).get("transform") or {}).get("rotation", {}).get("yaw", 0.0)
    )
    yaw = math.radians(yaw_deg)
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    right_x, right_y = -math.sin(yaw), math.cos(yaw)

    analysis_rows: List[Dict[str, object]] = []
    for frame_id in sorted(metric_by_frame):
        metric = metric_by_frame[frame_id]
        gt = gt_by_frame.get(frame_id)
        row: Dict[str, object] = {
            "frame_id": frame_id,
            "result_received": int(truthy(metric.get("result_received"))),
            "target_gt_present": int(gt is not None),
            "target_visible": int(truthy(gt.get("in_camera_frustum")) if gt else False),
            "target_forward_m": number(metric, "diagnostic_target_forward_m"),
            "target_lateral_m": number(metric, "diagnostic_target_lateral_m"),
            "target_radar_points": number(metric, "diagnostic_target_radar_points", 0.0),
            "target_pixel_x": number(gt, "projected_x") if gt else float("nan"),
            "candidate_vehicle_predictions": len(predictions_by_frame.get(frame_id, [])),
            "matched_prediction_score": float("nan"),
            "nearest_error_m": float("nan"),
            "forward_error_m": float("nan"),
            "lateral_error_m": float("nan"),
            "radar_support_score": float("nan"),
        }
        if gt is not None and truthy(metric.get("result_received")):
            gx, gy = number(gt, "origin_x"), number(gt, "origin_y")
            candidates = []
            for prediction in predictions_by_frame.get(frame_id, []):
                px, py = number(prediction, "world_x"), number(prediction, "world_y")
                if not all(math.isfinite(value) for value in (gx, gy, px, py)):
                    continue
                dx, dy = px - gx, py - gy
                candidates.append((math.hypot(dx, dy), dx, dy, prediction))
            if candidates:
                error, dx, dy, prediction = min(candidates, key=lambda item: item[0])
                row.update(
                    {
                        "matched_prediction_score": number(prediction, "score"),
                        "nearest_error_m": error,
                        "forward_error_m": dx * forward_x + dy * forward_y,
                        "lateral_error_m": dx * right_x + dy * right_y,
                        "radar_support_score": number(prediction, "radar_support_score"),
                    }
                )
        analysis_rows.append(row)

    opportunities = len(analysis_rows)
    received = sum(int(row["result_received"]) for row in analysis_rows)
    gt_present = sum(int(row["target_gt_present"]) for row in analysis_rows)
    visible = sum(int(row["target_visible"]) for row in analysis_rows)
    nearest_errors = finite(float(row["nearest_error_m"]) for row in analysis_rows)
    forward_errors = finite(float(row["forward_error_m"]) for row in analysis_rows)
    lateral_errors = finite(float(row["lateral_error_m"]) for row in analysis_rows)
    matched_scores = finite(float(row["matched_prediction_score"]) for row in analysis_rows)
    learned_radar_support = finite(float(row["radar_support_score"]) for row in analysis_rows)
    tight = profile_metrics(nearest_errors, float(args.tight_gate_m), opportunities)
    diagnostic = profile_metrics(nearest_errors, float(args.diagnostic_gate_m), opportunities)
    forward_values = finite(float(row["target_forward_m"]) for row in analysis_rows)
    lateral_values = finite(float(row["target_lateral_m"]) for row in analysis_rows)
    radar_counts = finite(float(row["target_radar_points"]) for row in analysis_rows)
    pixel_values = finite(float(row["target_pixel_x"]) for row in analysis_rows)
    image_width = int(config.get("camera_width", 0) or 0)
    pixel_offsets = [value - image_width / 2.0 for value in pixel_values] if image_width else []
    anchor_z = float(
        ((manifest.get("anchor") or {}).get("transform") or {}).get("location", {}).get("z", float("nan"))
    )
    target_z = float(
        (target_manifest.get("initial_transform") or {}).get("location", {}).get("z", float("nan"))
    )
    camera_world_z = float(
        (((manifest.get("camera") or {}).get("actual_world_transform") or {}).get("location") or {}).get(
            "z", float("nan")
        )
    )

    expected_forward_m = float(args.expected_forward_m)
    expected_lateral_m = float(args.expected_lateral_m)
    protocol_checks = {
        "profile_centered": str(config.get("experiment3_target_profile")) == "centered",
        "parked_ego": bool(config.get("ego_freeze")) and str(config.get("sensor_platform")) == "ego_vehicle",
        "spawn_80": int(config.get("ego_spawn_index", -1)) == 80,
        "settle_30_ticks": int(config.get("experiment3_settle_ticks", -1)) == 30,
        "training_camera_world_height": math.isfinite(camera_world_z)
        and abs(camera_world_z - 1.57) <= 0.15,
        "same_road_height": math.isfinite(anchor_z)
        and math.isfinite(target_z)
        and abs(anchor_z - target_z) <= 0.15,
        "expected_forward_distance": abs(
            float(config.get("experiment3_target_forward_m", 0.0)) - expected_forward_m
        )
        <= 1e-6,
        "expected_lateral_offset": abs(
            float(config.get("experiment3_target_lateral_m", 99.0)) - expected_lateral_m
        )
        <= 1e-6,
        "loopback": str(config.get("role")) == "loopback",
        "uint8": str(config.get("quantization_mode")) == "per_channel_uint8",
        "zlib": str(config.get("entropy_coder")) == "zlib",
        "roi0": abs(float(config.get("roi_threshold", 99.0))) <= 1e-9,
        "noae_checkpoint": str(config.get("fusion_checkpoint", "")).endswith(
            "ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
        ),
        "score_decode_floor_005": abs(float(config.get("object_score_threshold", 99.0)) - 0.05)
        <= 1e-9,
        "nms_radius_2": int(config.get("object_nms_radius_px", -1)) == 2,
        "topk_120": int(config.get("topk_objects", -1)) == 120,
        "camera_1280x720_fov120": int(config.get("camera_width", 0)) == 1280
        and int(config.get("camera_height", 0)) == 720
        and abs(float(config.get("camera_fov", 0.0)) - 120.0) <= 1e-6,
        "model_768x432": int(config.get("model_input_width", 0)) == 768
        and int(config.get("model_input_height", 0)) == 432,
        "camera_mount_training": abs(float(config.get("ego_camera_x", 0.0)) - 1.8) <= 1e-6
        and abs(float(config.get("ego_camera_z", 0.0)) - 1.55) <= 1e-6
        and abs(float(config.get("ego_camera_pitch", 0.0)) + 4.0) <= 1e-6,
        "radar_mount_training": abs(float(config.get("ego_radar_x", 0.0)) - 2.0) <= 1e-6
        and abs(float(config.get("ego_radar_z", 0.0)) - 1.0) <= 1e-6,
        "no_background_actors": int(config.get("npc_vehicles", -1)) == 0
        and int(config.get("npc_pedestrians", -1)) == 0,
        "pps_200k": int(config.get("radar_points_per_second", 0)) == 200000,
        "radar_hfov_120": abs(float(config.get("radar_hfov", 0.0)) - 120.0) <= 1e-6,
        "raster_radius_4": int(config.get("radar_raster_radius_px", 0)) == 4,
        "temporal_window_2": int(config.get("radar_temporal_window_frames", 0)) == 2,
        "exact_frame_count": opportunities == int(args.expected_centered_frames),
        "complete_delivery": received == opportunities,
        "target_gt_every_frame": gt_present == opportunities,
        "target_visible_every_frame": visible == opportunities,
        "placement_forward": bool(forward_values)
        and max(abs(value - expected_forward_m) for value in forward_values) <= 0.25,
        "placement_lateral": bool(lateral_values)
        and max(abs(value - expected_lateral_m) for value in lateral_values) <= 0.25,
    }
    center_accuracy_checks = {
        "tight_availability": tight["availability"] >= float(args.center_tight_availability_min),
        "conditional_mean": math.isfinite(tight["mean_error_m"])
        and tight["mean_error_m"] <= float(args.center_mean_max_m),
        "conditional_median": math.isfinite(tight["median_error_m"])
        and tight["median_error_m"] <= float(args.center_median_max_m),
    }
    protocol_pass = all(protocol_checks.values())
    accuracy_pass = all(center_accuracy_checks.values())
    gate_pass = protocol_pass and accuracy_pass

    summary = {
        "run_dir": str(run_dir),
        "target_role_name": target_role,
        "target_actor_id": target_actor_id,
        "score_threshold": float(args.score_threshold),
        "expected": {
            "forward_m": expected_forward_m,
            "lateral_m": expected_lateral_m,
            "center_mean_max_m": float(args.center_mean_max_m),
            "center_median_max_m": float(args.center_median_max_m),
            "tight_availability_min": float(args.center_tight_availability_min),
        },
        "opportunities": opportunities,
        "result_received": received,
        "target_gt_present": gt_present,
        "target_visible": visible,
        "frames_with_scored_vehicle_prediction": len(nearest_errors),
        "tight": tight,
        "diagnostic": diagnostic,
        "nearest_prediction_all": {
            "count": len(nearest_errors),
            "mean_error_m": mean(nearest_errors),
            "median_error_m": median(nearest_errors),
            "p90_error_m": percentile(nearest_errors, 0.90),
        },
        "error_vector_all": {
            "forward_mean_m": mean(forward_errors),
            "forward_median_m": median(forward_errors),
            "lateral_mean_m": mean(lateral_errors),
            "lateral_median_m": median(lateral_errors),
            "matched_score_mean": mean(matched_scores),
            "matched_score_median": median(matched_scores),
            "learned_radar_support_mean": mean(learned_radar_support),
        },
        "placement": {
            "forward_mean_m": mean(forward_values),
            "forward_min_m": min(forward_values) if forward_values else float("nan"),
            "forward_max_m": max(forward_values) if forward_values else float("nan"),
            "lateral_mean_m": mean(lateral_values),
            "lateral_min_m": min(lateral_values) if lateral_values else float("nan"),
            "lateral_max_m": max(lateral_values) if lateral_values else float("nan"),
            "pixel_offset_mean_px": mean(pixel_offsets),
            "pixel_offset_min_px": min(pixel_offsets) if pixel_offsets else float("nan"),
            "pixel_offset_max_px": max(pixel_offsets) if pixel_offsets else float("nan"),
            "ego_origin_z_m": anchor_z,
            "target_origin_z_m": target_z,
            "camera_world_z_m": camera_world_z,
        },
        "radar": {
            "target_points_mean": mean(radar_counts),
            "target_points_median": median(radar_counts),
            "target_support_frame_rate": (
                sum(value > 0 for value in radar_counts) / opportunities if opportunities else 0.0
            ),
        },
        "protocol_checks": protocol_checks,
        "center_accuracy_checks": center_accuracy_checks,
        "protocol_pass": protocol_pass,
        "accuracy_pass": accuracy_pass,
        "center_gate_pass": gate_pass,
    }

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame_path = analysis_dir / "target_frame_analysis.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(analysis_rows[0]) if analysis_rows else ["frame_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    result_path = run_dir / "CENTERED_RESULT.md"
    result_path.write_text(
        "\n".join(
            [
                "# Experiment 3 — parked-ego centered validation",
                "",
                f"**Stop/go result: {'PASS — lateral run may proceed' if gate_pass else 'FAIL — do not run laterally'}.**",
                "",
                f"- Opportunities / loopback results / visible target: {opportunities} / {received} / {visible}",
                f"- Score threshold: {float(args.score_threshold):.2f}",
                (
                    f"- ≤{float(args.tight_gate_m):.0f} m matches: {int(tight['matches'])}/{opportunities} "
                    f"({100.0 * tight['availability']:.1f}%); conditional mean/median/p90 "
                    f"{fmt(tight['mean_error_m'])}/{fmt(tight['median_error_m'])}/{fmt(tight['p90_error_m'])} m"
                ),
                (
                    f"- ≤{float(args.diagnostic_gate_m):.0f} m matches: {int(diagnostic['matches'])}/{opportunities} "
                    f"({100.0 * diagnostic['availability']:.1f}%); conditional mean/median/p90 "
                    f"{fmt(diagnostic['mean_error_m'])}/{fmt(diagnostic['median_error_m'])}/{fmt(diagnostic['p90_error_m'])} m"
                ),
                (
                    "- Placement mean: forward "
                    f"{fmt(summary['placement']['forward_mean_m'])} m, lateral "
                    f"{fmt(summary['placement']['lateral_mean_m'])} m, pixel offset "
                    f"{fmt(summary['placement']['pixel_offset_mean_px'], 2)} px"
                ),
                (
                    "- Raw radar support: mean/median "
                    f"{fmt(summary['radar']['target_points_mean'], 1)}/"
                    f"{fmt(summary['radar']['target_points_median'], 1)} points; supported frames "
                    f"{100.0 * summary['radar']['target_support_frame_rate']:.1f}%"
                ),
                (
                    "- Mean signed error (ego frame): forward "
                    f"{fmt(summary['error_vector_all']['forward_mean_m'])} m, lateral "
                    f"{fmt(summary['error_vector_all']['lateral_mean_m'])} m; mean target score "
                    f"{fmt(summary['error_vector_all']['matched_score_mean'])}"
                ),
                f"- Protocol checks: {'PASS' if protocol_pass else 'FAIL'}",
                f"- Accuracy checks: {'PASS' if accuracy_pass else 'FAIL'}",
                "",
                "Detailed machine-readable output: `analysis/summary.json` and `analysis/target_frame_analysis.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(result_path.read_text(encoding="utf-8"))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
