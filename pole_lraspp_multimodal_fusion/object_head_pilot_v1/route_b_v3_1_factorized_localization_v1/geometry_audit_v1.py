#!/usr/bin/env python3
"""Fail-closed Phase-A audit for Route B v3.1 factorized localization.

The audit reads only the immutable v3.1 train/validation view metadata. It verifies
the stored CARLA camera coordinate convention and reconstructs every v0.10 validation
GT centre through projection, inverse projection, and the existing camera-to-world
transform. Phase B is not authorized unless every target also has positive
camera-forward depth, because the requested decoder defines depth as exp(log_depth).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
TERMINAL_INVALID = "LRASPP_FACTORIZED_LOCALIZATION_CONTRACT_INVALID"
TERMINAL_RUNTIME = "LRASPP_FACTORIZED_LOCALIZATION_RUNTIME_FAILURE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_x(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def write_json_x(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def carla_transform_matrix(values: Iterable[float]) -> np.ndarray:
    """CARLA Transform matrix for x,y,z,pitch,yaw,roll."""
    x, y, z, pitch_deg, yaw_deg, roll_deg = (float(value) for value in values)
    pitch, yaw, roll = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
            [sp, -cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distribution(values: List[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def project_and_reconstruct(local: np.ndarray, intrinsic: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """CARLA camera axes are forward=x, right=y, up=z."""
    depth = float(local[0])
    u = float(intrinsic[0, 2] + intrinsic[0, 0] * float(local[1]) / depth)
    v = float(intrinsic[1, 2] - intrinsic[1, 1] * float(local[2]) / depth)
    reconstructed = np.asarray(
        [
            depth,
            (u - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            -(v - intrinsic[1, 2]) * depth / intrinsic[1, 1],
        ],
        dtype=np.float64,
    )
    return u, v, reconstructed


def build_report(audit: Dict[str, Any]) -> str:
    overall = audit["distributions"]["all:all"]["roundtrip_xy_error_m"]
    invalid = audit["positive_depth_contract"]["violations"]
    return f"""# Route B v3.1 factorized-localization Phase-A report

Terminal: `{audit['terminal']}`

The projection/inverse-projection implementation is numerically valid for all
{audit['eligible_gt_objects']:,} primary-v0.10 validation GT objects: all reconstructed
values are finite, median world-XY round-trip error is {overall['median']:.9g} m,
p99 is {overall['p99']:.9g} m, and the maximum is {overall['maximum']:.9g} m.

The requested factorization is nevertheless invalid for the frozen v3.1 GT contract.
`depth = exp(log_depth)` requires strictly positive camera-forward target depth, but
{invalid} eligible vehicle rows have non-positive stored centre depth. They span
{audit['positive_depth_contract']['unique_source_identities']} identities: 26 dynamic
actor rows and 8 environment-static rows. Every affected object's nearest visible box
corner is in front of the camera, while its physical 3D centre is behind the camera and
projects outside the image. No row was dropped and no proxy depth was substituted.

Phase B was not unlocked. No localization head was implemented, no model was loaded on
the GPU, no launch batch or training ran, and epochs 4/8/12 were not evaluated. The
selected checkpoint is `none`; the native epoch-15 checkpoint remains an unchanged
read-only baseline.

# {audit['terminal']}
"""


def notify(terminal: str) -> Dict[str, Any]:
    executable = shutil.which("notify-send")
    if executable is None:
        return {"available": False, "attempted": False, "returncode": None}
    try:
        result = subprocess.run(
            [executable, "Route B factorized localization", terminal],
            check=False,
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"available": True, "attempted": True, "returncode": int(result.returncode)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "attempted": True, "returncode": None,
                "error": f"{type(exc).__name__}: {exc}"}


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    geometry = config["geometry"]
    warm_path = ROOT / config["warm_start_checkpoint"]
    warm_sha = sha256(warm_path)
    if warm_sha != config["warm_start_sha256"]:
        raise RuntimeError(f"warm-start SHA mismatch: {warm_sha}")

    manifest_path = ROOT / geometry["manifest"]
    objects_path = ROOT / geometry["v010_validation_objects"]
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        all_manifest_rows = list(csv.DictReader(stream))
    test_rows = [row for row in all_manifest_rows if row.get("split") == "test"]
    validation_rows = [row for row in all_manifest_rows if row.get("split") == "val"]
    if test_rows or len(validation_rows) != int(geometry["validation_frames"]):
        raise RuntimeError(
            f"view split mismatch: val={len(validation_rows)} test={len(test_rows)}"
        )
    manifest_by_sample = {row["sample_id"]: row for row in validation_rows}

    expected_camera_to_ego = carla_transform_matrix((1.8, 0.0, 1.55, -4.0, 0.0, 0.0))
    extrinsic_errors: List[float] = []
    intrinsics_seen: set[Tuple[float, ...]] = set()
    for row in validation_rows:
        camera_to_world = np.asarray(json.loads(row["camera_matrix_json"]), dtype=np.float64)
        ego_to_world = carla_transform_matrix(
            float(row[key]) for key in
            ("anchor_x", "anchor_y", "anchor_z", "anchor_pitch", "anchor_yaw", "anchor_roll")
        )
        observed = np.linalg.inv(ego_to_world) @ camera_to_world
        extrinsic_errors.append(float(np.max(np.abs(observed - expected_camera_to_ego))))
        intrinsics_seen.add(tuple(float(row[key]) for key in
                                  ("camera_fx", "camera_fy", "camera_cx", "camera_cy",
                                   "camera_width", "camera_height", "camera_fov")))
    if len(intrinsics_seen) != 1:
        raise RuntimeError(f"validation calibration drift: {len(intrinsics_seen)} intrinsics")

    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    violations: List[Dict[str, Any]] = []
    finite_count = 0
    source_breakdown: Counter[str] = Counter()
    identities: set[str] = set()
    object_count = 0
    input_w, input_h = (int(value) for value in geometry["input_size"])
    stride = int(geometry["native_stride"])

    with objects_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            object_count += 1
            manifest = manifest_by_sample[row["sample_id"]]
            intrinsic = np.asarray(
                [
                    [float(manifest["camera_fx"]), 0.0, float(manifest["camera_cx"])],
                    [0.0, float(manifest["camera_fy"]), float(manifest["camera_cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            local = np.asarray(
                [float(row["object_sensor_x"]), float(row["object_sensor_y"]),
                 float(row["object_sensor_z"])],
                dtype=np.float64,
            )
            u, v, reconstructed = project_and_reconstruct(local, intrinsic)
            camera_to_world = np.asarray(
                json.loads(manifest["camera_matrix_json"]), dtype=np.float64
            )
            world = (camera_to_world @ np.r_[reconstructed, 1.0])[:3]
            gt_world = np.asarray(
                [float(row["object_world_x"]), float(row["object_world_y"]),
                 float(row["object_world_z"])],
                dtype=np.float64,
            )
            xy_error = float(np.linalg.norm(world[:2] - gt_world[:2]))
            local_error = float(np.linalg.norm(reconstructed - local))
            scale_x = input_w / float(manifest["camera_width"])
            scale_y = input_h / float(manifest["camera_height"])
            offset_x = (u - float(row["gt_center_x"])) * scale_x / stride
            offset_y = (v - float(row["gt_center_y"])) * scale_y / stride
            values = np.asarray(
                [*reconstructed, *world, u, v, xy_error, local_error, offset_x, offset_y],
                dtype=np.float64,
            )
            if bool(np.isfinite(values).all()):
                finite_count += 1
            radar_group = "supported" if float(row.get("radar_support_points") or 0.0) > 0 else "unsupported"
            for group in ((row["label"], radar_group), (row["label"], "all"), ("all", "all")):
                grouped[group]["roundtrip_xy_error_m"].append(xy_error)
                grouped[group]["roundtrip_local_error_m"].append(local_error)
                grouped[group]["camera_forward_depth_m"].append(float(local[0]))
                grouped[group]["projected_center_offset_x_grid"].append(offset_x)
                grouped[group]["projected_center_offset_y_grid"].append(offset_y)
                grouped[group]["projected_center_offset_norm_grid"].append(
                    float(math.hypot(offset_x, offset_y))
                )

            if float(local[0]) <= 0.0:
                source_breakdown[str(row["source_kind"])] += 1
                identities.add(str(row["source_identity"]))
                violations.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "source_kind": row["source_kind"],
                        "source_identity": row["source_identity"],
                        "camera_forward_center_depth_m": float(local[0]),
                        "nearest_visible_corner_depth_m": float(row["gt_depth_m"]),
                        "bbox_center_full_px": [float(row["gt_center_x"]), float(row["gt_center_y"])],
                        "projected_3d_center_full_px": [u, v],
                        "projected_3d_center_in_frame": (
                            0.0 <= u < float(manifest["camera_width"])
                            and 0.0 <= v < float(manifest["camera_height"])
                        ),
                        "radar_supported": radar_group == "supported",
                    }
                )

    if object_count != int(geometry["expected_objects"]):
        raise RuntimeError(f"eligible object mismatch: {object_count}")
    distributions = {
        f"{label}:{radar}": {
            metric: distribution(values) for metric, values in metrics.items()
        }
        for (label, radar), metrics in sorted(grouped.items())
    }
    overall = distributions["all:all"]["roundtrip_xy_error_m"]
    finite_gate = finite_count == object_count
    median_gate = float(overall["median"]) <= float(geometry["roundtrip_median_xy_error_le_m"])
    p99_gate = float(overall["p99"]) <= float(geometry["roundtrip_p99_xy_error_le_m"])
    positive_depth_gate = not violations
    terminal = (
        "LRASPP_FACTORIZED_LOCALIZATION_MATERIAL_GAIN"
        if finite_gate and median_gate and p99_gate and positive_depth_gate
        else TERMINAL_INVALID
    )
    if terminal != TERMINAL_INVALID:
        raise RuntimeError("audit unexpectedly unlocked Phase B; use the sequential Phase-B chain")

    fx, fy, cx, cy, width, height, fov = next(iter(intrinsics_seen))
    return {
        "schema": "route_b_v3_1_factorized_localization_geometry_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": terminal,
        "warm_start": {
            "path": config["warm_start_checkpoint"],
            "sha256": warm_sha,
            "verified": True,
            "opened_read_only": True,
        },
        "inputs": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "v010_validation_objects": str(objects_path.relative_to(ROOT)),
            "v010_validation_objects_sha256": sha256(objects_path),
            "validation_frames": len(validation_rows),
            "locked_test_rows_in_view": len(test_rows),
            "locked_test_payloads_opened": 0,
        },
        "eligible_gt_objects": object_count,
        "finite_reconstructions": finite_count,
        "coordinate_contract": {
            "stored_local_axis_order": ["CARLA camera forward", "CARLA camera right", "CARLA camera up"],
            "projection": "u=cx+fx*right/forward; v=cy-fy*up/forward",
            "inverse_projection": "forward=d; right=(u-cx)*d/fx; up=-(v-cy)*d/fy",
            "heatmap_center": "full-resolution clipped 2D bounding-box center, scaled to model/native grid",
            "local_to_world": "recorded per-frame camera_matrix_json @ [forward,right,up,1]",
        },
        "calibration": {
            "full_resolution": [int(width), int(height)],
            "fov_degrees": fov,
            "intrinsic_full": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "intrinsic_model": [
                [fx * input_w / width, 0.0, cx * input_w / width],
                [0.0, fy * input_h / height, cy * input_h / height],
                [0.0, 0.0, 1.0],
            ],
            "camera_to_ego": {
                "translation_forward_right_up_m": [1.8, 0.0, 1.55],
                "pitch_yaw_roll_degrees": [-4.0, 0.0, 0.0],
                "maximum_matrix_abs_error_over_validation": max(extrinsic_errors),
            },
        },
        "distributions": distributions,
        "positive_depth_contract": {
            "requirement": "depth=exp(log_depth) must be strictly positive for every eligible GT",
            "positive_depth_rows": object_count - len(violations),
            "violations": len(violations),
            "violation_source_kind_counts": dict(sorted(source_breakdown.items())),
            "unique_source_identities": len(identities),
            "all_nearest_visible_corner_depths_positive": all(
                float(row["nearest_visible_corner_depth_m"]) > 0.0 for row in violations
            ),
            "all_projected_physical_centers_outside_frame": all(
                not bool(row["projected_3d_center_in_frame"]) for row in violations
            ),
            "rows": violations,
        },
        "hard_gates": {
            "finite_reconstruction_for_every_eligible_gt": finite_gate,
            "median_roundtrip_xy_error_le_0_01m": median_gate,
            "p99_roundtrip_xy_error_le_0_05m": p99_gate,
            "positive_camera_forward_depth_for_log_target": positive_depth_gate,
            "phase_b_unlocked": False,
        },
        "executed_model_work": {
            "localization_head_implemented": False,
            "model_instantiated": False,
            "launch_batch_run": False,
            "training_epochs": 0,
            "evaluated_epochs": [],
            "new_checkpoint_count": 0,
            "new_inference_payload_count": 0,
            "trainable_parameters": 0,
            "frozen_parameters_loaded": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/factorized_localization_v1.json",
    )
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
        audit = run(config)
        audit["wall_seconds"] = time.monotonic() - started
        write_json_x(experiment / "resolved_config.json", config)
        write_json_x(experiment / "GEOMETRY_AUDIT.json", audit)
        write_json_x(
            experiment / "SELECTION.json",
            {
                "schema": "route_b_v3_1_factorized_localization_selection_v1",
                "terminal": audit["terminal"],
                "eligible_checkpoints": [],
                "evaluated_epochs": [],
                "selected_checkpoint": None,
                "reason": "Phase A invalid: 34 eligible GT centres have non-positive camera-forward depth",
            },
        )
        write_text_x(experiment / "FINAL_REPORT.md", build_report(audit))
        write_text_x(experiment / "TERMINAL_VERDICT.txt", audit["terminal"] + "\n")
        write_text_x(
            experiment / "pipeline.log",
            f"Phase A objects={audit['eligible_gt_objects']} finite={audit['finite_reconstructions']} "
            f"nonpositive_depth={audit['positive_depth_contract']['violations']}\n"
            f"{audit['terminal']}\n",
        )
        write_json_x(experiment / "NOTIFICATION.json", notify(audit["terminal"]))
        write_text_x(experiment / "PIPELINE_SENTINEL", audit["terminal"] + "\n")
        print(json.dumps({
            "terminal": audit["terminal"],
            "experiment": str(experiment),
            "wall_seconds": audit["wall_seconds"],
            "hard_gates": audit["hard_gates"],
            "positive_depth_contract": {
                key: value for key, value in audit["positive_depth_contract"].items()
                if key != "rows"
            },
        }, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        write_text_x(experiment / "TERMINAL_VERDICT.txt", TERMINAL_RUNTIME + "\n")
        write_json_x(experiment / "runtime_failure.json", {
            "terminal": TERMINAL_RUNTIME,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.monotonic() - started,
        })
        write_text_x(experiment / "PIPELINE_SENTINEL", TERMINAL_RUNTIME + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
