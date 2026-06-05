#!/usr/bin/env python3
"""Validate a saved SceneSense fusion training dataset folder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - validator still catches file/schema issues.
    cv2 = None  # type: ignore


REQUIRED_MANIFEST_FIELDS = (
    "experiment_id",
    "sample_id",
    "split",
    "rgb_path",
    "mask_path",
    "radar_tensor_path",
    "radar_points_path",
    "frame_id",
    "timestamp",
    "camera_width",
    "camera_height",
    "camera_matrix_json",
    "camera_inverse_matrix_json",
    "radar_matrix_json",
    "radar_inverse_matrix_json",
    "radar_to_camera_matrix_json",
    "vehicle_pixels",
    "person_pixels",
)

REQUIRED_OBJECT_FIELDS = (
    "sample_id",
    "label",
    "gt_actor_id",
    "gt_source",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_w",
    "gt_bbox_h",
    "gt_bbox_area_px",
    "gt_center_x",
    "gt_center_y",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "object_sensor_x",
    "object_sensor_y",
    "object_sensor_z",
    "object_yaw_deg",
    "radar_support_points",
)

RADAR_POINT_KEYS = (
    "world_xyz",
    "camera_xyz",
    "velocity_mps",
    "u",
    "v",
    "camera_depth_m",
    "stationary_age_s",
    "valid_projection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manifest/object/files for a fusion training dataset."
    )
    parser.add_argument("dataset_dir", help="Dataset folder containing manifest.csv.")
    parser.add_argument("--max-samples", type=int, default=20, help="Number of manifest rows to inspect deeply.")
    parser.add_argument("--write-summary", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def require_fields(rows: Sequence[Dict[str, str]], fields: Iterable[str], *, table: str, errors: List[str]) -> None:
    if not rows:
        errors.append(f"{table} has no rows.")
        return
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        errors.append(f"{table} missing fields: {', '.join(missing)}")


def path_from_row(dataset_dir: Path, row: Dict[str, str], field: str) -> Path:
    raw = str(row.get(field, "")).strip()
    path = Path(raw)
    return path if path.is_absolute() else dataset_dir / path


def validate_json_matrix(raw: str, *, field: str, errors: List[str]) -> None:
    try:
        matrix = np.asarray(json.loads(raw), dtype=np.float64)
    except Exception as exc:
        errors.append(f"{field} is not valid JSON matrix: {exc}")
        return
    if matrix.shape != (4, 4):
        errors.append(f"{field} shape is {matrix.shape}, expected (4, 4)")


def validate_sample_files(
    *,
    dataset_dir: Path,
    rows: Sequence[Dict[str, str]],
    errors: List[str],
    warnings: List[str],
) -> Dict[str, object]:
    inspected = 0
    radar_shapes = Counter()
    mask_shapes = Counter()
    rgb_shapes = Counter()
    mask_classes = Counter()
    for row in rows:
        inspected += 1
        sample_id = row.get("sample_id", f"row_{inspected}")
        for field in ("rgb_path", "mask_path", "radar_tensor_path", "radar_points_path"):
            path = path_from_row(dataset_dir, row, field)
            if not path.exists():
                errors.append(f"{sample_id}: missing {field}: {path}")
        for field in ("camera_matrix_json", "camera_inverse_matrix_json", "radar_matrix_json", "radar_inverse_matrix_json", "radar_to_camera_matrix_json"):
            validate_json_matrix(row.get(field, ""), field=f"{sample_id}.{field}", errors=errors)

        rgb_path = path_from_row(dataset_dir, row, "rgb_path")
        mask_path = path_from_row(dataset_dir, row, "mask_path")
        radar_tensor_path = path_from_row(dataset_dir, row, "radar_tensor_path")
        radar_points_path = path_from_row(dataset_dir, row, "radar_points_path")

        if cv2 is not None and rgb_path.exists():
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"{sample_id}: cv2 could not read RGB image {rgb_path}")
            else:
                rgb_shapes[tuple(int(v) for v in image.shape)] += 1
        elif cv2 is None:
            warnings.append("cv2 unavailable; skipped image/mask decode checks.")

        if cv2 is not None and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                errors.append(f"{sample_id}: cv2 could not read mask {mask_path}")
            else:
                mask_shapes[tuple(int(v) for v in mask.shape)] += 1
                for value in np.unique(mask):
                    mask_classes[int(value)] += 1
                unknown = [int(value) for value in np.unique(mask) if int(value) not in (0, 1, 2)]
                if unknown:
                    errors.append(f"{sample_id}: mask has unknown classes {unknown}")

        if radar_tensor_path.exists():
            try:
                tensor = np.load(radar_tensor_path)
                radar_shapes[tuple(int(v) for v in tensor.shape)] += 1
                if tensor.ndim != 3 or int(tensor.shape[0]) != 4:
                    errors.append(f"{sample_id}: radar tensor shape {tensor.shape}, expected (4,H,W)")
                if not np.isfinite(tensor).all():
                    errors.append(f"{sample_id}: radar tensor contains non-finite values")
            except Exception as exc:
                errors.append(f"{sample_id}: could not load radar tensor: {exc}")

        if radar_points_path.exists():
            try:
                with np.load(radar_points_path) as points:
                    missing = [key for key in RADAR_POINT_KEYS if key not in points.files]
                    if missing:
                        errors.append(f"{sample_id}: radar points missing keys {missing}")
            except Exception as exc:
                errors.append(f"{sample_id}: could not load radar points: {exc}")

    return {
        "inspected_samples": inspected,
        "rgb_shapes": {str(key): value for key, value in rgb_shapes.items()},
        "mask_shapes": {str(key): value for key, value in mask_shapes.items()},
        "radar_tensor_shapes": {str(key): value for key, value in radar_shapes.items()},
        "mask_classes_seen": dict(mask_classes),
    }


def validate_objects(
    manifest_rows: Sequence[Dict[str, str]],
    object_rows: Sequence[Dict[str, str]],
    errors: List[str],
    warnings: List[str],
) -> Dict[str, object]:
    manifest_ids = {row.get("sample_id", "") for row in manifest_rows}
    labels = Counter(row.get("label", "") for row in object_rows)
    source_counts = Counter(row.get("gt_source", "") for row in object_rows)
    by_sample = defaultdict(int)
    for row in object_rows:
        sample_id = row.get("sample_id", "")
        by_sample[sample_id] += 1
        if sample_id not in manifest_ids:
            errors.append(f"object row references unknown sample_id={sample_id}")
        for field in ("gt_bbox_w", "gt_bbox_h", "gt_bbox_area_px"):
            value = to_float(row.get(field, ""))
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"object row {sample_id}/{row.get('gt_actor_id', '')}: invalid {field}={row.get(field, '')}")
        for field in ("object_world_x", "object_world_y", "object_world_z", "object_sensor_x", "object_sensor_y", "object_sensor_z"):
            value = to_float(row.get(field, ""))
            if not math.isfinite(value):
                errors.append(f"object row {sample_id}/{row.get('gt_actor_id', '')}: invalid {field}")
    samples_without_objects = sorted(sample_id for sample_id in manifest_ids if by_sample[sample_id] == 0)
    if samples_without_objects:
        warnings.append(f"{len(samples_without_objects)} samples have no projected object rows.")
    return {
        "object_rows": len(object_rows),
        "label_counts": dict(labels),
        "gt_source_counts": dict(source_counts),
        "samples_without_objects": len(samples_without_objects),
        "objects_per_sample_mean": (
            sum(by_sample.values()) / len(manifest_ids) if manifest_ids else float("nan")
        ),
    }


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    errors: List[str] = []
    warnings: List[str] = []
    manifest_path = dataset_dir / "manifest.csv"
    object_path = dataset_dir / "object_boxes.csv"
    metadata_path = dataset_dir / "metadata.json"

    manifest_rows = read_csv(manifest_path)
    object_rows = read_csv(object_path) if object_path.exists() else []
    require_fields(manifest_rows, REQUIRED_MANIFEST_FIELDS, table="manifest.csv", errors=errors)
    require_fields(object_rows, REQUIRED_OBJECT_FIELDS, table="object_boxes.csv", errors=errors)
    if not metadata_path.exists():
        warnings.append("metadata.json is missing.")

    deep_rows = manifest_rows[: max(0, int(args.max_samples))]
    file_summary = validate_sample_files(
        dataset_dir=dataset_dir,
        rows=deep_rows,
        errors=errors,
        warnings=warnings,
    )
    object_summary = validate_objects(manifest_rows, object_rows, errors, warnings)
    split_counts = Counter(row.get("split", "") for row in manifest_rows)
    summary = {
        "dataset_dir": str(dataset_dir),
        "manifest_rows": len(manifest_rows),
        "split_counts": dict(split_counts),
        **file_summary,
        **object_summary,
        "warnings": warnings,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    if bool(args.write_summary):
        out_path = dataset_dir / "validation_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
