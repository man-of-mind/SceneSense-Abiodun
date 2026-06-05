#!/usr/bin/env python3
"""Dry-run SceneSense fusion training target construction.

This validates the step after file/schema validation: can manifest/object rows
from a saved dataset be converted into segmentation and object-head training
targets without starting a training job?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - target builder can still run.
    cv2 = None  # type: ignore


ABIODUN_DIR = Path(__file__).resolve().parents[1]
FUSION_PACKAGE_ROOT = ABIODUN_DIR / "pole_lraspp_multimodal_fusion"
if str(FUSION_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSION_PACKAGE_ROOT))

from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    build_object_targets,
    load_object_boxes,
    valid_vehicle_objects,
)


RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run fusion training target construction from manifest.csv and "
            "object_boxes.csv."
        )
    )
    parser.add_argument("dataset_dir", help="Dataset folder containing manifest.csv.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum manifest rows to inspect. Use <=0 for all rows.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Manifest split to inspect.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=0,
        help="Fusion model input width. Defaults to metadata model_input_size[0] or 768.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=0,
        help="Fusion model input height. Defaults to metadata model_input_size[1] or 432.",
    )
    parser.add_argument(
        "--min-gt-area-px",
        type=float,
        default=12.0,
        help="Minimum projected vehicle bbox area used by valid_vehicle_objects().",
    )
    parser.add_argument("--heatmap-radius-px", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument(
        "--require-positive-vehicle-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no inspected sample produces a positive vehicle object target.",
    )
    parser.add_argument("--write-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--summary-name",
        default="target_dry_run_summary.json",
        help="Summary JSON filename written inside the dataset folder.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def path_from_row(dataset_dir: Path, row: Dict[str, str], field: str) -> Path:
    raw = str(row.get(field, "")).strip()
    path = Path(raw)
    return path if path.is_absolute() else dataset_dir / path


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def finite_tensor(name: str, tensor: torch.Tensor, errors: List[str], sample_id: str) -> None:
    if not torch.isfinite(tensor).all():
        errors.append(f"{sample_id}: target tensor {name} contains non-finite values")


def shape_tuple(value: object) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    arr = np.asarray(value)
    return tuple(int(dim) for dim in arr.shape)


def load_metadata_input_size(dataset_dir: Path) -> Optional[Tuple[int, int]]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("model_input_size")
    if not isinstance(value, list) or len(value) != 2:
        return None
    width = to_int(value[0], 0)
    height = to_int(value[1], 0)
    if width <= 0 or height <= 0:
        return None
    return width, height


def resolve_input_size(args: argparse.Namespace, dataset_dir: Path) -> Tuple[int, int]:
    metadata_size = load_metadata_input_size(dataset_dir)
    width = int(args.input_width) if int(args.input_width) > 0 else 0
    height = int(args.input_height) if int(args.input_height) > 0 else 0
    if width > 0 and height > 0:
        return width, height
    if metadata_size is not None:
        return metadata_size
    return 768, 432


def select_rows(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    selected = [
        row for row in rows if args.split == "all" or str(row.get("split", "")) == str(args.split)
    ]
    if int(args.max_samples) > 0:
        return selected[: int(args.max_samples)]
    return list(selected)


def resize_radar_to_input(radar: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    if radar.ndim != 3:
        raise ValueError(f"radar tensor has shape {radar.shape}, expected (4,H,W)")
    if int(radar.shape[0]) != 4:
        raise ValueError(f"radar tensor has {radar.shape[0]} channels, expected 4")
    if tuple(radar.shape[1:]) == (input_height, input_width):
        return radar.astype(np.float32, copy=False)
    if cv2 is None:
        raise RuntimeError("cv2 unavailable; cannot resize radar tensor")
    channels = []
    for idx, channel in enumerate(radar):
        interp = cv2.INTER_NEAREST if idx == 0 else cv2.INTER_LINEAR
        channels.append(cv2.resize(channel, (input_width, input_height), interpolation=interp))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def build_input_tensor(
    *,
    dataset_dir: Path,
    row: Dict[str, str],
    input_size: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if cv2 is None:
        return None, None
    input_width, input_height = int(input_size[0]), int(input_size[1])
    rgb_path = path_from_row(dataset_dir, row, "rgb_path")
    radar_path = path_from_row(dataset_dir, row, "radar_tensor_path")
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"could not read RGB image {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if tuple(rgb.shape[:2]) != (input_height, input_width):
        rgb = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    rgb_chw = np.ascontiguousarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
    rgb_chw = (rgb_chw - RGB_MEAN) / RGB_STD
    radar = resize_radar_to_input(np.load(radar_path), input_size)
    feature = np.concatenate([rgb_chw, radar], axis=0).astype(np.float32, copy=False)

    mask_path = path_from_row(dataset_dir, row, "mask_path")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"could not read mask {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if tuple(mask.shape[:2]) != (input_height, input_width):
        mask = cv2.resize(mask, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return feature, mask.astype(np.int64, copy=False)


def counter_to_json(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def validate_target_shapes(
    *,
    targets: Dict[str, torch.Tensor],
    input_size: Tuple[int, int],
    max_objects: int,
    errors: List[str],
    sample_id: str,
) -> None:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    expected = {
        "center_heatmap": (1, input_height, input_width),
        "regression": (10, input_height, input_width),
        "regression_mask": (1, input_height, input_width),
        "gt_objects": (int(max_objects), 9),
    }
    for name, expected_shape in expected.items():
        actual = shape_tuple(targets[name])
        if actual != expected_shape:
            errors.append(f"{sample_id}: {name} shape {actual}, expected {expected_shape}")
        finite_tensor(name, targets[name], errors, sample_id)


def dry_run(args: argparse.Namespace) -> Dict[str, object]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    input_size = resolve_input_size(args, dataset_dir)
    manifest_rows = read_csv(dataset_dir / "manifest.csv")
    selected_rows = select_rows(manifest_rows, args)
    objects_by_sample = load_object_boxes(dataset_dir / "object_boxes.csv")

    errors: List[str] = []
    warnings: List[str] = []
    notes = [
        "The current object target helper consumes vehicle actor rows only; "
        "person rows are not object-head positives in this dry run."
    ]

    split_counts = Counter(row.get("split", "") for row in manifest_rows)
    label_counts = Counter()
    source_counts = Counter()
    feature_shapes = Counter()
    mask_shapes = Counter()
    mask_classes = Counter()
    target_heatmap_shapes = Counter()
    target_regression_shapes = Counter()
    target_gt_object_shapes = Counter()

    total_object_rows = 0
    total_vehicle_rows = 0
    total_actor_vehicle_rows = 0
    total_valid_vehicle_objects = 0
    total_target_gt_count = 0
    total_positive_pixels = 0
    total_radar_supported_targets = 0
    total_parked_targets = 0
    samples_with_positive_targets = 0
    samples_without_positive_targets: List[str] = []
    samples_with_feature_tensor = 0
    max_target_gt_count = 0

    if not selected_rows:
        errors.append(f"No manifest rows selected for split={args.split!r}.")

    for row in selected_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            errors.append("Manifest row missing sample_id.")
            continue
        object_rows = objects_by_sample.get(sample_id, [])
        total_object_rows += len(object_rows)
        for obj_row in object_rows:
            label_counts[obj_row.get("label", "")] += 1
            source_counts[obj_row.get("gt_source", "")] += 1
        vehicle_rows = [obj for obj in object_rows if obj.get("label") == "vehicle"]
        actor_vehicle_rows = [
            obj for obj in vehicle_rows if obj.get("gt_source") == "actor"
        ]
        total_vehicle_rows += len(vehicle_rows)
        total_actor_vehicle_rows += len(actor_vehicle_rows)

        image_width = to_int(row.get("camera_width", ""), 0)
        image_height = to_int(row.get("camera_height", ""), 0)
        if image_width <= 0 or image_height <= 0:
            errors.append(f"{sample_id}: invalid camera_width/camera_height in manifest")
            continue

        valid_objects = valid_vehicle_objects(
            object_rows,
            image_width=image_width,
            image_height=image_height,
            min_area_px=float(args.min_gt_area_px),
        )
        total_valid_vehicle_objects += len(valid_objects)
        total_radar_supported_targets += sum(
            1 for obj in valid_objects if float(obj.get("radar_support", 0.0)) > 0.5
        )
        total_parked_targets += sum(
            1 for obj in valid_objects if float(obj.get("parked", 0.0)) > 0.5
        )

        targets = build_object_targets(
            objects=valid_objects,
            original_size=(image_width, image_height),
            input_size=input_size,
            heatmap_radius_px=int(args.heatmap_radius_px),
            max_objects=int(args.max_objects),
        )
        validate_target_shapes(
            targets=targets,
            input_size=input_size,
            max_objects=int(args.max_objects),
            errors=errors,
            sample_id=sample_id,
        )

        gt_count = int(targets["gt_count"].item())
        positive_pixels = int(targets["regression_mask"].sum().item())
        total_target_gt_count += gt_count
        total_positive_pixels += positive_pixels
        max_target_gt_count = max(max_target_gt_count, gt_count)
        target_heatmap_shapes[shape_tuple(targets["center_heatmap"])] += 1
        target_regression_shapes[shape_tuple(targets["regression"])] += 1
        target_gt_object_shapes[shape_tuple(targets["gt_objects"])] += 1

        if gt_count > 0:
            samples_with_positive_targets += 1
        else:
            samples_without_positive_targets.append(sample_id)
        if positive_pixels < gt_count:
            warnings.append(
                f"{sample_id}: {gt_count} GT objects map to {positive_pixels} unique "
                "target pixels; object centers may overlap after resizing."
            )
        if len(valid_objects) > int(args.max_objects):
            warnings.append(
                f"{sample_id}: {len(valid_objects)} valid vehicles exceeds "
                f"--max-objects={int(args.max_objects)}; targets were truncated."
            )

        try:
            feature, seg_mask = build_input_tensor(
                dataset_dir=dataset_dir,
                row=row,
                input_size=input_size,
            )
        except Exception as exc:
            errors.append(f"{sample_id}: input/mask dry run failed: {exc}")
            continue
        if feature is None or seg_mask is None:
            warnings.append("cv2 unavailable; skipped RGB/radar feature and mask target dry run.")
            continue
        samples_with_feature_tensor += 1
        feature_shapes[tuple(int(dim) for dim in feature.shape)] += 1
        mask_shapes[tuple(int(dim) for dim in seg_mask.shape)] += 1
        if feature.shape != (7, int(input_size[1]), int(input_size[0])):
            errors.append(
                f"{sample_id}: input feature shape {feature.shape}, expected "
                f"(7,{int(input_size[1])},{int(input_size[0])})"
            )
        if not np.isfinite(feature).all():
            errors.append(f"{sample_id}: input feature tensor contains non-finite values")
        for value in np.unique(seg_mask):
            mask_classes[int(value)] += 1
        unknown_classes = [int(value) for value in np.unique(seg_mask) if int(value) not in (0, 1, 2)]
        if unknown_classes:
            errors.append(f"{sample_id}: segmentation target has unknown classes {unknown_classes}")

    if bool(args.require_positive_vehicle_target) and samples_with_positive_targets == 0:
        errors.append("No inspected sample produced a positive vehicle object target.")

    inspected = len(selected_rows)
    summary = {
        "dataset_dir": str(dataset_dir),
        "input_size": [int(input_size[0]), int(input_size[1])],
        "split_filter": str(args.split),
        "manifest_rows": len(manifest_rows),
        "inspected_samples": inspected,
        "split_counts": counter_to_json(split_counts),
        "object_rows_inspected": int(total_object_rows),
        "label_counts_inspected": counter_to_json(label_counts),
        "gt_source_counts_inspected": counter_to_json(source_counts),
        "vehicle_rows_inspected": int(total_vehicle_rows),
        "actor_vehicle_rows_inspected": int(total_actor_vehicle_rows),
        "valid_vehicle_objects": int(total_valid_vehicle_objects),
        "target_gt_count_total": int(total_target_gt_count),
        "target_gt_count_mean": (
            total_target_gt_count / inspected if inspected else float("nan")
        ),
        "target_positive_pixels_total": int(total_positive_pixels),
        "target_positive_pixels_mean": (
            total_positive_pixels / inspected if inspected else float("nan")
        ),
        "target_max_gt_count": int(max_target_gt_count),
        "target_samples_with_positive_vehicle": int(samples_with_positive_targets),
        "target_samples_without_positive_vehicle": len(samples_without_positive_targets),
        "target_samples_without_positive_vehicle_ids": samples_without_positive_targets[:20],
        "target_radar_supported_vehicle_objects": int(total_radar_supported_targets),
        "target_parked_vehicle_objects": int(total_parked_targets),
        "target_heatmap_shapes": counter_to_json(target_heatmap_shapes),
        "target_regression_shapes": counter_to_json(target_regression_shapes),
        "target_gt_object_shapes": counter_to_json(target_gt_object_shapes),
        "feature_samples_checked": int(samples_with_feature_tensor),
        "feature_tensor_shapes": counter_to_json(feature_shapes),
        "segmentation_target_shapes": counter_to_json(mask_shapes),
        "segmentation_classes_seen": counter_to_json(mask_classes),
        "settings": {
            "min_gt_area_px": float(args.min_gt_area_px),
            "heatmap_radius_px": int(args.heatmap_radius_px),
            "max_objects": int(args.max_objects),
            "require_positive_vehicle_target": bool(args.require_positive_vehicle_target),
        },
        "notes": notes,
        "warnings": warnings,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = dry_run(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if bool(args.write_summary):
        out_path = dataset_dir / str(args.summary_name)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
