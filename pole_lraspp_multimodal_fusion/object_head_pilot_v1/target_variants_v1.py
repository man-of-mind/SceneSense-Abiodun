#!/usr/bin/env python3
"""Candidate object-target construction for the object-head architecture pilot.

Control arm (A) uses the production
:func:`pole_lraspp_multimodal_fusion.object_targets.build_object_targets`
unchanged - current adaptive heatmap-radius behaviour.

Candidate arm (B) uses :func:`build_object_targets_capped` with
``vehicle_heatmap_radius_cap_px = 4``: adaptive person radii are retained
exactly, and only the *vehicle* class Gaussian radius is capped. Tensor shapes,
the decoder, the regression heads and every person target are untouched - the
cap applies solely to the vehicle channel of ``center_heatmap``.

This lives outside the production package on purpose: no production file is
edited for the pilot. :func:`assert_control_parity` is a runtime guard that
proves the copy is bit-identical to production when the cap is disabled, so the
duplicated function cannot silently drift from its source.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from pole_lraspp_multimodal_fusion.object_targets import (
    OBJECT_CLASS_NAMES,
    build_object_targets,
    draw_gaussian,
    gaussian_radius,
    object_reg_channels,
)

VEHICLE_CLASS_NAME = "vehicle"
DEFAULT_VEHICLE_RADIUS_CAP_PX = 4


def build_object_targets_capped(
    *,
    objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    heatmap_radius_px: int,
    max_objects: int,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    predict_bbox2d: bool = False,
    adaptive_heatmap_radius: bool = False,
    vehicle_heatmap_radius_cap_px: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Production target construction plus a vehicle-only adaptive-radius cap.

    With ``vehicle_heatmap_radius_cap_px=None`` this is a faithful copy of the
    production function; the cap is the single behavioural difference.
    """
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_width, original_height = int(original_size[0]), int(original_size[1])
    sx = input_width / max(1.0, float(original_width))
    sy = input_height / max(1.0, float(original_height))
    reg_channels = object_reg_channels(predict_bbox2d)
    class_names = tuple(object_class_names)
    object_class_count = max(1, len(class_names))
    vehicle_index = class_names.index(VEHICLE_CLASS_NAME) if VEHICLE_CLASS_NAME in class_names else None
    cap = None if vehicle_heatmap_radius_cap_px is None else int(vehicle_heatmap_radius_cap_px)

    heatmap = np.zeros((object_class_count, input_height, input_width), dtype=np.float32)
    regression = np.zeros((reg_channels, input_height, input_width), dtype=np.float32)
    reg_mask = np.zeros((1, input_height, input_width), dtype=np.float32)
    gt_objects = np.zeros((int(max_objects), 9), dtype=np.float32)
    gt_class_indices = np.zeros((int(max_objects),), dtype=np.int64)
    gt_count = 0
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        class_index = int(obj.get("class_index", 0))
        if class_index < 0 or class_index >= object_class_count:
            continue
        cx = float(obj["center_x"]) * sx
        cy = float(obj["center_y"]) * sy
        ix = int(round(cx))
        iy = int(round(cy))
        if ix < 0 or iy < 0 or ix >= input_width or iy >= input_height:
            continue
        if adaptive_heatmap_radius:
            bw_in = float(obj.get("bbox_w", 0.0)) * sx
            bh_in = float(obj.get("bbox_h", 0.0)) * sy
            radius = int(max(2, round(gaussian_radius(bh_in, bw_in))))
        else:
            radius = int(heatmap_radius_px)
        # The only behavioural difference from production: a near vehicle's
        # size-matched Gaussian spreads positives across many pixels, which is
        # what produces duplicate vehicle peaks after NMS. Person radii are left
        # adaptive so small/far pedestrians keep their calibrated footprint.
        if cap is not None and vehicle_index is not None and class_index == vehicle_index:
            radius = min(radius, cap)
        draw_gaussian(heatmap[class_index], cx, cy, radius)
        heatmap[class_index, iy, ix] = 1.0
        if reg_mask[0, iy, ix] < 0.5:
            values = [
                obj["local_x"], obj["local_y"], obj["local_z"],
                obj["size_x"], obj["size_y"], obj["size_z"],
                obj["yaw_sin"], obj["yaw_cos"],
                obj["parked"], obj["radar_support"],
            ]
            if predict_bbox2d:
                values.append(float(obj.get("bbox_w", 0.0)) * sx / max(1.0, float(input_width)))
                values.append(float(obj.get("bbox_h", 0.0)) * sy / max(1.0, float(input_height)))
            regression[:, iy, ix] = np.array(values, dtype=np.float32)
            reg_mask[0, iy, ix] = 1.0
        if gt_count < int(max_objects):
            gt_objects[gt_count] = np.array(
                [
                    obj["world_x"], obj["world_y"], obj["world_z"],
                    obj["size_x"], obj["size_y"], obj["size_z"],
                    obj["yaw_sin"], obj["yaw_cos"], obj["parked"],
                ],
                dtype=np.float32,
            )
            gt_class_indices[gt_count] = class_index
            gt_count += 1
    if gt_count > 0:
        assert float(heatmap.max()) >= 0.999, (
            "object center heatmap target has no peak >= 1.0 despite gt_count > 0"
        )
    return {
        "center_heatmap": torch.from_numpy(heatmap),
        "regression": torch.from_numpy(regression),
        "regression_mask": torch.from_numpy(reg_mask),
        "gt_objects": torch.from_numpy(gt_objects),
        "gt_class_indices": torch.from_numpy(gt_class_indices),
        "gt_count": torch.tensor(gt_count, dtype=torch.long),
    }


def assert_control_parity(sample_kwargs: Dict[str, Any]) -> None:
    """Prove the copy matches production exactly when the cap is disabled."""
    reference = build_object_targets(**sample_kwargs)
    candidate = build_object_targets_capped(**sample_kwargs, vehicle_heatmap_radius_cap_px=None)
    if set(reference) != set(candidate):
        raise AssertionError(
            f"target key drift: production={sorted(reference)} pilot={sorted(candidate)}"
        )
    for key, expected in reference.items():
        if not torch.equal(expected, candidate[key]):
            raise AssertionError(
                f"pilot target copy diverged from production on {key!r}; "
                "re-sync target_variants_v1.build_object_targets_capped"
            )


def install(vehicle_heatmap_radius_cap_px: Optional[int]) -> str:
    """Point train_fusion's dataset at the capped builder. Returns the arm label.

    ``None`` leaves the production function in place, so the control arm runs the
    unmodified code path rather than a copy of it.
    """
    from pole_lraspp_multimodal_fusion import train_fusion

    if vehicle_heatmap_radius_cap_px is None:
        return "control:production_adaptive_radius"

    cap = int(vehicle_heatmap_radius_cap_px)

    def _capped(**kwargs: Any) -> Dict[str, torch.Tensor]:
        return build_object_targets_capped(**kwargs, vehicle_heatmap_radius_cap_px=cap)

    train_fusion.build_object_targets = _capped
    return f"candidate:vehicle_heatmap_radius_cap_px={cap}"
