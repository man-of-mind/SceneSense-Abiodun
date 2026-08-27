#!/usr/bin/env python3
"""Native-grid CenterNet v2 targets and dataset.

Targets are built **directly at each head's native resolution**:

    grid centre  = projected input-image centre / head stride
    integer cell = floor(grid centre)
    offset       = grid centre - integer cell            (in [0,1) per axis)
    radius       = CenterNet Gaussian radius computed in that native grid
                   from the *resized* box dimensions divided by the stride
    regression   = supervised only at the native positive cell

Vehicles use stride 4 (108x192 for a 768x432 input); persons use stride 2
(216x384).  Each class owns a private regression tensor and mask, so one class
can never overwrite the other's regression target - the class-agnostic
``reg_mask`` defect in v1 is structurally impossible here.

Nothing is bilinearly enlarged, for training or for decoding.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from pole_lraspp_multimodal_fusion.object_targets import (
    OBJECT_CLASS_NAMES,
    gaussian_radius,
    valid_localization_objects,
)
from pole_lraspp_multimodal_fusion.train_fusion import FusionPoleMultiTaskDataset

BRANCHES: Tuple[Tuple[str, str, int], ...] = (
    # (prefix, class name, stride)
    ("veh", "vehicle", 4),
    ("per", "person", 2),
)
REG_FIELDS = 12


def draw_native_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius: int) -> None:
    """Draw a Gaussian on the native grid at the exact fractional centre.

    Uses the project's existing sigma convention (``sigma = max(1, radius/2)``)
    so the heatmap sharpness matches the rest of the codebase.
    """
    radius = max(1, int(radius))
    x0 = max(0, int(math.floor(cx)) - radius)
    y0 = max(0, int(math.floor(cy)) - radius)
    x1 = min(heatmap.shape[1], int(math.floor(cx)) + radius + 1)
    y1 = min(heatmap.shape[0], int(math.floor(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sigma = max(1.0, float(radius) / 2.0)
    values = np.exp(-((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], values.astype(np.float32))


def build_native_object_targets(
    *,
    objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
) -> Dict[str, torch.Tensor]:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_width, original_height = int(original_size[0]), int(original_size[1])
    sx = input_width / max(1.0, float(original_width))
    sy = input_height / max(1.0, float(original_height))
    name_to_index = {str(n): i for i, n in enumerate(object_class_names)}

    maps: Dict[str, np.ndarray] = {}
    grid: Dict[str, Tuple[int, int]] = {}
    for prefix, _cls, stride in BRANCHES:
        gh, gw = input_height // stride, input_width // stride
        grid[prefix] = (gh, gw)
        maps[f"{prefix}_hm"] = np.zeros((1, gh, gw), dtype=np.float32)
        maps[f"{prefix}_off"] = np.zeros((2, gh, gw), dtype=np.float32)
        maps[f"{prefix}_reg"] = np.zeros((REG_FIELDS, gh, gw), dtype=np.float32)
        maps[f"{prefix}_mask"] = np.zeros((1, gh, gw), dtype=np.float32)

    counts = {prefix: 0 for prefix, _c, _s in BRANCHES}
    # Larger objects are written first, so on a native-cell collision the larger
    # object keeps the regression target (same rule as v1).
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        class_index = int(obj.get("class_index", 0))
        if class_index < 0 or class_index >= len(BRANCHES):
            continue
        prefix, cls_name, stride = BRANCHES[class_index]
        if name_to_index.get(cls_name, class_index) != class_index:
            raise ValueError(f"class order mismatch for {cls_name!r}")
        gh, gw = grid[prefix]
        cx_in = float(obj["center_x"]) * sx
        cy_in = float(obj["center_y"]) * sy
        gx = cx_in / float(stride)
        gy = cy_in / float(stride)
        ix, iy = int(math.floor(gx)), int(math.floor(gy))
        if ix < 0 or iy < 0 or ix >= gw or iy >= gh:
            continue
        # Gaussian radius in *this* native grid, from the resized box dimensions.
        bw_grid = float(obj.get("bbox_w", 0.0)) * sx / float(stride)
        bh_grid = float(obj.get("bbox_h", 0.0)) * sy / float(stride)
        radius = max(1, int(round(gaussian_radius(bh_grid, bw_grid))))
        draw_native_gaussian(maps[f"{prefix}_hm"][0], gx, gy, radius)
        # The focal loss identifies positives by target == 1.0 exactly.
        maps[f"{prefix}_hm"][0, iy, ix] = 1.0
        if maps[f"{prefix}_mask"][0, iy, ix] < 0.5:
            maps[f"{prefix}_off"][0, iy, ix] = gx - float(ix)
            maps[f"{prefix}_off"][1, iy, ix] = gy - float(iy)
            values = [
                obj["local_x"], obj["local_y"], obj["local_z"],
                obj["size_x"], obj["size_y"], obj["size_z"],
                obj["yaw_sin"], obj["yaw_cos"],
                obj["parked"], obj["radar_support"],
                float(obj.get("bbox_w", 0.0)) * sx / max(1.0, float(input_width)),
                float(obj.get("bbox_h", 0.0)) * sy / max(1.0, float(input_height)),
            ]
            maps[f"{prefix}_reg"][:, iy, ix] = np.array(values, dtype=np.float32)
            maps[f"{prefix}_mask"][0, iy, ix] = 1.0
            counts[prefix] += 1

    out = {key: torch.from_numpy(value) for key, value in maps.items()}
    for prefix, _c, _s in BRANCHES:
        out[f"{prefix}_positives"] = torch.tensor(counts[prefix], dtype=torch.long)
        if counts[prefix] > 0:
            assert float(maps[f"{prefix}_hm"].max()) >= 0.999, (
                f"{prefix} heatmap has no exact 1.0 peak despite positives; "
                "focal loss would see zero positives."
            )
    return out


class NativeFusionDataset(FusionPoleMultiTaskDataset):
    """Same RGB/radar/mask input pipeline as v1; native multi-stride targets."""

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        original_width, original_height = image.size
        mask = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        radar = self._load_radar(row)
        image, mask, radar = self._augment(image, mask, radar)
        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        radar = self._resize_radar(radar)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        image_tensor = (image_tensor - self.rgb_mean) / self.rgb_std
        radar_tensor = torch.from_numpy(np.ascontiguousarray(radar)).to(torch.float32)
        fused = torch.cat([image_tensor, radar_tensor], dim=0)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=original_width,
            image_height=original_height,
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 12.0)),
            object_class_names=self.object_class_names,
            max_distance_m=(
                float(self.object_cfg["max_gt_distance_m"])
                if self.object_cfg.get("max_gt_distance_m") not in (None, "", 0)
                else None
            ),
        )
        targets = build_native_object_targets(
            objects=objects,
            original_size=(original_width, original_height),
            input_size=(self.input_width, self.input_height),
            object_class_names=self.object_class_names,
        )
        return fused, mask_tensor, targets
