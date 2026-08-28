#!/usr/bin/env python3
"""Native stride-4 object targets and the v3.1 ignore-aware dataset.

Targets are built DIRECTLY on the 192x108 grid. Nothing is drawn at 768x432 and then
resized: the supervision the head receives is the supervision at the resolution the
head actually predicts.

Everything that is not grid geometry is inherited unchanged from v3.1:
  * the same actor-origin GT rows, the same 40 m range gate and the same 12 px
    minimum projected-area gate (valid_localization_objects);
  * the same static-vehicle (parked) and person-visibility contracts, which live in
    the contract CSV columns and are copied straight through;
  * the same regression channel order, units and meanings (metres, metres, sin/cos,
    logits, input-image fractions for the 2D box);
  * the same -1 background-ignore sentinel understood by the focal loss, and the same
    "positives override ignore" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for _path in (str(PACKAGE_ROOT), str(BASE_PKG), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion import train_fusion as trainer  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    draw_gaussian,
    gaussian_radius,
    valid_localization_objects,
)

from model_v1 import NATIVE_STRIDE, OFFSET_CHANNELS, REG_CHANNELS  # noqa: E402

MIN_NATIVE_RADIUS_CELLS = 1


def build_native_object_targets(
    *,
    objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    max_objects: int,
    stride: int = NATIVE_STRIDE,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
) -> Dict[str, torch.Tensor]:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_width, original_height = int(original_size[0]), int(original_size[1])
    if input_width % stride or input_height % stride:
        raise ValueError(f"input {input_size} is not divisible by stride {stride}")
    grid_w, grid_h = input_width // int(stride), input_height // int(stride)
    scale_x = input_width / max(1.0, float(original_width))
    scale_y = input_height / max(1.0, float(original_height))
    class_count = max(1, len(tuple(object_class_names)))

    heatmap = np.zeros((class_count, grid_h, grid_w), dtype=np.float32)
    regression = np.zeros((REG_CHANNELS, grid_h, grid_w), dtype=np.float32)
    offset = np.zeros((OFFSET_CHANNELS, grid_h, grid_w), dtype=np.float32)
    reg_mask = np.zeros((1, grid_h, grid_w), dtype=np.float32)
    gt_objects = np.zeros((int(max_objects), 9), dtype=np.float32)
    gt_class_indices = np.zeros((int(max_objects),), dtype=np.int64)
    gt_count = 0
    cell_conflicts = 0

    # Largest-first, exactly as v3.1: when two centres land in one cell the larger
    # object owns the shared regression slot.
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        class_index = int(obj.get("class_index", 0))
        if class_index < 0 or class_index >= class_count:
            continue
        center_x = float(obj["center_x"]) * scale_x           # model pixel space
        center_y = float(obj["center_y"]) * scale_y
        grid_x = center_x / float(stride)                     # native grid space
        grid_y = center_y / float(stride)
        cell_x, cell_y = int(np.floor(grid_x)), int(np.floor(grid_y))
        if cell_x < 0 or cell_y < 0 or cell_x >= grid_w or cell_y >= grid_h:
            continue

        box_w = float(obj.get("bbox_w", 0.0)) * scale_x / float(stride)   # native cells
        box_h = float(obj.get("bbox_h", 0.0)) * scale_y / float(stride)
        radius = int(max(MIN_NATIVE_RADIUS_CELLS, round(gaussian_radius(box_h, box_w))))
        draw_gaussian(heatmap[class_index], grid_x, grid_y, radius)
        # The Gaussian is sampled at integer cells, so a sub-cell centre leaves the
        # peak below 1.0; the focal loss keys positives off target == 1.0.
        heatmap[class_index, cell_y, cell_x] = 1.0

        if reg_mask[0, cell_y, cell_x] < 0.5:
            values = [
                obj["local_x"], obj["local_y"], obj["local_z"],
                obj["size_x"], obj["size_y"], obj["size_z"],
                obj["yaw_sin"], obj["yaw_cos"],
                obj["parked"], obj["radar_support"],
                # 2D box stays a fraction of the INPUT image, i.e. grid-independent.
                float(obj.get("bbox_w", 0.0)) * scale_x / max(1.0, float(input_width)),
                float(obj.get("bbox_h", 0.0)) * scale_y / max(1.0, float(input_height)),
            ]
            regression[:, cell_y, cell_x] = np.array(values, dtype=np.float32)
            # Private stride-quantization offset in [0, 1).
            offset[0, cell_y, cell_x] = grid_x - float(cell_x)
            offset[1, cell_y, cell_x] = grid_y - float(cell_y)
            reg_mask[0, cell_y, cell_x] = 1.0
        else:
            cell_conflicts += 1

        if gt_count < int(max_objects):
            gt_objects[gt_count] = np.array(
                [obj["world_x"], obj["world_y"], obj["world_z"],
                 obj["size_x"], obj["size_y"], obj["size_z"],
                 obj["yaw_sin"], obj["yaw_cos"], obj["parked"]], dtype=np.float32,
            )
            gt_class_indices[gt_count] = class_index
            gt_count += 1

    if gt_count > 0 and float(heatmap.max()) < 0.999:
        raise AssertionError(
            "native centre heatmap has no peak >= 1.0 despite gt_count > 0; the focal "
            "positive count would be zero and the centre head would never learn."
        )
    return {
        "center_heatmap": torch.from_numpy(heatmap),
        "regression": torch.from_numpy(regression),
        "regression_mask": torch.from_numpy(reg_mask),
        "center_offset": torch.from_numpy(offset),
        "gt_objects": torch.from_numpy(gt_objects),
        "gt_class_indices": torch.from_numpy(gt_class_indices),
        "gt_count": torch.tensor(gt_count, dtype=torch.long),
        "cell_conflicts": torch.tensor(cell_conflicts, dtype=torch.long),
    }


def downsample_ignore_conservative(mask: np.ndarray, stride: int = NATIVE_STRIDE) -> np.ndarray:
    """Conservative ignore downsample: a native cell is ignored only if EVERY one of
    its stride x stride source pixels is ignored.

    Conservative in the direction that matters here - quarantine is never expanded
    into supervised territory, so the background focal term keeps the maximum amount
    of genuine negative supervision. Positives override ignore regardless.
    """
    height, width = mask.shape
    if height % stride or width % stride:
        raise ValueError(f"ignore mask {mask.shape} is not divisible by stride {stride}")
    blocks = mask.reshape(height // stride, stride, width // stride, stride)
    return blocks.all(axis=(1, 3))


class NativeGridDataset(trainer.FusionPoleMultiTaskDataset):
    """v3.1 dataset with native stride-4 object targets and downsampled ignore."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        original_width, original_height = image.size
        mask = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        radar = self._load_radar(row)
        image, mask, radar = self._augment(image, mask, radar)
        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        radar = self._resize_radar(radar)
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = (torch.from_numpy(array).permute(2, 0, 1) - self.rgb_mean) / self.rgb_std
        radar_tensor = torch.from_numpy(np.ascontiguousarray(radar)).to(torch.float32)
        fused = torch.cat([image_tensor, radar_tensor], dim=0)

        segmentation = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        segmentation[segmentation == 255] = -100  # PyTorch CrossEntropy ignore_index

        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=original_width, image_height=original_height,
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 24.0)),
            object_class_names=self.object_class_names,
            max_distance_m=(float(self.object_cfg["max_gt_distance_m"])
                            if self.object_cfg.get("max_gt_distance_m") not in (None, "", 0) else None),
        )
        targets = build_native_object_targets(
            objects=objects,
            original_size=(original_width, original_height),
            input_size=(self.input_width, self.input_height),
            max_objects=int(self.object_cfg.get("max_objects_per_frame", 64)),
            stride=NATIVE_STRIDE,
            object_class_names=self.object_class_names,
        )

        ignore_image = Image.open(self.dataset_dir / row["object_ignore_mask_path"]).convert("L")
        ignore_image = ignore_image.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        ignore_full = np.asarray(ignore_image, dtype=np.uint8) != 0
        ignore_cells = torch.from_numpy(downsample_ignore_conservative(ignore_full, NATIVE_STRIDE).copy())

        heatmap = targets["center_heatmap"]
        # -1 is the explicit sentinel the v3.1 focal loss treats as "no contribution".
        # Applied only where the target is exactly 0, so both exact peaks and their
        # Gaussian skirts survive: positives override ignore.
        background_ignore = ignore_cells.unsqueeze(0).expand_as(heatmap) & heatmap.eq(0.0)
        heatmap[background_ignore] = -1.0
        targets["center_heatmap"] = heatmap
        targets["object_ignore_mask"] = ignore_cells.unsqueeze(0)
        return fused, segmentation, targets
