#!/usr/bin/env python3
"""Geometry-consistent v3.1 training view for COCO person distillation.

The ordinary native-grid dataset cannot be used here because its object rows are
constructed after image augmentation and therefore cannot follow a geometric image
transform.  This implementation constructs all model-plane geometry first, samples
one affine, applies it to every image-plane modality, and only then regenerates the
native targets.  Metric/world values are copied without modification.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
for _path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    valid_localization_objects,
)
from targets_v1 import (  # noqa: E402
    NATIVE_STRIDE,
    build_native_object_targets,
    downsample_ignore_conservative,
)

MODEL_WIDTH, MODEL_HEIGHT = 768, 432
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _resize_radar(radar: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = (int(value) for value in size)
    if tuple(radar.shape[-2:]) == (height, width):
        return radar.astype(np.float32, copy=False)
    values = [
        cv2.resize(channel, (width, height),
                   interpolation=cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR)
        for index, channel in enumerate(radar)
    ]
    return np.stack(values, axis=0).astype(np.float32)


def _load_radar(path: Path) -> np.ndarray:
    payload = np.load(path)
    try:
        radar = payload["radar"] if isinstance(payload, np.lib.npyio.NpzFile) else payload
        value = np.asarray(radar, dtype=np.float32)
    finally:
        if hasattr(payload, "close"):
            payload.close()
    if value.ndim != 3 or int(value.shape[0]) != 4:
        raise ValueError(f"registered radar tensor must be [4,H,W], got {value.shape}: {path}")
    return value


def _model_objects(objects: Sequence[Dict[str, float]], original_size: Tuple[int, int]) -> List[Dict[str, float]]:
    """Copy objects into model-input pixels while retaining metric fields verbatim."""
    ow, oh = (float(value) for value in original_size)
    sx, sy = MODEL_WIDTH / ow, MODEL_HEIGHT / oh
    result: List[Dict[str, float]] = []
    for source in objects:
        item = dict(source)
        item["center_x"] = float(source["center_x"]) * sx
        item["center_y"] = float(source["center_y"]) * sy
        item["bbox_w"] = float(source.get("bbox_w", 0.0)) * sx
        item["bbox_h"] = float(source.get("bbox_h", 0.0)) * sy
        item["area"] = item["bbox_w"] * item["bbox_h"]
        result.append(item)
    return result


def _box(item: Dict[str, float]) -> np.ndarray:
    half_w, half_h = float(item["bbox_w"]) / 2.0, float(item["bbox_h"]) / 2.0
    return np.asarray([
        float(item["center_x"]) - half_w, float(item["center_y"]) - half_h,
        float(item["center_x"]) + half_w, float(item["center_y"]) + half_h,
    ], dtype=np.float32)


def _sample_person_affine(
    objects: Sequence[Dict[str, float]], rng: Any, *, attempts: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scale about a valid person centre; the implied translation fixes that centre.

    ``x' = s*x + (1-s)*anchor_x`` is one affine containing both scale and translation.
    Keeping the anchor fixed is the unique translation that cannot introduce an
    unregistered positional prior.  A scale is accepted only when its entire anchor
    box remains on canvas, as required by the registered resampling rule.
    """
    people = [item for item in objects if str(item.get("class_name")) == "person"]
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if not people:
        return identity, {"applied": False, "reason": "no_valid_person", "attempts": 0,
                          "scale": 1.0, "anchor_index": None}
    for attempt in range(1, int(attempts) + 1):
        anchor_index = int(rng.randint(0, len(people)))
        anchor = people[anchor_index]
        scale = float(rng.uniform(0.8, 1.4))
        cx, cy = float(anchor["center_x"]), float(anchor["center_y"])
        transformed = _box(anchor).copy()
        transformed[[0, 2]] = scale * transformed[[0, 2]] + (1.0 - scale) * cx
        transformed[[1, 3]] = scale * transformed[[1, 3]] + (1.0 - scale) * cy
        if (transformed[0] >= 0.0 and transformed[1] >= 0.0
                and transformed[2] <= MODEL_WIDTH and transformed[3] <= MODEL_HEIGHT):
            matrix = np.asarray(
                [[scale, 0.0, (1.0 - scale) * cx],
                 [0.0, scale, (1.0 - scale) * cy]], dtype=np.float32,
            )
            return matrix, {
                "applied": True, "reason": "registered_person_geometry",
                "attempts": attempt, "scale": scale, "anchor_index": anchor_index,
                "anchor_center_xy": [cx, cy], "matrix_2x3": matrix.tolist(),
            }
    return identity, {"applied": False, "reason": "anchor_resampling_exhausted",
                      "attempts": int(attempts), "scale": 1.0, "anchor_index": None}


def _transform_objects(
    objects: Sequence[Dict[str, float]], matrix: np.ndarray, ignore: np.ndarray,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Transform boxes/centres; quarantine and drop every partly off-canvas object."""
    kept: List[Dict[str, float]] = []
    dropped: List[Dict[str, Any]] = []
    scale = float(matrix[0, 0])
    for source in objects:
        item = dict(source)
        x = scale * float(source["center_x"]) + float(matrix[0, 2])
        y = scale * float(source["center_y"]) + float(matrix[1, 2])
        item["center_x"], item["center_y"] = x, y
        item["bbox_w"] = scale * float(source.get("bbox_w", 0.0))
        item["bbox_h"] = scale * float(source.get("bbox_h", 0.0))
        item["area"] = item["bbox_w"] * item["bbox_h"]
        box = _box(item)
        fully_inside = (
            box[0] >= 0.0 and box[1] >= 0.0
            and box[2] <= MODEL_WIDTH and box[3] <= MODEL_HEIGHT
            and 0.0 <= x < MODEL_WIDTH and 0.0 <= y < MODEL_HEIGHT
        )
        if fully_inside:
            kept.append(item)
            continue
        ix0 = max(0, min(MODEL_WIDTH, int(math.floor(float(box[0])))))
        iy0 = max(0, min(MODEL_HEIGHT, int(math.floor(float(box[1])))))
        ix1 = max(0, min(MODEL_WIDTH, int(math.ceil(float(box[2])))))
        iy1 = max(0, min(MODEL_HEIGHT, int(math.ceil(float(box[3])))))
        if ix1 > ix0 and iy1 > iy0:
            ignore[iy0:iy1, ix0:ix1] = True
        dropped.append({"class_name": str(item.get("class_name")), "box_xyxy": box.tolist(),
                        "visible_ignore_box_xyxy": [ix0, iy0, ix1, iy1]})
    return kept, dropped


def _warp_bundle(
    rgb: np.ndarray, radar: np.ndarray, segmentation: np.ndarray, ignore: np.ndarray,
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = (MODEL_WIDTH, MODEL_HEIGHT)
    rgb_out = cv2.warpAffine(rgb, matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    seg_out = cv2.warpAffine(segmentation, matrix, size, flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ignore_out = cv2.warpAffine(ignore.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=1) != 0
    radar_out = np.stack([
        cv2.warpAffine(channel, matrix, size,
                       flags=cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        for index, channel in enumerate(radar)
    ], axis=0).astype(np.float32)
    return rgb_out, radar_out, seg_out, ignore_out


def _intrinsics(row: Dict[str, str], matrix: np.ndarray) -> torch.Tensor:
    sx = MODEL_WIDTH / float(row["camera_width"])
    sy = MODEL_HEIGHT / float(row["camera_height"])
    base = np.asarray([
        [float(row["camera_fx"]) * sx, 0.0, float(row["camera_cx"]) * sx],
        [0.0, float(row["camera_fy"]) * sy, float(row["camera_cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    affine3 = np.eye(3, dtype=np.float32)
    affine3[:2] = matrix
    return torch.from_numpy(affine3 @ base)


class PersonCocoDistillationDataset(Dataset):
    """Seven-channel student + exact RGB teacher + variable person geometry."""

    def __init__(
        self, dataset_dir: Path, rows: Sequence[Dict[str, str]],
        object_rows: Dict[str, List[Dict[str, str]]], object_cfg: Dict[str, Any],
        *, augment: bool,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.rows = list(rows)
        self.object_rows = object_rows
        self.object_cfg = dict(object_cfg)
        self.object_class_names = tuple(object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        # Training may pass (dataset_index, augmentation_seed).  This makes every
        # draw reproducible across a recovery restart without depending on opaque
        # persistent-worker RNG state while retaining the registered 8 workers.
        if isinstance(index, (tuple, list)):
            sample_index, augmentation_seed = int(index[0]), int(index[1])
            # RandomState is intentionally used for worker/version stability; reduce
            # the deterministic 64-bit draw identity into its documented uint32 domain.
            rng: Any = np.random.RandomState(augmentation_seed % (2**32 - 1))
        else:
            sample_index, augmentation_seed = int(index), None
            rng = np.random
        row = self.rows[sample_index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        original_size = image.size
        segmentation_image = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        ignore_image = Image.open(self.dataset_dir / row["object_ignore_mask_path"]).convert("L")
        radar = _load_radar(self.dataset_dir / row["radar_tensor_path"])
        raw_objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=original_size[0], image_height=original_size[1],
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 12.0)),
            object_class_names=self.object_class_names,
            max_distance_m=float(self.object_cfg.get("max_gt_distance_m", 40.0)),
        )
        objects = _model_objects(raw_objects, original_size)

        # Existing strong photometric recipe: the same two independent operations and
        # probabilities/ranges used by FusionPoleMultiTaskDataset._augment.
        if self.augment and rng.rand() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.65, 1.35)))
        if self.augment and rng.rand() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.BILINEAR)
        segmentation_image = segmentation_image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.NEAREST)
        ignore_image = ignore_image.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.NEAREST)
        rgb = np.array(image, dtype=np.uint8, copy=True)
        segmentation = np.array(segmentation_image, dtype=np.uint8, copy=True)
        ignore = np.array(ignore_image, dtype=np.uint8, copy=True) != 0
        radar = _resize_radar(radar, (MODEL_WIDTH, MODEL_HEIGHT))

        matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        geometry: Dict[str, Any] = {"applied": False, "reason": "augmentation_off",
                                    "attempts": 0, "scale": 1.0, "anchor_index": None}
        has_person = any(str(item.get("class_name")) == "person" for item in objects)
        if self.augment and has_person and rng.rand() < 0.35:
            matrix, geometry = _sample_person_affine(objects, rng, attempts=8)
            if geometry["applied"]:
                rgb, radar, segmentation, ignore = _warp_bundle(
                    rgb, radar, segmentation, ignore, matrix,
                )
        transformed_objects, dropped = _transform_objects(objects, matrix, ignore)
        geometry["dropped_off_canvas"] = dropped

        targets = build_native_object_targets(
            objects=transformed_objects, original_size=(MODEL_WIDTH, MODEL_HEIGHT),
            input_size=(MODEL_WIDTH, MODEL_HEIGHT),
            max_objects=int(self.object_cfg.get("max_objects_per_frame", 64)),
            stride=NATIVE_STRIDE, object_class_names=self.object_class_names,
        )
        ignore_cells = torch.from_numpy(downsample_ignore_conservative(ignore, NATIVE_STRIDE).copy())
        heatmap = targets["center_heatmap"]
        heatmap[ignore_cells.unsqueeze(0).expand_as(heatmap) & heatmap.eq(0.0)] = -1.0
        targets["center_heatmap"] = heatmap
        targets["object_ignore_mask"] = ignore_cells.unsqueeze(0)

        person_objects = [item for item in transformed_objects if str(item.get("class_name")) == "person"]
        person_boxes = torch.as_tensor(
            np.stack([_box(item) for item in person_objects], axis=0)
            if person_objects else np.zeros((0, 4), dtype=np.float32), dtype=torch.float32,
        )
        person_cells = torch.as_tensor([
            [int(math.floor(float(item["center_y"]) / NATIVE_STRIDE)),
             int(math.floor(float(item["center_x"]) / NATIVE_STRIDE))]
            for item in person_objects
        ], dtype=torch.long).reshape(-1, 2)
        person_radar = torch.as_tensor(
            [float(item["radar_support"]) for item in person_objects], dtype=torch.float32,
        )

        rgb01 = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        fused = torch.cat([(rgb01 - RGB_MEAN) / RGB_STD,
                           torch.from_numpy(np.ascontiguousarray(radar)).float()], dim=0)
        segmentation_tensor = torch.from_numpy(np.ascontiguousarray(segmentation)).long()
        segmentation_tensor[segmentation_tensor == 255] = -100
        return {
            "student": fused, "teacher_rgb01": rgb01,
            "segmentation": segmentation_tensor, "targets": targets,
            "person_boxes": person_boxes, "person_cells": person_cells,
            "person_radar_support": person_radar,
            "camera_intrinsics": _intrinsics(row, matrix),
            "sample_id": str(row["sample_id"]), "episode_id": str(row["experiment_id"]),
            "geometry": geometry, "augmentation_seed": augmentation_seed,
        }


def person_distillation_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack fixed tensors and preserve variable person tensors as per-image lists."""
    if not samples:
        raise ValueError("empty distillation batch")
    target_keys = tuple(samples[0]["targets"])
    return {
        "student": torch.stack([item["student"] for item in samples]),
        "teacher_rgb01": torch.stack([item["teacher_rgb01"] for item in samples]),
        "segmentation": torch.stack([item["segmentation"] for item in samples]),
        "targets": {key: torch.stack([item["targets"][key] for item in samples])
                    for key in target_keys},
        "person_boxes": [item["person_boxes"] for item in samples],
        "person_cells": [item["person_cells"] for item in samples],
        "person_radar_support": [item["person_radar_support"] for item in samples],
        "camera_intrinsics": torch.stack([item["camera_intrinsics"] for item in samples]),
        "sample_ids": [item["sample_id"] for item in samples],
        "episode_ids": [item["episode_id"] for item in samples],
        "geometry": [item["geometry"] for item in samples],
        "augmentation_seeds": [item["augmentation_seed"] for item in samples],
    }


def geometry_contract_probe() -> Dict[str, Any]:
    """Deterministic synthetic proof of joint affine, metric invariance, and quarantine."""
    rgb = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
    radar = np.zeros((4, MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
    seg = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.uint8)
    ignore = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=bool)
    objects = [
        {"class_name": "person", "class_index": 1.0, "center_x": 384.0, "center_y": 216.0,
         "bbox_w": 40.0, "bbox_h": 80.0, "area": 3200.0, "local_x": 10.0,
         "local_y": 1.0, "local_z": 0.0, "world_x": 2.0, "world_y": 3.0,
         "world_z": 0.0, "size_x": 0.5, "size_y": 0.5, "size_z": 1.8,
         "yaw_sin": 0.0, "yaw_cos": 1.0, "parked": 0.0, "radar_support": 1.0},
        {"class_name": "vehicle", "class_index": 0.0, "center_x": 690.0, "center_y": 216.0,
         "bbox_w": 30.0, "bbox_h": 40.0, "area": 1200.0, "local_x": 15.0,
         "local_y": 2.0, "local_z": 0.0, "world_x": 5.0, "world_y": 6.0,
         "world_z": 0.0, "size_x": 4.0, "size_y": 2.0, "size_z": 1.5,
         "yaw_sin": 0.0, "yaw_cos": 1.0, "parked": 0.0, "radar_support": 0.0},
    ]
    scale = 1.25
    matrix = np.asarray([[scale, 0.0, (1.0-scale)*384.0],
                         [0.0, scale, (1.0-scale)*216.0]], dtype=np.float32)
    _rgb, _radar, _seg, warped_ignore = _warp_bundle(rgb, radar, seg, ignore, matrix)
    kept, dropped = _transform_objects(objects, matrix, warped_ignore)
    person = next(item for item in kept if item["class_name"] == "person")
    metric_keys = ("local_x", "local_y", "local_z", "world_x", "world_y", "world_z")
    return {
        "single_affine": matrix.tolist(), "scale": scale,
        "person_anchor_center_unchanged": [person["center_x"], person["center_y"]] == [384.0, 216.0],
        "world_and_local_values_unchanged": all(person[key] == objects[0][key] for key in metric_keys),
        "off_canvas_objects_dropped": len(dropped) == 1 and len(kept) == 1,
        "off_canvas_visible_region_ignored": bool(warped_ignore.any()),
        "joint_output_shapes": {
            "rgb": list(_rgb.shape), "radar": list(_radar.shape),
            "segmentation": list(_seg.shape), "ignore": list(warped_ignore.shape),
        },
    }
