#!/usr/bin/env python3
"""Route B RGB/radar/ROI dataset with one consistent geometric transform."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from pole_lraspp_multimodal_fusion.object_targets import valid_localization_objects


CLASS_TO_LABEL = {"vehicle": 1, "person": 2}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class RouteBFasterRCNNDataset(Dataset):
    """Returns resized RGB/radar, projected-box targets and semantic masks.

    The sole geometric augmentation is a synchronized horizontal reflection.
    In addition to pixels and boxes it reflects camera-local lateral position and
    camera-local yaw. World XYZ is never used as a training regression target.
    """

    def __init__(
        self,
        dataset_dir: Path,
        rows: Sequence[Dict[str, str]],
        object_rows: Dict[str, List[Dict[str, str]]],
        input_size: Tuple[int, int],
        *,
        training: bool,
        flip_probability: float = 0.5,
        min_gt_area_px: float = 12.0,
        max_gt_distance_m: float = 40.0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.rows = list(rows)
        self.object_rows = object_rows
        self.input_width, self.input_height = map(int, input_size)
        self.training = bool(training)
        self.flip_probability = float(flip_probability)
        self.min_gt_area_px = float(min_gt_area_px)
        self.max_gt_distance_m = float(max_gt_distance_m)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_radar(self, row: Dict[str, str]) -> np.ndarray:
        payload = np.load(self.dataset_dir / row["radar_tensor_path"])
        try:
            radar = (
                payload["radar"].astype(np.float32)
                if isinstance(payload, np.lib.npyio.NpzFile)
                else np.asarray(payload, dtype=np.float32)
            )
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.ndim != 3 or radar.shape[0] != 4:
            raise ValueError(f"expected radar [4,H,W], got {radar.shape}")
        resized = []
        for channel_index, channel in enumerate(radar):
            interpolation = cv2.INTER_NEAREST if channel_index == 0 else cv2.INTER_LINEAR
            resized.append(cv2.resize(channel, (self.input_width, self.input_height), interpolation=interpolation))
        return np.stack(resized).astype(np.float32)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        original_width, original_height = image.size
        mask = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        radar = self._load_radar(row)

        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        sx = self.input_width / float(original_width)
        sy = self.input_height / float(original_height)
        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=original_width,
            image_height=original_height,
            min_area_px=self.min_gt_area_px,
            object_class_names=("vehicle", "person"),
            max_distance_m=self.max_gt_distance_m,
        )

        boxes: List[List[float]] = []
        labels: List[int] = []
        roi_fields: List[List[float]] = []
        camera_yaw = math.radians(float(row.get("camera_yaw") or 0.0))
        for obj in objects:
            cx = float(obj["center_x"]) * sx
            cy = float(obj["center_y"]) * sy
            bw = float(obj["bbox_w"]) * sx
            bh = float(obj["bbox_h"]) * sy
            x0 = max(0.0, cx - 0.5 * bw)
            y0 = max(0.0, cy - 0.5 * bh)
            x1 = min(float(self.input_width), cx + 0.5 * bw)
            y1 = min(float(self.input_height), cy + 0.5 * bh)
            if x1 <= x0 or y1 <= y0:
                continue
            world_yaw = math.atan2(float(obj["yaw_sin"]), float(obj["yaw_cos"]))
            local_yaw = _wrap_pi(world_yaw - camera_yaw)
            boxes.append([x0, y0, x1, y1])
            labels.append(CLASS_TO_LABEL[str(obj["class_name"])])
            roi_fields.append(
                [
                    float(obj["local_x"]), float(obj["local_y"]), float(obj["local_z"]),
                    float(obj["size_x"]), float(obj["size_y"]), float(obj["size_z"]),
                    math.sin(local_yaw), math.cos(local_yaw),
                    float(obj["parked"]), float(obj["radar_support"]),
                ]
            )

        do_flip = self.training and bool(torch.rand(()) < self.flip_probability)
        if do_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            radar = np.flip(radar, axis=2).copy()
            for box, field in zip(boxes, roi_fields):
                old_x0, old_x1 = box[0], box[2]
                box[0] = float(self.input_width) - old_x1
                box[2] = float(self.input_width) - old_x0
                field[1] = -field[1]
                local_yaw = -math.atan2(field[6], field[7])
                field[6], field[7] = math.sin(local_yaw), math.cos(local_yaw)

        if self.training:
            if bool(torch.rand(()) < 0.35):
                image = ImageEnhance.Brightness(image).enhance(float(torch.empty(()).uniform_(0.8, 1.2)))
            if bool(torch.rand(()) < 0.35):
                image = ImageEnhance.Contrast(image).enhance(float(torch.empty(()).uniform_(0.8, 1.2)))

        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        radar_tensor = torch.from_numpy(np.ascontiguousarray(radar)).float()
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "roi_fields": torch.tensor(roi_fields, dtype=torch.float32).reshape(-1, 10),
            "segmentation": mask_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        metadata = {
            "index": index,
            "sample_id": str(row["sample_id"]),
            "original_width": original_width,
            "original_height": original_height,
            "camera_matrix_json": str(row.get("camera_matrix_json", "")),
            "camera_yaw_deg": float(row.get("camera_yaw") or 0.0),
            "frame_id": str(row.get("frame_id", "")),
            "radar_frame_id": str(row.get("radar_frame_id", "")),
            "timestamp": float(row.get("timestamp") or 0.0),
            "radar_timestamp": float(row.get("radar_timestamp") or 0.0),
            "flipped": do_flip,
        }
        return rgb, radar_tensor, target, metadata


def detection_collate(batch):
    rgb, radar, targets, metadata = zip(*batch)
    return list(rgb), list(radar), list(targets), list(metadata)

