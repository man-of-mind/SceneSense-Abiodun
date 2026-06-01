from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


OBJECT_REG_CHANNELS = 10
OBJECT_OUTPUT_CHANNELS = 1 + OBJECT_REG_CHANNELS
REG_LOCAL_XYZ = slice(0, 3)
REG_DIMS = slice(3, 6)
REG_YAW = slice(6, 8)
REG_PARKED = 8
REG_RADAR_SUPPORT = 9


def parse_matrix(value: str) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        arr = np.asarray(json.loads(value), dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return arr


def transform_point(matrix: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    point = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    return (matrix @ point)[:3]


def load_object_boxes(path: Path) -> Dict[str, List[Dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def valid_vehicle_objects(
    rows: Sequence[Dict[str, str]],
    *,
    image_width: int,
    image_height: int,
    min_area_px: float,
) -> List[Dict[str, float]]:
    objects: List[Dict[str, float]] = []
    for row in rows:
        if row.get("label") != "vehicle" or row.get("gt_source") != "actor":
            continue
        if row.get("object_sensor_x", "") == "" or row.get("object_world_x", "") == "":
            continue
        area = _float(row, "gt_bbox_area_px")
        if area < float(min_area_px):
            continue
        cx = _float(row, "gt_center_x")
        cy = _float(row, "gt_center_y")
        if not (0.0 <= cx < float(image_width) and 0.0 <= cy < float(image_height)):
            continue
        yaw_rad = math.radians(_float(row, "object_yaw_deg"))
        objects.append(
            {
                "center_x": cx,
                "center_y": cy,
                "bbox_w": _float(row, "gt_bbox_w"),
                "bbox_h": _float(row, "gt_bbox_h"),
                "area": area,
                "local_x": _float(row, "object_sensor_x"),
                "local_y": _float(row, "object_sensor_y"),
                "local_z": _float(row, "object_sensor_z"),
                "world_x": _float(row, "object_world_x"),
                "world_y": _float(row, "object_world_y"),
                "world_z": _float(row, "object_world_z"),
                "size_x": max(0.01, _float(row, "gt_size_x_m")),
                "size_y": max(0.01, _float(row, "gt_size_y_m")),
                "size_z": max(0.01, _float(row, "gt_size_z_m")),
                "yaw_sin": math.sin(yaw_rad),
                "yaw_cos": math.cos(yaw_rad),
                "parked": float(_float(row, "parked_label") >= 0.5),
                "radar_support": float(_float(row, "radar_support_points") > 0.0),
            }
        )
    return objects


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius: int) -> None:
    radius = max(0, int(radius))
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(heatmap.shape[1], int(round(cx)) + radius + 1)
    y1 = min(heatmap.shape[0], int(round(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    if radius <= 0:
        heatmap[int(round(cy)), int(round(cx))] = 1.0
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sigma = max(1.0, float(radius) / 2.0)
    values = np.exp(-((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], values.astype(np.float32))


def build_object_targets(
    *,
    objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    heatmap_radius_px: int,
    max_objects: int,
) -> Dict[str, torch.Tensor]:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_width, original_height = int(original_size[0]), int(original_size[1])
    sx = input_width / max(1.0, float(original_width))
    sy = input_height / max(1.0, float(original_height))
    heatmap = np.zeros((input_height, input_width), dtype=np.float32)
    regression = np.zeros((OBJECT_REG_CHANNELS, input_height, input_width), dtype=np.float32)
    reg_mask = np.zeros((1, input_height, input_width), dtype=np.float32)
    gt_objects = np.zeros((int(max_objects), 9), dtype=np.float32)
    gt_count = 0
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        cx = float(obj["center_x"]) * sx
        cy = float(obj["center_y"]) * sy
        ix = int(round(cx))
        iy = int(round(cy))
        if ix < 0 or iy < 0 or ix >= input_width or iy >= input_height:
            continue
        draw_gaussian(heatmap, cx, cy, heatmap_radius_px)
        # The gaussian is evaluated at integer pixel coordinates; with a sub-pixel
        # (cx, cy) the peak pixel only reaches exp(-d^2/(2 sigma^2)) < 1.0. The
        # focal heatmap loss treats positives via target == 1.0, so without this
        # the previous run had pos_count == 0 every batch and the center head
        # never learned (learned_object_f1 = 0).
        heatmap[iy, ix] = 1.0
        regression[:, iy, ix] = np.array(
            [
                obj["local_x"],
                obj["local_y"],
                obj["local_z"],
                obj["size_x"],
                obj["size_y"],
                obj["size_z"],
                obj["yaw_sin"],
                obj["yaw_cos"],
                obj["parked"],
                obj["radar_support"],
            ],
            dtype=np.float32,
        )
        reg_mask[0, iy, ix] = 1.0
        if gt_count < int(max_objects):
            gt_objects[gt_count] = np.array(
                [
                    obj["world_x"],
                    obj["world_y"],
                    obj["world_z"],
                    obj["size_x"],
                    obj["size_y"],
                    obj["size_z"],
                    obj["yaw_sin"],
                    obj["yaw_cos"],
                    obj["parked"],
                ],
                dtype=np.float32,
            )
            gt_count += 1
    if gt_count > 0:
        assert float(heatmap.max()) >= 0.999, (
            "object center heatmap target has no peak >= 1.0 despite gt_count > 0; "
            "focal loss positive count would be zero (learned_object_f1 = 0 regression)."
        )
    return {
        "center_heatmap": torch.from_numpy(heatmap[None, :, :]),
        "regression": torch.from_numpy(regression),
        "regression_mask": torch.from_numpy(reg_mask),
        "gt_objects": torch.from_numpy(gt_objects),
        "gt_count": torch.tensor(gt_count, dtype=torch.long),
    }


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, *, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.ge(1.0 - 1e-3).to(logits.dtype)
    neg = (1.0 - pos).to(logits.dtype)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg
    pos_count = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / pos_count


def multitask_object_loss(outputs: torch.Tensor, targets: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    center_logits = outputs[:, 0:1]
    regs = outputs[:, 1:]
    heatmap = targets["center_heatmap"].to(outputs.device)
    reg_target = targets["regression"].to(outputs.device)
    reg_mask = targets["regression_mask"].to(outputs.device)
    center_loss = focal_heatmap_loss(center_logits, heatmap)
    denom = reg_mask.sum().clamp(min=1.0)
    mask = reg_mask.expand_as(regs)
    loc_loss = F.smooth_l1_loss(regs[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ], reg_target[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ], reduction="sum") / denom
    dim_loss = F.smooth_l1_loss(regs[:, REG_DIMS] * mask[:, REG_DIMS], reg_target[:, REG_DIMS] * mask[:, REG_DIMS], reduction="sum") / denom
    yaw_pred = F.normalize(regs[:, REG_YAW], dim=1)
    yaw_loss = F.smooth_l1_loss(yaw_pred * mask[:, REG_YAW], reg_target[:, REG_YAW] * mask[:, REG_YAW], reduction="sum") / denom
    parked_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_PARKED : REG_PARKED + 1],
        reg_target[:, REG_PARKED : REG_PARKED + 1],
        weight=reg_mask,
        reduction="sum",
    ) / denom
    radar_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        reg_target[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        weight=reg_mask,
        reduction="sum",
    ) / denom
    total = (
        float(weights.get("center", 1.0)) * center_loss
        + float(weights.get("location", 0.05)) * loc_loss
        + float(weights.get("dimensions", 0.2)) * dim_loss
        + float(weights.get("yaw", 0.05)) * yaw_loss
        + float(weights.get("parked", 0.2)) * parked_loss
        + float(weights.get("radar_support", 0.1)) * radar_loss
    )
    parts = {
        "center_loss": float(center_loss.detach().item()),
        "loc_loss": float(loc_loss.detach().item()),
        "dim_loss": float(dim_loss.detach().item()),
        "yaw_loss": float(yaw_loss.detach().item()),
        "parked_loss": float(parked_loss.detach().item()),
        "radar_support_loss": float(radar_loss.detach().item()),
    }
    return total, parts


def decode_objects(
    object_output: torch.Tensor,
    *,
    camera_matrix: np.ndarray,
    topk: int,
    score_threshold: float,
    nms_radius_px: int,
) -> List[Dict[str, float]]:
    if object_output.ndim == 4:
        object_output = object_output[0]
    center = torch.sigmoid(object_output[0]).detach().cpu()
    regs = object_output[1:].detach().cpu().numpy()
    flat = center.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    if k <= 0:
        return []
    scores, indices = torch.topk(flat, k=k)
    height, width = int(center.shape[0]), int(center.shape[1])
    occupied = np.zeros((height, width), dtype=bool)
    predictions: List[Dict[str, float]] = []
    for score_t, index_t in zip(scores, indices):
        score = float(score_t.item())
        if score < float(score_threshold):
            continue
        idx = int(index_t.item())
        y, x = divmod(idx, width)
        y0, y1 = max(0, y - int(nms_radius_px)), min(height, y + int(nms_radius_px) + 1)
        x0, x1 = max(0, x - int(nms_radius_px)), min(width, x + int(nms_radius_px) + 1)
        if occupied[y0:y1, x0:x1].any():
            continue
        occupied[y0:y1, x0:x1] = True
        local = regs[REG_LOCAL_XYZ, y, x]
        dims = np.maximum(regs[REG_DIMS, y, x], 0.0)
        yaw_sin, yaw_cos = regs[REG_YAW, y, x]
        norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        world = transform_point(camera_matrix, local)
        predictions.append(
            {
                "score": score,
                "center_x_px": float(x),
                "center_y_px": float(y),
                "local_x": float(local[0]),
                "local_y": float(local[1]),
                "local_z": float(local[2]),
                "world_x": float(world[0]),
                "world_y": float(world[1]),
                "world_z": float(world[2]),
                "size_x": float(dims[0]),
                "size_y": float(dims[1]),
                "size_z": float(dims[2]),
                "yaw_sin": float(yaw_sin / norm),
                "yaw_cos": float(yaw_cos / norm),
                "parked_score": float(torch.sigmoid(object_output[1 + REG_PARKED, y, x]).item()),
                "radar_support_score": float(torch.sigmoid(object_output[1 + REG_RADAR_SUPPORT, y, x]).item()),
            }
        )
    return predictions


def greedy_match_predictions(
    predictions: Sequence[Dict[str, float]],
    gt_objects: Sequence[Dict[str, float]],
    *,
    max_distance_m: float,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(gt_objects):
            dist = float(np.hypot(float(pred["world_x"]) - float(gt["world_x"]), float(pred["world_y"]) - float(gt["world_y"])))
            if dist <= float(max_distance_m):
                candidates.append((dist, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0])
    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, dist))
    return matches
