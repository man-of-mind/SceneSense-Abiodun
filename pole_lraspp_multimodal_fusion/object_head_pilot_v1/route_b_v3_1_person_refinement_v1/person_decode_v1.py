#!/usr/bin/env python3
"""Vehicle-bit-preserving native decode plus the private refined person decode."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
for path in (str(NATIVE_PACKAGE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import model_v1 as native  # noqa: E402
from decode_v1 import TOPK_PER_CLASS, decode_native_objects, native_local_maxima  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    REG_BBOX_WH, REG_DIMS, REG_PARKED, REG_RADAR_SUPPORT, REG_YAW,
    transform_point,
)


def _softplus(value: float) -> float:
    return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))


def decode_person_refinement(
    base_object: torch.Tensor, refinement: dict[str, torch.Tensor], *,
    camera_matrix: np.ndarray, intrinsic_model: np.ndarray,
    range_edges: Sequence[float], offset_caps: Sequence[float],
    score_threshold: float, topk: int = TOPK_PER_CLASS,
    model_size: Sequence[int] = native.MODEL_SIZE,
) -> list[dict[str, float]]:
    if base_object.ndim == 4:
        base_object = base_object[0]
        refinement = {key: value[0] for key, value in refinement.items()}
    objectness = torch.sigmoid(
        base_object[1].float() + refinement["objectness_residual"][0].float()
    )
    quality = torch.sigmoid(refinement["localization_quality"][0].float())
    combined = objectness * quality
    peaks = native_local_maxima(combined[None, None])[0, 0].detach().cpu()
    flat = peaks.reshape(-1)
    count = min(int(topk), int(flat.numel()))
    if count <= 0:
        return []
    scores, indices = torch.topk(flat, k=count)
    regs = base_object[native.SL_REG].float().detach().cpu().numpy()
    grid_offset = base_object[native.SL_OFFSET].float().clamp(0.0, 1.0).detach().cpu().numpy()
    bin_logits = refinement["range_bin_logits"].float().detach().cpu()
    residual = torch.tanh(refinement["range_residual"][0].float()).detach().cpu().numpy()
    caps = np.asarray(offset_caps, dtype=np.float64)
    projected_offset = (
        torch.tanh(refinement["projected_center_offset"].float()).detach().cpu().numpy()
        * caps[:, None, None]
    )
    quality_np = quality.detach().cpu().numpy()
    edges = np.asarray(range_edges, dtype=np.float64)
    centers, half_widths = (edges[:-1] + edges[1:]) / 2.0, (edges[1:] - edges[:-1]) / 2.0
    model_width, model_height = int(model_size[0]), int(model_size[1])
    grid_h, grid_w = peaks.shape
    raw = base_object.float().detach().cpu()
    predictions: list[dict[str, float]] = []
    for score_tensor, index_tensor in zip(scores, indices):
        score = float(score_tensor.item())
        if score < float(score_threshold):
            break
        index = int(index_tensor.item())
        cell_y, cell_x = divmod(index, grid_w)
        selected_bin = int(torch.argmax(bin_logits[:, cell_y, cell_x]).item())
        depth = float(np.clip(
            centers[selected_bin] + residual[cell_y, cell_x] * half_widths[selected_bin],
            0.05, edges[-1],
        ))
        center_x_px = (cell_x + float(grid_offset[0, cell_y, cell_x])) * native.NATIVE_STRIDE
        center_y_px = (cell_y + float(grid_offset[1, cell_y, cell_x])) * native.NATIVE_STRIDE
        projected_u = center_x_px + float(projected_offset[0, cell_y, cell_x]) * native.NATIVE_STRIDE
        projected_v = center_y_px + float(projected_offset[1, cell_y, cell_x]) * native.NATIVE_STRIDE
        right = (projected_u - intrinsic_model[0, 2]) * depth / intrinsic_model[0, 0]
        up = -(projected_v - intrinsic_model[1, 2]) * depth / intrinsic_model[1, 1]
        local = np.asarray([depth, right, up], dtype=np.float64)
        world = transform_point(camera_matrix, local)
        dims = np.maximum(regs[REG_DIMS, cell_y, cell_x], 0.0)
        yaw_sin, yaw_cos = regs[REG_YAW, cell_y, cell_x]
        yaw_norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        box_w = _softplus(float(regs[REG_BBOX_WH.start, cell_y, cell_x])) * model_width
        box_h = _softplus(float(regs[REG_BBOX_WH.start + 1, cell_y, cell_x])) * model_height
        predictions.append({
            "class_index": 1.0, "class_name": "person", "score": score,
            "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
            "local_x": depth, "local_y": float(right), "local_z": float(up),
            "size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2]),
            "yaw_sin": float(yaw_sin / yaw_norm), "yaw_cos": float(yaw_cos / yaw_norm),
            "parked_score": float(torch.sigmoid(raw[native.HEATMAP_CHANNELS + REG_PARKED, cell_y, cell_x]).item()),
            "radar_support_score": float(torch.sigmoid(raw[native.HEATMAP_CHANNELS + REG_RADAR_SUPPORT, cell_y, cell_x]).item()),
            "center_x_px": center_x_px, "center_y_px": center_y_px,
            "bbox_w_px": box_w, "bbox_h_px": box_h,
            "bbox_x0": center_x_px - box_w / 2.0, "bbox_y0": center_y_px - box_h / 2.0,
            "bbox_x1": center_x_px + box_w / 2.0, "bbox_y1": center_y_px + box_h / 2.0,
            "native_cell_x": float(cell_x), "native_cell_y": float(cell_y),
            "localization_quality": float(quality_np[cell_y, cell_x]),
        })
    return predictions


def decode_all(outputs: dict[str, Any], *, camera_matrix: np.ndarray,
               intrinsic_model: np.ndarray, range_edges: Sequence[float],
               offset_caps: Sequence[float], score_threshold: float,
               model_size: Sequence[int]) -> list[dict[str, float]]:
    native_predictions = decode_native_objects(
        outputs["object"], camera_matrix=camera_matrix,
        score_threshold=score_threshold, topk=TOPK_PER_CLASS,
        stride=native.NATIVE_STRIDE, model_size=model_size,
        object_class_names=("vehicle", "person"),
    )
    vehicles = [value for value in native_predictions if value["class_name"] == "vehicle"]
    persons = decode_person_refinement(
        outputs["object"], outputs["person_refinement"], camera_matrix=camera_matrix,
        intrinsic_model=intrinsic_model, range_edges=range_edges,
        offset_caps=offset_caps, score_threshold=score_threshold,
        model_size=model_size,
    )
    return sorted(vehicles + persons, key=lambda item: (-float(item["score"]), item["class_name"]))
