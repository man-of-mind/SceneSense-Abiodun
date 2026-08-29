#!/usr/bin/env python3
"""Unchanged vehicle decoder plus the person-private visible-anchor decoder."""

from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

_DECODE_SPEC = importlib.util.spec_from_file_location(
    "route_b_v3_1_native_grid_decode_for_visible_anchor_v1",
    NATIVE_PACKAGE / "decode_v1.py",
)
if _DECODE_SPEC is None or _DECODE_SPEC.loader is None:
    raise ImportError("unable to load frozen native-grid decoder")
_native_decode = importlib.util.module_from_spec(_DECODE_SPEC)
_DECODE_SPEC.loader.exec_module(_native_decode)
TOPK_PER_CLASS = _native_decode.TOPK_PER_CLASS
decode_native_objects = _native_decode.decode_native_objects
native_local_maxima = _native_decode.native_local_maxima
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))
from pole_lraspp_multimodal_fusion.object_targets import transform_point  # noqa: E402

STRIDE = 4
MODEL_SIZE = (768, 432)


def decode_depth(raw: float, bounds_m: Sequence[float]) -> float:
    low, high = (float(value) for value in bounds_m)
    normalized = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, float(raw)))))
    return math.exp(math.log(low) + normalized * (math.log(high) - math.log(low)))


def decode_private_person(private: Mapping[str, torch.Tensor], *,
                          camera_matrix: np.ndarray, intrinsic_model: np.ndarray,
                          score_threshold: float, offset_scales: Mapping[str, float],
                          depth_bounds_m: Sequence[float], topk: int = TOPK_PER_CLASS,
                          model_size: Sequence[int] = MODEL_SIZE) -> list[dict[str, float]]:
    values = {name: tensor[0] if tensor.ndim == 4 else tensor for name, tensor in private.items()}
    heat = torch.sigmoid(values["visible_heatmap"].float())
    peaks = native_local_maxima(heat.unsqueeze(0))[0, 0].detach().cpu()
    flat = peaks.reshape(-1)
    scores, indices = torch.topk(flat, k=min(int(topk), int(flat.numel())))
    arrays = {name: tensor.float().detach().cpu().numpy() for name, tensor in values.items()}
    grid_h, grid_w = heat.shape[-2:]
    model_w, model_h = (int(value) for value in model_size)
    box_scale = float(offset_scales["box_center_grid_cells"])
    ray_scale = float(offset_scales["physical_ray_grid_cells"])
    output: list[dict[str, float]] = []
    for score_tensor, index_tensor in zip(scores, indices):
        score = float(score_tensor.item())
        if score < float(score_threshold):
            break
        cell_y, cell_x = divmod(int(index_tensor.item()), grid_w)
        subcell = np.clip(arrays["visible_subcell_offset"][:, cell_y, cell_x], 0.0, 1.0)
        visible_grid_x, visible_grid_y = cell_x + float(subcell[0]), cell_y + float(subcell[1])
        box_offset = arrays["visible_to_box_center_offset"][:, cell_y, cell_x] * box_scale
        ray_offset = arrays["visible_to_physical_ray_offset"][:, cell_y, cell_x] * ray_scale
        box_center_x = (visible_grid_x + float(box_offset[0])) * STRIDE
        box_center_y = (visible_grid_y + float(box_offset[1])) * STRIDE
        physical_u = (visible_grid_x + float(ray_offset[0])) * STRIDE
        physical_v = (visible_grid_y + float(ray_offset[1])) * STRIDE
        depth = decode_depth(
            float(arrays["positive_camera_forward_depth"][0, cell_y, cell_x]), depth_bounds_m,
        )
        local = np.asarray([
            depth,
            (physical_u - float(intrinsic_model[0, 2])) * depth / float(intrinsic_model[0, 0]),
            (float(intrinsic_model[1, 2]) - physical_v) * depth / float(intrinsic_model[1, 1]),
        ], dtype=np.float64)
        world = transform_point(camera_matrix, local)
        wh_raw = arrays["full_box_wh"][:, cell_y, cell_x]
        wh = np.log1p(np.exp(-np.abs(wh_raw))) + np.maximum(wh_raw, 0.0)
        box_w, box_h = float(wh[0]) * model_w, float(wh[1]) * model_h
        dims = np.maximum(arrays["person_dimensions"][:, cell_y, cell_x], 0.0)
        yaw = arrays["person_yaw"][:, cell_y, cell_x]
        yaw_norm = max(1e-6, float(np.hypot(yaw[0], yaw[1])))
        radar_raw = float(arrays["radar_support"][0, cell_y, cell_x])
        radar_score = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, radar_raw))))
        output.append({
            "class_index": 1.0, "class_name": "person", "score": score,
            "center_x_px": box_center_x, "center_y_px": box_center_y,
            "bbox_w_px": box_w, "bbox_h_px": box_h,
            "bbox_x0": box_center_x - box_w / 2.0,
            "bbox_y0": box_center_y - box_h / 2.0,
            "bbox_x1": box_center_x + box_w / 2.0,
            "bbox_y1": box_center_y + box_h / 2.0,
            "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
            "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
            "size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2]),
            "yaw_sin": float(yaw[0] / yaw_norm), "yaw_cos": float(yaw[1] / yaw_norm),
            "parked_score": 0.0, "radar_support_score": radar_score,
            "visible_anchor_x_px": visible_grid_x * STRIDE,
            "visible_anchor_y_px": visible_grid_y * STRIDE,
            "physical_ray_x_px": physical_u, "physical_ray_y_px": physical_v,
            "predicted_depth_m": depth,
        })
    return output


def decode_all(outputs: Mapping[str, Any], *, camera_matrix: np.ndarray,
               intrinsic_model: np.ndarray, score_threshold: float,
               offset_scales: Mapping[str, float], depth_bounds_m: Sequence[float],
               topk: int = TOPK_PER_CLASS,
               model_size: Sequence[int] = MODEL_SIZE) -> list[dict[str, float]]:
    inherited = decode_native_objects(
        outputs["object"], camera_matrix=camera_matrix, score_threshold=score_threshold,
        topk=topk, model_size=model_size,
    )
    vehicles = [row for row in inherited if row["class_name"] == "vehicle"]
    persons = decode_private_person(
        outputs["person_private"], camera_matrix=camera_matrix, intrinsic_model=intrinsic_model,
        score_threshold=score_threshold, offset_scales=offset_scales,
        depth_bounds_m=depth_bounds_m, topk=topk, model_size=model_size,
    )
    result = vehicles + persons
    result.sort(key=lambda item: (-float(item["score"]), str(item["class_name"])))
    return result


def algebraic_roundtrip(local_xyz: Sequence[float], intrinsic: np.ndarray) -> dict[str, Any]:
    depth, right, up = (float(value) for value in local_xyz)
    u = float(intrinsic[0, 2]) + float(intrinsic[0, 0]) * right / depth
    v = float(intrinsic[1, 2]) - float(intrinsic[1, 1]) * up / depth
    reconstructed = np.asarray([
        depth,
        (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0]),
        (float(intrinsic[1, 2]) - v) * depth / float(intrinsic[1, 1]),
    ], dtype=np.float64)
    expected = np.asarray(local_xyz, dtype=np.float64)
    error = float(np.max(np.abs(reconstructed - expected)))
    return {
        "local_xyz": expected.tolist(), "projected_uv": [u, v],
        "reconstructed_local_xyz": reconstructed.tolist(), "max_abs_error_m": error,
        "pass": bool(np.isfinite(reconstructed).all() and error <= 1e-9),
    }
