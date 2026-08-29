from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model import MODEL_SIZE_WH, NATIVE_STRIDE

TOPK_PER_CLASS = 120
LOCAL_MAX_KERNEL = 3


def inference_expm1_float64(value: torch.Tensor) -> torch.Tensor:
    """Unbounded inference-only metric decode in float64."""
    return torch.expm1(value.to(torch.float64))


def inference_exp_float64(value: torch.Tensor) -> torch.Tensor:
    """Unbounded inference-only dimension decode in float64."""
    return torch.exp(value.to(torch.float64))


def local_maxima(scores: torch.Tensor) -> torch.Tensor:
    pooled = F.max_pool2d(scores, LOCAL_MAX_KERNEL, stride=1, padding=LOCAL_MAX_KERNEL // 2)
    return scores * pooled.eq(scores).to(scores.dtype)


def decode_branch(branch: Mapping[str, torch.Tensor], class_name: str,
                  depth_anchors: torch.Tensor, depth_delta: torch.Tensor,
                  camera_matrix: np.ndarray, intrinsic_model: np.ndarray,
                  score_threshold: float, topk: int = TOPK_PER_CLASS) -> list[dict[str, Any]]:
    raw = {name: value[0].float() if value.ndim == 4 else value.float() for name, value in branch.items()}
    scores = local_maxima(torch.sigmoid(raw["heatmap"]).unsqueeze(0))[0, 0]
    flat = scores.reshape(-1)
    count = min(int(topk), flat.numel())
    values, indices = torch.topk(flat, count)
    grid_width = scores.shape[1]
    records: list[dict[str, Any]] = []
    anchors = depth_anchors.to(raw["heatmap"].device, dtype=torch.float32)
    delta = depth_delta.to(raw["heatmap"].device, dtype=torch.float32)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_model, dtype=np.float64)
    for score_tensor, index_tensor in zip(values, indices):
        score = float(score_tensor.item())
        if score < float(score_threshold):
            break
        index = int(index_tensor.item())
        cell_y, cell_x = divmod(index, grid_width)
        subcell = torch.sigmoid(raw["subcell"][:, cell_y, cell_x])
        grid_anchor_x = cell_x + float(subcell[0].item())
        grid_anchor_y = cell_y + float(subcell[1].item())
        box_delta = raw["box_center_delta"][:, cell_y, cell_x]
        box_center_x = NATIVE_STRIDE * (grid_anchor_x + float(box_delta[0].item()))
        box_center_y = NATIVE_STRIDE * (grid_anchor_y + float(box_delta[1].item()))
        box_wh = NATIVE_STRIDE * F.softplus(raw["box_wh"][:, cell_y, cell_x])
        physical_delta = raw["physical_ray_delta"][:, cell_y, cell_x]
        u_physical = NATIVE_STRIDE * (grid_anchor_x + float(physical_delta[0].item()))
        v_physical = NATIVE_STRIDE * (grid_anchor_y + float(physical_delta[1].item()))
        logits = raw["depth_bin_logits"][:, cell_y, cell_x]
        residuals = raw["depth_bin_residuals"][:, cell_y, cell_x]
        z = (torch.softmax(logits, dim=0) * (anchors + delta * residuals)).sum()
        depth = max(0.0, float(inference_expm1_float64(z).item()))
        local = np.asarray([
            depth,
            depth * (u_physical - intrinsic[0, 2]) / intrinsic[0, 0],
            depth * (intrinsic[1, 2] - v_physical) / intrinsic[1, 1],
        ], dtype=np.float64)
        world = (matrix @ np.asarray([local[0], local[1], local[2], 1.0]))[:3]
        dimensions = inference_exp_float64(raw["log_dimensions"][:, cell_y, cell_x])
        yaw = F.normalize(raw["yaw_sincos"][:, cell_y, cell_x], dim=0, eps=1e-6)
        width, height = float(box_wh[0].item()), float(box_wh[1].item())
        record: dict[str, Any] = {
            "class_index": 0 if class_name == "vehicle" else 1,
            "class_name": class_name, "score": score,
            "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
            "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
            "size_x": float(dimensions[0].item()), "size_y": float(dimensions[1].item()),
            "size_z": float(dimensions[2].item()),
            "yaw_sin": float(yaw[0].item()), "yaw_cos": float(yaw[1].item()),
            "parked_score": (float(torch.sigmoid(raw["parked"][0, cell_y, cell_x]).item())
                             if class_name == "vehicle" else 0.0),
            "radar_support_score": float(torch.sigmoid(raw["radar_support"][0, cell_y, cell_x]).item()),
            "center_x_px": float(box_center_x), "center_y_px": float(box_center_y),
            "bbox_w_px": width, "bbox_h_px": height,
            "bbox_x0": float(box_center_x - width / 2.0), "bbox_y0": float(box_center_y - height / 2.0),
            "bbox_x1": float(box_center_x + width / 2.0), "bbox_y1": float(box_center_y + height / 2.0),
            "physical_ray_x_px": float(u_physical), "physical_ray_y_px": float(v_physical),
            "actor_forward_depth_m": depth,
            "native_cell_x": cell_x, "native_cell_y": cell_y,
        }
        nonfinite = [
            name for name, value in record.items()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        ]
        if nonfinite:
            raise FloatingPointError(
                f"non-finite scored detection class={class_name} cell=({cell_y},{cell_x}) fields={nonfinite}"
            )
        records.append(record)
    return records


def decode_geometry(outputs: Mapping[str, Any], depth_anchors: torch.Tensor, depth_delta: torch.Tensor,
                    camera_matrix: np.ndarray, intrinsic_model: np.ndarray,
                    score_threshold: float, topk: int = TOPK_PER_CLASS) -> list[dict[str, Any]]:
    records = []
    for class_name in ("vehicle", "person"):
        records.extend(decode_branch(
            outputs["objects"][class_name], class_name, depth_anchors, depth_delta,
            camera_matrix, intrinsic_model, score_threshold, topk,
        ))
    records.sort(key=lambda value: (-float(value["score"]), str(value["class_name"])))
    return records


def intrinsic_from_row(row: Mapping[str, str]) -> np.ndarray:
    sx, sy = MODEL_SIZE_WH[0] / float(row["camera_width"]), MODEL_SIZE_WH[1] / float(row["camera_height"])
    return np.asarray([
        [float(row["camera_fx"]) * sx, 0.0, float(row["camera_cx"]) * sx],
        [0.0, float(row["camera_fy"]) * sy, float(row["camera_cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def camera_matrix_from_row(row: Mapping[str, str]) -> np.ndarray:
    import json
    return np.asarray(json.loads(row["camera_matrix_json"]), dtype=np.float64)


def external_record(record: Mapping[str, Any], sample_id: str, frame_id: str,
                    prediction_index: int) -> dict[str, Any]:
    fields = (
        "class_name", "score", "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
        "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score", "radar_support_score",
        "center_x_px", "center_y_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    )
    return {
        "sample_id": sample_id, "frame_id": frame_id, "prediction_index": prediction_index,
        **{name: record[name] for name in fields},
    }


EXTERNAL_FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
)
