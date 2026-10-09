"""Inference-only features of one SplitFusion-FCOS detection record.

Only fields present in a detection record (detections.csv schema) are read.  No GT,
no radar, no dataset path.  Model pixel space is 768x432 (content area; bottom pad excluded).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

MODEL_W, MODEL_H = 768.0, 432.0
EDGE_PX = 2.0
# Record fields the correction is allowed to read (asserted by tests).
INPUT_FIELDS = (
    "class_name", "score", "local_x", "local_y", "local_z", "world_x", "world_y", "world_z",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "level_index", "depth_residual",
)
FEATURE_NAMES = (
    "log_depth", "log_depth_sq", "log_box_h", "log_box_w", "abs_center_x_norm", "bottom_y_norm",
    "edge_touch", "edge_dist_norm", "score", "level_index", "depth_residual",
)


def record_features(rec: Mapping[str, Any]) -> np.ndarray:
    depth = max(float(rec["local_x"]), 0.5)  # camera-forward depth (== actor_forward_depth_m)
    x0, y0, x1, y1 = (float(rec[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    ld = math.log(depth)
    touch = float(x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= MODEL_W - EDGE_PX or y1 >= MODEL_H - EDGE_PX)
    edge_dist = min(x0, y0, MODEL_W - x1, MODEL_H - y1)
    return np.array([
        ld, ld * ld, math.log(h), math.log(w),
        abs((x0 + x1) / 2.0 - MODEL_W / 2.0) / (MODEL_W / 2.0), y1 / MODEL_H,
        touch, min(max(edge_dist, 0.0) / 100.0, 1.0),
        float(rec["score"]), float(rec["level_index"]), float(rec["depth_residual"]),
    ], dtype=np.float64)


def edge_touching(rec: Mapping[str, Any]) -> bool:
    return bool(record_features(rec)[6] > 0.5)
