"""The evaluator-facing registered z-buffer visibility primitives."""

from data_collection.route_b_publication_zbuffer_visibility_v2.core import (
    TAU_EMPTY_M,
    TAU_MATCH_M,
    compute_zbuffer_visibility,
    decode_depth_bgra,
    mask_iou,
)

__all__ = [
    "TAU_EMPTY_M",
    "TAU_MATCH_M",
    "compute_zbuffer_visibility",
    "decode_depth_bgra",
    "mask_iou",
]
