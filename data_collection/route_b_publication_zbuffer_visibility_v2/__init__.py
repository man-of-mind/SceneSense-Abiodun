"""Registered renderer z-buffer visibility protocol v2."""

from .core import (
    TAU_EMPTY_M,
    TAU_MATCH_M,
    ZBufferVisibilityError,
    compute_zbuffer_visibility,
    decode_depth_bgra,
)

__all__ = [
    "TAU_EMPTY_M",
    "TAU_MATCH_M",
    "ZBufferVisibilityError",
    "compute_zbuffer_visibility",
    "decode_depth_bgra",
]
