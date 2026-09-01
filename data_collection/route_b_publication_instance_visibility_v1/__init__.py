"""Renderer-derived Route B publication visibility ground truth."""

from .core import (
    decode_instance_bgra,
    instance_mask,
    measure_visibility,
    prove_actor_id_mapping,
    relative_transform_matrix,
    reproduce_transform_matrix,
)

__all__ = [
    "decode_instance_bgra",
    "instance_mask",
    "measure_visibility",
    "prove_actor_id_mapping",
    "relative_transform_matrix",
    "reproduce_transform_matrix",
]
