"""Detached publication handoff for the v2 optimized tail candidate."""

from __future__ import annotations

from .detached_tail import DetachedOptimizedTailAdapter
from .optimized_tail_v2 import OptimizedFrozenP025TailAdapterV2


class DetachedOptimizedTailAdapterV2(
    DetachedOptimizedTailAdapter, OptimizedFrozenP025TailAdapterV2
):
    """Combine the qualified detached handoff with v2 compute semantics.

    The method-resolution order is intentional: ``compute_product`` comes from
    the detached adapter and its ``super().__call__`` resolves to the v2 tail.
    Serialization remains CPU-only and frame-scoped.
    """

    pass


def require_v2_mro() -> tuple[str, ...]:
    names = tuple(cls.__name__ for cls in DetachedOptimizedTailAdapterV2.__mro__)
    required = (
        "DetachedOptimizedTailAdapterV2",
        "DetachedOptimizedTailAdapter",
        "OptimizedFrozenP025TailAdapterV2",
        "OptimizedFrozenP025TailAdapter",
    )
    if names[: len(required)] != required:
        raise RuntimeError(f"detached v2 method-resolution drift: {names}")
    return names


require_v2_mro()
