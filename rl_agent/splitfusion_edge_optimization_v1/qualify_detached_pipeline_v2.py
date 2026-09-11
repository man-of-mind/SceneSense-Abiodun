#!/usr/bin/env python3
"""Reuse the qualified detached-pipeline gate with the additive v2 tail."""

from __future__ import annotations

from rl_agent.splitfusion_edge_optimization_v1 import (
    qualify_detached_pipeline as qualification,
)

from .detached_tail_v2 import DetachedOptimizedTailAdapterV2


def main(argv: list[str] | None = None) -> int:
    qualification.DetachedOptimizedTailAdapter = DetachedOptimizedTailAdapterV2
    return qualification.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
