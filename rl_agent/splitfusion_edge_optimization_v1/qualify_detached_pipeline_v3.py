#!/usr/bin/env python3
"""Reuse the detached-pipeline gate with the complete v3 edge tail."""

from __future__ import annotations

from rl_agent.splitfusion_edge_optimization_v1 import (
    qualify_detached_pipeline as qualification,
)

from .detached_tail_v3 import DetachedOptimizedTailAdapterV3


def main(argv: list[str] | None = None) -> int:
    qualification.DetachedOptimizedTailAdapter = DetachedOptimizedTailAdapterV3
    return qualification.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
