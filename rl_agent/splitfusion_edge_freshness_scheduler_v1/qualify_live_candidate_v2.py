#!/usr/bin/env python3
"""SFD1 CUDA parity gate for the v2 resident UE/edge path."""

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v2 import (
    preload_detached_optimized_edge_v2,
)

from . import qualify_live_candidate as qualification


def main() -> int:
    qualification.preload_detached_optimized_edge = (
        preload_detached_optimized_edge_v2
    )
    return qualification.main()


if __name__ == "__main__":
    raise SystemExit(main())
