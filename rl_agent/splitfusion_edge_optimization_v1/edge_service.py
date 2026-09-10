#!/usr/bin/env python3
"""Run the proven timing-diagnostic service with the optimized edge preload."""

from __future__ import annotations

from rl_agent.splitfusion_timing_diagnostic_v1 import edge_service as diagnostic

from .edge_preload import preload_optimized_edge


def main(argv: list[str] | None = None) -> int:
    diagnostic.preload_instrumented_edge = preload_optimized_edge
    return diagnostic.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
