#!/usr/bin/env python3
"""M3.1 v2 map-server trampoline: the profiled Phase-6 direct map with map-side class preservation.

The file name keeps the retained process marker substring ``map_profiled_v1.py`` (sampler role, leftover markers).
It installs ``splitfusion_m3_route_b_v2.map_class`` and then runs ``map_profiled_v1.main`` unchanged.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_m3_route_b_v2.map_class import install_map_class_conversion  # noqa: E402
from rl_agent.splitfusion_single_gpu_profiler_v1 import map_profiled_v1  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - live process
    install_map_class_conversion()
    raise SystemExit(map_profiled_v1.main())
