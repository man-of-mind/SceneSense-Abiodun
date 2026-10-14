#!/usr/bin/env python3
"""M3 v20 map-server trampoline: the v3 class-preserving trampoline plus the display-only live-view install tap.

Deployed to ``rl_agent/splitfusion_single_gpu_profiler_v1/m3v20_live_view_map_profiled_v1.py``; the file name keeps
the retained process marker substring ``map_profiled_v1.py`` (sampler role, leftover markers).  It installs
``splitfusion_m3_route_b_v3.map_class`` exactly as ``m3v3_map_profiled_v1.py`` does, then
``splitfusion_live_view_tap_v1.tap.install_map_tap`` (outermost, post-install offer only), then runs
``map_profiled_v1.main`` unchanged.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.splitfusion_m3_route_b_v3.map_class import install_map_class_conversion  # noqa: E402
from rl_agent.splitfusion_live_view_tap_v1.tap import install_map_tap  # noqa: E402
from rl_agent.splitfusion_single_gpu_profiler_v1 import map_profiled_v1  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - live process
    install_map_class_conversion()
    install_map_tap()
    raise SystemExit(map_profiled_v1.main())
