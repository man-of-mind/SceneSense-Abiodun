#!/usr/bin/env python3
"""Run the unchanged registered selection/evaluation implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"
if str(V2_PACKAGE) not in sys.path:
    sys.path.insert(0, str(V2_PACKAGE))

spec = importlib.util.spec_from_file_location("route_b_evaluate_v2", V2_PACKAGE / "evaluate_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("unable to load registered evaluation implementation")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)

if __name__ == "__main__":
    raise SystemExit(implementation.main())
