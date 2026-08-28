#!/usr/bin/env python3
"""Run registered inference with the v3 FP32 localization model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"
for path in (str(PACKAGE_ROOT), str(V2_PACKAGE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_v3 import build_factorized_model  # noqa: E402

spec = importlib.util.spec_from_file_location("route_b_infer_v2", V2_PACKAGE / "infer_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("unable to load registered inference implementation")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
implementation.build_factorized_model = build_factorized_model

if __name__ == "__main__":
    raise SystemExit(implementation.main())
