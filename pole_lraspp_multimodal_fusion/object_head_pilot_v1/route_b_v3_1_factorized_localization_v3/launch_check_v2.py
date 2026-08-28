#!/usr/bin/env python3
"""Run the registered v2 launch contract against the v3 FP32 implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"
for path in (str(PACKAGE_ROOT), str(V2_PACKAGE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from losses_v3 import factorized_localization_loss  # noqa: E402
from model_v3 import (  # noqa: E402
    build_factorized_model, freeze_for_localization, load_native_warm_start,
    localization_parameters, parameter_report, split_boundary_report,
)

spec = importlib.util.spec_from_file_location("route_b_launch_check_v2", V2_PACKAGE / "launch_check_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("unable to load registered launch-check implementation")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
implementation.PACKAGE_ROOT = PACKAGE_ROOT
implementation.factorized_localization_loss = factorized_localization_loss
implementation.build_factorized_model = build_factorized_model
implementation.freeze_for_localization = freeze_for_localization
implementation.load_native_warm_start = load_native_warm_start
implementation.localization_parameters = localization_parameters
implementation.parameter_report = parameter_report
implementation.split_boundary_report = split_boundary_report

if __name__ == "__main__":
    raise SystemExit(implementation.main())
