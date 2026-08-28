#!/usr/bin/env python3
"""Run the registered v2 training loop with the v3 FP32 localization path."""

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
    localization_parameters, parameter_report,
)

spec = importlib.util.spec_from_file_location("route_b_train_v2", V2_PACKAGE / "train_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("unable to load registered training implementation")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
implementation.factorized_localization_loss = factorized_localization_loss
implementation.build_factorized_model = build_factorized_model
implementation.freeze_for_localization = freeze_for_localization
implementation.load_native_warm_start = load_native_warm_start
implementation.localization_parameters = localization_parameters
implementation.parameter_report = parameter_report

if __name__ == "__main__":
    raise SystemExit(implementation.main())
