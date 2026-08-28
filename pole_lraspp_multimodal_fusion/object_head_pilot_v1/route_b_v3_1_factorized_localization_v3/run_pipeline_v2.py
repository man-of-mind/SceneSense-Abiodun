#!/usr/bin/env python3
"""Run the registered sequential experiment once in a create-only v3 directory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"

spec = importlib.util.spec_from_file_location("route_b_pipeline_v2", V2_PACKAGE / "run_pipeline_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("unable to load registered pipeline implementation")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
implementation.PACKAGE_ROOT = PACKAGE_ROOT
implementation.EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_factorized_localization_v3"
implementation.TRAINING_SOURCE = PACKAGE_ROOT / "configs/factorized_localization_training_v2.json"
implementation.SELECTION_SOURCE = PACKAGE_ROOT / "configs/selection_contract_v2.json"
implementation.TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_FACTORIZED_LOCALIZATION_V3_REPORT.md"
registered_make_report = implementation.make_report


def make_v3_report(*args, **kwargs):
    """Label generated reports with the versioned numerical repair package."""
    report = registered_make_report(*args, **kwargs)
    return report.replace(
        "# Route B v3.1 factorized localization v2 report",
        "# Route B v3.1 factorized localization v3 FP32-repair report",
        1,
    ).replace(
        "It unprojects positive `exp(log_depth)`",
        "The complete new localization path executes in FP32. It unprojects positive `exp(log_depth)`",
        1,
    )


implementation.make_report = make_v3_report

if __name__ == "__main__":
    raise SystemExit(implementation.main())
