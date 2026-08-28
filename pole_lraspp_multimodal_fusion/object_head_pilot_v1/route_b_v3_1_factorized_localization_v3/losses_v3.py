#!/usr/bin/env python3
"""Unchanged v2 localization decode/unprojection/loss algebra forced to FP32."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"


def _load_v2() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_factorized_losses_v2_implementation", V2_PACKAGE / "losses_v2.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load committed factorized-localization v2 losses")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()


def factorized_localization_loss(
    localization: torch.Tensor, legacy_object: torch.Tensor,
    targets: Dict[str, torch.Tensor], weights: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute the unchanged depth decode, unprojection, and losses in FP32."""
    with torch.autocast(device_type=localization.device.type, enabled=False):
        return v2.factorized_localization_loss(
            localization.float(), legacy_object.float(), targets, weights
        )
