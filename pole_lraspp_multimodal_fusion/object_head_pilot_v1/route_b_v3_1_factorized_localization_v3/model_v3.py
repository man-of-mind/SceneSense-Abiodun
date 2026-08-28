#!/usr/bin/env python3
"""Factorized localization with the complete new localization path in FP32."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"


def _load_v2() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_factorized_model_v2_implementation", V2_PACKAGE / "model_v2.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load committed factorized-localization v2 model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
native = v2.native
LOCALIZATION_HIDDEN = v2.LOCALIZATION_HIDDEN
LOCALIZATION_CHANNELS = v2.LOCALIZATION_CHANNELS


class FactorizedLocalizationFusionLRASPP(v2.FactorizedLocalizationFusionLRASPP):
    """The v2 architecture with only its new localization path forced to FP32."""

    def _fp32_localization(self, native_feature: torch.Tensor) -> torch.Tensor:
        # This boundary is deliberately complete: the trunk and both heads never
        # inherit the surrounding detector AMP context. Decoding/unprojection and
        # localization losses are held inside an equivalent boundary in losses_v3.
        with torch.autocast(device_type=native_feature.device.type, enabled=False):
            loc_features_fp32 = native_feature.detach().float()
            loc_hidden = self.localization_trunk(loc_features_fp32)
            raw_log_depth = self.log_depth_head(loc_hidden)
            projected_offset = self.projected_3d_center_offset_head(loc_hidden)
            return torch.cat([raw_log_depth, projected_offset], dim=1)

    def native_tail_outputs(self, features: object) -> tuple[torch.Tensor, torch.Tensor]:
        native_feature = self.object_head.upsampler(
            self.object_head.shared_trunk(self._object_input(features))
        )
        object_output = torch.cat([
            self.object_head.vehicle_heatmap_head(native_feature),
            self.object_head.person_heatmap_head(native_feature),
            self.object_head.regression_head(native_feature),
            self.object_head.offset_head(native_feature),
        ], dim=1)
        return object_output, self._fp32_localization(native_feature)

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        if float(feature_drop_fraction) != 0.0:
            raise ValueError("factorized localization v3 is q=0 only")
        features = self.backbone(x)
        seg = self.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        object_output, localization = self.native_tail_outputs(features)
        return {"out": seg, "object": object_output, "localization": localization}

    def localization_training_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the inherited network frozen; train only the FP32 localization path."""
        with torch.no_grad():
            features = self.backbone(x)
            native_feature = self.object_head.upsampler(
                self.object_head.shared_trunk(self._object_input(features))
            )
            object_output = torch.cat([
                self.object_head.vehicle_heatmap_head(native_feature),
                self.object_head.person_heatmap_head(native_feature),
                self.object_head.regression_head(native_feature),
                self.object_head.offset_head(native_feature),
            ], dim=1)
        return {
            "object": object_output.detach(),
            "localization": self._fp32_localization(native_feature),
        }


def build_factorized_model(
    *, num_classes: int = 3, radar_channels: int = 4,
    hidden_channels: int = 128, head_depth: int = 3,
    localization_hidden: int = LOCALIZATION_HIDDEN,
    device: Optional[torch.device] = None,
) -> FactorizedLocalizationFusionLRASPP:
    base = native.build_fusion_lraspp(
        num_classes=num_classes, radar_channels=radar_channels,
        pretrained=False, init_checkpoint="", device=device,
    )
    model = FactorizedLocalizationFusionLRASPP(
        base, hidden_channels=hidden_channels, head_depth=head_depth,
        localization_hidden=localization_hidden,
    )
    return model.to(device) if device is not None else model


load_native_warm_start = v2.load_native_warm_start
localization_parameters = v2.localization_parameters
freeze_for_localization = v2.freeze_for_localization
parameter_report = v2.parameter_report
encode_front = v2.encode_front
decode_tail = v2.decode_tail
split_boundary_report = v2.split_boundary_report
