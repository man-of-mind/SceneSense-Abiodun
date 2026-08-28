#!/usr/bin/env python3
"""Frozen native-grid LR-ASPP plus one tail-side factorized-localization path."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PKG), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import model_v1 as native  # noqa: E402

LOCALIZATION_HIDDEN = 64
LOCALIZATION_CHANNELS = 3


class FactorizedLocalizationFusionLRASPP(native.NativeGridFusionLRASPP):
    """Native detector unchanged; new localization reads its frozen stride-4 feature."""

    def __init__(self, base_model: torch.nn.Module, *, hidden_channels: int = 128,
                 head_depth: int = 3, localization_hidden: int = LOCALIZATION_HIDDEN) -> None:
        super().__init__(base_model, hidden_channels=hidden_channels, head_depth=head_depth)
        self.localization_trunk = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, localization_hidden, 3, padding=1, bias=True),
            torch.nn.GroupNorm(8, localization_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(localization_hidden, localization_hidden, 3, padding=1, bias=True),
            torch.nn.GroupNorm(8, localization_hidden),
            torch.nn.ReLU(inplace=True),
        )
        self.log_depth_head = torch.nn.Conv2d(localization_hidden, 1, 1)
        self.projected_3d_center_offset_head = torch.nn.Conv2d(localization_hidden, 2, 1)
        self._init_localization()

    def _init_localization(self) -> None:
        for module in self.localization_trunk.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        with torch.no_grad():
            # Tiny non-zero output weights preserve the useful log(20 m)/zero-offset
            # priors while allowing the mandatory launch batch to prove that the
            # localization trunk itself receives a real gradient.
            torch.nn.init.normal_(self.log_depth_head.weight, mean=0.0, std=1e-3)
            self.log_depth_head.bias.fill_(math.log(20.0))
            torch.nn.init.normal_(
                self.projected_3d_center_offset_head.weight, mean=0.0, std=1e-3
            )
            self.projected_3d_center_offset_head.bias.zero_()

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
        localization_feature = self.localization_trunk(native_feature.detach())
        localization = torch.cat([
            self.log_depth_head(localization_feature),
            self.projected_3d_center_offset_head(localization_feature),
        ], dim=1)
        return object_output, localization

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        if float(feature_drop_fraction) != 0.0:
            raise ValueError("factorized localization v2 is q=0 only")
        features = self.backbone(x)
        seg = self.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        object_output, localization = self.native_tail_outputs(features)
        return {"out": seg, "object": object_output, "localization": localization}

    def localization_training_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the entire inherited network without autograd; train only new modules."""
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
        localization_feature = self.localization_trunk(native_feature.detach())
        localization = torch.cat([
            self.log_depth_head(localization_feature),
            self.projected_3d_center_offset_head(localization_feature),
        ], dim=1)
        return {"object": object_output.detach(), "localization": localization}


def build_factorized_model(*, num_classes: int = 3, radar_channels: int = 4,
                           hidden_channels: int = 128, head_depth: int = 3,
                           localization_hidden: int = LOCALIZATION_HIDDEN,
                           device: Optional[torch.device] = None) -> FactorizedLocalizationFusionLRASPP:
    base = native.build_fusion_lraspp(
        num_classes=num_classes, radar_channels=radar_channels,
        pretrained=False, init_checkpoint="", device=device,
    )
    model = FactorizedLocalizationFusionLRASPP(
        base, hidden_channels=hidden_channels, head_depth=head_depth,
        localization_hidden=localization_hidden,
    )
    return model.to(device) if device is not None else model


def load_native_warm_start(model: torch.nn.Module, checkpoint: Path,
                           *, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(payload["model"], strict=False)
    expected_prefixes = (
        "localization_trunk.", "log_depth_head.", "projected_3d_center_offset_head.",
    )
    only_new = all(key.startswith(expected_prefixes) for key in missing.missing_keys)
    if missing.unexpected_keys or not only_new:
        raise RuntimeError(
            f"native warm-start mapping failure missing={missing.missing_keys} "
            f"unexpected={missing.unexpected_keys}"
        )
    model.to(device)
    return {
        "checkpoint": str(checkpoint), "checkpoint_epoch": int(payload.get("epoch", -1)),
        "missing_new_keys": sorted(missing.missing_keys),
        "unexpected_keys": sorted(missing.unexpected_keys),
        "all_inherited_tensors_loaded": only_new and not missing.unexpected_keys,
    }


def localization_parameters(model: FactorizedLocalizationFusionLRASPP) -> list[torch.nn.Parameter]:
    return [
        *model.localization_trunk.parameters(), *model.log_depth_head.parameters(),
        *model.projected_3d_center_offset_head.parameters(),
    ]


def freeze_for_localization(model: FactorizedLocalizationFusionLRASPP) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in localization_parameters(model):
        parameter.requires_grad = True
    model.eval()
    model.localization_trunk.train()
    model.log_depth_head.train()
    model.projected_3d_center_offset_head.train()


def parameter_report(model: FactorizedLocalizationFusionLRASPP) -> Dict[str, Any]:
    groups = {
        "backbone": model.backbone,
        "segmentation_classifier": model.classifier,
        "native_shared_trunk": model.object_head.shared_trunk,
        "native_upsampler": model.object_head.upsampler,
        "vehicle_heatmap": model.object_head.vehicle_heatmap_head,
        "person_heatmap": model.object_head.person_heatmap_head,
        "legacy_regression": model.object_head.regression_head,
        "grid_offset": model.object_head.offset_head,
        "localization_trunk": model.localization_trunk,
        "log_depth_head": model.log_depth_head,
        "projected_3d_center_offset_head": model.projected_3d_center_offset_head,
    }
    output: Dict[str, Any] = {}
    for name, module in groups.items():
        total = sum(int(parameter.numel()) for parameter in module.parameters())
        trainable = sum(int(parameter.numel()) for parameter in module.parameters()
                        if parameter.requires_grad)
        output[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters()
                    if parameter.requires_grad)
    output["model_total"] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    return output


def encode_front(model: FactorizedLocalizationFusionLRASPP, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((str(name), value) for name, value in model.backbone(x).items())


def decode_tail(model: FactorizedLocalizationFusionLRASPP,
                features: "OrderedDict[str, torch.Tensor]") -> Dict[str, torch.Tensor]:
    seg = model.classifier(features)
    if isinstance(seg, dict):
        seg = seg["out"]
    object_output, localization = model.native_tail_outputs(features)
    return {"out": seg, "object": object_output, "localization": localization}


def split_boundary_report(model: FactorizedLocalizationFusionLRASPP,
                          x: torch.Tensor) -> Dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        monolithic = model(x)
        bundle = encode_front(model, x)
        tail = decode_tail(model, bundle)
    deltas = {key: float((monolithic[key].float() - tail[key].float()).abs().max().item())
              for key in monolithic}
    return {
        "transported_feature_names": sorted(bundle),
        "transported_feature_shapes": {key: list(value.shape) for key, value in bundle.items()},
        "tail_reads_only_low_high": sorted(bundle) == ["high", "low"],
        "tail_static_or_frame_metadata": ["camera_intrinsics", "camera_to_world"],
        "tail_raw_modality_side_channels": [],
        "monolithic_vs_split_max_abs_delta": deltas,
        "outputs_bit_identical": all(value == 0.0 for value in deltas.values()),
    }
