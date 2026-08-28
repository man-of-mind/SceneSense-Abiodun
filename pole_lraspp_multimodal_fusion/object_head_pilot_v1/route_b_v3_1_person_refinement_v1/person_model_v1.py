#!/usr/bin/env python3
"""Frozen recovered LR-ASPP plus one private person-only tail refinement."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import model_v1 as native  # noqa: E402

PRIVATE_FIXED_CHANNELS = 6  # object residual, quality, range residual, uv offset, mask residual


class PersonRefinementHead(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, groups: int,
                 range_bins: int) -> None:
        super().__init__()
        self.range_bins = int(range_bins)
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(groups, hidden_channels),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            torch.nn.GroupNorm(groups, hidden_channels),
            torch.nn.SiLU(inplace=True),
        )
        self.objectness_residual = torch.nn.Conv2d(hidden_channels, 1, 1)
        self.localization_quality = torch.nn.Conv2d(hidden_channels, 1, 1)
        self.range_bin_logits = torch.nn.Conv2d(hidden_channels, self.range_bins, 1)
        self.range_residual = torch.nn.Conv2d(hidden_channels, 1, 1)
        self.projected_center_offset = torch.nn.Conv2d(hidden_channels, 2, 1)
        self.person_mask_residual = torch.nn.Conv2d(hidden_channels, 1, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.trunk.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            for head in (
                self.objectness_residual, self.range_residual,
                self.projected_center_offset, self.person_mask_residual,
            ):
                head.weight.zero_()
                head.bias.zero_()
            self.localization_quality.weight.zero_()
            self.localization_quality.bias.fill_(math.log(0.99 / 0.01))
            torch.nn.init.normal_(self.range_bin_logits.weight, mean=0.0, std=1e-3)
            self.range_bin_logits.bias.zero_()

    def forward(self, native_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.trunk(native_feature)
        return {
            "objectness_residual": self.objectness_residual(feature),
            "localization_quality": self.localization_quality(feature),
            "range_bin_logits": self.range_bin_logits(feature),
            "range_residual": self.range_residual(feature),
            "projected_center_offset": self.projected_center_offset(feature),
            "person_mask_residual": self.person_mask_residual(feature),
        }


class PersonRefinementFusionLRASPP(native.NativeGridFusionLRASPP):
    """The recovered network is frozen; only a person-private tail is added.

    ``object`` retains the exact native channel order.  Vehicle heatmap and all
    shared regression channels are produced by the recovered modules.  The private
    dictionary is consumed only by the experimental person decoder and never crosses
    the transported front/tail boundary.
    """

    def __init__(self, base_model: torch.nn.Module, *, hidden_channels: int = 128,
                 head_depth: int = 3, person_hidden: int = 96,
                 group_norm_groups: int = 8, range_bins: int = 8) -> None:
        super().__init__(base_model, hidden_channels=hidden_channels, head_depth=head_depth)
        self.person_refinement = PersonRefinementHead(
            hidden_channels, person_hidden, group_norm_groups, range_bins,
        )
        self.person_range_bins = int(range_bins)

    def _native_feature(self, features: object) -> torch.Tensor:
        return self.object_head.upsampler(
            self.object_head.shared_trunk(self._object_input(features))
        )

    def _finite_class_heatmaps(self, native_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the two inherited 1x1 heatmap projections in FP32.

        Recovered epoch-40 heatmap weights can overflow at some background cells
        under FP16 although the input feature, weights, FP32 path, losses, and all
        decoded values are finite.  This is the single registered AMP numerical
        repair; regression, offsets, segmentation, and the new person tail remain
        under the caller's normal autocast policy.
        """
        with torch.autocast(device_type="cuda", enabled=False):
            feature = native_feature.float()
            return (
                self.object_head.vehicle_heatmap_head(feature),
                self.object_head.person_heatmap_head(feature),
            )

    def tail_outputs(self, features: object) -> dict[str, Any]:
        native_feature = self._native_feature(features)
        vehicle_heatmap, person_heatmap = self._finite_class_heatmaps(native_feature)
        base_object = torch.cat([
            vehicle_heatmap, person_heatmap,
            self.object_head.regression_head(native_feature),
            self.object_head.offset_head(native_feature),
        ], dim=1)
        refinement = self.person_refinement(native_feature.detach())
        segmentation = self.classifier(features)
        if isinstance(segmentation, dict):
            segmentation = segmentation["out"]
        if tuple(refinement["person_mask_residual"].shape[-2:]) != tuple(segmentation.shape[-2:]):
            mask_residual = torch.nn.functional.interpolate(
                refinement["person_mask_residual"], size=segmentation.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        else:
            mask_residual = refinement["person_mask_residual"]
        refined_segmentation = torch.cat([
            segmentation[:, :2], segmentation[:, 2:3] + mask_residual,
        ], dim=1)
        return {
            "out": refined_segmentation, "base_out": segmentation,
            "object": base_object, "person_refinement": refinement,
            "native_feature": native_feature,
        }

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> dict[str, Any]:
        if float(feature_drop_fraction) != 0.0:
            raise ValueError("person refinement is clean q=0 only")
        return self.tail_outputs(self.backbone(x))

    def training_outputs(self, x: torch.Tensor) -> dict[str, Any]:
        """Backbone/shared/vehicle/segmentation stay outside autograd.

        The recovered person heatmap 1x1 slice remains in the graph only when P2
        explicitly marks it trainable.
        """
        with torch.no_grad():
            features = self.backbone(x)
            native_feature = self._native_feature(features)
            vehicle_heatmap, _unused_person_heatmap = self._finite_class_heatmaps(native_feature)
            regression = self.object_head.regression_head(native_feature)
            grid_offset = self.object_head.offset_head(native_feature)
            segmentation = self.classifier(features)
            if isinstance(segmentation, dict):
                segmentation = segmentation["out"]
        with torch.autocast(device_type="cuda", enabled=False):
            person_heatmap = self.object_head.person_heatmap_head(native_feature.detach().float())
        base_object = torch.cat([
            vehicle_heatmap.detach(), person_heatmap, regression.detach(), grid_offset.detach(),
        ], dim=1)
        refinement = self.person_refinement(native_feature.detach())
        mask_residual = refinement["person_mask_residual"]
        if tuple(mask_residual.shape[-2:]) != tuple(segmentation.shape[-2:]):
            mask_residual = torch.nn.functional.interpolate(
                mask_residual, size=segmentation.shape[-2:], mode="bilinear", align_corners=False,
            )
        refined_segmentation = torch.cat([
            segmentation[:, :2].detach(), segmentation[:, 2:3].detach() + mask_residual,
        ], dim=1)
        return {
            "out": refined_segmentation, "base_out": segmentation.detach(),
            "object": base_object, "person_refinement": refinement,
        }


def build_model(*, radar_channels: int = 4, hidden_channels: int = 128,
                head_depth: int = 3, person_hidden: int = 96,
                group_norm_groups: int = 8, range_bins: int = 8,
                device: Optional[torch.device] = None) -> PersonRefinementFusionLRASPP:
    base = native.build_fusion_lraspp(
        num_classes=3, radar_channels=radar_channels, pretrained=False,
        init_checkpoint="", device=device,
    )
    model = PersonRefinementFusionLRASPP(
        base, hidden_channels=hidden_channels, head_depth=head_depth,
        person_hidden=person_hidden, group_norm_groups=group_norm_groups,
        range_bins=range_bins,
    )
    return model.to(device) if device is not None else model


def load_recovered_base(model: PersonRefinementFusionLRASPP, checkpoint: Path,
                        *, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    only_new = all(key.startswith("person_refinement.") for key in incompatible.missing_keys)
    if incompatible.unexpected_keys or not only_new:
        raise RuntimeError(
            f"recovered base mapping failure missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    return {
        "checkpoint": str(checkpoint), "checkpoint_epoch": int(payload["epoch"]),
        "all_recovered_tensors_loaded": only_new and not incompatible.unexpected_keys,
        "missing_new_keys": sorted(incompatible.missing_keys),
    }


def new_parameters(model: PersonRefinementFusionLRASPP) -> list[torch.nn.Parameter]:
    return list(model.person_refinement.parameters())


def inherited_person_parameters(model: PersonRefinementFusionLRASPP) -> list[torch.nn.Parameter]:
    return list(model.object_head.person_heatmap_head.parameters())


def configure_stage(model: PersonRefinementFusionLRASPP, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in new_parameters(model):
        parameter.requires_grad = True
    if stage == "P2":
        for parameter in inherited_person_parameters(model):
            parameter.requires_grad = True
    elif stage != "P1":
        raise ValueError(f"unknown stage {stage}")
    model.eval()
    model.person_refinement.train()
    if stage == "P2":
        model.object_head.person_heatmap_head.train()


def parameter_report(model: PersonRefinementFusionLRASPP) -> dict[str, Any]:
    groups = OrderedDict([
        ("backbone", model.backbone),
        ("segmentation", model.classifier),
        ("native_shared", model.object_head.shared_trunk),
        ("native_upsampler", model.object_head.upsampler),
        ("vehicle_heatmap", model.object_head.vehicle_heatmap_head),
        ("person_heatmap", model.object_head.person_heatmap_head),
        ("shared_regression", model.object_head.regression_head),
        ("grid_offset", model.object_head.offset_head),
        ("person_refinement", model.person_refinement),
    ])
    result: dict[str, Any] = {}
    for name, module in groups.items():
        total = sum(int(parameter.numel()) for parameter in module.parameters())
        trainable = sum(int(parameter.numel()) for parameter in module.parameters()
                        if parameter.requires_grad)
        result[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters()
                    if parameter.requires_grad)
    result["model_total"] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    return result


def encode_front(model: PersonRefinementFusionLRASPP,
                 x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((str(name), value) for name, value in model.backbone(x).items())


def decode_tail(model: PersonRefinementFusionLRASPP,
                features: OrderedDict[str, torch.Tensor]) -> dict[str, Any]:
    return model.tail_outputs(features)


def split_boundary_report(model: PersonRefinementFusionLRASPP,
                          x: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        monolithic = model(x)
        bundle = encode_front(model, x)
        split = decode_tail(model, bundle)
    tensor_keys = ("out", "base_out", "object", "native_feature")
    deltas = {
        key: float((monolithic[key].float() - split[key].float()).abs().max().item())
        for key in tensor_keys
    }
    for key in monolithic["person_refinement"]:
        name = f"person_refinement.{key}"
        deltas[name] = float((
            monolithic["person_refinement"][key].float()
            - split["person_refinement"][key].float()
        ).abs().max().item())
    return {
        "transported_feature_names": list(bundle.keys()),
        "transported_feature_shapes": {key: list(value.shape) for key, value in bundle.items()},
        "transported_feature_dtypes": {key: str(value.dtype) for key, value in bundle.items()},
        "tail_reads_only_low_high": list(bundle.keys()) == ["low", "high"],
        "tail_raw_modality_side_channels": [],
        "tail_frame_metadata": ["camera_intrinsics", "camera_to_world"],
        "external_object_shape": list(monolithic["object"].shape),
        "external_object_channel_order": [
            "vehicle_heatmap", "person_heatmap", "shared_regression_12", "center_offset_2"
        ],
        "monolithic_vs_split_max_abs_delta": deltas,
        "outputs_bit_identical": all(value == 0.0 for value in deltas.values()),
    }
