#!/usr/bin/env python3
"""Clean ResNet34-FPN CenterNet/CenterFusion-style Route B model.

The image detector and radar path are intentionally separate.  A pretrained
three-channel ResNet34 produces an RGB FPN at output stride four.  A distinct
four-channel radar encoder produces its own stride-four FPN.  The primary
CenterNet head operates on the RGB feature.  A second, radar-conditioned head
refines the two centre heatmaps and all twelve regression maps.  The lightweight
segmentation decoder consumes the fused RGB/radar feature.

Future UE/edge boundary
-----------------------
``encode_front(rgb, radar)`` returns the complete feature bundle consumed by
``decode_tail(bundle)``.  There is no raw-input or radar side channel around the
bundle.  A future q/AE/quantization operation therefore has a single accountable
boundary at the returned ``rgb_p2`` and ``radar_p2`` tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.model import OBJECT_HEAD_CHANNELS
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels

HEAD_ARCH_NAME = "resnet34_fpn_centerfusion_v1"
FPN_CHANNELS = 128
HEAD_CHANNELS = 128


def _gn(channels: int) -> torch.nn.GroupNorm:
    groups = min(16, int(channels))
    while int(channels) % groups:
        groups -= 1
    return torch.nn.GroupNorm(groups, int(channels))


class ConvGNAct(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _gn(out_channels),
            torch.nn.ReLU(inplace=True),
        )


class RGBResNet34Backbone(torch.nn.Module):
    """ResNet34 feature extractor exposing C2..C5 without unused FC weights."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        weights = ResNet34_Weights.IMAGENET1K_V1 if bool(pretrained) else None
        source = resnet34(weights=weights)
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.relu = source.relu
        self.maxpool = source.maxpool
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.relu(self.bn1(self.conv1(rgb)))
        x = self.maxpool(x)
        c2 = self.layer1(x)   # 1/4, 64
        c3 = self.layer2(c2)  # 1/8, 128
        c4 = self.layer3(c3)  # 1/16, 256
        c5 = self.layer4(c4)  # 1/32, 512
        return c2, c3, c4, c5


class RadarEncoder(torch.nn.Module):
    """Independent four-channel radar pyramid at the ResNet feature strides."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.stem = ConvGNAct(int(in_channels), 32, stride=2)
        self.stage2 = torch.nn.Sequential(ConvGNAct(32, 64, stride=2), ConvGNAct(64, 64))
        self.stage3 = torch.nn.Sequential(ConvGNAct(64, 96, stride=2), ConvGNAct(96, 96))
        self.stage4 = torch.nn.Sequential(ConvGNAct(96, 128, stride=2), ConvGNAct(128, 128))
        self.stage5 = torch.nn.Sequential(ConvGNAct(128, 160, stride=2), ConvGNAct(160, 160))

    def forward(self, radar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.stem(radar)
        r2 = self.stage2(x)
        r3 = self.stage3(r2)
        r4 = self.stage4(r3)
        r5 = self.stage5(r4)
        return r2, r3, r4, r5


class FeaturePyramid(torch.nn.Module):
    def __init__(self, channels: Tuple[int, int, int, int], out_channels: int) -> None:
        super().__init__()
        self.laterals = torch.nn.ModuleList(
            torch.nn.Conv2d(c, out_channels, 1) for c in channels
        )
        self.smooth = torch.nn.ModuleList(
            ConvGNAct(out_channels, out_channels) for _ in channels
        )

    def forward(self, levels: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        lateral = [layer(value) for layer, value in zip(self.laterals, levels)]
        p = self.smooth[-1](lateral[-1])
        for index in range(len(lateral) - 2, -1, -1):
            p = lateral[index] + F.interpolate(
                p, size=lateral[index].shape[-2:], mode="bilinear", align_corners=False
            )
            p = self.smooth[index](p)
        return p


class CenterNetHead(torch.nn.Module):
    """Two class-specific heatmaps and shared regression at output stride four."""

    def __init__(self, in_channels: int, reg_channels: int, hidden: int = HEAD_CHANNELS) -> None:
        super().__init__()
        self.shared_trunk = torch.nn.Sequential(
            ConvGNAct(in_channels, hidden),
            ConvGNAct(hidden, hidden),
        )
        self.vehicle_heatmap_head = torch.nn.Conv2d(hidden, 1, 1)
        self.person_heatmap_head = torch.nn.Conv2d(hidden, 1, 1)
        self.regression_head = torch.nn.Conv2d(hidden, int(reg_channels), 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        shared = self.shared_trunk(feature)
        return torch.cat(
            (
                self.vehicle_heatmap_head(shared),
                self.person_heatmap_head(shared),
                self.regression_head(shared),
            ),
            dim=1,
        )


class RadarRefinementHead(torch.nn.Module):
    """CenterFusion-style radar-conditioned dense second stage."""

    def __init__(self, feature_channels: int, object_channels: int, hidden: int) -> None:
        super().__init__()
        self.fusion = torch.nn.Sequential(
            ConvGNAct(feature_channels * 2 + object_channels, hidden),
            ConvGNAct(hidden, hidden),
        )
        self.vehicle_heatmap_head = torch.nn.Conv2d(hidden, 1, 1)
        self.person_heatmap_head = torch.nn.Conv2d(hidden, 1, 1)
        self.regression_head = torch.nn.Conv2d(hidden, object_channels - 2, 1)

    def forward(
        self, rgb_feature: torch.Tensor, radar_feature: torch.Tensor, primary: torch.Tensor
    ) -> torch.Tensor:
        shared = self.fusion(torch.cat((rgb_feature, radar_feature, primary), dim=1))
        return torch.cat(
            (
                self.vehicle_heatmap_head(shared),
                self.person_heatmap_head(shared),
                self.regression_head(shared),
            ),
            dim=1,
        )


class SegmentationDecoder(torch.nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ConvGNAct(in_channels, 128),
            ConvGNAct(128, 64),
            torch.nn.Conv2d(64, int(num_classes), 1),
        )


class CleanCenterFusionResNet34(torch.nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        radar_channels: int,
        object_channels: int,
        predict_bbox2d: bool,
        pretrained: bool,
        fpn_channels: int = FPN_CHANNELS,
        head_channels: int = HEAD_CHANNELS,
    ) -> None:
        super().__init__()
        self.head_arch = HEAD_ARCH_NAME
        self.radar_channels = int(radar_channels)
        self.object_channels = int(object_channels)
        self.predict_bbox2d = bool(predict_bbox2d)
        self.reg_channels = object_reg_channels(self.predict_bbox2d)
        self.heatmap_channels = self.object_channels - self.reg_channels
        if self.heatmap_channels != 2 or self.object_channels != 14:
            raise ValueError(
                "clean CenterNet model requires two heatmaps plus twelve regression maps "
                f"(14 total), got {self.object_channels}"
            )

        self.backbone = RGBResNet34Backbone(pretrained=bool(pretrained))
        self.radar_encoder = RadarEncoder(self.radar_channels)
        self.rgb_fpn = FeaturePyramid((64, 128, 256, 512), int(fpn_channels))
        self.radar_fpn = FeaturePyramid((64, 96, 128, 160), int(fpn_channels))
        self.object_head = CenterNetHead(int(fpn_channels), self.reg_channels, int(head_channels))
        self.refinement_head = RadarRefinementHead(
            int(fpn_channels), self.object_channels, int(head_channels)
        )
        self.fusion_projection = torch.nn.Sequential(
            torch.nn.Conv2d(int(fpn_channels) * 2, int(fpn_channels), 1, bias=False),
            _gn(int(fpn_channels)),
            torch.nn.ReLU(inplace=True),
        )
        self.classifier = SegmentationDecoder(int(fpn_channels), int(num_classes))
        self.feature_ae = None
        self._init_new_modules()

    def _init_new_modules(self) -> None:
        new_modules = (
            self.radar_encoder,
            self.rgb_fpn,
            self.radar_fpn,
            self.object_head,
            self.refinement_head,
            self.fusion_projection,
            self.classifier,
        )
        for root in new_modules:
            for module in root.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.GroupNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)

        # Standard CenterNet p=0.1 prior for trainable cold heatmap heads.
        for head in (
            self.object_head.vehicle_heatmap_head,
            self.object_head.person_heatmap_head,
        ):
            torch.nn.init.normal_(head.weight, std=0.001)
            torch.nn.init.constant_(head.bias, -2.19)

        # The radar second stage begins as a small, nonzero residual.  This keeps
        # the primary detector well-conditioned while providing gradient to every
        # refinement tensor on the launch batch.
        for head in (
            self.refinement_head.vehicle_heatmap_head,
            self.refinement_head.person_heatmap_head,
            self.refinement_head.regression_head,
        ):
            torch.nn.init.normal_(head.weight, std=1e-3)
            torch.nn.init.zeros_(head.bias)

        if self.predict_bbox2d:
            with torch.no_grad():
                self.object_head.regression_head.bias[-2:].fill_(-3.0)

    def encode_front(
        self, rgb: torch.Tensor, radar: torch.Tensor
    ) -> "OrderedDict[str, torch.Tensor]":
        rgb_p2 = self.rgb_fpn(self.backbone(rgb))
        radar_p2 = self.radar_fpn(self.radar_encoder(radar))
        return OrderedDict((("rgb_p2", rgb_p2), ("radar_p2", radar_p2)))

    def decode_tail(
        self, feature_bundle: Dict[str, torch.Tensor], out_hw: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        rgb_p2 = feature_bundle["rgb_p2"]
        radar_p2 = feature_bundle["radar_p2"]
        primary = self.object_head(rgb_p2)
        refinement = self.refinement_head(rgb_p2, radar_p2, primary)
        object_maps = primary + refinement
        fused = self.fusion_projection(torch.cat((rgb_p2, radar_p2), dim=1))
        segmentation = self.classifier(fused)
        return {
            "out": segmentation,
            "object": F.interpolate(
                object_maps, size=out_hw, mode="bilinear", align_corners=False
            ),
        }

    def _objectness_drop(
        self, bundle: Dict[str, torch.Tensor], q: float
    ) -> "OrderedDict[str, torch.Tensor]":
        with torch.no_grad():
            primary = self.object_head(bundle["rgb_p2"])
            objectness = torch.sigmoid(primary[:, :2]).amax(dim=1, keepdim=True)
        batch = objectness.shape[0]
        flat = objectness.reshape(batch, -1).float()
        count = int(round(float(q) * flat.shape[1]))
        if count <= 0:
            return OrderedDict(bundle)
        dropped = flat.argsort(dim=1)[:, :count]
        keep = torch.ones_like(flat).scatter_(1, dropped, 0.0)
        keep = keep.reshape(batch, 1, *objectness.shape[-2:])
        return OrderedDict(
            (name, value * keep.to(value.dtype)) for name, value in bundle.items()
        )

    def _apply_feature_ae(
        self, bundle: Dict[str, torch.Tensor]
    ) -> "OrderedDict[str, torch.Tensor]":
        # Reserved for future work.  Both modality tensors are part of the split
        # bundle, so any later AE must account for both rather than bypass radar.
        if self.feature_ae is None:
            return OrderedDict(bundle)
        raise RuntimeError("feature AE is intentionally disabled for the clean noAE qualification")

    def forward(
        self, inputs: torch.Tensor, feature_drop_fraction: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        rgb = inputs[:, :3]
        radar = inputs[:, 3 : 3 + self.radar_channels]
        bundle = self.encode_front(rgb, radar)
        if float(feature_drop_fraction) > 0.0:
            bundle = self._objectness_drop(bundle, float(feature_drop_fraction))
        if self.feature_ae is not None:
            bundle = self._apply_feature_ae(bundle)
        return self.decode_tail(bundle, (int(inputs.shape[-2]), int(inputs.shape[-1])))


def build_clean_centernet(
    *,
    num_classes: int,
    radar_channels: int,
    object_channels: int = OBJECT_HEAD_CHANNELS,
    object_hidden_channels: int = HEAD_CHANNELS,
    predict_bbox2d: bool = True,
    pretrained: bool = False,
    **_: object,
) -> CleanCenterFusionResNet34:
    return CleanCenterFusionResNet34(
        num_classes=int(num_classes),
        radar_channels=int(radar_channels),
        object_channels=int(object_channels),
        predict_bbox2d=bool(predict_bbox2d),
        pretrained=bool(pretrained),
        head_channels=int(object_hidden_channels),
    )


def install() -> None:
    """Dispatch the registered architecture through the existing train/eval code."""
    from pole_lraspp_multimodal_fusion import evaluate_fusion, model, train_fusion

    original = getattr(
        model, "_clean_centernet_original_builder", model.build_multitask_fusion_lraspp
    )

    def dispatch(*, head_arch: str = "shared", **kwargs: object) -> torch.nn.Module:
        if str(head_arch) != HEAD_ARCH_NAME:
            return original(head_arch=head_arch, **kwargs)
        return build_clean_centernet(pretrained=bool(kwargs.pop("pretrained", False)), **kwargs)

    model._clean_centernet_original_builder = original
    for module in (model, train_fusion, evaluate_fusion):
        module.build_multitask_fusion_lraspp = dispatch

