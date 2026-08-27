#!/usr/bin/env python3
"""Route B clean CenterNet **v2** - native dual-stride object heads.

What changed relative to ``centernet_clean_v1`` (and why), per the findings in
``CENTERNET_EVALUATION_CONTRACT_AUDIT.md``:

* v1 produced all 14 object channels on the stride-4 grid and then *bilinearly
  enlarged* them to the input resolution.  Targets were placed at full
  resolution, so regression was supervised on interpolated values and the
  decoder spent ~81% of its top-k budget on interpolated duplicates.
  v2 trains and decodes **only on native grids**; nothing is enlarged.
* v1 had no centre-offset head, so the decoded centre was quantised to the
  stride-4 cell.  v2 gives **each branch its own private 2-channel offset head**.
* Persons are thin (17% of person GT boxes are narrower than one stride-4 cell
  after the 0.6 resize).  v2 gives persons a **compact stride-2 branch** with its
  own heatmap, offsets and 12 regression fields.  Vehicles stay at stride 4.
* Class-specific regression maps: the vehicle and person branches own separate
  regression tensors, so the v1 class-agnostic ``reg_mask`` overwrite is
  structurally impossible.
* Segmentation gets a task-specific lightweight decoder: fused stride-4 context
  + a stride-2 RGB/radar skip + two learned (transposed-conv) upsamples.  No
  HRNet, no LR-ASPP.

Preserved from v1: ImageNet-pretrained ResNet34 RGB encoder, independent
four-channel radar encoder, RGB/radar feature fusion, three-class segmentation
output, the decoded object field schema (class, score, XYZ, dimensions, yaw,
parked, radar support, 2D box) and the ``encode_front`` / ``decode_tail`` split.

UE/edge split boundary
----------------------
``encode_front(rgb, radar)`` returns the complete feature bundle
(``rgb_p2``, ``radar_p2``, ``s2``).  ``decode_tail(bundle)`` reads **only** that
bundle - there is no raw-RGB or raw-radar side channel around the boundary, and
the stride-2 skip used by both the person branch and the segmentation decoder
crosses the boundary inside the bundle rather than bypassing it.  A future
q / AE / quantization stage therefore still has one accountable attachment
point covering every feature the tail consumes.  q and AE are disabled here.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

HEAD_ARCH_NAME = "resnet34_fpn_centernet_native_v2"
FPN_CHANNELS = 128
HEAD_CHANNELS = 128
STRIDE2_CHANNELS = 16
PERSON_CHANNELS = 64
VEHICLE_STRIDE = 4
PERSON_STRIDE = 2
REG_FIELDS = 12  # xyz(3) dims(3) yaw(2) parked(1) radar_support(1) bbox_wh(2)


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


class UpConvGNAct(torch.nn.Sequential):
    """Learned 2x upsample (transposed conv), used by the seg decoder and person branch."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            _gn(out_channels),
            torch.nn.ReLU(inplace=True),
        )


class RGBResNet34Backbone(torch.nn.Module):
    """ResNet34 feature extractor exposing the stride-2 stem plus C2..C5.

    Parameter names are identical to v1 so every tensor warm-starts from the
    epoch-12 checkpoint; only the returned tuple is wider (``c1`` added).
    """

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
        c1 = self.relu(self.bn1(self.conv1(rgb)))  # 1/2, 64
        x = self.maxpool(c1)
        c2 = self.layer1(x)   # 1/4, 64
        c3 = self.layer2(c2)  # 1/8, 128
        c4 = self.layer3(c3)  # 1/16, 256
        c5 = self.layer4(c4)  # 1/32, 512
        return c1, c2, c3, c4, c5


class RadarEncoder(torch.nn.Module):
    """Independent four-channel radar pyramid (v1 parameter names preserved)."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.stem = ConvGNAct(int(in_channels), 32, stride=2)
        self.stage2 = torch.nn.Sequential(ConvGNAct(32, 64, stride=2), ConvGNAct(64, 64))
        self.stage3 = torch.nn.Sequential(ConvGNAct(64, 96, stride=2), ConvGNAct(96, 96))
        self.stage4 = torch.nn.Sequential(ConvGNAct(96, 128, stride=2), ConvGNAct(128, 128))
        self.stage5 = torch.nn.Sequential(ConvGNAct(128, 160, stride=2), ConvGNAct(160, 160))

    def forward(self, radar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        r1 = self.stem(radar)  # 1/2, 32
        r2 = self.stage2(r1)   # 1/4, 64
        r3 = self.stage3(r2)
        r4 = self.stage4(r3)
        r5 = self.stage5(r4)
        return r1, r2, r3, r4, r5


class FeaturePyramid(torch.nn.Module):
    """Unchanged from v1 (parameter names preserved for warm start)."""

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


class RadarFusion(torch.nn.Module):
    """RGB/radar feature fusion at stride four, shared by both branches and seg."""

    def __init__(self, fpn_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(int(fpn_channels) * 2, int(out_channels), 1, bias=False),
            _gn(int(out_channels)),
            torch.nn.ReLU(inplace=True),
        )
        self.refine = ConvGNAct(int(out_channels), int(out_channels))

    def forward(self, rgb_p2: torch.Tensor, radar_p2: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(torch.cat((rgb_p2, radar_p2), dim=1)))


class NativeCenterHead(torch.nn.Module):
    """One class's native head: heatmap + private centre offsets + 12 regression fields."""

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            ConvGNAct(int(in_channels), int(hidden)),
            ConvGNAct(int(hidden), int(hidden)),
        )
        self.heatmap = torch.nn.Conv2d(int(hidden), 1, 1)
        self.offset = torch.nn.Conv2d(int(hidden), 2, 1)
        self.regression = torch.nn.Conv2d(int(hidden), REG_FIELDS, 1)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.trunk(feature)
        return {
            "heatmap": self.heatmap(shared),
            "offset": self.offset(shared),
            "reg": self.regression(shared),
        }


class PersonStride2Feature(torch.nn.Module):
    """Compact stride-2 RGB/radar feature: learned 2x upsample of the fused
    stride-4 context, concatenated with the stride-2 skip that crossed the split."""

    def __init__(self, context_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = UpConvGNAct(int(context_channels), int(out_channels))
        self.fuse = ConvGNAct(int(out_channels) + int(skip_channels), int(out_channels))

    def forward(self, context: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(context)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((up, skip), dim=1))


class SegmentationDecoderV2(torch.nn.Module):
    """Lightweight task-specific decoder: stride-4 fused context -> learned 2x ->
    stride-2 RGB/radar skip -> learned 2x to the final full-resolution output.

    Deliberately small: two transposed convs and two 3x3 convs.  No HRNet.
    Only the 3-channel logits ever exist at full resolution.
    """

    def __init__(self, context_channels: int, skip_channels: int, num_classes: int) -> None:
        super().__init__()
        self.context = ConvGNAct(int(context_channels), 64)
        self.up_to_half = UpConvGNAct(64, 48)
        self.fuse_skip = ConvGNAct(48 + int(skip_channels), 48)
        self.up_to_full = torch.nn.ConvTranspose2d(48, int(num_classes), 4, stride=2, padding=1)

    def forward(self, context: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_to_half(self.context(context))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_skip(torch.cat((x, skip), dim=1))
        return self.up_to_full(x)


class CleanCenterNetV2(torch.nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        radar_channels: int,
        pretrained: bool,
        fpn_channels: int = FPN_CHANNELS,
        head_channels: int = HEAD_CHANNELS,
        stride2_channels: int = STRIDE2_CHANNELS,
        person_channels: int = PERSON_CHANNELS,
    ) -> None:
        super().__init__()
        self.head_arch = HEAD_ARCH_NAME
        self.num_classes = int(num_classes)
        self.radar_channels = int(radar_channels)
        self.vehicle_stride = VEHICLE_STRIDE
        self.person_stride = PERSON_STRIDE
        self.reg_fields = REG_FIELDS
        self.object_class_names = ("vehicle", "person")

        # --- warm-startable front (v1 parameter names) ---
        self.backbone = RGBResNet34Backbone(pretrained=bool(pretrained))
        self.radar_encoder = RadarEncoder(self.radar_channels)
        self.rgb_fpn = FeaturePyramid((64, 128, 256, 512), int(fpn_channels))
        self.radar_fpn = FeaturePyramid((64, 96, 128, 160), int(fpn_channels))
        # --- new v2 front module: the compact stride-2 RGB/radar skip ---
        self.stride2_proj = torch.nn.Sequential(
            ConvGNAct(64 + 32, 32),
            torch.nn.Conv2d(32, int(stride2_channels), 1, bias=False),
            _gn(int(stride2_channels)),
            torch.nn.ReLU(inplace=True),
        )
        # --- tail ---
        self.fusion = RadarFusion(int(fpn_channels), int(fpn_channels))
        self.vehicle_head = NativeCenterHead(int(fpn_channels), int(head_channels))
        self.person_feature = PersonStride2Feature(
            int(fpn_channels), int(stride2_channels), int(person_channels)
        )
        self.person_head = NativeCenterHead(int(person_channels), int(person_channels))
        self.classifier = SegmentationDecoderV2(
            int(fpn_channels), int(stride2_channels), int(num_classes)
        )
        self.feature_ae = None  # future attachment point; disabled in this clean run
        self._init_new_modules()

    # ------------------------------------------------------------------ init
    def _new_modules(self):
        return (
            self.stride2_proj,
            self.fusion,
            self.vehicle_head,
            self.person_feature,
            self.person_head,
            self.classifier,
        )

    def _init_new_modules(self) -> None:
        for root in self._new_modules():
            for module in root.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.GroupNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
        # Standard CenterNet p=0.1 heatmap prior.
        for head in (self.vehicle_head.heatmap, self.person_head.heatmap):
            torch.nn.init.normal_(head.weight, std=0.001)
            torch.nn.init.constant_(head.bias, -2.19)
        # Offsets start at zero-mean, tiny: the centre begins at the cell centre.
        for head in (self.vehicle_head.offset, self.person_head.offset):
            torch.nn.init.normal_(head.weight, std=1e-3)
            torch.nn.init.constant_(head.bias, 0.5)  # sigmoid-free; target is in [0,1)
        # 2D-box channels start small and positive after softplus (v1 convention).
        for head in (self.vehicle_head.regression, self.person_head.regression):
            with torch.no_grad():
                head.bias[-2:].fill_(-3.0)

    # ----------------------------------------------------------- split front
    def encode_front(
        self, rgb: torch.Tensor, radar: torch.Tensor
    ) -> "OrderedDict[str, torch.Tensor]":
        c1, c2, c3, c4, c5 = self.backbone(rgb)
        r1, r2, r3, r4, r5 = self.radar_encoder(radar)
        rgb_p2 = self.rgb_fpn((c2, c3, c4, c5))
        radar_p2 = self.radar_fpn((r2, r3, r4, r5))
        if r1.shape[-2:] != c1.shape[-2:]:
            r1 = F.interpolate(r1, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.stride2_proj(torch.cat((c1, r1), dim=1))
        return OrderedDict((("rgb_p2", rgb_p2), ("radar_p2", radar_p2), ("s2", s2)))

    # ------------------------------------------------------------ split tail
    def decode_tail(
        self,
        feature_bundle: Dict[str, torch.Tensor],
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Consumes ONLY ``feature_bundle``.  No raw RGB/radar side channel."""
        rgb_p2 = feature_bundle["rgb_p2"]
        radar_p2 = feature_bundle["radar_p2"]
        s2 = feature_bundle["s2"]
        fused = self.fusion(rgb_p2, radar_p2)
        vehicle = self.vehicle_head(fused)
        person = self.person_head(self.person_feature(fused, s2))
        segmentation = self.classifier(fused, s2)
        if out_hw is not None and tuple(segmentation.shape[-2:]) != tuple(out_hw):
            segmentation = F.interpolate(
                segmentation, size=tuple(out_hw), mode="bilinear", align_corners=False
            )
        return {
            "out": segmentation,
            "veh_hm": vehicle["heatmap"],
            "veh_off": vehicle["offset"],
            "veh_reg": vehicle["reg"],
            "per_hm": person["heatmap"],
            "per_off": person["offset"],
            "per_reg": person["reg"],
        }

    def _apply_feature_ae(self, bundle):
        if self.feature_ae is None:
            return OrderedDict(bundle)
        raise RuntimeError("feature AE is intentionally disabled for the clean v2 run")

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = inputs[:, :3]
        radar = inputs[:, 3 : 3 + self.radar_channels]
        bundle = self.encode_front(rgb, radar)
        if self.feature_ae is not None:
            bundle = self._apply_feature_ae(bundle)
        return self.decode_tail(bundle, (int(inputs.shape[-2]), int(inputs.shape[-1])))


def build_centernet_v2(
    *, num_classes: int = 3, radar_channels: int = 4, pretrained: bool = False, **_: object
) -> CleanCenterNetV2:
    return CleanCenterNetV2(
        num_classes=int(num_classes),
        radar_channels=int(radar_channels),
        pretrained=bool(pretrained),
    )


# --------------------------------------------------------------- warm start
WARM_START_PREFIXES = ("backbone.", "rgb_fpn.", "radar_encoder.", "radar_fpn.")


def warm_start_from_v1(
    model: torch.nn.Module, checkpoint_state: Dict[str, torch.Tensor]
) -> Dict[str, object]:
    """Load every shape-compatible ResNet34 / RGB-FPN / radar-encoder / radar-FPN
    tensor from the v1 epoch-12 state dict.  Returns the full mapping report."""
    current = model.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    incompatible = []
    ignored_source = []
    for key, tensor in checkpoint_state.items():
        name = key[7:] if str(key).startswith("module.") else str(key)
        if not name.startswith(WARM_START_PREFIXES):
            ignored_source.append(name)
            continue
        if name not in current:
            incompatible.append({"tensor": name, "reason": "absent_in_v2"})
            continue
        if tuple(current[name].shape) != tuple(tensor.shape):
            incompatible.append(
                {
                    "tensor": name,
                    "reason": "shape_mismatch",
                    "v1_shape": list(tensor.shape),
                    "v2_shape": list(current[name].shape),
                }
            )
            continue
        loaded[name] = tensor
    missing = model.load_state_dict(loaded, strict=False).missing_keys
    return {
        "loaded_tensors": sorted(loaded),
        "loaded_count": len(loaded),
        "new_tensors": sorted(missing),
        "new_count": len(missing),
        "incompatible_tensors": incompatible,
        "incompatible_count": len(incompatible),
        "source_tensors_outside_warm_start_scope": sorted(ignored_source),
        "warm_start_prefixes": list(WARM_START_PREFIXES),
    }
