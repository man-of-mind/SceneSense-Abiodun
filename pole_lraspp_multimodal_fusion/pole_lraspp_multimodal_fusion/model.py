from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


OBJECT_HEAD_CHANNELS = 11


class MultiTaskFusionLRASPP(torch.nn.Module):
    """LR-ASPP segmentation backbone with learned object localization heads."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        object_channels: int = OBJECT_HEAD_CHANNELS,
        hidden_channels: int = 128,
        fuse_low_into_object_head: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        try:
            high_channels = int(base_model.classifier.cbr[0].in_channels)
        except Exception:
            high_channels = 960
        self.fuse_low_into_object_head = bool(fuse_low_into_object_head)
        if self.fuse_low_into_object_head:
            try:
                low_channels = int(base_model.classifier.low_classifier.in_channels)
            except Exception:
                low_channels = 40
            object_in_channels = high_channels + low_channels
        else:
            low_channels = 0
            object_in_channels = high_channels
        self.object_head = torch.nn.Sequential(
            torch.nn.Conv2d(int(object_in_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(int(hidden_channels)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(int(hidden_channels)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(hidden_channels), int(object_channels), kernel_size=1),
        )
        self.object_channels = int(object_channels)
        self._init_object_head()

    def _init_object_head(self) -> None:
        for module in self.object_head.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        final = self.object_head[-1]
        if isinstance(final, torch.nn.Conv2d) and final.bias is not None:
            with torch.no_grad():
                final.bias.zero_()
                final.bias[0] = -4.6

    def _high_feature(self, features: object) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        if isinstance(features, dict):
            if "high" in features:
                return features["high"]
            if "out" in features:
                return features["out"]
            return list(features.values())[-1]
        raise TypeError(f"Unsupported backbone feature type: {type(features)!r}")

    def _low_feature(self, features: object) -> torch.Tensor:
        if isinstance(features, dict) and "low" in features:
            return features["low"]
        raise RuntimeError(
            "fuse_low_into_object_head=True requires the backbone to expose a 'low' feature "
            "(LR-ASPP MobileNetV3-Large does)."
        )

    def _object_input(self, features: object) -> torch.Tensor:
        high = self._high_feature(features)
        if not self.fuse_low_into_object_head:
            return high
        low = self._low_feature(features)
        if tuple(high.shape[-2:]) != tuple(low.shape[-2:]):
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([low, high], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        seg = self.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        object_logits = self.object_head(self._object_input(features))
        if tuple(object_logits.shape[-2:]) != tuple(x.shape[-2:]):
            object_logits = F.interpolate(object_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": seg, "object": object_logits}


def build_lraspp(num_classes: int, pretrained: bool) -> torch.nn.Module:
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
    from torchvision.models.segmentation.lraspp import LRASPPHead

    try:
        if pretrained:
            model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    except Exception:
        model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    high_channels = int(model.classifier.cbr[0].in_channels)
    inter_channels = int(model.classifier.cbr[0].out_channels)
    low_channels = int(model.classifier.low_classifier.in_channels)
    try:
        model.classifier = LRASPPHead(low_channels, high_channels, int(num_classes), inter_channels)
    except TypeError:
        model.classifier = LRASPPHead(low_channels, high_channels, int(num_classes))
    return model


def _first_conv_parent(model: torch.nn.Module) -> Tuple[torch.nn.Module, str, torch.nn.Conv2d]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and int(module.in_channels) == 3:
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            return parent, child_name, module
    raise RuntimeError("Unable to find the first 3-channel Conv2d in LR-ASPP.")


def adapt_first_conv_in_channels(model: torch.nn.Module, in_channels: int) -> torch.nn.Module:
    parent, child_name, old_conv = _first_conv_parent(model)
    if int(old_conv.in_channels) == int(in_channels):
        return model
    new_conv = torch.nn.Conv2d(
        in_channels=int(in_channels),
        out_channels=int(old_conv.out_channels),
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :3].copy_(old_conv.weight)
        if int(in_channels) > 3:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for channel in range(3, int(in_channels)):
                new_conv.weight[:, channel : channel + 1].copy_(mean_weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    setattr(parent, child_name, new_conv)
    return model


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint did not contain a state_dict.")


def load_compatible_state_dict(model: torch.nn.Module, checkpoint_path: str, *, device: torch.device) -> Dict[str, int]:
    if not checkpoint_path:
        return {"loaded": 0, "skipped": 0}
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = _extract_state_dict(torch.load(path, map_location=device))
    current = model.state_dict()
    compatible: Dict[str, torch.Tensor] = OrderedDict()
    skipped = 0
    for key, tensor in state.items():
        key2 = key[7:] if key.startswith("module.") else key
        if key2 in current and tuple(current[key2].shape) == tuple(tensor.shape):
            compatible[key2] = tensor
        elif (
            key2 in current
            and current[key2].ndim == 4
            and tensor.ndim == 4
            and int(tensor.shape[1]) == 3
            and int(current[key2].shape[1]) > 3
            and tuple(current[key2].shape[0:1] + current[key2].shape[2:]) == tuple(tensor.shape[0:1] + tensor.shape[2:])
        ):
            expanded = current[key2].clone()
            expanded[:, :3].copy_(tensor)
            mean_weight = tensor.mean(dim=1, keepdim=True)
            for channel in range(3, int(current[key2].shape[1])):
                expanded[:, channel : channel + 1].copy_(mean_weight)
            compatible[key2] = expanded
        else:
            skipped += 1
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped}


def build_fusion_lraspp(
    *,
    num_classes: int,
    radar_channels: int,
    pretrained: bool,
    init_checkpoint: str = "",
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    device = device or torch.device("cpu")
    model = build_lraspp(num_classes, pretrained)
    adapt_first_conv_in_channels(model, 3 + int(radar_channels))
    if init_checkpoint:
        load_compatible_state_dict(model, init_checkpoint, device=device)
    return model


def build_multitask_fusion_lraspp(
    *,
    num_classes: int,
    radar_channels: int,
    pretrained: bool,
    init_checkpoint: str = "",
    object_channels: int = OBJECT_HEAD_CHANNELS,
    object_hidden_channels: int = 128,
    fuse_low_into_object_head: bool = False,
    device: Optional[torch.device] = None,
) -> MultiTaskFusionLRASPP:
    base = build_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=pretrained,
        init_checkpoint=init_checkpoint,
        device=device,
    )
    return MultiTaskFusionLRASPP(
        base,
        object_channels=int(object_channels),
        hidden_channels=int(object_hidden_channels),
        fuse_low_into_object_head=bool(fuse_low_into_object_head),
    )
