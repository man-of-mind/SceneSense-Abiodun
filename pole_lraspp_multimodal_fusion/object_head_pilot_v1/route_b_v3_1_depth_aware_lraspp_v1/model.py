from __future__ import annotations

import copy
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

MODEL_SIZE_WH = (768, 432)
NATIVE_STRIDE = 4
NATIVE_GRID_HW = (108, 192)
DEPTH_BINS = 32

COMMON_FIELDS = OrderedDict([
    ("heatmap", 1),
    ("subcell", 2),
    ("box_center_delta", 2),
    ("box_wh", 2),
    ("physical_ray_delta", 2),
    ("depth_bin_logits", DEPTH_BINS),
    ("depth_bin_residuals", DEPTH_BINS),
    ("log_dimensions", 3),
    ("yaw_sincos", 2),
    ("radar_support", 1),
])


class SevenChannelStem(torch.nn.Module):
    """Exactly one seven-channel convolution parameterized as RGB plus radar."""

    def __init__(self, official_stem: torch.nn.Module) -> None:
        super().__init__()
        source: torch.nn.Conv2d = official_stem[0]
        kwargs = dict(
            out_channels=source.out_channels,
            kernel_size=source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=False,
            padding_mode=source.padding_mode,
        )
        self.rgb_conv = torch.nn.Conv2d(in_channels=3, **kwargs)
        self.radar_conv = torch.nn.Conv2d(in_channels=4, **kwargs)
        self.norm = copy.deepcopy(official_stem[1])
        self.activation = copy.deepcopy(official_stem[2])
        with torch.no_grad():
            self.rgb_conv.weight.copy_(source.weight)
            self.radar_conv.weight.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 7:
            raise ValueError(f"expected [B,7,H,W], got {tuple(value.shape)}")
        fused = self.rgb_conv(value[:, :3]) + self.radar_conv(value[:, 3:])
        return self.activation(self.norm(fused))

    def concatenated_weight(self) -> torch.Tensor:
        return torch.cat([self.rgb_conv.weight, self.radar_conv.weight], dim=1)


class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )


class SharedDepthAwareNeck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.high_projection = ConvBNReLU(960, 128, 1)
        self.low_projection = ConvBNReLU(40, 48, 1)
        self.fusion = torch.nn.Sequential(ConvBNReLU(176, 128, 3), ConvBNReLU(128, 128, 3))
        self.upsample = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(features) != {"low", "high"}:
            raise ValueError(f"tail accepts only low/high, got {sorted(features)}")
        low = self.low_projection(features["low"])
        high = self.high_projection(features["high"])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.upsample(self.fusion(torch.cat([low, high], dim=1)))


class ObjectBranch(torch.nn.Module):
    def __init__(self, class_name: str) -> None:
        super().__init__()
        self.class_name = str(class_name)
        self.trunk = ConvBNReLU(128, 128, 3)
        fields = OrderedDict(COMMON_FIELDS)
        if self.class_name == "vehicle":
            fields["parked"] = 1
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Conv2d(128, channels, kernel_size=1)
            for name, channels in fields.items()
        })
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        with torch.no_grad():
            # Field heads represent registered output priors. Exact-zero final
            # weights make those priors effective without changing capacity;
            # the object trunk retains its Kaiming initialization.
            for head in self.heads.values():
                head.weight.zero_()
            self.heads["heatmap"].bias.fill_(-4.6)
            self.heads["subcell"].weight.zero_()
            self.heads["subcell"].bias.zero_()
            self.heads["box_wh"].bias.fill_(math.log(math.expm1(5.0)))
            dims = (4.0, 1.8, 1.6) if self.class_name == "vehicle" else (0.6, 0.6, 1.7)
            self.heads["log_dimensions"].bias.copy_(torch.log(torch.tensor(dims)))
            self.heads["yaw_sincos"].bias[1] = 1.0

    def forward(self, shared: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        value = self.trunk(shared)
        return OrderedDict((name, head(value)) for name, head in self.heads.items())


class DepthAwareLRASPP(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.depth_neck = SharedDepthAwareNeck()
        self.segmentation = torch.nn.Sequential(ConvBNReLU(128, 64, 3), torch.nn.Conv2d(64, 3, 1))
        self.dense_depth = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(64, 1, 1),
        )
        self.vehicle = ObjectBranch("vehicle")
        self.person = ObjectBranch("person")
        anchors = torch.linspace(0.0, math.log1p(40.0), DEPTH_BINS, dtype=torch.float32)
        self.register_buffer("depth_anchors", anchors, persistent=True)
        self.register_buffer("depth_delta", anchors[1] - anchors[0], persistent=True)
        self._initialize_new_tail()

    def _initialize_new_tail(self) -> None:
        # Never traverse the pretrained backbone here. Only newly constructed tail
        # modules are initialized; the two object branches initialize themselves.
        for root in (self.depth_neck, self.segmentation, self.dense_depth):
            for module in root.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)

    def encode_front(self, value: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        result = self.backbone(value)
        return OrderedDict((name, result[name]) for name in ("low", "high"))

    def decode_tail(self, features: Mapping[str, torch.Tensor], *, dense: bool = False) -> dict[str, Any]:
        shared = self.depth_neck(features)
        segmentation = self.segmentation(shared)
        segmentation = F.interpolate(segmentation, size=(MODEL_SIZE_WH[1], MODEL_SIZE_WH[0]), mode="bilinear", align_corners=False)
        result: dict[str, Any] = {
            "out": segmentation,
            "objects": OrderedDict([
                ("vehicle", self.vehicle(shared)),
                ("person", self.person(shared)),
            ]),
        }
        if dense:
            result["dense_depth_log1p"] = self.dense_depth(shared)
        return result

    def forward(self, value: torch.Tensor, *, dense: bool = False) -> dict[str, Any]:
        return self.decode_tail(self.encode_front(value), dense=dense)


def build_model(weight_path: Path, device: torch.device | None = None) -> tuple[DepthAwareLRASPP, dict[str, Any]]:
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    official = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
    base = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=3)
    backbone = base.backbone
    current = backbone.state_dict()
    compatible: OrderedDict[str, torch.Tensor] = OrderedDict()
    source_feature_keys = [name for name in official if name.startswith("features.")]
    incompatible: list[dict[str, Any]] = []
    for source_name in source_feature_keys:
        target_name = source_name[len("features."):]
        value = official[source_name]
        if target_name in current and tuple(current[target_name].shape) == tuple(value.shape):
            compatible[target_name] = value
        else:
            incompatible.append({"source": source_name, "target": target_name, "shape": list(value.shape)})
    missing = backbone.load_state_dict(compatible, strict=False)
    if incompatible or missing.unexpected_keys or missing.missing_keys:
        raise RuntimeError(
            f"official feature mapping incomplete incompatible={incompatible[:3]} "
            f"missing={missing.missing_keys[:3]} unexpected={missing.unexpected_keys[:3]}"
        )
    official_stem = backbone["0"]
    backbone["0"] = SevenChannelStem(official_stem)
    model = DepthAwareLRASPP(backbone)
    report = {
        "official_state_tensors": len(official),
        "official_feature_tensors": len(source_feature_keys),
        "compatible_feature_tensors_loaded": len(compatible),
        "incompatible_feature_tensors": incompatible,
        "classifier_tensors_loaded": 0,
        "rgb_stem_exact": torch.equal(model.backbone["0"].rgb_conv.weight, official["features.0.0.weight"]),
        "radar_stem_exact_zero": bool(torch.count_nonzero(model.backbone["0"].radar_conv.weight).item() == 0),
    }
    if not report["rgb_stem_exact"] or not report["radar_stem_exact_zero"]:
        raise RuntimeError("seven-channel stem initialization failed")
    if device is not None:
        model.to(device)
    return model, report


def freeze_bn_running_state(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_stage(model: DepthAwareLRASPP, stage: str) -> None:
    if stage not in {"A", "B"}:
        raise ValueError(stage)
    for parameter in model.parameters():
        parameter.requires_grad = True
    if stage == "A":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone["0"].radar_conv.parameters():
            parameter.requires_grad = True
    freeze_bn_running_state(model)


def stage_train_mode(model: DepthAwareLRASPP, stage: str) -> None:
    model.train()
    if stage == "A":
        model.backbone.eval()
        model.backbone["0"].radar_conv.train()
    freeze_bn_running_state(model)


def parameter_groups(model: DepthAwareLRASPP) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    result = {"new_decay": [], "new_no_decay": [], "backbone_decay": [], "backbone_no_decay": []}
    for name, parameter in model.named_parameters():
        backbone = name.startswith("backbone.") and not name.startswith("backbone.0.radar_conv")
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        group = ("backbone" if backbone else "new") + ("_no_decay" if no_decay else "_decay")
        result[group].append((name, parameter))
    return result


def parameter_report(model: DepthAwareLRASPP) -> dict[str, Any]:
    modules = OrderedDict([
        ("backbone", model.backbone),
        ("rgb_stem", model.backbone["0"].rgb_conv),
        ("radar_stem", model.backbone["0"].radar_conv),
        ("depth_neck", model.depth_neck),
        ("segmentation", model.segmentation),
        ("dense_depth", model.dense_depth),
        ("vehicle_branch", model.vehicle),
        ("person_branch", model.person),
    ])
    report: OrderedDict[str, Any] = OrderedDict()
    for name, module in modules.items():
        report[name] = {
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
        }
    report["model"] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    report["optimizer_groups"] = {
        name: {"tensors": len(values), "parameters": sum(parameter.numel() for _, parameter in values)}
        for name, values in parameter_groups(model).items()
    }
    report["class_private_heads"] = {
        class_name: {
            field: sum(parameter.numel() for parameter in head.parameters())
            for field, head in getattr(model, class_name).heads.items()
        }
        for class_name in ("vehicle", "person")
    }
    return report


def pretrained_backbone_state(model: DepthAwareLRASPP) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items()
        if name.startswith("backbone.") and not name.startswith("backbone.0.radar_conv")
    )


def split_report(model: DepthAwareLRASPP, value: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        monolithic = model(value, dense=True)
        bundle = model.encode_front(value)
        split = model.decode_tail(bundle, dense=True)
    tensors: list[tuple[str, torch.Tensor, torch.Tensor]] = [("out", monolithic["out"], split["out"])]
    tensors.append(("dense_depth_log1p", monolithic["dense_depth_log1p"], split["dense_depth_log1p"]))
    for class_name in ("vehicle", "person"):
        for field in monolithic["objects"][class_name]:
            tensors.append((f"objects.{class_name}.{field}", monolithic["objects"][class_name][field], split["objects"][class_name][field]))
    raw_bytes = sum(item.numel() * item.element_size() for item in bundle.values())
    serialized = b"".join(item.detach().cpu().contiguous().numpy().tobytes() for item in bundle.values())
    return {
        "transport_names": list(bundle),
        "shapes": {name: list(item.shape) for name, item in bundle.items()},
        "dtypes": {name: str(item.dtype) for name, item in bundle.items()},
        "raw_bytes": raw_bytes,
        "identity_serialized_bytes": len(serialized),
        "tail_inputs": ["low", "high"],
        "raw_equal": {name: torch.equal(left, right) for name, left, right in tensors},
        "all_raw_equal": all(torch.equal(left, right) for _, left, right in tensors),
    }


def stem_equivalence_report(model: DepthAwareLRASPP, value: torch.Tensor) -> dict[str, Any]:
    stem: SevenChannelStem = model.backbone["0"]
    ordinary = torch.nn.Conv2d(7, stem.rgb_conv.out_channels, stem.rgb_conv.kernel_size,
                              stem.rgb_conv.stride, stem.rgb_conv.padding,
                              dilation=stem.rgb_conv.dilation, groups=stem.rgb_conv.groups,
                              bias=False).to(device=value.device, dtype=value.dtype)
    with torch.no_grad():
        ordinary.weight.copy_(stem.concatenated_weight())
        parameterized = stem.rgb_conv(value[:, :3]) + stem.radar_conv(value[:, 3:])
        concatenated = ordinary(value)
    return {
        "rgb_bias": stem.rgb_conv.bias is not None,
        "radar_bias": stem.radar_conv.bias is not None,
        "weight_shape": list(stem.concatenated_weight().shape),
        "conv_outputs_equal": torch.equal(parameterized, concatenated),
        "max_abs_delta": float((parameterized - concatenated).abs().max().item()),
    }
