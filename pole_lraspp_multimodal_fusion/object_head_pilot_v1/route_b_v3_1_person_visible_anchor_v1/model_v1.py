#!/usr/bin/env python3
"""Frozen epoch-40 LR-ASPP with one append-only person-private tail."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_NATIVE_SPEC = importlib.util.spec_from_file_location(
    "route_b_v3_1_native_grid_model_for_visible_anchor_v1",
    NATIVE_PACKAGE / "model_v1.py",
)
if _NATIVE_SPEC is None or _NATIVE_SPEC.loader is None:
    raise ImportError("unable to load frozen native-grid model")
native = importlib.util.module_from_spec(_NATIVE_SPEC)
_NATIVE_SPEC.loader.exec_module(native)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

# Compatibility constants used by frozen native target/decoder modules when they are
# loaded beside this versioned package.
MODEL_SIZE = native.MODEL_SIZE
NATIVE_STRIDE = native.NATIVE_STRIDE
NATIVE_GRID = native.NATIVE_GRID
HEATMAP_CHANNELS = native.HEATMAP_CHANNELS
REG_CHANNELS = native.REG_CHANNELS
OFFSET_CHANNELS = native.OFFSET_CHANNELS
SL_REG = native.SL_REG
SL_OFFSET = native.SL_OFFSET


PRIVATE_HEAD_CHANNELS = OrderedDict([
    ("visible_heatmap", 1),
    ("visible_subcell_offset", 2),
    ("visible_to_box_center_offset", 2),
    ("full_box_wh", 2),
    ("visible_to_physical_ray_offset", 2),
    ("positive_camera_forward_depth", 1),
    ("person_dimensions", 3),
    ("person_yaw", 2),
    ("radar_support", 1),
])


class PersonPrivateTower(torch.nn.Module):
    """Private copy of the proven fused low/high tower plus factorized heads."""

    def __init__(self, shared_trunk: torch.nn.Module, upsampler: torch.nn.Module,
                 depth_bounds_m: tuple[float, float]) -> None:
        super().__init__()
        self.shared_trunk = copy.deepcopy(shared_trunk)
        self.upsampler = copy.deepcopy(upsampler)
        hidden = int(self.upsampler[0].out_channels)
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Conv2d(hidden, channels, kernel_size=1)
            for name, channels in PRIVATE_HEAD_CHANNELS.items()
        })
        self.depth_bounds_m = tuple(float(value) for value in depth_bounds_m)
        self._initialize_new_heads()
        self.freeze_private_batch_norm()

    def _initialize_new_heads(self) -> None:
        for head in self.heads.values():
            torch.nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(head.bias)
        low, high = self.depth_bounds_m
        prior = (math.log(20.0) - math.log(low)) / (math.log(high) - math.log(low))
        prior = min(1.0 - 1e-5, max(1e-5, prior))
        with torch.no_grad():
            self.heads["positive_camera_forward_depth"].bias.fill_(
                math.log(prior / (1.0 - prior))
            )

    def freeze_private_batch_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, fused_low_high: torch.Tensor) -> dict[str, torch.Tensor]:
        # The complete private path, including geometry heads, is explicitly FP32.
        device_type = fused_low_high.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            feature = self.upsampler(self.shared_trunk(fused_low_high.detach().float()))
            return {name: head(feature) for name, head in self.heads.items()}


class PersonVisibleAnchorLRASPP(native.NativeGridFusionLRASPP):
    """All inherited outputs are untouched; only a private person dictionary is added."""

    def __init__(self, base_model: torch.nn.Module, *, hidden_channels: int = 128,
                 head_depth: int = 3, depth_bounds_m: tuple[float, float] = (0.05, 40.0)) -> None:
        super().__init__(base_model, hidden_channels=hidden_channels, head_depth=head_depth)
        self.person_private = PersonPrivateTower(
            self.object_head.shared_trunk, self.object_head.upsampler, depth_bounds_m,
        )

    def tail_outputs(self, features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        # These two calls are the original native-grid inference equations.
        segmentation = self.classifier(features)
        if isinstance(segmentation, dict):
            segmentation = segmentation["out"]
        base_object = self.object_logits(features)
        private = self.person_private(self._object_input(features))
        return {"out": segmentation, "object": base_object, "person_private": private}

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> dict[str, Any]:
        if float(feature_drop_fraction) != 0.0:
            raise ValueError("visible-anchor experiment is clean q=0 only")
        return self.tail_outputs(self.backbone(x))

    def private_training_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # The entire inherited graph, including every BN running statistic, is frozen.
        with torch.no_grad():
            features = self.backbone(x)
            fused_low_high = self._object_input(features).detach()
        return self.person_private(fused_low_high)


def build_model(*, radar_channels: int = 4, hidden_channels: int = 128,
                head_depth: int = 3, depth_bounds_m: tuple[float, float] = (0.05, 40.0),
                device: Optional[torch.device] = None) -> PersonVisibleAnchorLRASPP:
    base = native.build_fusion_lraspp(
        num_classes=3, radar_channels=radar_channels, pretrained=False,
        init_checkpoint="", device=device,
    )
    model = PersonVisibleAnchorLRASPP(
        base, hidden_channels=hidden_channels, head_depth=head_depth,
        depth_bounds_m=depth_bounds_m,
    )
    return model.to(device) if device is not None else model


def _copy_slice(target: torch.nn.Conv2d, source: torch.nn.Conv2d, channels: slice) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight[channels])
        target.bias.copy_(source.bias[channels])


def initialize_private_from_inherited(model: PersonVisibleAnchorLRASPP) -> dict[str, Any]:
    """Copy only compatible epoch-40 tensors after the inherited checkpoint is loaded."""
    private = model.person_private
    private.shared_trunk.load_state_dict(model.object_head.shared_trunk.state_dict(), strict=True)
    private.upsampler.load_state_dict(model.object_head.upsampler.state_dict(), strict=True)
    _copy_slice(private.heads["visible_heatmap"], model.object_head.person_heatmap_head, slice(0, 1))
    _copy_slice(private.heads["visible_subcell_offset"], model.object_head.offset_head, slice(0, 2))
    _copy_slice(private.heads["full_box_wh"], model.object_head.regression_head, slice(10, 12))
    _copy_slice(private.heads["person_dimensions"], model.object_head.regression_head, slice(3, 6))
    _copy_slice(private.heads["person_yaw"], model.object_head.regression_head, slice(6, 8))
    _copy_slice(private.heads["radar_support"], model.object_head.regression_head, slice(9, 10))
    private.freeze_private_batch_norm()
    return {
        "tower": ["object_head.shared_trunk", "object_head.upsampler"],
        "copied_heads": {
            "visible_heatmap": "person_heatmap_head[0:1]",
            "visible_subcell_offset": "offset_head[0:2]",
            "full_box_wh": "regression_head[10:12]",
            "person_dimensions": "regression_head[3:6]",
            "person_yaw": "regression_head[6:8]",
            "radar_support": "regression_head[9:10]",
        },
        "new_heads": [
            "visible_to_box_center_offset", "visible_to_physical_ray_offset",
            "positive_camera_forward_depth",
        ],
    }


def load_epoch40(model: PersonVisibleAnchorLRASPP, checkpoint: Path,
                 *, device: torch.device, initialize_private: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    only_private = all(key.startswith("person_private.") for key in incompatible.missing_keys)
    if incompatible.unexpected_keys or not only_private:
        raise RuntimeError(
            f"epoch-40 mapping failure missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    mapping = initialize_private_from_inherited(model) if initialize_private else None
    model.to(device)
    return {
        "checkpoint": str(checkpoint), "checkpoint_epoch": int(payload["epoch"]),
        "inherited_tensor_count": len(payload["model"]),
        "missing_private_tensors": sorted(incompatible.missing_keys),
        "all_inherited_tensors_loaded": only_private and not incompatible.unexpected_keys,
        "private_initialization": mapping,
    }


def configure_private_training(model: PersonVisibleAnchorLRASPP) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.person_private.named_parameters():
        if "shared_trunk" in name or "upsampler" in name or "heads" in name:
            parameter.requires_grad = True
    model.person_private.freeze_private_batch_norm()
    model.eval()
    model.person_private.train()
    model.person_private.freeze_private_batch_norm()


def private_parameters(model: PersonVisibleAnchorLRASPP) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.person_private.parameters() if parameter.requires_grad]


def inherited_state(model: PersonVisibleAnchorLRASPP) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value) for name, value in model.state_dict().items()
        if not name.startswith("person_private.")
    )


def parameter_report(model: PersonVisibleAnchorLRASPP) -> dict[str, Any]:
    groups = OrderedDict([
        ("backbone", model.backbone),
        ("segmentation", model.classifier),
        ("native_shared_trunk", model.object_head.shared_trunk),
        ("native_upsampler", model.object_head.upsampler),
        ("vehicle_heatmap", model.object_head.vehicle_heatmap_head),
        ("inherited_person_heatmap", model.object_head.person_heatmap_head),
        ("shared_regression", model.object_head.regression_head),
        ("inherited_offset", model.object_head.offset_head),
        ("person_private", model.person_private),
    ])
    report: dict[str, Any] = {}
    for name, module in groups.items():
        total = sum(int(parameter.numel()) for parameter in module.parameters())
        trainable = sum(int(parameter.numel()) for parameter in module.parameters()
                        if parameter.requires_grad)
        report[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters()
                    if parameter.requires_grad)
    report["model_total"] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    report["private_heads"] = {
        name: {
            "total": sum(int(parameter.numel()) for parameter in head.parameters()),
            "trainable": sum(int(parameter.numel()) for parameter in head.parameters()
                             if parameter.requires_grad),
        }
        for name, head in model.person_private.heads.items()
    }
    return report


def encode_front(model: PersonVisibleAnchorLRASPP,
                 x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((str(name), value) for name, value in model.backbone(x).items())


def decode_tail(model: PersonVisibleAnchorLRASPP,
                features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return model.tail_outputs(features)


def split_boundary_report(model: PersonVisibleAnchorLRASPP,
                          x: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        monolithic = model(x)
        bundle = encode_front(model, x)
        split = decode_tail(model, bundle)
    deltas = {
        name: float((monolithic[name].float() - split[name].float()).abs().max().item())
        for name in ("out", "object")
    }
    for name in PRIVATE_HEAD_CHANNELS:
        deltas[f"person_private.{name}"] = float((
            monolithic["person_private"][name].float()
            - split["person_private"][name].float()
        ).abs().max().item())
    return {
        "transported_feature_names": list(bundle),
        "transported_feature_shapes": {name: list(value.shape) for name, value in bundle.items()},
        "transported_feature_elements": {name: int(value.numel()) for name, value in bundle.items()},
        "transported_feature_dtypes": {name: str(value.dtype) for name, value in bundle.items()},
        "tail_reads_only_low_high": list(bundle) == ["low", "high"],
        "tail_calibration_metadata": ["camera_intrinsics", "camera_to_world"],
        "tail_raw_sensor_side_channels": [],
        "monolithic_vs_split_max_abs_delta": deltas,
        "outputs_bit_identical": all(value == 0.0 for value in deltas.values()),
    }
