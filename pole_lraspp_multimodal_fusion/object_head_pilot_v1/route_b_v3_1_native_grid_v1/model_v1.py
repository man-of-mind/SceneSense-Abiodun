#!/usr/bin/env python3
"""Native stride-4 tail-side object decoder for the unchanged 7-channel LR-ASPP model.

Everything here lives strictly on the EDGE TAIL side of the split boundary. The front
still produces exactly the existing low/high backbone bundle; nothing is added to the
transported payload, so q / quantization / AE / zstd continue to operate on the same
feature bundle they operate on today.

Structure (only the bracketed parts are new):

    backbone -> {low: 40ch @ stride 8, high: 960ch @ stride 16}      <-- TRANSPORTED, unchanged
      tail:
        classifier(features) -> segmentation                          <-- unchanged
        _object_input(features) -> concat(low, high@stride8) = 1000ch @ 54x96
          shared_trunk (3 x conv3x3+BN+ReLU, 1000->128)               <-- warm-started
          [upsampler: ConvTranspose2d(128,128,4,2,1)+BN+ReLU]         <-- NEW, 108x192 (stride 4)
          vehicle_heatmap_head  Conv1x1(128,1)                        <-- warm-started (slice)
          person_heatmap_head   Conv1x1(128,1)                        <-- warm-started (slice)
          regression_head       Conv1x1(128,12)                       <-- warm-started (slice)
          [offset_head          Conv1x1(128,2)]                       <-- NEW, private

Object logits are returned at their NATIVE 192x108 resolution. They are never enlarged
to 768x432 before loss or peak selection.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for _path in (str(PACKAGE_ROOT), str(BASE_PKG), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.model import (  # noqa: E402
    MultiTaskFusionLRASPP,
    SplitClassHeatmapHead,
    build_fusion_lraspp,
)
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402

MODEL_SIZE = (768, 432)          # (width, height) of the model input
NATIVE_STRIDE = 4                # object output stride after the learned 2x upsample
NATIVE_GRID = (MODEL_SIZE[0] // NATIVE_STRIDE, MODEL_SIZE[1] // NATIVE_STRIDE)  # (192, 108)
HEATMAP_CHANNELS = 2             # vehicle, person
REG_CHANNELS = object_reg_channels(True)  # 12, incl. the 2 bbox2d channels
OFFSET_CHANNELS = 2              # private fractional centre offset (x, y)
OUTPUT_CHANNELS = HEATMAP_CHANNELS + REG_CHANNELS + OFFSET_CHANNELS  # 16

CH_VEHICLE = 0
CH_PERSON = 1
SL_REG = slice(HEATMAP_CHANNELS, HEATMAP_CHANNELS + REG_CHANNELS)
SL_OFFSET = slice(HEATMAP_CHANNELS + REG_CHANNELS, OUTPUT_CHANNELS)


def bilinear_kernel(channels: int, kernel_size: int = 4) -> torch.Tensor:
    """Per-channel bilinear upsampling kernel for ConvTranspose2d(stride=2).

    Initialising the new upsampler this way makes the block a (near) exact bilinear
    2x upsample at step zero, so the warm-started 1x1 output heads immediately see the
    same trunk features they were trained on - just carried onto the stride-4 grid.
    Training then moves the block away from interpolation toward sharp native peaks.
    """
    factor = (kernel_size + 1) // 2
    center = factor - 1.0 if kernel_size % 2 == 1 else factor - 0.5
    coords = torch.arange(kernel_size, dtype=torch.float32)
    ramp = 1.0 - torch.abs(coords - center) / factor
    kernel2d = torch.outer(ramp, ramp)
    weight = torch.zeros(channels, channels, kernel_size, kernel_size, dtype=torch.float32)
    for index in range(channels):
        weight[index, index] = kernel2d
    return weight


class NativeGridObjectHead(SplitClassHeatmapHead):
    """Shared trunk + learned 2x upsample + class heatmaps, shared regression, offsets."""

    def __init__(self, in_ch: int, hidden: int, reg_ch: int, depth: int) -> None:
        super().__init__(in_ch, hidden, reg_ch, depth)
        self.upsampler = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(int(hidden), int(hidden), kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(int(hidden)),
            torch.nn.ReLU(inplace=True),
        )
        self.offset_head = torch.nn.Conv2d(int(hidden), OFFSET_CHANNELS, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shared = self.upsampler(self.shared_trunk(inputs))
        return torch.cat(
            [
                self.vehicle_heatmap_head(shared),
                self.person_heatmap_head(shared),
                self.regression_head(shared),
                self.offset_head(shared),
            ],
            dim=1,
        )


class NativeGridFusionLRASPP(MultiTaskFusionLRASPP):
    """The unchanged fusion LR-ASPP with a native stride-4 tail-side object decoder."""

    def __init__(self, base_model: torch.nn.Module, *, hidden_channels: int = 128, head_depth: int = 3) -> None:
        super().__init__(
            base_model,
            object_channels=HEATMAP_CHANNELS + REG_CHANNELS,
            hidden_channels=int(hidden_channels),
            fuse_low_into_object_head=True,
            head_arch="split_class_heatmaps",
            head_depth=int(head_depth),
            predict_bbox2d=True,
        )
        head_in = int(self.object_head.shared_trunk[0].in_channels)
        self.object_head = NativeGridObjectHead(head_in, int(hidden_channels), REG_CHANNELS, int(head_depth))
        self._init_object_head()          # inherited kaiming init + focal/bbox bias priors
        self._init_native_object_head()   # new-module priors
        self.object_output_channels = OUTPUT_CHANNELS
        self.offset_channels = OFFSET_CHANNELS
        self.native_stride = NATIVE_STRIDE

    def _init_native_object_head(self) -> None:
        deconv, norm = self.object_head.upsampler[0], self.object_head.upsampler[1]
        with torch.no_grad():
            deconv.weight.copy_(bilinear_kernel(int(deconv.in_channels)))
            norm.weight.fill_(1.0)
            norm.bias.zero_()
            norm.running_mean.zero_()
            norm.running_var.fill_(1.0)
            # Offset prior: predict the cell centre (0.5, 0.5) before any evidence.
            self.object_head.offset_head.weight.zero_()
            self.object_head.offset_head.bias.fill_(0.5)

    def object_logits(self, features: object) -> torch.Tensor:
        """Native-resolution object logits. Deliberately NOT enlarged to 768x432."""
        return self.object_head(self._object_input(features))

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if float(feature_drop_fraction) > 0.0:
            features = self._objectness_drop(features, float(feature_drop_fraction))
        if getattr(self, "feature_ae", None) is not None:
            features = self._apply_feature_ae(features)
        seg = self.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        return {"out": seg, "object": self.object_logits(features)}


def build_native_grid_model(
    *,
    num_classes: int = 3,
    radar_channels: int = 4,
    hidden_channels: int = 128,
    head_depth: int = 3,
    device: Optional[torch.device] = None,
) -> NativeGridFusionLRASPP:
    base = build_fusion_lraspp(
        num_classes=int(num_classes), radar_channels=int(radar_channels),
        pretrained=False, init_checkpoint="", device=device,
    )
    model = NativeGridFusionLRASPP(base, hidden_channels=int(hidden_channels), head_depth=int(head_depth))
    return model.to(device) if device is not None else model


# --------------------------------------------------------------------------------------
# Warm start
# --------------------------------------------------------------------------------------

# The epoch-20 baseline uses head_arch="shared": one Sequential whose final 1x1 conv
# emits all 14 channels. Its output slices map onto the new separate branches.
_BASELINE_OUTPUT_CONV = "object_head.9"
_OUTPUT_SLICES = (
    ("object_head.vehicle_heatmap_head", slice(0, 1)),
    ("object_head.person_heatmap_head", slice(1, 2)),
    ("object_head.regression_head", slice(2, 14)),
)


def load_warm_start(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> Dict[str, Any]:
    """Warm-start every compatible backbone / classifier / object tensor from epoch 20.

    Returns the full tensor-level mapping: loaded, transformed, new and incompatible.
    """
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    source: Dict[str, torch.Tensor] = payload["model"]
    current = model.state_dict()

    staged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    loaded: list[str] = []
    transformed: list[Dict[str, Any]] = []
    incompatible: list[Dict[str, Any]] = []

    for key, tensor in source.items():
        name = key[7:] if key.startswith("module.") else key
        if name.startswith(f"{_BASELINE_OUTPUT_CONV}."):
            leaf = name.rsplit(".", 1)[-1]           # weight | bias
            for target_prefix, channels in _OUTPUT_SLICES:
                target = f"{target_prefix}.{leaf}"
                value = tensor[channels]
                if target in current and tuple(current[target].shape) == tuple(value.shape):
                    staged[target] = value.clone()
                    transformed.append({
                        "source": name, "target": target,
                        "operation": f"output_channel_slice[{channels.start}:{channels.stop}]",
                        "shape": list(value.shape),
                    })
                else:
                    incompatible.append({"source": name, "target": target,
                                         "reason": "shape_or_name_mismatch",
                                         "source_shape": list(value.shape)})
            continue
        # The baseline trunk is object_head.<index>.*; the new head nests it under
        # shared_trunk with identical indices, shapes and semantics.
        candidates = [name]
        if name.startswith("object_head."):
            candidates.insert(0, "object_head.shared_trunk." + name[len("object_head."):])
        target = next((item for item in candidates
                       if item in current and tuple(current[item].shape) == tuple(tensor.shape)), None)
        if target is None:
            incompatible.append({"source": name, "target": None, "reason": "no_compatible_target",
                                 "source_shape": list(tensor.shape)})
            continue
        staged[target] = tensor.clone()
        if target == name:
            loaded.append(target)
        else:
            transformed.append({"source": name, "target": target, "operation": "rename_into_shared_trunk",
                                "shape": list(tensor.shape)})

    new_tensors = sorted(set(current) - set(staged))
    missing = model.load_state_dict(staged, strict=False)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "source_tensor_count": len(source),
        "target_tensor_count": len(current),
        "loaded_tensors": sorted(loaded),
        "loaded_count": len(loaded),
        "transformed_tensors": transformed,
        "transformed_count": len(transformed),
        "new_tensors": new_tensors,
        "new_count": len(new_tensors),
        "incompatible_tensors": incompatible,
        "incompatible_count": len(incompatible),
        "unexpected_keys": list(missing.unexpected_keys),
        "reported_missing_keys": sorted(missing.missing_keys),
        # PyTorch omits BatchNorm num_batches_tracked from missing_keys, so the
        # assertion is containment: nothing uninitialised beyond the known-new set.
        "missing_keys_are_new_only": set(missing.missing_keys) <= set(new_tensors),
    }


# --------------------------------------------------------------------------------------
# Split boundary
# --------------------------------------------------------------------------------------

def encode_front(model: NativeGridFusionLRASPP, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    """UE front: exactly the existing low/high backbone bundle. Nothing added."""
    features = model.backbone(x)
    return OrderedDict((str(name), value) for name, value in features.items())


def decode_tail(
    model: NativeGridFusionLRASPP, features: "OrderedDict[str, torch.Tensor]"
) -> Dict[str, torch.Tensor]:
    """Edge tail: reads ONLY the transported bundle. No raw RGB, no raw radar."""
    seg = model.classifier(features)
    if isinstance(seg, dict):
        seg = seg["out"]
    return {"out": seg, "object": model.object_logits(features)}


def split_boundary_report(model: NativeGridFusionLRASPP, x: torch.Tensor) -> Dict[str, Any]:
    """Confirm the tail is a pure function of the transported bundle."""
    model.eval()
    with torch.inference_mode():
        monolithic = model(x)
        bundle = encode_front(model, x)
        tail = decode_tail(model, bundle)
    deltas = {
        key: float((monolithic[key].float() - tail[key].float()).abs().max().item())
        for key in ("out", "object")
    }
    return {
        "transported_feature_names": sorted(bundle),
        "transported_feature_shapes": {name: list(value.shape) for name, value in bundle.items()},
        "transported_feature_elements": {name: int(value.numel()) for name, value in bundle.items()},
        "tail_reads_only_transported_bundle": sorted(bundle) == ["high", "low"],
        "monolithic_vs_split_max_abs_delta": deltas,
        "outputs_match": all(value == 0.0 for value in deltas.values()),
        "object_grid": list(tail["object"].shape[-2:]),
        "object_grid_is_native": tuple(tail["object"].shape[-2:]) == (NATIVE_GRID[1], NATIVE_GRID[0]),
        "object_channels": int(tail["object"].shape[1]),
    }


def parameter_report(model: torch.nn.Module) -> Dict[str, Any]:
    groups = {
        "backbone": model.backbone,
        "classifier": model.classifier,
        "object_head_trunk": model.object_head.shared_trunk,
        "object_head_upsampler": model.object_head.upsampler,
        "object_head_vehicle_heatmap": model.object_head.vehicle_heatmap_head,
        "object_head_person_heatmap": model.object_head.person_heatmap_head,
        "object_head_regression": model.object_head.regression_head,
        "object_head_offset": model.object_head.offset_head,
    }
    report: Dict[str, Any] = {}
    for name, module in groups.items():
        total = sum(int(p.numel()) for p in module.parameters())
        trainable = sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
        report[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    report["model_total"] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    return report
