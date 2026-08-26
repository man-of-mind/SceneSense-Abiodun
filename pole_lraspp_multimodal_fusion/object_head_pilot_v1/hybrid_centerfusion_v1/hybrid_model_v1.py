#!/usr/bin/env python3
"""LR-ASPP / CenterFusion hybrid detector (``centerfusion_hybrid_v1``).

This is a *refinement* of the measured Route B noAE model, not a replacement
architecture. It keeps the MobileNetV3 LR-ASPP backbone and its segmentation
classifier, keeps the 14-channel object output order, keeps the production
decoder contract, and adds three things:

1. a dedicated four-channel shallow **radar encoder** (radar stops being four
   extra RGB-like colour planes fed to the image stem);
2. **feature-level RGB/radar fusion** in a lightweight FPN path that reaches
   1/4 input resolution, so the CenterNet-style detection branch no longer
   localizes on a 1/8 grid alone;
3. a **radar-conditioned refinement** of the centre heatmaps and of the
   XYZ / dimension / yaw regression, the one CenterFusion idea that the
   retained calibrated radar raster can support without new data plumbing.

Exact-parity warm start
-----------------------
The baseline stem is a single ``Conv2d(7 -> 16)`` over ``cat[rgb, radar]``.
Convolution is linear in the input channels, so

    conv7(cat[rgb, radar]) == conv3_rgb(rgb) + conv4_radar(radar)

The hybrid therefore restores a genuine 3-channel image stem and hands the four
radar planes to the radar encoder's own first convolution, warm-started from the
baseline stem's radar slice and summed back into the image stem. The measured
model's radar information path is preserved *exactly* while radar gains its own
encoder. Everything the segmentation branch consumes is bit-equivalent to the
baseline (up to float summation order), which is what makes the Phase C
warm-start parity check a real gate rather than a formality.

The new detection capacity is added as a residual on top of the warm-started
1/8 coarse head, with the refinement's output convolutions initialised at
std=1e-4. At initialisation the object output is the baseline's to ~1e-3 in
logit space, so training starts from the measured operating point instead of
from a cold head.

Split boundary (future UE/edge split; NOT wired to the live runtime here)
------------------------------------------------------------------------
    encode_front(rgb, radar) -> fused feature bundle
    decode_tail(fused, out_hw) -> {"out": segmentation, "object": 14-channel}

``q`` (objectness ROI drop) and a feature AE remain attachable to the fused
bundle exactly as on the baseline: both act on ``fused["high"]`` / the bundle,
and both are structural no-ops at q=0 with no AE attached.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.model import (
    OBJECT_HEAD_CHANNELS,
    SplitClassHeatmapHead,
    build_lraspp,
)
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels

HEAD_ARCH_NAME = "centerfusion_hybrid_v1"

# Backbone taps. '3' is the earliest practical 1/4-resolution MobileNetV3 stage
# (24 channels); '4' -> low (40 ch, 1/8) and '16' -> high (960 ch, 1/16) are the
# two the stock LR-ASPP already exposes.
QUARTER_RES_BACKBONE_KEY = "3"

RADAR_STEM_CHANNELS = 16
RADAR_WIDTHS = (32, 64, 96)   # at 1/4, 1/8, 1/16
FPN_CHANNELS = 96
REFINE_CHANNELS = 64
REFINE_INIT_STD = 1e-4


def _gn(channels: int) -> torch.nn.GroupNorm:
    """GroupNorm, not BatchNorm, for every new module.

    The production trial sets ``freeze_bn: true``, which puts *every* BatchNorm
    in the model into eval mode with frozen affine params. For the warm-started
    backbone that is intended. For a freshly initialised branch it would mean
    normalising by untrained running stats (mean 0, var 1), i.e. no
    normalisation at all. GroupNorm is unaffected by that switch.
    """
    return torch.nn.GroupNorm(min(8, int(channels)), int(channels))


class ShallowRadarEncoder(torch.nn.Module):
    """Four-channel radar encoder: stem at 1/2 then 1/4, 1/8, 1/16 stages.

    ``stem`` is warm-started from the baseline 7-channel image stem's radar
    slice, so ``stem(radar)`` reproduces the radar half of the measured model's
    first convolution exactly.
    """

    def __init__(self, in_channels: int = 4, stem_channels: int = RADAR_STEM_CHANNELS,
                 widths: Tuple[int, ...] = RADAR_WIDTHS) -> None:
        super().__init__()
        self.stem = torch.nn.Conv2d(int(in_channels), int(stem_channels),
                                    kernel_size=3, stride=2, padding=1, bias=False)
        channels = int(stem_channels)
        blocks: List[torch.nn.Sequential] = []
        for width in widths:
            blocks.append(torch.nn.Sequential(
                torch.nn.Conv2d(channels, int(width), kernel_size=3, stride=2, padding=1, bias=False),
                _gn(int(width)),
                torch.nn.ReLU(inplace=True),
            ))
            channels = int(width)
        self.block4, self.block8, self.block16 = blocks

    def forward(self, radar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(radar)          # 1/2
        r4 = self.block4(stem)           # 1/4
        r8 = self.block8(r4)             # 1/8
        r16 = self.block16(r8)           # 1/16
        return stem, r4, r8, r16


class HybridCenterFusionLRASPP(torch.nn.Module):
    """LR-ASPP segmentation + high-resolution radar-conditioned CenterNet head."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        radar_channels: int = 4,
        object_channels: int = OBJECT_HEAD_CHANNELS,
        hidden_channels: int = 128,
        head_depth: int = 3,
        predict_bbox2d: bool = True,
        fpn_channels: int = FPN_CHANNELS,
        refine_channels: int = REFINE_CHANNELS,
        refine_init_std: float = REFINE_INIT_STD,
    ) -> None:
        super().__init__()
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        self.head_arch = HEAD_ARCH_NAME
        self.fuse_low_into_object_head = True   # the coarse head always uses low+high
        self.use_coordconv = False
        self.use_groundplane_prior = False
        self.feature_ae = None                  # attachable later; unused here

        self.radar_channels = int(radar_channels)
        self.object_channels = int(object_channels)
        self.predict_bbox2d = bool(predict_bbox2d)
        self.reg_channels = object_reg_channels(self.predict_bbox2d)
        self.heatmap_channels = max(1, self.object_channels - self.reg_channels)
        if self.heatmap_channels != 2:
            raise ValueError(
                "centerfusion_hybrid_v1 expects exactly two centre heatmaps "
                f"(vehicle, person); got {self.heatmap_channels}."
            )
        self.head_depth = max(1, int(head_depth))

        high_channels = int(base_model.classifier.cbr[0].in_channels)
        low_channels = int(base_model.classifier.low_classifier.in_channels)
        quarter_channels = int(self.backbone[QUARTER_RES_BACKBONE_KEY].out_channels)
        self.high_channels, self.low_channels, self.quarter_channels = (
            high_channels, low_channels, quarter_channels)

        self.radar_encoder = ShallowRadarEncoder(self.radar_channels)
        r4, r8, r16 = RADAR_WIDTHS

        # Coarse 1/8 detection stage: byte-for-byte the baseline head's shape and
        # input (cat[low, high-upsampled]), so it warm-starts exactly.
        self.object_head = SplitClassHeatmapHead(
            low_channels + high_channels, int(hidden_channels), self.reg_channels, self.head_depth
        )

        # Lightweight FPN carrying fused RGB+radar features down to 1/4.
        fpn = int(fpn_channels)
        self.lat16 = torch.nn.Conv2d(high_channels + r16, fpn, kernel_size=1)
        self.norm16 = _gn(fpn)
        self.lat8 = torch.nn.Conv2d(low_channels + r8, fpn, kernel_size=1)
        self.norm8 = _gn(fpn)
        self.reduce8 = torch.nn.Conv2d(fpn, int(refine_channels), kernel_size=1)
        self.lat4 = torch.nn.Conv2d(quarter_channels + r4, int(refine_channels), kernel_size=1)
        self.norm4 = _gn(int(refine_channels))
        self.smooth4 = torch.nn.Sequential(
            torch.nn.Conv2d(int(refine_channels), int(refine_channels), kernel_size=3, padding=1, bias=False),
            _gn(int(refine_channels)),
            torch.nn.ReLU(inplace=True),
        )

        # Radar-conditioned refinement at 1/4, conditioned on the coarse stage's
        # own 14-channel prediction (the CenterFusion second-stage idea, applied
        # densely because the retained radar is a calibrated raster, not an
        # associable point set).
        self.refine_trunk = torch.nn.Sequential(
            torch.nn.Conv2d(int(refine_channels) + self.object_channels, int(refine_channels),
                            kernel_size=3, padding=1, bias=False),
            _gn(int(refine_channels)),
            torch.nn.ReLU(inplace=True),
        )
        self.refine_vehicle_heatmap_head = torch.nn.Conv2d(int(refine_channels), 1, kernel_size=1)
        self.refine_person_heatmap_head = torch.nn.Conv2d(int(refine_channels), 1, kernel_size=1)
        self.refine_regression_head = torch.nn.Conv2d(int(refine_channels), self.reg_channels, kernel_size=1)

        self._init_new_modules(float(refine_init_std))

    # ------------------------------------------------------------------ init
    def _init_new_modules(self, refine_init_std: float) -> None:
        new_modules = [
            self.radar_encoder, self.lat16, self.norm16, self.lat8, self.norm8,
            self.reduce8, self.lat4, self.norm4, self.smooth4, self.refine_trunk,
        ]
        for module in new_modules:
            for sub in module.modules():
                if isinstance(sub, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(sub.weight, mode="fan_out", nonlinearity="relu")
                    if sub.bias is not None:
                        torch.nn.init.zeros_(sub.bias)
        # Near-zero (not exactly zero) residual output: the object logits start at
        # the warm-started coarse stage's values, yet every parameter in the
        # refinement branch still receives a finite non-zero gradient on step 1.
        for head in (self.refine_vehicle_heatmap_head, self.refine_person_heatmap_head,
                     self.refine_regression_head):
            torch.nn.init.normal_(head.weight, mean=0.0, std=float(refine_init_std))
            if head.bias is not None:
                torch.nn.init.zeros_(head.bias)

    # ------------------------------------------------------- split boundary
    def encode_front(self, rgb: torch.Tensor, radar: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """UE-side front half: RGB + radar -> the fused intermediate feature bundle.

        This is the exact tensor set a future split would transmit; ``q`` and a
        feature AE attach here (see ``_objectness_drop`` / ``_apply_feature_ae``).
        """
        radar_stem, r4, r8, r16 = self.radar_encoder(radar)
        fused = self._backbone_forward(rgb, radar_stem)
        fused["radar4"], fused["radar8"], fused["radar16"] = r4, r8, r16
        return fused

    def _backbone_forward(self, rgb: torch.Tensor, radar_stem: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """MobileNetV3 forward with (a) radar injected into the image stem and
        (b) an extra 1/4-resolution tap, keeping the stock parameter names."""
        return_layers = dict(self.backbone.return_layers)
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        items = iter(self.backbone.items())
        name, block = next(items)
        # Split block '0' = Conv2dNormActivation(conv, bn, act): sum the two stems
        # *before* normalisation, reproducing the baseline's single 7-channel conv.
        hidden = block[0](rgb) + radar_stem
        for sub in list(block)[1:]:
            hidden = sub(hidden)
        if name in return_layers:
            out[return_layers[name]] = hidden
        if name == QUARTER_RES_BACKBONE_KEY:
            out["quarter"] = hidden
        for name, block in items:
            hidden = block(hidden)
            if name in return_layers:
                out[return_layers[name]] = hidden
            if name == QUARTER_RES_BACKBONE_KEY:
                out["quarter"] = hidden
        return out

    def decode_tail(self, fused: Dict[str, torch.Tensor],
                    out_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Edge-side back half: fused features -> segmentation + 14-channel object."""
        seg = self.classifier({"low": fused["low"], "high": fused["high"]})
        if isinstance(seg, dict):
            seg = seg["out"]

        low, high = fused["low"], fused["high"]
        quarter, r4, r8, r16 = fused["quarter"], fused["radar4"], fused["radar8"], fused["radar16"]

        # --- coarse 1/8 CenterNet stage (warm-started, baseline-identical) ---
        high_at_low = self._match(high, low)
        coarse = self.object_head(torch.cat([low, high_at_low], dim=1))

        # --- feature-level RGB/radar fusion, FPN down to 1/4 ---
        p16 = F.relu(self.norm16(self.lat16(torch.cat([high, self._match(r16, high)], dim=1))), inplace=True)
        p8 = F.relu(self.norm8(self.lat8(torch.cat([low, self._match(r8, low)], dim=1))
                               + self._match(p16, low)), inplace=True)
        p4 = F.relu(self.norm4(self.lat4(torch.cat([quarter, self._match(r4, quarter)], dim=1))
                               + self._match(self.reduce8(p8), quarter)), inplace=True)
        f4 = self.smooth4(p4)

        # --- radar-conditioned refinement of centres / XYZ / dims / yaw ---
        refine_in = torch.cat([f4, self._match(coarse, f4)], dim=1)
        shared = self.refine_trunk(refine_in)
        delta = torch.cat([
            self.refine_vehicle_heatmap_head(shared),
            self.refine_person_heatmap_head(shared),
            self.refine_regression_head(shared),
        ], dim=1)

        # Both stages are upsampled to the input grid independently and summed
        # there: the coarse term stays bit-identical to the baseline's decode
        # path, the 1/4 term adds the resolution the 1/8 grid cannot express.
        object_logits = (F.interpolate(coarse, size=out_hw, mode="bilinear", align_corners=False)
                         + F.interpolate(delta, size=out_hw, mode="bilinear", align_corners=False))
        return {"out": seg, "object": object_logits}

    @staticmethod
    def _match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape[-2:]) == tuple(reference.shape[-2:]):
            return source
        return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    # ------------------------------------------------ q / AE attach points
    def _apply_feature_ae(self, fused: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ae = self.feature_ae
        reconstructed = ae.decode(ae.encode(fused["high"]))
        return type(fused)((k, (reconstructed if k == "high" else v)) for k, v in fused.items())

    def _objectness_drop(self, fused: Dict[str, torch.Tensor], q: float) -> Dict[str, torch.Tensor]:
        """Rank-drop the lowest-objectness fraction q of every fused feature cell.

        Same contract as the baseline: objectness comes from the model's own
        (detached) coarse centre heatmaps, and the drop is by rank so exactly
        fraction q is removed regardless of the focal-biased score floor.
        """
        low, high = fused["low"], fused["high"]
        with torch.no_grad():
            coarse = self.object_head(torch.cat([low, self._match(high, low)], dim=1))
            objectness = torch.sigmoid(coarse[:, : self.heatmap_channels]).amax(dim=1, keepdim=True)

        def gate(feat: torch.Tensor) -> torch.Tensor:
            pooled = F.adaptive_max_pool2d(objectness, feat.shape[-2:])
            batch = pooled.shape[0]
            flat = pooled.reshape(batch, -1).float()
            k = int(round(float(q) * flat.shape[1]))
            if k <= 0:
                return feat
            drop_idx = flat.argsort(dim=1)[:, :k]
            keep = torch.ones_like(flat).scatter_(1, drop_idx, 0.0)
            keep = keep.reshape(batch, 1, feat.shape[-2], feat.shape[-1]).to(feat.dtype)
            return feat * keep

        return type(fused)((k, gate(v)) for k, v in fused.items())

    # ----------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        rgb = x[:, :3]
        radar = x[:, 3: 3 + self.radar_channels]
        fused = self.encode_front(rgb, radar)
        if float(feature_drop_fraction) > 0.0:
            fused = self._objectness_drop(fused, float(feature_drop_fraction))
        if getattr(self, "feature_ae", None) is not None:
            fused = self._apply_feature_ae(fused)
        return self.decode_tail(fused, (int(x.shape[-2]), int(x.shape[-1])))


# --------------------------------------------------------------------------
# Warm start
# --------------------------------------------------------------------------

def _source_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise ValueError("Checkpoint did not contain a state_dict.")


def warm_start_from_baseline(
    model: HybridCenterFusionLRASPP,
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Map every compatible baseline tensor onto the hybrid; record the rest.

    Never writes to ``checkpoint_path``. Returns a full mapping report:
    ``mapped`` (source -> target, with the transform used), ``new`` (hybrid
    tensors with no source) and ``incompatible`` (source tensors that could not
    be placed).
    """
    device = device or torch.device("cpu")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    source = _source_state_dict(torch.load(path, map_location=device, weights_only=False))
    source = {(k[7:] if str(k).startswith("module.") else str(k)): v for k, v in source.items()}
    current = model.state_dict()

    staged: Dict[str, torch.Tensor] = OrderedDict()
    mapped: List[Dict[str, str]] = []
    incompatible: List[Dict[str, str]] = []
    consumed: set[str] = set()

    def place(target: str, tensor: torch.Tensor, sources: str, transform: str) -> bool:
        if target not in current or tuple(current[target].shape) != tuple(tensor.shape):
            return False
        staged[target] = tensor.detach().clone().to(current[target].dtype)
        mapped.append({"source": sources, "target": target, "transform": transform,
                       "shape": str(tuple(tensor.shape))})
        return True

    # 1) The 7-channel image stem splits into a 3-channel image stem plus the
    #    radar encoder's own first convolution: conv7(cat) == conv3(rgb) + conv4(radar).
    stem_key = "backbone.0.0.weight"
    stem = source.get(stem_key)
    if stem is None or stem.ndim != 4 or int(stem.shape[1]) != 3 + model.radar_channels:
        raise ValueError(
            f"expected a {3 + model.radar_channels}-channel baseline stem at {stem_key}; "
            f"got {None if stem is None else tuple(stem.shape)}"
        )
    ok_rgb = place(stem_key, stem[:, :3], stem_key, "channel_slice[:, 0:3] (rgb half of the fused stem)")
    ok_radar = place("radar_encoder.stem.weight", stem[:, 3: 3 + model.radar_channels], stem_key,
                     f"channel_slice[:, 3:{3 + model.radar_channels}] (radar half of the fused stem)")
    if not (ok_rgb and ok_radar):
        raise ValueError("could not split the baseline stem into rgb / radar halves")
    consumed.add(stem_key)

    # 2) Same-name, same-shape backbone + segmentation-classifier tensors.
    for key, tensor in source.items():
        if key in consumed:
            continue
        if not (key.startswith("backbone.") or key.startswith("classifier.")):
            continue
        if place(key, tensor, key, "identity"):
            consumed.add(key)
        else:
            incompatible.append({"source": key, "shape": str(tuple(tensor.shape)),
                                 "reason": "no same-shape hybrid tensor"})
            consumed.add(key)

    # 3) The baseline's single shared object head -> the split-heatmap coarse
    #    stage. cat[W[0:1]x, W[1:2]x, W[2:]x] + split bias == Wx + b, so this is
    #    an exact re-parameterisation, not an approximation.
    head_prefix = "object_head."
    for key, tensor in source.items():
        if key in consumed or not key.startswith(head_prefix):
            continue
        suffix = key[len(head_prefix):]
        trunk_target = f"object_head.shared_trunk.{suffix}"
        if place(trunk_target, tensor, key, "identity (shared trunk)"):
            consumed.add(key)
            continue
        if tensor.ndim in (1, 4) and int(tensor.shape[0]) == model.object_channels:
            leaf = suffix.rsplit(".", 1)[-1]
            branches = (
                ("object_head.vehicle_heatmap_head." + leaf, tensor[0:1], "row_slice[0:1] (vehicle heatmap)"),
                ("object_head.person_heatmap_head." + leaf, tensor[1:2], "row_slice[1:2] (person heatmap)"),
                ("object_head.regression_head." + leaf, tensor[2:], "row_slice[2:] (shared regression)"),
            )
            if all(t in current and tuple(current[t].shape) == tuple(v.shape) for t, v, _ in branches):
                for target, value, transform in branches:
                    place(target, value, key, transform)
                consumed.add(key)
                continue
        incompatible.append({"source": key, "shape": str(tuple(tensor.shape)),
                             "reason": "no shape-compatible hybrid object-head tensor"})
        consumed.add(key)

    for key, tensor in source.items():
        if key not in consumed:
            incompatible.append({"source": key, "shape": str(tuple(tensor.shape)),
                                 "reason": "source tensor outside backbone/classifier/object_head"})

    missing, unexpected = model.load_state_dict(staged, strict=False)
    if unexpected:
        raise RuntimeError(f"warm start produced unexpected keys: {sorted(unexpected)[:8]}")

    # Bit-exactness guard: every mapped tensor must equal the value we staged.
    loaded = model.state_dict()
    for record in mapped:
        target = record["target"]
        if not torch.equal(loaded[target].detach().cpu(), staged[target].detach().cpu()):
            raise RuntimeError(f"warm-started tensor did not land bit-exactly: {target}")

    return {
        "source_checkpoint": str(path),
        "mapped": mapped,
        "mapped_target_tensors": len(mapped),
        "new": sorted(str(name) for name in missing),
        "new_tensors": len(missing),
        "incompatible": incompatible,
        "incompatible_source_tensors": len(incompatible),
        "source_tensors": len(source),
    }


# --------------------------------------------------------------------------
# Builder + production-entry-point patch
# --------------------------------------------------------------------------

def build_hybrid_centerfusion(
    *,
    num_classes: int,
    radar_channels: int,
    object_channels: int = OBJECT_HEAD_CHANNELS,
    object_hidden_channels: int = 128,
    head_depth: int = 3,
    predict_bbox2d: bool = True,
    init_checkpoint: str = "",
    device: Optional[torch.device] = None,
) -> HybridCenterFusionLRASPP:
    device = device or torch.device("cpu")
    base = build_lraspp(int(num_classes), False)   # 3-channel stem: NOT channel-adapted
    model = HybridCenterFusionLRASPP(
        base,
        radar_channels=int(radar_channels),
        object_channels=int(object_channels),
        hidden_channels=int(object_hidden_channels),
        head_depth=int(head_depth),
        predict_bbox2d=bool(predict_bbox2d),
    )
    model.warm_start_report = None
    if init_checkpoint:
        model.warm_start_report = warm_start_from_baseline(model, init_checkpoint, device=device)
    return model


def install() -> None:
    """Route ``head_arch == centerfusion_hybrid_v1`` to the hybrid builder.

    Production ``train_fusion`` / ``evaluate_fusion`` are not edited: both import
    ``build_multitask_fusion_lraspp`` by name, so replacing that name in their
    module namespaces (and in ``model``) is enough. Every other ``head_arch``
    falls through to the untouched production builder.
    """
    from pole_lraspp_multimodal_fusion import evaluate_fusion, model as model_module, train_fusion

    original = getattr(model_module, "_original_build_multitask_fusion_lraspp",
                       model_module.build_multitask_fusion_lraspp)

    def dispatch(*, head_arch: str = "shared", **kwargs):
        if str(head_arch) != HEAD_ARCH_NAME:
            return original(head_arch=head_arch, **kwargs)
        return build_hybrid_centerfusion(
            num_classes=int(kwargs["num_classes"]),
            radar_channels=int(kwargs["radar_channels"]),
            object_channels=int(kwargs.get("object_channels", OBJECT_HEAD_CHANNELS)),
            object_hidden_channels=int(kwargs.get("object_hidden_channels", 128)),
            head_depth=int(kwargs.get("head_depth", 3)),
            predict_bbox2d=bool(kwargs.get("predict_bbox2d", True)),
            init_checkpoint=str(kwargs.get("init_checkpoint", "") or ""),
            device=kwargs.get("device"),
        )

    model_module._original_build_multitask_fusion_lraspp = original
    for module in (model_module, train_fusion, evaluate_fusion):
        module.build_multitask_fusion_lraspp = dispatch
