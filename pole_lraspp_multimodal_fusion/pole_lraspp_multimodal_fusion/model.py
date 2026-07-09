from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .object_targets import OBJECT_OUTPUT_CHANNELS, OBJECT_REG_CHANNELS, object_reg_channels

OBJECT_HEAD_CHANNELS = OBJECT_OUTPUT_CHANNELS


class MultiTaskFusionLRASPP(torch.nn.Module):
    """LR-ASPP segmentation backbone with learned object localization heads."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        object_channels: int = OBJECT_HEAD_CHANNELS,
        hidden_channels: int = 128,
        fuse_low_into_object_head: bool = False,
        head_arch: str = "shared",
        use_coordconv: bool = False,
        head_depth: int = 2,
        predict_bbox2d: bool = False,
        use_groundplane_prior: bool = False,
        cam_fy: float = 369.5,
        cam_cy: float = 360.0,
        cam_height_m: float = 1.57,
        cam_pitch_deg: float = -4.16,
        cam_image_height: int = 720,
        groundplane_max_range_m: float = 80.0,
    ) -> None:
        super().__init__()
        # Camera geometry for the optional ground-plane depth prior (flat-ground IPM).
        self.use_groundplane_prior = bool(use_groundplane_prior)
        self.cam_fy = float(cam_fy)
        self.cam_cy = float(cam_cy)
        self.cam_height_m = float(cam_height_m)
        self.cam_pitch_deg = float(cam_pitch_deg)
        self.cam_image_height = int(cam_image_height)
        self.groundplane_max_range_m = float(groundplane_max_range_m)
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

        self.object_channels = int(object_channels)
        self.head_arch = str(head_arch).lower()
        self.use_coordconv = bool(use_coordconv)
        self.head_depth = max(1, int(head_depth))
        self.predict_bbox2d = bool(predict_bbox2d)
        self.reg_channels = object_reg_channels(self.predict_bbox2d)
        self.heatmap_channels = max(1, int(self.object_channels) - self.reg_channels)
        # CoordConv appends normalized (x, y) image-position channels so the head
        # can exploit image-row -> ground-plane depth priors (key for pedestrians).
        head_in = int(object_in_channels) + (2 if self.use_coordconv else 0)
        # Ground-plane prior appends one calibrated per-row depth channel.
        head_in += (1 if self.use_groundplane_prior else 0)

        if self.head_arch == "decoupled":
            # Separate detection (heatmap) and metric-regression branches so the
            # two objectives do not compete inside a single shallow trunk.
            self.heatmap_head = self._make_head(head_in, int(hidden_channels), self.heatmap_channels, self.head_depth)
            self.reg_head = self._make_head(head_in, int(hidden_channels), self.reg_channels, self.head_depth)
            self.object_head = None
        else:
            self.object_head = self._make_head(head_in, int(hidden_channels), self.object_channels, self.head_depth)
            self.heatmap_head = None
            self.reg_head = None
        self._init_object_head()

    @staticmethod
    def _make_head(in_ch: int, hidden: int, out_ch: int, depth: int) -> torch.nn.Sequential:
        layers = []
        c = int(in_ch)
        for _ in range(max(1, int(depth))):
            layers.append(torch.nn.Conv2d(c, int(hidden), kernel_size=3, padding=1, bias=False))
            layers.append(torch.nn.BatchNorm2d(int(hidden)))
            layers.append(torch.nn.ReLU(inplace=True))
            c = int(hidden)
        layers.append(torch.nn.Conv2d(int(hidden), int(out_ch), kernel_size=1))
        return torch.nn.Sequential(*layers)

    def _init_object_head(self) -> None:
        heads = [h for h in (self.object_head, self.heatmap_head, self.reg_head) if h is not None]
        for head in heads:
            for module in head.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
        # Bias the heatmap logits toward "no object" (focal-loss prior, p~0.01).
        if self.head_arch == "decoupled":
            final = self.heatmap_head[-1]
            heatmap_channels = min(self.heatmap_channels, int(final.bias.numel())) if final.bias is not None else 0
        else:
            final = self.object_head[-1]
            heatmap_channels = min(self.heatmap_channels, int(final.bias.numel())) if final.bias is not None else 0
        if isinstance(final, torch.nn.Conv2d) and final.bias is not None:
            with torch.no_grad():
                final.bias.zero_()
                final.bias[:heatmap_channels] = -4.6
        # Bias the 2D-box (last two regression) logits so softplus(bias) ~ 0.05, a
        # typical normalized object size, instead of softplus(0) ~ 0.69 (huge boxes).
        if self.predict_bbox2d:
            reg_final = self.reg_head[-1] if self.head_arch == "decoupled" else self.object_head[-1]
            if isinstance(reg_final, torch.nn.Conv2d) and reg_final.bias is not None and reg_final.bias.numel() >= 2:
                with torch.no_grad():
                    reg_final.bias[-2:] = -3.0

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
            base = high
        else:
            low = self._low_feature(features)
            if tuple(high.shape[-2:]) != tuple(low.shape[-2:]):
                high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
            base = torch.cat([low, high], dim=1)
        if self.use_coordconv:
            base = torch.cat([base, self._coord_channels(base)], dim=1)
        if self.use_groundplane_prior:
            base = torch.cat([base, self._groundplane_channel(base)], dim=1)
        return base

    def _groundplane_channel(self, ref: torch.Tensor) -> torch.Tensor:
        """Flat-ground IPM depth prior: per feature-map row, the ground distance a
        ground-contact object at that row would be at, normalized to [0, 1]. Above
        the horizon -> clamped to 1 (far). Constant across columns (roll ~ 0)."""
        n, _, h, w = ref.shape
        rows = torch.arange(h, device=ref.device, dtype=torch.float32)
        v = (rows + 0.5) / float(h) * float(self.cam_image_height)  # feature row -> full-image row
        a = torch.atan((v - self.cam_cy) / max(1e-3, self.cam_fy))  # angle below optical axis
        pitch_down = math.radians(-self.cam_pitch_deg)              # CARLA neg pitch = looking down
        total_down = pitch_down + a
        depth = torch.where(
            total_down > 1e-3,
            self.cam_height_m / torch.tan(total_down.clamp(min=1e-3)),
            torch.full_like(total_down, self.groundplane_max_range_m),
        )
        depth = depth.clamp(min=0.0, max=self.groundplane_max_range_m) / self.groundplane_max_range_m
        return depth.view(1, 1, h, 1).expand(n, 1, h, w).to(ref.dtype)

    @staticmethod
    def _coord_channels(ref: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.shape
        ys = torch.linspace(-1.0, 1.0, h, device=ref.device, dtype=ref.dtype).view(1, 1, h, 1).expand(n, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=ref.device, dtype=ref.dtype).view(1, 1, 1, w).expand(n, 1, h, w)
        return torch.cat([xs, ys], dim=1)

    def _objectness_drop(self, features: object, q: float) -> object:
        """Opt-in drop-aware training: zero the lowest-objectness fraction q of backbone-feature cells,
        using the model's OWN objectness (detached) — consistent with the inference-time ROI gate. Makes the
        model robust to ROI dropping across the range (train with q~U(0,q_max)). Default off (q=0 -> no-op)."""
        obj_in = self._object_input(features)
        with torch.no_grad():
            if self.head_arch == "decoupled":
                heat = self.heatmap_head(obj_in)
            else:
                heat = self.object_head(obj_in)[:, : self.heatmap_channels]
            objness = torch.sigmoid(heat).amax(dim=1, keepdim=True)  # (B,1,h,w)

        def gate(feat: torch.Tensor) -> torch.Tensor:
            pooled = F.adaptive_max_pool2d(objness, feat.shape[-2:])          # objectness at this feat res
            b = pooled.shape[0]
            flat = pooled.reshape(b, -1).float()                             # per-sample, fp32 (AMP -> half)
            n = flat.shape[1]
            k = int(round(float(q) * n))                                      # cells to drop (by rank)
            if k <= 0:
                return feat
            # Drop the k LOWEST-objectness cells by RANK, not a value threshold: the focal-biased
            # objectness is floor-dominated (most cells ~sigmoid(-4.6)), so a quantile threshold ties on
            # the floor and keeps everything. Rank-drop guarantees exactly fraction q is removed.
            drop_idx = flat.argsort(dim=1)[:, :k]
            keep = torch.ones_like(flat).scatter_(1, drop_idx, 0.0)
            keep = keep.reshape(b, 1, feat.shape[-2], feat.shape[-1]).to(feat.dtype)
            return feat * keep

        if isinstance(features, dict):
            return type(features)((k, gate(v)) for k, v in features.items())
        return gate(features)

    def forward(self, x: torch.Tensor, feature_drop_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if float(feature_drop_fraction) > 0.0:
            features = self._objectness_drop(features, float(feature_drop_fraction))
        seg = self.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        obj_in = self._object_input(features)
        if self.head_arch == "decoupled":
            # Channel order stays [heatmap..., regression...] to match the loss/decoder.
            object_logits = torch.cat([self.heatmap_head(obj_in), self.reg_head(obj_in)], dim=1)
        else:
            object_logits = self.object_head(obj_in)
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
    head_arch: str = "shared",
    use_coordconv: bool = False,
    head_depth: int = 2,
    predict_bbox2d: bool = False,
    use_groundplane_prior: bool = False,
    groundplane_params: Optional[Dict[str, float]] = None,
    device: Optional[torch.device] = None,
) -> MultiTaskFusionLRASPP:
    base = build_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=pretrained,
        init_checkpoint=init_checkpoint,
        device=device,
    )
    gp = dict(groundplane_params or {})
    return MultiTaskFusionLRASPP(
        base,
        object_channels=int(object_channels),
        hidden_channels=int(object_hidden_channels),
        fuse_low_into_object_head=bool(fuse_low_into_object_head),
        head_arch=str(head_arch),
        use_coordconv=bool(use_coordconv),
        head_depth=int(head_depth),
        predict_bbox2d=bool(predict_bbox2d),
        use_groundplane_prior=bool(use_groundplane_prior),
        cam_fy=float(gp.get("cam_fy", 369.5)),
        cam_cy=float(gp.get("cam_cy", 360.0)),
        cam_height_m=float(gp.get("cam_height_m", 1.57)),
        cam_pitch_deg=float(gp.get("cam_pitch_deg", -4.16)),
        cam_image_height=int(gp.get("cam_image_height", 720)),
        groundplane_max_range_m=float(gp.get("groundplane_max_range_m", 80.0)),
    )
