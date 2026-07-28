"""Shared model/input builders for the split-inference motivation study (E1-E5).

Builds the exact no-AE full model that evaluate_fusion.py builds, and pulls a REAL
fused input tensor (3 RGB + 4 radar channels) from the test split manifest.

FRONT (car)  = model.backbone
BACK  (edge) = model.classifier (seg) + model.object_head (detection), incl. the
               low/high concat done by model._object_input and the final interpolate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
PKG_ROOT = AB / "pole_lraspp_multimodal_fusion"
CONFIG = PKG_ROOT / "configs" / "fusion_full_run.yaml"
EXP_DIR = AB / "experiments" / "ae_integrated_20260710" / "noae_baseline"
CKPT = EXP_DIR / "checkpoints" / "mprime_joint_noae" / "best.pt"

for p in (str(PKG_ROOT), str(AB), str(AB / "rl_agent" / "feature_ae")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_full_model(device: torch.device):
    """Mirror evaluate_fusion.evaluate_checkpoint's model build, no-AE path."""
    from pole_lraspp_multimodal_fusion.common import load_config
    from pole_lraspp_multimodal_fusion.model import OBJECT_HEAD_CHANNELS, build_multitask_fusion_lraspp

    config = load_config(str(CONFIG))
    train_cfg = config["training"]
    fusion_cfg = config.get("fusion", {})
    object_cfg = config.get("object_heads", {})

    ckpt = torch.load(CKPT, map_location=device)
    assert isinstance(ckpt, dict), "expected dict checkpoint"
    # No-AE baseline must NOT carry an integrated AE.
    ae_bn = int((ckpt.get("trial") or {}).get("ae_bottleneck", 0))
    assert ae_bn == 0, f"expected no-AE checkpoint, got ae_bottleneck={ae_bn}"

    num_classes = int(train_cfg.get("num_classes", 3))
    input_size = tuple(int(v) for v in ckpt.get("input_size", train_cfg.get("input_size", [768, 432])))
    radar_channels = int(ckpt.get("radar_channels") or fusion_cfg.get("radar_channels", 4))
    object_channels = int(ckpt.get("object_channels") or object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS))
    fuse_low = bool(ckpt.get("fuse_low_into_object_head")) or bool(object_cfg.get("fuse_low_feature", False))

    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        object_channels=object_channels,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=fuse_low,
        head_arch=str(ckpt.get("object_head_arch") or object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(ckpt.get("object_use_coordconv")) or bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(ckpt.get("object_head_depth") or object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(ckpt.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(ckpt.get("object_use_groundplane_prior"))
        or bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(ckpt.get("object_groundplane_params") or object_cfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model, input_size, config


def get_real_input(device: torch.device, input_size: Tuple[int, int], index: int = 0):
    """Real fused 1x7xHxW tensor from the test split (RGB + 4-channel radar raster)."""
    from pole_lraspp_multimodal_fusion.common import read_manifest
    from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor

    dataset_dir = EXP_DIR / "dataset"
    rows = [r for r in read_manifest(dataset_dir / "manifest.csv") if r.get("split") == "test"]
    if not rows:
        raise RuntimeError("no test rows in manifest")
    row = rows[index % len(rows)]
    fused, orig_hw, _ = load_fused_tensor(row, dataset_dir, input_size, device)
    return fused, row, orig_hw


class BackWrapper(torch.nn.Module):
    """Edge-side heads operating on precomputed backbone features (dict of tensors)."""

    def __init__(self, model: torch.nn.Module, out_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.model = model
        self.out_hw = out_hw

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        m = self.model
        seg = m.classifier(features)
        if isinstance(seg, dict):
            seg = seg["out"]
        obj_in = m._object_input(features)
        if m.head_arch == "decoupled":
            object_logits = torch.cat([m.heatmap_head(obj_in), m.reg_head(obj_in)], dim=1)
        else:
            object_logits = m.object_head(obj_in)
        if tuple(object_logits.shape[-2:]) != tuple(self.out_hw):
            object_logits = F.interpolate(object_logits, size=self.out_hw, mode="bilinear", align_corners=False)
        return {"out": seg, "object": object_logits}


class FrontWrapper(torch.nn.Module):
    """Car-side backbone."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


class BackTensorWrapper(torch.nn.Module):
    """BACK with positional tensor args (low, high) so fvcore/torch.jit can trace it.

    fvcore traces with torch.jit, which cannot handle the OrderedDict feature input
    that BackWrapper takes; this rebuilds the dict internally. Compute is identical.
    """

    def __init__(self, model: torch.nn.Module, out_hw: Tuple[int, int], keys) -> None:
        super().__init__()
        self.inner = BackWrapper(model, out_hw)
        self.keys = list(keys)

    def forward(self, *tensors: torch.Tensor):
        from collections import OrderedDict

        return self.inner(OrderedDict(zip(self.keys, tensors)))
