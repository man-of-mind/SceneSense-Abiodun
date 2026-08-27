#!/usr/bin/env python3
"""Recovery-v2 model assembly: warm start, registered P2 anchor addition, freeze plan.

Architecture is FIXED. Relative to `fasterrcnn_radar_roi_v1/model_v1.py` the only structural
change is the pre-registered RPN anchor addition described in `P2_ANCHOR_REGISTRATION` below.
Backbone / detector family / radar path / split boundary are untouched.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator

from model_v1 import build_model, freeze_batch_norm  # noqa: F401


# =========================== REGISTERED BEFORE TRAINING ===========================
# Evidence (retained artifact `route_b_fasterrcnn_recovery_v1/20260827_004500/
# gt_best_proposal_and_roi_logits.csv.gz`, 3,588 val frames, 4,637 eligible person GT):
#
#   person GT area (ORIGINAL 1280x720 px)   n     proposal recall @IoU0.5
#   [50,150)                                805   0.6509      <-- specific small-person failure
#   [150,400)                              1465   0.9099
#   [400,1200)                             1203   0.8928
#   [1200,inf)                             1163   0.9553
#
# Input space is 1024x576, so that bucket is 32-96 px^2 (e.g. 6x14 .. 8x20 px). The smallest
# existing anchor is scale 32 at P2, whose smallest box is 22.6x45 = 1017 px^2; the maximum
# attainable IoU of any existing anchor with a 96 px^2 GT box is therefore < 0.10, i.e. below
# the RPN fg_iou_thresh 0.7 AND below 0.5. The failure is an anchor-coverage failure, not a
# scoring failure. This licenses the single minimal P2 anchor addition permitted by the plan.
#
# THE ONLY NEW ANCHOR SCALE INTRODUCED PYRAMID-WIDE IS 16, AT P2.
# torchvision shares one RPNHead across all FPN levels, so `num_anchors_per_location` must be
# identical on every level (see torchvision.models.detection.rpn.RPNHead: a single
# `cls_logits`/`bbox_pred` conv is applied to every level). Adding an anchor at P2 alone is
# therefore structurally impossible. The forced companion slots at P3..P6 are each set to a
# scale that ALREADY EXISTS in the pyramid at the adjacent level (P3 gets 32, P4 gets 64, ...),
# so they introduce no new scale and no duplicate anchor. The expanded head is initialised by
# exact replication of the pretrained per-aspect-ratio slots, so the warm start is preserved.
P2_ANCHOR_REGISTRATION = {
    "before": {"sizes": ((32,), (64,), (128,), (256,), (512,)), "ratios": (0.5, 1.0, 2.0),
               "anchors_per_location": 3},
    "after": {"sizes": ((16, 32), (32, 64), (64, 128), (128, 256), (256, 512)),
              "ratios": (0.5, 1.0, 2.0), "anchors_per_location": 6},
    "new_scales_introduced": [16],
    "new_scale_level": "P2 (FPN key '0', stride 4)",
    "companion_slots_reuse_existing_pyramid_scales": True,
    "rpn_head_init": "exact replication of the pretrained per-aspect-ratio cls/bbox slots",
    "reason": "person GT 50-150 px^2 (original space) proposal recall @IoU0.5 = 0.6509 vs "
              "0.89-0.96 for larger persons; max attainable IoU of any pre-existing anchor "
              "with such a box is < 0.10",
    "sweep_performed": False,
}

FROZEN_PREFIXES = (
    "detector.backbone.",          # RGB ResNet50 + FPN
    "radar_encoder.",              # radar 4-channel FPN
    "radar_roi_embed.",            # radar ROI embedding
    "roi_localization_head.",      # world XYZ / dimensions / yaw / parked / radar-support heads
)
TRAINABLE_PREFIXES = (
    "detector.rpn.",
    "detector.roi_heads.box_head.",
    "detector.roi_heads.box_predictor.",
    "segmentation_decoder.",
)
# ==================================================================================


def _new_anchor_generator() -> AnchorGenerator:
    spec = P2_ANCHOR_REGISTRATION["after"]
    return AnchorGenerator(sizes=tuple(spec["sizes"]),
                           aspect_ratios=tuple(tuple(spec["ratios"]) for _ in spec["sizes"]))


def expand_rpn_with_p2_anchor(model: nn.Module) -> Dict[str, object]:
    """Apply the registered anchor addition, replicating pretrained RPN head slots exactly."""
    rpn = model.detector.rpn
    old_ratios = len(P2_ANCHOR_REGISTRATION["before"]["ratios"])
    old_cls, old_box = rpn.head.cls_logits, rpn.head.bbox_pred
    if int(old_cls.out_channels) != old_ratios:
        raise RuntimeError(f"unexpected RPN head width {old_cls.out_channels}, expected {old_ratios}")
    n_scales = len(P2_ANCHOR_REGISTRATION["after"]["sizes"][0])
    new_a = old_ratios * n_scales
    device = old_cls.weight.device
    new_cls = nn.Conv2d(old_cls.in_channels, new_a, 1).to(device)
    new_box = nn.Conv2d(old_box.in_channels, new_a * 4, 1).to(device)
    with torch.no_grad():
        # AnchorGenerator emits anchors ratio-major: index = ratio_idx * n_scales + scale_idx.
        for ratio_idx in range(old_ratios):
            for scale_idx in range(n_scales):
                dst = ratio_idx * n_scales + scale_idx
                new_cls.weight[dst].copy_(old_cls.weight[ratio_idx])
                new_cls.bias[dst].copy_(old_cls.bias[ratio_idx])
                new_box.weight[dst * 4:(dst + 1) * 4].copy_(old_box.weight[ratio_idx * 4:(ratio_idx + 1) * 4])
                new_box.bias[dst * 4:(dst + 1) * 4].copy_(old_box.bias[ratio_idx * 4:(ratio_idx + 1) * 4])
    rpn.head.cls_logits, rpn.head.bbox_pred = new_cls, new_box
    rpn.anchor_generator = _new_anchor_generator()
    return dict(P2_ANCHOR_REGISTRATION,
                anchors_per_location_verified=rpn.anchor_generator.num_anchors_per_location())


def apply_freeze_plan(model: nn.Module) -> Dict[str, object]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad = True
    return freeze_audit(model)


def freeze_audit(model: nn.Module, frozen_box_head: nn.Module | None = None) -> Dict[str, object]:
    """Prove every protected branch has zero trainable parameters."""
    protected = {prefix: {"params": 0, "trainable": 0} for prefix in FROZEN_PREFIXES}
    trainable = {prefix: {"params": 0, "trainable": 0} for prefix in TRAINABLE_PREFIXES}
    total = trainable_total = 0
    unclassified: List[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        trainable_total += count if parameter.requires_grad else 0
        hit = False
        for group in (protected, trainable):
            for prefix, bucket in group.items():
                if name.startswith(prefix):
                    bucket["params"] += count
                    bucket["trainable"] += count if parameter.requires_grad else 0
                    hit = True
        if not hit:
            unclassified.append(name)
    frozen_copy = None
    if frozen_box_head is not None:
        frozen_copy = {
            "params": int(sum(p.numel() for p in frozen_box_head.parameters())),
            "trainable": int(sum(p.numel() for p in frozen_box_head.parameters() if p.requires_grad)),
        }
    audit = {
        "protected_branches": protected,
        "trainable_branches": trainable,
        "frozen_box_head_copy": frozen_copy,
        "unclassified_parameter_names": unclassified,
        "total_parameters": total,
        "trainable_parameters": trainable_total,
        "trainable_module_count": len(TRAINABLE_PREFIXES),
        "frozen_module_count": len(FROZEN_PREFIXES) + (1 if frozen_box_head is not None else 0),
    }
    # BatchNorm affine parameters are held frozen everywhere (freeze_batch_norm), exactly as in the
    # v1 run that produced the warm start. That is deliberate: batch size 4 makes BN statistics noisy,
    # and the warm start was itself trained with frozen BN. It affects only detector.roi_heads.box_head
    # (FastRCNNConvFCHead uses nn.BatchNorm2d); the RPN head uses no norm and the segmentation decoder
    # uses GroupNorm, so both are fully trainable.
    frozen_bn = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            frozen_bn += int(sum(p.numel() for p in module.parameters() if not p.requires_grad))
    audit["frozen_batchnorm_affine_parameters"] = frozen_bn
    audit["pass"] = bool(
        all(bucket["trainable"] == 0 for bucket in protected.values())
        and all(bucket["trainable"] > 0 for bucket in trainable.values())
        and not unclassified
        and (frozen_copy is None or frozen_copy["trainable"] == 0)
    )
    return audit


def build_recovery_model(checkpoint: Dict[str, object], device: torch.device) -> Tuple[nn.Module, nn.Module, Dict]:
    """Warm start -> strict load of the ORIGINAL state dict -> anchor addition -> freeze plan."""
    width, height = map(int, checkpoint["input_size"])
    model = build_model(pretrained=False, input_size=(width, height))
    model.load_state_dict(checkpoint["model"], strict=True)
    anchor_info = expand_rpn_with_p2_anchor(model)
    model.to(device)
    # The ROI localization branch reads detector.roi_heads.box_head, which this run RETRAINS.
    # A frozen deepcopy of the ORIGINAL box_head therefore feeds localization at inference so
    # the radar localization path is preserved bit-exactly.
    frozen_box_head = copy.deepcopy(model.detector.roi_heads.box_head).to(device).eval()
    for parameter in frozen_box_head.parameters():
        parameter.requires_grad = False
    apply_freeze_plan(model)
    return model, frozen_box_head, anchor_info


def load_recovery_checkpoint(path, device: torch.device):
    """Rebuild an already-expanded recovery-v2 checkpoint for evaluation."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    width, height = map(int, payload["input_size"])
    model = build_model(pretrained=False, input_size=(width, height))
    expand_rpn_with_p2_anchor(model)
    model.load_state_dict(payload["model"], strict=True)
    frozen_box_head = copy.deepcopy(model.detector.roi_heads.box_head)
    frozen_box_head.load_state_dict(payload["frozen_box_head"])
    model.to(device).eval()
    frozen_box_head.to(device).eval()
    return payload, model, frozen_box_head


def set_eval_mode_on_frozen(model: nn.Module) -> None:
    """Frozen branches stay in eval mode so no running statistic can drift."""
    freeze_batch_norm(model)
    model.detector.backbone.eval()
    model.radar_encoder.eval()
    model.radar_roi_embed.eval()
    model.roi_localization_head.eval()
