#!/usr/bin/env python3
"""Train-time-only distillation adapter and the two registered distillation losses.

Nothing in this module survives training. The adapter is a free-standing
``torch.nn.Module`` that is deliberately NOT attached to the LR-ASPP model, so the
deployable state dict keeps exactly its 351 tensors and gains no projector or teacher
key. It exists only to put the student's ROI-pooled native features and the teacher's
ROI-pooled FPN features into one embedding space so a cosine loss is meaningful.

Registered weights, applied verbatim:

    L_feat = 0.5     cosine distance, valid GT-positive person ROIs only
    L_obj  = 0.25    BCE, valid GT-positive person centre cells only
    L_kd_reg = 0.0   teacher box regression is never distilled

Both losses are computed in float32 regardless of the autocast policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

STUDENT_ROI_CHANNELS = 1000          # 40 low + 960 high, the transported bundle
EMBED_CHANNELS = 512                 # 256 teacher P2 + 256 teacher P3
GROUP_NORM_GROUPS = 32
L_FEAT_WEIGHT = 0.5
L_OBJ_WEIGHT = 0.25
L_KD_REG_WEIGHT = 0.0


class StudentRoiAdapter(torch.nn.Module):
    """Lightweight 1x1 projection from pooled student features to the teacher space."""

    def __init__(self, in_channels: int = STUDENT_ROI_CHANNELS, out_channels: int = EMBED_CHANNELS) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
            torch.nn.GroupNorm(GROUP_NORM_GROUPS, int(out_channels)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(out_channels), int(out_channels), kernel_size=1, bias=True),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.project(pooled.float())


def feature_distillation_loss(
    student_embedding: torch.Tensor, teacher_embedding: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Scale-invariant cosine distance over person ROIs. Teacher side is detached."""
    if student_embedding.numel() == 0 or teacher_embedding.numel() == 0:
        zero = student_embedding.sum() * 0.0 if student_embedding.numel() else None
        value = zero if zero is not None else torch.zeros((), device=teacher_embedding.device)
        return value, {"l_feat": 0.0, "l_feat_rois": 0.0, "l_feat_mean_cosine": float("nan")}
    student = student_embedding.float()
    teacher = teacher_embedding.float().detach()
    cosine = F.cosine_similarity(student, teacher, dim=1, eps=1e-8)     # [N, k, k]
    loss = (1.0 - cosine).mean()
    return loss, {
        "l_feat": float(loss.detach().item()),
        "l_feat_rois": float(student.shape[0]),
        "l_feat_mean_cosine": float(cosine.detach().mean().item()),
    }


def objectness_distillation_loss(
    person_logits: torch.Tensor, teacher_scores: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE at GT-positive person centre cells that carry IoU-matched teacher evidence.

    ``teacher_scores`` may contain ``nan`` for GT person boxes the teacher did not
    detect. Those sites are dropped, never converted into a target of either sign, so
    a teacher miss can never push a genuine CARLA GT positive down. Sites where the
    teacher fires but GT has no person are already absent from this tensor: it is
    indexed by GT, not by teacher output.
    """
    if person_logits.numel() == 0:
        return (person_logits.sum() * 0.0, {"l_obj": 0.0, "l_obj_sites": 0.0,
                                            "l_obj_mean_teacher_score": float("nan")})
    logits = person_logits.float()
    targets = teacher_scores.float()
    valid = torch.isfinite(targets)
    if not bool(valid.any().item()):
        return (logits.sum() * 0.0, {"l_obj": 0.0, "l_obj_sites": 0.0,
                                     "l_obj_mean_teacher_score": float("nan")})
    logits = logits[valid]
    targets = targets[valid].clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    return loss, {
        "l_obj": float(loss.detach().item()),
        "l_obj_sites": float(int(valid.sum().item())),
        "l_obj_mean_teacher_score": float(targets.mean().item()),
    }


def gather_person_logits(
    object_logits: torch.Tensor, cells: List[torch.Tensor], person_channel: int
) -> torch.Tensor:
    """Student person heatmap logits at the GT-positive person centre cells."""
    collected: List[torch.Tensor] = []
    for index, frame_cells in enumerate(cells):
        if frame_cells.numel() == 0:
            continue
        rows = frame_cells[:, 0].long()
        columns = frame_cells[:, 1].long()
        collected.append(object_logits[index, person_channel, rows, columns])
    if not collected:
        return object_logits.new_zeros((0,))
    return torch.cat(collected, dim=0)


def adapter_report(adapter: torch.nn.Module) -> Dict[str, Any]:
    total = sum(int(p.numel()) for p in adapter.parameters())
    trainable = sum(int(p.numel()) for p in adapter.parameters() if p.requires_grad)
    return {
        "module": type(adapter).__name__,
        "parameters": total,
        "trainable_parameters": trainable,
        "tensor_names": sorted(adapter.state_dict().keys()),
        "train_time_only": True,
        "discarded_at_deployment": True,
    }
