#!/usr/bin/env python3
"""Ignore-aware adapter around the unchanged MultiTaskFusionLRASPP trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
if str(FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSION_ROOT))

from pole_lraspp_multimodal_fusion import object_targets as object_target_module  # noqa: E402
from pole_lraspp_multimodal_fusion import train_fusion as trainer  # noqa: E402


class RouteBV31Dataset(trainer.FusionPoleMultiTaskDataset):
    """The established dataset plus exact segmentation/object ignore targets."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        fused, segmentation, targets = super().__getitem__(index)
        segmentation = segmentation.clone()
        segmentation[segmentation == 255] = -100  # PyTorch CrossEntropy ignore_index.
        row = self.rows[index]
        path = self.dataset_dir / row["object_ignore_mask_path"]
        ignore = Image.open(path).convert("L")
        ignore = ignore.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        ignore_tensor = torch.from_numpy((np.asarray(ignore, dtype=np.uint8) != 0).copy())
        heatmap = targets["center_heatmap"].clone()
        # -1 is an explicit sentinel understood by focal_heatmap_loss_v31.  Exact
        # positive peaks remain active even if a neighboring quarantine overlaps.
        background_ignore = ignore_tensor.unsqueeze(0).expand_as(heatmap) & heatmap.eq(0.0)
        heatmap[background_ignore] = -1.0
        targets["center_heatmap"] = heatmap
        targets["object_ignore_mask"] = ignore_tensor.unsqueeze(0)
        return fused, segmentation, targets


def focal_heatmap_loss_v31(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_weight: Optional[torch.Tensor] = None,
    weight_cap: float = 4.0,
    class_balanced: bool = False,
    stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Established CornerNet loss with exact zero contribution at target=-1."""
    valid = target.ge(0.0).to(logits.dtype)
    safe_target = target.clamp(min=0.0, max=1.0)
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = safe_target.ge(1.0 - 1e-3).to(logits.dtype) * valid
    neg = (1.0 - pos).to(logits.dtype) * valid
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - safe_target, beta) * neg
    pos_count = pos.sum().clamp(min=1.0)
    if pos_weight is None and not class_balanced:
        return (pos_loss.sum() + neg_loss.sum()) / pos_count
    cap = float(weight_cap)
    weights = torch.ones_like(pos) if pos_weight is None else pos_weight.to(logits.dtype).expand_as(pos).clamp(min=0.0, max=cap)
    per_class_mean: List[torch.Tensor] = []
    per_class_count: List[torch.Tensor] = []
    max_weight = 0.0
    for class_index in range(int(pos.shape[1])):
        pos_class = pos[:, class_index]
        count = pos_class.sum()
        if float(count.detach().item()) < 0.5:
            continue
        class_weights = weights[:, class_index] * pos_class
        class_weights = class_weights * (count / class_weights.sum().clamp(min=1e-6))
        per_class_mean.append((pos_loss[:, class_index] * class_weights).sum() / count)
        per_class_count.append(count)
        if stats is not None:
            max_weight = max(max_weight, float(class_weights.max().detach().item()))
            stats[f"pos_mean_w_class{class_index}"] = float((class_weights.sum() / count).detach().item())
            stats[f"pos_count_class{class_index}"] = float(count.detach().item())
            stats[f"pos_mean_loss_class{class_index}"] = float(per_class_mean[-1].detach().item())
    if not per_class_mean:
        pos_term = pos_loss.sum() * 0.0
    elif class_balanced:
        pos_term = torch.stack(per_class_mean).mean()
    else:
        counts = torch.stack(per_class_count)
        pos_term = (torch.stack(per_class_mean) * counts).sum() / counts.sum().clamp(min=1.0)
    if stats is not None:
        stats["pos_max_weight"] = max_weight
        stats["pos_classes_present"] = float(len(per_class_mean))
    return pos_term + neg_loss.sum() / pos_count


def lovasz_softmax_loss_v31(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Established Lovasz-Softmax calculation after removing ignore pixels."""
    probabilities = torch.softmax(logits, dim=1)
    num_classes = probabilities.shape[1]
    probabilities_flat = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels_flat = labels.reshape(-1)
    valid = labels_flat.ge(0) & labels_flat.lt(num_classes)
    probabilities_flat = probabilities_flat[valid]
    labels_flat = labels_flat[valid]
    if labels_flat.numel() == 0:
        return logits.new_zeros(())
    losses: List[torch.Tensor] = []
    for class_index in range(num_classes):
        foreground = (labels_flat == class_index).to(probabilities_flat.dtype)
        if foreground.sum() <= 0:
            continue
        errors = (foreground - probabilities_flat[:, class_index]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, trainer._lovasz_grad(foreground[permutation])))
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def install() -> None:
    trainer.FusionPoleMultiTaskDataset = RouteBV31Dataset
    object_target_module.focal_heatmap_loss = focal_heatmap_loss_v31
    trainer.lovasz_softmax_loss = lovasz_softmax_loss_v31


def main() -> None:
    install()
    trainer.main()


if __name__ == "__main__":
    main()
