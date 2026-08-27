#!/usr/bin/env python3
"""Registered recovery-v2 objectives. Fixed before training; no variant is swept."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


# =========================== REGISTERED BEFORE TRAINING ===========================
# ROI classification: class-weighted cross-entropy. Weights are the inverse square root of the
# MEASURED TRAIN-ONLY realized ROI label frequency under this run's registered sampler, capped.
#   measured (train split, 6,600 frames x 256 ROIs = 1,689,600 ROIs, artifact
#   route_b_fasterrcnn_recovery_v1/20260827_004500/finetune_history.json epoch 1):
#       background 1,399,834 (0.82850) | vehicle 189,713 (0.11228) | person 100,053 (0.05922)
#   w_c = clip(sqrt(f_background / f_c), 1.0, WEIGHT_CAP), w_background := 1.0
#       vehicle sqrt(0.82850/0.11228) = 2.71624
#       person  sqrt(0.82850/0.05922) = 3.74076 -> capped to 3.0
# The cap bounds the effective foreground:background gradient ratio at 3:1, which is what keeps
# the warm-started, already-calibrated classifier stable.
#
# NO hard-negative mining and NO hard-negative oversampling. The v1 recovery run's OHEM recipe
# (0.75 of negatives drawn from the highest-scoring background pool) made the classifier too
# conservative (vehicle ceiling -0.020, person ceiling -0.078) and is explicitly not repeated.
# Negatives here are sampled UNIFORMLY AT RANDOM.
ROI_LABEL_FREQUENCY_TRAIN = {"background": 0.828500, "vehicle": 0.112282, "person": 0.059217}
WEIGHT_CAP = 3.0
ROI_CE_WEIGHTS = (1.0, 2.71624, 3.0)

# Segmentation: median-frequency class balancing from TRAIN pixel counts (manifest columns
# `vehicle_pixels` / `person_pixels` over the 6,600 train rows), clipped to [0.2, 8.0], plus an
# equally-weighted 3-class soft Dice term. Dice is included because the gate is an IoU gate and
# person occupies 0.29% of pixels, where weighted CE alone optimises the wrong surrogate.
#   train pixel frequency: background 0.956725 | vehicle 0.040358 | person 0.002917
#   median frequency = 0.040358 (vehicle)
#   raw median/freq  = 0.04218 | 1.0 | 13.8318   -> clipped [0.2, 8.0] = 0.2 | 1.0 | 8.0
# Lower clip 0.2 is the v1 background weight, already proven stable on this decoder.
SEG_PIXEL_FREQUENCY_TRAIN = {"background": 0.956725, "vehicle": 0.040358, "person": 0.002917}
SEG_WEIGHT_CLIP = (0.2, 8.0)
SEG_CE_WEIGHTS = (0.2, 1.0, 8.0)
SEG_DICE_EPS = 1.0
LOSS_WEIGHTS = {"rpn": 1.0, "roi_classifier": 1.0, "roi_box_reg": 1.0, "segmentation": 1.0}
# ==================================================================================


def roi_losses(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: torch.Tensor,
    regression_targets: torch.Tensor,
    weights: Sequence[float] = ROI_CE_WEIGHTS,
) -> Dict[str, torch.Tensor]:
    """Class-weighted CE + torchvision-convention positive-only smooth L1 box regression."""
    weight = class_logits.new_tensor(tuple(weights))
    loss_classifier = F.cross_entropy(class_logits, labels, weight=weight)
    positive = torch.where(labels > 0)[0]
    if positive.numel() == 0:
        return {"loss_roi_classifier": loss_classifier, "loss_roi_box_reg": box_regression.sum() * 0.0}
    rows = box_regression.reshape(class_logits.shape[0], box_regression.size(-1) // 4, 4)
    loss_box_reg = F.smooth_l1_loss(
        rows[positive, labels[positive]], regression_targets[positive],
        beta=1.0 / 9.0, reduction="sum",
    ) / labels.numel()
    return {"loss_roi_classifier": loss_classifier, "loss_roi_box_reg": loss_box_reg}


def segmentation_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Sequence[float] = SEG_CE_WEIGHTS,
) -> Dict[str, torch.Tensor]:
    """Bounded class-balanced CE + equally-weighted 3-class soft Dice."""
    weight = logits.new_tensor(tuple(weights))
    loss_ce = F.cross_entropy(logits, targets, weight=weight)
    probabilities = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    cardinality = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + SEG_DICE_EPS) / (cardinality + SEG_DICE_EPS)
    return {"loss_segmentation_ce": loss_ce, "loss_segmentation_dice": 1.0 - dice.mean()}


def registration() -> Dict[str, object]:
    return {
        "roi_classification": {
            "formulation": "class-weighted cross-entropy",
            "train_label_frequency": ROI_LABEL_FREQUENCY_TRAIN,
            "rule": "w_c = clip(sqrt(f_background / f_c), 1.0, cap); w_background := 1.0",
            "weight_cap": WEIGHT_CAP,
            "weights_bg_vehicle_person": list(ROI_CE_WEIGHTS),
            "hard_negative_mining": False,
            "hard_negative_oversampling": False,
            "negative_sampling": "uniform at random",
        },
        "roi_box_regression": {"formulation": "positive-only smooth L1, beta=1/9, normalised by sampled ROI count"},
        "segmentation": {
            "formulation": "median-frequency-balanced CE + equally-weighted 3-class soft Dice",
            "train_pixel_frequency": SEG_PIXEL_FREQUENCY_TRAIN,
            "weight_clip": list(SEG_WEIGHT_CLIP),
            "ce_weights_bg_vehicle_person": list(SEG_CE_WEIGHTS),
            "dice_epsilon": SEG_DICE_EPS,
            "independence": "backbone is frozen and its features are detached, and the decoder "
                            "shares no parameter with the detection or localization branches, so "
                            "the segmentation gradient cannot reach detection or localization",
        },
        "loss_weights": LOSS_WEIGHTS,
        "rpn": {"formulation": "torchvision native objectness BCE + smooth L1, trained jointly "
                               "so proposal coverage keeps margin above the final recall targets"},
    }
