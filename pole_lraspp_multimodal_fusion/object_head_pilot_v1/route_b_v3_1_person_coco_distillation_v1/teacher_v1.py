#!/usr/bin/env python3
"""Frozen COCO Faster R-CNN teacher, used only during training and then discarded.

The teacher never appears in the deployed model, never sees radar, never runs at
validation or test, and never produces a label. It contributes exactly two training
signals, both restricted to regions that CARLA v3.1 actor GT already declares to be a
person:

  * FPN P2/P3 features, ROI-pooled over valid person GT boxes  -> L_feat
  * COCO class-1 (person) box scores, IoU-matched to those same GT boxes -> L_obj

Box regression, depth, dimensions, yaw and world localization are never read.

Coordinate alignment
--------------------
torchvision's ``GeneralizedRCNNTransform`` normally rescales an image to its own
min/max size, which would make teacher and student pixel frames different and would
force an approximate ROI mapping. That is not acceptable here, so the transform is
pinned to ``min_size=(432,), max_size=768``. For the fixed 432x768 student canvas the
implied scale factor is ``min(432/432, 768/768) = 1.0`` exactly, so the teacher image
frame IS the student model-input frame. Padding to the 32-divisible 448x768 canvas is
bottom/right only and therefore does not move any coordinate. Both facts are asserted
numerically before training; a failure is a hard CONTRACT_INVALID, never an
approximation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.ops import box_iou

from roi_v1 import roi_align_model_frame

TEACHER_CACHE = Path.home() / ".cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
TEACHER_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
TEACHER_BYTES = 175221657
COCO_PERSON_LABEL = 1

MODEL_HEIGHT, MODEL_WIDTH = 432, 768
P2_KEY, P2_SCALE = "0", 0.25
P3_KEY, P3_SCALE = "1", 0.125
ROI_OUTPUT = 7
ROI_SAMPLING_RATIO = 2
TEACHER_EMBED_CHANNELS = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_teacher_cache() -> Dict[str, Any]:
    """Hash the already-cached official weights. No download is ever attempted."""
    if not TEACHER_CACHE.is_file():
        raise RuntimeError(f"teacher weight cache absent: {TEACHER_CACHE}")
    digest = sha256_file(TEACHER_CACHE)
    size = TEACHER_CACHE.stat().st_size
    return {
        "path": str(TEACHER_CACHE),
        "sha256": digest,
        "bytes": size,
        "sha256_matches": digest == TEACHER_SHA256,
        "bytes_match": size == TEACHER_BYTES,
        "weights_enum": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "weights_url": FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url,
        "url_matches_provenance": FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url.endswith(
            "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        ),
        "torchvision_version": torchvision.__version__,
    }


def build_teacher(device: torch.device) -> torch.nn.Module:
    """Load the stock COCO detector, pin its transform, freeze it permanently."""
    teacher = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    teacher.transform.min_size = (MODEL_HEIGHT,)
    teacher.transform.max_size = MODEL_WIDTH
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.to(device)
    # eval() is re-asserted every epoch; train() is disabled outright so that no later
    # code path can silently put the teacher into training mode.
    teacher.train = _refuse_train.__get__(teacher, type(teacher))
    return teacher


def _refuse_train(self, mode: bool = True):  # noqa: ANN001, ANN202
    if mode:
        raise RuntimeError("the distillation teacher is permanently frozen in eval() mode")
    return self


def teacher_state(teacher: torch.nn.Module) -> Dict[str, Any]:
    total = sum(int(p.numel()) for p in teacher.parameters())
    trainable = sum(int(p.numel()) for p in teacher.parameters() if p.requires_grad)
    return {
        "training_mode": bool(teacher.training),
        "parameters": total,
        "trainable_parameters": trainable,
        "all_frozen": trainable == 0,
        "transform_min_size": list(teacher.transform.min_size),
        "transform_max_size": int(teacher.transform.max_size),
        "transform_size_divisible": int(teacher.transform.size_divisible),
        "image_mean": list(teacher.transform.image_mean),
        "image_std": list(teacher.transform.image_std),
    }


@torch.no_grad()
def verify_transform_identity(teacher: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Prove teacher pixel space == student model-input pixel space, exactly."""
    probe = torch.rand(2, 3, MODEL_HEIGHT, MODEL_WIDTH, device=device)
    images, _ = teacher.transform([probe[0], probe[1]], None)
    mean = torch.tensor(teacher.transform.image_mean, device=device).view(3, 1, 1)
    std = torch.tensor(teacher.transform.image_std, device=device).view(3, 1, 1)
    reference = (probe - mean) / std
    body = images.tensors[:, :, :MODEL_HEIGHT, :MODEL_WIDTH]
    identity_delta = float((body - reference).abs().max().item())
    pad_rows = images.tensors[:, :, MODEL_HEIGHT:, :]
    pad_cols = images.tensors[:, :, :, MODEL_WIDTH:]
    features = teacher.backbone(images.tensors)
    shapes = {key: list(value.shape) for key, value in features.items()}
    p2, p3 = features[P2_KEY], features[P3_KEY]
    return {
        "image_sizes": [list(size) for size in images.image_sizes],
        "padded_tensor_shape": list(images.tensors.shape),
        "scale_factor": 1.0,
        "identity_max_abs_delta": identity_delta,
        "transform_is_identity": identity_delta == 0.0,
        "image_sizes_are_model_size": all(
            tuple(size) == (MODEL_HEIGHT, MODEL_WIDTH) for size in images.image_sizes
        ),
        "padding_is_trailing_only": True,
        "padding_max_abs_value": float(
            max(
                pad_rows.abs().max().item() if pad_rows.numel() else 0.0,
                pad_cols.abs().max().item() if pad_cols.numel() else 0.0,
            )
        ),
        "fpn_level_shapes": shapes,
        "p2_stride": MODEL_WIDTH / float(p2.shape[-1]),
        "p3_stride": MODEL_WIDTH / float(p3.shape[-1]),
        "p2_stride_is_4": abs(MODEL_WIDTH / float(p2.shape[-1]) - 4.0) < 1e-12,
        "p3_stride_is_8": abs(MODEL_WIDTH / float(p3.shape[-1]) - 8.0) < 1e-12,
        "p2_covers_model_rows": int(p2.shape[-2]) * 4 >= MODEL_HEIGHT,
        "p3_covers_model_rows": int(p3.shape[-2]) * 8 >= MODEL_HEIGHT,
    }


@torch.no_grad()
def teacher_forward(
    teacher: torch.nn.Module, rgb01: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Run the frozen teacher on unnormalized RGB in [0, 1]. Returns (P2, P3, detections).

    The teacher's own transform performs its native normalization, so the teacher sees
    exactly the augmented RGB the student sees, in the teacher's own input convention.
    """
    if rgb01.dim() != 4 or int(rgb01.shape[1]) != 3:
        raise ValueError(f"teacher expects [B,3,H,W] RGB, got {tuple(rgb01.shape)}")
    original_sizes = [(int(rgb01.shape[-2]), int(rgb01.shape[-1]))] * int(rgb01.shape[0])
    images, _ = teacher.transform([image for image in rgb01], None)
    features = teacher.backbone(images.tensors)
    proposals, _ = teacher.rpn(images, features, None)
    detections, _ = teacher.roi_heads(features, proposals, images.image_sizes, None)
    detections = teacher.transform.postprocess(detections, images.image_sizes, original_sizes)
    return features[P2_KEY], features[P3_KEY], detections


def teacher_person_evidence(
    detections: Sequence[Dict[str, torch.Tensor]],
    person_boxes: Sequence[torch.Tensor],
    *,
    iou_threshold: float = 0.5,
    score_floor: float = 0.05,
) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    """IoU-match teacher COCO person detections to valid v3.1 person GT boxes.

    Returns one score tensor per image, aligned with ``person_boxes``. A GT box with no
    matched teacher person detection gets ``nan``: that site is OMITTED from L_obj
    rather than being given a signed target, so the teacher can never suppress a GT
    positive and can never create one.

    Teacher person detections that match no valid GT person box are counted and
    discarded. They are exactly the "teacher positive, GT absent or ignored" sites the
    contract forbids from becoming supervision of either sign.
    """
    per_image: List[torch.Tensor] = []
    stats = {
        "gt_person_boxes": 0,
        "gt_person_boxes_with_teacher_evidence": 0,
        "gt_person_boxes_without_teacher_evidence": 0,
        "teacher_person_detections_above_floor": 0,
        "omitted_teacher_positive_gt_absent": 0,
    }
    for detection, boxes in zip(detections, person_boxes):
        labels = detection["labels"]
        scores = detection["scores"]
        keep = (labels == COCO_PERSON_LABEL) & (scores >= float(score_floor))
        teacher_boxes = detection["boxes"][keep]
        teacher_scores = scores[keep]
        stats["teacher_person_detections_above_floor"] += int(teacher_boxes.shape[0])
        stats["gt_person_boxes"] += int(boxes.shape[0])
        if boxes.numel() == 0:
            stats["omitted_teacher_positive_gt_absent"] += int(teacher_boxes.shape[0])
            per_image.append(boxes.new_zeros((0,)))
            continue
        if teacher_boxes.numel() == 0:
            stats["gt_person_boxes_without_teacher_evidence"] += int(boxes.shape[0])
            per_image.append(torch.full((int(boxes.shape[0]),), float("nan"),
                                        device=boxes.device, dtype=torch.float32))
            continue
        overlap = box_iou(teacher_boxes.float(), boxes.float())          # [T, G]
        eligible = overlap >= float(iou_threshold)
        matched_scores = torch.where(
            eligible, teacher_scores.float().unsqueeze(1), torch.zeros_like(overlap)
        )
        best, _ = matched_scores.max(dim=0)                               # [G]
        has_evidence = eligible.any(dim=0)
        target = torch.where(has_evidence, best, torch.full_like(best, float("nan")))
        stats["gt_person_boxes_with_teacher_evidence"] += int(has_evidence.sum().item())
        stats["gt_person_boxes_without_teacher_evidence"] += int((~has_evidence).sum().item())
        stats["omitted_teacher_positive_gt_absent"] += int((~eligible.any(dim=1)).sum().item())
        per_image.append(target)
    return per_image, stats


def teacher_roi_embedding(
    p2: torch.Tensor, p3: torch.Tensor, boxes: List[torch.Tensor]
) -> torch.Tensor:
    """Fixed teacher-side ROI embedding: concat(ROIAlign(P2), ROIAlign(P3)) -> [N,512,7,7].

    The teacher-side adapter is deliberately the identity. A trainable adapter on both
    sides of a cosine objective admits a trivial constant-collapse solution, so the
    common embedding space is pinned to the teacher's own ROI feature space and only
    the student side is learned.
    """
    pooled_p2 = roi_align_model_frame(
        p2.float(), boxes, output_size=ROI_OUTPUT, spatial_scale=P2_SCALE,
        sampling_ratio=ROI_SAMPLING_RATIO,
    )
    pooled_p3 = roi_align_model_frame(
        p3.float(), boxes, output_size=ROI_OUTPUT, spatial_scale=P3_SCALE,
        sampling_ratio=ROI_SAMPLING_RATIO,
    )
    return torch.cat([pooled_p2, pooled_p3], dim=1)
