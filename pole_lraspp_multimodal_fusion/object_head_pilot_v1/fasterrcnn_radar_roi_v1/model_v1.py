#!/usr/bin/env python3
"""COCO Faster R-CNN v2 with radar-conditioned ROI localization and split boundary."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign

from split_runtime_adapter_v1 import reconstruct_image_list


ROUTE_CLASS_NAMES = ("vehicle", "person")
COCO_MAPPING = {0: (0,), 1: (3, 4, 6, 8), 2: (1,)}
ROI_FIELD_NAMES = (
    "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z",
    "local_yaw_sin", "local_yaw_cos",
    "parked_logit", "radar_support_logit",
)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(16, channels), channels)


class RadarPyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False), _norm(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), _norm(64), nn.SiLU(),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), _norm(64), nn.SiLU()),
                nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), _norm(64), nn.SiLU()),
                nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False), _norm(64), nn.SiLU()),
            ]
        )

    def forward(self, radar: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        p2 = self.stem(radar)
        p3 = self.stages[0](p2)
        p4 = self.stages[1](p3)
        p5 = self.stages[2](p4)
        return OrderedDict(
            [("0", p2), ("1", p3), ("2", p4), ("3", p5), ("pool", F.max_pool2d(p5, 1, stride=2))]
        )


class SemanticFPNDecoder(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {key: nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), _norm(64), nn.SiLU()) for key in ("0", "1", "2", "3")}
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False), _norm(128), nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), _norm(64), nn.SiLU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, features: Dict[str, torch.Tensor], output_size: Tuple[int, int]) -> torch.Tensor:
        base_size = features["0"].shape[-2:]
        parts = []
        for key in ("0", "1", "2", "3"):
            value = self.projections[key](features[key])
            if value.shape[-2:] != base_size:
                value = F.interpolate(value, size=base_size, mode="bilinear", align_corners=False)
            parts.append(value)
        logits = self.fuse(torch.cat(parts, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def remap_coco_predictor(detector: nn.Module) -> Dict[str, object]:
    """Build Route B bg/vehicle/person predictor while retaining useful COCO rows."""
    old = detector.roi_heads.box_predictor
    new = FastRCNNPredictor(old.cls_score.in_features, 3).to(old.cls_score.weight.device)
    with torch.no_grad():
        for route_label, coco_labels in COCO_MAPPING.items():
            index = torch.tensor(coco_labels, dtype=torch.long, device=old.cls_score.weight.device)
            new.cls_score.weight[route_label].copy_(old.cls_score.weight.index_select(0, index).mean(0))
            new.cls_score.bias[route_label].copy_(old.cls_score.bias.index_select(0, index).mean(0))
            bbox_rows = torch.cat(
                [torch.arange(label * 4, label * 4 + 4, device=index.device) for label in coco_labels]
            ).reshape(len(coco_labels), 4)
            new.bbox_pred.weight[route_label * 4 : route_label * 4 + 4].copy_(
                old.bbox_pred.weight.index_select(0, bbox_rows.reshape(-1)).reshape(len(coco_labels), 4, -1).mean(0)
            )
            new.bbox_pred.bias[route_label * 4 : route_label * 4 + 4].copy_(
                old.bbox_pred.bias.index_select(0, bbox_rows.reshape(-1)).reshape(len(coco_labels), 4).mean(0)
            )
    detector.roi_heads.box_predictor = new
    return {
        "route_background_0": [0],
        "route_vehicle_1_mean": [3, 4, 6, 8],
        "route_person_2": [1],
        "vehicle_categories": ["car", "motorcycle", "bus", "truck"],
        "person_category": "person",
        "classification_and_class_specific_box_rows_copied": True,
    }


class FasterRCNNRadarROI(nn.Module):
    """Split model. Only the complete feature bundle crosses the boundary."""

    def __init__(self, *, pretrained: bool, input_size: Tuple[int, int] = (1024, 576)) -> None:
        super().__init__()
        width, height = map(int, input_size)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            min_size=height,
            max_size=width,
            trainable_backbone_layers=5,
        )
        self.coco_mapping = remap_coco_predictor(self.detector) if pretrained else self._replace_predictor()
        self.detector.roi_heads.score_thresh = 0.02
        self.detector.roi_heads.nms_thresh = 0.5
        self.detector.roi_heads.detections_per_img = 100
        self.input_size = (width, height)

        self.radar_encoder = RadarPyramid()
        self.radar_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        self.radar_roi_embed = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), _norm(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.roi_localization_head = nn.Sequential(
            nn.Linear(1024 + 128, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 10),
        )
        self.segmentation_decoder = SemanticFPNDecoder(num_classes=3)

    def _replace_predictor(self) -> Dict[str, object]:
        old = self.detector.roi_heads.box_predictor
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(old.cls_score.in_features, 3)
        return {"checkpoint_load_build": True}

    def encode_front(self, rgb: Sequence[torch.Tensor], radar: Sequence[torch.Tensor]) -> Dict[str, object]:
        """Encode raw RGB3/radar4 into the complete transport boundary bundle."""
        original_sizes = [tuple(map(int, image.shape[-2:])) for image in rgb]
        images, _ = self.detector.transform(list(rgb), None)
        radar_batch = torch.stack(list(radar), dim=0)
        if radar_batch.shape[-2:] != images.tensors.shape[-2:]:
            radar_batch = F.interpolate(radar_batch, size=images.tensors.shape[-2:], mode="bilinear", align_corners=False)
        rgb_features = self.detector.backbone(images.tensors)
        if isinstance(rgb_features, torch.Tensor):
            rgb_features = OrderedDict([("0", rgb_features)])
        radar_features = self.radar_encoder(radar_batch)
        return {
            "image_batch_shape": list(images.tensors.shape),
            "image_sizes": list(images.image_sizes),
            "original_image_sizes": original_sizes,
            "rgb_fpn": rgb_features,
            "radar_fpn": radar_features,
        }

    def _roi_predictions(
        self,
        rgb_features: Dict[str, torch.Tensor],
        radar_features: Dict[str, torch.Tensor],
        boxes: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if sum(int(value.shape[0]) for value in boxes) == 0:
            return rgb_features["0"].new_zeros((0, 10))
        visual = self.detector.roi_heads.box_roi_pool(rgb_features, boxes, image_sizes)
        visual = self.detector.roi_heads.box_head(visual)
        radar = self.radar_roi_pool(radar_features, boxes, image_sizes)
        radar = self.radar_roi_embed(radar)
        return self.roi_localization_head(torch.cat([visual, radar], dim=1))

    def _localization_losses(
        self,
        predictions: torch.Tensor,
        labels: List[torch.Tensor],
        matched_indices: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        label_cat = torch.cat(labels, dim=0)
        positive = label_cat > 0
        if not bool(positive.any()):
            zero = predictions.sum() * 0.0
            return {key: zero for key in ("loss_roi_xyz", "loss_roi_dimensions", "loss_roi_yaw", "loss_roi_parked", "loss_roi_radar")}
        expected = torch.cat(
            [target["roi_fields"][matched[pos]] for target, matched, label in zip(targets, matched_indices, labels) for pos in [label > 0]],
            dim=0,
        )
        pred = predictions[positive]
        xyz = F.smooth_l1_loss(pred[:, 0:3], expected[:, 0:3])
        dims = F.smooth_l1_loss(F.softplus(pred[:, 3:6]), expected[:, 3:6])
        yaw = F.smooth_l1_loss(F.normalize(pred[:, 6:8], dim=1), expected[:, 6:8])
        parked = F.binary_cross_entropy_with_logits(pred[:, 8], expected[:, 8])
        radar = F.binary_cross_entropy_with_logits(pred[:, 9], expected[:, 9])
        return {
            "loss_roi_xyz": 1.5 * xyz,
            "loss_roi_dimensions": 0.6 * dims,
            "loss_roi_yaw": 0.3 * yaw,
            "loss_roi_parked": 0.2 * parked,
            "loss_roi_radar": 0.1 * radar,
        }

    def decode_tail(
        self,
        bundle: Dict[str, object],
        image_size: Sequence[Tuple[int, int]] | Tuple[int, int],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, object]:
        """Decode exclusively from the boundary bundle; no raw modality argument exists."""
        rgb_features = bundle["rgb_fpn"]
        radar_features = bundle["radar_fpn"]
        image_sizes = list(bundle["image_sizes"])
        batch_shape = list(bundle["image_batch_shape"])
        images = reconstruct_image_list(
            tuple(batch_shape), image_sizes, rgb_features["0"].device,
            dtype=rgb_features["0"].dtype,
        )
        if isinstance(image_size, tuple) and len(image_size) == 2 and isinstance(image_size[0], int):
            original_sizes = [image_size] * len(image_sizes)
        else:
            original_sizes = list(image_size)
        proposals, proposal_losses = self.detector.rpn(images, rgb_features, targets)
        seg_logits = self.segmentation_decoder(rgb_features, (batch_shape[-2], batch_shape[-1]))

        if self.training:
            if targets is None:
                raise ValueError("targets required in training mode")
            proposals, matched, labels, regression_targets = self.detector.roi_heads.select_training_samples(proposals, targets)
            visual = self.detector.roi_heads.box_roi_pool(rgb_features, proposals, image_sizes)
            visual = self.detector.roi_heads.box_head(visual)
            class_logits, box_regression = self.detector.roi_heads.box_predictor(visual)
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            radar = self.radar_roi_embed(self.radar_roi_pool(radar_features, proposals, image_sizes))
            local_predictions = self.roi_localization_head(torch.cat([visual, radar], dim=1))
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, **proposal_losses}
            losses.update(self._localization_losses(local_predictions, labels, matched, targets))
            seg_targets = torch.stack([target["segmentation"] for target in targets], dim=0)
            if seg_targets.shape[-2:] != seg_logits.shape[-2:]:
                seg_targets = F.interpolate(seg_targets[:, None].float(), size=seg_logits.shape[-2:], mode="nearest")[:, 0].long()
            class_weights = seg_logits.new_tensor([0.2, 1.0, 4.0])
            losses["loss_segmentation"] = F.cross_entropy(seg_logits, seg_targets, weight=class_weights)
            return {"losses": losses, "detections": [], "segmentation": seg_logits}

        detections, detector_losses = self.detector.roi_heads(rgb_features, proposals, image_sizes, None)
        detection_boxes = [item["boxes"] for item in detections]
        local_predictions = self._roi_predictions(rgb_features, radar_features, detection_boxes, image_sizes)
        offset = 0
        for item in detections:
            count = int(item["boxes"].shape[0])
            fields = local_predictions[offset : offset + count]
            offset += count
            item["local_xyz"] = fields[:, 0:3]
            item["dimensions"] = F.softplus(fields[:, 3:6])
            item["local_yaw_sincos"] = F.normalize(fields[:, 6:8], dim=1)
            item["parked_score"] = torch.sigmoid(fields[:, 8])
            item["radar_support_score"] = torch.sigmoid(fields[:, 9])
        detections = self.detector.transform.postprocess(detections, image_sizes, original_sizes)
        return {"losses": {**proposal_losses, **detector_losses}, "detections": detections, "segmentation": seg_logits}

    def forward(
        self,
        rgb: Sequence[torch.Tensor],
        radar: Sequence[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, object]:
        bundle = self.encode_front(rgb, radar)
        return self.decode_tail(bundle, bundle["original_image_sizes"], targets)


def build_model(*, pretrained: bool, input_size: Tuple[int, int] = (1024, 576)) -> FasterRCNNRadarROI:
    return FasterRCNNRadarROI(pretrained=pretrained, input_size=input_size)


def freeze_batch_norm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            for parameter in child.parameters():
                parameter.requires_grad = False


def boundary_manifest(bundle: Dict[str, object]) -> Dict[str, object]:
    tensors: Dict[str, torch.Tensor] = {}
    for group in ("rgb_fpn", "radar_fpn"):
        for name, value in bundle[group].items():
            tensors[f"{group}.{name}"] = value
    rows = []
    total = 0
    for name, value in tensors.items():
        size = int(value.numel() * value.element_size())
        total += size
        rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "bytes": size})
    return {
        "tensors": rows,
        "total_bytes": total,
        "metadata": {
            "image_batch_shape": list(bundle["image_batch_shape"]),
            "image_sizes": [list(value) for value in bundle["image_sizes"]],
            "original_image_sizes": [list(value) for value in bundle["original_image_sizes"]],
        },
        "raw_rgb_present": False,
        "raw_radar_present": False,
    }


def records_from_detections(
    detection: Dict[str, torch.Tensor], camera_matrix: np.ndarray, camera_yaw_deg: float
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for index in range(int(detection["boxes"].shape[0])):
        local = detection["local_xyz"][index].detach().cpu().numpy()
        world = (camera_matrix @ np.array([local[0], local[1], local[2], 1.0], dtype=np.float64))[:3]
        yaw_sc = detection["local_yaw_sincos"][index].detach().cpu().numpy()
        local_yaw = float(np.arctan2(yaw_sc[0], yaw_sc[1]))
        world_yaw = local_yaw + np.deg2rad(float(camera_yaw_deg))
        box = detection["boxes"][index].detach().cpu().numpy()
        label = int(detection["labels"][index].item())
        dims = detection["dimensions"][index].detach().cpu().numpy()
        records.append(
            {
                "class_index": float(label - 1),
                "class_name": ROUTE_CLASS_NAMES[label - 1],
                "score": float(detection["scores"][index].item()),
                "bbox_x0": float(box[0]), "bbox_y0": float(box[1]),
                "bbox_x1": float(box[2]), "bbox_y1": float(box[3]),
                "center_x_px": float(0.5 * (box[0] + box[2])),
                "center_y_px": float(0.5 * (box[1] + box[3])),
                "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
                "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
                "size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2]),
                "yaw_sin": float(np.sin(world_yaw)), "yaw_cos": float(np.cos(world_yaw)),
                "parked_score": float(detection["parked_score"][index].item()),
                "radar_support_score": float(detection["radar_support_score"][index].item()),
            }
        )
    return records
