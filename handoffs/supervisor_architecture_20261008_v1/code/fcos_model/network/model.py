from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

from common import sha256, tensor_hash
from data import CONTENT_H, CONTENT_W, DEPTH_BINS, NETWORK_H, depth_edges

LEVELS = ("p2", "p3", "p4", "p5", "p6", "p7")
ANCHOR_SIZES = (4, 8, 16, 32, 64, 128)
CLASS_NAMES = ("vehicle", "person")
OFFICIAL_WEIGHTS_SHA256 = "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"
OFFICIAL_WEIGHTS_BYTES = 129612099
_VERIFIED_WEIGHT_PATHS: set[Path] = set()


class SevenChannelConvFront(nn.Module):
    """One mathematical seven-channel convolution with separately registered slices."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        source: nn.Conv2d = body.conv1
        self.W_rgb = nn.Parameter(source.weight.detach().clone())
        self.W_radar = nn.Parameter(torch.zeros(
            source.out_channels, 4, *source.kernel_size, dtype=source.weight.dtype,
        ))
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.bn1 = copy.deepcopy(body.bn1)
        self.relu = copy.deepcopy(body.relu)
        self.maxpool = copy.deepcopy(body.maxpool)
        self.layer1 = copy.deepcopy(body.layer1)

    def concatenated_weight(self) -> torch.Tensor:
        return torch.cat([self.W_rgb, self.W_radar], dim=1)

    def forward(self, x7: torch.Tensor) -> torch.Tensor:
        if x7.ndim != 4 or tuple(x7.shape[1:])[:1] != (7,) or tuple(x7.shape[-2:]) != (NETWORK_H, CONTENT_W):
            raise ValueError(f"expected [B,7,{NETWORK_H},{CONTENT_W}], got {tuple(x7.shape)}")
        device_type = x7.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            value = F.conv2d(x7.float(), self.concatenated_weight().float(), bias=None,
                             stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            value = self.layer1(self.maxpool(self.relu(self.bn1(value))))
        if tuple(value.shape[1:]) != (256, 112, 192):
            raise RuntimeError(f"C2 boundary drift: {tuple(value.shape)}")
        return value.float()


class P2P7Tail(nn.Module):
    def __init__(self, official: nn.Module) -> None:
        super().__init__()
        body = official.backbone.body
        self.layer2 = copy.deepcopy(body.layer2)
        self.layer3 = copy.deepcopy(body.layer3)
        self.layer4 = copy.deepcopy(body.layer4)
        self.official_fpn_p3_p7 = copy.deepcopy(official.backbone.fpn)
        self.p2_lateral = nn.Conv2d(256, 256, kernel_size=1)
        self.p2_output = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        for layer in (self.p2_lateral, self.p2_output):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.zeros_(layer.bias)

    def resnet(self, c2: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict((("0", c3), ("1", c4), ("2", c5)))

    def forward(self, c2: torch.Tensor) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
        cs = self.resnet(c2)
        official = self.official_fpn_p3_p7(cs)
        p3_inner = self.official_fpn_p3_p7.get_result_from_inner_blocks(cs["0"], 0)
        p2_inner = self.p2_lateral(c2) + F.interpolate(p3_inner, size=c2.shape[-2:], mode="nearest")
        p2 = self.p2_output(p2_inner)
        values = OrderedDict((("p2", p2), ("p3", official["0"]), ("p4", official["1"]),
                              ("p5", official["2"]), ("p6", official["p6"]), ("p7", official["p7"])))
        return values, cs


class ConvGNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )


class P2TaskHead(nn.Module):
    def __init__(self, out_channels: int, bias: Sequence[float] | float = 0.0) -> None:
        super().__init__()
        self.tower = nn.Sequential(ConvGNReLU(256, 128), ConvGNReLU(128, 128))
        self.output = nn.Conv2d(128, out_channels, 1)
        self._initialize(bias)

    def _initialize(self, bias: Sequence[float] | float) -> None:
        for module in self.tower.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.output.weight, std=0.001)
        values = torch.as_tensor(bias, dtype=self.output.bias.dtype).flatten()
        with torch.no_grad():
            if values.numel() == 1:
                self.output.bias.fill_(float(values.item()))
            elif values.numel() == self.output.out_channels:
                self.output.bias.copy_(values)
            else:
                raise ValueError("task-head bias shape")

    def forward(self, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.output(self.tower(p2))
        full = F.interpolate(raw, size=(NETWORK_H, CONTENT_W), mode="bilinear", align_corners=False)
        return raw, full[..., :CONTENT_H, :CONTENT_W]


GEOMETRY_WIDTHS = OrderedDict((
    ("depth_bin_logits", DEPTH_BINS + 1),
    ("depth_bin_residuals", DEPTH_BINS),
    ("physical_ray", 2),
    ("log_dimensions", 3),
    ("yaw", 2),
))


class GeometryHead(nn.Module):
    def __init__(self, priors: Mapping[str, Any]) -> None:
        super().__init__()
        tower = []
        for _ in range(4):
            tower.append(ConvGNReLU(256, 256))
        self.tower = nn.Sequential(*tower)
        self.outputs = nn.ModuleDict({name: nn.Conv2d(256, 2 * width, 3, padding=1)
                                      for name, width in GEOMETRY_WIDTHS.items()})
        self._initialize(priors)

    def _initialize(self, priors: Mapping[str, Any]) -> None:
        for module in self.tower.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        for output in self.outputs.values():
            nn.init.normal_(output.weight, std=0.001)
            nn.init.zeros_(output.bias)
        with torch.no_grad():
            depth_bias = torch.tensor(priors["depth_bin_log_probability_bias"], dtype=torch.float32)
            self.outputs["depth_bin_logits"].bias.copy_(depth_bias.reshape(-1))
            dims = torch.tensor([priors["mean_log_dimensions"][name] for name in CLASS_NAMES])
            self.outputs["log_dimensions"].bias.copy_(dims.reshape(-1))
            yaw = self.outputs["yaw"].bias.view(2, 2)
            yaw[:, 0].zero_()
            yaw[:, 1].fill_(1.0)
            self.outputs["depth_bin_residuals"].bias.zero_()
            self.outputs["physical_ray"].bias.zero_()

    def forward(self, features: Sequence[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        levels = []
        for feature in features:
            shared = self.tower(feature)
            level: dict[str, torch.Tensor] = {}
            for name, width in GEOMETRY_WIDTHS.items():
                value = self.outputs[name](shared)
                batch, _channels, height, width_px = value.shape
                level[name] = value.view(batch, 2, width, height, width_px).permute(0, 3, 4, 1, 2).reshape(
                    batch, height * width_px, 2, width)
            levels.append(level)
        return levels


class SplitFusionFCOS(nn.Module):
    def __init__(self, official: nn.Module, priors: Mapping[str, Any]) -> None:
        super().__init__()
        self.front = SevenChannelConvFront(official.backbone.body)
        self.tail = P2P7Tail(official)
        official_class = official.head.classification_head
        self.classification_tower = copy.deepcopy(official_class.conv)
        self.project_classifier = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        with torch.no_grad():
            self.project_classifier.weight[0].copy_(official_class.cls_logits.weight[3])
            self.project_classifier.bias[0].copy_(official_class.cls_logits.bias[3])
            self.project_classifier.weight[1].copy_(official_class.cls_logits.weight[1])
            self.project_classifier.bias[1].copy_(official_class.cls_logits.bias[1])
        self.regression_head = copy.deepcopy(official.head.regression_head)
        self.box_coder = copy.deepcopy(official.head.box_coder)
        self.anchor_generator = AnchorGenerator(tuple((size,) for size in ANCHOR_SIZES), ((1.0,),) * 6)
        self.semantic = P2TaskHead(3, priors["semantic_log_frequency_bias"])
        self.dense_depth = P2TaskHead(1, priors["dense_log1p_bias"])
        self.geometry = GeometryHead(priors)
        self.register_buffer("depth_edges_m", depth_edges(), persistent=True)
        self.score_thresh, self.nms_thresh = 0.02, 0.60
        self.topk_candidates, self.detections_per_img = 1000, 100

    def encode_front(self, x7: torch.Tensor) -> torch.Tensor:
        return self.front(x7)

    @staticmethod
    def transport_encode(c2: torch.Tensor, config: Mapping[str, Any] | None = None) -> torch.Tensor:
        if c2.dtype != torch.float32 or tuple(c2.shape[1:]) != (256, 112, 192):
            raise ValueError("noAE transport requires raw FP32 C2")
        return c2

    @staticmethod
    def transport_decode(payload: torch.Tensor, config: Mapping[str, Any] | None = None) -> torch.Tensor:
        if payload.dtype != torch.float32 or tuple(payload.shape[1:]) != (256, 112, 192):
            raise ValueError("noAE transport payload drift")
        return payload

    def _detection_heads(self, features: Sequence[torch.Tensor]) -> dict[str, Any]:
        cls_levels, box_levels, ctr_levels = [], [], []
        class_tower_levels, regression_tower_levels = [], []
        for feature in features:
            cls_feature = self.classification_tower(feature)
            reg_feature = self.regression_head.conv(feature)
            cls = self.project_classifier(cls_feature)
            box = F.relu(self.regression_head.bbox_reg(reg_feature))
            ctr = self.regression_head.bbox_ctrness(reg_feature)
            batch, _, height, width = cls.shape
            cls_levels.append(cls.permute(0, 2, 3, 1).reshape(batch, height * width, 2))
            box_levels.append(box.permute(0, 2, 3, 1).reshape(batch, height * width, 4))
            ctr_levels.append(ctr.permute(0, 2, 3, 1).reshape(batch, height * width, 1))
            class_tower_levels.append(cls_feature)
            regression_tower_levels.append(reg_feature)
        return {
            "cls_logits": torch.cat(cls_levels, dim=1),
            "bbox_regression": torch.cat(box_levels, dim=1),
            "bbox_ctrness": torch.cat(ctr_levels, dim=1),
            "per_level": {"cls_logits": cls_levels, "bbox_regression": box_levels, "bbox_ctrness": ctr_levels},
            "classification_tower": class_tower_levels,
            "regression_tower": regression_tower_levels,
        }

    def anchors(self, c2: torch.Tensor, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        placeholder = c2.new_empty((c2.shape[0], 1, NETWORK_H, CONTENT_W))
        image_list = ImageList(placeholder, [(NETWORK_H, CONTENT_W)] * c2.shape[0])
        return self.anchor_generator(image_list, list(features))

    def decode_tail(self, c2_hat: torch.Tensor, calibration: Any = None,
                    metadata: Any = None, *, dense: bool = True) -> dict[str, Any]:
        if c2_hat.dtype != torch.float32:
            raise ValueError("tail boundary must be FP32")
        features_by_name, cs = self.tail(c2_hat)
        features = list(features_by_name.values())
        detection = self._detection_heads(features)
        semantic_raw, semantic = self.semantic(features_by_name["p2"])
        result: dict[str, Any] = {
            "c2": c2_hat,
            "features": features_by_name,
            "resnet_features": cs,
            "detection": detection,
            "geometry": self.geometry(features),
            "semantic_logits_stride4": semantic_raw,
            "semantic_logits": semantic,
            "anchors": self.anchors(c2_hat, features),
        }
        if dense:
            dense_raw, dense_full = self.dense_depth(features_by_name["p2"])
            result["dense_depth_log1p_stride4"] = dense_raw
            result["dense_depth_log1p"] = dense_full
        return result

    def forward(self, x7: torch.Tensor, *, dense: bool = True) -> dict[str, Any]:
        c2 = self.encode_front(x7)
        payload = self.transport_encode(c2)
        c2_hat = self.transport_decode(payload)
        return self.decode_tail(c2_hat, dense=dense)

    def _decode_geometry(self, raw: Mapping[str, torch.Tensor], anchors: torch.Tensor,
                         point_indices: torch.Tensor, labels: torch.Tensor,
                         intrinsic: torch.Tensor, extrinsic: torch.Tensor) -> dict[str, torch.Tensor]:
        device = anchors.device
        row = torch.arange(len(point_indices), device=device)
        gathered = {name: value[point_indices, labels] for name, value in raw.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            logits = gathered["depth_bin_logits"].float()
            bins = logits.argmax(dim=1)
            in_range = bins < DEPTH_BINS
            safe_bins = bins.clamp(max=DEPTH_BINS - 1)
            edges = self.depth_edges_m.float().to(device)
            lower = edges[safe_bins]
            upper = edges[safe_bins + 1]
            residuals = 0.5 * torch.tanh(gathered["depth_bin_residuals"].float())
            selected_residual = residuals[row, safe_bins]
            zl, zu = torch.log1p(lower), torch.log1p(upper)
            log_depth = 0.5 * (zl + zu) + selected_residual * (zu - zl)
            depth = torch.where(in_range, torch.expm1(log_depth).clamp(0.0, 40.0),
                                torch.full_like(log_depth, 40.0))
            sizes = anchors[point_indices, 2] - anchors[point_indices, 0]
            centers = (anchors[point_indices, :2] + anchors[point_indices, 2:]) / 2
            uv = centers + sizes[:, None] * gathered["physical_ray"].float()
            k = intrinsic.float().to(device)
            local = torch.stack((
                depth,
                depth * (uv[:, 0] - k[0, 2]) / k[0, 0],
                depth * (k[1, 2] - uv[:, 1]) / k[1, 1],
            ), dim=1)
            homogeneous = torch.cat((local.double(), torch.ones(len(local), 1, device=device, dtype=torch.float64)), dim=1)
            world = (homogeneous @ extrinsic.to(device=device, dtype=torch.float64).T)[:, :3]
            dimensions = torch.exp(gathered["log_dimensions"].double())
            yaw = F.normalize(gathered["yaw"].float(), dim=1, eps=1e-6)
        return {"local_xyz": local, "world_xyz": world, "dimensions": dimensions,
                "yaw": yaw, "physical_uv": uv, "depth_bin": bins,
                "depth_residual": selected_residual, "depth": depth}

    def postprocess(self, outputs: Mapping[str, Any], calibrations: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        per = outputs["detection"]["per_level"]
        anchors_by_image = outputs["anchors"]
        geometry_levels = outputs["geometry"]
        results = []
        for image_index, calibration in enumerate(calibrations):
            image_boxes, image_scores, image_labels = [], [], []
            image_level, image_point, image_geometry = [], [], []
            split_anchors = list(anchors_by_image[image_index].split(
                [level.shape[1] for level in per["cls_logits"]]))
            for level_index, level_name in enumerate(LEVELS):
                cls = per["cls_logits"][level_index][image_index].float()
                ctr = per["bbox_ctrness"][level_index][image_index].float()
                scores = torch.sqrt(torch.sigmoid(cls) * torch.sigmoid(ctr)).flatten()
                keep = scores > self.score_thresh
                candidate = torch.where(keep)[0]
                scores = scores[keep]
                if candidate.numel() > self.topk_candidates:
                    scores, order = scores.topk(self.topk_candidates)
                    candidate = candidate[order]
                point = torch.div(candidate, 2, rounding_mode="floor")
                label = candidate % 2
                anchors = split_anchors[level_index]
                boxes = self.box_coder.decode(
                    per["bbox_regression"][level_index][image_index][point].float(), anchors[point].float())
                boxes = box_ops.clip_boxes_to_image(boxes, (CONTENT_H, CONTENT_W))
                raw_geometry = {name: value[image_index] for name, value in geometry_levels[level_index].items()}
                decoded = self._decode_geometry(raw_geometry, anchors, point, label,
                                                calibration["intrinsic"], calibration["extrinsic"])
                image_boxes.append(boxes); image_scores.append(scores); image_labels.append(label)
                image_level.append(torch.full_like(point, level_index)); image_point.append(point)
                image_geometry.append(decoded)
            boxes, scores, labels = map(torch.cat, (image_boxes, image_scores, image_labels))
            levels, points = torch.cat(image_level), torch.cat(image_point)
            geometry = {name: torch.cat([value[name] for value in image_geometry]) for name in image_geometry[0]}
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
            result = {"boxes": boxes[keep], "scores": scores[keep], "labels_internal": labels[keep],
                      "labels_canonical": labels[keep] + 1, "level_indices": levels[keep],
                      "point_indices": points[keep],
                      "candidate_identity": torch.stack((
                          torch.full_like(levels[keep], image_index), levels[keep], points[keep], labels[keep],
                      ), dim=1)}
            result.update({name: value[keep] for name, value in geometry.items()})
            results.append(result)
        return results


def build_model(priors: Mapping[str, Any], device: torch.device | None = None) -> tuple[SplitFusionFCOS, dict[str, Any]]:
    weights = FCOS_ResNet50_FPN_Weights.COCO_V1
    weight_path = Path("/home/shr_aisvcs/.cache/torch/hub/checkpoints") / Path(weights.url).name
    official = fcos_resnet50_fpn(weights=weights, progress=False, trainable_backbone_layers=5)
    resolved_weight_path = weight_path.resolve(strict=True)
    if resolved_weight_path not in _VERIFIED_WEIGHT_PATHS:
        if resolved_weight_path.stat().st_size != OFFICIAL_WEIGHTS_BYTES or sha256(resolved_weight_path) != OFFICIAL_WEIGHTS_SHA256:
            raise RuntimeError("official FCOS checkpoint byte/hash verification failed")
        _VERIFIED_WEIGHT_PATHS.add(resolved_weight_path)
    if official.head.classification_head.num_classes != 91:
        raise RuntimeError("complete official 91-output model was not loaded")
    source_classifier = official.head.classification_head.cls_logits
    source_report = {
        "complete_official_outputs": 91,
        "official_weight_path": str(weight_path),
        "coco_car_index": 3,
        "coco_person_index": 1,
        "coco_car_weight_sha256": tensor_hash(source_classifier.weight[3]),
        "coco_car_bias_sha256": tensor_hash(source_classifier.bias[3]),
        "coco_person_weight_sha256": tensor_hash(source_classifier.weight[1]),
        "coco_person_bias_sha256": tensor_hash(source_classifier.bias[1]),
    }
    model = SplitFusionFCOS(official, priors)
    source_report.update({
        "project_vehicle_weight_sha256": tensor_hash(model.project_classifier.weight[0]),
        "project_vehicle_bias_sha256": tensor_hash(model.project_classifier.bias[0]),
        "project_person_weight_sha256": tensor_hash(model.project_classifier.weight[1]),
        "project_person_bias_sha256": tensor_hash(model.project_classifier.bias[1]),
        "rgb_stem_exact": torch.equal(model.front.W_rgb, official.backbone.body.conv1.weight),
        "radar_stem_exact_zero": int(torch.count_nonzero(model.front.W_radar)) == 0,
    })
    if not all(source_report[name] for name in ("rgb_stem_exact", "radar_stem_exact_zero")):
        raise RuntimeError("stem transfer failure")
    if device is not None:
        model.to(device)
    return model, source_report


def configure_trainability(model: SplitFusionFCOS, epoch: int) -> dict[str, list[str]]:
    warmup = int(epoch) <= 3
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    frozen_prefixes = []
    if warmup:
        frozen_prefixes = [
            "front.W_rgb", "front.bn1", "front.layer1", "tail.layer2", "tail.layer3", "tail.layer4",
            "tail.official_fpn_p3_p7", "classification_tower", "regression_head",
        ]
        for name, parameter in model.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in frozen_prefixes):
                parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return {
        "trainable": [name for name, value in model.named_parameters() if value.requires_grad],
        "frozen": [name for name, value in model.named_parameters() if not value.requires_grad],
    }


def optimizer_parameter_groups(model: SplitFusionFCOS) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {"pretrained_backbone": [], "pretrained_fpn_heads": [], "new": []}
    for name, parameter in model.named_parameters():
        if name == "front.W_rgb" or name.startswith(("front.bn1", "front.layer1", "tail.layer2", "tail.layer3", "tail.layer4")):
            groups["pretrained_backbone"].append((name, parameter))
        elif name.startswith(("tail.official_fpn_p3_p7", "classification_tower", "regression_head")):
            groups["pretrained_fpn_heads"].append((name, parameter))
        else:
            groups["new"].append((name, parameter))
    if len({id(parameter) for values in groups.values() for _, parameter in values}) != len(list(model.parameters())):
        raise RuntimeError("optimizer parameter group overlap/omission")
    return groups


def parameter_inventory(model: SplitFusionFCOS) -> dict[str, Any]:
    groups = optimizer_parameter_groups(model)
    tensors = []
    for name, parameter in model.named_parameters():
        group = next(group for group, values in groups.items() if any(parameter is item for _, item in values))
        tensors.append({"name": name, "shape": list(parameter.shape), "parameters": parameter.numel(),
                        "group": group, "trainable": parameter.requires_grad, "sha256": tensor_hash(parameter)})
    buffers = [{"name": name, "shape": list(value.shape), "elements": value.numel(), "sha256": tensor_hash(value)}
               for name, value in model.named_buffers()]
    return {
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "parameter_tensors": len(tensors),
        "buffer_tensors": len(buffers),
        "groups": {name: {"tensors": len(values), "parameters": sum(value.numel() for _, value in values)}
                   for name, values in groups.items()},
        "tensors": tensors,
        "buffers": buffers,
    }
