from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


OBJECT_CLASS_NAMES = ("vehicle", "person")
OBJECT_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(OBJECT_CLASS_NAMES)}
OBJECT_HEATMAP_CHANNELS = len(OBJECT_CLASS_NAMES)
OBJECT_REG_CHANNELS = 10
OBJECT_OUTPUT_CHANNELS = OBJECT_HEATMAP_CHANNELS + OBJECT_REG_CHANNELS
REG_LOCAL_XYZ = slice(0, 3)
REG_DIMS = slice(3, 6)
REG_YAW = slice(6, 8)
REG_PARKED = 8
REG_RADAR_SUPPORT = 9
# Optional 2D-box (per-instance, normalized width/height) regression appended after
# the base channels when predict_bbox2d=True. Center comes from the heatmap peak.
OBJECT_REG_CHANNELS_BBOX = 12
REG_BBOX_WH = slice(10, 12)


def object_reg_channels(predict_bbox2d: bool = False) -> int:
    return OBJECT_REG_CHANNELS_BBOX if predict_bbox2d else OBJECT_REG_CHANNELS


def parse_matrix(value: str) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        arr = np.asarray(json.loads(value), dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return arr


def transform_point(matrix: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    point = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    return (matrix @ point)[:3]


def load_object_boxes(path: Path) -> Dict[str, List[Dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def valid_localization_objects(
    rows: Sequence[Dict[str, str]],
    *,
    image_width: int,
    image_height: int,
    min_area_px: float,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    max_distance_m: float | None = None,
) -> List[Dict[str, float]]:
    class_to_index = {str(name): idx for idx, name in enumerate(object_class_names)}
    objects: List[Dict[str, float]] = []
    for row in rows:
        label = str(row.get("label", ""))
        if label not in class_to_index or row.get("gt_source") != "actor":
            continue
        if row.get("object_sensor_x", "") == "" or row.get("object_world_x", "") == "":
            continue
        area = _float(row, "gt_bbox_area_px")
        if area < float(min_area_px):
            continue
        # Operating-range gate: a sensor cannot reliably detect distant objects, so beyond
        # this range they are neither trained as targets nor scored at eval. Removes the
        # unfair >50 m "ghost" GT that dominated recall (~63% of GT was >50 m).
        if max_distance_m is not None and _float(row, "gt_distance_m") > float(max_distance_m):
            continue
        cx = _float(row, "gt_center_x")
        cy = _float(row, "gt_center_y")
        if not (0.0 <= cx < float(image_width) and 0.0 <= cy < float(image_height)):
            continue
        yaw_rad = math.radians(_float(row, "object_yaw_deg"))
        objects.append(
            {
                "class_index": float(class_to_index[label]),
                "class_name": label,
                "center_x": cx,
                "center_y": cy,
                "bbox_w": _float(row, "gt_bbox_w"),
                "bbox_h": _float(row, "gt_bbox_h"),
                "area": area,
                "local_x": _float(row, "object_sensor_x"),
                "local_y": _float(row, "object_sensor_y"),
                "local_z": _float(row, "object_sensor_z"),
                "world_x": _float(row, "object_world_x"),
                "world_y": _float(row, "object_world_y"),
                "world_z": _float(row, "object_world_z"),
                "size_x": max(0.01, _float(row, "gt_size_x_m")),
                "size_y": max(0.01, _float(row, "gt_size_y_m")),
                "size_z": max(0.01, _float(row, "gt_size_z_m")),
                "yaw_sin": math.sin(yaw_rad),
                "yaw_cos": math.cos(yaw_rad),
                "parked": float(_float(row, "parked_label") >= 0.5),
                "radar_support": float(_float(row, "radar_support_points") > 0.0),
            }
        )
    return objects


def valid_vehicle_objects(
    rows: Sequence[Dict[str, str]],
    *,
    image_width: int,
    image_height: int,
    min_area_px: float,
) -> List[Dict[str, float]]:
    """Compatibility wrapper for older vehicle-only experiments."""

    return valid_localization_objects(
        rows,
        image_width=image_width,
        image_height=image_height,
        min_area_px=min_area_px,
        object_class_names=("vehicle",),
    )


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius: int) -> None:
    radius = max(0, int(radius))
    x0 = max(0, int(round(cx)) - radius)
    y0 = max(0, int(round(cy)) - radius)
    x1 = min(heatmap.shape[1], int(round(cx)) + radius + 1)
    y1 = min(heatmap.shape[0], int(round(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    if radius <= 0:
        heatmap[int(round(cy)), int(round(cx))] = 1.0
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sigma = max(1.0, float(radius) / 2.0)
    values = np.exp(-((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], values.astype(np.float32))


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet Gaussian radius: largest radius keeping IoU >= min_overlap for a
    box of the given size. Scales the heatmap footprint to each object's size."""
    h, w = float(height), float(width)
    a1, b1, c1 = 1.0, (h + w), w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 - math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / (2 * a1)
    a2, b2, c2 = 4.0, 2.0 * (h + w), (1.0 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / (2 * a2)
    a3, b3, c3 = 4.0 * min_overlap, -2.0 * min_overlap * (h + w), (min_overlap - 1.0) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def build_object_targets(
    *,
    objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    heatmap_radius_px: int,
    max_objects: int,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    predict_bbox2d: bool = False,
    adaptive_heatmap_radius: bool = False,
) -> Dict[str, torch.Tensor]:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_width, original_height = int(original_size[0]), int(original_size[1])
    sx = input_width / max(1.0, float(original_width))
    sy = input_height / max(1.0, float(original_height))
    reg_channels = object_reg_channels(predict_bbox2d)
    object_class_count = max(1, len(tuple(object_class_names)))
    heatmap = np.zeros((object_class_count, input_height, input_width), dtype=np.float32)
    regression = np.zeros((reg_channels, input_height, input_width), dtype=np.float32)
    reg_mask = np.zeros((1, input_height, input_width), dtype=np.float32)
    gt_objects = np.zeros((int(max_objects), 9), dtype=np.float32)
    gt_class_indices = np.zeros((int(max_objects),), dtype=np.int64)
    gt_count = 0
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        class_index = int(obj.get("class_index", 0))
        if class_index < 0 or class_index >= object_class_count:
            continue
        cx = float(obj["center_x"]) * sx
        cy = float(obj["center_y"]) * sy
        ix = int(round(cx))
        iy = int(round(cy))
        if ix < 0 or iy < 0 or ix >= input_width or iy >= input_height:
            continue
        if adaptive_heatmap_radius:
            # Size-matched footprint: scale the Gaussian radius to this object's box
            # (in input-image px). Far/small objects get a tight peak; near/large
            # ones get a wider footprint -> better-calibrated positives for recall.
            bw_in = float(obj.get("bbox_w", 0.0)) * sx
            bh_in = float(obj.get("bbox_h", 0.0)) * sy
            radius = int(max(2, round(gaussian_radius(bh_in, bw_in))))
        else:
            radius = int(heatmap_radius_px)
        draw_gaussian(heatmap[class_index], cx, cy, radius)
        # The gaussian is evaluated at integer pixel coordinates; with a sub-pixel
        # (cx, cy) the peak pixel only reaches exp(-d^2/(2 sigma^2)) < 1.0. The
        # focal heatmap loss treats positives via target == 1.0, so without this
        # the previous run had pos_count == 0 every batch and the center head
        # never learned (learned_object_f1 = 0).
        heatmap[class_index, iy, ix] = 1.0
        if reg_mask[0, iy, ix] < 0.5:
            values = [
                obj["local_x"],
                obj["local_y"],
                obj["local_z"],
                obj["size_x"],
                obj["size_y"],
                obj["size_z"],
                obj["yaw_sin"],
                obj["yaw_cos"],
                obj["parked"],
                obj["radar_support"],
            ]
            if predict_bbox2d:
                # Normalized 2D-box width/height in input-image fraction (center = peak).
                values.append(float(obj.get("bbox_w", 0.0)) * sx / max(1.0, float(input_width)))
                values.append(float(obj.get("bbox_h", 0.0)) * sy / max(1.0, float(input_height)))
            regression[:, iy, ix] = np.array(values, dtype=np.float32)
            reg_mask[0, iy, ix] = 1.0
        if gt_count < int(max_objects):
            gt_objects[gt_count] = np.array(
                [
                    obj["world_x"],
                    obj["world_y"],
                    obj["world_z"],
                    obj["size_x"],
                    obj["size_y"],
                    obj["size_z"],
                    obj["yaw_sin"],
                    obj["yaw_cos"],
                    obj["parked"],
                ],
                dtype=np.float32,
            )
            gt_class_indices[gt_count] = class_index
            gt_count += 1
    if gt_count > 0:
        assert float(heatmap.max()) >= 0.999, (
            "object center heatmap target has no peak >= 1.0 despite gt_count > 0; "
            "focal loss positive count would be zero (learned_object_f1 = 0 regression)."
        )
    return {
        "center_heatmap": torch.from_numpy(heatmap),
        "regression": torch.from_numpy(regression),
        "regression_mask": torch.from_numpy(reg_mask),
        "gt_objects": torch.from_numpy(gt_objects),
        "gt_class_indices": torch.from_numpy(gt_class_indices),
        "gt_count": torch.tensor(gt_count, dtype=torch.long),
    }


def focal_heatmap_loss(
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
    """CornerNet-style focal heatmap loss.

    With ``pos_weight=None`` this is bit-identical to the original pooled form.

    With ``class_balanced=True`` the POSITIVE term becomes a macro-average over the
    object classes that actually have positives in this batch, and ``pos_weight`` is
    renormalised to mean 1.0 *within each class independently*. Two consequences that
    are the whole point of the change:

      * vehicle cell count can no longer dominate person learning - each class
        contributes one equally-weighted per-positive mean, regardless of how many
        cells it owns;
      * the positive-loss budget is fixed - both the pooled and the macro-averaged
        form are a "mean focal loss per positive cell", so reweighting can only
        redistribute gradient, never inflate it. A handful of far/small objects
        cannot take over training.

    The BACKGROUND (negative) term is left exactly as it was, including its original
    ``pos_count`` denominator, so the positive:negative balance is not silently moved.
    Classes with zero positives in a batch are skipped safely and contribute nothing.
    """
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.ge(1.0 - 1e-3).to(logits.dtype)
    neg = (1.0 - pos).to(logits.dtype)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg
    pos_count = pos.sum().clamp(min=1.0)

    if pos_weight is None and not class_balanced:
        return (pos_loss.sum() + neg_loss.sum()) / pos_count

    cap = float(weight_cap)
    if pos_weight is None:
        w = torch.ones_like(pos)
    else:
        w = pos_weight.to(logits.dtype).expand_as(pos).clamp(min=0.0, max=cap)

    num_classes = int(pos.shape[1])
    per_class_mean: List[torch.Tensor] = []
    per_class_count: List[torch.Tensor] = []
    max_w_seen = 0.0
    for c in range(num_classes):
        pos_c = pos[:, c]
        n_c = pos_c.sum()
        if float(n_c.detach().item()) < 0.5:
            # No positives for this class in this batch: skip it entirely. Never
            # divide by zero, never fabricate a gradient for an absent class.
            continue
        # Renormalise this class's weights to mean 1.0 over ITS OWN positives.
        w_c = w[:, c] * pos_c
        w_c = w_c * (n_c / w_c.sum().clamp(min=1e-6))
        per_class_mean.append((pos_loss[:, c] * w_c).sum() / n_c)
        per_class_count.append(n_c)
        if stats is not None:
            max_w_seen = max(max_w_seen, float(w_c.max().detach().item()))
            stats[f"pos_mean_w_class{c}"] = float((w_c.sum() / n_c).detach().item())
            stats[f"pos_count_class{c}"] = float(n_c.detach().item())
            stats[f"pos_mean_loss_class{c}"] = float(per_class_mean[-1].detach().item())

    if not per_class_mean:
        pos_term = pos_loss.sum() * 0.0
    elif class_balanced:
        # Macro-average: one equally-weighted vote per class present.
        pos_term = torch.stack(per_class_mean).mean()
    else:
        # Pooled (cell-count-weighted) mean, i.e. the original balance with weights.
        counts = torch.stack(per_class_count)
        pos_term = (torch.stack(per_class_mean) * counts).sum() / counts.sum().clamp(min=1.0)

    if stats is not None:
        stats["pos_max_weight"] = max_w_seen
        stats["pos_classes_present"] = float(len(per_class_mean))

    return pos_term + neg_loss.sum() / pos_count


def multitask_object_loss(outputs: torch.Tensor, targets: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    heatmap = targets["center_heatmap"].to(outputs.device)
    heatmap_channels = int(heatmap.shape[1])
    center_logits = outputs[:, :heatmap_channels]
    regs = outputs[:, heatmap_channels:]
    if int(regs.shape[1]) not in (OBJECT_REG_CHANNELS, OBJECT_REG_CHANNELS_BBOX):
        raise ValueError(
            f"Object head regression channels={int(regs.shape[1])}, expected "
            f"{OBJECT_REG_CHANNELS} or {OBJECT_REG_CHANNELS_BBOX}. "
            f"Output channels={int(outputs.shape[1])}, heatmap channels={heatmap_channels}."
        )
    reg_target = targets["regression"].to(outputs.device)
    has_bbox2d = int(regs.shape[1]) >= OBJECT_REG_CHANNELS_BBOX and int(reg_target.shape[1]) >= OBJECT_REG_CHANNELS_BBOX
    reg_mask = targets["regression_mask"].to(outputs.device)

    # Bounded range/size positive weighting, built from targets that are already on
    # device - no new dataloader field and no new target tensor.
    #   local_x/local_y (REG_LOCAL_XYZ[0:2]) are raw METRES  -> the 20-40 m band;
    #   REG_BBOX_WH are input-image FRACTIONS                -> the small-object test.
    # Weights are capped per element and renormalised to mean 1.0 within each class
    # by focal_heatmap_loss, so the positive-loss budget is fixed.
    pos_weight = None
    center_stats: Dict[str, float] = {}
    use_pos_weight = bool(weights.get("pos_weight_enable", False))
    class_balanced = bool(weights.get("class_balanced_center", False))
    if use_pos_weight:
        m = reg_mask  # (B,1,H,W), 1.0 at positive cells
        if has_bbox2d:
            gw = reg_target[:, REG_BBOX_WH.start: REG_BBOX_WH.start + 1].clamp(min=0.0)
            gh = reg_target[:, REG_BBOX_WH.start + 1: REG_BBOX_WH.start + 2].clamp(min=0.0)
            small = ((gw * gh) < float(weights.get("small_area_frac", 0.003))).to(reg_target.dtype) * m
        else:
            small = torch.zeros_like(m)
        rng = torch.linalg.vector_norm(reg_target[:, 0:2], dim=1, keepdim=True)
        lo = float(weights.get("range_band_lo_m", 20.0))
        hi = float(weights.get("range_band_hi_m", 40.0))
        band = ((rng >= lo) & (rng < hi)).to(reg_target.dtype) * m
        pos_weight = (
            (1.0 + float(weights.get("small_gain", 1.0)) * small)
            * (1.0 + float(weights.get("range_gain", 0.8)) * band)
        )
    center_loss = focal_heatmap_loss(
        center_logits,
        heatmap,
        pos_weight=pos_weight,
        weight_cap=float(weights.get("pos_weight_cap", 4.0)),
        class_balanced=class_balanced,
        stats=center_stats,
    )
    denom = reg_mask.sum().clamp(min=1.0)
    mask = reg_mask.expand_as(regs)
    loc_loss = F.smooth_l1_loss(regs[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ], reg_target[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ], reduction="sum") / denom
    dim_loss = F.smooth_l1_loss(regs[:, REG_DIMS] * mask[:, REG_DIMS], reg_target[:, REG_DIMS] * mask[:, REG_DIMS], reduction="sum") / denom
    yaw_pred = F.normalize(regs[:, REG_YAW], dim=1)
    yaw_loss = F.smooth_l1_loss(yaw_pred * mask[:, REG_YAW], reg_target[:, REG_YAW] * mask[:, REG_YAW], reduction="sum") / denom
    parked_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_PARKED : REG_PARKED + 1],
        reg_target[:, REG_PARKED : REG_PARKED + 1],
        weight=reg_mask,
        reduction="sum",
    ) / denom
    radar_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        reg_target[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        weight=reg_mask,
        reduction="sum",
    ) / denom
    if has_bbox2d:
        # Scale-invariant GIoU loss on the 2D box. Pred/GT boxes share the cell
        # center (center comes from the heatmap), so GIoU here supervises size.
        # softplus keeps predicted w/h positive; gives real gradient on the tiny
        # far-object boxes that smooth-L1 ignored.
        eps = 1e-6
        pw = F.softplus(regs[:, REG_BBOX_WH.start: REG_BBOX_WH.start + 1])
        ph = F.softplus(regs[:, REG_BBOX_WH.start + 1: REG_BBOX_WH.start + 2])
        gw = reg_target[:, REG_BBOX_WH.start: REG_BBOX_WH.start + 1].clamp(min=0.0)
        gh = reg_target[:, REG_BBOX_WH.start + 1: REG_BBOX_WH.start + 2].clamp(min=0.0)
        inter = torch.min(pw, gw) * torch.min(ph, gh)
        union = pw * ph + gw * gh - inter + eps
        iou = inter / union
        enclose = torch.max(pw, gw) * torch.max(ph, gh) + eps
        giou = iou - (enclose - union) / enclose
        bbox_loss = ((1.0 - giou) * reg_mask).sum() / denom
    else:
        bbox_loss = center_loss.new_zeros(())
    total = (
        float(weights.get("center", 1.0)) * center_loss
        + float(weights.get("location", 0.05)) * loc_loss
        + float(weights.get("dimensions", 0.2)) * dim_loss
        + float(weights.get("yaw", 0.05)) * yaw_loss
        + float(weights.get("parked", 0.2)) * parked_loss
        + float(weights.get("radar_support", 0.1)) * radar_loss
        + float(weights.get("bbox2d", 1.0)) * bbox_loss
    )
    parts = {
        "center_loss": float(center_loss.detach().item()),
        "loc_loss": float(loc_loss.detach().item()),
        "dim_loss": float(dim_loss.detach().item()),
        "yaw_loss": float(yaw_loss.detach().item()),
        "parked_loss": float(parked_loss.detach().item()),
        "radar_support_loss": float(radar_loss.detach().item()),
        "bbox2d_loss": float(bbox_loss.detach().item()),
    }
    parts.update(center_stats)
    return total, parts


def decode_objects(
    object_output: torch.Tensor,
    *,
    camera_matrix: np.ndarray,
    topk: int,
    score_threshold: float,
    nms_radius_px: int,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    predict_bbox2d: bool = False,
) -> List[Dict[str, float]]:
    if object_output.ndim == 4:
        object_output = object_output[0]
    heatmap_channels = max(1, int(object_output.shape[0]) - object_reg_channels(predict_bbox2d))
    center = torch.sigmoid(object_output[:heatmap_channels]).detach().cpu()
    regs = object_output[heatmap_channels:].detach().cpu().numpy()
    flat = center.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    if k <= 0:
        return []
    scores, indices = torch.topk(flat, k=k)
    height, width = int(center.shape[1]), int(center.shape[2])
    occupied = np.zeros((heatmap_channels, height, width), dtype=bool)
    predictions: List[Dict[str, float]] = []
    for score_t, index_t in zip(scores, indices):
        score = float(score_t.item())
        if score < float(score_threshold):
            continue
        idx = int(index_t.item())
        class_index, rem = divmod(idx, height * width)
        y, x = divmod(rem, width)
        y0, y1 = max(0, y - int(nms_radius_px)), min(height, y + int(nms_radius_px) + 1)
        x0, x1 = max(0, x - int(nms_radius_px)), min(width, x + int(nms_radius_px) + 1)
        if occupied[class_index, y0:y1, x0:x1].any():
            continue
        occupied[class_index, y0:y1, x0:x1] = True
        local = regs[REG_LOCAL_XYZ, y, x]
        dims = np.maximum(regs[REG_DIMS, y, x], 0.0)
        yaw_sin, yaw_cos = regs[REG_YAW, y, x]
        norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        world = transform_point(camera_matrix, local)
        class_name = (
            str(object_class_names[class_index])
            if class_index < len(object_class_names)
            else f"object_{class_index}"
        )
        bbox2d = {}
        if predict_bbox2d and regs.shape[0] >= OBJECT_REG_CHANNELS_BBOX:
            # softplus to match the GIoU-trained size encoding (stable for large logits).
            def _softplus(v):
                return float(np.log1p(np.exp(-abs(v))) + max(v, 0.0))
            bw = _softplus(regs[REG_BBOX_WH.start, y, x]) * float(width)
            bh = _softplus(regs[REG_BBOX_WH.start + 1, y, x]) * float(height)
            bbox2d = {
                "bbox_w_px": bw,
                "bbox_h_px": bh,
                "bbox_x0": float(x) - bw / 2.0,
                "bbox_y0": float(y) - bh / 2.0,
                "bbox_x1": float(x) + bw / 2.0,
                "bbox_y1": float(y) + bh / 2.0,
            }
        predictions.append(
            {
                "class_index": float(class_index),
                "class_name": class_name,
                "score": score,
                "center_x_px": float(x),
                "center_y_px": float(y),
                **bbox2d,
                "local_x": float(local[0]),
                "local_y": float(local[1]),
                "local_z": float(local[2]),
                "world_x": float(world[0]),
                "world_y": float(world[1]),
                "world_z": float(world[2]),
                "size_x": float(dims[0]),
                "size_y": float(dims[1]),
                "size_z": float(dims[2]),
                "yaw_sin": float(yaw_sin / norm),
                "yaw_cos": float(yaw_cos / norm),
                "parked_score": float(torch.sigmoid(object_output[heatmap_channels + REG_PARKED, y, x]).item()),
                "radar_support_score": float(torch.sigmoid(object_output[heatmap_channels + REG_RADAR_SUPPORT, y, x]).item()),
            }
        )
    return predictions


def greedy_match_predictions(
    predictions: Sequence[Dict[str, float]],
    gt_objects: Sequence[Dict[str, float]],
    *,
    max_distance_m: float,
    class_aware: bool = True,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(gt_objects):
            if bool(class_aware) and str(pred.get("class_name", "")) != str(gt.get("class_name", "")):
                continue
            dist = float(np.hypot(float(pred["world_x"]) - float(gt["world_x"]), float(pred["world_y"]) - float(gt["world_y"])))
            if dist <= float(max_distance_m):
                candidates.append((dist, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0])
    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, dist))
    return matches
