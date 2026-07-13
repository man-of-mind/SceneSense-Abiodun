from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from .common import (
    CLASS_NAMES,
    class_iou_from_confusion,
    load_config,
    read_manifest,
    save_json,
    set_reproducible_seeds,
    setup_logger,
    update_confusion,
    utc_iso,
)
from .model import OBJECT_HEAD_CHANNELS, build_fusion_lraspp, build_multitask_fusion_lraspp
from .object_targets import (
    OBJECT_CLASS_NAMES,
    build_object_targets,
    load_object_boxes,
    multitask_object_loss,
    object_reg_channels,
    valid_localization_objects,
)


class FusionPoleMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        rows: List[Dict[str, str]],
        object_rows: Dict[str, List[Dict[str, str]]],
        input_size: Tuple[int, int],
        object_cfg: Dict,
        augment_strength: str = "off",
        geometric_augment: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.object_rows = object_rows
        self.input_width, self.input_height = input_size
        self.object_cfg = object_cfg
        self.object_class_names = tuple(object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
        self.predict_bbox2d = bool(object_cfg.get("predict_bbox2d", False))
        self.adaptive_heatmap_radius = bool(object_cfg.get("adaptive_heatmap_radius", False))
        self.augment_strength = str(augment_strength)
        self.geometric_augment = bool(geometric_augment)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: Image.Image, mask: Image.Image, radar: np.ndarray) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        if self.augment_strength == "off":
            return image, mask, radar
        if self.geometric_augment and self.augment_strength in {"medium", "strong"}:
            image, mask, radar = self._augment_geometric(image, mask, radar)
        if self.augment_strength in {"light", "medium", "strong"}:
            if np.random.rand() < 0.35:
                if self.augment_strength == "light":
                    factor = float(np.random.uniform(0.85, 1.15))
                elif self.augment_strength == "medium":
                    factor = float(np.random.uniform(0.75, 1.25))
                else:
                    factor = float(np.random.uniform(0.65, 1.35))
                image = ImageEnhance.Brightness(image).enhance(factor)
            if np.random.rand() < 0.35:
                if self.augment_strength == "light":
                    factor = float(np.random.uniform(0.9, 1.1))
                elif self.augment_strength == "medium":
                    factor = float(np.random.uniform(0.8, 1.2))
                else:
                    factor = float(np.random.uniform(0.7, 1.3))
                image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask, radar

    def _augment_geometric(
        self,
        image: Image.Image,
        mask: Image.Image,
        radar: np.ndarray,
    ) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        # Only enable this for segmentation-only trials. Object-box targets are
        # built from original image geometry below, so geometric augmentation
        # would misalign the learned localization head.
        if np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            radar = np.flip(radar, axis=2).copy()

        if np.random.rand() < 0.75:
            width, height = image.size
            crop_scale = float(np.random.uniform(0.65, 1.0))
            crop_w = max(16, int(width * crop_scale))
            crop_h = max(16, int(height * crop_scale))
            if crop_w < width or crop_h < height:
                left = int(np.random.randint(0, max(1, width - crop_w + 1)))
                top = int(np.random.randint(0, max(1, height - crop_h + 1)))
                box = (left, top, left + crop_w, top + crop_h)
                image = image.crop(box)
                mask = mask.crop(box)
                radar_h, radar_w = radar.shape[1], radar.shape[2]
                r_left = int(round(left * radar_w / width))
                r_top = int(round(top * radar_h / height))
                r_right = int(round((left + crop_w) * radar_w / width))
                r_bottom = int(round((top + crop_h) * radar_h / height))
                r_right = max(r_left + 1, min(r_right, radar_w))
                r_bottom = max(r_top + 1, min(r_bottom, radar_h))
                radar = radar[:, r_top:r_bottom, r_left:r_right].copy()
        return image, mask, radar

    def _load_radar(self, row: Dict[str, str]) -> np.ndarray:
        radar_path = self.dataset_dir / row["radar_tensor_path"]
        payload = np.load(radar_path)
        try:
            if isinstance(payload, np.lib.npyio.NpzFile):
                radar = payload["radar"].astype(np.float32)
            else:
                radar = np.asarray(payload, dtype=np.float32)
        finally:
            if hasattr(payload, "close"):
                payload.close()
        if radar.ndim != 3:
            raise ValueError(f"Expected radar tensor [C,H,W] at {radar_path}, got {radar.shape}")
        return radar

    def _resize_radar(self, radar: np.ndarray) -> np.ndarray:
        if radar.shape[2] == self.input_width and radar.shape[1] == self.input_height:
            return radar
        resized = []
        for channel_idx, channel in enumerate(radar):
            interpolation = cv2.INTER_NEAREST if channel_idx == 0 else cv2.INTER_LINEAR
            resized.append(cv2.resize(channel, (self.input_width, self.input_height), interpolation=interpolation))
        return np.stack(resized, axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        original_width, original_height = image.size
        mask = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        radar = self._load_radar(row)
        image, mask, radar = self._augment(image, mask, radar)
        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        radar = self._resize_radar(radar)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        image_tensor = (image_tensor - self.rgb_mean) / self.rgb_std
        radar_tensor = torch.from_numpy(np.ascontiguousarray(radar)).to(torch.float32)
        fused = torch.cat([image_tensor, radar_tensor], dim=0)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=original_width,
            image_height=original_height,
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 24.0)),
            object_class_names=self.object_class_names,
            max_distance_m=(float(self.object_cfg["max_gt_distance_m"])
                            if self.object_cfg.get("max_gt_distance_m") not in (None, "", 0)
                            else None),
        )
        object_targets = build_object_targets(
            objects=objects,
            original_size=(original_width, original_height),
            input_size=(self.input_width, self.input_height),
            heatmap_radius_px=int(self.object_cfg.get("heatmap_radius_px", 2)),
            max_objects=int(self.object_cfg.get("max_objects_per_frame", 64)),
            object_class_names=self.object_class_names,
            predict_bbox2d=self.predict_bbox2d,
            adaptive_heatmap_radius=self.adaptive_heatmap_radius,
        )
        return fused, mask_tensor, object_targets


def split_rows(manifest_rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    splits = {"train": [], "val": [], "test": []}
    for row in manifest_rows:
        split = row.get("split", "train")
        splits.setdefault(split, []).append(row)
    return splits


def save_split_files(exp_dir: Path, splits: Dict[str, List[Dict[str, str]]]) -> None:
    split_dir = exp_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (split_dir / f"{split}.txt").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(str(row["sample_id"]) + "\n")


def _freeze_batch_norm(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
            for param in child.parameters():
                param.requires_grad = False


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)


def _eval_frozen_modules(model: torch.nn.Module, *, freeze_backbone: bool, freeze_classifier: bool, freeze_object_head: bool) -> None:
    if freeze_backbone:
        model.backbone.eval()
    if freeze_classifier:
        model.classifier.eval()
    if freeze_object_head:
        model.object_head.eval()


def _count_parameters(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(param.numel()) for param in module.parameters())
    trainable = sum(int(param.numel()) for param in module.parameters() if param.requires_grad)
    return total, trainable


def _checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise ValueError("Checkpoint did not contain a state_dict.")


def _load_object_head_checkpoint(model: torch.nn.Module, checkpoint_path: str, *, device: torch.device) -> Dict[str, int]:
    if not checkpoint_path:
        return {"loaded": 0, "skipped": 0}
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Object-head checkpoint not found: {path}")
    source = _checkpoint_state_dict(torch.load(path, map_location=device))
    current = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped = 0
    partial = 0
    for key, tensor in source.items():
        key2 = key[7:] if str(key).startswith("module.") else str(key)
        if not key2.startswith("object_head."):
            continue
        if key2 not in current:
            skipped += 1
            continue
        cur = current[key2]
        if tuple(cur.shape) == tuple(tensor.shape):
            compatible[key2] = tensor
        elif (
            cur.ndim == tensor.ndim
            and cur.shape[0] > tensor.shape[0]
            and tuple(cur.shape[1:]) == tuple(tensor.shape[1:])
        ):
            # Output channels grew (e.g. appended 2D-box channels). Copy the
            # overlapping leading channels (heatmap + existing regression, same
            # order/meaning) and keep the new channels' fresh init. Preserves the
            # warm-started detection/regression instead of cold-starting the layer.
            merged = cur.clone()
            merged[: tensor.shape[0]] = tensor.to(merged.dtype)
            compatible[key2] = merged
            partial += 1
        else:
            skipped += 1
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped, "partial": partial}


def _move_object_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    positives = gt_sorted.sum()
    intersection = positives - gt_sorted.float().cumsum(0)
    union = positives + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss for segmentation IoU optimization."""
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels_flat = labels.reshape(-1)
    losses: List[torch.Tensor] = []
    for class_idx in range(num_classes):
        foreground = (labels_flat == class_idx).to(probs_flat.dtype)
        if foreground.sum() <= 0:
            continue
        class_prob = probs_flat[:, class_idx]
        errors = (foreground - class_prob).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        foreground_sorted = foreground[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _lr_multiplier(epoch: int, max_epochs: int, scheduler: str, warmup_epochs: int, min_lr_ratio: float, poly_power: float) -> float:
    scheduler = str(scheduler or "none").lower()
    if scheduler in {"none", "off", "constant"}:
        return 1.0
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max(1e-3, float(epoch + 1) / float(warmup_epochs))
    decay_epochs = max(1, max_epochs - max(0, warmup_epochs))
    progress = min(1.0, max(0.0, float(epoch - warmup_epochs + 1) / float(decay_epochs)))
    if scheduler == "cosine":
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    if scheduler == "poly":
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * ((1.0 - progress) ** poly_power))
    raise ValueError(f"Unsupported lr_scheduler {scheduler!r}; use none, cosine, or poly.")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lrs: List[float], multiplier: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(multiplier)


def compute_losses(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    masks: torch.Tensor,
    object_targets: Dict[str, torch.Tensor],
    num_classes: int,
    loss_weights: Dict[str, float],
    class_weights: Optional[torch.Tensor] = None,
    lovasz_weight: float = 0.0,
    teacher: Optional[torch.nn.Module] = None,
    distill_weight: float = 0.0,
    distill_temp: float = 2.0,
    feature_drop_fraction: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    outputs = model(tensors, feature_drop_fraction=feature_drop_fraction)
    logits = outputs["out"]
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    ce_loss = F.cross_entropy(logits, masks, weight=class_weights)
    lovasz_loss = lovasz_softmax_loss(logits.float(), masks) if float(lovasz_weight) > 0.0 else logits.new_zeros(())
    seg_loss = ce_loss + float(lovasz_weight) * lovasz_loss
    # Seg-preservation distillation: anchor the student's seg logits to a frozen
    # teacher (the seg-only model) so a partially-unfrozen backbone cannot drift
    # segmentation while it adapts for localization.
    if teacher is not None and float(distill_weight) > 0.0:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            t_out = teacher(tensors.float())
            t_logits = t_out["out"] if isinstance(t_out, dict) else t_out
            if t_logits.shape[-2:] != logits.shape[-2:]:
                t_logits = F.interpolate(t_logits.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
        T = max(1e-3, float(distill_temp))
        s_log = F.log_softmax(logits.float() / T, dim=1)
        t_prob = F.softmax(t_logits.float() / T, dim=1)
        distill_loss = F.kl_div(s_log, t_prob, reduction="none").sum(dim=1).mean() * (T * T)
        seg_loss = seg_loss + float(distill_weight) * distill_loss
    else:
        distill_loss = seg_loss.detach().new_zeros(())
    # Run regression head losses in FP32 even when the surrounding context is AMP.
    # Smooth-L1 / BCE on small-magnitude regression targets are unstable in FP16
    # and can silently zero gradients into the object head.
    object_total_weight = float(loss_weights.get("object_total", 1.0))
    if object_total_weight > 0.0:
        # Run regression head losses in FP32 even when the surrounding context is AMP.
        # Smooth-L1 / BCE on small-magnitude regression targets are unstable in FP16
        # and can silently zero gradients into the object head.
        with torch.cuda.amp.autocast(enabled=False):
            object_logits_fp32 = outputs["object"].float()
            object_loss, object_parts = multitask_object_loss(
                object_logits_fp32, object_targets, loss_weights.get("object", {})
            )
    else:
        object_loss = seg_loss.detach().new_zeros(())
        object_parts = {}
    total = float(loss_weights.get("segmentation", 1.0)) * seg_loss + object_total_weight * object_loss
    parts = {
        "seg_loss": float(seg_loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "lovasz_loss": float(lovasz_loss.detach().item()),
        "distill_loss": float(distill_loss.detach().item()),
        "object_loss": float(object_loss.detach().item()),
        **object_parts,
    }
    return total, parts, logits


def _deep_merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _selection_score(metrics: Dict[str, float], mode: str) -> float:
    mode = str(mode or "default").lower()
    miou = float(metrics.get("miou", 0.0))
    vehicle_iou = float(metrics.get("vehicle_iou", miou))
    person_iou = float(metrics.get("person_iou", miou))
    loc_loss = float(metrics.get("loc_loss", 0.0))
    dim_loss = float(metrics.get("dim_loss", 0.0))
    if mode == "default":
        return float(miou - 0.05 * loc_loss - 0.05 * dim_loss)
    if mode in {"miou", "segmentation"}:
        return miou
    if mode == "vehicle_iou":
        return vehicle_iou
    if mode == "vehicle_miou":
        return float(0.7 * vehicle_iou + 0.3 * miou)
    if mode == "person_iou":
        return person_iou
    if mode == "person_miou":
        return float(0.7 * person_iou + 0.3 * miou)
    if mode in {"object_loss", "localization_loss"}:
        return -float(metrics.get("object_loss", 0.0))
    if mode in {"loc_loss", "xy_loss"}:
        return -loc_loss
    if mode in {"loc_dim_loss", "xy_dim_loss"}:
        return -(loc_loss + 0.25 * dim_loss)
    raise ValueError(
        "Unsupported selection_score_mode "
        f"{mode!r}; use default, miou, segmentation, vehicle_iou, vehicle_miou, "
        "person_iou, person_miou, object_loss, loc_loss, or loc_dim_loss."
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    loss_weights: Dict[str, float],
    class_weights: Optional[torch.Tensor] = None,
    selection_score_mode: str = "default",
    lovasz_weight: float = 0.0,
    feature_drop_fraction: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    losses: List[float] = []
    part_sums: Dict[str, float] = {}
    batches = 0
    object_counts = 0
    with torch.inference_mode():
        for tensors, masks, object_targets in loader:
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            object_targets = _move_object_targets(object_targets, device)
            loss, parts, logits = compute_losses(
                model,
                tensors,
                masks,
                object_targets,
                num_classes,
                loss_weights,
                class_weights=class_weights,
                lovasz_weight=lovasz_weight,
                feature_drop_fraction=feature_drop_fraction,
            )
            losses.append(float(loss.item()))
            batches += 1
            object_counts += int(object_targets["gt_count"].sum().item())
            for key, value in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            target = masks.detach().cpu().numpy().astype(np.int64)
            for p, t in zip(pred, target):
                update_confusion(confusion, p, t, num_classes)
    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "gt_objects": float(object_counts),
    }
    for key, value in part_sums.items():
        metrics[key] = float(value / max(1, batches))
    for idx, name in enumerate(CLASS_NAMES[:num_classes]):
        metrics[f"{name}_iou"] = float(ious[idx])
    metrics["selection_score"] = _selection_score(metrics, selection_score_mode)
    return metrics


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    dataset_dir = exp_dir / "dataset"
    manifest_path = dataset_dir / "manifest.csv"
    log = setup_logger(exp_dir / "supervisor.log")
    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No dataset rows found at {manifest_path}")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    splits = split_rows(rows)
    save_split_files(exp_dir, splits)
    if not splits["train"] or not splits["val"]:
        raise RuntimeError("Training requires non-empty train and val splits.")

    trial = json.loads(args.trial_json)
    trial_name = str(trial.get("name", "trial"))
    train_cfg = config["training"]
    fusion_cfg = config.get("fusion", {})
    object_cfg = _deep_merge_dicts(
        dict(config.get("object_heads", {})),
        trial.get("object_heads") if isinstance(trial.get("object_heads"), dict) else None,
    )
    object_class_names = tuple(object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
    input_width, input_height = [int(v) for v in trial.get("input_size", train_cfg.get("input_size", [512, 288]))]
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(fusion_cfg.get("radar_channels", 4))
    trial_seed = int(hashlib.sha1(trial_name.encode("utf-8")).hexdigest()[:8], 16)
    seed = int(config["collection"].get("seed", 17)) ^ trial_seed
    set_reproducible_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"Learned-localization fusion trial {trial_name} on {device}; "
        f"rows train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    trial_dir = exp_dir / "checkpoints" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics" / f"{trial_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    loss_weights = _deep_merge_dicts(
        dict(train_cfg.get("loss_weights", {})),
        trial.get("loss_weights") if isinstance(trial.get("loss_weights"), dict) else None,
    )
    object_total_weight = float(loss_weights.get("object_total", 1.0))
    geometric_augment = bool(trial.get("geometric_augment", False))
    if geometric_augment and object_total_weight > 0.0:
        raise ValueError("geometric_augment=true is only supported for segmentation-only trials with object_total=0.")

    train_ds = FusionPoleMultiTaskDataset(
        dataset_dir,
        splits["train"],
        object_rows,
        (input_width, input_height),
        object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
        geometric_augment=geometric_augment,
    )
    val_ds = FusionPoleMultiTaskDataset(dataset_dir, splits["val"], object_rows, (input_width, input_height), object_cfg, augment_strength="off")
    num_workers = int(trial.get("num_workers", train_cfg.get("num_workers", 4)))
    persistent_workers = bool(trial.get("persistent_workers", train_cfg.get("persistent_workers", True))) and num_workers > 0
    prefetch_factor = int(trial.get("prefetch_factor", train_cfg.get("prefetch_factor", 2)))
    data_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        data_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 6))),
        shuffle=True,
        drop_last=False,
        **data_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 6))),
        shuffle=False,
        **data_loader_kwargs,
    )

    init_checkpoint = str(trial.get("init_rgb_checkpoint", train_cfg.get("init_rgb_checkpoint", "")))
    fuse_low_into_object_head = bool(object_cfg.get("fuse_low_feature", False))
    object_head_arch = str(object_cfg.get("head_arch", "shared"))
    object_use_coordconv = bool(object_cfg.get("use_coordconv", False))
    object_head_depth = int(object_cfg.get("head_depth", 2))
    object_use_groundplane = bool(object_cfg.get("use_groundplane_prior", False))
    object_groundplane_params = dict(object_cfg.get("groundplane_params", {}) or {})
    object_predict_bbox2d = bool(object_cfg.get("predict_bbox2d", False))
    # Total object-head channels = #object-classes (heatmap) + regression channels.
    object_channels_total = len(object_class_names) + object_reg_channels(object_predict_bbox2d)
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=bool(train_cfg.get("pretrained", True)) and not init_checkpoint,
        init_checkpoint=init_checkpoint,
        object_channels=object_channels_total,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=fuse_low_into_object_head,
        head_arch=object_head_arch,
        use_coordconv=object_use_coordconv,
        head_depth=object_head_depth,
        predict_bbox2d=object_predict_bbox2d,
        use_groundplane_prior=object_use_groundplane,
        groundplane_params=object_groundplane_params,
        device=device,
    ).to(device)
    init_object_checkpoint = str(trial.get("init_object_checkpoint", train_cfg.get("init_object_checkpoint", "")))
    if init_object_checkpoint:
        object_load_stats = _load_object_head_checkpoint(model, init_object_checkpoint, device=device)
        log(
            f"Loaded object head from {init_object_checkpoint}; "
            f"loaded={object_load_stats['loaded']} partial={object_load_stats.get('partial', 0)} "
            f"skipped={object_load_stats['skipped']}."
        )
    # INTEGRATED feature-AE (end-to-end): attach after warm-start load, before the optimizer, so its
    # params train jointly with the backbone + heads (the whole model co-adapts to the bottleneck).
    ae_bottleneck = int(trial.get("ae_bottleneck", 0))
    if ae_bottleneck > 0:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rl_agent" / "feature_ae"))
        from ae_model import build_ae
        hi_ch = int(model.classifier.cbr[0].in_channels)
        ae = build_ae(str(trial.get("ae_arch", "v2")), hi_ch, ae_bottleneck).to(device)
        ae_init = str(trial.get("ae_init_checkpoint", ""))
        if ae_init and Path(ae_init).exists():
            _aeck = torch.load(ae_init, map_location=device)
            ae.load_state_dict(_aeck["ae_state"])
            log(f"AE warm-started from {ae_init}")
        model.feature_ae = ae
        log(f"Integrated feature-AE: arch={trial.get('ae_arch','v2')} bottleneck={ae_bottleneck} "
            f"in={hi_ch} params={sum(p.numel() for p in ae.parameters()):,}")
    freeze_backbone = bool(trial.get("freeze_backbone", train_cfg.get("freeze_backbone", False)))
    freeze_classifier = bool(
        trial.get(
            "freeze_classifier",
            trial.get("freeze_seg_head", train_cfg.get("freeze_classifier", train_cfg.get("freeze_seg_head", False))),
        )
    )
    freeze_object_head = bool(trial.get("freeze_object_head", train_cfg.get("freeze_object_head", False)))
    if freeze_backbone:
        _set_requires_grad(model.backbone, False)
    if freeze_classifier:
        _set_requires_grad(model.classifier, False)
    if freeze_object_head:
        _set_requires_grad(model.object_head, False)
    # Partial unfreeze: re-enable grads on only the last N backbone blocks so the
    # high-feature / object pathway can adapt while the low-detail (person-boundary)
    # pathway stays frozen. BN is re-frozen below regardless.
    unfreeze_backbone_last_n = int(trial.get("unfreeze_backbone_last_n", 0))
    if unfreeze_backbone_last_n > 0:
        bb_children = list(model.backbone.named_children())
        for name, child in bb_children[-unfreeze_backbone_last_n:]:
            _set_requires_grad(child, True)
        log(f"Partial unfreeze: last {unfreeze_backbone_last_n} backbone blocks "
            f"({[n for n, _ in bb_children[-unfreeze_backbone_last_n:]]}) set trainable.")
    freeze_bn = bool(trial.get("freeze_bn", train_cfg.get("freeze_bn", False)))
    if freeze_bn:
        _freeze_batch_norm(model)
    _eval_frozen_modules(
        model,
        freeze_backbone=freeze_backbone,
        freeze_classifier=freeze_classifier,
        freeze_object_head=freeze_object_head,
    )
    total_params, trainable_params = _count_parameters(model)
    trainable_param_list = [param for param in model.parameters() if param.requires_grad]
    if not trainable_param_list:
        raise ValueError("No trainable parameters remain after applying freeze options.")
    log(
        f"Trainable parameters: {trainable_params:,}/{total_params:,}; "
        f"freeze_backbone={freeze_backbone} freeze_classifier={freeze_classifier} "
        f"freeze_object_head={freeze_object_head}."
    )
    optimizer_name = str(trial.get("optimizer", train_cfg.get("optimizer", "adamw"))).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_param_list,
            lr=float(trial.get("lr", 1e-3)),
            momentum=float(trial.get("momentum", 0.9)),
            weight_decay=float(trial.get("weight_decay", 5e-4)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_param_list, lr=float(trial.get("lr", 2e-4)), weight_decay=float(trial.get("weight_decay", 1e-4)))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    class_loss_weights_cfg = trial.get("class_loss_weights", train_cfg.get("class_loss_weights"))
    selection_score_mode = str(trial.get("selection_score_mode", train_cfg.get("selection_score_mode", "default")))
    lovasz_weight = float(trial.get("lovasz_weight", train_cfg.get("lovasz_weight", 0.0)))
    # Drop-aware training (opt-in): per batch, drop a random objectness-ranked feature
    # fraction q ~ Uniform(0, feature_drop_max) so ONE model generalizes across ROI drop
    # thresholds. Default 0.0 => structural no-op, leaves the 200k recipe byte-identical.
    feature_drop_max = float(trial.get("feature_drop_max", train_cfg.get("feature_drop_max", 0.0)))
    # Validation/selection drop level: for a drop-aware run, select the checkpoint that is best at a
    # representative operating point (default q_max/2) instead of clean q=0 -> otherwise selection keeps
    # the least drop-adapted epoch. Clean q=0 is guarded separately at GATE A. 0 => clean val (default).
    feature_drop_val = float(trial.get("feature_drop_val", feature_drop_max * 0.5))
    # Seg-distillation teacher: a frozen copy of the seg model (from the distill
    # checkpoint, defaulting to the seg init checkpoint) anchors the student's seg
    # output while the backbone partially adapts for localization.
    distill_weight = float(trial.get("distill_weight", 0.0))
    distill_temp = float(trial.get("distill_temp", 2.0))
    teacher_model = None
    if distill_weight > 0.0:
        teacher_ckpt = str(trial.get("distill_teacher_checkpoint", init_checkpoint))
        teacher_model = build_fusion_lraspp(
            num_classes=num_classes,
            radar_channels=radar_channels,
            pretrained=False,
            init_checkpoint=teacher_ckpt,
            device=device,
        ).to(device)
        teacher_model.eval()
        _set_requires_grad(teacher_model, False)
        log(f"Distillation enabled: weight={distill_weight:g} temp={distill_temp:g} teacher={teacher_ckpt}")
    class_weights_tensor: Optional[torch.Tensor] = None
    if class_loss_weights_cfg is not None:
        class_weights_tensor = torch.tensor(
            [float(w) for w in class_loss_weights_cfg], dtype=torch.float32, device=device
        )
        if class_weights_tensor.numel() != num_classes:
            raise ValueError(
                f"training.class_loss_weights has {class_weights_tensor.numel()} entries but num_classes={num_classes}"
            )
    log(
        f"{trial_name} objective: selection_score_mode={selection_score_mode}; "
        f"class_loss_weights={class_loss_weights_cfg}; loss_weights={json.dumps(loss_weights, sort_keys=True)}; "
        f"lovasz_weight={lovasz_weight:g}; geometric_augment={geometric_augment}; freeze_bn={freeze_bn}"
    )

    start_epoch = 0
    best_score = -math.inf
    best_miou = -math.inf
    best_path = trial_dir / "best.pt"
    last_path = trial_dir / "last.pt"
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    lr_scheduler = str(trial.get("lr_scheduler", train_cfg.get("lr_scheduler", "none"))).lower()
    lr_warmup_epochs = int(trial.get("lr_warmup_epochs", train_cfg.get("lr_warmup_epochs", 0)))
    min_lr_ratio = float(trial.get("min_lr_ratio", train_cfg.get("min_lr_ratio", 0.05)))
    poly_power = float(trial.get("poly_power", train_cfg.get("poly_power", 0.9)))
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        base_lrs = [float(v) for v in ckpt.get("base_lrs", base_lrs)]
        if "resume_lr" in trial:
            resume_lr = float(trial["resume_lr"])
            for group in optimizer.param_groups:
                group["lr"] = resume_lr
            base_lrs = [resume_lr for _ in optimizer.param_groups]
            log(f"Overrode resumed optimizer lr to {resume_lr:g}.")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_selection_score", ckpt.get("best_miou", best_score)))
        best_miou = float(ckpt.get("best_miou", best_miou))
        log(f"Resumed {trial_name} from epoch {start_epoch}.")

    fieldnames = [
        "trial",
        "epoch",
        "train_loss",
        "val_loss",
        "selection_score",
        "miou",
        "vehicle_iou",
        "person_iou",
        "pixel_accuracy",
        "seg_loss",
        "ce_loss",
        "lovasz_loss",
        "object_loss",
        "center_loss",
        "loc_loss",
        "dim_loss",
        "yaw_loss",
        "parked_loss",
        "radar_support_loss",
        "gt_objects",
        "lr",
        "timestamp",
    ]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    deadline = time.monotonic() + float(args.training_budget_hours) * 3600.0 if float(args.training_budget_hours) > 0 else math.inf
    patience = int(trial.get("early_stop_patience", train_cfg.get("early_stop_patience", 3)))
    stale_epochs = 0
    max_epochs = int(trial.get("epochs", train_cfg.get("epochs", 8)))
    for epoch in range(start_epoch, max_epochs):
        if time.monotonic() >= deadline:
            log(f"Training budget exhausted during {trial_name}; checkpointing and stopping.")
            break
        lr_mult = _lr_multiplier(epoch, max_epochs, lr_scheduler, lr_warmup_epochs, min_lr_ratio, poly_power)
        _set_optimizer_lr(optimizer, base_lrs, lr_mult)
        model.train()
        _eval_frozen_modules(
            model,
            freeze_backbone=freeze_backbone,
            freeze_classifier=freeze_classifier,
            freeze_object_head=freeze_object_head,
        )
        if freeze_bn:
            _freeze_batch_norm(model)
        losses: List[float] = []
        for tensors, masks, object_targets in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            object_targets = _move_object_targets(object_targets, device)
            optimizer.zero_grad(set_to_none=True)
            q_drop = float(torch.rand(1).item()) * feature_drop_max if feature_drop_max > 0.0 else 0.0
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, _, _ = compute_losses(
                    model, tensors, masks, object_targets, num_classes, loss_weights,
                    class_weights=class_weights_tensor,
                    lovasz_weight=lovasz_weight,
                    teacher=teacher_model,
                    distill_weight=distill_weight,
                    distill_temp=distill_temp,
                    feature_drop_fraction=q_drop,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            num_classes,
            loss_weights,
            class_weights=class_weights_tensor,
            selection_score_mode=selection_score_mode,
            lovasz_weight=lovasz_weight,
            feature_drop_fraction=feature_drop_val,
        )
        # Maximin selection for drop-aware runs: also evaluate clean (q=0) and select the checkpoint
        # on the WORST of {clean, drop} so robustness cannot be bought by regressing clean accuracy
        # (and vice-versa). val_metrics keeps the drop-pass metrics for logging; clean is added alongside.
        if feature_drop_val > 0.0:
            clean_metrics = evaluate_model(
                model, val_loader, device, num_classes, loss_weights,
                class_weights=class_weights_tensor, selection_score_mode=selection_score_mode,
                lovasz_weight=lovasz_weight, feature_drop_fraction=0.0,
            )
            val_metrics["drop_selection_score"] = val_metrics["selection_score"]
            val_metrics["clean_selection_score"] = clean_metrics["selection_score"]
            val_metrics["clean_miou"] = clean_metrics["miou"]
            val_metrics["selection_score"] = min(
                float(val_metrics["selection_score"]), float(clean_metrics["selection_score"]))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "trial": trial_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "selection_score": val_metrics["selection_score"],
            "miou": val_metrics["miou"],
            "vehicle_iou": val_metrics.get("vehicle_iou", float("nan")),
            "person_iou": val_metrics.get("person_iou", float("nan")),
            "pixel_accuracy": val_metrics["pixel_accuracy"],
            "seg_loss": val_metrics.get("seg_loss", float("nan")),
            "ce_loss": val_metrics.get("ce_loss", float("nan")),
            "lovasz_loss": val_metrics.get("lovasz_loss", float("nan")),
            "object_loss": val_metrics.get("object_loss", float("nan")),
            "center_loss": val_metrics.get("center_loss", float("nan")),
            "loc_loss": val_metrics.get("loc_loss", float("nan")),
            "dim_loss": val_metrics.get("dim_loss", float("nan")),
            "yaw_loss": val_metrics.get("yaw_loss", float("nan")),
            "parked_loss": val_metrics.get("parked_loss", float("nan")),
            "radar_support_loss": val_metrics.get("radar_support_loss", float("nan")),
            "gt_objects": val_metrics.get("gt_objects", 0.0),
            "lr": float(optimizer.param_groups[0].get("lr", trial.get("lr", 2e-4))),
            "timestamp": utc_iso(),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)
        improved = float(val_metrics["selection_score"]) > best_score
        if improved:
            best_score = float(val_metrics["selection_score"])
            best_miou = float(val_metrics["miou"])
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_selection_score": best_score,
                    "best_miou": best_miou,
                    "trial": trial,
                    "config": config,
                    "input_size": [input_width, input_height],
                    "radar_channels": radar_channels,
                    "object_channels": object_channels_total,
                    "object_predict_bbox2d": object_predict_bbox2d,
                    "object_class_names": list(object_class_names),
                    "fuse_low_into_object_head": bool(fuse_low_into_object_head),
                    "object_head_arch": object_head_arch,
                    "object_use_coordconv": object_use_coordconv,
                    "object_head_depth": object_head_depth,
                    "object_use_groundplane_prior": object_use_groundplane,
                    "object_groundplane_params": object_groundplane_params,
                    "class_names": CLASS_NAMES[:num_classes],
                    "init_rgb_checkpoint": init_checkpoint,
                    "init_object_checkpoint": init_object_checkpoint,
                    "class_loss_weights": class_loss_weights_cfg,
                    "loss_weights": loss_weights,
                    "selection_score_mode": selection_score_mode,
                    "lovasz_weight": lovasz_weight,
                    "lr_scheduler": lr_scheduler,
                    "lr_warmup_epochs": lr_warmup_epochs,
                    "min_lr_ratio": min_lr_ratio,
                    "poly_power": poly_power,
                    "early_stop_patience": patience,
                    "freeze_backbone": freeze_backbone,
                    "freeze_classifier": freeze_classifier,
                    "freeze_object_head": freeze_object_head,
                    "model_task": "segmentation_plus_learned_object_localization",
                },
                best_path,
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_selection_score": best_score,
                "best_miou": best_miou,
                "trial": trial,
                "config": config,
                "input_size": [input_width, input_height],
                "radar_channels": radar_channels,
                "object_channels": object_channels_total,
                    "object_predict_bbox2d": object_predict_bbox2d,
                "object_class_names": list(object_class_names),
                "fuse_low_into_object_head": bool(fuse_low_into_object_head),
                "object_head_arch": object_head_arch,
                "object_use_coordconv": object_use_coordconv,
                "object_head_depth": object_head_depth,
                "init_rgb_checkpoint": init_checkpoint,
                "init_object_checkpoint": init_object_checkpoint,
                "class_loss_weights": class_loss_weights_cfg,
                "loss_weights": loss_weights,
                "selection_score_mode": selection_score_mode,
                "lovasz_weight": lovasz_weight,
                "lr_scheduler": lr_scheduler,
                "lr_warmup_epochs": lr_warmup_epochs,
                "min_lr_ratio": min_lr_ratio,
                "poly_power": poly_power,
                "base_lrs": base_lrs,
                "early_stop_patience": patience,
                "freeze_backbone": freeze_backbone,
                "freeze_classifier": freeze_classifier,
                "freeze_object_head": freeze_object_head,
                "model_task": "segmentation_plus_learned_object_localization",
            },
            last_path,
        )
        log(
            f"{trial_name} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_score(maximin)={val_metrics['selection_score']:.4f} val_miou@drop={val_metrics['miou']:.4f} "
            f"clean_miou={val_metrics.get('clean_miou', float('nan')):.4f} "
            f"vehicle_iou={val_metrics.get('vehicle_iou', float('nan')):.4f} "
            f"loc_loss={val_metrics.get('loc_loss', float('nan')):.4f} dim_loss={val_metrics.get('dim_loss', float('nan')):.4f}"
        )
        if patience > 0 and stale_epochs >= patience:
            log(f"Early stopping {trial_name} after {stale_epochs} stale epochs.")
            break

    save_json(
        trial_dir / "trial_summary.json",
        {
            "trial": trial,
            "best_selection_score": best_score,
            "best_miou": best_miou,
            "best_checkpoint": str(best_path),
            "updated_at": utc_iso(),
            "init_rgb_checkpoint": init_checkpoint,
            "init_object_checkpoint": init_object_checkpoint,
            "class_loss_weights": class_loss_weights_cfg,
            "loss_weights": loss_weights,
            "selection_score_mode": selection_score_mode,
            "lovasz_weight": lovasz_weight,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_epochs": lr_warmup_epochs,
            "min_lr_ratio": min_lr_ratio,
            "poly_power": poly_power,
            "early_stop_patience": patience,
            "freeze_backbone": freeze_backbone,
            "freeze_classifier": freeze_classifier,
            "freeze_object_head": freeze_object_head,
            "model_task": "segmentation_plus_learned_object_localization",
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--trial-json", required=True)
    parser.add_argument("--training-budget-hours", type=float, default=0.0)
    args = parser.parse_args()
    raise SystemExit(train(args))


if __name__ == "__main__":
    main()
