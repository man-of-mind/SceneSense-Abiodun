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
from .model import OBJECT_HEAD_CHANNELS, build_multitask_fusion_lraspp
from .object_targets import (
    OBJECT_CLASS_NAMES,
    build_object_targets,
    load_object_boxes,
    multitask_object_loss,
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
    ) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.object_rows = object_rows
        self.input_width, self.input_height = input_size
        self.object_cfg = object_cfg
        self.object_class_names = tuple(object_cfg.get("object_classes", OBJECT_CLASS_NAMES))
        self.augment_strength = str(augment_strength)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: Image.Image, mask: Image.Image, radar: np.ndarray) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        if self.augment_strength == "off":
            return image, mask, radar
        # Keep geometric alignment intact for learned localization; use photometric augmentation only.
        if self.augment_strength in {"light", "medium"}:
            if np.random.rand() < 0.35:
                factor = float(np.random.uniform(0.85, 1.15) if self.augment_strength == "light" else np.random.uniform(0.75, 1.25))
                image = ImageEnhance.Brightness(image).enhance(factor)
            if np.random.rand() < 0.35:
                factor = float(np.random.uniform(0.9, 1.1) if self.augment_strength == "light" else np.random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(factor)
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
        )
        object_targets = build_object_targets(
            objects=objects,
            original_size=(original_width, original_height),
            input_size=(self.input_width, self.input_height),
            heatmap_radius_px=int(self.object_cfg.get("heatmap_radius_px", 2)),
            max_objects=int(self.object_cfg.get("max_objects_per_frame", 64)),
            object_class_names=self.object_class_names,
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


def _move_object_targets(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def compute_losses(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    masks: torch.Tensor,
    object_targets: Dict[str, torch.Tensor],
    num_classes: int,
    loss_weights: Dict[str, float],
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    outputs = model(tensors)
    logits = outputs["out"]
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    seg_loss = F.cross_entropy(logits, masks, weight=class_weights)
    # Run regression head losses in FP32 even when the surrounding context is AMP.
    # Smooth-L1 / BCE on small-magnitude regression targets are unstable in FP16
    # and can silently zero gradients into the object head.
    with torch.cuda.amp.autocast(enabled=False):
        object_logits_fp32 = outputs["object"].float()
        object_loss, object_parts = multitask_object_loss(
            object_logits_fp32, object_targets, loss_weights.get("object", {})
        )
    total = float(loss_weights.get("segmentation", 1.0)) * seg_loss + float(loss_weights.get("object_total", 1.0)) * object_loss
    parts = {"seg_loss": float(seg_loss.detach().item()), "object_loss": float(object_loss.detach().item()), **object_parts}
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
    raise ValueError(
        "Unsupported selection_score_mode "
        f"{mode!r}; use default, miou, segmentation, vehicle_iou, or vehicle_miou."
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    loss_weights: Dict[str, float],
    class_weights: Optional[torch.Tensor] = None,
    selection_score_mode: str = "default",
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
            loss, parts, logits = compute_losses(model, tensors, masks, object_targets, num_classes, loss_weights, class_weights=class_weights)
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
    object_cfg = config.get("object_heads", {})
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

    train_ds = FusionPoleMultiTaskDataset(
        dataset_dir,
        splits["train"],
        object_rows,
        (input_width, input_height),
        object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    val_ds = FusionPoleMultiTaskDataset(dataset_dir, splits["val"], object_rows, (input_width, input_height), object_cfg, augment_strength="off")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 6))),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 6))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    init_checkpoint = str(trial.get("init_rgb_checkpoint", train_cfg.get("init_rgb_checkpoint", "")))
    fuse_low_into_object_head = bool(object_cfg.get("fuse_low_feature", False))
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=bool(train_cfg.get("pretrained", True)) and not init_checkpoint,
        init_checkpoint=init_checkpoint,
        object_channels=int(object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=fuse_low_into_object_head,
        device=device,
    ).to(device)
    optimizer_name = str(trial.get("optimizer", train_cfg.get("optimizer", "adamw"))).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(trial.get("lr", 1e-3)),
            momentum=float(trial.get("momentum", 0.9)),
            weight_decay=float(trial.get("weight_decay", 5e-4)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial.get("lr", 2e-4)), weight_decay=float(trial.get("weight_decay", 1e-4)))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    loss_weights = _deep_merge_dicts(
        dict(train_cfg.get("loss_weights", {})),
        trial.get("loss_weights") if isinstance(trial.get("loss_weights"), dict) else None,
    )
    class_loss_weights_cfg = trial.get("class_loss_weights", train_cfg.get("class_loss_weights"))
    selection_score_mode = str(trial.get("selection_score_mode", train_cfg.get("selection_score_mode", "default")))
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
        f"class_loss_weights={class_loss_weights_cfg}; loss_weights={json.dumps(loss_weights, sort_keys=True)}"
    )

    start_epoch = 0
    best_score = -math.inf
    best_miou = -math.inf
    best_path = trial_dir / "best.pt"
    last_path = trial_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "resume_lr" in trial:
            resume_lr = float(trial["resume_lr"])
            for group in optimizer.param_groups:
                group["lr"] = resume_lr
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
    patience = int(train_cfg.get("early_stop_patience", 3))
    stale_epochs = 0
    max_epochs = int(trial.get("epochs", train_cfg.get("epochs", 8)))
    for epoch in range(start_epoch, max_epochs):
        if time.monotonic() >= deadline:
            log(f"Training budget exhausted during {trial_name}; checkpointing and stopping.")
            break
        model.train()
        losses: List[float] = []
        for tensors, masks, object_targets in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            object_targets = _move_object_targets(object_targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, _, _ = compute_losses(
                    model, tensors, masks, object_targets, num_classes, loss_weights,
                    class_weights=class_weights_tensor,
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
        )
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
                    "object_channels": int(object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
                    "object_class_names": list(object_class_names),
                    "fuse_low_into_object_head": bool(fuse_low_into_object_head),
                    "class_names": CLASS_NAMES[:num_classes],
                    "init_rgb_checkpoint": init_checkpoint,
                    "class_loss_weights": class_loss_weights_cfg,
                    "loss_weights": loss_weights,
                    "selection_score_mode": selection_score_mode,
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
                "object_channels": int(object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
                "object_class_names": list(object_class_names),
                "fuse_low_into_object_head": bool(fuse_low_into_object_head),
                "init_rgb_checkpoint": init_checkpoint,
                "class_loss_weights": class_loss_weights_cfg,
                "loss_weights": loss_weights,
                "selection_score_mode": selection_score_mode,
                "model_task": "segmentation_plus_learned_object_localization",
            },
            last_path,
        )
        log(
            f"{trial_name} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_score={val_metrics['selection_score']:.4f} val_miou={val_metrics['miou']:.4f} "
            f"vehicle_iou={val_metrics.get('vehicle_iou', float('nan')):.4f} "
            f"loc_loss={val_metrics.get('loc_loss', float('nan')):.4f} dim_loss={val_metrics.get('dim_loss', float('nan')):.4f}"
        )
        if stale_epochs >= patience:
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
            "class_loss_weights": class_loss_weights_cfg,
            "loss_weights": loss_weights,
            "selection_score_mode": selection_score_mode,
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
