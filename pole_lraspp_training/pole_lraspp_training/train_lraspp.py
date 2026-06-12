from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def build_lraspp(num_classes: int, pretrained: bool) -> torch.nn.Module:
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
    from torchvision.models.segmentation.lraspp import LRASPPHead

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    try:
        model = lraspp_mobilenet_v3_large(weights=weights)
    except Exception:
        model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    high_channels = int(model.classifier.cbr[0].in_channels)
    inter_channels = int(model.classifier.cbr[0].out_channels)
    low_channels = int(model.classifier.low_classifier.in_channels)
    try:
        model.classifier = LRASPPHead(low_channels, high_channels, int(num_classes), inter_channels)
    except TypeError:
        model.classifier = LRASPPHead(low_channels, high_channels, int(num_classes))
    return model


class PoleSegDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        rows: List[Dict[str, str]],
        input_size: Tuple[int, int],
        augment_strength: str = "off",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.input_width, self.input_height = input_size
        self.augment_strength = str(augment_strength)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.augment_strength == "off":
            return image, mask
        if np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment_strength in {"light", "medium"}:
            if np.random.rand() < 0.35:
                factor = float(np.random.uniform(0.85, 1.15) if self.augment_strength == "light" else np.random.uniform(0.75, 1.25))
                image = ImageEnhance.Brightness(image).enhance(factor)
            if np.random.rand() < 0.35:
                factor = float(np.random.uniform(0.9, 1.1) if self.augment_strength == "light" else np.random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["rgb_path"]).convert("RGB")
        mask = Image.open(self.dataset_dir / row["mask_path"]).convert("L")
        image, mask = self._augment(image, mask)
        image = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)
        mask = mask.resize((self.input_width, self.input_height), Image.Resampling.NEAREST)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return image_tensor, mask_tensor


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


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    losses: List[float] = []
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)["out"]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, masks)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            target = masks.detach().cpu().numpy().astype(np.int64)
            for p, t in zip(pred, target):
                update_confusion(confusion, p, t, num_classes)
    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "miou": miou,
        "pixel_accuracy": pixel_acc,
    }
    for idx, name in enumerate(CLASS_NAMES[:num_classes]):
        metrics[f"{name}_iou"] = float(ious[idx])
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
    splits = split_rows(rows)
    save_split_files(exp_dir, splits)
    if not splits["train"] or not splits["val"]:
        raise RuntimeError("Training requires non-empty train and val splits.")

    trial = json.loads(args.trial_json)
    trial_name = str(trial.get("name", "trial"))
    train_cfg = config["training"]
    input_width, input_height = [int(v) for v in trial.get("input_size", train_cfg.get("input_size", [512, 288]))]
    num_classes = int(train_cfg.get("num_classes", 3))
    trial_seed = int(hashlib.sha1(trial_name.encode("utf-8")).hexdigest()[:8], 16)
    seed = int(config["collection"].get("seed", 17)) ^ trial_seed
    set_reproducible_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training trial {trial_name} on {device}; rows train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    trial_dir = exp_dir / "checkpoints" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics" / f"{trial_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = PoleSegDataset(dataset_dir, splits["train"], (input_width, input_height), augment_strength=str(trial.get("augment_strength", "off")))
    val_ds = PoleSegDataset(dataset_dir, splits["val"], (input_width, input_height), augment_strength="off")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 8))),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(trial.get("batch_size", train_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_lraspp(num_classes, bool(train_cfg.get("pretrained", True))).to(device)
    optimizer_name = str(trial.get("optimizer", train_cfg.get("optimizer", "adamw"))).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(trial.get("lr", 1e-3)),
            momentum=float(trial.get("momentum", 0.9)),
            weight_decay=float(trial.get("weight_decay", 5e-4)),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(trial.get("lr", 3e-4)),
            weight_decay=float(trial.get("weight_decay", 1e-4)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    start_epoch = 0
    best_miou = -math.inf
    best_path = trial_dir / "best.pt"
    last_path = trial_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_miou = float(ckpt.get("best_miou", best_miou))
        log(f"Resumed {trial_name} from epoch {start_epoch}.")

    fieldnames = ["trial", "epoch", "train_loss", "val_loss", "miou", "vehicle_iou", "person_iou", "pixel_accuracy", "lr", "timestamp"]
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    deadline = time.monotonic() + float(args.training_budget_hours) * 3600.0 if float(args.training_budget_hours) > 0 else math.inf
    patience = int(train_cfg.get("early_stop_patience", 3))
    stale_epochs = 0
    max_epochs = int(train_cfg.get("epochs", 8))
    for epoch in range(start_epoch, max_epochs):
        if time.monotonic() >= deadline:
            log(f"Training budget exhausted during {trial_name}; checkpointing and stopping.")
            break
        model.train()
        losses: List[float] = []
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)["out"]
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        val_metrics = evaluate_model(model, val_loader, device, num_classes)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "trial": trial_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "miou": val_metrics["miou"],
            "vehicle_iou": val_metrics.get("vehicle_iou", float("nan")),
            "person_iou": val_metrics.get("person_iou", float("nan")),
            "pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": float(trial.get("lr", 3e-4)),
            "timestamp": utc_iso(),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)
        improved = float(val_metrics["miou"]) > best_miou
        if improved:
            best_miou = float(val_metrics["miou"])
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "trial": trial,
                    "config": config,
                    "input_size": [input_width, input_height],
                    "class_names": CLASS_NAMES[:num_classes],
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
                "best_miou": best_miou,
                "trial": trial,
                "config": config,
                "input_size": [input_width, input_height],
            },
            last_path,
        )
        log(f"{trial_name} epoch={epoch} train_loss={train_loss:.4f} val_miou={val_metrics['miou']:.4f} vehicle_iou={val_metrics.get('vehicle_iou', float('nan')):.4f} person_iou={val_metrics.get('person_iou', float('nan')):.4f}")
        if stale_epochs >= patience:
            log(f"Early stopping {trial_name} after {stale_epochs} stale epochs.")
            break

    save_json(trial_dir / "trial_summary.json", {"trial": trial, "best_miou": best_miou, "best_checkpoint": str(best_path), "updated_at": utc_iso()})
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
