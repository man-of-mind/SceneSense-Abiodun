#!/usr/bin/env python3
"""CenterNet v2 trainer - self-contained, native dual-stride heads.

Deliberately does not go through ``train_fusion.train``: that trainer's target
builder, loss and checkpoint payload all assume one dense full-resolution object
tensor, which is exactly the geometry v2 removes.  The RGB/radar/mask input
pipeline *is* reused (``NativeFusionDataset`` subclasses the v1 dataset), so the
input contract is unchanged.

One fixed differential-LR schedule, one warmup/cosine schedule, 24 epochs, no
early stopping, every epoch saved create-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
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
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from pole_lraspp_multimodal_fusion.train_fusion import _freeze_batch_norm  # noqa: E402

from centernet_model_v2 import build_centernet_v2, warm_start_from_v1  # noqa: E402
from losses_v2 import DEFAULT_OBJECT_WEIGHTS, compute_v2_losses  # noqa: E402
from targets_v2 import NativeFusionDataset  # noqa: E402

BACKBONE_PREFIX = ("backbone.",)
WARM_PREFIX = ("rgb_fpn.", "radar_fpn.", "radar_encoder.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lr_multiplier(epoch: int, max_epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    """Linear warmup for ``warmup_epochs`` then cosine decay to ``min_lr_ratio``."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max(1e-3, float(epoch + 1) / float(warmup_epochs))
    decay_epochs = max(1, max_epochs - max(0, warmup_epochs))
    progress = min(1.0, max(0.0, float(epoch - warmup_epochs + 1) / float(decay_epochs)))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_param_groups(model: torch.nn.Module, trial: Dict) -> List[Dict]:
    """Fixed three-tier differential LR.

    highest  - every newly initialized tensor (native vehicle/person heads, the
               private offset heads, the stride-2 projection, the RGB/radar
               fusion and the segmentation decoder);
    middle   - warm-started RGB-FPN / radar-FPN / radar encoder;
    lowest   - the ImageNet-pretrained ResNet34 backbone.
    """
    groups = {"new": [], "warm": [], "backbone": []}
    names = {"new": [], "warm": [], "backbone": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(BACKBONE_PREFIX):
            key = "backbone"
        elif name.startswith(WARM_PREFIX):
            key = "warm"
        else:
            key = "new"
        groups[key].append(param)
        names[key].append(name)
    lrs = {
        "new": float(trial.get("lr_new", 3e-4)),
        "warm": float(trial.get("lr_warm", 1e-4)),
        "backbone": float(trial.get("lr_backbone", 3e-5)),
    }
    out = []
    for key in ("new", "warm", "backbone"):
        if groups[key]:
            out.append({"params": groups[key], "lr": lrs[key], "group_name": key})
    return out, names, lrs


def move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_validation(model, loader, device, num_classes, loss_cfg) -> Dict[str, float]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    losses: List[float] = []
    part_sums: Dict[str, float] = {}
    batches = 0
    with torch.inference_mode():
        for tensors, masks, targets in loader:
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = move(targets, device)
            with torch.autocast(device_type=device.type, enabled=loss_cfg["amp"]):
                outputs = model(tensors)
            with torch.autocast(device_type=device.type, enabled=False):
                loss, parts = compute_v2_losses(
                    outputs,
                    masks,
                    targets,
                    object_weights=loss_cfg["object_weights"],
                    segmentation_weight=loss_cfg["segmentation_weight"],
                    object_total_weight=loss_cfg["object_total_weight"],
                    class_weights=loss_cfg["class_weights"],
                    lovasz_weight=loss_cfg["lovasz_weight"],
                )
            losses.append(float(loss.item()))
            batches += 1
            for key, value in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
            logits = outputs["out"]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            target = masks.detach().cpu().numpy().astype(np.int64)
            for p, t in zip(pred, target):
                update_confusion(confusion, p, t, num_classes)
    miou, ious, pixel_acc = class_iou_from_confusion(confusion)
    metrics = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "miou": float(miou),
        "pixel_accuracy": float(pixel_acc),
    }
    for key, value in part_sums.items():
        metrics[key] = float(value / max(1, batches))
    for idx, name in enumerate(CLASS_NAMES[:num_classes]):
        metrics[f"{name}_iou"] = float(ious[idx])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--max-epochs", type=int, default=0, help="0 = use trial epochs")
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    config = load_config(str(args.config))
    trial = json.loads(Path(args.trial_json).read_text(encoding="utf-8"))
    trial_name = str(trial.get("name", "centernet_v2"))
    log = setup_logger(exp_dir / "supervisor.log")

    warm_path = args.warm_start.resolve()
    actual = sha256(warm_path)
    if actual != str(args.warm_start_sha256):
        raise SystemExit(f"warm-start SHA-256 mismatch: {actual} != {args.warm_start_sha256}")
    log(f"warm-start checkpoint verified: {warm_path} sha256={actual}")

    dataset_dir = exp_dir / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        splits.setdefault(row.get("split", "train"), []).append(row)
    log(f"splits train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    if splits["test"]:
        raise SystemExit("test split must be absent from this view; it stays locked")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")

    split_dir = exp_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        (split_dir / f"{split}.txt").write_text(
            "".join(f"{r['sample_id']}\n" for r in split_rows), encoding="utf-8"
        )

    train_cfg = config["training"]
    object_cfg = dict(config.get("object_heads", {}))
    fusion_cfg = config.get("fusion", {})
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(fusion_cfg.get("radar_channels", 4))
    input_width, input_height = [int(v) for v in trial.get("input_size", train_cfg["input_size"])]

    set_reproducible_seeds(int(trial.get("training_seed", 20260826)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NativeFusionDataset(
        dataset_dir,
        splits["train"],
        object_rows,
        (input_width, input_height),
        object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
        geometric_augment=False,
    )
    val_ds = NativeFusionDataset(
        dataset_dir, splits["val"], object_rows, (input_width, input_height), object_cfg,
        augment_strength="off",
    )
    num_workers = int(trial.get("num_workers", 8))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(trial.get("persistent_workers", True)) and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(trial.get("prefetch_factor", 2))
    batch_size = int(trial.get("batch_size", 24))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = build_centernet_v2(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=bool(train_cfg.get("pretrained", True)),
    ).to(device)
    warm_state = torch.load(warm_path, map_location="cpu", weights_only=False)
    state = warm_state.get("model", warm_state)
    mapping = warm_start_from_v1(model, state)
    mapping["warm_start_checkpoint"] = str(warm_path)
    mapping["warm_start_sha256"] = actual
    mapping["total_v2_tensors"] = len(model.state_dict())
    save_json(exp_dir / "checkpoint_mapping_report.json", mapping)
    log(
        f"warm start: loaded={mapping['loaded_count']} new={mapping['new_count']} "
        f"incompatible={mapping['incompatible_count']} of {mapping['total_v2_tensors']} v2 tensors"
    )

    if bool(trial.get("freeze_bn", True)):
        _freeze_batch_norm(model)
        log("freeze_bn=True: backbone BatchNorm layers held in eval with frozen affine params.")

    param_groups, group_names, group_lrs = build_param_groups(model, trial)
    optimizer = torch.optim.AdamW(
        param_groups, lr=float(trial.get("lr_new", 3e-4)),
        weight_decay=float(trial.get("weight_decay", 1e-4)),
    )
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    save_json(
        exp_dir / "param_group_report.json",
        {
            "base_lrs": {g["group_name"]: g["lr"] for g in param_groups},
            "counts": {k: len(v) for k, v in group_names.items()},
            "tensors": group_names,
            "schedule": {
                "optimizer": "adamw",
                "weight_decay": float(trial.get("weight_decay", 1e-4)),
                "warmup_epochs": int(trial.get("lr_warmup_epochs", 1)),
                "scheduler": "cosine",
                "min_lr_ratio": float(trial.get("min_lr_ratio", 0.01)),
                "epochs": int(trial.get("epochs", 24)),
            },
        },
    )
    group_summary = {k: {"lr": group_lrs[k], "tensors": len(v)} for k, v in group_names.items()}
    log("LR groups: " + json.dumps(group_summary, sort_keys=True))

    class_weights = None
    if trial.get("class_loss_weights"):
        class_weights = torch.tensor(
            [float(w) for w in trial["class_loss_weights"]], dtype=torch.float32, device=device
        )
    loss_weights = dict(trial.get("loss_weights", {}))
    object_weights = dict(DEFAULT_OBJECT_WEIGHTS)
    object_weights.update(loss_weights.get("object", {}))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    loss_cfg = {
        "object_weights": object_weights,
        "segmentation_weight": float(loss_weights.get("segmentation", 0.4)),
        "object_total_weight": float(loss_weights.get("object_total", 1.0)),
        "class_weights": class_weights,
        "lovasz_weight": float(trial.get("lovasz_weight", 0.5)),
        "amp": amp_enabled,
    }
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    trial_dir = exp_dir / "checkpoints" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics" / f"{trial_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial", "epoch", "train_loss", "val_loss", "miou", "background_iou", "vehicle_iou",
        "person_iou", "pixel_accuracy", "seg_loss", "ce_loss", "lovasz_loss", "object_loss",
        "veh_object_loss", "per_object_loss", "veh_center_loss", "per_center_loss",
        "veh_offset_loss", "per_offset_loss", "veh_loc_loss", "per_loc_loss", "veh_dim_loss",
        "per_dim_loss", "veh_bbox2d_loss", "per_bbox2d_loss", "veh_positives", "per_positives",
        "lr_new", "lr_warm", "lr_backbone", "epoch_seconds",
        "cuda_max_memory_allocated_mib", "cuda_max_memory_reserved_mib", "timestamp",
    ]

    start_epoch = 0
    last_path = trial_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["training_epoch_index"]) + 1
        log(f"resumed from {last_path} at epoch index {start_epoch}")
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    max_epochs = int(trial.get("epochs", 24))
    run_until = int(args.max_epochs) if int(args.max_epochs or 0) > 0 else max_epochs
    warmup_epochs = int(trial.get("lr_warmup_epochs", 1))
    min_lr_ratio = float(trial.get("min_lr_ratio", 0.01))

    for epoch in range(start_epoch, run_until):
        reported_epoch = epoch + 1
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        multiplier = lr_multiplier(epoch, max_epochs, warmup_epochs, min_lr_ratio)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = float(base_lr) * float(multiplier)
        model.train()
        if bool(trial.get("freeze_bn", True)):
            _freeze_batch_norm(model)
        losses: List[float] = []
        for tensors, masks, targets in train_loader:
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = move(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, cache_enabled=False):
                outputs = model(tensors)
            with torch.autocast(device_type=device.type, enabled=False):
                loss, _ = compute_v2_losses(
                    outputs, masks, targets,
                    object_weights=loss_cfg["object_weights"],
                    segmentation_weight=loss_cfg["segmentation_weight"],
                    object_total_weight=loss_cfg["object_total_weight"],
                    class_weights=loss_cfg["class_weights"],
                    lovasz_weight=loss_cfg["lovasz_weight"],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        val_metrics = run_validation(model, val_loader, device, num_classes, loss_cfg)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "trial": trial_name,
            "epoch": reported_epoch,
            "train_loss": train_loss,
            "lr_new": float(optimizer.param_groups[0]["lr"]),
            "lr_warm": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else float("nan"),
            "lr_backbone": float(optimizer.param_groups[2]["lr"]) if len(optimizer.param_groups) > 2 else float("nan"),
            "epoch_seconds": float(time.monotonic() - started),
            "cuda_max_memory_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if device.type == "cuda" else float("nan")
            ),
            "cuda_max_memory_reserved_mib": (
                float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
                if device.type == "cuda" else float("nan")
            ),
            "timestamp": utc_iso(),
        }
        for key in fieldnames:
            if key in row:
                continue
            row[key] = val_metrics.get(key, float("nan"))
        with metrics_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

        payload = {
            "model": model.state_dict(),
            "epoch": reported_epoch,
            "training_epoch_index": epoch,
            "trial": trial,
            "config": config,
            "input_size": [input_width, input_height],
            "radar_channels": radar_channels,
            "num_classes": num_classes,
            "object_class_names": ["vehicle", "person"],
            "vehicle_stride": 4,
            "person_stride": 2,
            "head_arch": trial_name,
            "warm_start_checkpoint": str(warm_path),
            "warm_start_sha256": actual,
            "model_task": "segmentation_plus_native_dual_stride_object_localization",
            "val_metrics": val_metrics,
        }
        epoch_path = trial_dir / f"epoch_{reported_epoch:03d}.pt"
        if epoch_path.exists():
            raise FileExistsError(f"refusing to overwrite {epoch_path}")
        torch.save(payload, epoch_path)
        torch.save(
            {**payload, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
            last_path,
        )
        log(
            f"{trial_name} epoch={reported_epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} miou={val_metrics['miou']:.4f} "
            f"vehicle_iou={val_metrics.get('vehicle_iou', float('nan')):.4f} "
            f"person_iou={val_metrics.get('person_iou', float('nan')):.4f} "
            f"veh_center={val_metrics.get('veh_center_loss', float('nan')):.4f} "
            f"per_center={val_metrics.get('per_center_loss', float('nan')):.4f} "
            f"secs={row['epoch_seconds']:.1f} peak_mib={row['cuda_max_memory_allocated_mib']:.0f}"
        )

    save_json(
        trial_dir / "trial_summary.json",
        {"trial": trial, "epochs_completed": run_until, "updated_at": utc_iso()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
