#!/usr/bin/env python3
"""Single fixed 12-epoch Faster R-CNN radar-ROI training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, HERE.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes

from dataset_v1 import RouteBFasterRCNNDataset, detection_collate
from model_v1 import build_model, freeze_batch_norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_create(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_checkpoint_create(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def move_targets(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def build_groups(model: torch.nn.Module, config: Dict) -> List[Dict]:
    groups = {"new": [], "detector": [], "backbone": []}
    names = {key: [] for key in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("detector.backbone."):
            key = "backbone"
        elif name.startswith(("radar_encoder.", "radar_roi_embed.", "roi_localization_head.", "segmentation_decoder.", "detector.roi_heads.box_predictor.")):
            key = "new"
        else:
            key = "detector"
        groups[key].append(parameter)
        names[key].append(name)
    lrs = {
        "new": float(config["lr_new"]),
        "detector": float(config["lr_detector"]),
        "backbone": float(config["lr_backbone"]),
    }
    return ([{"params": groups[key], "lr": lrs[key], "group_name": key} for key in ("new", "detector", "backbone")], names, lrs)


def lr_multiplier(epoch: int, config: Dict) -> float:
    epochs = int(config["epochs"])
    warmup = int(config["warmup_epochs"])
    minimum = float(config["min_lr_ratio"])
    if epoch <= warmup:
        return float(epoch) / max(1.0, float(warmup))
    progress = float(epoch - warmup) / max(1.0, float(epochs - warmup))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_backbone_trainable(model: torch.nn.Module, enabled: bool) -> None:
    for parameter in model.detector.backbone.parameters():
        parameter.requires_grad = bool(enabled)


def notify_completion(experiment_dir: Path, status: str) -> None:
    payload = {"status": status, "time_unix": time.time(), "experiment_dir": str(experiment_dir)}
    target = experiment_dir / "TRAINING_COMPLETION_NOTIFICATION.json"
    if not target.exists():
        write_json_create(target, payload)
    try:
        subprocess.run(["notify-send", "Route B Faster R-CNN training", f"{status}: {experiment_dir.name}"], check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = args.experiment_dir.resolve()
    dataset_dir = experiment_dir / "dataset"
    if not dataset_dir.exists():
        raise SystemExit(f"dataset view missing: {dataset_dir}")
    checkpoint_dir = experiment_dir / "checkpoints" / config["name"]
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise SystemExit(f"refusing nonempty checkpoint directory: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(dataset_dir / "manifest.csv")
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if (len(train_rows), len(val_rows), len(test_rows)) != (6600, 3588, 0):
        raise SystemExit(f"unexpected split counts {(len(train_rows), len(val_rows), len(test_rows))}")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    width, height = map(int, config["input_size"])
    dataset = RouteBFasterRCNNDataset(
        dataset_dir, train_rows, object_rows, (width, height), training=True,
        flip_probability=float(config["flip_probability"]),
    )

    set_reproducible_seeds(int(config["training_seed"]))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=detection_collate,
    )
    model = build_model(pretrained=True, input_size=(width, height)).to(device)
    param_groups, group_names, group_lrs = build_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(config["weight_decay"]))
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]) and amp_dtype == torch.float16)
    write_json_create(
        experiment_dir / "param_groups.json",
        {"base_lrs": group_lrs, "parameter_names": group_names, "counts": {key: len(value) for key, value in group_names.items()}},
    )
    write_json_create(experiment_dir / "coco_mapping.json", model.coco_mapping)

    metrics_path = experiment_dir / "training_metrics.csv"
    metrics_handle = metrics_path.open("x", newline="", encoding="utf-8")
    fieldnames = ["epoch", "phase", "lr_new", "lr_detector", "lr_backbone", "loss", "seconds", "peak_allocated_mib", "peak_reserved_mib"]
    writer = csv.DictWriter(metrics_handle, fieldnames=fieldnames)
    writer.writeheader()
    start = time.monotonic()
    status = "runtime_failure"
    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            phase = "head_adaptation" if epoch <= int(config["head_adaptation_epochs"]) else "joint_finetune"
            set_backbone_trainable(model, phase == "joint_finetune")
            model.train()
            freeze_batch_norm(model.detector)
            multiplier = lr_multiplier(epoch, config)
            for group in optimizer.param_groups:
                group["lr"] = float(group_lrs[group["group_name"]]) * multiplier
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.monotonic()
            loss_sum = 0.0
            batches = 0
            for rgb, radar, targets, _metadata in loader:
                rgb = [value.to(device, non_blocking=True) for value in rgb]
                radar = [value.to(device, non_blocking=True) for value in radar]
                targets = move_targets(targets, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(config["amp"])):
                    output = model(rgb, radar, targets)
                    loss = sum(output["losses"].values())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss epoch={epoch} batch={batches}: {float(loss)}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach().item())
                batches += 1
            elapsed = time.monotonic() - epoch_start
            row = {
                "epoch": epoch,
                "phase": phase,
                "lr_new": optimizer.param_groups[0]["lr"],
                "lr_detector": optimizer.param_groups[1]["lr"],
                "lr_backbone": optimizer.param_groups[2]["lr"],
                "loss": loss_sum / max(1, batches),
                "seconds": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
            }
            writer.writerow(row)
            metrics_handle.flush()
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "input_size": [width, height],
                "object_class_names": list(("vehicle", "person")),
                "coco_mapping": model.coco_mapping,
                "training_runtime_seconds": time.monotonic() - start,
                "peak_allocated_mib": row["peak_allocated_mib"],
                "peak_reserved_mib": row["peak_reserved_mib"],
            }
            target = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint_create(target, checkpoint)
            print(json.dumps({"event": "epoch_complete", **row, "checkpoint": str(target)}), flush=True)
        status = "complete"
        write_json_create(
            experiment_dir / "training_runtime.json",
            {
                "status": status,
                "runtime_seconds": time.monotonic() - start,
                "epochs": int(config["epochs"]),
                "batch_size": int(config["batch_size"]),
                "device": torch.cuda.get_device_name(device),
                "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in list(csv.DictReader(metrics_path.open()))),
                "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in list(csv.DictReader(metrics_path.open()))),
            },
        )
    finally:
        metrics_handle.close()
        notify_completion(experiment_dir, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
