#!/usr/bin/env python3
"""Train only the three factorized-localization components for exactly 12 epochs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from losses_v2 import factorized_localization_loss  # noqa: E402
from model_v2 import (  # noqa: E402
    build_factorized_model, freeze_for_localization, load_native_warm_start,
    localization_parameters, parameter_report,
)
from targets_v2 import FactorizedLocalizationDataset  # noqa: E402

CHECKPOINT_EPOCHS = (4, 8, 12)
METRIC_FIELDS = (
    "epoch", "train_loss", "log_depth_loss", "projected_center_offset_loss",
    "local_xy_endpoint_loss", "vehicle_positive_cells", "person_positive_cells",
    "learning_rate", "batches", "epoch_seconds", "cuda_max_memory_allocated_mib",
    "cuda_max_memory_reserved_mib", "created_utc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_experiment = args.contract_experiment.resolve()
    started = time.monotonic()

    if not (experiment / "LAUNCH_CHECKS_COMPLETE").is_file():
        raise RuntimeError("eight-check launch gate is absent")
    if config["epochs"] != 12 or tuple(config["checkpoint_epochs"]) != CHECKPOINT_EPOCHS:
        raise RuntimeError("fixed 12-epoch schedule drift")
    if config["batch_size"] != 16 or config["q"] != 0 or config["ae"] is not False:
        raise RuntimeError("fixed clean batch/q/AE recipe drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda")
    set_reproducible_seeds(int(config["training_seed"]))
    checkpoint = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != config["warm_start_sha256"]:
        raise RuntimeError("warm-start SHA drift after launch check")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    object_cfg = dict(payload["config"]["object_heads"])
    dataset_dir = contract_experiment / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if len(train_rows) != 6361 or len(val_rows) != 3345 or test_rows:
        raise RuntimeError(
            f"derived dataset split drift: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    dataset = FactorizedLocalizationDataset(
        dataset_dir, train_rows, object_rows, tuple(config["input_size"]), object_cfg,
        augment_strength=str(config["augment_strength"]),
        geometric_augment=bool(config["geometric_augment"]),
    )
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=int(config["prefetch_factor"]),
    )

    model = build_factorized_model(
        num_classes=int(payload["config"]["training"].get("num_classes", 3)),
        radar_channels=int(payload["radar_channels"]),
        hidden_channels=int(payload["object_hidden_channels"]),
        head_depth=int(payload["object_head_depth"]),
        localization_hidden=int(config["localization_hidden_channels"]), device=device,
    )
    warm_mapping = load_native_warm_start(model, checkpoint, device=device)
    freeze_for_localization(model)
    parameters = parameter_report(model)
    trainable = localization_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["localization_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=12, eta_min=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))

    checkpoint_dir = experiment / "checkpoints" / str(config["name"])
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir = experiment / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = metrics_dir / "training_metrics.csv"
    with metrics_path.open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()
    write_json_x(experiment / "TRAINING_STARTED.json", {
        "schema": "route_b_v3_1_factorized_localization_training_started_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "warm_start": warm_mapping, "parameters": parameters,
        "train_frames": len(train_rows), "validation_frames_not_used_for_training": len(val_rows),
        "only_trainable_components": config["trainable"],
    })

    rows_out: list[dict[str, Any]] = []
    peak_allocated = 0.0
    peak_reserved = 0.0
    for epoch in range(1, 13):
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        model.localization_trunk.train()
        model.log_depth_head.train()
        model.projected_3d_center_offset_head.train()
        sums = {key: 0.0 for key in (
            "total_loss", "log_depth_loss", "projected_center_offset_loss",
            "local_xy_endpoint_loss", "vehicle_positive_cells", "person_positive_cells",
        )}
        batches = 0
        lr = float(optimizer.param_groups[0]["lr"])
        for tensors, _masks, targets in loader:
            tensors = tensors.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", enabled=scaler.is_enabled(),
                cache_enabled=bool(config["autocast_cache_enabled"]),
            ):
                outputs = model.localization_training_outputs(tensors)
            with torch.autocast(device_type="cuda", enabled=False):
                loss, parts = factorized_localization_loss(
                    outputs["localization"].float(), outputs["object"].float(), targets,
                    config["losses"],
                )
            if not math.isfinite(float(loss.detach().item())):
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batches + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for key in sums:
                sums[key] += float(parts[key])
        scheduler.step()
        allocated = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
        peak_allocated = max(peak_allocated, allocated)
        peak_reserved = max(peak_reserved, reserved)
        row = {
            "epoch": epoch, "train_loss": sums["total_loss"] / batches,
            "log_depth_loss": sums["log_depth_loss"] / batches,
            "projected_center_offset_loss": sums["projected_center_offset_loss"] / batches,
            "local_xy_endpoint_loss": sums["local_xy_endpoint_loss"] / batches,
            "vehicle_positive_cells": int(sums["vehicle_positive_cells"]),
            "person_positive_cells": int(sums["person_positive_cells"]),
            "learning_rate": lr, "batches": batches,
            "epoch_seconds": time.monotonic() - epoch_started,
            "cuda_max_memory_allocated_mib": allocated,
            "cuda_max_memory_reserved_mib": reserved,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows_out.append(row)
        with metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)
        if epoch in CHECKPOINT_EPOCHS:
            output = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            torch.save({
                "schema": "route_b_v3_1_factorized_localization_checkpoint_v2",
                "model": model.state_dict(), "epoch": epoch, "config": config,
                "native_checkpoint": str(checkpoint),
                "native_checkpoint_sha256": config["warm_start_sha256"],
                "position_decoder": "factorized_xyz",
                "legacy_xyz_channels_retained": True,
                "legacy_xyz_channels_trained": False,
                "input_size": list(config["input_size"]),
                "radar_channels": int(payload["radar_channels"]),
                "object_hidden_channels": int(payload["object_hidden_channels"]),
                "object_head_depth": int(payload["object_head_depth"]),
                "object_class_names": list(payload["object_class_names"]),
                "native_stride": int(payload["native_stride"]),
                "native_grid": list(payload["native_grid"]),
                "parameter_report": parameters,
            }, output)
        print(
            f"[factorized train] epoch={epoch}/12 loss={row['train_loss']:.6f} "
            f"depth={row['log_depth_loss']:.6f} offset={row['projected_center_offset_loss']:.6f} "
            f"xy={row['local_xy_endpoint_loss']:.6f} lr={lr:.8g}", flush=True,
        )

    checkpoint_paths = [checkpoint_dir / f"epoch_{epoch:03d}.pt" for epoch in CHECKPOINT_EPOCHS]
    unexpected = [path.name for path in checkpoint_dir.glob("epoch_*.pt") if path not in checkpoint_paths]
    gates = {
        "exactly_12_epoch_rows": [row["epoch"] for row in rows_out] == list(range(1, 13)),
        "checkpoint_epochs_exactly_4_8_12": all(path.is_file() for path in checkpoint_paths) and not unexpected,
        "all_losses_finite": all(math.isfinite(float(row["train_loss"])) for row in rows_out),
        "no_early_stopping": len(rows_out) == 12,
        "q0_no_ae_batch16": config["q"] == 0 and config["ae"] is False and config["batch_size"] == 16,
        "only_new_parameters_trainable": parameters["model_total"]["trainable"]
        == sum(parameter.numel() for parameter in trainable),
    }
    if not all(gates.values()):
        raise RuntimeError(f"training completion gate failure: {gates}")
    result = {
        "schema": "route_b_v3_1_factorized_localization_training_complete_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gates": gates,
        "epochs_completed": 12, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "checkpoint_sha256": {path.name: sha256(path) for path in checkpoint_paths},
        "parameter_report": parameters, "warm_start_mapping": warm_mapping,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
    }
    write_json_x(experiment / "TRAINING_COMPLETE.json", result)
    (experiment / "TRAINING_COMPLETE").write_text("EXACTLY_12_EPOCHS_COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
