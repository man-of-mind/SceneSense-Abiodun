#!/usr/bin/env python3
"""Stage H + Stage J training for the native stride-4 object head.

A dedicated loop is required for exactly one reason: Stage J needs two parameter
groups at different learning rates (inherited backbone/classifier at 1e-5, native
object decoder/head at 1e-4), and the shared trainer builds a single flat group. Every
other ingredient - dataset, augmentation, AMP, frozen BN, segmentation objective,
object regression terms and their weights - is the frozen v3.1 recipe, reused by
import rather than reimplemented.

The schedule is fixed before the first step and is not changed after training starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for _path in (str(ROOT), str(FUSION_ROOT), str(BASE_PKG)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    class_iou_from_confusion, load_config, read_manifest, set_reproducible_seeds,
    update_confusion, utc_iso,
)
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from losses_v1 import (  # noqa: E402
    native_object_loss, segmentation_loss,
)
from model_v1 import (  # noqa: E402
    NATIVE_GRID, OUTPUT_CHANNELS, build_native_grid_model, load_warm_start, parameter_report,
)
from targets_v1 import (  # noqa: E402
    NativeGridDataset,
)

METRIC_FIELDS = [
    "trial", "epoch", "stage", "train_loss", "val_loss", "miou", "vehicle_iou", "person_iou",
    "pixel_accuracy", "seg_loss", "ce_loss", "lovasz_loss", "object_loss", "center_loss",
    "loc_loss", "dim_loss", "yaw_loss", "parked_loss", "radar_support_loss", "bbox2d_loss",
    "offset_loss", "positive_cells", "gt_objects", "lr_inherited", "lr_object",
    "epoch_seconds", "cuda_max_memory_allocated_mib", "cuda_max_memory_reserved_mib", "timestamp",
]


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def object_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """The native object decoder and head: trunk, new upsampler, all output branches."""
    return list(model.object_head.parameters())


def inherited_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return list(model.backbone.parameters()) + list(model.classifier.parameters())


def freeze_inherited_batch_norm(model: torch.nn.Module) -> None:
    """Frozen v3.1 recipe: every BatchNorm stays in eval mode with frozen affine."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def apply_stage(model: torch.nn.Module, stage: Dict[str, Any], freeze_bn: bool) -> None:
    frozen = bool(stage["freeze_backbone"]), bool(stage["freeze_classifier"])
    for param in model.backbone.parameters():
        param.requires_grad = not frozen[0]
    for param in model.classifier.parameters():
        param.requires_grad = not frozen[1]
    for param in model.object_head.parameters():
        param.requires_grad = True
    if freeze_bn:
        freeze_inherited_batch_norm(model)


def stage_train_mode(model: torch.nn.Module, stage: Dict[str, Any], freeze_bn: bool) -> None:
    model.train()
    if bool(stage["freeze_backbone"]):
        model.backbone.eval()
    if bool(stage["freeze_classifier"]):
        model.classifier.eval()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def compute_batch_losses(
    model: torch.nn.Module, tensors: torch.Tensor, masks: torch.Tensor,
    targets: Dict[str, torch.Tensor], loss_weights: Dict[str, Any],
    class_weights: Optional[torch.Tensor], lovasz_weight: float,
):
    outputs = model(tensors, feature_drop_fraction=0.0)
    seg_loss, seg_parts, seg_logits = segmentation_loss(
        outputs["out"], masks, class_weights=class_weights, lovasz_weight=lovasz_weight
    )
    # Regression terms run in FP32 even under AMP: small-magnitude smooth-L1/BCE
    # targets are unstable in FP16 and can silently zero the object-head gradient.
    with torch.autocast(device_type=tensors.device.type, enabled=False):
        object_loss, object_parts = native_object_loss(
            outputs["object"].float(), targets, loss_weights.get("object", {})
        )
    total = (float(loss_weights.get("segmentation", 0.3)) * seg_loss
             + float(loss_weights.get("object_total", 1.0)) * object_loss)
    parts = {**seg_parts, **object_parts, "object_loss": float(object_loss.detach().item())}
    return total, parts, seg_logits


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, loss_weights, class_weights, lovasz_weight):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    losses: List[float] = []
    sums: Dict[str, float] = {}
    batches = 0
    gt_objects = 0
    for tensors, masks, targets in loader:
        tensors = tensors.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
        loss, parts, logits = compute_batch_losses(
            model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
        )
        losses.append(float(loss.item()))
        batches += 1
        gt_objects += int(targets["gt_count"].sum().item())
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        prediction = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        target = masks.detach().cpu().numpy().astype(np.int64)
        for one_prediction, one_target in zip(prediction, target):
            update_confusion(confusion, one_prediction, one_target, num_classes)
    miou, ious, pixel_accuracy = class_iou_from_confusion(confusion)
    metrics = {"loss": float(np.mean(losses)) if losses else float("nan"),
               "miou": miou, "pixel_accuracy": pixel_accuracy, "gt_objects": float(gt_objects)}
    for key, value in sums.items():
        metrics[key] = float(value / max(1, batches))
    for index, name in enumerate(("background", "vehicle", "person")[:num_classes]):
        metrics[f"{name}_iou"] = float(ious[index])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    config = load_config(args.config.resolve())
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    started = time.monotonic()

    train_cfg = config["training"]
    object_cfg = dict(config.get("object_heads", {}))
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(config.get("fusion", {}).get("radar_channels", 4))
    input_width, input_height = (int(value) for value in trial["input_size"])
    trial_name = str(trial["name"])

    write_json_x(experiment / "TRAINING_STARTED.json", {
        "schema": "route_b_v3_1_native_grid_training_started_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial": trial, "config": str(args.config.resolve()),
        "single_authorized_training_launch": True,
    })

    try:
        set_reproducible_seeds(int(trial["training_seed"]))
        device = torch.device("cuda")

        dataset_dir = experiment / "dataset"
        rows = read_manifest(dataset_dir / "manifest.csv")
        object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
        train_rows = [row for row in rows if row.get("split") == "train"]
        val_rows = [row for row in rows if row.get("split") == "val"]
        test_rows = [row for row in rows if row.get("split") == "test"]
        if len(train_rows) != 6361 or len(val_rows) != 3345 or test_rows:
            raise RuntimeError(
                f"v3.1 view mismatch: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
            )

        loader_kwargs = {"num_workers": int(trial["num_workers"]), "pin_memory": True,
                         "persistent_workers": bool(trial["persistent_workers"]),
                         "prefetch_factor": int(trial["prefetch_factor"])}
        train_loader = DataLoader(
            NativeGridDataset(dataset_dir, train_rows, object_rows, (input_width, input_height),
                              object_cfg, augment_strength=str(trial["augment_strength"]),
                              geometric_augment=bool(trial["geometric_augment"])),
            batch_size=int(trial["batch_size"]), shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(
            NativeGridDataset(dataset_dir, val_rows, object_rows, (input_width, input_height),
                              object_cfg, augment_strength="off"),
            batch_size=int(trial["batch_size"]), shuffle=False, **loader_kwargs)

        model = build_native_grid_model(
            num_classes=num_classes, radar_channels=radar_channels,
            hidden_channels=int(object_cfg.get("hidden_channels", 128)),
            head_depth=int(object_cfg.get("head_depth", 3)), device=device)
        warm_start = load_warm_start(model, ROOT / trial["warm_start_checkpoint"], device=device)
        write_json_x(experiment / "WARM_START_MAPPING.json", warm_start)

        loss_weights = dict(trial["loss_weights"])
        lovasz_weight = float(trial["lovasz_weight"])
        class_weights = torch.tensor([float(v) for v in trial["class_loss_weights"]],
                                     dtype=torch.float32, device=device)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(trial["amp"]))
        freeze_bn = bool(trial["freeze_bn"])
        stages = {int(epoch): stage for stage in trial["stages"]
                  for epoch in range(int(stage["first_epoch"]), int(stage["last_epoch"]) + 1)}
        checkpoint_epochs = set(int(value) for value in trial["checkpoint_epochs"])
        checkpoint_dir = experiment / "checkpoints" / trial_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = experiment / "metrics" / f"{trial_name}_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

        # One optimizer with two named groups, built once. Stage H simply drives the
        # inherited group's LR to zero and freezes it; Stage J re-enables it at 1e-5.
        optimizer = torch.optim.AdamW(
            [{"params": inherited_parameters(model), "lr": 0.0, "name": "inherited"},
             {"params": object_parameters(model), "lr": 0.0, "name": "object"}],
            lr=0.0, weight_decay=float(trial["weight_decay"]))

        stage_parameters: Dict[str, Any] = {}
        peak_allocated = peak_reserved = 0.0
        for epoch in range(1, int(trial["total_epochs"]) + 1):
            stage = stages[epoch]
            apply_stage(model, stage, freeze_bn)
            lr_inherited = float(stage["lr"]["inherited"])
            lr_object = float(stage["lr"]["object"])
            for group in optimizer.param_groups:
                group["lr"] = lr_inherited if group["name"] == "inherited" else lr_object
            if stage["name"] not in stage_parameters:
                stage_parameters[stage["name"]] = {
                    "first_epoch": int(stage["first_epoch"]), "last_epoch": int(stage["last_epoch"]),
                    "lr_inherited": lr_inherited, "lr_object": lr_object,
                    "parameters": parameter_report(model),
                }

            epoch_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            stage_train_mode(model, stage, freeze_bn)
            losses: List[float] = []
            for tensors, masks, targets in train_loader:
                tensors = tensors.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled(), cache_enabled=False):
                    loss, _, _ = compute_batch_losses(
                        model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))

            val_metrics = evaluate(model, val_loader, device, num_classes,
                                   loss_weights, class_weights, lovasz_weight)
            train_loss = float(np.mean(losses)) if losses else float("nan")
            allocated = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            reserved = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
            peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
            row = {
                "trial": trial_name, "epoch": epoch, "stage": stage["name"],
                "train_loss": train_loss, "val_loss": val_metrics["loss"],
                "miou": val_metrics["miou"], "vehicle_iou": val_metrics.get("vehicle_iou"),
                "person_iou": val_metrics.get("person_iou"), "pixel_accuracy": val_metrics["pixel_accuracy"],
                "gt_objects": val_metrics.get("gt_objects"),
                "lr_inherited": lr_inherited, "lr_object": lr_object,
                "epoch_seconds": float(time.monotonic() - epoch_started),
                "cuda_max_memory_allocated_mib": allocated,
                "cuda_max_memory_reserved_mib": reserved, "timestamp": utc_iso(),
            }
            for key in ("seg_loss", "ce_loss", "lovasz_loss", "object_loss", "center_loss",
                        "loc_loss", "dim_loss", "yaw_loss", "parked_loss", "radar_support_loss",
                        "bbox2d_loss", "offset_loss", "positive_cells"):
                row[key] = val_metrics.get(key, float("nan"))
            with metrics_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)
            if not all(np.isfinite([train_loss, float(val_metrics["loss"])])):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}")

            if epoch in checkpoint_epochs:
                path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite {path}")
                torch.save({
                    "model": model.state_dict(), "epoch": epoch, "stage": stage["name"],
                    "trial": trial, "config": config,
                    "input_size": [input_width, input_height], "radar_channels": radar_channels,
                    "object_class_names": list(object_cfg["object_classes"]),
                    "object_output_channels": OUTPUT_CHANNELS,
                    "native_stride": int(object_cfg["native_stride"]),
                    "native_grid": list(NATIVE_GRID),
                    "object_hidden_channels": int(object_cfg.get("hidden_channels", 128)),
                    "object_head_depth": int(object_cfg.get("head_depth", 3)),
                    "warm_start": warm_start["checkpoint"],
                    "warm_start_sha256": trial["warm_start_sha256"],
                    "model_task": "segmentation_plus_native_grid_object_localization",
                }, path)
            print(f"[train] epoch={epoch} stage={stage['name']} train_loss={train_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} miou={val_metrics['miou']:.4f} "
                  f"center={val_metrics.get('center_loss', float('nan')):.4f} "
                  f"offset={val_metrics.get('offset_loss', float('nan')):.4f} "
                  f"loc={val_metrics.get('loc_loss', float('nan')):.4f}", flush=True)

        missing = [str(checkpoint_dir / f"epoch_{value:03d}.pt") for value in sorted(checkpoint_epochs)
                   if not (checkpoint_dir / f"epoch_{value:03d}.pt").is_file()]
        if missing:
            raise RuntimeError(f"missing checkpoints: {missing}")
        result = {
            "schema": "route_b_v3_1_native_grid_training_complete_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "trial": trial_name, "epochs_completed": int(trial["total_epochs"]),
            "checkpoint_epochs": sorted(checkpoint_epochs),
            "stage_parameters": stage_parameters,
            "warm_start": warm_start,
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        }
        write_json_x(experiment / "TRAINING_COMPLETE.json", result)
        (experiment / "PHASE_C_TRAINING_COMPLETE").write_text("TRAINING_COMPLETE\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "warm_start"}, indent=2), flush=True)
        return 0
    except Exception as exc:
        (experiment / "TERMINAL_VERDICT.txt").write_text("LRASPP_NATIVE_GRID_RUNTIME_FAILURE\n", encoding="utf-8")
        write_json_x(experiment / "training_failure.json", {
            "terminal": "LRASPP_NATIVE_GRID_RUNTIME_FAILURE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
