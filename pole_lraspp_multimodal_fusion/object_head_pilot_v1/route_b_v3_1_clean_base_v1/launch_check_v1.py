#!/usr/bin/env python3
"""Run the sole real q=0 AMP launch batch and verify all three model branches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
if str(FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSION_ROOT))

from pole_lraspp_multimodal_fusion import train_fusion as trainer  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402
from runtime_v1 import install  # noqa: E402


def branch_gradient(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [parameter.grad.detach().float() for parameter in module.parameters() if parameter.requires_grad and parameter.grad is not None]
    norm = math.sqrt(sum(float(torch.sum(gradient * gradient).item()) for gradient in gradients)) if gradients else 0.0
    finite = bool(gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients))
    return {"norm": norm, "finite": finite, "gradient_tensors": len(gradients)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    install()
    experiment = args.experiment.resolve()
    output = experiment / "LAUNCH_CHECK.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    config = trainer.load_config(str(args.config))
    training = config["training"]
    object_cfg = trainer._deep_merge_dicts(dict(config.get("object_heads", {})), trial["object_heads"])
    object_names = tuple(object_cfg["object_classes"])
    width, height = (int(value) for value in trial["input_size"])
    rows = trainer.read_manifest(experiment / "dataset/manifest.csv")
    objects = trainer.load_object_boxes(experiment / "dataset/object_boxes.csv")
    splits = trainer.split_rows(rows)
    dataset = trainer.FusionPoleMultiTaskDataset(
        experiment / "dataset", splits["train"], objects, (width, height), object_cfg,
        augment_strength=str(trial["augment_strength"]), geometric_augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=int(trial["batch_size"]), shuffle=True,
        num_workers=int(trial["num_workers"]), pin_memory=True,
        persistent_workers=bool(trial["persistent_workers"]),
        prefetch_factor=int(trial["prefetch_factor"]),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    model = build_multitask_fusion_lraspp(
        num_classes=int(training.get("num_classes", 3)),
        radar_channels=int(config.get("fusion", {}).get("radar_channels", 4)),
        pretrained=False, init_checkpoint=str(trial["init_rgb_checkpoint"]),
        object_channels=len(object_names) + object_reg_channels(bool(object_cfg["predict_bbox2d"])),
        object_hidden_channels=int(object_cfg["hidden_channels"]),
        fuse_low_into_object_head=bool(object_cfg["fuse_low_feature"]),
        head_arch=str(object_cfg["head_arch"]), use_coordconv=bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(object_cfg["head_depth"]), predict_bbox2d=bool(object_cfg["predict_bbox2d"]),
        use_groundplane_prior=bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(object_cfg.get("groundplane_params", {}) or {}), device=device,
    ).to(device)
    load_report = trainer._load_object_head_checkpoint(model, str(trial["init_object_checkpoint"]), device=device)
    tensors, masks, targets = next(iter(loader))
    tensors, masks = tensors.to(device), masks.to(device)
    targets = trainer._move_object_targets(targets, device)
    model.train()
    trainer._freeze_batch_norm(model)
    model.zero_grad(set_to_none=True)
    class_weights = torch.tensor(trial["class_loss_weights"], dtype=torch.float32, device=device)
    loss_weights = trainer._deep_merge_dicts(dict(training.get("loss_weights", {})), trial["loss_weights"])
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", enabled=bool(training.get("amp", True)), cache_enabled=False):
        loss, parts, _logits = trainer.compute_losses(
            model, tensors, masks, targets, int(training.get("num_classes", 3)), loss_weights,
            class_weights=class_weights, lovasz_weight=float(trial["lovasz_weight"]),
            feature_drop_fraction=0.0,
        )
    loss.backward()
    gradients = {
        "backbone": branch_gradient(model.backbone),
        "classifier": branch_gradient(model.classifier),
        "object_head": branch_gradient(model.object_head),
    }
    failures = [name for name, item in gradients.items() if not item["finite"] or item["norm"] <= 0.0]
    if not bool(torch.isfinite(loss).item()):
        failures.append("nonfinite_loss")
    report = {
        "verdict": "PASS" if not failures else "FAIL", "failures": failures,
        "q": 0.0, "amp": bool(training.get("amp", True)), "autocast_cache_enabled": False,
        "batch_size": int(trial["batch_size"]), "num_workers": int(trial["num_workers"]),
        "loss": float(loss.detach().item()), "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients": gradients, "warm_start_load": load_report,
        "positive_objects": int(targets["gt_count"].sum().item()),
        "ignored_object_cells": int(targets["object_ignore_mask"].sum().item()),
        "ignored_segmentation_pixels": int(masks.eq(-100).sum().item()),
        "loss_parts": parts,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
