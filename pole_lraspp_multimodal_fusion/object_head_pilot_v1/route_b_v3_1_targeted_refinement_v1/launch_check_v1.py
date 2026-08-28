#!/usr/bin/env python3
"""Minimum launch check for the one class-balanced clean-q continuation.

One real q=0 AMP batch. Verifies finite loss, finite nonzero gradients in the
backbone / segmentation classifier / object head, the verified warm-start SHA, and
that the class-balanced positive-center path is genuinely active with both the
vehicle and person positive classes present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for path in (str(FUSION_ROOT), str(BASE_PKG)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion import train_fusion as trainer  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402
from runtime_v1 import install  # noqa: E402

EXPECTED_WARM_START_SHA = "88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d"
OBJECT_CLASS_ORDER = ("vehicle", "person")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def branch_gradient(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [
        parameter.grad.detach().float()
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    norm = math.sqrt(sum(float(torch.sum(g * g).item()) for g in gradients)) if gradients else 0.0
    finite = bool(gradients and all(torch.isfinite(g).all().item() for g in gradients))
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
    if tuple(object_names) != OBJECT_CLASS_ORDER:
        raise RuntimeError(f"unexpected object class order: {object_names}")

    warm_start = Path(trial["init_object_checkpoint"])
    warm_start_sha = sha256(warm_start)
    if warm_start_sha != EXPECTED_WARM_START_SHA or trial["init_rgb_checkpoint"] != str(warm_start):
        raise RuntimeError(f"warm-start checkpoint SHA mismatch: {warm_start_sha}")

    loss_weights = trainer._deep_merge_dicts(dict(training.get("loss_weights", {})), trial["loss_weights"])
    object_weights = loss_weights["object"]
    if not bool(object_weights.get("class_balanced_center", False)):
        raise RuntimeError("class_balanced_center is not enabled in the resolved loss weights")
    if bool(object_weights.get("pos_weight_enable", False)):
        raise RuntimeError("pos_weight_enable must remain false")

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

    # With pos_weight_enable=false the per-class statistics are only emitted when the
    # class-balanced macro-average branch is taken, so their presence proves the path
    # is live rather than merely configured.
    vehicle_positives = float(parts.get("pos_count_class0", 0.0))
    person_positives = float(parts.get("pos_count_class1", 0.0))
    classes_present = float(parts.get("pos_classes_present", 0.0))
    class_balanced_active = "pos_classes_present" in parts
    both_classes_reported = vehicle_positives > 0.0 and person_positives > 0.0 and classes_present == 2.0

    failures = [name for name, item in gradients.items() if not item["finite"] or item["norm"] <= 0.0]
    if not bool(torch.isfinite(loss).item()):
        failures.append("nonfinite_loss")
    if not class_balanced_active:
        failures.append("class_balanced_path_inactive")
    if not both_classes_reported:
        failures.append("both_positive_classes_not_reported")

    report = {
        "verdict": "PASS" if not failures else "FAIL", "failures": failures,
        "q": 0.0, "amp": bool(training.get("amp", True)), "autocast_cache_enabled": False,
        "batch_size": int(trial["batch_size"]), "num_workers": int(trial["num_workers"]),
        "epochs": int(trial["epochs"]), "checkpoint_every_epochs": int(trial["checkpoint_every_epochs"]),
        "warm_start_checkpoint": str(warm_start), "warm_start_sha256": warm_start_sha,
        "warm_start_sha_verified": True,
        "class_balanced_center": True, "pos_weight_enable": False,
        "class_balanced_path_active": class_balanced_active,
        "positive_classes_present": classes_present,
        "vehicle_positive_cells": vehicle_positives,
        "person_positive_cells": person_positives,
        "vehicle_person_positive_ratio": (
            vehicle_positives / person_positives if person_positives > 0 else None
        ),
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
