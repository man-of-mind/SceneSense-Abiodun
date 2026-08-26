#!/usr/bin/env python3
"""One real-batch q=0 launch check and mapped warm-start snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
ABIODUN = PKG_ROOT.parent
for path in (str(PKG_ROOT), str(ABIODUN)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pole_lraspp_multimodal_fusion import train_fusion as tf  # noqa: E402
from pole_lraspp_multimodal_fusion.common import CLASS_NAMES  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial-json", required=True, type=Path)
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    config = tf.load_config(str(args.config))
    train_cfg = config["training"]
    object_cfg = tf._deep_merge_dicts(dict(config.get("object_heads", {})), trial["object_heads"])
    object_names = list(object_cfg.get("object_classes", ["vehicle", "person"]))
    width, height = [int(value) for value in trial["input_size"]]
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(config.get("fusion", {}).get("radar_channels", 4))
    device = torch.device("cuda")

    rows = tf.read_manifest(exp_dir / "dataset" / "manifest.csv")
    object_rows = tf.load_object_boxes(exp_dir / "dataset" / "object_boxes.csv")
    splits = tf.split_rows(rows)
    dataset = tf.FusionPoleMultiTaskDataset(
        exp_dir / "dataset",
        splits["train"],
        object_rows,
        (width, height),
        object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(trial["batch_size"]),
        shuffle=True,
        num_workers=int(trial.get("num_workers", 4)),
        pin_memory=True,
    )

    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        init_checkpoint=str(trial["init_rgb_checkpoint"]),
        object_channels=len(object_names) + object_reg_channels(bool(object_cfg["predict_bbox2d"])),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(object_cfg["fuse_low_feature"]),
        head_arch=str(object_cfg["head_arch"]),
        use_coordconv=bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(object_cfg["predict_bbox2d"]),
        use_groundplane_prior=bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(object_cfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    load_report = tf._load_object_head_checkpoint(
        model, str(trial["init_object_checkpoint"]), device=device
    )

    tensors, masks, targets = next(iter(loader))
    tensors = tensors.to(device)
    masks = masks.to(device)
    targets = tf._move_object_targets(targets, device)
    model.train()
    tf._freeze_batch_norm(model)
    model.zero_grad(set_to_none=True)
    class_weights = torch.tensor(trial["class_loss_weights"], dtype=torch.float32, device=device)
    loss_weights = tf._deep_merge_dicts(dict(train_cfg.get("loss_weights", {})), trial["loss_weights"])
    with torch.autocast(device_type="cuda", enabled=bool(train_cfg.get("amp", True)), cache_enabled=False):
        loss, parts, _ = tf.compute_losses(
            model,
            tensors,
            masks,
            targets,
            num_classes,
            loss_weights,
            class_weights=class_weights,
            lovasz_weight=float(trial["lovasz_weight"]),
            feature_drop_fraction=0.0,
        )
    loss.backward()

    gradients = {}
    failures = []
    branches = {
        "vehicle_head": model.object_head.vehicle_heatmap_head,
        "person_head": model.object_head.person_heatmap_head,
        "regression_head": model.object_head.regression_head,
    }
    for name, branch in branches.items():
        grad = branch.weight.grad
        norm = float(grad.detach().float().norm().item()) if grad is not None else 0.0
        finite = bool(grad is not None and torch.isfinite(grad).all().item())
        gradients[name] = {"norm": norm, "finite": finite}
        if not finite or norm <= 0.0:
            failures.append(f"{name}: norm={norm} finite={finite}")

    with torch.no_grad():
        features = model.backbone(tensors[:1])
        object_input = model._object_input(features)
    report = {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "q": 0.0,
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "gradients": gradients,
        "feature_resolution_hw": list(object_input.shape[-2:]),
        "object_output_resolution_hw": list(tensors.shape[-2:]),
        "low_high_fused": bool(model.fuse_low_into_object_head),
        "warm_start_load": load_report,
        "positive_loss": {
            key: value
            for key, value in parts.items()
            if key.startswith("pos_")
        },
    }
    if not report["loss_finite"]:
        failures.append("non-finite total loss")
        report["verdict"] = "FAIL"

    out_path = exp_dir / "launch_check.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not failures:
        source = torch.load(
            str(trial["init_object_checkpoint"]), map_location="cpu", weights_only=False
        )
        mapped = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": 0,
            "warm_start_untrained": True,
            "trial": trial,
            "config": config,
            "input_size": [width, height],
            "radar_channels": radar_channels,
            "object_channels": len(object_names) + object_reg_channels(bool(object_cfg["predict_bbox2d"])),
            "object_predict_bbox2d": bool(object_cfg["predict_bbox2d"]),
            "object_class_names": object_names,
            "fuse_low_into_object_head": bool(object_cfg["fuse_low_feature"]),
            "object_head_arch": str(object_cfg["head_arch"]),
            "object_use_coordconv": bool(object_cfg.get("use_coordconv", False)),
            "object_head_depth": int(object_cfg.get("head_depth", 2)),
            "object_use_groundplane_prior": bool(object_cfg.get("use_groundplane_prior", False)),
            "object_groundplane_params": dict(object_cfg.get("groundplane_params", {}) or {}),
            "class_names": CLASS_NAMES[:num_classes],
            "source_checkpoint_epoch": source.get("epoch") if isinstance(source, dict) else None,
            "warm_start_load_report": load_report,
            "model_task": "segmentation_plus_learned_object_localization",
        }
        checkpoint_dir = exp_dir / "checkpoints" / str(trial["name"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        warm_path = checkpoint_dir / "warm_start_mapped.pt"
        if warm_path.exists():
            raise FileExistsError(f"refusing to overwrite mapped warm start: {warm_path}")
        torch.save(mapped, warm_path)
        report["mapped_warm_start"] = str(warm_path)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
