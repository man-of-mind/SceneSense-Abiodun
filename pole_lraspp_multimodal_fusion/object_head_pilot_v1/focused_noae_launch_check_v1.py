#!/usr/bin/env python3
"""One-batch launch check for focused_noae_v1. Not a test suite.

Checks exactly four things on a single real training batch, then exits:
  C1  loss is finite at q=0.00 and at one degraded anchor (q=0.90);
  C2  the shared object head receives NONZERO gradient at BOTH anchors
      (the AMP cast-cache zero-gradient regression fixed in 59f031a);
  C3  positive mass is class-balanced: each present class's renormalised
      positive weights average to 1.0, and the center positive term is the
      macro-average of the per-class means (vehicle cell count cannot
      dominate person learning);
  C4  no positive weight exceeds the registered cap of 4.0.

Exit 0 = launch. Nonzero = do not launch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
ABIODUN = PKG_ROOT.parent
for p in (str(PKG_ROOT), str(ABIODUN)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from object_head_pilot_v1.target_variants_v1 import install  # noqa: E402
from pole_lraspp_multimodal_fusion import train_fusion as tf  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402

CAP = 4.0
ANCHOR_DEGRADED = 0.90


def main() -> int:
    cfg_path = HERE / "configs" / "route_b_noae_precision_pilot_v1.yaml"
    trial = json.loads((HERE / "configs" / "focused_noae_v1.json").read_text())
    exp_dir = Path((HERE / "FOCUSED_EXP_DIR.txt").read_text().strip())

    install(trial["object_heads"].get("vehicle_heatmap_radius_cap_px"))

    config = tf.load_config(str(cfg_path))
    train_cfg = config["training"]
    dataset_dir = exp_dir / "dataset"
    rows = tf.read_manifest(dataset_dir / "manifest.csv")
    object_rows = tf.load_object_boxes(dataset_dir / "object_boxes.csv")
    splits = tf.split_rows(rows)

    object_cfg = tf._deep_merge_dicts(dict(config.get("object_heads", {})), trial.get("object_heads"))
    object_class_names = list(object_cfg.get("object_classes", ["vehicle", "person"]))
    iw, ih = [int(v) for v in trial["input_size"]]
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(config.get("fusion", {}).get("radar_channels", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = tf.FusionPoleMultiTaskDataset(
        dataset_dir, splits["train"], object_rows, (iw, ih), object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    loader = DataLoader(ds, batch_size=int(trial["batch_size"]), shuffle=True, num_workers=4)

    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        init_checkpoint=str(trial["init_rgb_checkpoint"]),
        object_channels=len(object_class_names) + object_reg_channels(bool(object_cfg["predict_bbox2d"])),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(object_cfg.get("fuse_low_feature", False)),
        head_arch=str(object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(object_cfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params={},
        device=device,
    ).to(device)
    tf._load_object_head_checkpoint(model, str(trial["init_object_checkpoint"]), device=device)
    assert model.head_arch == "shared" and model.object_head is not None, "shared head required"
    model.train()
    tf._freeze_batch_norm(model)

    loss_weights = tf._deep_merge_dicts(dict(train_cfg.get("loss_weights", {})), trial.get("loss_weights"))
    class_weights = torch.tensor([float(v) for v in trial["class_loss_weights"]],
                                 dtype=torch.float32, device=device)

    tensors, masks, object_targets = next(iter(loader))
    tensors = tensors.to(device)
    masks = masks.to(device)
    object_targets = tf._move_object_targets(object_targets, device)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    final_conv = model.object_head[-1]
    report: dict = {"anchors": {}}
    failures: list[str] = []

    for q in (0.0, ANCHOR_DEGRADED):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled(), cache_enabled=False):
            loss, parts, _ = tf.compute_losses(
                model, tensors, masks, object_targets, num_classes, loss_weights,
                class_weights=class_weights, lovasz_weight=float(trial["lovasz_weight"]),
                feature_drop_fraction=float(q),
            )
        # Backward WITHOUT the loss scaler: we still exercise the autocast path that
        # carried the 59f031a zero-gradient bug (cache_enabled=False), but the measured
        # gradient is the true unscaled one. Scaling by GradScaler's initial 65536 would
        # overflow fp16 intermediates to inf and make "nonzero" meaningless.
        loss.backward()
        gw = final_conv.weight.grad
        gnorm = float(gw.detach().float().norm().item()) if gw is not None else 0.0
        grad_finite = bool(gw is not None and torch.isfinite(gw.detach()).all().item())
        entry = {
            "q": q,
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "object_head_grad_norm": gnorm,
            "object_head_grad_finite": grad_finite,
            "center_loss": parts.get("center_loss"),
            "pos_classes_present": parts.get("pos_classes_present"),
            "pos_max_weight": parts.get("pos_max_weight"),
        }
        for c in range(len(object_class_names)):
            for k in ("pos_mean_w_class", "pos_count_class", "pos_mean_loss_class"):
                if f"{k}{c}" in parts:
                    entry[f"{k}{c}"] = parts[f"{k}{c}"]
        report["anchors"][f"q={q:.2f}"] = entry

        # C1
        if not entry["loss_finite"]:
            failures.append(f"C1 q={q}: non-finite loss {entry['loss']}")
        # C2
        if not (gnorm > 0.0) or not grad_finite:
            failures.append(
                f"C2 q={q}: object-head final-conv grad norm={gnorm} finite={grad_finite}")
        # C3
        present = int(entry.get("pos_classes_present") or 0)
        if present < 1:
            failures.append(f"C3 q={q}: no object class had positives in this batch")
        for c in range(len(object_class_names)):
            mw = entry.get(f"pos_mean_w_class{c}")
            if mw is not None and abs(mw - 1.0) > 1e-3:
                failures.append(f"C3 q={q}: class{c} mean positive weight {mw} != 1.0")
        # C4
        mx = entry.get("pos_max_weight")
        if mx is None or mx > CAP + 1e-6:
            failures.append(f"C4 q={q}: max positive weight {mx} exceeds cap {CAP}")

    # C3 (macro-average): the center positive term must be the mean of the
    # per-class means, not the vehicle-dominated pooled mean.
    for tag, entry in report["anchors"].items():
        means = [entry[f"pos_mean_loss_class{c}"] for c in range(len(object_class_names))
                 if f"pos_mean_loss_class{c}" in entry]
        counts = [entry[f"pos_count_class{c}"] for c in range(len(object_class_names))
                  if f"pos_count_class{c}" in entry]
        if len(means) >= 2:
            macro = sum(means) / len(means)
            pooled = sum(m * n for m, n in zip(means, counts)) / max(1e-9, sum(counts))
            entry["macro_avg_pos_mean"] = macro
            entry["pooled_pos_mean"] = pooled
            entry["class_count_ratio"] = max(counts) / max(1e-9, min(counts))
            # center_loss = macro positive term + background term, so
            # center_loss - macro must be >= 0 (the background contribution).
            if entry["center_loss"] - macro < -1e-4:
                failures.append(f"C3 {tag}: center_loss {entry['center_loss']} below macro-average {macro}")

    report["cap"] = CAP
    report["verdict"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    out = exp_dir / "focused_noae_launch_check_v1.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"\nwritten: {out}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
