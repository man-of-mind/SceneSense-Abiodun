#!/usr/bin/env python3
"""ONE bounded 12-epoch Route B Faster R-CNN recovery run (recovery-v2).

Warm start: the ORIGINAL route_b_fasterrcnn_radar_roi_v1 epoch-12 checkpoint.
Trainable: RPN, ROI box head, ROI classifier/2D box predictor, semantic segmentation decoder.
Frozen: RGB ResNet50+FPN backbone, radar encoder, radar ROI embed, world XYZ/dimension/yaw
        localization head, and a frozen deepcopy of the ORIGINAL box_head that preserves the
        radar ROI localization path bit-exactly.
Train split only. No validation row and no validation FP row is ever used as a training example.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import common_v2 as C
from pole_lraspp_multimodal_fusion.common import set_reproducible_seeds
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes, valid_localization_objects
from dataset_v1 import RouteBFasterRCNNDataset, detection_collate
from split_runtime_adapter_v1 import reconstruct_image_list
from model_patch_v2 import (
    P2_ANCHOR_REGISTRATION, build_recovery_model, freeze_audit, set_eval_mode_on_frozen,
)
from losses_v2 import LOSS_WEIGHTS, ROI_CE_WEIGHTS, roi_losses, segmentation_losses, registration


def person_frame_weights(rows, object_rows, weight: float) -> tuple[List[float], int]:
    weights, person_frames = [], 0
    for row in rows:
        objects = valid_localization_objects(
            object_rows.get(row["sample_id"], []),
            image_width=int(row["camera_width"]), image_height=int(row["camera_height"]),
            min_area_px=12.0, object_class_names=C.CLASSES, max_distance_m=40.0,
        )
        has_person = any(obj["class_name"] == "person" for obj in objects)
        person_frames += int(has_person)
        weights.append(float(weight) if has_person else 1.0)
    return weights, person_frames


def sample_rois(roi_heads, proposals, targets, config, device, counters):
    """Registered ROI sampler: GT-augmented proposals, per-class positive quota, UNIFORM negatives."""
    boxes, labels, regression_targets = [], [], []
    for index, proposal in enumerate(proposals):
        gt_boxes = targets[index]["boxes"]
        gt_labels = targets[index]["labels"]
        candidates = torch.cat([proposal, gt_boxes], dim=0) if gt_boxes.numel() else proposal
        if candidates.numel() == 0:
            continue
        matched, assigned = roi_heads.assign_targets_to_proposals([candidates], [gt_boxes], [gt_labels])
        matched, assigned = matched[0], assigned[0]
        person = torch.where(assigned == 2)[0]
        vehicle = torch.where(assigned == 1)[0]
        negative = torch.where(assigned == 0)[0]
        keep_person = person[torch.randperm(person.numel(), device=device)[: config["max_person_positives"]]]
        keep_vehicle = vehicle[torch.randperm(vehicle.numel(), device=device)[: config["max_vehicle_positives"]]]
        n_negative = max(1, int(config["rois_per_image"]) - keep_person.numel() - keep_vehicle.numel())
        if negative.numel() > n_negative:
            keep_negative = negative[torch.randperm(negative.numel(), device=device)[:n_negative]]
        else:
            keep_negative = negative
        keep = torch.cat([keep_person, keep_vehicle, keep_negative])
        selected = candidates[keep]
        selected_labels = assigned[keep]
        matched_gt = gt_boxes[matched[keep]] if gt_boxes.numel() else selected
        boxes.append(selected)
        labels.append(selected_labels)
        regression_targets.append(roi_heads.box_coder.encode([matched_gt], [selected])[0])
        counters["person_positive_rois"] += int(keep_person.numel())
        counters["vehicle_positive_rois"] += int(keep_vehicle.numel())
        counters["negative_rois"] += int(keep_negative.numel())
    return boxes, labels, regression_targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=C.IMPL_V2 / "configs" / "recovery_v2.json")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=0, help="launch qualification only; 0 = full run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    roi_config = config["roi_sampler"]
    experiment_dir = args.experiment_dir.resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise SystemExit(f"refusing nonempty checkpoint directory: {checkpoint_dir}")

    parent_sha = C.verify_warm_start()
    print(f"[ok] warm-start SHA verified {parent_sha}", flush=True)
    splits = C.load_split_rows()
    train_rows = splits["train"]
    object_rows = load_object_boxes(C.DATASET_DIR / "object_boxes.csv")
    weights, person_frames = person_frame_weights(
        train_rows, object_rows, float(config["frame_sampler"]["person_bearing_frame_weight"]))
    print(f"[ok] train {len(train_rows)} frames, {person_frames} person-bearing", flush=True)

    set_reproducible_seeds(int(config["seed"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    payload = torch.load(C.WARM_START, map_location="cpu", weights_only=False)
    width, height = map(int, payload["input_size"])
    model, frozen_box_head, anchor_info = build_recovery_model(payload, device)
    set_eval_mode_on_frozen(model)          # audit the state the training loop actually runs in
    audit = freeze_audit(model, frozen_box_head)
    if not audit["pass"]:
        raise SystemExit(f"freeze audit failed: {json.dumps(audit, indent=2, default=str)}")
    print(f"[ok] trainable {audit['trainable_parameters']:,}/{audit['total_parameters']:,} "
          f"({100 * audit['trainable_parameters'] / audit['total_parameters']:.2f}%)", flush=True)

    dataset = RouteBFasterRCNNDataset(C.DATASET_DIR, train_rows, object_rows, (width, height), training=True)
    sampler = WeightedRandomSampler(weights, num_samples=int(config["frame_sampler"]["num_samples"]), replacement=True)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), sampler=sampler,
                        num_workers=int(config["num_workers"]), pin_memory=True,
                        persistent_workers=True, prefetch_factor=2, collate_fn=detection_collate)

    roi_heads = model.detector.roi_heads
    groups = [
        {"params": [p for p in model.detector.rpn.parameters() if p.requires_grad], "lr": float(config["lr_rpn"]), "group_name": "rpn"},
        {"params": list(roi_heads.box_head.parameters()), "lr": float(config["lr_box_head"]), "group_name": "box_head"},
        {"params": list(roi_heads.box_predictor.parameters()), "lr": float(config["lr_box_predictor"]), "group_name": "box_predictor"},
        {"params": list(model.segmentation_decoder.parameters()), "lr": float(config["lr_segmentation_decoder"]), "group_name": "segmentation_decoder"},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16

    registered = {
        "config": config, "parent_checkpoint_sha256": parent_sha,
        "anchor_registration": anchor_info, "objective_registration": registration(),
        "freeze_audit": audit, "train_frames": len(train_rows),
        "person_bearing_train_frames": person_frames,
        "roi_ce_weights_bg_vehicle_person": list(ROI_CE_WEIGHTS),
        "loss_weights": LOSS_WEIGHTS,
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(device),
    }
    if args.max_steps <= 0:
        C.write_json_create(experiment_dir / "registered_config.json", registered)

    metrics_path = experiment_dir / "training_metrics.csv"
    fields = ["epoch", "loss", "loss_objectness", "loss_rpn_box_reg", "loss_roi_classifier",
              "loss_roi_box_reg", "loss_segmentation_ce", "loss_segmentation_dice",
              "person_positive_rois", "vehicle_positive_rois", "negative_rois",
              "lr_rpn", "seconds", "peak_allocated_mib", "peak_reserved_mib"]
    handle = metrics_path.open("x", newline="", encoding="utf-8") if args.max_steps <= 0 else None
    writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
    if writer:
        writer.writeheader()

    epochs = int(config["epochs"])
    total_steps = len(loader) * epochs
    global_step = 0
    history: List[Dict] = []
    status = "runtime_failure"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            set_eval_mode_on_frozen(model)
            counters = {"person_positive_rois": 0, "vehicle_positive_rois": 0, "negative_rois": 0}
            sums = {key: 0.0 for key in ("loss", "loss_objectness", "loss_rpn_box_reg", "loss_roi_classifier",
                                         "loss_roi_box_reg", "loss_segmentation_ce", "loss_segmentation_dice")}
            batches = 0
            epoch_start = time.monotonic()
            for rgb, _radar, targets, _metadata in loader:
                global_step += 1
                warm = min(1.0, global_step / float(config["warmup_steps"]))
                progress = global_step / max(1, total_steps)
                ratio = float(config["min_lr_ratio"])
                cosine = ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * warm * cosine

                rgb = [value.to(device, non_blocking=True) for value in rgb]
                targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()}
                           for target in targets]
                optimizer.zero_grad(set_to_none=True)
                # Frozen backbone: forward without grad, then DETACH. The detachment is what makes
                # RPN / ROI / segmentation training provably unable to alter backbone features.
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=bool(config["amp"])):
                    images, _ = model.detector.transform(list(rgb), None)
                    raw_features = model.detector.backbone(images.tensors)
                if isinstance(raw_features, torch.Tensor):
                    raw_features = OrderedDict([("0", raw_features)])
                features = OrderedDict((key, value.detach()) for key, value in raw_features.items())
                image_sizes = list(images.image_sizes)
                batch_shape = tuple(images.tensors.shape)

                with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(config["amp"])):
                    image_list = reconstruct_image_list(batch_shape, image_sizes,
                                                        features["0"].device, dtype=features["0"].dtype)
                    proposals, rpn_loss = model.detector.rpn(image_list, features, targets)
                with torch.no_grad():
                    boxes, labels, regression_targets = sample_rois(
                        roi_heads, proposals, targets, roi_config, device, counters)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=bool(config["amp"])):
                    visual = roi_heads.box_head(roi_heads.box_roi_pool(features, boxes, image_sizes))
                    class_logits, box_regression = roi_heads.box_predictor(visual)
                    roi_loss = roi_losses(class_logits.float(), box_regression.float(),
                                          torch.cat(labels), torch.cat(regression_targets))
                    seg_logits = model.segmentation_decoder(features, (batch_shape[-2], batch_shape[-1]))
                    seg_targets = torch.stack([target["segmentation"] for target in targets], dim=0)
                    if seg_targets.shape[-2:] != seg_logits.shape[-2:]:
                        seg_targets = F.interpolate(seg_targets[:, None].float(), size=seg_logits.shape[-2:],
                                                    mode="nearest")[:, 0].long()
                    seg_loss = segmentation_losses(seg_logits.float(), seg_targets)

                parts = {
                    "loss_objectness": LOSS_WEIGHTS["rpn"] * rpn_loss["loss_objectness"].float(),
                    "loss_rpn_box_reg": LOSS_WEIGHTS["rpn"] * rpn_loss["loss_rpn_box_reg"].float(),
                    "loss_roi_classifier": LOSS_WEIGHTS["roi_classifier"] * roi_loss["loss_roi_classifier"],
                    "loss_roi_box_reg": LOSS_WEIGHTS["roi_box_reg"] * roi_loss["loss_roi_box_reg"],
                    "loss_segmentation_ce": LOSS_WEIGHTS["segmentation"] * seg_loss["loss_segmentation_ce"],
                    "loss_segmentation_dice": LOSS_WEIGHTS["segmentation"] * seg_loss["loss_segmentation_dice"],
                }
                loss = sum(parts.values())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss epoch={epoch} step={global_step}: {float(loss)}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               float(config["grad_clip_norm"]))
                optimizer.step()
                sums["loss"] += float(loss.detach())
                for key, value in parts.items():
                    sums[key] += float(value.detach())
                batches += 1
                if args.max_steps > 0 and batches >= args.max_steps:
                    return qualification_report(model, frozen_box_head, parts, loss, audit,
                                                anchor_info, counters, device)
                if global_step % 300 == 0:
                    print(f"[train] e{epoch} step {global_step}/{total_steps} "
                          f"loss={sums['loss']/batches:.4f} cls={sums['loss_roi_classifier']/batches:.4f} "
                          f"obj={sums['loss_objectness']/batches:.4f} "
                          f"segdice={sums['loss_segmentation_dice']/batches:.4f} "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

            row = {"epoch": epoch, "seconds": time.monotonic() - epoch_start,
                   "lr_rpn": optimizer.param_groups[0]["lr"],
                   "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                   "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
                   **{key: value / max(1, batches) for key, value in sums.items()}, **counters}
            history.append(row)
            if writer:
                writer.writerow(row)
                handle.flush()
            C.save_checkpoint_create(checkpoint_dir / f"recovery_v2_epoch_{epoch:03d}.pt", {
                "epoch": epoch, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "frozen_box_head": {k: v.cpu() for k, v in frozen_box_head.state_dict().items()},
                "input_size": [width, height], "config": config,
                "anchor_registration": anchor_info, "freeze_audit": audit,
                "parent_checkpoint_sha256": parent_sha,
                "object_class_names": list(C.CLASSES),
                "training_runtime_seconds": time.monotonic() - started,
            })
            print("[epoch] " + json.dumps(row), flush=True)
        status = "complete"
        C.write_json_create(experiment_dir / "training_runtime.json", {
            "status": status, "runtime_seconds": time.monotonic() - started,
            "epochs": epochs, "batch_size": int(config["batch_size"]),
            "device": torch.cuda.get_device_name(device),
            "peak_allocated_mib": max(r["peak_allocated_mib"] for r in history),
            "peak_reserved_mib": max(r["peak_reserved_mib"] for r in history),
            "history": history,
        })
    finally:
        if handle:
            handle.close()
        if args.max_steps <= 0:
            C.notify("Route B Faster R-CNN recovery v2 training",
                     f"{status}: {experiment_dir.name}",
                     experiment_dir / "TRAINING_COMPLETION_NOTIFICATION.json")
    return 0


def qualification_report(model, frozen_box_head, parts, loss, audit, anchor_info, counters, device):
    """Phase C: one real AMP batch proving finite nonzero gradients in every trainable branch."""
    branches = {
        "rpn": model.detector.rpn,
        "roi_box_head": model.detector.roi_heads.box_head,
        "roi_box_predictor": model.detector.roi_heads.box_predictor,
        "segmentation_decoder": model.segmentation_decoder,
    }
    protected = {
        "detector.backbone": model.detector.backbone,
        "radar_encoder": model.radar_encoder,
        "radar_roi_embed": model.radar_roi_embed,
        "roi_localization_head": model.roi_localization_head,
        "frozen_box_head": frozen_box_head,
    }
    report = {"losses": {key: float(value.detach()) for key, value in parts.items()},
              "total_loss": float(loss.detach()), "roi_label_counts": counters,
              "freeze_audit": audit, "anchor_registration": anchor_info,
              "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
              "gradients": {}, "protected_gradients": {}}
    ok = True
    for name, module in branches.items():
        norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
        finite = all(math.isfinite(value) for value in norms)
        nonzero = any(value > 0.0 for value in norms)
        report["gradients"][name] = {"tensors_with_grad": len(norms),
                                     "total_tensors": sum(1 for _ in module.parameters()),
                                     "grad_norm_sum": float(sum(norms)),
                                     "finite": finite, "nonzero": nonzero}
        ok = ok and finite and nonzero and len(norms) > 0
    for name, module in protected.items():
        with_grad = [n for n, p in module.named_parameters() if p.grad is not None]
        report["protected_gradients"][name] = {"tensors_with_grad": len(with_grad),
                                               "names": with_grad[:5]}
        ok = ok and not with_grad
    ok = ok and all(math.isfinite(v) for v in report["losses"].values())
    ok = ok and all(v > 0.0 for v in report["losses"].values())
    report["pass"] = bool(ok)
    print("LAUNCH_QUALIFICATION " + json.dumps(report, indent=2, default=str), flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
