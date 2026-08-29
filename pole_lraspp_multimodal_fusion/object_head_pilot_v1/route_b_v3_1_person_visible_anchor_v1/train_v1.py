#!/usr/bin/env python3
"""The single registered 24-epoch person-visible-anchor scientific run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

from common_v1 import (  # noqa: E402
    read_csv, restore_rng, rng_states, seed_everything, sha256, tensor_state_hash,
    utc_now, write_json_atomic, write_json_x, write_text_x,
)
from losses_v1 import private_person_loss  # noqa: E402
from model_v1 import (  # noqa: E402
    build_model, configure_private_training, inherited_state, load_epoch40,
    parameter_report, private_parameters,
)
from targets_v1 import VisibleAnchorDataset, load_visible_rows  # noqa: E402
from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402

CHECKPOINT_EPOCHS = (6, 12, 18, 24)
LOSS_NAMES = (
    "visible_heatmap", "visible_subcell_offset", "visible_to_box_center_offset",
    "full_box_wh_smooth_l1", "full_box_wh_giou", "physical_ray_offset",
    "bounded_log_depth", "local_xyz_endpoint", "person_dimensions", "person_yaw",
    "radar_support",
)
FIELDS = (
    "epoch", "lr_start", "lr_end", "total_loss", "positive_cells",
    *tuple(f"unweighted_{name}" for name in LOSS_NAMES),
    *tuple(f"weighted_{name}" for name in LOSS_NAMES),
    "decoded_depth_min_m", "decoded_depth_max_m", "person_cell_collisions",
    "batches", "optimizer_steps", "epoch_seconds", "cuda_allocated_peak_mib",
    "cuda_reserved_peak_mib", "inherited_state_hash", "created_utc",
)


def _loss_design(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth_bounds_m": config["person_private"]["depth_bounds_m"],
        "dimension_normalization_m": config["person_private"]["dimension_normalization_m"],
        "endpoint_normalization_m": config["person_private"]["endpoint_normalization_m"],
        "loss_weights": config["training"]["loss_weights"],
    }


def learning_rate(step: int, steps_per_epoch: int, total_epochs: int,
                  peak: float, warmup_start_ratio: float,
                  final_ratio: float) -> float:
    total_steps = int(steps_per_epoch) * int(total_epochs)
    warmup_steps = int(steps_per_epoch)
    if step < warmup_steps:
        progress = step / float(max(1, warmup_steps - 1))
        ratio = warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
    else:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
        ratio = final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(peak) * float(ratio)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    digest = sha256(path)
    verify = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "rng_states", "epoch",
                "optimizer_steps", "resolved_config_sha256", "numerical_policy"}
    if not required.issubset(verify) or int(verify["epoch"]) != int(payload["epoch"]):
        raise RuntimeError(f"checkpoint integrity failure: {path}")
    if not all(torch.isfinite(value).all().item() for value in verify["model"].values()):
        raise RuntimeError(f"nonfinite checkpoint tensor: {path}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--end-epoch", required=True, type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    config_path = experiment / "RESOLVED_CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = config["training"]
    if (int(design["epochs"]) != 24 or tuple(design["checkpoint_epochs"]) != CHECKPOINT_EPOCHS
            or int(design["batch_size"]) != 16 or design["optimizer"] != "AdamW"
            or int(design["warmup_epochs"]) != 1 or design["numerical_policy"] != "full_fp32"
            or design["private_fp16_forbidden"] is not True or design["geometric_augment"] is not False
            or design["q"] != 0 or design["ae"] is not False):
        raise RuntimeError("registered training design drift")
    if args.end_epoch not in (12, 24):
        raise RuntimeError("training may stop only for the epoch-12 gate or final epoch 24")
    registration = json.loads((experiment / "REGISTERED_DESIGN.json").read_text(encoding="utf-8"))
    preflight = json.loads((experiment / "PREFLIGHT.json").read_text(encoding="utf-8"))
    numerical = json.loads((experiment / "NUMERICAL_QUALIFICATION.json").read_text(encoding="utf-8"))
    if (not preflight["all_pass"] or registration["optimizer_steps_before_registration"] != 0
            or sha256(config_path) != registration["resolved_config_sha256"]
            or numerical["selected_policy"] != "full_fp32" or numerical["private_fp16_used"]):
        raise RuntimeError("preflight/design/numerical registration is not closed")
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA environment unavailable")

    checkpoint_path = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint_path) != config["warm_start_sha256"]:
        raise RuntimeError("epoch-40 warm-start SHA drift")
    base = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = read_manifest(experiment / "dataset/manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    if len(train_rows) != 16827 or len(val_rows) != 3345 or any(row["split"] == "test" for row in manifest):
        raise RuntimeError("training split population drift")
    visible_rows, target_parameters = load_visible_rows(
        experiment / "derived_targets/visible_anchor_targets_v010.csv",
    )
    if target_parameters["offset_scales"] != config["resolved_offset_scales"]:
        raise RuntimeError("train-only offset normalization drift")
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    object_cfg = dict(base["config"]["object_heads"])
    dataset = VisibleAnchorDataset(
        experiment / "dataset", train_rows, object_rows, tuple(config["model_size_wh"]),
        object_cfg, augment_strength=str(design["augment_strength"]),
        geometric_augment=False, visible_rows=visible_rows,
        offset_scales=config["resolved_offset_scales"],
        depth_bounds_m=config["person_private"]["depth_bounds_m"],
        dimension_scale_m=config["person_private"]["dimension_normalization_m"],
        endpoint_scale_m=config["person_private"]["endpoint_normalization_m"],
    )
    sampler_registration = json.loads((experiment / "SAMPLER_REGISTRATION.json").read_text())
    weights = torch.as_tensor(sampler_registration["normalized_weights"], dtype=torch.double)
    if len(weights) != len(dataset) or sampler_registration["validation_used_for_sampling_or_parameters"] != 0:
        raise RuntimeError("sampler registration drift")

    device = torch.device("cuda")
    seed_everything(int(design["training_seed"]))
    model = build_model(
        radar_channels=int(base["radar_channels"]),
        hidden_channels=int(base["object_hidden_channels"]),
        head_depth=int(base["object_head_depth"]),
        depth_bounds_m=tuple(config["person_private"]["depth_bounds_m"]), device=device,
    )
    load_epoch40(model, checkpoint_path, device=device, initialize_private=True)
    configure_private_training(model)
    optimizer = torch.optim.AdamW(
        private_parameters(model), lr=0.0, weight_decay=float(design["weight_decay"]),
    )
    if not all(parameter.dtype == torch.float32 for parameter in private_parameters(model)):
        raise RuntimeError("private optimizer parameter is not FP32")
    start_epoch, optimizer_steps = 1, 0
    inherited_hash = tensor_state_hash(inherited_state(model))
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve(strict=True)
        if not args.resume_sha256 or sha256(resume_path) != args.resume_sha256:
            raise RuntimeError("recovery checkpoint SHA mismatch")
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (resume["resolved_config_sha256"] != sha256(config_path)
                or resume["registered_design_sha256"] != sha256(experiment / "REGISTERED_DESIGN.json")
                or resume["numerical_policy"] != "full_fp32"):
            raise RuntimeError("recovery checkpoint registration mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        restore_rng(resume["rng_states"])
        start_epoch = int(resume["epoch"]) + 1
        optimizer_steps = int(resume["optimizer_steps"])
        if tensor_state_hash(inherited_state(model)) != inherited_hash:
            raise RuntimeError("recovery checkpoint changed inherited state")
    if args.end_epoch < start_epoch:
        raise RuntimeError("requested training boundary precedes recovery state")

    metrics_dir = experiment / "training_metrics"; metrics_dir.mkdir(exist_ok=True)
    metrics_path = metrics_dir / "per_epoch_training.csv"
    if start_epoch == 1:
        if (experiment / "SCIENTIFIC_ATTEMPT_1_STARTED.json").exists():
            raise RuntimeError("a scientific attempt was already launched")
        write_json_x(experiment / "SCIENTIFIC_ATTEMPT_1_STARTED.json", {
            "schema": "route_b_v3_1_person_visible_anchor_scientific_attempt_v1",
            "created_utc": utc_now(), "attempt": 1, "start_epoch": 1,
            "planned_final_epoch": 24, "resolved_config_sha256": sha256(config_path),
            "registered_design_sha256": sha256(experiment / "REGISTERED_DESIGN.json"),
            "warm_start_sha256": sha256(checkpoint_path), "numerical_policy": "full_fp32",
        })
        with metrics_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    else:
        existing = read_csv(metrics_path)
        if not existing or int(existing[-1]["epoch"]) != start_epoch - 1:
            raise RuntimeError("training CSV/recovery boundary mismatch")

    checkpoint_dir = experiment / "checkpoints" / config["name"]
    recovery_dir = experiment / "recovery_checkpoints"; recovery_dir.mkdir(exist_ok=True)
    peak_allocated = peak_reserved = 0.0
    steps_per_epoch = math.ceil(int(design["sampling"]["num_samples_per_epoch"])
                                / int(design["batch_size"]))
    for epoch in range(start_epoch, args.end_epoch + 1):
        configure_private_training(model)
        generator = torch.Generator(); generator.manual_seed(int(design["training_seed"]) + epoch)
        sampler = WeightedRandomSampler(
            weights, num_samples=int(design["sampling"]["num_samples_per_epoch"]),
            replacement=True, generator=generator,
        )
        loader = DataLoader(
            dataset, batch_size=int(design["batch_size"]), sampler=sampler,
            drop_last=False, num_workers=int(design["num_workers"]), pin_memory=True,
            persistent_workers=bool(design["persistent_workers"]),
            prefetch_factor=int(design["prefetch_factor"]),
        )
        if len(loader) != steps_per_epoch:
            raise RuntimeError("registered steps-per-epoch drift")
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.monotonic(); sums: dict[str, float] = {}; batches = 0
        first_lr = last_lr = 0.0
        collision_sum = 0.0
        for batch_index, (tensors, _masks, targets) in enumerate(loader):
            step = (epoch - 1) * steps_per_epoch + batch_index
            lr = learning_rate(
                step, steps_per_epoch, int(design["epochs"]), float(design["peak_lr"]),
                float(design["warmup_start_ratio"]), float(design["cosine_final_ratio"]),
            )
            if batch_index == 0:
                first_lr = lr
            last_lr = lr; optimizer.param_groups[0]["lr"] = lr
            tensors = tensors.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model.private_training_outputs(tensors)
            loss, parts = private_person_loss(
                outputs, targets, design=_loss_design(config),
                offset_scales=config["resolved_offset_scales"],
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"nonfinite loss epoch={epoch} batch={batch_index + 1}")
            loss.backward()
            bad_gradients = [name for name, parameter in model.person_private.named_parameters()
                             if parameter.requires_grad and (parameter.grad is None
                             or not torch.isfinite(parameter.grad).all().item())]
            if bad_gradients:
                raise RuntimeError(f"missing/nonfinite gradients: {bad_gradients}")
            if any(parameter.grad is not None for name, parameter in model.named_parameters()
                   if not name.startswith("person_private.")):
                raise RuntimeError("inherited gradient appeared during training")
            optimizer.step(); optimizer_steps += 1; batches += 1
            for key, value in parts.items():
                if key.startswith("share_"):
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
            collision_sum += float(targets["person_cell_collisions"].sum().item())
        current_inherited_hash = tensor_state_hash(inherited_state(model))
        if current_inherited_hash != inherited_hash:
            raise RuntimeError(f"inherited state drift after epoch {epoch}")
        allocated = torch.cuda.max_memory_allocated(device) / 2**20
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
        if reserved >= 12 * 1024:
            raise RuntimeError(f"training memory exceeded 12 GiB: {reserved}")
        peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
        row = {
            "epoch": epoch, "lr_start": first_lr, "lr_end": last_lr,
            **{key: sums.get(key, 0.0) / batches for key in FIELDS if key in sums},
            "person_cell_collisions": collision_sum, "batches": batches,
            "optimizer_steps": optimizer_steps, "epoch_seconds": time.monotonic() - epoch_started,
            "cuda_allocated_peak_mib": allocated, "cuda_reserved_peak_mib": reserved,
            "inherited_state_hash": current_inherited_hash, "created_utc": utc_now(),
        }
        with metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        write_json_x(metrics_dir / f"epoch_{epoch:03d}.json", row)
        designated = epoch in CHECKPOINT_EPOCHS
        checkpoint_file = ((checkpoint_dir if designated else recovery_dir)
                           / f"epoch_{epoch:03d}.pt")
        payload = {
            "schema": "route_b_v3_1_person_visible_anchor_checkpoint_v1",
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": {
                "schema": "one_epoch_linear_warmup_then_full_schedule_cosine_v1",
                "step": optimizer_steps, "steps_per_epoch": steps_per_epoch,
                "peak_lr": design["peak_lr"], "warmup_epochs": 1,
                "warmup_start_ratio": design["warmup_start_ratio"],
                "final_ratio": design["cosine_final_ratio"], "total_epochs": 24,
            },
            "rng_states": rng_states(), "epoch": epoch, "optimizer_steps": optimizer_steps,
            "resolved_config": config, "resolved_config_sha256": sha256(config_path),
            "registered_design_sha256": sha256(experiment / "REGISTERED_DESIGN.json"),
            "target_view_sha256": sha256(experiment / "derived_targets/visible_anchor_targets_v010.csv"),
            "numerical_policy": "full_fp32", "grad_scaler_enabled": False,
            "warm_start_checkpoint": str(checkpoint_path),
            "warm_start_sha256": sha256(checkpoint_path),
            "radar_channels": int(base["radar_channels"]),
            "object_hidden_channels": int(base["object_hidden_channels"]),
            "object_head_depth": int(base["object_head_depth"]),
            "parameter_report": parameter_report(model),
            "inherited_state_hash": inherited_hash,
        }
        checkpoint_hash = save_checkpoint(checkpoint_file, payload)
        old_latest = (json.loads((experiment / "LATEST_SAFE.json").read_text())
                      if (experiment / "LATEST_SAFE.json").is_file() else None)
        write_json_atomic(experiment / "LATEST_SAFE.json", {
            "epoch": epoch, "path": str(checkpoint_file), "sha256": checkpoint_hash,
            "optimizer_steps": optimizer_steps, "created_utc": utc_now(),
        })
        if old_latest:
            old_path = Path(old_latest["path"])
            if old_path.parent == recovery_dir and old_path != checkpoint_file and old_path.is_file():
                old_path.unlink()
        print(
            f"[visible anchor train] epoch={epoch}/24 loss={row['total_loss']:.6f} "
            f"lr={last_lr:.8g} steps={optimizer_steps} vram={reserved:.1f}MiB",
            flush=True,
        )

    stage = {
        "schema": "route_b_v3_1_person_visible_anchor_training_stage_complete_v1",
        "created_utc": utc_now(), "start_epoch": start_epoch, "end_epoch": args.end_epoch,
        "optimizer_steps": optimizer_steps, "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved, "validation_rows_used": 0,
        "scientific_attempt": 1, "numerical_policy": "full_fp32",
    }
    write_json_x(experiment / f"TRAINING_STAGE_COMPLETE_{args.end_epoch:03d}.json", stage)
    if args.end_epoch == 24:
        checkpoints = []
        for epoch in CHECKPOINT_EPOCHS:
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            checkpoints.append({"epoch": epoch, "path": str(path), "sha256": sha256(path)})
        write_json_x(experiment / "TRAINING_COMPLETE.json", {
            **stage, "schema": "route_b_v3_1_person_visible_anchor_training_complete_v1",
            "epochs_completed": 24, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "checkpoints": checkpoints, "exactly_one_scientific_attempt": True,
        })
        write_text_x(experiment / "TRAINING_COMPLETE", "EXACTLY_ONE_24_EPOCH_SCIENTIFIC_RUN_COMPLETE\n")
    print(json.dumps(stage, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
