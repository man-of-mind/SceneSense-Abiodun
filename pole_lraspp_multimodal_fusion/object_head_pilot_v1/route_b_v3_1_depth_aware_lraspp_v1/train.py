from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from common import (CONFIG_PATH, load_json, read_csv, restore_rng_state, rng_state, seed_everything,
                    sha256, tensor_state_hash, utc_now, write_json_x, write_text_x,
                    write_torch_atomic_create)
from data import DepthCache, TrainingDataset, collate_training, load_objects, load_visible_anchors
from losses import compute_losses
from model import (build_model, configure_stage, freeze_bn_running_state, parameter_groups,
                   pretrained_backbone_state, stage_train_mode)

SCIENTIFIC_COMPONENTS = (
    "segmentation", "heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray",
    "depth_bin", "depth_residual", "endpoint", "dimensions", "yaw", "parked",
    "radar_support", "dense_depth", "radar_consistency",
)


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    groups = parameter_groups(model)
    specs = []
    for name, values in groups.items():
        specs.append({
            "params": [parameter for _key, parameter in values], "lr": 0.0,
            "weight_decay": 0.0 if name.endswith("no_decay") else 1e-4,
            "name": name,
        })
    return torch.optim.AdamW(specs, betas=(0.9, 0.999), eps=1e-8)


def scheduled_lrs(epoch: int, update_in_epoch: int, updates_in_epoch: int,
                  global_update: int) -> tuple[float, float]:
    if epoch <= 5:
        warm = min(1000, updates_in_epoch)
        if epoch == 1 and update_in_epoch <= warm:
            fraction = max(0.0, min(1.0, (update_in_epoch - 1) / max(1, warm - 1)))
            new = 3e-5 + fraction * (3e-4 - 3e-5)
        else:
            new = 3e-4
        return new, 0.0
    if epoch == 6:
        fraction = max(0.0, min(1.0, (update_in_epoch - 1) / max(1, updates_in_epoch - 1)))
        return 1e-4, 1e-6 + fraction * (1e-5 - 1e-6)
    total = 34 * updates_in_epoch
    index = (epoch - 7) * updates_in_epoch + (update_in_epoch - 1)
    fraction = max(0.0, min(1.0, index / max(1, total - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return 1e-6 + (1e-4 - 1e-6) * cosine, 1e-7 + (1e-5 - 1e-7) * cosine


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, new_lr: float, backbone_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = backbone_lr if str(group["name"]).startswith("backbone") else new_lr


def task_groups(parts: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> dict[str, torch.Tensor]:
    detection_names = ("heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray",
                       "dimensions", "yaw", "parked", "radar_support")
    actor_names = ("depth_bin", "depth_residual", "endpoint")
    dense_names = ("dense_depth", "radar_consistency")
    return {
        "detection": sum(float(weights[name]) * parts[name] for name in detection_names),
        "actor_depth": sum(float(weights[name]) * parts[name] for name in actor_names),
        "dense_depth": sum(float(weights[name]) * parts[name] for name in dense_names),
        "segmentation": float(weights["segmentation"]) * parts["segmentation"],
    }


def telemetry(model: torch.nn.Module, batch: Mapping[str, Any], weights: Mapping[str, float]) -> dict[str, Any]:
    cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    training = model.training
    model.eval(); freeze_bn_running_state(model)
    total, parts, denominators, _outputs = compute_losses(model, batch, weights)
    shared = [parameter for parameter in model.depth_neck.parameters() if parameter.requires_grad]
    groups = task_groups(parts, weights)
    gradients: dict[str, list[torch.Tensor]] = {}
    norms: dict[str, float] = {}
    for index, (name, value) in enumerate(groups.items()):
        values = torch.autograd.grad(value, shared, retain_graph=index < len(groups) - 1, allow_unused=False)
        gradients[name] = [item.detach() for item in values]
        norms[name] = math.sqrt(sum(float(item.float().pow(2).sum().item()) for item in values))
    cosines = {}
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            dot = sum(float((a.float() * b.float()).sum().item()) for a, b in zip(gradients[left], gradients[right]))
            cosines[f"{left}__{right}"] = dot / max(1e-20, norms[left] * norms[right])
    record = {
        "raw_normalized_losses": {name: float(parts[name].detach().item()) for name in SCIENTIFIC_COMPONENTS},
        "configured_weights": {name: float(weights[name]) for name in SCIENTIFIC_COMPONENTS},
        "weighted_scalar_contributions": {name: float((float(weights[name]) * parts[name]).detach().item())
                                          for name in SCIENTIFIC_COMPONENTS},
        "valid_denominators": denominators,
        "shared_depth_neck_gradient_norms": norms,
        "gradient_cosines": cosines,
        "total": float(total.detach().item()), "optimizer_step": False,
        "batch_sample_ids": list(batch["sample_id"]),
    }
    for parameter in model.parameters():
        parameter.grad = None
    random.setstate(cpu_rng[0]); np.random.set_state(cpu_rng[1]); torch.set_rng_state(cpu_rng[2])
    torch.cuda.set_rng_state_all(cuda_rng)
    if training:
        model.train(); freeze_bn_running_state(model)
    return record


def latest_checkpoint(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("epoch_*.pt"), reverse=True) if directory.exists() else []
    for candidate in candidates:
        sidecar = candidate.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            record = load_json(sidecar)
            if (bool(record["complete"])
                    and int(record["epoch"]) == int(candidate.stem.split("_")[1])
                    and int(record["bytes"]) == candidate.stat().st_size
                    and record["sha256"] == sha256(candidate)):
                return candidate
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    if not (experiment / "QUALIFICATION_COMPLETE").is_file():
        raise RuntimeError("scientific training requires completed qualification")
    config = load_json(CONFIG_PATH)
    runtime = load_json(experiment / "QUALIFIED_RUNTIME.json")
    physical = int(runtime["physical_batch"]); accumulation = int(runtime["gradient_accumulation"])
    seed = int(config["scientific_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("scientific CUDA runtime unavailable")
    device = torch.device("cuda")
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "train"]
    if len(rows) != 16827 or len({row["sample_id"] for row in rows}) != 16827:
        raise RuntimeError("scientific train population drift")
    objects = load_objects(dataset_root)
    visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    cache = DepthCache(experiment / "depth_cache/train", rows)
    dataset = TrainingDataset(dataset_root, rows, objects, visible, cache, seed)
    fixed_rows = rows[:min(4, len(rows))]
    # Ensure the fixed telemetry batch contains person supervision.
    for index, row in enumerate(rows):
        if any(item["label"] == "person" for item in objects.get(row["sample_id"], ())):
            fixed_rows[-1] = row; break
    fixed_dataset = TrainingDataset(dataset_root, fixed_rows, objects, visible, cache, seed)
    fixed_dataset.set_epoch(1)
    fixed_batch = next(iter(DataLoader(fixed_dataset, batch_size=len(fixed_rows), num_workers=0,
                                       collate_fn=collate_training)))
    model, loading = build_model(Path(config["pretrained"]["path"]), device)
    optimizer = build_optimizer(model)
    config_sha = sha256(CONFIG_PATH)
    initial_pretrained_hash = tensor_state_hash(pretrained_backbone_state(model))
    checkpoint_dir = experiment / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    metrics_dir = experiment / "training_metrics"; metrics_dir.mkdir(exist_ok=True)
    telemetry_dir = experiment / "gradient_telemetry"; telemetry_dir.mkdir(exist_ok=True)
    checkpoint = latest_checkpoint(checkpoint_dir)
    start_epoch = 1; global_step = 0; training_started = time.time(); cumulative_wall = 0.0
    source_provenance = (load_json(experiment / "SOURCE_PROVENANCE.json")
                         if (experiment / "SOURCE_PROVENANCE.json").is_file()
                         else {"repair_code_commit": config["source_commit"],
                               "scientific_launch_commit": None})
    if checkpoint is None:
        write_json_x(experiment / "TRAINING_STARTED.json", {
            "schema": "route_b_v3_1_depth_aware_lraspp_training_started_v1", "created_utc": utc_now(),
            "seed": seed, "seeds": {"python": seed, "numpy": seed, "torch": seed, "cuda": seed,
                                      "sampler": seed, "workers": [seed + index + 1 for index in range(8)]},
            "physical_batch": physical, "gradient_accumulation": accumulation, "effective_batch": 16,
            "epochs": 40, "precision": "full_fp32", "validation_during_training": False,
            "official_loading": loading, "initial_pretrained_state_hash": initial_pretrained_hash,
            "resolved_config_sha256": config_sha,
        })
        initial_payload = {
            "schema": "route_b_v3_1_depth_aware_lraspp_recovery_checkpoint_v2",
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "epoch": 0, "global_step": 0,
            "gradient_accumulation_phase": 0, "rng_state": rng_state(),
            "sampler_state": {"epoch": 1, "seed": seed + 1, "visited": 0,
                              "unique": 0, "complete": False},
            "scheduler_state": {"next_epoch": 1, "global_update": 0,
                                "schedule": "registered_stage_a_stage_b_stateless_v1"},
            "resolved_config_sha256": config_sha, "source_commit": config["source_commit"],
            "code_provenance": source_provenance,
            "physical_batch": physical, "gradient_accumulation": accumulation,
            "initial_pretrained_state_hash": initial_pretrained_hash,
            "cumulative_wall_seconds": 0.0, "precision": "full_fp32",
            "bn_running_state": "frozen", "compression": "identity_disabled",
        }
        initial_path = checkpoint_dir / "epoch_000.pt"
        write_torch_atomic_create(initial_path, initial_payload)
        write_json_x(checkpoint_dir / "epoch_000.json", {
            "epoch": 0, "path": str(initial_path), "bytes": initial_path.stat().st_size,
            "sha256": sha256(initial_path), "complete": True,
        })
        if latest_checkpoint(checkpoint_dir) != initial_path:
            raise RuntimeError("atomic epoch_000 checkpoint verification failed")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["resolved_config_sha256"] != config_sha or payload["source_commit"] != config["source_commit"]:
            raise RuntimeError("resume provenance mismatch")
        model.load_state_dict(payload["model"], strict=True); model.to(device)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
        cumulative_wall = float(payload.get("cumulative_wall_seconds", 0.0))
        initial_pretrained_hash = payload["initial_pretrained_state_hash"]
    started_monotonic = time.monotonic()
    weights = config["loss_weights"]
    batches_per_epoch = math.ceil(len(rows) / physical)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    operational_ceiling = float(config.get("operational", {}).get(
        "scientific_wall_clock_ceiling_seconds", 16 * 3600,
    ))
    for epoch in range(start_epoch, 41):
        if cumulative_wall + (time.monotonic() - started_monotonic) >= operational_ceiling:
            raise TimeoutError("sixteen-hour scientific wall-clock ceiling exhausted")
        stage = "A" if epoch <= 5 else "B"
        configure_stage(model, stage); stage_train_mode(model, stage)
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        loader = DataLoader(dataset, batch_size=physical, sampler=sampler, num_workers=8,
                            pin_memory=True, persistent_workers=False, drop_last=False,
                            collate_fn=collate_training)
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
        sums: defaultdict[str, float] = defaultdict(float); denominator_sums: defaultdict[str, int] = defaultdict(int)
        visited: list[str] = []; clipping_count = 0; preclip_norm_sum = 0.0; optimizer_updates = 0
        last_new_lr = last_backbone_lr = 0.0
        group_micro_count = accumulation
        for batch_index, batch in enumerate(loader, 1):
            if (batch_index - 1) % accumulation == 0:
                group_micro_count = min(accumulation, batches_per_epoch - batch_index + 1)
            total, parts, denominators, _outputs = compute_losses(model, batch, weights)
            if not torch.isfinite(total).item():
                raise FloatingPointError(f"nonfinite scientific loss epoch={epoch} batch={batch_index}")
            (total / group_micro_count).backward()
            for name in SCIENTIFIC_COMPONENTS:
                sums[name] += float(parts[name].detach().item())
            sums["total"] += float(total.detach().item())
            for name, value in denominators.items(): denominator_sums[name] += int(value)
            visited.extend(batch["sample_id"])
            boundary = batch_index % accumulation == 0 or batch_index == batches_per_epoch
            if boundary:
                optimizer_updates += 1; global_step += 1
                last_new_lr, last_backbone_lr = scheduled_lrs(epoch, optimizer_updates, updates_per_epoch, global_step)
                set_optimizer_lrs(optimizer, last_new_lr, last_backbone_lr)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                norm_value = float(norm.item()); preclip_norm_sum += norm_value
                clipping_count += int(norm_value > 5.0)
                if not math.isfinite(norm_value):
                    raise FloatingPointError(f"nonfinite scientific gradient epoch={epoch} update={optimizer_updates}")
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if len(visited) != 16827 or len(set(visited)) != 16827:
            raise RuntimeError(f"epoch {epoch} did not visit every train frame exactly once")
        telemetry_record = telemetry(model, fixed_batch, weights)
        telemetry_record.update({"schema": "route_b_v3_1_depth_aware_lraspp_gradient_telemetry_v1",
                                 "created_utc": utc_now(), "epoch": epoch})
        write_json_x(telemetry_dir / f"epoch_{epoch:03d}.json", telemetry_record)
        current_wall = cumulative_wall + (time.monotonic() - started_monotonic)
        stage_a_hash = tensor_state_hash(pretrained_backbone_state(model)) if epoch <= 5 else None
        if epoch <= 5 and stage_a_hash != initial_pretrained_hash:
            raise RuntimeError(f"pretrained RGB/backbone/BN state changed during Stage A epoch {epoch}")
        metric = {
            "schema": "route_b_v3_1_depth_aware_lraspp_epoch_metrics_v1", "created_utc": utc_now(),
            "epoch": epoch, "stage": stage, "frames_visited": len(visited),
            "unique_frames_visited": len(set(visited)), "batches": batches_per_epoch,
            "optimizer_updates": optimizer_updates, "global_step": global_step,
            "mean_losses": {name: value / batches_per_epoch for name, value in sums.items()},
            "denominators": dict(denominator_sums), "new_lr_last": last_new_lr,
            "backbone_lr_last": last_backbone_lr, "clipping_count": clipping_count,
            "mean_preclip_gradient_norm": preclip_norm_sum / optimizer_updates,
            "epoch_seconds": time.monotonic() - epoch_started, "cumulative_wall_seconds": current_wall,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "stage_a_pretrained_state_hash": stage_a_hash,
        }
        write_json_x(metrics_dir / f"epoch_{epoch:03d}.json", metric)
        checkpoint_payload = {
            "schema": "route_b_v3_1_depth_aware_lraspp_recovery_checkpoint_v1",
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
            "gradient_accumulation_phase": 0, "rng_state": rng_state(),
            "sampler_state": {"epoch": epoch, "seed": seed + epoch, "visited": len(visited),
                              "unique": len(set(visited)), "complete": True},
            "scheduler_state": {"next_epoch": epoch + 1, "global_update": global_step,
                                "schedule": "registered_stage_a_stage_b_stateless_v1"},
            "resolved_config_sha256": config_sha, "source_commit": config["source_commit"],
            "code_provenance": source_provenance,
            "physical_batch": physical, "gradient_accumulation": accumulation,
            "initial_pretrained_state_hash": initial_pretrained_hash,
            "cumulative_wall_seconds": current_wall, "precision": "full_fp32",
            "bn_running_state": "frozen", "compression": "identity_disabled",
        }
        path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        write_torch_atomic_create(path, checkpoint_payload)
        checkpoint_hash = sha256(path)
        write_json_x(checkpoint_dir / f"epoch_{epoch:03d}.json", {
            "epoch": epoch, "path": str(path), "bytes": path.stat().st_size,
            "sha256": checkpoint_hash, "complete": True,
        })
        print(json.dumps({"epoch": epoch, "stage": stage, "loss": metric["mean_losses"]["total"],
                          "seconds": metric["epoch_seconds"], "global_step": global_step,
                          "checkpoint_sha256": checkpoint_hash}), flush=True)
    total_wall = cumulative_wall + (time.monotonic() - started_monotonic)
    write_json_x(experiment / "TRAINING_COMPLETE.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_training_complete_v1", "created_utc": utc_now(),
        "epochs_completed": 40, "scientific_runs": 1, "global_step": global_step,
        "evaluation_during_training": 0, "checkpoint_epochs": list(range(1, 41)),
        "wall_seconds": total_wall, "stage_a_pretrained_state_hash": initial_pretrained_hash,
        "epoch5_pretrained_bit_identical": True,
    })
    write_text_x(experiment / "TRAINING_COMPLETE", "40_EPOCHS_COMPLETE\n")
    print(json.dumps({"training_complete": True, "epochs": 40, "wall_seconds": total_wall}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
