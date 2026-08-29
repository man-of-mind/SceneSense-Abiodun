from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from common import (CONFIG_PATH, load_json, read_csv, restore_rng_state, rng_state, seed_everything,
                    sha256, utc_now, write_json_x, write_text_x, write_torch_atomic_create)
from data import DepthCache, TrainingDataset, collate_training, load_objects, load_visible_anchors
from losses import private_object_losses, representation_losses
from model import build_model, configure_two_stage, freeze_bn_running_state
from two_stage import (assert_allowlist, build_optimizer, latest_checkpoint, model_finite,
                       optimizer_finite, parameter_allowlist, representation_state, scheduled_lr,
                       set_lrs, state_hash, is_representation)

STAGE1_COMPONENTS = ("segmentation", "dense_depth", "radar_consistency")
STAGE2_COMPONENTS = ("heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray",
                     "depth_bin", "depth_residual", "endpoint", "dimensions", "yaw", "parked",
                     "radar_support")


def write_checkpoint(path: Path, payload: dict) -> None:
    write_torch_atomic_create(path, payload)
    write_json_x(path.with_suffix(".json"), {
        "epoch": int(payload["epoch"]), "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256(path), "complete": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True); stage = args.stage
    if not (experiment / "QUALIFICATION_COMPLETE").is_file():
        raise RuntimeError("training requires bounded qualification")
    if stage == "stage2" and not (experiment / "STAGE2_AUTHORIZED").is_file():
        raise RuntimeError("Stage 2 was not authorized by the preregistered Stage-1 gates")
    config = load_json(CONFIG_PATH); resolved_hash = sha256(experiment / "RESOLVED_CONFIG.json")
    registered = load_json(experiment / "REGISTERED_TWO_STAGE_DESIGN.json")
    seed = int(config[f"{stage}_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("scientific CUDA runtime unavailable")
    device = torch.device("cuda")
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "train"]
    if len(rows) != 16827 or len({row["sample_id"] for row in rows}) != 16827:
        raise RuntimeError("train population drift")
    objects = load_objects(dataset_root); visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    cache = DepthCache(experiment / "depth_cache/train", rows)
    dataset = TrainingDataset(dataset_root, rows, objects, visible, cache, seed)
    model, loading = build_model(Path(config["pretrained"]["path"]), device)
    configure_two_stage(model, stage)
    allowlist = registered["parameter_allowlists"][stage]
    assert_allowlist(model, stage, allowlist)
    optimizer = build_optimizer(model, stage)
    assert_allowlist(model, stage, allowlist)
    stage_root = experiment / stage; checkpoint_dir = stage_root / "checkpoints"
    metric_dir = stage_root / "training_metrics"; gradient_dir = stage_root / "gradient_telemetry"
    checkpoint_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True)
    gradient_dir.mkdir(parents=True, exist_ok=True)
    prefix = "epoch" if stage == "stage1" else "stage2_epoch"
    checkpoint = latest_checkpoint(checkpoint_dir, prefix, resolved_hash)
    max_epochs = 20 if stage == "stage1" else 30
    start_epoch = 1; global_step = 0; cumulative_wall = 0.0
    frozen_reference_hash = None
    if checkpoint is None:
        raise RuntimeError(f"missing qualified atomic {stage} epoch-000 checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True); model.to(device); configure_two_stage(model, stage)
    optimizer.load_state_dict(payload["optimizer"]); restore_rng_state(payload["rng_state"])
    start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
    cumulative_wall = float(payload.get("cumulative_wall_seconds", 0.0))
    frozen_reference_hash = payload.get("frozen_representation_hash")
    if stage == "stage2":
        actual = state_hash(model, is_representation)
        if not frozen_reference_hash or actual != frozen_reference_hash:
            raise RuntimeError("Stage-2 epoch-000 frozen representation hash mismatch")
    marker = stage_root / "TRAINING_STARTED.json"
    if not marker.exists():
        write_json_x(marker, {
            "schema": f"two_stage_lraspp_{stage}_training_started_v1", "created_utc": utc_now(),
            "stage": stage, "seed": seed, "official_loading": loading,
            "epochs": max_epochs, "batch": 16, "accumulation": 1, "precision": "full_fp32",
            "validation_during_optimization": False, "parameter_allowlist": allowlist,
            "resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": sha256(checkpoint),
        })
    batches = math.ceil(len(dataset) / 16); updates = batches
    weights = config["loss_weights"]
    loss_function = representation_losses if stage == "stage1" else private_object_losses
    components = STAGE1_COMPONENTS if stage == "stage1" else STAGE2_COMPONENTS
    started = time.monotonic(); ceiling = float(config["operational"]["scientific_wall_clock_ceiling_seconds"])
    for epoch in range(start_epoch, max_epochs + 1):
        if cumulative_wall + time.monotonic() - started >= ceiling:
            raise TimeoutError("eight-hour operational ceiling exhausted")
        configure_two_stage(model, stage); assert_allowlist(model, stage, allowlist)
        dataset.set_epoch(epoch)
        sampler = RandomSampler(dataset, replacement=False,
                                generator=torch.Generator().manual_seed(seed + epoch))
        loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=8,
                            pin_memory=True, persistent_workers=False, drop_last=False,
                            collate_fn=collate_training)
        optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(device)
        sums: defaultdict[str, float] = defaultdict(float); denoms: defaultdict[str, int] = defaultdict(int)
        visited: list[str] = []; epoch_start = time.monotonic(); clip_count = 0; grad_sum = 0.0
        first_gradient = None; last_new = last_pretrained = 0.0
        for batch_index, batch in enumerate(loader, 1):
            total, parts, denominators, _ = loss_function(model, batch, weights)
            if not torch.isfinite(total).item():
                raise FloatingPointError(f"nonfinite {stage} loss epoch={epoch} batch={batch_index}")
            total.backward(); global_step += 1
            last_new, last_pretrained = scheduled_lr(stage, epoch, batch_index, updates)
            set_lrs(optimizer, stage, last_new, last_pretrained)
            norm = torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], 5.0)
            norm_value = float(norm.item())
            if not math.isfinite(norm_value):
                raise FloatingPointError(f"nonfinite {stage} gradient epoch={epoch} batch={batch_index}")
            if first_gradient is None:
                first_gradient = {name: (None if value.grad is None else float(value.grad.float().norm().item()))
                                  for name, value in model.named_parameters() if value.requires_grad}
            grad_sum += norm_value; clip_count += int(norm_value > 5.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for name in components: sums[name] += float(parts[name].detach().item())
            sums["total"] += float(total.detach().item())
            for name, value in denominators.items(): denoms[name] += int(value)
            visited.extend(batch["sample_id"])
        if len(visited) != 16827 or len(set(visited)) != 16827:
            raise RuntimeError(f"{stage} epoch {epoch} did not visit every train frame exactly once")
        if not model_finite(model) or not optimizer_finite(optimizer):
            raise FloatingPointError(f"nonfinite {stage} state epoch={epoch}")
        frozen_hash = state_hash(model, is_representation) if stage == "stage2" else None
        if stage == "stage2" and frozen_hash != frozen_reference_hash:
            raise RuntimeError(f"frozen representation drift at Stage-2 epoch {epoch}")
        wall = cumulative_wall + time.monotonic() - started
        metric = {
            "schema": f"two_stage_lraspp_{stage}_epoch_metrics_v1", "created_utc": utc_now(),
            "epoch": epoch, "frames_visited": len(visited), "unique_frames_visited": len(set(visited)),
            "batches": batches, "optimizer_updates": batches, "global_step": global_step,
            "mean_losses": {name: value / batches for name, value in sums.items()},
            "denominators": dict(denoms), "new_lr_last": last_new,
            "pretrained_lr_last": last_pretrained if stage == "stage1" else None,
            "clipping_count": clip_count, "mean_preclip_gradient_norm": grad_sum / batches,
            "epoch_seconds": time.monotonic() - epoch_start, "cumulative_wall_seconds": wall,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "frozen_representation_hash": frozen_hash,
        }
        write_json_x(metric_dir / f"epoch_{epoch:03d}.json", metric)
        write_json_x(gradient_dir / f"epoch_{epoch:03d}.json", {
            "schema": f"two_stage_lraspp_{stage}_gradient_telemetry_v1", "created_utc": utc_now(),
            "epoch": epoch, "first_update_parameter_gradient_norms": first_gradient,
            "all_finite": all(value is not None and math.isfinite(value) for value in first_gradient.values()),
        })
        checkpoint_payload = {
            "schema": f"two_stage_lraspp_{stage}_checkpoint_v1", "model": {
                name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
            "rng_state": rng_state(), "sampler_state": {"epoch": epoch, "seed": seed + epoch,
                "visited": len(visited), "unique": len(set(visited)), "complete": True},
            "scheduler_state": {"next_epoch": epoch + 1, "schedule": f"registered_{stage}_warmup_cosine_v1"},
            "resolved_config_sha256": resolved_hash, "source_commit": config["source_commit"],
            "batch": 16, "accumulation": 1, "cumulative_wall_seconds": wall,
            "frozen_representation_hash": frozen_reference_hash,
        }
        path = checkpoint_dir / f"{prefix}_{epoch:03d}.pt"; write_checkpoint(path, checkpoint_payload)
        print(json.dumps({"stage": stage, "epoch": epoch, "loss": metric["mean_losses"]["total"],
                          "seconds": metric["epoch_seconds"], "peak_reserved_mib": metric["cuda_peak_reserved_mib"],
                          "checkpoint_sha256": sha256(path)}), flush=True)
    total_wall = cumulative_wall + time.monotonic() - started
    write_json_x(stage_root / "TRAINING_COMPLETE.json", {
        "schema": f"two_stage_lraspp_{stage}_training_complete_v1", "created_utc": utc_now(),
        "epochs_completed": max_epochs, "scientific_runs": 1, "global_step": global_step,
        "evaluation_during_training": 0, "wall_seconds": total_wall,
        "checkpoint_epochs": list(range(1, max_epochs + 1)),
        "frozen_representation_hash": frozen_reference_hash,
    })
    write_text_x(stage_root / "TRAINING_COMPLETE", f"{max_epochs}_EPOCHS_COMPLETE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
