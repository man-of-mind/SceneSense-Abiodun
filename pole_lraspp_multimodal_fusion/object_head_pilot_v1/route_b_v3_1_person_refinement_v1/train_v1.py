#!/usr/bin/env python3
"""Train the registered person-only refinement for exactly 18 epochs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_v1 import NATIVE_GRID  # noqa: E402
from person_losses_v1 import person_refinement_loss  # noqa: E402
from person_model_v1 import (  # noqa: E402
    build_model, configure_stage, inherited_person_parameters, load_recovered_base,
    new_parameters, parameter_report,
)
from person_targets_v1 import PersonRefinementDataset  # noqa: E402
from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402

CHECKPOINT_EPOCHS = (6, 12, 18)
FIELDS = (
    "epoch", "stage", "new_lr_start", "new_lr_end", "inherited_person_lr_start",
    "inherited_person_lr_end", "total_loss", "center_loss", "quality_loss",
    "range_bin_loss", "range_residual_loss", "projected_offset_loss",
    "local_xy_endpoint_loss", "person_mask_loss", "local_xy_error_mean_m",
    "person_positive_cells", "center_hard_negatives_selected",
    "mask_hard_negatives_selected", "clipped_offset_targets", "batches",
    "optimizer_steps", "epoch_seconds", "cuda_allocated_peak_mib",
    "cuda_reserved_peak_mib", "created_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(states: dict[str, Any]) -> None:
    if set(states) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise RuntimeError("candidate recovery RNG state is incomplete")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    torch.cuda.set_rng_state_all(states["torch_cuda"])


def learning_rates(epoch: int, batch_index: int, batches: int,
                   design: dict[str, Any]) -> tuple[float, float]:
    if epoch <= 6:
        stage = design["stage_p1"]
        first, last = 1, 6
    else:
        stage = design["stage_p2"]
        first, last = 7, 18
    total_steps = (last - first + 1) * batches
    index = (epoch - first) * batches + batch_index
    progress = index / float(max(1, total_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = float(design["final_lr_ratio"]) + (1.0 - float(design["final_lr_ratio"])) * cosine
    return float(stage["new_head_lr"]) * ratio, float(stage["inherited_person_lr"]) * ratio


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    digest = sha256(path)
    verify = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "grad_scaler", "rng_states", "epoch", "registration"}
    if not required.issubset(verify) or int(verify["epoch"]) != int(payload["epoch"]):
        raise RuntimeError(f"checkpoint integrity failure: {path}")
    if not all(torch.isfinite(value).all().item() for value in verify["model"].values()):
        raise RuntimeError(f"nonfinite checkpoint tensor: {path}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--numerical-policy-registration", required=True, type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    design = config["person_design"]
    registration_path = experiment / "REGISTRATION.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    numerical_policy_path = args.numerical_policy_registration.resolve(strict=True)
    numerical_policy = json.loads(numerical_policy_path.read_text(encoding="utf-8"))
    if not registration["all_frozen_before_training"]:
        raise RuntimeError("refinement registration is not closed")
    if sha256(args.config) != registration["resolved_config_sha256"]:
        raise RuntimeError("resolved refinement configuration changed after registration")
    required_policy = {
        "frozen_feature_compute": "existing_fp16_no_grad",
        "detached_native_feature": "fp32",
        "complete_person_refinement_tail": "fp32",
        "inherited_trainable_person_heatmap": "fp32",
        "person_losses_and_unprojection": "fp32",
        "optimizer_parameters_and_state": "fp32",
        "grad_scaler_enabled": False,
    }
    if (
        numerical_policy.get("schema")
        != "route_b_v3_1_person_refinement_numerical_policy_registration_v3"
        or numerical_policy.get("authorized") is not True
        or numerical_policy.get("policy") != required_policy
        or numerical_policy.get("source_registration_sha256") != sha256(registration_path)
        or numerical_policy.get("base_checkpoint_sha256") != args.base_sha256
    ):
        raise RuntimeError("full-FP32 person numerical policy registration is invalid")
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA environment unavailable")
    if int(design["epochs"]) != 18 or tuple(design["checkpoint_epochs"]) != CHECKPOINT_EPOCHS:
        raise RuntimeError("registered 18-epoch/6-12-18 schedule drift")
    if design["geometric_augment"] is not False or design["q"] != 0 or design["ae"] is not False:
        raise RuntimeError("clean, geometry-free q0/no-AE contract drift")

    device = torch.device("cuda")
    set_reproducible_seeds(int(design["training_seed"]))
    base_path = args.base_checkpoint.resolve(strict=True)
    base_hash = sha256(base_path)
    if base_hash != args.base_sha256 or base_hash != registration["base_checkpoint_sha256"]:
        raise RuntimeError("recovered epoch-40 base SHA mismatch")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    object_cfg = dict(base["config"]["object_heads"])
    rows = read_manifest(experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 16827 or len(val_rows) != 3345 or {row["split"] for row in rows} != {"train", "val"}:
        raise RuntimeError("expanded train/validation population drift")
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    dataset = PersonRefinementDataset(
        experiment / "dataset", train_rows, object_rows,
        tuple(config["registered_input_size"]), object_cfg,
        augment_strength=str(design["augment_strength"]),
        geometric_augment=False,
        range_edges=registration["range_bins"]["edges_m"],
        offset_caps=design["projected_offset_cap_grid_xy"],
    )
    weights = torch.as_tensor(registration["sampler"]["normalized_weights"], dtype=torch.double)
    if len(weights) != len(dataset) or registration["sampler"]["validation_rows_used_by_sampler_or_mining"] != 0:
        raise RuntimeError("registered sampler population drift")

    model = build_model(
        radar_channels=int(base["radar_channels"]),
        hidden_channels=int(base["object_hidden_channels"]),
        head_depth=int(base["object_head_depth"]),
        person_hidden=int(design["hidden_channels"]),
        group_norm_groups=int(design["group_norm_groups"]),
        range_bins=int(design["range_bins"]), device=device,
    )
    load_mapping = load_recovered_base(model, base_path, device=device)
    configure_stage(model, "P1")
    optimizer = torch.optim.AdamW([
        {"params": new_parameters(model), "lr": 0.0, "name": "new_person_tail"},
        {"params": inherited_person_parameters(model), "lr": 0.0, "name": "inherited_person_heatmap"},
    ], lr=0.0, weight_decay=float(design["weight_decay"]))
    if not all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for group in optimizer.param_groups for parameter in group["params"]
    ):
        raise RuntimeError("person optimizer parameters are not FP32")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, optimizer_steps = 1, 0
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve(strict=True)
        if sha256(resume_path) != args.resume_sha256:
            raise RuntimeError("candidate retry checkpoint SHA mismatch")
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["grad_scaler"])
        restore_rng(resume["rng_states"])
        start_epoch = int(resume["epoch"]) + 1
        optimizer_steps = int(resume["optimizer_steps"])
        if resume["registration_sha256"] != sha256(registration_path):
            raise RuntimeError("candidate retry registration mismatch")
        if resume.get("numerical_policy_registration_sha256") != sha256(numerical_policy_path):
            raise RuntimeError("candidate retry numerical policy mismatch")

    checkpoint_dir = experiment / "checkpoints" / config["name"]
    recovery_dir = experiment / "candidate_recovery_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = experiment / "person_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "training_metrics.csv"
    if start_epoch == 1:
        with metrics_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        write_json_x(experiment / "PERSON_TRAINING_STARTED.json", {
            "schema": "route_b_v3_1_person_refinement_training_started_v1",
            "created_utc": utc_now(), "attempt": args.attempt, "config": config,
            "registration_sha256": sha256(registration_path), "base_mapping": load_mapping,
            "numerical_policy_registration": str(numerical_policy_path),
            "numerical_policy_registration_sha256": sha256(numerical_policy_path),
            "base_checkpoint_sha256": base_hash, "train_frames": len(train_rows),
            "validation_frames_not_used_for_training_or_mining": len(val_rows),
            "sampling_draws_per_epoch": int(design["sampling"]["num_samples_per_epoch"]),
        })
    existing_rows = []
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            existing_rows = list(csv.DictReader(stream))
    if existing_rows and int(existing_rows[-1]["epoch"]) != start_epoch - 1:
        raise RuntimeError("candidate metrics/checkpoint recovery boundary mismatch")

    peak_allocated = 0.0
    peak_reserved = 0.0
    for epoch in range(start_epoch, 19):
        stage = "P1" if epoch <= 6 else "P2"
        configure_stage(model, stage)
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(int(design["training_seed"]) + epoch)
        sampler = WeightedRandomSampler(
            weights, num_samples=int(design["sampling"]["num_samples_per_epoch"]),
            replacement=True, generator=epoch_generator,
        )
        loader = DataLoader(
            dataset, batch_size=int(design["batch_size"]), sampler=sampler,
            drop_last=False, num_workers=int(design["num_workers"]), pin_memory=True,
            persistent_workers=bool(design["persistent_workers"]),
            prefetch_factor=int(design["prefetch_factor"]),
        )
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        sums: dict[str, float] = {}
        batches = 0
        first_lr: tuple[float, float] | None = None
        last_lr: tuple[float, float] | None = None
        for batch_index, (tensors, masks, targets) in enumerate(loader):
            new_lr, inherited_lr = learning_rates(epoch, batch_index, len(loader), design)
            first_lr = first_lr or (new_lr, inherited_lr)
            last_lr = (new_lr, inherited_lr)
            optimizer.param_groups[0]["lr"] = new_lr
            optimizer.param_groups[1]["lr"] = inherited_lr
            tensors = tensors.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", enabled=bool(design["amp"]), dtype=torch.float16,
                cache_enabled=bool(design["autocast_cache_enabled"]),
            ):
                outputs = model.training_outputs(tensors)
            with torch.autocast(device_type="cuda", enabled=False):
                loss, parts = person_refinement_loss(
                    outputs, masks, targets,
                    range_edges=registration["range_bins"]["edges_m"],
                    offset_caps=design["projected_offset_cap_grid_xy"], design=design,
                )
            if not math.isfinite(float(loss.detach().item())):
                raise RuntimeError(f"nonfinite refinement loss epoch={epoch} batch={batch_index + 1}")
            loss.backward()
            nonfinite_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            ]
            if nonfinite_gradients:
                raise RuntimeError(
                    f"nonfinite full-FP32 gradients epoch={epoch} batch={batch_index + 1} "
                    f"parameters={nonfinite_gradients}"
                )
            optimizer.step()
            optimizer_steps += 1
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        assert first_lr is not None and last_lr is not None
        if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
            raise RuntimeError(f"nonfinite model state after epoch {epoch}")
        non_fp32_optimizer_state = [
            (group_index, state_name, str(value.dtype))
            for group_index, group in enumerate(optimizer.param_groups)
            for parameter in group["params"]
            for state_name, value in optimizer.state.get(parameter, {}).items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            and value.dtype != torch.float32
        ]
        if non_fp32_optimizer_state:
            raise RuntimeError(
                f"non-FP32 optimizer state after epoch {epoch}: {non_fp32_optimizer_state}"
            )
        allocated = torch.cuda.max_memory_allocated(device) / 2**20
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
        peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
        row = {
            "epoch": epoch, "stage": stage,
            "new_lr_start": first_lr[0], "new_lr_end": last_lr[0],
            "inherited_person_lr_start": first_lr[1], "inherited_person_lr_end": last_lr[1],
            **{key: sums.get(key, 0.0) / batches for key in FIELDS if key in sums},
            "batches": batches, "optimizer_steps": optimizer_steps,
            "epoch_seconds": time.monotonic() - epoch_started,
            "cuda_allocated_peak_mib": allocated, "cuda_reserved_peak_mib": reserved,
            "created_utc": utc_now(),
        }
        with metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        write_json_x(metrics_dir / f"epoch_{epoch:03d}.json", row)
        designated = epoch in CHECKPOINT_EPOCHS
        checkpoint_path = (
            checkpoint_dir / f"epoch_{epoch:03d}.pt" if designated
            else recovery_dir / f"epoch_{epoch:03d}.pt"
        )
        payload = {
            "schema": "route_b_v3_1_person_refinement_checkpoint_v1",
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(), "rng_states": rng_states(),
            "epoch": epoch, "optimizer_steps": optimizer_steps, "stage": stage,
            "config": config, "registration": registration,
            "registration_sha256": sha256(registration_path),
            "numerical_policy_registration": numerical_policy,
            "numerical_policy_registration_sha256": sha256(numerical_policy_path),
            "native_checkpoint": str(base_path), "native_checkpoint_sha256": base_hash,
            "radar_channels": int(base["radar_channels"]),
            "object_hidden_channels": int(base["object_hidden_channels"]),
            "object_head_depth": int(base["object_head_depth"]),
            "object_class_names": list(base["object_class_names"]),
            "native_stride": int(base["native_stride"]),
            "native_grid": list(base.get("native_grid", NATIVE_GRID)),
            "parameter_report": parameter_report(model),
        }
        checkpoint_hash = save_checkpoint(checkpoint_path, payload)
        latest = {
            "epoch": epoch, "path": str(checkpoint_path), "sha256": checkpoint_hash,
            "optimizer_steps": optimizer_steps, "created_utc": utc_now(),
        }
        old = json.loads((experiment / "PERSON_LATEST_SAFE.json").read_text()) if (experiment / "PERSON_LATEST_SAFE.json").is_file() else None
        write_json_atomic(experiment / "PERSON_LATEST_SAFE.json", latest)
        if old:
            old_path = Path(old["path"])
            if old_path.parent == recovery_dir and old_path != checkpoint_path and old_path.is_file():
                old_path.unlink()
        print(
            f"[person train] epoch={epoch}/18 stage={stage} loss={row['total_loss']:.6f} "
            f"xy={row['local_xy_error_mean_m']:.4f} new_lr={last_lr[0]:.8g} "
            f"person_lr={last_lr[1]:.8g}", flush=True,
        )

    paths = [checkpoint_dir / f"epoch_{epoch:03d}.pt" for epoch in CHECKPOINT_EPOCHS]
    result = {
        "schema": "route_b_v3_1_person_refinement_training_complete_v1",
        "created_utc": utc_now(), "epochs_completed": 18,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "checkpoints": [{"epoch": epoch, "path": str(path), "sha256": sha256(path)}
                        for epoch, path in zip(CHECKPOINT_EPOCHS, paths)],
        "optimizer_steps": optimizer_steps, "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "validation_rows_used_for_training_or_mining": 0,
        "numerical_policy_registration": str(numerical_policy_path),
        "numerical_policy_registration_sha256": sha256(numerical_policy_path),
        "grad_scaler_enabled": False,
        "no_ordinary_early_stop": True,
    }
    write_json_x(experiment / "PERSON_TRAINING_COMPLETE.json", result)
    (experiment / "PERSON_TRAINING_COMPLETE").write_text("EXACTLY_18_EPOCHS_COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
