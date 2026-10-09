from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import numpy as np
from torch.utils.data import DataLoader

from common import (CONFIG_PATH, ROOT, atomic_json, atomic_text, atomic_torch, capture_rng,
                    canonical_hash, desktop_notify, finite_tree, load_json, named_tensor_hash, package_hashes,
                    restore_rng, seed_everything, sha256, utc_now)
from data import DepthCache, FrozenEpochSampler, RouteBDataset, collate, load_split_rows
from losses import compute_loss_groups, scalar_components
from model import SplitFusionFCOS, build_model, configure_trainability, optimizer_parameter_groups


def build_optimizer(model: SplitFusionFCOS, config: Mapping[str, Any]) -> torch.optim.SGD:
    groups = optimizer_parameter_groups(model)
    optimizer = torch.optim.SGD([
        {"params": [parameter for _, parameter in groups[name]], "lr": 0.0, "name": name}
        for name in ("pretrained_backbone", "pretrained_fpn_heads", "new")
    ], momentum=float(config["training"]["momentum"]), weight_decay=float(config["training"]["weight_decay"]))
    if optimizer.state:
        raise RuntimeError("fresh optimizer unexpectedly has state")
    return optimizer


def scheduled_lrs(config: Mapping[str, Any], epoch: int, optimizer_update: int) -> dict[str, float]:
    bases = config["training"]["base_lrs"]
    if epoch <= 3:
        progress = min(1.0, max(0.0, (optimizer_update - 1) / max(1, config["training"]["warmup_updates"] - 1)))
        factor = config["training"]["warmup_start_factor"] + progress * (1.0 - config["training"]["warmup_start_factor"])
        return {"pretrained_backbone": 0.0, "pretrained_fpn_heads": 0.0, "new": float(bases["new"]) * factor}
    decay = 0 if epoch <= 16 else 1 if epoch <= 22 else 2
    return {name: float(value) * float(config["training"]["gamma"]) ** decay for name, value in bases.items()}


def set_lrs(optimizer: torch.optim.Optimizer, values: Mapping[str, float]) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(values[group["name"]])


def all_model_finite(model: torch.nn.Module) -> bool:
    return all(not value.dtype.is_floating_point or bool(torch.isfinite(value).all())
               for value in itertools.chain(model.parameters(), model.buffers()))


def all_gradients_finite(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    return all(not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point or bool(torch.isfinite(value).all())
               for state in optimizer.state.values() for value in state.values())


def gradient_norm(parameter: torch.nn.Parameter) -> float:
    return float(parameter.grad.detach().float().norm()) if parameter.grad is not None else 0.0


def require_registered_source(experiment: Path) -> None:
    registered = load_json(experiment / "SCIENTIFIC_REGISTRATION.json")["source_state"]
    current = package_hashes()
    current_hash = canonical_hash(current)
    if current == registered["files"] and current_hash == registered["canonical_sha256"]:
        return
    expected_hash = registered["canonical_sha256"]
    amendments = sorted(experiment.glob("SOURCE_AMENDMENT_*.json"))
    if not amendments:
        raise RuntimeError("scientific source differs from Phase A registration without an amendment")
    for path in amendments:
        amendment = load_json(path)
        if (amendment.get("base_source_state_sha256",
                          amendment.get("base_registration_source_state_sha256")) != expected_hash
                or amendment.get("scientific_settings_changed") is not False
                or amendment.get("scope") not in {"diagnostic_runtime_only", "training_runtime_guard_only"}):
            raise RuntimeError(f"source-amendment chain drift: {path}")
        expected_hash = amendment["amended_source_state_sha256"]
    if expected_hash != current_hash or load_json(amendments[-1])["amended_source_files"] != current:
        raise RuntimeError("final amended source-state provenance drift")


def required_gradient_evidence(model: SplitFusionFCOS) -> dict[str, dict[str, Any]]:
    selectors = {
        "radar_stem": lambda name: name == "front.W_radar",
        "rgb_stem": lambda name: name == "front.W_rgb",
        "p2": lambda name: name.startswith("tail.p2_"),
        "project_classifier": lambda name: name.startswith("project_classifier"),
        "fcos_box_regression": lambda name: name.startswith("regression_head.bbox_reg"),
        "fcos_centerness": lambda name: name.startswith("regression_head.bbox_ctrness"),
        "semantic": lambda name: name.startswith("semantic"),
        "dense_depth": lambda name: name.startswith("dense_depth"),
        "geometry_tower": lambda name: name.startswith("geometry.tower"),
        "geometry_depth_bins": lambda name: name.startswith("geometry.outputs.depth_bin_logits"),
        "geometry_depth_residual": lambda name: name.startswith("geometry.outputs.depth_bin_residuals"),
        "geometry_ray": lambda name: name.startswith("geometry.outputs.physical_ray"),
        "geometry_dimensions": lambda name: name.startswith("geometry.outputs.log_dimensions"),
        "geometry_yaw": lambda name: name.startswith("geometry.outputs.yaw"),
    }
    report = {}
    named = list(model.named_parameters())
    for group, selector in selectors.items():
        parameters = [parameter for name, parameter in named if selector(name) and parameter.requires_grad]
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        report[group] = {
            "required_this_stage": bool(parameters),
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(gradients),
            "finite": bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients),
            "nonzero": bool(gradients) and any(bool(torch.count_nonzero(value)) for value in gradients),
            "l2": math.sqrt(sum(float(value.detach().float().norm()) ** 2 for value in gradients)),
        }
    return report


def checkpoint_payload(model: SplitFusionFCOS, optimizer: torch.optim.Optimizer, config: Mapping[str, Any],
                       experiment: Path, epoch: int, global_update: int, sampler_state: Mapping[str, Any],
                       lrs: Mapping[str, float]) -> dict[str, Any]:
    return {
        "schema": "splitfusion_fcos_atomic_recovery_checkpoint_v1", "created_utc": utc_now(),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scheduler": {"kind": "absolute_epoch_multistep", "lrs": dict(lrs),
                                                             "milestones_after_epochs": [16, 22], "gamma": 0.1},
        "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": None},
        "epoch": int(epoch), "global_optimizer_update": int(global_update), "rng": capture_rng(),
        "sampler": dict(sampler_state), "config_sha256": sha256(CONFIG_PATH),
        "registration_hashes": load_json(experiment / "REGISTRATION_HASHES.json"),
        "source_hashes": package_hashes(), "validation_accessed": False,
    }


def diagnostic_telemetry(model: SplitFusionFCOS, dataset: RouteBDataset, indices_sets: Sequence[Sequence[int]],
                         multipliers: Mapping[str, float], physical: int) -> dict[str, Any]:
    rng = capture_rng(); was_training = model.training; rgb_requires_grad = model.front.W_rgb.requires_grad
    rows = []
    try:
        # Telemetry measures both stem slices even while the scientific warm-up
        # freezes RGB. This flag change is diagnostic-only: autograd.grad never
        # writes .grad, no optimizer is called, and the original flag is restored.
        model.front.W_rgb.requires_grad_(True)
        model.train()
        for batch_number, indices in enumerate(indices_sets):
            chunks = [indices[start:start + physical] for start in range(0, len(indices), physical)]
            scale = 1.0 / len(chunks); groups = ("D", "G", "S", "A")
            squared = {group: 0.0 for group in groups}
            cross = {pair: 0.0 for pair in itertools.combinations(groups, 2)}
            stem_sum = {group: {"rgb": None, "radar": None} for group in groups}
            p2_rows, sample_ids = [], []
            for chunk in chunks:
                batch = collate([dataset[index] for index in chunk]); sample_ids.extend(batch["sample_ids"])
                _total, parts, audit, outputs = compute_loss_groups(model, batch, multipliers)
                local = {}
                for group in groups:
                    value = torch.autograd.grad(parts[group], outputs["c2"], retain_graph=True)[0].detach().float() * scale
                    local[group] = value.reshape(-1); squared[group] += float(local[group].dot(local[group]))
                for pair in cross:
                    cross[pair] += float(local[pair[0]].dot(local[pair[1]]))
                for group in groups:
                    rgb, radar = torch.autograd.grad(parts[group], (model.front.W_rgb, model.front.W_radar),
                                                     retain_graph=True, allow_unused=True)
                    for name, value in (("rgb", rgb), ("radar", radar)):
                        if value is not None:
                            value = value.detach().float() * scale
                            stem_sum[group][name] = value if stem_sum[group][name] is None else stem_sum[group][name] + value
                p2_rows.append(audit["assignment"]["p2_loss_fraction"])
                del batch, _total, parts, audit, outputs, local
            gradients = {group: math.sqrt(value) for group, value in squared.items()}
            cosines = {f"{left}:{right}": cross[(left, right)] /
                       max(1e-12, gradients[left] * gradients[right]) for left, right in cross}
            stem = {group: {name: float(value.norm()) if value is not None else 0.0
                            for name, value in stem_sum[group].items()} for group in groups}
            p2_fraction = {name: sum(row[name] for row in p2_rows) / len(p2_rows) for name in p2_rows[0]}
            rows.append({"batch": batch_number, "sample_ids": sample_ids,
                         "sample_ids_sha256": __import__("hashlib").sha256("\n".join(sample_ids).encode()).hexdigest(),
                         "frames": len(sample_ids), "physical_microbatches": len(chunks),
                         "c2_gradient_norms": gradients, "pairwise_cosines": cosines,
                         "stem_gradient_norms": stem, "p2_loss_fraction": p2_fraction})
    finally:
        model.front.W_rgb.requires_grad_(rgb_requires_grad)
        model.train(was_training); restore_rng(rng)
    return {"schema": "splitfusion_fcos_epoch_gradient_telemetry_v1", "created_utc": utc_now(),
            "batches": rows, "optimizer_updates": 0, "rng_and_mode_restored": True,
            "rgb_requires_grad_restored": model.front.W_rgb.requires_grad == rgb_requires_grad,
            "diagnostic_only": True, "validation_accessed": False}


def dataloader(dataset: RouteBDataset, seed: int, epoch: int, physical: int,
               workers: int, start_index: int = 0) -> tuple[DataLoader, FrozenEpochSampler]:
    sampler = FrozenEpochSampler(len(dataset), seed, epoch, start_index)
    loader = DataLoader(dataset, batch_size=physical, sampler=sampler, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0, drop_last=False, collate_fn=collate)
    return loader, sampler


def microbatch_groups(loader: DataLoader, accumulation: int) -> Iterable[list[Mapping[str, Any]]]:
    iterator = iter(loader)
    while True:
        values = []
        for _ in range(accumulation):
            try:
                values.append(next(iterator))
            except StopIteration:
                break
        if not values:
            break
        yield values


def run_updates(model: SplitFusionFCOS, optimizer: torch.optim.Optimizer, loader: DataLoader,
                config: Mapping[str, Any], multipliers: Mapping[str, float], epoch: int,
                global_update: int, accumulation: int, maximum_updates: int | None = None,
                enforce_required_nonzero: bool = False) -> tuple[int, dict[str, Any]]:
    model.train(); totals = defaultdict(float); component_count = 0; update_records = []
    assignment_totals = Counter(); p2_fraction_sums = Counter(); p2_audit_count = 0; radar_norms = []; rgb_norms = []
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for update_in_epoch, microbatches in enumerate(microbatch_groups(loader, accumulation), 1):
        if maximum_updates is not None and update_in_epoch > maximum_updates:
            break
        lrs = scheduled_lrs(config, epoch, global_update + 1); set_lrs(optimizer, lrs)
        optimizer.zero_grad(set_to_none=True); update_loss = 0.0
        update_parts = defaultdict(float); update_audits = []
        for batch in microbatches:
            total, parts, audit, _outputs = compute_loss_groups(model, batch, multipliers)
            scalars = scalar_components(parts)
            if not finite_tree(scalars): raise FloatingPointError(f"nonfinite individual loss epoch={epoch} update={update_in_epoch}")
            (total / len(microbatches)).backward()
            update_loss += float(total.detach()) / len(microbatches)
            for name, value in scalars.items(): update_parts[name] += value / len(microbatches)
            update_audits.append(audit)
            del total, parts, audit, _outputs
        if not all_gradients_finite(model): raise FloatingPointError(f"nonfinite gradient epoch={epoch} update={update_in_epoch}")
        required = required_gradient_evidence(model)
        failed = {name: value for name, value in required.items()
                  if value["required_this_stage"] and
                  (not value["finite"] or (enforce_required_nonzero and not value["nonzero"]))}
        if failed:
            raise RuntimeError(f"required-gradient qualification failure epoch={epoch} update={update_in_epoch}: {failed}")
        radar_norm, rgb_norm = gradient_norm(model.front.W_radar), gradient_norm(model.front.W_rgb)
        optimizer.step(); global_update += 1
        if not all_model_finite(model): raise FloatingPointError(f"nonfinite parameter epoch={epoch} update={update_in_epoch}")
        if not optimizer_finite(optimizer): raise FloatingPointError(f"nonfinite optimizer epoch={epoch} update={update_in_epoch}")
        radar_norms.append(radar_norm); rgb_norms.append(rgb_norm)
        for name, value in update_parts.items(): totals[name] += value
        component_count += 1
        for audit in update_audits:
            p2_audit_count += 1
            assignment_totals["foreground"] += audit["assignment"]["foreground"]
            p2_foreground = sum(image["per_class_level"][class_name]["p2"]
                                for image in audit["assignment"]["per_image"]
                                for class_name in ("vehicle", "person"))
            assignment_totals["p2_foreground"] += p2_foreground
            assignment_totals["p3_p7_foreground"] += audit["assignment"]["foreground"] - p2_foreground
            for name, value in audit["assignment"]["p2_loss_fraction"].items(): p2_fraction_sums[name] += value
        update_records.append({"update_in_epoch": update_in_epoch, "global_update": global_update,
                               "loss": update_loss, "lrs": lrs, "microbatches": len(microbatches),
                               "radar_stem_gradient_norm": radar_norm, "rgb_stem_gradient_norm": rgb_norm,
                               "required_gradient_evidence": required, "finite": True})
        if update_in_epoch == 1 or update_in_epoch % 100 == 0:
            print(json.dumps({"epoch": epoch, "update": update_in_epoch, "global": global_update,
                              "loss": update_loss, "lrs": lrs}), flush=True)
    denominator = max(1, component_count)
    summary = {"epoch": epoch, "updates": component_count, "global_update": global_update,
               "mean_components": {name: value / denominator for name, value in totals.items()},
               "last_lrs": update_records[-1]["lrs"] if update_records else {},
               "radar_stem_gradient_norm_mean": float(np.mean(radar_norms)) if radar_norms else 0.0,
               "rgb_stem_gradient_norm_mean": float(np.mean(rgb_norms)) if rgb_norms else 0.0,
               "radar_stem_gradient_nonzero_updates": sum(value > 0 for value in radar_norms),
               "rgb_stem_gradient_nonzero_updates": sum(value > 0 for value in rgb_norms),
               "p2_loss_fraction_mean": {name: value / max(1, p2_audit_count)
                                         for name, value in p2_fraction_sums.items()},
               "foreground_locations": dict(assignment_totals),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
               "wall_seconds": time.monotonic() - started, "all_updates_finite": True,
               "update_boundary_records": update_records}
    return global_update, summary


def prepare(experiment: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], RouteBDataset, torch.device]:
    config = load_json(CONFIG_PATH); runtime = load_json(experiment / "QUALIFIED_RUNTIME.json")
    calibration = load_json(experiment / "LOSS_CALIBRATION.json")
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    rows = load_split_rows(dataset_root, "train")
    cache = DepthCache((ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, (12288 * 2**20) / total), device)
    return config, runtime, calibration, dataset, device


def qualification(experiment: Path) -> int:
    if not (experiment / "STRUCTURAL_QUALIFICATION_COMPLETE").is_file(): raise RuntimeError("structural qualification incomplete")
    if not (experiment / "P2_ASSIGNMENT_AUDIT_COMPLETE").is_file(): raise RuntimeError("full assignment audit incomplete")
    require_registered_source(experiment)
    config, runtime, calibration, dataset, device = prepare(experiment)
    seed = int(config["scientific_seed"]); priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json")
    registration = load_json(experiment / "SCIENTIFIC_REGISTRATION.json")
    seed_everything(seed); model, _ = build_model(priors, device); configure_trainability(model, 1)
    initial_hash = named_tensor_hash(model.state_dict().items())
    if initial_hash != registration["initial_model"]["state_sha256"]: raise RuntimeError("disposable initial-state drift")
    optimizer = build_optimizer(model, config)
    physical, accumulation = int(runtime["physical_batch"]), int(runtime["gradient_accumulation"])
    dataset.set_epoch(1); loader, sampler = dataloader(dataset, seed, 1, physical, config["training"]["workers"])
    global_update, epoch_summary = run_updates(model, optimizer, loader, config, calibration["multipliers"],
                                               1, 0, accumulation, enforce_required_nonzero=True)
    configure_trainability(model, 4); dataset.set_epoch(4)
    joint_loader, joint_sampler = dataloader(dataset, seed, 4, physical, config["training"]["workers"])
    global_update, joint_summary = run_updates(model, optimizer, joint_loader, config, calibration["multipliers"],
                                               4, global_update, accumulation, maximum_updates=32,
                                               enforce_required_nonzero=True)
    archive = experiment / "QUALIFICATION_ONLY_DO_NOT_USE"; archive.mkdir(parents=True, exist_ok=False)
    atomic_torch(archive / "disposable_final_state.pt", checkpoint_payload(
        model, optimizer, config, experiment, 1, global_update, joint_sampler.state_dict(),
        joint_summary["last_lrs"]))
    atomic_text(archive / "NOT_A_SCIENTIFIC_CHECKPOINT", "QUALIFICATION_ONLY_DO_NOT_USE\n", overwrite=False)
    del model, optimizer; torch.cuda.empty_cache()
    seed_everything(seed); fresh, _ = build_model(priors, device); configure_trainability(fresh, 1)
    reconstructed_hash = named_tensor_hash(fresh.state_dict().items()); fresh_optimizer = build_optimizer(fresh, config)
    if reconstructed_hash != initial_hash or fresh_optimizer.state:
        raise RuntimeError("fresh scientific reconstruction after disposable qualification failed")
    report = {"schema": "splitfusion_fcos_disposable_epoch_qualification_v1", "created_utc": utc_now(),
              "complete_frames": len(dataset), "epoch1": epoch_summary, "joint_updates": joint_summary,
              "disposable_optimizer_updates": global_update, "all_checks_every_update": True,
              "archive": str(archive), "archive_sha256": sha256(archive / "disposable_final_state.pt"),
              "initial_state_sha256": initial_hash, "reconstructed_state_sha256": reconstructed_hash,
              "transferred_new_tensor_hashes_match": reconstructed_hash == initial_hash,
              "fresh_optimizer_empty": not bool(fresh_optimizer.state), "validation_accessed": False,
              "pass": epoch_summary["all_updates_finite"] and joint_summary["all_updates_finite"]
                      and reconstructed_hash == initial_hash and not bool(fresh_optimizer.state)}
    atomic_json(experiment / "DISPOSABLE_QUALIFICATION.json", report, overwrite=False)
    atomic_text(experiment / "QUALIFICATION_COMPLETE", "FULL_DISPOSABLE_EPOCH_AND_32_JOINT_UPDATES_QUALIFIED\n", overwrite=False)
    atomic_json(experiment / "STATUS.json", {"phase": "B", "state": "complete", "created_utc": utc_now(),
                                              "validation_accessed": False, "scientific_optimizer_steps": 0})
    atomic_json(experiment / "NOTIFICATION_QUALIFICATION_COMPLETE.json", desktop_notify(
        "SplitFusion FCOS", "Phase B disposable epoch and 32 joint updates qualified; scientific state reconstructed."), overwrite=False)
    print(json.dumps({"pass": report["pass"], "epoch1_updates": epoch_summary["updates"],
                      "joint_updates": joint_summary["updates"], "reconstructed": reconstructed_hash}, indent=2))
    return 0


def scientific(experiment: Path, resume: bool) -> int:
    if not (experiment / "QUALIFICATION_COMPLETE").is_file(): raise RuntimeError("Phase B incomplete")
    require_registered_source(experiment)
    config, runtime, calibration, dataset, device = prepare(experiment)
    seed = int(config["scientific_seed"]); priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json")
    registration = load_json(experiment / "SCIENTIFIC_REGISTRATION.json")
    seed_everything(seed); model, _ = build_model(priors, device); optimizer = build_optimizer(model, config)
    checkpoints = experiment / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    start_epoch, global_update = 1, 0
    if resume:
        candidates = sorted(checkpoints.glob("epoch_*.pt"))
        if not candidates: raise RuntimeError("resume requested without checkpoint")
        state = torch.load(candidates[-1], map_location="cpu", weights_only=False)
        current_source = package_hashes()
        source_compatible = state["source_hashes"] == current_source
        if not source_compatible:
            amendment_path = experiment / "SOURCE_AMENDMENT_002.json"
            if amendment_path.is_file():
                amendment = load_json(amendment_path)
                source_compatible = (amendment.get("base_source_files") == state["source_hashes"]
                                     and amendment.get("amended_source_files") == current_source
                                     and amendment.get("scope") == "training_runtime_guard_only"
                                     and amendment.get("scientific_settings_changed") is False)
        if (state["config_sha256"] != sha256(CONFIG_PATH) or not source_compatible
                or state["registration_hashes"] != load_json(experiment / "REGISTRATION_HASHES.json")):
            raise RuntimeError("resume checkpoint source/config/registration provenance drift")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"]); start_epoch = int(state["epoch"]) + 1; global_update = int(state["global_optimizer_update"])
    else:
        initial_hash = named_tensor_hash(model.state_dict().items())
        if initial_hash != registration["initial_model"]["state_sha256"] or optimizer.state:
            raise RuntimeError("scientific launch state/optimizer drift")
        configure_trainability(model, 1)
        lrs = scheduled_lrs(config, 1, 1); set_lrs(optimizer, lrs)
        atomic_torch(checkpoints / "epoch_000.pt", checkpoint_payload(
            model, optimizer, config, experiment, 0, 0, {"epoch": 0, "start_index": 0}, lrs))
        atomic_json(experiment / "SCIENTIFIC_TRAINING_STARTED.json", {
            "created_utc": utc_now(), "initial_state_sha256": initial_hash, "optimizer_state_empty": True,
            "physical_batch": runtime["physical_batch"], "accumulation": runtime["gradient_accumulation"],
            "effective_batch": 16, "epochs": 26, "validation_accessed": False}, overwrite=False)
    registered_batches = registration["calibration_batches"]
    physical = int(runtime["physical_batch"]); diagnostic_sets = [registered_batches[0]["indices"],
                                                                   registered_batches[-1]["indices"]]
    training_metrics = experiment / "training_metrics"; telemetry_root = experiment / "gradient_telemetry"
    training_metrics.mkdir(exist_ok=True); telemetry_root.mkdir(exist_ok=True)
    for epoch in range(start_epoch, 27):
        configure_trainability(model, epoch); dataset.set_epoch(epoch)
        loader, sampler = dataloader(dataset, seed, epoch, physical, config["training"]["workers"])
        global_update, summary = run_updates(model, optimizer, loader, config, calibration["multipliers"],
                                             epoch, global_update, int(runtime["gradient_accumulation"]))
        telemetry = diagnostic_telemetry(model, dataset, diagnostic_sets, calibration["multipliers"], physical)
        atomic_json(training_metrics / f"epoch_{epoch:03d}.json", summary, overwrite=False)
        atomic_json(telemetry_root / f"epoch_{epoch:03d}.json", telemetry, overwrite=False)
        payload = checkpoint_payload(model, optimizer, config, experiment, epoch, global_update,
                                     sampler.state_dict(), summary["last_lrs"])
        checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"; atomic_torch(checkpoint, payload)
        atomic_json(checkpoints / f"epoch_{epoch:03d}.json", {
            "epoch": epoch, "path": str(checkpoint), "sha256": sha256(checkpoint),
            "global_optimizer_update": global_update, "created_utc": utc_now()}, overwrite=False)
        atomic_json(experiment / "STATUS.json", {"phase": "C", "state": "training", "epoch_complete": epoch,
                                                  "global_optimizer_update": global_update, "created_utc": utc_now(),
                                                  "validation_accessed": False})
        atomic_json(experiment / f"NOTIFICATION_EPOCH_{epoch:03d}.json", desktop_notify(
            "SplitFusion FCOS", f"Scientific epoch {epoch}/26 complete; checkpoint durable."), overwrite=False)
        print(json.dumps({"epoch_complete": epoch, "updates": summary["updates"],
                          "checkpoint_sha256": sha256(checkpoint), "wall_seconds": summary["wall_seconds"]}), flush=True)
    atomic_text(experiment / "TRAINING_COMPLETE", "EXACTLY_26_SCIENTIFIC_EPOCHS_COMPLETE\n", overwrite=False)
    atomic_json(experiment / "TRAINING_COMPLETE.json", {"created_utc": utc_now(), "epochs": 26,
                                                         "global_optimizer_update": global_update,
                                                         "validation_accessed_during_training": False}, overwrite=False)
    atomic_json(experiment / "STATUS.json", {"phase": "C", "state": "complete", "epoch_complete": 26,
                                              "global_optimizer_update": global_update, "created_utc": utc_now(),
                                              "validation_accessed": False})
    atomic_json(experiment / "NOTIFICATION_TRAINING_COMPLETE.json", desktop_notify(
        "SplitFusion FCOS", "All 26 scientific epochs complete; fixed validation may begin."), overwrite=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("qualification", "scientific")); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    return qualification(experiment) if args.mode == "qualification" else scientific(experiment, args.resume)


if __name__ == "__main__": raise SystemExit(main())
