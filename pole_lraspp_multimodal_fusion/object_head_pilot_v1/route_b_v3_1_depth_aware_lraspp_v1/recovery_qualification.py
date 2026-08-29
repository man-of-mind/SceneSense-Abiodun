from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from common import (CONFIG_PATH, load_json, read_csv, seed_everything, sha256, tensor_state_hash,
                    utc_now, write_json_x, write_torch_atomic_create)
from data import (DepthCache, TrainingDataset, collate_training, load_objects,
                  load_visible_anchors)
from decode import camera_matrix_from_row, decode_geometry, intrinsic_from_row
from losses import compute_losses
from model import (build_model, configure_stage, freeze_bn_running_state, pretrained_backbone_state,
                   stage_train_mode)
from numerical_audit import batch_hashes, canonical_hash, named_tensor_hash
from train import (SCIENTIFIC_COMPONENTS, build_optimizer, scheduled_lrs, set_optimizer_lrs,
                   task_groups)


DETECTION_NAMES = (
    "heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray",
    "dimensions", "yaw", "parked", "radar_support",
)
ACTOR_DEPTH_NAMES = ("depth_bin", "depth_residual", "endpoint")


def fixed_rows(rows: Sequence[dict[str, str]],
               objects: Mapping[str, Sequence[dict[str, str]]]) -> list[dict[str, str]]:
    result = list(rows[:min(4, len(rows))])
    for row in rows:
        if any(item["label"] == "person" for item in objects.get(row["sample_id"], ())):
            result[-1] = row
            break
    return result


def prepare(cache_path: Path, stage: str = "A") -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    seed = int(config["scientific_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification unavailable")
    device = torch.device("cuda")
    root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(root / "dataset/manifest.csv") if row["split"] == "train"]
    objects = load_objects(root)
    visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    cache = DepthCache(cache_path.resolve(strict=True), rows)
    dataset = TrainingDataset(root, rows, objects, visible, cache, seed)
    selected = fixed_rows(rows, objects)
    fixed_dataset = TrainingDataset(root, selected, objects, visible, cache, seed)
    fixed_dataset.set_epoch(1)
    fixed_batch = next(iter(DataLoader(
        fixed_dataset, batch_size=len(selected), num_workers=0, collate_fn=collate_training,
    )))
    model, loading = build_model(Path(config["pretrained"]["path"]), device)
    optimizer = build_optimizer(model)
    configure_stage(model, stage)
    stage_train_mode(model, stage)
    return {
        "config": config, "seed": seed, "device": device, "root": root, "rows": rows,
        "objects": objects, "visible": visible, "cache": cache, "dataset": dataset,
        "fixed_rows": selected, "fixed_batch": fixed_batch, "model": model,
        "optimizer": optimizer, "loading": loading,
    }


def scalar_losses(parts: Mapping[str, torch.Tensor], weights: Mapping[str, float],
                  total: torch.Tensor) -> dict[str, Any]:
    unweighted = {name: float(parts[name].detach().item()) for name in SCIENTIFIC_COMPONENTS}
    weighted = {name: float(weights[name]) * unweighted[name] for name in SCIENTIFIC_COMPONENTS}
    result = {"unweighted": unweighted, "weighted": weighted, "total": float(total.detach().item())}
    result["all_finite"] = all(math.isfinite(value) for values in (unweighted, weighted)
                                  for value in values.values()) and math.isfinite(result["total"])
    return result


def tensor_collection_finite(values: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    count = 0
    elements = 0
    nonfinite = []
    nonzero = []
    absolute_maximum = 0.0
    for name, value in values:
        tensor = value.detach()
        count += 1
        elements += tensor.numel()
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all().item()):
            if len(nonfinite) < 20:
                nonfinite.append(name)
            finite_values = tensor[torch.isfinite(tensor)]
        else:
            finite_values = tensor
        if tensor.numel() and int(torch.count_nonzero(tensor)):
            if len(nonzero) < 200:
                nonzero.append(name)
        if finite_values.numel():
            absolute_maximum = max(absolute_maximum, float(finite_values.abs().max().item()))
    return {"tensors": count, "elements": elements, "finite": not nonfinite,
            "nonfinite_names": nonfinite, "nonzero_names": nonzero,
            "finite_absolute_maximum": absolute_maximum}


def gradients(model: torch.nn.Module, with_hash: bool = True) -> dict[str, Any]:
    values = [(name, parameter.grad) for name, parameter in model.named_parameters()
              if parameter.grad is not None]
    report = tensor_collection_finite(values)
    if with_hash:
        report["sha256"] = named_tensor_hash(values)
    return report


def optimizer_values(optimizer: torch.optim.Optimizer) -> Iterable[tuple[str, torch.Tensor]]:
    for parameter_index, (parameter, state) in enumerate(optimizer.state.items()):
        for name, value in sorted(state.items()):
            if isinstance(value, torch.Tensor):
                yield f"parameter_{parameter_index}.{name}", value


def optimizer_finite(optimizer: torch.optim.Optimizer, with_hash: bool = False) -> dict[str, Any]:
    values = list(optimizer_values(optimizer))
    report = tensor_collection_finite(values)
    if with_hash:
        report["sha256"] = canonical_hash(optimizer.state_dict())
    return report


def model_finite(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "parameters": tensor_collection_finite(model.named_parameters()),
        "buffers": tensor_collection_finite(model.named_buffers()),
    }


def loss_shares(parts: Mapping[str, torch.Tensor], weights: Mapping[str, float],
                total: torch.Tensor) -> dict[str, Any]:
    losses = scalar_losses(parts, weights, total)
    denominator = losses["total"]
    contributions = losses["weighted"]
    shares = {name: value / denominator for name, value in contributions.items()}
    detection = sum(contributions[name] for name in DETECTION_NAMES)
    actor = sum(contributions[name] for name in ACTOR_DEPTH_NAMES)
    groups = {
        "detection": detection, "actor_depth": actor,
        "dense_depth": contributions["dense_depth"] + contributions["radar_consistency"],
        "segmentation": contributions["segmentation"],
    }
    return {**losses, "shares": shares, "maximum_individual_share": max(shares.values()),
            "group_contributions": groups,
            "group_shares": {name: value / denominator for name, value in groups.items()}}


def grad_norm(value: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    values = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=False)
    finite = all(bool(torch.isfinite(item).all().item()) for item in values)
    norm = math.sqrt(sum(float(item.detach().float().pow(2).sum().item()) for item in values))
    return {"tensors": len(values), "finite": finite, "norm": norm, "nonzero": norm > 0.0}


def mode_loss_pressure(args: argparse.Namespace) -> int:
    context = prepare(args.cache, "A")
    model, optimizer = context["model"], context["optimizer"]
    weights = context["config"]["loss_weights"]
    fixed_batch = context["fixed_batch"]
    updates_per_epoch = math.ceil(len(context["rows"]) / 16)
    optimizer.zero_grad(set_to_none=True)
    total1, parts1, denominators1, _outputs = compute_losses(model, fixed_batch, weights)
    initial = loss_shares(parts1, weights, total1)
    total1.backward()
    update1_head_gradients = {}
    for class_name in ("vehicle", "person"):
        branch = getattr(model, class_name)
        update1_head_gradients[class_name] = {
            name: {
                "finite": head.weight.grad is not None and bool(torch.isfinite(head.weight.grad).all().item()),
                "norm": (float(head.weight.grad.float().norm().item()) if head.weight.grad is not None else 0.0),
                "nonzero": head.weight.grad is not None and int(torch.count_nonzero(head.weight.grad)) > 0,
            }
            for name, head in branch.heads.items()
        }
    # Vehicle anchors are defined as their 2-D box centres, so the registered
    # vehicle box-centre-delta target is identically zero.  Its zero-weight,
    # zero-bias prior is already at the exact Smooth-L1 optimum and therefore
    # must have a zero update-1 gradient.  Prove that property from this fixed
    # batch and treat only that field as structurally inapplicable to the
    # nonzero-gradient gate; all other final field heads remain required.
    vehicle_box_center_targets = fixed_batch["owners"]["vehicle"]["box_center_delta"]
    structural_optimum = {
        "field": "vehicle.box_center_delta",
        "registered_reason": "vehicle anchor equals 2-D ground-truth box centre",
        "target_elements": int(vehicle_box_center_targets.numel()),
        "all_targets_exact_zero": int(torch.count_nonzero(vehicle_box_center_targets)) == 0,
        "expected_update_1_weight_gradient_exact_zero": True,
        "observed_update_1_weight_gradient_exact_zero": not update1_head_gradients["vehicle"]["box_center_delta"]["nonzero"],
    }
    preclip1 = gradients(model)
    norm1 = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    postclip1 = gradients(model)
    new_lr1, backbone_lr1 = scheduled_lrs(1, 1, updates_per_epoch, 1)
    set_optimizer_lrs(optimizer, new_lr1, backbone_lr1)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)

    total2, parts2, denominators2, _outputs = compute_losses(model, fixed_batch, weights)
    after_one_update = loss_shares(parts2, weights, total2)
    grouped = task_groups(parts2, weights)
    shared = [parameter for parameter in model.depth_neck.parameters() if parameter.requires_grad]
    group_gradients = {name: grad_norm(value, shared) for name, value in grouped.items()}
    object_value = grouped["detection"] + grouped["actor_depth"]
    trunk_parameters = [
        parameter for class_name in ("vehicle", "person")
        for parameter in getattr(model, class_name).trunk.parameters() if parameter.requires_grad
    ]
    object_trunk_task_gradient = grad_norm(object_value, trunk_parameters)
    object_neck_task_gradient = grad_norm(object_value, shared)
    total2.backward()
    preclip2 = gradients(model)
    trunk_gradient_names = [
        name for name, parameter in model.named_parameters()
        if (name.startswith("vehicle.trunk") or name.startswith("person.trunk"))
        and parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) > 0
    ]
    norm2 = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    postclip2 = gradients(model)
    new_lr2, backbone_lr2 = scheduled_lrs(1, 2, updates_per_epoch, 2)
    set_optimizer_lrs(optimizer, new_lr2, backbone_lr2)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    final_state = {"model": model_finite(model), "optimizer": optimizer_finite(optimizer)}
    required_head_gradients = {
        f"{class_name}.{name}": value
        for class_name, branch in update1_head_gradients.items()
        for name, value in branch.items()
        if f"{class_name}.{name}" != structural_optimum["field"]
    }
    all_required_heads = all(value["finite"] and value["nonzero"]
                             for value in required_head_gradients.values())
    structural_optimum_pass = (
        structural_optimum["target_elements"] > 0
        and structural_optimum["all_targets_exact_zero"]
        and structural_optimum["observed_update_1_weight_gradient_exact_zero"]
        and update1_head_gradients["vehicle"]["box_center_delta"]["finite"]
    )
    critical_groups = all(value["finite"] and value["nonzero"] for value in group_gradients.values())
    pressure_pass = (
        initial["all_finite"] and initial["maximum_individual_share"] <= 0.50
        and initial["group_shares"]["detection"] >= 0.05
        and initial["group_shares"]["actor_depth"] >= 0.05
    )
    passed = (
        pressure_pass and all_required_heads and structural_optimum_pass and critical_groups
        and object_trunk_task_gradient["finite"] and object_trunk_task_gradient["nonzero"]
        and object_neck_task_gradient["finite"] and object_neck_task_gradient["nonzero"]
        and bool(trunk_gradient_names) and preclip1["finite"] and postclip1["finite"]
        and preclip2["finite"] and postclip2["finite"]
        and math.isfinite(float(norm1)) and math.isfinite(float(norm2))
        and final_state["model"]["parameters"]["finite"]
        and final_state["model"]["buffers"]["finite"] and final_state["optimizer"]["finite"]
    )
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_loss_pressure_v1", "created_utc": utc_now(),
        "fixed_sample_ids": list(fixed_batch["sample_id"]), "initial": initial,
        "after_one_update": after_one_update, "denominators_update_1": denominators1,
        "denominators_update_2": denominators2, "update_1_field_head_weight_gradients": update1_head_gradients,
        "update_1_required_field_head_weight_gradients": required_head_gradients,
        "update_1_structural_optimum_exception": structural_optimum,
        "all_required_update_1_field_head_weight_gradients_finite_nonzero": all_required_heads,
        "structural_optimum_gate_pass": structural_optimum_pass,
        "update_2_shared_neck_task_gradients": group_gradients,
        "update_2_object_trunk_task_gradient": object_trunk_task_gradient,
        "update_2_object_neck_task_gradient": object_neck_task_gradient,
        "update_2_nonzero_trunk_gradient_names": trunk_gradient_names,
        "preclip_update_1": preclip1, "postclip_update_1": postclip1,
        "preclip_update_2": preclip2, "postclip_update_2": postclip2,
        "clip_norms": [float(norm1), float(norm2)], "final_state": final_state,
        "pressure_gate": {"maximum_individual_le_50pct": initial["maximum_individual_share"] <= 0.50,
                          "detection_ge_5pct": initial["group_shares"]["detection"] >= 0.05,
                          "actor_depth_ge_5pct": initial["group_shares"]["actor_depth"] >= 0.05},
        "pass": passed, "validation_accessed": False,
    }
    write_json_x(args.output, report)
    print(json.dumps({"pass": passed, "initial_shares": initial["shares"],
                      "group_shares": initial["group_shares"], "output": str(args.output)}, indent=2))
    return 0 if passed else 2


def mode_short(args: argparse.Namespace) -> int:
    started = time.monotonic()
    context = prepare(args.cache, "A")
    model, optimizer = context["model"], context["optimizer"]
    config, dataset = context["config"], context["dataset"]
    seed = context["seed"]
    initial = {
        "model": canonical_hash(model.state_dict()), "parameters": named_tensor_hash(model.named_parameters()),
        "buffers": named_tensor_hash(model.named_buffers()), "optimizer": canonical_hash(optimizer.state_dict()),
    }
    dataset.set_epoch(1)
    sampler = RandomSampler(dataset, replacement=False,
                            generator=torch.Generator().manual_seed(seed + 1))
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=False, drop_last=False, collate_fn=collate_training)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    weights = config["loss_weights"]
    updates_per_epoch = math.ceil(len(context["rows"]) / 16)
    batches, updates = [], []
    torch.cuda.reset_peak_memory_stats(context["device"])
    for update in range(1, 16):
        batch = next(iterator)
        batch_record = batch_hashes(batch); batch_record["batch_index"] = update; batches.append(batch_record)
        total, parts, denominators, _outputs = compute_losses(model, batch, weights)
        losses = scalar_losses(parts, weights, total)
        if not losses["all_finite"]:
            raise FloatingPointError(f"short replay {args.run} non-finite loss update {update}")
        total.backward()
        preclip = gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        postclip = gradients(model)
        new_lr, backbone_lr = scheduled_lrs(1, update, updates_per_epoch, update)
        set_optimizer_lrs(optimizer, new_lr, backbone_lr)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state = model_finite(model); adam = optimizer_finite(optimizer)
        finite = (preclip["finite"] and postclip["finite"] and math.isfinite(float(norm))
                  and state["parameters"]["finite"] and state["buffers"]["finite"] and adam["finite"])
        if not finite:
            raise FloatingPointError(f"short replay {args.run} non-finite state update {update}")
        updates.append({"update": update, "sample_ids": list(batch["sample_id"]), "losses": losses,
                        "denominators": denominators, "preclip": preclip, "postclip": postclip,
                        "clip_norm": float(norm), "new_lr": new_lr, "backbone_lr": backbone_lr,
                        "model_state": state, "optimizer_state": adam, "finite": finite})
    state_payload = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "run": args.run, "updates": 15,
    }
    write_torch_atomic_create(args.state, state_payload)
    state_hash = sha256(args.state)
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_repaired_short_replay_v1",
        "created_utc": utc_now(), "run": args.run, "initial": initial,
        "batches_1_through_15": batches, "updates_1_through_15": updates,
        "categorical_verdict": "REPAIRED_EXECUTION_FINITE", "all_finite": True,
        "state_path": str(args.state), "state_sha256": state_hash,
        "end_model_sha256": canonical_hash(model.state_dict()),
        "end_optimizer_sha256": canonical_hash(optimizer.state_dict()),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(context["device"]) / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(context["device"]) / 2**20,
        "wall_seconds": time.monotonic() - started, "validation_accessed": False,
        "scientific_candidate": False,
    }
    write_json_x(args.output, report)
    print(json.dumps({"run": args.run, "verdict": report["categorical_verdict"],
                      "batch14_total": updates[13]["losses"]["total"],
                      "state_sha256": state_hash, "wall_seconds": report["wall_seconds"]}, indent=2))
    return 0


def drift(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    differences = []
    left_norm_sq = 0.0
    difference_norm_sq = 0.0
    for name in sorted(left):
        a, b = left[name].detach().double().reshape(-1), right[name].detach().double().reshape(-1)
        if a.shape != b.shape:
            raise RuntimeError(f"drift shape mismatch {name}")
        delta = (a - b).abs().cpu()
        differences.append(delta)
        left_norm_sq += float(a.pow(2).sum().item())
        difference_norm_sq += float((a - b).pow(2).sum().item())
    values = torch.cat(differences) if differences else torch.empty(0, dtype=torch.float64)
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float64))
    return {"elements": len(values), "maximum_absolute_difference": float(values.max()) if len(values) else 0.0,
            "relative_l2_difference": math.sqrt(difference_norm_sq) / max(1e-30, math.sqrt(left_norm_sq)),
            "absolute_difference_percentiles": {name: float(value) for name, value in zip(
                ("p50", "p90", "p95", "p99", "p100"), quantiles.tolist())}}


def optimizer_tensor_map(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    result = {}
    for parameter, values in state["state"].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                result[f"{parameter}.{name}"] = value
    return result


def mode_compare_short(args: argparse.Namespace) -> int:
    left, right = load_json(args.left), load_json(args.right)
    left_state = torch.load(args.left_state, map_location="cpu", weights_only=False)
    right_state = torch.load(args.right_state, map_location="cpu", weights_only=False)
    initial_equal = left["initial"] == right["initial"]
    batch_equal = all(
        a["batch"] == b["batch"] and a["input"] == b["input"] and a["targets"] == b["targets"]
        and a["sample_ids_ordered"] == b["sample_ids_ordered"]
        for a, b in zip(left["batches_1_through_15"], right["batches_1_through_15"])
    )
    loss_comparison = {}
    for name in (*SCIENTIFIC_COMPONENTS, "total"):
        a = (left["updates_1_through_15"][13]["losses"]["total"] if name == "total"
             else left["updates_1_through_15"][13]["losses"]["unweighted"][name])
        b = (right["updates_1_through_15"][13]["losses"]["total"] if name == "total"
             else right["updates_1_through_15"][13]["losses"]["unweighted"][name])
        loss_comparison[name] = {"left": a, "right": b, "absolute_delta": abs(a - b),
                                 "within_rtol_1e_3_atol_1e_5": math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-5)}
    same_verdict = left["categorical_verdict"] == right["categorical_verdict"] == "REPAIRED_EXECUTION_FINITE"
    all_losses_agree = all(value["within_rtol_1e_3_atol_1e_5"] for value in loss_comparison.values())
    model_drift = drift(left_state["model"], right_state["model"])
    optimizer_drift = drift(
        optimizer_tensor_map(left_state["optimizer"]), optimizer_tensor_map(right_state["optimizer"]),
    )
    passed = initial_equal and batch_equal and same_verdict and all_losses_agree and left["all_finite"] and right["all_finite"]
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_repaired_short_replay_comparison_v1",
        "created_utc": utc_now(), "corrected_initial_state_identical": initial_equal,
        "all_15_batches_identical": batch_equal, "same_finite_categorical_verdict": same_verdict,
        "batch_14_scalar_losses": loss_comparison, "all_batch_14_losses_within_tolerance": all_losses_agree,
        "model_drift_diagnostic_only": model_drift, "optimizer_drift_diagnostic_only": optimizer_drift,
        "hash_or_state_drift_is_stopping_gate": False, "pass": passed,
    }
    write_json_x(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def mode_full_epoch(args: argparse.Namespace) -> int:
    started = time.monotonic()
    context = prepare(args.cache, "A")
    model, optimizer = context["model"], context["optimizer"]
    config, dataset, rows = context["config"], context["dataset"], context["rows"]
    initial_pretrained = tensor_state_hash(pretrained_backbone_state(model))
    dataset.set_epoch(1)
    sampler = RandomSampler(dataset, replacement=False,
                            generator=torch.Generator().manual_seed(context["seed"] + 1))
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=False, drop_last=False, collate_fn=collate_training)
    weights = config["loss_weights"]
    updates_per_epoch = math.ceil(len(rows) / 16)
    optimizer.zero_grad(set_to_none=True)
    visited = []
    records = []
    module_nonzero = set()
    maximum_share = {name: 0.0 for name in SCIENTIFIC_COMPONENTS}
    torch.cuda.reset_peak_memory_stats(context["device"])
    epoch_started = time.monotonic()
    for update, batch in enumerate(loader, 1):
        total, parts, denominators, _outputs = compute_losses(model, batch, weights)
        losses = scalar_losses(parts, weights, total)
        if not losses["all_finite"]:
            raise FloatingPointError(f"disposable epoch non-finite loss update {update}")
        for name, value in losses["weighted"].items():
            maximum_share[name] = max(maximum_share[name], value / losses["total"])
        total.backward()
        preclip = gradients(model, with_hash=False)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and int(torch.count_nonzero(parameter.grad)):
                module_nonzero.add(name)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        postclip = gradients(model, with_hash=False)
        new_lr, backbone_lr = scheduled_lrs(1, update, updates_per_epoch, update)
        set_optimizer_lrs(optimizer, new_lr, backbone_lr)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state = model_finite(model); adam = optimizer_finite(optimizer)
        finite = (preclip["finite"] and postclip["finite"] and math.isfinite(float(norm))
                  and state["parameters"]["finite"] and state["buffers"]["finite"] and adam["finite"])
        if not finite:
            raise FloatingPointError(f"disposable epoch non-finite state update {update}")
        visited.extend(batch["sample_id"])
        records.append({"update": update, "total": losses["total"],
                        "unweighted": losses["unweighted"], "weighted": losses["weighted"],
                        "denominators": denominators, "clip_norm": float(norm),
                        "new_lr": new_lr, "backbone_lr": backbone_lr, "finite": finite})
        if update % 100 == 0:
            print(json.dumps({"disposable_epoch_update": update, "total": losses["total"]}), flush=True)
    torch.cuda.synchronize(context["device"])
    epoch_seconds = time.monotonic() - epoch_started
    stage_train_mode(model, "A")
    with torch.no_grad():
        end_total, end_parts, end_denominators, _ = compute_losses(model, context["fixed_batch"], weights)
    end_shares = loss_shares(end_parts, weights, end_total)
    model.eval(); freeze_bn_running_state(model)
    fixed_value = context["fixed_batch"]["input"][0:1].to(context["device"])
    with torch.inference_mode():
        deployable = model(fixed_value, dense=False)
    raw_finite = bool(torch.isfinite(deployable["out"]).all().item()) and all(
        bool(torch.isfinite(value).all().item())
        for branch in deployable["objects"].values() for value in branch.values()
    )
    row = context["fixed_rows"][0]
    decoded = decode_geometry(deployable, model.depth_anchors, model.depth_delta,
                              camera_matrix_from_row(row), intrinsic_from_row(row), 0.02, 120)
    decoded_finite = all(
        math.isfinite(float(value)) for record in decoded for value in record.values()
        if isinstance(value, (int, float))
    )
    required_prefixes = (
        "backbone.0.radar_conv", "depth_neck", "segmentation", "dense_depth",
        "vehicle.trunk", "person.trunk", "vehicle.heads", "person.heads",
    )
    required_gradients = {
        prefix: any(name.startswith(prefix) for name in module_nonzero) for prefix in required_prefixes
    }
    peak_reserved = torch.cuda.max_memory_reserved(context["device"]) / 2**20
    passed = (
        len(visited) == 16827 and len(set(visited)) == 16827 and len(records) == updates_per_epoch
        and all(record["finite"] for record in records) and all(required_gradients.values())
        and raw_finite and decoded_finite and peak_reserved <= 11059.2
        and tensor_state_hash(pretrained_backbone_state(model)) == initial_pretrained
        and end_shares["all_finite"]
    )
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_disposable_full_epoch_v1",
        "created_utc": utc_now(), "pass": passed, "epoch": 1,
        "frames_visited": len(visited), "unique_frames": len(set(visited)),
        "updates": len(records), "expected_updates": updates_per_epoch,
        "all_updates": records, "maximum_weighted_loss_shares": maximum_share,
        "end_fixed_batch_loss_pressure": end_shares, "end_fixed_batch_denominators": end_denominators,
        "required_nonzero_gradient_modules": required_gradients,
        "stage_a_pretrained_state_bit_identical": tensor_state_hash(pretrained_backbone_state(model)) == initial_pretrained,
        "deployable_fixed_train_raw_finite": raw_finite, "deployable_fixed_train_records": len(decoded),
        "deployable_fixed_train_decoded_finite": decoded_finite,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(context["device"]) / 2**20,
        "cuda_peak_reserved_mib": peak_reserved, "registered_reserved_limit_mib": 11059.2,
        "epoch_seconds": epoch_seconds, "frames_per_second": len(visited) / epoch_seconds,
        "wall_seconds": time.monotonic() - started, "validation_accessed": False,
        "scientific_candidate": False, "state_discarded": True,
    }
    write_json_x(args.output, report)
    print(json.dumps({"pass": passed, "epoch_seconds": epoch_seconds,
                      "frames_per_second": report["frames_per_second"],
                      "peak_reserved_mib": peak_reserved, "end_shares": end_shares["shares"]}, indent=2))
    return 0 if passed else 2


def benchmark_stage(context: Mapping[str, Any], stage: str, measured: int, warmup: int) -> dict[str, Any]:
    model, optimizer, dataset = context["model"], context["optimizer"], context["dataset"]
    configure_stage(model, stage); stage_train_mode(model, stage)
    dataset.set_epoch(1 if stage == "A" else 6)
    sampler = RandomSampler(dataset, replacement=False,
                            generator=torch.Generator().manual_seed(context["seed"] + (1 if stage == "A" else 6)))
    loader = iter(DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=8, pin_memory=True,
                             persistent_workers=False, drop_last=False, collate_fn=collate_training))
    weights = context["config"]["loss_weights"]
    times = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(context["device"])
    for index in range(warmup + measured):
        started = time.monotonic()
        batch = next(loader)
        total, _parts, _denominators, _outputs = compute_losses(model, batch, weights)
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f"throughput stage {stage} non-finite loss")
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        set_optimizer_lrs(optimizer, 3e-4 if stage == "A" else 1e-4, 0.0 if stage == "A" else 1e-5)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(context["device"])
        if index >= warmup:
            times.append(time.monotonic() - started)
    return {"stage": stage, "warmup_batches": warmup, "measured_batches": measured,
            "mean_seconds_per_batch": float(np.mean(times)), "median_seconds_per_batch": float(np.median(times)),
            "p95_seconds_per_batch": float(np.percentile(times, 95)),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(context["device"]) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(context["device"]) / 2**20,
            "_model": model, "_optimizer": optimizer, "_batch": batch}


def mode_throughput(args: argparse.Namespace) -> int:
    started = time.monotonic()
    stage_a_context = prepare(args.cache, "A")
    stage_a = benchmark_stage(stage_a_context, "A", args.measured_batches, args.warmup_batches)
    del stage_a["_model"], stage_a["_optimizer"], stage_a["_batch"], stage_a_context
    torch.cuda.empty_cache()
    stage_b_context = prepare(args.cache, "B")
    stage_b = benchmark_stage(stage_b_context, "B", args.measured_batches, args.warmup_batches)
    model, optimizer, batch = stage_b.pop("_model"), stage_b.pop("_optimizer"), stage_b.pop("_batch")
    checkpoint_payload = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "epoch": 1,
    }
    temporary = tempfile.NamedTemporaryFile(prefix="depth_aware_checkpoint_", suffix=".pt", delete=False)
    temporary_path = Path(temporary.name); temporary.close()
    checkpoint_started = time.monotonic(); torch.save(checkpoint_payload, temporary_path)
    checkpoint_seconds = time.monotonic() - checkpoint_started
    checkpoint_bytes = temporary_path.stat().st_size; temporary_path.unlink()
    model.eval(); freeze_bn_running_state(model)
    value = batch["input"][0:1].to(stage_b_context["device"])
    with torch.inference_mode():
        for _ in range(50):
            model(value, dense=False)
        torch.cuda.synchronize(stage_b_context["device"])
        inference_times = []
        for _ in range(200):
            begin = time.monotonic(); model(value, dense=False); torch.cuda.synchronize(stage_b_context["device"])
            inference_times.append(time.monotonic() - begin)
    batches_per_epoch = math.ceil(16827 / 16)
    training_seconds = batches_per_epoch * (
        5 * stage_a["mean_seconds_per_batch"] + 35 * stage_b["mean_seconds_per_batch"]
    )
    checkpoint_total = 41 * checkpoint_seconds
    # Conservatively reserves two hours for four persisted inference traversals,
    # scoring, and auxiliary dense-depth diagnostics. GPU-only train-sample latency
    # is recorded but is not used to reduce this allowance.
    evaluation_allowance = 2 * 3600.0
    projected = training_seconds + checkpoint_total + evaluation_allowance
    ceiling = 16 * 3600.0
    required_maximum = ceiling * 0.85
    passed = projected <= required_maximum and max(stage_a["peak_reserved_mib"], stage_b["peak_reserved_mib"]) <= 11059.2
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_throughput_projection_v1",
        "created_utc": utc_now(), "stage_a": stage_a, "stage_b": stage_b,
        "batches_per_epoch": batches_per_epoch, "projected_training_seconds": training_seconds,
        "measured_checkpoint_seconds": checkpoint_seconds, "measured_checkpoint_bytes": checkpoint_bytes,
        "projected_41_checkpoint_seconds": checkpoint_total,
        "train_sample_deployable_inference_median_ms": float(np.median(inference_times) * 1000.0),
        "train_sample_deployable_inference_p95_ms": float(np.percentile(inference_times, 95) * 1000.0),
        "evaluation_and_diagnostics_allowance_seconds": evaluation_allowance,
        "projected_total_seconds": projected, "projected_total_hours": projected / 3600.0,
        "ceiling_seconds": ceiling, "required_15pct_margin_maximum_seconds": required_maximum,
        "projected_margin_fraction": (ceiling - projected) / ceiling,
        "pass": passed, "validation_accessed": False, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=("loss-pressure", "short", "compare-short", "full-epoch", "throughput"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", type=int)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--left", type=Path); parser.add_argument("--right", type=Path)
    parser.add_argument("--left-state", type=Path); parser.add_argument("--right-state", type=Path)
    parser.add_argument("--measured-batches", type=int, default=80)
    parser.add_argument("--warmup-batches", type=int, default=10)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.cache is not None:
        args.cache = args.cache.resolve(strict=True)
    if args.mode == "loss-pressure": return mode_loss_pressure(args)
    if args.mode == "short":
        if args.run not in (1, 2) or args.state is None: raise ValueError("short requires --run and --state")
        args.state = args.state.resolve()
        return mode_short(args)
    if args.mode == "compare-short":
        return mode_compare_short(args)
    if args.mode == "full-epoch": return mode_full_epoch(args)
    if args.mode == "throughput": return mode_throughput(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
