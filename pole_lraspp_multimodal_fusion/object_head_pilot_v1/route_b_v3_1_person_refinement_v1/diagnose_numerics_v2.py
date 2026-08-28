#!/usr/bin/env python3
"""Replay epoch-1 batches 133--135 and compare person-tail numeric policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from person_losses_v1 import person_refinement_loss  # noqa: E402
from person_model_v1 import (  # noqa: E402
    build_model, configure_stage, inherited_person_parameters, load_recovered_base,
    new_parameters, split_boundary_report,
)
from person_targets_v1 import PersonRefinementDataset  # noqa: E402
from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from train_v1 import learning_rates  # noqa: E402

POLICIES = ("existing_fp16", "person_tail_bf16", "person_tail_fp32")
DIAGNOSTIC_BATCHES = (133, 134, 135)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    floating = detached.is_floating_point() or detached.is_complex()
    if floating:
        nan = int(torch.isnan(detached).sum().item())
        pos_inf = int(torch.isposinf(detached).sum().item())
        neg_inf = int(torch.isneginf(detached).sum().item())
        finite_mask = torch.isfinite(detached)
    else:
        nan = pos_inf = neg_inf = 0
        finite_mask = torch.ones_like(detached, dtype=torch.bool)
    finite = detached[finite_mask]
    if finite.numel():
        finite_float = finite.float()
        min_value = float(finite_float.min().item())
        max_value = float(finite_float.max().item())
        minimum_absolute = float(finite_float.abs().min().item())
        maximum_absolute = float(finite_float.abs().max().item())
        absolute_sum = float(finite_float.abs().sum().item())
    else:
        min_value = max_value = minimum_absolute = maximum_absolute = absolute_sum = None
    return {
        "dtype": str(detached.dtype), "shape": list(detached.shape),
        "minimum_finite": min_value, "maximum_finite": max_value,
        "minimum_absolute_activation": minimum_absolute,
        "maximum_absolute_activation": maximum_absolute,
        "nan": nan, "positive_infinity": pos_inf, "negative_infinity": neg_inf,
        "nonfinite": nan + pos_inf + neg_inf,
        "finite": nan + pos_inf + neg_inf == 0,
        "finite_absolute_sum": absolute_sum,
    }


def named_tensor_stats(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: tensor_stats(value) for name, value in sorted(values.items())}


def scalar_or_none(value: torch.Tensor) -> float | None:
    result = float(value.detach().float().item())
    return result if math.isfinite(result) else None


def nonfinite_scalar_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(nonfinite_scalar_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(nonfinite_scalar_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(prefix)
    return paths


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class OperationRecorder:
    def __init__(self, policy: str, batch: int) -> None:
        self.policy = policy
        self.batch = batch
        self.events: list[dict[str, Any]] = []
        self.digests: dict[str, str] = {}
        self.handles: list[Any] = []

    def record(self, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self.events.append({
                "order": len(self.events) + 1, "operation": name,
                **tensor_stats(value),
            })
            if name in {
                "transport.low", "transport.high", "native_feature",
                "vehicle_heatmap", "shared_regression", "grid_offset",
            }:
                self.digests[name] = tensor_digest(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.record(f"{name}.{key}", item)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.record(f"{name}.{index}", item)

    def hook(self, name: str):
        def observe(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.record(name, output)
        return observe

    def install(self, model: torch.nn.Module) -> None:
        modules = [
            ("transport", model.backbone),
            ("native_feature", model.object_head.upsampler),
            ("person_trunk.conv1", model.person_refinement.trunk[0]),
            ("person_trunk.groupnorm1", model.person_refinement.trunk[1]),
            ("person_trunk.silu1", model.person_refinement.trunk[2]),
            ("person_trunk.conv2", model.person_refinement.trunk[3]),
            ("person_trunk.groupnorm2", model.person_refinement.trunk[4]),
            ("person_trunk.silu2", model.person_refinement.trunk[5]),
            ("objectness_residual", model.person_refinement.objectness_residual),
            ("quality_head", model.person_refinement.localization_quality),
            ("range_bin_logits", model.person_refinement.range_bin_logits),
            ("range_residual", model.person_refinement.range_residual),
            ("projected_offset", model.person_refinement.projected_center_offset),
            ("mask_residual", model.person_refinement.person_mask_residual),
            ("inherited_person_heatmap", model.object_head.person_heatmap_head),
            ("vehicle_heatmap", model.object_head.vehicle_heatmap_head),
            ("shared_regression", model.object_head.regression_head),
            ("grid_offset", model.object_head.offset_head),
        ]
        self.handles = [module.register_forward_hook(self.hook(name)) for name, module in modules]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def first_nonfinite(self) -> dict[str, Any] | None:
        return next((event for event in self.events if event["nonfinite"] > 0), None)


def make_model(base: dict[str, Any], base_path: Path, design: dict[str, Any],
               device: torch.device) -> torch.nn.Module:
    model = build_model(
        radar_channels=int(base["radar_channels"]),
        hidden_channels=int(base["object_hidden_channels"]),
        head_depth=int(base["object_head_depth"]),
        person_hidden=int(design["hidden_channels"]),
        group_norm_groups=int(design["group_norm_groups"]),
        range_bins=int(design["range_bins"]), device=device,
    )
    load_recovered_base(model, base_path, device=device)
    configure_stage(model, "P1")
    return model


def make_optimizer(model: torch.nn.Module, design: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW([
        {"params": new_parameters(model), "lr": 0.0, "name": "new_person_tail"},
        {"params": inherited_person_parameters(model), "lr": 0.0,
         "name": "inherited_person_heatmap"},
    ], lr=0.0, weight_decay=float(design["weight_decay"]))


def policy_outputs(model: torch.nn.Module, tensors: torch.Tensor,
                   policy: str) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(policy)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", enabled=True, dtype=torch.float16, cache_enabled=False,
    ):
        features = model.backbone(tensors)
        native_feature = model._native_feature(features)
        vehicle_heatmap, _unused_person_heatmap = model._finite_class_heatmaps(native_feature)
        regression = model.object_head.regression_head(native_feature)
        grid_offset = model.object_head.offset_head(native_feature)
        segmentation = model.classifier(features)
        if isinstance(segmentation, dict):
            segmentation = segmentation["out"]
    with torch.autocast(device_type="cuda", enabled=False):
        person_heatmap = model.object_head.person_heatmap_head(native_feature.detach().float())
    tail_bf16 = policy == "person_tail_bf16"
    tail_fp16 = policy == "existing_fp16"
    tail_feature = native_feature.detach() if tail_fp16 else native_feature.detach().float()
    with torch.autocast(
        device_type="cuda", enabled=tail_bf16 or tail_fp16,
        dtype=torch.float16 if tail_fp16 else torch.bfloat16,
        cache_enabled=False,
    ):
        refinement = model.person_refinement(tail_feature)
        mask_residual = refinement["person_mask_residual"]
        if tuple(mask_residual.shape[-2:]) != tuple(segmentation.shape[-2:]):
            mask_residual = F.interpolate(
                mask_residual, size=segmentation.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        refined_segmentation = torch.cat([
            segmentation[:, :2].detach(), segmentation[:, 2:3].detach() + mask_residual,
        ], dim=1)
    base_object = torch.cat([
        vehicle_heatmap.detach(), person_heatmap, regression.detach(), grid_offset.detach(),
    ], dim=1)
    return {
        "out": refined_segmentation, "base_out": segmentation.detach(),
        "object": base_object, "person_refinement": refinement,
    }


def output_max_abs_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    deltas = {
        "out": float((left["out"].float() - right["out"].float()).abs().max().item()),
        "base_out": float((left["base_out"].float() - right["base_out"].float()).abs().max().item()),
        "object": float((left["object"].float() - right["object"].float()).abs().max().item()),
    }
    for name in left["person_refinement"]:
        deltas[f"person_refinement.{name}"] = float((
            left["person_refinement"][name].float()
            - right["person_refinement"][name].float()
        ).abs().max().item())
    return deltas


def trainable_gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters[name] = None if parameter.grad is None else tensor_stats(parameter.grad)
    module_prefixes = {
        "person_trunk": "person_refinement.trunk.",
        "objectness_residual": "person_refinement.objectness_residual.",
        "quality_head": "person_refinement.localization_quality.",
        "range_bin_logits": "person_refinement.range_bin_logits.",
        "range_residual": "person_refinement.range_residual.",
        "projected_offset": "person_refinement.projected_center_offset.",
        "mask_residual": "person_refinement.person_mask_residual.",
        "inherited_person_heatmap": "object_head.person_heatmap_head.",
    }
    modules: dict[str, Any] = {}
    named = dict(model.named_parameters())
    for module_name, prefix in module_prefixes.items():
        gradients = [parameter.grad for name, parameter in named.items()
                     if name.startswith(prefix) and parameter.requires_grad]
        present = [gradient for gradient in gradients if gradient is not None]
        finite_absolute_sum = float(sum(
            value.detach().float()[torch.isfinite(value)].abs().sum().item()
            for value in present
        ))
        nonfinite = int(sum((~torch.isfinite(value)).sum().item() for value in present))
        modules[module_name] = {
            "trainable_parameter_tensors": len(gradients),
            "gradient_tensors": len(present),
            "all_finite": bool(present) and all(torch.isfinite(value).all().item() for value in present),
            "finite_absolute_sum": finite_absolute_sum,
            "nonfinite": nonfinite,
        }
    frozen_with_gradient = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    return {
        "parameters": parameters, "modules": modules,
        "all_trainable_gradients_present": all(value is not None for value in parameters.values()),
        "all_trainable_gradients_finite": all(
            value is not None and value["finite"] for value in parameters.values()
        ),
        "expected_modules_nonzero": all(
            payload["finite_absolute_sum"] > 0.0 for name, payload in modules.items()
            if name != "inherited_person_heatmap"
        ),
        "frozen_gradients_absent": not frozen_with_gradient,
        "frozen_parameters_with_gradient": frozen_with_gradient,
    }


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    return named_tensor_stats({name: parameter for name, parameter in model.named_parameters()})


def optimizer_report(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    states: dict[str, Any] = {}
    for parameter, state in optimizer.state.items():
        name = names.get(parameter, "unknown")
        states[name] = {
            key: tensor_stats(value) if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
    floating_state_fp32 = all(
        payload["dtype"] == "torch.float32"
        for state in states.values() for payload in state.values()
        if isinstance(payload, dict) and payload.get("dtype", "").startswith("torch.float")
    )
    return {"states": states, "all_floating_state_fp32": floating_state_fp32}


def all_finite_report(report: dict[str, Any]) -> bool:
    return all(value["finite"] for value in report.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA runtime unavailable")
    source_experiment = args.source_experiment.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    base_path = args.base_checkpoint.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"create-only diagnostic output exists: {output_path}")
    if sha256(base_path) != args.base_sha256:
        raise RuntimeError("recovered checkpoint SHA mismatch")
    registration_path = source_experiment / "REGISTRATION.json"
    if sha256(registration_path) != args.registration_sha256:
        raise RuntimeError("registered experiment artifact SHA mismatch")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = config["person_design"]
    if sha256(config_path) != registration["resolved_config_sha256"]:
        raise RuntimeError("registered config SHA mismatch")
    if int(design["training_seed"]) != 20260828 or int(design["batch_size"]) != 16:
        raise RuntimeError("registered sampler seed or batch size drift")

    device = torch.device("cuda")
    set_reproducible_seeds(int(design["training_seed"]))
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    rows = read_manifest(source_experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 16827 or len(val_rows) != 3345:
        raise RuntimeError("registered dataset population drift")
    object_rows = load_object_boxes(source_experiment / "dataset/object_boxes.csv")
    dataset = PersonRefinementDataset(
        source_experiment / "dataset", train_rows, object_rows,
        tuple(config["registered_input_size"]), dict(base["config"]["object_heads"]),
        augment_strength=str(design["augment_strength"]), geometric_augment=False,
        range_edges=registration["range_bins"]["edges_m"],
        offset_caps=design["projected_offset_cap_grid_xy"],
    )
    weights = torch.as_tensor(registration["sampler"]["normalized_weights"], dtype=torch.double)
    if len(weights) != len(dataset):
        raise RuntimeError("registered sampler population drift")

    replay_model = make_model(base, base_path, design, device)
    replay_optimizer = make_optimizer(replay_model, design)
    replay_scaler = torch.amp.GradScaler("cuda", enabled=True)
    epoch_generator = torch.Generator().manual_seed(int(design["training_seed"]) + 1)
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
    identity_generator = torch.Generator().manual_seed(int(design["training_seed"]) + 1)
    sampled_indices = torch.multinomial(
        weights, int(design["sampling"]["num_samples_per_epoch"]), True,
        generator=identity_generator,
    ).tolist()
    batch_identity = {
        str(batch): {
            "dataset_indices": sampled_indices[(batch - 1) * 16:batch * 16],
            "sample_ids": [train_rows[index]["sample_id"]
                           for index in sampled_indices[(batch - 1) * 16:batch * 16]],
        } for batch in DIAGNOSTIC_BATCHES
    }

    captured_batches: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
    replay_steps = 0
    for batch_index, (tensors, masks, targets) in enumerate(loader):
        batch_number = batch_index + 1
        if batch_number in DIAGNOSTIC_BATCHES:
            captured_batches[batch_number] = (tensors, masks, targets)
            if batch_number == DIAGNOSTIC_BATCHES[-1]:
                break
            continue
        if batch_number > 132:
            continue
        new_lr, inherited_lr = learning_rates(1, batch_index, len(loader), design)
        replay_optimizer.param_groups[0]["lr"] = new_lr
        replay_optimizer.param_groups[1]["lr"] = inherited_lr
        tensors = tensors.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
        replay_optimizer.zero_grad(set_to_none=True)
        outputs = policy_outputs(replay_model, tensors, "existing_fp16")
        with torch.autocast(device_type="cuda", enabled=False):
            loss, _parts = person_refinement_loss(
                outputs, masks, targets,
                range_edges=registration["range_bins"]["edges_m"],
                offset_caps=design["projected_offset_cap_grid_xy"], design=design,
            )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"unexpected replay failure before target window at batch {batch_number}")
        replay_scaler.scale(loss).backward()
        replay_scaler.step(replay_optimizer)
        replay_scaler.update()
        replay_steps += 1
    if set(captured_batches) != set(DIAGNOSTIC_BATCHES) or replay_steps != 132:
        raise RuntimeError("failed to capture exact batch window after 132 optimizer iterations")

    snapshot_model = copy.deepcopy(replay_model.state_dict())
    snapshot_optimizer = copy.deepcopy(replay_optimizer.state_dict())
    snapshot_scaler = copy.deepcopy(replay_scaler.state_dict())
    del loader, replay_model, replay_optimizer, replay_scaler
    torch.cuda.empty_cache()

    policy_reports: dict[str, Any] = {}
    comparison_digests: dict[str, dict[int, dict[str, str]]] = {}
    for policy in POLICIES:
        model = make_model(base, base_path, design, device)
        model.load_state_dict(snapshot_model, strict=True)
        configure_stage(model, "P1")
        optimizer = make_optimizer(model, design)
        optimizer.load_state_dict(copy.deepcopy(snapshot_optimizer))
        scaler = torch.amp.GradScaler("cuda", enabled=policy != "person_tail_fp32")
        if scaler.is_enabled():
            scaler.load_state_dict(copy.deepcopy(snapshot_scaler))
        batches: dict[str, Any] = {}
        comparison_digests[policy] = {}
        for batch_number in DIAGNOSTIC_BATCHES:
            batch_index = batch_number - 1
            new_lr, inherited_lr = learning_rates(1, batch_index, 1052, design)
            optimizer.param_groups[0]["lr"] = new_lr
            optimizer.param_groups[1]["lr"] = inherited_lr
            tensors_cpu, masks_cpu, targets_cpu = captured_batches[batch_number]
            tensors = tensors_cpu.to(device, non_blocking=True)
            masks = masks_cpu.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True)
                       for key, value in targets_cpu.items()}
            input_checks = {"tensors": tensor_stats(tensors), "masks": tensor_stats(masks)}
            target_checks = named_tensor_stats(targets)
            parameters_before = parameter_report(model)
            optimizer_before = optimizer_report(model, optimizer)
            optimizer.zero_grad(set_to_none=True)
            recorder = OperationRecorder(policy, batch_number)
            recorder.install(model)
            try:
                outputs = policy_outputs(model, tensors, policy)
                actual_policy_deltas: dict[str, float] | None = None
                if policy == "person_tail_fp32":
                    recorder.close()
                    with torch.autocast(
                        device_type="cuda", enabled=True, dtype=torch.float16,
                        cache_enabled=False,
                    ):
                        actual_outputs = model.training_outputs(tensors)
                    actual_policy_deltas = output_max_abs_deltas(outputs, actual_outputs)
                    del actual_outputs
                recorder.record(
                    "mask_residual.interpolated",
                    outputs["out"][:, 2:3].float() - outputs["base_out"][:, 2:3].float(),
                )
                recorder.record("output.object", outputs["object"])
                recorder.record("output.segmentation", outputs["out"])
                recorder.record("output.base_segmentation", outputs["base_out"])
                recorder.record("output.person_refinement", outputs["person_refinement"])
                with torch.autocast(device_type="cuda", enabled=False):
                    loss, parts = person_refinement_loss(
                        outputs, masks, targets,
                        range_edges=registration["range_bins"]["edges_m"],
                        offset_caps=design["projected_offset_cap_grid_xy"], design=design,
                        tensor_observer=recorder.record,
                    )
                loss_finite = bool(torch.isfinite(loss).item())
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                gradients = trainable_gradient_report(model)
                step_applied = False
                if loss_finite:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                        step_applied = True
                    elif gradients["all_trainable_gradients_finite"]:
                        optimizer.step()
                        step_applied = True
                output_checks = {
                    "object": tensor_stats(outputs["object"]),
                    "segmentation": tensor_stats(outputs["out"]),
                    "base_segmentation": tensor_stats(outputs["base_out"]),
                    **{f"person_refinement.{name}": tensor_stats(value)
                       for name, value in outputs["person_refinement"].items()},
                }
                comparison_digests[policy][batch_number] = {
                    **recorder.digests,
                    "vehicle_object_slice": tensor_digest(outputs["object"][:, 0:1]),
                    "shared_object_slice": tensor_digest(outputs["object"][:, 2:]),
                }
                first_nonfinite = recorder.first_nonfinite()
                batches[str(batch_number)] = {
                    "input_checks": input_checks,
                    "target_checks": target_checks,
                    "parameters_before": parameters_before,
                    "optimizer_before": optimizer_before,
                    "forward_and_loss_events": recorder.events,
                    "first_nonfinite_operation": first_nonfinite,
                    "loss_value": scalar_or_none(loss),
                    "loss_finite": loss_finite,
                    "loss_detail_finite": all(math.isfinite(float(value)) for value in parts.values()),
                    "output_checks": output_checks,
                    "gradients": gradients,
                    "step_applied": step_applied,
                    "grad_scaler_enabled": scaler.is_enabled(),
                    "grad_scaler_state": copy.deepcopy(scaler.state_dict()),
                    "actual_full_fp32_implementation_max_abs_deltas": actual_policy_deltas,
                    "actual_full_fp32_implementation_bit_identical": (
                        actual_policy_deltas is None
                        or all(value == 0.0 for value in actual_policy_deltas.values())
                    ),
                    "learning_rates": {"new_person_tail": new_lr,
                                       "inherited_person_heatmap": inherited_lr},
                }
            finally:
                recorder.close()
        policy_reports[policy] = {"batches": batches}
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    invariant_comparison: dict[str, Any] = {}
    reference = comparison_digests["existing_fp16"]
    for policy in POLICIES:
        invariant_comparison[policy] = {}
        for batch_number in DIAGNOSTIC_BATCHES:
            current = comparison_digests[policy][batch_number]
            keys = sorted(set(reference[batch_number]) & set(current))
            invariant_comparison[policy][str(batch_number)] = {
                key: current[key] == reference[batch_number][key] for key in keys
            }

    fp16_first = next(
        (policy_reports["existing_fp16"]["batches"][str(batch)]["first_nonfinite_operation"]
         for batch in DIAGNOSTIC_BATCHES
         if policy_reports["existing_fp16"]["batches"][str(batch)]["first_nonfinite_operation"]),
        None,
    )
    fp16_failure_batch = next(
        (batch for batch in DIAGNOSTIC_BATCHES
         if not policy_reports["existing_fp16"]["batches"][str(batch)]["loss_finite"]),
        None,
    )
    fp32_batches = policy_reports["person_tail_fp32"]["batches"]
    fp32_forward_loss_gradient_pass = all(
        payload["loss_finite"] and payload["loss_detail_finite"]
        and all(value["finite"] for value in payload["output_checks"].values())
        and payload["gradients"]["all_trainable_gradients_finite"]
        and payload["gradients"]["expected_modules_nonzero"]
        and payload["gradients"]["frozen_gradients_absent"]
        and payload["actual_full_fp32_implementation_bit_identical"]
        and all(value["finite"] for value in payload["input_checks"].values())
        and all(value["finite"] for value in payload["target_checks"].values())
        and all(value["finite"] for value in payload["parameters_before"].values())
        and payload["optimizer_before"]["all_floating_state_fp32"]
        for payload in fp32_batches.values()
    )
    invariant_pass = all(
        all(flags.values()) for policy, batches in invariant_comparison.items()
        for flags in batches.values()
    )
    p2_model = make_model(base, base_path, design, device)
    p2_model.load_state_dict(snapshot_model, strict=True)
    configure_stage(p2_model, "P2")
    p2_tensors_cpu, p2_masks_cpu, p2_targets_cpu = captured_batches[134]
    p2_tensors = p2_tensors_cpu.to(device)
    p2_masks = p2_masks_cpu.to(device)
    p2_targets = {key: value.to(device) for key, value in p2_targets_cpu.items()}
    p2_model.zero_grad(set_to_none=True)
    p2_recorder = OperationRecorder("person_tail_fp32_p2", 134)
    p2_recorder.install(p2_model)
    try:
        with torch.autocast(
            device_type="cuda", enabled=True, dtype=torch.float16, cache_enabled=False,
        ):
            p2_outputs = p2_model.training_outputs(p2_tensors)
        with torch.autocast(device_type="cuda", enabled=False):
            p2_loss, _p2_parts = person_refinement_loss(
                p2_outputs, p2_masks, p2_targets,
                range_edges=registration["range_bins"]["edges_m"],
                offset_caps=design["projected_offset_cap_grid_xy"], design=design,
            )
        p2_loss.backward()
        p2_gradients = trainable_gradient_report(p2_model)
    finally:
        p2_recorder.close()
    inherited_events = [
        event for event in p2_recorder.events
        if event["operation"] == "inherited_person_heatmap"
    ]
    p2_proof = {
        "batch": 134,
        "loss": scalar_or_none(p2_loss), "loss_dtype": str(p2_loss.dtype),
        "all_person_refinement_outputs_fp32": all(
            value.dtype == torch.float32 for value in p2_outputs["person_refinement"].values()
        ),
        "inherited_person_heatmap_events": inherited_events,
        "inherited_person_heatmap_fp32": bool(inherited_events)
        and all(event["dtype"] == "torch.float32" for event in inherited_events),
        "gradients": p2_gradients,
        "all_parameters_fp32": all(
            not parameter.is_floating_point() or parameter.dtype == torch.float32
            for parameter in p2_model.parameters()
        ),
    }
    inherited_gradient = p2_gradients["modules"]["inherited_person_heatmap"]
    p2_pass = (
        p2_loss.dtype == torch.float32 and torch.isfinite(p2_loss).item()
        and p2_proof["all_person_refinement_outputs_fp32"]
        and p2_proof["inherited_person_heatmap_fp32"]
        and inherited_gradient["all_finite"]
        and inherited_gradient["finite_absolute_sum"] > 0.0
        and p2_gradients["all_trainable_gradients_finite"]
        and p2_gradients["frozen_gradients_absent"]
        and p2_proof["all_parameters_fp32"]
    )
    p2_proof["all_pass"] = bool(p2_pass)
    del p2_model, p2_outputs, p2_loss
    parity_model = make_model(base, base_path, design, device)
    parity_model.load_state_dict(snapshot_model, strict=True)
    parity_sample = captured_batches[133][0][:1].to(device)
    parity = split_boundary_report(parity_model, parity_sample)
    del parity_model

    result = {
        "schema": "route_b_v3_1_person_refinement_numerical_reproduction_v2",
        "created_utc": utc_now(), "source_experiment": str(source_experiment),
        "source_terminal": (source_experiment / "TERMINAL_VERDICT.txt").read_text().strip(),
        "base_checkpoint": str(base_path), "base_checkpoint_sha256": sha256(base_path),
        "registration": str(registration_path),
        "registration_sha256": sha256(registration_path),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "sampler": {
            "training_seed": int(design["training_seed"]), "epoch_seed": int(design["training_seed"]) + 1,
            "batch_size": int(design["batch_size"]), "replacement": True,
            "num_samples": int(design["sampling"]["num_samples_per_epoch"]),
            "num_workers": int(design["num_workers"]),
            "prefetch_factor": int(design["prefetch_factor"]),
            "persistent_workers": bool(design["persistent_workers"]),
            "batch_identity": batch_identity,
        },
        "replayed_optimizer_iterations_before_window": replay_steps,
        "snapshot": {
            "all_parameters_finite": all(
                torch.isfinite(value).all().item() for value in snapshot_model.values()
            ),
            "grad_scaler_state": snapshot_scaler,
        },
        "policies": policy_reports,
        "invariant_digest_comparison_against_existing_fp16": invariant_comparison,
        "transport_and_vehicle_outputs_exactly_unchanged": invariant_pass,
        "monolithic_split_parity": parity,
        "exact_first_fp16_nonfinite_operation": fp16_first,
        "exact_fp16_nonfinite_loss_batch": fp16_failure_batch,
        "deterministic_batch_134_failure_reproduced": fp16_failure_batch == 134,
        "full_fp32_batches_133_135_pass": fp32_forward_loss_gradient_pass,
        "full_fp32_p2_inherited_person_proof": p2_proof,
        "full_fp32_policy_authorized": (
            fp16_failure_batch == 134 and fp32_forward_loss_gradient_pass
            and p2_pass and invariant_pass and parity["outputs_bit_identical"]
        ),
        "wall_seconds": time.monotonic() - started,
    }
    result["nonfinite_scalar_paths_sanitized"] = nonfinite_scalar_paths(result)
    serialized = json_safe(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(serialized, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "fp16_failure_batch": fp16_failure_batch,
        "fp16_first_nonfinite": fp16_first,
        "fp32_pass": fp32_forward_loss_gradient_pass,
        "policy_authorized": result["full_fp32_policy_authorized"],
        "wall_seconds": result["wall_seconds"],
    }, indent=2, sort_keys=True), flush=True)
    return 0 if result["full_fp32_policy_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
