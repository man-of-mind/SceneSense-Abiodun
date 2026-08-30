from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .audit import ForwardAudit, audit_tree, require_finite_audit
from .base_runtime import load_base
from .contracts import atomic_json, load_json, load_recovery_config, resolve_repo_path, verify_original_provenance
from .guards import proposed_sgd_metrics
from .recovery_losses import compute_loss_groups as recovered_loss
from .recovery_model import build_recovery_model
from .state_guard import model_hash, optimizer_hash


def _norm(values: Sequence[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(value.detach().double().pow(2).sum()) for value in values if value is not None))


def _parameter_gradients(model: torch.nn.Module) -> dict[str, float]:
    selectors = {
        "geometry_tower": lambda name: name.startswith("geometry.tower"),
        "geometry_yaw": lambda name: name.startswith("geometry.outputs.yaw"),
        "p2": lambda name: name.startswith("tail.p2_"),
        "c2_front": lambda name: name.startswith(("front.W_rgb", "front.W_radar", "front.bn1", "front.layer1")),
        "rgb_stem": lambda name: name == "front.W_rgb",
        "radar_stem": lambda name: name == "front.W_radar",
    }
    named = list(model.named_parameters())
    return {group: _norm([value.grad for name, value in named if selector(name)]) for group, selector in selectors.items()}


def _autograd_diagnostics(model: torch.nn.Module, parts: Mapping[str, torch.Tensor],
                          outputs: Mapping[str, Any]) -> dict[str, Any]:
    raw_yaw = tuple(level["yaw"] for level in outputs["geometry"])
    yaw_parameters = tuple((name, parameter) for name, parameter in model.named_parameters()
                           if name.startswith("geometry.outputs.yaw"))
    losses = [name for name in ("D", "G", "S", "A", "geometry_yaw", "geometry_endpoint",
                                "geometry_dimensions", "geometry_physical_ray") if name in parts]
    result: dict[str, Any] = {}
    for name in losses:
        yaw_grad = torch.autograd.grad(parts[name], raw_yaw, retain_graph=True, allow_unused=True)
        parameter_grad = torch.autograd.grad(parts[name], tuple(value for _key, value in yaw_parameters),
                                             retain_graph=True, allow_unused=True)
        c2_grad = torch.autograd.grad(parts[name], outputs["c2"], retain_graph=True, allow_unused=True)
        result[name] = {"raw_yaw_gradient_l2": _norm(yaw_grad), "c2_gradient_l2": _norm(c2_grad),
                        "raw_yaw_gradient_by_level": [_norm([value]) for value in yaw_grad],
                        "yaw_head_parameter_gradient_l2": _norm(parameter_grad),
                        "yaw_head_parameter_gradient_by_name": {
                            key: _norm([gradient]) for (key, _parameter), gradient in zip(yaw_parameters, parameter_grad)}}
    return result


def _training_decode_audit(model: torch.nn.Module, outputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Audit train-time FCOS scores/decoded boxes without selecting detections."""
    records: dict[str, dict[str, Any]] = {}
    per = outputs["detection"]["per_level"]
    counts = [value.shape[1] for value in per["cls_logits"]]
    for image_index, anchors_all in enumerate(outputs["anchors"]):
        anchors = list(anchors_all.split(counts))
        for level_index, (cls, ctr, box) in enumerate(zip(per["cls_logits"], per["bbox_ctrness"],
                                                          per["bbox_regression"])):
            score = torch.sqrt(torch.sigmoid(cls[image_index].detach().float()) *
                               torch.sigmoid(ctr[image_index].detach().float()))
            decoded = model.box_coder.decode(box[image_index].detach().float(), anchors[level_index].detach().float())
            records.update(audit_tree(score, f"train_decode.image{image_index}.level{level_index}.scores"))
            records.update(audit_tree(decoded, f"train_decode.image{image_index}.level{level_index}.boxes"))
    require_finite_audit(records)
    return records


def _original_carrier_identities(base: Any, outputs: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                                 matched_images: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
    num = [value.shape[1] for value in outputs["detection"]["per_level"]["cls_logits"]]
    records = []
    for image_index, target in enumerate(targets):
        offset = 0
        for level_index, count in enumerate(num):
            match = matched_images[image_index][offset:offset + count]; offset += count
            positive = match >= 0
            if not bool(positive.any()):
                continue
            point = torch.where(positive)[0]; actor = match[positive]
            labels = target["labels"].to(point.device)[actor]
            raw = outputs["geometry"][level_index]["yaw"][image_index, point, labels].float()
            scale = raw.abs().amax(dim=-1, keepdim=True); zero = scale == 0
            scaled = raw / torch.where(zero, torch.ones_like(scale), scale)
            norm = (scale * torch.sqrt(scaled.square().sum(dim=-1, keepdim=True) + zero.to(raw.dtype))).squeeze(-1)
            normalized = torch.nn.functional.normalize(raw, dim=1, eps=1e-6)
            for local_index in range(len(point)):
                actor_index = int(actor[local_index]); sources = target.get("source_identity", [])
                records.append({"image_index": image_index, "sample_id": target.get("sample_id"),
                    "actor_index": actor_index, "source_identity": sources[actor_index] if sources else None,
                    "class_index": int(labels[local_index]), "fpn_level": base.model.LEVELS[level_index],
                    "point_index": int(point[local_index]), "raw_yaw": raw[local_index].detach().cpu().tolist(),
                    "raw_yaw_norm": float(norm[local_index]),
                    "normalized_yaw": normalized[local_index].detach().cpu().tolist(),
                    "below_original_eps": bool(norm[local_index] < 1e-6),
                    "normalizer": "original_F.normalize_eps_1e-6"})
    return records


def original_loss_instrumented(model: torch.nn.Module, batch: Mapping[str, Any],
                               multipliers: Mapping[str, float] | None = None, *, use_amp: bool = True,
                               audit_detail: bool = False) -> tuple[Any, ...]:
    """Byte-equation original loss plus raw carrier identity diagnostics."""
    base = load_base(); inputs = batch["input"].to(next(model.parameters()).device, non_blocking=True)
    targets = batch["targets"]; amp = inputs.device.type == "cuda" and use_amp
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = model(inputs, dense=True)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        detection, detection_parts, matched, assignment = base.losses.detection_losses(model, outputs, targets)
        geometry, geometry_parts, geometry_audit = base.losses.geometry_losses(model, outputs, targets, matched)
        semantic, semantic_parts = base.losses.semantic_loss(outputs["semantic_logits"], targets)
        auxiliary, auxiliary_parts, auxiliary_audit = base.losses.dense_losses(outputs, batch)
    geometry_audit = dict(geometry_audit)
    geometry_audit["carrier_identities"] = (
        _original_carrier_identities(base, outputs, targets, matched) if audit_detail else [])
    geometry_audit["detailed_audit_enabled"] = audit_detail
    groups = {"D": detection, "G": geometry, "S": semantic, "A": auxiliary}
    weights = {"D": 1.0, "G": 1.0, "S": 1.0, "A": 1.0}
    if multipliers is not None:
        weights.update({name: float(multipliers[name]) for name in ("G", "S", "A")})
    total = sum(weights[name] * groups[name] for name in ("D", "G", "S", "A")); pressure = total.detach().abs().clamp_min(1e-12)
    components = {**{f"fcos_{name}": value for name, value in detection_parts.items()},
                  **{f"geometry_{name}": value for name, value in geometry_parts.items()},
                  **semantic_parts, **auxiliary_parts, **groups,
                  **{f"weighted_{name}": weights[name] * groups[name] for name in groups},
                  **{f"optimization_fraction_{name}": (weights[name] * groups[name]).detach() / pressure for name in groups},
                  "total": total}
    return total, components, {"assignment": assignment, "geometry": geometry_audit,
                               "auxiliary": auxiliary_audit, "multipliers": weights}, outputs


def _finite_model_optimizer(base: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not base.train.all_model_finite(model):
        raise FloatingPointError("nonfinite model parameter/buffer")
    if not base.train.optimizer_finite(optimizer):
        raise FloatingPointError("nonfinite optimizer state")


def prepare_runtime(tau: float | None, normalization: str) -> tuple[Any, Mapping[str, Any], Mapping[str, Any],
                                                                    torch.nn.Module, torch.optim.Optimizer, Any, Any]:
    provenance = verify_original_provenance(checkpoint_metadata=True)
    immutable = load_recovery_config(); base = load_base()
    experiment = resolve_repo_path(immutable["original"]["experiment"])
    original_config = load_json(resolve_repo_path(immutable["original"]["config"]))
    runtime = load_json(experiment / "QUALIFIED_RUNTIME.json")
    calibration = load_json(experiment / "LOSS_CALIBRATION.json")
    priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("real replay requires CUDA; no CPU fallback is permitted")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12288 * 2**20 / total_memory), device)
    if normalization == "original":
        model, _ = base.model.build_model(priors, device)
        loss_function: Callable[..., Any] = original_loss_instrumented
    elif normalization == "candidate":
        if tau is None:
            raise RuntimeError("candidate replay requires a preregistered tau")
        model, _ = build_recovery_model(priors, tau, device)
        loss_function = recovered_loss
    else:
        raise ValueError(normalization)
    optimizer = base.train.build_optimizer(model, original_config)
    checkpoint = torch.load(Path(provenance["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"])
    base.common.restore_rng(checkpoint["rng"])
    base.model.configure_trainability(model, 10)
    _finite_model_optimizer(base, model, optimizer)
    dataset_root = (base.common.ROOT / original_config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / original_config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(original_config["scientific_seed"]), cache, augment=True)
    dataset.set_epoch(10)
    loader, sampler = base.train.dataloader(dataset, int(original_config["scientific_seed"]), 10,
                                             int(runtime["physical_batch"]), int(original_config["training"]["workers"]))
    return base, original_config, calibration, model, optimizer, loader, loss_function


def run_replay_once(output: Path, expected_sample_ids: Sequence[str], *, normalization: str,
                    tau: float | None = None, stop_update: int = 447, step_stop_update: bool = False,
                    use_amp: bool = True) -> dict[str, Any]:
    if Path(output).exists():
        raise FileExistsError(output)
    Path(output).mkdir(parents=True, exist_ok=False)
    base, config, calibration, model, optimizer, loader, loss_function = prepare_runtime(tau, normalization)
    accumulation = 4; global_update = 9468; healthy_records = []; boundary: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats()
    for update_in_epoch, microbatches in enumerate(base.train.microbatch_groups(loader, accumulation), 1):
        if update_in_epoch > stop_update:
            break
        if update_in_epoch == stop_update and len(microbatches) != 4:
            raise RuntimeError("registered update boundary is not four physical microbatches")
        lrs = base.train.scheduled_lrs(config, 10, global_update + 1); base.train.set_lrs(optimizer, lrs)
        optimizer.zero_grad(set_to_none=True)
        model_before = model_hash(model); optimizer_before = optimizer_hash(optimizer)
        sample_ids: list[str] = []; micro_records = []; accumulated_parts: dict[str, float] = defaultdict(float)
        for physical_index, batch in enumerate(microbatches):
            sample_ids.extend(batch["sample_ids"])
            audit_context = ForwardAudit(model) if update_in_epoch == stop_update else None
            if audit_context is not None:
                audit_context.__enter__(); audit_context.add("batch.input", batch["input"])
            try:
                total, parts, audit, outputs = loss_function(
                    model, batch, calibration["multipliers"], use_amp=use_amp,
                    audit_detail=update_in_epoch == stop_update)
                scalar = base.losses.scalar_components(parts)
                if not base.common.finite_tree(scalar):
                    raise FloatingPointError(f"nonfinite individual loss at epoch10 update{update_in_epoch}")
                for name, value in scalar.items():
                    accumulated_parts[name] += value / len(microbatches)
                diagnostics = _autograd_diagnostics(model, parts, outputs) if update_in_epoch == stop_update else None
                if audit_context is not None:
                    audit_context.add("forward.outputs", outputs)
                    audit_context.add("loss.parts", parts)
                (total / len(microbatches)).backward()
                if update_in_epoch == stop_update:
                    raw_audit = audit_tree(outputs, "outputs")
                    require_finite_audit(raw_audit)
                    decode_audit = _training_decode_audit(model, outputs)
                    micro_records.append({"physical_microbatch": physical_index + 1,
                        "sample_ids": list(batch["sample_ids"]), "losses": scalar,
                        "group_and_per_loss_autograd": diagnostics, "geometry": audit.get("geometry", {}),
                        "tensor_audit": audit_context.records if audit_context is not None else raw_audit,
                        "score_and_box_decode_audit": decode_audit})
            finally:
                if audit_context is not None:
                    audit_context.__exit__(None, None, None)
            del total, parts, audit, outputs
        if not base.train.all_gradients_finite(model):
            raise FloatingPointError(f"nonfinite gradient at epoch10 update{update_in_epoch}")
        metrics = proposed_sgd_metrics(model, optimizer)
        record = {"source": "EPOCH10_EXPLICIT_REPLAY", "epoch": 10, "update_in_epoch": update_in_epoch,
                  "global_update_if_stepped": global_update + 1, "finite": metrics["finite"], "metrics": metrics}
        if update_in_epoch <= 446:
            healthy_records.append(record)
        if update_in_epoch == stop_update:
            if list(sample_ids) != list(expected_sample_ids):
                raise RuntimeError(f"update-447 sample identity mismatch: {sample_ids}")
            boundary = {"epoch": 10, "update_in_epoch": update_in_epoch, "global_update_if_stepped": global_update + 1,
                        "sample_ids": sample_ids, "accumulated_loss_components": dict(accumulated_parts),
                        "physical_microbatches": micro_records,
                        "parameter_gradient_groups": _parameter_gradients(model), "pre_step": metrics,
                        "model_hash_before_forward_backward": model_before,
                        "model_hash_after_forward_backward": model_hash(model),
                        "optimizer_hash_before_forward_backward": optimizer_before,
                        "optimizer_hash_after_forward_backward": optimizer_hash(optimizer),
                        "model_optimizer_unchanged_by_forward_backward": model_before == model_hash(model)
                            and optimizer_before == optimizer_hash(optimizer),
                        "optimizer_step_executed": bool(step_stop_update)}
            if not step_stop_update:
                break
        optimizer.step(); global_update += 1
        _finite_model_optimizer(base, model, optimizer)
    if boundary is None:
        raise RuntimeError(f"requested replay boundary {stop_update} not reached")
    report = {"schema": "splitfusion_fcos_explicit_epoch10_replay_v1", "normalization": normalization,
              "tau": tau, "source_checkpoint_epoch": 9, "source_global_update": 9468,
              "latest_checkpoint_discovery_used": False, "healthy_records": healthy_records,
              "boundary": boundary, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "vram_cap_mib": 12288, "validation_accessed": False}
    atomic_json(Path(output) / "replay.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gated real replay; never selects tau")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-update447", required=True, type=Path)
    parser.add_argument("--normalization", required=True, choices=("original", "candidate"))
    parser.add_argument("--tau", type=float)
    parser.add_argument("--execute-real-replay", required=True, choices=("I_UNDERSTAND_THIS_RUNS_REAL_TRAINING_CODE",))
    args = parser.parse_args()
    expected = load_json(args.expected_update447)
    if expected.get("preregistered") is not True or len(expected.get("sample_ids", [])) != 16:
        raise RuntimeError("reviewed preregistered update-447 sample IDs are required")
    run_replay_once(args.output, expected["sample_ids"], normalization=args.normalization, tau=args.tau)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
