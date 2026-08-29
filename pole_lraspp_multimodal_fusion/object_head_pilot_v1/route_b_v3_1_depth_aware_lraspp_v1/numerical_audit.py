from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from common import CONFIG_PATH, load_json, read_csv, rng_state, seed_everything, utc_now, write_json_x
from data import (CLASS_NAMES, DepthCache, TrainingDataset, _owner_record, collate_training,
                  load_objects, load_visible_anchors)
from losses import _gather, compute_losses
from model import build_model, configure_stage, freeze_bn_running_state, stage_train_mode
from train import (SCIENTIFIC_COMPONENTS, build_optimizer, scheduled_lrs, set_optimizer_lrs)


FAILED_BATCH = 14
ORIGINAL_CACHE = Path("experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_042423/depth_cache/train")
ORIGINAL_FAILURE = Path("experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_042423/SCIENTIFIC_RUNTIME_FAILURE.json")


def _feed_hash(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0" + str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _feed_hash(digest, value.item())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _feed_hash(digest, key)
            _feed_hash(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update((b"tuple\0" if isinstance(value, tuple) else b"list\0"))
        digest.update(struct.pack(">Q", len(value)))
        for item in value:
            _feed_hash(digest, item)
    elif isinstance(value, bytes):
        digest.update(b"bytes\0" + struct.pack(">Q", len(value)) + value)
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str\0" + struct.pack(">Q", len(encoded)) + encoded)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + bytes([value]))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack(">d", value))
    else:
        raise TypeError(f"unsupported hash value {type(value)}")


def canonical_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _feed_hash(digest, value)
    return digest.hexdigest()


def named_tensor_hash(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    return canonical_hash({name: value for name, value in values})


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach()
    count = tensor.numel()
    floating = tensor.dtype.is_floating_point or tensor.dtype.is_complex
    finite = torch.isfinite(tensor) if floating else torch.ones_like(tensor, dtype=torch.bool)
    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "elements": count,
        "finite_count": finite_count, "nonfinite_count": count - finite_count,
    }
    if finite_count:
        selected = tensor[finite] if finite_count != count else tensor
        if selected.dtype.is_complex:
            magnitudes = selected.abs()
            result.update({"finite_minimum": None, "finite_maximum": None,
                           "finite_absolute_maximum": float(magnitudes.max().item())})
        else:
            result.update({
                "finite_minimum": float(selected.min().item()),
                "finite_maximum": float(selected.max().item()),
                "finite_absolute_maximum": float(selected.abs().max().item()),
            })
    else:
        result.update({"finite_minimum": None, "finite_maximum": None, "finite_absolute_maximum": None})
    return result


def state_finiteness(values: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    tensors = 0
    elements = 0
    nonfinite_elements = 0
    nonfinite_names: list[str] = []
    nonzero_tensors = 0
    absolute_maximum = 0.0
    for name, value in values:
        tensors += 1
        elements += value.numel()
        summary = tensor_summary(value)
        nonfinite_elements += summary["nonfinite_count"]
        if summary["nonfinite_count"] and len(nonfinite_names) < 20:
            nonfinite_names.append(name)
        if value.numel() and bool(torch.count_nonzero(value.detach()).item()):
            nonzero_tensors += 1
        if summary["finite_absolute_maximum"] is not None:
            absolute_maximum = max(absolute_maximum, summary["finite_absolute_maximum"])
    return {
        "tensor_count": tensors, "element_count": elements,
        "nonfinite_elements": nonfinite_elements, "finite": nonfinite_elements == 0,
        "nonfinite_names": nonfinite_names, "nonzero_tensors": nonzero_tensors,
        "finite_absolute_maximum": absolute_maximum,
    }


def gradient_state(model: torch.nn.Module) -> dict[str, Any]:
    values = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    result = state_finiteness((name, value) for name, value in values)
    result["sha256"] = named_tensor_hash((name, value) for name, value in values)
    return result


def optimizer_tensor_values(optimizer: torch.optim.Optimizer) -> Iterable[tuple[str, torch.Tensor]]:
    name_by_id = {id(parameter): name for name, parameter in optimizer._model_named_parameters}  # type: ignore[attr-defined]
    for parameter, state in optimizer.state.items():
        name = name_by_id[id(parameter)]
        for key, value in sorted(state.items()):
            if isinstance(value, torch.Tensor):
                yield f"{name}.{key}", value


def optimizer_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    values = list(optimizer_tensor_values(optimizer))
    report = state_finiteness(values)
    report["sha256"] = canonical_hash(optimizer.state_dict())
    report["state_tensor_sha256"] = named_tensor_hash(values)
    exp_avg = [(name, value) for name, value in values if name.endswith(".exp_avg")]
    exp_avg_sq = [(name, value) for name, value in values if name.endswith(".exp_avg_sq")]
    steps = [(name, value) for name, value in values if name.endswith(".step")]
    report["exp_avg"] = state_finiteness(exp_avg)
    report["exp_avg_sq"] = state_finiteness(exp_avg_sq)
    report["step"] = state_finiteness(steps)
    return report


def loss_report(parts: Mapping[str, torch.Tensor], weights: Mapping[str, float],
                total: torch.Tensor) -> dict[str, Any]:
    unweighted = {name: tensor_summary(parts[name]) for name in SCIENTIFIC_COMPONENTS}
    weighted = {name: tensor_summary(float(weights[name]) * parts[name]) for name in SCIENTIFIC_COMPONENTS}
    return {"unweighted": unweighted, "weighted": weighted, "total": tensor_summary(total)}


def batch_hashes(batch: Mapping[str, Any]) -> dict[str, Any]:
    targets = {key: value for key, value in batch.items() if key not in ("input", "sample_id")}
    return {
        "batch": canonical_hash(batch),
        "input": canonical_hash(batch["input"]),
        "targets": canonical_hash(targets),
        "sample_ids": canonical_hash(batch["sample_id"]),
        "sample_ids_ordered": list(batch["sample_id"]),
    }


def selected_owner_metadata(
    batch: Mapping[str, Any], frame_lookup: Mapping[str, Mapping[str, str]],
    objects: Mapping[str, Sequence[Mapping[str, Any]]],
    visible: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
    for batch_index, sample_id in enumerate(batch["sample_id"]):
        frame = frame_lookup[sample_id]
        candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
        for row in objects.get(sample_id, ()):
            if row.get("contract_state") != "POSITIVE":
                continue
            item = _owner_record(row, frame, visible.get((sample_id, row["source_identity"])))
            candidates[item["class_name"]].append(item)
        for class_name in CLASS_NAMES:
            by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for item in candidates[class_name]:
                by_cell[(item["cell_y"], item["cell_x"])].append(item)
            selected = [
                sorted(items, key=lambda value: (-value["area"], value["depth"], value["source_identity"]))[0]
                for items in by_cell.values()
            ]
            selected.sort(key=lambda item: (item["cell_y"], item["cell_x"], item["source_identity"]))
            for item in selected:
                result[class_name].append({
                    "batch_index": batch_index, "sample_id": sample_id,
                    "source_identity": item["source_identity"],
                    "cell_y": item["cell_y"], "cell_x": item["cell_x"],
                    "target_depth_m": item["depth"],
                })
    for class_name in CLASS_NAMES:
        cells = batch["owners"][class_name]["cells"].tolist()
        expected = [[item["batch_index"], item["cell_y"], item["cell_x"]] for item in result[class_name]]
        if cells != expected:
            raise RuntimeError(f"owner metadata ordering mismatch for {class_name}")
    return result


def nonfinite_origins(output: torch.Tensor, metadata: Sequence[Mapping[str, Any]],
                      input_value: torch.Tensor | None) -> list[dict[str, Any]]:
    if output.ndim == 0 or output.shape[0] != len(metadata):
        return []
    mask = ~torch.isfinite(output.detach())
    rows = sorted(set(int(value) for value in torch.nonzero(mask, as_tuple=False)[:, 0].tolist()))
    result = []
    for row in rows[:20]:
        item = dict(metadata[row])
        positions = torch.nonzero(mask[row], as_tuple=False).reshape(-1).tolist()
        item["nonfinite_component_indices"] = positions
        if input_value is not None and input_value.ndim and input_value.shape[0] == len(metadata):
            raw = input_value.detach()[row].reshape(-1).cpu().tolist()
            item["causal_input_values"] = [
                float(value) if math.isfinite(float(value)) else (
                    "Infinity" if float(value) > 0.0 else "-Infinity" if float(value) < 0.0 else "NaN"
                )
                for value in raw
            ]
        result.append(item)
    return result


def append_event(events: list[dict[str, Any]], operation: str, value: torch.Tensor,
                 metadata: Sequence[Mapping[str, Any]] = (),
                 input_value: torch.Tensor | None = None) -> torch.Tensor:
    record = {"sequence": len(events), "operation": operation, "summary": tensor_summary(value)}
    if input_value is not None:
        record["input_summary"] = tensor_summary(input_value)
    if record["summary"]["nonfinite_count"]:
        record["origins"] = nonfinite_origins(value, metadata, input_value)
    events.append(record)
    return value


class ModuleProbe:
    def __init__(self, model: torch.nn.Module) -> None:
        self.active = False
        self.events: list[dict[str, Any]] = []
        self.handles = []
        for name, module in model.named_modules():
            if name and not any(True for _ in module.children()):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if not self.active:
                return
            values: list[tuple[str, torch.Tensor]] = []
            if isinstance(output, torch.Tensor):
                values.append(("", output))
            elif isinstance(output, Mapping):
                values.extend((f".{key}", value) for key, value in output.items() if isinstance(value, torch.Tensor))
            elif isinstance(output, (tuple, list)):
                values.extend((f".{index}", value) for index, value in enumerate(output) if isinstance(value, torch.Tensor))
            for suffix, value in values:
                append_event(self.events, f"module.{name}{suffix}", value)
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def explicit_operation_probe(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], parts: Mapping[str, torch.Tensor],
    total: torch.Tensor, model: torch.nn.Module, frame_lookup: Mapping[str, Mapping[str, str]],
    owner_metadata: Mapping[str, Sequence[Mapping[str, Any]]], module_events: list[dict[str, Any]],
) -> dict[str, Any]:
    events = list(module_events)
    append_event(events, "loss.segmentation", parts["segmentation"])
    for class_index, class_name in enumerate(CLASS_NAMES):
        branch = outputs["objects"][class_name]
        cells = batch["owners"][class_name]["cells"].to(next(model.parameters()).device)
        metadata = owner_metadata[class_name]
        heatmap_prediction = torch.sigmoid(branch["heatmap"]).clamp(1e-4, 1.0 - 1e-4)
        append_event(events, f"{class_name}.heatmap.sigmoid_clamp", heatmap_prediction)
        gathered: dict[str, torch.Tensor] = {}
        for field in ("subcell", "box_center_delta", "box_wh", "physical_ray_delta",
                      "depth_bin_logits", "depth_bin_residuals", "log_dimensions",
                      "yaw_sincos", "radar_support"):
            gathered[field] = _gather(branch[field], cells)
            append_event(events, f"{class_name}.gather.{field}", gathered[field], metadata)
        if class_name == "vehicle":
            gathered["parked"] = _gather(branch["parked"], cells)
            append_event(events, "vehicle.gather.parked", gathered["parked"], metadata)
        if not len(cells):
            continue
        subcell = append_event(events, f"{class_name}.subcell.sigmoid", torch.sigmoid(gathered["subcell"]), metadata)
        append_event(events, f"{class_name}.box_wh.softplus", F.softplus(gathered["box_wh"]), metadata)
        logits = gathered["depth_bin_logits"]
        residuals = gathered["depth_bin_residuals"]
        log_probabilities = append_event(events, f"{class_name}.depth.log_softmax", F.log_softmax(logits, dim=1), metadata)
        del log_probabilities
        probabilities = append_event(events, f"{class_name}.depth.softmax", F.softmax(logits, dim=1), metadata)
        anchors = model.depth_anchors[None, :]
        candidates = append_event(
            events, f"{class_name}.depth.anchor_plus_delta_residual",
            anchors + model.depth_delta * residuals, metadata,
        )
        contributions = append_event(
            events, f"{class_name}.depth.probability_weighted_contribution",
            probabilities * candidates, metadata,
        )
        z_prediction = append_event(
            events, f"{class_name}.depth.summed_log_depth", contributions.sum(dim=1), metadata,
        )
        expm1_depth = append_event(
            events, f"{class_name}.depth.expm1", torch.expm1(z_prediction), metadata,
            input_value=z_prediction,
        )
        decoded_depth = append_event(
            events, f"{class_name}.depth.nonnegative_lower_guard", expm1_depth.clamp_min(0.0), metadata,
            input_value=expm1_depth,
        )
        batch_index, cell_y, cell_x = cells[:, 0], cells[:, 1].float(), cells[:, 2].float()
        grid_anchor_x = cell_x + subcell[:, 0]
        grid_anchor_y = cell_y + subcell[:, 1]
        u_physical = append_event(
            events, f"{class_name}.geometry.u_physical",
            4.0 * (grid_anchor_x + gathered["physical_ray_delta"][:, 0]), metadata,
        )
        v_physical = append_event(
            events, f"{class_name}.geometry.v_physical",
            4.0 * (grid_anchor_y + gathered["physical_ray_delta"][:, 1]), metadata,
        )
        intrinsic = batch["owners"][class_name]["intrinsic"].to(decoded_depth.device)
        ray_components = append_event(events, f"{class_name}.geometry.ray_components", torch.stack([
            torch.ones_like(decoded_depth),
            (u_physical - intrinsic[:, 2]) / intrinsic[:, 0],
            (intrinsic[:, 3] - v_physical) / intrinsic[:, 1],
        ], dim=1), metadata)
        local = append_event(
            events, f"{class_name}.geometry.local_xyz", decoded_depth[:, None] * ray_components, metadata,
        )
        world_values = []
        for index, item in enumerate(metadata):
            matrix = torch.tensor(json.loads(frame_lookup[item["sample_id"]]["camera_matrix_json"]),
                                  device=local.device, dtype=torch.float64)
            homogeneous = torch.cat([local[index].double(), torch.ones(1, device=local.device, dtype=torch.float64)])
            world_values.append((matrix @ homogeneous)[:3])
        world = torch.stack(world_values) if world_values else local.new_empty((0, 3), dtype=torch.float64)
        append_event(events, f"{class_name}.geometry.world_xyz", world, metadata)
        log_dimensions = gathered["log_dimensions"]
        dimensions = append_event(
            events, f"{class_name}.dimensions.exp", torch.exp(log_dimensions), metadata,
            input_value=log_dimensions,
        )
        clamped_dimensions = append_event(
            events, f"{class_name}.dimensions.clamp_min", dimensions.clamp_min(1e-6), metadata,
            input_value=dimensions,
        )
        append_event(
            events, f"{class_name}.dimensions.log", torch.log(clamped_dimensions), metadata,
            input_value=clamped_dimensions,
        )
        append_event(events, f"{class_name}.yaw.normalize", F.normalize(gathered["yaw_sincos"], dim=1, eps=1e-6), metadata)
        append_event(events, f"{class_name}.radar_support.logits", gathered["radar_support"], metadata)
        if class_name == "vehicle":
            append_event(events, "vehicle.parked.logits", gathered["parked"], metadata)
    append_event(events, "dense.raw_log1p", outputs["dense_depth_log1p"])
    valid = batch["dense_valid"].to(outputs["dense_depth_log1p"].device)
    append_event(events, "dense.valid_predictions", outputs["dense_depth_log1p"][:, 0][valid])
    radar_samples = []
    for batch_index, points_cpu in enumerate(batch["radar_points"]):
        if not points_cpu.numel():
            continue
        points = points_cpu.to(outputs["dense_depth_log1p"].device)
        grid = points[:, :2].view(1, 1, -1, 2)
        radar_samples.append(F.grid_sample(
            outputs["dense_depth_log1p"][batch_index:batch_index + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        ).reshape(-1))
    append_event(events, "dense.radar_sampled_predictions", torch.cat(radar_samples))
    for name in SCIENTIFIC_COMPONENTS:
        append_event(events, f"loss.unweighted.{name}", parts[name])
    append_event(events, "loss.total", total)
    first = next((record for record in events if record["summary"]["nonfinite_count"]), None)
    return {"events": events, "first_nonfinite": first, "event_count": len(events)}


def fixed_batch_rng_consumption(rows: Sequence[dict[str, str]], dataset_root: Path,
                                objects: Mapping[str, Sequence[dict[str, str]]],
                                visible: Mapping[tuple[str, str], Mapping[str, Any]],
                                cache: DepthCache, seed: int) -> None:
    fixed_rows = list(rows[:min(4, len(rows))])
    for row in rows:
        if any(item["label"] == "person" for item in objects.get(row["sample_id"], ())):
            fixed_rows[-1] = row
            break
    fixed_dataset = TrainingDataset(dataset_root, fixed_rows, objects, visible, cache, seed)
    fixed_dataset.set_epoch(1)
    next(iter(DataLoader(fixed_dataset, batch_size=len(fixed_rows), num_workers=0,
                         collate_fn=collate_training)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ORIGINAL_CACHE)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    started_wall = time.monotonic()
    config = load_json(CONFIG_PATH)
    failure = load_json(ORIGINAL_FAILURE)
    seed = int(config["scientific_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for deterministic numerical reproduction")
    device = torch.device("cuda")
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "train"]
    frame_lookup = {row["sample_id"]: row for row in rows}
    index_lookup = {row["sample_id"]: index for index, row in enumerate(rows)}
    objects = load_objects(dataset_root)
    visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    cache = DepthCache(args.cache.resolve(strict=True), rows)
    dataset = TrainingDataset(dataset_root, rows, objects, visible, cache, seed)
    fixed_batch_rng_consumption(rows, dataset_root, objects, visible, cache, seed)
    model, loading = build_model(Path(config["pretrained"]["path"]), device)
    optimizer = build_optimizer(model)
    optimizer._model_named_parameters = list(model.named_parameters())  # type: ignore[attr-defined]
    initial = {
        "model_state_sha256": canonical_hash(model.state_dict()),
        "parameters_sha256": named_tensor_hash(model.named_parameters()),
        "buffers_sha256": named_tensor_hash(model.named_buffers()),
        "optimizer_sha256": canonical_hash(optimizer.state_dict()),
        "rng_sha256": canonical_hash(rng_state()),
        "rng_components": {
            "python": canonical_hash(random.getstate()),
            "numpy": canonical_hash(np.random.get_state()),
            "torch_cpu": canonical_hash(torch.get_rng_state()),
            "torch_cuda": canonical_hash(torch.cuda.get_rng_state_all()),
        },
    }
    configure_stage(model, "A")
    stage_train_mode(model, "A")
    dataset.set_epoch(1)
    sampler_generator = torch.Generator().manual_seed(seed + 1)
    sampler = RandomSampler(dataset, replacement=False, generator=sampler_generator)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=False, drop_last=False, collate_fn=collate_training)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    weights = config["loss_weights"]
    updates_per_epoch = math.ceil(len(rows) / 16)
    updates = []
    batches = []
    probe = ModuleProbe(model)
    batch14_record: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(1, 15):
        batch = next(iterator)
        hashes = batch_hashes(batch)
        hashes["batch_index"] = batch_index
        batches.append(hashes)
        if batch_index < FAILED_BATCH:
            total, parts, denominators, _outputs = compute_losses(model, batch, weights)
            losses = loss_report(parts, weights, total)
            if losses["total"]["nonfinite_count"]:
                raise RuntimeError(f"unexpected non-finite before failed batch: {batch_index}")
            total.backward()
            preclip = gradient_state(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            postclip = gradient_state(model)
            new_lr, backbone_lr = scheduled_lrs(1, batch_index, updates_per_epoch, batch_index)
            set_optimizer_lrs(optimizer, new_lr, backbone_lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            parameter_state = state_finiteness(model.named_parameters())
            buffer_state = state_finiteness(model.named_buffers())
            optimizer_state = optimizer_report(optimizer)
            update = {
                "update": batch_index, "sample_ids": list(batch["sample_id"]), "losses": losses,
                "denominators": denominators, "preclip_gradients": preclip,
                "clip_returned_norm": float(norm.item()), "postclip_gradients": postclip,
                "new_lr": new_lr, "backbone_lr": backbone_lr,
                "parameters": parameter_state, "buffers": buffer_state,
                "optimizer": optimizer_state,
            }
            if not all((preclip["finite"], postclip["finite"], parameter_state["finite"],
                        buffer_state["finite"], optimizer_state["finite"], math.isfinite(float(norm.item())))):
                raise RuntimeError(f"state became non-finite before batch 14 at update {batch_index}")
            updates.append(update)
        else:
            append_event(probe.events, "model.input", batch["input"])
            probe.active = True
            total, parts, denominators, outputs = compute_losses(model, batch, weights)
            probe.active = False
            metadata = selected_owner_metadata(batch, frame_lookup, objects, visible)
            operations = explicit_operation_probe(
                outputs, batch, parts, total, model, frame_lookup, metadata, probe.events,
            )
            batch14_record = {
                "sample_ids": list(batch["sample_id"]),
                "matches_failed_record": list(batch["sample_id"]) == failure["failed_batch_input_audit"]["sample_ids"],
                "indices_match_failed_record": [index_lookup[sample_id] for sample_id in batch["sample_id"]]
                == failure["failed_batch_input_audit"]["dataset_indices"],
                "losses": loss_report(parts, weights, total), "denominators": denominators,
                "forward_and_loss_only": True, "backward": False, "optimizer_step": False,
                "owner_metadata": metadata, "operation_probe": operations,
            }
    batch15 = next(iterator)
    batch15_hashes = batch_hashes(batch15)
    batch15_hashes["batch_index"] = 15
    batches.append(batch15_hashes)
    probe.close()
    if batch14_record is None:
        raise RuntimeError("batch 14 was not observed")
    post13 = {
        "model_state_sha256": canonical_hash(model.state_dict()),
        "parameters_sha256": named_tensor_hash(model.named_parameters()),
        "buffers_sha256": named_tensor_hash(model.named_buffers()),
        "optimizer_sha256": canonical_hash(optimizer.state_dict()),
        "rng_sha256": canonical_hash(rng_state()),
        "rng_components": {
            "python": canonical_hash(random.getstate()),
            "numpy": canonical_hash(np.random.get_state()),
            "torch_cpu": canonical_hash(torch.get_rng_state()),
            "torch_cuda": canonical_hash(torch.cuda.get_rng_state_all()),
        },
        "parameters": state_finiteness(model.named_parameters()),
        "buffers": state_finiteness(model.named_buffers()),
        "optimizer": optimizer_report(optimizer),
    }
    first = batch14_record["operation_probe"]["first_nonfinite"]
    reproduced = (
        batch14_record["matches_failed_record"]
        and batch14_record["indices_match_failed_record"]
        and batch14_record["losses"]["total"]["nonfinite_count"] > 0
        and first is not None
    )
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_numerical_reproduction_v1",
        "created_utc": utc_now(), "reproduction": args.run,
        "failed_implementation_commit": "049f7029d9156871e02b2aed34da3cdbcbd842ef",
        "scientific_seed": seed, "sampler_seed": seed + 1, "precision": "full_fp32",
        "physical_batch": 16, "gradient_accumulation": 1,
        "official_loading": loading, "initial": initial, "batches_1_through_15": batches,
        "updates_1_through_13": updates, "post_update_13": post13,
        "batch_14": batch14_record, "original_failure_reproduced": reproduced,
        "first_nonfinite_operation": first,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "wall_seconds": time.monotonic() - started_wall,
        "validation_accessed": False, "test_accessed": False,
    }
    write_json_x(output, report)
    print(json.dumps({
        "reproduction": args.run, "reproduced": reproduced,
        "first_nonfinite_operation": first,
        "initial_model": initial["model_state_sha256"],
        "post13_model": post13["model_state_sha256"],
        "post13_optimizer": post13["optimizer_sha256"],
        "output": str(output), "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0 if reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
