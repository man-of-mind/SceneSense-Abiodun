from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn

from common import (CONFIG_PATH, ROOT, atomic_json, atomic_text, capture_rng, desktop_notify,
                    canonical_hash, finite_tree, load_json, named_tensor_hash, package_hashes,
                    restore_rng, seed_everything,
                    tensor_hash, utc_now)
from data import DepthCache, InferenceDataset, RouteBDataset, collate, load_split_rows
from losses import compute_loss_groups, scalar_components
from model import (LEVELS, SplitFusionFCOS, build_model, configure_trainability,
                   optimizer_parameter_groups, parameter_inventory)

MAX_MIB = 12288.0


def close(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left, right = left.detach().float(), right.detach().float()
    delta = (left - right).abs()
    return {"shape": list(left.shape), "allclose": bool(torch.allclose(left, right, rtol=1e-5, atol=1e-6)),
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean()) if delta.numel() else 0.0, "rtol": 1e-5, "atol": 1e-6}


def require_allclose(report: Mapping[str, Any], name: str) -> None:
    if not report["allclose"]:
        raise RuntimeError(f"parity failure {name}: {report}")


def build_optimizer(model: SplitFusionFCOS, lrs: Mapping[str, float]) -> torch.optim.SGD:
    groups = optimizer_parameter_groups(model)
    return torch.optim.SGD([
        {"params": [value for _, value in groups[name]], "lr": float(lrs[name]), "name": name}
        for name in ("pretrained_backbone", "pretrained_fpn_heads", "new")
    ], momentum=0.9, weight_decay=1e-4)


def optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
                return False
    return True


def gradients(model: torch.nn.Module) -> dict[str, Any]:
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        result[name] = {"norm": float(parameter.grad.detach().float().norm()),
                        "finite": bool(torch.isfinite(parameter.grad).all()),
                        "nonzero": bool(torch.count_nonzero(parameter.grad))}
    return result


def group_gradient_evidence(gradients_by_name: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selectors: dict[str, Callable[[str], bool]] = {
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
    for group, selector in selectors.items():
        rows = [value for name, value in gradients_by_name.items() if selector(name)]
        report[group] = {"tensors": len(rows), "finite": bool(rows) and all(row["finite"] for row in rows),
                         "nonzero": bool(rows) and any(row["nonzero"] for row in rows),
                         "l2": math.sqrt(sum(row["norm"] ** 2 for row in rows))}
    return report


def official_parity(model: SplitFusionFCOS, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    official = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1,
                                 progress=False, trainable_backbone_layers=5).to(device).eval()
    x7 = batch["input"].to(device)
    rgb_normalized = x7[:, :3]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb_01 = (rgb_normalized[..., :432, :] * std + mean).clamp(0, 1)
    with torch.inference_mode():
        smoke = official([rgb_01[0]])
        body = official.backbone.body
        official_stem = body.relu(body.bn1(body.conv1(rgb_normalized)))
        official_c2 = body.layer1(body.maxpool(official_stem))
        project_stem = model.front.relu(model.front.bn1(torch.nn.functional.conv2d(
            x7, model.front.concatenated_weight(), stride=model.front.stride, padding=model.front.padding)))
        project_c2 = model.encode_front(x7)
        official_c3 = body.layer2(official_c2); official_c4 = body.layer3(official_c3); official_c5 = body.layer4(official_c4)
        official_cs = {"0": official_c3, "1": official_c4, "2": official_c5}
        official_fpn = official.backbone.fpn(official_cs)
        project_features, project_cs = model.tail(project_c2)
        parity = {
            "stem": close(project_stem, official_stem), "c2": close(project_c2, official_c2),
            "c3": close(project_cs["0"], official_c3), "c4": close(project_cs["1"], official_c4),
            "c5": close(project_cs["2"], official_c5),
        }
        for project_name, official_name in zip(("p3", "p4", "p5", "p6", "p7"), ("0", "1", "2", "p6", "p7")):
            parity[project_name] = close(project_features[project_name], official_fpn[official_name])
        project_head = model._detection_heads(list(project_features.values())[1:])
        official_head = official.head(list(official_fpn.values()))
        parity["classification_tower_p3"] = close(project_head["classification_tower"][0],
                                                    official.head.classification_head.conv(official_fpn["0"]))
        parity["regression_tower_p3"] = close(project_head["regression_tower"][0],
                                                official.head.regression_head.conv(official_fpn["0"]))
        parity["box_regression_p3_p7"] = close(project_head["bbox_regression"], official_head["bbox_regression"])
        parity["centerness_p3_p7"] = close(project_head["bbox_ctrness"], official_head["bbox_ctrness"])
    for name, value in parity.items():
        require_allclose(value, name)
    return {
        "official_transform_smoke": {"input_shape": list(rgb_01[0].shape),
                                     "detections": int(len(smoke[0]["scores"])),
                                     "finite": finite_tree(smoke)},
        "same_external_normalized_padded_rgb_with_real_radar": True,
        "radar_finite_nonzero": bool(torch.isfinite(x7[:, 3:]).all() and torch.count_nonzero(x7[:, 3:])),
        "radar_weights_zero": int(torch.count_nonzero(model.front.W_radar)) == 0,
        "parity": parity,
        "p2_disabled_p3_p7_unchanged": all(parity[name]["allclose"] for name in ("p3", "p4", "p5", "p6", "p7")),
    }


def roundtrip_tests(target: Mapping[str, Any]) -> dict[str, Any]:
    intrinsic = target["intrinsic"].double()
    local = target["local_xyz"].double()
    if not len(local):
        raise RuntimeError("roundtrip batch needs actors")
    depth = local[:, 0]
    uv = torch.stack((intrinsic[0, 2] + intrinsic[0, 0] * local[:, 1] / depth,
                      intrinsic[1, 2] - intrinsic[1, 1] * local[:, 2] / depth), dim=1)
    recovered = torch.stack((depth, depth * (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                             depth * (intrinsic[1, 2] - uv[:, 1]) / intrinsic[1, 1]), dim=1)
    world = torch.cat((local, torch.ones(len(local), 1, dtype=torch.float64)), dim=1) @ target["extrinsic"].T
    inverse = torch.linalg.inv(target["extrinsic"])
    recovered_local = world @ inverse.T
    source_intrinsic = intrinsic.clone()
    source_intrinsic[0] /= (768 / 1280); source_intrinsic[1] /= (432 / 720)
    scaled = source_intrinsic.clone(); scaled[0] *= 768 / 1280; scaled[1] *= 432 / 720
    report = {
        "projection_unprojection": close(recovered, local),
        "camera_world_camera": close(recovered_local[:, :3], local),
        "intrinsic_scale_roundtrip": close(scaled, intrinsic),
        "bottom_padding_intrinsics_unchanged": True,
        "boxes_original_coordinate_frame": bool((target["boxes"][:, 2] <= 768).all() and (target["boxes"][:, 3] <= 432).all()),
    }
    for name, value in report.items():
        if isinstance(value, dict): require_allclose(value, name)
    return report


def lineage_test(model: SplitFusionFCOS, outputs: Mapping[str, Any], calibration: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    def detached_clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, Mapping):
            return {key: detached_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detached_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detached_clone(item) for item in value)
        return value
    synthetic = detached_clone(outputs)
    for level_index, level_name in enumerate(LEVELS):
        cls = synthetic["detection"]["per_level"]["cls_logits"][level_index]
        ctr = synthetic["detection"]["per_level"]["bbox_ctrness"][level_index]
        box = synthetic["detection"]["per_level"]["bbox_regression"][level_index]
        cls.fill_(-100.0); ctr.fill_(100.0)
        box.zero_()
        for field in synthetic["geometry"][level_index].values(): field.zero_()
        count = cls.shape[1]
        active = min(count, 1100 if level_index == 0 else 20)
        indices = torch.arange(active, device=cls.device)
        classes = indices % 2
        cls[0, indices, classes] = -5.0
        high = min(20, active)
        cls[0, indices[:high], classes[:high]] = (
            5.0 - torch.arange(high, device=cls.device) * 0.1 - level_index * 0.001)
        code = level_index * 100000 + indices * 2 + classes
        dims = synthetic["geometry"][level_index]["log_dimensions"]
        dims[0, indices, classes, 0] = -2.0 + code.float() * 1e-7
        dims[0, indices, classes, 1:] = -2.0
        synthetic["geometry"][level_index]["yaw"][0, indices, classes, 1] = 1.0
    result = model.postprocess(synthetic, [calibration])[0]
    identities = result["candidate_identity"]
    observed = result["dimensions"][:, 0].double()
    code = identities[:, 1] * 100000 + identities[:, 2] * 2 + identities[:, 3]
    expected = torch.exp((-2.0 + code.double() * 1e-7))
    exact = bool(torch.allclose(observed, expected, rtol=1e-6, atol=1e-8))
    report = {"survivors": len(identities), "final_truncation": len(identities) <= 100,
              "identity_unique": len(torch.unique(identities, dim=0)) == len(identities),
              "gather_exact": exact, "image_index_exact": bool((identities[:, 0] == 0).all()),
              "levels_present": sorted(set(identities[:, 1].tolist())),
              "all_levels_survive": sorted(set(identities[:, 1].tolist())) == list(range(len(LEVELS))),
              "operations": ["per-level flatten", "class flatten", "score filtering", "top-k reordering",
                             "level concatenation", "classwise NMS", "final truncation"]}
    if not all(report[name] for name in ("final_truncation", "identity_unique", "gather_exact",
                                         "image_index_exact", "all_levels_survive")):
        raise RuntimeError(f"synthetic lineage failure: {report}")
    return report


def real_lineage_test(model: SplitFusionFCOS, outputs: Mapping[str, Any], calibration: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result = model.postprocess(outputs, [calibration])[0]
    checks = []
    for row, identity in enumerate(result["candidate_identity"]):
        image, level, point, label = map(int, identity.tolist())
        if image != 0:
            raise RuntimeError("real geometry image-identity drift")
        raw = outputs["geometry"][level]["log_dimensions"][0, point, label].double()
        checks.append(bool(torch.allclose(result["dimensions"][row], torch.exp(raw), rtol=1e-6, atol=1e-8)))
    report = {"survivors": len(checks), "all_exact_level_point_class_gathers": all(checks),
              "identity_fields": ["image", "level", "flattened_point", "internal_class"]}
    if not checks or not all(checks): raise RuntimeError("real geometry lineage failure or no retained real candidate")
    return report


def measure_latency(model: SplitFusionFCOS, value: torch.Tensor) -> dict[str, Any]:
    def timed(function: Callable[[], Any], warmup: int = 10, iterations: int = 30) -> dict[str, float]:
        for _ in range(warmup): function()
        torch.cuda.synchronize()
        values = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); function(); end.record(); torch.cuda.synchronize()
            values.append(start.elapsed_time(end))
        return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
    with torch.inference_mode():
        c2 = model.encode_front(value)
        front = timed(lambda: model.encode_front(value))
        payload = timed(lambda: model.transport_encode(c2))
        tail = timed(lambda: model.decode_tail(model.transport_decode(c2), dense=False))
        monolithic = timed(lambda: model(value, dense=False))
        copy_values = []
        for _ in range(10):
            start = time.perf_counter(); blob = c2.detach().cpu().contiguous().numpy().tobytes(); torch.cuda.synchronize()
            copy_values.append((time.perf_counter() - start) * 1000)
    return {"front": front, "transport_identity_gpu": payload, "payload_copy_serialization": {
                "median_ms": float(np.median(copy_values)), "p95_ms": float(np.percentile(copy_values, 95)),
                "serialized_bytes": len(blob)}, "tail": tail, "monolithic": monolithic,
            "warmup": 10, "iterations": 30}


def batch_probe(config: Mapping[str, Any], priors: Mapping[str, Any], dataset: RouteBDataset,
                device: torch.device) -> tuple[int, list[dict[str, Any]]]:
    reports = []
    for physical in config["training"]["physical_batch_candidates"]:
        torch.cuda.empty_cache(); seed_everything(int(config["scientific_seed"]))
        model, _ = build_model(priors, device); configure_trainability(model, 4)
        optimizer = build_optimizer(model, config["training"]["base_lrs"])
        # Materialize the exact SGD-momentum state footprint without performing
        # an optimizer step; loss calibration must temporally precede all steps.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    optimizer.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = collate([dataset[index] for index in range(physical)])
            optimizer.zero_grad(set_to_none=True)
            total, parts, _audit, _outputs = compute_loss_groups(model, batch, {"G": 1, "S": 1, "A": 1})
            total.backward()
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            finite = finite_tree(parts) and all(torch.isfinite(value).all() for value in model.parameters()) and optimizer_finite(optimizer)
            accepted = finite and peak <= MAX_MIB and 16 % physical == 0
            reports.append({"physical_batch": physical, "peak_allocated_mib": peak,
                            "finite_full_joint_forward_backward": finite, "optimizer_step": False,
                            "sgd_momentum_footprint_materialized": True, "accepted": accepted, "error": None})
            del batch, total, parts, _audit, _outputs, model, optimizer
            torch.cuda.empty_cache()
            if accepted:
                return physical, reports
        except torch.cuda.OutOfMemoryError as error:
            reports.append({"physical_batch": physical, "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                            "finite_full_joint_forward_backward": False, "optimizer_step": False,
                            "sgd_momentum_footprint_materialized": True, "accepted": False,
                            "error": "CUDA_OUT_OF_MEMORY_AT_12_GIB_CAP"})
            del model, optimizer
            torch.cuda.empty_cache()
    raise RuntimeError(f"no physical batch qualified: {reports}")


def calibrate(config: Mapping[str, Any], registration: Mapping[str, Any], priors: Mapping[str, Any],
              dataset: RouteBDataset, physical: int, device: torch.device) -> dict[str, Any]:
    seed_everything(int(config["scientific_seed"]))
    model, _ = build_model(priors, device); configure_trainability(model, 1); model.train()
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = named_tensor_hash(initial_state.items()); rng = capture_rng()
    expected_hash = registration["initial_model"]["state_sha256"]
    if initial_hash != expected_hash:
        raise RuntimeError(f"calibration initial hash drift {initial_hash} != {expected_hash}")
    norms = {name: [] for name in ("D", "G", "S", "A")}
    p2_fractions = []
    for registered in registration["calibration_batches"]:
        group_sq = {name: 0.0 for name in norms}
        indices = registered["indices"]
        chunks = [indices[index:index + physical] for index in range(0, len(indices), physical)]
        for indices_chunk in chunks:
            batch = collate([dataset[index] for index in indices_chunk])
            _total, parts, audit, outputs = compute_loss_groups(
                model, batch, {"G": 1, "S": 1, "A": 1}, use_amp=False)
            for name in norms:
                gradient = torch.autograd.grad(parts[name], outputs["c2"], retain_graph=True, allow_unused=False)[0]
                gradient = gradient.float() / len(chunks)
                group_sq[name] += float(gradient.pow(2).sum())
            p2_fractions.append(audit["assignment"]["p2_loss_fraction"])
            del batch, _total, outputs, parts, audit
        for name in norms:
            value = math.sqrt(group_sq[name])
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"invalid calibration gradient {name}: {value}")
            norms[name].append(value)
    medians = {name: statistics.median(values) for name, values in norms.items()}
    eps = float(config["losses"]["group_calibration"]["eps"])
    low, high = config["losses"]["group_calibration"]["clip"]
    multipliers = {
        "D": 1.0,
        "G": min(high, max(low, 0.50 * medians["D"] / max(medians["G"], eps))),
        "S": min(high, max(low, 0.25 * medians["D"] / max(medians["S"], eps))),
        "A": min(high, max(low, 0.10 * medians["D"] / max(medians["A"], eps))),
    }
    model.load_state_dict(initial_state, strict=True); restore_rng(rng)
    restored_hash = named_tensor_hash(model.state_dict().items())
    if restored_hash != initial_hash:
        raise RuntimeError("model state not restored after loss calibration")
    return {"schema": "splitfusion_fcos_loss_group_calibration_v1", "created_utc": utc_now(),
            "calibration_batch_hash": registration["calibration_batches_sha256"],
            "physical_batch": physical, "microbatches_per_effective_batch": 16 // physical,
            "norms": norms, "medians": medians, "multipliers": multipliers,
            "state_before_sha256": initial_hash, "state_after_restore_sha256": restored_hash,
            "rng_restored": True, "optimizer_steps": 0, "validation_accessed": False,
            "p2_loss_fractions_over_microbatches": p2_fractions}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "PREREGISTRATION_COMPLETE").is_file(): raise RuntimeError("Phase A incomplete")
    config = load_json(CONFIG_PATH); registration = load_json(experiment / "SCIENTIFIC_REGISTRATION.json")
    current_source_files = package_hashes()
    current_source_hash = canonical_hash(current_source_files)
    if (current_source_files != registration["source_state"]["files"]
            or current_source_hash != registration["source_state"]["canonical_sha256"]):
        raise RuntimeError("source tree differs from Phase A registration; qualification forbidden")
    priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json"); seed = int(config["scientific_seed"])
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda:0"); total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, (MAX_MIB * 2**20) / total_memory), device)
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    train_rows = load_split_rows(dataset_root, "train")
    cache = DepthCache((ROOT / config["train_depth_cache"]).resolve(strict=True), train_rows)
    dataset = RouteBDataset(dataset_root, "train", seed, cache, augment=False)
    person_index = next(index for index, row in enumerate(dataset.rows)
                        if any(item["label"] == "person" for item in dataset.objects.get(row["sample_id"], ())))
    fixed_indices = [person_index, 0]
    seed_everything(seed); model, transfer = build_model(priors, device); model.eval()
    initial_hash = named_tensor_hash(model.state_dict().items())
    if initial_hash != registration["initial_model"]["state_sha256"]: raise RuntimeError("registered launch-state drift")
    parity = official_parity(model, collate([dataset[person_index]]), device)
    with torch.inference_mode():
        one = collate([dataset[person_index]])
        one_input = one["input"].to(device)
        monolithic = model(one_input, dense=True)
        c2 = model.encode_front(one_input); payload = model.transport_encode(c2); c2_hat = model.transport_decode(payload)
        split = model.decode_tail(c2_hat, dense=True)
    split_parity = {name: close(monolithic[name], split[name]) for name in
                    ("semantic_logits", "dense_depth_log1p", "semantic_logits_stride4", "dense_depth_log1p_stride4")}
    for family in ("cls_logits", "bbox_regression", "bbox_ctrness"):
        split_parity[f"detection_{family}"] = close(
            monolithic["detection"][family], split["detection"][family])
    for level_index, level_name in enumerate(LEVELS):
        for field in ("depth_bin_logits", "depth_bin_residuals", "physical_ray", "log_dimensions", "yaw"):
            split_parity[f"geometry_{level_name}_{field}"] = close(
                monolithic["geometry"][level_index][field], split["geometry"][level_index][field])
    for name, value in split_parity.items(): require_allclose(value, name)
    split_parity["c2_exact"] = bool(torch.equal(c2, payload) and torch.equal(payload, c2_hat))
    split_parity["same_storage_identity"] = c2.data_ptr() == payload.data_ptr() == c2_hat.data_ptr()
    split_parity["raw_bytes"] = c2[0].numel() * c2[0].element_size()
    if split_parity["raw_bytes"] != 22020096: raise RuntimeError("payload byte drift")
    calibration_map = {"intrinsic": one["targets"][0]["intrinsic"].to(device),
                       "extrinsic": one["targets"][0]["extrinsic"].to(device)}
    monolithic_selected = model.postprocess(monolithic, [calibration_map])[0]
    split_selected = model.postprocess(split, [calibration_map])[0]
    split_parity["candidate_identity_exact"] = bool(torch.equal(
        monolithic_selected["candidate_identity"], split_selected["candidate_identity"]))
    for field in ("boxes", "scores", "local_xyz", "world_xyz", "dimensions", "yaw", "physical_uv", "depth"):
        split_parity[f"selected_{field}"] = close(monolithic_selected[field], split_selected[field])
    split_parity["selected_outputs_finite"] = finite_tree(split_selected)
    lineage = lineage_test(model, split, calibration_map)
    real_lineage = real_lineage_test(model, split, calibration_map)
    roundtrips = roundtrip_tests(one["targets"][0])
    padding = {"semantic_output_shape": list(split["semantic_logits"].shape),
               "dense_output_shape": list(split["dense_depth_log1p"].shape),
               "semantic_raw_shape": list(split["semantic_logits_stride4"].shape),
               "dense_raw_shape": list(split["dense_depth_log1p_stride4"].shape),
               "padded_semantic_rows_excluded": split["semantic_logits"].shape[-2] == 432,
               "padded_dense_rows_excluded": split["dense_depth_log1p"].shape[-2] == 432,
               "network_input_padding_exact_zero": int(torch.count_nonzero(one_input[..., 432:, :])) == 0}
    physical, probes = batch_probe(config, priors, dataset, device)
    loss_calibration = calibrate(config, registration, priors, dataset, 1, device)
    seed_everything(seed); model, transfer = build_model(priors, device); configure_trainability(model, 4); model.train()
    optimizer = build_optimizer(model, config["training"]["base_lrs"])
    probe_indices = list(dict.fromkeys([person_index, *range(physical)]))[:physical]
    probe_batch = collate([dataset[index] for index in probe_indices])
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    total, parts, audit, outputs = compute_loss_groups(model, probe_batch, {"G": 1, "S": 1, "A": 1})
    total.backward(); grad = gradients(model); evidence = group_gradient_evidence(grad)
    if not all(value["finite"] and value["nonzero"] for value in evidence.values()):
        raise RuntimeError(f"required gradient qualification failure: {evidence}")
    optimizer.step()
    actual_joint_peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    if actual_joint_peak_mib > MAX_MIB:
        raise RuntimeError(f"qualified joint optimizer step exceeded 12 GiB: {actual_joint_peak_mib}")
    finite = {"individual_losses": finite_tree(parts), "total": bool(torch.isfinite(total)),
              "gradients": all(value["finite"] for value in grad.values()),
              "parameters": all(bool(torch.isfinite(value).all()) for value in model.parameters()),
              "optimizer_state": optimizer_finite(optimizer)}
    qualified_loss_components = scalar_components(parts); qualified_loss_audit = audit
    optimizer.zero_grad(set_to_none=True)
    del probe_batch, total, parts, audit, outputs, optimizer
    torch.cuda.empty_cache()
    warm_allowlist = configure_trainability(model, 1); joint_allowlist = configure_trainability(model, 4)
    norm_state = {"frozen_batchnorm_modules": sum(module.__class__.__name__ == "FrozenBatchNorm2d" for module in model.modules()),
                  "groupnorm_modules": sum(isinstance(module, torch.nn.GroupNorm) for module in model.modules()),
                  "batchnorm_modules": sum(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules())}
    model.eval(); latency = measure_latency(model, one_input)
    edge_source = inspect.getsource(SplitFusionFCOS.decode_tail)
    inference_source = inspect.getsource(InferenceDataset)
    edge_isolation = {"signature": str(inspect.signature(SplitFusionFCOS.decode_tail)),
                      "only_learned_input": "c2_hat", "forbidden_parameters_absent": all(
                          name not in str(inspect.signature(SplitFusionFCOS.decode_tail)) for name in
                          ("rgb", "radar", "depth_gt", "semantic_gt")),
                      "generalized_rcnn_transform_modules": sum(module.__class__.__name__ == "GeneralizedRCNNTransform" for module in model.modules()),
                      "source_sha256": tensor_hash(torch.tensor(list(edge_source.encode()), dtype=torch.uint8)),
                      "inference_dataset_source_sha256": tensor_hash(torch.tensor(list(inference_source.encode()), dtype=torch.uint8)),
                      "inference_dataset_has_no_target_loader": all(token not in inference_source for token in
                          ("RouteBDataset", "load_objects", "DepthCache", "_mask("))}
    atomic_json(experiment / "LOSS_CALIBRATION.json", loss_calibration, overwrite=False)
    runtime = {"schema": "splitfusion_fcos_qualified_runtime_v1", "created_utc": utc_now(),
               "physical_batch": physical, "gradient_accumulation": 16 // physical, "effective_batch": 16,
               "precision": "BF16 tail; FP32 front boundary and sensitive losses/geometry/transport",
               "max_allocated_cap_mib": MAX_MIB, "probe_attempts": probes,
               "loss_calibration_physical_batch": 1,
               "qualified_joint_optimizer_step_after_calibration": True,
               "qualified_joint_optimizer_step_peak_allocated_mib": actual_joint_peak_mib}
    atomic_json(experiment / "QUALIFIED_RUNTIME.json", runtime, overwrite=False)
    report = {
        "schema": "splitfusion_fcos_structural_qualification_v1", "created_utc": utc_now(),
        "initial_state_sha256": initial_hash, "transfer": transfer, "official_parity": parity,
        "padding": padding, "roundtrips": roundtrips, "synthetic_lineage": lineage,
        "real_train_lineage": real_lineage, "split_parity": split_parity, "losses": qualified_loss_components,
        "loss_audit": qualified_loss_audit, "finite": finite, "gradient_evidence": evidence,
        "warmup_allowlist": warm_allowlist, "joint_allowlist": joint_allowlist,
        "normalization_state": norm_state, "bf16_qualified": all(finite.values()),
        "runtime": runtime, "latency": latency, "edge_isolation": edge_isolation,
        "parameter_inventory": parameter_inventory(model), "validation_accessed": False,
        "pass": all(finite.values()) and all(value["allclose"] for value in parity["parity"].values())
                and split_parity["c2_exact"] and edge_isolation["forbidden_parameters_absent"]
                and split_parity["candidate_identity_exact"] and split_parity["selected_outputs_finite"]
                and edge_isolation["inference_dataset_has_no_target_loader"]
                and actual_joint_peak_mib <= MAX_MIB
                and edge_isolation["generalized_rcnn_transform_modules"] == 0,
    }
    if not report["pass"]: raise RuntimeError("structural qualification failed")
    atomic_json(experiment / "STRUCTURAL_QUALIFICATION.json", report, overwrite=False)
    atomic_json(experiment / "STATUS.json", {"phase": "B", "state": "structural_and_calibration_complete",
                                              "created_utc": utc_now(), "optimizer_steps": 0,
                                              "validation_accessed": False})
    atomic_text(experiment / "STRUCTURAL_QUALIFICATION_COMPLETE", "STRUCTURAL_NUMERICAL_AND_CALIBRATION_QUALIFIED\n", overwrite=False)
    atomic_json(experiment / "NOTIFICATION_QUALIFICATION_STRUCTURE.json", desktop_notify(
        "SplitFusion FCOS", f"Structural qualification complete; physical batch {physical}, accumulation {16 // physical}."), overwrite=False)
    print(json.dumps({"pass": report["pass"], "physical_batch": physical,
                      "multipliers": loss_calibration["multipliers"], "latency": latency}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
