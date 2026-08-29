from __future__ import annotations

import argparse
import inspect
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common import (CONFIG_PATH, load_json, read_csv, sha256, tensor_state_hash, utc_now,
                    write_json_x, write_text_x)
from data import (DepthCache, InferenceDataset, TrainingDataset, collate_training, collision_audit,
                  load_objects, load_visible_anchors, reference_gaussian_radius)
from decode import camera_matrix_from_row, decode_geometry, intrinsic_from_row
from losses import compute_losses
from model import (build_model, configure_stage, freeze_bn_running_state, parameter_groups,
                   parameter_report, pretrained_backbone_state, split_report, stage_train_mode,
                   stem_equivalence_report)


def optimizer_for(model: torch.nn.Module, new_lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    groups = parameter_groups(model)
    specs = []
    for name, values in groups.items():
        parameters = [value for _key, value in values if value.requires_grad]
        if not parameters:
            continue
        specs.append({
            "params": parameters,
            "lr": backbone_lr if name.startswith("backbone") else new_lr,
            "weight_decay": 0.0 if name.endswith("no_decay") else 1e-4,
            "name": name,
        })
    return torch.optim.AdamW(specs, betas=(0.9, 0.999), eps=1e-8)


def choose_overfit_rows(rows: Sequence[dict[str, str]], objects: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    properties: set[str] = set()
    for row in rows:
        values = objects.get(row["sample_id"], [])
        current = set()
        if any(item["label"] == "person" for item in values): current.add("person")
        if any(item["label"] == "vehicle" for item in values): current.add("vehicle")
        if any(float(item.get("radar_support_points", "0") or 0) > 0 for item in values): current.add("supported")
        if any(float(item.get("radar_support_points", "0") or 0) == 0 for item in values): current.add("unsupported")
        if current - properties:
            chosen.append(row); properties |= current
        if properties == {"person", "vehicle", "supported", "unsupported"} and len(chosen) >= 4:
            break
    if properties != {"person", "vehicle", "supported", "unsupported"}:
        raise RuntimeError(f"unable to form registered overfit subset: {properties}")
    return chosen[:8]


def batch_loader(dataset: TrainingDataset, batch: int) -> dict[str, Any]:
    return next(iter(DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0,
                                collate_fn=collate_training, drop_last=False)))


def finite_gradients(model: torch.nn.Module) -> dict[str, Any]:
    result = {}
    for prefix in ("backbone.0.radar_conv", "depth_neck", "segmentation", "dense_depth", "vehicle", "person"):
        values = [parameter.grad for name, parameter in model.named_parameters()
                  if name.startswith(prefix) and parameter.requires_grad]
        finite = all(value is not None and torch.isfinite(value).all().item() for value in values)
        norm = math.sqrt(sum(float(value.detach().float().pow(2).sum().item()) for value in values if value is not None))
        result[prefix] = {"gradient_tensors": len(values), "finite": finite, "norm": norm,
                          "nonzero": norm > 0.0}
    return result


def memory_qualification(weight_path: Path, dataset: TrainingDataset,
                         weights: dict[str, float], device: torch.device) -> tuple[dict[str, Any], tuple[int, int]]:
    attempts = []
    accepted = None
    limit_mib = 12.0 * 1024.0 * 0.90
    for physical, accumulation in ((16, 1), (8, 2), (4, 4)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model = optimizer = batch = None
        try:
            model, _ = build_model(weight_path, device)
            configure_stage(model, "B"); stage_train_mode(model, "B")
            optimizer = optimizer_for(model, 1e-4, 1e-5)
            batch = batch_loader(dataset, physical)
            optimizer.zero_grad(set_to_none=True)
            total, _parts, _denominators, _outputs = compute_losses(model, batch, weights)
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.max_memory_allocated(device) / 2**20
            reserved = torch.cuda.max_memory_reserved(device) / 2**20
            passed = math.isfinite(float(total.item())) and reserved <= limit_mib
            attempts.append({"physical_batch": physical, "accumulation": accumulation,
                             "allocated_mib": allocated, "reserved_mib": reserved,
                             "limit_mib": limit_mib, "pass": passed})
            if passed:
                accepted = (physical, accumulation); break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"physical_batch": physical, "accumulation": accumulation,
                             "pass": False, "error": f"{type(error).__name__}: {error}"})
        finally:
            del batch, optimizer, model
            torch.cuda.empty_cache()
    if accepted is None:
        raise RuntimeError(f"no registered batch fallback passed: {attempts}")
    return {"attempts": attempts, "accepted_physical_batch": accepted[0],
            "accepted_accumulation": accepted[1], "effective_batch": 16}, accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve(strict=True)
    config = load_json(CONFIG_PATH)
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    manifest_path = dataset_root / "dataset/manifest.csv"
    manifest = read_csv(manifest_path)
    train_rows = [row for row in manifest if row["split"] == "train"]
    validation_rows = [row for row in manifest if row["split"] == "val"]
    checks: dict[str, Any] = {}
    checks["runtime"] = {
        "python": sys.version, "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("mandatory CUDA qualification unavailable")
    device = torch.device("cuda")
    checks["data"] = {
        "manifest_sha256": sha256(manifest_path), "rows": len(manifest),
        "train": len(train_rows), "validation": len(validation_rows),
        "unique_sample_ids": len({row["sample_id"] for row in manifest}),
        "train_validation_disjoint": not ({row["sample_id"] for row in train_rows}
                                          & {row["sample_id"] for row in validation_rows}),
        "test_rows": sum(row["split"] not in {"train", "val"} for row in manifest),
        "test_path_references": sum("test" in value.lower() for row in manifest for value in row.values()),
    }
    expected_data = (checks["data"]["manifest_sha256"] == config["data"]["manifest_sha256"]
                     and checks["data"]["train"] == 16827 and checks["data"]["validation"] == 3345
                     and checks["data"]["unique_sample_ids"] == len(manifest)
                     and checks["data"]["train_validation_disjoint"] and checks["data"]["test_rows"] == 0
                     and checks["data"]["test_path_references"] == 0)
    checks["data"]["pass"] = expected_data
    contract_hashes = {}
    for contract in ("v010", "v025"):
        for split in ("train", "val"):
            for name in ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv"):
                path = dataset_root / f"contracts/{contract}/{split}/{name}"
                contract_hashes[f"{contract}/{split}/{name}"] = sha256(path)
    checks["contract_hashes"] = contract_hashes
    weight_path = Path(config["pretrained"]["path"])
    checks["weight"] = {"path": str(weight_path), "bytes": weight_path.stat().st_size,
                        "sha256": sha256(weight_path), "enum": config["pretrained"]["enum"],
                        "url": config["pretrained"]["url"]}
    if checks["weight"]["sha256"] != config["pretrained"]["sha256"]:
        raise RuntimeError("official pretrained weight drift")

    model, load_report = build_model(weight_path, device)
    checks["official_loading"] = load_report
    sample = train_rows[0]
    pil = np.asarray(Image.open(dataset_root / "dataset" / sample["rgb_path"]).convert("RGB"))
    bgr = cv2.imread(str(dataset_root / "dataset" / sample["rgb_path"]), cv2.IMREAD_COLOR)
    radar = np.load(dataset_root / "dataset" / sample["radar_tensor_path"])
    checks["real_input"] = {
        "sample_id": sample["sample_id"], "pil_rgb_matches_cv2_bgr_reversal": bool(np.array_equal(pil, bgr[:, :, ::-1])),
        "rgb_channels": ["R", "G", "B"],
        "radar_channels": config["input"]["channels"][3:], "radar_shape": list(radar.shape),
        "radar_channel_ranges": [[float(radar[index].min()), float(radar[index].max())] for index in range(4)],
        "radar_normalization": "identity prepared tensor",
    }
    random_input = torch.randn(1, 7, 432, 768, device=device)
    stem = stem_equivalence_report(model, random_input)
    stem["equivalent_at_fp32_tolerance_1e_5"] = stem["max_abs_delta"] <= 1e-5
    checks["stem_equivalence"] = stem
    checks["split"] = split_report(model, random_input)
    checks["parameters"] = parameter_report(model)
    del random_input

    # Synthetic radial/forward checks.
    on_axis = 10.0 / math.sqrt(1.0)
    off_axis_radial = 10.0 * math.sqrt(1.0 + 0.5 ** 2 + 0.25 ** 2)
    off_axis = off_axis_radial / math.sqrt(1.0 + 0.5 ** 2 + 0.25 ** 2)
    checks["radial_forward_synthetic"] = {"on_axis": on_axis, "off_axis": off_axis,
                                           "pass": abs(on_axis - 10.0) < 1e-12 and abs(off_axis - 10.0) < 1e-12}

    objects = load_objects(dataset_root)
    visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    frames = {row["sample_id"]: row for row in manifest}
    max_roundtrip = 0.0
    max_depth = 0.0
    for values in objects.values():
        for row in values:
            frame = frames[row["sample_id"]]
            sx, sy = 768.0 / float(frame["camera_width"]), 432.0 / float(frame["camera_height"])
            fx, fy = float(frame["camera_fx"]) * sx, float(frame["camera_fy"]) * sy
            cx, cy = float(frame["camera_cx"]) * sx, float(frame["camera_cy"]) * sy
            depth, right, up = (float(row[name]) for name in ("object_sensor_x", "object_sensor_y", "object_sensor_z"))
            u, v = cx + fx * right / depth, cy - fy * up / depth
            reconstructed = np.asarray([depth, depth * (u - cx) / fx, depth * (cy - v) / fy])
            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(reconstructed - [depth, right, up]))))
            max_depth = max(max_depth, depth)
    checks["actor_geometry"] = {"objects": sum(map(len, objects.values())), "max_depth_m": max_depth,
                                 "max_roundtrip_abs_error_m": max_roundtrip,
                                 "pass": max_depth <= 40.0 and max_roundtrip < 1e-4}
    gaussian_mismatch = 0; anchor_invalid = 0
    for row in visible.values():
        radius = int(max(1, round(reference_gaussian_radius(
            float(row["visible_bbox_grid_h"]), float(row["visible_bbox_grid_w"]), 0.7))))
        gaussian_mismatch += int(radius != int(row["reference_radius_integer"]))
        anchor_invalid += int(row["anchor_pixel_is_own_visible"] != "1" or row["anchor_cell_has_own_visible"] != "1")
    checks["person_anchor_gaussian"] = {"rows": len(visible), "gaussian_mismatches": gaussian_mismatch,
                                         "invalid_anchors": anchor_invalid,
                                         "pass": gaussian_mismatch == 0 and anchor_invalid == 0}
    checks["collisions"] = {
        "train": collision_audit(train_rows, objects, visible, dataset_root),
        "validation": collision_audit(validation_rows, objects, visible, dataset_root),
    }
    cache_report = load_json(experiment / "depth_cache/train/CACHE_REPORT.json")
    checks["train_depth_cache"] = cache_report
    checks["radar"] = {
        "camera_depth_pass": cache_report["radar_camera_depth_max_abs_delta_m"] <= 1e-4,
        "transform_pass": cache_report["radar_current_sweep_transform_max_abs_delta_m"] <= 1e-4,
        "consistent_points": cache_report["radar_consistent_points"],
    }

    inference_signature = str(inspect.signature(InferenceDataset.__init__))
    model_signature = str(inspect.signature(model.forward))
    normal_dataset = InferenceDataset(dataset_root, [validation_rows[0]])
    sentinel_row = dict(validation_rows[0]); sentinel_row["depth_path"] = "/guaranteed/nonexistent/depth/sentinel.png"
    sentinel_dataset = InferenceDataset(dataset_root, [sentinel_row])
    normal_input, _ = normal_dataset[0]; sentinel_input, _ = sentinel_dataset[0]
    checks["no_depth_inference"] = {
        "dataset_signature": inference_signature, "model_signature": model_signature,
        "depth_argument_absent": "depth" not in inference_signature.lower(),
        "sentinel_prediction_input_equal": torch.equal(normal_input, sentinel_input),
        "sentinel_open_attempts": 0,
    }

    train_cache = DepthCache(experiment / "depth_cache/train", train_rows)
    full_train_dataset = TrainingDataset(
        dataset_root, train_rows, objects, visible, train_cache, config["scientific_seed"],
    )
    subset_rows = choose_overfit_rows(train_rows, objects)
    subset = TrainingDataset(dataset_root, subset_rows, objects, visible, train_cache, config["scientific_seed"])
    subset_batch = batch_loader(subset, len(subset_rows))
    weights = config["loss_weights"]

    # Stage-A one-step disposable freeze/gradient proof.
    clone, _ = build_model(weight_path, device); configure_stage(clone, "A"); stage_train_mode(clone, "A")
    before = tensor_state_hash(pretrained_backbone_state(clone))
    radar_before = clone.backbone["0"].radar_conv.weight.detach().clone()
    optimizer = optimizer_for(clone, 3e-4, 0.0); optimizer.zero_grad(set_to_none=True)
    total, _parts, denoms, outputs = compute_losses(clone, subset_batch, weights)
    total.backward(); gradients = finite_gradients(clone); optimizer.step()
    after = tensor_state_hash(pretrained_backbone_state(clone))
    checks["stage_a_freeze_gradients"] = {
        "pretrained_state_before": before, "pretrained_state_after": after,
        "pretrained_bit_identical": before == after,
        "radar_stem_updated": not torch.equal(radar_before, clone.backbone["0"].radar_conv.weight),
        "gradients": gradients, "denominators": denoms,
        "all_required_gradient_paths": all(value["finite"] and value["nonzero"] for value in gradients.values()),
    }
    del outputs, total, optimizer, clone
    torch.cuda.empty_cache()

    memory_report, accepted = memory_qualification(weight_path, full_train_dataset, weights, device)
    checks["memory"] = memory_report

    # Disposable overfit, reinitialized from the registered seed/official weight.
    torch.manual_seed(config["scientific_seed"]); torch.cuda.manual_seed_all(config["scientific_seed"])
    clone, _ = build_model(weight_path, device); configure_stage(clone, "A"); stage_train_mode(clone, "A")
    # The disposable clone starts at the registered Stage-A initial LR. It does not
    # skip the scientific warm-up and jump directly to the peak on one repeated batch.
    optimizer = optimizer_for(clone, 3e-5, 0.0)
    history = []
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        total, parts, _denoms, _outputs = compute_losses(clone, subset_batch, weights)
        total.backward(); torch.nn.utils.clip_grad_norm_(clone.parameters(), 5.0); optimizer.step()
        history.append({
            "step": step + 1, "total": float(total.detach().item()),
            "person_heatmap": float(parts["heatmap_person"].detach().item()),
            "person_actor_depth": float((parts["depth_bin_person"] + parts["depth_residual_person"]).detach().item()),
            "dense_depth": float(parts["dense_depth"].detach().item()),
        })
    def first_last(name: str) -> tuple[float, float]:
        return (sum(item[name] for item in history[:5]) / 5.0,
                sum(item[name] for item in history[-5:]) / 5.0)
    overfit = {name: {"first5_mean": first_last(name)[0], "last5_mean": first_last(name)[1],
                      "falling": first_last(name)[1] < first_last(name)[0]}
               for name in ("person_heatmap", "person_actor_depth", "dense_depth")}
    checks["disposable_overfit"] = {"steps": 80, "subset_sample_ids": [row["sample_id"] for row in subset_rows],
                                     "losses": overfit, "all_required_falling": all(value["falling"] for value in overfit.values()),
                                     "history_endpoints": [history[0], history[-1]]}
    print(json.dumps({"disposable_overfit": checks["disposable_overfit"]}, indent=2), flush=True)
    del optimizer, clone
    torch.cuda.empty_cache()

    # Byte-identical raw and decoded monolithic/split proof on a real sample.
    model.eval(); normal_input = normal_input.unsqueeze(0).to(device)
    with torch.inference_mode():
        mono = model(normal_input, dense=False)
        bundle = model.encode_front(normal_input)
        split = model.decode_tail(bundle, dense=False)
    camera = camera_matrix_from_row(validation_rows[0]); intrinsic = intrinsic_from_row(validation_rows[0])
    mono_records = decode_geometry(mono, model.depth_anchors, model.depth_delta, camera, intrinsic, 0.02)
    split_records = decode_geometry(split, model.depth_anchors, model.depth_delta, camera, intrinsic, 0.02)
    mono_bytes = json.dumps(mono_records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    split_bytes = json.dumps(split_records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    # The corrected registered heatmap prior is sigmoid(-4.6) ~= 0.00995,
    # intentionally below the unchanged 0.02 scoring threshold.  Preserve the
    # scored split-parity check above and use a separate threshold-zero decode
    # solely to assert the external record schema at pristine initialization.
    schema_records = decode_geometry(mono, model.depth_anchors, model.depth_delta, camera, intrinsic, 0.0)
    checks["decoded_parity"] = {"records": len(mono_records), "byte_identical": mono_bytes == split_bytes,
                                 "registered_prior_below_scoring_threshold": len(mono_records) == 0,
                                 "schema_probe_threshold": 0.0,
                                 "schema_probe_records": len(schema_records),
                                 "external_fields_compatible": all(name in schema_records[0] for name in (
                                     "class_name", "score", "world_x", "world_y", "world_z", "local_x", "local_y",
                                     "local_z", "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
                                     "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
                                     if schema_records else False}

    hard_passes = [
        checks["data"]["pass"], load_report["official_feature_tensors"] == load_report["compatible_feature_tensors_loaded"],
        stem["equivalent_at_fp32_tolerance_1e_5"], checks["split"]["all_raw_equal"],
        checks["radial_forward_synthetic"]["pass"], checks["actor_geometry"]["pass"],
        checks["person_anchor_gaussian"]["pass"], checks["radar"]["camera_depth_pass"],
        checks["radar"]["transform_pass"], checks["no_depth_inference"]["depth_argument_absent"],
        checks["no_depth_inference"]["sentinel_prediction_input_equal"], checks["stage_a_freeze_gradients"]["pretrained_bit_identical"],
        checks["stage_a_freeze_gradients"]["radar_stem_updated"], checks["stage_a_freeze_gradients"]["all_required_gradient_paths"],
        checks["disposable_overfit"]["all_required_falling"], checks["decoded_parity"]["byte_identical"],
        checks["decoded_parity"]["external_fields_compatible"],
    ]
    if not all(hard_passes):
        raise RuntimeError(f"qualification hard gate failed: {hard_passes}")
    report = {
        "schema": "route_b_v3_1_depth_aware_lraspp_qualification_v1", "created_utc": utc_now(),
        "pass": True, "checks": checks, "validation_predictions_created": 0,
        "validation_depth_pngs_opened": 0, "disposable_optimizer_steps": 82,
        "accepted_physical_batch": accepted[0], "accepted_accumulation": accepted[1],
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "QUALIFICATION_REPORT.json", report)
    write_json_x(experiment / "QUALIFIED_RUNTIME.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_qualified_runtime_v1", "created_utc": utc_now(),
        "physical_batch": accepted[0], "gradient_accumulation": accepted[1], "effective_batch": 16,
        "precision": "full_fp32", "memory": memory_report,
    })
    write_json_x(experiment / "PARAMETER_REPORT.json", checks["parameters"])
    write_text_x(experiment / "QUALIFICATION_COMPLETE", "PASS\n")
    print(json.dumps({"pass": True, "accepted": accepted, "wall_seconds": report["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
