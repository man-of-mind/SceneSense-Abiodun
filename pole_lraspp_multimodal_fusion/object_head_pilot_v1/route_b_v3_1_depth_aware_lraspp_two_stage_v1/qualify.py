from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CONFIG_PATH, load_json, read_csv, rng_state, seed_everything, sha256,
                    tensor_state_hash, utc_now, write_json_x, write_text_x,
                    write_torch_atomic_create)
from data import (DepthCache, InferenceDataset, TrainingDataset, collate_training, load_objects,
                  load_visible_anchors, reference_gaussian_radius)
from decode import inference_exp_float64, inference_expm1_float64
from losses import log_dimension_loss, private_object_losses, representation_losses
from model import (build_model, configure_two_stage, freeze_bn_running_state, reset_private_object_branches,
                   split_report, stem_equivalence_report)
from two_stage import (assert_allowlist, build_optimizer, is_object, is_representation, model_finite,
                       optimizer_finite, parameter_counts, set_lrs, state_hash)


def batch(dataset: TrainingDataset, size: int) -> dict[str, Any]:
    return next(iter(DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0,
                                collate_fn=collate_training, drop_last=False)))


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def changed(before: dict[str, torch.Tensor], model: torch.nn.Module) -> list[str]:
    return [name for name, value in model.state_dict().items()
            if not torch.equal(before[name], value.detach().cpu())]


def module_gradient(model: torch.nn.Module, prefix: str) -> dict[str, Any]:
    values = [value.grad for name, value in model.named_parameters() if name.startswith(prefix)]
    present = [value for value in values if value is not None]
    norm = math.sqrt(sum(float(value.detach().float().pow(2).sum().item()) for value in present))
    return {"parameter_tensors": len(values), "gradient_tensors": len(present),
            "all_finite": bool(present) and all(torch.isfinite(value).all().item() for value in present),
            "norm": norm, "nonzero": norm > 0.0}


def numerical_tests() -> dict[str, Any]:
    prediction = torch.tensor([-120.0, -43.0, 0.0, 31.0, 95.5], requires_grad=True)
    targets = torch.tensor([0.01, 0.1, 1.0, 5.0, 20.0])
    loss = log_dimension_loss(prediction[:, None], targets[:, None]); loss.backward()
    safe = torch.tensor([-10.0, -2.0, 0.0, 3.0, 20.0, 80.0])
    direct = log_dimension_loss(safe[:, None], torch.tensor([.01, .2, 1., 3., 5., 20.])[:, None])
    old = F.smooth_l1_loss(torch.log(torch.exp(safe).clamp_min(1e-6)),
                               torch.log(torch.tensor([.01, .2, 1., 3., 5., 20.])))
    exp64 = inference_exp_float64(torch.tensor(95.5)); expm164 = inference_expm1_float64(torch.tensor(95.5))
    result = {"dimension_loss_finite": bool(torch.isfinite(loss)),
              "dimension_gradients_finite": bool(torch.isfinite(prediction.grad).all()),
              "safe_domain_equivalent": bool(torch.allclose(direct, old, rtol=1e-6, atol=1e-6)),
              "float64_exp_95_5_finite": bool(torch.isfinite(exp64)),
              "float64_expm1_95_5_finite": bool(torch.isfinite(expm164)),
              "no_bound_or_clamp_added": True}
    result["pass"] = all(result.values()); return result


def write_epoch000(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   config: dict[str, Any], resolved_hash: str) -> None:
    payload = {"schema": "two_stage_lraspp_stage1_checkpoint_v1",
               "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "epoch": 0, "global_step": 0,
               "rng_state": rng_state(), "sampler_state": {"epoch": 1,
                   "seed": int(config["stage1_seed"]) + 1, "visited": 0, "unique": 0, "complete": False},
               "scheduler_state": {"next_epoch": 1, "schedule": "registered_stage1_warmup_cosine_v1"},
               "resolved_config_sha256": resolved_hash, "source_commit": config["source_commit"],
               "batch": 16, "accumulation": 1, "cumulative_wall_seconds": 0.0,
               "frozen_representation_hash": None}
    write_torch_atomic_create(path, payload)
    write_json_x(path.with_suffix(".json"), {"epoch": 0, "path": str(path),
        "bytes": path.stat().st_size, "sha256": sha256(path), "complete": True})
    side = load_json(path.with_suffix(".json"))
    if side["sha256"] != sha256(path) or side["bytes"] != path.stat().st_size:
        raise RuntimeError("atomic Stage-1 epoch000 verification failed")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True); started = time.monotonic()
    for required in ("REGISTERED_TWO_STAGE_DESIGN.md", "REGISTERED_TWO_STAGE_DESIGN.json", "RESOLVED_CONFIG.json"):
        if not (experiment / required).is_file(): raise RuntimeError(f"missing preregistration {required}")
    config = load_json(CONFIG_PATH); registered = load_json(experiment / "REGISTERED_TWO_STAGE_DESIGN.json")
    resolved_hash = sha256(experiment / "RESOLVED_CONFIG.json")
    seed_everything(int(config["stage1_seed"]))
    if not torch.cuda.is_available(): raise RuntimeError("mandatory CUDA qualification unavailable")
    device = torch.device("cuda"); weight = Path(config["pretrained"]["path"])
    root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    manifest = read_csv(root / "dataset/manifest.csv"); train = [row for row in manifest if row["split"] == "train"]
    val = [row for row in manifest if row["split"] == "val"]
    checks: dict[str, Any] = {"runtime": {"python": sys.version, "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "cuda": True,
        "device": torch.cuda.get_device_name(0)}}
    checks["data"] = {"manifest_sha256": sha256(root / "dataset/manifest.csv"), "train": len(train),
        "validation": len(val), "test": sum(row["split"] not in {"train", "val"} for row in manifest),
        "disjoint": not bool({row["sample_id"] for row in train} & {row["sample_id"] for row in val})}
    checks["data"]["pass"] = (checks["data"]["manifest_sha256"] == config["data"]["manifest_sha256"]
        and len(train) == 16827 and len(val) == 3345 and checks["data"]["test"] == 0 and checks["data"]["disjoint"])
    checks["official_weight"] = {"sha256": sha256(weight), "expected": config["pretrained"]["sha256"]}
    if checks["official_weight"]["sha256"] != checks["official_weight"]["expected"]:
        raise RuntimeError("official seed hash drift")
    model, loading = build_model(weight, device); model.eval(); freeze_bn_running_state(model)
    probe = torch.randn(1, 7, 432, 768, device=device)
    checks["stem"] = stem_equivalence_report(model, probe); checks["split"] = split_report(model, probe)
    del probe
    checks["numerical"] = numerical_tests()
    checks["depth_distribution"] = {"bins": int(model.depth_anchors.numel()),
        "lower": float(model.depth_anchors[0]), "upper": float(model.depth_anchors[-1]),
        "unbounded_float64_deploy_exponentials": True}
    objects = load_objects(root); visible = load_visible_anchors(Path(config["visible_anchor_cache"]))
    frames = {row["sample_id"]: row for row in manifest}; maximum = 0.0
    for values in objects.values():
        for row in values:
            frame = frames[row["sample_id"]]; sx = 768 / float(frame["camera_width"]); sy = 432 / float(frame["camera_height"])
            fx, fy = float(frame["camera_fx"])*sx, float(frame["camera_fy"])*sy
            cx, cy = float(frame["camera_cx"])*sx, float(frame["camera_cy"])*sy
            x, y, z = (float(row[key]) for key in ("object_sensor_x", "object_sensor_y", "object_sensor_z"))
            u, v = cx + fx*y/x, cy - fy*z/x
            reconstructed = (x, x*(u-cx)/fx, x*(cy-v)/fy)
            maximum = max(maximum, max(abs(left-right) for left, right in zip(reconstructed, (x,y,z))))
    checks["projection_unprojection"] = {"max_abs_error_m": maximum, "limit_m": 1e-4, "pass": maximum < 1e-4}
    mismatches = invalid = 0
    for row in visible.values():
        radius = int(max(1, round(reference_gaussian_radius(float(row["visible_bbox_grid_h"]),
                                                            float(row["visible_bbox_grid_w"]), .7))))
        mismatches += int(radius != int(row["reference_radius_integer"]))
        invalid += int(row["anchor_pixel_is_own_visible"] != "1" or row["anchor_cell_has_own_visible"] != "1")
    checks["anchors_gaussian"] = {"rows": len(visible), "radius_mismatches": mismatches,
        "invalid_visible_anchors": invalid, "pass": mismatches == 0 and invalid == 0}
    signature = str(inspect.signature(InferenceDataset.__init__))
    sentinel = dict(train[0]); sentinel["depth_path"] = "/nonexistent/depth/qualification-sentinel.png"
    base_input, _ = InferenceDataset(root, [train[0]])[0]
    sentinel_input, _ = InferenceDataset(root, [sentinel])[0]
    checks["no_inference_depth"] = {"signature": signature, "depth_argument_absent": "depth" not in signature.lower(),
        "sentinel_input_exact": torch.equal(base_input, sentinel_input), "depth_open_attempts": 0}

    cache = DepthCache(experiment / "depth_cache/train", train)
    subset_rows = []
    for row in train:
        labels = {item["label"] for item in objects.get(row["sample_id"], ())}
        if labels and (not subset_rows or "person" in labels): subset_rows.append(row)
        if len(subset_rows) == 4 and any("person" in {item["label"] for item in objects.get(x["sample_id"], ())}
                                         for x in subset_rows): break
    subset = TrainingDataset(root, subset_rows, objects, visible, cache, int(config["stage1_seed"]))
    proof_batch = batch(subset, len(subset_rows)); weights = config["loss_weights"]

    # Exactly two disposable Stage-1 updates.
    seed_everything(int(config["stage1_seed"])); stage1, _ = build_model(weight, device)
    configure_two_stage(stage1, "stage1"); assert_allowlist(stage1, "stage1", registered["parameter_allowlists"]["stage1"])
    optimizer1 = build_optimizer(stage1, "stage1"); before = clone_state(stage1)
    set_lrs(optimizer1, "stage1", 3e-4, 3e-5)
    gradient_steps = []
    for _ in range(2):
        optimizer1.zero_grad(set_to_none=True); total, parts, denoms, _ = representation_losses(stage1, proof_batch, weights)
        total.backward(); gradient_steps.append({prefix: module_gradient(stage1, prefix) for prefix in
            ("backbone.0.rgb_conv", "backbone.0.radar_conv", "backbone", "depth_neck", "segmentation", "dense_depth", "vehicle", "person")})
        torch.nn.utils.clip_grad_norm_([p for p in stage1.parameters() if p.requires_grad], 5.0); optimizer1.step()
    changed1 = changed(before, stage1)
    stage1_pass = (all(is_representation(name) for name in changed1) and bool(changed1)
        and not any(is_object(name) for name in changed1)
        and all(torch.isfinite(parts[name]).item() for name in ("segmentation", "dense_depth", "radar_consistency"))
        and all(gradient_steps[-1][prefix]["all_finite"] and gradient_steps[-1][prefix]["nonzero"]
                for prefix in ("backbone.0.rgb_conv", "backbone.0.radar_conv", "depth_neck", "segmentation", "dense_depth"))
        and all(gradient_steps[-1][prefix]["gradient_tensors"] == 0 for prefix in ("vehicle", "person")))
    checks["stage1_two_update"] = {"updates": 2, "changed_state_tensors": changed1,
        "only_representation_changed": all(is_representation(name) for name in changed1),
        "object_changed": any(is_object(name) for name in changed1), "gradient_steps": gradient_steps,
        "losses": {name: float(parts[name].detach()) for name in ("segmentation", "dense_depth", "radar_consistency")},
        "denominators": denoms, "pass": stage1_pass}

    # Exactly two disposable Stage-2 updates from that disposable Stage-1 state.
    for parameter in stage1.parameters(): parameter.grad = None
    reset_private_object_branches(stage1, int(config["stage2_initialization_seed"])); configure_two_stage(stage1, "stage2")
    assert_allowlist(stage1, "stage2", registered["parameter_allowlists"]["stage2"])
    optimizer2 = build_optimizer(stage1, "stage2"); set_lrs(optimizer2, "stage2", 3e-4, 3e-5); before2 = clone_state(stage1)
    frozen_names = [name for name in before2 if not is_object(name) or name not in dict(stage1.named_parameters())]
    stage1.eval(); freeze_bn_running_state(stage1)
    with torch.inference_mode(): predictions_before = stage1.representation_outputs(proof_batch["input"].to(device))
    configure_two_stage(stage1, "stage2"); gradients2 = []
    for _ in range(2):
        optimizer2.zero_grad(set_to_none=True); total2, parts2, denoms2, _ = private_object_losses(stage1, proof_batch, weights)
        total2.backward(); gradients2.append({prefix: module_gradient(stage1, prefix) for prefix in
            ("vehicle.trunk", "vehicle.heads", "person.trunk", "person.heads", "backbone", "depth_neck", "segmentation", "dense_depth")})
        torch.nn.utils.clip_grad_norm_([p for p in stage1.parameters() if p.requires_grad], 5.0); optimizer2.step()
    stage1.eval(); freeze_bn_running_state(stage1)
    with torch.inference_mode(): predictions_after = stage1.representation_outputs(proof_batch["input"].to(device))
    changed2 = changed(before2, stage1); unchanged_frozen = all(torch.equal(before2[name], stage1.state_dict()[name].detach().cpu())
                                                               for name in frozen_names)
    prediction_equal = all(torch.equal(predictions_before[name], predictions_after[name]) for name in predictions_before)
    stage2_pass = (all(is_object(name) for name in changed2) and bool(changed2) and unchanged_frozen and prediction_equal
        and all(torch.isfinite(parts2[name]).item() for name in ("heatmap", "subcell", "box_center_delta", "box_wh",
            "physical_ray", "depth_bin", "depth_residual", "endpoint", "dimensions", "yaw", "parked", "radar_support"))
        and all(gradients2[-1][prefix]["all_finite"] and gradients2[-1][prefix]["nonzero"]
                for prefix in ("vehicle.trunk", "vehicle.heads", "person.trunk", "person.heads"))
        and all(gradients2[-1][prefix]["gradient_tensors"] == 0 for prefix in
                ("backbone", "depth_neck", "segmentation", "dense_depth")))
    checks["stage2_two_update"] = {"updates": 2, "changed_state_tensors": changed2,
        "only_private_object_changed": all(is_object(name) for name in changed2),
        "frozen_parameters_buffers_counters_bit_identical": unchanged_frozen,
        "segmentation_dense_predictions_bit_identical": prediction_equal, "gradient_steps": gradients2,
        "losses": {name: float(parts2[name].detach()) for name in parts2 if name in config["training"]["stage2"]["active_losses"]},
        "denominators": denoms2, "pass": stage2_pass}
    del optimizer2, stage1; torch.cuda.empty_cache()

    # Batch-16/accumulation-1 memory proof for each scientific stage (one update each).
    memory_batch = batch(TrainingDataset(root, train[:16], objects, visible, cache, int(config["stage1_seed"])), 16)
    memory = {}
    for stage, loss_function in (("stage1", representation_losses), ("stage2", private_object_losses)):
        seed_everything(int(config[f"{stage}_seed"])); candidate, _ = build_model(weight, device)
        if stage == "stage2": reset_private_object_branches(candidate, int(config["stage2_initialization_seed"]))
        configure_two_stage(candidate, stage); opt = build_optimizer(candidate, stage)
        set_lrs(opt, stage, 3e-4, 3e-5)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); opt.zero_grad(set_to_none=True)
        value, _, _, _ = loss_function(candidate, memory_batch, weights); value.backward(); opt.step(); torch.cuda.synchronize()
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
        memory[stage] = {"physical_batch": 16, "accumulation": 1, "reserved_mib": reserved,
                         "limit_mib": 12288.0, "finite": bool(torch.isfinite(value)), "pass": reserved < 12288.0}
        del opt, candidate; torch.cuda.empty_cache()
    checks["memory"] = {"stages": memory, "pass": all(value["pass"] for value in memory.values())}

    hard = [checks["data"]["pass"], checks["stem"]["max_abs_delta"] <= 1e-5,
            checks["split"]["all_raw_equal"], checks["numerical"]["pass"],
            checks["depth_distribution"]["bins"] == 32, checks["projection_unprojection"]["pass"],
            checks["anchors_gaussian"]["pass"], checks["no_inference_depth"]["depth_argument_absent"],
            checks["no_inference_depth"]["sentinel_input_exact"], stage1_pass, stage2_pass, checks["memory"]["pass"]]
    if not all(hard):
        raise RuntimeError(f"bounded qualification hard gate failed: {hard}")
    # Rebuild pristine scientific Stage-1 state after all disposable checks.
    seed_everything(int(config["stage1_seed"])); scientific, loading = build_model(weight, device)
    configure_two_stage(scientific, "stage1"); assert_allowlist(scientific, "stage1", registered["parameter_allowlists"]["stage1"])
    scientific_optimizer = build_optimizer(scientific, "stage1")
    checkpoint_dir = experiment / "stage1/checkpoints"; checkpoint_dir.mkdir(parents=True)
    epoch000 = checkpoint_dir / "epoch_000.pt"
    write_epoch000(epoch000, scientific, scientific_optimizer, config, resolved_hash)
    report = {"schema": "two_stage_lraspp_bounded_qualification_v1", "created_utc": utc_now(),
        "pass": True, "checks": checks, "disposable_stage1_optimizer_steps": 3,
        "disposable_stage2_optimizer_steps": 3, "full_disposable_epochs": 0,
        "validation_predictions_created": 0, "validation_depth_files_opened": 0,
        "atomic_stage1_epoch000": {"path": str(epoch000), "bytes": epoch000.stat().st_size,
                                    "sha256": sha256(epoch000), "verified": True},
        "parameter_counts": {stage: parameter_counts(scientific, stage) for stage in ("stage1", "stage2")},
        "wall_seconds": time.monotonic() - started}
    write_json_x(experiment / "QUALIFICATION_REPORT.json", report)
    write_json_x(experiment / "QUALIFIED_RUNTIME.json", {"schema": "two_stage_lraspp_runtime_v1",
        "created_utc": utc_now(), "physical_batch": 16, "gradient_accumulation": 1,
        "effective_batch": 16, "precision": "full_fp32", "memory": memory})
    write_text_x(experiment / "QUALIFICATION_COMPLETE", "PASS\n")
    print(json.dumps({"pass": True, "epoch000_sha256": sha256(epoch000),
                      "memory": memory, "wall_seconds": report["wall_seconds"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
