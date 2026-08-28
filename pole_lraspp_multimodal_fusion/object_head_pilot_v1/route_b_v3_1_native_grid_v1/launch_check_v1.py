#!/usr/bin/env python3
"""Minimum pre-run checks for the native stride-4 object head. Exactly six, no more.

  1. py_compile for the new package
  2. parse the config
  3. verify the warm-start SHA
  4. one synthetic target case: native centre and fractional offset mapping
  5. one real v3.1 q=0 AMP forward/backward batch
  6. split-boundary check
"""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for _path in (str(ROOT), str(FUSION_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from losses_v1 import (  # noqa: E402
    native_object_loss, segmentation_loss,
)
from model_v1 import (  # noqa: E402
    NATIVE_GRID, NATIVE_STRIDE, OUTPUT_CHANNELS, bilinear_kernel, build_native_grid_model,
    load_warm_start, parameter_report, split_boundary_report,
)
from targets_v1 import (  # noqa: E402
    NativeGridDataset, build_native_object_targets, downsample_ignore_conservative,
)

EXPECTED_WARM_START_SHA = "88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_nonfinite(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        return [item for key, sub in value.items() for item in find_nonfinite(sub, f"{path}.{key}")]
    if isinstance(value, list):
        return [item for index, sub in enumerate(value) for item in find_nonfinite(sub, f"{path}[{index}]")]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}={value}"]
    return []


def json_safe(value: Any) -> Any:
    """Non-finite floats are recorded as strings so the artifact still writes; they are
    also listed explicitly under nonfinite_fields and gated by the checks."""
    if isinstance(value, dict):
        return {key: json_safe(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [json_safe(sub) for sub in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def branch_gradient(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [p.grad.detach().float() for p in module.parameters()
                 if p.requires_grad and p.grad is not None]
    norm = math.sqrt(sum(float((g * g).sum().item()) for g in gradients)) if gradients else 0.0
    return {
        "gradient_tensors": len(gradients),
        "norm": norm,
        "finite": bool(gradients and all(bool(torch.isfinite(g).all().item()) for g in gradients)),
        "nonzero": norm > 0.0,
    }


def check_synthetic_target() -> dict[str, Any]:
    """One object whose centre deliberately falls at a known sub-cell position."""
    # Source 1280x720 -> model 768x432 (x0.6). Choose a source centre so the model
    # centre is 402.25, 218.75 px: native cell (100, 54), offset (0.5625, 0.6875).
    source_x, source_y = 402.25 / 0.6, 218.75 / 0.6
    obj = {
        "class_index": 1.0, "class_name": "person", "center_x": source_x, "center_y": source_y,
        "bbox_w": 13.0, "bbox_h": 37.0, "area": 481.0,
        "local_x": 12.5, "local_y": -3.25, "local_z": 0.75,
        "world_x": 1.0, "world_y": 2.0, "world_z": 0.5,
        "size_x": 0.8, "size_y": 0.7, "size_z": 1.8,
        "yaw_sin": 0.0, "yaw_cos": 1.0, "parked": 0.0, "radar_support": 1.0,
    }
    targets = build_native_object_targets(
        objects=[obj], original_size=(1280, 720), input_size=(768, 432), max_objects=64,
    )
    heatmap = targets["center_heatmap"].numpy()
    offset = targets["center_offset"].numpy()
    regression = targets["regression"].numpy()
    mask = targets["regression_mask"].numpy()

    model_x, model_y = source_x * 0.6, source_y * 0.6
    expected_cell = (int(model_x // NATIVE_STRIDE), int(model_y // NATIVE_STRIDE))
    expected_offset = (model_x / NATIVE_STRIDE - expected_cell[0], model_y / NATIVE_STRIDE - expected_cell[1])
    cell_x, cell_y = expected_cell
    peaks = np.argwhere(heatmap >= 1.0 - 1e-6)

    # Recovering the centre from (cell + offset) * stride must return the exact input pixel.
    recovered = ((cell_x + float(offset[0, cell_y, cell_x])) * NATIVE_STRIDE,
                 (cell_y + float(offset[1, cell_y, cell_x])) * NATIVE_STRIDE)

    ignore_full = np.zeros((432, 768), dtype=bool)
    ignore_full[8:16, 8:16] = True     # exactly 2x2 native cells, fully covered
    ignore_full[40:43, 40:43] = True   # partial coverage of one native cell
    ignore_cells = downsample_ignore_conservative(ignore_full, NATIVE_STRIDE)

    return {
        "grid_shape": list(heatmap.shape),
        "expected_cell": list(expected_cell),
        "peak_cells": [[int(p[2]), int(p[1])] for p in peaks],
        "peak_is_expected_cell": len(peaks) == 1 and int(peaks[0][1]) == cell_y and int(peaks[0][2]) == cell_x,
        "peak_class_is_person": len(peaks) == 1 and int(peaks[0][0]) == 1,
        "expected_offset": list(expected_offset),
        "target_offset": [float(offset[0, cell_y, cell_x]), float(offset[1, cell_y, cell_x])],
        "offset_exact": all(abs(float(offset[i, cell_y, cell_x]) - expected_offset[i]) < 1e-6 for i in (0, 1)),
        "offset_in_unit_interval": bool((offset >= 0.0).all() and (offset < 1.0).all()),
        "model_pixel_centre": [model_x, model_y],
        "recovered_centre": list(recovered),
        "centre_recovery_exact": all(abs(recovered[i] - (model_x, model_y)[i]) < 1e-4 for i in (0, 1)),
        "regression_written_at_centre_cell": float(mask[0, cell_y, cell_x]) == 1.0,
        "regression_positive_cells": int(mask.sum()),
        "regression_local_xyz_at_centre": [float(v) for v in regression[0:3, cell_y, cell_x]],
        "regression_local_xyz_preserved": bool(
            np.allclose(regression[0:3, cell_y, cell_x], [12.5, -3.25, 0.75])),
        "bbox2d_target_is_input_fraction": bool(
            abs(float(regression[10, cell_y, cell_x]) - 13.0 * 0.6 / 768.0) < 1e-6
            and abs(float(regression[11, cell_y, cell_x]) - 37.0 * 0.6 / 432.0) < 1e-6),
        "ignore_full_pixels": int(ignore_full.sum()),
        "ignore_native_cells": int(ignore_cells.sum()),
        "ignore_downsample_is_conservative": int(ignore_cells.sum()) == 4,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    result: dict[str, Any] = {
        "schema": "route_b_v3_1_native_grid_launch_check_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    # 1. py_compile the new package.
    compiled = compileall.compile_dir(str(PACKAGE_ROOT), quiet=1, force=True)
    result["py_compile"] = {"package": str(PACKAGE_ROOT.relative_to(ROOT)), "ok": bool(compiled)}

    # 2. Parse the config.
    config = load_config(args.config.resolve())
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    object_cfg = config["object_heads"]
    result["config"] = {
        "parsed": True,
        "input_size": list(config["training"]["input_size"]),
        "native_stride": int(object_cfg["native_stride"]),
        "native_grid": list(object_cfg["native_grid"]),
        "stage_schedule": [
            {"stage": s["name"], "epochs": [s["first_epoch"], s["last_epoch"]], "lr": s["lr"],
             "freeze_backbone": s["freeze_backbone"], "freeze_classifier": s["freeze_classifier"]}
            for s in trial["stages"]],
        "resolved_object_loss_weights": trial["loss_weights"]["object"],
        "segmentation_weight": trial["loss_weights"]["segmentation"],
        "class_loss_weights": trial["class_loss_weights"],
        "lovasz_weight": trial["lovasz_weight"],
        "batch_size": int(trial["batch_size"]),
        "matches_model_constants": (list(object_cfg["native_grid"]) == list(NATIVE_GRID)
                                    and int(object_cfg["native_stride"]) == NATIVE_STRIDE),
    }

    # 3. Verify the warm-start SHA.
    warm_start_path = (ROOT / trial["warm_start_checkpoint"]).resolve(strict=True)
    warm_start_sha = sha256(warm_start_path)
    result["warm_start"] = {
        "path": str(warm_start_path.relative_to(ROOT)),
        "sha256": warm_start_sha,
        "expected_sha256": EXPECTED_WARM_START_SHA,
        "sha_verified": warm_start_sha == EXPECTED_WARM_START_SHA == str(trial["warm_start_sha256"]),
    }

    # 4. One synthetic target case.
    result["synthetic_target"] = check_synthetic_target()

    # 5. One real v3.1 q=0 AMP forward/backward batch.
    set_reproducible_seeds(int(trial["training_seed"]))
    device = torch.device("cuda")
    model = build_native_grid_model(
        num_classes=int(config["training"].get("num_classes", 3)),
        radar_channels=int(config["fusion"]["radar_channels"]),
        hidden_channels=int(object_cfg["hidden_channels"]),
        head_depth=int(object_cfg["head_depth"]), device=device)

    # Confirm the new upsampler is a bilinear 2x upsample at initialisation, which is
    # what carries the warm-started output heads onto the stride-4 grid intact.
    with torch.no_grad():
        probe = torch.rand(2, int(object_cfg["hidden_channels"]), 54, 96, device=device)
        model.eval()
        upsampled = model.object_head.upsampler(probe)
        reference = torch.nn.functional.interpolate(probe, scale_factor=2, mode="bilinear", align_corners=False)
        interior = slice(2, -2)
        result["upsampler_init"] = {
            "kernel_is_bilinear": bool(torch.allclose(
                model.object_head.upsampler[0].weight.detach().cpu(),
                bilinear_kernel(int(object_cfg["hidden_channels"])), atol=1e-6)),
            "output_shape": list(upsampled.shape),
            "max_abs_interior_deviation_from_bilinear": float(
                (upsampled[..., interior, interior] - reference[..., interior, interior]).abs().max().item()),
            "offset_head_bias_is_cell_centre": bool(
                torch.allclose(model.object_head.offset_head.bias.detach().cpu(), torch.full((2,), 0.5))),
        }

    warm_start = load_warm_start(model, warm_start_path, device=device)
    result["warm_start_mapping"] = {
        "loaded_count": warm_start["loaded_count"],
        "transformed_count": warm_start["transformed_count"],
        "new_count": warm_start["new_count"],
        "incompatible_count": warm_start["incompatible_count"],
        "new_tensors": warm_start["new_tensors"],
        "transformed_tensors": warm_start["transformed_tensors"],
        "incompatible_tensors": warm_start["incompatible_tensors"],
        "unexpected_keys": warm_start["unexpected_keys"],
        "missing_keys_are_new_only": warm_start["missing_keys_are_new_only"],
    }

    dataset_dir = experiment / "dataset"
    rows = [row for row in read_manifest(dataset_dir / "manifest.csv") if row.get("split") == "train"]
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    loader = DataLoader(
        NativeGridDataset(dataset_dir, rows, object_rows,
                          tuple(int(v) for v in config["training"]["input_size"]), object_cfg,
                          augment_strength=str(trial["augment_strength"])),
        batch_size=int(trial["batch_size"]), shuffle=True, num_workers=4)
    tensors, masks, targets = next(iter(loader))
    tensors, masks = tensors.to(device), masks.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    # An explicit modest init_scale keeps the first backward off the fp16 overflow that
    # the default dynamic scaler would simply skip and back off from. Gradients are read
    # AFTER unscale_, i.e. in true (unscaled) units.
    probe_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=2.0 ** 8)
    with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        outputs = model(tensors, feature_drop_fraction=0.0)
        seg_loss, seg_parts, _seg_logits = segmentation_loss(
            outputs["out"], masks,
            class_weights=torch.tensor([float(v) for v in trial["class_loss_weights"]], device=device),
            lovasz_weight=float(trial["lovasz_weight"]))
        with torch.autocast(device_type="cuda", enabled=False):
            object_loss, object_parts = native_object_loss(
                outputs["object"].float(), targets, trial["loss_weights"]["object"])
        total = (float(trial["loss_weights"]["segmentation"]) * seg_loss
                 + float(trial["loss_weights"]["object_total"]) * object_loss)
    scaler.scale(total).backward()
    scaler.unscale_(probe_optimizer)

    gradients = {
        "upsampler": branch_gradient(model.object_head.upsampler),
        "vehicle_heatmap_head": branch_gradient(model.object_head.vehicle_heatmap_head),
        "person_heatmap_head": branch_gradient(model.object_head.person_heatmap_head),
        "offset_head": branch_gradient(model.object_head.offset_head),
        "regression_head": branch_gradient(model.object_head.regression_head),
        "shared_trunk": branch_gradient(model.object_head.shared_trunk),
    }
    result["real_batch"] = {
        "batch_size": int(tensors.shape[0]),
        "amp_enabled": True,
        "grad_scale_used": float(scaler.get_scale()),
        "gradients_are_unscaled": True,
        "input_shape": list(tensors.shape),
        "input_channels_are_7": int(tensors.shape[1]) == 7,
        "object_grid": list(outputs["object"].shape[-2:]),
        "object_grid_is_192x108": list(outputs["object"].shape[-2:]) == [NATIVE_GRID[1], NATIVE_GRID[0]],
        "object_channels": int(outputs["object"].shape[1]),
        "object_channels_expected": OUTPUT_CHANNELS,
        "target_grid": list(targets["center_heatmap"].shape[-2:]),
        "total_loss": float(total.detach().item()),
        "loss_finite": bool(math.isfinite(float(total.detach().item()))),
        "loss_parts": {**seg_parts, **object_parts},
        "positive_cells_in_batch": object_parts["positive_cells"],
        "ignore_cells_in_batch": int(targets["center_heatmap"].eq(-1.0).sum().item()),
        "gradients": gradients,
        "all_new_and_output_branches_have_finite_nonzero_gradients": all(
            value["finite"] and value["nonzero"] for value in gradients.values()),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2),
        "device_total_mib": float(torch.cuda.get_device_properties(device).total_memory) / (1024.0 ** 2),
    }
    result["parameters"] = parameter_report(model)

    # 6. Split-boundary check.
    model.zero_grad(set_to_none=True)
    result["split_boundary"] = split_boundary_report(model, tensors[:2])

    checks = {
        "py_compile_ok": result["py_compile"]["ok"],
        "config_parsed_and_consistent": result["config"]["matches_model_constants"],
        "warm_start_sha_verified": result["warm_start"]["sha_verified"],
        "warm_start_only_new_tensors_uninitialised": (
            result["warm_start_mapping"]["incompatible_count"] == 0
            and result["warm_start_mapping"]["missing_keys_are_new_only"]
            and not result["warm_start_mapping"]["unexpected_keys"]),
        "synthetic_centre_and_offset_exact": (
            result["synthetic_target"]["peak_is_expected_cell"]
            and result["synthetic_target"]["offset_exact"]
            and result["synthetic_target"]["centre_recovery_exact"]),
        "synthetic_ignore_downsample_conservative": result["synthetic_target"]["ignore_downsample_is_conservative"],
        "real_batch_loss_finite": result["real_batch"]["loss_finite"],
        "real_batch_gradients_finite_nonzero": result["real_batch"]["all_new_and_output_branches_have_finite_nonzero_gradients"],
        "object_grid_is_192x108": result["real_batch"]["object_grid_is_192x108"],
        "tail_reads_only_low_high_bundle": result["split_boundary"]["tail_reads_only_transported_bundle"],
        "monolithic_matches_encode_decode": result["split_boundary"]["outputs_match"],
        "batch_size_fits_with_headroom": (
            result["real_batch"]["peak_reserved_mib"] < 0.80 * result["real_batch"]["device_total_mib"]),
    }
    result["nonfinite_fields"] = find_nonfinite(result, "result")
    checks["no_nonfinite_recorded_values"] = not result["nonfinite_fields"]
    result["checks"] = checks
    result["launch_ready"] = all(checks.values())

    experiment.mkdir(parents=True, exist_ok=True)
    write_json_x(experiment / "LAUNCH_CHECK.json", result)
    (experiment / "LAUNCH_CHECK_COMPLETE").write_text(
        ("LAUNCH_READY" if result["launch_ready"] else "LAUNCH_BLOCKED") + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"launch_ready": result["launch_ready"], "checks": checks,
                      "nonfinite_fields": result["nonfinite_fields"],
                      "object_grid": result["real_batch"]["object_grid"],
                      "object_channels": result["real_batch"]["object_channels"],
                      "warm_start": result["warm_start_mapping"]["loaded_count"],
                      "transformed": result["warm_start_mapping"]["transformed_count"],
                      "new": result["warm_start_mapping"]["new_tensors"],
                      "upsampler_bilinear_deviation":
                          result["upsampler_init"]["max_abs_interior_deviation_from_bilinear"],
                      "loss_parts": result["real_batch"]["loss_parts"],
                      "gradients": {k: round(v["norm"], 6) for k, v in gradients.items()},
                      "peak_reserved_mib": result["real_batch"]["peak_reserved_mib"]}),
                     indent=2), flush=True)
    return 0 if result["launch_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
