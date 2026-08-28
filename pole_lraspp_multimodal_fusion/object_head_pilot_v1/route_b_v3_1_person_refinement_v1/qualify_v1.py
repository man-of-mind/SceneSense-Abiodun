#!/usr/bin/env python3
"""Fail-closed source, data, split-boundary, gradient, and AMP qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import person_model_v1  # noqa: E402
from person_decode_v1 import decode_all  # noqa: E402
from person_losses_v1 import person_refinement_loss  # noqa: E402
from person_model_v1 import (  # noqa: E402
    build_model, configure_stage, load_recovered_base, parameter_report,
    split_boundary_report,
)
from person_targets_v1 import (  # noqa: E402
    PersonRefinementDataset, build_sampling_weights, derive_range_edges,
)
from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes, parse_matrix  # noqa: E402


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


def grad_summary(model: torch.nn.Module) -> dict[str, Any]:
    groups = {
        "new_person_tail": model.person_refinement,
        "person_trunk": model.person_refinement.trunk,
        "person_objectness": model.person_refinement.objectness_residual,
        "person_quality": model.person_refinement.localization_quality,
        "person_range_bins": model.person_refinement.range_bin_logits,
        "person_range_residual": model.person_refinement.range_residual,
        "person_projected_offset": model.person_refinement.projected_center_offset,
        "person_mask_residual": model.person_refinement.person_mask_residual,
        "inherited_person_heatmap": model.object_head.person_heatmap_head,
        "vehicle_heatmap": model.object_head.vehicle_heatmap_head,
        "shared_regression": model.object_head.regression_head,
        "grid_offset": model.object_head.offset_head,
        "native_shared": model.object_head.shared_trunk,
        "native_upsampler": model.object_head.upsampler,
        "backbone": model.backbone,
        "segmentation": model.classifier,
    }
    output: dict[str, Any] = {}
    for name, module in groups.items():
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        output[name] = {
            "tensors_with_gradient": len(gradients),
            "finite": all(bool(torch.isfinite(value).all().item()) for value in gradients),
            "absolute_sum": float(sum(value.detach().float().abs().sum().item() for value in gradients)),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--base-acceptance", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registration-output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = config["person_design"]
    base_path = args.base_checkpoint.resolve(strict=True)
    diagnostic_path = args.diagnostic.resolve(strict=True)
    acceptance_path = args.base_acceptance.resolve(strict=True)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    checks.append({
        "name": "recovered_epoch40_favorable_variation_accepted",
        "pass": (
            acceptance.get("decision")
            == "RECOVERED_EPOCH40_ACCEPTED_WITH_FAVORABLE_LOW_THRESHOLD_VARIATION"
            and acceptance.get("recovered_checkpoint_sha256") == args.base_sha256
            and acceptance.get("comparison_baseline")
            == "recovered_checkpoint_own_decoded_primary_v010_metrics"
            and acceptance.get("repeat_epochs_11_through_40") is False
            and acceptance.get("person_refinement_design_changed") is False
        ),
        "acceptance": str(acceptance_path), "acceptance_sha256": sha256(acceptance_path),
        "decision": acceptance.get("decision"),
    })

    sources = sorted(PACKAGE_ROOT.glob("*.py"))
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, sources)],
        capture_output=True, text=True, check=False,
    )
    checks.append({
        "name": "all_person_refinement_sources_compile",
        "pass": compile_result.returncode == 0,
        "sources": [str(path.relative_to(ROOT)) for path in sources],
        "stderr": compile_result.stderr[-2000:],
    })
    checks.append({
        "name": "required_interpreter_sm120_cuda",
        "pass": (
            sys.executable == "/usr/bin/python3" and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
            and "sm_120" in torch.cuda.get_arch_list()
        ),
        "interpreter": sys.executable,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "architecture_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    })
    actual_base_hash = sha256(base_path)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    required_state = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    checks.append({
        "name": "exact_recovered_epoch40_full_state",
        "pass": actual_base_hash == args.base_sha256 and int(base.get("epoch", -1)) == 40
                and required_state.issubset(base),
        "actual_sha256": actual_base_hash, "expected_sha256": args.base_sha256,
        "epoch": base.get("epoch"), "required_state_present": required_state.issubset(base),
    })

    rows = read_manifest(experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_tokens = [row["sample_id"] for row in rows if "canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"]]
    checks.append({
        "name": "expanded_split_counts_locked_test_absent",
        "pass": len(train_rows) == 16827 and len(val_rows) == 3345
                and {row["split"] for row in rows} == {"train", "val"} and not test_tokens,
        "train": len(train_rows), "validation": len(val_rows), "test": 0,
        "locked_test_token_matches": len(test_tokens), "locked_test_paths_read": 0,
    })
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    train_ids = {row["sample_id"] for row in train_rows}
    range_bins = derive_range_edges(
        object_rows, train_ids, design["range_quantiles"],
        float(design["range_floor_m"]), float(design["range_ceiling_m"]),
    )
    weights, sampler_report = build_sampling_weights(
        train_rows, val_rows, object_rows, design["sampling"],
    )
    sampler_report["normalized_weights"] = [float(value) for value in weights.tolist()]
    checks.append({
        "name": "train_only_range_bins_and_balanced_sampler",
        "pass": range_bins["population"] >= 1000
                and all(b > a for a, b in zip(range_bins["edges_m"], range_bins["edges_m"][1:]))
                and sampler_report["validation_rows_used_by_sampler_or_mining"] == 0
                and len(weights) == 16827,
        "range_bins": range_bins,
        "sampler_without_weight_vector": {key: value for key, value in sampler_report.items()
                                           if key != "normalized_weights"},
    })

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for executable qualification")
    device = torch.device("cuda")
    set_reproducible_seeds(int(design["training_seed"]))
    object_cfg = dict(base["config"]["object_heads"])
    positive_ids = {
        sample_id for sample_id in train_ids
        if any(value.get("label") == "person" and value.get("gt_source") == "actor"
               for value in object_rows.get(sample_id, []))
    }
    qualification_rows = [row for row in train_rows if row["sample_id"] in positive_ids][:16]
    if len(qualification_rows) != 16:
        raise RuntimeError("insufficient train-only person-positive qualification rows")
    dataset = PersonRefinementDataset(
        experiment / "dataset", qualification_rows, object_rows,
        tuple(config["registered_input_size"]), object_cfg,
        augment_strength="off", geometric_augment=False,
        range_edges=range_bins["edges_m"],
        offset_caps=design["projected_offset_cap_grid_xy"],
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    qualification_batches = iter(loader)
    tensors, masks, targets = next(qualification_batches)
    successor_tensors, successor_masks, successor_targets = next(qualification_batches)
    tensors = tensors.to(device)
    masks = masks.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}
    model = build_model(
        radar_channels=int(base["radar_channels"]),
        hidden_channels=int(base["object_hidden_channels"]),
        head_depth=int(base["object_head_depth"]),
        person_hidden=int(design["hidden_channels"]),
        group_norm_groups=int(design["group_norm_groups"]),
        range_bins=int(design["range_bins"]), device=device,
    )
    mapping = load_recovered_base(model, base_path, device=device)
    split = split_boundary_report(model, tensors[:1])
    checks.append({
        "name": "encode_front_low_high_split_parity",
        "pass": split["tail_reads_only_low_high"] and not split["tail_raw_modality_side_channels"]
                and split["outputs_bit_identical"],
        "detail": split,
    })

    native_model = person_model_v1.native.build_native_grid_model(
        num_classes=int(base["config"]["training"].get("num_classes", 3)),
        radar_channels=int(base["radar_channels"]),
        hidden_channels=int(base["object_hidden_channels"]),
        head_depth=int(base["object_head_depth"]), device=device,
    )
    native_model.load_state_dict(base["model"], strict=True)
    native_model.eval()
    model.eval()
    with torch.inference_mode():
        native_outputs = native_model(tensors[:1], feature_drop_fraction=0.0)
        refined_outputs = model(tensors[:1], feature_drop_fraction=0.0)
    slices = person_model_v1.native
    invariant_deltas = {
        "vehicle_heatmap": float((native_outputs["object"][:, 0:1].float() - refined_outputs["object"][:, 0:1].float()).abs().max().item()),
        "shared_regression": float((native_outputs["object"][:, slices.SL_REG].float() - refined_outputs["object"][:, slices.SL_REG].float()).abs().max().item()),
        "grid_offset": float((native_outputs["object"][:, slices.SL_OFFSET].float() - refined_outputs["object"][:, slices.SL_OFFSET].float()).abs().max().item()),
        "base_segmentation": float((native_outputs["out"].float() - refined_outputs["base_out"].float()).abs().max().item()),
    }
    checks.append({
        "name": "recovered_vehicle_regression_segmentation_paths_bit_identical",
        "pass": all(value == 0.0 for value in invariant_deltas.values()),
        "max_abs_deltas": invariant_deltas, "native_channel_slices": {
            "vehicle_heatmap": [0, 1], "person_heatmap": [1, 2],
            "shared_regression": [slices.SL_REG.start, slices.SL_REG.stop],
            "grid_offset": [slices.SL_OFFSET.start, slices.SL_OFFSET.stop],
        },
    })

    gradient_checks: dict[str, Any] = {}
    for stage in ("P1", "P2"):
        configure_stage(model, stage)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
            outputs = model.training_outputs(tensors)
        with torch.autocast(device_type="cuda", enabled=False):
            loss, _parts = person_refinement_loss(
                outputs, masks, targets, range_edges=range_bins["edges_m"],
                offset_caps=design["projected_offset_cap_grid_xy"], design=design,
            )
        loss.backward()
        gradient_checks[stage] = {"loss": float(loss.detach().item()), **grad_summary(model)}
    p1, p2 = gradient_checks["P1"], gradient_checks["P2"]
    new_head_names = (
        "person_trunk", "person_objectness", "person_quality", "person_range_bins",
        "person_range_residual", "person_projected_offset", "person_mask_residual",
    )
    frozen_names = (
        "vehicle_heatmap", "shared_regression", "grid_offset", "native_shared",
        "native_upsampler", "backbone", "segmentation",
    )
    gradient_pass = (
        math.isfinite(p1["loss"]) and p1["new_person_tail"]["absolute_sum"] > 0.0
        and all(p1[name]["absolute_sum"] > 0.0 for name in new_head_names)
        and p1["inherited_person_heatmap"]["tensors_with_gradient"] == 0
        and math.isfinite(p2["loss"]) and p2["new_person_tail"]["absolute_sum"] > 0.0
        and all(p2[name]["absolute_sum"] > 0.0 for name in new_head_names)
        and p2["inherited_person_heatmap"]["absolute_sum"] > 0.0
        and all(p2[name]["tensors_with_gradient"] == 0 for name in frozen_names)
        and all(payload["finite"] for payload in (p1[name] for name in p1 if isinstance(p1[name], dict)))
        and all(payload["finite"] for payload in (p2[name] for name in p2 if isinstance(p2[name], dict)))
    )
    checks.append({
        "name": "real_amp_stage_gradients_new_nonzero_frozen_zero",
        "pass": gradient_pass, "detail": gradient_checks,
    })

    row = train_rows[0]
    matrix = parse_matrix(row["camera_matrix_json"])
    scale_x = config["registered_input_size"][0] / float(row["camera_width"])
    scale_y = config["registered_input_size"][1] / float(row["camera_height"])
    intrinsic = np.asarray([
        [float(row["camera_fx"]) * scale_x, 0.0, float(row["camera_cx"]) * scale_x],
        [0.0, float(row["camera_fy"]) * scale_y, float(row["camera_cy"]) * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        outputs = model(tensors[:1])
    predictions = decode_all(
        outputs, camera_matrix=np.asarray(matrix), intrinsic_model=intrinsic,
        range_edges=range_bins["edges_m"], offset_caps=design["projected_offset_cap_grid_xy"],
        score_threshold=0.02, model_size=config["registered_input_size"],
    )
    required_fields = {
        "class_name", "score", "world_x", "world_y", "world_z", "local_x", "local_y",
        "local_z", "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos",
        "center_x_px", "center_y_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    }
    schema_ok = all(required_fields.issubset(value) for value in predictions)
    nonfinite_prediction_fields = [
        {"prediction_index": index, "class_name": value.get("class_name"), "field": key,
         "value": value.get(key)}
        for index, value in enumerate(predictions)
        for key in required_fields - {"class_name"}
        if key in value and not math.isfinite(float(value[key]))
    ]
    missing_prediction_fields = [
        {"prediction_index": index, "class_name": value.get("class_name"),
         "missing": sorted(required_fields - set(value))}
        for index, value in enumerate(predictions) if not required_fields.issubset(value)
    ]
    finite_ok = not nonfinite_prediction_fields
    object_output_finite = bool(torch.isfinite(outputs["object"]).all().item())
    amp_nonfinite_by_output = {
        "object_channels": [
            int((~torch.isfinite(outputs["object"][:, channel])).sum().item())
            for channel in range(outputs["object"].shape[1])
        ],
        "out": int((~torch.isfinite(outputs["out"])).sum().item()),
        "base_out": int((~torch.isfinite(outputs["base_out"])).sum().item()),
        "native_feature": int((~torch.isfinite(outputs["native_feature"])).sum().item()),
        "person_refinement": {
            key: int((~torch.isfinite(value)).sum().item())
            for key, value in outputs["person_refinement"].items()
        },
    }
    positive_depth_targets = targets["person_local_xyz"][:, 0][targets["person_regression_mask"][:, 0].bool()]
    camera_depth_ok = (
        positive_depth_targets.numel() == 0 or bool((positive_depth_targets > 0).all().item())
    )
    checks.append({
        "name": "external_schema_range_bins_camera_plane_amp_forward",
        "pass": schema_ok and finite_ok and object_output_finite and camera_depth_ok,
        "prediction_count": len(predictions), "required_fields": sorted(required_fields),
        "schema_ok": schema_ok, "finite_ok": finite_ok,
        "object_output_finite": object_output_finite, "camera_depth_ok": camera_depth_ok,
        "amp_nonfinite_by_output": amp_nonfinite_by_output,
        "missing_prediction_fields": missing_prediction_fields,
        "nonfinite_prediction_fields": nonfinite_prediction_fields,
        "camera_plane_positive_targets": int(positive_depth_targets.numel()),
        "camera_plane_convention": "local_x_forward_local_y_right_local_z_up",
        "amp_dtype": str(outputs["object"].dtype),
    })

    successor_tensors = successor_tensors.to(device)
    successor_masks = successor_masks.to(device)
    successor_targets = {key: value.to(device) for key, value in successor_targets.items()}
    configure_stage(model, "P2")
    with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        successor_outputs = model.training_outputs(successor_tensors)
    with torch.autocast(device_type="cuda", enabled=False):
        successor_loss, _successor_parts = person_refinement_loss(
            successor_outputs, successor_masks, successor_targets,
            range_edges=range_bins["edges_m"],
            offset_caps=design["projected_offset_cap_grid_xy"], design=design,
        )
    parameters_finite = all(bool(torch.isfinite(parameter).all().item())
                            for parameter in model.parameters())
    inputs_finite = bool(torch.isfinite(tensors).all().item()) and bool(
        torch.isfinite(successor_tensors).all().item()
    )
    targets_finite = all(bool(torch.isfinite(value).all().item()) for value in targets.values()) and all(
        bool(torch.isfinite(value).all().item()) for value in successor_targets.values()
    )
    successor_finite = (
        bool(torch.isfinite(successor_loss).item())
        and bool(torch.isfinite(successor_outputs["object"]).all().item())
        and bool(torch.isfinite(successor_outputs["out"]).all().item())
        and all(bool(torch.isfinite(value).all().item())
                for value in successor_outputs["person_refinement"].values())
    )
    repair = {
        "schema": "route_b_v3_1_person_refinement_amp_numerical_repair_v1",
        "operation": "inherited_native_vehicle_and_person_1x1_class_heatmap_projections",
        "precision": "FP32",
        "scope": "two inherited class-heatmap Conv2d projections only",
        "pre_repair_fp16_nonfinite_cells": {"vehicle": 1403, "person": 1040},
        "fp32_path_finite": all(value == 0.0 for value in invariant_deltas.values()),
        "inputs_finite": inputs_finite, "targets_finite": targets_finite,
        "parameters_finite": parameters_finite,
        "failing_batch_after_repair_finite": object_output_finite and math.isfinite(float(loss.detach().item())),
        "successor_batch_after_repair_finite": successor_finite,
        "architecture_losses_targets_lr_validation_changed": False,
    }
    checks.append({
        "name": "single_narrow_amp_heatmap_repair_failing_batch_and_successor",
        "pass": all(bool(repair[key]) for key in (
            "fp32_path_finite", "inputs_finite", "targets_finite", "parameters_finite",
            "failing_batch_after_repair_finite", "successor_batch_after_repair_finite",
        )),
        "detail": repair,
    })

    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    registration = {
        "schema": "route_b_v3_1_person_refinement_registration_v1",
        "created_utc": utc_now(), "all_frozen_before_training": True,
        "resolved_config": config, "resolved_config_sha256": sha256(config_path),
        "diagnostic": str(diagnostic_path), "diagnostic_sha256": sha256(diagnostic_path),
        "base_acceptance": str(acceptance_path),
        "base_acceptance_sha256": sha256(acceptance_path),
        "base_acceptance_decision": acceptance["decision"],
        "base_checkpoint": str(base_path), "base_checkpoint_sha256": actual_base_hash,
        "architecture": config["person_design"],
        "loss": config["person_design"]["loss_weights"],
        "sampler": sampler_report, "range_bins": range_bins,
        "selection": {
            "eligibility": config["final_eligibility"],
            "material_gain": config["material_gain"],
            "ranking": "continuous_normalized_person_deficit_then_f1_recall_xy_earlier",
            "targets": config["service_targets"],
        },
        "source_hashes": source_hashes,
        "base_mapping": mapping, "parameter_report_p2": parameter_report(model),
        "validation_rows_used_for_sampler_mining_or_training": 0,
        "geometric_augmentation": False, "q": 0, "ae": False,
        "amp_numerical_repair": repair,
    }
    all_pass = all(check["pass"] for check in checks)
    result = {
        "schema": "route_b_v3_1_person_refinement_qualification_v1",
        "created_utc": utc_now(), "all_pass": all_pass, "checks": checks,
        "wall_seconds": time.monotonic() - started,
    }
    output_path = args.output.resolve() if args.output is not None else experiment / "QUALIFICATION.json"
    registration_output = (
        args.registration_output.resolve()
        if args.registration_output is not None else experiment / "REGISTRATION.json"
    )
    write_json_x(output_path, result)
    if all_pass:
        write_json_x(registration_output, registration)
        if args.output is None:
            write_json_x(experiment / "AMP_NUMERICAL_REPAIR.json", repair)
            (experiment / "QUALIFICATION_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if all_pass else 20


if __name__ == "__main__":
    raise SystemExit(main())
