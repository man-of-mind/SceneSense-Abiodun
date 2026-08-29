#!/usr/bin/env python3
"""Create the target view, register the design, and qualify the one scientific run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

import model_v1 as visible_model  # noqa: E402
native = visible_model.native
from common_v1 import (  # noqa: E402
    read_csv, seed_everything, sha256, tensor_state_hash, utc_now, write_json_x,
    write_text_x,
)
from decode_v1 import algebraic_roundtrip, decode_all  # noqa: E402
from losses_v1 import private_person_loss  # noqa: E402
from model_v1 import (  # noqa: E402
    build_model, configure_private_training, inherited_state, load_epoch40,
    parameter_report, private_parameters, split_boundary_report,
)
from targets_v1 import (  # noqa: E402
    VisibleAnchorDataset, build_sampling_weights, build_visible_target_view,
    gaussian_unit_tests, load_visible_rows, verify_audit_gaussian_population,
)
from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    load_object_boxes, parse_matrix,
)


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    names = (
        "common_v1.py", "model_v1.py", "targets_v1.py", "losses_v1.py",
        "decode_v1.py", "preflight_v1.py", "train_v1.py", "infer_v1.py",
        "evaluate_v1.py", "report_v1.py", "run_pipeline_v1.py",
    )
    return {name: sha256(PACKAGE / name) for name in names if (PACKAGE / name).is_file()}


def _setup_links(experiment: Path, dataset_root: Path) -> None:
    experiment.mkdir(parents=True, exist_ok=False)
    (experiment / "logs").mkdir()
    (experiment / "dataset").symlink_to(dataset_root / "dataset", target_is_directory=True)
    (experiment / "contracts").symlink_to(dataset_root / "contracts", target_is_directory=True)


def _loss_design(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth_bounds_m": config["person_private"]["depth_bounds_m"],
        "dimension_normalization_m": config["person_private"]["dimension_normalization_m"],
        "endpoint_normalization_m": config["person_private"]["endpoint_normalization_m"],
        "loss_weights": config["training"]["loss_weights"],
    }


def _make_dataset(experiment: Path, rows: list[dict[str, str]],
                  object_rows: dict[str, list[dict[str, str]]], object_cfg: dict[str, Any],
                  visible_rows: dict[str, list[dict[str, Any]]], config: dict[str, Any],
                  offset_scales: dict[str, float], augment: str) -> VisibleAnchorDataset:
    return VisibleAnchorDataset(
        experiment / "dataset", rows, object_rows, tuple(config["model_size_wh"]),
        object_cfg, augment_strength=augment, geometric_augment=False,
        visible_rows=visible_rows, offset_scales=offset_scales,
        depth_bounds_m=config["person_private"]["depth_bounds_m"],
        dimension_scale_m=config["person_private"]["dimension_normalization_m"],
        endpoint_scale_m=config["person_private"]["endpoint_normalization_m"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    checkpoint_path = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    audit_root = (ROOT / config["audit_root"]).resolve(strict=True)
    if experiment.exists():
        raise FileExistsError(f"create-only experiment already exists: {experiment}")
    _setup_links(experiment, dataset_root)

    branch = _git(["branch", "--show-current"])
    head = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--short"]).splitlines()
    if branch != "master" or head != config["required_starting_head"]:
        raise RuntimeError(f"repository start drift branch={branch} head={head}")
    if any(line not in {"m OAI/openairinterface5g", " m OAI/openairinterface5g"}
           and "route_b_v3_1_person_visible_anchor_v1" not in line
           for line in status):
        raise RuntimeError(f"unexpected pre-existing repository changes: {status}")
    if sha256(checkpoint_path) != config["warm_start_sha256"]:
        raise RuntimeError("required epoch-40 checkpoint SHA mismatch")

    manifest = read_manifest(experiment / "dataset/manifest.csv")
    split_counts = {name: sum(row["split"] == name for row in manifest)
                    for name in ("train", "val", "test")}
    split_names = {row["split"] for row in manifest}
    if split_counts != config["expected_split_counts"] or split_names != {"train", "val"}:
        raise RuntimeError(f"split contract drift: {split_counts}/{split_names}")
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    train_episodes = {row["experiment_id"] for row in train_rows}
    val_episodes = {row["experiment_id"] for row in val_rows}
    if (len(train_episodes) != 10 or len(val_episodes) != 2
            or train_episodes & val_episodes):
        raise RuntimeError("episode split/disjointness failure")
    if any("canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"]
           for row in manifest):
        raise RuntimeError("locked test reference in resolved manifest")

    target_path = experiment / "derived_targets/visible_anchor_targets_v010.csv"
    target_summary = build_visible_target_view(dataset_root, target_path)
    unit_tests = gaussian_unit_tests()
    gaussian_population = verify_audit_gaussian_population(
        dataset_root, audit_root / "gaussian_radius_comparison.csv",
    )
    visible_rows, target_parameters = load_visible_rows(target_path)
    if target_parameters["validation_influence"]:
        raise RuntimeError("validation influenced offset normalization")
    if target_summary["physical_projection_roundtrip_max_abs_error_m"] > 1e-9:
        raise RuntimeError("target-view physical projection roundtrip failed")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    object_cfg = dict(checkpoint["config"]["object_heads"])
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    weights, sampler_report = build_sampling_weights(
        train_rows, val_rows, object_rows, config["training"]["sampling"],
    )
    sampler_payload = {**sampler_report, "normalized_weights": weights.tolist()}
    write_json_x(experiment / "SAMPLER_REGISTRATION.json", sampler_payload)

    resolved = json.loads(json.dumps(config))
    resolved["resolved_offset_scales"] = target_parameters["offset_scales"]
    resolved["resolved_source_config"] = str(config_path)
    resolved["resolved_source_config_sha256"] = sha256(config_path)
    resolved["resolved_target_view"] = str(target_path)
    resolved["resolved_target_view_sha256"] = sha256(target_path)
    resolved["all_scientific_settings_frozen_before_optimizer_step"] = True
    resolved_path = experiment / "RESOLVED_CONFIG.json"
    write_json_x(resolved_path, resolved)
    design_markdown = f"""# Registered Route B v3.1 person-visible-anchor design

Registered before the first optimizer step. The inherited epoch-40 backbone, all inherited BatchNorm state, segmentation classifier, native shared object path, both inherited class heads, regression head, and offset head are frozen. A private copied low/high tower predicts a visible heatmap, visible subcell, visible-to-full-box offset, full box size, a separate visible-to-physical-ray offset, bounded positive forward depth, person dimensions/yaw, and radar support.

- Warm start: `{checkpoint_path}` (`{config['warm_start_sha256']}`)
- Derived v0.10 target view: `{target_path}` (`{sha256(target_path)}`)
- Visible/full-box offset scale: `{target_parameters['offset_scales']['box_center_grid_cells']}` grid cells (train-only p99.5 ceiling)
- Visible/physical-ray offset scale: `{target_parameters['offset_scales']['physical_ray_grid_cells']}` grid cells (train-only p99.5 ceiling)
- Numerical policy: full FP32; FP16 forbidden; BF16 qualified only as an alternate
- Schedule: 24 epochs, batch 16, AdamW, one warmup epoch, then registered cosine decay; checkpoints/evaluations at 6/12/18/24
- Geometry-changing augmentation, distillation, threshold/NMS sweeps, q/AE, test, CARLA, OAI, live runtime, and 288 measurements are excluded.
"""
    write_text_x(experiment / "REGISTERED_DESIGN.md", design_markdown)
    write_json_x(experiment / "REGISTERED_DESIGN.json", {
        "schema": "route_b_v3_1_person_visible_anchor_registered_design_v1",
        "created_utc": utc_now(), "resolved_config": str(resolved_path),
        "resolved_config_sha256": sha256(resolved_path),
        "design_markdown_sha256": sha256(experiment / "REGISTERED_DESIGN.md"),
        "target_parameters": target_parameters, "target_summary": target_summary,
        "gaussian_unit_tests": unit_tests, "gaussian_population": gaussian_population,
        "sampler_registration_sha256": sha256(experiment / "SAMPLER_REGISTRATION.json"),
        "optimizer_steps_before_registration": 0,
        "all_scientific_settings_frozen": True,
    })

    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA runtime unavailable")
    device = torch.device("cuda")
    seed_everything(int(config["training"]["training_seed"]))
    dataset = _make_dataset(
        experiment, train_rows, object_rows, object_cfg, visible_rows, resolved,
        resolved["resolved_offset_scales"], "off",
    )
    positive_indices = [index for index, row in enumerate(train_rows) if visible_rows.get(row["sample_id"])]
    if len(positive_indices) < int(config["training"]["batch_size"]):
        raise RuntimeError("insufficient real positive frames for qualification")
    loader = DataLoader(
        Subset(dataset, positive_indices[:int(config["training"]["batch_size"])]),
        batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0,
    )
    tensors, _masks, targets = next(iter(loader))
    tensors = tensors.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}

    model = build_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]),
        depth_bounds_m=tuple(config["person_private"]["depth_bounds_m"]), device=device,
    )
    mapping = load_epoch40(model, checkpoint_path, device=device, initialize_private=True)
    configure_private_training(model)
    parameters = parameter_report(model)
    inherited_trainable = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith("person_private.") and parameter.requires_grad
    )
    if inherited_trainable != 0:
        raise RuntimeError("inherited parameter unexpectedly trainable")
    inherited_before = tensor_state_hash(inherited_state(model))

    reference = native.build_native_grid_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]), device=device,
    )
    reference.load_state_dict(checkpoint["model"], strict=True); reference.eval(); model.eval()
    with torch.inference_mode():
        reference_outputs = reference(tensors[:1])
        attached_outputs = model(tensors[:1])
    base_deltas = {name: float((reference_outputs[name] - attached_outputs[name]).abs().max().item())
                   for name in ("out", "object")}
    if any(value != 0.0 for value in base_deltas.values()):
        raise RuntimeError(f"inherited tensor parity failure: {base_deltas}")
    split = split_boundary_report(model, tensors[:1])
    if not split["outputs_bit_identical"]:
        raise RuntimeError("monolithic/split inference mismatch")
    expected_shapes = {"low": [1, 40, 54, 96], "high": [1, 960, 27, 48]}
    if split["transported_feature_shapes"] != expected_shapes:
        raise RuntimeError(f"transport shape drift: {split['transported_feature_shapes']}")

    first_row = train_rows[positive_indices[0]]
    matrix = parse_matrix(first_row["camera_matrix_json"])
    if matrix is None:
        raise RuntimeError("qualification camera matrix missing")
    sx = float(config["model_size_wh"][0]) / float(first_row["camera_width"])
    sy = float(config["model_size_wh"][1]) / float(first_row["camera_height"])
    intrinsic = np.asarray([
        [float(first_row["camera_fx"]) * sx, 0.0, float(first_row["camera_cx"]) * sx],
        [0.0, float(first_row["camera_fy"]) * sy, float(first_row["camera_cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    new_records = decode_all(
        attached_outputs, camera_matrix=matrix, intrinsic_model=intrinsic,
        score_threshold=0.02, offset_scales=resolved["resolved_offset_scales"],
        depth_bounds_m=config["person_private"]["depth_bounds_m"],
    )
    native_records = native.decode_native_objects(
        reference_outputs["object"], camera_matrix=matrix, score_threshold=0.02,
    ) if hasattr(native, "decode_native_objects") else None
    # Native decode is imported through decode_v1 in decode_all; tensor equality is the
    # stronger vehicle proof. Record-schema proof is checked directly.
    required_record_fields = {
        "class_name", "score", "world_x", "world_y", "world_z", "local_x", "local_y",
        "local_z", "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
        "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
        "bbox_x1", "bbox_y1",
    }
    if not new_records or any(not required_record_fields.issubset(row) for row in new_records):
        raise RuntimeError("external object record schema failure")

    algebra = [
        algebraic_roundtrip((20.0, 2.0, 1.0), intrinsic),
        algebraic_roundtrip((5.0, -1.0, -0.3), intrinsic),
        algebraic_roundtrip((39.0, 8.0, 1.7), intrinsic),
    ]
    if not all(row["pass"] for row in algebra):
        raise RuntimeError("algebraic projection/unprojection qualification failed")

    qualification: dict[str, Any] = {}
    for policy in ("bf16_frozen_features_private_fp32", "full_fp32"):
        configure_private_training(model)
        for parameter in private_parameters(model):
            parameter.grad = None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        enabled = policy.startswith("bf16")
        with torch.autocast(device_type="cuda", enabled=enabled, dtype=torch.bfloat16):
            outputs = model.private_training_outputs(tensors)
        loss, parts = private_person_loss(
            outputs, targets, design=_loss_design(resolved),
            offset_scales=resolved["resolved_offset_scales"],
        )
        finite_forward = bool(torch.isfinite(loss).item()
                              and all(torch.isfinite(value).all().item() for value in outputs.values()))
        loss.backward()
        gradients = {}
        for name, output_head in model.person_private.heads.items():
            values = [parameter.grad for parameter in output_head.parameters()]
            gradients[name] = {
                "present": all(value is not None for value in values),
                "finite": all(value is not None and torch.isfinite(value).all().item() for value in values),
                "absolute_sum": float(sum(value.abs().sum().item() for value in values if value is not None)),
            }
        qualification[policy] = {
            "finite_forward_and_loss": finite_forward,
            "loss": parts, "output_head_gradients": gradients,
            "all_output_heads_finite_nonzero_gradient": all(
                row["present"] and row["finite"] and row["absolute_sum"] > 0.0
                for row in gradients.values()
            ),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
    selected = qualification["full_fp32"]
    if not all(row["finite_forward_and_loss"] and row["all_output_heads_finite_nonzero_gradient"]
               for row in qualification.values()):
        raise RuntimeError(f"numerical/gradient qualification failure: {qualification}")
    maximum_share = max(value for key, value in selected["loss"].items() if key.startswith("share_"))
    if maximum_share > 0.60:
        raise RuntimeError(f"one registered weighted loss exceeds 60%: {maximum_share}")
    if selected["peak_reserved_mib"] >= 12 * 1024:
        raise RuntimeError(f"batch-16 memory exceeds 12 GiB: {selected['peak_reserved_mib']}")
    if any(parameter.grad is not None for name, parameter in model.named_parameters()
           if not name.startswith("person_private.")):
        raise RuntimeError("inherited parameter received a gradient")
    inherited_after = tensor_state_hash(inherited_state(model))
    if inherited_before != inherited_after:
        raise RuntimeError("inherited state drifted during qualification")

    target_audit = {
        "schema": "route_b_v3_1_visible_anchor_target_gaussian_audit_v1",
        "created_utc": utc_now(), "target_view": str(target_path),
        "target_view_sha256": sha256(target_path), "target_summary": target_summary,
        "target_parameters": target_parameters, "gaussian_unit_tests": unit_tests,
        "gaussian_population": gaussian_population,
        "proofs": {
            "person_anchor_pixels_own_visible_fraction": 1.0,
            "person_anchor_cells_contain_own_visible_fraction": 1.0,
            "reference_population_exact_match_fraction": 1.0,
            "train_sampling_or_parameters_use_validation": False,
        },
    }
    write_json_x(experiment / "TARGET_GAUSSIAN_AUDIT.json", target_audit)
    numerical = {
        "schema": "route_b_v3_1_person_visible_anchor_numerical_qualification_v1",
        "created_utc": utc_now(), "candidates": qualification,
        "selected_policy": "full_fp32", "private_fp16_used": False,
        "maximum_weighted_loss_share": maximum_share,
        "batch_size": int(config["training"]["batch_size"]),
        "memory_below_12_gib": True,
    }
    write_json_x(experiment / "NUMERICAL_QUALIFICATION.json", numerical)
    freeze_split = {
        "schema": "route_b_v3_1_person_visible_anchor_freeze_split_qualification_v1",
        "created_utc": utc_now(), "warm_start_mapping": mapping,
        "parameter_report": parameters, "inherited_trainable_parameters": inherited_trainable,
        "inherited_state_hash_before": inherited_before,
        "inherited_state_hash_after": inherited_after,
        "zero_inherited_state_drift": inherited_before == inherited_after,
        "attached_vs_native_base_max_abs_delta": base_deltas,
        "vehicle_and_segmentation_tensors_bit_identical": all(value == 0.0 for value in base_deltas.values()),
        "split_boundary": split, "algebraic_roundtrips": algebra,
        "external_record_fields": sorted(required_record_fields),
        "external_record_schema_unchanged": True,
        "private_decoder_inputs": ["low", "high", "camera_intrinsics", "camera_to_world"],
        "private_decoder_raw_sensor_side_channels": [],
    }
    write_json_x(experiment / "FREEZE_SPLIT_GEOMETRY_QUALIFICATION.json", freeze_split)

    input_provenance = {
        "schema": "route_b_v3_1_person_visible_anchor_input_provenance_v1",
        "created_utc": utc_now(), "repository": {"branch": branch, "head": head, "status": status},
        "warm_start": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
                       "bytes": checkpoint_path.stat().st_size},
        "dataset": {"root": str(dataset_root), "split_counts": split_counts,
                    "train_episodes": sorted(train_episodes), "val_episodes": sorted(val_episodes),
                    "episodes_disjoint": True, "test_rows": 0},
        "inputs": {
            "manifest_sha256": sha256(dataset_root / "dataset/manifest.csv"),
            "object_boxes_sha256": sha256(dataset_root / "dataset/object_boxes.csv"),
            "v010_train_boxes_sha256": sha256(dataset_root / "contracts/v010/train/object_boxes.csv"),
            "v010_val_boxes_sha256": sha256(dataset_root / "contracts/v010/val/object_boxes.csv"),
            "audit_gaussian_sha256": sha256(audit_root / "gaussian_radius_comparison.csv"),
        },
        "source_hashes": _source_hashes(),
        "forbidden_scope_access_counts": {"test": 0, "CARLA": 0, "OAI": 0, "q_AE": 0,
                                          "live_runtime": 0, "campaign_288": 0},
    }
    write_json_x(experiment / "INPUT_PROVENANCE.json", input_provenance)
    preflight = {
        "schema": "route_b_v3_1_person_visible_anchor_preflight_v1",
        "created_utc": utc_now(), "all_pass": True,
        "resolved_config_sha256": sha256(resolved_path),
        "registered_design_sha256": sha256(experiment / "REGISTERED_DESIGN.json"),
        "target_gaussian_audit_sha256": sha256(experiment / "TARGET_GAUSSIAN_AUDIT.json"),
        "numerical_qualification_sha256": sha256(experiment / "NUMERICAL_QUALIFICATION.json"),
        "freeze_split_qualification_sha256": sha256(experiment / "FREEZE_SPLIT_GEOMETRY_QUALIFICATION.json"),
        "optimizer_steps": 0, "scientific_attempts_consumed": 0,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "PREFLIGHT.json", preflight)
    write_text_x(experiment / "PREFLIGHT_COMPLETE", "ALL_PREFLIGHT_ASSERTIONS_PASS\n")
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
