#!/usr/bin/env python3
"""Autonomous, create-only Route B v3.1 COCO distillation supervisor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
EXPANDED_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_expanded_training_v2"
for _path in (str(PACKAGE_ROOT), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from dataset_v1 import (  # noqa: E402
    MODEL_HEIGHT, MODEL_WIDTH, PersonCocoDistillationDataset,
    geometry_contract_probe, person_distillation_collate,
)
from distill_v1 import (  # noqa: E402
    L_FEAT_WEIGHT, L_KD_REG_WEIGHT, L_OBJ_WEIGHT, StudentRoiAdapter,
    adapter_report, feature_distillation_loss, gather_person_logits,
    objectness_distillation_loss,
)
from evaluation_v1 import (  # noqa: E402
    baseline_deltas, catastrophic_gates, eligibility_gates, material_gain_gates,
    nondominated, person_slices, rank_key, service_gates, teacher_adoption_guard,
)
from roi_v1 import student_roi_embedding, verify_round_trip  # noqa: E402
from student_v1 import (  # noqa: E402
    SegmentationRowGuard, batch_norm_snapshot, build_student, component_gradient_sums,
    configure_trainable, deployable_state_report, enforce_train_mode, finite_parameter_tree,
    forward_once, freeze_batch_norm, registered_parameter_groups, snapshots_equal,
    split_boundary_report, supervised_loss,
)
from teacher_v1 import (  # noqa: E402
    build_teacher, teacher_forward, teacher_person_evidence, teacher_roi_embedding,
    teacher_state, verify_teacher_cache, verify_transform_identity,
)

CONFIG_PATH = PACKAGE_ROOT / "configs/person_coco_distillation_v1.json"
CONFIG: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE_VIEW = ROOT / "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
BASELINE_DECODE = ROOT / "experiments/route_b_v3_1_person_refinement_v1/20260828_163100/decisions/epoch_040_decode.json"
TERMINALS = {
    "LRASPP_COCO_DISTILLATION_SERVICE_READY",
    "LRASPP_COCO_DISTILLATION_MATERIAL_GAIN",
    "LRASPP_COCO_DISTILLATION_NO_GAIN",
    "LRASPP_COCO_DISTILLATION_TEACHER_NOT_ADOPTED",
    "LRASPP_COCO_DISTILLATION_CONTRACT_INVALID",
    "LRASPP_COCO_DISTILLATION_RUNTIME_FAILURE",
}
PROGRESS_FIELDS = (
    "created_utc", "phase", "event", "attempt", "epoch", "train_total_loss",
    "validation_total_loss", "l_feat", "l_obj", "feature_cosine", "objectness_sites",
    "person_f1", "person_recall", "vehicle_f1", "wall_seconds", "detail",
)


class ContractInvalid(RuntimeError):
    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


class TeacherNotAdopted(RuntimeError):
    pass


class CatastrophicStop(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_progress(experiment: Path, **values: Any) -> None:
    row = {key: values.get(key, "") for key in PROGRESS_FIELDS}
    row["created_utc"] = utc_now()
    with (experiment / "PROGRESS.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writerow(row)


def update_status(experiment: Path, **values: Any) -> None:
    path = experiment / "STATUS.json"
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(values)
    current["updated_utc"] = utc_now()
    json_atomic(path, current)


def tensor_tree_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensor_tree_finite(item) for item in value)
    return True


def input_hashes(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    paths = {
        "registered_config": CONFIG_PATH,
        "lineage_audit": ROOT / config["registered_design"],
        "lineage_manifest": ROOT / config["lineage_manifest"],
        "student_warm_start": ROOT / config["student_warm_start"],
        "baseline_reconciliation": ROOT / config["baseline_reference_epoch40"]["source"],
        "dataset_manifest": SOURCE_VIEW / "dataset/manifest.csv",
        "dataset_object_boxes": SOURCE_VIEW / "dataset/object_boxes.csv",
        "dataset_summary": SOURCE_VIEW / "CAMERA_PLANE_CONTRACT_SUMMARY.json",
        "dataset_config": SOURCE_VIEW / "resolved_config.json",
        "teacher_cache": Path(config["teacher"]["local_cache"]).expanduser(),
        "dimension_yaw_scorer": ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/score_contract_v1.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ContractInvalid("required local payloads are missing", {"missing_inputs": missing})
    return {name: {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()}


def cpu_contract_audit(config: Mapping[str, Any], hashes: Mapping[str, Any]) -> Dict[str, Any]:
    if hashes["student_warm_start"]["sha256"] != config["student_warm_start_sha256"]:
        raise ContractInvalid("student warm-start SHA-256 mismatch")
    if hashes["teacher_cache"]["sha256"] != config["teacher"]["cache_sha256"]:
        raise ContractInvalid("teacher cache SHA-256 mismatch")
    if hashes["teacher_cache"]["bytes"] != int(config["teacher"]["cache_bytes"]):
        raise ContractInvalid("teacher cache size mismatch")
    if hashes["dimension_yaw_scorer"]["sha256"] != config["dimension_yaw_scorer_sha256"]:
        raise ContractInvalid("fixed dimension/yaw scorer SHA-256 mismatch")
    audit_text = (ROOT / config["registered_design"]).read_text(encoding="utf-8")
    lineage = json.loads((ROOT / config["lineage_manifest"]).read_text(encoding="utf-8"))
    audit_terminal = config["required_audit_terminal"]
    if audit_terminal not in audit_text or lineage.get("terminal") != audit_terminal:
        raise ContractInvalid("required lineage audit terminal is absent")
    summary = json.loads((SOURCE_VIEW / "CAMERA_PLANE_CONTRACT_SUMMARY.json").read_text())
    expected_dataset_hashes = summary["dataset"]
    if hashes["dataset_manifest"]["sha256"] != expected_dataset_hashes["manifest_sha256"]:
        raise ContractInvalid("expanded manifest SHA-256 mismatch")
    if hashes["dataset_object_boxes"]["sha256"] != expected_dataset_hashes["object_boxes_sha256"]:
        raise ContractInvalid("expanded object-box SHA-256 mismatch")
    rows = read_manifest(SOURCE_VIEW / "dataset/manifest.csv")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    episodes = {split: {row["experiment_id"] for row in rows if row["split"] == split}
                for split in ("train", "val")}
    sample_ids = {split: {row["sample_id"] for row in rows if row["split"] == split}
                  for split in ("train", "val")}
    if counts != {"train": 16827, "val": 3345, "test": 0}:
        raise ContractInvalid("expanded dataset population drift", counts)
    if episodes["train"] & episodes["val"] or sample_ids["train"] & sample_ids["val"]:
        raise ContractInvalid("train/validation disjointness failure")
    if len(episodes["train"]) != 10 or len(episodes["val"]) != 2:
        raise ContractInvalid("episode-count drift")
    baseline = json.loads((ROOT / config["baseline_reference_epoch40"]["source"]).read_text())
    recomputed = baseline[config["baseline_reference_epoch40"]["source_field"]]
    registered = config["baseline_reference_epoch40"]
    mismatches = {key: {"registered": registered[key], "recomputed": recomputed[key]}
                  for key in recomputed if key in registered
                  and not math.isclose(float(registered[key]), float(recomputed[key]), rel_tol=0.0, abs_tol=1e-15)}
    if mismatches:
        raise ContractInvalid("baseline metric reconciliation mismatch", mismatches)
    return {
        "dataset_counts": counts,
        "train_episodes": sorted(episodes["train"]), "validation_episodes": sorted(episodes["val"]),
        "episode_disjoint": True, "sample_id_disjoint": True, "test_rows": 0,
        "test_payload_accessed": False, "audit_terminal": audit_terminal,
        "baseline_reconciliation": {"source": str(ROOT / config["baseline_reference_epoch40"]["source"]),
                                    "metrics_exact": True, "source_all_pass": baseline.get("all_pass")},
        "dataset_summary_hard_gates": summary["hard_gates"],
    }


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "student": batch["student"].to(device),
        "teacher_rgb01": batch["teacher_rgb01"].to(device),
        "segmentation": batch["segmentation"].to(device),
        "targets": {key: value.to(device) for key, value in batch["targets"].items()},
        "person_boxes": [value.to(device) for value in batch["person_boxes"]],
        "person_cells": [value.to(device) for value in batch["person_cells"]],
        "person_radar_support": [value.to(device) for value in batch["person_radar_support"]],
        "camera_intrinsics": batch["camera_intrinsics"].to(device),
        "sample_ids": list(batch["sample_ids"]), "episode_ids": list(batch["episode_ids"]),
        "geometry": list(batch["geometry"]),
    }


@torch.no_grad()
def chunked_teacher(
    teacher: torch.nn.Module, rgb01: torch.Tensor, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    p2_values, p3_values, detections = [], [], []
    for start in range(0, int(rgb01.shape[0]), int(chunk_size)):
        p2, p3, one = teacher_forward(teacher, rgb01[start:start + int(chunk_size)])
        p2_values.append(p2); p3_values.append(p3); detections.extend(one)
    return torch.cat(p2_values), torch.cat(p3_values), detections


def _loss_bundle(
    model: torch.nn.Module, adapter: torch.nn.Module, teacher: torch.nn.Module,
    batch: Mapping[str, Any], config: Mapping[str, Any], policy: str, chunk_size: int,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    features, outputs = forward_once(model, batch["student"], policy=policy)
    p2, p3, detections = chunked_teacher(teacher, batch["teacher_rgb01"], chunk_size)
    evidence, evidence_stats = teacher_person_evidence(
        detections, batch["person_boxes"],
        iou_threshold=float(config["teacher"]["detection_to_gt_iou_threshold"]),
        score_floor=float(config["teacher"]["teacher_positive_score_floor"]),
    )
    with torch.autocast(device_type=batch["student"].device.type, enabled=False):
        student_pooled = student_roi_embedding(features["low"], features["high"], batch["person_boxes"])
        student_embedding = adapter(student_pooled.float())
        teacher_embedding = teacher_roi_embedding(p2, p3, batch["person_boxes"])
        feat_loss, feat_parts = feature_distillation_loss(student_embedding, teacher_embedding)
        person_logits = gather_person_logits(outputs["object"].float(), batch["person_cells"], 1)
        teacher_scores = torch.cat(evidence) if evidence else person_logits.new_zeros((0,))
        obj_loss, obj_parts = objectness_distillation_loss(person_logits, teacher_scores)
        supervised, supervised_parts, segmentation_logits = supervised_loss(
            outputs, batch["segmentation"], batch["targets"], class_weights=class_weights,
            loss_weights={"segmentation": config["existing_losses_unchanged"]["segmentation"],
                          "object_total": config["existing_losses_unchanged"]["object_total"],
                          "object": config["existing_losses_unchanged"]["object"]},
            lovasz_weight=float(config["existing_losses_unchanged"]["lovasz_weight"]),
        )
        total = supervised.float() + L_FEAT_WEIGHT * feat_loss.float() + L_OBJ_WEIGHT * obj_loss.float()
    parts = {**supervised_parts, **feat_parts, **obj_parts,
             "total_loss": float(total.detach().item()),
             "weighted_l_feat": L_FEAT_WEIGHT * float(feat_loss.detach().item()),
             "weighted_l_obj": L_OBJ_WEIGHT * float(obj_loss.detach().item()),
             "l_kd_reg": L_KD_REG_WEIGHT}
    evidence_stats = {**evidence_stats,
                      "student_feature_shapes": {key: list(value.shape) for key, value in features.items()},
                      "teacher_feature_shapes": {"p2": list(p2.shape), "p3": list(p3.shape)},
                      "teacher_outputs_require_grad": bool(p2.requires_grad or p3.requires_grad),
                      "segmentation_shape": list(segmentation_logits.shape)}
    return total, parts, evidence_stats


def _optimizer_state_finite(optimizer: torch.optim.Optimizer) -> bool:
    return tensor_tree_finite(optimizer.state_dict())


def _qualify_policy(
    policy: str, teacher: torch.nn.Module, batch_cpu: Mapping[str, Any],
    config: Mapping[str, Any], device: torch.device, chunk_size: int,
) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, _payload = build_student(ROOT / config["student_warm_start"], device)
    trainability = configure_trainable(model)
    adapter = StudentRoiAdapter().to(device).float()
    row_guard = SegmentationRowGuard(model)
    groups = registered_parameter_groups(model, adapter)
    optimizer = torch.optim.AdamW(groups, lr=0.0, weight_decay=float(config["weight_decay"]))
    class_weights = torch.tensor(config["existing_losses_unchanged"]["class_loss_weights"],
                                 dtype=torch.float32, device=device)
    batch = _device_batch(batch_cpu, device)
    enforce_train_mode(model)
    before_bn = batch_norm_snapshot(model)
    optimizer.zero_grad(set_to_none=True)
    failure: str | None = None
    try:
        loss, parts, evidence = _loss_bundle(
            model, adapter, teacher, batch, config, policy, chunk_size, class_weights,
        )
        activations_finite = math.isfinite(float(loss.detach().item())) and all(
            math.isfinite(float(value)) for value in parts.values()
            if isinstance(value, (int, float)) and not math.isnan(float(value))
        )
        loss.backward()
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in list(model.parameters()) + list(adapter.parameters())
        )
        gradients = component_gradient_sums(model, adapter)
        nonzero_components = {name: value > 0.0 and math.isfinite(value)
                              for name, value in gradients.items()}
        frozen_gradients = [name for name, parameter in model.named_parameters()
                            if not parameter.requires_grad and parameter.grad is not None
                            and float(parameter.grad.abs().sum().item()) != 0.0]
        teacher_gradients = [name for name, parameter in teacher.named_parameters()
                             if parameter.grad is not None and float(parameter.grad.abs().sum().item()) != 0.0]
        row_gradients_zero = {}
        for name, parameter in row_guard.parameters.items():
            row_gradients_zero[name] = (
                parameter.grad is not None
                and bool(parameter.grad.detach()[:2].eq(0).all().item())
                and float(parameter.grad.detach()[2:].abs().sum().item()) > 0.0
            )
        optimizer.step()
        row_guard.restore()
        after_bn = batch_norm_snapshot(model)
        report = {
            "policy": policy, "pass": False, "losses": parts, "teacher_evidence": evidence,
            "activations_and_losses_finite": activations_finite,
            "gradients_finite": gradient_finite, "gradient_absolute_sums": gradients,
            "every_registered_component_nonzero": all(nonzero_components.values()),
            "nonzero_component_gates": nonzero_components,
            "frozen_nonzero_gradients": frozen_gradients,
            "teacher_nonzero_gradients": teacher_gradients,
            "teacher_outputs_require_grad": evidence["teacher_outputs_require_grad"],
            "segmentation_masked_row_gradients": row_gradients_zero,
            "segmentation_row_restoration": row_guard.report(),
            "batch_norm_stats_unchanged": snapshots_equal(before_bn, after_bn),
            "parameters_finite_after_step": finite_parameter_tree(
                list(model.parameters()) + list(adapter.parameters())),
            "optimizer_state_finite": _optimizer_state_finite(optimizer),
            "deployable_state": deployable_state_report(model),
            "trainability": trainability,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "fp16_used": False, "grad_scaler_enabled": False,
        }
        report["pass"] = bool(
            report["activations_and_losses_finite"] and report["gradients_finite"]
            and report["every_registered_component_nonzero"]
            and not frozen_gradients and not teacher_gradients
            and not report["teacher_outputs_require_grad"]
            and all(row_gradients_zero.values())
            and report["segmentation_row_restoration"]["all_exact"]
            and report["batch_norm_stats_unchanged"]
            and report["parameters_finite_after_step"] and report["optimizer_state_finite"]
            and report["deployable_state"]["no_teacher_or_projector_keys"]
            and report["peak_reserved_mib"] <= float(config["budget"]["vram_budget_gib"]) * 1024.0
        )
        return report
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        return {"policy": policy, "pass": False, "error": failure,
                "traceback": traceback.format_exc(), "fp16_used": False,
                "grad_scaler_enabled": False,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
    finally:
        del optimizer, adapter, model, batch
        torch.cuda.empty_cache()


def gpu_preflight(
    config: Mapping[str, Any], cpu_audit: Mapping[str, Any], hashes: Mapping[str, Any],
) -> Dict[str, Any]:
    if sys.executable != "/usr/bin/python3":
        raise ContractInvalid(f"registered interpreter is /usr/bin/python3, got {sys.executable}")
    if not torch.cuda.is_available():
        raise ContractInvalid("CUDA is unavailable for required numerical qualification")
    device = torch.device("cuda")
    set_reproducible_seeds(int(config["training_seed"]))
    teacher_cache = verify_teacher_cache()
    if not all(teacher_cache[key] for key in ("sha256_matches", "bytes_match", "url_matches_provenance")):
        raise ContractInvalid("official teacher cache provenance failure", teacher_cache)
    teacher = build_teacher(device)
    teacher_report = teacher_state(teacher)
    transform = verify_transform_identity(teacher, device)
    if not (teacher_report["all_frozen"] and not teacher_report["training_mode"]
            and transform["transform_is_identity"] and transform["image_sizes_are_model_size"]
            and transform["p2_stride_is_4"] and transform["p3_stride_is_8"]):
        raise ContractInvalid("teacher freeze/transform preflight failed",
                              {"teacher": teacher_report, "transform": transform})

    rows = read_manifest(SOURCE_VIEW / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    object_rows = load_object_boxes(SOURCE_VIEW / "dataset/object_boxes.csv")
    base_payload = torch.load(ROOT / config["student_warm_start"], map_location="cpu", weights_only=False)
    object_cfg = dict(base_payload["config"]["object_heads"])
    dataset = PersonCocoDistillationDataset(SOURCE_VIEW / "dataset", train_rows, object_rows,
                                             object_cfg, augment=True)
    reproduction = json.loads((ROOT / config["numerical_policy"]["previously_problematic_batch"]["source"]).read_text())
    identity = reproduction["sampler"]["batch_identity"]["134"]
    indices = [int(value) for value in identity["dataset_indices"]]
    observed_ids = [train_rows[index]["sample_id"] for index in indices]
    if observed_ids != identity["sample_ids"]:
        raise ContractInvalid("previously failing batch-134 identity drift")
    batch_cpu = person_distillation_collate([
        dataset[(index, int(config["training_seed"]) * 100000 + position)]
        for position, index in enumerate(indices)
    ])
    if sum(int(value.shape[0]) for value in batch_cpu["person_boxes"]) == 0:
        raise ContractInvalid("batch-134 numerical qualification has no person GT")
    reconstructed = (batch_cpu["student"][:, :3] * torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
                     + torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
    rgb_delta = float((reconstructed - batch_cpu["teacher_rgb01"]).abs().max().item())

    student, _ = build_student(ROOT / config["student_warm_start"], device)
    student.eval(); freeze_batch_norm(student)
    one = batch_cpu["student"][:1].to(device)
    parity = split_boundary_report(student, one)
    with torch.inference_mode():
        bundle, _outputs = forward_once(student, one, policy="full_fp32")
    real_boxes = torch.cat([value for value in batch_cpu["person_boxes"] if value.numel()], dim=0).to(device)
    teacher_shapes = transform["fpn_level_shapes"]
    levels = [
        {"name": "teacher_p2", "height": teacher_shapes["0"][-2], "width": teacher_shapes["0"][-1], "spatial_scale": 0.25},
        {"name": "teacher_p3", "height": teacher_shapes["1"][-2], "width": teacher_shapes["1"][-1], "spatial_scale": 0.125},
        {"name": "student_low", "height": bundle["low"].shape[-2], "width": bundle["low"].shape[-1], "spatial_scale": 0.125},
        {"name": "student_high", "height": (bundle["high"].shape[-2] + 1) // 2,
         "width": (bundle["high"].shape[-1] + 1) // 2, "spatial_scale": 0.03125},
    ]
    roi = verify_round_trip(levels=levels, real_boxes=real_boxes, output_size=7,
                            sampling_ratio=2, device=device)
    deploy = deployable_state_report(student)
    transport = {
        "keys": list(bundle.keys()),
        "shapes": {key: list(value.shape) for key, value in bundle.items()},
        "elements": {key: int(value.numel()) for key, value in bundle.items()},
        "estimated_bytes_fp32": sum(int(value.numel()) * 4 for value in bundle.values()),
        "tail_raw_modality_side_channels": [],
    }
    del student, bundle, one, real_boxes
    torch.cuda.empty_cache()
    structural_gates = {
        "geometry_contract": all(value for key, value in geometry_contract_probe().items()
                                 if key.endswith("unchanged") or key.startswith("off_canvas")),
        "teacher_frozen_eval": teacher_report["all_frozen"] and not teacher_report["training_mode"],
        "teacher_transform_identity": transform["transform_is_identity"],
        "roi_synthetic_and_real": roi["pass"],
        "monolithic_split_parity": parity["outputs_match"],
        "transport_keys": transport["keys"] == ["low", "high"],
        "no_raw_modality_side_channel": not transport["tail_raw_modality_side_channels"],
        "deployable_state_clean": deploy["no_teacher_or_projector_keys"] and deploy["tensor_count"] == 351,
        "teacher_rgb_retained_exactly": rgb_delta <= 2.5e-7,
        "variable_box_collate": len({int(value.shape[0]) for value in batch_cpu["person_boxes"]}) > 1,
        "test_inaccessible": cpu_audit["test_rows"] == 0 and not cpu_audit["test_payload_accessed"],
    }
    if not all(structural_gates.values()):
        raise ContractInvalid("structural preflight gate failed", structural_gates)

    # The registered half-batch teacher execution is frozen here, before epoch 1.
    chunk_size = int(config["batch_size"]) // 2
    qualifications = [
        _qualify_policy(policy, teacher, batch_cpu, config, device, chunk_size)
        for policy in config["numerical_policy"]["candidate_policies"]
    ]
    by_name = {item["policy"]: item for item in qualifications}
    preferred = config["numerical_policy"]["preferred"]
    if by_name.get(preferred, {}).get("pass"):
        selected = preferred
    elif by_name.get("full_fp32", {}).get("pass"):
        selected = "full_fp32"
    else:
        raise ContractInvalid("neither registered numerical policy qualified", qualifications)
    result = {
        "schema": "route_b_v3_1_person_coco_distillation_preflight_v1",
        "created_utc": utc_now(), "all_pass": True,
        "input_hashes": hashes, "cpu_contract": cpu_audit,
        "teacher_cache": teacher_cache, "teacher_state": teacher_report,
        "teacher_transform": transform, "geometry_probe": geometry_contract_probe(),
        "batch_134": {"dataset_indices": indices, "sample_ids": observed_ids,
                      "person_boxes": sum(int(value.shape[0]) for value in batch_cpu["person_boxes"]),
                      "teacher_rgb_reconstruction_max_abs_delta": rgb_delta},
        "roi_round_trip": roi, "split_parity": parity, "transport": transport,
        "deployable_state": deploy, "structural_gates": structural_gates,
        "numerical_candidates": qualifications, "selected_numerical_policy": selected,
        "teacher_execution": {"mode": "registered_half_batch_fallback", "chunk_size": chunk_size,
                              "student_batch_size": int(config["batch_size"]),
                              "reason": "preflight-frozen execution under the registered 12 GiB budget"},
        "fp16_used": False, "selected_before_epoch_1": True,
    }
    del teacher, batch_cpu, dataset
    torch.cuda.empty_cache()
    return result


class EpochPermutationSampler(Sampler[Any]):
    """Deterministic shuffle plus per-draw augmentation seeds for exact restart."""

    def __init__(self, size: int, seed: int) -> None:
        self.size, self.seed, self.epoch = int(size), int(seed), 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):  # noqa: ANN204
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for position, index in enumerate(order):
            augmentation_seed = self.seed * 1000003 + self.epoch * 200003 + position
            yield (int(index), int(augmentation_seed))

    def __len__(self) -> int:
        return self.size


def rng_states() -> Dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(states: Mapping[str, Any]) -> None:
    random.setstate(states["python"]); np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"]); torch.cuda.set_rng_state_all(states["torch_cuda"])


def lr_factor(epoch: int, batch_index: int, batches: int, epochs: int, minimum: float) -> float:
    if epoch == 1:
        return (batch_index + 1) / float(max(1, batches))
    total = max(1, (epochs - 1) * batches - 1)
    index = (epoch - 2) * batches + batch_index
    progress = index / float(total)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum) + (1.0 - float(minimum)) * cosine


def save_torch_x(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    return sha256(path)


def recovery_payload(
    *, model: torch.nn.Module, adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer, epoch: int, optimizer_steps: int,
    policy: str, config_hash: str, hashes: Mapping[str, Any], base: Mapping[str, Any],
) -> Dict[str, Any]:
    deploy = deployable_state_report(model)
    if not deploy["no_teacher_or_projector_keys"] or deploy["tensor_count"] != 351:
        raise RuntimeError("deployable state acquired teacher/projector keys")
    return {
        "schema": "route_b_v3_1_person_coco_distillation_recovery_v1",
        "model": model.state_dict(), "train_time_adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": {
            "kind": "one_warmup_epoch_then_cosine", "min_lr_ratio": 0.1,
            "optimizer_steps": int(optimizer_steps)},
        "epoch": int(epoch), "optimizer_steps": int(optimizer_steps), "rng_states": rng_states(),
        "numerical_policy": policy, "config_sha256": config_hash,
        "input_hashes": hashes, "deployable_state_report": deploy,
        "base_checkpoint": str(ROOT / CONFIG["student_warm_start"]),
        "base_checkpoint_sha256": CONFIG["student_warm_start_sha256"],
        "config": base["config"], "input_size": list(base["input_size"]),
        "radar_channels": int(base["radar_channels"]),
        "object_class_names": list(base["object_class_names"]),
        "object_output_channels": int(base["object_output_channels"]),
        "native_stride": int(base["native_stride"]), "native_grid": list(base["native_grid"]),
        "object_hidden_channels": int(base["object_hidden_channels"]),
        "object_head_depth": int(base["object_head_depth"]),
        "model_task": "segmentation_plus_native_grid_object_localization",
    }


def deploy_payload(
    model: torch.nn.Module, epoch: int, base: Mapping[str, Any], policy: str,
    config_hash: str, hashes: Mapping[str, Any], parameter_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    deploy = deployable_state_report(model)
    if not deploy["no_teacher_or_projector_keys"] or deploy["tensor_count"] != 351:
        raise RuntimeError("invalid deployable checkpoint state")
    return {
        "schema": "route_b_v3_1_person_coco_distillation_deployable_checkpoint_v1",
        "model": model.state_dict(), "epoch": int(epoch), "config": base["config"],
        "input_size": list(base["input_size"]), "radar_channels": int(base["radar_channels"]),
        "object_class_names": list(base["object_class_names"]),
        "object_output_channels": int(base["object_output_channels"]),
        "native_stride": int(base["native_stride"]), "native_grid": list(base["native_grid"]),
        "object_hidden_channels": int(base["object_hidden_channels"]),
        "object_head_depth": int(base["object_head_depth"]),
        "model_task": "segmentation_plus_native_grid_object_localization",
        "numerical_policy": policy, "config_sha256": config_hash, "input_hashes": hashes,
        "deployable_state_report": deploy, "parameter_counts": parameter_counts,
        "teacher_present": False, "adapter_present": False,
    }


@torch.inference_mode()
def validation_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, policy: str,
    config: Mapping[str, Any], class_weights: torch.Tensor,
) -> float:
    model.eval(); total = 0.0; batches = 0
    for batch_cpu in loader:
        batch = _device_batch(batch_cpu, device)
        _features, outputs = forward_once(model, batch["student"], policy=policy)
        loss, _parts, _segmentation = supervised_loss(
            outputs, batch["segmentation"], batch["targets"], class_weights=class_weights,
            loss_weights={"segmentation": config["existing_losses_unchanged"]["segmentation"],
                          "object_total": config["existing_losses_unchanged"]["object_total"],
                          "object": config["existing_losses_unchanged"]["object"]},
            lovasz_weight=float(config["existing_losses_unchanged"]["lovasz_weight"]),
        )
        total += float(loss.item()); batches += 1
    return total / max(1, batches)


def run_evaluation(
    experiment: Path, checkpoint: Path, checkpoint_hash: str, epoch: int,
    config: Mapping[str, Any], teacher: torch.nn.Module, device: torch.device,
) -> Dict[str, Any]:
    augmented_path = experiment / "decisions" / f"epoch_{epoch:03d}_evaluation.json"
    if augmented_path.is_file():
        return json.loads(augmented_path.read_text())
    tag = f"coco_distill_epoch_{epoch:03d}"
    prediction_root = experiment / "predictions" / tag
    teacher.cpu(); torch.cuda.empty_cache()
    try:
        score_path = experiment / "decisions" / f"epoch_{epoch:03d}_primary.json"
        # A reporting-only recovery reuses a fully completed, hash-verified inference
        # and primary score instead of violating the one-pass evaluation contract.
        if not (prediction_root / "INFERENCE_COMPLETE").is_file():
            inference = [sys.executable, str(NATIVE_PACKAGE / "infer_native_v1.py"),
                         "--experiment", str(experiment), "--checkpoint", str(checkpoint),
                         "--checkpoint-sha256", checkpoint_hash, "--tag", tag]
            if subprocess.run(inference, check=False).returncode != 0:
                raise RuntimeError(f"native inference failed at epoch {epoch}")
        if not score_path.is_file():
            scoring = [sys.executable, str(EXPANDED_PACKAGE / "score_continuation_v3.py"),
                       "--mode", "primary", "--experiment", str(experiment),
                       "--prediction-root", str(prediction_root), "--output", str(score_path),
                       "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_hash,
                       "--epoch", str(epoch)]
            if subprocess.run(scoring, check=False).returncode != 0:
                raise RuntimeError(f"fixed v0.10 scoring failed at epoch {epoch}")
        record = json.loads(score_path.read_text())
        inference_manifest = json.loads((prediction_root / "inference_manifest.json").read_text())
        if (inference_manifest["inference_pass_count"] != 1
                or inference_manifest["checkpoint_sha256"] != checkpoint_hash
                or sha256(prediction_root / "detections.csv") != inference_manifest["detections_sha256"]):
            raise RuntimeError(f"reused inference provenance failure at epoch {epoch}")
        record["checkpoint"] = str(checkpoint)
        record["checkpoint_sha256"] = checkpoint_hash
        record["prediction_root"] = str(prediction_root)
        record["person_slices_v010_at_0_20"] = person_slices(experiment, prediction_root, dict(config))
        record["baseline_deltas"] = baseline_deltas(record["metrics"], config["baseline_reference_epoch40"])
        record["baseline_deltas"]["vehicle_duplicate_fp"] = (
            int(record["vehicle_duplicate_fp"])
            - int(config["baseline_reference_epoch40"]["vehicle_duplicate_fp"])
        )
        baseline_decode = json.loads(BASELINE_DECODE.read_text())
        baseline_vehicle_count = (int(baseline_decode["metrics"]["vehicle_tp"])
                                  + int(baseline_decode["metrics"]["vehicle_fp"]))
        record["eligibility_gates"] = eligibility_gates(
            record, config, baseline_vehicle_count=baseline_vehicle_count)
        record["eligible"] = all(record["eligibility_gates"].values())
        record["material_gain"] = material_gain_gates(record, config)
        record["service_gates"] = service_gates(record, config)
        record["teacher_adoption_guard_fired"] = teacher_adoption_guard(record, config)
        record["catastrophic_gates"] = catastrophic_gates(record, config)
        json_x(augmented_path, record)
        return record
    finally:
        teacher.to(device)


def _training_loader(
    dataset: PersonCocoDistillationDataset, sampler: Sampler[Any], config: Mapping[str, Any],
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), sampler=sampler, drop_last=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=int(config["prefetch_factor"]), collate_fn=person_distillation_collate,
    )


def _validation_loader(
    dataset: PersonCocoDistillationDataset, config: Mapping[str, Any],
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False, drop_last=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=int(config["prefetch_factor"]), collate_fn=person_distillation_collate,
    )


def train_attempt(
    experiment: Path, config: Mapping[str, Any], preflight: Mapping[str, Any],
    hashes: Mapping[str, Any], *, attempt: int, resume: Path | None,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    policy = str(preflight["selected_numerical_policy"])
    chunk_size = int(preflight["teacher_execution"]["chunk_size"])
    config_hash = hashes["registered_config"]["sha256"]
    set_reproducible_seeds(int(config["training_seed"]))
    rows = read_manifest(experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    model, base = build_student(ROOT / config["student_warm_start"], device)
    trainability = configure_trainable(model)
    adapter = StudentRoiAdapter().to(device).float()
    adapter_counts = adapter_report(adapter)
    row_guard = SegmentationRowGuard(model)
    optimizer = torch.optim.AdamW(
        registered_parameter_groups(model, adapter), lr=0.0,
        weight_decay=float(config["weight_decay"]),
    )
    teacher = build_teacher(device)
    teacher.eval()
    class_weights = torch.tensor(config["existing_losses_unchanged"]["class_loss_weights"],
                                 dtype=torch.float32, device=device)
    training = PersonCocoDistillationDataset(experiment / "dataset", train_rows, object_rows,
                                              dict(base["config"]["object_heads"]), augment=True)
    validation = PersonCocoDistillationDataset(experiment / "dataset", val_rows, object_rows,
                                                dict(base["config"]["object_heads"]), augment=False)
    sampler = EpochPermutationSampler(len(training), int(config["training_seed"]))
    train_loader = _training_loader(training, sampler, config)
    val_loader = _validation_loader(validation, config)
    start_epoch, optimizer_steps = 1, 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if (payload.get("config_sha256") != config_hash
                or payload.get("numerical_policy") != policy
                or payload.get("base_checkpoint_sha256") != config["student_warm_start_sha256"]):
            raise RuntimeError("recovery checkpoint registration mismatch")
        model.load_state_dict(payload["model"], strict=True)
        adapter.load_state_dict(payload["train_time_adapter"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng_states"])
        start_epoch = int(payload["epoch"]) + 1
        optimizer_steps = int(payload["optimizer_steps"])
        row_guard.restore()

    metrics_path = experiment / "metrics/training_metrics.csv"
    fields = [
        "epoch", "attempt", "train_total_loss", "validation_total_loss", "supervised_loss",
        "l_feat", "l_obj", "l_feat_mean_cosine", "l_feat_rois", "l_obj_sites",
        "l_obj_mean_teacher_score", "teacher_gt_boxes", "teacher_evidence_boxes",
        "teacher_missed_boxes", "teacher_false_positive_omissions", "lr_factor_start",
        "lr_factor_end", "optimizer_steps", "epoch_seconds", "peak_allocated_mib",
        "peak_reserved_mib", "geometric_samples", "created_utc",
    ]
    if start_epoch == 1 and not metrics_path.exists():
        with metrics_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    checkpoint_dir = experiment / "checkpoints" / str(config["name"])
    recovery_dir = experiment / "recovery_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        epoch0 = recovery_dir / "epoch_000.pt"
        save_torch_x(epoch0, recovery_payload(
            model=model, adapter=adapter, optimizer=optimizer, epoch=0,
            optimizer_steps=0, policy=policy, config_hash=config_hash,
            hashes=hashes, base=base,
        ))
        json_atomic(experiment / "LATEST_SAFE.json", {"epoch": 0, "path": str(epoch0),
                                                       "sha256": sha256(epoch0)})

    total_epochs = int(config["epochs"])
    checkpoint_epochs = {int(value) for value in config["checkpoint_epochs"]}
    evaluation_epochs = {int(value) for value in config["evaluation_epochs"]}
    if total_epochs != 18 or checkpoint_epochs != {6, 12, 18} or evaluation_epochs != {6, 12, 18}:
        raise RuntimeError("registered epoch/checkpoint schedule drift")
    records: list[Dict[str, Any]] = []
    # A retry can enter after a designated checkpoint was saved but before its fixed
    # evaluation completed. Complete that boundary before taking another train step.
    previous_epoch = start_epoch - 1
    if previous_epoch in evaluation_epochs:
        checkpoint = checkpoint_dir / f"epoch_{previous_epoch:03d}.pt"
        if checkpoint.is_file():
            records.append(run_evaluation(experiment, checkpoint, sha256(checkpoint),
                                          previous_epoch, config, teacher, device))

    wall_start = time.monotonic()
    peak_allocated = peak_reserved = 0.0
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.monotonic()
        sampler.set_epoch(epoch)
        enforce_train_mode(model); adapter.train(); teacher.eval()
        before_bn = batch_norm_snapshot(model)
        torch.cuda.reset_peak_memory_stats(device)
        sums: Dict[str, float] = {}
        evidence_sums = {"gt_person_boxes": 0, "gt_person_boxes_with_teacher_evidence": 0,
                         "gt_person_boxes_without_teacher_evidence": 0,
                         "omitted_teacher_positive_gt_absent": 0}
        batches = geometric_samples = 0
        first_factor = last_factor = 0.0
        for batch_index, batch_cpu in enumerate(train_loader):
            factor = lr_factor(epoch, batch_index, len(train_loader), total_epochs,
                               float(config["min_lr_ratio"]))
            first_factor = factor if batches == 0 else first_factor; last_factor = factor
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * factor
            batch = _device_batch(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts, evidence = _loss_bundle(
                model, adapter, teacher, batch, config, policy, chunk_size, class_weights,
            )
            if not math.isfinite(float(loss.detach().item())):
                raise RuntimeError(f"nonfinite loss epoch={epoch} batch={batch_index + 1}")
            loss.backward()
            bad_gradients = [name for name, parameter in list(model.named_parameters())
                             if parameter.requires_grad and parameter.grad is not None
                             and not bool(torch.isfinite(parameter.grad).all().item())]
            bad_adapter = [name for name, parameter in adapter.named_parameters()
                           if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())]
            if bad_gradients or bad_adapter:
                raise RuntimeError(f"nonfinite gradients: student={bad_gradients} adapter={bad_adapter}")
            optimizer.step(); row_guard.restore(); optimizer_steps += 1
            if not row_guard.report()["all_exact"]:
                raise RuntimeError("segmentation non-person rows drifted")
            batches += 1
            geometric_samples += sum(bool(item.get("applied")) for item in batch_cpu["geometry"])
            for key, value in parts.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    sums[key] = sums.get(key, 0.0) + float(value)
            for key in evidence_sums:
                evidence_sums[key] += int(evidence.get(key, 0))
        if not snapshots_equal(before_bn, batch_norm_snapshot(model)):
            raise RuntimeError(f"frozen BatchNorm statistics changed at epoch {epoch}")
        if not finite_parameter_tree(list(model.parameters()) + list(adapter.parameters())):
            raise RuntimeError(f"nonfinite parameter state at epoch {epoch}")
        if not _optimizer_state_finite(optimizer):
            raise RuntimeError(f"nonfinite optimizer state at epoch {epoch}")
        validation_value = validation_loss(model, val_loader, device, policy, config, class_weights)
        allocated = torch.cuda.max_memory_allocated(device) / 2**20
        reserved = torch.cuda.max_memory_reserved(device) / 2**20
        peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
        row = {
            "epoch": epoch, "attempt": attempt,
            "train_total_loss": sums.get("total_loss", 0.0) / max(1, batches),
            "validation_total_loss": validation_value,
            "supervised_loss": sums.get("supervised_loss", 0.0) / max(1, batches),
            "l_feat": sums.get("l_feat", 0.0) / max(1, batches),
            "l_obj": sums.get("l_obj", 0.0) / max(1, batches),
            "l_feat_mean_cosine": sums.get("l_feat_mean_cosine", 0.0) / max(1, batches),
            "l_feat_rois": sums.get("l_feat_rois", 0.0),
            "l_obj_sites": sums.get("l_obj_sites", 0.0),
            "l_obj_mean_teacher_score": sums.get("l_obj_mean_teacher_score", 0.0) / max(1, batches),
            "teacher_gt_boxes": evidence_sums["gt_person_boxes"],
            "teacher_evidence_boxes": evidence_sums["gt_person_boxes_with_teacher_evidence"],
            "teacher_missed_boxes": evidence_sums["gt_person_boxes_without_teacher_evidence"],
            "teacher_false_positive_omissions": evidence_sums["omitted_teacher_positive_gt_absent"],
            "lr_factor_start": first_factor, "lr_factor_end": last_factor,
            "optimizer_steps": optimizer_steps, "epoch_seconds": time.monotonic() - epoch_start,
            "peak_allocated_mib": allocated, "peak_reserved_mib": reserved,
            "geometric_samples": geometric_samples, "created_utc": utc_now(),
        }
        if not all(math.isfinite(float(row[key])) for key in (
                "train_total_loss", "validation_total_loss", "l_feat", "l_obj")):
            raise RuntimeError(f"nonfinite epoch aggregate at epoch {epoch}")
        with metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        json_x(experiment / "metrics" / f"epoch_{epoch:03d}.json", row)
        recovery = recovery_dir / f"epoch_{epoch:03d}.pt"
        recovery_hash = save_torch_x(recovery, recovery_payload(
            model=model, adapter=adapter, optimizer=optimizer, epoch=epoch,
            optimizer_steps=optimizer_steps, policy=policy, config_hash=config_hash,
            hashes=hashes, base=base,
        ))
        json_atomic(experiment / "LATEST_SAFE.json", {"epoch": epoch, "path": str(recovery),
                                                       "sha256": recovery_hash})
        append_progress(experiment, phase="training", event="epoch_complete", attempt=attempt,
                        epoch=epoch, train_total_loss=row["train_total_loss"],
                        validation_total_loss=validation_value, l_feat=row["l_feat"],
                        l_obj=row["l_obj"], feature_cosine=row["l_feat_mean_cosine"],
                        objectness_sites=row["l_obj_sites"], wall_seconds=time.monotonic()-wall_start)
        update_status(experiment, phase="training", state="running", attempt=attempt,
                      epoch=epoch, epochs_total=total_epochs, optimizer_steps=optimizer_steps,
                      latest_recovery=str(recovery))
        print(f"[coco distill] epoch={epoch}/18 loss={row['train_total_loss']:.6f} "
              f"val={validation_value:.6f} feat={row['l_feat']:.6f} obj={row['l_obj']:.6f}", flush=True)

        if epoch in checkpoint_epochs:
            checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            checkpoint_hash = save_torch_x(checkpoint, deploy_payload(
                model, epoch, base, policy, config_hash, hashes,
                {"student": trainability, "adapter": adapter_counts},
            ))
            if epoch in evaluation_epochs:
                update_status(experiment, phase="evaluation", state="running", epoch=epoch)
                record = run_evaluation(experiment, checkpoint, checkpoint_hash,
                                        epoch, config, teacher, device)
                records.append(record)
                metric = record["metrics"]
                append_progress(experiment, phase="evaluation", event="evaluation_complete",
                                attempt=attempt, epoch=epoch, person_f1=metric["person_f1"],
                                person_recall=metric["person_recall"], vehicle_f1=metric["vehicle_f1"],
                                wall_seconds=time.monotonic()-wall_start)
                if record["teacher_adoption_guard_fired"]:
                    raise TeacherNotAdopted(f"teacher-adoption guard fired at epoch {epoch}")
                if epoch == int(config["catastrophic_limits"]["applied_at_epoch"]):
                    if not all(record["catastrophic_gates"].values()):
                        raise CatastrophicStop("registered catastrophic gate failed at epoch 12")
        if time.monotonic() - wall_start > float(config["budget"]["estimated_wall_seconds"]) * float(config["budget"]["abort_multiple"]):
            raise RuntimeError("registered 2x wall-time budget exceeded")

    return {
        "epochs_completed": total_epochs, "optimizer_steps": optimizer_steps,
        "evaluation_records": records, "trainability": trainability,
        "adapter": adapter_counts, "wall_seconds": time.monotonic() - wall_start,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        "teacher_execution_train_frames": total_epochs * len(train_rows),
        "teacher_execution_validation_frames": 0, "teacher_execution_test_frames": 0,
    }


def setup_experiment(requested: Path | None) -> tuple[Path, Dict[str, Any]]:
    if requested is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = ROOT / "experiments/route_b_v3_1_person_coco_distillation_v1" / timestamp
    else:
        experiment = requested.resolve()
    experiment.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "metrics", "decisions", "predictions", "resolved_configs"):
        (experiment / name).mkdir()
    os.symlink(SOURCE_VIEW / "dataset", experiment / "dataset", target_is_directory=True)
    os.symlink(SOURCE_VIEW / "contracts", experiment / "contracts", target_is_directory=True)
    shutil.copy2(CONFIG_PATH, experiment / "resolved_configs/person_coco_distillation_v1.json")
    with (experiment / "PROGRESS.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writeheader()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    git = {"starting_head": head, "branch": branch, "starting_status": status,
           "required_ancestor": "a5af817a567553d6e50f41bc98b7ab1ff7dca2db"}
    if branch != "master" or subprocess.run(
        ["git", "merge-base", "--is-ancestor", git["required_ancestor"], "HEAD"],
        cwd=ROOT, check=False,
    ).returncode != 0:
        raise ContractInvalid("repository lineage is not the registered local master")
    json_x(experiment / "PIPELINE_STARTED.json", {
        "schema": "route_b_v3_1_person_coco_distillation_pipeline_started_v1",
        "created_utc": utc_now(), "experiment": str(experiment), "git": git,
        "single_scientific_attempt": True, "q": 0, "ae": False,
    })
    json_atomic(experiment / "STATUS.json", {
        "schema": "route_b_v3_1_person_coco_distillation_status_v1",
        "phase": "preflight", "state": "running", "epoch": 0, "epochs_total": 18,
        "attempt": 0, "experiment": str(experiment), "starting_head": head,
    })
    append_progress(experiment, phase="preflight", event="pipeline_started", attempt=0,
                    epoch=0, detail=head)
    return experiment, git


def prepare_reporting_recovery(experiment: Path, source: Path) -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Carry a verified epoch boundary into a fresh create-only supervisor directory."""
    source = source.resolve(strict=True)
    required = [source / name for name in (
        "INPUT_HASHES.json", "PREFLIGHT.json", "NUMERICAL_POLICY_REGISTRATION.json", "LATEST_SAFE.json")]
    if any(not path.is_file() for path in required):
        raise ContractInvalid("reporting-recovery source is incomplete",
                              {"missing": [str(path) for path in required if not path.is_file()]})
    hashes = json.loads((source / "INPUT_HASHES.json").read_text())
    preflight = json.loads((source / "PREFLIGHT.json").read_text())
    if not preflight.get("all_pass"):
        raise ContractInvalid("cannot recover from a non-green preflight")
    latest = json.loads((source / "LATEST_SAFE.json").read_text())
    epoch = int(latest["epoch"]); source_recovery = Path(latest["path"])
    if epoch <= 0 or not source_recovery.is_file() or sha256(source_recovery) != latest["sha256"]:
        raise ContractInvalid("reporting-recovery checkpoint provenance failure")
    for name in ("INPUT_HASHES.json", "PREFLIGHT.json", "NUMERICAL_POLICY_REGISTRATION.json"):
        shutil.copy2(source / name, experiment / name)
    shutil.copy2(source / "metrics/training_metrics.csv", experiment / "metrics/training_metrics.csv")
    for path in sorted((source / "metrics").glob("epoch_*.json")):
        shutil.copy2(path, experiment / "metrics" / path.name)
    target_recovery = experiment / "recovery_checkpoints" / f"epoch_{epoch:03d}.pt"
    target_recovery.parent.mkdir(parents=True)
    os.link(source_recovery, target_recovery)
    source_checkpoint = source / "checkpoints" / str(CONFIG["name"]) / f"epoch_{epoch:03d}.pt"
    target_checkpoint = experiment / "checkpoints" / str(CONFIG["name"]) / f"epoch_{epoch:03d}.pt"
    target_checkpoint.parent.mkdir(parents=True)
    os.link(source_checkpoint, target_checkpoint)
    tag = f"coco_distill_epoch_{epoch:03d}"
    os.symlink(source / "predictions" / tag, experiment / "predictions" / tag, target_is_directory=True)
    shutil.copy2(source / "decisions" / f"epoch_{epoch:03d}_primary.json",
                 experiment / "decisions" / f"epoch_{epoch:03d}_primary.json")
    json_atomic(experiment / "LATEST_SAFE.json", {"epoch": epoch, "path": str(target_recovery),
                                                   "sha256": sha256(target_recovery)})
    json_x(experiment / "REPORTING_RECOVERY_ORIGIN.json", {
        "schema": "route_b_v3_1_person_coco_distillation_reporting_recovery_v1",
        "created_utc": utc_now(), "source_experiment": str(source), "epoch": epoch,
        "recovery_checkpoint": str(target_recovery), "recovery_sha256": sha256(target_recovery),
        "deployable_checkpoint": str(target_checkpoint), "deployable_sha256": sha256(target_checkpoint),
        "inference_reused": str(experiment / "predictions" / tag),
        "reason": "resume after a post-scoring Python namespace/report enrichment bug; no scientific change",
    })
    update_status(experiment, phase="reporting_recovery", state="resuming", attempt=2,
                  epoch=epoch, numerical_policy=preflight["selected_numerical_policy"])
    append_progress(experiment, phase="reporting_recovery", event="verified_epoch_resume",
                    attempt=2, epoch=epoch, detail=str(source))
    return hashes, preflight, target_recovery


def _checkpoint_records(experiment: Path, config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    values = []
    for epoch in config["evaluation_epochs"]:
        path = experiment / "decisions" / f"epoch_{int(epoch):03d}_evaluation.json"
        if path.is_file():
            values.append(json.loads(path.read_text()))
    return values


def run_sensitivity(experiment: Path, selected: Mapping[str, Any]) -> Dict[str, Any]:
    output = experiment / "decisions/SELECTED_V025_SENSITIVITY.json"
    command = [sys.executable, str(EXPANDED_PACKAGE / "score_continuation_v3.py"),
               "--mode", "sensitivity", "--experiment", str(experiment),
               "--prediction-root", str(selected["prediction_root"]), "--output", str(output)]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError("selected-only fixed v0.25 sensitivity scoring failed")
    return json.loads(output.read_text())


def make_decision(
    experiment: Path, config: Mapping[str, Any], preflight: Mapping[str, Any],
    training: Mapping[str, Any] | None, *, forced_terminal: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    records = _checkpoint_records(experiment, config)
    eligible = sorted([record for record in records if record.get("eligible")], key=rank_key)
    selected = eligible[0] if eligible else None
    sensitivity = run_sensitivity(experiment, selected) if selected is not None else None
    material = selected.get("material_gain") if selected else None
    service = selected.get("service_gates") if selected else None
    if forced_terminal is not None:
        terminal = forced_terminal
    elif selected is not None and material and material["pass"]:
        terminal = ("LRASPP_COCO_DISTILLATION_SERVICE_READY" if all(service.values())
                    else "LRASPP_COCO_DISTILLATION_MATERIAL_GAIN")
    else:
        terminal = "LRASPP_COCO_DISTILLATION_NO_GAIN"
    if terminal not in TERMINALS:
        raise RuntimeError(f"unregistered terminal: {terminal}")
    checkpoint_dir = experiment / "checkpoints" / str(config["name"])
    retained = []
    for epoch in config["checkpoint_epochs"]:
        path = checkpoint_dir / f"epoch_{int(epoch):03d}.pt"
        if path.is_file():
            retained.append({"epoch": int(epoch), "path": str(path), "sha256": sha256(path)})
    return {
        "schema": "route_b_v3_1_person_coco_distillation_decision_v1",
        "created_utc": utc_now(), "terminal": terminal,
        "epochs_completed": int(training.get("epochs_completed", 0)) if training else (
            max([int(item["epoch"]) for item in records], default=0)),
        "evaluation_epochs_completed": [int(item["epoch"]) for item in records],
        "records": records, "eligible_epochs": [int(item["epoch"]) for item in eligible],
        "ranking": [int(item["epoch"]) for item in eligible],
        "non_dominated_epochs": nondominated(records),
        "selected": ({"epoch": int(selected["epoch"]), "checkpoint": selected["checkpoint"],
                      "checkpoint_sha256": selected["checkpoint_sha256"],
                      "metrics_v010": selected["metrics"], "baseline_deltas": selected["baseline_deltas"]}
                     if selected else None),
        "selected_v025": sensitivity, "material_gain": material, "service_gates": service,
        "teacher_adoption_guard_fired": any(item.get("teacher_adoption_guard_fired") for item in records),
        "retained_checkpoints": retained, "training": training, "preflight": {
            "selected_numerical_policy": preflight.get("selected_numerical_policy"),
            "teacher_execution": preflight.get("teacher_execution"),
            "all_pass": preflight.get("all_pass", False)},
        "failure": failure, "q": 0, "ae": False, "q_ae_eligible": False,
        "q_ae_reason": "clean noAE verdict is terminal; q/AE continuation is explicitly forbidden",
    }


def _metric_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| Epoch | Vehicle P/R/F1 | Person P/R/F1 | V/P XY m | V/P IoU | FG mIoU |",
             "|---:|---|---|---|---|---:|"]
    for record in records:
        m = record["metrics"]
        lines.append(
            f"| {record['epoch']} | {m['vehicle_precision']:.4f}/{m['vehicle_recall']:.4f}/{m['vehicle_f1']:.4f} "
            f"| {m['person_precision']:.4f}/{m['person_recall']:.4f}/{m['person_f1']:.4f} "
            f"| {m['vehicle_xy_mae_m']:.4f}/{m['person_xy_mae_m']:.4f} "
            f"| {m['vehicle_iou']:.4f}/{m['person_box_mask_iou']:.4f} | {m['foreground_miou']:.4f} |"
        )
    return "\n".join(lines) if records else "No checkpoint evaluation was authorized."


def write_report(
    experiment: Path, decision: Mapping[str, Any], hashes: Mapping[str, Any],
    git: Mapping[str, Any], preflight: Mapping[str, Any], started: float,
) -> None:
    records = decision.get("records", [])
    selected = decision.get("selected")
    training = decision.get("training") or {}
    trajectory = []
    metrics_csv = experiment / "metrics/training_metrics.csv"
    if metrics_csv.is_file():
        trajectory = list(csv.DictReader(metrics_csv.open(encoding="utf-8", newline="")))
    report = f"""# LR-ASPP COCO Person Distillation Final Report

Terminal: `{decision['terminal']}`

## Contract and provenance

- Starting HEAD: `{git['starting_head']}` on `{git['branch']}`; required commit `a5af817` is an ancestor.
- Scientific config SHA-256: `{hashes.get('registered_config', {}).get('sha256')}` (unchanged).
- Student SHA-256: `{hashes.get('student_warm_start', {}).get('sha256')}`.
- Teacher SHA-256: `{hashes.get('teacher_cache', {}).get('sha256')}`; cached official COCO weight, no download.
- Dataset manifest/object-box SHA-256: `{hashes.get('dataset_manifest', {}).get('sha256')}` / `{hashes.get('dataset_object_boxes', {}).get('sha256')}`.
- Dataset: 16,827 train / 3,345 validation / 0 test; train/validation episodes disjoint; locked test unopened.
- q=0, AE=false. No CARLA, OAI, live-split, q/AE, or 288-measurement action was run.

## Preflight and implementation

- Numerical policy: `{preflight.get('selected_numerical_policy')}`; FP16 was never used; GradScaler disabled.
- Teacher execution: `{json.dumps(preflight.get('teacher_execution'))}`.
- Teacher is frozen/eval/no-grad, RGB-only, and executed on train only. Validation/test teacher frames: 0/0.
- Transform identity, synthetic+real ROI round trip, monolithic/split parity, `{{low, high}}` transport, no-side-channel, frozen-BN, allowlist, person-row masking/restoration, baseline reconciliation, clean deployable state, batch-134 finite qualification, and test-inaccessibility gates passed before epoch 1.
- Claude's teacher/ROI/distillation modules were retained. Completed code adds joint affine dataset/collation, one-pass student integration, preflight, bounded training/evaluation, gates, recovery, status/progress, report, sentinel, and notification.

## Fixed v0.10 checkpoint results

{_metric_table(records)}

Each record in `DECISION.json` also contains TP/FP/FN/neutral counts, recall@0.02, dimension/yaw errors, duplicate-FP and person-FN taxonomy, person area/distance/radar-support slices, baseline deltas, and exact eligibility/material/service gates. Exactly one inference pass at floor 0.02 produced each checkpoint's predictions; score 0.20 was derived offline.

## Distillation and resource trajectory

- Epoch rows recorded: {len(trajectory)}. Final feature cosine: {trajectory[-1].get('l_feat_mean_cosine') if trajectory else None}; final objectness sites: {trajectory[-1].get('l_obj_sites') if trajectory else None}.
- Deployable trainable/frozen parameters: {training.get('trainability', {}).get('trainable_parameters')} / {training.get('trainability', {}).get('frozen_parameters')}.
- Train-time-only adapter parameters: {training.get('adapter', {}).get('parameters')}.
- Runtime: {training.get('wall_seconds', time.monotonic()-started)} s; peak allocated/reserved: {training.get('peak_allocated_mib')} / {training.get('peak_reserved_mib')} MiB.
- Bundle: {json.dumps(preflight.get('transport'))}.

## Decision

- Eligible epochs: {decision.get('eligible_epochs')}; non-dominated epochs retained: {decision.get('non_dominated_epochs')}.
- Selected checkpoint: {json.dumps(selected)}.
- Material gain: {json.dumps(decision.get('material_gain'))}.
- Service gates: {json.dumps(decision.get('service_gates'))}.
- Teacher-adoption guard fired: {decision.get('teacher_adoption_guard_fired')}.
- Retained designated checkpoints: {json.dumps(decision.get('retained_checkpoints'))}.
- q/AE eligibility: **NO**. The registered recovery contract forbids q/AE continuation; this is the clean noAE terminal.

Registered caveat: a gain would not isolate pretraining from extra supervision signal; a randomized-teacher control was deliberately excluded.
"""
    (experiment / "LRASPP_COCO_DISTILLATION_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def notify(experiment: Path, terminal: str) -> Dict[str, Any]:
    command = ["notify-send", "LR-ASPP COCO distillation complete", f"{terminal}\n{experiment}"]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def complete(
    experiment: Path, decision: Mapping[str, Any], hashes: Mapping[str, Any],
    git: Mapping[str, Any], preflight: Mapping[str, Any], started: float,
) -> None:
    terminal = str(decision["terminal"])
    json_x(experiment / "DECISION.json", decision)
    write_report(experiment, decision, hashes, git, preflight, started)
    update_status(experiment, phase="complete", state="terminal", terminal=terminal,
                  epoch=decision.get("epochs_completed", 0), completed_utc=utc_now())
    append_progress(experiment, phase="complete", event="terminal", attempt="",
                    epoch=decision.get("epochs_completed", 0), detail=terminal,
                    wall_seconds=time.monotonic()-started)
    (experiment / "COMPLETION_SENTINEL").write_text(terminal + "\n", encoding="utf-8")
    json_x(experiment / "NOTIFICATION.json", notify(experiment, terminal))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--resume-source", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment, git = setup_experiment(args.experiment)
    hashes: Dict[str, Any] = {}
    preflight: Dict[str, Any] = {"all_pass": False}
    initial_resume: Path | None = None
    initial_attempt = 1
    try:
        if args.resume_source is not None:
            hashes, preflight, initial_resume = prepare_reporting_recovery(
                experiment, args.resume_source)
            initial_attempt = 2
        else:
            hashes = input_hashes(CONFIG)
            json_x(experiment / "INPUT_HASHES.json", hashes)
            cpu = cpu_contract_audit(CONFIG, hashes)
            preflight = gpu_preflight(CONFIG, cpu, hashes)
            json_x(experiment / "PREFLIGHT.json", preflight)
            json_x(experiment / "NUMERICAL_POLICY_REGISTRATION.json", {
            "schema": "route_b_v3_1_person_coco_distillation_numerical_policy_v1",
            "created_utc": utc_now(), "frozen_before_epoch_1": True,
            "selected": preflight["selected_numerical_policy"],
            "candidate_results": preflight["numerical_candidates"],
            "teacher_execution": preflight["teacher_execution"],
            "config_sha256": hashes["registered_config"]["sha256"],
            })
        update_status(experiment, phase="training", state="running", attempt=initial_attempt,
                      epoch=(int(torch.load(initial_resume, map_location="cpu", weights_only=False)["epoch"])
                             if initial_resume else 0),
                      numerical_policy=preflight["selected_numerical_policy"])
        append_progress(experiment, phase="preflight", event="preflight_passed", attempt=0,
                        epoch=0, detail=preflight["selected_numerical_policy"])
    except ContractInvalid as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "details": exc.details,
                   "traceback": traceback.format_exc()}
        if not (experiment / "PREFLIGHT.json").exists():
            json_x(experiment / "PREFLIGHT.json", {"all_pass": False, "failure": failure,
                                                    "input_hashes": hashes})
        decision = make_decision(experiment, CONFIG, preflight, None,
                                 forced_terminal="LRASPP_COCO_DISTILLATION_CONTRACT_INVALID",
                                 failure=failure)
        complete(experiment, decision, hashes, git, preflight, started)
        print(json.dumps({"terminal": decision["terminal"], "experiment": str(experiment),
                          "failure": failure}, indent=2), flush=True)
        return 2

    training: Dict[str, Any] | None = None
    forced_terminal: str | None = None
    failure: Dict[str, Any] | None = None
    attempt = initial_attempt
    resume: Path | None = initial_resume
    transient_retries = 0
    while True:
        try:
            training = train_attempt(experiment, CONFIG, preflight, hashes,
                                     attempt=attempt, resume=resume)
            break
        except TeacherNotAdopted as exc:
            forced_terminal = "LRASPP_COCO_DISTILLATION_TEACHER_NOT_ADOPTED"
            failure = {"type": type(exc).__name__, "message": str(exc)}
            break
        except CatastrophicStop as exc:
            forced_terminal = "LRASPP_COCO_DISTILLATION_NO_GAIN"
            failure = {"type": type(exc).__name__, "message": str(exc)}
            break
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            transient = any(token in message.lower() for token in (
                "cuda", "cudnn", "resource temporarily unavailable", "input/output error", "i/o error"))
            latest_path = experiment / "LATEST_SAFE.json"
            if transient and transient_retries < 1 and latest_path.is_file():
                latest = json.loads(latest_path.read_text())
                resume = Path(latest["path"])
                if not resume.is_file() or sha256(resume) != latest["sha256"]:
                    transient = False
                else:
                    transient_retries += 1
                    attempt += 1
                    append_progress(experiment, phase="recovery", event="single_transient_retry",
                                    attempt=attempt, epoch=latest["epoch"], detail=message)
                    update_status(experiment, phase="recovery", state="retrying", attempt=attempt,
                                  epoch=latest["epoch"], latest_recovery=str(resume))
                    torch.cuda.empty_cache()
                    continue
            forced_terminal = "LRASPP_COCO_DISTILLATION_RUNTIME_FAILURE"
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "transient": transient, "attempt": attempt,
                       "traceback": traceback.format_exc()}
            break
    try:
        decision = make_decision(experiment, CONFIG, preflight, training,
                                 forced_terminal=forced_terminal, failure=failure)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "stage": "selection_or_sensitivity", "traceback": traceback.format_exc()}
        decision = make_decision(experiment, CONFIG, preflight, training,
                                 forced_terminal="LRASPP_COCO_DISTILLATION_RUNTIME_FAILURE",
                                 failure=failure)
    complete(experiment, decision, hashes, git, preflight, started)
    print(json.dumps({"terminal": decision["terminal"], "experiment": str(experiment),
                      "epochs_completed": decision["epochs_completed"],
                      "selected": decision["selected"]}, indent=2), flush=True)
    return 0 if decision["terminal"] not in {
        "LRASPP_COCO_DISTILLATION_CONTRACT_INVALID", "LRASPP_COCO_DISTILLATION_RUNTIME_FAILURE"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
