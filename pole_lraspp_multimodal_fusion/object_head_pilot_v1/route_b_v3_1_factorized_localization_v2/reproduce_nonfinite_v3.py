#!/usr/bin/env python3
"""Deterministically reproduce and identify the first v2 non-finite operation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
NATIVE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
for path in (str(PACKAGE_ROOT), str(NATIVE_PKG), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from model_v1 import NATIVE_STRIDE, SL_OFFSET  # noqa: E402
from model_v2 import (  # noqa: E402
    build_factorized_model, freeze_for_localization, load_native_warm_start,
    localization_parameters,
)
from targets_v2 import FactorizedLocalizationDataset  # noqa: E402

RECORD_EPOCH = 2
RECORD_FIRST_BATCH = 130


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


class WithSampleId(Dataset):
    """Attach provenance without changing the wrapped dataset's access order or RNG."""

    def __init__(self, dataset: FactorizedLocalizationDataset, rows: list[dict[str, str]]) -> None:
        self.dataset = dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        tensors, masks, targets = self.dataset[index]
        return tensors, masks, targets, self.rows[index]["sample_id"]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    output: dict[str, Any] = {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape), "numel": int(value.numel()),
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "positive_inf_count": int(torch.isposinf(value).sum().item()),
        "negative_inf_count": int(torch.isneginf(value).sum().item()),
        "all_finite": bool(finite.all().item()),
    }
    if finite_values.numel():
        output.update({
            "finite_min": float(finite_values.min().item()),
            "finite_max": float(finite_values.max().item()),
            "finite_mean": float(finite_values.mean().item()),
        })
    else:
        output.update({"finite_min": None, "finite_max": None, "finite_mean": None})
    return output


def masked_values(tensor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[positive]
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1)[positive]
    raise ValueError(f"unsupported masked tensor rank {tensor.ndim}")


def module_parameter_stats(model) -> dict[str, Any]:
    modules = {
        "localization_trunk": model.localization_trunk,
        "log_depth_head": model.log_depth_head,
        "projected_3d_center_offset_head": model.projected_3d_center_offset_head,
    }
    return {
        name: tensor_stats(torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()]))
        for name, module in modules.items()
    }


def module_gradient_stats(model) -> dict[str, Any]:
    modules = {
        "localization_trunk": model.localization_trunk,
        "log_depth_head": model.log_depth_head,
        "projected_3d_center_offset_head": model.projected_3d_center_offset_head,
    }
    output: dict[str, Any] = {}
    for name, module in modules.items():
        gradients = [parameter.grad.detach().reshape(-1) for parameter in module.parameters()
                     if parameter.grad is not None]
        if not gradients:
            output[name] = {"gradient_tensor_count": 0, "all_finite": True, "l2_norm": 0.0}
            continue
        combined = torch.cat(gradients)
        stats = tensor_stats(combined)
        stats["gradient_tensor_count"] = len(gradients)
        stats["l2_norm"] = float(torch.linalg.vector_norm(combined.float()).item())
        output[name] = stats
    return output


def geometry_and_losses(
    localization: torch.Tensor, legacy_object: torch.Tensor, targets: Dict[str, torch.Tensor],
    weights: Dict[str, Any], camera_matrices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    raw_log_depth = localization[:, 0]
    predicted_offset = localization[:, 1:3]
    target_log_depth = targets["factorized_log_depth"][:, 0].to(localization.dtype)
    target_offset = targets["projected_3d_center_offset"].to(localization.dtype)
    target_local_xy = targets["factorized_local_xy"].to(localization.dtype)
    target_local_z = targets["regression"][:, 2:3].to(localization.dtype)
    class_index = targets["factorized_class_index"][:, 0]
    positive = targets["regression_mask"][:, 0] > 0.5
    intrinsic = targets["camera_intrinsic_model"].to(localization.dtype)
    legacy_offset = legacy_object[:, SL_OFFSET].detach().to(localization.dtype).clamp(0.0, 1.0)
    batch, _channels, height, width = localization.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=localization.device, dtype=localization.dtype),
        torch.arange(width, device=localization.device, dtype=localization.dtype), indexing="ij",
    )
    box_grid_x = xx.unsqueeze(0) + legacy_offset[:, 0]
    box_grid_y = yy.unsqueeze(0) + legacy_offset[:, 1]
    projected_u = (box_grid_x + predicted_offset[:, 0]) * float(NATIVE_STRIDE)
    projected_v = (box_grid_y + predicted_offset[:, 1]) * float(NATIVE_STRIDE)
    decoded_depth = torch.exp(raw_log_depth)
    fx = intrinsic[:, 0, 0].view(batch, 1, 1)
    fy = intrinsic[:, 1, 1].view(batch, 1, 1)
    cx = intrinsic[:, 0, 2].view(batch, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch, 1, 1)
    camera_right = (projected_u - cx) * decoded_depth / fx
    camera_up = -(projected_v - cy) * decoded_depth / fy
    camera_xyz = torch.stack([decoded_depth, camera_right, camera_up], dim=1)
    local_xy = camera_xyz[:, :2]
    rotation = camera_matrices[:, :3, :3].to(localization.dtype)
    translation = camera_matrices[:, :3, 3].to(localization.dtype)
    world_xyz = torch.einsum("bij,bjhw->bihw", rotation, camera_xyz) + translation[:, :, None, None]
    target_camera_xyz = torch.cat([target_local_xy, target_local_z], dim=1)
    target_world_xyz = (torch.einsum("bij,bjhw->bihw", rotation, target_camera_xyz)
                        + translation[:, :, None, None])

    depth_terms, offset_terms, endpoint_terms = [], [], []
    for class_id in range(2):
        mask = positive & class_index.eq(class_id)
        if not bool(mask.any()):
            continue
        depth_terms.append(F.smooth_l1_loss(
            raw_log_depth[mask], target_log_depth[mask], beta=1.0, reduction="mean"
        ))
        offset_terms.append(F.smooth_l1_loss(
            predicted_offset.permute(0, 2, 3, 1)[mask],
            target_offset.permute(0, 2, 3, 1)[mask], beta=1.0, reduction="mean",
        ))
        endpoint_terms.append(F.smooth_l1_loss(
            local_xy.permute(0, 2, 3, 1)[mask],
            target_local_xy.permute(0, 2, 3, 1)[mask],
            beta=float(weights["local_xy_endpoint_smooth_l1_beta_m"]), reduction="mean",
        ))
    depth_loss = torch.stack(depth_terms).mean()
    offset_loss = torch.stack(offset_terms).mean()
    endpoint_loss = torch.stack(endpoint_terms).mean()
    total = (
        float(weights["log_depth_smooth_l1_weight"]) * depth_loss
        + float(weights["projected_center_offset_smooth_l1_weight"]) * offset_loss
        + float(weights["local_xy_endpoint_weight"]) * endpoint_loss
    )
    tensors = {
        "raw_depth_logits": masked_values(raw_log_depth, positive),
        "decoded_depth_after_exp": masked_values(decoded_depth, positive),
        "target_camera_forward_depth": torch.exp(masked_values(target_log_depth, positive)),
        "predicted_projected_center_offset": masked_values(predicted_offset, positive),
        "target_projected_center_offset": masked_values(target_offset, positive),
        "reconstructed_camera_xyz": masked_values(camera_xyz, positive),
        "reconstructed_local_xy": masked_values(local_xy, positive),
        "reconstructed_world_xy": masked_values(world_xyz[:, :2], positive),
        "target_local_xy": masked_values(target_local_xy, positive),
        "target_world_xy": masked_values(target_world_xyz[:, :2], positive),
        "log_depth_loss": depth_loss.reshape(1),
        "projected_center_offset_loss": offset_loss.reshape(1),
        "local_xy_endpoint_loss": endpoint_loss.reshape(1),
        "total_loss": total.reshape(1),
    }
    operation_order = [
        ("localization_head_output", "raw_depth_logits"),
        ("unconstrained_exp", "decoded_depth_after_exp"),
        ("target_depth_decode", "target_camera_forward_depth"),
        ("projected_offset_head_output", "predicted_projected_center_offset"),
        ("projected_offset_target", "target_projected_center_offset"),
        ("camera_unprojection", "reconstructed_camera_xyz"),
        ("local_xy_reconstruction", "reconstructed_local_xy"),
        ("world_xy_conversion", "reconstructed_world_xy"),
        ("smooth_l1_log_depth", "log_depth_loss"),
        ("smooth_l1_projected_offset", "projected_center_offset_loss"),
        ("smooth_l1_local_xy_endpoint", "local_xy_endpoint_loss"),
        ("weighted_loss_sum", "total_loss"),
    ]
    stats = {name: tensor_stats(value) for name, value in tensors.items()}
    first = next(({"operation": operation, "tensor": name, "stats": stats[name]}
                  for operation, name in operation_order if not stats[name]["all_finite"]), None)
    return total, tensors, {"tensor_statistics": stats, "first_nonfinite": first}


def activation_path(model, tensors: torch.Tensor, *, amp_enabled: bool) -> dict[str, Any]:
    """Locate the earliest non-finite boundary without changing model state or RNG."""
    stages: list[tuple[str, torch.Tensor]] = [("fused_input", tensors)]
    with torch.no_grad(), torch.autocast(
        device_type="cuda", enabled=amp_enabled, cache_enabled=False
    ):
        features = model.backbone(tensors)
        for name, value in features.items():
            stages.append((f"backbone_{name}", value))
        value = model._object_input(features)
        stages.append(("native_object_input", value))
        for index, module in enumerate(model.object_head.shared_trunk):
            value = module(value)
            stages.append((f"native_shared_trunk_{index}_{type(module).__name__}", value))
        for index, module in enumerate(model.object_head.upsampler):
            value = module(value)
            stages.append((f"native_upsampler_{index}_{type(module).__name__}", value))
        native_feature = value
        value = native_feature.detach()
        for index, module in enumerate(model.localization_trunk):
            value = module(value)
            stages.append((f"localization_trunk_{index}_{type(module).__name__}", value))
        stages.append(("log_depth_head_Conv2d", model.log_depth_head(value)))
        stages.append(("projected_center_offset_head_Conv2d",
                       model.projected_3d_center_offset_head(value)))
    statistics = {name: tensor_stats(value) for name, value in stages}
    first = next(({"operation": name, "stats": statistics[name]}
                  for name, _value in stages if not statistics[name]["all_finite"]), None)
    return {"amp_enabled": amp_enabled, "stages": statistics, "first_nonfinite": first}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_experiment = args.contract_experiment.resolve()
    started = time.monotonic()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required interpreter /usr/bin/python3, got {sys.executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    set_reproducible_seeds(int(config["training_seed"]))
    device = torch.device("cuda")
    checkpoint = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != config["warm_start_sha256"]:
        raise RuntimeError("warm-start SHA mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    object_cfg = dict(payload["config"]["object_heads"])
    dataset_dir = contract_experiment / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    train_rows = [row for row in rows if row.get("split") == "train"]
    row_by_id = {row["sample_id"]: row for row in train_rows}
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    base_dataset = FactorizedLocalizationDataset(
        dataset_dir, train_rows, object_rows, tuple(config["input_size"]), object_cfg,
        augment_strength=str(config["augment_strength"]),
        geometric_augment=bool(config["geometric_augment"]),
    )
    loader = DataLoader(
        WithSampleId(base_dataset, train_rows), batch_size=int(config["batch_size"]),
        shuffle=True, drop_last=False, num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=int(config["prefetch_factor"]),
    )
    model = build_factorized_model(
        num_classes=int(payload["config"]["training"].get("num_classes", 3)),
        radar_channels=int(payload["radar_channels"]),
        hidden_channels=int(payload["object_hidden_channels"]),
        head_depth=int(payload["object_head_depth"]),
        localization_hidden=int(config["localization_hidden_channels"]), device=device,
    )
    load_native_warm_start(model, checkpoint, device=device)
    freeze_for_localization(model)
    optimizer = torch.optim.AdamW(
        localization_parameters(model), lr=float(config["localization_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    records: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None

    for epoch in range(1, 13):
        model.eval()
        model.localization_trunk.train()
        model.log_depth_head.train()
        model.projected_3d_center_offset_head.train()
        for batch_number, (tensors, _masks, targets, sample_ids) in enumerate(loader, 1):
            tensors = tensors.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            camera_matrices = torch.tensor(
                [json.loads(row_by_id[sample_id]["camera_matrix_json"]) for sample_id in sample_ids],
                dtype=torch.float32, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            parameters_before = module_parameter_stats(model) if (
                epoch == RECORD_EPOCH and batch_number >= RECORD_FIRST_BATCH
            ) else None
            scale_before = float(scaler.get_scale())
            with torch.autocast(
                device_type="cuda", enabled=scaler.is_enabled(),
                cache_enabled=bool(config["autocast_cache_enabled"]),
            ):
                outputs = model.localization_training_outputs(tensors)
            with torch.autocast(device_type="cuda", enabled=False):
                actual_loss, _actual_tensors, actual = geometry_and_losses(
                    outputs["localization"].float(), outputs["object"].float(), targets,
                    config["losses"], camera_matrices,
                )
            is_recorded = epoch == RECORD_EPOCH and batch_number >= RECORD_FIRST_BATCH
            record: dict[str, Any] | None = None
            if is_recorded:
                record = {
                    "epoch": epoch, "batch": batch_number, "sample_ids": list(sample_ids),
                    "grad_scaler_scale_before": scale_before,
                    "trainable_parameters_before_step": parameters_before,
                    "actual_amp_forward_fp32_geometry": actual,
                }
            if not math.isfinite(float(actual_loss.detach().item())):
                amp_activation_path = activation_path(model, tensors, amp_enabled=True)
                fp32_activation_path = activation_path(model, tensors, amp_enabled=False)
                with torch.autocast(device_type="cuda", enabled=False):
                    fp32_outputs = model.localization_training_outputs(tensors.float())
                    fp32_loss, _fp32_tensors, fp32 = geometry_and_losses(
                        fp32_outputs["localization"].float(), fp32_outputs["object"].float(),
                        targets, config["losses"], camera_matrices,
                    )
                assert record is not None
                record["explicit_fp32_forward_and_geometry"] = fp32
                record["amp_activation_path"] = amp_activation_path
                record["fp32_activation_path"] = fp32_activation_path
                record["gradient_action"] = "not_run_stop_at_first_nonfinite_forward_or_loss"
                records.append(record)
                actual_first = actual["first_nonfinite"]
                fp32_first = fp32["first_nonfinite"]
                authorized_depth = bool(
                    actual_first is not None and actual_first["operation"] in {
                        "unconstrained_exp", "camera_unprojection", "local_xy_reconstruction",
                        "world_xy_conversion", "smooth_l1_local_xy_endpoint", "weighted_loss_sum",
                    }
                    and actual["tensor_statistics"]["raw_depth_logits"]["all_finite"]
                    and not actual["tensor_statistics"]["decoded_depth_after_exp"]["all_finite"]
                )
                failure = {
                    "epoch": epoch, "batch": batch_number, "sample_ids": list(sample_ids),
                    "first_nonfinite_tensor": actual_first,
                    "fp32_first_nonfinite_tensor": fp32_first,
                    "amp_activation_path_first_nonfinite": amp_activation_path["first_nonfinite"],
                    "fp32_activation_path_first_nonfinite": fp32_activation_path["first_nonfinite"],
                    "cause_classification": (
                        "UNCONSTRAINED_DEPTH_EXPONENTIATION"
                        if authorized_depth else "UNRESOLVED_OR_UNAUTHORIZED_NUMERICAL_CAUSE"
                    ),
                    "authorized_bounded_depth_repair": authorized_depth,
                    "raw_logits_finite_before_exp": actual["tensor_statistics"]["raw_depth_logits"]["all_finite"],
                    "decoded_depth_nonfinite_after_exp": not actual["tensor_statistics"]["decoded_depth_after_exp"]["all_finite"],
                    "amp_total_loss": actual["tensor_statistics"]["total_loss"],
                    "fp32_total_loss": fp32["tensor_statistics"]["total_loss"],
                    "trainable_parameters_before_failure": parameters_before,
                }
                break
            scaler.scale(actual_loss).backward()
            if is_recorded:
                assert record is not None
                record["scaled_gradients_before_unscale"] = module_gradient_stats(model)
                scaler.unscale_(optimizer)
                record["unscaled_gradients_before_optimizer_step"] = module_gradient_stats(model)
            scaler.step(optimizer)
            scaler.update()
            if is_recorded:
                assert record is not None
                record["grad_scaler_scale_after"] = float(scaler.get_scale())
                record["trainable_parameters_after_step"] = module_parameter_stats(model)
                records.append(record)
        if failure is not None:
            break
        scheduler.step()
        print(f"[reproduce] completed epoch {epoch}", flush=True)

    if failure is None:
        failure = {
            "cause_classification": "FAILURE_NOT_REPRODUCED",
            "authorized_bounded_depth_repair": False,
        }
    result = {
        "schema": "route_b_v3_1_factorized_localization_nonfinite_root_cause_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "seed": int(config["training_seed"]),
        "warm_start": str(checkpoint), "warm_start_sha256": sha256(checkpoint),
        "data_order_contract": {
            "manifest_order_reused": True, "shuffle": True,
            "data_loader_generator": "implicit PyTorch global generator, identical to committed train_v2",
            "workers": int(config["num_workers"]), "batch_size": int(config["batch_size"]),
            "persistent_workers": bool(config["persistent_workers"]),
            "prefetch_factor": int(config["prefetch_factor"]),
        },
        "optimizer": config["optimizer"], "learning_rate": config["localization_lr"],
        "schedule": config["schedule"], "amp": config["amp"],
        "autocast_cache_enabled": config["autocast_cache_enabled"],
        "recorded_batches": records, "root_cause": failure,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(output / "ROOT_CAUSE.json", result)
    (output / "REPRODUCTION_COMPLETE").write_text(
        failure["cause_classification"] + "\n", encoding="utf-8"
    )
    print(json.dumps({"root_cause": failure, "recorded_batch_count": len(records),
                      "output": str(output), "wall_seconds": result["wall_seconds"]},
                     indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if failure.get("authorized_bounded_depth_repair") else 2


if __name__ == "__main__":
    raise SystemExit(main())
