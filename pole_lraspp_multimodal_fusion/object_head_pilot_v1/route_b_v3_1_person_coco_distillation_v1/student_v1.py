#!/usr/bin/env python3
"""Exact epoch-40 student integration and registered trainability controls."""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
CLEAN_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_clean_base_v1"
for _path in (str(NATIVE_PACKAGE), str(CLEAN_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from losses_v1 import native_object_loss, segmentation_loss  # noqa: E402
from model_v1 import (  # noqa: E402
    build_native_grid_model,
    decode_tail,
    encode_front,
    parameter_report,
    split_boundary_report,
)

TRAINABLE_PREFIXES = (
    "object_head.person_heatmap_head.",
    "object_head.shared_trunk.6.",
    "classifier.low_classifier.",
    "classifier.high_classifier.",
    "backbone.13.", "backbone.14.", "backbone.15.", "backbone.16.",
)
SEGMENTATION_ROW_PARAMETERS = (
    "classifier.low_classifier.weight", "classifier.low_classifier.bias",
    "classifier.high_classifier.weight", "classifier.high_classifier.bias",
)


def build_student(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload["model"]
    if len(state) != 351:
        raise RuntimeError(f"registered deployable state must contain 351 tensors, got {len(state)}")
    model = build_native_grid_model(
        num_classes=int(payload["config"]["training"].get("num_classes", 3)),
        radar_channels=int(payload["radar_channels"]),
        hidden_channels=int(payload["object_hidden_channels"]),
        head_depth=int(payload["object_head_depth"]), device=device,
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict epoch-40 load failed: {incompatible}")
    return model, payload


def freeze_batch_norm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(False)


def enforce_train_mode(model: torch.nn.Module) -> None:
    """Enable train behavior only where registered; every BatchNorm remains frozen."""
    model.train()
    freeze_batch_norm(model)


def configure_trainable(model: torch.nn.Module) -> Dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    freeze_batch_norm(model)
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    unexpected = [name for name in trainable if not any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)]
    missing_prefixes = [prefix for prefix in TRAINABLE_PREFIXES
                        if not any(name.startswith(prefix) for name in trainable)]
    bn_trainable = [name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and _is_batch_norm_parameter(model, name)]
    if unexpected or missing_prefixes or bn_trainable:
        raise RuntimeError(
            f"registered trainable allowlist failure: unexpected={unexpected} "
            f"missing={missing_prefixes} bn_trainable={bn_trainable}"
        )
    return {
        "trainable_names": trainable,
        "frozen_names": sorted(name for name, parameter in model.named_parameters()
                               if not parameter.requires_grad),
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(int(parameter.numel()) for parameter in model.parameters()
                                    if parameter.requires_grad),
        "frozen_parameters": sum(int(parameter.numel()) for parameter in model.parameters()
                                 if not parameter.requires_grad),
        "native_parameter_report": parameter_report(model),
        "allowlist_exact": True,
    }


def _is_batch_norm_parameter(model: torch.nn.Module, parameter_name: str) -> bool:
    module_name = parameter_name.rsplit(".", 1)[0]
    modules = dict(model.named_modules())
    return isinstance(modules.get(module_name), torch.nn.modules.batchnorm._BatchNorm)


class SegmentationRowGuard:
    """Zero non-person row gradients and bit-exactly restore rows 0/1 after AdamW."""

    def __init__(self, model: torch.nn.Module) -> None:
        named = dict(model.named_parameters())
        self.parameters: Dict[str, torch.nn.Parameter] = {}
        self.reference: Dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []
        for name in SEGMENTATION_ROW_PARAMETERS:
            parameter = named.get(name)
            if parameter is None or int(parameter.shape[0]) != 3:
                raise RuntimeError(f"missing 3-row segmentation parameter: {name}")
            self.parameters[name] = parameter
            self.reference[name] = parameter.detach()[:2].cpu().clone()

            def mask_rows(gradient: torch.Tensor) -> torch.Tensor:
                result = gradient.clone()
                result[:2].zero_()
                return result

            self.hooks.append(parameter.register_hook(mask_rows))

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.parameters.items():
            parameter[:2].copy_(self.reference[name].to(parameter.device, dtype=parameter.dtype))

    def report(self) -> Dict[str, Any]:
        exact = {
            name: bool(torch.equal(parameter.detach()[:2].cpu(), self.reference[name]))
            for name, parameter in self.parameters.items()
        }
        return {"masked_rows": [0, 1], "trained_row": 2,
                "row_restore_exact": exact, "all_exact": all(exact.values())}


def registered_parameter_groups(
    model: torch.nn.Module, adapter: torch.nn.Module,
) -> list[Dict[str, Any]]:
    named = dict(model.named_parameters())
    definitions = (
        ("person_heatmap", 1.0e-4, ("object_head.person_heatmap_head.",)),
        ("shared_trunk_6", 2.0e-5, ("object_head.shared_trunk.6.",)),
        ("person_segmentation_rows", 2.0e-5,
         ("classifier.low_classifier.", "classifier.high_classifier.")),
        ("backbone_13_16", 5.0e-6,
         ("backbone.13.", "backbone.14.", "backbone.15.", "backbone.16.")),
    )
    groups: list[Dict[str, Any]] = []
    used: set[str] = set()
    for group_name, lr, prefixes in definitions:
        names = [name for name, parameter in named.items()
                 if parameter.requires_grad and any(name.startswith(prefix) for prefix in prefixes)]
        if not names:
            raise RuntimeError(f"registered optimizer group is empty: {group_name}")
        used.update(names)
        groups.append({"params": [named[name] for name in names], "lr": lr,
                       "base_lr": lr, "name": group_name, "parameter_names": names})
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if used != expected:
        raise RuntimeError(f"optimizer/model trainable mismatch: missing={sorted(expected-used)} extra={sorted(used-expected)}")
    adapter_parameters = list(adapter.parameters())
    if not adapter_parameters or any(not parameter.requires_grad for parameter in adapter_parameters):
        raise RuntimeError("distillation adapter trainability failure")
    groups.append({"params": adapter_parameters, "lr": 1.0e-4, "base_lr": 1.0e-4,
                   "name": "distillation_student_adapter",
                   "parameter_names": [f"adapter.{name}" for name, _ in adapter.named_parameters()]})
    return groups


def forward_once(
    model: torch.nn.Module, tensors: torch.Tensor, *, policy: str,
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, torch.Tensor]]:
    enabled = policy == "student_bf16_teacher_fp32_losses_fp32"
    if policy not in {"student_bf16_teacher_fp32_losses_fp32", "full_fp32"}:
        raise ValueError(f"unregistered numerical policy: {policy}")
    with torch.autocast(device_type=tensors.device.type, enabled=enabled,
                        dtype=torch.bfloat16, cache_enabled=False):
        features = encode_front(model, tensors)
        outputs = decode_tail(model, features)
    return features, outputs


def supervised_loss(
    outputs: Dict[str, torch.Tensor], masks: torch.Tensor,
    targets: Dict[str, torch.Tensor], *, class_weights: torch.Tensor,
    loss_weights: Dict[str, Any], lovasz_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    # Reductions and the entire native regression path are explicitly FP32.
    with torch.autocast(device_type=outputs["out"].device.type, enabled=False):
        seg, seg_parts, seg_logits = segmentation_loss(
            outputs["out"].float(), masks, class_weights=class_weights,
            lovasz_weight=float(lovasz_weight),
        )
        obj, obj_parts = native_object_loss(
            outputs["object"].float(), targets, loss_weights.get("object", {}),
        )
        total = (float(loss_weights.get("segmentation", 0.3)) * seg
                 + float(loss_weights.get("object_total", 1.0)) * obj)
    return total, {**seg_parts, **obj_parts, "object_loss": float(obj.detach().item()),
                   "supervised_loss": float(total.detach().item())}, seg_logits


def batch_norm_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            result[f"{name}.running_mean"] = module.running_mean.detach().cpu().clone()
            result[f"{name}.running_var"] = module.running_var.detach().cpu().clone()
            result[f"{name}.num_batches_tracked"] = module.num_batches_tracked.detach().cpu().clone()
    return result


def snapshots_equal(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def deployable_state_report(model: torch.nn.Module) -> Dict[str, Any]:
    keys = sorted(model.state_dict())
    forbidden = [key for key in keys if any(token in key.lower()
                                            for token in ("teacher", "adapter", "projector"))]
    return {"tensor_count": len(keys), "forbidden_keys": forbidden,
            "no_teacher_or_projector_keys": not forbidden}


def component_gradient_sums(model: torch.nn.Module, adapter: torch.nn.Module) -> Dict[str, float]:
    prefixes = {
        "person_heatmap": ("object_head.person_heatmap_head.",),
        "shared_trunk_6": ("object_head.shared_trunk.6.",),
        "person_segmentation_rows": ("classifier.low_classifier.", "classifier.high_classifier."),
        "backbone_13_16": ("backbone.13.", "backbone.14.", "backbone.15.", "backbone.16."),
    }
    result = {}
    for group, group_prefixes in prefixes.items():
        result[group] = float(sum(
            parameter.grad.detach().abs().sum().item()
            for name, parameter in model.named_parameters()
            if any(name.startswith(prefix) for prefix in group_prefixes) and parameter.grad is not None
        ))
    result["distillation_student_adapter"] = float(sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in adapter.parameters() if parameter.grad is not None
    ))
    return result


def finite_parameter_tree(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in parameters)
