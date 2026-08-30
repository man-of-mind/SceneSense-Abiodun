from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .audit import audit_tree, require_finite_audit
from .recovery_losses import compute_loss_groups
from .state_guard import DiagnosticStateGuard


def _norm(values: Sequence[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(value.detach().double().pow(2).sum()) for value in values if value is not None))


def _one_precision(base: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataset: Any,
                   registered_batches: Sequence[Mapping[str, Any]], multipliers: Mapping[str, float],
                   physical_batch: int, use_amp: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    with DiagnosticStateGuard(model, optimizer) as guard:
        model.train()
        for registered in registered_batches:
            optimizer.zero_grad(set_to_none=True)
            group_sums = {name: 0.0 for name in ("D", "G", "S", "A")}; component_sums: dict[str, float] = {}
            c2_squared = {name: 0.0 for name in group_sums}; yaw_norms = []; yaw_below_tau = 0; sample_ids = []
            indices = list(registered["indices"]); chunks = [indices[x:x + physical_batch] for x in range(0, 16, physical_batch)]
            for chunk in chunks:
                batch = base.data.collate([dataset[index] for index in chunk]); sample_ids.extend(batch["sample_ids"])
                total, parts, audit, outputs = compute_loss_groups(
                    model, batch, multipliers, use_amp=use_amp, audit_detail=True)
                require_finite_audit(audit_tree(outputs, "outputs"))
                for name in group_sums:
                    group_sums[name] += float(parts[name].detach()) / len(chunks)
                    c2 = torch.autograd.grad(parts[name], outputs["c2"], retain_graph=True, allow_unused=True)
                    c2_squared[name] += _norm(c2) ** 2 / len(chunks) ** 2
                for name, value in parts.items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(value.detach()) / len(chunks)
                for row in audit["geometry"]["carrier_identities"]:
                    yaw_norms.append(float(row["raw_yaw_norm"]))
                    yaw_below_tau += int(bool(row["below_tau"]))
                (total / len(chunks)).backward()
            if sample_ids != list(registered["sample_ids"]):
                raise RuntimeError(f"registered calibration sample order drift in batch {registered['batch']}")
            required = base.train.required_gradient_evidence(model)
            if not base.train.all_gradients_finite(model):
                raise FloatingPointError("nonfinite calibration comparison gradients")
            yaw_array = np.asarray(yaw_norms, dtype=np.float64)
            rows.append({"batch": registered["batch"], "sample_ids": sample_ids, "loss_groups": group_sums,
                         "loss_components": component_sums,
                         "c2_gradient_norms": {name: math.sqrt(value) for name, value in c2_squared.items()},
                         "yaw_raw_norm": {"count": len(yaw_norms), "min": min(yaw_norms) if yaw_norms else None,
                                          "max": max(yaw_norms) if yaw_norms else None,
                                          "median": float(np.percentile(yaw_array, 50)) if yaw_norms else None,
                                          "p99": float(np.percentile(yaw_array, 99)) if yaw_norms else None,
                                          "mean": sum(yaw_norms) / len(yaw_norms) if yaw_norms else None,
                                          "below_tau_count": yaw_below_tau,
                                          "below_tau_fraction": yaw_below_tau / len(yaw_norms) if yaw_norms else 0.0},
                         "required_parameter_gradients": required,
                         "all_outputs_losses_gradients_finite": True, "optimizer_step": False})
    return rows, guard.report


def compare_registered_precisions(base: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                  dataset: Any, registered_batches: Sequence[Mapping[str, Any]],
                                  multipliers: Mapping[str, float], physical_batch: int = 4) -> dict[str, Any]:
    if len(registered_batches) != 8:
        raise RuntimeError("precision comparison requires exactly eight registered train-only batches")
    fp32, fp32_state = _one_precision(base, model, optimizer, dataset, registered_batches, multipliers,
                                      physical_batch, use_amp=False)
    bf16, bf16_state = _one_precision(base, model, optimizer, dataset, registered_batches, multipliers,
                                      physical_batch, use_amp=True)
    comparisons = []
    for left, right in zip(fp32, bf16):
        ratios = {}
        for name in ("D", "G", "S", "A"):
            ratios[f"loss_{name}"] = right["loss_groups"][name] / max(abs(left["loss_groups"][name]), 1e-30)
            ratios[f"c2_grad_{name}"] = right["c2_gradient_norms"][name] / max(left["c2_gradient_norms"][name], 1e-30)
        for name, fp_value in left["required_parameter_gradients"].items():
            bf_value = right["required_parameter_gradients"][name]
            ratios[f"required_grad_{name}"] = bf_value["l2"] / max(fp_value["l2"], 1e-30)
        for name in ("min", "median", "p99", "max", "mean"):
            fp_value, bf_value = left["yaw_raw_norm"][name], right["yaw_raw_norm"][name]
            ratios[f"yaw_raw_norm_{name}"] = None if fp_value is None or bf_value is None else bf_value / max(abs(fp_value), 1e-30)
        fp_order = sorted(("D", "G", "S", "A"), key=lambda name: (-left["loss_groups"][name], name))
        bf_order = sorted(("D", "G", "S", "A"), key=lambda name: (-right["loss_groups"][name], name))
        divergent = [name for name, value in ratios.items()
                     if value is not None and (not math.isfinite(float(value)) or float(value) < 0.1 or float(value) > 10.0)]
        comparisons.append({"batch": left["batch"], "bf16_over_fp32_ratios": ratios,
                            "loss_order_fp32": fp_order, "loss_order_bf16": bf_order,
                            "loss_order_diverged": fp_order != bf_order,
                            "order_of_magnitude_divergences": divergent})
    return {"schema": "splitfusion_fcos_train_only_bf16_fp32_comparison_v1", "registered_batches": 8,
            "fp32": fp32, "bf16": bf16, "comparison": comparisons,
            "state_restoration": {"fp32": fp32_state, "bf16": bf16_state},
            "loss_calibration_recomputed": False, "group_multipliers_changed": False,
            "all_outputs_losses_gradients_finite": all(
                row["all_outputs_losses_gradients_finite"] for row in fp32 + bf16),
            "optimizer_steps": 0, "validation_accessed": False}
