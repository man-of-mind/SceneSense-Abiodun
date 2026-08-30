from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import atomic_json


def _squared_l2(value: torch.Tensor) -> float:
    """Reduce one tensor directly to an FP64 scalar without retaining an FP64 copy."""
    scalar = torch.linalg.vector_norm(value.detach(), ord=2, dtype=torch.float64)
    result = float(scalar) ** 2
    del scalar
    return result


def parameter_groups(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, list[torch.nn.Parameter]]:
    names = {id(value): name for name, value in model.named_parameters()}
    result: dict[str, list[torch.nn.Parameter]] = {}
    seen: set[int] = set()
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        if name in result:
            raise RuntimeError(f"duplicate optimizer group name: {name}")
        values = list(group["params"])
        for value in values:
            if id(value) not in names or id(value) in seen:
                raise RuntimeError("optimizer parameter group omission/overlap")
            seen.add(id(value))
        result[name] = values
    if seen != set(names):
        raise RuntimeError("optimizer does not cover every model parameter exactly once")
    return result


def proposed_sgd_metrics(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Stream exact SGD metrics one tensor at a time without mutating state."""
    groups = parameter_groups(model, optimizer)
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    group_gradients: dict[str, float] = {}
    group_momentum: dict[str, float] = {}
    group_updates: dict[str, float] = {}
    group_parameters: dict[str, float] = {}
    group_optimizer_state: dict[str, float] = {}
    max_relative = 0.0
    max_relative_name: str | None = None
    nonzero_gradients = 0
    global_gradient_squared = 0.0
    global_parameter_squared = 0.0
    for group in optimizer.param_groups:
        name = str(group["name"])
        gradient_squared = momentum_squared = update_squared = 0.0
        parameter_squared = optimizer_state_squared = 0.0
        lr, momentum = float(group["lr"]), float(group.get("momentum", 0.0))
        dampening, weight_decay = float(group.get("dampening", 0.0)), float(group.get("weight_decay", 0.0))
        nesterov, maximize = bool(group.get("nesterov", False)), bool(group.get("maximize", False))
        for parameter in groups[name]:
            parameter_l2_squared = _squared_l2(parameter)
            parameter_squared += parameter_l2_squared
            global_parameter_squared += parameter_l2_squared
            state = optimizer.state.get(parameter, {})
            momentum_buffer_l2_squared = None
            for state_name, state_value in state.items():
                if isinstance(state_value, torch.Tensor) and state_value.dtype.is_floating_point:
                    state_l2_squared = _squared_l2(state_value)
                    optimizer_state_squared += state_l2_squared
                    if state_name == "momentum_buffer":
                        momentum_buffer_l2_squared = state_l2_squared
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            gradient_l2_squared = _squared_l2(gradient)
            gradient_squared += gradient_l2_squared
            global_gradient_squared += gradient_l2_squared
            nonzero_gradients += int(gradient_l2_squared > 0.0)
            direction = -gradient if maximize else gradient
            if weight_decay:
                direction = direction.add(parameter.detach(), alpha=weight_decay)
            buffer = state.get("momentum_buffer")
            if buffer is not None:
                prior = buffer.detach()
                momentum_squared += (momentum_buffer_l2_squared
                                     if momentum_buffer_l2_squared is not None else _squared_l2(prior))
                next_buffer = prior.mul(momentum).add(direction, alpha=1.0 - dampening)
            else:
                next_buffer = direction
            if momentum:
                effective = direction.add(next_buffer, alpha=momentum) if nesterov else next_buffer
            else:
                effective = direction
            effective_l2_squared = _squared_l2(effective)
            update_squared += lr * lr * effective_l2_squared
            relative_value = abs(lr) * math.sqrt(effective_l2_squared) / max(
                math.sqrt(parameter_l2_squared), torch.finfo(torch.float64).tiny)
            if max_relative_name is None or relative_value > max_relative:
                max_relative = relative_value
                max_relative_name = parameter_names[id(parameter)]
            del direction, next_buffer, effective
        group_gradients[name] = math.sqrt(gradient_squared)
        group_momentum[name] = math.sqrt(momentum_squared)
        group_updates[name] = math.sqrt(update_squared)
        group_parameters[name] = math.sqrt(parameter_squared)
        group_optimizer_state[name] = math.sqrt(optimizer_state_squared)
    record = {
        "gradient_norm": {**group_gradients, "global": math.sqrt(global_gradient_squared)},
        "momentum_norm": group_momentum,
        "proposed_sgd_update_norm": group_updates,
        "parameter_norm": {**group_parameters, "global": math.sqrt(global_parameter_squared)},
        "optimizer_state_norm": group_optimizer_state,
        "max_parameter_relative_update": max_relative,
        "max_parameter_relative_update_name": max_relative_name,
        "gradient_tensors": sum(parameter.grad is not None for parameter in model.parameters()),
        "nonzero_gradient_tensors": nonzero_gradients,
        "zero_gradients_are_diagnostic_only": True,
        "streaming_scalar_reductions": True,
        "retained_fp64_tensor_copies": 0,
    }
    floating = [value for family in (record["gradient_norm"], record["momentum_norm"],
                                      record["proposed_sgd_update_norm"]) for value in family.values()]
    floating.extend(record["parameter_norm"].values()); floating.extend(record["optimizer_state_norm"].values())
    floating.append(record["max_parameter_relative_update"])
    record["finite"] = all(math.isfinite(float(value)) for value in floating)
    return record


class PreStepBreaker:
    """Post-accumulation/pre-step fail-closed breaker; never clips or skips."""

    def __init__(self, ceilings: Mapping[str, Any], failure_root: Path) -> None:
        required = ("gradient_norm", "momentum_norm", "proposed_sgd_update_norm", "max_parameter_relative_update")
        if any(name not in ceilings or ceilings[name] is None for name in required):
            raise RuntimeError("pre-step breaker ceilings are not qualified")
        self.ceilings, self.failure_root = ceilings, Path(failure_root)

    def check(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, epoch: int,
              update_in_epoch: int, global_update_if_stepped: int, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        metrics = proposed_sgd_metrics(model, optimizer)
        violations: list[dict[str, Any]] = []
        if not metrics["finite"]:
            violations.append({"kind": "nonfinite_pre_step_metric"})
        for family in ("gradient_norm", "momentum_norm", "proposed_sgd_update_norm"):
            for group, value in metrics[family].items():
                ceiling = float(self.ceilings[family][group])
                if not math.isfinite(ceiling) or value > ceiling:
                    violations.append({"kind": family, "group": group, "value": value, "ceiling": ceiling})
        value = float(metrics["max_parameter_relative_update"])
        ceiling = float(self.ceilings["max_parameter_relative_update"])
        if not math.isfinite(ceiling) or value > ceiling:
            violations.append({"kind": "max_parameter_relative_update", "value": value, "ceiling": ceiling})
        context_record = dict(context or {})
        record = {"schema": "splitfusion_fcos_pre_step_breaker_v1", "epoch": int(epoch),
                  "update_in_epoch": int(update_in_epoch), "global_update_if_stepped": int(global_update_if_stepped),
                  "sample_ids": list(context_record.get("sample_ids", [])),
                  "metrics": metrics, "ceilings": self.ceilings, "violations": violations,
                  "nonmutation_proof": "unit_tests_and_qualification_boundary_hashes",
                  "action": "abort_before_optimizer_step" if violations else "allow_step",
                  "no_clipping": True, "no_skip_policy": True, "loss_is_not_a_breaker_criterion": True,
                  "context": context_record}
        if violations:
            self.failure_root.mkdir(parents=True, exist_ok=True)
            path = self.failure_root / f"BREAKER_E{epoch:03d}_U{update_in_epoch:04d}.json"
            atomic_json(path, record)
            raise FloatingPointError(f"pre-step numerical breaker aborted before optimizer.step: {path}")
        return record
