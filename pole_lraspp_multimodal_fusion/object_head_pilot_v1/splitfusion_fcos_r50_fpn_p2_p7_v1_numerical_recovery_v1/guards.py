from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import atomic_json
from .state_guard import model_hash, optimizer_hash


def _l2(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().pow(2).sum()) for value in values))


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
    """Calculate the exact next SGD delta without mutating parameters or state."""
    groups = parameter_groups(model, optimizer)
    group_gradients: dict[str, float] = {}
    group_momentum: dict[str, float] = {}
    group_updates: dict[str, float] = {}
    group_parameters: dict[str, float] = {}
    group_optimizer_state: dict[str, float] = {}
    relative: dict[str, float] = {}
    nonzero_gradients = 0
    for group in optimizer.param_groups:
        name = str(group["name"])
        gradients, momentums, updates = [], [], []
        parameter_values = [parameter.detach().double() for parameter in groups[name]]
        optimizer_values = [value.detach().double() for parameter in groups[name]
                            for value in optimizer.state.get(parameter, {}).values()
                            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point]
        lr, momentum = float(group["lr"]), float(group.get("momentum", 0.0))
        dampening, weight_decay = float(group.get("dampening", 0.0)), float(group.get("weight_decay", 0.0))
        nesterov, maximize = bool(group.get("nesterov", False)), bool(group.get("maximize", False))
        for parameter in groups[name]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().double()
            gradients.append(gradient)
            nonzero_gradients += int(bool(torch.count_nonzero(gradient)))
            direction = -gradient if maximize else gradient
            if weight_decay:
                direction = direction + parameter.detach().double() * weight_decay
            buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if buffer is not None:
                prior = buffer.detach().double()
                momentums.append(prior)
                next_buffer = prior * momentum + direction * (1.0 - dampening)
            else:
                next_buffer = direction
            if momentum:
                effective = direction + momentum * next_buffer if nesterov else next_buffer
            else:
                effective = direction
            delta = effective * (-lr)
            updates.append(delta)
            parameter_norm = float(parameter.detach().double().norm())
            relative_name = next(key for key, value in model.named_parameters() if value is parameter)
            relative[relative_name] = float(delta.norm()) / max(parameter_norm, torch.finfo(torch.float64).tiny)
        group_gradients[name] = _l2(gradients)
        group_momentum[name] = _l2(momentums)
        group_updates[name] = _l2(updates)
        group_parameters[name] = _l2(parameter_values)
        group_optimizer_state[name] = _l2(optimizer_values)
    all_gradients = [value.grad.detach().double() for value in model.parameters() if value.grad is not None]
    record = {
        "gradient_norm": {**group_gradients, "global": _l2(all_gradients)},
        "momentum_norm": group_momentum,
        "proposed_sgd_update_norm": group_updates,
        "parameter_norm": {**group_parameters, "global": _l2([value.detach().double() for value in model.parameters()])},
        "optimizer_state_norm": group_optimizer_state,
        "max_parameter_relative_update": max(relative.values(), default=0.0),
        "max_parameter_relative_update_name": max(relative, key=relative.get) if relative else None,
        "gradient_tensors": len(all_gradients), "nonzero_gradient_tensors": nonzero_gradients,
        "zero_gradients_are_diagnostic_only": True,
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
        before = {"model": model_hash(model), "optimizer": optimizer_hash(optimizer)}
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
        after = {"model": model_hash(model), "optimizer": optimizer_hash(optimizer)}
        if before != after:
            raise RuntimeError("breaker calculation mutated model or optimizer")
        context_record = dict(context or {})
        record = {"schema": "splitfusion_fcos_pre_step_breaker_v1", "epoch": int(epoch),
                  "update_in_epoch": int(update_in_epoch), "global_update_if_stepped": int(global_update_if_stepped),
                  "sample_ids": list(context_record.get("sample_ids", [])),
                  "metrics": metrics, "ceilings": self.ceilings, "violations": violations,
                  "model_optimizer_unchanged": True, "action": "abort_before_optimizer_step" if violations else "allow_step",
                  "no_clipping": True, "no_skip_policy": True, "loss_is_not_a_breaker_criterion": True,
                  "context": context_record}
        if violations:
            self.failure_root.mkdir(parents=True, exist_ok=True)
            path = self.failure_root / f"BREAKER_E{epoch:03d}_U{update_in_epoch:04d}.json"
            atomic_json(path, record)
            raise FloatingPointError(f"pre-step numerical breaker aborted before optimizer.step: {path}")
        return record
