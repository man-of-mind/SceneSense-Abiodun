from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
from typing import Any, Mapping

import torch


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    record: dict[str, Any] = {"dtype": str(value.dtype), "shape": list(value.shape),
                              "floating": value.dtype.is_floating_point, "numel": value.numel()}
    if not value.dtype.is_floating_point:
        record["finite"] = True; record["nonfinite_count"] = 0
        if value.numel():
            numeric = value.detach().double()
            record.update({"min": float(numeric.min()), "max": float(numeric.max()),
                           "absmax": float(numeric.abs().max())})
        else:
            record.update({"min": None, "max": None, "absmax": None})
        return record
    finite = torch.isfinite(value)
    record["finite"] = bool(finite.all())
    record["nonfinite_count"] = int((~finite).sum())
    if value.numel() and bool(finite.any()):
        values = value.detach()[finite].double()
        record.update({"min": float(values.min()), "max": float(values.max()), "absmax": float(values.abs().max())})
    else:
        record.update({"min": None, "max": None, "absmax": None})
    return record


def audit_tree(value: Any, prefix: str = "root") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = tensor_record(value)
    elif isinstance(value, Mapping):
        for name, item in value.items():
            result.update(audit_tree(item, f"{prefix}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(audit_tree(item, f"{prefix}[{index}]"))
    return result


def require_finite_audit(records: Mapping[str, Mapping[str, Any]]) -> None:
    failed = [name for name, record in records.items() if not record.get("finite", False)]
    if failed:
        raise FloatingPointError(f"nonfinite tensor audit: {failed[:20]}")


class ForwardAudit(AbstractContextManager["ForwardAudit"]):
    """Hooks input, C2/FPN/head modules; explicit output audits cover decode tensors."""

    TOKENS = ("front", "tail", "classification_tower", "regression_head", "project_classifier",
              "geometry", "semantic", "dense_depth")

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.records: dict[str, dict[str, Any]] = {}
        self.counts: Counter[str] = Counter()
        self._handles: list[Any] = []

    def __enter__(self) -> "ForwardAudit":
        def hook(name: str):
            def capture(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
                self.counts[name] += 1
                self.records.update(audit_tree(inputs, f"module.{name}.input.{self.counts[name]}"))
                self.records.update(audit_tree(output, f"module.{name}.output.{self.counts[name]}"))
            return capture
        for name, module in self.model.named_modules():
            if name and any(name == token or name.startswith(token + ".") for token in self.TOKENS):
                self._handles.append(module.register_forward_hook(hook(name)))
        return self

    def add(self, name: str, value: Any) -> None:
        self.records.update(audit_tree(value, name))

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is None:
            require_finite_audit(self.records)
