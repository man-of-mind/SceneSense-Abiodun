from __future__ import annotations

import hashlib
import json
import random
from contextlib import AbstractContextManager
from typing import Any, Mapping

import numpy as np
import torch


def _tensor_digest(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
    digest.update(str(tuple(tensor.shape)).encode() + b"\0" + tensor.numpy().tobytes())


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _tensor_digest(digest, name, value)
    return digest.hexdigest()


def optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    for group_index, group in enumerate(state["param_groups"]):
        metadata = {key: value for key, value in group.items() if key != "params"}
        digest.update(json.dumps([group_index, metadata, group["params"]], sort_keys=True, default=str).encode())
    for parameter_id, values in sorted(state["state"].items()):
        for name, value in sorted(values.items()):
            if isinstance(value, torch.Tensor):
                _tensor_digest(digest, f"{parameter_id}.{name}", value)
            else:
                digest.update(repr((parameter_id, name, value)).encode())
    return digest.hexdigest()


def control_hash(model: torch.nn.Module) -> str:
    """Hash modes, reachability flags, and gradients outside state_dict."""
    digest = hashlib.sha256()
    digest.update(json.dumps({name: module.training for name, module in model.named_modules()},
                             sort_keys=True).encode())
    for name, parameter in sorted(model.named_parameters()):
        digest.update(repr((name, parameter.requires_grad, parameter.grad is None)).encode())
        if parameter.grad is not None:
            _tensor_digest(digest, f"gradient.{name}", parameter.grad)
    return digest.hexdigest()


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def rng_hash(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(repr((value["python"], value["numpy"][0], value["numpy"][1].tolist(),
                                  value["numpy"][2:])).encode())
    digest.update(value["torch"].cpu().numpy().tobytes())
    for item in value["cuda"]:
        digest.update(item.cpu().numpy().tobytes())
    return digest.hexdigest()


class DiagnosticStateGuard(AbstractContextManager["DiagnosticStateGuard"]):
    """Restore model, optimizer, grads, modes, requires-grad, and all RNG in finally."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model, self.optimizer = model, optimizer
        self.report: dict[str, Any] = {}

    def __enter__(self) -> "DiagnosticStateGuard":
        self._model = _cpu_clone(self.model.state_dict())
        self._optimizer = _cpu_clone(self.optimizer.state_dict())
        self._rng = capture_rng()
        self._modes = {name: module.training for name, module in self.model.named_modules()}
        self._requires = {name: value.requires_grad for name, value in self.model.named_parameters()}
        self._grads = {name: None if value.grad is None else value.grad.detach().cpu().clone()
                       for name, value in self.model.named_parameters()}
        self.report["before"] = {"model": model_hash(self.model), "optimizer": optimizer_hash(self.optimizer),
                                 "rng": rng_hash(self._rng), "modes_requires_grad_and_gradients": control_hash(self.model)}
        self.report["normalization_state_accounted"] = ["FrozenBatchNorm parameters/buffers/modes", "GroupNorm parameters/modes"]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.model.load_state_dict(self._model, strict=True)
            self.optimizer.load_state_dict(self._optimizer)
            for name, module in self.model.named_modules():
                module.train(self._modes[name])
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(self._requires[name])
                saved = self._grads[name]
                parameter.grad = None if saved is None else saved.to(parameter.device).clone()
            restore_rng(self._rng)
            after_rng = capture_rng()
            self.report["after"] = {"model": model_hash(self.model), "optimizer": optimizer_hash(self.optimizer),
                                    "rng": rng_hash(after_rng),
                                    "modes_requires_grad_and_gradients": control_hash(self.model)}
            self.report["restored_exactly"] = self.report["before"] == self.report["after"]
            if not self.report["restored_exactly"] and exc_type is None:
                raise RuntimeError(f"diagnostic state restoration mismatch: {self.report}")
        finally:
            del self._model, self._optimizer, self._rng, self._modes, self._requires, self._grads
