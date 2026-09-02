from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from . import contract


class HybridQError(RuntimeError):
    """Base fail-closed error for the hybrid-q transport path."""


class HybridQConfigError(HybridQError):
    """Invalid q, temperature or other caller-supplied configuration."""


class HybridQNumericalError(HybridQError):
    """Non-finite scores, features or decoded values."""


class HybridQPayloadError(HybridQError):
    """Malformed header, bitmask, value block or retained index set."""


class HybridQOwnershipError(HybridQError):
    """Frozen perception parameters reached a trainable optimizer group."""


def require_valid_q(q: float, *, registered_only: bool = True) -> float:
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise HybridQConfigError(f"q must be a real number, got {type(q).__name__}")
    value = float(q)
    if value != value or value in (float("inf"), float("-inf")):
        raise HybridQConfigError("q must be finite")
    if not 0.0 <= value < 1.0:
        raise HybridQConfigError(f"q must satisfy 0 <= q < 1, got {value!r}")
    if registered_only and not contract.is_registered_q(value):
        raise HybridQConfigError(
            f"q={value!r} is not a registered value {contract.REGISTERED_Q_VALUES}"
        )
    return value


def require_positive_temperature(temperature: float | None) -> float:
    if temperature is None:
        raise HybridQConfigError(
            "straight-through temperature must be supplied explicitly by configuration"
        )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise HybridQConfigError("temperature must be a real number")
    value = float(temperature)
    if not value > 0.0 or value != value or value == float("inf"):
        raise HybridQConfigError(f"temperature must be finite and positive, got {value!r}")
    return value


def require_finite(tensor: torch.Tensor, what: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise HybridQNumericalError(f"{what} contains non-finite values")
    return tensor


def require_c2_tensor(
    tensor: torch.Tensor,
    *,
    channels: int = contract.SPLIT_CHANNELS,
    dtype: torch.dtype = torch.float32,
    what: str = "C2 tensor",
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tensor.dim() != 3:
        raise HybridQPayloadError(f"{what} must be [C,H,W], got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != channels:
        raise HybridQPayloadError(
            f"{what} must have {channels} channels, got {tensor.shape[0]}"
        )
    if tensor.dtype is not dtype:
        raise HybridQPayloadError(f"{what} must be {dtype}, got {tensor.dtype}")
    return tensor


def require_keep_cardinality(observed: int, expected: int) -> int:
    if int(observed) != int(expected):
        raise HybridQPayloadError(
            f"keep cardinality mismatch: observed {observed}, contract requires {expected}"
        )
    return int(observed)


def require_sorted_unique_indices(indices: torch.Tensor, cells: int) -> torch.Tensor:
    if indices.dim() != 1:
        raise HybridQPayloadError("retained indices must be a 1-D tensor")
    if indices.dtype not in (torch.int32, torch.int64):
        raise HybridQPayloadError("retained indices must be integer typed")
    if indices.numel() == 0:
        return indices
    if int(indices.min()) < 0 or int(indices.max()) >= int(cells):
        raise HybridQPayloadError("retained index out of spatial range")
    deltas = indices[1:] - indices[:-1]
    if indices.numel() > 1 and int(deltas.min()) <= 0:
        raise HybridQPayloadError("retained indices must be strictly ascending and unique")
    return indices


def require_frozen_perception(modules: Iterable[torch.nn.Module]) -> None:
    """Every parameter of the frozen perception stack must be non-trainable."""
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise HybridQOwnershipError(
                    f"frozen perception parameter '{name}' has requires_grad=True"
                )


def require_optimizer_owns_only(
    optimizer: torch.optim.Optimizer, allowed: Iterable[torch.nn.Parameter]
) -> None:
    allowed_ids = {id(parameter) for parameter in allowed}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in allowed_ids:
                raise HybridQOwnershipError(
                    "optimizer owns a parameter outside the ranker parameter set"
                )


def snapshot_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def require_parameters_unchanged(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    current = dict(module.named_parameters())
    if set(current) != set(snapshot):
        raise HybridQOwnershipError("frozen parameter set changed identity")
    for name, value in current.items():
        if not torch.equal(value.detach(), snapshot[name]):
            raise HybridQOwnershipError(f"frozen parameter '{name}' changed value")
