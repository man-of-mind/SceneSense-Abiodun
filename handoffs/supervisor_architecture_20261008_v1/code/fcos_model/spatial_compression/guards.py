from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from . import contract


class HybridQError(RuntimeError):
    """Base fail-closed error for the hybrid-q transport path."""


class HybridQConfigError(HybridQError):
    """Invalid q, temperature or other caller-supplied configuration."""


class HybridQNumericalError(HybridQError):
    """Non-finite scores, features, losses, gradients or decoded values."""


class HybridQPayloadError(HybridQError):
    """Malformed header, bitmask, value block, shape or retained index set."""


class HybridQOwnershipError(HybridQError):
    """Frozen perception state reached a trainable optimizer group or changed."""


class HybridQQualificationError(HybridQError):
    """A ranker parameter failed the gradient qualification window."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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
            "temperature must be supplied explicitly by the locked configuration"
        )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise HybridQConfigError("temperature must be a real number")
    value = float(temperature)
    if not value > 0.0 or value != value or value == float("inf"):
        raise HybridQConfigError(f"temperature must be finite and positive, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Frozen C2 boundary
# ---------------------------------------------------------------------------


def require_finite(tensor: torch.Tensor, what: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise HybridQNumericalError(f"{what} contains non-finite values")
    return tensor


def require_frozen_c2(
    tensor: torch.Tensor, *, what: str = "C2 tensor", check_finite: bool = True
) -> torch.Tensor:
    """Production boundary: exactly one frame of [256,112,192] FP32, finite."""
    if not isinstance(tensor, torch.Tensor):
        raise HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(tensor.shape) != contract.SPLIT_SHAPE:
        raise HybridQPayloadError(
            f"{what} must be {list(contract.SPLIT_SHAPE)}, got {list(tensor.shape)}"
        )
    if tensor.dtype is not torch.float32:
        raise HybridQPayloadError(f"{what} must be float32, got {tensor.dtype}")
    if check_finite:
        require_finite(tensor, what)
    return tensor


def require_frozen_batched_c2(
    tensor: torch.Tensor, *, what: str = "batched C2 tensor", check_finite: bool = True
) -> torch.Tensor:
    """Ranker boundary: [256,112,192] or [B,256,112,192] FP32, finite."""
    if not isinstance(tensor, torch.Tensor):
        raise HybridQPayloadError(f"{what} must be a torch.Tensor")
    shape = tuple(tensor.shape)
    if shape != contract.SPLIT_SHAPE and shape[1:] != contract.SPLIT_SHAPE:
        raise HybridQPayloadError(
            f"{what} must be {list(contract.SPLIT_SHAPE)} or "
            f"[B, {', '.join(str(size) for size in contract.SPLIT_SHAPE)}], got {list(shape)}"
        )
    if tensor.dim() == 4 and tensor.shape[0] < 1:
        raise HybridQPayloadError(f"{what} must have at least one frame")
    if tensor.dtype is not torch.float32:
        raise HybridQPayloadError(f"{what} must be float32, got {tensor.dtype}")
    if check_finite:
        require_finite(tensor, what)
    return tensor


def require_frozen_scores(scores: torch.Tensor, *, what: str = "ranker scores") -> torch.Tensor:
    """Production boundary for a single frame of per-cell scores: [112,192]."""
    if not isinstance(scores, torch.Tensor):
        raise HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(scores.shape) != contract.SPLIT_SPATIAL_SHAPE:
        raise HybridQPayloadError(
            f"{what} must be {list(contract.SPLIT_SPATIAL_SHAPE)}, got {list(scores.shape)}"
        )
    return require_finite(scores, what)


def require_frozen_header_dims(
    channels: int, height: int, width: int, dtype_code: int, *, fp32_code: int
) -> None:
    """A decoded header must describe exactly the frozen C2 tensor in FP32."""
    if (channels, height, width) != contract.SPLIT_SHAPE:
        raise HybridQPayloadError(
            f"header describes {[channels, height, width]}, "
            f"contract requires {list(contract.SPLIT_SHAPE)}"
        )
    if dtype_code != fp32_code:
        raise HybridQPayloadError("header dtype is not FP32")


def require_generic_c2(
    tensor: torch.Tensor,
    *,
    channels: int,
    dtype: torch.dtype = torch.float32,
    what: str = "C2 tensor",
) -> torch.Tensor:
    """Shape/dtype check for the private generic path used by layout tests."""
    if not isinstance(tensor, torch.Tensor):
        raise HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tensor.dim() != 3:
        raise HybridQPayloadError(f"{what} must be [C,H,W], got {list(tensor.shape)}")
    if tensor.shape[0] != channels:
        raise HybridQPayloadError(
            f"{what} must have {channels} channels, got {tensor.shape[0]}"
        )
    if tensor.dtype is not dtype:
        raise HybridQPayloadError(f"{what} must be {dtype}, got {tensor.dtype}")
    return tensor


# ---------------------------------------------------------------------------
# Selection integrity
# ---------------------------------------------------------------------------


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


def require_selection_integrity(
    selection,
    q: float,
    *,
    cells: int,
    spatial_shape: tuple[int, int],
) -> None:
    """Cross-check a selection against the requested q before it reaches the wire.

    Verifies q agreement, cell count, keep/drop counts against the registered
    formula, mask shape, mask popcount, index ordering/uniqueness, and that the
    mask and the index set describe exactly the same cells.
    """
    if contract._q_to_e4(float(selection.q)) != contract._q_to_e4(float(q)):
        raise HybridQPayloadError(
            f"selection q={selection.q!r} does not match requested q={q!r}"
        )
    if int(selection.cells) != int(cells):
        raise HybridQPayloadError(
            f"selection covers {selection.cells} cells, contract requires {cells}"
        )
    expected_keep = contract.keep_count(q, cells)
    expected_drop = contract.drop_count(q, cells)
    require_keep_cardinality(int(selection.keep_count), expected_keep)
    if int(selection.drop_count) != expected_drop:
        raise HybridQPayloadError(
            f"selection drop count {selection.drop_count} != registered {expected_drop}"
        )
    if tuple(selection.keep_mask.shape) != tuple(spatial_shape):
        raise HybridQPayloadError(
            f"selection mask shape {list(selection.keep_mask.shape)} != "
            f"{list(spatial_shape)}"
        )
    if selection.keep_mask.dtype is not torch.bool:
        raise HybridQPayloadError("selection mask must be boolean")
    flat_mask = selection.keep_mask.reshape(-1)
    if int(flat_mask.numel()) != int(cells):
        raise HybridQPayloadError("selection mask does not cover every cell")
    if int(flat_mask.sum()) != expected_keep:
        raise HybridQPayloadError(
            f"selection mask popcount {int(flat_mask.sum())} != keep count {expected_keep}"
        )
    indices = require_sorted_unique_indices(
        selection.keep_indices.to(torch.int64).cpu(), cells
    )
    require_keep_cardinality(int(indices.numel()), expected_keep)
    from_mask = torch.nonzero(flat_mask.cpu(), as_tuple=False).reshape(-1)
    if not torch.equal(from_mask, indices):
        raise HybridQPayloadError(
            "selection mask and retained index set describe different cells"
        )


# ---------------------------------------------------------------------------
# Frozen ownership and module state
# ---------------------------------------------------------------------------


def require_frozen_perception(modules: Iterable[torch.nn.Module]) -> None:
    """Every parameter of the frozen perception stack must be non-trainable."""
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise HybridQOwnershipError(
                    f"frozen perception parameter '{name}' has requires_grad=True"
                )


def require_eval_mode(modules: Iterable[torch.nn.Module]) -> None:
    """The frozen perception stack must run in evaluation mode."""
    for module in modules:
        for name, submodule in module.named_modules():
            if submodule.training:
                raise HybridQOwnershipError(
                    f"frozen perception module '{name or type(module).__name__}' "
                    "is in training mode"
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


def snapshot_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot every parameter *and* buffer of a frozen module."""
    snapshot = {
        f"param:{name}": value.detach().clone()
        for name, value in module.named_parameters()
    }
    snapshot.update(
        {
            f"buffer:{name}": value.detach().clone()
            for name, value in module.named_buffers()
            if isinstance(value, torch.Tensor)
        }
    )
    return snapshot


def require_parameters_unchanged(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    current = dict(module.named_parameters())
    if set(current) != set(snapshot):
        raise HybridQOwnershipError("frozen parameter set changed identity")
    for name, value in current.items():
        if not torch.equal(value.detach(), snapshot[name]):
            raise HybridQOwnershipError(f"frozen parameter '{name}' changed value")


def require_module_state_unchanged(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    """Every frozen parameter and buffer must be exactly unchanged."""
    current = snapshot_module_state(module)
    if set(current) != set(snapshot):
        raise HybridQOwnershipError("frozen module state set changed identity")
    for name, value in current.items():
        if not torch.equal(value, snapshot[name]):
            raise HybridQOwnershipError(f"frozen module state '{name}' changed value")


def require_module_parameters_finite(module: torch.nn.Module, what: str) -> None:
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            raise HybridQNumericalError(f"{what} parameter '{name}' is non-finite")


def require_finite_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """All tensor-valued optimizer state must remain finite after a step."""
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise HybridQNumericalError(
                    f"optimizer state '{key}' is non-finite for a ranker parameter"
                )
