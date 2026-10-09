from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import load_recovery_config


@dataclass(frozen=True)
class YawResult:
    value: torch.Tensor
    raw_norm: torch.Tensor
    below_tau: torch.Tensor
    diagnostics: Mapping[str, Any]


def require_candidate_tau(tau: float | None) -> float:
    if tau is None or not math.isfinite(float(tau)):
        raise RuntimeError("yaw tau is missing or nonfinite")
    candidates = tuple(float(value) for value in load_recovery_config()["yaw"]["candidate_tau"])
    value = float(tau)
    if value not in candidates:
        raise RuntimeError(f"yaw tau {value!r} is not preregistered")
    return value


def normalize_yaw_fp32(raw_yaw: torch.Tensor, tau: float | None) -> YawResult:
    """Shared train/inference yaw map; no fallback and no target change."""
    floor = require_candidate_tau(tau)
    if not isinstance(raw_yaw, torch.Tensor) or raw_yaw.ndim < 1 or raw_yaw.shape[-1] != 2:
        raise ValueError("raw yaw must be a tensor with final dimension two")
    raw = raw_yaw.float()
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("nonfinite raw yaw")
    # A scaled FP32 L2 norm avoids sum-of-squares overflow.  The additive one
    # exists only on exactly-zero rows, preventing an undefined sqrt gradient;
    # it does not perturb any nonzero norm.
    scale = raw.abs().amax(dim=-1, keepdim=True)
    zero = scale == 0
    scaled = raw / torch.where(zero, torch.ones_like(scale), scale)
    norm = scale * torch.sqrt(scaled.square().sum(dim=-1, keepdim=True) + zero.to(raw.dtype))
    if not bool(torch.isfinite(norm).all()):
        raise FloatingPointError("nonfinite raw yaw norm")
    below = norm < floor
    value = raw / norm.clamp_min(floor)
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("nonfinite normalized yaw")
    count = int(norm.numel())
    affected = int(below.sum().item())
    diagnostics = {
        "tau": floor,
        "count": count,
        "below_tau_count": affected,
        "below_tau_fraction": affected / count if count else 0.0,
        "raw_norm_min": float(norm.detach().min()) if count else None,
        "raw_norm_max": float(norm.detach().max()) if count else None,
        "raw_norm_mean": float(norm.detach().mean()) if count else None,
        "calculation_dtype": str(raw.dtype),
        "fallback_used": False,
    }
    return YawResult(value=value, raw_norm=norm.squeeze(-1), below_tau=below.squeeze(-1), diagnostics=diagnostics)


def exp_dimensions_fp64(log_dimensions: torch.Tensor) -> torch.Tensor:
    """Fail-closed FP64 dimension exponential without a physical clamp."""
    if not isinstance(log_dimensions, torch.Tensor) or log_dimensions.shape[-1] != 3:
        raise ValueError("log dimensions must have final dimension three")
    value = log_dimensions.double()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("nonfinite log dimensions")
    if value.numel():
        lower = math.log(math.nextafter(0.0, 1.0))
        upper = math.log(torch.finfo(torch.float64).max)
        if bool(((value < lower) | (value > upper)).any()):
            raise OverflowError(f"log dimension outside finite FP64 exp range [{lower}, {upper}]")
    decoded = torch.exp(value)
    if not bool(torch.isfinite(decoded).all()) or bool((decoded <= 0).any()):
        raise FloatingPointError("dimension exponential produced invalid physical dimension")
    return decoded
