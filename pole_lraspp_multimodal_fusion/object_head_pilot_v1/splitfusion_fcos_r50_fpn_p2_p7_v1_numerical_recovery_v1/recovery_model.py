from __future__ import annotations

from typing import Any, Mapping

import torch

from .audit import audit_tree, require_finite_audit
from .base_runtime import load_base
from .safe_math import exp_dimensions_fp64, normalize_yaw_fp32, require_candidate_tau


def _decode_geometry(self: torch.nn.Module, raw: Mapping[str, torch.Tensor], anchors: torch.Tensor,
                     point_indices: torch.Tensor, labels: torch.Tensor, intrinsic: torch.Tensor,
                     extrinsic: torch.Tensor) -> dict[str, torch.Tensor]:
    """Original decode equation with only shared checked yaw and checked FP64 exp."""
    base = load_base()
    device = anchors.device
    row = torch.arange(len(point_indices), device=device)
    gathered = {name: value[point_indices, labels] for name, value in raw.items()}
    require_finite_audit(audit_tree(gathered, "decode.gathered"))
    with torch.autocast(device_type=device.type, enabled=False):
        logits = gathered["depth_bin_logits"].float()
        probabilities = torch.softmax(logits, dim=1)
        bins = logits.argmax(dim=1)
        in_range = bins < base.data.DEPTH_BINS
        safe_bins = bins.clamp(max=base.data.DEPTH_BINS - 1)
        edges = self.depth_edges_m.float().to(device)
        lower, upper = edges[safe_bins], edges[safe_bins + 1]
        residuals = 0.5 * torch.tanh(gathered["depth_bin_residuals"].float())
        selected_residual = residuals[row, safe_bins]
        zl, zu = torch.log1p(lower), torch.log1p(upper)
        log_depth = 0.5 * (zl + zu) + selected_residual * (zu - zl)
        depth = torch.where(in_range, torch.expm1(log_depth).clamp(0.0, 40.0), torch.full_like(log_depth, 40.0))
        sizes = anchors[point_indices, 2] - anchors[point_indices, 0]
        centers = (anchors[point_indices, :2] + anchors[point_indices, 2:]) / 2
        uv = centers + sizes[:, None] * gathered["physical_ray"].float()
        k = intrinsic.float().to(device)
        if not bool(torch.isfinite(k).all()) or bool((torch.stack((k[0, 0], k[1, 1])) == 0).any()):
            raise FloatingPointError("invalid intrinsic division denominator")
        local = torch.stack((depth, depth * (uv[:, 0] - k[0, 2]) / k[0, 0],
                             depth * (k[1, 2] - uv[:, 1]) / k[1, 1]), dim=1)
        homogeneous = torch.cat((local.double(), torch.ones(len(local), 1, device=device, dtype=torch.float64)), dim=1)
        world = (homogeneous @ extrinsic.to(device=device, dtype=torch.float64).T)[:, :3]
        dimensions = exp_dimensions_fp64(gathered["log_dimensions"])
        yaw_result = normalize_yaw_fp32(gathered["yaw"], self._recovery_tau)
    result = {"local_xyz": local, "world_xyz": world, "dimensions": dimensions,
              "yaw": yaw_result.value, "yaw_raw_norm": yaw_result.raw_norm,
              "yaw_below_tau": yaw_result.below_tau, "physical_uv": uv, "depth_bin": bins,
              "depth_residual": selected_residual, "depth": depth,
              "depth_bin_logits": logits, "depth_bin_probabilities": probabilities,
              "bounded_depth_residuals": residuals, "physical_ray_offsets": gathered["physical_ray"].float(),
              "log_dimensions": gathered["log_dimensions"].double(), "raw_yaw": gathered["yaw"].float()}
    require_finite_audit(audit_tree(result, "decode.result"))
    return result


def build_recovery_model(priors: Mapping[str, Any], tau: float, device: torch.device | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    floor = require_candidate_tau(tau)
    base = load_base()
    model, report = base.model.build_model(priors, device)
    recovery_type = type("NumericalRecoverySplitFusionFCOS", (base.model.SplitFusionFCOS,),
                         {"_decode_geometry": _decode_geometry, "__module__": __name__})
    model.__class__ = recovery_type
    model._recovery_tau = floor
    report = dict(report)
    report.update({"recovery_yaw_tau": floor, "shared_training_inference_normalizer": True,
                   "architecture_and_state_dict_unchanged": True})
    return model, report
