from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .audit import audit_tree, require_finite_audit
from .base_runtime import load_base
from .safe_math import exp_dimensions_fp64, normalize_yaw_fp32


def geometry_losses(model: torch.nn.Module, outputs: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                    matched_images: Sequence[torch.Tensor], *,
                    audit_detail: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    """Original registered geometry loss, changing only the shared yaw map."""
    base = load_base()
    num = [value.shape[1] for value in outputs["detection"]["per_level"]["cls_logits"]]
    anchors_by_image = [list(value.split(num)) for value in outputs["anchors"]]
    actor_terms: dict[str, list[torch.Tensor]] = {name: [] for name in base.losses.GEOMETRY_INTERNAL}
    carrier_counts, yaw_diagnostics, carrier_identities, numerical_tensor_audits = [], [], [], []
    for image_index, target in enumerate(targets):
        offset = 0
        target_on = {name: value.to(outputs["c2"].device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for name, value in target.items()}
        image_actor_losses: dict[str, list[torch.Tensor]] = {name: [] for name in base.losses.GEOMETRY_INTERNAL}
        image_actor_indices = []
        for level_index, count in enumerate(num):
            match = matched_images[image_index][offset:offset + count]; offset += count
            positive = match >= 0
            if not bool(positive.any()):
                continue
            point = torch.where(positive)[0]; actor = match[positive]; label = target_on["labels"][actor]
            raw = outputs["geometry"][level_index]
            row = torch.arange(len(point), device=point.device)
            gathered = {name: value[image_index, point, label] for name, value in raw.items()}
            logits = gathered["depth_bin_logits"].float(); target_bin = target_on["depth_bin"][actor]
            bin_loss = F.cross_entropy(logits, target_bin, reduction="none")
            bounded_residuals = 0.5 * torch.tanh(gathered["depth_bin_residuals"].float())
            selected = bounded_residuals[row, target_bin]
            residual_loss = F.smooth_l1_loss(selected, target_on["depth_residual"][actor], reduction="none")
            anchors = anchors_by_image[image_index][level_index]
            centers = (anchors[point, :2] + anchors[point, 2:]) / 2; sizes = anchors[point, 2] - anchors[point, 0]
            if bool((sizes == 0).any()):
                raise FloatingPointError("zero anchor size in physical-ray division")
            ray_target = (target_on["physical_uv"][actor] - centers) / sizes[:, None]
            ray_loss = F.smooth_l1_loss(gathered["physical_ray"].float(), ray_target, reduction="none").mean(1)
            edges = model.depth_edges_m.float().to(point.device)
            zl, zu = torch.log1p(edges[:-1]), torch.log1p(edges[1:])
            decoded_bins = torch.expm1(0.5 * (zl + zu)[None] + bounded_residuals * (zu - zl)[None]).clamp(0.0, 40.0)
            probabilities = F.softmax(logits, dim=1)
            decoded_depth = (probabilities[:, :base.data.DEPTH_BINS] * decoded_bins).sum(1) + probabilities[:, base.data.DEPTH_BINS] * 40.0
            uv = centers + sizes[:, None] * gathered["physical_ray"].float(); intrinsic = target_on["intrinsic"].float()
            if bool((torch.stack((intrinsic[0, 0], intrinsic[1, 1])) == 0).any()):
                raise FloatingPointError("zero intrinsic division denominator")
            local = torch.stack((decoded_depth, decoded_depth * (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                                 decoded_depth * (intrinsic[1, 2] - uv[:, 1]) / intrinsic[1, 1]), dim=1)
            endpoint = F.smooth_l1_loss(local / 3.0, target_on["local_xyz"][actor] / 3.0,
                                        reduction="none").mean(1)
            if bool((target_on["dimensions"][actor] <= 0).any()):
                raise FloatingPointError("nonpositive dimension target before logarithm")
            dimensions = F.smooth_l1_loss(gathered["log_dimensions"].float(),
                                           torch.log(target_on["dimensions"][actor]), reduction="none").mean(1)
            yaw_result = normalize_yaw_fp32(gathered["yaw"], model._recovery_tau)
            yaw_loss = F.smooth_l1_loss(yaw_result.value, target_on["yaw"][actor], reduction="none").mean(1)
            if audit_detail:
                yaw_diagnostics.append({"image_index": image_index, "fpn_level": base.model.LEVELS[level_index],
                                        **yaw_result.diagnostics})
                with torch.no_grad():
                    dimensions_decoded = exp_dimensions_fp64(gathered["log_dimensions"].detach())
                    homogeneous = torch.cat((local.detach().double(), torch.ones(
                        len(local), 1, device=local.device, dtype=torch.float64)), dim=1)
                    world = (homogeneous @ target_on["extrinsic"].to(
                        device=local.device, dtype=torch.float64).T)[:, :3]
                    tensors = {
                        "depth_bin_logits": logits.detach(), "depth_bin_probabilities": probabilities.detach(),
                        "bounded_depth_residuals": bounded_residuals.detach(), "decoded_depth": decoded_depth.detach(),
                        "physical_ray_offsets": gathered["physical_ray"].detach(), "physical_uv": uv.detach(),
                        "local_xyz": local.detach(), "world_xyz": world,
                        "log_dimensions": gathered["log_dimensions"].detach(),
                        "exponentiated_dimensions": dimensions_decoded,
                        "raw_yaw": gathered["yaw"].detach(), "raw_yaw_norm": yaw_result.raw_norm.detach(),
                        "normalized_yaw": yaw_result.value.detach(),
                    }
                    records = audit_tree(tensors, "geometry")
                    require_finite_audit(records)
                    numerical_tensor_audits.append({"image_index": image_index,
                        "fpn_level": base.model.LEVELS[level_index], "records": records})
                for local_index in range(len(point)):
                    actor_index = int(actor[local_index]); source = target_on.get("source_identity", [])
                    carrier_identities.append({"image_index": image_index, "sample_id": target_on.get("sample_id"),
                        "actor_index": actor_index, "source_identity": source[actor_index] if source else None,
                        "class_index": int(label[local_index]), "fpn_level": base.model.LEVELS[level_index],
                        "point_index": int(point[local_index]),
                        "raw_yaw": gathered["yaw"][local_index].detach().float().cpu().tolist(),
                        "raw_yaw_norm": float(yaw_result.raw_norm[local_index]),
                        "normalized_yaw": yaw_result.value[local_index].detach().float().cpu().tolist(),
                        "below_tau": bool(yaw_result.below_tau[local_index])})
            for name, value in (("depth_bin", bin_loss), ("depth_residual", residual_loss),
                                ("physical_ray", ray_loss), ("endpoint", endpoint),
                                ("dimensions", dimensions), ("yaw", yaw_loss)):
                image_actor_losses[name].append(value)
            image_actor_indices.append(actor)
        if image_actor_indices:
            actors = torch.cat(image_actor_indices); carrier_counts.append(int(len(actors)))
            for name in base.losses.GEOMETRY_INTERNAL:
                carrier_loss = torch.cat(image_actor_losses[name])
                for actor_index in range(len(target_on["labels"])):
                    mask = actors == actor_index
                    if bool(mask.any()):
                        actor_terms[name].append(carrier_loss[mask].mean())
        else:
            carrier_counts.append(0)
    reference = outputs["c2"]
    parts = {name: torch.stack(values).mean() if values else reference.sum() * 0.0
             for name, values in actor_terms.items()}
    total = sum(weight * parts[name] for name, weight in base.losses.GEOMETRY_INTERNAL.items())
    return total, parts, {"carriers_per_image": carrier_counts, "normalization": "per_actor_over_complete_batch",
                          "yaw_normalization": yaw_diagnostics, "carrier_identities": carrier_identities,
                          "numerical_tensor_audits": numerical_tensor_audits,
                          "detailed_audit_enabled": audit_detail,
                          "target_and_loss_unchanged": True}


def compute_loss_groups(model: torch.nn.Module, batch: Mapping[str, Any], multipliers: Mapping[str, float] | None = None,
                        *, use_amp: bool = True,
                        audit_detail: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any], Mapping[str, Any]]:
    base = load_base()
    inputs = batch["input"].to(next(model.parameters()).device, non_blocking=True)
    targets = batch["targets"]; amp = inputs.device.type == "cuda" and use_amp
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = model(inputs, dense=True)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        detection, detection_parts, matched, assignment = base.losses.detection_losses(model, outputs, targets)
        geometry, geometry_parts, geometry_audit = geometry_losses(
            model, outputs, targets, matched, audit_detail=audit_detail)
        semantic, semantic_parts = base.losses.semantic_loss(outputs["semantic_logits"], targets)
        auxiliary, auxiliary_parts, auxiliary_audit = base.losses.dense_losses(outputs, batch)
    groups = {"D": detection, "G": geometry, "S": semantic, "A": auxiliary}
    weights = {"D": 1.0, "G": 1.0, "S": 1.0, "A": 1.0}
    if multipliers is not None:
        weights.update({name: float(multipliers[name]) for name in ("G", "S", "A")})
    total = sum(weights[name] * groups[name] for name in ("D", "G", "S", "A"))
    pressure = total.detach().abs().clamp_min(1e-12)
    components = {**{f"fcos_{name}": value for name, value in detection_parts.items()},
                  **{f"geometry_{name}": value for name, value in geometry_parts.items()},
                  **semantic_parts, **auxiliary_parts, **groups,
                  **{f"weighted_{name}": weights[name] * groups[name] for name in groups},
                  **{f"optimization_fraction_{name}": (weights[name] * groups[name]).detach() / pressure for name in groups},
                  "total": total}
    audit = {"assignment": assignment, "geometry": geometry_audit, "auxiliary": auxiliary_audit,
             "multipliers": weights, "yaw_train_inference_shared": True}
    return total, components, audit, outputs
