from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .guards import PreStepBreaker
from .recovery_losses import compute_loss_groups


def run_guarded_epoch(base: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loader: Any,
                      config: Mapping[str, Any], multipliers: Mapping[str, float], epoch: int,
                      global_update: int, accumulation: int, breaker: PreStepBreaker,
                      *, maximum_updates: int | None = None, enforce_required_nonzero: bool = False) -> tuple[int, dict[str, Any]]:
    """Original per-update checks plus the post-accumulation/pre-step breaker."""
    model.train(); totals = defaultdict(float); update_records = []; radar_norms = []; rgb_norms = []
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for update_in_epoch, microbatches in enumerate(base.train.microbatch_groups(loader, accumulation), 1):
        if maximum_updates is not None and update_in_epoch > maximum_updates:
            break
        lrs = base.train.scheduled_lrs(config, epoch, global_update + 1); base.train.set_lrs(optimizer, lrs)
        optimizer.zero_grad(set_to_none=True); update_loss = 0.0; update_parts = defaultdict(float)
        update_sample_ids: list[str] = []
        for batch in microbatches:
            update_sample_ids.extend(str(value) for value in batch.get("sample_ids", []))
            total, parts, _audit, _outputs = compute_loss_groups(model, batch, multipliers)
            scalars = base.losses.scalar_components(parts)
            if not base.common.finite_tree(scalars):
                raise FloatingPointError(f"nonfinite individual loss epoch={epoch} update={update_in_epoch}")
            (total / len(microbatches)).backward(); update_loss += float(total.detach()) / len(microbatches)
            for name, value in scalars.items():
                update_parts[name] += value / len(microbatches)
            del total, parts, _audit, _outputs
        if not base.train.all_gradients_finite(model):
            raise FloatingPointError(f"nonfinite gradient epoch={epoch} update={update_in_epoch}")
        required = base.train.required_gradient_evidence(model)
        failed = {name: value for name, value in required.items() if value["required_this_stage"] and
                  (not value["finite"] or (enforce_required_nonzero and not value["nonzero"]))}
        if failed:
            raise RuntimeError(f"required-gradient failure epoch={epoch} update={update_in_epoch}: {failed}")
        breaker_record = breaker.check(model, optimizer, epoch=epoch, update_in_epoch=update_in_epoch,
                                       global_update_if_stepped=global_update + 1,
                                       context={"loss": update_loss, "microbatches": len(microbatches),
                                                "sample_ids": update_sample_ids})
        radar_norm = base.train.gradient_norm(model.front.W_radar); rgb_norm = base.train.gradient_norm(model.front.W_rgb)
        optimizer.step(); global_update += 1
        if not base.train.all_model_finite(model):
            raise FloatingPointError(f"nonfinite parameter epoch={epoch} update={update_in_epoch}")
        if not base.train.optimizer_finite(optimizer):
            raise FloatingPointError(f"nonfinite optimizer epoch={epoch} update={update_in_epoch}")
        radar_norms.append(radar_norm); rgb_norms.append(rgb_norm)
        for name, value in update_parts.items():
            totals[name] += value
        update_records.append({"update_in_epoch": update_in_epoch, "global_update": global_update,
                               "loss": update_loss, "lrs": lrs, "microbatches": len(microbatches),
                               "radar_stem_gradient_norm": radar_norm, "rgb_stem_gradient_norm": rgb_norm,
                               "required_gradient_evidence": required, "numerical_guard": breaker_record, "finite": True})
    count = len(update_records)
    return global_update, {"epoch": epoch, "updates": count, "global_update": global_update,
        "mean_components": {name: value / max(1, count) for name, value in totals.items()},
        "last_lrs": update_records[-1]["lrs"] if update_records else {},
        "radar_stem_gradient_norm_mean": float(np.mean(radar_norms)) if radar_norms else 0.0,
        "rgb_stem_gradient_norm_mean": float(np.mean(rgb_norms)) if rgb_norms else 0.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "wall_seconds": time.monotonic() - started, "all_updates_finite": True,
        "pre_step_breaker_checked_every_update": True, "update_boundary_records": update_records}
