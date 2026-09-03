"""Phase-9C AE128 scientific training: 4 Stage-A epochs then 8 Stage-B epochs.

One scientific run. AE128 is the only learned component: the frozen perception
stack and the stable epoch-4 ranker stay in eval mode with `requires_grad=False`,
the optimizer owns exactly the eight named AE tensors, and every optimizer batch
is drawn from the 13,543 registered fit frames. The reserved train-holdout is
never opened here — selection is a separate command — and no validation or test
frame is read.

Stage A is dense q=0 for epochs 1-4. Stage B is a continuous round robin over
q = {0, 0.30, 0.50, 0.70} for epochs 5-12, one q per batch, with the cycle
position carried across epoch boundaries so the 6,776 Stage-B updates split
exactly 1,694 per q. q=0.90 and q=0.98 are evaluation/emergency settings and are
never optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards, training
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    encode_front,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    require_no_validation_or_test,
    source_delta,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_training import (
    epoch_loader,
    epoch_order,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.teacher_cache import (
    SplitPartition,
    build_split_partition,
)
from . import ae_composition, ae_contract, ae_loss, ae_training_common as common

EXECUTE_TOKEN = "SPLITFUSION_AE128_PHASE9C_TRAINING"
TERMINAL = "SPLITFUSION_AE128_TRAINING_COMPLETE"
SCHEMA = common.AE_TRAINING_SCHEMA
DATALOADER_WORKERS = 8

# The holdout must not be reachable from this command at all.
HOLDOUT_MODULE = (
    "pole_lraspp_multimodal_fusion.object_head_pilot_v1."
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1.ae_holdout_selection"
)


def require_holdout_unopened() -> None:
    """Structural guard: the selection runner is a separate command, full stop."""
    if HOLDOUT_MODULE in sys.modules:
        raise guards.HybridQOwnershipError(
            "the holdout selection module is loaded inside the training command"
        )


# ---------------------------------------------------------------------------
# One update
# ---------------------------------------------------------------------------


def _total_grad_norm(module: torch.nn.Module) -> float:
    squares = [
        float(parameter.grad.detach().norm()) ** 2
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return float(math.sqrt(sum(squares)))


def optimizer_update(
    autoencoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    frozen: Mapping[str, torch.nn.Module],
    loss: torch.Tensor,
    qualification: training.GradientQualification,
) -> dict[str, Any]:
    """One update with the full runtime-safety contract enforced around it."""
    if not torch.isfinite(loss.detach()).all():
        raise guards.HybridQNumericalError("non-finite total AE training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nonzero = qualification.observe(autoencoder, loss=loss)
    # clip_grad_norm_ returns the norm observed *before* clipping.
    pre_clip = float(training.clip_ranker_gradients(autoencoder))
    post_clip = _total_grad_norm(autoencoder)
    if not (math.isfinite(pre_clip) and math.isfinite(post_clip)):
        raise guards.HybridQNumericalError("non-finite AE gradient norm")
    optimizer.step()
    training.require_post_step_health(autoencoder, optimizer)
    guards.require_optimizer_owns_only(optimizer, autoencoder.parameters())
    for name, module in frozen.items():
        for parameter_name, parameter in module.named_parameters():
            if parameter.grad is not None:
                raise guards.HybridQOwnershipError(
                    f"frozen {name} parameter '{parameter_name}' received a gradient"
                )
    return {
        "pre_clip_global_grad_norm": pre_clip,
        "post_clip_global_grad_norm": post_clip,
        "clipped": pre_clip > common.AE_GRAD_CLIP_GLOBAL_NORM,
        "all_tensors_nonzero": bool(nonzero),
    }


# ---------------------------------------------------------------------------
# Epoch accounting
# ---------------------------------------------------------------------------


@dataclass
class QAccumulator:
    updates: int = 0
    frames: int = 0
    total: float = 0.0
    plain: float = 0.0
    combined: float = 0.0
    worst_total: float = 0.0

    def observe(self, total: float, plain: float, combined: float, frames: int) -> None:
        self.updates += 1
        self.frames += int(frames)
        self.total += float(total)
        self.plain += float(plain)
        self.combined += float(combined)
        self.worst_total = max(self.worst_total, float(total))

    def summary(self) -> dict[str, Any]:
        updates = max(1, self.updates)
        return {
            "updates": self.updates,
            "frames": self.frames,
            "mean_total_loss": self.total / updates,
            "mean_plain_reconstruction": self.plain / updates,
            "mean_combined_importance_reconstruction": self.combined / updates,
            "max_batch_total_loss": self.worst_total,
        }


@dataclass
class EpochAccumulator:
    epoch: int
    stage: str
    learning_rate: float
    updates: int = 0
    frames: int = 0
    clipped_updates: int = 0
    max_pre_clip_norm: float = 0.0
    per_q: dict[str, QAccumulator] = field(default_factory=dict)
    group_availability: dict[str, int] = field(default_factory=dict)
    group_exclusions: dict[str, int] = field(default_factory=dict)
    min_valid_groups: int = len(ae_contract.AE_TASK_GROUPS)

    def observe(self, q: float, report: Mapping[str, Any], health: Mapping[str, Any]) -> None:
        key = f"{float(q):.2f}"
        bucket = self.per_q.setdefault(key, QAccumulator())
        bucket.observe(
            report["total"],
            report["plain_reconstruction"],
            report["combined_importance_reconstruction"],
            report["frames"],
        )
        self.updates += 1
        self.frames += int(report["frames"])
        self.clipped_updates += int(health["clipped"])
        self.max_pre_clip_norm = max(
            self.max_pre_clip_norm, float(health["pre_clip_global_grad_norm"])
        )
        for name, count in report["group_availability"].items():
            self.group_availability[name] = self.group_availability.get(name, 0) + int(count)
        for name, reasons in report["excluded_groups"].items():
            self.group_exclusions[name] = self.group_exclusions.get(name, 0) + sum(
                int(value) for value in reasons.values()
            )
        self.min_valid_groups = min(
            self.min_valid_groups, int(report["min_valid_groups_observed"])
        )

    def summary(
        self,
        *,
        seconds: float,
        peak_allocated: float,
        peak_reserved: float,
        qualification: training.GradientQualification,
    ) -> dict[str, Any]:
        updates = max(1, self.updates)
        totals = {key: bucket.summary() for key, bucket in sorted(self.per_q.items())}
        return {
            "epoch": self.epoch,
            "stage": self.stage,
            "learning_rate": self.learning_rate,
            "optimizer_updates": self.updates,
            "frames": self.frames,
            "mean_total_loss_this_stage": sum(
                bucket.total for bucket in self.per_q.values()
            )
            / updates,
            "per_q": totals,
            "loss_comparability_note": (
                "losses are recorded separately by stage and q; a Stage-A epoch is "
                "dense q=0 only, so its aggregate is not comparable with a Stage-B "
                "aggregate that mixes four different keep budgets"
            ),
            "teacher_group_availability": dict(sorted(self.group_availability.items())),
            "teacher_group_exclusions": dict(sorted(self.group_exclusions.items())),
            "min_valid_groups_observed": self.min_valid_groups,
            "clipped_updates": self.clipped_updates,
            "clipped_fraction": self.clipped_updates / updates,
            "max_pre_clip_global_grad_norm": self.max_pre_clip_norm,
            "clipping_is_not_a_failure": True,
            "peak_allocated_vram_mib": peak_allocated,
            "peak_reserved_vram_mib": peak_reserved,
            "wall_seconds": seconds,
            "gradient_qualification": {
                "window": qualification.window,
                "observed_updates": qualification.seen,
                "named_ae_tensors": list(qualification.parameter_names),
                "qualified": qualification.qualified(),
                "disconnected_tensors": list(qualification.disconnected()),
                "never_nonzero_tensors": list(qualification.never_nonzero()),
                "zero_gradient_batch_count": len(qualification.zero_gradient_batches),
                "missing_gradient_batch_count": len(qualification.missing_gradient_batches),
                "policy": "each named AE tensor must be finite and nonzero at least "
                "once per epoch, not on every batch",
            },
        }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    global_update_index: int = 0
    stage_b_position: int = 0


def run_epoch(
    *,
    epoch: int,
    model: torch.nn.Module,
    base: Any,
    ranker: torch.nn.Module,
    autoencoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Any,
    partition: SplitPartition,
    store: common.AeTeacherStore,
    device: torch.device,
    state: TrainingState,
    workers: int,
    batch_limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    require_holdout_unopened()
    stage = common.stage_for_epoch(epoch)
    learning_rate = common.set_learning_rate(
        optimizer, common.learning_rate_for_stage(stage)
    )
    order = epoch_order(partition, epoch)
    loader = epoch_loader(base, dataset, order, workers=workers)
    expected_ids = [dataset.rows[index]["sample_id"] for index in order]
    planned = len(loader) if batch_limit is None else min(len(loader), batch_limit)
    if batch_limit is None and planned != common.batches_per_epoch():
        raise guards.HybridQConfigError(
            f"epoch {epoch} planned {planned} batches != {common.batches_per_epoch()}"
        )
    accumulator = EpochAccumulator(epoch=epoch, stage=stage, learning_rate=learning_rate)
    qualification = training.GradientQualification.for_module(autoencoder, window=planned)
    frozen = {"perception": model, "ranker": ranker}
    observed_ids: list[str] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(loader):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        sample_ids = [str(value) for value in batch["sample_ids"]]
        observed_ids.extend(sample_ids)
        if store.other_split_ids.intersection(sample_ids):
            raise guards.HybridQOwnershipError(
                "a reserved holdout frame entered an optimizer batch"
            )
        teacher = store.batch(sample_ids)

        if stage == common.AE_STAGE_A:
            q = float(common.AE_STAGE_A_Q)
        else:
            q = common.stage_b_q_at(state.stage_b_position)
        ae_loss.require_optimization_q(q)

        c2 = encode_front(model, batch, device)
        composition = ae_composition.compose_batch(c2, autoencoder, ranker, q)
        reconstructed = autoencoder.decode(
            composition.masked_latent, composition.keep_mask
        )
        loss = ae_loss.task_aware_reconstruction_loss(c2, reconstructed, teacher)
        health = optimizer_update(autoencoder, optimizer, frozen, loss.total, qualification)

        if stage == common.AE_STAGE_B:
            state.stage_b_position += 1
        state.global_update_index += 1
        accumulator.observe(q, loss.report(), health)
        del c2, composition, reconstructed, loss, teacher, batch

        if progress and planned >= 100 and (batch_index + 1) % max(1, planned // 4) == 0:
            print(
                f"[ae128] epoch {epoch} ({stage}) {batch_index + 1}/{planned} batches, "
                f"{(time.time() - started) / 60.0:.1f} min",
                flush=True,
            )

    del loader
    seconds = time.time() - started
    if batch_limit is None:
        if observed_ids != expected_ids:
            raise guards.HybridQConfigError(
                "epoch batch order drifted from the seeded order"
            )
        if len(set(observed_ids)) != contract.TRAIN_FIT_FRAMES:
            raise guards.HybridQConfigError(
                "an epoch did not cover every fit frame exactly once"
            )
        if set(observed_ids) & store.other_split_ids:
            raise guards.HybridQOwnershipError(
                "a reserved holdout frame entered an optimizer batch"
            )
        qualification.require_qualified()

    summary = accumulator.summary(
        seconds=seconds,
        peak_allocated=torch.cuda.max_memory_allocated(device) / 2 ** 20,
        peak_reserved=torch.cuda.max_memory_reserved(device) / 2 ** 20,
        qualification=qualification,
    )
    summary["epoch_shuffle_seed"] = contract.epoch_shuffle_seed(epoch)
    summary["epoch_sample_id_sha256"] = contract.sample_id_digest(observed_ids)
    summary["holdout_frames_in_optimizer_batches"] = 0
    summary["global_update_index"] = state.global_update_index
    summary["stage_b_cycle_position"] = state.stage_b_position
    return summary


def order_identity(partition: SplitPartition, epoch: int, dataset: Any) -> dict[str, Any]:
    """Exactly what an exact resume needs to reproduce this epoch's sampler."""
    order = epoch_order(partition, epoch)
    return {
        "epoch": int(epoch),
        "shuffle_seed": contract.epoch_shuffle_seed(epoch),
        "batch_size": common.AE_BATCH_SIZE,
        "drop_last": common.AE_DROP_LAST,
        "frames": len(order),
        "batches": common.batches_per_epoch(),
        "sample_id_sha256": contract.sample_id_digest(
            dataset.rows[index]["sample_id"] for index in order
        ),
    }


def latest_recovery(recovery_dir: Path) -> tuple[int, Path] | None:
    candidates = sorted(recovery_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    path = candidates[-1]
    return int(path.stem.split("_")[1]), path


def load_recovery(
    path: Path,
    autoencoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != common.AE_RECOVERY_SCHEMA:
        raise guards.HybridQConfigError("AE recovery schema drift")
    for name in (
        "perception_checkpoint_sha256",
        "stable_ranker_sha256",
        "locked_config_sha256",
        "teacher_cache_manifest_sha256",
    ):
        expected = {
            "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
            "stable_ranker_sha256": binding["stable_epoch4_ranker"]["sha256"],
            "locked_config_sha256": binding["hybrid_q_locked_config"]["sha256"],
            "teacher_cache_manifest_sha256": binding["teacher_cache_manifest"]["sha256"],
        }[name]
        if payload[name] != expected:
            raise guards.HybridQConfigError(f"recovery {name} drift; refusing to resume")
    if payload["configuration"] != common.training_configuration():
        raise guards.HybridQConfigError(
            "recovery was written under a different training configuration"
        )
    autoencoder.load_state_dict(payload["autoencoder"])
    optimizer.load_state_dict(payload["optimizer"])
    state.global_update_index = int(payload["global_update_index"])
    state.stage_b_position = int(payload["stage_b_cycle_position"])
    common.restore_rng(payload["rng"])
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-9C SplitFusion AE128 scientific training (fit frames only)"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    require_holdout_unopened()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-9C AE128 training requires CUDA")
    device = torch.device("cuda:0")

    seed = ae_contract.AE_INIT_BASE_SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    binding = common.bind_frozen_inputs()
    delta = source_delta(binding)
    schedule_counts = common.require_balanced_stage_b()
    print(
        f"[ae128] inputs bound: {binding['teacher_cache_shards']['verified']}/66 shards; "
        f"Stage-B schedule {schedule_counts}",
        flush=True,
    )

    model, base, perception = load_frozen_perception(device)
    common.freeze(model)
    ranker = common.load_stable_ranker(device)
    guards.require_frozen_perception([model, ranker])
    guards.require_eval_mode([model, ranker])
    model_hashes, model_aggregate = common.state_hashes(model)
    ranker_hashes, ranker_aggregate = common.state_hashes(ranker)
    frozen_model_state = guards.snapshot_module_state(model)
    frozen_ranker_state = guards.snapshot_module_state(ranker)

    dataset = build_train_dataset(base)
    if getattr(dataset, "augment", False) != common.AE_AUGMENTATION:
        raise guards.HybridQConfigError("augmentation must be off for AE training")
    partition = build_split_partition(dataset)
    split_check = require_no_validation_or_test(
        dataset, (row["sample_id"] for row in dataset.rows)
    )
    store = common.load_ae_teacher_store(partition, "fit")
    if store.split != "fit" or store.frames != contract.TRAIN_FIT_FRAMES:
        raise guards.HybridQConfigError("the optimizer teacher store is not the fit split")
    print(
        f"[ae128] fit teacher store: {store.bytes / 2 ** 30:.2f} GiB, "
        f"{store.frames} maps; {len(store.other_split_ids)} holdout maps withheld",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    recovery_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    recovery_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    autoencoder = common.build_ae(device)
    optimizer = common.build_ae_optimizer(
        autoencoder,
        lr=common.learning_rate_for_stage(common.AE_STAGE_A),
        frozen_modules=(model, ranker),
    )
    state = TrainingState()
    epoch_summaries: list[dict[str, Any]] = []
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    started_epoch = 1

    if args.resume:
        found = latest_recovery(recovery_dir)
        if found is not None:
            completed, path = found
            load_recovery(path, autoencoder, optimizer, state, binding)
            autoencoder.to(device)
            started_epoch = completed + 1
            summary_path = output / "epoch_summaries.json"
            if summary_path.is_file():
                epoch_summaries = json.loads(summary_path.read_text(encoding="utf-8"))[
                    "epochs"
                ]
            print(
                f"[ae128] resumed from completed epoch {completed}; continuing at "
                f"epoch {started_epoch} with Stage-B cycle position "
                f"{state.stage_b_position}",
                flush=True,
            )

    run_started = time.time()
    for epoch in range(started_epoch, common.AE_TRAINING_EPOCHS + 1):
        identity = order_identity(partition, epoch, dataset)
        summary = run_epoch(
            epoch=epoch,
            model=model,
            base=base,
            ranker=ranker,
            autoencoder=autoencoder,
            optimizer=optimizer,
            dataset=dataset,
            partition=partition,
            store=store,
            device=device,
            state=state,
            workers=int(args.workers),
        )
        if summary["epoch_sample_id_sha256"] != identity["sample_id_sha256"]:
            raise guards.HybridQConfigError("epoch order identity drift")

        guards.require_module_state_unchanged(model, frozen_model_state)
        guards.require_module_state_unchanged(ranker, frozen_ranker_state)
        after_model, after_model_aggregate = common.state_hashes(model)
        after_ranker, after_ranker_aggregate = common.state_hashes(ranker)
        if (
            after_model != model_hashes
            or after_ranker != ranker_hashes
            or after_model_aggregate != model_aggregate
            or after_ranker_aggregate != ranker_aggregate
        ):
            raise guards.HybridQOwnershipError(
                f"a frozen parameter or buffer hash changed during epoch {epoch}"
            )
        summary["frozen_perception_state_unchanged"] = True
        summary["frozen_ranker_state_unchanged"] = True
        summary["frozen_tensors_compared"] = len(model_hashes) + len(ranker_hashes)

        recovery_path = recovery_dir / f"epoch_{epoch:02d}.pt"
        recovery_hashes[recovery_path.name] = common.save_recovery(
            recovery_path,
            epoch=epoch,
            autoencoder=autoencoder,
            optimizer=optimizer,
            global_update_index=state.global_update_index,
            stage_b_position=state.stage_b_position,
            order_identity=identity,
            summary=summary,
            binding=binding,
        )
        if epoch in common.AE_CANDIDATE_EPOCHS:
            candidate_path = checkpoint_dir / common.candidate_filename(epoch)
            candidate_hashes[candidate_path.name] = common.save_candidate(
                candidate_path,
                epoch=epoch,
                autoencoder=autoencoder,
                global_update_index=state.global_update_index,
                stage_b_position=state.stage_b_position,
                binding=binding,
            )
            summary["candidate_checkpoint"] = candidate_path.name
            summary["candidate_checkpoint_sha256"] = candidate_hashes[candidate_path.name]

        epoch_summaries.append(summary)
        (output / "epoch_summaries.json").write_text(
            json.dumps({"schema": SCHEMA, "epochs": epoch_summaries}, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "stage": summary["stage"],
                    "lr": summary["learning_rate"],
                    "updates": summary["optimizer_updates"],
                    "per_q_mean_total": {
                        key: round(row["mean_total_loss"], 6)
                        for key, row in summary["per_q"].items()
                    },
                    "clipped_fraction": round(summary["clipped_fraction"], 6),
                    "minutes": round(summary["wall_seconds"] / 60.0, 2),
                }
            ),
            flush=True,
        )

    if state.global_update_index != common.AE_TRAINING_EPOCHS * common.batches_per_epoch():
        raise guards.HybridQConfigError("total optimizer update count drift")
    if state.stage_b_position != common.stage_b_updates_total():
        raise guards.HybridQConfigError("Stage-B cycle position drift")
    observed_q_counts: dict[str, int] = {}
    for summary in epoch_summaries:
        if summary["stage"] != common.AE_STAGE_B:
            continue
        for key, row in summary["per_q"].items():
            observed_q_counts[key] = observed_q_counts.get(key, 0) + int(row["updates"])
    if observed_q_counts != schedule_counts:
        raise guards.HybridQConfigError(
            f"realized Stage-B q counts {observed_q_counts} != locked {schedule_counts}"
        )
    if set(candidate_hashes) != {
        common.candidate_filename(epoch) for epoch in common.AE_CANDIDATE_EPOCHS
    }:
        raise guards.HybridQConfigError("candidate checkpoint set drift")
    require_holdout_unopened()

    report = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - run_started,
        "scope": {
            "family": "AE128",
            "ae64_or_ae32_trained": False,
            "optimization_split": "fit",
            "optimization_frames": contract.TRAIN_FIT_FRAMES,
            "holdout_opened_here": False,
            "holdout_frames_in_optimizer_batches": 0,
            "validation_or_test_accessed": False,
            "augmentation": common.AE_AUGMENTATION,
            "fake_quantization_or_zstd_in_training": False,
            "carla_launched": False,
            "checkpoint_selection_performed_here": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "configuration": common.training_configuration(),
        "realized_stage_b_q_counts": observed_q_counts,
        "binding": binding,
        "perception_binding": perception,
        "hybrid_q_source_delta_since_phase4": delta,
        "split_check": split_check,
        "frozen_state": {
            "perception_aggregate_sha256": model_aggregate,
            "ranker_aggregate_sha256": ranker_aggregate,
            "tensors_compared": len(model_hashes) + len(ranker_hashes),
            "unchanged_at_every_epoch_boundary": True,
        },
        "epochs": epoch_summaries,
        "recovery_checkpoints": recovery_hashes,
        "candidate_checkpoints": candidate_hashes,
        "next_step": (
            "run the separate holdout-selection command; it is not launched here"
        ),
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / "TRAINING_COMPLETE").write_text(
        f"{sha256_file(report_path)}\n", encoding="utf-8"
    )
    print(json.dumps({"report_sha256": sha256_file(report_path), "output": str(output)}))
    print(TERMINAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
