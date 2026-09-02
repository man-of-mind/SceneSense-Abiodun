"""Phase-5 hybrid-q ranker training: 4 distillation epochs then 8 q-aware epochs.

One scientific run. The ranker is the only learned component: the frozen epoch-26
perception stack stays in eval mode with `requires_grad=False`, the optimizer owns
exactly the 2,144 ranker parameters, and every optimizer batch is drawn from the
eight fit episodes only. No validation or test frame is read; the two reserved
train-holdout episodes never enter an optimizer batch.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import contract, guards, training
from .gpu_qualification import (
    build_train_dataset,
    encode_front,
    load_frozen_perception,
    loss_groups_from_c2,
    sha256_file,
)
from .phase5_common import (
    TeacherStore,
    bind_inputs,
    load_teacher_store,
    require_no_validation_or_test,
    source_delta,
)
from .ranker import build_ranker
from .teacher_cache import SplitPartition, build_split_partition

EXECUTE_TOKEN = "HYBRID_Q_PHASE5_RANKER_TRAINING"
SCHEMA = "splitfusion_fcos_hybrid_q_phase5_training_v1"
DATALOADER_WORKERS = 8
SMOKE_DISTILLATION_BATCHES = 2
SMOKE_Q_AWARE_BATCHES = 3


# ---------------------------------------------------------------------------
# Deterministic epoch order
# ---------------------------------------------------------------------------


def epoch_order(partition: SplitPartition, epoch: int) -> list[int]:
    """Every fit frame exactly once, in a deterministic epoch-specific shuffle."""
    generator = torch.Generator()
    generator.manual_seed(contract.epoch_shuffle_seed(epoch))
    permutation = torch.randperm(len(partition.fit_indices), generator=generator).tolist()
    order = [partition.fit_indices[position] for position in permutation]
    if len(order) != contract.TRAIN_FIT_FRAMES or len(set(order)) != contract.TRAIN_FIT_FRAMES:
        raise guards.HybridQConfigError("epoch order is not a permutation of the fit partition")
    if set(order) != set(partition.fit_indices):
        raise guards.HybridQConfigError("epoch order left the fit partition")
    return order


def epoch_loader(base: Any, dataset: Any, order: Sequence[int], *, workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(order)),
        batch_size=contract.TRAINING_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        collate_fn=base.data.collate,
        drop_last=contract.DROP_LAST_TRAINING_BATCH,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def distillation_loss(scores: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean per-frame listwise soft cross-entropy at the locked temperature."""
    if scores.shape != teacher.shape:
        raise guards.HybridQPayloadError("score and cached teacher batch shapes differ")
    return torch.stack([
        training.ranker_distillation_loss(scores[index], teacher[index])
        for index in range(scores.shape[0])
    ]).mean()


def masked_group_losses(
    model: torch.nn.Module, base: Any, c2: torch.Tensor, scores: torch.Tensor,
    batch: Mapping[str, Any], q: float, *, use_amp: bool,
) -> dict[str, torch.Tensor]:
    """Hard exact mask with the locked surrogate, then the registered D/G/S/A losses."""
    masked = torch.stack([
        training.masked_c2_forward(
            c2[index], training.straight_through_mask(scores[index], q)
        )
        for index in range(c2.shape[0])
    ])
    _outputs, groups = loss_groups_from_c2(model, base, masked, batch, use_amp=use_amp)
    del _outputs, masked
    return groups


def normalized_task_term(
    groups: Mapping[str, torch.Tensor], references: training.ReferenceMedians
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Split the registered groups into contributing and excluded.

    A batch whose geometry loss is exactly zero carries no matched FCOS location and
    therefore no geometry supervision at all. Dividing that zero by the frozen median
    would enter the equal mean as perfect geometry preservation, which is the opposite
    of what it means, so G is excluded from that batch's mean instead.
    """
    valid: dict[str, torch.Tensor] = {}
    excluded: dict[str, str] = {}
    for name in contract.TEACHER_GROUPS:
        loss = groups.get(name)
        if loss is None:
            excluded[name] = "absent_loss_group"
            continue
        value = float(loss.detach())
        if not math.isfinite(value):
            raise guards.HybridQNumericalError(f"masked task loss for group {name} is non-finite")
        if name == "G" and value == 0.0:
            excluded[name] = "zero_geometry_supervision"
            continue
        references.require(name)
        valid[name] = loss
    if not valid:
        raise guards.HybridQConfigError("no registered task group supervises this batch")
    return valid, excluded


# ---------------------------------------------------------------------------
# One optimizer update
# ---------------------------------------------------------------------------


def _total_grad_norm(module: torch.nn.Module) -> float:
    squares = [
        float(parameter.grad.detach().norm()) ** 2
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return float(math.sqrt(sum(squares)))


@dataclass
class EpochAccumulator:
    epoch: int
    stage: str
    updates: int = 0
    total_loss: float = 0.0
    distillation_loss: float = 0.0
    clipped_updates: int = 0
    q_counts: dict[str, int] = field(default_factory=dict)
    group_sums: dict[str, float] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    group_exclusions: dict[str, int] = field(default_factory=dict)
    max_pre_clip_norm: float = 0.0
    frames: int = 0

    def observe_groups(self, contributions: Mapping[str, float], excluded: Mapping[str, str]) -> None:
        for name, value in contributions.items():
            self.group_sums[name] = self.group_sums.get(name, 0.0) + float(value)
            self.group_counts[name] = self.group_counts.get(name, 0) + 1
        for name in excluded:
            self.group_exclusions[name] = self.group_exclusions.get(name, 0) + 1

    def summary(self, *, seconds: float, peak_allocated: float, peak_reserved: float,
                qualification: training.GradientQualification) -> dict[str, Any]:
        updates = max(1, self.updates)
        return {
            "epoch": self.epoch,
            "stage": self.stage,
            "optimizer_updates": self.updates,
            "frames": self.frames,
            "mean_total_loss": self.total_loss / updates,
            "mean_distillation_loss": self.distillation_loss / updates,
            "mean_normalized_group_contribution": {
                name: self.group_sums[name] / self.group_counts[name]
                for name in sorted(self.group_sums)
            },
            "group_contributing_updates": dict(sorted(self.group_counts.items())),
            "group_excluded_updates": dict(sorted(self.group_exclusions.items())),
            "q_update_counts": dict(sorted(self.q_counts.items())),
            "clipped_updates": self.clipped_updates,
            "clipped_fraction": self.clipped_updates / updates,
            "max_pre_clip_global_grad_norm": self.max_pre_clip_norm,
            "peak_allocated_vram_mib": peak_allocated,
            "peak_reserved_vram_mib": peak_reserved,
            "wall_seconds": seconds,
            "gradient_qualification": {
                "window": qualification.window,
                "observed_updates": qualification.seen,
                "named_ranker_tensors": list(qualification.parameter_names),
                "qualified": qualification.qualified(),
                "disconnected_tensors": list(qualification.disconnected()),
                "never_nonzero_tensors": list(qualification.never_nonzero()),
                "zero_gradient_batches": [
                    {"update": index, "tensors": list(names)}
                    for index, names in qualification.zero_gradient_batches
                ],
                "missing_gradient_batches": [
                    {"update": index, "tensors": list(names)}
                    for index, names in qualification.missing_gradient_batches
                ],
            },
        }


def optimizer_update(
    ranker: torch.nn.Module, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
    loss: torch.Tensor, qualification: training.GradientQualification,
) -> dict[str, Any]:
    """One locked update with the full runtime-safety contract enforced around it."""
    if not torch.isfinite(loss.detach()).all():
        raise guards.HybridQNumericalError("non-finite total training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nonzero = qualification.observe(ranker, loss=loss)
    pre_clip = float(training.clip_ranker_gradients(ranker))
    post_clip = _total_grad_norm(ranker)
    if not (math.isfinite(pre_clip) and math.isfinite(post_clip)):
        raise guards.HybridQNumericalError("non-finite ranker gradient norm")
    optimizer.step()
    training.require_post_step_health(ranker, optimizer)
    guards.require_optimizer_owns_only(optimizer, ranker.parameters())
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            raise guards.HybridQOwnershipError(
                f"frozen perception parameter '{name}' received a gradient"
            )
    return {
        "pre_clip_global_grad_norm": pre_clip,
        "post_clip_global_grad_norm": post_clip,
        "clipped": pre_clip > contract.GRAD_CLIP_GLOBAL_NORM,
        "all_tensors_nonzero": bool(nonzero),
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    global_update_index: int = 0
    global_q_update_index: int = 0


def run_epoch(
    *, epoch: int, model: torch.nn.Module, base: Any, ranker: torch.nn.Module,
    optimizer: torch.optim.Optimizer, dataset: Any, partition: SplitPartition,
    store: TeacherStore, references: training.ReferenceMedians, device: torch.device,
    state: TrainingState, use_amp: bool, workers: int, batch_limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    stage = contract.stage_for_epoch(epoch)
    order = epoch_order(partition, epoch)
    loader = epoch_loader(base, dataset, order, workers=workers)
    expected_ids = [dataset.rows[index]["sample_id"] for index in order]
    accumulator = EpochAccumulator(epoch=epoch, stage=stage)
    planned = len(loader) if batch_limit is None else min(len(loader), batch_limit)
    qualification = training.GradientQualification.for_module(ranker, window=planned)
    observed_ids: list[str] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(loader):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        sample_ids = [str(value) for value in batch["sample_ids"]]
        observed_ids.extend(sample_ids)
        teacher = store.fit_batch(sample_ids, device)

        c2 = encode_front(model, batch, device)
        scores = ranker(c2)
        distillation = distillation_loss(scores, teacher)
        if not torch.isfinite(distillation.detach()).all():
            raise guards.HybridQNumericalError("non-finite distillation loss")

        if stage == "distillation":
            q = None
            loss = distillation
            contributions: dict[str, float] = {}
            excluded: dict[str, str] = {}
        else:
            q = training.q_for_update(state.global_q_update_index)
            if contract.drop_count(q) == 0:
                raise guards.HybridQConfigError("q=0 must never be a training update")
            groups = masked_group_losses(
                model, base, c2, scores, batch, q, use_amp=use_amp
            )
            valid, excluded = normalized_task_term(groups, references)
            loss = training.q_aware_objective(valid, distillation, references)
            contributions = {
                name: float(value.detach()) / references.require(name)
                for name, value in valid.items()
            }
            state.global_q_update_index += 1
            key = f"{q:.2f}"
            accumulator.q_counts[key] = accumulator.q_counts.get(key, 0) + 1
            del groups, valid

        health = optimizer_update(ranker, optimizer, model, loss, qualification)
        state.global_update_index += 1
        accumulator.updates += 1
        accumulator.frames += len(sample_ids)
        accumulator.total_loss += float(loss.detach())
        accumulator.distillation_loss += float(distillation.detach())
        accumulator.clipped_updates += int(health["clipped"])
        accumulator.max_pre_clip_norm = max(
            accumulator.max_pre_clip_norm, health["pre_clip_global_grad_norm"]
        )
        accumulator.observe_groups(contributions, excluded)
        del c2, scores, distillation, loss, teacher

        if progress and planned >= 100 and (batch_index + 1) % max(1, planned // 4) == 0:
            print(
                f"[phase5] epoch {epoch} ({stage}) "
                f"{batch_index + 1}/{planned} batches, "
                f"{(time.time() - started) / 60.0:.1f} min",
                flush=True,
            )

    del loader
    seconds = time.time() - started
    if batch_limit is None:
        if observed_ids != expected_ids:
            raise guards.HybridQConfigError("epoch batch order drifted from the seeded order")
        if len(set(observed_ids)) != contract.TRAIN_FIT_FRAMES:
            raise guards.HybridQConfigError("an epoch did not cover every fit frame exactly once")
        if set(observed_ids) & store.holdout_ids:
            raise guards.HybridQOwnershipError("a reserved holdout frame entered an optimizer batch")
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
    summary["global_q_update_index"] = state.global_q_update_index
    return summary


# ---------------------------------------------------------------------------
# Recovery and candidate checkpoints
# ---------------------------------------------------------------------------


def rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def save_recovery(
    path: Path, *, epoch: int, ranker: torch.nn.Module, optimizer: torch.optim.Optimizer,
    state: TrainingState, summary: Mapping[str, Any], binding: Mapping[str, Any],
) -> str:
    payload = {
        "schema": "splitfusion_fcos_hybrid_q_phase5_recovery_v1",
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "next_epoch_shuffle_seed": (
            contract.epoch_shuffle_seed(epoch + 1)
            if epoch < contract.TRAINING_EPOCHS else None
        ),
        "stage": contract.stage_for_epoch(epoch),
        "global_update_index": state.global_update_index,
        "global_q_update_index": state.global_q_update_index,
        "ranker": ranker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "epoch_sample_id_sha256": summary["epoch_sample_id_sha256"],
        "epoch_summary": dict(summary),
        "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
        "locked_config_sha256": binding["hybrid_q_locked_config"]["sha256"],
        "teacher_cache_manifest_sha256": binding["teacher_cache_manifest"]["sha256"],
        "fit_reference_medians_sha256": binding["fit_reference_medians"]["sha256"],
        "hybrid_q_source_sha256": binding["hybrid_q_source_sha256"],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def save_candidate(
    path: Path, *, epoch: int, ranker: torch.nn.Module, state: TrainingState,
    binding: Mapping[str, Any],
) -> str:
    payload = {
        "schema": "splitfusion_fcos_hybrid_q_phase5_candidate_v1",
        "epoch": int(epoch),
        "stage": contract.stage_for_epoch(epoch),
        "ranker": ranker.state_dict(),
        "parameter_count": sum(p.numel() for p in ranker.parameters()),
        "global_update_index": state.global_update_index,
        "global_q_update_index": state.global_q_update_index,
        "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
        "locked_config_sha256": binding["hybrid_q_locked_config"]["sha256"],
        "teacher_cache_manifest_sha256": binding["teacher_cache_manifest"]["sha256"],
        "fit_reference_medians_sha256": binding["fit_reference_medians"]["sha256"],
        "hybrid_q_source_sha256": binding["hybrid_q_source_sha256"],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def load_recovery(path: Path, ranker: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  state: TrainingState, binding: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "splitfusion_fcos_hybrid_q_phase5_recovery_v1":
        raise guards.HybridQConfigError("recovery schema drift")
    for name, key in (
        ("perception_checkpoint_sha256", contract.FROZEN_CHECKPOINT_SHA256),
        ("locked_config_sha256", binding["hybrid_q_locked_config"]["sha256"]),
        ("teacher_cache_manifest_sha256", binding["teacher_cache_manifest"]["sha256"]),
        ("fit_reference_medians_sha256", binding["fit_reference_medians"]["sha256"]),
    ):
        if payload[name] != key:
            raise guards.HybridQConfigError(f"recovery {name} drift; refusing to resume")
    ranker.load_state_dict(payload["ranker"])
    optimizer.load_state_dict(payload["optimizer"])
    state.global_update_index = int(payload["global_update_index"])
    state.global_q_update_index = int(payload["global_q_update_index"])
    rng = payload["rng"]
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    return payload


def latest_recovery(recovery_dir: Path) -> tuple[int, Path] | None:
    candidates = sorted(recovery_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    path = candidates[-1]
    return int(path.stem.split("_")[1]), path


# ---------------------------------------------------------------------------
# Bounded pre-flight smoke on a disposable ranker
# ---------------------------------------------------------------------------


def smoke(
    *, model: torch.nn.Module, base: Any, dataset: Any, partition: SplitPartition,
    store: TeacherStore, references: training.ReferenceMedians, device: torch.device,
    frozen_snapshot: Mapping[str, torch.Tensor], use_amp: bool, workers: int,
) -> dict[str, Any]:
    """Exercise both stages end to end on a ranker whose state is then discarded.

    This is a runner pre-flight, not a scientific run: it constructs its own ranker
    and optimizer, takes a handful of updates on fit batches, and throws both away.
    Nothing it produces is saved or carried into the 12-epoch run.
    """
    started = time.time()
    disposable = build_ranker().to(device)
    optimizer = training.build_ranker_optimizer(disposable, frozen_modules=[model])
    state = TrainingState()
    stages = []
    for epoch, limit in ((1, SMOKE_DISTILLATION_BATCHES), (5, SMOKE_Q_AWARE_BATCHES)):
        summary = run_epoch(
            epoch=epoch, model=model, base=base, ranker=disposable, optimizer=optimizer,
            dataset=dataset, partition=partition, store=store, references=references,
            device=device, state=state, use_amp=use_amp, workers=workers,
            batch_limit=limit, progress=False,
        )
        stages.append({
            "stage": summary["stage"],
            "optimizer_updates": summary["optimizer_updates"],
            "mean_total_loss": summary["mean_total_loss"],
            "mean_distillation_loss": summary["mean_distillation_loss"],
            "q_update_counts": summary["q_update_counts"],
            "mean_normalized_group_contribution": summary["mean_normalized_group_contribution"],
            "group_excluded_updates": summary["group_excluded_updates"],
        })
    guards.require_module_state_unchanged(model, frozen_snapshot)
    del disposable, optimizer
    torch.cuda.empty_cache()
    return {
        "passed": True,
        "seconds": time.time() - started,
        "stages": stages,
        "global_q_update_index_after_smoke": state.global_q_update_index,
        "disposable_ranker_retained": False,
        "checks": [
            "both stages executed on the real data, mask, loss and optimizer path",
            "cached teacher maps joined by exact sample id",
            "frozen perception parameters and buffers exactly unchanged",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid-q Phase-5 ranker training")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.resume:
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-5 ranker training requires CUDA")
    device = torch.device("cuda:0")

    torch.manual_seed(contract.RANKER_INIT_SEED)
    torch.cuda.manual_seed_all(contract.RANKER_INIT_SEED)
    random.seed(contract.RANKER_INIT_SEED)
    np.random.seed(contract.RANKER_INIT_SEED)

    binding = bind_inputs()
    delta = source_delta(binding)
    print(f"[phase5] inputs bound: {binding['teacher_cache_shards']['verified']}/66 shards, "
          f"medians {binding['fit_reference_medians']['sha256'][:12]}", flush=True)

    model, base, perception = load_frozen_perception(device)
    dataset = build_train_dataset(base)
    partition = build_split_partition(dataset)
    split_check = require_no_validation_or_test(
        dataset, (row["sample_id"] for row in dataset.rows)
    )
    store = load_teacher_store(binding, partition)
    references = training.ReferenceMedians(
        medians=binding["fit_reference_medians"]["medians"]
    )
    frozen_snapshot = guards.snapshot_module_state(model)
    print(f"[phase5] teacher store preloaded: {store.bytes / 2 ** 30:.2f} GiB, "
          f"{len(store.fit_ids)} fit / {len(store.holdout_ids)} holdout maps", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    recovery_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    recovery_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    ranker = build_ranker().to(device)
    optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[model])
    guards.require_optimizer_owns_only(optimizer, ranker.parameters())
    owned = sum(
        parameter.numel()
        for group in optimizer.param_groups for parameter in group["params"]
    )
    if owned != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQOwnershipError(
            f"optimizer owns {owned} parameters != {contract.RANKER_PARAMETER_COUNT}"
        )
    state = TrainingState()
    epoch_summaries: list[dict[str, Any]] = []
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    smoke_report: dict[str, Any] | None = None
    started_epoch = 1

    resumed = None
    if args.resume:
        found = latest_recovery(recovery_dir)
        if found is not None:
            completed, path = found
            resumed = load_recovery(path, ranker, optimizer, state, binding)
            ranker.to(device)
            started_epoch = completed + 1
            summary_path = output / "epoch_summaries.json"
            if summary_path.is_file():
                epoch_summaries = json.loads(summary_path.read_text(encoding="utf-8"))["epochs"]
            print(f"[phase5] resumed from completed epoch {completed}; "
                  f"continuing at epoch {started_epoch}", flush=True)

    if started_epoch == 1 and not args.skip_smoke:
        smoke_report = smoke(
            model=model, base=base, dataset=dataset, partition=partition, store=store,
            references=references, device=device, frozen_snapshot=frozen_snapshot,
            use_amp=True, workers=min(4, int(args.workers)),
        )
        print(f"[phase5] pre-flight smoke passed in {smoke_report['seconds']:.1f} s "
              "on a disposable ranker; beginning the scientific run", flush=True)
        # The smoke's disposable optimizer never touched these; rebuild anyway so the
        # scientific run starts from the registered deterministic initialization.
        torch.manual_seed(contract.RANKER_INIT_SEED)
        torch.cuda.manual_seed_all(contract.RANKER_INIT_SEED)
        random.seed(contract.RANKER_INIT_SEED)
        np.random.seed(contract.RANKER_INIT_SEED)
        ranker = build_ranker().to(device)
        optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[model])
        guards.require_optimizer_owns_only(optimizer, ranker.parameters())

    run_started = time.time()
    frozen_comparisons: list[dict[str, Any]] = []
    for epoch in range(started_epoch, contract.TRAINING_EPOCHS + 1):
        summary = run_epoch(
            epoch=epoch, model=model, base=base, ranker=ranker, optimizer=optimizer,
            dataset=dataset, partition=partition, store=store, references=references,
            device=device, state=state, use_amp=True, workers=int(args.workers),
        )
        guards.require_module_state_unchanged(model, frozen_snapshot)
        summary["frozen_perception_state_unchanged"] = True
        if epoch in contract.CHECKPOINT_EPOCHS:
            frozen_comparisons.append({
                "epoch": epoch,
                "parameters_and_buffers_exactly_unchanged": True,
                "compared_tensors": len(frozen_snapshot),
            })
        recovery_path = recovery_dir / f"epoch_{epoch:02d}.pt"
        recovery_hashes[recovery_path.name] = save_recovery(
            recovery_path, epoch=epoch, ranker=ranker, optimizer=optimizer,
            state=state, summary=summary, binding=binding,
        )
        if epoch in contract.CHECKPOINT_EPOCHS:
            candidate_path = checkpoint_dir / f"ranker_epoch_{epoch:02d}.pt"
            candidate_hashes[candidate_path.name] = save_candidate(
                candidate_path, epoch=epoch, ranker=ranker, state=state, binding=binding,
            )
            summary["candidate_checkpoint"] = candidate_path.name
            summary["candidate_checkpoint_sha256"] = candidate_hashes[candidate_path.name]
        epoch_summaries.append(summary)
        (output / "epoch_summaries.json").write_text(
            json.dumps({"schema": SCHEMA, "epochs": epoch_summaries}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "epoch": epoch, "stage": summary["stage"],
            "updates": summary["optimizer_updates"],
            "mean_total_loss": round(summary["mean_total_loss"], 6),
            "mean_distillation_loss": round(summary["mean_distillation_loss"], 6),
            "clipped_fraction": round(summary["clipped_fraction"], 6),
            "minutes": round(summary["wall_seconds"] / 60.0, 2),
        }), flush=True)

    if len(candidate_hashes) != len(contract.CHECKPOINT_EPOCHS) and started_epoch == 1:
        raise guards.HybridQConfigError("candidate checkpoint set incomplete")

    report = {
        "schema": SCHEMA,
        "terminal": "HYBRID_Q_PHASE5_TRAINING_COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "train_fit_only_optimizer_batches": True,
            "holdout_frames_in_optimizer_batches": 0,
            "validation_or_test_accessed": False,
            "augmentation": contract.AUGMENTATION_ENABLED,
            "seed": contract.RANKER_INIT_SEED,
            "epochs": contract.TRAINING_EPOCHS,
            "carla_launched": False,
            "quantization_or_zstd_run": False,
            "holdout_evaluated_here": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "training_precision": "bf16 autocast tail, fp32 C2 boundary and losses (registered)",
        },
        "binding": {k: v for k, v in binding.items() if k != "teacher_cache_shards"},
        "teacher_cache_shards": {
            "count": binding["teacher_cache_shards"]["count"],
            "verified": binding["teacher_cache_shards"]["verified"],
            "total_bytes": binding["teacher_cache_shards"]["total_bytes"],
            "sha256": binding["teacher_cache_shards"]["sha256"],
        },
        "source_delta": delta,
        "perception_binding": perception,
        "split": {
            "fit_frames": len(partition.fit_indices),
            "holdout_frames": len(partition.holdout_indices),
            "fit_sample_id_sha256": contract.sample_id_digest(partition.fit_sample_ids),
            "holdout_sample_id_sha256": contract.sample_id_digest(partition.holdout_sample_ids),
            **split_check,
        },
        "teacher_store": {
            "frames": len(store.index),
            "bytes": store.bytes,
            "gib": store.bytes / 2 ** 30,
            "preloaded_once": True,
            "c2_tensors_cached": False,
            "holdout_maps_blocked_from_training": True,
        },
        "optimization": {
            "optimizer": contract.OPTIMIZER,
            "learning_rate": contract.LEARNING_RATE,
            "weight_decay": contract.WEIGHT_DECAY,
            "lr_schedule": contract.LR_SCHEDULE,
            "grad_clip_global_norm": contract.GRAD_CLIP_GLOBAL_NORM,
            "physical_batch": contract.TRAINING_BATCH_SIZE,
            "effective_batch": contract.TRAINING_BATCH_SIZE,
            "gradient_accumulation_steps": contract.GRADIENT_ACCUMULATION_STEPS,
            "final_partial_batch_retained": not contract.DROP_LAST_TRAINING_BATCH,
            "owned_parameters": owned,
            "trainable_component": "ranker only",
        },
        "reference_medians": binding["fit_reference_medians"]["medians"],
        "smoke": smoke_report,
        "resumed_from": None if resumed is None else {
            "epoch": int(resumed["epoch"]),
            "global_update_index": int(resumed["global_update_index"]),
        },
        "epochs": epoch_summaries,
        "totals": {
            "optimizer_updates": state.global_update_index,
            "q_aware_updates": state.global_q_update_index,
            "wall_seconds": time.time() - run_started,
        },
        "frozen_state_comparisons": frozen_comparisons,
        "recovery_checkpoints": recovery_hashes,
        "candidate_checkpoints": candidate_hashes,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / "TRAINING_COMPLETE").write_text(
        f"HYBRID_Q_PHASE5_TRAINING_COMPLETE {report['generated_utc']}\n", encoding="utf-8"
    )
    print(json.dumps({
        "terminal": report["terminal"],
        "optimizer_updates": state.global_update_index,
        "q_aware_updates": state.global_q_update_index,
        "minutes": round(report["totals"]["wall_seconds"] / 60.0, 1),
        "candidates": sorted(candidate_hashes),
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - runner entry point
    raise SystemExit(main())
