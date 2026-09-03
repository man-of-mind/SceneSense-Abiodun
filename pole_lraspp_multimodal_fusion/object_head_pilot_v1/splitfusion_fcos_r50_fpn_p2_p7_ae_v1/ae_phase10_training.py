"""Phase-10A AE64/AE32 scientific training: one family per command, per process.

The completed AE128 Phase-9C procedure, extended to the two smaller registered
families without editing it. The schedule, objective, optimizer, data join,
isolation rules and the candidate/summary/recovery commit order are the AE128
ones, imported and reused; the only things that change are the bottleneck-
dependent architecture and the deterministic per-family initialization.

    python3 -m ...ae_v1.ae_phase10_training \\
      --execute SPLITFUSION_AE64_PHASE10_TRAINING --bottleneck 64 --output <dir>

    python3 -m ...ae_v1.ae_phase10_training \\
      --execute SPLITFUSION_AE32_PHASE10_TRAINING --bottleneck 32 --output <dir>

The execute token and `--bottleneck` must name the same family or the command
refuses to start, and a process may bind exactly one family, so AE64 and AE32
cannot be trained together.

Stage A is dense q=0 for epochs 1-4. Stage B is a continuous round robin over
q = {0, 0.30, 0.50, 0.70} for epochs 5-12, one q per batch, with the cycle
position carried across epoch boundaries so the 6,776 Stage-B updates split
exactly 1,694 per q. q=0.90 and q=0.98 are evaluation/emergency settings and are
never optimized.

Per epoch the writes are ordered candidate checkpoint, then the epoch-summary
file, then the recovery checkpoint, each atomic. `recovery/ae<B>_recovery_epoch_E.pt`
is therefore the sole durable declaration that epoch E completed, and it embeds
that epoch's final summary including its candidate metadata.

Selection is a separate command. The reserved train-holdout is never opened
here, no validation or test frame is read, and training refuses to start if a
holdout-selection module is imported or if this run already holds a selection
output.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from . import ae_composition, ae_loss
from . import ae_phase10_common as family
from . import ae_training_common as common

# The family-independent epoch mechanics of the proven AE128 trainer: the
# safety-checked optimizer update, the per-q/per-epoch accounting, the counter
# carrier and the sampler-order identity. None of them emits a family label, so
# they are reused rather than re-implemented per family.
from .ae_training import (
    EpochAccumulator,
    TrainingState,
    expected_stage_b_position,
    optimizer_update,
    order_identity,
)

DATALOADER_WORKERS = 8

# No selection runner -- for any family -- may be reachable from this command.
HOLDOUT_MODULES = tuple(
    "pole_lraspp_multimodal_fusion.object_head_pilot_v1."
    f"splitfusion_fcos_r50_fpn_p2_p7_ae_v1.{name}"
    for name in ("ae_holdout_selection", "ae_phase10_holdout_selection")
)


def require_holdout_unopened() -> None:
    """Structural guard: the selection runner is a separate command, full stop."""
    loaded = [name for name in HOLDOUT_MODULES if name in sys.modules]
    if loaded:
        raise guards.HybridQOwnershipError(
            f"holdout selection module(s) {loaded} are loaded inside the "
            "training command"
        )


def require_no_selection_output(output: Path, bottleneck: int) -> list[str]:
    """Refuse to train into a directory that already holds a selection result.

    Selection consumes a completed training run. If one has already been
    produced here, continuing to write into the same directory would train
    against an existing selection, so the command stops instead.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    output = Path(output)
    if not output.is_dir():
        return []
    existing = sorted(
        child.name
        for child in output.iterdir()
        if child.is_dir() and child.name.startswith("holdout_selection")
    )
    if existing:
        raise guards.HybridQOwnershipError(
            f"{family.family_label(size)} training refuses to run: {output} "
            f"already holds selection output {existing}"
        )
    return existing


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def run_epoch(
    *,
    bottleneck: int,
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
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_slug(size)
    require_holdout_unopened()
    if autoencoder.bottleneck != size:
        raise guards.HybridQOwnershipError(
            "the optimized AE is not the family this run bound"
        )
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
                f"[{label}] epoch {epoch} ({stage}) {batch_index + 1}/{planned} "
                f"batches, {(time.time() - started) / 60.0:.1f} min",
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
    summary.update(family.family_fields(size))
    summary["epoch_shuffle_seed"] = contract.epoch_shuffle_seed(epoch)
    summary["epoch_sample_id_sha256"] = contract.sample_id_digest(observed_ids)
    summary["holdout_frames_in_optimizer_batches"] = 0
    summary["global_update_index"] = state.global_update_index
    summary["stage_b_cycle_position"] = state.stage_b_position
    return summary


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def completed_recovery_epochs(recovery_dir: Path, bottleneck: int) -> list[tuple[int, Path]]:
    """Every recovery file for *this family*, as (epoch, path), oldest first.

    The directory is scanned for this family's filenames only, and any other AE
    checkpoint sitting beside them is refused rather than parsed: a recovery
    directory that mixes families is not a resumable run.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    found: list[tuple[int, Path]] = []
    for path in sorted(Path(recovery_dir).glob("*.pt")):
        if not path.name.startswith(f"ae{size}_recovery_epoch_"):
            raise guards.HybridQOwnershipError(
                f"{path.name} is not an {family.family_label(size)} recovery "
                "checkpoint; this recovery directory holds another family"
            )
        found.append((family.epoch_from_recovery_filename(size, path.name), path))
    return found


def stale_candidate_files(checkpoint_dir: Path, bottleneck: int, completed: int) -> list[str]:
    """Candidate files for epochs past the last verified recovery checkpoint.

    Those epochs did not complete, so their candidate is not evidence of
    anything. They are left alone here and atomically replaced when the epoch is
    rerun.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    return sorted(
        family.candidate_filename(size, epoch)
        for epoch in common.AE_CANDIDATE_EPOCHS
        if epoch > int(completed)
        and (Path(checkpoint_dir) / family.candidate_filename(size, epoch)).is_file()
    )


def read_recovery(
    path: Path,
    bottleneck: int,
    epoch: int,
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one recovery checkpoint completely, without mutating any state."""
    size = family.require_phase10_bottleneck(bottleneck)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != family.recovery_schema(size):
        raise guards.HybridQConfigError(f"{path.name} recovery schema drift")
    family.require_family_fields(payload, size, what=f"recovery {path.name}")
    if int(payload["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(
            f"{path.name} carries epoch {payload['epoch']}; the filename says {epoch}"
        )
    if payload["stage"] != common.stage_for_epoch(epoch):
        raise guards.HybridQConfigError(f"{path.name} stage drift")
    # Every saved binding, including the per-file source hashes.
    common.require_bindings(payload, binding, what=f"recovery {path.name}")
    if payload["configuration"] != family.training_configuration(size):
        raise guards.HybridQConfigError(
            "recovery was written under a different training configuration"
        )
    # The completed-epoch count implies both counters exactly.
    expected_updates = int(epoch) * common.batches_per_epoch()
    if int(payload["global_update_index"]) != expected_updates:
        raise guards.HybridQConfigError(
            f"{path.name} global update index {payload['global_update_index']} != "
            f"{expected_updates} after {epoch} completed epochs"
        )
    if int(payload["stage_b_cycle_position"]) != expected_stage_b_position(epoch):
        raise guards.HybridQConfigError(
            f"{path.name} Stage-B cycle position {payload['stage_b_cycle_position']} "
            f"!= {expected_stage_b_position(epoch)} after {epoch} completed epochs"
        )
    if dict(payload["order_identity"]) != dict(identity):
        raise guards.HybridQConfigError(f"{path.name} sampler/order identity drift")
    summary = dict(payload["epoch_summary"])
    if int(summary["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(f"{path.name} epoch-summary epoch drift")
    if summary["epoch_sample_id_sha256"] != identity["sample_id_sha256"]:
        raise guards.HybridQConfigError(f"{path.name} epoch-summary order drift")
    # Written after the candidate, so a candidate epoch's recovery must name it.
    expected_candidate = (
        family.candidate_filename(size, epoch)
        if int(epoch) in common.AE_CANDIDATE_EPOCHS
        else None
    )
    if summary.get("candidate_checkpoint") != expected_candidate:
        raise guards.HybridQConfigError(
            f"{path.name} embedded summary names candidate "
            f"{summary.get('candidate_checkpoint')!r}, expected {expected_candidate!r}"
        )
    return payload


def load_recovery(
    path: Path,
    bottleneck: int,
    epoch: int,
    autoencoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify, then restore weights, optimizer state, counters and RNG."""
    payload = read_recovery(path, bottleneck, epoch, binding, identity)
    autoencoder.load_state_dict(payload["autoencoder"])
    optimizer.load_state_dict(payload["optimizer"])
    state.global_update_index = int(payload["global_update_index"])
    state.stage_b_position = int(payload["stage_b_cycle_position"])
    common.restore_rng(payload["rng"])
    return payload


def verify_external_summaries(
    output: Path, bottleneck: int, canonical: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Check whatever summary file survived against the canonical record.

    The file is written before the epoch's recovery checkpoint, so exactly two
    disagreements are expected after an interruption and both are tolerated:
    it may be missing or truncated (the process died before or during the write),
    and it may carry one extra tail entry for an epoch whose recovery checkpoint
    was never written. Anything it does say about a completed epoch must agree.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    path = Path(output) / family.epoch_summaries_filename(size)
    report: dict[str, Any] = {
        "path": path.name,
        "present": path.is_file(),
        "canonical_epochs": len(canonical),
        "verified_prefix_epochs": 0,
        "tolerated": [],
    }
    if not path.is_file():
        report["tolerated"].append("absent; rewritten from the recovery checkpoints")
        return report
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        report["tolerated"].append(
            "unreadable/truncated; rewritten from the recovery checkpoints"
        )
        return report
    if document.get("schema") != family.training_schema(size):
        raise guards.HybridQConfigError("epoch-summary schema drift")
    family.require_family_fields(document, size, what=path.name)

    external = list(document.get("epochs", []))
    report["external_epochs"] = len(external)
    if len(external) < len(canonical):
        report["tolerated"].append(
            f"truncated at {len(external)} of {len(canonical)} completed epochs"
        )
    elif len(external) > len(canonical):
        report["tolerated"].append(
            f"{len(external) - len(canonical)} tail entr"
            f"{'y' if len(external) - len(canonical) == 1 else 'ies'} past the last "
            "recovery checkpoint discarded"
        )
    for position, summary in enumerate(canonical):
        if position >= len(external):
            break
        observed = external[position]
        if int(observed["epoch"]) != int(summary["epoch"]):
            raise guards.HybridQConfigError(
                f"{path.name} position {position} holds epoch "
                f"{observed['epoch']}, the recovery record says {summary['epoch']}"
            )
        for field_name in ("epoch_sample_id_sha256", "candidate_checkpoint_sha256"):
            if field_name in observed and observed[field_name] != summary.get(field_name):
                raise guards.HybridQConfigError(
                    f"{path.name} epoch {observed['epoch']} {field_name} "
                    "disagrees with its recovery checkpoint"
                )
        report["verified_prefix_epochs"] = position + 1
    return report


def restore_completed_epochs(
    *,
    bottleneck: int,
    output: Path,
    recovery_dir: Path,
    checkpoint_dir: Path,
    autoencoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    binding: Mapping[str, Any],
    partition: SplitPartition,
    dataset: Any,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], dict[str, str], dict[str, str]]:
    """Rebuild the full bookkeeping of a partially completed family run.

    The contiguous verified recovery checkpoints are authoritative: each one is
    written last for its epoch and embeds that epoch's final summary, candidate
    metadata included, so the canonical record is reconstructed from them rather
    than from the external summary file. Every candidate epoch at or below the
    last verified recovery must have its candidate checkpoint, and it is
    verified against the hash the recovery checkpoint recorded; candidate files
    for later epochs belong to an epoch that did not complete and are ignored.
    The summary file is then rewritten atomically from that canonical record.
    Only the last recovery checkpoint restores state; no epoch holding a valid
    recovery checkpoint is replayed.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_slug(size)
    output = Path(output)
    recovery_dir = Path(recovery_dir)
    checkpoint_dir = Path(checkpoint_dir)
    if not output.is_dir():
        raise guards.HybridQConfigError(
            f"--resume requires an existing training directory: {output} does not exist"
        )
    if not recovery_dir.is_dir():
        raise guards.HybridQConfigError(
            f"--resume requires {recovery_dir}; refusing to resume into a new directory"
        )
    found = completed_recovery_epochs(recovery_dir, size)
    if not found:
        raise guards.HybridQConfigError(
            "--resume requires at least one recovery checkpoint; refusing to "
            "resume into an empty directory"
        )
    epochs = [epoch for epoch, _ in found]
    completed = epochs[-1]
    if epochs != list(range(1, completed + 1)):
        raise guards.HybridQConfigError(
            f"recovery checkpoints {epochs} are not the contiguous set 1..{completed}"
        )

    epoch_summaries: list[dict[str, Any]] = []
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for epoch, path in found:
        identity = order_identity(partition, epoch, dataset)
        if epoch == completed:
            payload = load_recovery(
                path, size, epoch, autoencoder, optimizer, state, binding, identity
            )
            autoencoder.to(device)
        else:
            # Verified in full, but only the last checkpoint restores state.
            payload = read_recovery(path, size, epoch, binding, identity)
        summary = dict(payload["epoch_summary"])
        del payload
        epoch_summaries.append(summary)
        recovery_hashes[path.name] = sha256_file(path)
        if epoch not in common.AE_CANDIDATE_EPOCHS:
            continue
        candidate_path = checkpoint_dir / family.candidate_filename(size, epoch)
        if not candidate_path.is_file():
            raise guards.HybridQConfigError(
                f"epoch {epoch} completed but {candidate_path.name} is missing"
            )
        digest = sha256_file(candidate_path)
        if summary.get("candidate_checkpoint_sha256") != digest:
            raise guards.HybridQConfigError(f"{candidate_path.name} sha256 drift")
        # Verifies the schema, family, epoch, every binding and finiteness.
        restored, _metadata = family.load_candidate(
            candidate_path, size, epoch, torch.device("cpu"), binding
        )
        del restored
        candidate_hashes[candidate_path.name] = digest

    external = verify_external_summaries(output, size, epoch_summaries)
    family.write_epoch_summaries(output, size, epoch_summaries)
    if external["tolerated"]:
        print(
            f"[{label}] {external['path']}: {'; '.join(external['tolerated'])}",
            flush=True,
        )
    return completed, epoch_summaries, recovery_hashes, candidate_hashes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-10A SplitFusion AE64/AE32 scientific training (fit frames only)"
        )
    )
    parser.add_argument("--execute", required=True, choices=family.TRAINING_EXECUTE_TOKENS)
    parser.add_argument(
        "--bottleneck", required=True, type=int, choices=family.AE_PHASE10_BOTTLENECKS
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # The token and the bottleneck must name the same family, and this process
    # may bind exactly one family for its whole life.
    bottleneck = family.require_token_agrees_with_bottleneck(
        args.execute, args.bottleneck, kind="training"
    )
    family.bind_process_family(bottleneck)
    label = family.family_slug(bottleneck)
    terminal = family.training_terminal(bottleneck)
    schema = family.training_schema(bottleneck)

    require_holdout_unopened()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if args.resume and not output.is_dir():
        raise guards.HybridQConfigError(
            f"--resume requires an existing training directory: {output} does not exist"
        )
    require_no_selection_output(output, bottleneck)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Phase-10A {family.family_label(bottleneck)} training requires CUDA"
        )
    device = torch.device("cuda:0")

    seed = family.process_seed(bottleneck)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    binding = common.bind_frozen_inputs()
    delta = source_delta(binding)
    schedule_counts = common.require_balanced_stage_b()
    configuration = family.training_configuration(bottleneck)
    print(
        f"[{label}] inputs bound: {binding['teacher_cache_shards']['verified']}/66 "
        f"shards; Stage-B schedule {schedule_counts}",
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
    store_provenance = store.provenance()
    print(
        f"[{label}] fit teacher store: {store.bytes / 2 ** 30:.2f} GiB, "
        f"{store.frames} maps from {len(store.loaded_shards)} fit shards; "
        f"{len(store.other_split_ids)} holdout IDs excluded, "
        f"{len(store.withheld_shards)} holdout shards never opened",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    recovery_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    recovery_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    autoencoder = family.build_family_ae(bottleneck, device)
    optimizer = family.build_family_optimizer(
        autoencoder,
        lr=common.learning_rate_for_stage(common.AE_STAGE_A),
        frozen_modules=(model, ranker),
    )
    state = TrainingState()
    epoch_summaries: list[dict[str, Any]] = []
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    started_epoch = 1

    resumed_from: int | None = None
    if args.resume:
        (
            completed,
            epoch_summaries,
            recovery_hashes,
            candidate_hashes,
        ) = restore_completed_epochs(
            bottleneck=bottleneck,
            output=output,
            recovery_dir=recovery_dir,
            checkpoint_dir=checkpoint_dir,
            autoencoder=autoencoder,
            optimizer=optimizer,
            state=state,
            binding=binding,
            partition=partition,
            dataset=dataset,
            device=device,
        )
        resumed_from = completed
        started_epoch = completed + 1
        stale = stale_candidate_files(checkpoint_dir, bottleneck, completed)
        print(
            f"[{label}] resumed from completed epoch {completed}; continuing at "
            f"epoch {started_epoch} with global update {state.global_update_index} "
            f"and Stage-B cycle position {state.stage_b_position}; "
            f"{len(recovery_hashes)} recovery and {len(candidate_hashes)} candidate "
            "checkpoints reconstructed, no completed epoch replayed"
            + (
                f"; ignoring {stale} from an epoch that did not complete"
                if stale
                else ""
            ),
            flush=True,
        )

    run_started = time.time()
    for epoch in range(started_epoch, common.AE_TRAINING_EPOCHS + 1):
        identity = order_identity(partition, epoch, dataset)
        summary = run_epoch(
            bottleneck=bottleneck,
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

        # Commit order, all atomic: the candidate first, then the external
        # summary file, then the recovery checkpoint last.
        # `recovery/ae<B>_recovery_epoch_E.pt` is the sole durable declaration
        # that epoch E completed, so an interruption can never leave a declared
        # epoch whose candidate or summary is missing.
        if epoch in common.AE_CANDIDATE_EPOCHS:
            candidate_path = checkpoint_dir / family.candidate_filename(bottleneck, epoch)
            candidate_hashes[candidate_path.name] = family.save_candidate(
                candidate_path,
                bottleneck=bottleneck,
                epoch=epoch,
                autoencoder=autoencoder,
                global_update_index=state.global_update_index,
                stage_b_position=state.stage_b_position,
                binding=binding,
            )
            summary["candidate_checkpoint"] = candidate_path.name
            summary["candidate_checkpoint_sha256"] = candidate_hashes[candidate_path.name]

        epoch_summaries.append(summary)
        family.write_epoch_summaries(output, bottleneck, epoch_summaries)

        recovery_path = recovery_dir / family.recovery_filename(bottleneck, epoch)
        recovery_hashes[recovery_path.name] = family.save_recovery(
            recovery_path,
            bottleneck=bottleneck,
            epoch=epoch,
            autoencoder=autoencoder,
            optimizer=optimizer,
            global_update_index=state.global_update_index,
            stage_b_position=state.stage_b_position,
            order_identity=identity,
            # The final summary, candidate metadata included, so this checkpoint
            # alone can reconstruct the canonical record of the epoch.
            summary=summary,
            binding=binding,
        )
        print(
            json.dumps(
                {
                    "family": family.family_label(bottleneck),
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
        family.candidate_filename(bottleneck, epoch)
        for epoch in common.AE_CANDIDATE_EPOCHS
    }:
        raise guards.HybridQConfigError("candidate checkpoint set drift")
    require_holdout_unopened()
    require_no_selection_output(output, bottleneck)

    report = {
        "schema": schema,
        "terminal": terminal,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - run_started,
        "scope": {
            **family.family_fields(bottleneck),
            "trained_families": [family.family_label(bottleneck)],
            "other_family_trained_in_this_process": False,
            "ae128_touched": False,
            "optimization_split": "fit",
            "optimization_frames": contract.TRAIN_FIT_FRAMES,
            "holdout_opened_here": False,
            "holdout_frames_in_optimizer_batches": 0,
            "holdout_teacher_shards_deserialized": 0,
            "holdout_teacher_maps_loaded": 0,
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
        "configuration": configuration,
        "configuration_delta_from_ae128": family.configuration_delta(bottleneck),
        "teacher_store": store_provenance,
        "resumed_from_completed_epoch": resumed_from,
        "epoch_commit_order": (
            "per epoch, all atomic: candidate checkpoint, then the family epoch "
            "summary file, then the recovery checkpoint last; "
            f"recovery/{family.recovery_filename(bottleneck, 1)} and its siblings "
            "are the sole durable declaration that an epoch completed and embed "
            "that epoch's final summary"
        ),
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
            "run the separate Phase-10A holdout-selection command for this "
            "family; it is not launched here"
        ),
    }
    report_path = output / family.training_report_filename(bottleneck)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / terminal).write_text(f"{sha256_file(report_path)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "family": family.family_label(bottleneck),
                "report_sha256": sha256_file(report_path),
                "output": str(output),
            }
        )
    )
    print(terminal)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
