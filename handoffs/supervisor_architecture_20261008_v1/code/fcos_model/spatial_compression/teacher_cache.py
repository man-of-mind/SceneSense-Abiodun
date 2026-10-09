"""Phase-4 frozen train-only teacher cache and fit-reference loss medians.

Generates, for every registered training frame, the combined D/G/S/A teacher
importance map at q=0 behind the frozen epoch-26 perception lock, and freezes the
four fit-partition reference medians the q-aware objective will divide by.

Scope, deliberately: no ranker, no optimizer, no optimizer step, no validation or
test access, no evaluation, no CARLA. The frozen stack is loaded in eval mode with
requires_grad=False; gradients exist only to reach the C2 leaf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from . import contract, guards, training
from .gpu_qualification import (
    build_train_dataset,
    encode_front,
    load_frozen_perception,
    loss_groups_from_c2,
    package_source_hashes,
    sha256_file,
)

EXECUTE_TOKEN = "HYBRID_Q_PHASE4_TEACHER_CACHE"
SHARD_SCHEMA = "splitfusion_fcos_hybrid_q_teacher_cache_shard_v1"
MANIFEST_SCHEMA = "splitfusion_fcos_hybrid_q_teacher_cache_manifest_v1"
DATALOADER_WORKERS = 8

# One cached map is [112,192] FP32; a full shard of 256 frames is exactly this many
# bytes of tensor payload. A leaked C2 tensor would add ~22 MB per frame, so the
# serialized shard staying inside this budget plus a small metadata allowance is an
# independent structural check that no C2 tensor was written.
MAP_BYTES = contract.SPLIT_CELLS * 4
SHARD_METADATA_BUDGET_BYTES = 4 << 20


# ---------------------------------------------------------------------------
# Locked split partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPartition:
    """Registered-order fit and holdout index lists over the train dataset."""

    fit_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    fit_sample_ids: tuple[str, ...]
    holdout_sample_ids: tuple[str, ...]

    def indices_for(self, split: str) -> tuple[int, ...]:
        return self.fit_indices if split == "fit" else self.holdout_indices


def build_split_partition(dataset: Any) -> SplitPartition:
    """Partition the registered train frames by episode and fail closed on drift."""
    if len(dataset.rows) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError(
            f"train frame count {len(dataset.rows)} != {contract.TRAIN_TOTAL_FRAMES}"
        )
    episodes = {row["experiment_id"] for row in dataset.rows}
    expected_episodes = set(contract.TRAIN_FIT_EPISODES) | set(contract.TRAIN_HOLDOUT_EPISODES)
    if episodes != expected_episodes:
        raise guards.HybridQConfigError(
            f"registered training episodes drift: {sorted(episodes ^ expected_episodes)}"
        )

    fit_indices: list[int] = []
    holdout_indices: list[int] = []
    fit_ids: list[str] = []
    holdout_ids: list[str] = []
    for index, row in enumerate(dataset.rows):
        # split_for_episode raises on any episode outside the locked partition.
        if contract.split_for_episode(row["experiment_id"]) == "holdout":
            holdout_indices.append(index)
            holdout_ids.append(row["sample_id"])
        else:
            fit_indices.append(index)
            fit_ids.append(row["sample_id"])

    if len(fit_indices) != contract.TRAIN_FIT_FRAMES:
        raise guards.HybridQConfigError(
            f"fit frame count {len(fit_indices)} != {contract.TRAIN_FIT_FRAMES}"
        )
    if len(holdout_indices) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError(
            f"holdout frame count {len(holdout_indices)} != {contract.TRAIN_HOLDOUT_FRAMES}"
        )
    if contract.sample_id_digest(fit_ids) != contract.TRAIN_FIT_SAMPLE_ID_SHA256:
        raise guards.HybridQConfigError("fit frame identity drift")
    if contract.sample_id_digest(holdout_ids) != contract.TRAIN_HOLDOUT_SAMPLE_ID_SHA256:
        raise guards.HybridQConfigError("holdout frame identity drift")
    if set(fit_ids) & set(holdout_ids):
        raise guards.HybridQConfigError("fit and holdout frames overlap")

    return SplitPartition(
        fit_indices=tuple(fit_indices),
        holdout_indices=tuple(holdout_indices),
        fit_sample_ids=tuple(fit_ids),
        holdout_sample_ids=tuple(holdout_ids),
    )


def partition_loader(base: Any, dataset: Any, indices: Sequence[int], *, workers: int) -> DataLoader:
    """Registered-order batches of the locked size; never mixes fit and holdout."""
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=contract.TEACHER_CACHE_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        collate_fn=base.data.collate,
        drop_last=False,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Per-batch teacher records
# ---------------------------------------------------------------------------


def teacher_records_for_batch(
    model: torch.nn.Module, base: Any, batch: Mapping[str, Any], device: torch.device,
    *, split: str, use_amp: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame D/G/S/A teacher maps and the batch's dense q=0 task losses.

    Unlike the Phase-3 qualification this tolerates group validity differing
    across frames of a batch: a legitimately absent or zero-gradient group is
    recorded per frame with its reason. A frame with *no* valid group is still a
    hard failure, because its combined teacher map would not exist.
    """
    c2 = encode_front(model, batch, device)
    leaf = c2.detach().clone().float().requires_grad_(True)
    del c2
    _outputs, groups = loss_groups_from_c2(model, base, leaf, batch, use_amp=use_amp)

    task_losses: dict[str, float] = {}
    batch_exclusions: dict[str, str] = {}
    backward_order: list[str] = []
    for name in contract.TEACHER_GROUPS:
        loss = groups.get(name)
        if loss is None:
            batch_exclusions[name] = "absent_loss_group"
            continue
        value = float(loss.detach())
        task_losses[name] = value
        if not math.isfinite(value):
            # Never differentiate a non-finite loss; record it and move on.
            batch_exclusions[name] = "non_finite_task_loss"
            continue
        backward_order.append(name)

    group_grads: dict[str, torch.Tensor | None] = {}
    for position, name in enumerate(backward_order):
        grad, = torch.autograd.grad(
            groups[name], leaf,
            retain_graph=position < len(backward_order) - 1,
            allow_unused=True,
        )
        if grad is None:
            batch_exclusions[name] = "no_gradient_path_to_c2"
            group_grads[name] = None
            continue
        group_grads[name] = grad.detach()
    del _outputs, groups

    sample_ids = list(batch["sample_ids"])
    episode_ids = [row["experiment_id"] for row in batch["rows"]]
    records: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        frame_grads = {
            name: (None if grad is None else grad[index])
            for name, grad in group_grads.items()
        }
        result = training.build_teacher_maps(
            leaf[index].detach(), frame_grads, task_losses=task_losses
        )
        if not result.is_supervisable:
            raise guards.HybridQNumericalError(
                f"frame {sample_id} has no valid teacher group"
            )
        importance = result.importance
        guards.require_finite(importance, "combined teacher map")
        if bool((importance < 0).any()) or float(importance.sum()) <= 0.0:
            raise guards.HybridQNumericalError(
                f"combined teacher map for {sample_id} is not positive"
            )
        # A group dropped for the whole batch keeps its specific batch-level reason
        # rather than the generic per-frame "absent".
        excluded = dict(result.excluded_groups)
        excluded.update(batch_exclusions)
        records.append({
            "sample_id": sample_id,
            "episode_id": episode_ids[index],
            "split": split,
            "importance": importance.detach().to("cpu", torch.float32).contiguous(),
            "valid_groups": tuple(result.valid_groups),
            "excluded_groups": excluded,
            "gradient_mass": {k: float(v) for k, v in result.gradient_mass.items()},
            "task_losses": {k: float(v) for k, v in task_losses.items()},
        })

    del leaf, group_grads
    summary = {
        "split": split,
        "frames": len(records),
        "task_losses": dict(task_losses),
        "excluded_groups": dict(batch_exclusions),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Sharded, uncompressed, atomically written cache
# ---------------------------------------------------------------------------


class ShardWriter:
    """Buffers records and writes fixed-size, split-pure, uncompressed shards."""

    def __init__(self, shard_dir: Path, common: Mapping[str, Any], *, frames_per_shard: int) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=False)
        self.common = dict(common)
        self.frames_per_shard = int(frames_per_shard)
        self.entries: list[dict[str, Any]] = []
        self._buffer: list[dict[str, Any]] = []
        self._next_index = 0
        self._cursor = 0

    def add(self, record: Mapping[str, Any]) -> None:
        self._buffer.append(dict(record))
        if len(self._buffer) >= self.frames_per_shard:
            self.flush()

    def flush(self) -> None:
        """Close the current shard, if any. Called at every partition boundary."""
        if not self._buffer:
            return
        buffer, self._buffer = self._buffer, []
        splits = {record["split"] for record in buffer}
        if len(splits) != 1:
            raise guards.HybridQConfigError("a shard must not mix fit and holdout frames")

        maps = torch.stack([record["importance"] for record in buffer]).contiguous()
        if maps.dtype != torch.float32:
            raise guards.HybridQPayloadError("cached teacher maps must be FP32")
        if tuple(maps.shape[1:]) != contract.SPLIT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError("cached teacher maps must be [112,192]")

        shard_index = self._next_index
        self._next_index += 1
        payload = {
            "schema": SHARD_SCHEMA,
            "shard_index": shard_index,
            "frames": len(buffer),
            "split": buffer[0]["split"],
            "cache_index_start": self._cursor,
            "sample_ids": [record["sample_id"] for record in buffer],
            "episode_ids": [record["episode_id"] for record in buffer],
            "splits": [record["split"] for record in buffer],
            "importance": maps,
            "valid_groups": [list(record["valid_groups"]) for record in buffer],
            "excluded_groups": [dict(record["excluded_groups"]) for record in buffer],
            "gradient_mass": [dict(record["gradient_mass"]) for record in buffer],
            "task_losses": [dict(record["task_losses"]) for record in buffer],
            **self.common,
        }

        name = f"teacher_shard_{shard_index:05d}.pt"
        final = self.shard_dir / name
        temporary = self.shard_dir / f".{name}.tmp"
        # Uncompressed zip container, then an atomic rename into place.
        torch.save(payload, temporary, _use_new_zipfile_serialization=True)
        os.replace(temporary, final)

        size = final.stat().st_size
        budget = len(buffer) * MAP_BYTES + SHARD_METADATA_BUDGET_BYTES
        if size > budget:
            raise guards.HybridQPayloadError(
                f"shard {name} is {size} B, above the {budget} B teacher-map budget; "
                "a non-teacher tensor may have been serialized"
            )
        self.entries.append({
            "shard_index": shard_index,
            "path": f"shards/{name}",
            "split": buffer[0]["split"],
            "frames": len(buffer),
            "fit_frames": sum(1 for record in buffer if record["split"] == "fit"),
            "holdout_frames": sum(1 for record in buffer if record["split"] == "holdout"),
            "cache_index_start": self._cursor,
            "cache_index_end": self._cursor + len(buffer),
            "first_sample_id": buffer[0]["sample_id"],
            "last_sample_id": buffer[-1]["sample_id"],
            "bytes": size,
            "sha256": sha256_file(final),
        })
        self._cursor += len(buffer)


def load_shard(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _tensors(value: Any, path: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}[{index}]")


def verify_shard(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read one shard from disk and re-check every structural cache gate."""
    path = root / entry["path"]
    size = path.stat().st_size
    if size != int(entry["bytes"]):
        raise guards.HybridQPayloadError(f"{entry['path']} byte size drift")
    if sha256_file(path) != entry["sha256"]:
        raise guards.HybridQPayloadError(f"{entry['path']} sha256 drift")

    payload = load_shard(path)
    if payload["schema"] != SHARD_SCHEMA:
        raise guards.HybridQConfigError(f"{entry['path']} schema drift")

    tensors = dict(_tensors(payload))
    if set(tensors) != {".importance"}:
        raise guards.HybridQPayloadError(
            f"{entry['path']} serializes unexpected tensors: {sorted(tensors)}"
        )
    maps = tensors[".importance"]
    frames = int(entry["frames"])
    if maps.dtype != torch.float32:
        raise guards.HybridQPayloadError(f"{entry['path']} maps are not FP32")
    if tuple(maps.shape) != (frames,) + contract.SPLIT_SPATIAL_SHAPE:
        raise guards.HybridQPayloadError(f"{entry['path']} map shape drift")
    if not torch.isfinite(maps).all():
        raise guards.HybridQNumericalError(f"{entry['path']} holds a non-finite map")
    if bool((maps < 0).any()):
        raise guards.HybridQNumericalError(f"{entry['path']} holds a negative map")
    masses = maps.reshape(frames, -1).sum(dim=1)
    if bool((masses <= 0).any()):
        raise guards.HybridQNumericalError(f"{entry['path']} holds a zero-mass map")

    if len(payload["sample_ids"]) != frames or len(payload["splits"]) != frames:
        raise guards.HybridQPayloadError(f"{entry['path']} identifier count drift")
    if set(payload["splits"]) - set(contract.SPLIT_LABELS):
        raise guards.HybridQConfigError(f"{entry['path']} has an unregistered split label")
    if len(set(payload["splits"])) != 1 or payload["splits"][0] != entry["split"]:
        raise guards.HybridQConfigError(f"{entry['path']} split label drift")
    for sample_id, episode_id, label in zip(
        payload["sample_ids"], payload["episode_ids"], payload["splits"]
    ):
        if contract.split_for_episode(episode_id) != label:
            raise guards.HybridQConfigError(
                f"{sample_id} is labelled {label} but belongs to episode {episode_id}"
            )
    if payload["perception_checkpoint_sha256"] != contract.FROZEN_CHECKPOINT_SHA256:
        raise guards.HybridQConfigError(f"{entry['path']} checkpoint hash drift")

    return {
        "sample_ids": list(payload["sample_ids"]),
        "splits": list(payload["splits"]),
        "valid_groups": [tuple(item) for item in payload["valid_groups"]],
        "excluded_groups": list(payload["excluded_groups"]),
        "map_mass_min": float(masses.min()),
        "map_mass_max": float(masses.max()),
        "map_min": float(maps.min()),
    }


# ---------------------------------------------------------------------------
# Frozen fit reference medians
# ---------------------------------------------------------------------------


def fit_reference_medians(
    fit_batch_losses: Sequence[Mapping[str, float]]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Conventional median of the finite positive q=0 fit-batch task losses."""
    medians: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for group in contract.TEACHER_GROUPS:
        admitted = [
            float(losses[group]) for losses in fit_batch_losses
            if group in losses and math.isfinite(losses[group]) and float(losses[group]) > 0.0
        ]
        rejected = len(fit_batch_losses) - len(admitted)
        if not admitted:
            raise guards.HybridQNumericalError(
                f"group {group} has no finite positive fit task loss"
            )
        # statistics.median averages the two central values for an even count.
        value = float(statistics.median(admitted))
        if not math.isfinite(value) or value <= 0.0:
            raise guards.HybridQNumericalError(
                f"reference median for {group} is not finite and positive"
            )
        medians[group] = value
        detail[group] = {
            "median": value,
            "contributing_fit_batches": len(admitted),
            "rejected_fit_batches": rejected,
            "min": min(admitted),
            "max": max(admitted),
            "mean": float(statistics.fmean(admitted)),
        }
    return medians, detail


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationOutcome:
    frames: int = 0
    batches: int = 0
    fit_batch_losses: list[dict[str, float]] = field(default_factory=list)
    holdout_batches: int = 0
    batch_summaries: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


def generate(
    model: torch.nn.Module, base: Any, dataset: Any, partition: SplitPartition,
    writer: ShardWriter, device: torch.device, *,
    use_amp: bool, workers: int, batch_limit: int | None = None,
    progress: str | None = None,
) -> GenerationOutcome:
    """One create-only pass: fit partition first, then holdout, both in order."""
    outcome = GenerationOutcome()
    planned = 0
    for split in contract.SPLIT_LABELS:
        count = len(partition.indices_for(split))
        batches = (count + contract.TEACHER_CACHE_BATCH_SIZE - 1) // contract.TEACHER_CACHE_BATCH_SIZE
        planned += batches if batch_limit is None else min(batches, batch_limit)
    milestones = {max(1, (planned * fraction) // 100): fraction for fraction in (25, 50, 75)}
    started = time.time()

    for split in contract.SPLIT_LABELS:
        indices = partition.indices_for(split)
        loader = partition_loader(base, dataset, indices, workers=workers)
        for batch_index, batch in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            records, summary = teacher_records_for_batch(
                model, base, batch, device, split=split, use_amp=use_amp
            )
            for record in records:
                writer.add(record)
            outcome.frames += len(records)
            outcome.batches += 1
            summary["batch_index"] = batch_index
            outcome.batch_summaries.append(summary)
            if split == "fit":
                outcome.fit_batch_losses.append(dict(summary["task_losses"]))
            else:
                outcome.holdout_batches += 1
            if progress and outcome.batches in milestones:
                elapsed = time.time() - started
                print(
                    f"[{progress}] {milestones[outcome.batches]}% "
                    f"({outcome.batches}/{planned} batches, {outcome.frames} frames, "
                    f"{elapsed / 60.0:.1f} min elapsed)",
                    flush=True,
                )
        # Never let a shard straddle the fit/holdout boundary.
        writer.flush()
        del loader

    writer.flush()
    outcome.seconds = time.time() - started
    return outcome


def smoke(
    model: torch.nn.Module, base: Any, dataset: Any, partition: SplitPartition,
    device: torch.device, smoke_dir: Path, common: Mapping[str, Any], *,
    use_amp: bool, frozen_snapshot: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """One fit batch and one holdout batch through the real write/read path."""
    writer = ShardWriter(smoke_dir / "shards", common, frames_per_shard=contract.TEACHER_CACHE_BATCH_SIZE)
    outcome = generate(
        model, base, dataset, partition, writer, device,
        use_amp=use_amp, workers=2, batch_limit=1,
    )
    if len(writer.entries) != 2:
        raise guards.HybridQPayloadError("the smoke must write exactly two shards")

    observed: list[str] = []
    labels: list[str] = []
    for entry in writer.entries:
        verified = verify_shard(smoke_dir, entry)
        observed.extend(verified["sample_ids"])
        labels.extend(verified["splits"])

    expected_fit = list(partition.fit_sample_ids[:contract.TEACHER_CACHE_BATCH_SIZE])
    expected_holdout = list(partition.holdout_sample_ids[:contract.TEACHER_CACHE_BATCH_SIZE])
    if observed != expected_fit + expected_holdout:
        raise guards.HybridQConfigError("smoke sample ids do not reconcile with the locked order")
    if labels != ["fit"] * len(expected_fit) + ["holdout"] * len(expected_holdout):
        raise guards.HybridQConfigError("smoke split labels do not reconcile")
    guards.require_module_state_unchanged(model, frozen_snapshot)

    return {
        "passed": True,
        "shards": writer.entries,
        "frames": outcome.frames,
        "batches": outcome.batches,
        "seconds": outcome.seconds,
        "fit_batch_sample_ids": expected_fit,
        "holdout_batch_sample_ids": expected_holdout,
        "checks": [
            "shard write and read back succeeded",
            "sample ids and split labels reconcile with the locked registered order",
            "maps are FP32 [112,192]",
            "maps are finite, non-negative and positive-mass",
            "the only serialized tensor is the teacher map; no C2 tensor is present",
            "frozen perception parameters and buffers exactly unchanged",
        ],
        "frozen_state_unchanged": True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid-q Phase-4 train-only teacher cache")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")

    if not torch.cuda.is_available():
        raise RuntimeError("the Phase-4 teacher cache requires CUDA")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / 2 ** 30
    if total_gib < contract.TEACHER_CACHE_MIN_GPU_TOTAL_GIB:
        raise guards.HybridQConfigError(
            f"total GPU memory {total_gib:.2f} GiB is below the required "
            f"{contract.TEACHER_CACHE_MIN_GPU_TOTAL_GIB} GiB; the cache batch size is part "
            "of the registered reference-loss contract and must not be reduced silently"
        )

    # Cache payload plus headroom for the in-flight temporary shard and the smoke.
    projected_bytes = contract.TRAIN_TOTAL_FRAMES * MAP_BYTES
    required_bytes = projected_bytes + (1 << 30)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise guards.HybridQConfigError(
            f"free disk {free_bytes / 2 ** 30:.2f} GiB is below the required "
            f"{required_bytes / 2 ** 30:.2f} GiB for the cache plus temporary space"
        )

    torch.manual_seed(contract.RANKER_INIT_SEED)
    locked = contract.load_locked_config()
    locked_sha256 = sha256_file(contract.locked_config_path())
    source_hashes = package_source_hashes()

    model, base, binding = load_frozen_perception(device)
    dataset = build_train_dataset(base)
    partition = build_split_partition(dataset)
    frozen_snapshot = guards.snapshot_module_state(model)
    torch.cuda.reset_peak_memory_stats(device)

    common = {
        "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
        "locked_config_sha256": locked_sha256,
        "hybrid_q_source_sha256": source_hashes,
        "normalization": contract.TEACHER_NORMALIZATION,
        "combination": contract.TEACHER_GROUP_COMBINATION,
        "q": 0.0,
        "seed": contract.RANKER_INIT_SEED,
    }

    output.mkdir(parents=True, exist_ok=False)
    print(f"[phase4] start: {contract.TRAIN_TOTAL_FRAMES} train frames "
          f"({contract.TRAIN_FIT_FRAMES} fit, {contract.TRAIN_HOLDOUT_FRAMES} holdout) "
          f"on {properties.name}, {total_gib:.2f} GiB", flush=True)

    smoke_dir = output / "smoke_tmp"
    smoke_report = smoke(
        model, base, dataset, partition, device, smoke_dir, common,
        use_amp=True, frozen_snapshot=frozen_snapshot,
    )
    shutil.rmtree(smoke_dir)
    smoke_report["temporary_output_removed"] = True
    print(f"[phase4] smoke passed in {smoke_report['seconds']:.1f} s; "
          "beginning the single full create-only generation", flush=True)

    writer = ShardWriter(
        output / "shards", common, frames_per_shard=contract.TEACHER_CACHE_SHARD_FRAMES
    )
    outcome = generate(
        model, base, dataset, partition, writer, device,
        use_amp=True, workers=int(args.workers), progress="phase4",
    )
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 20
    guards.require_module_state_unchanged(model, frozen_snapshot)

    # ---- completion gates -------------------------------------------------
    observed_ids: list[str] = []
    observed_splits: list[str] = []
    valid_counts = {group: 0 for group in contract.TEACHER_GROUPS}
    exclusion_counts: dict[str, dict[str, int]] = {group: {} for group in contract.TEACHER_GROUPS}
    map_mass_min = math.inf
    map_min = math.inf
    for entry in writer.entries:
        verified = verify_shard(output, entry)
        observed_ids.extend(verified["sample_ids"])
        observed_splits.extend(verified["splits"])
        for groups in verified["valid_groups"]:
            for group in groups:
                valid_counts[group] += 1
        for excluded in verified["excluded_groups"]:
            for group, reason in excluded.items():
                exclusion_counts[group][reason] = exclusion_counts[group].get(reason, 0) + 1
        map_mass_min = min(map_mass_min, verified["map_mass_min"])
        map_min = min(map_min, verified["map_min"])

    registered_ids = [row["sample_id"] for row in dataset.rows]
    if len(observed_ids) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError(f"cached {len(observed_ids)} frames")
    if len(set(observed_ids)) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError("the cache holds duplicate frames")
    if set(observed_ids) != set(registered_ids):
        raise guards.HybridQConfigError(
            "the cache does not match the registered train frame set exactly"
        )
    if observed_ids != list(partition.fit_sample_ids) + list(partition.holdout_sample_ids):
        raise guards.HybridQConfigError("cache order drift against the locked partition")
    fit_cached = observed_splits.count("fit")
    holdout_cached = observed_splits.count("holdout")
    if fit_cached != contract.TRAIN_FIT_FRAMES or holdout_cached != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError(
            f"cached split counts {fit_cached}/{holdout_cached} drift from the locked split"
        )
    if not math.isfinite(map_mass_min) or map_mass_min <= 0.0 or map_min < 0.0:
        raise guards.HybridQNumericalError("a cached teacher map is not positive-mass")

    expected_fit_batches = -(-contract.TRAIN_FIT_FRAMES // contract.TEACHER_CACHE_BATCH_SIZE)
    if len(outcome.fit_batch_losses) != expected_fit_batches:
        raise guards.HybridQConfigError(
            f"{len(outcome.fit_batch_losses)} fit batches != {expected_fit_batches}"
        )
    medians, median_detail = fit_reference_medians(outcome.fit_batch_losses)
    references = training.ReferenceMedians(medians=medians)
    if set(references.medians) != set(contract.TEACHER_GROUPS):
        raise guards.HybridQConfigError("the frozen reference medians are incomplete")

    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": generated,
        "terminal": "HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE",
        "scope": "frozen train-only q=0 teacher cache; no ranker, no optimizer, no evaluation",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": properties.name,
            "total_gpu_gib": round(total_gib, 3),
        },
        "perception_binding": binding,
        "locked_config_sha256": locked_sha256,
        "hybrid_q_source_sha256": source_hashes,
        "split": {
            "fit_frames": fit_cached,
            "holdout_frames": holdout_cached,
            "total_frames": len(observed_ids),
            "fit_sample_id_sha256": contract.sample_id_digest(partition.fit_sample_ids),
            "holdout_sample_id_sha256": contract.sample_id_digest(partition.holdout_sample_ids),
            "holdout_episodes": list(contract.TRAIN_HOLDOUT_EPISODES),
            "validation_or_test_frames": 0,
        },
        "generation": {
            "batch_size": contract.TEACHER_CACHE_BATCH_SIZE,
            "batches": outcome.batches,
            "fit_batches": len(outcome.fit_batch_losses),
            "holdout_batches": outcome.holdout_batches,
            "augmentation": False,
            "seed": contract.RANKER_INIT_SEED,
            "precision": "bf16 autocast tail, fp32 C2 boundary and losses (registered)",
            "seconds": round(outcome.seconds, 1),
            "peak_allocated_vram_mib": peak_allocated,
            "peak_reserved_vram_mib": peak_reserved,
            "optimizer_steps": 0,
            "ranker_constructed": False,
            "frozen_state_unchanged_at_end": True,
        },
        "teacher": {
            "importance": "I_t(h,w) = sum_c |C2(c,h,w) * grad_t(c,h,w)|",
            "normalization": contract.TEACHER_NORMALIZATION,
            "combination": contract.TEACHER_GROUP_COMBINATION,
            "valid_frame_counts": valid_counts,
            "exclusion_counts": exclusion_counts,
            "map_dtype": "float32",
            "map_shape": list(contract.SPLIT_SPATIAL_SHAPE),
            "min_map_value_over_cache": map_min,
            "min_map_mass_over_cache": map_mass_min,
        },
        "disk": {
            "projected_map_bytes": projected_bytes,
            "required_free_bytes": required_bytes,
            "free_bytes_before_run": free_bytes,
        },
        "shards": {
            "frames_per_shard": contract.TEACHER_CACHE_SHARD_FRAMES,
            "count": len(writer.entries),
            "compression": "none",
            "total_bytes": sum(entry["bytes"] for entry in writer.entries),
            "all_hashes_verified": True,
            "entries": writer.entries,
        },
        "smoke": smoke_report,
    }
    (output / "teacher_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "fit_reference_medians.json").write_text(
        json.dumps({
            "schema": "splitfusion_fcos_hybrid_q_fit_reference_medians_v1",
            "generated_utc": generated,
            "source": contract.REFERENCE_MEDIAN_SOURCE,
            "statistic": "conventional median; the two central values are averaged for an even count",
            "admitted": "finite positive q=0 task losses only",
            "batch_size": contract.TEACHER_CACHE_BATCH_SIZE,
            "fit_batches": len(outcome.fit_batch_losses),
            "holdout_contribution": "none",
            "frozen": True,
            "medians": medians,
            "detail": median_detail,
            "fit_batch_task_losses": outcome.fit_batch_losses,
            "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
            "locked_config_sha256": locked_sha256,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "TEACHER_CACHE_COMPLETE").write_text(
        f"HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE {generated}\n", encoding="utf-8"
    )
    print(f"[phase4] complete: {len(observed_ids)} frames in {len(writer.entries)} shards, "
          f"{outcome.seconds / 60.0:.1f} min, peak allocated {peak_allocated:.1f} MiB", flush=True)
    print("HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - runner entry point
    raise SystemExit(main())
