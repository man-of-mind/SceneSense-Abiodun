"""Shared Phase-9C configuration, schedule, data join and checkpoint helpers.

Everything both AE128 runners need and neither should restate: the locked
training configuration, the exact Stage-A/Stage-B schedule, the sample-id-keyed
Phase-4 teacher store, and the atomic exact-resume checkpoint format.

Nothing here loads a checkpoint, touches CUDA, reads a dataset or cache, trains,
infers or evaluates at import time. The frozen hybrid-q package, the AE model,
the AE loss, the ranker, the perception model and the p025 scorer are imported
and reused, never modified.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    phase5_common,
    teacher_cache,
    training,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import sha256_file
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.ranker import build_ranker
from . import ae_contract, ae_loss
from .ae_gpu_qualification import bind_inputs as bind_frozen_inputs
from .ae_gpu_qualification import freeze, state_hashes
from .ae_model import SplitFeatureAE, ae_parameters, build_split_feature_ae


# ---------------------------------------------------------------------------
# Locked Phase-9C training configuration
# ---------------------------------------------------------------------------

AE_TRAINING_SCHEMA = "splitfusion_fcos_ae128_phase9c_training_v1"
AE_RECOVERY_SCHEMA = "splitfusion_fcos_ae128_phase9c_recovery_v1"
AE_CANDIDATE_SCHEMA = "splitfusion_fcos_ae128_phase9c_candidate_v1"
AE_HOLDOUT_SCHEMA = "splitfusion_fcos_ae128_phase9c_holdout_selection_v1"

# AE128 only. AE64 and AE32 are out of scope for Phase 9C entirely.
AE_TRAINING_BOTTLENECK = 128

# Stage A is dense q=0; Stage B is the locked four-value cycle. The epoch split
# and the 12-epoch length are the frozen Phase-5 shape, reused unchanged.
AE_STAGE_A = "stage_a"
AE_STAGE_B = "stage_b"
AE_STAGE_A_EPOCHS = contract.DISTILLATION_EPOCHS  # 4
AE_STAGE_B_EPOCHS = contract.Q_AWARE_EPOCHS  # 8
AE_TRAINING_EPOCHS = contract.TRAINING_EPOCHS  # 12
AE_CANDIDATE_EPOCHS = contract.CHECKPOINT_EPOCHS  # 4, 8, 12

AE_STAGE_A_Q = ae_contract.AE_STAGE_A_Q  # 0.00
AE_STAGE_B_Q_CYCLE = ae_contract.AE_STAGE_B_Q_CYCLE  # 0.00, 0.30, 0.50, 0.70
AE_EXCLUDED_Q = ae_contract.AE_OPTIMIZATION_EXCLUDED_Q  # 0.90, 0.98

# Optimizer. AdamW state is preserved across the Stage-A/Stage-B transition;
# only the learning rate changes.
AE_STAGE_A_LEARNING_RATE = 1e-3
AE_STAGE_B_LEARNING_RATE = 3e-4
AE_WEIGHT_DECAY = contract.WEIGHT_DECAY  # 1e-4
AE_GRAD_CLIP_GLOBAL_NORM = contract.GRAD_CLIP_GLOBAL_NORM  # 5.0

# Physical batch 16, final short batch retained, augmentation off.
AE_BATCH_SIZE = contract.TRAINING_BATCH_SIZE  # 16
AE_DROP_LAST = contract.DROP_LAST_TRAINING_BATCH  # False
AE_AUGMENTATION = False

# Holdout selection. FP32 AE latent reconstruction only at this phase.
AE_HOLDOUT_Q_VALUES = tuple(AE_STAGE_B_Q_CYCLE)  # 0.00, 0.30, 0.50, 0.70
AE_HOLDOUT_QUANTIZER = "fp32_latent_no_uint8_no_zstd"

# The frozen noAE same-q reference this phase compares against: the completed
# Phase-5 train-holdout evaluation, whose q>0 rows were produced by exactly the
# bound stable epoch-4 ranker and whose q=0 row is the ranker-free baseline.
NOAE_HOLDOUT_RELPATH = (
    "experiments/splitfusion_fcos_hybrid_q_v1/"
    "20260901_185725_phase5_ranker_training/holdout/holdout_evaluation.json"
)
NOAE_HOLDOUT_SHA256 = (
    "b86df9aeea6a9d5bb269f7a9d1f185f6bd4a93ffe87d3dd04a0ade1b15922717"
)
NOAE_REFERENCE_RANKER_EPOCH = contract.VALIDATION_RANKER_EPOCH  # 4


# ---------------------------------------------------------------------------
# Exact schedule
# ---------------------------------------------------------------------------


def require_training_epoch(epoch: int) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise guards.HybridQConfigError("epoch must be an int")
    if not 1 <= epoch <= AE_TRAINING_EPOCHS:
        raise guards.HybridQConfigError(
            f"epoch {epoch} is outside the locked 1..{AE_TRAINING_EPOCHS} schedule"
        )
    return int(epoch)


def stage_for_epoch(epoch: int) -> str:
    """Epochs 1-4 are Stage A (q=0 only); epochs 5-12 are Stage B."""
    return AE_STAGE_A if require_training_epoch(epoch) <= AE_STAGE_A_EPOCHS else AE_STAGE_B


def learning_rate_for_stage(stage: str) -> float:
    if stage == AE_STAGE_A:
        return AE_STAGE_A_LEARNING_RATE
    if stage == AE_STAGE_B:
        return AE_STAGE_B_LEARNING_RATE
    raise guards.HybridQConfigError(f"{stage!r} is not a registered AE training stage")


def batches_per_epoch(frames: int = contract.TRAIN_FIT_FRAMES) -> int:
    """Every fit frame exactly once at batch 16 with the final short batch kept."""
    if AE_DROP_LAST:
        raise guards.HybridQConfigError("the locked schedule never drops the last batch")
    return -(-int(frames) // AE_BATCH_SIZE)


def stage_b_updates_total() -> int:
    return AE_STAGE_B_EPOCHS * batches_per_epoch()


def stage_b_q_at(position: int) -> float:
    """Continuous round robin: the cycle position carries across epoch boundaries.

    `position` is the global count of Stage-B updates already taken, so the
    cycle is never restarted at an epoch boundary. 847 batches per epoch is not
    a multiple of four, so restarting per epoch would bias the four q values;
    carrying the position makes the totals exactly equal.
    """
    if isinstance(position, bool) or not isinstance(position, int):
        raise guards.HybridQConfigError("Stage-B cycle position must be an int")
    if position < 0:
        raise guards.HybridQConfigError("Stage-B cycle position must be non-negative")
    return float(AE_STAGE_B_Q_CYCLE[position % len(AE_STAGE_B_Q_CYCLE)])


def stage_b_q_counts(updates: int | None = None) -> dict[str, int]:
    """Exact per-q Stage-B update counts over the whole locked schedule."""
    total = stage_b_updates_total() if updates is None else int(updates)
    counts = {f"{float(q):.2f}": 0 for q in AE_STAGE_B_Q_CYCLE}
    for position in range(total):
        counts[f"{stage_b_q_at(position):.2f}"] += 1
    return counts


def require_balanced_stage_b() -> dict[str, int]:
    """Fail closed unless the locked schedule is exactly balanced over the four q."""
    counts = stage_b_q_counts()
    expected = stage_b_updates_total() // len(AE_STAGE_B_Q_CYCLE)
    if stage_b_updates_total() % len(AE_STAGE_B_Q_CYCLE) != 0:
        raise guards.HybridQConfigError(
            "the Stage-B schedule length is not a whole number of cycles"
        )
    if set(counts.values()) != {expected}:
        raise guards.HybridQConfigError(f"Stage-B q counts are unbalanced: {counts}")
    for q in AE_STAGE_B_Q_CYCLE:
        ae_loss.require_optimization_q(float(q))
    return counts


def training_configuration() -> dict[str, Any]:
    """The locked Phase-9C configuration as data, for the run record."""
    return {
        "family": ae_contract.family_name(
            ae_contract.family_for_bottleneck(AE_TRAINING_BOTTLENECK)
        ),
        "bottleneck": AE_TRAINING_BOTTLENECK,
        "initialization": "committed deterministic SplitFeatureAE initialization",
        "init_seed": ae_contract.ae_init_seed(AE_TRAINING_BOTTLENECK),
        "trainable_scope": ae_contract.AE_TRAINABLE_SCOPE,
        "epochs": AE_TRAINING_EPOCHS,
        "stage_a_epochs": AE_STAGE_A_EPOCHS,
        "stage_b_epochs": AE_STAGE_B_EPOCHS,
        "stage_a_q": AE_STAGE_A_Q,
        "stage_b_q_cycle": list(AE_STAGE_B_Q_CYCLE),
        "stage_b_cycle_carries_across_epochs": True,
        "stage_b_updates_total": stage_b_updates_total(),
        "stage_b_q_update_counts": stage_b_q_counts(),
        "excluded_from_optimization_q": list(AE_EXCLUDED_Q),
        "optimizer": "AdamW",
        "stage_a_learning_rate": AE_STAGE_A_LEARNING_RATE,
        "stage_b_learning_rate": AE_STAGE_B_LEARNING_RATE,
        "weight_decay": AE_WEIGHT_DECAY,
        "adamw_state_preserved_across_stage_transition": True,
        "grad_clip_global_norm": AE_GRAD_CLIP_GLOBAL_NORM,
        "batch_size": AE_BATCH_SIZE,
        "drop_last": AE_DROP_LAST,
        "batches_per_epoch": batches_per_epoch(),
        "augmentation": AE_AUGMENTATION,
        "augmentation_rationale": (
            "the Phase-4 importance maps were produced on the unaugmented frames, "
            "so an augmented frame would be supervised by a map of a different image"
        ),
        "optimization_split": "fit",
        "optimization_frames": contract.TRAIN_FIT_FRAMES,
        "selection_split": "reserved_train_holdout",
        "selection_frames": contract.TRAIN_HOLDOUT_FRAMES,
        "validation_or_test_accessed": False,
        "epoch_shuffle_seed_rule": "20260829 + epoch",
        "objective": "plain_reconstruction + combined_importance_reconstruction",
        "teacher_fields_consumed": ["importance", "valid_groups", "excluded_groups"],
        "per_group_maps_used": False,
        "fake_quantization_in_training": False,
        "zstd_in_training": False,
        "candidate_epochs": list(AE_CANDIDATE_EPOCHS),
    }


# ---------------------------------------------------------------------------
# Frozen stack
# ---------------------------------------------------------------------------


def load_stable_ranker(device: torch.device) -> torch.nn.Module:
    """The bound stable epoch-4 ranker, frozen and in eval mode."""
    payload = torch.load(
        contract.repository_root() / contract.VALIDATION_RANKER_RELPATH,
        map_location="cpu",
        weights_only=False,
    )
    if int(payload["epoch"]) != contract.VALIDATION_RANKER_EPOCH:
        raise guards.HybridQConfigError("stable ranker epoch drift")
    if int(payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError("stable ranker parameter-count drift")
    ranker = build_ranker()
    ranker.load_state_dict(payload["ranker"])
    ranker = ranker.to(device)
    freeze(ranker)
    del payload
    return ranker


def build_ae(device: torch.device) -> SplitFeatureAE:
    """AE128 from the committed deterministic initialization, on `device`."""
    autoencoder = build_split_feature_ae(AE_TRAINING_BOTTLENECK).to(device)
    if autoencoder.init_seed != ae_contract.ae_init_seed(AE_TRAINING_BOTTLENECK):
        raise guards.HybridQConfigError("AE initialization seed drift")
    return autoencoder


def build_ae_optimizer(
    autoencoder: SplitFeatureAE,
    *,
    lr: float,
    frozen_modules: Sequence[torch.nn.Module],
) -> torch.optim.Optimizer:
    """Locked AdamW owning exactly the eight named AE tensors and nothing else."""
    optimizer = training.build_ranker_optimizer(
        autoencoder,
        lr=float(lr),
        weight_decay=AE_WEIGHT_DECAY,
        frozen_modules=tuple(frozen_modules),
    )
    ae_loss.require_ae_only_optimizer(optimizer, autoencoder)
    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected = ae_parameters(autoencoder)
    if len(owned) != len(expected):
        raise guards.HybridQOwnershipError(
            f"optimizer owns {len(owned)} tensors, AE128 has {len(expected)}"
        )
    return optimizer


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> float:
    """Change only the learning rate; AdamW moment state is left untouched."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    return float(lr)


# ---------------------------------------------------------------------------
# Sample-id-keyed Phase-4 teacher store, one split at a time
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AeTeacherStore:
    """Cached teacher records for exactly one split, keyed by sample ID.

    Only the shards the manifest labels as `split` are ever deserialized. The
    opposite split's shards are named, counted and hash-verified from manifest
    metadata and then left closed, so this store cannot hold a single map of the
    split it is not for. Asking for a frame of the other split raises.
    """

    split: str
    maps: torch.Tensor  # [frames,112,192] FP32
    index: dict[str, int]
    valid_groups: tuple[tuple[str, ...], ...]
    excluded_groups: tuple[dict[str, str], ...]
    other_split_ids: frozenset[str]
    manifest_sha256: str = ""
    loaded_shards: tuple[str, ...] = ()
    withheld_shards: tuple[str, ...] = ()

    @property
    def frames(self) -> int:
        return int(self.maps.shape[0])

    @property
    def bytes(self) -> int:
        return int(self.maps.numel()) * self.maps.element_size()

    def provenance(self) -> dict[str, Any]:
        """What was opened and what was excluded, for the run record."""
        other = next(
            label for label in contract.SPLIT_LABELS if label != self.split
        )
        return {
            "split_loaded": self.split,
            "teacher_cache_manifest_sha256": self.manifest_sha256,
            "shards_opened": len(self.loaded_shards),
            "shards_withheld_unopened": len(self.withheld_shards),
            f"{other}_shards_deserialized": 0,
            f"{other}_maps_loaded": 0,
            f"{other}_ids_excluded": len(self.other_split_ids),
            "maps_loaded": self.frames,
            "policy": (
                "manifest and binary shard hashes are inspected in full; "
                f"torch.load is called only on {self.split} shards"
            ),
        }

    def record(self, sample_id: str) -> tuple[torch.Tensor, tuple[str, ...], dict[str, str]]:
        key = str(sample_id)
        if key in self.other_split_ids:
            raise guards.HybridQOwnershipError(
                f"{key} is not a {self.split} frame and must not be served here"
            )
        position = self.index.get(key)
        if position is None:
            raise guards.HybridQConfigError(f"{key} is not in the Phase-4 teacher cache")
        return (
            self.maps[position],
            self.valid_groups[position],
            self.excluded_groups[position],
        )

    def batch(self, sample_ids: Sequence[str]) -> ae_loss.CachedTeacherBatch:
        """Join by exact sample ID and build the committed cached-teacher batch."""
        rows = [self.record(str(name)) for name in sample_ids]
        return ae_loss.CachedTeacherBatch(
            importance=torch.stack([row[0] for row in rows]).contiguous(),
            valid_groups=tuple(tuple(row[1]) for row in rows),
            excluded_groups=tuple(dict(row[2]) for row in rows),
        )


@dataclass(frozen=True)
class TeacherShardPlan:
    """Which shards one split may open, decided from manifest metadata alone.

    Nothing here interprets a shard payload. The split decision, the frame
    accounting and the exact sample-ID coverage come from the manifest and from
    the registered `SplitPartition`; the opposite split's shards are named only
    so the run record can show they stayed closed.
    """

    split: str
    other_split: str
    manifest_sha256: str
    selected: tuple[Mapping[str, Any], ...]
    withheld: tuple[str, ...]
    expected_ids: tuple[str, ...]
    other_split_ids: frozenset[str]
    selected_frames: int
    withheld_frames: int


def plan_teacher_shards(
    cache_root: Path, partition: teacher_cache.SplitPartition, split: str
) -> TeacherShardPlan:
    """Read the whole manifest for provenance, then admit one split's shards.

    Reading manifest metadata and hashing shard bytes is provenance, not
    supervision. Deserializing an opposite-split tensor payload is what must not
    happen, so the manifest alone decides what may be opened.
    """
    if split not in contract.SPLIT_LABELS:
        raise guards.HybridQConfigError(f"{split!r} is not a registered split label")
    other_split = next(label for label in contract.SPLIT_LABELS if label != split)

    manifest_path = cache_root / "teacher_cache_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != contract.TEACHER_CACHE_MANIFEST_SHA256:
        raise guards.HybridQConfigError("Phase-4 teacher-cache manifest sha256 drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != teacher_cache.MANIFEST_SCHEMA:
        raise guards.HybridQConfigError("teacher-cache manifest schema drift")
    if manifest["terminal"] != "HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE":
        raise guards.HybridQConfigError("teacher-cache manifest is not a complete cache")
    if (
        manifest["perception_binding"]["checkpoint_sha256"]
        != contract.FROZEN_CHECKPOINT_SHA256
    ):
        raise guards.HybridQConfigError("teacher-cache manifest checkpoint binding drift")
    if int(manifest["split"]["validation_or_test_frames"]) != 0:
        raise guards.HybridQConfigError("teacher cache reports validation/test frames")

    entries = list(manifest["shards"]["entries"])
    if len(entries) != contract.TEACHER_CACHE_SHARD_COUNT:
        raise guards.HybridQConfigError("teacher-cache shard count drift")

    selected: list[Mapping[str, Any]] = []
    withheld: list[str] = []
    frames_by_split = {label: 0 for label in contract.SPLIT_LABELS}
    for entry in entries:
        path = str(entry["path"])
        label = str(entry["split"])
        if label not in contract.SPLIT_LABELS:
            raise guards.HybridQConfigError(f"{path} has an unregistered split label")
        frames = int(entry["frames"])
        if frames <= 0:
            raise guards.HybridQConfigError(f"{path} declares {frames} frames")
        # Split purity is required of every admitted shard, from the manifest's
        # own per-split frame counts, before anything is deserialized.
        counts = {name: int(entry[f"{name}_frames"]) for name in contract.SPLIT_LABELS}
        if counts[label] != frames or sum(counts.values()) != frames:
            raise guards.HybridQConfigError(
                f"{path} is not split-pure: {counts} under label {label!r}"
            )
        frames_by_split[label] += frames
        if label == split:
            selected.append(dict(entry))
        else:
            withheld.append(path)

    if sum(frames_by_split.values()) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError(
            f"the manifest accounts for {sum(frames_by_split.values())} frames "
            f"!= {contract.TRAIN_TOTAL_FRAMES}"
        )
    if int(manifest["split"]["total_frames"]) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError("teacher-cache manifest total frame-count drift")
    for label in contract.SPLIT_LABELS:
        if int(manifest["split"][f"{label}_frames"]) != frames_by_split[label]:
            raise guards.HybridQConfigError(
                f"manifest {label} frame count disagrees with its shard entries"
            )

    expected_ids = tuple(
        partition.fit_sample_ids if split == "fit" else partition.holdout_sample_ids
    )
    other_ids = frozenset(
        partition.holdout_sample_ids if split == "fit" else partition.fit_sample_ids
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise guards.HybridQConfigError(f"the registered {split} ids are not unique")
    if set(expected_ids) & other_ids:
        raise guards.HybridQConfigError("the registered split partitions overlap")
    # Exact sample-ID coverage of both splits, checked against the manifest
    # digests without opening a shard of either split.
    for label, ids in (
        ("fit", partition.fit_sample_ids),
        ("holdout", partition.holdout_sample_ids),
    ):
        if manifest["split"][f"{label}_sample_id_sha256"] != contract.sample_id_digest(ids):
            raise guards.HybridQConfigError(
                f"the manifest {label} sample-ID digest disagrees with the "
                "registered partition"
            )
    if frames_by_split[split] != len(expected_ids):
        raise guards.HybridQConfigError(
            f"the manifest holds {frames_by_split[split]} {split} frames, the "
            f"registered partition has {len(expected_ids)}"
        )
    if frames_by_split[other_split] != len(other_ids):
        raise guards.HybridQConfigError(
            f"the manifest holds {frames_by_split[other_split]} {other_split} "
            f"frames, the registered partition has {len(other_ids)}"
        )
    if not selected:
        raise guards.HybridQConfigError(f"the teacher cache holds no {split} shard")

    selected.sort(key=lambda entry: int(entry["shard_index"]))
    return TeacherShardPlan(
        split=str(split),
        other_split=str(other_split),
        manifest_sha256=manifest_sha256,
        selected=tuple(selected),
        withheld=tuple(withheld),
        expected_ids=expected_ids,
        other_split_ids=other_ids,
        selected_frames=frames_by_split[split],
        withheld_frames=frames_by_split[other_split],
    )


def load_ae_teacher_store(
    partition: teacher_cache.SplitPartition,
    split: str,
    *,
    shard_loader: Any = None,
) -> AeTeacherStore:
    """Deserialize exactly one split's teacher shards and nothing else.

    Training calls this with `split="fit"` and never deserializes a holdout
    shard; holdout selection calls it with `split="holdout"` and never
    deserializes a fit shard. `shard_loader` exists so a test can observe which
    paths are opened; production always uses `teacher_cache.load_shard`.
    """
    cache_root = phase5_common.teacher_cache_root()
    plan = plan_teacher_shards(cache_root, partition, split)
    load = teacher_cache.load_shard if shard_loader is None else shard_loader

    blocks: list[torch.Tensor] = []
    index: dict[str, int] = {}
    valid: list[tuple[str, ...]] = []
    excluded: list[dict[str, str]] = []
    loaded: list[str] = []
    for entry in plan.selected:
        path = cache_root / entry["path"]
        # Hash first, deserialize second.
        if path.stat().st_size != int(entry["bytes"]):
            raise guards.HybridQPayloadError(f"{entry['path']} byte size drift")
        if sha256_file(path) != entry["sha256"]:
            raise guards.HybridQPayloadError(f"{entry['path']} sha256 drift")

        payload = load(path)
        if payload["schema"] != teacher_cache.SHARD_SCHEMA:
            raise guards.HybridQConfigError(f"{entry['path']} schema drift")
        if payload["perception_checkpoint_sha256"] != contract.FROZEN_CHECKPOINT_SHA256:
            raise guards.HybridQConfigError(f"{entry['path']} checkpoint binding drift")
        labels = {str(value) for value in payload["splits"]}
        if str(payload["split"]) != split or labels != {split}:
            raise guards.HybridQOwnershipError(
                f"{entry['path']} is not a pure {split} shard: {sorted(labels)}"
            )
        frames = int(entry["frames"])
        if len(payload["sample_ids"]) != frames or len(payload["splits"]) != frames:
            raise guards.HybridQPayloadError(f"{entry['path']} identifier count drift")
        maps = payload["importance"]
        if maps.dtype is not torch.float32:
            raise guards.HybridQPayloadError(f"{entry['path']} maps are not FP32")
        if tuple(maps.shape) != (frames,) + contract.SPLIT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError(f"{entry['path']} map shape drift")

        for offset, sample_id in enumerate(payload["sample_ids"]):
            key = str(sample_id)
            if key in index:
                raise guards.HybridQConfigError(f"duplicate cached frame {key}")
            if key in plan.other_split_ids:
                raise guards.HybridQOwnershipError(
                    f"{key} is a registered {plan.other_split} frame but was "
                    f"cached in a {split} shard"
                )
            if contract.split_for_episode(str(payload["episode_ids"][offset])) != split:
                raise guards.HybridQConfigError(
                    f"{key} split label disagrees with its episode"
                )
            index[key] = len(index)
            groups = tuple(str(name) for name in payload["valid_groups"][offset])
            if len(groups) < ae_contract.AE_MIN_VALID_TASK_GROUPS:
                raise guards.HybridQConfigError(
                    f"{key} carries {len(groups)} valid teacher groups, below "
                    f"{ae_contract.AE_MIN_VALID_TASK_GROUPS}"
                )
            valid.append(groups)
            excluded.append(
                {
                    str(name): str(reason)
                    for name, reason in dict(payload["excluded_groups"][offset]).items()
                }
            )
        blocks.append(maps.contiguous())
        loaded.append(str(entry["path"]))
        del payload, maps

    if len(index) != plan.selected_frames:
        raise guards.HybridQConfigError(
            f"the opened {split} shards carry {len(index)} frames != the "
            f"manifest's {plan.selected_frames}"
        )
    if set(index) != set(plan.expected_ids):
        raise guards.HybridQConfigError(
            f"cached {split} frames disagree with the registered partition"
        )
    maps = torch.cat(blocks, dim=0)
    del blocks
    if int(maps.shape[0]) != len(index):
        raise guards.HybridQConfigError("teacher store map count drift")
    if not torch.isfinite(maps).all():
        raise guards.HybridQNumericalError("a cached teacher map is non-finite")
    if bool((maps < 0).any()):
        raise guards.HybridQNumericalError("a cached teacher map is negative")
    return AeTeacherStore(
        split=str(split),
        maps=maps,
        index=index,
        valid_groups=tuple(valid),
        excluded_groups=tuple(excluded),
        other_split_ids=plan.other_split_ids,
        manifest_sha256=plan.manifest_sha256,
        loaded_shards=tuple(loaded),
        withheld_shards=plan.withheld,
    )


# ---------------------------------------------------------------------------
# Atomic exact-resume checkpoints
# ---------------------------------------------------------------------------


def rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _atomic_write(path: Path, write: Callable[[Any], None]) -> str:
    """Write beside the destination, fsync, rename into place, fsync the directory.

    After this returns, `path` either does not exist or holds the complete new
    contents; a torn or half-written file is not a reachable state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    with staging.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    return _atomic_write(path, lambda stream: torch.save(dict(payload), stream))


def atomic_write_json(document: Mapping[str, Any], path: Path) -> str:
    """A JSON document written with exactly a checkpoint's durability."""
    raw = (json.dumps(dict(document), indent=2, default=str) + "\n").encode("utf-8")
    return _atomic_write(path, lambda stream: stream.write(raw))


def binding_fields(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Every saved source binding, all of which a load must reproduce exactly."""
    return {
        "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
        "stable_ranker_sha256": binding["stable_epoch4_ranker"]["sha256"],
        "locked_config_sha256": binding["hybrid_q_locked_config"]["sha256"],
        "teacher_cache_manifest_sha256": binding["teacher_cache_manifest"]["sha256"],
        "hybrid_q_source_sha256": binding["hybrid_q_source_sha256"],
        "ae_package_source_sha256": binding["ae_package_source_sha256"],
    }


def require_bindings(
    payload: Mapping[str, Any], binding: Mapping[str, Any], *, what: str
) -> dict[str, Any]:
    """Fail closed unless every saved binding field matches the live one.

    Every field `binding_fields()` writes is enforced, including the per-file
    `hybrid_q_source_sha256` and `ae_package_source_sha256` maps: a source,
    mapping or configuration edit between writing a checkpoint and loading it is
    drift, not a detail to skip.
    """
    expected = binding_fields(binding)
    for name, value in expected.items():
        if name not in payload:
            raise guards.HybridQConfigError(f"{what} does not carry {name}")
        if payload[name] != value:
            raise guards.HybridQConfigError(f"{what} {name} drift")
    return expected


def save_recovery(
    path: Path,
    *,
    epoch: int,
    autoencoder: SplitFeatureAE,
    optimizer: torch.optim.Optimizer,
    global_update_index: int,
    stage_b_position: int,
    order_identity: Mapping[str, Any],
    summary: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> str:
    """Everything an exact resume needs, written atomically."""
    require_training_epoch(epoch)
    payload = {
        "schema": AE_RECOVERY_SCHEMA,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "next_epoch_shuffle_seed": (
            contract.epoch_shuffle_seed(epoch + 1)
            if epoch < AE_TRAINING_EPOCHS
            else None
        ),
        "stage": stage_for_epoch(epoch),
        "next_stage": (
            stage_for_epoch(epoch + 1) if epoch < AE_TRAINING_EPOCHS else None
        ),
        "global_update_index": int(global_update_index),
        "stage_b_cycle_position": int(stage_b_position),
        "next_stage_b_q": stage_b_q_at(int(stage_b_position)),
        "autoencoder": autoencoder.state_dict(),
        "bottleneck": autoencoder.bottleneck,
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "order_identity": dict(order_identity),
        "epoch_summary": dict(summary),
        "configuration": training_configuration(),
        **binding_fields(binding),
    }
    return _atomic_torch_save(payload, path)


def save_candidate(
    path: Path,
    *,
    epoch: int,
    autoencoder: SplitFeatureAE,
    global_update_index: int,
    stage_b_position: int,
    binding: Mapping[str, Any],
) -> str:
    """One selection candidate: AE weights plus the identity to bind them."""
    if int(epoch) not in AE_CANDIDATE_EPOCHS:
        raise guards.HybridQConfigError(
            f"epoch {epoch} is not a registered candidate epoch {AE_CANDIDATE_EPOCHS}"
        )
    payload = {
        "schema": AE_CANDIDATE_SCHEMA,
        "epoch": int(epoch),
        "stage": stage_for_epoch(epoch),
        "autoencoder": autoencoder.state_dict(),
        "bottleneck": autoencoder.bottleneck,
        "family_id": autoencoder.family_id,
        "parameter_count": autoencoder.parameter_count(),
        "global_update_index": int(global_update_index),
        "stage_b_cycle_position": int(stage_b_position),
        "configuration": training_configuration(),
        **binding_fields(binding),
    }
    return _atomic_torch_save(payload, path)


def candidate_filename(epoch: int) -> str:
    return f"ae128_epoch_{int(epoch):02d}.pt"


def load_candidate(
    path: Path, epoch: int, device: torch.device, binding: Mapping[str, Any]
) -> tuple[SplitFeatureAE, dict[str, Any]]:
    """Load one candidate AE128 for evaluation, frozen and in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != AE_CANDIDATE_SCHEMA:
        raise guards.HybridQConfigError(f"{path.name} candidate schema drift")
    if int(payload["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(f"{path.name} epoch drift")
    if int(payload["bottleneck"]) != AE_TRAINING_BOTTLENECK:
        raise guards.HybridQConfigError(f"{path.name} bottleneck drift")
    require_bindings(payload, binding, what=path.name)
    autoencoder = build_split_feature_ae(AE_TRAINING_BOTTLENECK)
    autoencoder.load_state_dict(payload["autoencoder"])
    if autoencoder.parameter_count() != int(payload["parameter_count"]):
        raise guards.HybridQConfigError(f"{path.name} parameter-count drift")
    autoencoder = autoencoder.to(device)
    freeze(autoencoder)
    guards.require_module_parameters_finite(autoencoder, f"candidate {path.name}")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("autoencoder", "configuration")
    }
    del payload
    return autoencoder, metadata


def load_noae_holdout_reference() -> dict[float, dict[str, Any]]:
    """The frozen noAE same-q holdout rows this phase compares against.

    q=0 is the Phase-5 ranker-free baseline; q>0 are the Phase-5 rows produced by
    exactly the bound stable epoch-4 ranker, which is the same ranker every AE
    pass uses. Nothing is recomputed here.
    """
    root = contract.repository_root()
    path = (root / NOAE_HOLDOUT_RELPATH).resolve(strict=True)
    if sha256_file(path) != NOAE_HOLDOUT_SHA256:
        raise guards.HybridQConfigError("frozen noAE holdout reference sha256 drift")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["schema"] != "splitfusion_fcos_hybrid_q_phase5_holdout_v1":
        raise guards.HybridQConfigError("noAE holdout reference schema drift")
    if int(document["scope"]["holdout_frames"]) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("noAE holdout reference frame-count drift")
    if bool(document["scope"]["validation_or_test_accessed"]):
        raise guards.HybridQConfigError("noAE holdout reference reports reserved access")

    rows: dict[float, dict[str, Any]] = {}
    baseline = document["q0_baseline"]
    if bool(baseline["ranker_invoked"]):
        raise guards.HybridQConfigError("the noAE q=0 baseline invoked a ranker")
    rows[continuous_q.quantize_q(0.0).wire_q] = {
        "q": 0.0,
        "source": "phase5_q0_baseline",
        "ranker_epoch": None,
        "metrics": dict(baseline["metrics"]),
    }
    for row in document["checkpoint_q_evaluations"]:
        if int(row["epoch"]) != NOAE_REFERENCE_RANKER_EPOCH:
            continue
        plan = continuous_q.quantize_q(float(row["q"]))
        if plan.wire_q in rows:
            raise guards.HybridQConfigError(
                f"the noAE reference carries q={plan.wire_q!r} twice"
            )
        rows[plan.wire_q] = {
            "q": plan.wire_q,
            "source": "phase5_stable_epoch4_ranker",
            "ranker_epoch": NOAE_REFERENCE_RANKER_EPOCH,
            "metrics": dict(row["metrics"]),
        }
    wanted = {continuous_q.quantize_q(float(q)).wire_q for q in AE_HOLDOUT_Q_VALUES}
    if set(rows) != wanted:
        raise guards.HybridQConfigError(
            f"noAE reference covers {sorted(rows)}, this phase needs {sorted(wanted)}"
        )
    for row in rows.values():
        if set(row["metrics"]) != set(contract.PROTECTED_METRICS):
            raise guards.HybridQConfigError("noAE reference protected metric set drift")
    return rows


__all__ = [
    "AeTeacherStore",
    "TeacherShardPlan",
    "atomic_write_json",
    "bind_frozen_inputs",
    "binding_fields",
    "build_ae",
    "build_ae_optimizer",
    "freeze",
    "load_ae_teacher_store",
    "load_candidate",
    "load_noae_holdout_reference",
    "load_stable_ranker",
    "plan_teacher_shards",
    "require_bindings",
    "state_hashes",
    "training_configuration",
]
