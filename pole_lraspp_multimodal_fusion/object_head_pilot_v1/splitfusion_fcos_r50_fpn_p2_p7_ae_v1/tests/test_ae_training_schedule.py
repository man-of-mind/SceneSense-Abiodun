"""Focused CPU checks over the Phase-9C schedule, ranking rule and bookkeeping.

No real checkpoint, dataset, teacher-cache shard or CUDA context is touched, and
nothing is trained, inferred or evaluated. Every check is arithmetic over the
locked configuration, over synthetic ranking records, or over a synthetic cache
manifest / run directory built inside a temporary directory:

1. the Stage-A/Stage-B schedule and the balanced Stage-B q counts;
2. the preregistered checkpoint ranking rule;
3. split-pure teacher loading — asking for `fit` never deserializes a holdout
   shard, and the reverse;
4. exact-resume bookkeeping after epoch 8, plus source-hash drift rejection;
5. the global holdout reconstruction loss is independent of batch grouping.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

import numpy as np
import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards, teacher_cache
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import sha256_file
from .. import ae_contract, ae_holdout_selection, ae_loss, ae_training
from .. import ae_training_common as common
from ..ae_model import build_split_feature_ae


def synthetic_record(
    epoch: int, q: float, *, gates_passed: int, worst_normalized: float, loss: float
) -> dict:
    """Exactly the fields the ranking rule reads, and nothing else."""
    return {
        "epoch": epoch,
        "q": q,
        "gate_result": {
            "gates_passed": gates_passed,
            "all_passed": gates_passed == ae_holdout_selection.GATE_COUNT,
            "worst_normalized_degradation": worst_normalized,
        },
        "reconstruction": {"global_total_loss": loss},
    }


def synthetic_checkpoint(
    epoch: int, passes: tuple[int, int, int, int], worst: float, losses: tuple[float, ...]
) -> list[dict]:
    return [
        synthetic_record(
            epoch,
            float(q),
            gates_passed=passes[index],
            worst_normalized=worst,
            loss=losses[index],
        )
        for index, q in enumerate(common.AE_HOLDOUT_Q_VALUES)
    ]


# ---------------------------------------------------------------------------
# Synthetic fixtures: a teacher-cache manifest and a partially completed run
# ---------------------------------------------------------------------------


FIT_EPISODE = contract.TRAIN_FIT_EPISODES[0]
VALID_GROUPS = list(ae_contract.AE_TASK_GROUPS)


def synthetic_partition(
    fit_ids: tuple[str, ...], holdout_ids: tuple[str, ...]
) -> teacher_cache.SplitPartition:
    """A registered-shaped partition built directly, without a dataset."""
    return teacher_cache.SplitPartition(
        fit_indices=tuple(range(len(fit_ids))),
        holdout_indices=tuple(range(len(fit_ids), len(fit_ids) + len(holdout_ids))),
        fit_sample_ids=tuple(fit_ids),
        holdout_sample_ids=tuple(holdout_ids),
    )


def write_synthetic_cache(
    root: Path, partition: teacher_cache.SplitPartition, *, fit_shard_frames: int
) -> str:
    """A 66-entry manifest whose fit shards exist and whose holdout shards do not.

    The holdout shard files are deliberately never written: if anything tried to
    deserialize one it would fail outright, so "no holdout shard was opened" is
    enforced by the filesystem as well as by the spy.
    """
    shards = root / "shards"
    shards.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260902)
    entries: list[dict[str, Any]] = []
    cursor = 0

    fit_ids = list(partition.fit_sample_ids)
    for start in range(0, len(fit_ids), fit_shard_frames):
        block = fit_ids[start : start + fit_shard_frames]
        index = len(entries)
        maps = (
            torch.rand(
                (len(block),) + contract.SPLIT_SPATIAL_SHAPE, generator=generator
            )
            + 0.1
        ).float()
        payload = {
            "schema": teacher_cache.SHARD_SCHEMA,
            "shard_index": index,
            "frames": len(block),
            "split": "fit",
            "cache_index_start": cursor,
            "sample_ids": list(block),
            "episode_ids": [FIT_EPISODE] * len(block),
            "splits": ["fit"] * len(block),
            "importance": maps,
            "valid_groups": [list(VALID_GROUPS) for _ in block],
            "excluded_groups": [{} for _ in block],
            "perception_checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256,
        }
        path = shards / f"teacher_shard_{index:05d}.pt"
        torch.save(payload, path)
        entries.append(
            {
                "shard_index": index,
                "path": f"shards/{path.name}",
                "split": "fit",
                "frames": len(block),
                "fit_frames": len(block),
                "holdout_frames": 0,
                "cache_index_start": cursor,
                "cache_index_end": cursor + len(block),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        cursor += len(block)

    remaining = contract.TEACHER_CACHE_SHARD_COUNT - len(entries)
    holdout_total = len(partition.holdout_sample_ids)
    per_shard = [holdout_total // remaining] * remaining
    for position in range(holdout_total % remaining):
        per_shard[position] += 1
    for frames in per_shard:
        index = len(entries)
        entries.append(
            {
                "shard_index": index,
                "path": f"shards/teacher_shard_{index:05d}.pt",
                "split": "holdout",
                "frames": frames,
                "fit_frames": 0,
                "holdout_frames": frames,
                "cache_index_start": cursor,
                "cache_index_end": cursor + frames,
                "bytes": 0,
                "sha256": "0" * 64,
            }
        )
        cursor += frames

    manifest = {
        "schema": teacher_cache.MANIFEST_SCHEMA,
        "terminal": "HYBRID_Q_PHASE4_TEACHER_CACHE_COMPLETE",
        "perception_binding": {"checkpoint_sha256": contract.FROZEN_CHECKPOINT_SHA256},
        "split": {
            "fit_frames": len(partition.fit_sample_ids),
            "holdout_frames": holdout_total,
            "total_frames": contract.TRAIN_TOTAL_FRAMES,
            "fit_sample_id_sha256": contract.sample_id_digest(partition.fit_sample_ids),
            "holdout_sample_id_sha256": contract.sample_id_digest(
                partition.holdout_sample_ids
            ),
            "validation_or_test_frames": 0,
        },
        "shards": {"count": len(entries), "entries": entries},
    }
    return write_manifest(root, manifest)


def write_manifest(root: Path, manifest: dict[str, Any]) -> str:
    path = root / "teacher_cache_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def synthetic_binding(*, ae_source: str = "a" * 64, hybrid_source: str = "b" * 64) -> dict:
    """Every field `binding_fields()` reads, with no file access at all."""
    return {
        "stable_epoch4_ranker": {"sha256": "c" * 64},
        "hybrid_q_locked_config": {"sha256": "d" * 64},
        "teacher_cache_manifest": {"sha256": "e" * 64},
        "hybrid_q_source_sha256": {"contract.py": hybrid_source},
        "ae_package_source_sha256": {"ae_model.py": ae_source},
    }


class StubTrainRows:
    """Just the `rows` attribute `order_identity` reads."""

    def __init__(self, sample_ids: Sequence[str]) -> None:
        self.rows = [{"sample_id": sample_id} for sample_id in sample_ids]


def registered_shape_partition() -> tuple[teacher_cache.SplitPartition, StubTrainRows]:
    """A partition of the registered sizes, so `epoch_order` accepts it."""
    fit_ids = tuple(f"fit_{index:06d}" for index in range(contract.TRAIN_FIT_FRAMES))
    holdout_ids = tuple(
        f"hold_{index:06d}" for index in range(contract.TRAIN_HOLDOUT_FRAMES)
    )
    return synthetic_partition(fit_ids, holdout_ids), StubTrainRows(fit_ids + holdout_ids)


def cpu_rng_state() -> dict[str, Any]:
    """`common.rng_state` without the CUDA branch, so no context is created."""
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def write_partial_run(
    output: Path, partition: teacher_cache.SplitPartition, dataset: Any, binding: dict,
    *, through_epoch: int, uncommitted_epoch: bool = False,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Write epochs 1..`through_epoch` in the trainer's exact commit order.

    Per epoch: the candidate checkpoint, then `epoch_summaries.json`, then the
    recovery checkpoint last. With `uncommitted_epoch=True` the next epoch's
    candidate and summary are also written but its recovery checkpoint is not —
    exactly what an interruption between those two writes leaves on disk.
    """
    recovery_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    recovery_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    autoencoder = build_split_feature_ae(common.AE_TRAINING_BOTTLENECK)
    optimizer = common.build_ae_optimizer(
        autoencoder,
        lr=common.learning_rate_for_stage(common.AE_STAGE_A),
        frozen_modules=(),
    )
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    last = through_epoch + 1 if uncommitted_epoch else through_epoch
    with mock.patch.object(common, "rng_state", cpu_rng_state):
        for epoch in range(1, last + 1):
            identity = ae_training.order_identity(partition, epoch, dataset)
            updates = epoch * common.batches_per_epoch()
            position = ae_training.expected_stage_b_position(epoch)
            summary = {
                "epoch": epoch,
                "stage": common.stage_for_epoch(epoch),
                "epoch_sample_id_sha256": identity["sample_id_sha256"],
                "global_update_index": updates,
                "stage_b_cycle_position": position,
            }
            if epoch in common.AE_CANDIDATE_EPOCHS:
                candidate = common.candidate_filename(epoch)
                candidate_hashes[candidate] = common.save_candidate(
                    checkpoint_dir / candidate,
                    epoch=epoch,
                    autoencoder=autoencoder,
                    global_update_index=updates,
                    stage_b_position=position,
                    binding=binding,
                )
                summary["candidate_checkpoint"] = candidate
                summary["candidate_checkpoint_sha256"] = candidate_hashes[candidate]
            summaries.append(summary)
            ae_training.write_epoch_summaries(output, summaries)
            if epoch > through_epoch:
                # The interruption: this epoch is never declared complete.
                break
            name = ae_training.recovery_filename(epoch)
            recovery_hashes[name] = common.save_recovery(
                recovery_dir / name,
                epoch=epoch,
                autoencoder=autoencoder,
                optimizer=optimizer,
                global_update_index=updates,
                stage_b_position=position,
                order_identity=identity,
                summary=summary,
                binding=binding,
            )
    return recovery_hashes, candidate_hashes, summaries


class AeTrainingScheduleChecks(unittest.TestCase):
    def test_stage_schedule_and_balanced_stage_b_q_counts(self) -> None:
        # Stage boundaries and the two learning rates.
        for epoch in range(1, common.AE_STAGE_A_EPOCHS + 1):
            self.assertEqual(common.stage_for_epoch(epoch), common.AE_STAGE_A)
        for epoch in range(common.AE_STAGE_A_EPOCHS + 1, common.AE_TRAINING_EPOCHS + 1):
            self.assertEqual(common.stage_for_epoch(epoch), common.AE_STAGE_B)
        self.assertEqual(common.learning_rate_for_stage(common.AE_STAGE_A), 1e-3)
        self.assertEqual(common.learning_rate_for_stage(common.AE_STAGE_B), 3e-4)
        for outside in (0, common.AE_TRAINING_EPOCHS + 1):
            with self.assertRaises(guards.HybridQConfigError):
                common.stage_for_epoch(outside)

        # Every fit frame exactly once per epoch, final short batch retained.
        self.assertFalse(common.AE_DROP_LAST)
        self.assertEqual(common.AE_BATCH_SIZE, 16)
        self.assertEqual(common.batches_per_epoch(), 847)
        self.assertEqual(
            common.batches_per_epoch() * common.AE_BATCH_SIZE - contract.TRAIN_FIT_FRAMES,
            9,  # the final batch carries 7 of 16 frames
        )
        self.assertEqual(common.stage_b_updates_total(), 6776)

        # Walk the schedule exactly as the trainer does.
        position = 0
        updates = 0
        stage_a_q: set[float] = set()
        stage_b_counts: dict[str, int] = {
            f"{float(q):.2f}": 0 for q in common.AE_STAGE_B_Q_CYCLE
        }
        first_q_of_epoch: dict[int, float] = {}
        for epoch in range(1, common.AE_TRAINING_EPOCHS + 1):
            stage = common.stage_for_epoch(epoch)
            for batch in range(common.batches_per_epoch()):
                if stage == common.AE_STAGE_A:
                    q = float(common.AE_STAGE_A_Q)
                    stage_a_q.add(q)
                else:
                    q = common.stage_b_q_at(position)
                    stage_b_counts[f"{q:.2f}"] += 1
                    position += 1
                if batch == 0:
                    first_q_of_epoch[epoch] = q
                # The trainer admits every scheduled q through this guard.
                ae_loss.require_optimization_q(q)
                updates += 1

        self.assertEqual(updates, common.AE_TRAINING_EPOCHS * 847)
        self.assertEqual(stage_a_q, {0.00})
        self.assertEqual(position, 6776)
        self.assertEqual(sum(stage_b_counts.values()), 6776)
        self.assertEqual(
            stage_b_counts, {"0.00": 1694, "0.30": 1694, "0.50": 1694, "0.70": 1694}
        )
        self.assertEqual(common.require_balanced_stage_b(), stage_b_counts)

        # The cycle carries across epoch boundaries: 847 is not a multiple of
        # four, so a per-epoch restart would put q=0.00 first in every Stage-B
        # epoch and unbalance the totals.
        starts = [first_q_of_epoch[epoch] for epoch in range(5, 13)]
        self.assertNotEqual(set(starts), {0.00})
        self.assertEqual(starts[0], 0.00)
        self.assertEqual(starts[1], common.AE_STAGE_B_Q_CYCLE[847 % 4])

        # The two stress values are never scheduled and are refused outright.
        self.assertEqual(tuple(common.AE_EXCLUDED_Q), (0.90, 0.98))
        for excluded in common.AE_EXCLUDED_Q:
            self.assertNotIn(f"{float(excluded):.2f}", stage_b_counts)
            with self.assertRaises(guards.HybridQConfigError):
                ae_loss.require_optimization_q(float(excluded))
        self.assertEqual(
            tuple(common.AE_STAGE_B_Q_CYCLE), tuple(ae_contract.AE_STAGE_B_Q_CYCLE)
        )

    def test_preregistered_checkpoint_ranking_is_deterministic(self) -> None:
        full = ae_holdout_selection.GATE_COUNT
        losses = (0.5, 0.5, 0.5, 0.5)

        def decide(records: list[dict]) -> dict:
            shuffled = list(records)
            random.Random(20260829).shuffle(shuffled)
            decision = ae_holdout_selection.rank_checkpoints(shuffled)
            # Order of the input must not matter.
            self.assertEqual(
                decision["ranking"],
                ae_holdout_selection.rank_checkpoints(records)["ranking"],
            )
            return decision

        # 1) the worst same-q gate count wins, even against a larger total.
        decision = decide(
            synthetic_checkpoint(4, (10, 10, 10, 10), 0.5, losses)
            + synthetic_checkpoint(8, (full, full, full, 9), 0.1, losses)
        )
        self.assertEqual(decision["selected_epoch"], 4)
        self.assertEqual(decision["decided_at_criterion"], "min_same_q_gates_passed")
        self.assertEqual(decision["selected"]["min_same_q_gates_passed"], 10)
        self.assertEqual(decision["selected"]["total_gates_passed"], 40)

        # 2) equal minima: the larger total wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.1, losses)
            + synthetic_checkpoint(8, (9, 11, 11, 11), 0.9, losses)
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(decision["decided_at_criterion"], "total_gates_passed")

        # 3) equal minima and totals: the smaller worst normalized degradation wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.90, losses)
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, losses)
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(
            decision["decided_at_criterion"], "worst_normalized_degradation"
        )

        # 4) equal through three: the smaller mean holdout reconstruction loss wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, (0.9, 0.9, 0.9, 0.9))
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, (0.4, 0.4, 0.4, 0.4))
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(
            decision["decided_at_criterion"], "mean_holdout_reconstruction_loss"
        )
        self.assertAlmostEqual(
            decision["selected"]["mean_holdout_reconstruction_loss"], 0.4
        )

        # 5) fully tied: the earlier epoch wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, losses)
            + synthetic_checkpoint(12, (9, 9, 9, 9), 0.25, losses)
        )
        self.assertEqual(decision["selected_epoch"], 4)
        self.assertEqual(decision["decided_at_criterion"], "epoch")
        self.assertEqual([row["epoch"] for row in decision["ranking"]], [4, 8, 12])

        # Selecting a checkpoint is never a service-ready claim.
        self.assertFalse(decision["selection_is_a_service_ready_claim"])
        self.assertFalse(decision["selected"]["all_gates_passed_at_every_q"])
        passing = decide(
            synthetic_checkpoint(4, (full, full, full, full), -0.2, losses)
        )
        self.assertTrue(passing["selected"]["all_gates_passed_at_every_q"])
        self.assertFalse(passing["selection_is_a_service_ready_claim"])

        # An incomplete q sweep for a candidate fails closed.
        incomplete = synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)[:3]
        with self.assertRaises(guards.HybridQConfigError):
            ae_holdout_selection.rank_checkpoints(incomplete)
        duplicated = synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)
        duplicated.append(duplicated[0])
        with self.assertRaises(guards.HybridQConfigError):
            ae_holdout_selection.rank_checkpoints(duplicated)


class AeSplitPureTeacherLoadingChecks(unittest.TestCase):
    """Requesting one split must never deserialize the other split's shards."""

    def test_fit_never_invokes_the_shard_loader_on_a_holdout_entry(self) -> None:
        fit_ids = tuple(f"fit_{index:05d}" for index in range(32))
        holdout_ids = tuple(
            f"hold_{index:05d}" for index in range(contract.TRAIN_TOTAL_FRAMES - 32)
        )
        partition = synthetic_partition(fit_ids, holdout_ids)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            digest = write_synthetic_cache(root, partition, fit_shard_frames=16)
            opened: list[str] = []
            real_loader = teacher_cache.load_shard

            def spy(path: Path) -> dict:
                opened.append(str(path))
                return real_loader(path)

            with mock.patch.object(
                common.phase5_common, "teacher_cache_root", lambda: root
            ), mock.patch.object(
                common.contract, "TEACHER_CACHE_MANIFEST_SHA256", digest
            ), mock.patch.object(
                common.teacher_cache, "load_shard", spy
            ):
                # The production call path: no loader is injected here.
                store = common.load_ae_teacher_store(partition, "fit")
                fit_plan = common.plan_teacher_shards(root, partition, "fit")
                holdout_plan = common.plan_teacher_shards(root, partition, "holdout")

            # Exactly the two fit shards were deserialized, in shard order.
            self.assertEqual(
                opened,
                [
                    str(root / "shards/teacher_shard_00000.pt"),
                    str(root / "shards/teacher_shard_00001.pt"),
                ],
            )
            holdout_paths = {
                str(root / entry["path"]) for entry in holdout_plan.selected
            }
            self.assertEqual(len(holdout_paths), 64)
            self.assertFalse(holdout_paths.intersection(opened))
            # Those files were never even created, so opening one would fail.
            self.assertFalse(any(Path(path).exists() for path in holdout_paths))

            self.assertEqual(store.split, "fit")
            self.assertEqual(store.frames, len(fit_ids))
            self.assertEqual(set(store.index), set(fit_ids))
            self.assertEqual(store.other_split_ids, frozenset(holdout_ids))
            self.assertEqual(store.loaded_shards, tuple(fit_plan.selected[index]["path"] for index in range(2)))
            self.assertEqual(len(store.withheld_shards), 64)
            provenance = store.provenance()
            self.assertEqual(provenance["holdout_maps_loaded"], 0)
            self.assertEqual(provenance["holdout_shards_deserialized"], 0)
            self.assertEqual(provenance["holdout_ids_excluded"], len(holdout_ids))
            self.assertEqual(provenance["shards_opened"], 2)
            self.assertEqual(provenance["teacher_cache_manifest_sha256"], digest)

            # Each plan admits its own split only, in both directions.
            self.assertEqual({entry["split"] for entry in fit_plan.selected}, {"fit"})
            self.assertEqual(
                {entry["split"] for entry in holdout_plan.selected}, {"holdout"}
            )
            self.assertEqual(fit_plan.other_split_ids, frozenset(holdout_ids))
            self.assertEqual(holdout_plan.other_split_ids, frozenset(fit_ids))
            self.assertEqual(fit_plan.selected_frames, len(fit_ids))
            self.assertEqual(holdout_plan.selected_frames, len(holdout_ids))

            # A holdout id cannot be served out of a fit store.
            with self.assertRaises(guards.HybridQOwnershipError):
                store.record(holdout_ids[0])

    def test_a_shard_that_is_not_split_pure_is_refused_before_any_load(self) -> None:
        fit_ids = tuple(f"fit_{index:05d}" for index in range(32))
        holdout_ids = tuple(
            f"hold_{index:05d}" for index in range(contract.TRAIN_TOTAL_FRAMES - 32)
        )
        partition = synthetic_partition(fit_ids, holdout_ids)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_synthetic_cache(root, partition, fit_shard_frames=16)
            manifest = json.loads(
                (root / "teacher_cache_manifest.json").read_text(encoding="utf-8")
            )
            # One fit-labelled shard now claims a holdout frame as well.
            entry = manifest["shards"]["entries"][0]
            entry["holdout_frames"] = 1
            entry["fit_frames"] = entry["frames"] - 1
            digest = write_manifest(root, manifest)
            opened: list[str] = []

            with mock.patch.object(
                common.phase5_common, "teacher_cache_root", lambda: root
            ), mock.patch.object(
                common.contract, "TEACHER_CACHE_MANIFEST_SHA256", digest
            ), mock.patch.object(
                common.teacher_cache, "load_shard", lambda path: opened.append(str(path))
            ):
                with self.assertRaises(guards.HybridQConfigError):
                    common.load_ae_teacher_store(partition, "fit")
            self.assertEqual(opened, [])

    def test_manifest_sample_id_coverage_must_match_the_registered_partition(self) -> None:
        fit_ids = tuple(f"fit_{index:05d}" for index in range(32))
        holdout_ids = tuple(
            f"hold_{index:05d}" for index in range(contract.TRAIN_TOTAL_FRAMES - 32)
        )
        partition = synthetic_partition(fit_ids, holdout_ids)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_synthetic_cache(root, partition, fit_shard_frames=16)
            manifest = json.loads(
                (root / "teacher_cache_manifest.json").read_text(encoding="utf-8")
            )
            manifest["split"]["holdout_sample_id_sha256"] = "0" * 64
            digest = write_manifest(root, manifest)
            with mock.patch.object(
                common.phase5_common, "teacher_cache_root", lambda: root
            ), mock.patch.object(
                common.contract, "TEACHER_CACHE_MANIFEST_SHA256", digest
            ):
                with self.assertRaises(guards.HybridQConfigError):
                    common.plan_teacher_shards(root, partition, "fit")


class AeExactResumeChecks(unittest.TestCase):
    """Resume must rebuild the whole record and refuse anything it cannot verify."""

    def test_resume_after_epoch_eight_reconstructs_recovery_and_candidate_hashes(
        self,
    ) -> None:
        partition, dataset = registered_shape_partition()
        binding = synthetic_binding()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "run"
            written_recovery, written_candidates, _ = write_partial_run(
                output, partition, dataset, binding, through_epoch=8
            )
            self.assertEqual(len(written_recovery), 8)
            # Epochs 4 and 8 are the candidate epochs reached so far; 12 is not.
            self.assertEqual(
                set(written_candidates), {"ae128_epoch_04.pt", "ae128_epoch_08.pt"}
            )

            autoencoder = build_split_feature_ae(common.AE_TRAINING_BOTTLENECK)
            optimizer = common.build_ae_optimizer(
                autoencoder,
                lr=common.learning_rate_for_stage(common.AE_STAGE_A),
                frozen_modules=(),
            )
            state = ae_training.TrainingState()
            (
                completed,
                epoch_summaries,
                recovery_hashes,
                candidate_hashes,
            ) = ae_training.restore_completed_epochs(
                output=output,
                recovery_dir=output / "recovery",
                checkpoint_dir=output / "checkpoints",
                autoencoder=autoencoder,
                optimizer=optimizer,
                state=state,
                binding=binding,
                partition=partition,
                dataset=dataset,
                device=torch.device("cpu"),
            )

            self.assertEqual(completed, 8)
            # Counters continue rather than replay.
            self.assertEqual(state.global_update_index, 8 * common.batches_per_epoch())
            self.assertEqual(
                state.stage_b_position, 4 * common.batches_per_epoch()
            )
            self.assertEqual(
                [summary["epoch"] for summary in epoch_summaries], list(range(1, 9))
            )
            # Every already-completed file is back in the bookkeeping.
            self.assertEqual(recovery_hashes, written_recovery)
            self.assertEqual(
                set(candidate_hashes), {"ae128_epoch_04.pt", "ae128_epoch_08.pt"}
            )
            self.assertEqual(candidate_hashes, written_candidates)

            # Finishing epochs 9..12 then satisfies the trainer's final check.
            with mock.patch.object(common, "rng_state", cpu_rng_state):
                final = common.candidate_filename(12)
                candidate_hashes[final] = common.save_candidate(
                    output / "checkpoints" / final,
                    epoch=12,
                    autoencoder=autoencoder,
                    global_update_index=12 * common.batches_per_epoch(),
                    stage_b_position=common.stage_b_updates_total(),
                    binding=binding,
                )
            self.assertEqual(
                set(candidate_hashes),
                {common.candidate_filename(epoch) for epoch in common.AE_CANDIDATE_EPOCHS},
            )

    def test_resume_survives_both_interruption_windows(self) -> None:
        """A recovery checkpoint is the only durable declaration of an epoch.

        Window 1: the recovery checkpoint exists but the external summary file
        was never finished. Window 2: the candidate and the summary exist for the
        next epoch but its recovery checkpoint does not. Both must resume from
        the last verified recovery epoch, rebuild the canonical inventories from
        the recovery checkpoints, and replay nothing already completed.
        """
        partition, dataset = registered_shape_partition()
        binding = synthetic_binding()

        def resume(output: Path):
            autoencoder = build_split_feature_ae(common.AE_TRAINING_BOTTLENECK)
            state = ae_training.TrainingState()
            result = ae_training.restore_completed_epochs(
                output=output,
                recovery_dir=output / "recovery",
                checkpoint_dir=output / "checkpoints",
                autoencoder=autoencoder,
                optimizer=common.build_ae_optimizer(
                    autoencoder,
                    lr=common.learning_rate_for_stage(common.AE_STAGE_A),
                    frozen_modules=(),
                ),
                state=state,
                binding=binding,
                partition=partition,
                dataset=dataset,
                device=torch.device("cpu"),
            )
            return state, result

        def on_disk(output: Path) -> list[int]:
            document = json.loads(
                (output / "epoch_summaries.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["schema"], ae_training.SCHEMA)
            return [int(summary["epoch"]) for summary in document["epochs"]]

        with tempfile.TemporaryDirectory() as raw:
            # Window 1: epoch 8 is declared complete, the summary file is not.
            for damage in ("absent", "truncated"):
                output = Path(raw) / f"window1_{damage}"
                written_recovery, written_candidates, _ = write_partial_run(
                    output, partition, dataset, binding, through_epoch=8
                )
                summary_path = output / "epoch_summaries.json"
                if damage == "absent":
                    summary_path.unlink()
                else:
                    # A torn write: the file exists but does not parse.
                    text = summary_path.read_text(encoding="utf-8")
                    summary_path.write_text(text[: len(text) // 2], encoding="utf-8")

                state, (
                    completed,
                    summaries,
                    recovery_hashes,
                    candidate_hashes,
                ) = resume(output)

                self.assertEqual(completed, 8)
                self.assertEqual(
                    state.global_update_index, 8 * common.batches_per_epoch()
                )
                self.assertEqual(
                    state.stage_b_position, 4 * common.batches_per_epoch()
                )
                # Canonical record rebuilt from the recovery checkpoints alone.
                self.assertEqual(
                    [summary["epoch"] for summary in summaries], list(range(1, 9))
                )
                self.assertEqual(recovery_hashes, written_recovery)
                self.assertEqual(candidate_hashes, written_candidates)
                self.assertEqual(
                    set(candidate_hashes), {"ae128_epoch_04.pt", "ae128_epoch_08.pt"}
                )
                # And the external file is rewritten from it.
                self.assertEqual(on_disk(output), list(range(1, 9)))
                self.assertEqual(ae_training.stale_candidate_files(output / "checkpoints", 8), [])

            # Window 2: epoch 8's candidate and summary landed, its recovery
            # checkpoint did not, so epoch 7 is the last completed epoch.
            output = Path(raw) / "window2"
            written_recovery, written_candidates, written_summaries = write_partial_run(
                output, partition, dataset, binding, through_epoch=7,
                uncommitted_epoch=True,
            )
            self.assertEqual(sorted(written_recovery), [f"epoch_{e:02d}.pt" for e in range(1, 8)])
            self.assertEqual(
                set(written_candidates), {"ae128_epoch_04.pt", "ae128_epoch_08.pt"}
            )
            self.assertEqual(len(written_summaries), 8)
            self.assertEqual(on_disk(output), list(range(1, 9)))
            self.assertTrue(
                (output / "checkpoints" / common.candidate_filename(8)).is_file()
            )
            self.assertFalse(
                (output / "recovery" / ae_training.recovery_filename(8)).is_file()
            )

            state, (completed, summaries, recovery_hashes, candidate_hashes) = resume(
                output
            )

            self.assertEqual(completed, 7)
            self.assertEqual(state.global_update_index, 7 * common.batches_per_epoch())
            self.assertEqual(state.stage_b_position, 3 * common.batches_per_epoch())
            self.assertEqual(
                [summary["epoch"] for summary in summaries], list(range(1, 8))
            )
            self.assertEqual(recovery_hashes, written_recovery)
            # Epoch 8's candidate is from an epoch that never completed.
            self.assertEqual(set(candidate_hashes), {"ae128_epoch_04.pt"})
            self.assertEqual(
                ae_training.stale_candidate_files(output / "checkpoints", completed),
                ["ae128_epoch_08.pt"],
            )
            # The extra tail entry is discarded from the external record.
            self.assertEqual(on_disk(output), list(range(1, 8)))

    def test_resume_refuses_an_empty_or_new_directory_and_a_broken_record(self) -> None:
        partition, dataset = registered_shape_partition()
        binding = synthetic_binding()

        def resume(output: Path) -> None:
            autoencoder = build_split_feature_ae(common.AE_TRAINING_BOTTLENECK)
            ae_training.restore_completed_epochs(
                output=output,
                recovery_dir=output / "recovery",
                checkpoint_dir=output / "checkpoints",
                autoencoder=autoencoder,
                optimizer=common.build_ae_optimizer(
                    autoencoder,
                    lr=common.learning_rate_for_stage(common.AE_STAGE_A),
                    frozen_modules=(),
                ),
                state=ae_training.TrainingState(),
                binding=binding,
                partition=partition,
                dataset=dataset,
                device=torch.device("cpu"),
            )

        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "never_created"
            with self.assertRaises(guards.HybridQConfigError):
                resume(missing)

            empty = Path(raw) / "empty"
            (empty / "recovery").mkdir(parents=True)
            (empty / "checkpoints").mkdir()
            with self.assertRaises(guards.HybridQConfigError):
                resume(empty)

            # A gap in the recovery sequence is not a resumable run.
            gapped = Path(raw) / "gapped"
            write_partial_run(gapped, partition, dataset, binding, through_epoch=8)
            (gapped / "recovery" / ae_training.recovery_filename(5)).unlink()
            with self.assertRaises(guards.HybridQConfigError):
                resume(gapped)

            # A truncated summary file is tolerated (it is written before the
            # recovery checkpoint), but an external summary that *disagrees*
            # about a completed epoch is another run's file and is refused.
            foreign = Path(raw) / "foreign_summaries"
            write_partial_run(foreign, partition, dataset, binding, through_epoch=8)
            document = json.loads(
                (foreign / "epoch_summaries.json").read_text(encoding="utf-8")
            )
            document["epochs"][2]["epoch_sample_id_sha256"] = "0" * 64
            (foreign / "epoch_summaries.json").write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(guards.HybridQConfigError):
                resume(foreign)

            # A foreign schema is refused too.
            schema = Path(raw) / "foreign_schema"
            write_partial_run(schema, partition, dataset, binding, through_epoch=8)
            document = json.loads(
                (schema / "epoch_summaries.json").read_text(encoding="utf-8")
            )
            document["schema"] = "some_other_run_v1"
            (schema / "epoch_summaries.json").write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(guards.HybridQConfigError):
                resume(schema)

            # A missing candidate for a completed candidate epoch is refused.
            without = Path(raw) / "without_candidate"
            write_partial_run(without, partition, dataset, binding, through_epoch=8)
            (without / "checkpoints" / common.candidate_filename(4)).unlink()
            with self.assertRaises(guards.HybridQConfigError):
                resume(without)

    def test_every_saved_source_binding_is_enforced_on_load(self) -> None:
        partition, dataset = registered_shape_partition()
        binding = synthetic_binding()
        drifted = (
            synthetic_binding(ae_source="f" * 64),
            synthetic_binding(hybrid_source="9" * 64),
        )
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "run"
            write_partial_run(output, partition, dataset, binding, through_epoch=4)
            recovery = output / "recovery" / ae_training.recovery_filename(4)
            candidate = output / "checkpoints" / common.candidate_filename(4)
            identity = ae_training.order_identity(partition, 4, dataset)
            device = torch.device("cpu")

            # The unmodified binding is accepted by both loaders.
            ae_training.read_recovery(recovery, 4, binding, identity)
            common.load_candidate(candidate, 4, device, binding)

            # A drifted source-hash map is refused by both, not skipped.
            for candidate_binding in drifted:
                with self.assertRaises(guards.HybridQConfigError):
                    common.load_candidate(candidate, 4, device, candidate_binding)
                with self.assertRaises(guards.HybridQConfigError):
                    ae_training.read_recovery(recovery, 4, candidate_binding, identity)

            self.assertIn("hybrid_q_source_sha256", common.binding_fields(binding))
            self.assertIn("ae_package_source_sha256", common.binding_fields(binding))

            # The filename epoch must equal the embedded epoch.
            with self.assertRaises(guards.HybridQConfigError):
                ae_training.read_recovery(
                    recovery, 5, binding, ae_training.order_identity(partition, 5, dataset)
                )
            # And the sampler/order identity must match the epoch being resumed.
            with self.assertRaises(guards.HybridQConfigError):
                ae_training.read_recovery(
                    recovery, 4, binding, ae_training.order_identity(partition, 3, dataset)
                )


class AeHoldoutReconstructionTotalsChecks(unittest.TestCase):
    """Criterion 4 must not depend on how the holdout frames were batched."""

    def _frames(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(20260902)
        shape = (count, 3) + contract.SPLIT_SPATIAL_SHAPE
        target = torch.rand(shape, generator=generator) * 4.0 - 2.0
        estimate = target + 0.05 * torch.randn(shape, generator=generator)
        importance = (
            torch.rand((count,) + contract.SPLIT_SPATIAL_SHAPE, generator=generator) + 0.01
        )
        return target.float(), estimate.float(), importance.float()

    def _accumulate(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
        importance: torch.Tensor,
        groups: Sequence[int],
    ) -> dict[str, float]:
        totals = ae_holdout_selection.HoldoutReconstructionTotals()
        start = 0
        for size in groups:
            stop = start + size
            teacher = ae_loss.CachedTeacherBatch(
                importance=importance[start:stop].contiguous(),
                valid_groups=tuple(tuple(VALID_GROUPS) for _ in range(size)),
                excluded_groups=tuple({} for _ in range(size)),
            )
            totals.observe(target[start:stop], estimate[start:stop], teacher)
            start = stop
        self.assertEqual(start, int(target.shape[0]))
        return totals.totals()

    def test_global_loss_is_unchanged_under_regrouping(self) -> None:
        count = 12
        target, estimate, importance = self._frames(count)
        reference = self._accumulate(target, estimate, importance, (12,))
        for groups in ((8, 4), (5, 5, 2), (1,) * 12, (7, 1, 3, 1)):
            observed = self._accumulate(target, estimate, importance, groups)
            self.assertEqual(observed, reference, f"batch grouping {groups} moved the loss")

        self.assertEqual(reference["frames"], count)
        self.assertAlmostEqual(
            reference["global_total_loss"],
            reference["global_plain_reconstruction"]
            + reference["global_combined_importance_reconstruction"],
        )
        self.assertAlmostEqual(
            reference["global_plain_reconstruction"],
            reference["plain_squared_error_numerator"]
            / reference["plain_reference_energy_denominator"],
        )
        self.assertAlmostEqual(
            reference["global_combined_importance_reconstruction"],
            reference["combined_importance_numerator"]
            / reference["combined_importance_reference_energy_denominator"],
        )

    def test_a_single_batch_reproduces_the_committed_loss_exactly(self) -> None:
        """On one real 256-channel batch the totals are the committed loss."""
        generator = torch.Generator().manual_seed(20260903)
        shape = (2,) + contract.SPLIT_SHAPE
        target = (torch.rand(shape, generator=generator) * 4.0 - 2.0).float()
        estimate = (target + 0.05 * torch.randn(shape, generator=generator)).float()
        importance = (
            torch.rand((2,) + contract.SPLIT_SPATIAL_SHAPE, generator=generator) + 0.01
        ).float()
        teacher = ae_loss.CachedTeacherBatch(
            importance=importance,
            valid_groups=(tuple(VALID_GROUPS), tuple(VALID_GROUPS)),
            excluded_groups=({}, {}),
        )
        committed = ae_loss.task_aware_reconstruction_loss(target, estimate, teacher)
        totals = ae_holdout_selection.HoldoutReconstructionTotals()
        totals.observe(target, estimate, teacher)
        observed = totals.totals()
        self.assertAlmostEqual(
            observed["global_plain_reconstruction"], float(committed.plain), places=6
        )
        self.assertAlmostEqual(
            observed["global_combined_importance_reconstruction"],
            float(committed.combined_importance),
            places=6,
        )
        self.assertAlmostEqual(
            observed["global_total_loss"], float(committed.total), places=6
        )

    def test_the_unweighted_mean_of_batch_ratios_is_the_number_that_moves(self) -> None:
        """The diagnostic the ranking no longer uses does depend on batching."""
        count = 12
        target, estimate, importance = self._frames(count)

        def batch_ratio_mean(groups: Sequence[int]) -> float:
            # One accumulator per batch reproduces exactly the committed loss for
            # that batch; averaging those ratios is the discarded diagnostic.
            values = []
            start = 0
            for size in groups:
                values.append(
                    self._accumulate(
                        target[start : start + size],
                        estimate[start : start + size],
                        importance[start : start + size],
                        (size,),
                    )["global_total_loss"]
                )
                start += size
            return float(np.mean(values))

        self.assertNotAlmostEqual(
            batch_ratio_mean((8, 4)), batch_ratio_mean((1,) * 12), places=9
        )
        # The ranked number, over exactly the same two groupings, does not move.
        self.assertEqual(
            self._accumulate(target, estimate, importance, (8, 4)),
            self._accumulate(target, estimate, importance, (1,) * 12),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
