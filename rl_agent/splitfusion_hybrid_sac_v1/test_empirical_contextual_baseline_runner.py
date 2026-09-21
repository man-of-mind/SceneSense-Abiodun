"""Focused schedule, artifacts, and resume tests for the preliminary baseline."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .empirical_contextual_baseline_runner import (
    BASELINE_CHECKPOINT_INTERVAL_UPDATES,
    BASELINE_PROVENANCE_LABEL,
    PHASE_LABEL,
    REGISTERED_BASELINE_CONFIG,
)
from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
)
from .empirical_contextual_smoke_runner import EmpiricalSmokeConfigV1
from .run_empirical_contextual_baseline import (
    _load_checkpoint,
    run_seed_to_directory,
)


class RegisteredBaselineScheduleTest(unittest.TestCase):
    def test_exact_preliminary_schedule(self) -> None:
        config = REGISTERED_BASELINE_CONFIG
        self.assertEqual(config.seeds, (17, 29, 43))
        self.assertEqual(config.warmup_transitions, 1024)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.collect_per_update, 4)
        self.assertEqual(config.update_count, 5000)
        self.assertEqual(config.replay_capacity, 32768)
        self.assertEqual(config.cpu_threads, 1)
        self.assertEqual(config.total_transitions, 21024)
        self.assertEqual(BASELINE_CHECKPOINT_INTERVAL_UPDATES, 500)
        self.assertEqual(config.scope, PHASE_LABEL)
        self.assertIn("NOT_PROVENANCE_COMPLETE", BASELINE_PROVENANCE_LABEL)


class SmallScheduleArtifactTest(unittest.TestCase):
    def test_bit_exact_resume_and_output_shapes(self) -> None:
        config = EmpiricalSmokeConfigV1(
            seeds=(17,),
            warmup_transitions=4,
            batch_size=4,
            collect_per_update=2,
            update_count=3,
            replay_capacity=16,
            scope=PHASE_LABEL,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct_dir = root / "direct"
            resumed_dir = root / "resumed"
            direct = run_seed_to_directory(
                output_directory=direct_dir,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
            )
            partial = run_seed_to_directory(
                output_directory=resumed_dir,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
                stop_after_updates=1,
            )
            self.assertEqual(partial["status"], "IN_PROGRESS")
            resumed = run_seed_to_directory(
                output_directory=resumed_dir,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
            )

            self.assertEqual(direct["status"], "COMPLETE")
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(direct["summary"], resumed["summary"])
            self.assertEqual(
                direct["checkpoint_sha256"], resumed["checkpoint_sha256"]
            )
            self.assertEqual(
                (direct_dir / "metrics.csv").read_bytes(),
                (resumed_dir / "metrics.csv").read_bytes(),
            )

            expected_files = {
                "bindings.json",
                "checkpoint_latest.pt",
                "checkpoints",
                "config.json",
                "metrics.csv",
                "report.json",
            }
            self.assertEqual(
                {path.name for path in resumed_dir.iterdir()}, expected_files
            )
            with (resumed_dir / "metrics.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["update_index"] for row in rows], ["1", "2", "3"])
            self.assertTrue(
                all(float(row["target_reward_max_abs_diff"]) == 0.0 for row in rows)
            )
            bindings = json.loads(
                (resumed_dir / "bindings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(bindings["sampling_split"], "train")
            self.assertEqual(
                bindings["fit_partition_sha256"],
                REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            )
            self.assertEqual(bindings["provenance_label"], BASELINE_PROVENANCE_LABEL)
            report = json.loads(
                (resumed_dir / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["completed_updates"], 3)
            self.assertEqual(report["summary"]["transition_count"], 10)
            numbered = sorted((resumed_dir / "checkpoints").glob("checkpoint_*.pt"))
            self.assertEqual(
                [path.name for path in numbered],
                [
                    "checkpoint_000001.pt",
                    "checkpoint_000002.pt",
                    "checkpoint_000003.pt",
                ],
            )
            checkpoint = _load_checkpoint(resumed_dir / "checkpoint_latest.pt")
            self.assertEqual(checkpoint.update_count, 3)
            self.assertEqual(checkpoint.collection_seq, 10)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
