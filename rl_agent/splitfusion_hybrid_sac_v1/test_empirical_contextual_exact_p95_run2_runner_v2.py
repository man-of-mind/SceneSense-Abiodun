"""Focused train-only, checkpoint, and resume tests for Run-2-v2."""

from __future__ import annotations

import inspect
import random
import struct
import unittest
from dataclasses import replace

import torch

from .empirical_contextual_exact_p95_run2_runner_v2 import (
    CHECKPOINT_SCHEMA,
    EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2,
    PHASE_LABEL,
    REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    RUNNER_SCHEMA,
    ExactP95Run2CheckpointV2,
    ExactP95Run2RunnerConfigV2,
    ExactP95Run2RunnerErrorV2,
    ExactP95Run2RunnerV2,
    registered_warmup_mode_counts_v2,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_smoke_runner import EmpiricalSmokeConfigV1


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return tuple(left) == tuple(right) and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class RegisteredRun2V2ScheduleTest(unittest.TestCase):
    def test_exact_registered_schedule_and_distinct_identity(self) -> None:
        config = REGISTERED_EXACT_P95_RUN2_CONFIG_V2
        self.assertEqual(config.seeds, (17, 29, 43))
        self.assertEqual(config.warmup_transitions, 1024)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.collect_per_update, 4)
        self.assertEqual(config.update_count, 5000)
        self.assertEqual(config.replay_capacity, 32768)
        self.assertEqual(config.cpu_threads, 1)
        self.assertEqual(config.total_transitions, 21024)
        self.assertEqual(EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2, 500)
        self.assertEqual(config.scope, PHASE_LABEL)
        self.assertIn("run2_runner.v2", RUNNER_SCHEMA)
        self.assertIn("run2_checkpoint.v2", CHECKPOINT_SCHEMA)
        self.assertNotEqual(
            config.to_canonical_dict()["record"],
            EmpiricalSmokeConfigV1().to_canonical_dict()["record"],
        )
        for seed in config.seeds:
            counts = registered_warmup_mode_counts_v2(seed)
            self.assertEqual(len(counts), 12)
            self.assertEqual(sum(counts), 1024)
            self.assertTrue(all(count > 0 for count in counts))

    def test_schedule_forbids_eviction_and_foreign_v1_config(self) -> None:
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            replace(
                REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
                replay_capacity=21023,
            )
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            replace(REGISTERED_EXACT_P95_RUN2_CONFIG_V2, cpu_threads=2)
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            replace(REGISTERED_EXACT_P95_RUN2_CONFIG_V2, scope="RUN1")
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            ExactP95Run2RunnerV2(seed=17, config=EmpiricalSmokeConfigV1())

    def test_runner_source_has_no_development_panel_or_evaluator_dependency(self) -> None:
        from . import empirical_contextual_exact_p95_run2_runner_v2 as module

        source = inspect.getsource(module)
        banned_imports = (
            "from .empirical_contextual_fit_validation_panel import",
            "from .empirical_contextual_fit_validation_evaluator import",
            "import empirical_contextual_fit_validation_panel",
            "import empirical_contextual_fit_validation_evaluator",
        )
        for banned in banned_imports:
            self.assertNotIn(banned, source)


class ExactP95Run2V2EndToEndTest(unittest.TestCase):
    """One tiny real-evidence run covers update zero and exact resumption."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExactP95Run2RunnerConfigV2(
            seeds=(17,),
            warmup_transitions=4,
            batch_size=4,
            collect_per_update=2,
            update_count=3,
            replay_capacity=16,
        )
        cls.python_rng_before = random.getstate()
        cls.torch_rng_before = torch.get_rng_state().clone()
        cls.thread_count_before = torch.get_num_threads()
        cls.cuda_before = torch.cuda.is_initialized()
        cls.runner = ExactP95Run2RunnerV2(seed=17, config=cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runner.close()
        if random.getstate() != cls.python_rng_before:
            raise AssertionError("Run-2-v2 test advanced global Python RNG")
        if not torch.equal(torch.get_rng_state(), cls.torch_rng_before):
            raise AssertionError("Run-2-v2 test advanced global Torch RNG")
        if torch.get_num_threads() != cls.thread_count_before:
            raise AssertionError("Run-2-v2 test changed process CPU thread count")
        if not cls.cuda_before and torch.cuda.is_initialized():
            raise AssertionError("Run-2-v2 test initialized CUDA")

    def test_00_real_hash_attested_update_zero_checkpoint(self) -> None:
        runner = self.runner
        checkpoint = runner.checkpoint()
        self.assertIs(type(checkpoint), ExactP95Run2CheckpointV2)
        checkpoint.require_valid()
        self.assertEqual(checkpoint.schema, CHECKPOINT_SCHEMA)
        self.assertEqual(checkpoint.update_count, 0)
        self.assertEqual(checkpoint.collection_seq, 0)
        self.assertEqual(checkpoint.warmup_collected, 0)
        self.assertEqual(checkpoint.post_warmup_collected, 0)
        self.assertEqual(checkpoint.environment_state.d1_state.reset_count, 0)
        self.assertEqual(checkpoint.d1_transition_history, ())
        self.assertEqual(checkpoint.run2_v2_transition_history, ())
        self.assertEqual(checkpoint.metrics, ())
        self.assertIsNone(checkpoint.replay_binding)
        self.assertFalse(checkpoint.trainer_initialized)
        self.assertEqual(checkpoint.actor_optimizer_state["state"], {})
        self.assertEqual(checkpoint.critic_optimizer_state["state"], {})
        self.assertRegex(checkpoint.checkpoint_sha256, r"^[0-9a-f]{64}$")
        self.assertNotEqual(checkpoint.checkpoint_sha256, "0" * 64)
        runner.load_checkpoint(checkpoint)
        self.assertEqual(
            runner.checkpoint().checkpoint_sha256,
            checkpoint.checkpoint_sha256,
        )
        summary = runner.run_until_updates(0)
        self.assertEqual(summary.transition_count, 0)
        self.assertEqual(summary.checkpoint_sha256, checkpoint.checkpoint_sha256)
        self.assertIsNone(summary.raw_d1_reward_mean)
        self.assertIsNone(summary.replay_binding_sha256)
        self.assertFalse(summary.completed_training_hard_gates_passed)

    def test_01_train_only_reward_audit_and_exact_resume(self) -> None:
        runner = self.runner
        split_summary = runner.run_until_updates(1)
        split_checkpoint = runner.checkpoint()
        self.assertEqual(split_summary.transition_count, 6)
        self.assertEqual(split_checkpoint.replay_evicted_count, 0)
        self.assertTrue(split_checkpoint.trainer_initialized)
        self.assertIsNotNone(split_checkpoint.replay_binding)

        partition = load_registered_empirical_fit_partition()
        train_scene_ids = {
            item.sample_id
            for item in partition.scene_assignments
            if item.split == TRAIN_SPLIT
        }
        held_scene_ids = {
            item.sample_id
            for item in partition.scene_assignments
            if item.split == FIT_VALIDATION_SPLIT
        }
        train_radio_rows = {
            item.csv_row_number
            for item in partition.radio_assignments
            if item.split == TRAIN_SPLIT
        }
        held_radio_rows = {
            item.csv_row_number
            for item in partition.radio_assignments
            if item.split == FIT_VALIDATION_SPLIT
        }
        self.assertTrue(set(runner.sampled_scene_ids) <= train_scene_ids)
        self.assertTrue(set(runner.sampled_scene_ids).isdisjoint(held_scene_ids))
        self.assertTrue(
            set(runner.sampled_radio_csv_row_numbers) <= train_radio_rows
        )
        self.assertTrue(
            set(runner.sampled_radio_csv_row_numbers).isdisjoint(held_radio_rows)
        )

        reward_difference_count = 0
        for source, shaped in zip(
            runner.d1_transition_history,
            runner.run2_v2_transition_history,
        ):
            self.assertEqual(source.canonical_sha256(), shaped.source_d1_transition.canonical_sha256())
            self.assertEqual(source.reward, shaped.source_d1_reward64)
            self.assertEqual(
                struct.pack(">f", shaped.emitted_reward_float32),
                struct.pack(">f", float(torch.tensor(shaped.shaped_reward64, dtype=torch.float32))),
            )
            if source.reward != shaped.emitted_reward_float32:
                reward_difference_count += 1
        self.assertGreater(reward_difference_count, 0)
        self.assertTrue(
            all(
                metric.target_reward_signed_bit_mismatch_count == 0
                for metric in runner.metrics
            )
        )

        uninterrupted_summary = runner.run()
        uninterrupted_checkpoint = runner.checkpoint()
        uninterrupted_d1_hashes = tuple(
            item.canonical_sha256() for item in runner.d1_transition_history
        )
        uninterrupted_v2_hashes = tuple(
            item.canonical_sha256() for item in runner.run2_v2_transition_history
        )
        uninterrupted_metrics = tuple(item.as_dict() for item in runner.metrics)

        runner.load_checkpoint(split_checkpoint)
        resumed_summary = runner.run()
        resumed_checkpoint = runner.checkpoint()
        self.assertEqual(uninterrupted_summary, resumed_summary)
        self.assertEqual(
            uninterrupted_checkpoint.checkpoint_sha256,
            resumed_checkpoint.checkpoint_sha256,
        )
        self.assertEqual(
            uninterrupted_d1_hashes,
            tuple(item.canonical_sha256() for item in runner.d1_transition_history),
        )
        self.assertEqual(
            uninterrupted_v2_hashes,
            tuple(item.canonical_sha256() for item in runner.run2_v2_transition_history),
        )
        self.assertEqual(
            uninterrupted_metrics,
            tuple(item.as_dict() for item in runner.metrics),
        )
        for name in (
            "actor_state",
            "critics_state",
            "actor_optimizer_state",
            "critic_optimizer_state",
            "collection_rng_state",
            "replay_rng_state",
            "actor_update_rng_state",
        ):
            self.assertTrue(
                _state_equal(
                    getattr(uninterrupted_checkpoint, name),
                    getattr(resumed_checkpoint, name),
                ),
                name,
            )
        self.assertEqual(resumed_summary.transition_count, 10)
        self.assertEqual(resumed_summary.replay_resident_count, 10)
        self.assertEqual(resumed_summary.replay_eviction_count, 0)
        self.assertEqual(resumed_summary.sampling_split, TRAIN_SPLIT)
        self.assertEqual(
            resumed_summary.target_reward_signed_bit_mismatch_count, 0
        )
        self.assertTrue(resumed_summary.completed_training_hard_gates_passed)

    def test_02_foreign_and_malformed_checkpoint_fail_before_mutation(self) -> None:
        runner = self.runner
        before = runner.checkpoint()
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            runner.load_checkpoint(object())
        after_foreign = runner.checkpoint()
        self.assertEqual(
            before.checkpoint_sha256, after_foreign.checkpoint_sha256
        )

        malformed = replace(before, collection_seq=before.collection_seq + 1)
        with self.assertRaises(ExactP95Run2RunnerErrorV2):
            runner.load_checkpoint(malformed)
        after_malformed = runner.checkpoint()
        self.assertEqual(
            before.checkpoint_sha256, after_malformed.checkpoint_sha256
        )


if __name__ == "__main__":
    unittest.main()
