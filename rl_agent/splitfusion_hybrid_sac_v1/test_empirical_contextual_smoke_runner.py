"""Focused offline CPU tests for the D2b mechanics-only smoke runner."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

import torch

from .empirical_contextual_contract import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    require_supported_action,
)
from .empirical_contextual_smoke_runner import (
    PHASE_LABEL,
    REGISTERED_SMOKE_CONFIG,
    SCOPE_DISCLOSURE,
    EmpiricalContextualSmokeRunnerV1,
    EmpiricalSmokeConfigV1,
    EmpiricalSmokeError,
    registered_warmup_mode_counts,
)
from .state_reward_transition_contract import FORBIDDEN_POLICY_FEATURE_SUBSTRINGS


class RegisteredScheduleTest(unittest.TestCase):
    def test_exact_three_seed_mechanical_schedule(self) -> None:
        config = REGISTERED_SMOKE_CONFIG
        self.assertEqual(config.seeds, (17, 29, 43))
        self.assertEqual(config.warmup_transitions, 1024)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.collect_per_update, 4)
        self.assertEqual(config.update_count, 500)
        self.assertEqual(config.replay_capacity, 8192)
        self.assertEqual(config.cpu_threads, 1)
        self.assertEqual(config.total_transitions, 3024)
        self.assertEqual(
            config.canonical_sha256(),
            "d0f991545bf7d4bcf4cf513bd8882bba5c952274ef3a0a8f9d9b53e680645431",
        )
        self.assertIn("NO_VALIDATION", PHASE_LABEL)
        self.assertIn("not the committed", SCOPE_DISCLOSURE)
        self.assertIn("generalization", SCOPE_DISCLOSURE)
        for seed in config.seeds:
            counts = registered_warmup_mode_counts(seed)
            self.assertEqual(len(counts), 12)
            self.assertEqual(sum(counts), 1024)
            self.assertTrue(all(count > 0 for count in counts))

    def test_schedule_refuses_eviction_or_non_cpu_contracts(self) -> None:
        for mutation in (
            {"cpu_threads": 2},
            {"warmup_transitions": 2, "batch_size": 3},
            {
                "warmup_transitions": 4,
                "batch_size": 4,
                "collect_per_update": 2,
                "update_count": 3,
                "replay_capacity": 9,
            },
        ):
            with self.assertRaises(EmpiricalSmokeError):
                replace(REGISTERED_SMOKE_CONFIG, **mutation)


class EndToEndMechanicalTest(unittest.TestCase):
    """One small schedule exercises real registered D1 without live systems."""

    def test_rng_isolation_exact_target_support_and_bit_exact_resume(self) -> None:
        config = EmpiricalSmokeConfigV1(
            seeds=(17,),
            warmup_transitions=4,
            batch_size=4,
            collect_per_update=2,
            update_count=3,
            replay_capacity=16,
        )
        python_rng_before = random.getstate()
        torch_rng_before = torch.get_rng_state().clone()
        thread_count_before = torch.get_num_threads()

        runner = EmpiricalContextualSmokeRunnerV1(seed=17, config=config)
        try:
            runner.run_until_updates(1)
            split_checkpoint = runner.checkpoint()
            uninterrupted = runner.run()
            uninterrupted_checkpoint = runner.checkpoint()
            uninterrupted_actions = tuple(
                (item.action.mode_id, item.action.q_e4)
                for item in runner.transition_history
            )
            uninterrupted_metrics = tuple(item.as_dict() for item in runner.metrics)

            runner.load_checkpoint(split_checkpoint)
            resumed = runner.run()
            resumed_checkpoint = runner.checkpoint()

            self.assertEqual(
                uninterrupted_checkpoint.checkpoint_sha256,
                resumed_checkpoint.checkpoint_sha256,
            )
            self.assertEqual(uninterrupted, resumed)
            self.assertEqual(
                uninterrupted_actions,
                tuple(
                    (item.action.mode_id, item.action.q_e4)
                    for item in runner.transition_history
                ),
            )
            self.assertEqual(
                uninterrupted_metrics,
                tuple(item.as_dict() for item in runner.metrics),
            )
            self.assertEqual(uninterrupted.transition_count, 10)
            self.assertEqual(uninterrupted.warmup_transition_count, 4)
            self.assertEqual(uninterrupted.post_warmup_transition_count, 6)
            self.assertEqual(uninterrupted.replay_resident_count, 10)
            self.assertEqual(uninterrupted.replay_eviction_count, 0)
            self.assertEqual(uninterrupted.support_violation_count, 0)
            self.assertEqual(uninterrupted.target_reward_max_abs_diff, 0.0)
            self.assertTrue(uninterrupted.all_metrics_finite)
            self.assertLess(uninterrupted.reward_min, uninterrupted.reward_max)
            self.assertGreater(uninterrupted.actor_parameter_delta_from_init, 0.0)
            self.assertGreater(uninterrupted.critic_1_parameter_delta_from_init, 0.0)
            self.assertGreater(uninterrupted.critic_2_parameter_delta_from_init, 0.0)
            self.assertTrue(uninterrupted.global_python_rng_unchanged)
            self.assertTrue(uninterrupted.global_torch_rng_unchanged)
            self.assertFalse(uninterrupted.cuda_initialized_by_runner)
            self.assertTrue(uninterrupted.mechanical_acceptance_passed)
            self.assertEqual(
                uninterrupted.modeled_smoke_support_sha256,
                MODELED_SMOKE_SUPPORT_SHA256,
            )
            self.assertEqual(uninterrupted.phase_label, PHASE_LABEL)

            history = runner.transition_history
            self.assertEqual(
                tuple(item.collection_seq for item in history), tuple(range(10))
            )
            self.assertEqual(len({item.logical_key for item in history}), 10)
            for transition in history:
                self.assertEqual(
                    require_supported_action(
                        transition.action.mode_id, transition.action.q_e4
                    ),
                    transition.action,
                )
                self.assertEqual(
                    transition.result.audit.executed_mode_id,
                    transition.action.mode_id,
                )
                self.assertEqual(
                    transition.result.audit.executed_q_e4,
                    transition.action.q_e4,
                )
                names = tuple(name.lower() for name in transition.observation.policy_feature_order)
                for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
                    self.assertTrue(all(forbidden.lower() not in name for name in names))
            for metric in runner.metrics:
                metric.assert_finite()
                self.assertEqual(metric.target_reward_max_abs_diff, 0.0)
        finally:
            runner.close()

        self.assertEqual(random.getstate(), python_rng_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before))
        self.assertEqual(torch.get_num_threads(), thread_count_before)


if __name__ == "__main__":
    unittest.main()
