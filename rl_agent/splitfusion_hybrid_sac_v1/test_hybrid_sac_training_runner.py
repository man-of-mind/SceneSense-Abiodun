"""Focused tests for the bounded Hybrid-SAC algorithm qualification runner."""

from __future__ import annotations

import copy
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from .action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX
from .hybrid_sac_training_runner import (
    PHASE_LABEL,
    AcceptanceThresholdsV1,
    HybridSacAlgorithmQualificationRunnerV1,
    QualificationEnvironmentV1,
    QualificationError,
    QualificationReplayV1,
    QualificationRunnerConfigV1,
    run_seed_suite,
)
from .replay_buffer import ReplayBufferV1, ReplayTensorBatchV1
from .state_reward_transition_contract import POLICY_FEATURE_COUNT


def _small_config(seed: int = 17) -> QualificationRunnerConfigV1:
    return QualificationRunnerConfigV1(
        seed=seed,
        max_updates=12,
        replay_capacity=48,
        batch_size=4,
        warmup_transitions=8,
        collect_per_update=2,
        episode_horizon=5,
        gamma_per_tensor=0.99,
        alpha_d=0.10,
        alpha_c=0.05,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )


def _assert_nested_equal(test: unittest.TestCase, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertTrue(torch.equal(left, right))
        return
    if isinstance(left, dict):
        test.assertIsInstance(right, dict)
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
        return
    if isinstance(left, (tuple, list)):
        test.assertIsInstance(right, type(left))
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            _assert_nested_equal(test, first, second)
        return
    test.assertEqual(left, right)


class EnvironmentTests(unittest.TestCase):
    def test_all_modes_have_distinct_strictly_interior_optima(self) -> None:
        environment = QualificationEnvironmentV1(horizon=4)
        self.assertEqual(len(environment.Q_OPTIMA_E4), EXPECTED_MODE_COUNT)
        self.assertEqual(len(set(environment.Q_OPTIMA_E4)), EXPECTED_MODE_COUNT)
        self.assertTrue(all(0 < value < Q_E4_MAX for value in environment.Q_OPTIMA_E4))
        for context in range(EXPECTED_MODE_COUNT):
            state = environment.observation(context, 0.5)
            self.assertEqual(state.shape, (POLICY_FEATURE_COUNT,))
            self.assertEqual(state.dtype, torch.float32)
            mode, q_e4 = environment.oracle_action(context)
            # This analytic mechanics task deliberately exposes both targets;
            # these states are not an independent generalization split.
            self.assertEqual(int(state[:EXPECTED_MODE_COUNT].argmax()), mode)
            self.assertEqual(round(float(state[12]) * Q_E4_MAX), q_e4)
            self.assertEqual(mode, context)
            self.assertEqual(environment.reward_for(context, mode, q_e4), 1.0)
            self.assertLess(environment.reward_for(context, (mode + 1) % 12, q_e4), 1.0)
            self.assertLess(environment.reward_for(context, mode, 0), 1.0)

    def test_environment_round_trip_restores_exact_next_transition(self) -> None:
        first = QualificationEnvironmentV1(horizon=3)
        generator = torch.Generator().manual_seed(123)
        state = first.reset(generator)
        first.step(0, 1000, generator)
        checkpoint = copy.deepcopy(first.state_dict())
        rng = generator.get_state().clone()
        expected = first.step(1, 2000, generator)

        second = QualificationEnvironmentV1(horizon=3)
        second.load_state_dict(checkpoint)
        generator_2 = torch.Generator().manual_seed(999)
        generator_2.set_state(rng)
        observed = second.step(1, 2000, generator_2)
        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertEqual(expected[1:], observed[1:])
        self.assertTrue(torch.isfinite(state).all())


class ReplayIsolationTests(unittest.TestCase):
    def test_qualification_replay_is_not_production_replay(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        self.assertIsInstance(runner.replay, QualificationReplayV1)
        self.assertNotIsInstance(runner.replay, ReplayBufferV1)
        source = inspect.getsource(
            __import__(
                "rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_training_runner",
                fromlist=["*"],
            )
        )
        self.assertNotIn("from .replay_buffer import ReplayBufferV1", source)
        self.assertNotIn("experiments/", source)

    def test_direct_batch_is_explicitly_qualification_only(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        for _ in range(8):
            runner.collect_one(force_random=True)
        batch = runner.replay.sample(4, runner.generators["replay"])
        self.assertIs(type(batch), ReplayTensorBatchV1)
        self.assertEqual(batch.batch_size, 4)
        self.assertTrue(all(record["evidence_class"] == PHASE_LABEL for record in batch.audit))
        self.assertTrue(
            all("NOT_SPLITFUSION_EVIDENCE" in record["source"] for record in batch.audit)
        )
        self.assertTrue(torch.isfinite(batch.reward).all())
        self.assertTrue(torch.isfinite(batch.discount()).all())

    def test_replay_round_trip_preserves_sampling(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        for _ in range(10):
            runner.collect_one(force_random=True)
        state = copy.deepcopy(runner.replay.state_dict())
        rng_state = runner.generators["replay"].get_state().clone()
        expected = runner.replay.sample(4, runner.generators["replay"])

        restored = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        restored.replay.load_state_dict(state)
        restored.generators["replay"].set_state(rng_state)
        observed = restored.replay.sample(4, restored.generators["replay"])
        for name in ("state", "next_state", "mode_id", "q_e4", "reward", "duration"):
            self.assertTrue(torch.equal(getattr(expected, name), getattr(observed, name)))
        self.assertTrue(torch.equal(expected.discount(), observed.discount()))


class RunnerTests(unittest.TestCase):
    def test_five_explicit_rng_streams_are_distinct_and_global_rng_is_untouched(self) -> None:
        torch.manual_seed(789)
        before = torch.random.get_rng_state().clone()
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        after = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(
            set(runner.generators),
            {"collection", "replay", "target", "actor", "evaluation"},
        )
        self.assertEqual(len({id(value) for value in runner.generators.values()}), 5)
        for generator in runner.generators.values():
            self.assertIsNot(generator, torch.default_generator)
            self.assertEqual(generator.device.type, "cpu")

    def test_bounded_multi_update_emits_only_finite_audit_metrics(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        metrics = runner.advance(3)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(runner.update_count, 3)
        self.assertGreaterEqual(len(runner.replay), runner.config.warmup_transitions)
        for metric in metrics:
            metric.assert_finite()
            for value in metric.as_dict().values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.assertTrue(math.isfinite(float(value)))
        with self.assertRaises(QualificationError):
            runner.advance(runner.config.max_updates)

    def test_uninterrupted_and_resumed_runs_are_bitwise_identical(self) -> None:
        uninterrupted = HybridSacAlgorithmQualificationRunnerV1(_small_config(33))
        uninterrupted.advance(2)
        checkpoint = uninterrupted.checkpoint_state()

        resumed = HybridSacAlgorithmQualificationRunnerV1.from_checkpoint(checkpoint)
        expected_metrics = uninterrupted.advance(3)
        observed_metrics = resumed.advance(3)
        self.assertEqual(
            [metric.as_dict() for metric in expected_metrics],
            [metric.as_dict() for metric in observed_metrics],
        )
        _assert_nested_equal(
            self, uninterrupted.checkpoint_state(), resumed.checkpoint_state()
        )

    def test_checkpoint_includes_every_required_mutable_component(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        runner.advance(1)
        runner.evaluate(24)
        checkpoint = runner.checkpoint_state()
        self.assertTrue(
            {
                "actor",
                "critics",
                "actor_optimizer",
                "critic_optimizer",
                "generators",
                "fixed_evaluation_rng_state",
                "replay",
                "environment",
                "current_state",
                "collected_transitions",
                "trainer_update_count",
                "update_history",
            }.issubset(checkpoint)
        )
        self.assertEqual(set(checkpoint["generators"]), set(runner.generators))
        self.assertTrue(
            torch.equal(
                checkpoint["fixed_evaluation_rng_state"],
                runner.fixed_evaluation_rng_state,
            )
        )

    def test_checkpoint_file_round_trip(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(7))
        runner.advance(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.pt"
            digest = runner.save_checkpoint(path)
            self.assertTrue(path.is_file())
            self.assertEqual(len(digest), 64)
            restored = HybridSacAlgorithmQualificationRunnerV1.load_checkpoint(
                path, expected_sha256=digest
            )
            with self.assertRaises(QualificationError):
                HybridSacAlgorithmQualificationRunnerV1.load_checkpoint(
                    path, expected_sha256="0" * 64
                )
            source = inspect.getsource(
                HybridSacAlgorithmQualificationRunnerV1.load_checkpoint
            )
            self.assertNotIn("weights_only=False", source)
            self.assertEqual(source.count("weights_only=True"), 2)
        _assert_nested_equal(self, runner.checkpoint_state(), restored.checkpoint_state())

    def test_tampered_replay_discount_is_rejected_without_mutation(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(19))
        runner.advance(1)
        before = runner.checkpoint_state()
        tampered = copy.deepcopy(before)
        tampered["replay"]["rows"][0]["discount"] = 0.123
        with self.assertRaisesRegex(QualificationError, "binding-derived"):
            runner.load_checkpoint_state(tampered)
        _assert_nested_equal(self, before, runner.checkpoint_state())

    def test_tampered_fixed_evaluation_origin_is_rejected_transactionally(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(20))
        runner.advance(1)
        before = runner.checkpoint_state()
        tampered = copy.deepcopy(before)
        tampered["fixed_evaluation_rng_state"] = (
            torch.Generator(device="cpu").manual_seed(999).get_state()
        )
        with self.assertRaisesRegex(QualificationError, "differs from seed"):
            runner.load_checkpoint_state(tampered)
        _assert_nested_equal(self, before, runner.checkpoint_state())

    def test_late_checkpoint_failure_is_transactional(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(23))
        runner.advance(1)
        before = runner.checkpoint_state()
        donor = HybridSacAlgorithmQualificationRunnerV1(_small_config(23))
        donor.advance(2)
        malformed = donor.checkpoint_state()
        malformed["update_history"] = []
        with self.assertRaisesRegex(QualificationError, "history"):
            runner.load_checkpoint_state(malformed)
        _assert_nested_equal(self, before, runner.checkpoint_state())

    def test_fixed_evaluation_has_random_fixed_and_oracle_baselines(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config())
        thresholds = AcceptanceThresholdsV1(
            minimum_improvement_over_random=-100.0,
            maximum_oracle_regret=100.0,
        )
        result = runner.evaluate(
            24,
            fixed_mode=2,
            fixed_q_e4=3200,
            thresholds=thresholds,
        )
        result.assert_finite()
        self.assertEqual(result.steps, 24)
        self.assertEqual(result.oracle_mean_reward, 1.0)
        self.assertAlmostEqual(
            result.improvement_over_random,
            result.policy_mean_reward - result.random_mean_reward,
        )
        self.assertAlmostEqual(
            result.oracle_regret,
            result.oracle_mean_reward - result.policy_mean_reward,
        )
        self.assertEqual(result.fixed_mode, 2)
        self.assertEqual(result.fixed_q_e4, 3200)
        self.assertEqual(
            result.minimum_improvement_over_random,
            thresholds.minimum_improvement_over_random,
        )
        self.assertEqual(
            result.maximum_oracle_regret,
            thresholds.maximum_oracle_regret,
        )
        # Loose reward thresholds alone cannot manufacture an acceptance:
        # all 12 hybrid modes and interior q values are structural gates.
        self.assertFalse(result.accepted)

    def test_recorded_fixed_evaluation_reproduces_after_checkpoint_load(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(91))
        runner.advance(2)
        runner.reset_fixed_evaluation_stream()
        expected = runner.evaluate(36).as_dict()
        # Save after evaluation has advanced the active evaluation generator.
        checkpoint = runner.checkpoint_state()
        restored = HybridSacAlgorithmQualificationRunnerV1.from_checkpoint(checkpoint)
        restored.reset_fixed_evaluation_stream()
        observed = restored.evaluate(36).as_dict()
        self.assertEqual(expected, observed)

    def test_checkpoint_history_uses_qualification_label(self) -> None:
        runner = HybridSacAlgorithmQualificationRunnerV1(_small_config(92))
        metrics = runner.advance(2)
        self.assertTrue(all(metric.phase_label == PHASE_LABEL for metric in metrics))
        self.assertTrue(
            all(
                record["phase_label"] == PHASE_LABEL
                for record in runner.checkpoint_state()["update_history"]
            )
        )

    def test_three_seed_suite_support_is_bounded(self) -> None:
        config = _small_config(0)
        results = run_seed_suite(
            config,
            seeds=(3, 5, 7),
            updates=1,
            evaluation_steps=24,
        )
        self.assertEqual([result.seed for result in results], [3, 5, 7])
        self.assertTrue(all(result.update_count == 1 for result in results))
        self.assertTrue(all(result.evaluation.steps == 24 for result in results))
        with self.assertRaises(QualificationError):
            run_seed_suite(config, seeds=(3, 3, 7), updates=0)

    def test_default_acceptance_is_non_vacuous_across_three_seeds(self) -> None:
        untrained = HybridSacAlgorithmQualificationRunnerV1(
            QualificationRunnerConfigV1(seed=1, max_updates=500)
        )
        initial = untrained.evaluate(120)
        self.assertFalse(initial.accepted)

        results = run_seed_suite(
            QualificationRunnerConfigV1(seed=0, max_updates=500),
            seeds=(4, 5, 6),
            updates=500,
            evaluation_steps=120,
        )
        self.assertEqual(len(results), 3)
        for result in results:
            evaluation = result.evaluation
            self.assertTrue(evaluation.accepted)
            self.assertEqual(evaluation.policy_selected_mode_count, 12)
            self.assertEqual(evaluation.policy_interior_q_fraction, 1.0)
            self.assertGreaterEqual(evaluation.policy_mode_accuracy, 11.0 / 12.0)
            self.assertLess(evaluation.policy_mean_absolute_q_error, 0.11)
            self.assertEqual(evaluation.minimum_improvement_over_random, 0.05)
            self.assertEqual(evaluation.maximum_oracle_regret, 0.35)

    def test_evaluation_cannot_change_subsequent_training(self) -> None:
        first = HybridSacAlgorithmQualificationRunnerV1(_small_config(87))
        second = HybridSacAlgorithmQualificationRunnerV1(_small_config(87))
        first.advance(2)
        second.advance(2)
        first.evaluate(24)
        expected = first.advance(2)
        observed = second.advance(2)
        self.assertEqual(
            [metric.as_dict() for metric in expected],
            [metric.as_dict() for metric in observed],
        )
        for left, right in zip(first.actor.parameters(), second.actor.parameters()):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
