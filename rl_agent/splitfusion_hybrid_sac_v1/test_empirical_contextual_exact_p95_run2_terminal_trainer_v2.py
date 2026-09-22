"""Adversarial numerical tests for the isolated Run-2 v2 trainer."""

from __future__ import annotations

import ast
import copy
import inspect
import math
import unittest
from unittest import mock

import torch

from . import empirical_contextual_exact_p95_run2_replay as replay_v1
from . import empirical_contextual_exact_p95_run2_replay_v2 as replay_v2
from . import empirical_contextual_exact_p95_run2_terminal_trainer_v2 as trainer
from . import empirical_contextual_terminal_replay as d1_replay
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT
from .test_empirical_contextual_terminal_replay import make_transition


def make_v2_batch(count: int = 12, *, seed: int = 13):
    store = replay_v2.ExactP95Run2ReplayV2(capacity=count)
    reward_binding = None
    for index in range(count):
        mode_id = index % len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode_id]
        q_e4 = lower + ((upper - lower) * ((index % 5) + 1)) // 6
        source = make_transition(
            collection_seq=index,
            mode_id=mode_id,
            q_e4=q_e4,
            state_offset=(index % 7) / 10.0,
            q_perc=0.45 + 0.01 * (index % 10),
        )
        shaped = replay_v2.ExactP95Run2ShapedTransitionV2.from_validated_d1(
            source, reward_binding=reward_binding
        )
        reward_binding = shaped.reward_binding
        store.insert(shaped)
    return store.sample(count, torch.Generator().manual_seed(seed))


def make_d1_batch(count: int = 2):
    store = d1_replay.EmpiricalTerminalReplayV1(capacity=count)
    for index in range(count):
        store.insert(make_transition(collection_seq=1000 + index))
    return store.sample(count, torch.Generator().manual_seed(1))


def make_v1_batch():
    source = make_transition(collection_seq=2000)
    shaped = replay_v1.ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
    store = replay_v1.ExactP95Run2ReplayV1(capacity=1)
    store.insert(shaped)
    return store.sample(1, torch.Generator().manual_seed(1))


def build_trainer(batch, *, actor_seed: int = 100, critic_seed: int = 200):
    model_config = HybridSacModelConfig(
        dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )
    actor = build_actor(model_config, seed=actor_seed)
    critics = build_twin_critics(model_config, seed=critic_seed)
    config = trainer.ExactP95Run2TerminalTrainerConfigV2(
        batch_size=batch.batch_size
    )
    return trainer.ExactP95Run2TerminalHybridSacTrainerV2(
        actor,
        critics,
        config,
        expected_binding=batch.binding,
        actor_generator=torch.Generator().manual_seed(300),
    )


def parameter_snapshot(instance):
    return tuple(
        parameter.detach().clone()
        for parameter in list(instance.actor.parameters())
        + list(instance.critics.parameters())
    )


def assert_parameters_equal(test, instance, before):
    after = tuple(
        parameter.detach()
        for parameter in list(instance.actor.parameters())
        + list(instance.critics.parameters())
    )
    test.assertEqual(len(before), len(after))
    test.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))


class ExactV2AcceptanceTest(unittest.TestCase):
    def test_exact_v2_batch_updates_and_target_critics_are_not_evaluated(self) -> None:
        batch = make_v2_batch()
        instance = build_trainer(batch)
        with mock.patch.object(
            instance.critics.target_1,
            "forward",
            side_effect=AssertionError("target_1 forward forbidden"),
        ), mock.patch.object(
            instance.critics.target_2,
            "forward",
            side_effect=AssertionError("target_2 forward forbidden"),
        ):
            metrics = instance.update_once(batch)
        self.assertEqual(metrics.target_reward_signed_bit_mismatch_count, 0)
        self.assertEqual(metrics.target_mean, metrics.reward_mean)
        self.assertEqual(metrics.target_min, metrics.reward_min)
        self.assertEqual(metrics.target_max, metrics.reward_max)
        self.assertEqual(instance.update_count, 1)

    def test_actor_and_critic_gradients_are_isolated_and_nonzero(self) -> None:
        batch = make_v2_batch()
        instance = build_trainer(batch)
        metrics = instance.update_once(batch)
        for head_name in ("mean_head", "log_std_head", "logit_head"):
            gradient = getattr(instance.actor, head_name).weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool((gradient.detach().abs().sum(dim=1) > 0.0).all()))
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in instance._online_critic_parameters
            )
        )
        self.assertGreater(metrics.actor_grad_norm, 0.0)
        self.assertGreater(metrics.critic_grad_norm, 0.0)
        self.assertGreater(metrics.actor_param_delta_norm, 0.0)
        self.assertGreater(metrics.online_critic_param_delta_norm, 0.0)
        self.assertGreater(metrics.target_param_delta_norm, 0.0)

    def test_metrics_are_finite_and_hash_bound(self) -> None:
        batch = make_v2_batch(8)
        instance = build_trainer(batch)
        metrics = instance.update_once(batch)
        metrics.assert_finite()
        self.assertEqual(len(metrics.canonical_sha256()), 64)
        self.assertEqual(
            metrics.replay_binding_sha256, batch.binding.canonical_sha256()
        )
        self.assertEqual(
            metrics.trainer_config_sha256,
            instance.config.canonical_sha256(),
        )
        self.assertEqual(metrics.phase_label, trainer.PHASE_LABEL)
        self.assertEqual(
            metrics.replay_phase_label, trainer.REPLAY_PHASE_LABEL
        )
        self.assertEqual(
            metrics.modeled_smoke_support_sha256,
            MODELED_SMOKE_SUPPORT.canonical_sha256(),
        )

    def test_critic_coordinate_is_derived_only_from_integer_q_e4(self) -> None:
        batch = make_v2_batch(5)
        instance = build_trainer(batch)
        observed = []
        real_q_values = instance.critics.q_values

        def capture(state, mode_onehot, q_normalized):
            observed.append(q_normalized.detach().clone())
            return real_q_values(state, mode_onehot, q_normalized)

        with mock.patch.object(instance.critics, "q_values", side_effect=capture):
            instance.update_once(batch)
        self.assertGreaterEqual(len(observed), 1)
        expected = batch.q_e4.to(torch.float32) / 9800.0
        self.assertTrue(torch.equal(observed[0], expected))


class ExactBoundaryAndOrderingTest(unittest.TestCase):
    def assert_zero_mutation(self, instance, candidate, error_type) -> None:
        before = parameter_snapshot(instance)
        actor_optimizer = copy.deepcopy(instance.actor_optimizer.state_dict())
        critic_optimizer = copy.deepcopy(instance.critic_optimizer.state_dict())
        generator = instance._actor_generator.get_state().clone()
        parameters = tuple(
            list(instance.actor.parameters()) + list(instance.critics.parameters())
        )
        gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        )
        update_count = instance.update_count
        with self.assertRaises(error_type):
            instance.update_once(candidate)
        assert_parameters_equal(self, instance, before)
        self.assertEqual(instance.actor_optimizer.state_dict(), actor_optimizer)
        self.assertEqual(instance.critic_optimizer.state_dict(), critic_optimizer)
        self.assertTrue(
            torch.equal(generator, instance._actor_generator.get_state())
        )
        for parameter, expected in zip(parameters, gradients):
            if expected is None:
                self.assertIsNone(parameter.grad)
            else:
                self.assertTrue(torch.equal(parameter.grad, expected))
        self.assertEqual(instance.update_count, update_count)

    def test_d1_v1_subclass_and_foreign_batches_reject_before_mutation(self) -> None:
        exact = make_v2_batch(2)
        candidates = (make_d1_batch(), make_v1_batch(), {"reward": exact.reward})
        for candidate in candidates:
            self.assert_zero_mutation(
                build_trainer(exact),
                candidate,
                trainer.Run2V2TerminalTrainerPreflightError,
            )

        class BatchSubclass(replay_v2.ExactP95Run2BatchV2):
            pass

        subclass = object.__new__(BatchSubclass)
        instance = build_trainer(exact)
        with mock.patch.object(
            replay_v2.ExactP95Run2BatchV2, "revalidate"
        ) as revalidate:
            self.assert_zero_mutation(
                instance,
                subclass,
                trainer.Run2V2TerminalTrainerPreflightError,
            )
            revalidate.assert_not_called()

    def test_revalidation_failure_precedes_tensor_access_and_mutation(self) -> None:
        batch = make_v2_batch(3)
        instance = build_trainer(batch)
        source = inspect.getsource(instance._preflight)
        self.assertLess(source.index("batch.revalidate()"), source.index("batch.state"))
        with mock.patch.object(
            replay_v2.ExactP95Run2BatchV2,
            "revalidate",
            autospec=True,
            side_effect=replay_v2.ExactP95Run2ReplayV2Error("tamper"),
        ) as revalidate:
            self.assert_zero_mutation(
                instance,
                batch,
                trainer.Run2V2TerminalTrainerPreflightError,
            )
            revalidate.assert_called_once_with(batch)

    def test_authenticated_tensor_tamper_is_rejected_before_mutation(self) -> None:
        batch = make_v2_batch(3)
        corrupted = batch.reward
        corrupted[0] = torch.nextafter(
            corrupted[0], torch.tensor(math.inf, dtype=torch.float32)
        )
        object.__setattr__(batch, "_reward", corrupted)
        self.assert_zero_mutation(
            build_trainer(make_v2_batch(3)),
            batch,
            trainer.Run2V2TerminalTrainerPreflightError,
        )

    def test_signed_bits_detect_old_collision_boundary_and_negative_zero(self) -> None:
        old_feasible = torch.tensor(
            [-0.13886236214769926], dtype=torch.float32
        )
        old_infeasible = torch.tensor(
            [-0.13886236214769937], dtype=torch.float32
        )
        selected = torch.tensor(
            [-0.1388623639941216], dtype=torch.float32
        )
        self.assertEqual(trainer._signed_bit_mismatch_count(old_feasible, old_infeasible), 0)
        self.assertEqual(trainer._signed_bit_mismatch_count(old_feasible, selected), 1)
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(positive_zero, negative_zero))
        self.assertEqual(
            trainer._signed_bit_mismatch_count(positive_zero, negative_zero), 1
        )

    def test_binding_digest_tamper_fails_before_mutation(self) -> None:
        batch = make_v2_batch(3)
        instance = build_trainer(batch)
        instance.expected_binding = copy.deepcopy(instance.expected_binding)
        original = instance.expected_binding.float_dtype
        object.__setattr__(
            instance.expected_binding, "float_dtype", "torch.float64"
        )
        try:
            self.assert_zero_mutation(
                instance, batch, trainer.Run2V2TerminalTrainerStateError
            )
        finally:
            object.__setattr__(instance.expected_binding, "float_dtype", original)


class StateIntegrityAndTransactionTest(unittest.TestCase):
    def test_model_optimizer_and_config_drift_fail_closed(self) -> None:
        batch = make_v2_batch(4)
        cases = []
        instance = build_trainer(batch)
        next(instance.actor.parameters()).requires_grad_(False)
        cases.append(instance)
        instance = build_trainer(batch)
        next(instance.critics.target_1.parameters()).requires_grad_(True)
        cases.append(instance)
        instance = build_trainer(batch)
        object.__setattr__(instance.config, "tau", 0.25)
        cases.append(instance)
        instance = build_trainer(batch)
        instance.actor_optimizer.add_param_group(
            {"params": [next(instance.critics.target_1.parameters())]}
        )
        cases.append(instance)
        instance = build_trainer(batch)
        instance.actor_optimizer.param_groups[0]["lr"] = 0.1
        cases.append(instance)
        instance = build_trainer(batch)
        instance.critics.critic_1 = build_twin_critics(
            instance.actor.config, seed=987
        ).critic_1
        cases.append(instance)
        instance = build_trainer(batch)
        object.__setattr__(instance.actor.config, "hidden_width", 64)
        cases.append(instance)
        for candidate in cases:
            ExactBoundaryAndOrderingTest().assert_zero_mutation(
                candidate,
                batch,
                trainer.Run2V2TerminalTrainerStateError,
            )

    def test_configured_batch_size_is_exact(self) -> None:
        configured = make_v2_batch(4)
        different_size = make_v2_batch(3)
        ExactBoundaryAndOrderingTest().assert_zero_mutation(
            build_trainer(configured),
            different_size,
            trainer.Run2V2TerminalTrainerPreflightError,
        )

    def test_update_uses_only_local_rng_and_is_deterministic(self) -> None:
        batch = make_v2_batch(6)
        first = build_trainer(batch)
        second = build_trainer(batch)
        torch.manual_seed(9182)
        global_before = torch.get_rng_state().clone()
        local_before = first._actor_generator.get_state().clone()
        first_metrics = first.update_once(batch).as_dict()
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertFalse(
            torch.equal(local_before, first._actor_generator.get_state())
        )
        second_metrics = second.update_once(batch).as_dict()
        self.assertEqual(first_metrics, second_metrics)
        first_parameters = list(first.actor.parameters()) + list(first.critics.parameters())
        second_parameters = list(second.actor.parameters()) + list(second.critics.parameters())
        self.assertTrue(
            all(
                torch.equal(left.detach(), right.detach())
                for left, right in zip(first_parameters, second_parameters)
            )
        )

    def test_injected_post_critic_failure_rolls_back_everything(self) -> None:
        batch = make_v2_batch(5)
        instance = build_trainer(batch)
        real_objective = trainer.actor_objective

        def advance_local_rng_then_fail(*args, **kwargs):
            real_objective(*args, **kwargs)
            raise RuntimeError("injected post-critic failure")

        with mock.patch.object(
            trainer,
            "actor_objective",
            side_effect=advance_local_rng_then_fail,
        ):
            ExactBoundaryAndOrderingTest().assert_zero_mutation(
                instance, batch, RuntimeError
            )


class SourceScopeSentinelTest(unittest.TestCase):
    def test_source_has_no_target_recomputation_or_private_reward_reads(self) -> None:
        source = inspect.getsource(trainer)
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(
            any(
                module
                and (
                    module.endswith("empirical_contextual_terminal_trainer")
                    or module.endswith("empirical_contextual_exact_p95_deadline_penalty_v2")
                )
                for module in imported_modules
            )
        )
        forbidden_attributes = {
            "source_d1_reward64",
            "p95_base_reward64",
            "shaped_reward64",
            "reward_binding",
            "discount",
        }
        observed_attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertTrue(forbidden_attributes.isdisjoint(observed_attributes))
        forbidden_names = {
            "critic_target",
            "soft_state_value",
            "RUN2_V2_DEADLINE_PENALTY",
            "base_p95_expected_utility64_v2",
            "shaped_p95_expected_utility64_v2",
        }
        observed_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertTrue(forbidden_names.isdisjoint(observed_names))
        update_source = inspect.getsource(
            trainer.ExactP95Run2TerminalHybridSacTrainerV2._update_once_after_preflight
        )
        self.assertIn("target = batch.terminal_target()", update_source)
        self.assertIn(
            "q_normalized = q_e4.to(torch.float32) / float(Q_E4_MAX)",
            update_source,
        )

    def test_only_exact_v2_replay_types_are_imported(self) -> None:
        tree = ast.parse(inspect.getsource(trainer))
        replay_imports = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.endswith(
                    "empirical_contextual_exact_p95_run2_replay_v2"
                )
            ):
                replay_imports.extend(alias.name for alias in node.names)
        self.assertEqual(
            replay_imports,
            ["ExactP95Run2BatchV2", "ExactP95Run2ReplayBindingV2"],
        )


if __name__ == "__main__":
    unittest.main()
