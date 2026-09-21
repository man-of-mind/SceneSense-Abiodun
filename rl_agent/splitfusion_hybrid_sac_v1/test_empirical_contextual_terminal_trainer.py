"""CPU-only numerical and fail-closed tests for the D2a terminal trainer."""

from __future__ import annotations

import copy
import inspect
import unittest
from dataclasses import replace
from unittest import mock

import torch

from . import empirical_contextual_terminal_replay as replay
from . import empirical_contextual_terminal_trainer as trainer
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT
from .test_empirical_contextual_terminal_replay import (
    make_binding,
    make_transition,
)


def make_batch(count: int = 24, *, seed: int = 13):
    store = replay.EmpiricalTerminalReplayV1(capacity=count)
    bounds = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds
    for index in range(count):
        mode = index % len(bounds)
        lower, upper = bounds[mode]
        q_e4 = lower + ((upper - lower) * ((index % 5) + 1)) // 6
        store.insert(
            make_transition(
                collection_seq=index,
                mode_id=mode,
                q_e4=q_e4,
                state_offset=(index % 7) / 10.0,
                q_perc=0.45 + 0.01 * (index % 10),
            )
        )
    return store.sample(count, torch.Generator().manual_seed(seed))


def build_trainer(batch, *, actor_seed: int = 100, critic_seed: int = 200):
    model_config = HybridSacModelConfig(
        dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )
    actor = build_actor(model_config, seed=actor_seed)
    critics = build_twin_critics(model_config, seed=critic_seed)
    config = trainer.EmpiricalTerminalTrainerConfigV1(batch_size=batch.batch_size)
    instance = trainer.EmpiricalTerminalHybridSacTrainerV1(
        actor,
        critics,
        config,
        expected_binding=batch.binding,
        actor_generator=torch.Generator().manual_seed(300),
    )
    return instance


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


class ConstructionTest(unittest.TestCase):
    def test_requires_bounded_cpu_float32_models_and_local_generator(self) -> None:
        batch = make_batch(4)
        bounded = HybridSacModelConfig(
            modeled_smoke_support=MODELED_SMOKE_SUPPORT
        )
        critics = build_twin_critics(bounded, seed=2)
        config = trainer.EmpiricalTerminalTrainerConfigV1(batch_size=4)
        for generator in (None, torch.default_generator, "rng"):
            with self.assertRaises(trainer.TrainerStateError):
                trainer.EmpiricalTerminalHybridSacTrainerV1(
                    build_actor(bounded, seed=1),
                    critics,
                    config,
                    expected_binding=batch.binding,
                    actor_generator=generator,
                )

        unbounded = build_actor(HybridSacModelConfig(), seed=1)
        with self.assertRaises(trainer.TrainerStateError):
            trainer.EmpiricalTerminalHybridSacTrainerV1(
                unbounded,
                critics,
                config,
                expected_binding=batch.binding,
                actor_generator=torch.Generator().manual_seed(3),
            )

        double_config = HybridSacModelConfig(
            dtype=torch.float64,
            modeled_smoke_support=MODELED_SMOKE_SUPPORT,
        )
        with self.assertRaises(trainer.TrainerStateError):
            trainer.EmpiricalTerminalHybridSacTrainerV1(
                build_actor(double_config, seed=1),
                build_twin_critics(double_config, seed=2),
                config,
                expected_binding=batch.binding,
                actor_generator=torch.Generator().manual_seed(3),
            )

    def test_config_rejects_nonterminal_smoke_hyperparameters(self) -> None:
        for kwargs in (
            {"alpha_d": 0.0},
            {"alpha_c": float("nan")},
            {"tau": 1.1},
            {"actor_lr": -1.0},
            {"critic_lr": 0.0},
            {"batch_size": True},
            {"float_dtype": torch.float64},
        ):
            with self.assertRaises(trainer.TrainerPreflightError):
                trainer.EmpiricalTerminalTrainerConfigV1(**kwargs)


class UpdateMechanicsTest(unittest.TestCase):
    def test_target_is_reward_verbatim_and_target_critics_are_never_evaluated(self) -> None:
        batch = make_batch()
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
        self.assertEqual(metrics.target_reward_max_abs_diff, 0.0)
        self.assertEqual(metrics.target_mean, metrics.reward_mean)
        self.assertEqual(metrics.target_min, metrics.reward_min)
        self.assertEqual(metrics.target_max, metrics.reward_max)
        self.assertEqual(instance.update_count, 1)

    def test_actor_enumerates_all_modes_and_critic_gradients_stay_clear(self) -> None:
        batch = make_batch()
        instance = build_trainer(batch)
        metrics = instance.update_once(batch)
        for head_name in ("mean_head", "log_std_head", "logit_head"):
            gradient = getattr(instance.actor, head_name).weight.grad
            self.assertIsNotNone(gradient)
            row_activity = gradient.detach().abs().sum(dim=1)
            self.assertTrue(bool((row_activity > 0.0).all()), head_name)
        self.assertTrue(
            all(parameter.grad is None for parameter in instance._online_critic_parameters)
        )
        self.assertGreater(metrics.actor_grad_norm, 0.0)
        self.assertGreater(metrics.critic_grad_norm, 0.0)
        self.assertGreater(metrics.actor_param_delta_norm, 0.0)
        self.assertGreater(metrics.online_critic_param_delta_norm, 0.0)
        self.assertGreater(metrics.target_param_delta_norm, 0.0)

    def test_metrics_are_support_bound_and_finite(self) -> None:
        batch = make_batch(12)
        metrics = build_trainer(batch).update_once(batch)
        metrics.assert_finite()
        self.assertEqual(
            metrics.continuous_log_prob_coordinate,
            "NORMALIZED_Z_DENSITY",
        )
        self.assertEqual(
            metrics.modeled_smoke_support_sha256,
            batch.binding.modeled_smoke_support_sha256,
        )
        self.assertEqual(
            metrics.replay_binding_sha256, batch.binding.canonical_sha256()
        )
        self.assertEqual(
            metrics.trainer_config_sha256,
            trainer.EmpiricalTerminalTrainerConfigV1(
                batch_size=batch.batch_size
            ).canonical_sha256(),
        )
        self.assertGreaterEqual(metrics.q_saturation_fraction, 0.0)
        self.assertLessEqual(metrics.q_saturation_fraction, 1.0)

    def test_update_uses_only_local_rng_stream(self) -> None:
        batch = make_batch(8)
        instance = build_trainer(batch)
        torch.manual_seed(9182)
        global_before = torch.get_rng_state().clone()
        local_before = instance._actor_generator.get_state().clone()
        instance.update_once(batch)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertFalse(
            torch.equal(local_before, instance._actor_generator.get_state())
        )

    def test_two_identical_trainers_and_rngs_are_deterministic(self) -> None:
        batch = make_batch(10)
        first = build_trainer(batch)
        second = build_trainer(batch)
        first_metrics = first.update_once(batch).as_dict()
        second_metrics = second.update_once(batch).as_dict()
        self.assertEqual(first_metrics, second_metrics)
        first_values = [p.detach() for p in first.actor.parameters()]
        second_values = [p.detach() for p in second.actor.parameters()]
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(first_values, second_values))
        )


class FailClosedTest(unittest.TestCase):
    @staticmethod
    def altered_batch(batch, **changes):
        values = {
            "_state": batch.state,
            "_mode_id": batch.mode_id,
            "_q_e4": batch.q_e4,
            "_reward": batch.reward,
            "binding": batch.binding,
            "audit": batch.audit,
            "float_dtype": batch.float_dtype,
        }
        candidate = replay.EmpiricalTerminalBatchV1(**values)
        # Deliberately bypass the batch's own constructor gate to prove the
        # trainer independently rechecks a hostile, post-construction object.
        for name, value in changes.items():
            object.__setattr__(candidate, name, value)
        return candidate

    def assert_zero_mutation(self, instance, candidate, error_type) -> None:
        before = parameter_snapshot(instance)
        actor_state = copy.deepcopy(instance.actor_optimizer.state_dict())
        critic_state = copy.deepcopy(instance.critic_optimizer.state_dict())
        generator_state = instance._actor_generator.get_state().clone()
        parameters = tuple(
            list(instance.actor.parameters()) + list(instance.critics.parameters())
        )
        gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        )
        count = instance.update_count
        with self.assertRaises(error_type):
            instance.update_once(candidate)
        assert_parameters_equal(self, instance, before)
        self.assertEqual(instance.actor_optimizer.state_dict(), actor_state)
        self.assertEqual(instance.critic_optimizer.state_dict(), critic_state)
        self.assertTrue(
            torch.equal(generator_state, instance._actor_generator.get_state())
        )
        for parameter, expected in zip(parameters, gradients):
            if expected is None:
                self.assertIsNone(parameter.grad)
            else:
                self.assertTrue(torch.equal(parameter.grad, expected))
        self.assertEqual(instance.update_count, count)

    def test_shape_dtype_nonfinite_and_support_fail_before_mutation(self) -> None:
        batch = make_batch(6)
        cases = []
        bad_state = batch.state
        bad_state[0, 0] = float("nan")
        cases.append(self.altered_batch(batch, _state=bad_state))
        cases.append(
            self.altered_batch(batch, _state=batch.state.to(torch.float64))
        )
        bad_mode = batch.mode_id
        bad_mode[0] = 12
        cases.append(self.altered_batch(batch, _mode_id=bad_mode))
        bad_q = batch.q_e4
        bad_q[0] = 0
        cases.append(self.altered_batch(batch, _q_e4=bad_q))
        for candidate in cases:
            self.assert_zero_mutation(
                build_trainer(batch), candidate, trainer.TrainerPreflightError
            )

    def test_foreign_binding_and_wrong_batch_type_fail_before_mutation(self) -> None:
        batch = make_batch(4)
        other = make_transition(
            collection_seq=999,
            binding=make_binding(network_sha="a" * 64),
        )
        other_store = replay.EmpiricalTerminalReplayV1(capacity=1)
        other_store.insert(other)
        other_batch = other_store.sample(1, torch.Generator().manual_seed(1))
        self.assert_zero_mutation(
            build_trainer(batch), other_batch, trainer.TrainerPreflightError
        )
        self.assert_zero_mutation(
            build_trainer(batch), {"reward": batch.reward}, trainer.TrainerPreflightError
        )

    def test_float32_loss_overflow_is_refused_before_mutation(self) -> None:
        batch = make_batch(4)
        huge = self.altered_batch(
            batch, _reward=torch.full((4,), 1e20, dtype=torch.float32)
        )
        self.assert_zero_mutation(build_trainer(batch), huge, trainer.TrainerError)

    def test_optimizer_foreign_or_target_parameter_is_detected(self) -> None:
        batch = make_batch(4)
        instance = build_trainer(batch)
        target_parameter = next(instance.critics.target_1.parameters())
        instance.actor_optimizer.add_param_group({"params": [target_parameter]})
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

    def test_replaced_critic_module_is_detected_against_live_parameters(self) -> None:
        batch = make_batch(4)
        instance = build_trainer(batch)
        replacement = build_twin_critics(
            instance.actor.config, seed=987
        ).critic_1
        instance.critics.critic_1 = replacement
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

    def test_all_five_networks_must_be_parameter_identity_disjoint(self) -> None:
        batch = make_batch(4)
        instance = build_trainer(batch)
        instance.critics.critic_2.value_head.weight = (
            instance.critics.critic_1.value_head.weight
        )
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

        instance = build_trainer(batch)
        instance.actor.logit_head.weight = (
            instance.critics.critic_1.value_head.weight
        )
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

    def test_requires_grad_and_config_tampering_fail_before_mutation(self) -> None:
        batch = make_batch(4)
        for mutate in (
            lambda item: next(item.actor.parameters()).requires_grad_(False),
            lambda item: next(
                item.critics.critic_1.parameters()
            ).requires_grad_(False),
            lambda item: next(
                item.critics.target_1.parameters()
            ).requires_grad_(True),
            lambda item: object.__setattr__(item.config, "tau", float("nan")),
            lambda item: object.__setattr__(item.config, "tau", 0.25),
            lambda item: object.__setattr__(
                item.actor.config, "log_std_min", float("nan")
            ),
        ):
            instance = build_trainer(batch)
            mutate(instance)
            self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

    def test_post_preflight_exception_rolls_back_the_whole_update(self) -> None:
        batch = make_batch(6)
        instance = build_trainer(batch)
        real_objective = trainer.actor_objective

        def advance_rng_then_fail(*args, **kwargs):
            real_objective(*args, **kwargs)
            raise RuntimeError("injected post-critic failure")

        with mock.patch.object(
            trainer, "actor_objective", side_effect=advance_rng_then_fail
        ):
            self.assert_zero_mutation(instance, batch, RuntimeError)

    def test_mutated_actor_support_and_expected_binding_are_detected(self) -> None:
        batch = make_batch(4)
        instance = build_trainer(batch)
        instance.actor._support_q_e4_lower[0] += 1
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)

        instance = build_trainer(batch)
        original = instance.expected_binding.float_dtype
        object.__setattr__(instance.expected_binding, "float_dtype", "torch.float64")
        self.assert_zero_mutation(instance, batch, trainer.TrainerStateError)
        object.__setattr__(instance.expected_binding, "float_dtype", original)


class ScopeSentinelTest(unittest.TestCase):
    def test_update_source_has_no_trajectory_or_bootstrap_path(self) -> None:
        source = inspect.getsource(
            trainer.EmpiricalTerminalHybridSacTrainerV1._update_once_after_preflight
        )
        forbidden = (
            "soft_state_value",
            "gamma",
            "discount",
            "duration",
            "next_state",
            "torch.pow",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("target = reward.clone()", source)

    def test_batch_schema_has_no_production_or_trajectory_fields(self) -> None:
        names = set(replay.EmpiricalTerminalBatchV1.__dataclass_fields__)
        self.assertEqual(
            names,
            {
                "_state",
                "_mode_id",
                "_q_e4",
                "_reward",
                "binding",
                "audit",
                "float_dtype",
            },
        )


if __name__ == "__main__":
    unittest.main()
