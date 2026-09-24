"""Adversarial tests for the verifier-gated Run-4 SAC trainer."""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import inspect
import math
import random
import socket
import subprocess
import textwrap
import types
import unittest
from unittest import mock

import torch

from . import trainer as subject
from .models import build_run4_models
from .replay import ReplayBindingV1, ReplayTensorBatchV1
from .run4_contract import POLICY_FEATURE_COUNT


def _binding(*, gamma: float = 0.99, discriminator: str = "a") -> ReplayBindingV1:
    return ReplayBindingV1._for_test_only(
        gamma=gamma,
        freshness_policy_sha256=discriminator * 64,
        empirical_scaling_sha256="b" * 64,
        calibration_evidence_sha256="c" * 64,
        queue_kernel_evidence_sha256="d" * 64,
    )


def _batch(
    binding: ReplayBindingV1,
    *,
    size: int = 4,
    successor: bool = True,
    discount: float = 0.99**2,
) -> ReplayTensorBatchV1:
    state = torch.linspace(
        -0.4,
        0.8,
        steps=size * POLICY_FEATURE_COUNT,
        dtype=torch.float32,
    ).reshape(size, POLICY_FEATURE_COUNT)
    next_state = state.flip(1).clone() if successor else torch.zeros_like(state)
    has_next = torch.full((size,), successor, dtype=torch.bool)
    terminated = torch.full((size,), not successor, dtype=torch.bool)
    return ReplayTensorBatchV1(
        _state=state,
        _next_state=next_state,
        _mode_id=torch.arange(size, dtype=torch.int64) % 12,
        _q_e4=torch.arange(3000, 3000 + size, dtype=torch.int64),
        _reward=torch.linspace(-0.2, 0.3, steps=size, dtype=torch.float32),
        _duration=torch.full((size,), 2, dtype=torch.int64),
        _discount=torch.full((size,), discount, dtype=torch.float32),
        _has_next_state=has_next,
        _bootstrap=has_next.clone(),
        _terminated=terminated,
        _truncated=torch.zeros((size,), dtype=torch.bool),
        _binding=binding,
        _audit=tuple({"row": index} for index in range(size)),
    )


def _sequential_batch(binding: ReplayBindingV1) -> ReplayTensorBatchV1:
    """Two linked decisions with unchanged current telemetry and new history."""

    external = torch.tensor([0.1, 0.2, 12.0 / 28.0, 0.4])

    def features(
        *, mode: int, q_e4: int, quality: float, latency_fraction: float
    ) -> torch.Tensor:
        result = torch.zeros(POLICY_FEATURE_COUNT, dtype=torch.float32)
        result[:4] = external
        result[4 + mode] = 1.0
        result[16] = q_e4 / 9800.0
        result[17] = quality
        result[18] = latency_fraction
        result[19] = 1.0
        result[20] = 1.0
        return result

    state_0 = features(
        mode=4, q_e4=4500, quality=0.55, latency_fraction=0.6
    )
    # This state is the exact successor of decision 0 below: current telemetry
    # is unchanged, while the immediately previous action/outcome has changed.
    state_1 = features(
        mode=0, q_e4=3000, quality=0.5, latency_fraction=0.5
    )
    state_2 = features(
        mode=1, q_e4=3001, quality=0.4, latency_fraction=0.4
    )
    return ReplayTensorBatchV1(
        _state=torch.stack((state_0, state_1)),
        _next_state=torch.stack((state_1, state_2)),
        _mode_id=torch.tensor([0, 1], dtype=torch.int64),
        _q_e4=torch.tensor([3000, 3001], dtype=torch.int64),
        _reward=torch.tensor([0.375, 0.3], dtype=torch.float32),
        _duration=torch.tensor([2, 2], dtype=torch.int64),
        _discount=torch.tensor([0.99**2, 0.99**2], dtype=torch.float32),
        _has_next_state=torch.tensor([True, True]),
        _bootstrap=torch.tensor([True, True]),
        _terminated=torch.tensor([False, False]),
        _truncated=torch.tensor([False, False]),
        _binding=binding,
        _audit=({"decision": 0}, {"decision": 1}),
    )


def _clone_optimizer_state(optimizer: torch.optim.Optimizer):
    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return copy.deepcopy(value)

    return clone(optimizer.state_dict())


def _assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class TrainerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = _binding()
        bundle = build_run4_models(actor_seed=17, critic_seed=29)
        self.actor = bundle.actor
        self.critics = bundle.critics
        self.target_generator = torch.Generator(device="cpu").manual_seed(101)
        self.actor_generator = torch.Generator(device="cpu").manual_seed(202)
        self.config = subject.TrainerConfigV1(alpha_d=0.05, alpha_c=0.05)

    def mechanics_trainer(self) -> subject._TestOnlyRun4HybridSacTrainerV1:
        return subject._TestOnlyRun4HybridSacTrainerV1(
            actor=self.actor,
            critics=self.critics,
            config=self.config,
            expected_binding=self.binding,
            target_generator=self.target_generator,
            actor_generator=self.actor_generator,
        )


class ProductionEvidenceGateTest(TrainerFixture):
    def test_production_constructor_rejects_test_only_evidence(self) -> None:
        self.assertEqual(self.binding.evidence_eligibility, "TEST_ONLY_MECHANICS")
        with self.assertRaisesRegex(Exception, "verifier-attested"):
            subject.Run4HybridSacTrainerV1(
                actor=self.actor,
                critics=self.critics,
                config=self.config,
                expected_binding=self.binding,
                target_generator=self.target_generator,
                actor_generator=self.actor_generator,
            )

    def test_test_harness_is_private_and_labelled(self) -> None:
        self.assertNotIn(
            "_TestOnlyRun4HybridSacTrainerV1", subject.__all__
        )
        trainer = self.mechanics_trainer()
        self.assertEqual(
            trainer.expected_binding.evidence_eligibility,
            "TEST_ONLY_MECHANICS",
        )


class ImportPurityTest(unittest.TestCase):
    def test_import_has_no_io_rng_process_socket_or_cuda_side_effect(self) -> None:
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.is_initialized()
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem access")
        ) as opened, mock.patch.object(random, "seed") as seeded, mock.patch.object(
            random, "random"
        ) as sampled, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process launched")
        ) as popen:
            importlib.reload(subject)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(cuda_before, torch.cuda.is_initialized())


class BindingAndPreflightTest(TrainerFixture):
    def test_exact_21_feature_width_is_enforced(self) -> None:
        trainer = self.mechanics_trainer()
        batch = _batch(self.binding)
        object.__setattr__(
            batch,
            "_state",
            torch.zeros((batch.batch_size, 20), dtype=torch.float32),
        )
        with self.assertRaisesRegex(subject.TrainerPreflightError, "state"):
            trainer.update_once(batch)

    def test_different_evidence_binding_is_rejected(self) -> None:
        trainer = self.mechanics_trainer()
        foreign = _binding(discriminator="e")
        with self.assertRaisesRegex(
            subject.TrainerPreflightError, "evidence/binding"
        ):
            trainer.update_once(_batch(foreign))

    def test_ineligible_bootstrap_mask_is_rejected(self) -> None:
        trainer = self.mechanics_trainer()
        batch = _batch(self.binding, successor=False)
        object.__setattr__(
            batch,
            "_bootstrap",
            torch.ones((batch.batch_size,), dtype=torch.bool),
        )
        with self.assertRaisesRegex(
            subject.TrainerPreflightError, "real-successor"
        ):
            trainer.update_once(batch)

    def test_preflight_failure_mutates_nothing(self) -> None:
        trainer = self.mechanics_trainer()
        trainer.update_once(_batch(self.binding))
        parameters = list(
            chain_parameters(trainer.actor, trainer.critics)
        )
        before_parameters = [item.detach().clone() for item in parameters]
        before_actor_optimizer = _clone_optimizer_state(trainer.actor_optimizer)
        before_critic_optimizer = _clone_optimizer_state(trainer.critic_optimizer)
        before_target_rng = trainer._target_generator.get_state().clone()
        before_actor_rng = trainer._actor_generator.get_state().clone()
        before_count = trainer.update_count

        malformed = _batch(self.binding)
        object.__setattr__(
            malformed,
            "_duration",
            torch.ones((malformed.batch_size,), dtype=torch.int64),
        )
        with self.assertRaises(subject.TrainerPreflightError):
            trainer.update_once(malformed)

        for before, after in zip(before_parameters, parameters):
            self.assertTrue(torch.equal(before, after))
        _assert_nested_equal(
            self, before_actor_optimizer, trainer.actor_optimizer.state_dict()
        )
        _assert_nested_equal(
            self, before_critic_optimizer, trainer.critic_optimizer.state_dict()
        )
        self.assertTrue(
            torch.equal(before_target_rng, trainer._target_generator.get_state())
        )
        self.assertTrue(
            torch.equal(before_actor_rng, trainer._actor_generator.get_state())
        )
        self.assertEqual(before_count, trainer.update_count)

    def test_optimizer_target_foreign_and_duplicate_parameters_are_refused(self) -> None:
        def fresh(seed: int):
            binding = _binding()
            bundle = build_run4_models(actor_seed=seed, critic_seed=seed + 100)
            result = subject._TestOnlyRun4HybridSacTrainerV1(
                actor=bundle.actor,
                critics=bundle.critics,
                config=subject.TrainerConfigV1(alpha_d=0.05, alpha_c=0.05),
                expected_binding=binding,
                target_generator=torch.Generator(device="cpu").manual_seed(seed),
                actor_generator=torch.Generator(device="cpu").manual_seed(seed + 1),
            )
            return binding, result

        for index, case in enumerate(("target", "foreign", "duplicate"), 1):
            with self.subTest(case=case):
                binding, trainer = fresh(300 + index)
                if case == "target":
                    injected = next(trainer.critics.target_1.parameters())
                elif case == "foreign":
                    injected = torch.nn.Parameter(torch.zeros(1))
                else:
                    injected = trainer.actor_optimizer.param_groups[0]["params"][0]
                trainer.actor_optimizer.param_groups[0]["params"].append(injected)
                parameters = list(chain_parameters(trainer.actor, trainer.critics))
                before = [item.detach().clone() for item in parameters]
                target_rng = trainer._target_generator.get_state().clone()
                actor_rng = trainer._actor_generator.get_state().clone()
                with self.assertRaises(subject.TrainerStateError):
                    trainer.update_once(_batch(binding))
                for old, new in zip(before, parameters):
                    self.assertTrue(torch.equal(old, new))
                self.assertEqual(trainer.update_count, 0)
                self.assertEqual(len(trainer.actor_optimizer.state), 0)
                self.assertEqual(len(trainer.critic_optimizer.state), 0)
                self.assertTrue(
                    torch.equal(target_rng, trainer._target_generator.get_state())
                )
                self.assertTrue(
                    torch.equal(actor_rng, trainer._actor_generator.get_state())
                )


def chain_parameters(actor, critics):
    yield from actor.parameters()
    yield from critics.critic_1.parameters()
    yield from critics.critic_2.parameters()
    yield from critics.target_1.parameters()
    yield from critics.target_2.parameters()


class UpdateEquationTest(TrainerFixture):
    def test_next_action_consumes_exact_previous_action_and_outcome_slots(self) -> None:
        trainer = self.mechanics_trainer()
        batch = _sequential_batch(self.binding)
        self.assertTrue(torch.equal(batch.next_state[0], batch.state[1]))
        self.assertTrue(torch.equal(batch.state[0, :4], batch.state[1, :4]))
        self.assertFalse(torch.equal(batch.state[0, 4:], batch.state[1, 4:]))
        self.assertTrue(bool((batch.state[:, 19] == 1.0).all()))
        self.assertTrue(bool((batch.state[:, 20] == 1.0).all()))
        self.assertEqual(tuple(batch.state.shape), (2, 21))

        with mock.patch.object(
            subject,
            "actor_objective",
            wraps=subject.actor_objective,
        ) as actor_call, mock.patch.object(
            subject,
            "soft_state_value",
            wraps=subject.soft_state_value,
        ) as target_call:
            trainer.update_once(batch)
        actor_input = actor_call.call_args.args[2]
        target_input = target_call.call_args.args[2]
        self.assertTrue(torch.equal(actor_input, batch.state))
        self.assertTrue(torch.equal(target_input, batch.next_state))

    def test_stored_discount_is_used_verbatim(self) -> None:
        self.binding = _binding(gamma=0.99999999)
        trainer = self.mechanics_trainer()
        emitted = float(torch.tensor(0.9999985098838806, dtype=torch.float32))
        batch = _batch(self.binding, size=1, discount=emitted)
        fixed_value = torch.tensor([1_000_000.5], dtype=torch.float32)
        expected = float(
            batch.reward[0] + batch.discount()[0] * fixed_value[0]
        )
        dtype_first_wrong = float(
            batch.reward[0]
            + torch.tensor(self.binding.gamma, dtype=torch.float32)
            * fixed_value[0]
        )
        self.assertGreater(abs(expected - dtype_first_wrong), 1.0)

        replacement = types.SimpleNamespace(value=fixed_value)
        with mock.patch.object(subject, "soft_state_value", return_value=replacement):
            metrics = trainer.update_once(batch)
        self.assertEqual(metrics.target_mean, expected)
        self.assertNotEqual(metrics.target_mean, dtype_first_wrong)

    def test_source_has_no_discount_exponentiation_or_legacy_target(self) -> None:
        tree = ast.parse(inspect.getsource(subject))
        update_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(subject._Run4TrainerCore.update_once)
            )
        )
        update_names = {
            node.id for node in ast.walk(update_tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("gamma", update_names)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("pow", calls)
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("critic_target", names)

    def test_no_successor_never_reaches_soft_value(self) -> None:
        trainer = self.mechanics_trainer()
        batch = _batch(self.binding, successor=False)
        with mock.patch.object(
            subject,
            "soft_state_value",
            side_effect=AssertionError("sentinel was evaluated"),
        ) as target_call:
            metrics = trainer.update_once(batch)
        target_call.assert_not_called()
        self.assertEqual(metrics.bootstrap_count, 0)
        self.assertAlmostEqual(metrics.target_mean, float(batch.reward.mean()))

    def test_actor_receives_q_gradient_without_critic_grad_accumulation(self) -> None:
        trainer = self.mechanics_trainer()
        metrics = trainer.update_once(_batch(self.binding, size=8))
        self.assertGreater(metrics.actor_delta_norm, 0.0)
        self.assertGreater(metrics.critic_delta_norm, 0.0)
        self.assertGreater(metrics.target_delta_norm, 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in list(trainer.critics.critic_1.parameters())
                + list(trainer.critics.critic_2.parameters())
            )
        )
        for head_name in ("mean_head", "log_std_head"):
            gradients = [
                parameter.grad
                for parameter in getattr(trainer.actor, head_name).parameters()
            ]
            self.assertTrue(all(item is not None for item in gradients))
            self.assertGreater(
                sum(float(item.detach().abs().sum()) for item in gradients), 0.0
            )
            weight_gradient = getattr(trainer.actor, head_name).weight.grad
            self.assertIsNotNone(weight_gradient)
            row_norms = weight_gradient.detach().abs().sum(dim=1)
            self.assertEqual(tuple(row_norms.shape), (12,))
            self.assertTrue(bool((row_norms > 0.0).all()))

    def test_finite_reward_with_overflowing_mse_is_refused_before_step(self) -> None:
        trainer = self.mechanics_trainer()
        batch = _batch(self.binding, size=2, successor=False)
        object.__setattr__(
            batch,
            "_reward",
            torch.full((2,), 1e20, dtype=torch.float32),
        )
        parameters = list(chain_parameters(trainer.actor, trainer.critics))
        before = [item.detach().clone() for item in parameters]
        target_rng = trainer._target_generator.get_state().clone()
        actor_rng = trainer._actor_generator.get_state().clone()
        with self.assertRaisesRegex(subject.TrainerError, "critic loss"):
            trainer.update_once(batch)
        for old, new in zip(before, parameters):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(trainer.update_count, 0)
        self.assertEqual(len(trainer.actor_optimizer.state), 0)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)
        self.assertTrue(
            torch.equal(target_rng, trainer._target_generator.get_state())
        )
        self.assertTrue(torch.equal(actor_rng, trainer._actor_generator.get_state()))


class ConfigurationTest(unittest.TestCase):
    def test_configuration_rejects_non_float32_or_invalid_values(self) -> None:
        with self.assertRaises(subject.TrainerStateError):
            subject.TrainerConfigV1(
                alpha_d=0.1, alpha_c=0.1, float_dtype=torch.float64
            )
        for name in ("alpha_d", "alpha_c", "tau", "actor_lr", "critic_lr"):
            values = dict(
                alpha_d=0.1,
                alpha_c=0.1,
                tau=0.005,
                actor_lr=3e-4,
                critic_lr=3e-4,
            )
            values[name] = math.nan
            with self.subTest(name=name), self.assertRaises(
                subject.TrainerStateError
            ):
                subject.TrainerConfigV1(**values)

    def test_default_and_shared_generators_are_rejected(self) -> None:
        binding = _binding()
        for case in ("default", "shared"):
            with self.subTest(case=case):
                bundle = build_run4_models(actor_seed=41, critic_seed=42)
                local = torch.Generator(device="cpu").manual_seed(7)
                if case == "default":
                    target, actor = torch.default_generator, local
                else:
                    target = actor = local
                with self.assertRaises(subject.TrainerStateError):
                    subject._TestOnlyRun4HybridSacTrainerV1(
                        actor=bundle.actor,
                        critics=bundle.critics,
                        config=subject.TrainerConfigV1(
                            alpha_d=0.05, alpha_c=0.05
                        ),
                        expected_binding=binding,
                        target_generator=target,
                        actor_generator=actor,
                    )


if __name__ == "__main__":
    unittest.main()
