"""Focused CPU-only deterministic tests for the Hybrid-SAC neural models.

Phase 4b.1 -- ``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.  These tests qualify
shapes, the probability mathematics, the executed-``q`` quantization boundary,
the exact 12-mode enumeration, gradient routing and fail-closed behaviour.
They train nothing, read no evidence and launch no runtime.

Every test runs on CPU in float64.  Hand calculations are recomputed with the
standard library rather than with the module under test, so a sign error in
the Jacobian or the soft value cannot agree with itself.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from . import action_contract as ac
from . import hybrid_sac_models as hsm
from .state_reward_transition_contract import POLICY_FEATURE_COUNT

DTYPE = torch.float64
REFERENCE_CONFIG = hsm.HybridSacModelConfig(dtype=DTYPE)


def _state(batch: int = 5, seed: int = 17) -> torch.Tensor:
    """A deterministic finite policy-state batch."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch, hsm.STATE_DIM, dtype=DTYPE, generator=generator
    )


class ModelsTestBase(unittest.TestCase):
    """Shared deterministic actor/critic construction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.actor = hsm.build_actor(REFERENCE_CONFIG, seed=1234)
        cls.critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        cls.state = _state()

    def generator(self, seed: int = 99) -> torch.Generator:
        return torch.Generator().manual_seed(seed)


class ContractBindingTest(ModelsTestBase):
    """Dimensions come from the frozen contracts, not from local literals."""

    def test_dimensions_are_bound_to_the_frozen_contracts(self) -> None:
        self.assertEqual(hsm.STATE_DIM, POLICY_FEATURE_COUNT)
        self.assertEqual(hsm.STATE_DIM, 31)
        self.assertEqual(hsm.MODE_COUNT, ac.EXPECTED_MODE_COUNT)
        self.assertEqual(hsm.MODE_COUNT, 12)
        self.assertEqual(hsm.CRITIC_INPUT_DIM, 31 + 12 + 1)
        self.assertEqual(hsm.Q_SQUASH_SCALE, ac.Q_MAX / 2.0)
        self.assertEqual(hsm.Q_SQUASH_SCALE, 0.49)
        self.assertEqual(hsm.PHASE_LABEL, "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY")
        self.assertEqual((hsm.LOG_STD_MIN, hsm.LOG_STD_MAX), (-5.0, 2.0))

    def test_default_training_dtype_is_float32(self) -> None:
        config = hsm.HybridSacModelConfig()
        self.assertEqual(config.dtype, torch.float32)
        actor = hsm.build_actor(config, seed=7)
        critics = hsm.build_twin_critics(config, seed=8)
        self.assertTrue(all(p.dtype == torch.float32 for p in actor.parameters()))
        self.assertTrue(
            all(p.dtype == torch.float32 for p in critics.parameters())
        )

    def test_architecture_is_two_hidden_layers_of_128(self) -> None:
        linear = [m for m in self.actor.encoder if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear), 2)
        self.assertEqual(linear[0].in_features, 31)
        self.assertEqual(linear[0].out_features, 128)
        self.assertEqual(linear[1].in_features, 128)
        self.assertEqual(linear[1].out_features, 128)
        for head in (
            self.actor.logit_head,
            self.actor.mean_head,
            self.actor.log_std_head,
        ):
            self.assertEqual(head.in_features, 128)
            self.assertEqual(head.out_features, 12)

        critic_linear = [
            m for m in self.critics.critic_1.trunk if isinstance(m, torch.nn.Linear)
        ]
        self.assertEqual(len(critic_linear), 2)
        self.assertEqual(critic_linear[0].in_features, 44)
        self.assertEqual(self.critics.critic_1.value_head.out_features, 1)

    def test_parameter_counts_are_exact(self) -> None:
        actor_params = sum(p.numel() for p in self.actor.parameters())
        # 31*128+128 + 128*128+128 + 3 * (128*12+12)
        self.assertEqual(actor_params, 4096 + 16512 + 3 * 1548)
        self.assertEqual(actor_params, 25252)

        critic_params = sum(p.numel() for p in self.critics.critic_1.parameters())
        # 44*128+128 + 128*128+128 + 128*1+1
        self.assertEqual(critic_params, 5760 + 16512 + 129)
        self.assertEqual(critic_params, 22401)
        self.assertEqual(
            sum(p.numel() for p in self.critics.critic_2.parameters()), 22401
        )
        trainable = sum(
            p.numel() for p in self.critics.parameters() if p.requires_grad
        )
        self.assertEqual(trainable, 2 * 22401)
        self.assertEqual(sum(p.numel() for p in self.critics.parameters()), 4 * 22401)

    def test_models_stay_on_cpu(self) -> None:
        for parameter in list(self.actor.parameters()) + list(
            self.critics.parameters()
        ):
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.dtype, DTYPE)


class ShapeTest(ModelsTestBase):
    """Exact actor and critic tensor shapes."""

    def test_actor_head_shapes(self) -> None:
        for batch in (1, 5, 16):
            heads = self.actor(_state(batch))
            self.assertEqual(tuple(heads.logits.shape), (batch, 12))
            self.assertEqual(tuple(heads.mean.shape), (batch, 12))
            self.assertEqual(tuple(heads.log_std.shape), (batch, 12))

    def test_sample_shapes_cover_every_mode(self) -> None:
        sample = self.actor.sample_all_modes(
            self.state, generator=self.generator()
        )
        batch = self.state.shape[0]
        for name in (
            "pre_squash",
            "q",
            "q_e4",
            "q_normalized_executed",
            "q_normalized_straight_through",
            "log_prob_continuous",
            "log_prob_discrete",
            "probs",
        ):
            self.assertEqual(
                tuple(getattr(sample, name).shape), (batch, 12), f"{name} shape"
            )
        self.assertEqual(sample.q_e4.dtype, torch.long)

    def test_critic_shapes(self) -> None:
        batch = self.state.shape[0]
        modes = torch.zeros(batch, dtype=torch.long)
        one_hot = hsm.mode_one_hot(modes, dtype=DTYPE)
        self.assertEqual(tuple(one_hot.shape), (batch, 12))
        q_norm = torch.full((batch,), 0.5, dtype=DTYPE)
        q1, q2 = self.critics.q_values(self.state, one_hot, q_norm)
        self.assertEqual(tuple(q1.shape), (batch,))
        self.assertEqual(tuple(q2.shape), (batch,))

        q_all = torch.full((batch, 12), 0.5, dtype=DTYPE)
        self.assertEqual(
            tuple(self.critics.critic_1.q_all_modes(self.state, q_all).shape),
            (batch, 12),
        )
        self.assertEqual(
            tuple(self.critics.min_q_all_modes(self.state, q_all).shape),
            (batch, 12),
        )

    def test_q_all_modes_agrees_with_per_mode_forward(self) -> None:
        batch = self.state.shape[0]
        q_all = torch.rand(batch, 12, dtype=DTYPE, generator=self.generator(4))
        enumerated = self.critics.critic_1.q_all_modes(self.state, q_all)
        for mode in range(12):
            index = torch.full((batch,), mode, dtype=torch.long)
            single = self.critics.critic_1(
                self.state, hsm.mode_one_hot(index, dtype=DTYPE), q_all[:, mode]
            )
            torch.testing.assert_close(enumerated[:, mode], single)


class CategoricalTest(ModelsTestBase):
    """``pi_d`` is a proper categorical distribution over the 12 modes."""

    def test_probabilities_are_finite_nonnegative_and_normalized(self) -> None:
        for seed in (1, 2, 3):
            state = _state(8, seed=seed)
            log_probs, probs = self.actor.mode_log_probs(state)
            self.assertEqual(tuple(probs.shape), (8, 12))
            self.assertTrue(torch.isfinite(probs).all())
            self.assertTrue(torch.isfinite(log_probs).all())
            self.assertTrue((probs >= 0.0).all())
            self.assertTrue((probs <= 1.0).all())
            torch.testing.assert_close(
                probs.sum(dim=-1), torch.ones(8, dtype=DTYPE)
            )
            torch.testing.assert_close(log_probs.exp(), probs)

    def test_extreme_logits_stay_normalized(self) -> None:
        # A saturated state must not produce NaN probabilities.
        state = torch.full((3, hsm.STATE_DIM), 50.0, dtype=DTYPE)
        _, probs = self.actor.mode_log_probs(state)
        self.assertTrue(torch.isfinite(probs).all())
        torch.testing.assert_close(
            probs.sum(dim=-1), torch.ones(3, dtype=DTYPE)
        )


class BoundedQualityTest(ModelsTestBase):
    """Sampled ``q`` and executed ``q_e4`` never leave the registered range."""

    def test_sampled_and_executed_quality_are_bounded(self) -> None:
        for seed in range(6):
            sample = self.actor.sample_all_modes(
                _state(8, seed=seed), generator=self.generator(seed)
            )
            self.assertTrue((sample.q >= ac.Q_MIN).all())
            self.assertTrue((sample.q <= ac.Q_MAX).all())
            self.assertTrue((sample.q_e4 >= ac.Q_E4_MIN).all())
            self.assertTrue((sample.q_e4 <= ac.Q_E4_MAX).all())
            self.assertTrue((sample.q_normalized_executed >= 0.0).all())
            self.assertTrue((sample.q_normalized_executed <= 1.0).all())

    def test_saturated_pre_squash_stays_in_range(self) -> None:
        # +/- 40 saturates tanh to +/- 1 in float64; q must clamp to the bounds
        # rather than overshoot them.
        heads = hsm.ActorHeads(
            logits=torch.zeros(1, 12, dtype=DTYPE),
            mean=torch.zeros(1, 12, dtype=DTYPE),
            log_std=torch.zeros(1, 12, dtype=DTYPE),
        )
        pre = torch.tensor(
            [[-40.0, 40.0] + [0.0] * 10], dtype=DTYPE
        )
        sample = self.actor._finish_sample(heads, pre)
        self.assertAlmostEqual(float(sample.q[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(sample.q[0, 1]), ac.Q_MAX, places=12)
        self.assertEqual(int(sample.q_e4[0, 0]), 0)
        self.assertEqual(int(sample.q_e4[0, 1]), 9800)


class QuantizationTest(ModelsTestBase):
    """Tensor quantization is the registered contract rule, not a copy of it."""

    #: Boundary values and exact five-decimal ties.
    CASES = (
        0.0,
        0.00005,
        0.0001,
        0.00014999,
        0.12345,
        0.29995,
        0.3,
        0.30005,
        0.5,
        0.54325,
        0.70005,
        0.9,
        0.97995,
        0.98,
        0.9800499,
    )

    def test_tensor_quantization_matches_the_contract_on_boundaries_and_ties(
        self,
    ) -> None:
        tensor = torch.tensor(self.CASES, dtype=DTYPE)
        produced = hsm.quantize_q_e4(tensor)
        expected = [ac.round_half_up_q_e4(value) for value in self.CASES]
        self.assertEqual(produced.tolist(), expected)
        # Spot-check that decimal half-up really is exercised: these are the
        # cases a naive floor(10000*q + 0.5) would get wrong.
        self.assertEqual(ac.round_half_up_q_e4(0.70005), 7001)
        self.assertEqual(int(produced[self.CASES.index(0.70005)]), 7001)
        self.assertEqual(int(produced[self.CASES.index(0.12345)]), 1235)

    def test_quantization_clips_to_the_mechanical_bounds(self) -> None:
        tensor = torch.tensor([-1.0, -0.0001, 0.99, 5.0], dtype=DTYPE)
        self.assertEqual(hsm.quantize_q_e4(tensor).tolist(), [0, 0, 9800, 9800])

    def test_quantization_preserves_shape_and_dtype(self) -> None:
        tensor = torch.rand(4, 12, dtype=DTYPE, generator=self.generator(2)) * 0.98
        produced = hsm.quantize_q_e4(tensor)
        self.assertEqual(tuple(produced.shape), (4, 12))
        self.assertEqual(produced.dtype, torch.long)

    def test_float32_fast_path_matches_contract_around_every_half_grid(self) -> None:
        # For each q_e4 half-step, exercise the nearest representable float32
        # value and both adjacent floats.  Expected values are computed from
        # the tensor's actual binary value, not from a decimal literal that
        # float32 cannot represent.
        thresholds = (
            torch.arange(0, ac.Q_E4_MAX, dtype=torch.float64) + 0.5
        ) / float(ac.Q_E4_SCALE)
        center = thresholds.to(torch.float32)
        below = torch.nextafter(center, torch.full_like(center, float("-inf")))
        above = torch.nextafter(center, torch.full_like(center, float("inf")))
        values = torch.stack((below, center, above), dim=1).reshape(-1)
        expected = torch.tensor(
            [ac.round_half_up_q_e4(float(value)) for value in values],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(hsm._quantize_q_e4_training(values), expected))

    def test_float32_execution_boundary_delegates_to_contract(self) -> None:
        values = (
            torch.rand(1024, dtype=torch.float32, generator=self.generator(91))
            * ac.Q_MAX
        )
        expected = [ac.round_half_up_q_e4(float(value)) for value in values]
        self.assertEqual(hsm.quantize_q_e4(values).tolist(), expected)

    def test_straight_through_forward_is_the_exact_executed_value(self) -> None:
        sample = self.actor.sample_all_modes(
            self.state, generator=self.generator(8)
        )
        exact = sample.q_e4.to(DTYPE) / 9800.0
        torch.testing.assert_close(
            sample.q_normalized_straight_through, exact, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            sample.q_normalized_executed, exact, rtol=0.0, atol=0.0
        )
        # The executed value is generally NOT the unquantized request.
        self.assertFalse(
            torch.equal(sample.q_normalized_straight_through, sample.q / ac.Q_MAX)
        )
        # But it still carries a gradient path back to the actor.
        self.assertTrue(sample.q_normalized_straight_through.requires_grad)

    def test_straight_through_gradient_is_the_identity_gradient(self) -> None:
        q = torch.tensor([0.4321], dtype=DTYPE, requires_grad=True)
        q_e4 = hsm.quantize_q_e4(q)
        st = hsm._straight_through_executed_normalized_q(q, q_e4)
        st.sum().backward()
        # d(q / 0.98)/dq = 1 / 0.98
        self.assertAlmostEqual(float(q.grad), 1.0 / ac.Q_MAX, places=12)


class DeterministicExecutionTest(ModelsTestBase):
    """Evaluation selects the argmax mode at its conditional mean."""

    def test_deterministic_execution_uses_argmax_mode_and_mean(self) -> None:
        with torch.no_grad():
            heads = self.actor(self.state)
        execution = self.actor.deterministic_execution(self.state)
        expected_mode = torch.argmax(heads.logits, dim=-1)
        self.assertTrue(torch.equal(execution.mode_index, expected_mode))

        for index in range(self.state.shape[0]):
            mode = int(expected_mode[index])
            mean = float(heads.mean[index, mode])
            expected_q = 0.49 * (math.tanh(mean) + 1.0)
            self.assertAlmostEqual(float(execution.q[index]), expected_q, places=12)
            self.assertEqual(
                int(execution.q_e4[index]), ac.round_half_up_q_e4(expected_q)
            )
            self.assertAlmostEqual(
                float(execution.q_executed[index]),
                int(execution.q_e4[index]) / 10000.0,
                places=12,
            )
            self.assertAlmostEqual(
                float(execution.q_normalized_executed[index]),
                int(execution.q_e4[index]) / 9800.0,
                places=12,
            )

    def test_deterministic_execution_is_repeatable_and_gradient_free(self) -> None:
        first = self.actor.deterministic_execution(self.state)
        second = self.actor.deterministic_execution(self.state)
        self.assertTrue(torch.equal(first.mode_index, second.mode_index))
        self.assertTrue(torch.equal(first.q_e4, second.q_e4))
        self.assertFalse(first.q.requires_grad)


class ContinuousLogProbTest(ModelsTestBase):
    """The transformed density carries the complete tanh and 0.49 Jacobian."""

    def test_matches_a_hand_calculation(self) -> None:
        mean, log_std, pre = 0.3, -0.5, 0.8
        produced = float(
            hsm.continuous_log_prob(
                torch.tensor([pre], dtype=DTYPE),
                torch.tensor([mean], dtype=DTYPE),
                torch.tensor([log_std], dtype=DTYPE),
            )
        )
        # Recomputed independently with the standard library.
        sigma = math.exp(log_std)
        z = (pre - mean) / sigma
        log_normal = -0.5 * z * z - log_std - 0.5 * math.log(2.0 * math.pi)
        jacobian = math.log(0.49) + math.log(1.0 - math.tanh(pre) ** 2)
        self.assertAlmostEqual(produced, log_normal - jacobian, places=12)

    def test_matches_hand_calculation_across_a_grid(self) -> None:
        for mean in (-1.5, 0.0, 2.0):
            for log_std in (-2.0, 0.0, 1.0):
                for pre in (-3.0, -0.25, 0.0, 1.75):
                    produced = float(
                        hsm.continuous_log_prob(
                            torch.tensor([pre], dtype=DTYPE),
                            torch.tensor([mean], dtype=DTYPE),
                            torch.tensor([log_std], dtype=DTYPE),
                        )
                    )
                    sigma = math.exp(log_std)
                    z = (pre - mean) / sigma
                    expected = (
                        -0.5 * z * z
                        - log_std
                        - 0.5 * math.log(2.0 * math.pi)
                        - math.log(0.49)
                        - math.log(1.0 - math.tanh(pre) ** 2)
                    )
                    self.assertAlmostEqual(produced, expected, places=10)

    def test_jacobian_is_stable_for_saturated_samples(self) -> None:
        # Direct log(1 - tanh(u)^2) underflows near |u| = 20; the stable form
        # must stay finite.
        pre = torch.tensor([-30.0, -20.0, 20.0, 30.0], dtype=DTYPE)
        produced = hsm.continuous_log_prob(
            pre, torch.zeros(4, dtype=DTYPE), torch.zeros(4, dtype=DTYPE)
        )
        self.assertTrue(torch.isfinite(produced).all())

    def test_omitting_the_scale_term_would_change_the_result(self) -> None:
        # Guards against a Jacobian that forgets the 0.49 scale: the constant
        # is -log(0.49) = 0.7133...
        produced = float(
            hsm.continuous_log_prob(
                torch.tensor([0.0], dtype=DTYPE),
                torch.tensor([0.0], dtype=DTYPE),
                torch.tensor([0.0], dtype=DTYPE),
            )
        )
        without_scale = -0.5 * math.log(2.0 * math.pi) - math.log(1.0)
        self.assertAlmostEqual(produced - without_scale, -math.log(0.49), places=12)


class EnumerationTest(ModelsTestBase):
    """The actor objective and soft value enumerate all 12 modes exactly."""

    def test_actor_objective_enumerates_all_twelve_modes(self) -> None:
        breakdown = hsm.actor_objective(
            self.actor, self.critics, self.state, 0.2, 0.05,
            generator=self.generator(21),
        )
        batch = self.state.shape[0]
        self.assertEqual(tuple(breakdown.per_mode_term.shape), (batch, 12))
        self.assertEqual(tuple(breakdown.probs.shape), (batch, 12))
        self.assertEqual(tuple(breakdown.objective.shape), ())
        torch.testing.assert_close(
            breakdown.probs.sum(dim=-1), torch.ones(batch, dtype=DTYPE)
        )
        # The scalar really is the pi_d-weighted sum over all 12 modes.
        recomputed = (breakdown.probs * breakdown.per_mode_term).sum(dim=-1).mean()
        torch.testing.assert_close(breakdown.objective, recomputed)

    def test_every_mode_head_influences_the_objective(self) -> None:
        # Perturbing any single mode's Gaussian head must move J_pi.  If a mode
        # were dropped from the enumeration, its perturbation would be inert.
        def objective_value(actor: hsm.ConditionalHybridActor) -> float:
            return float(
                hsm.actor_objective(
                    actor, self.critics, self.state, 0.2, 0.05,
                    generator=self.generator(21),
                ).objective.detach()
            )

        baseline = objective_value(self.actor)
        for mode in range(12):
            actor = hsm.build_actor(REFERENCE_CONFIG, seed=1234)
            with torch.no_grad():
                actor.mean_head.bias[mode] += 1.0
            self.assertNotAlmostEqual(
                objective_value(actor),
                baseline,
                places=9,
                msg=f"mode {mode} does not influence the actor objective",
            )

    def test_soft_state_value_enumerates_all_twelve_modes(self) -> None:
        breakdown = hsm.soft_state_value(
            self.actor, self.critics, self.state, 0.2, 0.05,
            generator=self.generator(33),
        )
        batch = self.state.shape[0]
        self.assertEqual(tuple(breakdown.value.shape), (batch,))
        self.assertEqual(tuple(breakdown.per_mode_term.shape), (batch, 12))
        self.assertEqual(tuple(breakdown.min_target_q.shape), (batch, 12))
        torch.testing.assert_close(
            (breakdown.probs * breakdown.per_mode_term).sum(dim=-1),
            breakdown.value,
        )
        self.assertFalse(breakdown.value.requires_grad)

    def test_soft_value_matches_a_hand_assembled_enumeration(self) -> None:
        breakdown = hsm.soft_state_value(
            self.actor, self.critics, self.state, 0.3, 0.07,
            generator=self.generator(44),
        )
        # Reassemble V(s') from its published parts with explicit Python loops.
        probs = breakdown.probs
        term = breakdown.per_mode_term
        for index in range(self.state.shape[0]):
            total = 0.0
            for mode in range(12):
                total += float(probs[index, mode]) * float(term[index, mode])
            self.assertAlmostEqual(float(breakdown.value[index]), total, places=12)

    def test_soft_value_uses_the_minimum_of_the_target_twins(self) -> None:
        generator_state = self.generator(55).get_state()
        generator = torch.Generator()
        generator.set_state(generator_state)
        breakdown = hsm.soft_state_value(
            self.actor, self.critics, self.state, 0.2, 0.05, generator=generator
        )
        replay = torch.Generator()
        replay.set_state(generator_state)
        with torch.no_grad():
            sample = self.actor.sample_all_modes(self.state, generator=replay)
            t1 = self.critics.target_1.q_all_modes(
                self.state, sample.q_normalized_executed
            )
            t2 = self.critics.target_2.q_all_modes(
                self.state, sample.q_normalized_executed
            )
        torch.testing.assert_close(breakdown.min_target_q, torch.minimum(t1, t2))
        self.assertTrue((breakdown.min_target_q <= t1).all())
        self.assertTrue((breakdown.min_target_q <= t2).all())


class GradientRoutingTest(ModelsTestBase):
    """Gradients reach every actor head and never reach the targets."""

    def test_every_conditional_head_receives_finite_gradients(self) -> None:
        actor = hsm.build_actor(REFERENCE_CONFIG, seed=1234)
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        actor.zero_grad(set_to_none=True)
        breakdown = hsm.actor_objective(
            actor, critics, self.state, 0.2, 0.05, generator=self.generator(21)
        )
        breakdown.sample.q_normalized_straight_through.retain_grad()
        breakdown.objective.backward()

        q_grad = breakdown.sample.q_normalized_straight_through.grad
        self.assertIsNotNone(q_grad)
        self.assertTrue(torch.isfinite(q_grad).all())
        self.assertTrue((q_grad.abs() > 0).any())

        for name, head in (
            ("mean_head", actor.mean_head),
            ("log_std_head", actor.log_std_head),
            ("logit_head", actor.logit_head),
        ):
            self.assertIsNotNone(head.weight.grad, f"{name} weight has no grad")
            self.assertIsNotNone(head.bias.grad, f"{name} bias has no grad")
            self.assertTrue(
                torch.isfinite(head.weight.grad).all(), f"{name} weight grad"
            )
            self.assertTrue(
                torch.isfinite(head.bias.grad).all(), f"{name} bias grad"
            )
            # All 12 output units -- one per joint mode -- must be reached.
            per_mode = head.weight.grad.abs().sum(dim=1)
            self.assertEqual(tuple(per_mode.shape), (12,))
            self.assertTrue(
                (per_mode > 0).all(),
                f"{name}: modes {torch.nonzero(per_mode == 0).flatten().tolist()} "
                f"received no gradient",
            )

        for parameter in actor.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        # Actor loss must differentiate through Q with respect to q without
        # allocating or contaminating gradients on critic parameters.
        for online in (critics.critic_1, critics.critic_2):
            for parameter in online.parameters():
                self.assertTrue(parameter.requires_grad)
                self.assertIsNone(parameter.grad)

    def test_target_critics_receive_no_gradients(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        for target in (critics.target_1, critics.target_2):
            for parameter in target.parameters():
                self.assertFalse(parameter.requires_grad)

        actor = hsm.build_actor(REFERENCE_CONFIG, seed=1234)
        value = hsm.soft_state_value(
            actor, critics, self.state, 0.2, 0.05, generator=self.generator(21)
        )
        self.assertFalse(value.value.requires_grad)

        target = hsm.critic_target(
            reward=torch.zeros(self.state.shape[0], dtype=DTYPE),
            done=torch.zeros(self.state.shape[0], dtype=DTYPE),
            next_value=value.value,
            gamma=0.99,
            duration=torch.full((self.state.shape[0],), 2, dtype=torch.long),
        )
        one_hot = hsm.mode_one_hot(
            torch.zeros(self.state.shape[0], dtype=torch.long), dtype=DTYPE
        )
        q_norm = torch.full((self.state.shape[0],), 0.5, dtype=DTYPE)
        q1, q2 = critics.q_values(self.state, one_hot, q_norm)
        loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        loss.backward()

        for target_net in (critics.target_1, critics.target_2):
            for parameter in target_net.parameters():
                self.assertIsNone(parameter.grad)
        for online in (critics.critic_1, critics.critic_2):
            for parameter in online.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


class TwinIndependenceTest(ModelsTestBase):
    """The twins are genuinely two networks, not two views of one."""

    def test_twin_critics_share_no_parameter_storage(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        first = list(critics.critic_1.parameters())
        second = list(critics.critic_2.parameters())
        self.assertEqual(len(first), len(second))

        self.assertTrue(
            {id(p) for p in first}.isdisjoint({id(p) for p in second}),
            "twin critics share a parameter object",
        )
        self.assertTrue(
            {p.data_ptr() for p in first}.isdisjoint(
                {p.data_ptr() for p in second}
            ),
            "twin critics share parameter storage",
        )
        # Independently initialized, so their weights genuinely differ.
        self.assertFalse(
            any(torch.equal(a, b) for a, b in zip(first, second)),
            "twin critics were initialized identically",
        )
        # Mutating one must not touch the other.
        before = second[0].detach().clone()
        with torch.no_grad():
            first[0].add_(1.0)
        self.assertTrue(torch.equal(second[0], before))

    def test_targets_start_equal_to_their_own_online_critic(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        for online, target in (
            (critics.critic_1, critics.target_1),
            (critics.critic_2, critics.target_2),
        ):
            for online_p, target_p in zip(
                online.parameters(), target.parameters()
            ):
                self.assertTrue(torch.equal(online_p, target_p))
                self.assertNotEqual(online_p.data_ptr(), target_p.data_ptr())


class PolyakTest(ModelsTestBase):
    """The soft update matches ``tau * online + (1 - tau) * target`` exactly."""

    def test_polyak_update_matches_a_hand_computed_example(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        with torch.no_grad():
            for parameter in critics.critic_1.parameters():
                parameter.fill_(1.0)
            for parameter in critics.critic_2.parameters():
                parameter.fill_(3.0)
            for parameter in critics.target_1.parameters():
                parameter.fill_(5.0)
            for parameter in critics.target_2.parameters():
                parameter.fill_(-1.0)

        critics.polyak_update(0.25)

        # target_1 = 0.25 * 1.0 + 0.75 * 5.0 = 4.0
        for parameter in critics.target_1.parameters():
            self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, 4.0)))
        # target_2 = 0.25 * 3.0 + 0.75 * (-1.0) = 0.0
        for parameter in critics.target_2.parameters():
            self.assertTrue(torch.allclose(parameter, torch.zeros_like(parameter)))

    def test_tau_one_copies_the_online_critics(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        with torch.no_grad():
            for parameter in critics.critic_1.parameters():
                parameter.add_(0.5)
        critics.polyak_update(1.0)
        for online, target in (
            (critics.critic_1, critics.target_1),
            (critics.critic_2, critics.target_2),
        ):
            for online_p, target_p in zip(
                online.parameters(), target.parameters()
            ):
                self.assertTrue(torch.equal(online_p, target_p))

    def test_polyak_keeps_targets_gradient_free(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        critics.polyak_update(0.5)
        for target in (critics.target_1, critics.target_2):
            for parameter in target.parameters():
                self.assertFalse(parameter.requires_grad)

    def test_invalid_tau_fails_closed(self) -> None:
        critics = hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678)
        for tau in (0.0, -0.1, 1.5, float("nan"), float("inf")):
            with self.assertRaises(hsm.InvalidHyperparameterError):
                critics.polyak_update(tau)


class CriticTargetTest(ModelsTestBase):
    """The semi-Markov duration exponent is applied, not assumed to be one."""

    def test_hand_built_target_distinguishes_gamma_squared_from_cubed(self) -> None:
        next_value = torch.tensor([10.0, 10.0], dtype=DTYPE)
        reward = torch.tensor([1.0, 1.0], dtype=DTYPE)
        done = torch.zeros(2, dtype=DTYPE)
        gamma = 0.9

        two = hsm.critic_target(
            reward, done, next_value, gamma, torch.tensor([2, 2])
        )
        three = hsm.critic_target(
            reward, done, next_value, gamma, torch.tensor([3, 3])
        )
        # y = 1 + 0.81 * 10 = 9.1   versus   1 + 0.729 * 10 = 8.29
        self.assertAlmostEqual(float(two[0]), 1.0 + (0.9 ** 2) * 10.0, places=12)
        self.assertAlmostEqual(float(two[0]), 9.1, places=12)
        self.assertAlmostEqual(float(three[0]), 1.0 + (0.9 ** 3) * 10.0, places=12)
        self.assertAlmostEqual(float(three[0]), 8.29, places=12)
        self.assertNotAlmostEqual(float(two[0]), float(three[0]), places=6)

        # Mixed durations in one batch are discounted independently.
        mixed = hsm.critic_target(
            reward, done, next_value, gamma, torch.tensor([2, 3])
        )
        self.assertAlmostEqual(float(mixed[0]), 9.1, places=12)
        self.assertAlmostEqual(float(mixed[1]), 8.29, places=12)

    def test_done_suppresses_the_bootstrap(self) -> None:
        target = hsm.critic_target(
            reward=torch.tensor([2.0, 2.0], dtype=DTYPE),
            done=torch.tensor([1.0, 0.0], dtype=DTYPE),
            next_value=torch.tensor([10.0, 10.0], dtype=DTYPE),
            gamma=0.9,
            duration=torch.tensor([2, 2]),
        )
        self.assertAlmostEqual(float(target[0]), 2.0, places=12)
        self.assertAlmostEqual(float(target[1]), 2.0 + 0.81 * 10.0, places=12)

    def test_invalid_duration_and_gamma_fail_closed(self) -> None:
        reward = torch.zeros(2, dtype=DTYPE)
        done = torch.zeros(2, dtype=DTYPE)
        value = torch.ones(2, dtype=DTYPE)
        with self.assertRaises(hsm.InvalidHyperparameterError):
            hsm.critic_target(reward, done, value, 0.9, torch.tensor([0, 2]))
        with self.assertRaises(hsm.InvalidHyperparameterError):
            hsm.critic_target(reward, done, value, 0.9, torch.tensor([1, 2]))
        with self.assertRaises(hsm.InvalidHyperparameterError):
            hsm.critic_target(
                reward, done, value, 0.9, torch.tensor([2.5, 2.0], dtype=DTYPE)
            )
        for gamma in (0.0, -0.5, 1.5, float("nan")):
            with self.assertRaises(hsm.InvalidHyperparameterError):
                hsm.critic_target(reward, done, value, gamma, torch.tensor([2, 2]))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.critic_target(
                reward, torch.tensor([0.5, 0.0], dtype=DTYPE), value, 0.9,
                torch.tensor([2, 2]),
            )

    def test_integer_bellman_values_are_refused_without_truncation(self) -> None:
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.critic_target(
                reward=torch.tensor([1.0], dtype=DTYPE),
                done=torch.tensor([0]),
                next_value=torch.tensor([10]),
                gamma=0.9,
                duration=torch.tensor([2]),
            )
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.critic_target(
                reward=torch.tensor([1]),
                done=torch.tensor([0]),
                next_value=torch.tensor([10.0], dtype=DTYPE),
                gamma=0.9,
                duration=torch.tensor([2]),
            )

    def test_target_is_detached_by_construction(self) -> None:
        reward = torch.tensor([1.0], dtype=DTYPE, requires_grad=True)
        value = torch.tensor([10.0], dtype=DTYPE, requires_grad=True)
        target = hsm.critic_target(
            reward, torch.tensor([0]), value, 0.9, torch.tensor([2])
        )
        self.assertFalse(target.requires_grad)
        self.assertAlmostEqual(float(target[0]), 9.1, places=12)


class FailClosedTest(ModelsTestBase):
    """Malformed or non-finite input is refused, never silently repaired."""

    def test_non_finite_state_is_refused(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            state = _state(3).clone()
            state[1, 4] = bad
            with self.assertRaises(hsm.InvalidTensorError):
                self.actor(state)
            with self.assertRaises(hsm.InvalidTensorError):
                self.critics.critic_1.q_all_modes(
                    state, torch.zeros(3, 12, dtype=DTYPE)
                )

    def test_malformed_state_dimensions_are_refused(self) -> None:
        for shape in ((5, 30), (5, 32), (5, 0), (0, 31)):
            with self.assertRaises(hsm.InvalidTensorError):
                self.actor(torch.zeros(*shape, dtype=DTYPE))
        for tensor in (
            torch.zeros(31, dtype=DTYPE),
            torch.zeros(2, 5, 31, dtype=DTYPE),
        ):
            with self.assertRaises(hsm.InvalidTensorError):
                self.actor(tensor)
        with self.assertRaises(hsm.InvalidTensorError):
            self.actor(torch.zeros(5, 31, dtype=torch.long))
        with self.assertRaises(hsm.InvalidTensorError):
            self.actor([[0.0] * 31])

    def test_malformed_critic_inputs_are_refused(self) -> None:
        state = _state(4)
        good_onehot = hsm.mode_one_hot(torch.zeros(4, dtype=torch.long), dtype=DTYPE)
        good_q = torch.zeros(4, dtype=DTYPE)
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(state, torch.zeros(4, 11, dtype=DTYPE), good_q)
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(state, good_onehot, torch.zeros(4, 1, dtype=DTYPE))
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(state, good_onehot, torch.zeros(3, dtype=DTYPE))
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1.q_all_modes(state, torch.zeros(4, 11, dtype=DTYPE))
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(
                state, good_onehot, torch.full((4,), float("nan"), dtype=DTYPE)
            )
        malformed_onehot = good_onehot.clone()
        malformed_onehot[0, 1] = 1.0
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(state, malformed_onehot, good_q)
        fractional_onehot = good_onehot.clone()
        fractional_onehot[0, 0] = 0.5
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(state, fractional_onehot, good_q)
        for bad_q in (-0.001, 1.001):
            with self.assertRaises(hsm.InvalidTensorError):
                self.critics.critic_1(
                    state, good_onehot, torch.full((4,), bad_q, dtype=DTYPE)
                )
        with self.assertRaises(hsm.InvalidTensorError):
            self.critics.critic_1(
                state, good_onehot, torch.zeros(4, dtype=torch.long)
            )

    def test_non_finite_quantization_input_is_refused(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(hsm.InvalidTensorError):
                hsm.quantize_q_e4(torch.tensor([0.5, bad], dtype=DTYPE))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.quantize_q_e4(torch.tensor([1, 2], dtype=torch.long))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.quantize_q_e4([0.5])

    def test_invalid_temperatures_are_refused(self) -> None:
        for alpha_d, alpha_c in ((0.0, 0.1), (0.1, 0.0), (-1.0, 0.1), (0.1, float("inf"))):
            with self.assertRaises(hsm.InvalidHyperparameterError):
                hsm.actor_objective(
                    self.actor, self.critics, self.state, alpha_d, alpha_c
                )
            with self.assertRaises(hsm.InvalidHyperparameterError):
                hsm.soft_state_value(
                    self.actor, self.critics, self.state, alpha_d, alpha_c
                )

    def test_invalid_configuration_is_refused(self) -> None:
        for kwargs in (
            {"state_dim": 0},
            {"mode_count": -1},
            {"hidden_width": 0},
            {"hidden_depth": 0},
            {"log_std_min": 2.0, "log_std_max": -5.0},
            {"log_std_min": float("-inf")},
        ):
            with self.assertRaises(hsm.InvalidHyperparameterError):
                hsm.HybridSacModelConfig(**kwargs)
        with self.assertRaises(hsm.InvalidHyperparameterError):
            hsm.HybridSacModelConfig(dtype=torch.long)

    def test_malformed_mode_index_is_refused(self) -> None:
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.mode_one_hot(torch.zeros(2, 2, dtype=torch.long))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.mode_one_hot(torch.zeros(2, dtype=DTYPE))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.mode_one_hot(torch.tensor([12]))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.mode_one_hot(torch.tensor([-1]))
        with self.assertRaises(hsm.InvalidTensorError):
            hsm.mode_one_hot(torch.tensor([], dtype=torch.long))

    def test_log_std_is_clamped_to_the_registered_interval(self) -> None:
        actor = hsm.build_actor(REFERENCE_CONFIG, seed=1234)
        with torch.no_grad():
            actor.log_std_head.bias.fill_(50.0)
        heads = actor(self.state)
        self.assertTrue((heads.log_std <= hsm.LOG_STD_MAX + 1e-12).all())
        with torch.no_grad():
            actor.log_std_head.bias.fill_(-50.0)
        heads = actor(self.state)
        self.assertTrue((heads.log_std >= hsm.LOG_STD_MIN - 1e-12).all())


class DeterminismTest(ModelsTestBase):
    """The same seed gives byte-identical CPU results."""

    def test_same_seed_gives_byte_identical_parameters(self) -> None:
        first = hsm.build_actor(REFERENCE_CONFIG, seed=2024)
        second = hsm.build_actor(REFERENCE_CONFIG, seed=2024)
        other = hsm.build_actor(REFERENCE_CONFIG, seed=2025)
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(sorted(first_state), sorted(second_state))
        for key, value in first_state.items():
            self.assertEqual(
                value.numpy().tobytes(),
                second_state[key].numpy().tobytes(),
                f"{key} differs between identically seeded actors",
            )
        self.assertNotEqual(
            first_state["mean_head.weight"].numpy().tobytes(),
            other.state_dict()["mean_head.weight"].numpy().tobytes(),
        )

    def test_same_seed_gives_byte_identical_sampling(self) -> None:
        first = hsm.build_actor(REFERENCE_CONFIG, seed=2024)
        second = hsm.build_actor(REFERENCE_CONFIG, seed=2024)
        sample_a = first.sample_all_modes(
            self.state, generator=torch.Generator().manual_seed(7)
        )
        sample_b = second.sample_all_modes(
            self.state, generator=torch.Generator().manual_seed(7)
        )
        for name in ("q", "log_prob_continuous", "log_prob_discrete", "probs"):
            self.assertEqual(
                getattr(sample_a, name).detach().numpy().tobytes(),
                getattr(sample_b, name).detach().numpy().tobytes(),
                f"{name} is not byte-identical under the same seed",
            )
        self.assertEqual(sample_a.q_e4.tolist(), sample_b.q_e4.tolist())

    def test_building_a_model_does_not_disturb_the_global_rng(self) -> None:
        torch.manual_seed(4242)
        expected = torch.randn(3, dtype=DTYPE)
        torch.manual_seed(4242)
        hsm.build_actor(REFERENCE_CONFIG, seed=99)
        hsm.build_twin_critics(REFERENCE_CONFIG, seed=100)
        produced = torch.randn(3, dtype=DTYPE)
        self.assertEqual(expected.numpy().tobytes(), produced.numpy().tobytes())

    def test_objectives_are_reproducible(self) -> None:
        values = [
            float(
                hsm.actor_objective(
                    hsm.build_actor(REFERENCE_CONFIG, seed=1234),
                    hsm.build_twin_critics(REFERENCE_CONFIG, seed=5678),
                    self.state,
                    0.2,
                    0.05,
                    generator=torch.Generator().manual_seed(11),
                ).objective.detach()
            )
            for _ in range(3)
        ]
        self.assertEqual(len(set(values)), 1, f"non-reproducible objective {values}")


class ImportPurityTest(unittest.TestCase):
    """Importing the module reads no evidence and launches no runtime."""

    def test_import_performs_no_evidence_read_or_runtime_launch(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        probe = """
import json, sys
violations = []

def hook(event, args):
    try:
        if event == "open":
            path = str(args[0])
            low = path.lower()
            if "site-packages" in low or "dist-packages" in low:
                return
            if (
                low.endswith(".csv")
                or "/experiments/" in low
                or "action_catalog" in low
            ):
                violations.append([event, path])
        elif event in (
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "socket.socket",
            "socket.connect",
        ):
            violations.append([event, str(args)[:120]])
    except Exception:
        pass

sys.addaudithook(hook)
import rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models as m
assert m.STATE_DIM == 31
assert m.MODE_COUNT == 12
print("VIOLATIONS:" + json.dumps(violations))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            completed.returncode, 0, f"probe failed: {completed.stderr[-2000:]}"
        )
        marker = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("VIOLATIONS:")
        ]
        self.assertEqual(len(marker), 1, completed.stdout[-2000:])
        violations = json.loads(marker[0][len("VIOLATIONS:") :])
        self.assertEqual(violations, [], f"import had side effects: {violations}")

    def test_module_exposes_no_cuda_or_training_entry_points(self) -> None:
        source = Path(hsm.__file__).read_text(encoding="utf-8")
        for forbidden in (
            ".cuda(",
            "torch.cuda",
            "device='cuda'",
            'device="cuda"',
            "set_default_dtype",
            "set_default_device",
        ):
            self.assertNotIn(forbidden, source, f"module references {forbidden}")
        # No training loop, replay buffer or optimizer belongs in this phase.
        for forbidden in ("torch.optim", "ReplayBuffer", "def train("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
