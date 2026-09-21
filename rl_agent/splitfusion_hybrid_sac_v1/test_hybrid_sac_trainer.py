"""Focused tests for the one-update Hybrid-SAC trainer (Phase 4b.2, part C).

``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.  CPU-only and deterministic.

These prove *mechanics*: that one update is finite and reproducible, that the
target consumes the replay batch's emitted discount rather than re-deriving
it, that gradients reach what they should and nothing else, and that a
malformed batch mutates nothing.  No test here asserts learning progress, and
none could: a single update cannot show convergence.

Every batch is drawn from a real :class:`ReplayBufferV1` filled with genuine
eligible ``ReplayTransitionV1`` records built through the public contracts.
"""

from __future__ import annotations

import copy
import math
import unittest
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

import torch

from . import hybrid_sac_models as hsm
from . import hybrid_sac_trainer as hst
from . import modeled_smoke_support as mss
from . import replay_buffer as rbuf
from . import state_reward_transition_contract as src
from .test_replay_buffer import ReplayBufferTestBase

DTYPE = torch.float32
ANCHORS = (0, 3000, 5000, 7000, 9000, 9800)


class TrainerTestBase(ReplayBufferTestBase):
    """Builds a real buffer, a real batch and a float32 actor/critic pair."""

    def _models(
        self, actor_seed: int = 10, critic_seed: int = 11
    ) -> Tuple[hsm.ConditionalHybridActor, hsm.TwinHybridCritics]:
        config = hsm.HybridSacModelConfig(dtype=DTYPE)
        return (
            hsm.build_actor(config, seed=actor_seed),
            hsm.build_twin_critics(config, seed=critic_seed),
        )

    def _buffer(self, count: int = 6, **kwargs) -> rbuf.ReplayBufferV1:
        buffer = rbuf.ReplayBufferV1(capacity=max(count, 1), float_dtype=DTYPE)
        for index in range(count):
            buffer.insert(
                self._transition(
                    mode_id=index % hsm.MODE_COUNT,
                    q_e4=ANCHORS[index % len(ANCHORS)],
                    tag=f"tr{index}",
                    **kwargs,
                )
            )
        return buffer

    def _batch(self, count: int = 6, seed: int = 1, **kwargs):
        buffer = self._buffer(count, **kwargs)
        return buffer.sample(count, torch.Generator().manual_seed(seed))

    def _trainer(
        self,
        batch,
        actor=None,
        critics=None,
        *,
        target_seed: int = 100,
        actor_seed: int = 200,
        alpha_d: float = 0.2,
        alpha_c: float = 0.05,
        tau: float = 0.005,
    ) -> hst.HybridSacTrainerV1:
        if actor is None or critics is None:
            actor, critics = self._models()
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=alpha_d,
            alpha_c=alpha_c,
            tau=tau,
        )
        return hst.HybridSacTrainerV1(
            actor,
            critics,
            config,
            expected_binding=batch.binding,
            target_generator=torch.Generator().manual_seed(target_seed),
            actor_generator=torch.Generator().manual_seed(actor_seed),
        )


class SingleUpdateTest(TrainerTestBase):
    """One update runs, is finite, and reports complete diagnostics."""

    def test_one_update_is_finite_and_complete(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        self.assertEqual(trainer.update_count, 0)
        metrics = trainer.update_once(batch)
        self.assertEqual(trainer.update_count, 1)

        metrics.assert_finite()
        self.assertIsInstance(metrics, hst.UpdateMetricsV1)
        self.assertEqual(metrics.batch_size, 6)
        self.assertEqual(metrics.phase_label, "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY")
        self.assertEqual(
            metrics.gamma_per_tensor, batch.binding.gamma_per_tensor
        )
        self.assertEqual(
            metrics.continuous_log_prob_coordinate,
            hsm.PHYSICAL_Q_DENSITY,
        )
        self.assertIsNone(metrics.modeled_smoke_support_sha256)
        for name, value in metrics.as_dict().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertTrue(
                    math.isfinite(float(value)), f"{name} is not finite"
                )
        self.assertGreater(metrics.critic_loss_total, 0.0)
        self.assertGreaterEqual(metrics.twin_gap_mean, 0.0)
        self.assertGreaterEqual(metrics.bootstrap_fraction, 0.0)
        self.assertLessEqual(metrics.bootstrap_fraction, 1.0)
        self.assertGreaterEqual(metrics.duration_min, 2)
        self.assertGreaterEqual(metrics.q_requested_min, 0.0)
        self.assertLessEqual(metrics.q_requested_max, 0.98)
        self.assertGreaterEqual(metrics.q_executed_min, 0.0)
        self.assertLessEqual(metrics.q_executed_max, 0.98)
        self.assertGreater(metrics.discrete_entropy, 0.0)
        self.assertLessEqual(metrics.discrete_entropy, math.log(hsm.MODE_COUNT))

    def test_modeled_smoke_update_records_coordinate_support_and_bounds(self) -> None:
        batch = self._batch()
        model_config = hsm.HybridSacModelConfig(
            dtype=DTYPE,
            modeled_smoke_support=mss.MODELED_SMOKE_SUPPORT,
        )
        actor = hsm.build_actor(model_config, seed=10)
        critics = hsm.build_twin_critics(model_config, seed=11)
        with torch.no_grad():
            actor.mean_head.weight.zero_()
            actor.mean_head.bias.copy_(
                torch.tensor(
                    [-100.0 if mode % 2 == 0 else 100.0 for mode in range(12)],
                    dtype=DTYPE,
                )
            )
        metrics = self._trainer(
            batch, actor=actor, critics=critics
        ).update_once(batch)
        metrics.assert_finite()
        self.assertEqual(
            metrics.continuous_log_prob_coordinate,
            hsm.NORMALIZED_Z_DENSITY,
        )
        self.assertEqual(
            metrics.modeled_smoke_support_sha256,
            mss.MODELED_SMOKE_SUPPORT_SHA256,
        )
        self.assertEqual(metrics.q_saturation_fraction, 1.0)
        document = metrics.as_dict()
        self.assertEqual(
            document["continuous_log_prob_coordinate"],
            hsm.NORMALIZED_Z_DENSITY,
        )
        self.assertEqual(
            document["modeled_smoke_support_sha256"],
            mss.MODELED_SMOKE_SUPPORT_SHA256,
        )

    def test_tampered_modeled_smoke_support_is_refused_before_update(self) -> None:
        batch = self._batch(terminated=True)
        self.assertFalse(batch.bootstrap.any())
        model_config = hsm.HybridSacModelConfig(
            dtype=DTYPE,
            modeled_smoke_support=mss.MODELED_SMOKE_SUPPORT,
        )
        actor = hsm.build_actor(model_config, seed=10)
        critics = hsm.build_twin_critics(model_config, seed=11)
        trainer = self._trainer(batch, actor=actor, critics=critics)
        before_actor = [parameter.detach().clone() for parameter in actor.parameters()]
        before_critics = [
            parameter.detach().clone() for parameter in critics.parameters()
        ]
        actor_optimizer_before = copy.deepcopy(
            trainer.actor_optimizer.state_dict()
        )
        critic_optimizer_before = copy.deepcopy(
            trainer.critic_optimizer.state_dict()
        )
        with torch.no_grad():
            actor._support_q_e4_lower[0] = 0
        with self.assertRaisesRegex(
            hst.TrainerPreflightError, "changed after trainer construction"
        ):
            trainer.update_once(batch)
        for before, after in zip(before_actor, actor.parameters()):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(before_critics, critics.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertEqual(
            actor_optimizer_before, trainer.actor_optimizer.state_dict()
        )
        self.assertEqual(
            critic_optimizer_before, trainer.critic_optimizer.state_dict()
        )
        self.assertEqual(trainer.update_count, 0)

    def test_provisional_hyperparameters_are_labelled(self) -> None:
        self.assertEqual(
            hst.PROVISIONAL_HYPERPARAMETERS["status"],
            "PROVISIONAL_SMOKE_HYPOTHESIS_NOT_FROZEN",
        )
        config = hst.TrainerConfigV1(
            gamma_per_tensor=0.99, alpha_d=0.2, alpha_c=0.05
        )
        self.assertEqual(config.actor_lr, 3e-4)
        self.assertEqual(config.critic_lr, 3e-4)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.tau, 0.005)
        self.assertEqual(
            config.hyperparameter_status,
            "PROVISIONAL_SMOKE_HYPOTHESIS_NOT_FROZEN",
        )

    def test_online_actor_and_critics_change(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        actor_before = [p.detach().clone() for p in actor.parameters()]
        online_before = [
            p.detach().clone()
            for p in chain(
                critics.critic_1.parameters(), critics.critic_2.parameters()
            )
        ]
        metrics = self._trainer(batch, actor, critics).update_once(batch)

        self.assertGreater(metrics.actor_param_delta_norm, 0.0)
        self.assertGreater(metrics.online_critic_param_delta_norm, 0.0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(actor_before, actor.parameters())
            ),
            "no actor parameter changed",
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    online_before,
                    chain(
                        critics.critic_1.parameters(),
                        critics.critic_2.parameters(),
                    ),
                )
            ),
            "no online critic parameter changed",
        )

    def test_exactly_one_step_per_optimizer(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        with mock.patch.object(
            trainer.actor_optimizer,
            "step",
            wraps=trainer.actor_optimizer.step,
        ) as actor_step, mock.patch.object(
            trainer.critic_optimizer,
            "step",
            wraps=trainer.critic_optimizer.step,
        ) as critic_step, mock.patch.object(
            hsm.TwinHybridCritics,
            "polyak_update",
            autospec=True,
            wraps=hsm.TwinHybridCritics.polyak_update,
        ) as polyak:
            trainer.update_once(batch)
        self.assertEqual(actor_step.call_count, 1)
        self.assertEqual(critic_step.call_count, 1)
        self.assertEqual(polyak.call_count, 1)


class DiscountDriftSentinelTest(TrainerTestBase):
    """The target consumes the batch's emitted discount, never a recomputation."""

    def test_target_uses_emitted_discount_not_dtype_first_recomputation(
        self,
    ) -> None:
        gamma = 0.99999999
        spec = self._reward_spec(gamma_per_tensor=gamma)
        transition = self._transition(
            spec=spec, extra_reuses=148, tag="sentinel"
        )
        self.assertEqual(transition.hold_duration_tensors, 150)

        buffer = rbuf.ReplayBufferV1(capacity=2, float_dtype=DTYPE)
        buffer.insert(transition)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))

        self.assertEqual(int(batch.duration[0]), 150)
        self.assertTrue(bool(batch.bootstrap[0]))
        emitted = float(batch.discount()[0])
        self.assertAlmostEqual(emitted, 0.9999985098838806, places=12)
        self.assertNotEqual(emitted, 1.0)

        # Amplify V so the 1.49e-6 relative difference between the emitted
        # discount and the recomputed 1.0 is far above float32 resolution.
        actor, critics = self._models()
        with torch.no_grad():
            for target in (critics.target_1, critics.target_2):
                target.value_head.bias.fill_(1.0e6)

        # An identical, untouched pair reproduces V exactly, because the
        # target is formed before any parameter changes and from an
        # identically seeded stream.
        mirror_actor = copy.deepcopy(actor)
        mirror_critics = copy.deepcopy(critics)
        expected_v = hsm.soft_state_value(
            mirror_actor,
            mirror_critics,
            batch.next_state,
            0.2,
            0.05,
            generator=torch.Generator().manual_seed(100),
        ).value

        metrics = self._trainer(
            batch, actor, critics, target_seed=100
        ).update_once(batch)

        self.assertAlmostEqual(
            metrics.next_value_mean, float(expected_v[0]), places=3
        )
        reward = batch.reward
        correct = float(
            (
                reward
                + torch.tensor(emitted, dtype=DTYPE) * expected_v.to(DTYPE)
            )[0]
        )
        recomputed_discount = float(
            torch.pow(
                torch.tensor(gamma, dtype=DTYPE),
                torch.tensor(150.0, dtype=DTYPE),
            )
        )
        self.assertEqual(recomputed_discount, 1.0)
        wrong = float(
            (
                reward
                + torch.tensor(recomputed_discount, dtype=DTYPE)
                * expected_v.to(DTYPE)
            )[0]
        )

        self.assertNotAlmostEqual(
            correct, wrong, delta=1.0, msg="sentinel is not separable"
        )
        self.assertAlmostEqual(metrics.target_mean, correct, places=2)
        self.assertGreater(
            abs(metrics.target_mean - wrong),
            1.0,
            "the target used the dtype-first recomputed discount",
        )
        self.assertEqual(metrics.discount_min, emitted)
        self.assertEqual(metrics.discount_max, emitted)

    def test_trainer_never_recomputes_the_discount(self) -> None:
        source = Path(hst.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[2]  # skip the module docstring
        for forbidden in ("critic_target", "torch.pow", "** duration", "**duration"):
            self.assertNotIn(
                forbidden,
                body,
                f"the trainer body references {forbidden!r}",
            )
        self.assertNotIn("critic_target", dir(hst))
        self.assertIn("batch.discount()", body)

    def test_gamma_is_binding_metadata_and_must_match_exactly(self) -> None:
        # A gamma that disagrees with the frozen binding is now refused at
        # construction, before any optimizer exists.
        batch = self._batch()
        actor, critics = self._models()
        config = hst.TrainerConfigV1(
            gamma_per_tensor=0.98,  # batch binding is 0.99
            alpha_d=0.2,
            alpha_c=0.05,
        )
        with self.assertRaises(hst.TrainerStateError) as caught:
            hst.HybridSacTrainerV1(
                actor,
                critics,
                config,
                expected_binding=batch.binding,
                target_generator=torch.Generator().manual_seed(1),
                actor_generator=torch.Generator().manual_seed(2),
            )
        self.assertIn("gamma_per_tensor", str(caught.exception))


class BootstrapRoutingTest(TrainerTestBase):
    """Only bootstrap-eligible successor states reach the target evaluation."""

    def test_non_bootstrap_rows_never_reach_soft_state_value(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=8, float_dtype=DTYPE)
        cases = (
            {"terminated": True, "with_next": True},    # terminal w/ successor
            {"terminated": True, "with_next": False},
            {"truncated": True, "with_next": False},
            {},                                          # ordinary bootstrap
            {"truncated": True, "with_next": True},      # truncation bootstrap
        )
        for index, case in enumerate(cases):
            buffer.insert(self._transition(tag=f"route{index}", **case))
        batch = buffer.sample(5, torch.Generator().manual_seed(3))
        expected_rows = int(batch.bootstrap.sum())
        self.assertEqual(expected_rows, 2)

        seen = {}
        real = hst.soft_state_value

        def spy(actor, critics, next_state, *args, **kwargs):
            seen["states"] = next_state.clone()
            return real(actor, critics, next_state, *args, **kwargs)

        with mock.patch.object(hst, "soft_state_value", spy):
            metrics = self._trainer(batch).update_once(batch)

        self.assertIn("states", seen)
        self.assertEqual(seen["states"].shape[0], expected_rows)
        # Exactly the bootstrap rows' successor vectors, in order.
        expected = batch.next_state[batch.bootstrap]
        self.assertTrue(torch.equal(seen["states"], expected))
        # And no zero sentinel row was evaluated.
        self.assertFalse(
            bool((seen["states"].abs().sum(dim=1) == 0.0).any()),
            "a zero-filled sentinel reached the target critics",
        )
        self.assertEqual(metrics.bootstrap_count, expected_rows)

    def test_no_bootstrap_rows_skips_evaluation_entirely(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4, float_dtype=DTYPE)
        for index in range(3):
            buffer.insert(
                self._transition(
                    terminated=True, with_next=False, tag=f"noboot{index}"
                )
            )
        batch = buffer.sample(3, torch.Generator().manual_seed(0))
        self.assertFalse(bool(batch.bootstrap.any()))

        with mock.patch.object(hst, "soft_state_value") as never:
            metrics = self._trainer(batch).update_once(batch)
        never.assert_not_called()

        self.assertEqual(metrics.bootstrap_count, 0)
        self.assertEqual(metrics.next_value_mean, 0.0)
        # y collapses to the reward exactly.
        self.assertAlmostEqual(
            metrics.target_mean, metrics.reward_mean, places=6
        )

    def test_terminal_row_with_a_successor_contributes_no_bootstrap(
        self,
    ) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4, float_dtype=DTYPE)
        buffer.insert(
            self._transition(terminated=True, with_next=True, tag="term-next")
        )
        batch = buffer.sample(1, torch.Generator().manual_seed(0))
        self.assertTrue(bool(batch.has_next_state[0]))
        self.assertFalse(bool(batch.bootstrap[0]))
        self.assertFalse(bool(torch.all(batch.next_state[0] == 0.0)))

        with mock.patch.object(hst, "soft_state_value") as never:
            metrics = self._trainer(batch).update_once(batch)
        never.assert_not_called()
        self.assertAlmostEqual(
            metrics.target_mean, metrics.reward_mean, places=6
        )


class GradientRoutingTest(TrainerTestBase):
    """Gradients reach the actor's twelve branches and never the critics."""

    def test_no_critic_gradient_contamination_during_the_actor_step(
        self,
    ) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)

        observed = {}
        real_step = trainer.actor_optimizer.step

        def capture(*args, **kwargs):
            # At the instant the actor steps, no online critic may hold grad.
            observed["critic_grads"] = [
                p.grad is not None
                for p in chain(
                    critics.critic_1.parameters(),
                    critics.critic_2.parameters(),
                )
            ]
            observed["actor_grads"] = [
                p.grad is not None for p in actor.parameters()
            ]
            return real_step(*args, **kwargs)

        with mock.patch.object(trainer.actor_optimizer, "step", capture):
            trainer.update_once(batch)

        self.assertTrue(observed["critic_grads"])
        self.assertFalse(
            any(observed["critic_grads"]),
            "an online critic held a gradient during the actor step",
        )
        self.assertTrue(all(observed["actor_grads"]))
        # requires_grad is restored afterwards, not left disabled.
        for parameter in chain(
            critics.critic_1.parameters(), critics.critic_2.parameters()
        ):
            self.assertTrue(parameter.requires_grad)

    def test_all_twelve_actor_branches_receive_gradients(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)

        captured = {}
        real_step = trainer.actor_optimizer.step

        def capture(*args, **kwargs):
            for name, head in (
                ("mean_head", actor.mean_head),
                ("log_std_head", actor.log_std_head),
                ("logit_head", actor.logit_head),
            ):
                captured[name] = head.weight.grad.detach().clone()
            return real_step(*args, **kwargs)

        with mock.patch.object(trainer.actor_optimizer, "step", capture):
            metrics = trainer.update_once(batch)

        for name, grad in captured.items():
            self.assertEqual(tuple(grad.shape), (12, 128), name)
            per_mode = grad.abs().sum(dim=1)
            self.assertTrue(
                torch.isfinite(grad).all(), f"{name} gradient not finite"
            )
            self.assertTrue(
                bool((per_mode > 0).all()),
                f"{name}: modes "
                f"{torch.nonzero(per_mode == 0).flatten().tolist()} got no "
                f"gradient",
            )
        self.assertGreater(metrics.actor_grad_norm, 0.0)
        self.assertGreater(metrics.critic_grad_norm, 0.0)

    def test_dq_dq_still_reaches_the_actor_while_critics_are_frozen(
        self,
    ) -> None:
        # Freezing critic parameters must not cut the path from the critic's
        # q input back to the actor's continuous heads.
        actor, critics = self._models()
        state = torch.randn(4, 31, dtype=DTYPE)
        actor.zero_grad(set_to_none=True)
        with hst._FrozenParameters(
            list(
                chain(
                    critics.critic_1.parameters(),
                    critics.critic_2.parameters(),
                )
            )
        ):
            objective = hsm.actor_objective(
                actor,
                critics,
                state,
                0.2,
                0.05,
                generator=torch.Generator().manual_seed(1),
            )
            objective.objective.backward()
        self.assertIsNotNone(actor.mean_head.weight.grad)
        self.assertTrue(bool((actor.mean_head.weight.grad.abs() > 0).any()))
        for parameter in chain(
            critics.critic_1.parameters(), critics.critic_2.parameters()
        ):
            self.assertIsNone(parameter.grad)
            self.assertTrue(parameter.requires_grad)


class ExecutedActionTest(TrainerTestBase):
    """The critic is evaluated at the executed mode and executed q_e4/9800."""

    def test_critics_receive_executed_q_and_mode(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        captured = {}
        real = hsm.TwinHybridCritics.q_values

        def spy(self_critics, state, mode_onehot, q_normalized):
            captured["mode_onehot"] = mode_onehot.clone()
            captured["q"] = q_normalized.clone()
            captured["state"] = state.clone()
            return real(self_critics, state, mode_onehot, q_normalized)

        with mock.patch.object(hsm.TwinHybridCritics, "q_values", spy):
            trainer.update_once(batch)

        self.assertTrue(
            torch.equal(captured["q"], batch.q_normalized_executed)
        )
        expected_q = batch.q_e4.to(DTYPE) / 9800.0
        self.assertTrue(torch.equal(captured["q"], expected_q))
        self.assertTrue(
            torch.equal(
                captured["mode_onehot"].argmax(dim=1).to(torch.int64),
                batch.mode_id,
            )
        )
        self.assertTrue(torch.equal(captured["state"], batch.state))
        # Executed q lies in [0, 1] and includes both registered bounds here.
        self.assertGreaterEqual(float(captured["q"].min()), 0.0)
        self.assertLessEqual(float(captured["q"].max()), 1.0)


class PolyakTest(TrainerTestBase):
    """Target movement is exactly the Polyak equation, once."""

    def test_target_update_is_exactly_the_polyak_equation(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        tau = 0.25
        trainer = self._trainer(batch, actor, critics, tau=tau)

        target_params = list(
            chain(
                critics.target_1.parameters(), critics.target_2.parameters()
            )
        )
        target_before = [p.detach().clone() for p in target_params]
        trainer.update_once(batch)
        online_after = [
            p.detach().clone()
            for p in chain(
                critics.critic_1.parameters(), critics.critic_2.parameters()
            )
        ]
        for before, online, after in zip(
            target_before, online_after, target_params
        ):
            expected = (1.0 - tau) * before + tau * online
            torch.testing.assert_close(after, expected, rtol=0.0, atol=1e-7)

    def test_targets_are_absent_from_both_optimizers(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        target_ids = {
            id(p)
            for p in chain(
                critics.target_1.parameters(), critics.target_2.parameters()
            )
        }
        self.assertGreater(len(target_ids), 0)
        for optimizer in (trainer.actor_optimizer, trainer.critic_optimizer):
            optimizer_ids = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }
            self.assertTrue(target_ids.isdisjoint(optimizer_ids))
        # And targets never accumulate gradient.
        trainer.update_once(batch)
        for parameter in chain(
            critics.target_1.parameters(), critics.target_2.parameters()
        ):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_a_target_in_an_optimizer_is_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        trainer = hst.HybridSacTrainerV1(
            actor,
            critics,
            config,
            expected_binding=batch.binding,
            target_generator=torch.Generator().manual_seed(1),
            actor_generator=torch.Generator().manual_seed(2),
        )
        trainer.critic_optimizer.add_param_group(
            {"params": list(critics.target_1.parameters())}
        )
        with self.assertRaises(hst.TrainerStateError):
            trainer._assert_no_target_parameters_in_optimizers()


class DeterminismTest(TrainerTestBase):
    """Identical seeds reproduce exactly; different seeds diverge."""

    def test_identical_seeds_give_identical_updates(self) -> None:
        batch = self._batch()
        results = []
        for _ in range(2):
            actor, critics = self._models()
            metrics = self._trainer(batch, actor, critics).update_once(batch)
            results.append(
                (
                    metrics.as_dict(),
                    [p.detach().clone() for p in actor.parameters()],
                    [
                        p.detach().clone()
                        for p in chain(
                            critics.critic_1.parameters(),
                            critics.critic_2.parameters(),
                            critics.target_1.parameters(),
                            critics.target_2.parameters(),
                        )
                    ],
                )
            )
        self.assertEqual(results[0][0], results[1][0])
        for first, second in zip(results[0][1], results[1][1]):
            self.assertEqual(
                first.numpy().tobytes(), second.numpy().tobytes()
            )
        for first, second in zip(results[0][2], results[1][2]):
            self.assertEqual(
                first.numpy().tobytes(), second.numpy().tobytes()
            )

    def test_different_generator_seeds_diverge(self) -> None:
        batch = self._batch()
        actor_a, critics_a = self._models()
        first = self._trainer(
            batch, actor_a, critics_a, target_seed=100, actor_seed=200
        ).update_once(batch)
        actor_b, critics_b = self._models()
        second = self._trainer(
            batch, actor_b, critics_b, target_seed=777, actor_seed=888
        ).update_once(batch)
        self.assertNotEqual(first.actor_loss, second.actor_loss)
        self.assertNotEqual(
            first.critic_loss_total, second.critic_loss_total
        )

    def test_update_does_not_disturb_the_global_rng(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        torch.manual_seed(4242)
        before = torch.get_rng_state()
        trainer.update_once(batch)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_target_and_actor_streams_must_be_separate(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        shared = torch.Generator().manual_seed(5)
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        with self.assertRaises(hst.TrainerStateError):
            hst.HybridSacTrainerV1(
                actor,
                critics,
                config,
                expected_binding=batch.binding,
                target_generator=shared,
                actor_generator=shared,
            )
        for bad in (torch.default_generator, None, 5):
            with self.assertRaises(hst.TrainerStateError):
                hst.HybridSacTrainerV1(
                    actor,
                    critics,
                    config,
                    expected_binding=batch.binding,
                    target_generator=bad,
                    actor_generator=torch.Generator().manual_seed(1),
                )


class PreflightTest(TrainerTestBase):
    """A malformed batch mutates neither parameters nor optimizer state."""

    def _mutable_copy(self, batch, **overrides):
        """A directly built batch, for exercising preflight only."""
        fields = {
            "_state": batch.state,
            "_next_state": batch.next_state,
            "_mode_id": batch.mode_id,
            "_q_e4": batch.q_e4,
            "_reward": batch.reward,
            "_duration": batch.duration,
            "_discount": batch.discount(),
            "_has_next_state": batch.has_next_state,
            "_bootstrap": batch.bootstrap,
            "_terminated": batch.terminated,
            "_truncated": batch.truncated,
            "binding": batch.binding,
            "audit": batch.audit,
            "float_dtype": batch.float_dtype,
        }
        fields.update(overrides)
        return rbuf.ReplayTensorBatchV1(**fields)

    def _assert_no_mutation(self, bad_batch) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        params = list(actor.parameters()) + list(
            chain(
                critics.critic_1.parameters(),
                critics.critic_2.parameters(),
                critics.target_1.parameters(),
                critics.target_2.parameters(),
            )
        )
        before = [p.detach().clone() for p in params]
        self.assertEqual(len(trainer.actor_optimizer.state), 0)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)

        with self.assertRaises(hst.TrainerError):
            trainer.update_once(bad_batch)

        for old, new in zip(before, params):
            self.assertTrue(torch.equal(old, new), "a parameter changed")
        self.assertEqual(
            len(trainer.actor_optimizer.state), 0, "actor optimizer state grew"
        )
        self.assertEqual(
            len(trainer.critic_optimizer.state),
            0,
            "critic optimizer state grew",
        )
        self.assertEqual(trainer.update_count, 0)

    def test_nan_and_inf_batches_are_refused_without_mutation(self) -> None:
        batch = self._batch()
        for field, filler in (
            ("_reward", float("nan")),
            ("_reward", float("inf")),
            ("_state", float("nan")),
            ("_next_state", float("inf")),
            ("_discount", float("nan")),
        ):
            corrupt = getattr(batch, field.lstrip("_"))
            if field == "_discount":
                corrupt = batch.discount()
            corrupt = corrupt.clone()
            corrupt.view(-1)[0] = filler
            self._assert_no_mutation(self._mutable_copy(batch, **{field: corrupt}))

    def test_out_of_range_fields_are_refused_without_mutation(self) -> None:
        batch = self._batch()
        cases = (
            {"_mode_id": torch.full_like(batch.mode_id, 12)},
            {"_mode_id": torch.full_like(batch.mode_id, -1)},
            {"_q_e4": torch.full_like(batch.q_e4, 9801)},
            {"_duration": torch.ones_like(batch.duration)},
            {"_discount": torch.full_like(batch.discount(), 1.5)},
            {"_discount": torch.full_like(batch.discount(), -0.1)},
            {"_bootstrap": ~batch.bootstrap},
            {"_terminated": torch.ones_like(batch.terminated)},
        )
        for overrides in cases:
            self._assert_no_mutation(self._mutable_copy(batch, **overrides))

    def test_foreign_batch_types_are_refused(self) -> None:
        for bad in (None, 42, {"state": 1}, object()):
            self._assert_no_mutation(bad)

    def test_dtype_and_config_mismatches_are_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        # float64 models against the float32 replay binding.
        wide = hsm.HybridSacModelConfig(dtype=torch.float64)
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        with self.assertRaises(hst.TrainerStateError):
            hst.HybridSacTrainerV1(
                hsm.build_actor(wide, seed=1),
                hsm.build_twin_critics(wide, seed=2),
                config,
                expected_binding=batch.binding,
                target_generator=torch.Generator().manual_seed(1),
                actor_generator=torch.Generator().manual_seed(2),
            )
        with self.assertRaises(hst.TrainerPreflightError):
            hst.TrainerConfigV1(
                gamma_per_tensor=0.99,
                alpha_d=0.2,
                alpha_c=0.05,
                float_dtype=torch.float64,
            )

    def test_invalid_configurations_are_refused(self) -> None:
        for kwargs in (
            {"alpha_d": 0.0},
            {"alpha_c": -1.0},
            {"alpha_d": float("inf")},
            {"gamma_per_tensor": 0.0},
            {"gamma_per_tensor": 1.5},
            {"tau": 0.0},
            {"tau": 1.5},
            {"actor_lr": 0.0},
            {"critic_lr": -1.0},
            {"batch_size": 0},
        ):
            base = {
                "gamma_per_tensor": 0.99,
                "alpha_d": 0.2,
                "alpha_c": 0.05,
            }
            base.update(kwargs)
            with self.assertRaises(hst.TrainerPreflightError):
                hst.TrainerConfigV1(**base)


class HardeningTest(TrainerTestBase):
    """Phase C.1: nothing malformed may reach an optimizer or a parameter."""

    def _assert_construction_refused(self, actor, critics, batch) -> None:
        """Construction must fail, leaving no optimizer and no mutation."""
        before = [p.detach().clone() for p in actor.parameters()]
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        with self.assertRaises(hst.TrainerStateError) as caught:
            hst.HybridSacTrainerV1(
                actor,
                critics,
                config,
                expected_binding=batch.binding,
                target_generator=torch.Generator().manual_seed(1),
                actor_generator=torch.Generator().manual_seed(2),
            )
        for old, new in zip(before, actor.parameters()):
            self.assertTrue(torch.equal(old, new))
        return caught.exception

    # -- 3. structural validation before any optimizer exists ------------- #

    def test_actor_with_foreign_state_dim_is_refused(self) -> None:
        batch = self._batch()
        wrong = hsm.HybridSacModelConfig(state_dim=32, dtype=DTYPE)
        actor = hsm.build_actor(wrong, seed=1)
        _, critics = self._models()
        self.assertEqual(
            hsm.ConditionalHybridActor(wrong).encoder[0].in_features, 32
        )
        error = self._assert_construction_refused(actor, critics, batch)
        self.assertIn("state_dim", str(error))

    def test_actor_doubled_with_stale_config_metadata_is_refused(self) -> None:
        # config.dtype still claims float32 while every parameter is float64.
        batch = self._batch()
        actor, critics = self._models()
        actor.double()
        self.assertIs(actor.config.dtype, torch.float32)
        self.assertIs(next(actor.parameters()).dtype, torch.float64)
        error = self._assert_construction_refused(actor, critics, batch)
        self.assertIn("float64", str(error))
        self.assertIn("not trusted", str(error))

    def test_doubled_critics_are_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        critics.critic_2.double()
        error = self._assert_construction_refused(actor, critics, batch)
        self.assertIn("critic_2", str(error))

    def test_doubled_target_critics_are_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        critics.target_1.double()
        error = self._assert_construction_refused(actor, critics, batch)
        self.assertIn("target_1", str(error))

    # -- 2. the complete binding is frozen -------------------------------- #

    def _binding(self, batch, **overrides) -> rbuf.ReplayBindingV1:
        fields = {
            name: getattr(batch.binding, name)
            for name in (
                "reward_spec_sha256",
                "state_normalization_spec_sha256",
                "freshness_policy_sha256",
                "gamma_per_tensor",
                "schema_id",
                "schema_version",
                "schema_sha256",
                "catalog_sha256",
                "policy_feature_order",
                "policy_feature_count",
            )
        }
        fields.update(overrides)
        return rbuf.ReplayBindingV1(**fields)

    def test_foreign_schema_catalog_or_feature_order_is_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        foreign = (
            {"schema_id": "some_other_schema_v9"},
            {"schema_version": src.SCHEMA_VERSION + 1},
            {"schema_sha256": "0" * 64},
            {"catalog_sha256": "0" * 64},
            {
                "policy_feature_order": tuple(
                    reversed(src.POLICY_FEATURE_ORDER)
                )
            },
            {"policy_feature_count": 30},
        )
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        for overrides in foreign:
            name = next(iter(overrides))
            with self.assertRaises(hst.TrainerStateError, msg=name) as caught:
                hst.HybridSacTrainerV1(
                    actor,
                    critics,
                    config,
                    expected_binding=self._binding(batch, **overrides),
                    target_generator=torch.Generator().manual_seed(1),
                    actor_generator=torch.Generator().manual_seed(2),
                )
            self.assertIn(name, str(caught.exception))

    def test_reversed_feature_order_has_the_right_length(self) -> None:
        # The order check must be element-wise: a reversal passes a length
        # check and is semantically wrong for all 31 features.
        reversed_order = tuple(reversed(src.POLICY_FEATURE_ORDER))
        self.assertEqual(len(reversed_order), len(src.POLICY_FEATURE_ORDER))
        self.assertNotEqual(reversed_order, tuple(src.POLICY_FEATURE_ORDER))

    def test_non_binding_expected_binding_is_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        config = hst.TrainerConfigV1(
            gamma_per_tensor=batch.binding.gamma_per_tensor,
            alpha_d=0.2,
            alpha_c=0.05,
        )
        for bad in (None, {"gamma_per_tensor": 0.99}, 0.99):
            with self.assertRaises(hst.TrainerStateError):
                hst.HybridSacTrainerV1(
                    actor,
                    critics,
                    config,
                    expected_binding=bad,
                    target_generator=torch.Generator().manual_seed(1),
                    actor_generator=torch.Generator().manual_seed(2),
                )

    def test_a_batch_from_a_different_binding_is_refused(self) -> None:
        # Two buffers, two reward specs: the same trainer may not learn from
        # both.  Caught at update time, before any mutation.
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        trainer.update_once(batch)
        self.assertEqual(trainer.update_count, 1)

        other_spec = self._reward_spec(r_registered_failure=-2.0)
        other = rbuf.ReplayBufferV1(capacity=4, float_dtype=DTYPE)
        for index in range(2):
            other.insert(
                self._transition(spec=other_spec, tag=f"otherbind{index}")
            )
        foreign_batch = other.sample(2, torch.Generator().manual_seed(0))
        self.assertNotEqual(foreign_batch.binding, batch.binding)

        params = list(actor.parameters()) + list(
            chain(
                critics.critic_1.parameters(), critics.critic_2.parameters()
            )
        )
        before = [p.detach().clone() for p in params]
        with self.assertRaises(hst.TrainerPreflightError) as caught:
            trainer.update_once(foreign_batch)
        self.assertIn("reward_spec_sha256", str(caught.exception))
        for old, new in zip(before, params):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(trainer.update_count, 1)

    # -- 4. optimizer wiring is revalidated every update ------------------ #

    def test_target_added_to_an_optimizer_after_construction_is_refused(
        self,
    ) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        trainer.critic_optimizer.add_param_group(
            {"params": list(critics.target_1.parameters())}
        )
        params = list(actor.parameters()) + list(
            chain(
                critics.critic_1.parameters(),
                critics.critic_2.parameters(),
                critics.target_1.parameters(),
                critics.target_2.parameters(),
            )
        )
        before = [p.detach().clone() for p in params]
        with self.assertRaises(hst.TrainerStateError) as caught:
            trainer.update_once(batch)
        self.assertIn("target", str(caught.exception))
        for old, new in zip(before, params):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(trainer.update_count, 0)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)
        self.assertEqual(len(trainer.actor_optimizer.state), 0)

    def test_foreign_parameter_added_to_an_optimizer_is_refused(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        trainer.actor_optimizer.add_param_group(
            {"params": [torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))]}
        )
        with self.assertRaises(hst.TrainerStateError) as caught:
            trainer.update_once(batch)
        self.assertIn("foreign", str(caught.exception))
        self.assertEqual(trainer.update_count, 0)

    def test_duplicated_parameter_in_an_optimizer_is_refused(self) -> None:
        batch = self._batch()
        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        trainer.actor_optimizer.param_groups[0]["params"].append(
            actor.mean_head.weight
        )
        with self.assertRaises(hst.TrainerStateError) as caught:
            trainer.update_once(batch)
        self.assertIn("duplicated", str(caught.exception))
        self.assertEqual(trainer.update_count, 0)

    # -- 5. finite loss and diagnostic overflow --------------------------- #

    def test_finite_reward_with_nonfinite_float32_mse_is_refused(self) -> None:
        # r_registered_failure = -1e20 is contract-finite and survives the
        # replay dtype, but (q - y)^2 overflows float32 to inf.
        spec = self._reward_spec(r_registered_failure=-1e20)
        transition = self._transition(spec=spec, tag="mse-overflow")
        self.assertTrue(math.isfinite(transition.scalar_reward))
        buffer = rbuf.ReplayBufferV1(capacity=2, float_dtype=DTYPE)
        buffer.insert(transition)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(batch.reward).all())

        actor, critics = self._models()
        trainer = self._trainer(batch, actor, critics)
        params = list(actor.parameters()) + list(
            chain(
                critics.critic_1.parameters(),
                critics.critic_2.parameters(),
                critics.target_1.parameters(),
                critics.target_2.parameters(),
            )
        )
        before = [p.detach().clone() for p in params]

        with self.assertRaises(hst.TrainerError) as caught:
            trainer.update_once(batch)
        self.assertIn("critic_1_loss", str(caught.exception))
        for old, new in zip(before, params):
            self.assertTrue(torch.equal(old, new), "a parameter moved")
        self.assertEqual(len(trainer.actor_optimizer.state), 0)
        self.assertEqual(len(trainer.critic_optimizer.state), 0)
        self.assertEqual(trainer.update_count, 0)

    def test_large_but_representable_reward_still_updates(self) -> None:
        # -1e19 gives a finite float32 MSE and finite gradients.  It must not
        # be rejected: squaring the gradient in float32 would overflow to inf
        # even though the float64 norm is about 4.1e19.
        spec = self._reward_spec(r_registered_failure=-1e19)
        transition = self._transition(spec=spec, tag="big-but-ok")
        buffer = rbuf.ReplayBufferV1(capacity=2, float_dtype=DTYPE)
        buffer.insert(transition)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))

        metrics = self._trainer(batch).update_once(batch)
        metrics.assert_finite()
        self.assertTrue(math.isfinite(metrics.critic_grad_norm))
        self.assertGreater(metrics.critic_grad_norm, 1e18)
        self.assertTrue(math.isfinite(metrics.critic_loss_total))

    def test_grad_norm_accumulates_in_float64(self) -> None:
        # Direct proof that the float32 square overflows where float64 does not.
        parameter = torch.nn.Parameter(torch.zeros(4, dtype=DTYPE))
        parameter.grad = torch.full((4,), 1e19, dtype=DTYPE)
        naive = float(torch.sum(parameter.grad**2))
        self.assertEqual(naive, float("inf"))
        widened = hst._grad_norm([parameter])
        self.assertTrue(math.isfinite(widened))
        self.assertAlmostEqual(widened / 2e19, 1.0, places=6)

    def test_update_count_increments_only_after_metrics_validate(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        self.assertEqual(trainer.update_count, 0)
        with mock.patch.object(
            hst.UpdateMetricsV1,
            "assert_finite",
            side_effect=hst.TrainerError("synthetic diagnostic failure"),
        ):
            with self.assertRaises(hst.TrainerError):
                trainer.update_once(batch)
        self.assertEqual(
            trainer.update_count, 0, "a failed update was still counted"
        )
        trainer.update_once(batch)
        self.assertEqual(trainer.update_count, 1)


class ReplayIntegrationTest(TrainerTestBase):
    """Buffer to update, using only genuine eligible contract records."""

    def test_end_to_end_from_genuine_transitions(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=12, float_dtype=DTYPE)
        for index in range(8):
            buffer.insert(
                self._transition(
                    mode_id=index % hsm.MODE_COUNT,
                    q_e4=ANCHORS[index % len(ANCHORS)],
                    extra_reuses=index % 3,
                    tag=f"e2e{index}",
                )
            )
        self.assertEqual(len(buffer), 8)
        for transition in buffer.stored_transitions():
            self.assertTrue(transition.learning_eligible)
            self.assertTrue(math.isfinite(transition.scalar_reward))
            self.assertIs(
                transition.eligibility,
                type(transition.eligibility).ELIGIBLE,
            )

        # Replay sampling is caller-side with its own generator.
        replay_generator = torch.Generator().manual_seed(9)
        batch = buffer.sample(6, replay_generator)
        metrics = self._trainer(batch).update_once(batch)
        metrics.assert_finite()
        self.assertEqual(metrics.batch_size, 6)
        self.assertGreater(metrics.actor_param_delta_norm, 0.0)
        self.assertGreater(metrics.online_critic_param_delta_norm, 0.0)
        self.assertGreater(metrics.target_param_delta_norm, 0.0)
        self.assertEqual(set(batch.duration.tolist()) - {2, 3, 4}, set())
        # The trainer never sampled: the replay stream advanced only where
        # the caller drew the batch.
        self.assertEqual(metrics.gamma_per_tensor, 0.99)

    def test_trainer_holds_no_buffer_and_never_samples(self) -> None:
        batch = self._batch()
        trainer = self._trainer(batch)
        self.assertFalse(hasattr(trainer, "buffer"))
        self.assertFalse(hasattr(trainer, "replay"))
        source = Path(hst.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[2]
        for forbidden in (".sample(", "ReplayBufferV1", "randperm"):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
