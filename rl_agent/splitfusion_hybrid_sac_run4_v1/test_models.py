"""Tests for the Run-4 21-D Hybrid-SAC model binding."""

from __future__ import annotations

import copy
import unittest

import torch

from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)

from .models import (
    RUN4_MODEL_BINDING,
    RUN4_MODEL_BINDING_SHA256,
    Run4ModelBundleV1,
    Run4ModelError,
    build_run4_models,
    run4_model_config,
    validate_run4_models,
)
from .run4_contract import POLICY_FEATURE_COUNT, POLICY_FEATURE_ORDER


class Run4ModelsTest(unittest.TestCase):
    def test_binding_and_dimensions_are_exact(self) -> None:
        config = run4_model_config()
        self.assertEqual(config.state_dim, 21)
        self.assertEqual(config.state_dim, POLICY_FEATURE_COUNT)
        self.assertEqual(config.critic_input_dim, 34)
        self.assertEqual(
            RUN4_MODEL_BINDING["policy_feature_order"],
            tuple(POLICY_FEATURE_ORDER),
        )
        self.assertEqual(len(RUN4_MODEL_BINDING_SHA256), 64)

    def test_build_is_deterministic_and_rng_neutral(self) -> None:
        before = torch.get_rng_state().clone()
        first = build_run4_models(actor_seed=17, critic_seed=29)
        after = torch.get_rng_state().clone()
        second = build_run4_models(actor_seed=17, critic_seed=29)
        self.assertTrue(torch.equal(before, after))
        for left, right in zip(
            first.actor.state_dict().values(), second.actor.state_dict().values()
        ):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(
            first.critics.state_dict().values(),
            second.critics.state_dict().values(),
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(torch.cuda.is_initialized())

    def test_forward_and_deterministic_execution_use_21_features(self) -> None:
        bundle = build_run4_models(actor_seed=1, critic_seed=2)
        state = torch.zeros((3, POLICY_FEATURE_COUNT), dtype=torch.float32)
        heads = bundle.actor(state)
        self.assertEqual(heads.logits.shape, (3, 12))
        executed = bundle.actor.deterministic_execution(state)
        self.assertEqual(executed.mode_index.shape, (3,))
        self.assertTrue(bool((executed.q_e4 >= 0).all()))
        self.assertTrue(bool((executed.q_e4 <= 9800).all()))

    def test_legacy_31_feature_or_unbounded_models_are_rejected(self) -> None:
        legacy = HybridSacModelConfig(dtype=torch.float32)
        with self.assertRaisesRegex(Run4ModelError, "configuration differs"):
            validate_run4_models(build_actor(legacy, seed=1), build_twin_critics(legacy, seed=2))

        unbounded = HybridSacModelConfig(
            state_dim=POLICY_FEATURE_COUNT,
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(Run4ModelError, "configuration differs"):
            validate_run4_models(
                build_actor(unbounded, seed=1),
                build_twin_critics(unbounded, seed=2),
            )

    def test_mutated_binding_and_targets_fail_closed(self) -> None:
        bundle = build_run4_models(actor_seed=1, critic_seed=2)
        with self.assertRaisesRegex(Run4ModelError, "binding digest"):
            Run4ModelBundleV1(
                actor=bundle.actor,
                critics=bundle.critics,
                binding_sha256="0" * 64,
            )
        damaged = copy.deepcopy(bundle.critics)
        next(damaged.target_1.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(Run4ModelError, "must remain frozen"):
            validate_run4_models(bundle.actor, damaged)

    def test_bad_seeds_are_rejected(self) -> None:
        for value in (-1, 1.0, True):
            with self.subTest(value=value), self.assertRaises(Run4ModelError):
                build_run4_models(actor_seed=value, critic_seed=2)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
