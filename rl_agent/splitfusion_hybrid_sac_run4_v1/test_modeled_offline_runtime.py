"""Adversarial tests for the dedicated modeled-composite offline runtime."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    mcs_transition_acceptance,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    modeled_composite_training as modeled,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    modeled_offline_runtime as runtime,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import replay
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer
from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    test_modeled_composite_training as modeled_fixture,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1.models import build_run4_models


class ModeledOfflineRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        modeled_fixture.ModeledCompositeTrainingTest.setUpClass()
        cls.mcs_report = mcs_transition_acceptance.load_registered_acceptance()

    def setUp(self) -> None:
        self.fixture = modeled_fixture.ModeledCompositeTrainingTest(
            "test_binding_is_explicitly_offline_and_not_empirical"
        )

    def accepted_modeled_binding(
        self,
    ) -> modeled.ModeledCompositeBindingV1:
        base = self.fixture.binding()
        disclosures = []
        for item in base.component_disclosures:
            if item.role is modeled.ComponentRole.UL_MCS_TRANSITION:
                item = replace(
                    item,
                    source_evidence_sha256=(
                        self.mcs_report["source_evidence_sha256"]
                    ),
                    fit_support_sha256=(
                        self.mcs_report["model_binding_sha256"]
                    ),
                )
            disclosures.append(item)
        return replace(base, component_disclosures=tuple(disclosures))

    def wrapper(
        self,
        binding: modeled.ModeledCompositeBindingV1 | None = None,
    ) -> modeled.ModeledCompositeOfflineTransitionV1:
        authority = binding or self.accepted_modeled_binding()
        transition = self.fixture.transition()
        envelope = modeled.ModeledCompositeTrainingIssuerV1(authority).issue(
            transition=transition,
            support_use=self.fixture.support(),
            latency_projection=(
                modeled.LatencyProjectionV1.from_ordered_endpoints(
                    self.fixture.endpoints(150_000_000)
                )
            ),
        )
        return envelope.export_for_offline_training()

    def replay_binding(
        self,
        wrapper: modeled.ModeledCompositeOfflineTransitionV1 | None = None,
        authority: modeled.ModeledCompositeBindingV1 | None = None,
    ) -> runtime.ModeledReplayBindingV1:
        candidate = wrapper or self.wrapper(authority)
        modeled_authority = authority or self.accepted_modeled_binding()
        return runtime.ModeledReplayBindingV1.from_modeled_composite(
            modeled_binding=modeled_authority,
            gamma=candidate.gamma,
            freshness_policy_sha256=candidate.freshness_policy_sha256,
            empirical_scaling_sha256=candidate.empirical_scaling_sha256,
        )

    def runner(
        self,
    ) -> tuple[
        runtime.ModeledCompositeOfflineRunnerV1,
        modeled.ModeledCompositeOfflineTransitionV1,
    ]:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        runner = runtime.ModeledCompositeOfflineFactoryV1.build(
            modeled_binding=authority,
            gamma=wrapper.gamma,
            freshness_policy_sha256=wrapper.freshness_policy_sha256,
            empirical_scaling_sha256=wrapper.empirical_scaling_sha256,
            capacity=8,
            trainer_config=trainer.TrainerConfigV1(
                alpha_d=0.2,
                alpha_c=0.2,
            ),
            actor_seed=11,
            critic_seed=12,
            replay_seed=13,
            target_seed=14,
            trainer_actor_seed=15,
        )
        return runner, wrapper

    def empirical_binding(self, wrapper):
        return replay.ReplayBindingV1._for_test_only(
            gamma=wrapper.gamma,
            freshness_policy_sha256=wrapper.freshness_policy_sha256,
            empirical_scaling_sha256=wrapper.empirical_scaling_sha256,
            calibration_evidence_sha256="1" * 64,
            queue_kernel_evidence_sha256="2" * 64,
        )

    def test_binding_requires_registered_mcs_source_and_model(self) -> None:
        wrapper = self.wrapper()
        with self.assertRaisesRegex(
            replay.BindingMismatchError, "accepted source/model"
        ):
            runtime.ModeledReplayBindingV1.from_modeled_composite(
                modeled_binding=self.fixture.binding(),
                gamma=wrapper.gamma,
                freshness_policy_sha256=wrapper.freshness_policy_sha256,
                empirical_scaling_sha256=wrapper.empirical_scaling_sha256,
            )

    def test_binding_loads_registered_acceptance_not_provider_only(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        with mock.patch.object(
            mcs_transition_acceptance,
            "load_registered_acceptance",
            side_effect=mcs_transition_acceptance.McsAcceptanceError("missing"),
        ) as loader:
            with self.assertRaises(
                mcs_transition_acceptance.McsAcceptanceError
            ):
                runtime.ModeledReplayBindingV1.from_modeled_composite(
                    modeled_binding=authority,
                    gamma=wrapper.gamma,
                    freshness_policy_sha256=wrapper.freshness_policy_sha256,
                    empirical_scaling_sha256=wrapper.empirical_scaling_sha256,
                )
        loader.assert_called_once_with()

    def test_binding_preserves_mcs_and_non_deployment_provenance(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        binding = self.replay_binding(wrapper, authority)
        self.assertEqual(
            binding.mcs_acceptance_result_sha256,
            mcs_transition_acceptance.REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256,
        )
        self.assertEqual(
            binding.mcs_model_binding_sha256,
            self.mcs_report["model_binding_sha256"],
        )
        self.assertEqual(
            binding.mcs_source_evidence_sha256,
            self.mcs_report["source_evidence_sha256"],
        )
        self.assertEqual(
            binding.training_evidence_class,
            "MODELED_COMPOSITE_TRAINING",
        )
        self.assertEqual(
            binding.evidence_eligibility,
            "OFFLINE_MODELED_TRAINING_ONLY",
        )
        self.assertTrue(binding.offline_training_only)
        self.assertFalse(binding.measured_runtime_evidence)
        self.assertFalse(binding.calibrated_empirical_evidence)
        self.assertFalse(binding.production_authorized)
        self.assertFalse(binding.deployment_claim_allowed)

    def test_mcs_acceptance_post_binding_tamper_fails_closed(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        binding = self.replay_binding(wrapper, authority)
        old = binding.mcs_model_binding_sha256
        try:
            object.__setattr__(
                binding, "mcs_model_binding_sha256", "0" * 64
            )
            with self.assertRaises(replay.BindingMismatchError):
                binding.revalidate()
        finally:
            object.__setattr__(binding, "mcs_model_binding_sha256", old)
        binding.revalidate()

    def test_private_handoff_requires_exact_modeled_binding(self) -> None:
        wrapper = self.wrapper()
        with self.assertRaises(modeled.BindingError):
            wrapper._sealed_transition_for_modeled_replay("0" * 64)
        wrapper.require_attested()

    def test_buffer_accepts_wrapper_and_rejects_bare_transition(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        buffer = runtime.ModeledCompositeReplayBufferV1(
            4, self.replay_binding(wrapper, authority)
        )
        with self.assertRaises(replay.TransitionRejectedError):
            buffer.insert(wrapper._transition)
        self.assertEqual(len(buffer), 0)
        buffer.insert(wrapper)
        self.assertEqual(len(buffer), 1)

    def test_production_empirical_buffer_still_rejects_wrapper(self) -> None:
        wrapper = self.wrapper()
        empirical = replay._TestOnlyReplayBufferV1(
            4, self.empirical_binding(wrapper)
        )
        with self.assertRaises(replay.TransitionRejectedError):
            empirical.insert(wrapper)
        self.assertEqual(len(empirical), 0)
        with self.assertRaises(replay.ReplayBufferError):
            replay.ReplayBufferV1(
                4, self.replay_binding(wrapper)
            )  # type: ignore[arg-type]

    def test_wrapper_tamper_is_rejected_before_buffer_mutation(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        buffer = runtime.ModeledCompositeReplayBufferV1(
            4, self.replay_binding(wrapper, authority)
        )
        old = wrapper.support_use_sha256
        try:
            object.__setattr__(wrapper, "support_use_sha256", "0" * 64)
            with self.assertRaises(replay.TransitionRejectedError):
                buffer.insert(wrapper)
            self.assertEqual(len(buffer), 0)
            self.assertEqual(buffer.accepted_count, 0)
            self.assertEqual(buffer.seen_digest_count, 0)
            self.assertEqual(buffer.seen_identity_count, 0)
        finally:
            object.__setattr__(wrapper, "support_use_sha256", old)
        buffer.insert(wrapper)
        self.assertEqual(buffer.accepted_count, 1)

    def test_duplicate_transition_is_lifetime_rejected(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        buffer = runtime.ModeledCompositeReplayBufferV1(
            1, self.replay_binding(wrapper, authority)
        )
        buffer.insert(wrapper)
        with self.assertRaises(replay.DuplicateTransitionError):
            buffer.insert(wrapper)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.seen_digest_count, 1)

    def test_buffer_stores_no_transition_object_or_export_seam(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        buffer = runtime.ModeledCompositeReplayBufferV1(
            2, self.replay_binding(wrapper, authority)
        )
        buffer.insert(wrapper)
        row = buffer._rows[0]
        self.assertFalse(hasattr(row, "transition"))
        self.assertFalse(hasattr(row, "_transition"))
        self.assertFalse(hasattr(buffer, "export_for_replay"))
        with self.assertRaises(modeled.ProductionEvidenceRejected):
            wrapper.export_for_replay()

    def test_tensorization_is_exact_and_audit_survives(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        buffer = runtime.ModeledCompositeReplayBufferV1(
            2, self.replay_binding(wrapper, authority)
        )
        buffer.insert(wrapper)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(5)
        batch = buffer.sample(1, generator)
        transition = wrapper._transition
        self.assertTrue(
            torch.equal(
                batch.state[0],
                torch.tensor(
                    transition.state_features.as_tuple(),
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(int(batch.mode_id[0]), transition.action.mode_id)
        self.assertEqual(int(batch.q_e4[0]), transition.action.q_e4)
        self.assertEqual(
            float(batch.reward[0]),
            float(torch.tensor(transition.reward, dtype=torch.float32)),
        )
        self.assertEqual(
            float(batch.discount()[0]),
            float(torch.tensor(transition.discount, dtype=torch.float32)),
        )
        audit = batch.audit[0]
        self.assertEqual(
            audit["evidence_class"], "MODELED_COMPOSITE_TRAINING"
        )
        self.assertEqual(
            audit["modeled_binding_sha256"],
            wrapper.modeled_binding_sha256,
        )
        self.assertEqual(
            audit["source_envelope_sha256"],
            wrapper.source_envelope_sha256,
        )
        self.assertEqual(
            audit["support_use_sha256"], wrapper.support_use_sha256
        )
        self.assertEqual(
            audit["latency_projection_sha256"],
            wrapper.latency_projection_sha256,
        )
        self.assertFalse(audit["production_authorized"])

    def test_batch_accessors_do_not_alias_private_storage(self) -> None:
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        batch = runner.replay_buffer.sample(
            1, torch.Generator(device="cpu").manual_seed(7)
        )
        state = batch.state
        state.fill_(999.0)
        self.assertFalse(torch.equal(state, batch.state))
        discount = batch.discount()
        discount.fill_(0.0)
        self.assertGreater(float(batch.discount()[0]), 0.0)
        mutable_audit = dict(batch.audit[0])
        mutable_audit["production_authorized"] = True
        self.assertFalse(batch.audit[0]["production_authorized"])

    def test_sampling_does_not_advance_global_rng(self) -> None:
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        before = torch.random.get_rng_state().clone()
        local = torch.Generator(device="cpu")
        local.manual_seed(19)
        runner.replay_buffer.sample(1, local)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_modeled_trainer_reuses_exact_numerical_update(self) -> None:
        self.assertIs(
            runtime.ModeledCompositeHybridSacTrainerV1.update_once,
            trainer._Run4TrainerCore.update_once,
        )
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        metrics = runner.train_once(1)
        self.assertEqual(metrics.update_index, 1)
        self.assertEqual(runner.trainer.update_count, 1)
        metrics.require_finite()

    def test_modeled_trainer_rejects_empirical_batch(self) -> None:
        runner, wrapper = self.runner()
        raw_buffer = replay._TestOnlyReplayBufferV1(
            2, self.empirical_binding(wrapper)
        )
        raw_buffer.insert(wrapper._transition)
        empirical_batch = raw_buffer.sample(
            1, torch.Generator(device="cpu").manual_seed(23)
        )
        with self.assertRaises(trainer.TrainerPreflightError):
            runner.trainer.update_once(empirical_batch)
        self.assertEqual(runner.trainer.update_count, 0)

    def test_empirical_trainer_rejects_modeled_batch(self) -> None:
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        modeled_batch = runner.replay_buffer.sample(
            1, torch.Generator(device="cpu").manual_seed(29)
        )
        models = build_run4_models(actor_seed=31, critic_seed=32)
        empirical_binding = self.empirical_binding(wrapper)
        empirical_trainer = trainer._TestOnlyRun4HybridSacTrainerV1(
            actor=models.actor,
            critics=models.critics,
            config=trainer.TrainerConfigV1(alpha_d=0.2, alpha_c=0.2),
            expected_binding=empirical_binding,
            target_generator=torch.Generator(device="cpu").manual_seed(33),
            actor_generator=torch.Generator(device="cpu").manual_seed(34),
        )
        with self.assertRaises(trainer.TrainerPreflightError):
            empirical_trainer.update_once(modeled_batch)
        self.assertEqual(empirical_trainer.update_count, 0)

    def test_batch_audit_cannot_be_relabelled_empirical(self) -> None:
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        batch = runner.replay_buffer.sample(
            1, torch.Generator(device="cpu").manual_seed(37)
        )
        altered = dict(batch.audit[0])
        altered["evidence_class"] = "CALIBRATED_EMPIRICAL"
        with self.assertRaises(replay.ReplayBufferError):
            replace(batch, _audit=(altered,))

    def test_runner_audit_reports_modeled_and_mcs_provenance(self) -> None:
        runner, wrapper = self.runner()
        runner.ingest(wrapper)
        audit = runner.audit_summary()
        self.assertEqual(audit["accepted_count"], 1)
        self.assertEqual(
            audit["evidence_class"], "MODELED_COMPOSITE_TRAINING"
        )
        self.assertTrue(audit["offline_training_only"])
        self.assertFalse(audit["production_authorized"])
        self.assertFalse(audit["deployment_claim_allowed"])
        self.assertEqual(
            audit["mcs_acceptance_result_sha256"],
            mcs_transition_acceptance.REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256,
        )
        self.assertEqual(
            audit["mcs_model_binding_sha256"],
            self.mcs_report["model_binding_sha256"],
        )

    def test_factory_rng_streams_are_private_and_global_rng_is_unchanged(self) -> None:
        before = torch.random.get_rng_state().clone()
        runner, _ = self.runner()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIsNot(
            runner._replay_generator, runner.trainer._target_generator
        )
        self.assertIsNot(
            runner._replay_generator, runner.trainer._actor_generator
        )
        self.assertIsNot(
            runner.trainer._target_generator,
            runner.trainer._actor_generator,
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_factory_refuses_shared_rng_seed(self) -> None:
        authority = self.accepted_modeled_binding()
        wrapper = self.wrapper(authority)
        with self.assertRaises(trainer.TrainerStateError):
            runtime.ModeledCompositeOfflineFactoryV1.build(
                modeled_binding=authority,
                gamma=wrapper.gamma,
                freshness_policy_sha256=wrapper.freshness_policy_sha256,
                empirical_scaling_sha256=wrapper.empirical_scaling_sha256,
                capacity=8,
                trainer_config=trainer.TrainerConfigV1(
                    alpha_d=0.2, alpha_c=0.2
                ),
                actor_seed=41,
                critic_seed=42,
                replay_seed=43,
                target_seed=43,
                trainer_actor_seed=44,
            )


if __name__ == "__main__":
    unittest.main()
