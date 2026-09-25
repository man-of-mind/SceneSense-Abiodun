"""Adversarial tests for the offline-only modeled-composite evidence seam."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_composite_training as src
from rl_agent.splitfusion_hybrid_sac_run4_v1 import replay
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_run4_v1.test_environment import (
    CLOCK,
    SESSION,
    UE_ID,
    FixtureKernel,
    FixtureProvider,
)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


def _h(char: str) -> str:
    return char * 64


class ModeledCompositeTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = action_contract.load_contract()

    def action(self, mode_id: int = 7, q_e4: int = 7000) -> ExecutedActionIdentity:
        executable = self.catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.catalog
        )

    @staticmethod
    def binding() -> src.ModeledCompositeBindingV1:
        roles = tuple(src.ComponentRole)
        natures = (
            src.ComponentEvidenceNature.MEASURED_SOURCE,
            src.ComponentEvidenceNature.FIT_DERIVED_MODEL,
            src.ComponentEvidenceNature.FIT_DERIVED_MODEL,
            src.ComponentEvidenceNature.DETERMINISTIC_CONTRACT_TRANSFORM,
            src.ComponentEvidenceNature.FIT_DERIVED_MODEL,
        )
        disclosures = tuple(
            src.ComponentEvidenceDisclosureV1(
                role=role,
                nature=nature,
                source_evidence_sha256=_h(str(index + 1)),
                fit_support_sha256=_h(chr(ord("a") + index)),
                source_scope=f"fit-only-{role.value.lower()}",
            )
            for index, (role, nature) in enumerate(zip(roles, natures))
        )
        return src.ModeledCompositeBindingV1(
            binding_id="run4-offline-modeled-composite",
            binding_version=1,
            component_disclosures=disclosures,
            provider_implementation_sha256=_h("e"),
            verifier_manifest_sha256=_h("f"),
        )

    @staticmethod
    def support(
        *,
        target_profile: str = "FAVORABLE_STABLE",
        target_mode: int = 7,
        source_profiles: tuple[str, ...] = ("FAVORABLE_STABLE",),
        source_modes: tuple[int, ...] = (7,),
    ) -> src.ModeledCompositeSupportUseV1:
        profile_direct = target_profile in source_profiles
        mode_direct = target_mode in source_modes
        return src.ModeledCompositeSupportUseV1(
            target_profile_label=target_profile,
            target_mode_id=target_mode,
            source_profile_labels=source_profiles,
            source_mode_ids=source_modes,
            profile_transfer_status=(
                src.ProfileTransferStatus.PROFILE_WITHIN_DIRECT_FIT_SUPPORT
                if profile_direct
                else src.ProfileTransferStatus.PROFILE_TRANSFER_UNVALIDATED
            ),
            mode_transfer_status=(
                src.ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT
                if mode_direct
                else src.ModeTransferStatus.MODE_TRANSFER_UNVALIDATED
            ),
            payload_in_fit_support=True,
            backlog_in_fit_support=True,
            mcs_in_fit_support=True,
            quality_in_fit_support=True,
            total_latency_residual_in_fit_support=True,
            widened_uncertainty_applied=not (profile_direct and mode_direct),
            support_evidence_sha256=_h("9"),
        )

    def transition(
        self,
        *,
        kind: contract.RewardEventKind = contract.RewardEventKind.DELIVERED_SUCCESS,
        resolution_offset_ns: int = 150_000_000,
        cycle_offset_ns: int = 200_000_000,
        q_perc: float | None = 0.7,
        mode_id: int = 7,
    ) -> contract.SemiMarkovTransitionV2:
        provider = FixtureProvider()
        kernel = FixtureKernel(
            kind=kind,
            resolution_offset_ns=resolution_offset_ns,
            cycle_offset_ns=cycle_offset_ns,
            q_perc=q_perc,
        )
        calibrated = environment.CalibrationBindingV1(
            calibration_id="test-only-transition-constructor",
            calibration_version=1,
            evidence_sha256=_h("1"),
            verifier_report_sha256=_h("2"),
            kernel_binding_sha256=_h("3"),
            state_provider_binding_sha256=_h("4"),
        )
        with mock.patch.object(
            environment,
            "REGISTERED_CALIBRATION_BINDING_SHA256",
            calibrated.canonical_sha256(),
        ):
            env = environment.Run4SequentialEnvironmentV1(
                state_provider=provider,
                kernel=kernel,
                gamma=0.99,
                evidence_class=environment.EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL,
                calibration_binding=calibrated,
            )
        env.reset(session_uuid=SESSION, ue_id=UE_ID)
        result = env.step(self.action(mode_id=mode_id))
        self.assertIs(type(result), environment.CalibratedEmpiricalCycleV1)
        assert isinstance(result, environment.CalibratedEmpiricalCycleV1)
        return result.export_for_replay()

    @staticmethod
    def endpoints(total_ns: int) -> src.FeedbackEndpointPairV1:
        actor_inference_ns = 750_000
        quantization_dispatch_ns = 250_000
        return src.FeedbackEndpointPairV1(
            action_open_timestamp_ns=2_000_000_000,
            feedback_received_timestamp_ns=2_000_000_000 + total_ns,
            clock_domain=CLOCK,
            source_row_sha256=_h("8"),
            fixed_action_source_total_ns=(
                total_ns - actor_inference_ns - quantization_dispatch_ns
            ),
            fixed_action_source_total_evidence_sha256=_h("7"),
            actor_inference_ns=actor_inference_ns,
            actor_inference_evidence_sha256=_h("6"),
            quantization_dispatch_ns=quantization_dispatch_ns,
            quantization_dispatch_evidence_sha256=_h("5"),
        )

    def issued_success(self) -> src.ModeledCompositeTrainingEnvelopeV1:
        transition = self.transition()
        return src.ModeledCompositeTrainingIssuerV1(self.binding()).issue(
            transition=transition,
            support_use=self.support(),
            latency_projection=src.LatencyProjectionV1.from_ordered_endpoints(
                self.endpoints(150_000_000)
            ),
        )

    def test_binding_is_explicitly_offline_and_not_empirical(self) -> None:
        binding = self.binding()
        self.assertIs(binding.evidence_class, src.EVIDENCE_CLASS)
        self.assertTrue(binding.offline_training_only)
        self.assertFalse(binding.measured_runtime_evidence)
        self.assertFalse(binding.calibrated_empirical_evidence)
        self.assertFalse(binding.production_authorized)
        self.assertFalse(binding.deployment_claim_allowed)
        self.assertEqual(binding.latency_schema_version, 3)
        self.assertEqual(
            {item.role for item in binding.component_disclosures},
            set(src.ComponentRole),
        )

    def test_binding_refuses_any_production_or_measured_claim(self) -> None:
        base = self.binding()
        for field in (
            "measured_runtime_evidence",
            "calibrated_empirical_evidence",
            "production_authorized",
            "deployment_claim_allowed",
        ):
            with self.subTest(field=field):
                with self.assertRaises(src.BindingError):
                    replace(base, **{field: True})

    def test_direct_support_and_two_transfer_flags_are_independent(self) -> None:
        direct = self.support()
        self.assertIs(
            direct.profile_transfer_status,
            src.ProfileTransferStatus.PROFILE_WITHIN_DIRECT_FIT_SUPPORT,
        )
        self.assertIs(
            direct.mode_transfer_status,
            src.ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT,
        )
        self.assertFalse(direct.widened_uncertainty_applied)

        profile_only = self.support(target_profile="MID_VARIABLE")
        self.assertIs(
            profile_only.profile_transfer_status,
            src.ProfileTransferStatus.PROFILE_TRANSFER_UNVALIDATED,
        )
        self.assertIs(
            profile_only.mode_transfer_status,
            src.ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT,
        )
        self.assertTrue(profile_only.widened_uncertainty_applied)

        mode_only = self.support(target_mode=3)
        self.assertIs(
            mode_only.profile_transfer_status,
            src.ProfileTransferStatus.PROFILE_WITHIN_DIRECT_FIT_SUPPORT,
        )
        self.assertIs(
            mode_only.mode_transfer_status,
            src.ModeTransferStatus.MODE_TRANSFER_UNVALIDATED,
        )
        self.assertTrue(mode_only.widened_uncertainty_applied)

    def test_unvalidated_transfer_requires_widened_uncertainty(self) -> None:
        with self.assertRaisesRegex(src.SupportError, "widened_uncertainty"):
            replace(
                self.support(target_profile="MID_VARIABLE"),
                widened_uncertainty_applied=False,
            )

    def test_every_numerical_support_gate_is_mandatory(self) -> None:
        direct = self.support()
        for field in (
            "payload_in_fit_support",
            "backlog_in_fit_support",
            "mcs_in_fit_support",
            "quality_in_fit_support",
            "total_latency_residual_in_fit_support",
        ):
            with self.subTest(field=field):
                with self.assertRaises(src.SupportError):
                    replace(direct, **{field: False})

    def test_ordered_endpoints_define_authoritative_integer_total(self) -> None:
        projection = src.LatencyProjectionV1.from_ordered_endpoints(
            self.endpoints(133_123_456)
        )
        self.assertEqual(projection.terminal_elapsed_ns, 133_123_456)
        self.assertIsNotNone(projection.latency)
        assert projection.latency is not None
        self.assertEqual(
            projection.latency.action_open_to_feedback_ns, 133_123_456
        )
        self.assertEqual(
            projection.latency.action_open_to_feedback_evidence_sha256,
            projection.endpoints.canonical_sha256,
        )
        self.assertFalse(projection.latency.has_diagnostic_breakdown)

    def test_fixed_action_source_cannot_silently_assume_zero_actor_cost(self) -> None:
        endpoint = self.endpoints(150_000_000)
        with self.assertRaisesRegex(
            src.ModeledCompositeContractError, "positive measured/modeled actor"
        ):
            replace(
                endpoint,
                fixed_action_source_total_ns=150_000_000,
                actor_inference_ns=0,
                quantization_dispatch_ns=0,
            )

    def test_actor_augmentation_must_close_the_ordered_endpoint_total(self) -> None:
        with self.assertRaisesRegex(
            src.ModeledCompositeContractError, "must equal fixed-action"
        ):
            replace(self.endpoints(150_000_000), actor_inference_ns=750_001)

    def test_inclusive_170ms_endpoint_is_still_success_eligible(self) -> None:
        projection = src.LatencyProjectionV1.from_ordered_endpoints(
            self.endpoints(contract.REWARD_DEADLINE_NS)
        )
        self.assertIs(
            projection.terminal_kind,
            sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK,
        )
        self.assertEqual(
            projection.terminal_elapsed_ns, contract.REWARD_DEADLINE_NS
        )

    def test_one_ns_late_is_timeout_and_never_success_latency(self) -> None:
        actual = contract.REWARD_DEADLINE_NS + 1
        projection = src.LatencyProjectionV1.from_ordered_endpoints(
            self.endpoints(actual)
        )
        self.assertIs(
            projection.terminal_kind, sequential_kernel.KernelTerminalKind.TIMEOUT
        )
        self.assertEqual(
            projection.terminal_elapsed_ns,
            sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS,
        )
        self.assertIsNone(projection.latency)
        self.assertEqual(
            projection.late_orphan_action_open_to_feedback_ns, actual
        )

    def test_late_projection_cannot_be_relabelled_delivered(self) -> None:
        late = src.LatencyProjectionV1.from_ordered_endpoints(
            self.endpoints(200_000_000)
        )
        with self.assertRaisesRegex(src.ModeledCompositeContractError, "TIMEOUT"):
            replace(
                late,
                terminal_kind=(
                    sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK
                ),
            )

    def test_issuer_binds_on_time_projection_to_exact_success_transition(self) -> None:
        envelope = self.issued_success()
        self.assertTrue(envelope.is_attested)
        self.assertTrue(envelope.offline_training_export_allowed)
        exported = envelope.export_for_offline_training()
        self.assertIs(type(exported), src.ModeledCompositeOfflineTransitionV1)
        self.assertTrue(exported.is_attested)
        self.assertEqual(exported.latency_ms, 150.0)
        audit = envelope.to_audit_dict()
        self.assertEqual(audit["evidence_class"], "MODELED_COMPOSITE_TRAINING")
        self.assertFalse(audit["production_authorized"])
        self.assertFalse(audit["deployment_claim_allowed"])

    def test_issuer_binds_late_arrival_only_to_timeout_transition(self) -> None:
        transition = self.transition(
            kind=contract.RewardEventKind.TIMEOUT,
            resolution_offset_ns=(
                sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS
            ),
            q_perc=None,
        )
        projection = src.LatencyProjectionV1.from_ordered_endpoints(
            self.endpoints(220_000_000)
        )
        envelope = src.ModeledCompositeTrainingIssuerV1(self.binding()).issue(
            transition=transition,
            support_use=self.support(),
            latency_projection=projection,
        )
        exported = envelope.export_for_offline_training()
        self.assertIs(exported.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertEqual(exported.reward, -1.0)
        self.assertIsNone(exported.q_perc)
        self.assertIsNone(exported.latency_ms)

    def test_late_projection_is_refused_for_success_transition(self) -> None:
        with self.assertRaisesRegex(
            src.ModeledCompositeContractError, "timeout reward path"
        ):
            src.ModeledCompositeTrainingIssuerV1(self.binding()).issue(
                transition=self.transition(),
                support_use=self.support(),
                latency_projection=src.LatencyProjectionV1.from_ordered_endpoints(
                    self.endpoints(220_000_000)
                ),
            )

    def test_mode_support_must_match_executed_action(self) -> None:
        with self.assertRaisesRegex(src.SupportError, "target mode"):
            src.ModeledCompositeTrainingIssuerV1(self.binding()).issue(
                transition=self.transition(mode_id=7),
                support_use=self.support(target_mode=3),
                latency_projection=src.LatencyProjectionV1.from_ordered_endpoints(
                    self.endpoints(150_000_000)
                ),
            )

    def test_existing_environment_rejects_modeled_evidence_class(self) -> None:
        with self.assertRaises(environment.EnvironmentStateError):
            environment.Run4SequentialEnvironmentV1(
                state_provider=FixtureProvider(),
                kernel=FixtureKernel(),
                gamma=0.99,
                evidence_class=src.EVIDENCE_CLASS,  # type: ignore[arg-type]
            )

    def test_existing_calibrated_envelope_rejects_modeled_class(self) -> None:
        issued = self.issued_success()
        with self.assertRaises(environment.CalibrationUnavailableError):
            environment.CalibratedEmpiricalCycleV1(
                evidence_class=src.EVIDENCE_CLASS,  # type: ignore[arg-type]
                calibration_binding_sha256=_h("1"),
                kernel_terminal_closure_timestamp_ns=2_200_000_000,
                reward_request_flags=(True, False),
                _transition=self.transition(),
            )

    def test_production_named_export_always_fails(self) -> None:
        envelope = self.issued_success()
        self.assertFalse(envelope.replay_export_allowed)
        with self.assertRaises(src.ProductionEvidenceRejected):
            envelope.export_for_replay()

    def test_offline_export_cannot_enter_existing_replay_after_export(self) -> None:
        envelope = self.issued_success()
        exported = envelope.export_for_offline_training()
        self.assertIs(type(exported), src.ModeledCompositeOfflineTransitionV1)
        self.assertIsNot(type(exported), contract.SemiMarkovTransitionV2)
        self.assertFalse(exported.replay_export_allowed)
        with self.assertRaises(src.ProductionEvidenceRejected):
            exported.export_for_replay()
        binding = replay.ReplayBindingV1._for_test_only(
            gamma=exported.gamma,
            freshness_policy_sha256=exported.freshness_policy_sha256,
            empirical_scaling_sha256=exported.empirical_scaling_sha256,
            calibration_evidence_sha256=_h("1"),
            queue_kernel_evidence_sha256=_h("2"),
        )
        buffer = replay._TestOnlyReplayBufferV1(8, binding)
        with self.assertRaises(replay.TransitionRejectedError):
            buffer.insert(envelope.export_for_offline_training())
        self.assertEqual(len(buffer), 0)

    def test_unissued_envelope_cannot_export_offline(self) -> None:
        issued = self.issued_success()
        forged = replace(issued, _attestation=None)
        self.assertFalse(forged.is_attested)
        with self.assertRaisesRegex(
            src.ModeledCompositeContractError, "absent, forged, or stale"
        ):
            forged.export_for_offline_training()

    def test_post_issue_transition_tamper_invalidates_envelope(self) -> None:
        envelope = self.issued_success()
        exported = envelope.export_for_offline_training()
        transition = exported._transition
        old_discount = transition.discount
        try:
            object.__setattr__(transition, "discount", 0.5)
            self.assertFalse(envelope.is_attested)
            self.assertFalse(exported.is_attested)
            with self.assertRaises(src.ModeledCompositeContractError):
                envelope.export_for_offline_training()
            with self.assertRaises(src.ModeledCompositeContractError):
                exported.require_attested()
        finally:
            object.__setattr__(transition, "discount", old_discount)


if __name__ == "__main__":
    unittest.main()
