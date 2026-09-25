"""Adversarial CPU-only tests for the Run-4 sequential kernel reducer."""

from __future__ import annotations

import ast
import inspect
import math
import unittest
from dataclasses import fields, replace

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import held_payload
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models
from rl_agent.splitfusion_hybrid_sac_run4_v1 import quality_adapter
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel as src
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    EndpointEvidence,
    PolicySceneView,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)


SESSION = "11111111-1111-4111-8111-111111111111"
UE_ID = "ue-1"
CLOCK = "RUN4_KERNEL_TEST_CLOCK"


def _d(character: str) -> str:
    return character * 64


class Run4SequentialKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = action_contract.load_contract()
        executable = cls.catalog.resolve(0, 0.3)
        cls.action = ExecutedActionIdentity.from_executable_action(
            executable, cls.catalog
        )

    @staticmethod
    def support() -> src.KernelSupportV1:
        latency = src.LatencySupportV1(
            ue_action_path_ns=src.NumericSupportV1(0, 200_000_000),
            feature_uplink_ns=src.NumericSupportV1(0, 200_000_000),
            edge_decompression_ns=src.NumericSupportV1(0, 200_000_000),
            model_tail_ns=src.NumericSupportV1(0, 200_000_000),
            quality_evaluation_ns=src.NumericSupportV1(0, 200_000_000),
            feedback_downlink_ns=src.NumericSupportV1(0, 200_000_000),
        )
        split = src.FitValidationSplitV1(
            fit_cell_ids=("fit-a", "fit-b"),
            validation_cell_ids=("validation-a",),
            assignment_evidence_sha256=_d("1"),
        )
        return src.KernelSupportV1(
            support_id="run4-test-support",
            support_version=1,
            evidence_sha256=_d("2"),
            per_tensor_payload_bytes=src.NumericSupportV1(500, 2_000),
            pre_enqueue_backlog_bytes=src.NumericSupportV1(0, 5_000),
            observed_prior_ul_mcs=(0, 7, 12),
            latency=latency,
            maximum_hold_tensors=4,
            calibration_split=split,
        )

    @classmethod
    def prerequisites(cls) -> src.KernelVerifierPrerequisitesV1:
        support = cls.support()
        provenance = src.KernelProvenanceBindingV1(
            binding_id="run4-test-kernel",
            binding_version=1,
            raw_campaign_manifest_sha256=_d("3"),
            raw_decisions_sha256=_d("4"),
            corrected_analysis_generation=2,
            corrected_analysis_v2_sha256=_d("5"),
            corrected_decisions_v2_sha256=_d("6"),
            corrected_analysis_verdict=src.ACCEPTED_ANALYSIS_VERDICT,
            transport_fit_sha256=_d("7"),
            queue_transition_fit_sha256=_d("8"),
            ue_action_path_latency_evidence_sha256=_d("e"),
            tail_latency_evidence_sha256=_d("9"),
            quality_evaluation_latency_evidence_sha256=_d("a"),
            feedback_ack_latency_evidence_sha256=_d("b"),
            quality_feedback_report_sha256=_d("a"),
            quality_feedback_manifest_sha256=_d("b"),
            quality_adapter_binding_sha256=_d("c"),
            held_provider_binding_sha256=_d("d"),
            actor_feature_schema_sha256=contract.FEATURE_SCHEMA_SHA256,
            empirical_scaling_sha256=_d("e"),
            freshness_policy_sha256=_d("f"),
            support_sha256=support.canonical_sha256,
        )
        return src.KernelVerifierPrerequisitesV1(
            provenance=provenance,
            support=support,
            fit_report_sha256=_d("e"),
            validation_report_sha256=_d("f"),
            fitted_model_sha256=_d("0"),
            fit_validation_disjoint=True,
            sequential_queue_state_validated=True,
            per_tensor_payload_support_covered=True,
            full_feedback_frame_join_validated=True,
            missingness_preserved=True,
            actor_profile_label_absent=True,
            empirical_scaling_fit_only=True,
            freshness_policy_validated=True,
            prior_action_outcome_chain_validated=True,
            actor_state_group_variation_validated=True,
            fit_prediction_count=100,
            validation_prediction_count=20,
        )

    @staticmethod
    def state(
        sequence: int,
        *,
        mcs: int | None = 0,
        backlog: int | None = 0,
    ) -> src.RadioQueueStateV1:
        return src.RadioQueueStateV1(
            session_uuid=SESSION,
            ue_id=UE_ID,
            decision_seq=sequence,
            prior_ul_mcs=src.IntegerObservationV1(
                value=mcs,
                missing_reason=None if mcs is not None else "MCS_NOT_AVAILABLE",
                source_decision_seq=max(0, sequence - 1),
                provenance_sha256=_d("1"),
            ),
            pre_enqueue_backlog_bytes=src.IntegerObservationV1(
                value=backlog,
                missing_reason=(
                    None if backlog is not None else "BACKLOG_NOT_AVAILABLE"
                ),
                source_decision_seq=max(0, sequence - 1),
                provenance_sha256=_d("2"),
            ),
        )

    @classmethod
    def reward_tensor(
        cls, sequence: int, *, payload: int = 1_000, quality: float = 0.7
    ) -> quality_adapter.RewardTensorResultV1:
        scene = PolicySceneView(camera_si=100.0, radar_p40=0.4)
        scene_sha = canonical_sha256(
            {
                "record": "splitfusion_run4_policy_scene_si_p40_v1",
                "value": scene.to_dict(),
            }
        )
        evidence = quality_adapter.RewardTensorEvidenceV1(
            adapter_binding_sha256=_d("c"),
            fit_selection_sha256=_d("3"),
            action_sha256=cls.action.canonical_sha256(),
            mode_id=cls.action.mode_id,
            q_e4=cls.action.q_e4,
            q_perc=quality,
            offered_payload_bytes=payload,
            payload_evidence_class=(
                contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            ),
            policy_scene_sha256=scene_sha,
            sample_id="fit-sample",
            episode_id="fit-episode",
            frame_id=100 + sequence,
            surface_evidence_status=quality_adapter.EXACT_GRID_ROW_EVIDENCE,
            q_perc_status="VALID",
            endpoints=(
                EndpointEvidence(
                    q_e4=cls.action.q_e4,
                    row_key_sha256=_d("4"),
                    row_sha256=_d("5"),
                    datagram_count=1,
                    quality_valid=True,
                    quality_status="VALID",
                ),
            ),
        )
        tensor = contract.HoldTensorV1(
            tensor_seq=sequence * 10,
            offered_payload_bytes=payload,
            payload_evidence_class=(
                contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            ),
            payload_provenance_sha256=evidence.canonical_sha256,
            reward_requested=True,
        )
        return quality_adapter.RewardTensorResultV1(
            action=cls.action,
            policy_scene=scene,
            q_perc=quality,
            tensor=tensor,
            evidence=evidence,
        )

    @classmethod
    def held_tensor(
        cls, sequence: int, *, payload: int = 900
    ) -> quality_adapter.HeldTensorResultV1:
        estimate = held_payload.HeldPayloadEstimateV1(
            evidence_label=held_payload.EVIDENCE_LABEL,
            provider_binding_sha256=_d("d"),
            selection_identity_sha256=_d("6"),
            counter=held_payload.HeldSelectionCounterV1(
                session_id="run4-kernel-test",
                decision_seq=sequence,
                held_ordinal=1,
                rng_stream_id="run4-kernel-test-rng",
            ),
            rng_draw=0.25,
            selected_sample_id="held-sample",
            selected_episode_id="held-episode",
            selected_frame_id=200 + sequence,
            selected_scene_source_sha256=_d("7"),
            selected_curve_sha256=_d("8"),
            source_selection_sha256=_d("9"),
            source_database_sha256=_d("a"),
            inclusion_probability=0.5,
            sampling_weight=2.0,
            marginal_selection_probability=0.5,
            mode_id=cls.action.mode_id,
            q_e4=cls.action.q_e4,
            total_transmitted_bytes=payload,
            interpolation_status=held_payload.EXACT_NODE_STATUS,
            endpoints=(
                held_payload.EndpointEvidenceV1(
                    q_e4=cls.action.q_e4,
                    total_transmitted_bytes=payload,
                    source_row_sha256=_d("b"),
                ),
            ),
        )
        tensor = contract.HoldTensorV1(
            tensor_seq=sequence * 10 + 1,
            offered_payload_bytes=payload,
            payload_evidence_class=(
                contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
            ),
            payload_provenance_sha256=estimate.canonical_sha256,
            reward_requested=False,
        )
        return quality_adapter.HeldTensorResultV1(
            action=cls.action, tensor=tensor, estimate=estimate
        )

    @classmethod
    def decision(
        cls,
        sequence: int,
        *,
        partition: src.KernelCalibrationPartition = src.KernelCalibrationPartition.FIT,
        payload: int = 1_000,
        quality: float = 0.7,
        current_radio_state: src.RadioQueueStateV1 | None = None,
    ) -> src.KernelDecisionInputV1:
        if current_radio_state is None:
            current_radio_state = (
                cls.state(0)
                if sequence == 0
                else cls.state(sequence, mcs=7, backlog=100)
            )
        return src.KernelDecisionInputV1(
            identity=contract.DecisionIdentityV1(SESSION, UE_ID, sequence),
            current_radio_state_sha256=current_radio_state.canonical_sha256,
            action=cls.action,
            reward_tensor=cls.reward_tensor(
                sequence, payload=payload, quality=quality
            ),
            held_tensors=(cls.held_tensor(sequence),),
            action_open_timestamp_ns=1_000_000_000 + sequence * 200_000_000,
            clock_domain=CLOCK,
            calibration_partition=partition,
        )

    @staticmethod
    def latency(
        *, feature_uplink_ns: int = 50_000_000
    ) -> src.FeedbackLatencyBreakdownV1:
        return src.FeedbackLatencyBreakdownV1(
            ue_action_path_ns=20_000_000,
            feature_uplink_ns=feature_uplink_ns,
            edge_decompression_ns=10_000_000,
            model_tail_ns=20_000_000,
            quality_evaluation_ns=3_000_000,
            feedback_downlink_ns=5_000_000,
            ue_action_path_evidence_sha256=_d("e"),
            feature_transport_evidence_sha256=_d("7"),
            tail_evidence_sha256=_d("9"),
            quality_evaluation_evidence_sha256=_d("a"),
            feedback_ack_evidence_sha256=_d("b"),
        )

    @classmethod
    def prediction(
        cls,
        decision: src.KernelDecisionInputV1,
        *,
        current_radio_state: src.RadioQueueStateV1 | None = None,
        terminal: src.KernelTerminalKind = src.KernelTerminalKind.DELIVERED_FEEDBACK,
        latency: src.FeedbackLatencyBreakdownV1 | None = None,
        source_cell_id: str = "fit-a",
        next_mcs: int | None = 7,
        next_backlog: int | None = 100,
    ) -> src.EmpiricalStepPredictionV1:
        prereq = cls.prerequisites()
        if current_radio_state is None:
            current_radio_state = (
                cls.state(0)
                if decision.identity.decision_seq == 0
                else cls.state(decision.identity.decision_seq, mcs=7, backlog=100)
            )
        request = decision.to_prediction_request(current_radio_state)
        if terminal is src.KernelTerminalKind.DELIVERED_FEEDBACK:
            breakdown = cls.latency() if latency is None else latency
            elapsed = breakdown.full_feedback_ns
        elif terminal is src.KernelTerminalKind.TIMEOUT:
            breakdown = None
            elapsed = src.TIMEOUT_RESOLUTION_ELAPSED_NS
        else:
            breakdown = None
            elapsed = 50_000_000
        return src.EmpiricalStepPredictionV1(
            prediction_request_sha256=request.canonical_sha256,
            kernel_provenance_sha256=prereq.provenance.canonical_sha256,
            fitted_model_sha256=prereq.fitted_model_sha256,
            calibration_partition=decision.calibration_partition,
            source_cell_id=source_cell_id,
            terminal_kind=terminal,
            terminal_elapsed_ns=elapsed,
            latency=breakdown,
            next_state=cls.state(
                decision.identity.decision_seq + 1,
                mcs=next_mcs,
                backlog=next_backlog,
            ),
            source_row_sha256=_d("f"),
        )

    @classmethod
    def kernel(
        cls,
        *,
        partition: src.KernelCalibrationPartition = src.KernelCalibrationPartition.FIT,
    ) -> src.Run4SequentialRadioQueueKernelV1:
        prerequisites = cls.prerequisites()
        authorization = src._issue_test_only_authorization(prerequisites)
        return src.Run4SequentialRadioQueueKernelV1(
            prerequisites=prerequisites,
            authorization=authorization,
            calibration_partition=partition,
            initial_state=cls.state(0),
        )

    @classmethod
    def policy_features_with_previous(
        cls,
        previous: contract.PreviousOutcomeV1,
    ) -> contract.PolicyFeatureVectorV2:
        """Hold the four current measurements fixed and vary only history."""

        current_identity = contract.DecisionIdentityV1(SESSION, UE_ID, 1)
        sample_identity = contract.SampleIdentityV1(SESSION, UE_ID, 101)

        def observation(
            *,
            kind: contract.MeasurementKind,
            observer: contract.Observer,
            direction: contract.LinkDirection,
            value: float | int,
        ) -> contract.ScalarObservationV1:
            metadata = contract.MeasurementMetadataV1(
                identity=sample_identity,
                kind=kind,
                observer=observer,
                link_direction=direction,
                source=f"run4-sequential-kernel-test:{kind.value}",
                source_timestamp_ns=1_000_000_000,
                available_timestamp_ns=1_010_000_000,
                clock_domain=CLOCK,
                valid=True,
            )
            return contract.ScalarObservationV1(
                value=value,
                metadata=metadata,
                missing_reason=None,
            )

        prior_mcs_scalar = observation(
            kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
            observer=contract.Observer.UE,
            direction=contract.LinkDirection.UPLINK,
            value=12,
        )
        state = contract.PolicyStateV2(
            identity=current_identity,
            camera_si=observation(
                kind=contract.MeasurementKind.CAMERA_SI,
                observer=contract.Observer.SCENE_PIPELINE,
                direction=contract.LinkDirection.NOT_APPLICABLE,
                value=20.0,
            ),
            radar_p40=observation(
                kind=contract.MeasurementKind.RADAR_P40,
                observer=contract.Observer.SCENE_PIPELINE,
                direction=contract.LinkDirection.NOT_APPLICABLE,
                value=0.4,
            ),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=prior_mcs_scalar,
                mcs_table=contract.UL_MCS_TABLE_ID,
                harq_round=0,
                new_data_indicator=1,
                grant_identity="run4-sequential-kernel-test-grant",
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
            ),
            pre_action_rlc_backlog=observation(
                kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                observer=contract.Observer.UE,
                direction=contract.LinkDirection.UPLINK,
                value=1023,
            ),
            previous=previous,
        )
        boundary = contract.DecisionBoundaryV1(
            identity=current_identity,
            state_commit_timestamp_ns=1_050_000_000,
            action_open_timestamp_ns=1_060_000_000,
            clock_domain=CLOCK,
        )
        freshness = contract.FreshnessPolicyV2(
            policy_id="run4-sequential-kernel-test-freshness",
            policy_version=1,
            evidence_sha256=_d("4"),
            camera_si_max_age_ns=100_000_000,
            radar_p40_max_age_ns=100_000_000,
            prior_ul_mcs_max_age_ns=100_000_000,
            pre_action_rlc_backlog_max_age_ns=100_000_000,
        )
        scaling = contract.EmpiricalScalingV2(
            scaling_id="run4-sequential-kernel-test-scaling",
            scaling_version=1,
            evidence_sha256=_d("5"),
            camera_si_center=10.0,
            camera_si_scale=5.0,
            backlog_log1p_scale=math.log1p(1023),
        )
        guarded = contract.guard_state_for_action(state, boundary, freshness)
        return contract.build_policy_features(guarded, scaling)

    def test_production_verifier_is_fail_closed_until_v2_is_registered(self) -> None:
        self.assertIsNone(src.REGISTERED_KERNEL_PREREQUISITES_SHA256)
        with self.assertRaises(src.CorrectedEvidenceUnavailable):
            src.verify_kernel_prerequisites(self.prerequisites())

    def test_stale_generation_one_analysis_is_structurally_rejected(self) -> None:
        provenance = self.prerequisites().provenance
        with self.assertRaisesRegex(src.EvidenceBindingError, "analysis_v1"):
            replace(provenance, corrected_analysis_generation=1)

    def test_valid_zero_is_not_missing_and_missing_is_not_zero(self) -> None:
        zero = self.state(0, mcs=0, backlog=0)
        self.assertTrue(zero.actor_ready)
        self.assertEqual(zero.prior_ul_mcs.value, 0)
        self.assertEqual(zero.pre_enqueue_backlog_bytes.value, 0)
        missing = self.state(0, mcs=None, backlog=None)
        self.assertFalse(missing.actor_ready)
        self.assertIsNone(missing.prior_ul_mcs.value)
        with self.assertRaisesRegex(src.SupportViolation, "never be replaced by zero"):
            missing.require_actor_ready()

    def test_success_uses_exact_quality_and_full_frame_level_latency(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        prediction = self.prediction(decision)
        result = kernel.advance(decision=decision, prediction=prediction)
        resolution = contract.resolve_reward(result.reward_event)
        self.assertEqual(resolution.terminal, contract.RewardTerminal.SUCCESS)
        self.assertEqual(resolution.q_perc, decision.reward_tensor.q_perc)
        self.assertEqual(resolution.latency_ms, 108.0)
        self.assertAlmostEqual(
            resolution.reward, 0.7 - 0.25 * (108.0 / 170.0), places=12
        )
        self.assertEqual(result.latency.transport_ns, 55_000_000)
        self.assertEqual(result.latency.non_network_ns, 53_000_000)
        self.assertEqual(result.latency.full_feedback_ns, 108_000_000)
        self.assertEqual(kernel.current_state.decision_seq, 1)
        self.assertFalse(result.replay_export_allowed)

    def test_delivered_after_deadline_is_structurally_rejected(self) -> None:
        decision = self.decision(0)
        latency = self.latency(feature_uplink_ns=120_000_001)
        with self.assertRaisesRegex(src.PredictionViolation, "late-orphan"):
            self.prediction(decision, latency=latency)

    def test_timeout_must_be_strictly_after_inclusive_boundary(self) -> None:
        decision = self.decision(0)
        prediction = self.prediction(
            decision, terminal=src.KernelTerminalKind.TIMEOUT
        )
        with self.assertRaisesRegex(src.PredictionViolation, "strictly after"):
            replace(prediction, terminal_elapsed_ns=contract.REWARD_DEADLINE_NS)

    def test_timeout_closes_at_first_representable_instant_not_late_ack(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        prediction = self.prediction(
            decision, terminal=src.KernelTerminalKind.TIMEOUT
        )
        self.assertEqual(
            prediction.terminal_elapsed_ns,
            contract.REWARD_DEADLINE_NS + 1,
        )
        with self.assertRaisesRegex(src.PredictionViolation, "first nanosecond"):
            replace(
                prediction,
                terminal_elapsed_ns=contract.REWARD_DEADLINE_NS + 2,
            )
        result = kernel.advance(decision=decision, prediction=prediction)
        self.assertEqual(
            result.cycle_end_timestamp_ns,
            decision.action_open_timestamp_ns + contract.REWARD_DEADLINE_NS + 1,
        )
        resolution = contract.resolve_reward(result.reward_event)
        self.assertEqual(resolution.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertEqual(resolution.reward, -1.0)

    def test_registered_delivery_failure_carries_no_quality_or_latency(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        prediction = self.prediction(
            decision,
            terminal=src.KernelTerminalKind.REGISTERED_DELIVERY_FAILURE,
        )
        result = kernel.advance(decision=decision, prediction=prediction)
        resolution = contract.resolve_reward(result.reward_event)
        self.assertEqual(
            resolution.terminal,
            contract.RewardTerminal.REGISTERED_DELIVERY_FAILURE,
        )
        self.assertEqual(resolution.reward, -1.0)
        self.assertIsNone(resolution.q_perc)
        self.assertIsNone(result.latency)

    def test_wrong_calibration_split_cell_is_rejected_transactionally(self) -> None:
        kernel = self.kernel()
        before = kernel.checkpoint().canonical_sha256
        decision = self.decision(0)
        prediction = self.prediction(decision, source_cell_id="validation-a")
        with self.assertRaises(src.SupportViolation):
            kernel.advance(decision=decision, prediction=prediction)
        self.assertEqual(kernel.checkpoint().canonical_sha256, before)

    def test_payload_outside_support_is_rejected_before_state_mutation(self) -> None:
        kernel = self.kernel()
        before = kernel.checkpoint().canonical_sha256
        decision = self.decision(0, payload=2_001)
        prediction = self.prediction(decision)
        with self.assertRaisesRegex(src.SupportViolation, "offered_payload_bytes"):
            kernel.advance(decision=decision, prediction=prediction)
        self.assertEqual(kernel.checkpoint().canonical_sha256, before)

    def test_latency_evidence_sources_cannot_be_conflated(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        latency = replace(
            self.latency(), feature_transport_evidence_sha256=_d("8")
        )
        prediction = self.prediction(decision, latency=latency)
        with self.assertRaisesRegex(src.EvidenceBindingError, "transport"):
            kernel.advance(decision=decision, prediction=prediction)

    def test_prediction_must_bind_to_exact_decision(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        prediction = replace(
            self.prediction(decision), prediction_request_sha256=_d("1")
        )
        with self.assertRaisesRegex(src.PredictionViolation, "another causal"):
            kernel.advance(decision=decision, prediction=prediction)

    def test_decision_and_prediction_cannot_be_reused_across_radio_states(self) -> None:
        prerequisites = self.prerequisites()
        authorization = src._issue_test_only_authorization(prerequisites)
        source_state = self.state(0, mcs=0, backlog=0)
        other_state = self.state(0, mcs=12, backlog=5_000)
        decision = self.decision(0, current_radio_state=source_state)
        prediction = self.prediction(
            decision, current_radio_state=source_state
        )
        other_kernel = src.Run4SequentialRadioQueueKernelV1(
            prerequisites=prerequisites,
            authorization=authorization,
            calibration_partition=src.KernelCalibrationPartition.FIT,
            initial_state=other_state,
        )
        before = other_kernel.checkpoint().canonical_sha256
        with self.assertRaisesRegex(src.SequenceViolation, "different current"):
            other_kernel.advance(decision=decision, prediction=prediction)
        self.assertEqual(other_kernel.checkpoint().canonical_sha256, before)
        self.assertEqual(other_kernel.current_state, other_state)

    def test_exact_previous_action_and_outcome_reach_next_actor_input(self) -> None:
        previous_identity = contract.DecisionIdentityV1(SESSION, UE_ID, 0)
        alternate_executable = self.catalog.resolve(5, 0.7)
        alternate_action = ExecutedActionIdentity.from_executable_action(
            alternate_executable, self.catalog
        )
        success = contract.resolve_reward(
            contract.RewardEventV1(
                identity=previous_identity,
                action=self.action,
                kind=contract.RewardEventKind.DELIVERED_SUCCESS,
                action_open_timestamp_ns=800_000_000,
                resolution_timestamp_ns=900_000_000,
                clock_domain=CLOCK,
                source="run4-sequential-kernel-test",
                q_perc=0.8,
            )
        )
        failure = contract.resolve_reward(
            contract.RewardEventV1(
                identity=previous_identity,
                action=alternate_action,
                kind=contract.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
                action_open_timestamp_ns=800_000_000,
                resolution_timestamp_ns=900_000_000,
                clock_domain=CLOCK,
                source="run4-sequential-kernel-test",
            )
        )
        success_features = self.policy_features_with_previous(
            contract.PreviousOutcomeV1.from_resolution(success)
        )
        failure_features = self.policy_features_with_previous(
            contract.PreviousOutcomeV1.from_resolution(failure)
        )
        success_named = success_features.as_dict()
        failure_named = failure_features.as_dict()
        current_names = contract.POLICY_FEATURE_ORDER[:4]
        self.assertTrue(all(success_named[name] != 0.0 for name in current_names))
        self.assertEqual(
            tuple(success_named[name] for name in current_names),
            tuple(failure_named[name] for name in current_names),
        )
        self.assertNotEqual(success_features.as_tuple(), failure_features.as_tuple())
        self.assertEqual(success_named["prev_present"], 1.0)
        self.assertEqual(failure_named["prev_present"], 1.0)
        self.assertEqual(success_named["prev_success"], 1.0)
        self.assertEqual(failure_named["prev_success"], 0.0)
        self.assertEqual(success_named["prev_quality_qperc"], 0.8)
        self.assertEqual(failure_named["prev_quality_qperc"], 0.0)

        bundle = models.build_run4_models(actor_seed=17, critic_seed=23)
        success_tensor = torch.tensor(
            [success_features.as_tuple()], dtype=torch.float32
        )
        failure_tensor = torch.tensor(
            [failure_features.as_tuple()], dtype=torch.float32
        )
        with torch.no_grad():
            success_heads = bundle.actor(success_tensor)
            failure_heads = bundle.actor(failure_tensor)
            success_execution = bundle.actor.deterministic_execution(success_tensor)
            failure_execution = bundle.actor.deterministic_execution(failure_tensor)
        self.assertFalse(torch.equal(success_heads.logits, failure_heads.logits))
        self.assertFalse(torch.equal(success_heads.mean, failure_heads.mean))
        self.assertNotEqual(
            (
                int(success_execution.mode_index.item()),
                int(success_execution.q_e4.item()),
            ),
            (
                int(failure_execution.mode_index.item()),
                int(failure_execution.q_e4.item()),
            ),
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_next_state_remains_missing_and_blocks_next_action(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        prediction = self.prediction(
            decision, next_mcs=None, next_backlog=None
        )
        kernel.advance(decision=decision, prediction=prediction)
        self.assertFalse(kernel.current_state.actor_ready)
        next_decision = self.decision(
            1, current_radio_state=kernel.current_state
        )
        next_prediction = self.prediction(
            next_decision, current_radio_state=kernel.current_state
        )
        with self.assertRaisesRegex(src.SupportViolation, "external fallback"):
            kernel.advance(decision=next_decision, prediction=next_prediction)

    def test_exact_checkpoint_restore_and_continuation(self) -> None:
        prerequisites = self.prerequisites()
        authorization = src._issue_test_only_authorization(prerequisites)
        kernel = src.Run4SequentialRadioQueueKernelV1(
            prerequisites=prerequisites,
            authorization=authorization,
            calibration_partition=src.KernelCalibrationPartition.FIT,
            initial_state=self.state(0),
        )
        first = self.decision(0)
        kernel.advance(decision=first, prediction=self.prediction(first))
        checkpoint = kernel.checkpoint()
        restored = src.Run4SequentialRadioQueueKernelV1.restore(
            checkpoint=checkpoint,
            prerequisites=prerequisites,
            authorization=authorization,
        )
        self.assertEqual(
            restored.checkpoint().canonical_sha256, checkpoint.canonical_sha256
        )
        second = self.decision(1)
        result = restored.advance(
            decision=second, prediction=self.prediction(second)
        )
        self.assertEqual(result.next_radio_state.decision_seq, 2)
        self.assertEqual(restored.completed_steps, 2)

    def test_checkpoint_from_another_model_is_rejected(self) -> None:
        kernel = self.kernel()
        checkpoint = kernel.checkpoint()
        prerequisites = replace(
            self.prerequisites(), fitted_model_sha256=_d("1")
        )
        authorization = src._issue_test_only_authorization(prerequisites)
        with self.assertRaisesRegex(src.CheckpointError, "binding differs"):
            src.Run4SequentialRadioQueueKernelV1.restore(
                checkpoint=checkpoint,
                prerequisites=prerequisites,
                authorization=authorization,
            )

    def test_validation_kernel_accepts_only_validation_cells(self) -> None:
        kernel = self.kernel(partition=src.KernelCalibrationPartition.VALIDATION)
        decision = self.decision(
            0, partition=src.KernelCalibrationPartition.VALIDATION
        )
        prediction = self.prediction(
            decision, source_cell_id="validation-a"
        )
        result = kernel.advance(decision=decision, prediction=prediction)
        self.assertEqual(
            result.calibration_partition,
            src.KernelCalibrationPartition.VALIDATION,
        )

    def test_result_converts_without_changing_hold_or_reward_event(self) -> None:
        kernel = self.kernel()
        decision = self.decision(0)
        result = kernel.advance(
            decision=decision, prediction=self.prediction(decision)
        )
        envelope = result.to_environment_result()
        self.assertEqual(envelope.hold.canonical_sha256(), result.hold.canonical_sha256())
        self.assertEqual(
            envelope.reward_event.canonical_sha256(),
            result.reward_event.canonical_sha256(),
        )

    def test_no_profile_label_or_current_outcome_enters_decision_input(self) -> None:
        names = tuple(item.name for item in fields(src.KernelDecisionInputV1))
        joined = " ".join(names).lower()
        self.assertNotIn("profile", joined)
        self.assertNotIn("outcome", joined)
        self.assertEqual(
            tuple(contract.POLICY_FEATURE_ORDER),
            tuple(contract.POLICY_FEATURE_ORDER),
        )

    def test_prediction_request_exposes_exact_state_and_ordered_payloads(self) -> None:
        state = self.state(0, mcs=12, backlog=4_321)
        decision = self.decision(0, current_radio_state=state)
        request = decision.to_prediction_request(state)
        self.assertEqual(request.current_radio_state, state)
        self.assertEqual(request.current_radio_state.prior_ul_mcs.value, 12)
        self.assertEqual(
            request.current_radio_state.prior_ul_mcs.provenance_sha256,
            state.prior_ul_mcs.provenance_sha256,
        )
        self.assertEqual(
            request.current_radio_state.pre_enqueue_backlog_bytes.value,
            4_321,
        )
        self.assertEqual(request.action, decision.action)
        self.assertEqual(
            request.reward_payload.offered_payload_bytes,
            decision.reward_tensor.tensor.offered_payload_bytes,
        )
        self.assertEqual(
            tuple(item.offered_payload_bytes for item in request.held_payloads),
            tuple(
                item.tensor.offered_payload_bytes
                for item in decision.held_tensors
            ),
        )
        alternate_state = self.state(0, mcs=0, backlog=0)
        alternate = self.decision(0, current_radio_state=alternate_state)
        alternate_request = alternate.to_prediction_request(alternate_state)
        self.assertNotEqual(
            request.canonical_sha256, alternate_request.canonical_sha256
        )

    def test_prediction_request_excludes_quality_scene_and_frame_identity(self) -> None:
        state = self.state(0, mcs=7, backlog=100)
        low_quality = self.decision(
            0, quality=0.2, current_radio_state=state
        )
        high_quality = self.decision(
            0, quality=0.9, current_radio_state=state
        )
        low_request = low_quality.to_prediction_request(state)
        high_request = high_quality.to_prediction_request(state)
        # Perception quality changes the reward-side decision evidence but not
        # the radio/queue model's causal inputs.
        self.assertNotEqual(
            low_quality.canonical_sha256, high_quality.canonical_sha256
        )
        self.assertEqual(
            low_request.canonical_sha256, high_request.canonical_sha256
        )
        serialized_names = " ".join(low_request.to_dict()).lower()
        public_fields = " ".join(
            item.name
            for item in fields(src.PredictionRequestV1)
            if not item.name.startswith("_")
        ).lower()
        for forbidden in (
            "q_perc",
            "quality",
            "scene",
            "frame",
            "sample",
            "outcome",
            "terminal",
        ):
            self.assertNotIn(forbidden, serialized_names)
            self.assertNotIn(forbidden, public_fields)

    def test_prediction_request_is_attested_and_tamper_evident(self) -> None:
        state = self.state(0, mcs=7, backlog=100)
        decision = self.decision(0, current_radio_state=state)
        request = decision.to_prediction_request(state)
        self.assertEqual(
            request.canonical_sha256,
            decision.prediction_request_sha256,
        )
        with self.assertRaisesRegex(src.PredictionViolation, "attestation"):
            replace(
                request,
                current_radio_state=self.state(0, mcs=12, backlog=5_000),
            )
        unattested = src.PredictionRequestV1(
            identity=request.identity,
            current_radio_state=request.current_radio_state,
            action=request.action,
            reward_payload=request.reward_payload,
            held_payloads=request.held_payloads,
            calibration_partition=request.calibration_partition,
        )
        with self.assertRaisesRegex(src.PredictionViolation, "derived"):
            _ = unattested.canonical_sha256
        with self.assertRaisesRegex(src.SequenceViolation, "different current"):
            decision.to_prediction_request(
                self.state(0, mcs=12, backlog=5_000)
            )

    def test_fit_and_validation_cells_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(src.EvidenceBindingError, "overlap"):
            src.FitValidationSplitV1(
                fit_cell_ids=("cell-a",),
                validation_cell_ids=("cell-a",),
                assignment_evidence_sha256=_d("1"),
            )

    def test_module_has_no_runtime_or_accelerator_dependency(self) -> None:
        tree = ast.parse(inspect.getsource(src))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        forbidden_roots = {"carla", "docker", "socket", "subprocess", "torch"}
        self.assertTrue(
            forbidden_roots.isdisjoint(name.split(".", 1)[0] for name in imported)
        )


if __name__ == "__main__":
    unittest.main()
