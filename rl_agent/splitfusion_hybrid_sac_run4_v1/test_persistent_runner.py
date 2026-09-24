"""CPU-only causal and recovery tests for the persistent Run-4 runner."""

from __future__ import annotations

import math
import builtins
import importlib
import random
import socket
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    canonical_sha256,
)

from . import exploration, fit_scene_provider, held_payload, models
from . import persistent_runner as src, quality_adapter
from . import production_state_provider, replay, run4_contract as contract
from . import sequential_kernel, trainer
from . import test_fit_scene_provider as _fit_fixture


SESSION = "22222222-2222-4222-8222-222222222222"
UE_ID = "ue-run4-persistent-test"
CLOCK = "RUN4_PERSISTENT_TEST_CLOCK"


def _d(label: str) -> str:
    return canonical_sha256({"run4_persistent_test": label})


class _MeasuredStateStager:
    """Deterministic test source whose successor times are measured outputs."""

    binding_sha256 = _d("state-stager-binding")

    def __init__(self) -> None:
        self.count = 0

    def stage(
        self,
        *,
        identity: contract.DecisionIdentityV1,
        scene_draw,
        radio_state: sequential_kernel.RadioQueueStateV1,
        previous: contract.PreviousOutcomeV1 | None,
        minimum_state_commit_timestamp_ns: int,
    ) -> production_state_provider.StagedDecisionInputsV1:
        if identity.decision_seq == 0:
            if minimum_state_commit_timestamp_ns != 0 or previous is not None:
                raise AssertionError("genesis must have no completed outcome")
            commit_ns = 1_000_000_000
        else:
            if previous is None:
                raise AssertionError("a continuation requires the exact outcome")
            # These are the fixture's measured times.  The runner supplies
            # only the causal lower bound; it does not prescribe action-open.
            commit_ns = minimum_state_commit_timestamp_ns + 5_000_000
        open_ns = commit_ns + 7_000_000
        source_ns = commit_ns - 3_000_000
        available_ns = commit_ns - 1_000_000
        timing = production_state_provider.ObservationTimingV1(
            source="run4-persistent-test-measurement",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=CLOCK,
        )
        boundary = contract.DecisionBoundaryV1(
            identity=identity,
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=open_ns,
            clock_domain=CLOCK,
        )
        self.count += 1
        return production_state_provider.StagedDecisionInputsV1(
            identity=identity,
            boundary=boundary,
            scene_draw=scene_draw,
            scene_timing=timing,
            radio_state=radio_state,
            prior_grant=production_state_provider.PriorGrantProvenanceV1(
                radio_observation_provenance_sha256=(
                    radio_state.prior_ul_mcs.provenance_sha256
                ),
                grant_identity=f"grant-{identity.decision_seq}",
                new_data_indicator=identity.decision_seq % 2,
                harq_round=0,
                mcs_table=contract.UL_MCS_TABLE_ID,
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                timing=timing,
            ),
            backlog=production_state_provider.BacklogProvenanceV1(
                radio_observation_provenance_sha256=(
                    radio_state.pre_enqueue_backlog_bytes.provenance_sha256
                ),
                timing=timing,
            ),
            expected_previous_sha256=(
                None if previous is None else previous.canonical_sha256()
            ),
        )

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state) -> None:
        if set(state) != {"count"} or type(state["count"]) is not int:
            raise ValueError("invalid stager checkpoint")
        self.count = state["count"]


class _HeldAwarePredictionProvider:
    """External test fitter whose queue successor uses the held payload."""

    binding_sha256 = _d("prediction-provider-binding")

    def __init__(
        self,
        prerequisites: sequential_kernel.KernelVerifierPrerequisitesV1,
    ) -> None:
        self.prerequisites = prerequisites
        self.count = 0
        self.held_payloads: list[float] = []

    def predict(
        self, decision: sequential_kernel.KernelDecisionInputV1
    ) -> sequential_kernel.EmpiricalStepPredictionV1:
        held_payload = decision.hold.held_offered_payload_bytes
        self.held_payloads.append(held_payload)
        sequence = decision.identity.decision_seq
        next_backlog = min(9_000_000, 100 + int(held_payload // 10))
        next_mcs = 7 if sequence % 2 == 0 else 12
        next_state = sequential_kernel.RadioQueueStateV1(
            session_uuid=decision.identity.session_uuid,
            ue_id=decision.identity.ue_id,
            decision_seq=sequence + 1,
            prior_ul_mcs=sequential_kernel.IntegerObservationV1(
                value=next_mcs,
                missing_reason=None,
                source_decision_seq=sequence,
                provenance_sha256=_d(f"mcs-{sequence + 1}"),
            ),
            pre_enqueue_backlog_bytes=sequential_kernel.IntegerObservationV1(
                value=next_backlog,
                missing_reason=None,
                source_decision_seq=sequence,
                provenance_sha256=_d(f"backlog-{sequence + 1}"),
            ),
        )
        if sequence == 1:
            kind = sequential_kernel.KernelTerminalKind.TIMEOUT
            latency = None
            elapsed = contract.REWARD_DEADLINE_NS + 1
        else:
            kind = sequential_kernel.KernelTerminalKind.DELIVERED_FEEDBACK
            provenance = self.prerequisites.provenance
            latency = sequential_kernel.FeedbackLatencyBreakdownV1(
                ue_action_path_ns=20_000_000,
                feature_uplink_ns=50_000_000,
                edge_decompression_ns=10_000_000,
                model_tail_ns=20_000_000,
                quality_evaluation_ns=3_000_000,
                feedback_downlink_ns=5_000_000,
                ue_action_path_evidence_sha256=(
                    provenance.ue_action_path_latency_evidence_sha256
                ),
                feature_transport_evidence_sha256=provenance.transport_fit_sha256,
                tail_evidence_sha256=provenance.tail_latency_evidence_sha256,
                quality_evaluation_evidence_sha256=(
                    provenance.quality_evaluation_latency_evidence_sha256
                ),
                feedback_ack_evidence_sha256=(
                    provenance.feedback_ack_latency_evidence_sha256
                ),
            )
            elapsed = latency.full_feedback_ns
        self.count += 1
        return sequential_kernel.EmpiricalStepPredictionV1(
            decision_input_sha256=decision.canonical_sha256,
            kernel_provenance_sha256=(
                self.prerequisites.provenance.canonical_sha256
            ),
            fitted_model_sha256=self.prerequisites.fitted_model_sha256,
            calibration_partition=decision.calibration_partition,
            source_cell_id="fit-a",
            terminal_kind=kind,
            terminal_elapsed_ns=elapsed,
            latency=latency,
            next_state=next_state,
            source_row_sha256=_d(f"prediction-row-{sequence}"),
        )

    def state_dict(self):
        return {"count": self.count, "held_payloads": tuple(self.held_payloads)}

    def load_state_dict(self, state) -> None:
        if set(state) != {"count", "held_payloads"}:
            raise ValueError("invalid prediction-provider checkpoint")
        self.count = state["count"]
        self.held_payloads = list(state["held_payloads"])


class _SubstitutingStateStager(_MeasuredStateStager):
    """Adversary returning a valid-looking but different radio observation."""

    def stage(self, **kwargs):
        staged = super().stage(**kwargs)
        radio = staged.radio_state
        backlog = radio.pre_enqueue_backlog_bytes
        substituted_radio = replace(
            radio,
            pre_enqueue_backlog_bytes=replace(
                backlog,
                value=(backlog.value or 0) + 1,
            ),
        )
        return replace(staged, radio_state=substituted_radio)


class PersistentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _fit_fixture.Run4FitSceneProviderTests.setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        _fit_fixture.Run4FitSceneProviderTests.tearDownClass()

    @classmethod
    def _fit_provider(cls):
        fixture = _fit_fixture.Run4FitSceneProviderTests
        curves = _fit_fixture._synthetic_train_curves(fixture.inventory)
        held_provider = held_payload.HeldPayloadProviderV1(
            curves,
            expected_inventory_sha256=(
                held_payload.payload_curve_inventory_sha256(curves)
            ),
        )
        adapter = quality_adapter.Run4QualityPayloadAdapterV1(
            surface=fixture.surface,
            held_provider=held_provider,
            expected_surface_binding_sha256=fixture.surface_binding_sha,
            expected_held_provider_binding_sha256=(
                held_provider.binding_sha256
            ),
        )
        return fit_scene_provider.Run4FitSceneProviderV1(
            envelope=fixture.envelope,
            adapter=adapter,
            expected_envelope_sha256=fixture.envelope.canonical_sha256,
            scene_rng_seed=1701,
            held_rng_seed=2903,
            scene_rng_stream_id="run4-test-fit-scenes",
            held_rng_stream_id="run4-test-held-scenes",
        )

    @staticmethod
    def _scaling() -> contract.EmpiricalScalingV2:
        return contract.EmpiricalScalingV2(
            scaling_id="run4-persistent-test-scaling",
            scaling_version=1,
            evidence_sha256=_d("scaling"),
            camera_si_center=0.0,
            camera_si_scale=1_000.0,
            backlog_log1p_scale=math.log1p(10_000_000),
        )

    @staticmethod
    def _freshness() -> contract.FreshnessPolicyV2:
        return contract.FreshnessPolicyV2(
            policy_id="run4-persistent-test-freshness",
            policy_version=1,
            evidence_sha256=_d("freshness"),
            camera_si_max_age_ns=500_000_000,
            radar_p40_max_age_ns=500_000_000,
            prior_ul_mcs_max_age_ns=500_000_000,
            pre_action_rlc_backlog_max_age_ns=500_000_000,
        )

    @staticmethod
    def _support() -> sequential_kernel.KernelSupportV1:
        interval = sequential_kernel.NumericSupportV1(0, 200_000_000)
        return sequential_kernel.KernelSupportV1(
            support_id="run4-persistent-test-support",
            support_version=1,
            evidence_sha256=_d("kernel-support"),
            per_tensor_payload_bytes=sequential_kernel.NumericSupportV1(
                1, 10_000_000
            ),
            pre_enqueue_backlog_bytes=sequential_kernel.NumericSupportV1(
                0, 10_000_000
            ),
            observed_prior_ul_mcs=(7, 9, 12),
            latency=sequential_kernel.LatencySupportV1(
                ue_action_path_ns=interval,
                feature_uplink_ns=interval,
                edge_decompression_ns=interval,
                model_tail_ns=interval,
                quality_evaluation_ns=interval,
                feedback_downlink_ns=interval,
            ),
            maximum_hold_tensors=2,
            calibration_split=sequential_kernel.FitValidationSplitV1(
                fit_cell_ids=("fit-a",),
                validation_cell_ids=("validation-a",),
                assignment_evidence_sha256=_d("fit-validation-split"),
            ),
        )

    @classmethod
    def _kernel_prerequisites(
        cls,
        *,
        fit_provider,
        scaling: contract.EmpiricalScalingV2,
        freshness: contract.FreshnessPolicyV2,
    ) -> sequential_kernel.KernelVerifierPrerequisitesV1:
        support = cls._support()
        adapter = fit_provider._adapter.binding
        provenance = sequential_kernel.KernelProvenanceBindingV1(
            binding_id="run4-persistent-test-kernel",
            binding_version=1,
            raw_campaign_manifest_sha256=_d("raw-manifest"),
            raw_decisions_sha256=_d("raw-decisions"),
            corrected_analysis_generation=2,
            corrected_analysis_v2_sha256=_d("corrected-analysis"),
            corrected_decisions_v2_sha256=_d("corrected-decisions"),
            corrected_analysis_verdict=(
                sequential_kernel.ACCEPTED_ANALYSIS_VERDICT
            ),
            transport_fit_sha256=_d("transport-fit"),
            queue_transition_fit_sha256=_d("queue-fit"),
            ue_action_path_latency_evidence_sha256=_d("ue-path-latency"),
            tail_latency_evidence_sha256=_d("tail-latency"),
            quality_evaluation_latency_evidence_sha256=_d("quality-latency"),
            feedback_ack_latency_evidence_sha256=_d("feedback-latency"),
            quality_feedback_report_sha256=_d("quality-report"),
            quality_feedback_manifest_sha256=_d("quality-manifest"),
            quality_adapter_binding_sha256=adapter.canonical_sha256,
            held_provider_binding_sha256=adapter.held_provider_binding_sha256,
            actor_feature_schema_sha256=contract.FEATURE_SCHEMA_SHA256,
            empirical_scaling_sha256=scaling.canonical_sha256(),
            freshness_policy_sha256=freshness.canonical_sha256(),
            support_sha256=support.canonical_sha256,
        )
        return sequential_kernel.KernelVerifierPrerequisitesV1(
            provenance=provenance,
            support=support,
            fit_report_sha256=_d("fit-report"),
            validation_report_sha256=_d("validation-report"),
            fitted_model_sha256=_d("fitted-model"),
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
    def _gate() -> exploration.CoverageGateConfig:
        state_thresholds = tuple(
            exploration.StateFeatureThreshold(
                name=name,
                min_finite_count=2,
                min_unique_values=2,
                min_span=1e-9,
                min_per_mode_finite_count=2,
                min_per_mode_unique_values=2,
                min_per_mode_span=1e-9,
                saturation_lower=-1.0,
                saturation_upper=20_000_000.0,
                saturation_tolerance=0.0,
                max_boundary_saturation_fraction=1.0,
            )
            for name in exploration.REQUIRED_STATE_FEATURES
        )
        outcome_thresholds = tuple(
            exploration.OutcomeMetricThreshold(
                name=name,
                min_finite_count=2,
                min_unique_values=2,
                min_span=1e-9,
            )
            for name in exploration.REQUIRED_OUTCOME_METRICS
        )
        return exploration.CoverageGateConfig(
            preregistration_id="run4-persistent-test-gate",
            min_current_actions_per_mode=1,
            min_current_actions_per_q_bin=1,
            min_previous_actions_per_mode=1,
            min_previous_actions_per_q_bin=1,
            min_previous_present=1,
            min_previous_success=1,
            min_previous_failure=1,
            state_thresholds=state_thresholds,
            outcome_thresholds=outcome_thresholds,
        )

    @classmethod
    def factory(cls) -> src._TestOnlyPersistentRunnerV1:
        fit_provider = cls._fit_provider()
        scaling = cls._scaling()
        freshness = cls._freshness()
        state_prerequisites = production_state_provider.StateProviderPrerequisitesV1(
            fit_scene_provider_binding_sha256=(
                fit_provider.binding.canonical_sha256
            ),
            radio_queue_evidence_sha256=_d("radio-queue"),
            calibration_evidence_sha256=_d("calibration"),
            verifier_report_sha256=_d("state-verifier"),
            scaling=scaling,
            freshness=freshness,
            verification_status=(
                production_state_provider.TEST_ONLY_VERIFICATION_STATUS
            ),
        )
        state_authorization = (
            production_state_provider.authorize_test_only_state_provider(
                state_prerequisites
            )
        )
        state_provider = production_state_provider.Run4ProductionStateProviderV1(
            prerequisites=state_prerequisites,
            authorization=state_authorization,
        )
        kernel_prerequisites = cls._kernel_prerequisites(
            fit_provider=fit_provider, scaling=scaling, freshness=freshness
        )
        initial_radio = sequential_kernel.RadioQueueStateV1(
            session_uuid=SESSION,
            ue_id=UE_ID,
            decision_seq=0,
            prior_ul_mcs=sequential_kernel.IntegerObservationV1(
                value=9,
                missing_reason=None,
                source_decision_seq=0,
                provenance_sha256=_d("initial-mcs"),
            ),
            pre_enqueue_backlog_bytes=sequential_kernel.IntegerObservationV1(
                value=0,
                missing_reason=None,
                source_decision_seq=0,
                provenance_sha256=_d("initial-backlog"),
            ),
        )
        kernel = sequential_kernel.Run4SequentialRadioQueueKernelV1(
            prerequisites=kernel_prerequisites,
            authorization=sequential_kernel._issue_test_only_authorization(
                kernel_prerequisites
            ),
            calibration_partition=sequential_kernel.KernelCalibrationPartition.FIT,
            initial_state=initial_radio,
        )
        state_stager = _MeasuredStateStager()
        prediction_provider = _HeldAwarePredictionProvider(kernel_prerequisites)
        warmup = exploration.StratifiedWarmupSchedule(
            exploration.registered_modeled_support_config(
                q_bin_count=1, samples_per_q_bin=1, master_seed=73
            )
        )
        gate = cls._gate()
        replay_binding = replay.ReplayBindingV1._for_test_only(
            gamma=0.99,
            freshness_policy_sha256=freshness.canonical_sha256(),
            empirical_scaling_sha256=scaling.canonical_sha256(),
            calibration_evidence_sha256=state_prerequisites.calibration_evidence_sha256,
            queue_kernel_evidence_sha256=kernel_prerequisites.canonical_sha256,
        )
        buffer = replay._TestOnlyReplayBufferV1(64, replay_binding)
        bundle = models.build_run4_models(actor_seed=101, critic_seed=202)
        target_generator = torch.Generator(device="cpu").manual_seed(301)
        actor_generator = torch.Generator(device="cpu").manual_seed(302)
        sac_trainer = trainer._TestOnlyRun4HybridSacTrainerV1(
            actor=bundle.actor,
            critics=bundle.critics,
            config=trainer.TrainerConfigV1(
                alpha_d=0.1, alpha_c=0.1, nominal_batch_size=2
            ),
            expected_binding=replay_binding,
            target_generator=target_generator,
            actor_generator=actor_generator,
        )
        prerequisites = src.CompositeRunnerPrerequisitesV1(
            fit_scene_provider_binding_sha256=(
                fit_provider.binding.canonical_sha256
            ),
            state_provider_binding_sha256=state_provider.binding.canonical_sha256,
            kernel_prerequisites_sha256=kernel_prerequisites.canonical_sha256,
            kernel_support_sha256=kernel_prerequisites.support.canonical_sha256,
            state_stager_binding_sha256=state_stager.binding_sha256,
            prediction_provider_binding_sha256=prediction_provider.binding_sha256,
            replay_binding_sha256=canonical_sha256(
                replay_binding.to_canonical_dict()
            ),
            replay_capacity=buffer.capacity,
            model_binding_sha256=bundle.binding_sha256,
            trainer_config_sha256=src._trainer_config_sha256(
                sac_trainer.config
            ),
            warmup_schedule_id=warmup.config.schedule_id,
            exploration_gate_config_sha256=gate.config_sha256,
            verifier_manifest_sha256=_d("composite-verifier"),
            near_capacity_kernel_validated=False,
            scaling_and_freshness_validated=False,
            prior_outcome_chain_validated=False,
        )
        return src._TestOnlyPersistentRunnerV1(
            prerequisites=prerequisites,
            authorization=src._authorize_test_only(prerequisites),
            fit_provider=fit_provider,
            state_provider=state_provider,
            state_stager=state_stager,
            kernel=kernel,
            prediction_provider=prediction_provider,
            warmup_schedule=warmup,
            coverage_gate=gate,
            replay_buffer=buffer,
            model_bundle=bundle,
            sac_trainer=sac_trainer,
            decision_q_generator=torch.Generator(device="cpu").manual_seed(401),
            decision_mode_generator=torch.Generator(device="cpu").manual_seed(402),
            replay_generator=torch.Generator(device="cpu").manual_seed(403),
        )

    @staticmethod
    def _parameters(bundle: models.Run4ModelBundleV1) -> tuple[torch.Tensor, ...]:
        return tuple(
            parameter.detach().clone()
            for parameter in (
                *bundle.actor.parameters(),
                *bundle.critics.parameters(),
            )
        )

    def test_persistent_chain_uses_previous_outcome_and_held_payload(self) -> None:
        runner = self.factory()
        runner.start(session_uuid=SESSION, ue_id=UE_ID)
        genesis = runner.current_features
        before = self._parameters(runner.model_bundle)

        first = runner.step()
        after_success = runner.current_features
        self.assertEqual(first.decision.identity.decision_seq, 0)
        self.assertEqual(first.decision.hold.duration, 2)
        self.assertEqual(
            tuple(item.reward_requested for item in first.decision.hold.tensors),
            (True, False),
        )
        self.assertEqual(first.transition.reward_resolution.reward > -1.0, True)
        self.assertEqual(first.successor_staged.identity.decision_seq, 1)
        self.assertEqual(after_success[19], 1.0)  # previous-present
        self.assertEqual(after_success[20], 1.0)  # previous-success
        mode_offset = 4 + first.decision.action.mode_id
        self.assertEqual(after_success[mode_offset], 1.0)
        self.assertEqual(
            after_success[16], first.decision.action.q_e4 / 9800.0
        )
        self.assertEqual(
            after_success[17], first.transition.reward_resolution.q_perc
        )
        self.assertEqual(
            after_success[18],
            first.transition.reward_resolution.latency_ms / 170.0,
        )
        self.assertNotEqual(genesis, after_success)
        held = first.decision.hold.held_offered_payload_bytes
        self.assertEqual(
            first.prediction.next_state.pre_enqueue_backlog_bytes.value,
            min(9_000_000, 100 + int(held // 10)),
        )

        second = runner.step()
        after_timeout = runner.current_features
        self.assertEqual(second.decision.identity.decision_seq, 1)
        self.assertEqual(
            second.transition.reward_resolution.terminal,
            contract.RewardTerminal.TIMEOUT,
        )
        self.assertEqual(second.transition.reward_resolution.reward, -1.0)
        self.assertEqual(after_timeout[19], 1.0)
        self.assertEqual(after_timeout[20], 0.0)
        self.assertEqual(after_timeout[17:19], (0.0, 0.0))
        self.assertEqual(runner.environment.current_state.state.state.identity.decision_seq, 2)
        self.assertEqual(len(runner.replay_buffer), 2)

        with self.assertRaises(src.RunnerTrainingUnavailable):
            runner.train_once(2)
        self.assertEqual(runner.trainer.update_count, 0)
        after = self._parameters(runner.model_bundle)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, after)))

    def test_checkpoint_resume_is_exact_and_deterministic(self) -> None:
        uninterrupted = self.factory()
        uninterrupted.start(session_uuid=SESSION, ue_id=UE_ID)
        uninterrupted.step()
        checkpoint = uninterrupted.checkpoint()
        restored = src._TestOnlyPersistentRunnerV1.restore(
            checkpoint, fresh_factory=self.factory
        )
        self.assertEqual(restored.checkpoint().checkpoint_sha256, checkpoint.checkpoint_sha256)

        expected = uninterrupted.step()
        observed = restored.step()
        self.assertEqual(expected.canonical_sha256, observed.canonical_sha256)
        self.assertEqual(uninterrupted.current_features, restored.current_features)
        self.assertEqual(
            uninterrupted.checkpoint().checkpoint_sha256,
            restored.checkpoint().checkpoint_sha256,
        )

    def test_mismatched_checkpoint_binding_fails_without_mutating_source(self) -> None:
        runner = self.factory()
        runner.start(session_uuid=SESSION, ue_id=UE_ID)
        runner.step()
        checkpoint = runner.checkpoint()
        before = runner.checkpoint().checkpoint_sha256
        mismatched = replace(
            checkpoint,
            runner_binding_sha256=_d("different-runner-binding"),
            checkpoint_sha256="",
        )
        with self.assertRaisesRegex(src.RunnerCheckpointError, "binding mismatch"):
            src._TestOnlyPersistentRunnerV1.restore(
                mismatched, fresh_factory=self.factory
            )
        self.assertEqual(runner.checkpoint().checkpoint_sha256, before)

    def test_production_authorization_remains_fail_closed(self) -> None:
        runner = self.factory()
        with self.assertRaises(src.RunnerAuthorizationError):
            src.verify_composite_prerequisites(runner.prerequisites)

    def test_import_has_no_io_rng_process_socket_or_cuda_side_effect(self) -> None:
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
            importlib.reload(src)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_genesis_substitution_faults_runner_and_cannot_be_retried(self) -> None:
        runner = self.factory()
        runner.state_stager = _SubstitutingStateStager()
        with self.assertRaisesRegex(src.RunnerStateError, "substituted radio"):
            runner.start(session_uuid=SESSION, ue_id=UE_ID)
        self.assertEqual(runner.fit_provider.selected_count, 1)
        with self.assertRaisesRegex(src.RunnerStateError, "already started or faulted"):
            runner.start(session_uuid=SESSION, ue_id=UE_ID)

    def test_actor_mechanics_follow_completed_warmup_but_gradients_stay_blocked(self) -> None:
        runner = self.factory()
        runner.start(session_uuid=SESSION, ue_id=UE_ID)
        for _ in range(len(runner.warmup_schedule)):
            runner.step()
        self.assertEqual(runner.decision_count, len(runner.warmup_schedule))
        proposal = runner.select_action()
        proposal.require_reconciled()
        with self.assertRaises(src.RunnerTrainingUnavailable):
            runner.train_once(2)
        self.assertEqual(runner.trainer.update_count, 0)


if __name__ == "__main__":
    unittest.main()
