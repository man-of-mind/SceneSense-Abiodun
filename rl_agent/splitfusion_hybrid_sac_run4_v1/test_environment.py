"""Adversarial tests for the calibration-independent Run-4 environment."""

from __future__ import annotations

import builtins
import importlib
import random
import socket
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment as src
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as actions
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


SESSION = "11111111-1111-4111-8111-111111111111"
NEXT_SESSION = "22222222-2222-4222-8222-222222222222"
UE_ID = "ue-1"
CLOCK = "RUN4_ENV_TEST_MONOTONIC"
SYNTHETIC_EVIDENCE = "a" * 64


class FixtureProvider:
    """Contract-valid synthetic provider used only for mechanics tests."""

    def __init__(self) -> None:
        self.requests: list[src.DecisionStateRequestV1] = []
        self.fail_at_sequence: int | None = None
        self.wrong_previous = False
        self.freshness = contract.FreshnessPolicyV2(
            policy_id="synthetic-test-freshness",
            policy_version=1,
            evidence_sha256=SYNTHETIC_EVIDENCE,
            camera_si_max_age_ns=50_000_000,
            radar_p40_max_age_ns=50_000_000,
            prior_ul_mcs_max_age_ns=50_000_000,
            pre_action_rlc_backlog_max_age_ns=50_000_000,
        )
        self.scaling = contract.EmpiricalScalingV2(
            scaling_id="synthetic-test-scaling",
            scaling_version=1,
            evidence_sha256=SYNTHETIC_EVIDENCE,
            camera_si_center=0.0,
            camera_si_scale=1.0,
            backlog_log1p_scale=10.0,
        )

    @staticmethod
    def _metadata(
        identity: contract.DecisionIdentityV1,
        *,
        sample_seq: int,
        kind: contract.MeasurementKind,
        observer: contract.Observer,
        direction: contract.LinkDirection,
        source_ns: int,
        available_ns: int,
    ) -> contract.MeasurementMetadataV1:
        return contract.MeasurementMetadataV1(
            identity=contract.SampleIdentityV1(
                identity.session_uuid, identity.ue_id, sample_seq
            ),
            kind=kind,
            observer=observer,
            link_direction=direction,
            source="synthetic-environment-mechanics-test",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=CLOCK,
            valid=True,
        )

    def build_state(
        self, request: src.DecisionStateRequestV1
    ) -> src.DecisionStateBundleV1:
        self.requests.append(request)
        sequence = request.identity.decision_seq
        if sequence == self.fail_at_sequence:
            raise contract.ExternalFallbackRequired("fixture observation missing")
        open_ns = (
            2_000_000_000
            if request.required_action_open_timestamp_ns is None
            else request.required_action_open_timestamp_ns
        )
        commit_ns = open_ns - 10_000_000
        source_ns = open_ns - 30_000_000
        available_ns = open_ns - 20_000_000
        scene_sample = 100 + sequence

        camera_metadata = self._metadata(
            request.identity,
            sample_seq=scene_sample,
            kind=contract.MeasurementKind.CAMERA_SI,
            observer=contract.Observer.SCENE_PIPELINE,
            direction=contract.LinkDirection.NOT_APPLICABLE,
            source_ns=source_ns,
            available_ns=available_ns,
        )
        radar_metadata = self._metadata(
            request.identity,
            sample_seq=scene_sample,
            kind=contract.MeasurementKind.RADAR_P40,
            observer=contract.Observer.SCENE_PIPELINE,
            direction=contract.LinkDirection.NOT_APPLICABLE,
            source_ns=source_ns,
            available_ns=available_ns,
        )
        mcs_metadata = self._metadata(
            request.identity,
            sample_seq=200 + sequence,
            kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
            observer=contract.Observer.UE,
            direction=contract.LinkDirection.UPLINK,
            source_ns=source_ns,
            available_ns=available_ns,
        )
        backlog_metadata = self._metadata(
            request.identity,
            sample_seq=300 + sequence,
            kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
            observer=contract.Observer.UE,
            direction=contract.LinkDirection.UPLINK,
            source_ns=source_ns,
            available_ns=available_ns,
        )
        previous = None if self.wrong_previous else request.previous
        state = contract.PolicyStateV2(
            identity=request.identity,
            camera_si=contract.ScalarObservationV1(
                0.25 + sequence * 0.01, camera_metadata, None
            ),
            radar_p40=contract.ScalarObservationV1(
                0.4 + sequence * 0.01, radar_metadata, None
            ),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=contract.ScalarObservationV1(
                    12 + sequence, mcs_metadata, None
                ),
                mcs_table=contract.UL_MCS_TABLE_ID,
                harq_round=0,
                new_data_indicator=sequence % 2,
                grant_identity=f"synthetic-grant-{sequence}",
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
            ),
            pre_action_rlc_backlog=contract.ScalarObservationV1(
                1000 + sequence, backlog_metadata, None
            ),
            previous=previous,
        )
        boundary = contract.DecisionBoundaryV1(
            identity=request.identity,
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=open_ns,
            clock_domain=CLOCK,
        )
        guarded = contract.guard_state_for_action(
            state, boundary, self.freshness
        )
        features = contract.build_policy_features(guarded, self.scaling)
        return src.DecisionStateBundleV1(guarded, features)


class FixtureKernel:
    def __init__(
        self,
        *,
        kind: contract.RewardEventKind = contract.RewardEventKind.DELIVERED_SUCCESS,
        boundary: contract.EpisodeBoundary = contract.EpisodeBoundary.CONTINUES,
        resolution_offset_ns: int = 150_000_000,
        cycle_offset_ns: int = 200_000_000,
        q_perc: float | None = 0.7,
    ) -> None:
        self.kind = kind
        self.boundary = boundary
        self.resolution_offset_ns = resolution_offset_ns
        self.cycle_offset_ns = cycle_offset_ns
        self.q_perc = q_perc
        self.requests: list[src.KernelCycleRequestV1] = []
        self.last_result: src.KernelCycleResultV1 | None = None

    def execute_cycle(
        self, request: src.KernelCycleRequestV1
    ) -> src.KernelCycleResultV1:
        self.requests.append(request)
        identity = request.state.state.state.identity
        opened = request.state.state.boundary.action_open_timestamp_ns
        base_seq = identity.decision_seq * 10
        provenance = "b" * 64
        hold = contract.ActionHoldV1(
            identity=identity,
            action=request.action,
            tensors=(
                contract.HoldTensorV1(
                    tensor_seq=base_seq,
                    offered_payload_bytes=10_000,
                    payload_evidence_class=(
                        contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
                    ),
                    payload_provenance_sha256=provenance,
                    reward_requested=True,
                ),
                contract.HoldTensorV1(
                    tensor_seq=base_seq + 1,
                    offered_payload_bytes=10_001,
                    payload_evidence_class=(
                        contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
                    ),
                    payload_provenance_sha256=provenance,
                    reward_requested=False,
                ),
            ),
        )
        q_perc = (
            self.q_perc
            if self.kind is contract.RewardEventKind.DELIVERED_SUCCESS
            else None
        )
        event = contract.RewardEventV1(
            identity=identity,
            action=request.action,
            kind=self.kind,
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=opened + self.resolution_offset_ns,
            clock_domain=CLOCK,
            source="synthetic-environment-mechanics-test",
            q_perc=q_perc,
        )
        self.last_result = src.KernelCycleResultV1(
            hold=hold,
            reward_event=event,
            cycle_end_timestamp_ns=opened + self.cycle_offset_ns,
            episode_boundary=self.boundary,
        )
        return self.last_result


class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.action_contract = actions.load_contract()

    def action(self, mode_id: int = 3, q_e4: int = 5000) -> ExecutedActionIdentity:
        executable = self.action_contract.resolve(
            mode_id, q_e4 / float(actions.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.action_contract
        )

    def environment(
        self,
        provider: FixtureProvider | None = None,
        kernel: FixtureKernel | None = None,
    ) -> tuple[src.Run4SequentialEnvironmentV1, FixtureProvider, FixtureKernel]:
        provider = FixtureProvider() if provider is None else provider
        kernel = FixtureKernel() if kernel is None else kernel
        environment = src.Run4SequentialEnvironmentV1(
            state_provider=provider,
            kernel=kernel,
            gamma=0.99,
            evidence_class=(
                src.EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE
            ),
        )
        environment.reset(session_uuid=SESSION, ue_id=UE_ID)
        return environment, provider, kernel

    def test_success_carries_exact_previous_action_quality_and_latency(self) -> None:
        environment, provider, _ = self.environment()
        action = self.action(mode_id=7, q_e4=7000)
        result = environment.step(action)
        current = environment.current_state
        previous = current.state.state.previous
        self.assertIsNotNone(previous)
        assert previous is not None
        self.assertEqual(previous.action, action)
        self.assertEqual(previous.action.mode_id, 7)
        self.assertEqual(previous.action.q_e4, 7000)
        self.assertAlmostEqual(previous.q_perc, 0.7)
        self.assertAlmostEqual(previous.latency_ms, 150.0)
        self.assertEqual(previous.derived_reward, result.reward)
        self.assertEqual(current.state.state.identity.decision_seq, 1)
        self.assertEqual(provider.requests[-1].previous, previous)
        named = current.features.as_dict()
        self.assertEqual(named["prev_joint_mode_7_one_hot"], 1.0)
        self.assertEqual(named["prev_q_normalized"], 7000 / 9800.0)
        self.assertEqual(named["prev_quality_qperc"], 0.7)
        self.assertEqual(named["prev_latency_normalized"], 150.0 / 170.0)
        self.assertEqual(named["prev_present"], 1.0)
        self.assertEqual(named["prev_success"], 1.0)

        # Run-2/3's placeholder-state defect must not recur: after the one
        # genesis observation, all four *current* scene/radio inputs remain
        # genuine provider values while the exact prior outcome is appended.
        self.assertGreater(named["camera_si_scaled"], 0.0)
        self.assertGreater(named["radar_p40"], 0.0)
        self.assertGreater(named["prior_ul_mcs_normalized"], 0.0)
        self.assertGreater(
            named["pre_action_rlc_backlog_log1p_scaled"], 0.0
        )

    def test_session_ue_and_decision_sequence_persist_across_cycles(self) -> None:
        environment, provider, _ = self.environment()
        first = self.action(mode_id=3, q_e4=3000)
        second = self.action(mode_id=8, q_e4=6000)
        environment.step(first)
        environment.step(second)
        current = environment.current_state.state.state
        self.assertEqual(current.identity.session_uuid, SESSION)
        self.assertEqual(current.identity.ue_id, UE_ID)
        self.assertEqual(current.identity.decision_seq, 2)
        self.assertEqual(current.previous.action, second)
        self.assertEqual(
            [request.identity.decision_seq for request in provider.requests],
            [0, 1, 2],
        )

    def test_reset_cannot_abandon_an_active_decision_sequence(self) -> None:
        environment, provider, _ = self.environment()
        active_sha256 = environment.current_state.state.canonical_sha256()

        with self.assertRaisesRegex(src.EnvironmentStateError, "cannot reset an active"):
            environment.reset(session_uuid=NEXT_SESSION, ue_id=UE_ID)

        self.assertEqual(
            environment.current_state.state.canonical_sha256(), active_sha256
        )
        self.assertEqual(len(provider.requests), 1)

    def test_restart_after_boundary_requires_a_fresh_session_uuid(self) -> None:
        kernel = FixtureKernel(boundary=contract.EpisodeBoundary.TERMINATED)
        environment, provider, _ = self.environment(kernel=kernel)
        environment.step(self.action())
        self.assertTrue(environment.requires_reset)

        with self.assertRaisesRegex(src.EnvironmentStateError, "fresh session_uuid"):
            environment.reset(session_uuid=SESSION, ue_id=UE_ID)

        restarted = environment.reset(session_uuid=NEXT_SESSION, ue_id=UE_ID)
        self.assertEqual(restarted.state.state.identity.session_uuid, NEXT_SESSION)
        self.assertEqual(restarted.state.state.identity.decision_seq, 0)
        self.assertIsNone(restarted.state.state.previous)
        self.assertEqual(
            [request.identity.session_uuid for request in provider.requests],
            [SESSION, NEXT_SESSION],
        )

    def test_hold_reuses_exact_action_and_only_first_tensor_requests_reward(self) -> None:
        environment, _, kernel = self.environment()
        action = self.action(mode_id=5, q_e4=4321)
        result = environment.step(action)
        assert kernel.last_result is not None
        self.assertEqual(kernel.last_result.hold.action, action)
        self.assertEqual(kernel.requests[-1].action, action)
        self.assertEqual(result.action_sha256, action.canonical_sha256())
        self.assertEqual(result.duration, 2)
        self.assertEqual(result.reward_request_flags, (True, False))

    def test_timeout_is_punished_and_carried_as_failed_previous_outcome(self) -> None:
        kernel = FixtureKernel(
            kind=contract.RewardEventKind.TIMEOUT,
            resolution_offset_ns=171_000_000,
            q_perc=None,
        )
        environment, _, _ = self.environment(kernel=kernel)
        action = self.action(mode_id=2, q_e4=8000)
        result = environment.step(action)
        self.assertEqual(result.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertEqual(result.reward, -1.0)
        previous = environment.current_state.state.state.previous
        assert previous is not None
        self.assertEqual(previous.action, action)
        self.assertFalse(previous.success)
        self.assertIsNone(previous.q_perc)
        self.assertIsNone(previous.latency_ms)
        named = environment.current_state.features.as_dict()
        self.assertEqual(named["prev_joint_mode_2_one_hot"], 1.0)
        self.assertEqual(named["prev_q_normalized"], 8000 / 9800.0)
        self.assertEqual(named["prev_quality_qperc"], 0.0)
        self.assertEqual(named["prev_latency_normalized"], 0.0)
        self.assertEqual(named["prev_present"], 1.0)
        self.assertEqual(named["prev_success"], 0.0)

    def test_registered_service_failure_is_learning_included(self) -> None:
        kernel = FixtureKernel(
            kind=contract.RewardEventKind.REGISTERED_SERVICE_FAILURE,
            q_perc=None,
        )
        environment, _, _ = self.environment(kernel=kernel)
        result = environment.step(self.action())
        self.assertEqual(
            result.terminal, contract.RewardTerminal.REGISTERED_SERVICE_FAILURE
        )
        self.assertEqual(result.reward, -1.0)

    def test_infrastructure_fault_is_excluded_and_produces_no_transition(self) -> None:
        kernel = FixtureKernel(
            kind=contract.RewardEventKind.INFRASTRUCTURE_FAULT,
            q_perc=None,
        )
        environment, provider, _ = self.environment(kernel=kernel)
        result = environment.step(self.action())
        self.assertIs(type(result), src.ExcludedCycleV1)
        self.assertFalse(result.replay_export_allowed)
        with self.assertRaises(src.SyntheticEvidenceRejected):
            result.export_for_replay()
        self.assertTrue(environment.requires_reset)
        self.assertEqual(len(provider.requests), 1)
        with self.assertRaises(src.EnvironmentStateError):
            _ = environment.current_state

    def test_continuing_cycle_requires_real_successor(self) -> None:
        provider = FixtureProvider()
        provider.fail_at_sequence = 1
        environment, _, _ = self.environment(provider=provider)
        with self.assertRaises(src.SuccessorUnavailableError):
            environment.step(self.action())
        self.assertTrue(environment.requires_reset)

    def test_true_episode_boundary_does_not_request_successor(self) -> None:
        kernel = FixtureKernel(boundary=contract.EpisodeBoundary.TERMINATED)
        environment, provider, _ = self.environment(kernel=kernel)
        result = environment.step(self.action())
        self.assertIsNone(result.next_state_sha256)
        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(environment.requires_reset)
        with self.assertRaises(src.EnvironmentStateError):
            _ = environment.current_state

    def test_missing_genesis_observation_propagates_external_fallback(self) -> None:
        provider = FixtureProvider()
        provider.fail_at_sequence = 0
        environment = src.Run4SequentialEnvironmentV1(
            state_provider=provider,
            kernel=FixtureKernel(),
            gamma=0.99,
            evidence_class=(
                src.EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE
            ),
        )
        with self.assertRaises(contract.ExternalFallbackRequired):
            environment.reset(session_uuid=SESSION, ue_id=UE_ID)

    def test_provider_cannot_drop_exact_previous_outcome(self) -> None:
        provider = FixtureProvider()
        provider.wrong_previous = True
        environment, _, _ = self.environment(provider=provider)
        with self.assertRaises(src.SuccessorUnavailableError):
            environment.step(self.action())
        self.assertTrue(environment.requires_reset)

    def test_synthetic_result_contains_no_bare_transition_and_cannot_export(self) -> None:
        environment, _, _ = self.environment()
        result = environment.step(self.action())
        self.assertIs(type(result), src.SyntheticMechanicsCycleV1)
        self.assertFalse(hasattr(result, "transition"))
        self.assertFalse(hasattr(result, "_transition"))
        self.assertFalse(result.replay_export_allowed)
        with self.assertRaises(src.SyntheticEvidenceRejected):
            result.export_for_replay()

    def test_calibrated_mode_requires_binding_and_remains_fail_closed(self) -> None:
        with self.assertRaises(src.CalibrationUnavailableError):
            src.Run4SequentialEnvironmentV1(
                state_provider=FixtureProvider(),
                kernel=FixtureKernel(),
                gamma=0.99,
                evidence_class=src.EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL,
            )
        binding = src.CalibrationBindingV1(
            calibration_id="externally-supplied-candidate",
            calibration_version=1,
            evidence_sha256="1" * 64,
            verifier_report_sha256="2" * 64,
            kernel_binding_sha256="3" * 64,
            state_provider_binding_sha256="4" * 64,
        )
        self.assertIsNone(src.REGISTERED_CALIBRATION_BINDING_SHA256)
        with self.assertRaises(src.CalibrationUnavailableError):
            src.Run4SequentialEnvironmentV1(
                state_provider=FixtureProvider(),
                kernel=FixtureKernel(),
                gamma=0.99,
                evidence_class=src.EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL,
                calibration_binding=binding,
            )

    def test_kernel_action_mismatch_fails_and_requires_reset(self) -> None:
        environment, _, kernel = self.environment()
        original_execute = kernel.execute_cycle

        def mismatched(request: src.KernelCycleRequestV1) -> src.KernelCycleResultV1:
            result = original_execute(request)
            other = self.action(mode_id=4, q_e4=5000)
            return replace(
                result,
                hold=replace(result.hold, action=other),
                reward_event=replace(result.reward_event, action=other),
            )

        kernel.execute_cycle = mismatched  # type: ignore[method-assign]
        with self.assertRaises(src.KernelResultError):
            environment.step(self.action(mode_id=3, q_e4=5000))
        self.assertTrue(environment.requires_reset)

    def test_import_has_no_rng_filesystem_socket_or_process_side_effect(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
