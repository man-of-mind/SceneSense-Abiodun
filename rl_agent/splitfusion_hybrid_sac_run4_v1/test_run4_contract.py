"""Adversarial tests for the minimal Run-4 contract."""

from __future__ import annotations

import builtins
import importlib
import math
import random
import socket
import subprocess
import unittest
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as src
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as actions
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


SESSION = "11111111-1111-4111-8111-111111111111"
UE_ID = "ue-1"
CLOCK = "RUN4_VIRTUAL_MONOTONIC"
EVIDENCE_A = "a" * 64
EVIDENCE_B = "b" * 64


class Run4ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Explicit test setup is allowed to load the frozen catalog.  Importing
        # run4_contract itself never calls this filesystem-backed function.
        cls.action_contract = actions.load_contract()

    def action(self, mode_id: int = 3, q_e4: int = 5000) -> ExecutedActionIdentity:
        executable = self.action_contract.resolve(
            mode_id, q_e4 / float(actions.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.action_contract
        )

    def decision(self, sequence: int = 0) -> src.DecisionIdentityV1:
        return src.DecisionIdentityV1(SESSION, UE_ID, sequence)

    def metadata(
        self,
        kind: src.MeasurementKind,
        *,
        sample_seq: int = 10,
        source_ns: int = 1_000_000_000,
        available_ns: int = 1_010_000_000,
        valid: bool = True,
        observer: src.Observer | None = None,
        direction: src.LinkDirection | None = None,
    ) -> src.MeasurementMetadataV1:
        defaults = {
            src.MeasurementKind.CAMERA_SI: (
                src.Observer.SCENE_PIPELINE,
                src.LinkDirection.NOT_APPLICABLE,
            ),
            src.MeasurementKind.RADAR_P40: (
                src.Observer.SCENE_PIPELINE,
                src.LinkDirection.NOT_APPLICABLE,
            ),
            src.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX: (
                src.Observer.UE,
                src.LinkDirection.UPLINK,
            ),
            src.MeasurementKind.UE_DL_SNR_DB: (
                src.Observer.UE,
                src.LinkDirection.DOWNLINK,
            ),
            src.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES: (
                src.Observer.UE,
                src.LinkDirection.UPLINK,
            ),
            src.MeasurementKind.GNB_UL_PUSCH_SNR_DB: (
                src.Observer.GNB,
                src.LinkDirection.UPLINK,
            ),
        }
        default_observer, default_direction = defaults[kind]
        return src.MeasurementMetadataV1(
            identity=src.SampleIdentityV1(SESSION, UE_ID, sample_seq),
            kind=kind,
            observer=default_observer if observer is None else observer,
            link_direction=(
                default_direction if direction is None else direction
            ),
            source=f"unit-test:{kind.value}",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=CLOCK,
            valid=valid,
        )

    def observation(
        self,
        kind: src.MeasurementKind,
        value: float | int | None,
        **metadata_overrides: object,
    ) -> src.ScalarObservationV1:
        valid = bool(metadata_overrides.pop("valid", value is not None))
        return src.ScalarObservationV1(
            value=value,
            metadata=self.metadata(kind, valid=valid, **metadata_overrides),
            missing_reason=None if valid else "not reported",
        )

    def prior_grant(
        self,
        value: int | float | None = 12,
        *,
        kind: src.MeasurementKind | None = None,
        mcs_table: int = src.UL_MCS_TABLE_ID,
        harq_round: int | None = 0,
        new_data_indicator: int | None = 1,
        grant_identity: str | None = "ue-dci-grant:10",
        scheduler_policy_id: str = src.UL_MCS_POLICY_ID,
        selection_rule_id: str = src.UL_MCS_SELECTION_RULE_ID,
        **metadata_overrides: object,
    ) -> src.PriorUlGrantObservationV1:
        if kind is None:
            kind = src.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX
        valid = bool(metadata_overrides.get("valid", value is not None))
        return src.PriorUlGrantObservationV1(
            observation=self.observation(
                kind,
                value,
                **metadata_overrides,
            ),
            mcs_table=mcs_table,
            harq_round=harq_round if valid else None,
            new_data_indicator=(new_data_indicator if valid else None),
            grant_identity=grant_identity if valid else None,
            scheduler_policy_id=scheduler_policy_id,
            selection_rule_id=selection_rule_id,
        )

    def state(
        self,
        *,
        sequence: int = 0,
        previous: src.PreviousOutcomeV1 | None = None,
        source_ns: int = 1_000_000_000,
        available_ns: int = 1_010_000_000,
        sample_seq: int = 10,
        camera: src.ScalarObservationV1 | None = None,
        radar: src.ScalarObservationV1 | None = None,
        mcs: src.PriorUlGrantObservationV1 | None = None,
        backlog: src.ScalarObservationV1 | None = None,
    ) -> src.PolicyStateV2:
        common = dict(
            source_ns=source_ns,
            available_ns=available_ns,
            sample_seq=sample_seq,
        )
        return src.PolicyStateV2(
            identity=self.decision(sequence),
            camera_si=(
                camera
                if camera is not None
                else self.observation(src.MeasurementKind.CAMERA_SI, 20.0, **common)
            ),
            radar_p40=(
                radar
                if radar is not None
                else self.observation(src.MeasurementKind.RADAR_P40, 0.4, **common)
            ),
            prior_ul_mcs=(
                mcs
                if mcs is not None
                else self.prior_grant(12, **common)
            ),
            pre_action_rlc_backlog=(
                backlog
                if backlog is not None
                else self.observation(
                    src.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                    1023,
                    **common,
                )
            ),
            previous=previous,
        )

    def boundary(
        self,
        *,
        sequence: int = 0,
        commit_ns: int = 1_050_000_000,
        action_ns: int = 1_060_000_000,
    ) -> src.DecisionBoundaryV1:
        return src.DecisionBoundaryV1(
            identity=self.decision(sequence),
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=action_ns,
            clock_domain=CLOCK,
        )

    def freshness(self, **overrides: int) -> src.FreshnessPolicyV2:
        values = dict(
            camera_si_max_age_ns=100_000_000,
            radar_p40_max_age_ns=100_000_000,
            prior_ul_mcs_max_age_ns=100_000_000,
            pre_action_rlc_backlog_max_age_ns=100_000_000,
        )
        values.update(overrides)
        return src.FreshnessPolicyV2(
            policy_id="unit-test-freshness",
            policy_version=1,
            evidence_sha256=EVIDENCE_A,
            **values,
        )

    def scaling(self) -> src.EmpiricalScalingV2:
        return src.EmpiricalScalingV2(
            scaling_id="unit-test-only-not-production",
            scaling_version=1,
            evidence_sha256=EVIDENCE_B,
            camera_si_center=10.0,
            camera_si_scale=5.0,
            backlog_log1p_scale=math.log1p(1023),
        )

    def guarded(
        self,
        state: src.PolicyStateV2 | None = None,
        boundary: src.DecisionBoundaryV1 | None = None,
    ) -> src.GuardedPolicyStateV2:
        return src.guard_state_for_action(
            self.state() if state is None else state,
            self.boundary() if boundary is None else boundary,
            self.freshness(),
        )

    def event(
        self,
        *,
        kind: src.RewardEventKind | None = None,
        latency_ns: int = 100_000_000,
        sequence: int = 0,
        action: ExecutedActionIdentity | None = None,
        q_perc: float | None = 0.8,
    ) -> src.RewardEventV1:
        if kind is None:
            kind = src.RewardEventKind.DELIVERED_SUCCESS
        if kind is not src.RewardEventKind.DELIVERED_SUCCESS:
            q_perc = None
        opened = 1_060_000_000
        return src.RewardEventV1(
            identity=self.decision(sequence),
            action=self.action() if action is None else action,
            kind=kind,
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=opened + latency_ns,
            clock_domain=CLOCK,
            source="unit-test-controller",
            q_perc=q_perc,
        )

    def success(self, **kwargs: object) -> src.RewardResolutionV1:
        return src.resolve_reward(self.event(**kwargs))

    def successor(
        self,
        resolution: src.RewardResolutionV1,
        *,
        sample_seq: int = 7,
    ) -> tuple[src.GuardedPolicyStateV2, src.PolicyFeatureVectorV2]:
        previous = src.PreviousOutcomeV1.from_resolution(resolution)
        next_state = self.state(
            sequence=1,
            previous=previous,
            source_ns=1_170_000_000,
            available_ns=1_180_000_000,
            sample_seq=sample_seq,
        )
        next_boundary = self.boundary(
            sequence=1,
            commit_ns=1_200_000_000,
            action_ns=1_210_000_000,
        )
        guarded = self.guarded(next_state, next_boundary)
        return guarded, src.build_policy_features(guarded, self.scaling())

    # ------------------------------------------------------------------
    # Feature contract and leakage
    # ------------------------------------------------------------------

    def test_exact_feature_order_count_and_genesis_encoding(self) -> None:
        expected = (
            "camera_si_scaled",
            "radar_p40",
            "prior_ul_mcs_normalized",
            "pre_action_rlc_backlog_log1p_scaled",
            *(f"prev_joint_mode_{i}_one_hot" for i in range(12)),
            "prev_q_normalized",
            "prev_quality_qperc",
            "prev_latency_normalized",
            "prev_present",
            "prev_success",
        )
        self.assertEqual(src.POLICY_FEATURE_ORDER, expected)
        self.assertEqual(src.POLICY_FEATURE_COUNT, 21)

        vector = src.build_policy_features(self.guarded(), self.scaling())
        self.assertEqual(vector.feature_names, expected)
        self.assertEqual(len(vector.as_tuple()), 21)
        self.assertEqual(vector.as_tuple()[:4], (2.0, 0.4, 12.0 / 28.0, 1.0))
        self.assertEqual(vector.as_tuple()[4:], (0.0,) * 17)

    def test_forbidden_actor_leakage_and_redundant_reward_are_absent(self) -> None:
        src.assert_policy_feature_schema()
        joined = " ".join(src.POLICY_FEATURE_ORDER).lower()
        for forbidden in src.FORBIDDEN_POLICY_FEATURE_TERMS:
            self.assertNotIn(forbidden, joined)
        for forbidden in (
            "tbs",
            "grant",
            "network_profile",
            "gnb",
            "age",
            "validity",
            "frame_id",
            "prev_reward",
        ):
            self.assertNotIn(forbidden, joined)

        resolution = self.success(latency_ns=100_000_000, q_perc=0.8)
        previous = src.PreviousOutcomeV1.from_resolution(resolution)
        self.assertAlmostEqual(previous.derived_reward, float(resolution.reward))
        self.assertNotIn("prev_reward", src.POLICY_FEATURE_ORDER)

    def test_empirical_scaling_has_no_defaults_and_is_hash_bound(self) -> None:
        with self.assertRaises(TypeError):
            src.EmpiricalScalingV2()  # type: ignore[call-arg]
        scale = self.scaling()
        self.assertEqual(len(scale.canonical_sha256()), 64)
        changed = src.EmpiricalScalingV2(
            scaling_id=scale.scaling_id,
            scaling_version=scale.scaling_version,
            evidence_sha256="c" * 64,
            camera_si_center=scale.camera_si_center,
            camera_si_scale=scale.camera_si_scale,
            backlog_log1p_scale=scale.backlog_log1p_scale,
        )
        self.assertNotEqual(scale.canonical_sha256(), changed.canonical_sha256())

    # ------------------------------------------------------------------
    # Metadata, direction and external fallback
    # ------------------------------------------------------------------

    def test_timestamp_causality_fails_closed(self) -> None:
        with self.assertRaises(src.MetadataError):
            self.metadata(
                src.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
                source_ns=1_020,
                available_ns=1_019,
            )
        with self.assertRaises(src.MetadataError):
            self.boundary(commit_ns=100, action_ns=100)

        late_mcs = self.prior_grant(
            12,
            source_ns=1_049_000_000,
            available_ns=1_051_000_000,
        )
        with self.assertRaisesRegex(
            src.ExternalFallbackRequired, "not available"
        ):
            self.guarded(self.state(mcs=late_mcs), self.boundary())

    def test_missing_stale_and_zero_fill_rules(self) -> None:
        missing_mcs = self.prior_grant(
            None,
            valid=False,
        )
        with self.assertRaises(src.ExternalFallbackRequired):
            self.guarded(self.state(mcs=missing_mcs), self.boundary())

        missing_backlog = self.observation(
            src.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
            None,
            valid=False,
        )
        with self.assertRaises(src.ExternalFallbackRequired):
            self.guarded(self.state(backlog=missing_backlog), self.boundary())

        with self.assertRaisesRegex(src.MetadataError, "zero-fill"):
            src.ScalarObservationV1(
                value=0,
                metadata=self.metadata(
                    src.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
                    valid=False,
                ),
                missing_reason="missing",
            )

        # A real, valid empty RLC queue is distinct from missing and is admitted.
        zero_backlog = self.observation(
            src.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES, 0
        )
        vector = src.build_policy_features(
            self.guarded(self.state(backlog=zero_backlog), self.boundary()),
            self.scaling(),
        )
        self.assertEqual(
            vector.as_dict()["pre_action_rlc_backlog_log1p_scaled"], 0.0
        )

        with self.assertRaisesRegex(src.ExternalFallbackRequired, "stale"):
            src.guard_state_for_action(
                self.state(),
                self.boundary(),
                self.freshness(prior_ul_mcs_max_age_ns=50_000_000),
            )

    def test_prior_ul_mcs_cannot_be_replaced_by_physical_snr(self) -> None:
        for kind in (
            src.MeasurementKind.GNB_UL_PUSCH_SNR_DB,
            src.MeasurementKind.UE_DL_SNR_DB,
        ):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                src.MetadataError, "UE-observed uplink MCS evidence"
            ):
                self.prior_grant(18.0, kind=kind)

        prior_mcs = self.prior_grant(18)
        admitted = self.guarded(self.state(mcs=prior_mcs), self.boundary())
        self.assertTrue(admitted.is_guarded)

    def test_prior_ul_mcs_provenance_is_hash_bound_and_fail_closed(self) -> None:
        for overrides, message in (
            ({"mcs_table": 1}, "mcs_table"),
            ({"harq_round": 1}, "HARQ round 0"),
            ({"new_data_indicator": 2}, "NDI"),
            ({"grant_identity": ""}, "grant_identity"),
            ({"scheduler_policy_id": "foreign"}, "scheduler_policy_id"),
            ({"selection_rule_id": "foreign"}, "selection_rule_id"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                src.MetadataError, message
            ):
                self.prior_grant(12, **overrides)

        valid = self.prior_grant(12)
        same = src.PriorUlGrantObservationV1(
            observation=valid.observation,
            mcs_table=src.UL_MCS_TABLE_ID,
            harq_round=0,
            new_data_indicator=1,
            grant_identity="ue-dci-grant:10",
            scheduler_policy_id=src.UL_MCS_POLICY_ID,
            selection_rule_id=src.UL_MCS_SELECTION_RULE_ID,
        )
        self.assertEqual(valid.canonical_sha256(), same.canonical_sha256())
        self.assertEqual(
            valid.to_canonical_dict()["record_type"],
            "prior_ul_grant_observation_v1",
        )

    def test_prior_ul_mcs_wire_domain_and_normalization_are_exact(self) -> None:
        self.assertEqual(src.UL_MCS_TABLE_ID, 0)
        self.assertEqual(src.UL_MCS_INDEX_MIN, 0)
        self.assertEqual(src.UL_MCS_INDEX_MAX, 28)
        self.assertEqual(src.UL_MCS_POLICY_ID, "SCENESENSE_MCS_POLICY=sinr")

        for raw, expected in ((0, 0.0), (28, 1.0)):
            with self.subTest(raw=raw):
                observation = self.prior_grant(raw)
                vector = src.build_policy_features(
                    self.guarded(self.state(mcs=observation), self.boundary()),
                    self.scaling(),
                )
                self.assertEqual(
                    vector.as_dict()["prior_ul_mcs_normalized"], expected
                )

        for raw in (-1, 29, 12.0):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                src.MetadataError, "exact table-0 index"
            ):
                self.prior_grant(raw)

        with self.assertRaisesRegex(src.MetadataError, "finite real scalar"):
            self.observation(
                src.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
                True,
            )

    def test_backlog_must_be_available_before_action(self) -> None:
        post_action_backlog = self.observation(
            src.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
            100,
            source_ns=1_060_000_000,
            available_ns=1_060_000_000,
        )
        with self.assertRaisesRegex(
            src.ExternalFallbackRequired, "not available"
        ):
            self.guarded(
                self.state(backlog=post_action_backlog),
                self.boundary(),
            )

    def test_camera_and_radar_require_one_exact_scene_sample_identity(self) -> None:
        radar_other_sample = self.observation(
            src.MeasurementKind.RADAR_P40,
            0.4,
            sample_seq=11,
        )
        with self.assertRaisesRegex(
            src.ExternalFallbackRequired, "exact current scene SampleIdentity"
        ):
            self.guarded(self.state(radar=radar_other_sample), self.boundary())

        # UE radio telemetry is intentionally asynchronous and need not share
        # the scene sample sequence.
        async_mcs = self.prior_grant(12, sample_seq=99)
        self.assertTrue(
            self.guarded(self.state(mcs=async_mcs), self.boundary()).is_guarded
        )

    # ------------------------------------------------------------------
    # Previous result and reward
    # ------------------------------------------------------------------

    def test_success_and_failure_previous_encoding(self) -> None:
        success = self.success(latency_ns=85_000_000, q_perc=0.75)
        previous_success = src.PreviousOutcomeV1.from_resolution(success)
        guarded_success, vector_success = self.successor(success)
        named = vector_success.as_dict()
        self.assertEqual(named["prev_present"], 1.0)
        self.assertEqual(named["prev_success"], 1.0)
        self.assertEqual(named["prev_quality_qperc"], 0.75)
        self.assertEqual(named["prev_latency_normalized"], 0.5)
        self.assertEqual(
            named[f"prev_joint_mode_{previous_success.action.mode_id}_one_hot"],
            1.0,
        )
        self.assertEqual(sum(named[name] for name in src.POLICY_FEATURE_ORDER[4:16]), 1.0)
        self.assertIsNotNone(guarded_success.state.previous)

        failure = src.resolve_reward(
            self.event(kind=src.RewardEventKind.REGISTERED_DELIVERY_FAILURE)
        )
        previous_failure = src.PreviousOutcomeV1.from_resolution(failure)
        _, vector_failure = self.successor(failure)
        failed = vector_failure.as_dict()
        self.assertEqual(failed["prev_present"], 1.0)
        self.assertEqual(failed["prev_success"], 0.0)
        self.assertEqual(failed["prev_quality_qperc"], 0.0)
        self.assertEqual(failed["prev_latency_normalized"], 0.0)
        self.assertEqual(
            failed[f"prev_joint_mode_{previous_failure.action.mode_id}_one_hot"],
            1.0,
        )
        self.assertGreater(failed["prev_q_normalized"], 0.0)

        with self.assertRaises(src.MetadataError):
            src.PreviousOutcomeV1(
                identity=self.decision(),
                action=self.action(),
                terminal=src.RewardTerminal.REGISTERED_DELIVERY_FAILURE,
                q_perc=0.0,
                latency_ms=0.0,
                available_timestamp_ns=1_100_000_000,
                clock_domain=CLOCK,
                reward_resolution_sha256="c" * 64,
            )

    def test_reward_boundary_is_inclusive_at_exactly_170_ms(self) -> None:
        for latency_ns in (169_999_000, 170_000_000):
            with self.subTest(latency_ns=latency_ns):
                resolved = self.success(latency_ns=latency_ns, q_perc=0.8)
                self.assertEqual(resolved.terminal, src.RewardTerminal.SUCCESS)
                latency_ms = latency_ns / 1_000_000.0
                self.assertAlmostEqual(
                    float(resolved.reward),
                    0.8 - 0.25 * (latency_ms / 170.0),
                    places=14,
                )

        late = self.success(latency_ns=170_001_000, q_perc=0.8)
        self.assertEqual(late.terminal, src.RewardTerminal.TIMEOUT)
        self.assertEqual(late.reward, -1.0)
        self.assertIsNone(late.q_perc)
        self.assertIsNone(late.latency_ms)

    def test_registered_failures_score_minus_one_and_faults_are_excluded(self) -> None:
        for event_kind, terminal in (
            (
                src.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
                src.RewardTerminal.REGISTERED_DELIVERY_FAILURE,
            ),
            (
                src.RewardEventKind.REGISTERED_SERVICE_FAILURE,
                src.RewardTerminal.REGISTERED_SERVICE_FAILURE,
            ),
            (src.RewardEventKind.TIMEOUT, src.RewardTerminal.TIMEOUT),
        ):
            latency = 170_000_001 if event_kind is src.RewardEventKind.TIMEOUT else 10
            resolved = src.resolve_reward(
                self.event(kind=event_kind, latency_ns=latency)
            )
            self.assertEqual(resolved.terminal, terminal)
            self.assertTrue(resolved.learning_included)
            self.assertEqual(resolved.reward, -1.0)
            self.assertIsNone(resolved.q_perc)
            self.assertIsNone(resolved.latency_ms)

        for event_kind in (
            src.RewardEventKind.INFRASTRUCTURE_FAULT,
            src.RewardEventKind.EVALUATOR_FAULT,
        ):
            excluded = src.resolve_reward(self.event(kind=event_kind, latency_ns=10))
            self.assertFalse(excluded.learning_included)
            self.assertIsNone(excluded.reward)
            with self.assertRaisesRegex(src.MetadataError, "cannot create"):
                src.PreviousOutcomeV1.from_resolution(excluded)

    # ------------------------------------------------------------------
    # Semi-Markov hold/transition
    # ------------------------------------------------------------------

    def hold(
        self,
        action: ExecutedActionIdentity | None = None,
        tensors: tuple[src.HoldTensorV1, ...] | None = None,
    ) -> src.ActionHoldV1:
        if tensors is None:
            tensors = (
                self.hold_tensor(20, 1000, True),
                self.hold_tensor(21, 600.5, False),
            )
        return src.ActionHoldV1(
            identity=self.decision(),
            action=self.action() if action is None else action,
            tensors=tensors,
        )

    def hold_tensor(
        self,
        tensor_seq: int,
        payload: int | float,
        reward_requested: bool,
        evidence_class: src.PayloadEvidenceClass | None = None,
    ) -> src.HoldTensorV1:
        if evidence_class is None:
            evidence_class = (
                src.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
                if type(payload) is int
                else src.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION
            )
        return src.HoldTensorV1(
            tensor_seq=tensor_seq,
            offered_payload_bytes=payload,
            payload_evidence_class=evidence_class,
            payload_provenance_sha256=EVIDENCE_A,
            reward_requested=reward_requested,
        )

    def test_hold_requires_two_tensors_and_held_payload_has_no_reward(self) -> None:
        with self.assertRaisesRegex(src.TransitionError, "at least 2"):
            self.hold(tensors=(self.hold_tensor(20, 1000, True),))
        with self.assertRaisesRegex(src.TransitionError, "must not request"):
            self.hold(
                tensors=(
                    self.hold_tensor(20, 1000, True),
                    self.hold_tensor(21, 600, True),
                )
            )
        hold = self.hold()
        self.assertEqual(hold.duration, 2)
        self.assertEqual(hold.total_offered_payload_bytes, 1600.5)
        self.assertEqual(hold.held_offered_payload_bytes, 600.5)
        self.assertEqual(
            hold.tensors[1].offered_payload_bytes, 600.5
        )  # no silent rounding
        self.assertEqual(
            hold.tensors[1].payload_evidence_class,
            src.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION,
        )
        self.assertEqual(
            sum(tensor.reward_requested for tensor in hold.tensors), 1
        )
        self.assertEqual(src.TRANSMIT_CADENCE_HZ, 10)

        with self.assertRaisesRegex(src.TransitionError, "must be an exact int"):
            self.hold_tensor(
                22,
                600.5,
                False,
                src.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE,
            )

    def transition_inputs(self) -> dict[str, object]:
        action = self.action()
        state = self.guarded()
        features = src.build_policy_features(state, self.scaling())
        resolution = self.success(action=action)
        next_state, next_features = self.successor(resolution, sample_seq=7)
        return dict(
            state=state,
            state_features=features,
            action=action,
            hold=self.hold(action=action),
            reward_resolution=resolution,
            next_state=next_state,
            next_state_features=next_features,
            episode_boundary=src.EpisodeBoundary.CONTINUES,
            duration=2,
            cycle_end_timestamp_ns=1_210_000_000,
            elapsed_virtual_ns=150_000_000,
            gamma=0.99,
            discount=0.99**2,
        )

    def test_transition_is_feedback_cycle_with_exact_duration_and_discount(self) -> None:
        values = self.transition_inputs()
        transition = src.build_transition(**values)  # type: ignore[arg-type]
        self.assertTrue(transition.is_attested)
        self.assertEqual(
            transition.to_canonical_dict()["record_type"],
            "semi_markov_transition_v2",
        )
        self.assertEqual(transition.duration, 2)
        self.assertEqual(transition.elapsed_virtual_ns, 150_000_000)
        self.assertAlmostEqual(transition.discount, 0.99**2)
        self.assertTrue(transition.bootstrap_allowed)
        self.assertEqual(transition.bootstrap_discount, 0.99**2)
        self.assertEqual(transition.reward, values["reward_resolution"].reward)
        self.assertEqual(
            transition.next_state.state.previous.reward_resolution_sha256,
            transition.reward_resolution.canonical_sha256(),
        )
        self.assertEqual(
            transition.to_canonical_dict()["evidence_class"],
            src.TRAINING_EVIDENCE_CLASS,
        )
        self.assertEqual(
            transition.to_canonical_dict()["live_evidence_status"],
            src.LIVE_EVIDENCE_STATUS,
        )

        bad_discount = dict(values, discount=0.99)
        with self.assertRaisesRegex(src.TransitionError, "gamma \*\* duration"):
            src.build_transition(**bad_discount)  # type: ignore[arg-type]

        bad_duration = dict(values, duration=3, discount=0.99**3)
        with self.assertRaisesRegex(src.TransitionError, "exact transmitted"):
            src.build_transition(**bad_duration)  # type: ignore[arg-type]

        with self.assertRaises(src.TransitionError):
            src.build_transition(  # type: ignore[arg-type]
                **dict(values, elapsed_virtual_ns=0)
            )

        with self.assertRaisesRegex(src.TransitionError, "must equal cycle_end"):
            src.build_transition(  # type: ignore[arg-type]
                **dict(values, elapsed_virtual_ns=149_000_000)
            )

        with self.assertRaisesRegex(src.TransitionError, "cannot precede"):
            src.build_transition(  # type: ignore[arg-type]
                **dict(
                    values,
                    episode_boundary=src.EpisodeBoundary.TRUNCATED,
                    next_state=None,
                    next_state_features=None,
                    cycle_end_timestamp_ns=1_150_000_000,
                    elapsed_virtual_ns=90_000_000,
                )
            )

        quick_failure = src.resolve_reward(
            self.event(
                kind=src.RewardEventKind.REGISTERED_SERVICE_FAILURE,
                latency_ns=10_000_000,
                action=values["action"],  # type: ignore[arg-type]
            )
        )
        with self.assertRaisesRegex(src.TransitionError, "10-Hz hold cadence"):
            src.build_transition(  # type: ignore[arg-type]
                **dict(
                    values,
                    reward_resolution=quick_failure,
                    episode_boundary=src.EpisodeBoundary.TRUNCATED,
                    next_state=None,
                    next_state_features=None,
                    cycle_end_timestamp_ns=1_110_000_000,
                    elapsed_virtual_ns=50_000_000,
                )
            )

    def test_transition_does_not_claim_scene_frame_contiguity(self) -> None:
        # Current samples use sample_seq=10; successor samples deliberately use
        # 7.  Exact successor status comes from decision identity/outcome, not
        # a fabricated adjacent frame-id rule.
        values = self.transition_inputs()
        current_seq = values["state"].state.camera_si.metadata.identity.sample_seq
        next_seq = values[
            "next_state"
        ].state.camera_si.metadata.identity.sample_seq
        self.assertEqual((current_seq, next_seq), (10, 7))
        transition = src.build_transition(**values)  # type: ignore[arg-type]
        self.assertTrue(transition.is_attested)

    def test_trace_boundaries_stop_bootstrap_but_timeout_can_continue(self) -> None:
        values = self.transition_inputs()
        truncated = src.build_transition(  # type: ignore[arg-type]
            **dict(
                values,
                episode_boundary=src.EpisodeBoundary.TRUNCATED,
                next_state=None,
                next_state_features=None,
                cycle_end_timestamp_ns=1_160_000_000,
                elapsed_virtual_ns=100_000_000,
            )
        )
        self.assertTrue(truncated.truncated)
        self.assertFalse(truncated.terminated)
        self.assertFalse(truncated.bootstrap_allowed)
        self.assertEqual(truncated.bootstrap_discount, 0.0)
        self.assertIsNone(truncated.next_state)

        terminated = src.build_transition(  # type: ignore[arg-type]
            **dict(
                values,
                episode_boundary=src.EpisodeBoundary.TERMINATED,
                next_state=None,
                next_state_features=None,
                cycle_end_timestamp_ns=1_160_000_000,
                elapsed_virtual_ns=100_000_000,
            )
        )
        self.assertTrue(terminated.terminated)
        self.assertFalse(terminated.truncated)
        self.assertEqual(terminated.bootstrap_discount, 0.0)

        action = self.action()
        timeout = src.resolve_reward(
            self.event(
                kind=src.RewardEventKind.TIMEOUT,
                latency_ns=170_000_001,
                action=action,
            )
        )
        previous = src.PreviousOutcomeV1.from_resolution(timeout)
        next_raw = self.state(
            sequence=1,
            previous=previous,
            source_ns=1_220_000_000,
            available_ns=1_230_000_001,
            sample_seq=37,
        )
        next_boundary = self.boundary(
            sequence=1,
            commit_ns=1_240_000_000,
            action_ns=1_250_000_000,
        )
        next_guarded = self.guarded(next_raw, next_boundary)
        next_features = src.build_policy_features(next_guarded, self.scaling())
        current = self.guarded()
        continued_timeout = src.build_transition(
            state=current,
            state_features=src.build_policy_features(current, self.scaling()),
            action=action,
            hold=self.hold(action=action),
            reward_resolution=timeout,
            next_state=next_guarded,
            next_state_features=next_features,
            episode_boundary=src.EpisodeBoundary.CONTINUES,
            duration=2,
            cycle_end_timestamp_ns=1_250_000_000,
            elapsed_virtual_ns=190_000_000,
            gamma=0.99,
            discount=0.99**2,
        )
        self.assertEqual(
            continued_timeout.reward_resolution.terminal,
            src.RewardTerminal.TIMEOUT,
        )
        self.assertTrue(continued_timeout.bootstrap_allowed)
        self.assertFalse(continued_timeout.terminated)
        self.assertFalse(continued_timeout.truncated)

    def test_excluded_fault_cannot_create_learning_transition(self) -> None:
        values = self.transition_inputs()
        excluded = src.resolve_reward(
            self.event(
                kind=src.RewardEventKind.INFRASTRUCTURE_FAULT,
                action=values["action"],  # type: ignore[arg-type]
                latency_ns=10,
            )
        )
        with self.assertRaisesRegex(src.TransitionError, "excluded"):
            src.build_transition(  # type: ignore[arg-type]
                **dict(values, reward_resolution=excluded)
            )

    # ------------------------------------------------------------------
    # Versioning and import purity
    # ------------------------------------------------------------------

    def test_all_semantic_surfaces_are_versioned_and_hashed(self) -> None:
        for schema_id, version, digest, expected_suffix in (
            (src.SCHEMA_ID, src.SCHEMA_VERSION, src.SCHEMA_SHA256, "_v2"),
            (
                src.FEATURE_SCHEMA_ID,
                src.FEATURE_SCHEMA_VERSION,
                src.FEATURE_SCHEMA_SHA256,
                "_v2",
            ),
            (
                src.REWARD_SCHEMA_ID,
                src.REWARD_SCHEMA_VERSION,
                src.REWARD_SCHEMA_SHA256,
                "_v1",
            ),
            (
                src.TRANSITION_SCHEMA_ID,
                src.TRANSITION_SCHEMA_VERSION,
                src.TRANSITION_SCHEMA_SHA256,
                "_v2",
            ),
        ):
            self.assertTrue(schema_id.endswith(expected_suffix))
            self.assertEqual(version, int(expected_suffix.removeprefix("_v")))
            self.assertEqual(len(digest), 64)
            int(digest, 16)

        changed_records = (
            self.freshness(),
            self.scaling(),
            self.state(),
            self.guarded(),
            src.build_policy_features(self.guarded(), self.scaling()),
        )
        for record in changed_records:
            self.assertTrue(
                record.to_canonical_dict()["record_type"].endswith("_v2")
            )

    def test_import_has_no_filesystem_rng_socket_or_process_side_effect(self) -> None:
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
