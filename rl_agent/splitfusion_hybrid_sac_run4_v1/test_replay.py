"""Adversarial tests for the attested Run-4 replay boundary."""

from __future__ import annotations

import dataclasses
import math
import unittest
from dataclasses import replace

import torch

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as actions
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)

from . import replay
from . import run4_contract as contract


SESSION = "11111111-1111-4111-8111-111111111111"
UE_ID = "ue-1"
CLOCK = "RUN4_REPLAY_TEST_CLOCK"
FRESHNESS_EVIDENCE = "a" * 64
SCALING_EVIDENCE = "b" * 64
CALIBRATION_EVIDENCE = "c" * 64
QUEUE_KERNEL_EVIDENCE = "d" * 64


class Fixture:
    """Build valid records only through the public Run-4 contract."""

    def __init__(self) -> None:
        self.catalog = actions.load_contract()

    def action(self, mode_id: int = 3, q_e4: int = 5000) -> ExecutedActionIdentity:
        executable = self.catalog.resolve(
            mode_id, q_e4 / float(actions.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, self.catalog
        )

    def freshness(self, evidence: str = FRESHNESS_EVIDENCE) -> contract.FreshnessPolicyV2:
        return contract.FreshnessPolicyV2(
            policy_id="run4-replay-test-freshness",
            policy_version=1,
            evidence_sha256=evidence,
            camera_si_max_age_ns=500_000_000,
            radar_p40_max_age_ns=500_000_000,
            prior_ul_mcs_max_age_ns=500_000_000,
            pre_action_rlc_backlog_max_age_ns=500_000_000,
        )

    def scaling(
        self,
        evidence: str = SCALING_EVIDENCE,
        *,
        camera_center: float = 10.0,
        camera_scale: float = 5.0,
    ) -> contract.EmpiricalScalingV2:
        return contract.EmpiricalScalingV2(
            scaling_id="run4-replay-test-scaling",
            scaling_version=1,
            evidence_sha256=evidence,
            camera_si_center=camera_center,
            camera_si_scale=camera_scale,
            backlog_log1p_scale=math.log1p(1023),
        )

    @staticmethod
    def _base(sequence: int) -> int:
        return 1_000_000_000 + sequence * 20_000_000_000

    def metadata(
        self,
        *,
        session: str,
        kind: contract.MeasurementKind,
        sample_seq: int,
        source_ns: int,
        available_ns: int,
    ) -> contract.MeasurementMetadataV1:
        observer, direction = {
            contract.MeasurementKind.CAMERA_SI: (
                contract.Observer.SCENE_PIPELINE,
                contract.LinkDirection.NOT_APPLICABLE,
            ),
            contract.MeasurementKind.RADAR_P40: (
                contract.Observer.SCENE_PIPELINE,
                contract.LinkDirection.NOT_APPLICABLE,
            ),
            contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX: (
                contract.Observer.UE,
                contract.LinkDirection.UPLINK,
            ),
            contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES: (
                contract.Observer.UE,
                contract.LinkDirection.UPLINK,
            ),
        }[kind]
        return contract.MeasurementMetadataV1(
            identity=contract.SampleIdentityV1(session, UE_ID, sample_seq),
            kind=kind,
            observer=observer,
            link_direction=direction,
            source=f"run4-replay-test:{kind.value}",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=CLOCK,
            valid=True,
        )

    def observation(
        self,
        *,
        session: str,
        kind: contract.MeasurementKind,
        value: float | int,
        sample_seq: int,
        source_ns: int,
        available_ns: int,
    ) -> contract.ScalarObservationV1:
        return contract.ScalarObservationV1(
            value=value,
            metadata=self.metadata(
                session=session,
                kind=kind,
                sample_seq=sample_seq,
                source_ns=source_ns,
                available_ns=available_ns,
            ),
            missing_reason=None,
        )

    def raw_state(
        self,
        *,
        sequence: int,
        session: str,
        source_ns: int,
        available_ns: int,
        previous: contract.PreviousOutcomeV1 | None,
        camera: float = 20.0,
    ) -> contract.PolicyStateV2:
        sample_seq = sequence + 100
        common = dict(
            session=session,
            sample_seq=sample_seq,
            source_ns=source_ns,
            available_ns=available_ns,
        )
        mcs_observation = self.observation(
            kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
            value=12,
            **common,
        )
        return contract.PolicyStateV2(
            identity=contract.DecisionIdentityV1(session, UE_ID, sequence),
            camera_si=self.observation(
                kind=contract.MeasurementKind.CAMERA_SI,
                value=camera,
                **common,
            ),
            radar_p40=self.observation(
                kind=contract.MeasurementKind.RADAR_P40,
                value=0.4,
                **common,
            ),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=mcs_observation,
                mcs_table=contract.UL_MCS_TABLE_ID,
                harq_round=0,
                new_data_indicator=1,
                grant_identity=f"grant:{session}:{sequence}",
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
            ),
            pre_action_rlc_backlog=self.observation(
                kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                value=1023,
                **common,
            ),
            previous=previous,
        )

    def guarded_features(
        self,
        *,
        sequence: int,
        session: str,
        source_ns: int,
        available_ns: int,
        commit_ns: int,
        action_ns: int,
        previous: contract.PreviousOutcomeV1 | None,
        freshness: contract.FreshnessPolicyV2,
        scaling: contract.EmpiricalScalingV2,
        camera: float = 20.0,
    ) -> tuple[contract.GuardedPolicyStateV2, contract.PolicyFeatureVectorV2]:
        state = self.raw_state(
            sequence=sequence,
            session=session,
            source_ns=source_ns,
            available_ns=available_ns,
            previous=previous,
            camera=camera,
        )
        boundary = contract.DecisionBoundaryV1(
            identity=state.identity,
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=action_ns,
            clock_domain=CLOCK,
        )
        guarded = contract.guard_state_for_action(state, boundary, freshness)
        return guarded, contract.build_policy_features(guarded, scaling)

    def resolved_predecessor(
        self, *, sequence: int, session: str, current_base_ns: int
    ) -> contract.PreviousOutcomeV1 | None:
        """Return a real resolved predecessor for every non-genesis state."""

        if sequence == 0:
            return None
        previous_action = self.action(mode_id=2, q_e4=4000)
        opened = current_base_ns - 300_000_000
        event = contract.RewardEventV1(
            identity=contract.DecisionIdentityV1(session, UE_ID, sequence - 1),
            action=previous_action,
            kind=contract.RewardEventKind.DELIVERED_SUCCESS,
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=opened + 100_000_000,
            clock_domain=CLOCK,
            source="run4-replay-unit-test-predecessor",
            q_perc=0.7,
        )
        return contract.PreviousOutcomeV1.from_resolution(
            contract.resolve_reward(event)
        )

    def transition(
        self,
        *,
        sequence: int = 0,
        session: str = SESSION,
        mode_id: int = 3,
        q_e4: int = 5000,
        boundary: contract.EpisodeBoundary = contract.EpisodeBoundary.CONTINUES,
        duration: int = 2,
        gamma: float = 0.99,
        freshness: contract.FreshnessPolicyV2 | None = None,
        scaling: contract.EmpiricalScalingV2 | None = None,
        camera: float = 20.0,
        event_kind: contract.RewardEventKind = (
            contract.RewardEventKind.DELIVERED_SUCCESS
        ),
    ) -> contract.SemiMarkovTransitionV2:
        freshness = self.freshness() if freshness is None else freshness
        scaling = self.scaling() if scaling is None else scaling
        action = self.action(mode_id, q_e4)
        base = self._base(sequence)
        opened = base + 60_000_000
        predecessor = self.resolved_predecessor(
            sequence=sequence, session=session, current_base_ns=base
        )
        state, features = self.guarded_features(
            sequence=sequence,
            session=session,
            source_ns=base,
            available_ns=base + 10_000_000,
            commit_ns=base + 50_000_000,
            action_ns=opened,
            previous=predecessor,
            freshness=freshness,
            scaling=scaling,
            camera=camera,
        )
        resolution_latency_ns = (
            100_000_000
            if event_kind is not contract.RewardEventKind.TIMEOUT
            else contract.REWARD_DEADLINE_NS + 1
        )
        event = contract.RewardEventV1(
            identity=state.state.identity,
            action=action,
            kind=event_kind,
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=opened + resolution_latency_ns,
            clock_domain=CLOCK,
            source="run4-replay-unit-test",
            q_perc=(
                0.8
                if event_kind is contract.RewardEventKind.DELIVERED_SUCCESS
                else None
            ),
        )
        resolution = contract.resolve_reward(event)
        tensors = tuple(
            contract.HoldTensorV1(
                tensor_seq=sequence * 1000 + index,
                offered_payload_bytes=1000 + index,
                payload_evidence_class=(
                    contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE
                ),
                payload_provenance_sha256="e" * 64,
                reward_requested=index == 0,
            )
            for index in range(duration)
        )
        hold = contract.ActionHoldV1(
            identity=state.state.identity,
            action=action,
            tensors=tensors,
        )
        elapsed = max(
            150_000_000,
            (duration - 1) * 100_000_000 + 10_000_000,
            resolution_latency_ns + 20_000_000,
        )
        cycle_end = opened + elapsed
        if boundary is contract.EpisodeBoundary.CONTINUES:
            previous = contract.PreviousOutcomeV1.from_resolution(resolution)
            next_state, next_features = self.guarded_features(
                sequence=sequence + 1,
                session=session,
                source_ns=cycle_end - 40_000_000,
                available_ns=cycle_end - 20_000_000,
                commit_ns=cycle_end - 10_000_000,
                action_ns=cycle_end,
                previous=previous,
                freshness=freshness,
                scaling=scaling,
            )
        else:
            next_state = None
            next_features = None
        return contract.build_transition(
            state=state,
            state_features=features,
            action=action,
            hold=hold,
            reward_resolution=resolution,
            next_state=next_state,
            next_state_features=next_features,
            episode_boundary=boundary,
            duration=duration,
            cycle_end_timestamp_ns=cycle_end,
            elapsed_virtual_ns=elapsed,
            gamma=gamma,
            discount=gamma**duration,
        )

    def binding(
        self,
        *,
        gamma: float = 0.99,
        freshness: contract.FreshnessPolicyV2 | None = None,
        scaling: contract.EmpiricalScalingV2 | None = None,
        calibration_hash: str = CALIBRATION_EVIDENCE,
        queue_hash: str = QUEUE_KERNEL_EVIDENCE,
    ) -> replay.ReplayBindingV1:
        freshness = self.freshness() if freshness is None else freshness
        scaling = self.scaling() if scaling is None else scaling
        return replay.ReplayBindingV1._for_test_only(
            gamma=gamma,
            freshness_policy_sha256=freshness.canonical_sha256(),
            empirical_scaling_sha256=scaling.canonical_sha256(),
            calibration_evidence_sha256=calibration_hash,
            queue_kernel_evidence_sha256=queue_hash,
        )


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fx = Fixture()

    def buffer(self, capacity: int = 8, **binding_kwargs: object) -> replay.ReplayBufferV1:
        return replay._TestOnlyReplayBufferV1(
            capacity=capacity,
            binding=self.fx.binding(**binding_kwargs),
        )

    def one(self, transition: contract.SemiMarkovTransitionV2) -> replay.ReplayTensorBatchV1:
        buffer = self.buffer(gamma=transition.gamma)
        buffer.insert(transition)
        return buffer.sample(1, torch.Generator(device="cpu").manual_seed(3))

    # ------------------------------------------------------------------
    # Binding and exact type gates
    # ------------------------------------------------------------------

    def test_binding_pins_every_registered_surface_and_external_evidence(self) -> None:
        binding = self.fx.binding()
        self.assertEqual(binding.schema_sha256, contract.SCHEMA_SHA256)
        self.assertEqual(binding.feature_schema_sha256, contract.FEATURE_SCHEMA_SHA256)
        self.assertEqual(binding.reward_schema_sha256, contract.REWARD_SCHEMA_SHA256)
        self.assertEqual(
            binding.transition_schema_sha256, contract.TRANSITION_SCHEMA_SHA256
        )
        self.assertEqual(binding.catalog_sha256, actions.CATALOG_SHA256)
        self.assertEqual(binding.policy_feature_order, contract.POLICY_FEATURE_ORDER)
        self.assertEqual(binding.policy_feature_count, 21)
        self.assertEqual(binding.calibration_evidence_sha256, CALIBRATION_EVIDENCE)
        self.assertEqual(binding.queue_kernel_evidence_sha256, QUEUE_KERNEL_EVIDENCE)
        self.assertEqual(
            binding.training_evidence_class, contract.TRAINING_EVIDENCE_CLASS
        )
        self.assertEqual(binding.to_canonical_dict()["gamma"], 0.99)

    def test_binding_has_no_evidence_defaults_and_rejects_static_pin_drift(self) -> None:
        with self.assertRaises(TypeError):
            replay.ReplayBindingV1.from_verified_evidence(  # type: ignore[call-arg]
                gamma=0.99
            )
        bad = replace(self.fx.binding(), feature_schema_sha256="f" * 64)
        with self.assertRaises(replay.BindingMismatchError):
            bad.revalidate()
        with self.assertRaises(replay.ReplayBufferError):
            replay.ReplayBindingV1._for_test_only(
                gamma=0.99,
                freshness_policy_sha256="short",
                empirical_scaling_sha256="b" * 64,
                calibration_evidence_sha256="c" * 64,
                queue_kernel_evidence_sha256="d" * 64,
            )

    def test_buffer_requires_positive_capacity_and_exact_binding(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(replay.ReplayBufferError):
                replay._TestOnlyReplayBufferV1(  # type: ignore[arg-type]
                    value, self.fx.binding()
                )
        with self.assertRaises(replay.ReplayBufferError):
            replay._TestOnlyReplayBufferV1(1, object())  # type: ignore[arg-type]

    def test_production_buffer_fails_closed_without_verifier_attestation(self) -> None:
        test_binding = self.fx.binding()
        with self.assertRaises(replay.EvidenceEligibilityError):
            replay.ReplayBufferV1(1, test_binding)

        forged = replay._VerifiedEvidenceAttestationV1(
            calibration_evidence_sha256=CALIBRATION_EVIDENCE,
            queue_kernel_evidence_sha256=QUEUE_KERNEL_EVIDENCE,
            verifier_manifest_sha256="f" * 64,
            _token=object(),
        )
        with self.assertRaises(replay.EvidenceEligibilityError):
            replay.ReplayBindingV1.from_verified_evidence(
                gamma=0.99,
                freshness_policy_sha256=(
                    self.fx.freshness().canonical_sha256()
                ),
                empirical_scaling_sha256=(
                    self.fx.scaling().canonical_sha256()
                ),
                evidence_attestation=forged,
            )

    def test_exact_transition_type_refuses_objects_and_subclasses(self) -> None:
        buffer = self.buffer()
        with self.assertRaises(replay.TransitionRejectedError):
            buffer.insert(object())

        class SyntheticFixtureRecord:
            evidence_class = "SYNTHETIC_MECHANICS_ONLY"

            def as_transition(self) -> contract.SemiMarkovTransitionV2:
                return self.fx.transition()  # pragma: no cover - must not run

        synthetic = SyntheticFixtureRecord()
        synthetic.fx = self.fx
        with self.assertRaises(replay.TransitionRejectedError):
            buffer.insert(synthetic)
        original = self.fx.transition()

        class DerivedTransition(contract.SemiMarkovTransitionV2):
            pass

        values = {
            field.name: getattr(original, field.name)
            for field in dataclasses.fields(contract.SemiMarkovTransitionV2)
        }
        derived = DerivedTransition(**values)
        self.assertTrue(derived.is_attested)
        with self.assertRaises(replay.TransitionRejectedError):
            buffer.insert(derived)

    # ------------------------------------------------------------------
    # Exact transition content and bootstrap semantics
    # ------------------------------------------------------------------

    def test_valid_transition_tensorizes_exactly_on_cpu_float32(self) -> None:
        transition = self.fx.transition()
        batch = self.one(transition)
        self.assertEqual(batch.state.device.type, "cpu")
        self.assertEqual(batch.state.dtype, torch.float32)
        self.assertEqual(batch.state.shape, (1, 21))
        self.assertEqual(batch.mode_id.dtype, torch.int64)
        self.assertEqual(batch.q_e4.dtype, torch.int64)
        self.assertEqual(batch.duration.dtype, torch.int64)
        self.assertEqual(batch.bootstrap.dtype, torch.bool)
        self.assertTrue(
            torch.equal(
                batch.state[0],
                torch.tensor(transition.state_features.as_tuple(), dtype=torch.float32),
            )
        )
        self.assertEqual(batch.mode_id.item(), transition.action.mode_id)
        self.assertEqual(batch.q_e4.item(), transition.action.q_e4)
        self.assertAlmostEqual(batch.reward.item(), transition.reward, places=6)
        self.assertEqual(batch.duration.item(), transition.duration)

    def test_real_successor_and_previous_outcome_are_preserved(self) -> None:
        transition = self.fx.transition()
        batch = self.one(transition)
        self.assertTrue(batch.has_next_state.item())
        self.assertTrue(batch.bootstrap.item())
        self.assertTrue(
            torch.equal(
                batch.next_state[0],
                torch.tensor(
                    transition.next_state_features.as_tuple(), dtype=torch.float32
                ),
            )
        )
        expected = contract.PreviousOutcomeV1.from_resolution(
            transition.reward_resolution
        )
        actual = transition.next_state.state.previous
        self.assertIsNotNone(actual)
        self.assertEqual(actual.canonical_sha256(), expected.canonical_sha256())
        # The prior outcome is visible to learning only through the registered
        # previous-action/quality/latency/success slots in the successor vector.
        successor = transition.next_state_features.as_dict()
        self.assertEqual(successor["prev_present"], 1.0)
        self.assertEqual(successor["prev_success"], 1.0)
        self.assertEqual(successor["prev_quality_qperc"], 0.8)
        self.assertGreater(successor["prev_latency_normalized"], 0.0)

    def test_non_genesis_source_state_contains_real_previous_outcome(self) -> None:
        transition = self.fx.transition(sequence=1)
        previous = transition.state.state.previous
        self.assertIsNotNone(previous)
        self.assertEqual(previous.identity.decision_seq, 0)
        self.assertEqual(previous.action.mode_id, 2)
        self.assertEqual(previous.action.q_e4, 4000)
        self.assertEqual(previous.q_perc, 0.7)
        self.assertEqual(previous.latency_ms, 100.0)
        named = transition.state_features.as_dict()
        self.assertEqual(named["prev_present"], 1.0)
        self.assertEqual(named["prev_success"], 1.0)
        self.assertEqual(named["prev_quality_qperc"], 0.7)
        self.assertEqual(named["prev_latency_normalized"], 100.0 / 170.0)
        self.assertEqual(named["prev_joint_mode_2_one_hot"], 1.0)
        self.assertEqual(
            named["prev_q_normalized"], 4000 / float(actions.Q_E4_MAX)
        )

    def test_failure_successor_preserves_action_and_failure_not_zero_history(self) -> None:
        for event_kind in (
            contract.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
            contract.RewardEventKind.TIMEOUT,
        ):
            with self.subTest(event_kind=event_kind):
                transition = self.fx.transition(event_kind=event_kind)
                batch = self.one(transition)
                successor = transition.next_state_features.as_dict()
                self.assertEqual(batch.reward.item(), contract.REGISTERED_FAILURE_REWARD)
                self.assertTrue(batch.bootstrap.item())
                self.assertEqual(successor["prev_present"], 1.0)
                self.assertEqual(successor["prev_success"], 0.0)
                self.assertEqual(
                    successor[
                        f"prev_joint_mode_{transition.action.mode_id}_one_hot"
                    ],
                    1.0,
                )
                self.assertEqual(
                    successor["prev_q_normalized"],
                    transition.action.q_e4 / float(actions.Q_E4_MAX),
                )
                # Zero quality/latency here are registered failure encoding,
                # disambiguated by prev_present=1 and prev_success=0; they are
                # not a fabricated all-zero genesis state.
                self.assertEqual(successor["prev_quality_qperc"], 0.0)
                self.assertEqual(successor["prev_latency_normalized"], 0.0)

    def test_bootstrap_is_exactly_next_present_and_continues(self) -> None:
        for boundary, expected_terminated, expected_truncated in (
            (contract.EpisodeBoundary.CONTINUES, False, False),
            (contract.EpisodeBoundary.TERMINATED, True, False),
            (contract.EpisodeBoundary.TRUNCATED, False, True),
        ):
            with self.subTest(boundary=boundary):
                transition = self.fx.transition(boundary=boundary)
                batch = self.one(transition)
                expected = boundary is contract.EpisodeBoundary.CONTINUES
                self.assertEqual(batch.has_next_state.item(), expected)
                self.assertEqual(batch.bootstrap.item(), expected)
                self.assertEqual(batch.terminated.item(), expected_terminated)
                self.assertEqual(batch.truncated.item(), expected_truncated)
                if not expected:
                    self.assertTrue(torch.equal(batch.next_state, torch.zeros(1, 21)))

    def test_post_attestation_tamper_is_rejected_without_mutation(self) -> None:
        transition = self.fx.transition()
        buffer = self.buffer()
        original_discount = transition.discount
        object.__setattr__(transition, "discount", 0.5)
        with self.assertRaises(contract.Run4ContractError):
            buffer.insert(transition)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.accepted_count, 0)
        self.assertEqual(buffer.seen_digest_count, 0)
        self.assertEqual(buffer.seen_identity_count, 0)
        object.__setattr__(transition, "discount", original_discount)
        buffer.insert(transition)
        self.assertEqual(len(buffer), 1)

    def test_tampered_successor_previous_outcome_is_rejected(self) -> None:
        transition = self.fx.transition()
        self.assertIsNotNone(transition.next_state)
        object.__setattr__(transition.next_state.state, "previous", None)
        buffer = self.buffer()
        with self.assertRaises(contract.Run4ContractError):
            buffer.insert(transition)
        self.assertEqual((len(buffer), buffer.seen_identity_count), (0, 0))

    # ------------------------------------------------------------------
    # Homogeneous bindings and numerical boundary
    # ------------------------------------------------------------------

    def test_transition_freshness_scaling_and_gamma_must_match(self) -> None:
        alternatives = (
            self.fx.transition(freshness=self.fx.freshness("1" * 64)),
            self.fx.transition(scaling=self.fx.scaling("2" * 64)),
            self.fx.transition(gamma=0.98),
        )
        for transition in alternatives:
            with self.subTest(transition=transition.canonical_sha256()):
                with self.assertRaises(replay.BindingMismatchError):
                    self.buffer().insert(transition)

    def test_batch_carries_exact_calibration_and_queue_binding(self) -> None:
        binding = self.fx.binding()
        buffer = replay._TestOnlyReplayBufferV1(2, binding)
        buffer.insert(self.fx.transition())
        batch = buffer.sample(1, torch.Generator().manual_seed(1))
        binding.assert_exactly(batch.binding)
        self.assertEqual(
            batch.to_canonical_metadata()["binding"], binding.to_canonical_dict()
        )
        self.assertEqual(
            batch.to_canonical_metadata()["binding"]["evidence_eligibility"],
            "TEST_ONLY_MECHANICS",
        )
        different = self.fx.binding(queue_hash="9" * 64)
        with self.assertRaises(replay.BindingMismatchError):
            binding.assert_exactly(different)

    def test_float32_feature_overflow_is_rejected_before_mutation(self) -> None:
        transition = self.fx.transition(camera=1e300)
        buffer = self.buffer()
        with self.assertRaises(replay.NonFiniteInReplayDtypeError):
            buffer.insert(transition)
        self.assertEqual((len(buffer), buffer.accepted_count), (0, 0))

    def test_discount_is_contract_value_not_dtype_first_recomputation(self) -> None:
        gamma = 0.99999999
        transition = self.fx.transition(duration=150, gamma=gamma)
        batch = self.one(transition)
        emitted = batch.discount().item()
        correct = torch.tensor(transition.discount, dtype=torch.float32).item()
        wrong = torch.pow(
            torch.tensor(gamma, dtype=torch.float32),
            torch.tensor(150, dtype=torch.int64),
        ).item()
        self.assertEqual(emitted, correct)
        self.assertNotEqual(emitted, wrong)
        self.assertEqual(wrong, 1.0)

    def test_bootstrap_discount_underflow_is_rejected_but_boundary_is_safe(self) -> None:
        gamma = 0.1
        continuing = self.fx.transition(duration=150, gamma=gamma)
        with self.assertRaises(replay.NonFiniteInReplayDtypeError):
            self.buffer(gamma=gamma).insert(continuing)
        terminal = self.fx.transition(
            duration=150,
            gamma=gamma,
            boundary=contract.EpisodeBoundary.TERMINATED,
        )
        batch = self.one(terminal)
        self.assertEqual(batch.discount().item(), 0.0)
        self.assertFalse(batch.bootstrap.item())

    # ------------------------------------------------------------------
    # Lifetime identity, FIFO and sampling isolation
    # ------------------------------------------------------------------

    def test_fifo_capacity_and_lifetime_duplicate_indexes_survive_eviction(self) -> None:
        first = self.fx.transition(sequence=0)
        second = self.fx.transition(sequence=1)
        buffer = self.buffer(capacity=1)
        buffer.insert(first)
        buffer.insert(second)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.evicted_count, 1)
        self.assertEqual(buffer.accepted_count, 2)
        self.assertEqual(buffer.seen_digest_count, 2)
        self.assertEqual(buffer.seen_identity_count, 2)
        self.assertEqual(
            buffer.resident_transition_digests(), (second.canonical_sha256(),)
        )
        with self.assertRaises(replay.DuplicateTransitionError):
            buffer.insert(first)

    def test_logical_identity_conflict_survives_eviction(self) -> None:
        first = self.fx.transition(sequence=0, mode_id=3, q_e4=5000)
        conflicting = self.fx.transition(sequence=0, mode_id=4, q_e4=7000)
        buffer = self.buffer(capacity=1)
        buffer.insert(first)
        buffer.insert(self.fx.transition(sequence=1))
        with self.assertRaises(replay.IdentityConflictError):
            buffer.insert(conflicting)
        self.assertEqual(buffer.accepted_count, 2)

    def test_same_decision_seq_in_different_session_is_not_a_conflict(self) -> None:
        buffer = self.buffer()
        buffer.insert(self.fx.transition(session=SESSION))
        buffer.insert(
            self.fx.transition(session="22222222-2222-4222-8222-222222222222")
        )
        self.assertEqual(len(buffer), 2)

    def test_sampling_is_without_replacement_and_seed_reproducible(self) -> None:
        buffer = self.buffer(capacity=8)
        for sequence in range(5):
            buffer.insert(self.fx.transition(sequence=sequence))
        first = buffer.sample(4, torch.Generator().manual_seed(17))
        second = buffer.sample(4, torch.Generator().manual_seed(17))
        first_ids = [item["decision_seq"] for item in first.audit]
        second_ids = [item["decision_seq"] for item in second.audit]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))

    def test_sampling_advances_only_explicit_local_generator(self) -> None:
        buffer = self.buffer()
        for sequence in range(3):
            buffer.insert(self.fx.transition(sequence=sequence))
        torch.manual_seed(123)
        global_before = torch.random.get_rng_state().clone()
        unrelated = torch.Generator().manual_seed(55)
        unrelated_before = unrelated.get_state().clone()
        local = torch.Generator().manual_seed(77)
        local_before = local.get_state().clone()
        buffer.sample(2, local)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        self.assertTrue(torch.equal(unrelated.get_state(), unrelated_before))
        self.assertFalse(torch.equal(local.get_state(), local_before))

    def test_sampling_rejects_default_generator_and_bad_sizes(self) -> None:
        buffer = self.buffer()
        with self.assertRaises(replay.ReplaySamplingError):
            buffer.sample(1, torch.Generator())
        buffer.insert(self.fx.transition())
        with self.assertRaises(replay.ReplaySamplingError):
            buffer.sample(1, torch.default_generator)
        for size in (0, -1, True, 1.5, 2):
            with self.subTest(size=size), self.assertRaises(replay.ReplaySamplingError):
                buffer.sample(size, torch.Generator())  # type: ignore[arg-type]

    def test_batch_accessors_do_not_alias_private_storage(self) -> None:
        batch = self.one(self.fx.transition())
        accessors = (
            "state",
            "next_state",
            "mode_id",
            "q_e4",
            "reward",
            "duration",
            "has_next_state",
            "bootstrap",
            "terminated",
            "truncated",
        )
        for name in accessors:
            first = getattr(batch, name)
            second = getattr(batch, name)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            before = second.clone()
            first.fill_(0 if first.dtype is not torch.bool else True)
            self.assertTrue(torch.equal(getattr(batch, name), before))
        first_discount = batch.discount()
        second_discount = batch.discount()
        first_discount.zero_()
        self.assertFalse(torch.equal(first_discount, second_discount))

    def test_audit_identifiers_never_enter_policy_tensor(self) -> None:
        transition = self.fx.transition()
        batch = self.one(transition)
        self.assertEqual(batch.state.shape[1], contract.POLICY_FEATURE_COUNT)
        self.assertEqual(
            tuple(contract.POLICY_FEATURE_ORDER), batch.binding.policy_feature_order
        )
        self.assertEqual(batch.audit[0]["session_uuid"], SESSION)
        joined = " ".join(contract.POLICY_FEATURE_ORDER)
        self.assertNotIn("session", joined)
        self.assertNotIn("decision_seq", joined)


if __name__ == "__main__":
    unittest.main()
