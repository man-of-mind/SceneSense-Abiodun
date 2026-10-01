"""Adversarial CPU-only tests for the Run-5B 21-D state, transport prior and reward."""

from __future__ import annotations

import dataclasses
import json
import math
import random
import struct
import subprocess
import sys
import unittest
from dataclasses import replace

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as C
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity

SESSION = "11111111-1111-4111-8111-111111111111"
FOREIGN = "22222222-2222-4222-8222-222222222222"
UE_ID = "ue-unit-test"
RAW = SNR.LIVE_CLOCK_DOMAIN
K = R4.MeasurementKind


def bits(value: float) -> bytes:
    return struct.pack("<d", value)


class Fixtures:
    catalog = action_contract.load_contract()

    def action(self, mode_id=3, q_e4=5000) -> ExecutedActionIdentity:
        executable = self.catalog.resolve(mode_id, q_e4 / float(action_contract.Q_E4_SCALE))
        return ExecutedActionIdentity.from_executable_action(executable, self.catalog)

    def decision(self, sequence=0, session=SESSION):
        return R4.DecisionIdentityV1(session, UE_ID, sequence)

    def obs(self, kind, value, *, source_ns, available_ns, sample_seq=10, clock=RAW,
            session=SESSION):
        observer, direction = {
            K.CAMERA_SI: (R4.Observer.SCENE_PIPELINE, R4.LinkDirection.NOT_APPLICABLE),
            K.RADAR_P40: (R4.Observer.SCENE_PIPELINE, R4.LinkDirection.NOT_APPLICABLE),
            K.UE_PRIOR_NEW_DATA_UL_MCS_INDEX: (R4.Observer.UE, R4.LinkDirection.UPLINK),
            K.UE_PRE_ACTION_RLC_BACKLOG_BYTES: (R4.Observer.UE, R4.LinkDirection.UPLINK),
        }[kind]
        valid = value is not None
        return R4.ScalarObservationV1(value, R4.MeasurementMetadataV1(
            identity=R4.SampleIdentityV1(session, UE_ID, sample_seq), kind=kind,
            observer=observer, link_direction=direction, source="unit-test",
            source_timestamp_ns=source_ns, available_timestamp_ns=available_ns,
            clock_domain=clock, valid=valid), None if valid else "not reported")

    def grant(self, mcs, **kw):
        return R4.PriorUlGrantObservationV1(
            observation=self.obs(K.UE_PRIOR_NEW_DATA_UL_MCS_INDEX, mcs, **kw),
            mcs_table=R4.UL_MCS_TABLE_ID, harq_round=0 if mcs is not None else None,
            new_data_indicator=1 if mcs is not None else None,
            grant_identity="grant-1" if mcs is not None else None,
            scheduler_policy_id=R4.UL_MCS_POLICY_ID,
            selection_rule_id=R4.UL_MCS_SELECTION_RULE_ID)

    def slots(self, commit, camera=20.0, radar=0.4, mcs=12, backlog=1023, **overrides):
        common = dict(source_ns=commit - 20_000_000, available_ns=commit - 10_000_000)
        slots = dict(camera_si=self.obs(K.CAMERA_SI, camera, **common),
                     radar_p40=self.obs(K.RADAR_P40, radar, **common),
                     prior_ul_mcs=self.grant(mcs, **common),
                     pre_action_rlc_backlog=self.obs(K.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                                                     backlog, **common))
        slots.update(overrides)
        return slots

    def boundary(self, sequence=0, commit=1_250_000_000, opened=1_260_000_000, clock=RAW):
        return R4.DecisionBoundaryV1(identity=self.decision(sequence),
                                     state_commit_timestamp_ns=commit,
                                     action_open_timestamp_ns=opened, clock_domain=clock)

    def state(self, sequence=0, previous=None, commit=1_250_000_000, **kw):
        return C.Run5BPolicyStateV1(identity=self.decision(sequence), previous=previous,
                                    **self.slots(commit, **kw))

    def run4_state(self, sequence=0, previous=None, commit=1_250_000_000, **kw):
        return R4.PolicyStateV2(identity=self.decision(sequence), previous=previous,
                                **self.slots(commit, **kw))

    def freshness(self):
        return R4.FreshnessPolicyV2("unit-test-freshness", 1, "a" * 64, 100_000_000,
                                    100_000_000, 100_000_000, 100_000_000)

    def scaling(self):
        return R4.EmpiricalScalingV2("unit-test-scaling", 1, "b" * 64, 10.0, 5.0,
                                     math.log1p(1023))

    def lease(self, max_age=200_000_000):
        return SNR.SnrLeasePolicyV1("unit-test-lease", 1, "e" * 64, max_age)

    def snr(self, boundary, value=12.0, ack_lead=20_000_000, beat_lead=5_000_000,
            session=SESSION, **ack):
        adapter = SNR.ModeledLeaseSnrAdapterV1(provider_id="unit-test", session_uuid=session,
                                               ue_id=UE_ID)
        commit = boundary.state_commit_timestamp_ns
        adapter.record_command_ack_at(at_ns=commit - ack_lead, command_id="c1",
                                      status=ack.get("status", "ACK"),
                                      clamped=ack.get("clamped", False), target_snr_db=value)
        adapter.record_heartbeat_at(at_ns=commit - beat_lead, active_command_id="c1")
        return adapter.observe(boundary)

    def resolution(self, kind=R4.RewardEventKind.DELIVERED_SUCCESS, latency_ns=85_000_000,
                   q_perc=0.75, sequence=0, action=None):
        opened = 1_060_000_000
        return R4.resolve_reward(R4.RewardEventV1(
            identity=self.decision(sequence), action=action or self.action(), kind=kind,
            action_open_timestamp_ns=opened, resolution_timestamp_ns=opened + latency_ns,
            clock_domain=RAW, source="unit-test",
            q_perc=q_perc if kind is R4.RewardEventKind.DELIVERED_SUCCESS else None))

    def features(self, state, boundary=None, snr=None, lease=None):
        boundary = boundary or self.boundary(state.identity.decision_seq)
        guarded = C.guard_run5b_state(state, snr if snr is not None else self.snr(boundary),
                                      boundary, self.freshness(), lease or self.lease())
        return C.build_run5b_policy_features(guarded, self.scaling())


class Run5BStateContractTest(unittest.TestCase, Fixtures):
    # -- schema ------------------------------------------------------------
    def test_exactly_21_features_run4b_order_plus_snr(self) -> None:
        self.assertEqual(C.RUN5B_POLICY_FEATURE_COUNT, 21)
        self.assertEqual(len(C.RUN5B_POLICY_FEATURE_ORDER), 21)
        expected = [n for n in R4.POLICY_FEATURE_ORDER if n != "prev_quality_qperc"]
        self.assertEqual(list(C.RUN5B_POLICY_FEATURE_ORDER[:20]), expected)
        self.assertEqual(C.RUN5B_POLICY_FEATURE_ORDER[20], R5V1.SNR_FEATURE_NAME)
        self.assertNotIn("prev_quality_qperc", C.RUN5B_POLICY_FEATURE_ORDER)
        self.assertNotEqual(C.FEATURE_SCHEMA_SHA256, R4.FEATURE_SCHEMA_SHA256)
        self.assertNotEqual(C.FEATURE_SCHEMA_SHA256, SNR.FEATURE_SCHEMA_SHA256)
        self.assertNotEqual(C.FEATURE_SCHEMA_SHA256, R5V1.FEATURE_SCHEMA_SHA256)
        # Run-4 and Run-5B are both 21 wide but have different orders.
        self.assertEqual(len(R4.POLICY_FEATURE_ORDER), 21)
        self.assertNotEqual(tuple(R4.POLICY_FEATURE_ORDER), C.RUN5B_POLICY_FEATURE_ORDER)

    def test_leakage_terms_are_denied_in_feature_names(self) -> None:
        original = C.RUN5B_POLICY_FEATURE_ORDER
        for leaked in ("prev_quality_qperc", "prev_reward", "prev_q_perc", "gt_quality",
                       "profile_id", "future_snr", "gnb_pusch_snr"):
            with self.subTest(leaked):
                try:
                    C.RUN5B_POLICY_FEATURE_ORDER = (*original[:20], leaked)
                    with self.assertRaises(C.Run5BContractError):
                        C.assert_run5b_feature_schema()
                finally:
                    C.RUN5B_POLICY_FEATURE_ORDER = original
        C.assert_run5b_feature_schema()

    # -- transport prior ---------------------------------------------------
    def test_transport_prior_type_has_no_quality_field(self) -> None:
        names = {f.name for f in dataclasses.fields(C.TransportPriorOutcomeV1)}
        for forbidden in ("q_perc", "quality", "reward", "gt", "ground_truth"):
            self.assertFalse(any(forbidden in n for n in names), forbidden)
        prior = C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution())
        with self.assertRaises((AttributeError, TypeError)):
            prior.q_perc  # noqa: B018 - slots: the attribute does not exist
        self.assertNotIn("q_perc", json.dumps(prior.to_canonical_dict()))
        self.assertNotIn("0.75", json.dumps(prior.to_canonical_dict()))

    def test_successful_prior_is_valid_without_any_qperc(self) -> None:
        prior = C.TransportPriorOutcomeV1(
            identity=self.decision(0), action=self.action(), terminal=C.TransportTerminal.SUCCESS,
            operational_latency_ms=85.0, available_timestamp_ns=1_145_000_000,
            clock_domain=RAW, source=C.TransportPriorSource.OPERATIONAL_ACK,
            evidence_sha256="f" * 64)
        values = self.features(self.state(1, prior)).as_tuple()
        self.assertEqual(values[C.PREV_LATENCY_INDEX], 85.0 / 170.0)
        self.assertEqual(values[C.PREV_PRESENT_INDEX:C.PREV_SUCCESS_INDEX + 1], (1.0, 1.0))
        # The Run-4 previous type cannot even be built without Q_perc.
        with self.assertRaises(R4.MetadataError):
            R4.PreviousOutcomeV1(identity=self.decision(0), action=self.action(),
                                 terminal=R4.RewardTerminal.SUCCESS, q_perc=None,
                                 latency_ms=85.0, available_timestamp_ns=1_145_000_000,
                                 clock_domain=RAW, reward_resolution_sha256="f" * 64)

    def test_prior_projection_never_reads_qperc(self) -> None:
        low = C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution(q_perc=0.05))
        high = C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution(q_perc=0.95))
        self.assertEqual(low.canonical_bytes(), high.canonical_bytes())
        self.assertEqual(self.features(self.state(1, low)).as_tuple(),
                         self.features(self.state(1, high)).as_tuple())

    def test_run4_previous_outcome_is_refused_as_prior(self) -> None:
        previous = R4.PreviousOutcomeV1.from_resolution(self.resolution())
        with self.assertRaisesRegex(R4.MetadataError, "Q_perc"):
            self.state(1, previous)

    def test_prior_validity_rules(self) -> None:
        base = dict(identity=self.decision(0), action=self.action(),
                    available_timestamp_ns=1_145_000_000, clock_domain=RAW,
                    source=C.TransportPriorSource.OPERATIONAL_ACK, evidence_sha256="f" * 64)
        for terminal, latency in ((C.TransportTerminal.SUCCESS, None),
                                  (C.TransportTerminal.SUCCESS, 170.001),
                                  (C.TransportTerminal.SUCCESS, -1.0),
                                  (C.TransportTerminal.SUCCESS, float("nan")),
                                  (C.TransportTerminal.TIMEOUT, 85.0),
                                  (C.TransportTerminal.REGISTERED_DELIVERY_FAILURE, 0.0)):
            with self.subTest(terminal=terminal, latency=latency):
                with self.assertRaises(R4.MetadataError):
                    C.TransportPriorOutcomeV1(terminal=terminal, operational_latency_ms=latency,
                                              **base)
        for kind in (R4.RewardEventKind.INFRASTRUCTURE_FAULT, R4.RewardEventKind.EVALUATOR_FAULT):
            with self.assertRaises(R4.MetadataError):
                C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution(kind=kind))
        forged = replace(self.resolution(), _attestation=None)
        with self.assertRaises(R4.RewardError):
            C.TransportPriorOutcomeV1.from_reward_resolution(forged)

    def test_previous_must_be_the_immediately_preceding_decision(self) -> None:
        prior = C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution(sequence=0))
        with self.assertRaises(R4.MetadataError):
            self.state(2, prior)
        with self.assertRaises(R4.MetadataError):
            self.state(1, None)
        with self.assertRaises(R4.MetadataError):
            self.state(0, prior)
        foreign = replace(prior, identity=R4.DecisionIdentityV1(FOREIGN, UE_ID, 0))
        with self.assertRaises(R4.MetadataError):
            self.state(1, foreign)

    # -- previous-outcome encoding -------------------------------------------
    def test_genesis_success_failure_timeout_encodings(self) -> None:
        genesis = self.features(self.state(0)).as_tuple()
        self.assertEqual(genesis[C.PREVIOUS_SLICE], (0.0,) * 16)
        vectors = {}
        for kind, latency in ((R4.RewardEventKind.DELIVERED_SUCCESS, 85_000_000),
                              (R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE, 85_000_000),
                              (R4.RewardEventKind.REGISTERED_SERVICE_FAILURE, 85_000_000),
                              (R4.RewardEventKind.TIMEOUT, 171_000_000)):
            prior = C.TransportPriorOutcomeV1.from_reward_resolution(
                self.resolution(kind=kind, latency_ns=latency, action=self.action(7, 4000)))
            vectors[kind] = self.features(self.state(1, prior)).as_tuple()
        success = vectors[R4.RewardEventKind.DELIVERED_SUCCESS]
        self.assertEqual(success[4 + 7], 1.0)
        self.assertEqual(sum(success[C.PREV_MODE_SLICE]), 1.0)
        self.assertEqual(success[C.PREV_Q_INDEX], 4000 / float(action_contract.Q_E4_MAX))
        self.assertEqual(success[C.PREV_LATENCY_INDEX], 85.0 / 170.0)
        self.assertEqual(success[C.PREV_PRESENT_INDEX:C.PREV_SUCCESS_INDEX + 1], (1.0, 1.0))
        failure = vectors[R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE]
        self.assertEqual(failure[C.PREV_LATENCY_INDEX:C.PREV_SUCCESS_INDEX + 1], (0.0, 1.0, 0.0))
        self.assertEqual(failure[C.PREV_Q_INDEX], success[C.PREV_Q_INDEX])
        for kind in (R4.RewardEventKind.REGISTERED_SERVICE_FAILURE, R4.RewardEventKind.TIMEOUT):
            self.assertEqual(vectors[kind], failure)
        self.assertNotEqual(failure, success)

    def test_transport_features_bit_identical_to_run4_values_over_random_scalars(self) -> None:
        # Audit only: every Run-5B transport feature equals the Run-4 value of the
        # same measured quantity; Run-4's prev_quality_qperc has no counterpart.
        rng = random.Random(20260930)
        mapping = [R4.POLICY_FEATURE_ORDER.index(n) for n in C.RUN4B_POLICY_FEATURE_ORDER]
        kinds = list(R4.RewardEventKind)[:4]
        for trial in range(200):
            kw = dict(camera=rng.uniform(0.0, 80.0), radar=rng.random(),
                      mcs=rng.randint(0, 28), backlog=rng.randint(0, 50_000_000))
            sequence, previous, prior = 0, None, None
            if trial % 4:
                kind = kinds[trial % 4]
                latency = 171_000_000 if kind is R4.RewardEventKind.TIMEOUT else rng.randint(
                    0, 170_000_000)
                resolution = self.resolution(kind=kind, latency_ns=latency,
                                             q_perc=rng.random(),
                                             action=self.action(rng.randint(0, 11), 5000))
                sequence, previous = 1, R4.PreviousOutcomeV1.from_resolution(resolution)
                prior = C.TransportPriorOutcomeV1.from_reward_resolution(resolution)
            snr_db = rng.uniform(5.5, 24.5)
            boundary = self.boundary(sequence)
            run4 = R4.build_policy_features(R4.guard_state_for_action(
                self.run4_state(sequence, previous, **kw), boundary, self.freshness()),
                self.scaling()).as_tuple()
            values = self.features(self.state(sequence, prior, **kw), boundary,
                                   self.snr(boundary, snr_db)).as_tuple()
            self.assertEqual([bits(v) for v in values[:20]], [bits(run4[i]) for i in mapping])
            self.assertEqual(values[20], (snr_db - 5.5) / 19.0)
            if prior is not None and prior.success:
                self.assertNotIn(previous.q_perc, values)

    # -- reward (unchanged Run-4 semantics) -------------------------------------
    def test_reward_formula_and_inclusive_deadline(self) -> None:
        exact = self.resolution(latency_ns=170_000_000, q_perc=0.6)
        self.assertEqual(exact.terminal, R4.RewardTerminal.SUCCESS)
        self.assertEqual(exact.reward, 0.6 - 0.25 * (170.0 / 170.0))
        late = self.resolution(latency_ns=170_000_001, q_perc=0.6)
        self.assertEqual((late.terminal, late.reward), (R4.RewardTerminal.TIMEOUT, -1.0))
        mid = self.resolution(latency_ns=85_000_000, q_perc=0.75)
        self.assertEqual(mid.reward, 0.75 - 0.25 * (85.0 / 170.0))
        for kind in (R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
                     R4.RewardEventKind.REGISTERED_SERVICE_FAILURE):
            self.assertEqual(self.resolution(kind=kind).reward, -1.0)
        for kind in (R4.RewardEventKind.INFRASTRUCTURE_FAULT, R4.RewardEventKind.EVALUATOR_FAULT):
            fault = self.resolution(kind=kind)
            self.assertFalse(fault.learning_included)
            self.assertIsNone(fault.reward)
        prior = C.TransportPriorOutcomeV1.from_reward_resolution(exact)
        self.assertEqual(prior.operational_latency_ms, 170.0)
        self.assertEqual(C.FEATURE_SCHEMA_DESCRIPTOR["reward"]["schema_sha256"],
                         R4.REWARD_SCHEMA_SHA256)

    # -- guard fail-closed ------------------------------------------------------
    def test_snr_failures_fall_back_and_never_zero_fill(self) -> None:
        state, boundary = self.state(0), self.boundary(0)
        cases = {
            "absent": None,
            "clamped": self.snr(boundary, clamped=True),
            "errored": self.snr(boundary, status="ERROR"),
            "out_of_support_low": self.snr(boundary, 5.49),
            "out_of_support_high": self.snr(boundary, 24.51),
            "stale_lease": self.snr(boundary, beat_lead=250_000_000, ack_lead=260_000_000),
            "foreign_session": self.snr(boundary, session=FOREIGN),
        }
        for name, observation in cases.items():
            with self.subTest(name):
                with self.assertRaises(R4.ExternalFallbackRequired):
                    C.guard_run5b_state(state, observation, boundary, self.freshness(),
                                        self.lease())
        late = self.snr(boundary, ack_lead=-1, beat_lead=-2)
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.guard_run5b_state(state, late, boundary, self.freshness(), self.lease())
        other_clock = self.boundary(0, clock="CLOCK_MONOTONIC")
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.guard_run5b_state(self.state(0), self.snr(boundary), other_clock,
                                self.freshness(), self.lease())

    def test_measurement_failures_fall_back(self) -> None:
        commit = 1_250_000_000
        late = dict(source_ns=commit - 5_000_000, available_ns=commit + 1)
        stale = dict(source_ns=commit - 200_000_000, available_ns=commit - 190_000_000)
        cases = {
            "missing_backlog": dict(pre_action_rlc_backlog=self.obs(
                K.UE_PRE_ACTION_RLC_BACKLOG_BYTES, None, source_ns=commit - 20_000_000,
                available_ns=commit - 10_000_000)),
            "camera_after_commit": dict(camera_si=self.obs(K.CAMERA_SI, 20.0, **late)),
            "stale_mcs": dict(prior_ul_mcs=self.grant(12, **stale)),
            "radar_other_scene": dict(radar_p40=self.obs(
                K.RADAR_P40, 0.4, source_ns=commit - 20_000_000,
                available_ns=commit - 10_000_000, sample_seq=11)),
            "foreign_clock": dict(camera_si=self.obs(
                K.CAMERA_SI, 20.0, source_ns=commit - 20_000_000,
                available_ns=commit - 10_000_000, clock="OTHER")),
            "negative_camera": dict(camera=-1.0),
            "radar_out_of_range": dict(radar=1.5),
            "float_backlog": dict(backlog=10.5),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                with self.assertRaises(R4.ExternalFallbackRequired):
                    self.features(self.state(0, **kw))
        prior = C.TransportPriorOutcomeV1.from_reward_resolution(self.resolution())
        unavailable = replace(prior, available_timestamp_ns=commit + 1)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.features(self.state(1, unavailable))
        other = replace(prior, clock_domain="OTHER")
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.features(self.state(1, other))

    def test_features_require_guard_attestation(self) -> None:
        state, boundary = self.state(0), self.boundary(0)
        guarded = C.guard_run5b_state(state, self.snr(boundary), boundary, self.freshness(),
                                      self.lease())
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.build_run5b_policy_features(replace(guarded, _attestation=None), self.scaling())
        tampered = replace(guarded, heartbeat_age_ns=1, _attestation=guarded._attestation)
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.build_run5b_policy_features(tampered, self.scaling())
        vector = C.build_run5b_policy_features(guarded, self.scaling())
        forged = replace(vector, values=(0.0,) * 21)
        with self.assertRaises(R4.ScalingError):
            forged.as_tuple()
        with self.assertRaises(R4.MetadataError):
            C.guard_run5b_state(self.run4_state(0), self.snr(boundary), boundary,
                                self.freshness(), self.lease())

    def test_import_has_no_filesystem_socket_process_or_torch_side_effect(self) -> None:
        probe = r'''
import json, sys
violations = []
def audit(event, args):
    try:
        if event == "open":
            path = str(args[0]).lower()
            if "/experiments/" in path or path.endswith((".csv", ".json")):
                violations.append([event, path])
        elif event in ("subprocess.Popen", "os.system", "socket.socket", "socket.connect"):
            violations.append([event, str(args)[:120]])
    except Exception:
        pass
sys.addaudithook(audit)
import rl_agent.splitfusion_hybrid_sac_run5b_v1.run5b_state_contract as c
assert c.RUN5B_POLICY_FEATURE_COUNT == 21
assert "torch" not in sys.modules
print("VIOLATIONS:" + json.dumps(violations))
'''
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                   text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        line = [l for l in completed.stdout.splitlines() if l.startswith("VIOLATIONS:")][0]
        self.assertEqual(json.loads(line[len("VIOLATIONS:"):]), [])


if __name__ == "__main__":
    unittest.main()
