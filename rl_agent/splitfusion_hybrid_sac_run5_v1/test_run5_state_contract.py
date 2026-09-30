"""Adversarial CPU-only tests for the Run-5 22-D state contract and provider."""

from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
import unittest
from dataclasses import replace

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1 import test_run4_contract as R4T
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as C

SESSION = R4T.SESSION
FOREIGN_SESSION = "22222222-2222-4222-8222-222222222222"
UE_ID = R4T.UE_ID
CLOCK = R4T.CLOCK
EVIDENCE_C = "c" * 64
EVIDENCE_D = "d" * 64


def bits(value: float) -> bytes:
    return struct.pack("<d", value)


class _Fixtures(R4T.Run4ContractTest):
    """Run-4 fixtures only; never collected (no test methods are inherited)."""


# Borrow fixtures without re-collecting Run 4's tests under this module.
for _name in list(vars(R4T.Run4ContractTest)):
    if _name.startswith("test_"):
        setattr(_Fixtures, _name, None)
del _name


class Run5StateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _Fixtures.setUpClass()
        cls.fx = _Fixtures("setUp")
        cls.fx.action_contract = _Fixtures.action_contract

    # -- fixtures --------------------------------------------------------

    def snr_freshness(self, max_age_ns: int = 250_000_000) -> C.SnrProxyFreshnessV1:
        return C.SnrProxyFreshnessV1("unit-test-snr-freshness", 1, EVIDENCE_C, max_age_ns)

    def snr_scaling(self, low: float = 5.0, high: float = 25.0) -> C.SnrProxyScalingV1:
        return C.SnrProxyScalingV1("unit-test-only-not-production", 1, EVIDENCE_D,
                                   15.0, 10.0, low, high)

    def command(self, seq: int, ack_ns: int, target: float | None = 12.5, *,
                status: C.RfsimCommandStatus = C.RfsimCommandStatus.ACK,
                clamped: bool | None = False, session: str = SESSION,
                clock: str = CLOCK, send_ns: int | None = None) -> C.RfsimSnrCommandRecordV1:
        return C.RfsimSnrCommandRecordV1(
            session_uuid=session, command_seq=seq, clock_domain=clock,
            send_timestamp_ns=ack_ns - 200_000 if send_ns is None else send_ns,
            ack_timestamp_ns=ack_ns, status=status,
            clamped=None if target is None else clamped, target_snr_db=target)

    def provider(self, *records: C.RfsimSnrCommandRecordV1) -> C.RfsimEffectiveSnrProviderV1:
        return C.RfsimEffectiveSnrProviderV1(
            provider_id="unit-test-rfsim", session_uuid=SESSION, ue_id=UE_ID,
            clock_domain=CLOCK, records=records)

    def valid_snr(self, boundary: R4.DecisionBoundaryV1, value: float = 12.5,
                  ack_ns: int | None = None) -> C.UlSnrProxyObservationV1:
        ack = boundary.state_commit_timestamp_ns - 50_000_000 if ack_ns is None else ack_ns
        observation = self.provider(self.command(0, ack, value)).observe(boundary)
        self.assertTrue(observation.valid)
        return observation

    def guard(self, state=None, boundary=None, snr=None, **kw) -> C.GuardedRun5StateV1:
        boundary = self.fx.boundary() if boundary is None else boundary
        return C.guard_run5_state_for_action(
            self.fx.state() if state is None else state,
            self.valid_snr(boundary) if snr is None else snr,
            boundary, self.fx.freshness(),
            kw.get("freshness", self.snr_freshness()),
            kw.get("scaling", self.snr_scaling()))

    # -- schema ----------------------------------------------------------

    def test_exactly_22_features_and_positions_0_to_20_are_run4(self) -> None:
        self.assertEqual(C.RUN5_POLICY_FEATURE_COUNT, 22)
        self.assertEqual(len(C.RUN5_POLICY_FEATURE_ORDER), 22)
        self.assertEqual(C.RUN5_POLICY_FEATURE_ORDER[:21], R4.POLICY_FEATURE_ORDER)
        self.assertEqual(C.RUN5_POLICY_FEATURE_ORDER[21],
                         "effective_external_ul_snr_proxy_scaled")
        self.assertEqual(C.SNR_FEATURE_INDEX, 21)
        self.assertNotEqual(C.FEATURE_SCHEMA_SHA256, R4.FEATURE_SCHEMA_SHA256)
        self.assertEqual(C.SNR_PROXY_LABEL, "SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB")

    def test_run4_frozen_semantics_are_untouched(self) -> None:
        self.assertEqual(R4.POLICY_FEATURE_COUNT, 21)
        self.assertEqual(R4.REWARD_DEADLINE_MS, 170.0)
        self.assertEqual(R4.REWARD_LATENCY_WEIGHT, 0.25)
        self.assertEqual(R4.REGISTERED_FAILURE_REWARD, -1.0)
        descriptor = json.dumps(C.FEATURE_SCHEMA_DESCRIPTOR,
                                default=lambda value: dict(value))
        self.assertIn(R4.FEATURE_SCHEMA_SHA256, descriptor)
        self.assertIn(R4.SCHEMA_SHA256, descriptor)

    def test_leakage_terms_are_denied_in_feature_names(self) -> None:
        for term in ("profile", "trace", "markov", "future", "noise", "gnb", "pusch",
                     "timestamp", "valid", "target"):
            self.assertIn(term, C.RUN5_FORBIDDEN_FEATURE_TERMS)
        original = C.RUN5_POLICY_FEATURE_ORDER
        try:
            for bad in ("snr_profile_id", "trace_index_scaled", "gnb_pusch_snr_db",
                        "future_target_snr", "rfsim_noise_db", "markov_hidden"):
                C.RUN5_POLICY_FEATURE_ORDER = (*original[:21], bad)
                with self.assertRaises(C.Run5ContractError):
                    C.assert_run5_feature_schema()
        finally:
            C.RUN5_POLICY_FEATURE_ORDER = original
        C.assert_run5_feature_schema()

    # -- bit identity ----------------------------------------------------

    def test_prefix_is_bit_identical_to_run4_for_genesis_success_and_failure(self) -> None:
        scaling = self.fx.scaling()
        cases = []
        genesis_state = self.fx.state()
        cases.append((genesis_state, self.fx.boundary()))
        for resolution in (
            self.fx.success(latency_ns=85_000_000, q_perc=0.75),
            R4.resolve_reward(self.fx.event(kind=R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE)),
            R4.resolve_reward(self.fx.event(kind=R4.RewardEventKind.TIMEOUT,
                                            latency_ns=171_000_000)),
        ):
            previous = R4.PreviousOutcomeV1.from_resolution(resolution)
            action_ns = 1_260_000_000
            state = self.fx.state(sequence=1, previous=previous,
                                  source_ns=action_ns - 40_000_000,
                                  available_ns=action_ns - 30_000_000, sample_seq=7)
            cases.append((state, self.fx.boundary(sequence=1, commit_ns=action_ns - 10_000_000,
                                                  action_ns=action_ns)))
        for state, boundary in cases:
            run4 = R4.build_policy_features(
                R4.guard_state_for_action(state, boundary, self.fx.freshness()), scaling)
            guarded = self.guard(state=state, boundary=boundary)
            run5 = C.build_run5_policy_features(guarded, scaling, self.snr_scaling())
            values = run5.as_tuple()
            self.assertEqual(len(values), 22)
            self.assertEqual([bits(v) for v in values[:21]],
                             [bits(v) for v in run4.as_tuple()])
            self.assertEqual(values[21], (12.5 - 15.0) / 10.0)

    def test_prefix_bit_identity_over_randomized_scalars(self) -> None:
        rng = random.Random(20260929)
        scaling = self.fx.scaling()
        for _ in range(200):
            camera = rng.uniform(0.0, 80.0)
            radar = rng.random()
            mcs = rng.randint(0, 28)
            backlog = rng.randint(0, 50_000_000)
            snr_db = rng.uniform(5.0, 25.0)
            common = dict(source_ns=1_000_000_000, available_ns=1_010_000_000, sample_seq=10)
            state = self.fx.state(
                camera=self.fx.observation(R4.MeasurementKind.CAMERA_SI, camera, **common),
                radar=self.fx.observation(R4.MeasurementKind.RADAR_P40, radar, **common),
                mcs=self.fx.prior_grant(mcs, **common),
                backlog=self.fx.observation(
                    R4.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES, backlog, **common))
            boundary = self.fx.boundary()
            run4 = R4.build_policy_features(
                R4.guard_state_for_action(state, boundary, self.fx.freshness()), scaling)
            guarded = self.guard(state=state, boundary=boundary,
                                 snr=self.valid_snr(boundary, snr_db))
            run5 = C.build_run5_policy_features(guarded, scaling, self.snr_scaling())
            self.assertEqual([bits(v) for v in run5.as_tuple()[:21]],
                             [bits(v) for v in run4.as_tuple()])

    # -- guard fail-closed -----------------------------------------------

    def test_missing_snr_never_zero_fills(self) -> None:
        boundary = self.fx.boundary()
        with self.assertRaises(R4.MetadataError):
            C.UlSnrProxyObservationV1(
                identity=R4.SampleIdentityV1(SESSION, UE_ID, 0),
                kind=C.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB,
                provider_id="p", selection_rule_id="r", value_db=0.0,
                source_timestamp_ns=None, available_timestamp_ns=None,
                clock_domain=CLOCK, valid=False, missing_reason="absent")
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(R4.MetadataError):
                C.UlSnrProxyObservationV1(
                    identity=R4.SampleIdentityV1(SESSION, UE_ID, 0),
                    kind=C.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB,
                    provider_id="p", selection_rule_id="r", value_db=bad,
                    source_timestamp_ns=1, available_timestamp_ns=1,
                    clock_domain=CLOCK, valid=True, missing_reason=None)
        missing = self.provider().observe(boundary)
        self.assertFalse(missing.valid)
        self.assertIsNone(missing.value_db)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(boundary=boundary, snr=missing)
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.guard_run5_state_for_action(self.fx.state(), None, boundary,
                                          self.fx.freshness(), self.snr_freshness(),
                                          self.snr_scaling())

    def test_stale_out_of_support_foreign_and_late_snr_fall_back(self) -> None:
        boundary = self.fx.boundary()
        commit = boundary.state_commit_timestamp_ns
        opened = boundary.action_open_timestamp_ns
        # stale: age == max is admitted, max + 1 is refused
        at_limit = self.valid_snr(boundary, ack_ns=opened - 250_000_000)
        self.guard(boundary=boundary, snr=at_limit)
        stale = self.valid_snr(boundary, ack_ns=opened - 250_000_001)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(boundary=boundary, snr=stale)
        # support is closed at both ends
        for value, admitted in ((5.0, True), (25.0, True), (4.999, False), (25.001, False)):
            snr = self.valid_snr(boundary, value)
            if admitted:
                self.guard(boundary=boundary, snr=snr)
            else:
                with self.assertRaises(R4.ExternalFallbackRequired):
                    self.guard(boundary=boundary, snr=snr)
        good = self.valid_snr(boundary)
        for forged in (
            replace(good, identity=R4.SampleIdentityV1(FOREIGN_SESSION, UE_ID, 0)),
            replace(good, identity=R4.SampleIdentityV1(SESSION, "ue-2", 0)),
            replace(good, clock_domain="OTHER_CLOCK"),
            replace(good, available_timestamp_ns=commit + 1),
        ):
            with self.assertRaises(R4.ExternalFallbackRequired):
                self.guard(boundary=boundary, snr=forged)

    def test_run4_guard_failures_still_fall_back_with_valid_snr(self) -> None:
        boundary = self.fx.boundary()
        stale_mcs = self.fx.state(mcs=self.fx.prior_grant(
            12, source_ns=900_000_000, available_ns=910_000_000))
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(state=stale_mcs, boundary=boundary)

    def test_features_require_guard_and_matching_scaling(self) -> None:
        guarded = self.guard()
        with self.assertRaises(R4.ScalingError):
            C.build_run5_policy_features(guarded, self.fx.scaling(),
                                         self.snr_scaling(low=0.0))
        forged_guard = replace(guarded, _attestation=None)
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.build_run5_policy_features(forged_guard, self.fx.scaling(), self.snr_scaling())
        vector = C.build_run5_policy_features(guarded, self.fx.scaling(), self.snr_scaling())
        forged = C.Run5PolicyFeatureVectorV1(
            values=vector.values, run4_prefix_sha256=vector.run4_prefix_sha256,
            guarded_state_sha256=vector.guarded_state_sha256,
            snr_scaling_sha256=vector.snr_scaling_sha256)
        with self.assertRaises(R4.ScalingError):
            forged.as_tuple()
        with self.assertRaises(R4.ScalingError):
            C.Run5PolicyFeatureVectorV1(
                values=vector.values[:21], run4_prefix_sha256=vector.run4_prefix_sha256,
                guarded_state_sha256=vector.guarded_state_sha256,
                snr_scaling_sha256=vector.snr_scaling_sha256)

    def test_bindings_have_no_defaults(self) -> None:
        with self.assertRaises(TypeError):
            C.SnrProxyScalingV1()  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            C.SnrProxyFreshnessV1()  # type: ignore[call-arg]
        with self.assertRaises(R4.ScalingError):
            C.SnrProxyScalingV1("s", 1, EVIDENCE_D, 15.0, 0.0, 5.0, 25.0)
        with self.assertRaises(R4.ScalingError):
            C.SnrProxyScalingV1("s", 1, EVIDENCE_D, 15.0, 10.0, 25.0, 5.0)

    # -- RFsim provider causality ----------------------------------------

    def test_selects_latest_ack_at_or_before_commit_never_after(self) -> None:
        boundary = self.fx.boundary()
        commit = boundary.state_commit_timestamp_ns
        provider = self.provider(
            self.command(0, commit - 100_000_000, 10.0),
            self.command(1, commit, 11.0),                # exactly at commit: admitted
            self.command(2, commit + 300_000, 99.0,       # sent before, ACKed after
                         send_ns=commit),
            self.command(3, boundary.action_open_timestamp_ns + 5, 98.0),  # future
        )
        observation = provider.observe(boundary)
        self.assertEqual(observation.value_db, 11.0)
        self.assertEqual(observation.source_timestamp_ns, commit)
        self.assertLess(observation.source_timestamp_ns, boundary.action_open_timestamp_ns)

    def test_future_records_do_not_change_a_past_observation(self) -> None:
        boundary = self.fx.boundary()
        commit = boundary.state_commit_timestamp_ns
        base = [self.command(0, commit - 90_000_000, 13.0)]
        before = self.provider(*base).observe(boundary)
        future = [*base, self.command(1, commit + 10, 30.0), self.command(2, commit + 10**9, 1.0)]
        after = self.provider(*future).observe(boundary)
        self.assertEqual(before.value_db, after.value_db)
        self.assertEqual(before.source_timestamp_ns, after.source_timestamp_ns)

    def test_in_flight_newer_command_keeps_previous_ack_and_is_flagged(self) -> None:
        boundary = self.fx.boundary()
        commit = boundary.state_commit_timestamp_ns
        observation = self.provider(
            self.command(0, commit - 90_000_000, 13.0),
            self.command(1, commit + 400_000, 20.0, send_ns=commit - 100_000),
        ).observe(boundary)
        self.assertTrue(observation.valid)
        self.assertEqual(observation.value_db, 13.0)
        self.assertTrue(observation.newer_command_in_flight)

    def test_clamped_errored_or_targetless_latest_is_invalid_not_resurrected(self) -> None:
        boundary = self.fx.boundary()
        commit = boundary.state_commit_timestamp_ns
        older = self.command(0, commit - 90_000_000, 13.0)
        for latest, reason in (
            (self.command(1, commit - 10, 14.0, clamped=True), "LATEST_EFFECTIVE_COMMAND_CLAMPED"),
            (self.command(1, commit - 10, 14.0, status=C.RfsimCommandStatus.ERROR),
             "LATEST_EFFECTIVE_COMMAND_ERRORED"),
            (self.command(1, commit - 10, None), "LATEST_EFFECTIVE_COMMAND_HAS_NO_TARGET"),
        ):
            observation = self.provider(older, latest).observe(boundary)
            self.assertFalse(observation.valid)
            self.assertIsNone(observation.value_db)
            self.assertEqual(observation.missing_reason, reason)
            with self.assertRaises(R4.ExternalFallbackRequired):
                self.guard(boundary=boundary, snr=observation)

    def test_no_command_before_cutoff_is_missing_even_if_later_exist(self) -> None:
        boundary = self.fx.boundary()
        observation = self.provider(
            self.command(0, boundary.action_open_timestamp_ns + 1, 13.0)).observe(boundary)
        self.assertFalse(observation.valid)
        self.assertEqual(observation.missing_reason, "NO_EFFECTIVE_COMMAND_BEFORE_CUTOFF")

    def test_foreign_session_clock_order_and_overlap_are_rejected(self) -> None:
        provider = self.provider(self.command(0, 1_000_000_000))
        with self.assertRaises(R4.MetadataError):
            provider.ingest(self.command(1, 1_100_000_000, session=FOREIGN_SESSION))
        with self.assertRaises(R4.MetadataError):
            provider.ingest(self.command(1, 1_100_000_000, clock="OTHER_CLOCK"))
        with self.assertRaises(R4.MetadataError):
            provider.ingest(self.command(0, 1_100_000_000))
        with self.assertRaises(R4.MetadataError):
            provider.ingest(self.command(1, 1_100_000_000, send_ns=999_999_000))
        foreign_boundary = R4.DecisionBoundaryV1(
            identity=R4.DecisionIdentityV1(FOREIGN_SESSION, UE_ID, 0),
            state_commit_timestamp_ns=2_000_000_000, action_open_timestamp_ns=2_000_000_001,
            clock_domain=CLOCK)
        observation = provider.observe(foreign_boundary)
        self.assertFalse(observation.valid)
        self.assertEqual(observation.missing_reason, "DECISION_IDENTITY_FOREIGN_TO_PROVIDER")

    def test_log_entry_allow_list_drops_profile_trace_and_noise(self) -> None:
        entry = {
            "ack_monotonic_ns": 1_000_200_000, "send_monotonic_ns": 1_000_000_000,
            "status": "ACK", "clamped": False, "target_snr_db": 17.25,
            "profile_id": "ADVERSE_STABLE", "step_index": 42, "reason": "PROFILE_REPLAY",
            "commanded_noise_power_db": -3.25, "markov_state": 2, "trace_id": "t",
        }
        record = C.RfsimSnrCommandRecordV1.from_log_entry(
            entry, session_uuid=SESSION, command_seq=0, clock_domain=CLOCK)
        for name in C.RFSIM_LOG_FIELDS_NEVER_EXPOSED:
            self.assertFalse(hasattr(record, name), name)
        boundary = self.fx.boundary()
        observation = self.provider(record).observe(boundary)
        text = observation.canonical_bytes().decode()
        for leaked in ("ADVERSE_STABLE", "PROFILE_REPLAY", "-3.25", "step_index", "profile"):
            self.assertNotIn(leaked, text)
        with self.assertRaises(R4.MetadataError):
            C.RfsimSnrCommandRecordV1.from_log_entry(
                {**entry, "status": "TIMEOUT"}, session_uuid=SESSION, command_seq=0,
                clock_domain=CLOCK)

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
import rl_agent.splitfusion_hybrid_sac_run5_v1.run5_state_contract as c
assert c.RUN5_POLICY_FEATURE_COUNT == 22
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
