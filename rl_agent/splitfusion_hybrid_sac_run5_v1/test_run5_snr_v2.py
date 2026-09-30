"""CPU-only tests: v2 SNR support/scaling, lease validity, live RAW adapter, HOLD."""

from __future__ import annotations

import os
import time
import unittest
from dataclasses import replace
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S
from rl_agent.splitfusion_hybrid_sac_run5_v1 import successor_mcs_snr_audit as SA
from rl_agent.splitfusion_hybrid_sac_run5_v1 import test_run5_state_contract as T1

EVIDENCE_ROOT = Path(os.environ.get(
    "RUN5_EVIDENCE_ROOT", Path(__file__).resolve().parents[3] / "abiodun"))
SESSION = T1.SESSION
UE_ID = T1.UE_ID
RAW = S.LIVE_CLOCK_DOMAIN


class _Clock:
    def __init__(self, start: int) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now


class _TestAdapter(S.RfsimLeaseSnrAdapterV1):
    """Deterministic clock for tests only; production uses CLOCK_MONOTONIC_RAW."""


class SnrV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        T1._Fixtures.setUpClass()
        cls.fx = T1._Fixtures("setUp")
        cls.fx.action_contract = T1._Fixtures.action_contract

    def boundary(self, commit=1_050_000_000, opened=1_060_000_000, clock=RAW):
        return R4.DecisionBoundaryV1(identity=R4.DecisionIdentityV1(SESSION, UE_ID, 0),
                                     state_commit_timestamp_ns=commit,
                                     action_open_timestamp_ns=opened, clock_domain=clock)

    def state(self, commit=1_050_000_000):
        common = dict(source_ns=commit - 20_000_000, available_ns=commit - 10_000_000,
                      sample_seq=10)
        state = self.fx.state()
        # Re-issue every Run-4 slot on the RAW clock the v2 contract requires.
        def obs(kind, value):
            observation = self.fx.observation(kind, value, **common)
            return R4.ScalarObservationV1(value, replace(observation.metadata, clock_domain=RAW),
                                          None)
        grant = self.fx.prior_grant(12, **common)
        grant = replace(grant, observation=R4.ScalarObservationV1(
            12, replace(grant.observation.metadata, clock_domain=RAW), None))
        return replace(state, camera_si=obs(R4.MeasurementKind.CAMERA_SI, 20.0),
                       radar_p40=obs(R4.MeasurementKind.RADAR_P40, 0.4), prior_ul_mcs=grant,
                       pre_action_rlc_backlog=obs(
                           R4.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES, 1023))

    def lease(self, max_age=200_000_000):
        return S.SnrLeasePolicyV1("unit-test-lease", 1, "e" * 64, max_age)

    def adapter(self, start=1_000_000_000):
        adapter = _TestAdapter(provider_id="unit-test", session_uuid=SESSION, ue_id=UE_ID)
        adapter._clock = _Clock(start)
        return adapter

    def guard(self, observation, boundary=None, lease=None):
        boundary = boundary or self.boundary()
        return S.guard_run5_state_v2(self.state(boundary.state_commit_timestamp_ns),
                                     observation, boundary, self.fx.freshness(),
                                     lease or self.lease())

    # -- scaling / support ------------------------------------------------
    def test_registered_scaling_and_support_without_clipping(self) -> None:
        self.assertEqual(S.scale_snr_db(5.5), 0.0)
        self.assertEqual(S.scale_snr_db(24.5), 1.0)
        self.assertEqual(S.scale_snr_db(15.0), 9.5 / 19.0)
        for bad in (5.499999, 24.500001, float("nan"), float("inf"), -3.0):
            with self.assertRaises(R4.ExternalFallbackRequired):
                S.scale_snr_db(bad)
        self.assertEqual(S.NETWORK_PROFILE_DESIGN_SHA256,
                         __import__("hashlib").sha256((Path(__file__).resolve().parents[1]
                                                       / "configs/network_profile_design_v2.json")
                                                      .read_bytes()).hexdigest())

    # -- lease adapter ------------------------------------------------------
    def test_held_command_stays_valid_while_heartbeat_is_fresh(self) -> None:
        adapter = self.adapter()
        adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                   target_snr_db=12.0)
        for tick in range(1, 20):  # 1.9 s of HOLD ticks: no new command, heartbeats only
            adapter._clock.now = 1_000_000_000 + tick * 100_000_000
            adapter.record_heartbeat(active_command_id="c1")
        commit = adapter._clock.now + 1
        boundary = self.boundary(commit, commit + 10_000_000)
        observation = adapter.observe(boundary)
        self.assertTrue(observation.valid)
        self.assertEqual(observation.value_db, 12.0)
        self.assertEqual(observation.effective_since_ns, 1_000_000_000)
        self.assertEqual(observation.heartbeat_ns, commit - 1)
        guarded = self.guard(observation, boundary)
        values = S.build_run5_features_v2(guarded, self.fx.scaling()).as_tuple()
        self.assertEqual(values[21], (12.0 - 5.5) / 19.0)

    def test_stale_or_mismatched_lease_falls_back(self) -> None:
        adapter = self.adapter()
        adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                   target_snr_db=12.0)
        adapter.record_heartbeat(active_command_id="c1")
        stale = self.boundary(1_150_000_000, 1_200_000_001)   # 200 ms + 1 ns
        observation = adapter.observe(stale)
        self.assertTrue(observation.valid)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(observation, stale)
        at_limit = self.boundary(1_150_000_000, 1_200_000_000)
        valid_at_limit = adapter.observe(at_limit)
        self.guard(valid_at_limit, at_limit)
        adapter._clock.now += 1
        adapter.record_heartbeat(active_command_id="c0")
        self.assertEqual(adapter.observe(self.boundary(1_050_000_000, 1_060_000_000))
                         .missing_reason, "CONTROLLER_LEASE_NAMES_ANOTHER_COMMAND")
        forged = valid_at_limit
        self.guard(forged, at_limit)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(replace(forged, heartbeat_command_id="other"), at_limit)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(replace(forged, controller_session_uuid=T1.FOREIGN_SESSION), at_limit)

    def test_no_heartbeat_errored_clamped_or_targetless_is_invalid(self) -> None:
        adapter = self.adapter()
        adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                   target_snr_db=12.0)
        self.assertEqual(adapter.observe(self.boundary()).missing_reason,
                         "NO_CONTROLLER_LEASE_BEFORE_CUTOFF")
        for index, (status, clamped, target, reason) in enumerate((
                ("ERROR", False, 13.0, "ACTIVE_COMMAND_ERRORED"),
                ("ACK", True, 13.0, "ACTIVE_COMMAND_CLAMPED"),
                ("ACK", None, None, "ACTIVE_COMMAND_HAS_NO_TARGET"))):
            adapter._clock.now += 1_000
            adapter.record_command_ack(command_id=f"bad{index}", status=status, clamped=clamped,
                                       target_snr_db=target)
            adapter.record_heartbeat(active_command_id=f"bad{index}")
            observation = adapter.observe(self.boundary())
            self.assertFalse(observation.valid)
            self.assertIsNone(observation.value_db)
            self.assertEqual(observation.missing_reason, reason)
            with self.assertRaises(R4.ExternalFallbackRequired):
                self.guard(observation)

    def test_events_after_commit_are_invisible(self) -> None:
        adapter = self.adapter()
        adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                   target_snr_db=12.0)
        adapter.record_heartbeat(active_command_id="c1")
        adapter._clock.now = 1_050_000_001
        adapter.record_command_ack(command_id="c2", status="ACK", clamped=False,
                                   target_snr_db=20.0)
        adapter.record_heartbeat(active_command_id="c2")
        observation = adapter.observe(self.boundary())
        self.assertEqual((observation.value_db, observation.active_command_id), (12.0, "c1"))

    def test_out_of_support_value_falls_back_not_clipped(self) -> None:
        adapter = self.adapter()
        adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                   target_snr_db=24.6)
        adapter.record_heartbeat(active_command_id="c1")
        observation = adapter.observe(self.boundary())
        self.assertTrue(observation.valid)
        with self.assertRaises(R4.ExternalFallbackRequired):
            self.guard(observation)

    def test_live_clock_is_raw_and_non_raw_decisions_are_refused(self) -> None:
        adapter = S.RfsimLeaseSnrAdapterV1(provider_id="p", session_uuid=SESSION, ue_id=UE_ID)
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        stamped = adapter.record_command_ack(command_id="c1", status="ACK", clamped=False,
                                             target_snr_db=10.0)
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        self.assertTrue(before <= stamped <= after)
        self.assertEqual(adapter.CLOCK_DOMAIN, "CLOCK_MONOTONIC_RAW")
        for name in ("offset_ns", "bridge", "clock_bridge", "set_offset"):
            self.assertFalse(hasattr(adapter, name))
        observation = adapter.observe(self.boundary(stamped + 10, stamped + 20,
                                                    clock="CLOCK_MONOTONIC"))
        self.assertEqual(observation.missing_reason, "DECISION_CLOCK_IS_NOT_CLOCK_MONOTONIC_RAW")
        adapter._clock = _Clock(0)
        with self.assertRaises(R4.MetadataError):
            adapter.record_heartbeat(active_command_id="c1")

    def test_observation_carries_no_profile_trace_noise_or_gnb_field(self) -> None:
        fields = set(S.UlSnrLeaseObservationV1.__dataclass_fields__)
        for forbidden in ("profile", "trace", "markov", "hidden", "noise", "future",
                          "gnb", "pusch", "step_index"):
            self.assertFalse(any(forbidden in name for name in fields), forbidden)
        with self.assertRaises(R4.MetadataError):
            replace(self.adapter().observe(self.boundary()), value_db=0.0)

    # -- HOLD handling on the retained capture --------------------------------
    def test_hold_rows_inherit_the_previous_acked_command(self) -> None:
        snr, report = SA.reconstruct_snr(SA.Ledger(EVIDENCE_ROOT))
        for profile in ("MID_VARIABLE", "FADE_RECOVERY"):
            self.assertGreater(report[profile]["hold_rows"], 0)
            self.assertEqual(report[profile]["hold_rows"],
                             report[profile]["hold_rows_whose_schedule_target_differs_from_effective"])
            self.assertEqual(report[profile]["future_joins"], 0)
        import csv
        rows = list(csv.DictReader((EVIDENCE_ROOT / SA.EV.SOURCE_RUN_RELATIVE_PATH
                                    / "cells/00__mid_variable/profile_schedule.csv").open()))
        effective = None
        for row in rows:
            if row["command_status"] != "HOLD":
                effective = float(row["target_snr_db"])
            else:
                self.assertNotEqual(snr[("MID_VARIABLE", int(row["step_index"]))],
                                    float(row["target_snr_db"]))
            self.assertEqual(snr[("MID_VARIABLE", int(row["step_index"]))], effective)


if __name__ == "__main__":
    unittest.main()
