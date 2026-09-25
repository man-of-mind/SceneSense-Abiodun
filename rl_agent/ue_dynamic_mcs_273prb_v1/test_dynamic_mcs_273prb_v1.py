"""Offline tests for the target-radio dynamic prior-MCS preregistration."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rl_agent.ue_dynamic_mcs_273prb_v1 import analysis as A
from rl_agent.ue_dynamic_mcs_273prb_v1 import contract as C
from rl_agent.ue_dynamic_mcs_273prb_v1 import probe_sender as S
from rl_agent.ue_dynamic_mcs_273prb_v1 import runner as R
from rl_agent.ue_mcs_backlog_calibration_v1.decision_join import UeUlGrant


def grant(stamp: int, mcs: int = 7) -> UeUlGrant:
    return UeUlGrant(
        monotonic_ns=stamp, rnti=0x1234, dci_frame=1, dci_slot=2,
        sched_frame=1, sched_slot=4, mcs=mcs, mcs_table=0, rb_size=8,
        tbs=1000, harq_pid=3, ndi=0, rv=0, harq_round=0,
    )


class ContractTests(unittest.TestCase):
    def test_target_radio_is_exact(self) -> None:
        self.assertEqual(C.RADIO_PROFILE_ID, "OAI_N78_100MHZ_273PRB_4D5U_V1")
        self.assertEqual(C.RADIO["bandwidth_mhz"], 100)
        self.assertEqual(C.RADIO["prb"], 273)
        self.assertEqual((C.RADIO["downlink_slots"], C.RADIO["uplink_slots"]), (4, 5))

    def test_only_dynamic_profiles_are_registered(self) -> None:
        self.assertEqual(C.PROFILE_IDS, ("MID_VARIABLE", "FADE_RECOVERY"))
        plan = C.build_plan()
        self.assertEqual([row.run_index for row in plan], [0, 1])
        self.assertEqual({row.profile_id for row in plan}, set(C.PROFILE_IDS))

    def test_real_probe_action_is_pinned(self) -> None:
        self.assertEqual(C.PROBE_ACTION_ID, 68)
        self.assertEqual(C.PROBE_PROFILE_ID, "split_ae32_uint4_q5000")
        self.assertEqual(C.PROBE_PAYLOAD_BYTES, 129_707)
        self.assertEqual(C.PROBE_CHUNKS_PER_FRAME, 3)

    def test_partitions_are_disjoint_and_complete(self) -> None:
        self.assertEqual(C.partition_for(0), C.FIT)
        self.assertEqual(C.partition_for(209), C.FIT)
        self.assertEqual(C.partition_for(210), C.INTERNAL_VALIDATION)
        self.assertEqual(C.partition_for(299), C.INTERNAL_VALIDATION)

    def test_duration_two_never_crosses_reset(self) -> None:
        pairs = C.registered_transition_indices()
        self.assertEqual(len(pairs), 296)
        self.assertIn((207, 209), pairs)
        self.assertIn((210, 212), pairs)
        for forbidden in (208, 209, 298, 299):
            with self.assertRaises(C.ContractError):
                C.successor_index(forbidden)
        for current, successor in pairs:
            self.assertEqual(successor - current, 2)
            self.assertEqual(C.partition_for(current), C.partition_for(successor))

    def test_design_excludes_hidden_target_from_policy_boundary(self) -> None:
        design = C.design_record()
        selection = design["selection"]
        self.assertIn("NEVER_POLICY_INPUT", selection["target_snr_and_profile_id"])
        self.assertEqual(selection["source"], "UE_DECODED_UL_DCI_ONLY")

    def test_all_frozen_sources_rehash(self) -> None:
        report = C.verify_sources(C.ROOT)
        self.assertTrue(report["verified"])
        self.assertFalse(report["problems"])


class CausalSelectionTests(unittest.TestCase):
    def test_equal_timestamp_is_not_prior(self) -> None:
        rows = [grant(100, 4), grant(200, 9)]
        selected, status, age = A._select_prior(rows, [100, 200], 200)
        self.assertEqual(selected.mcs, 4)
        self.assertEqual(status, "VALID")
        self.assertEqual(age, 0.0001)

    def test_no_prior_is_explicit_missing(self) -> None:
        selected, status, age = A._select_prior([grant(100)], [100], 100)
        self.assertIsNone(selected)
        self.assertEqual(status, "MISSING_NO_PRIOR_GRANT")
        self.assertIsNone(age)

    def test_stale_is_not_converted_to_numeric_state(self) -> None:
        selected, status, age = A._select_prior(
            [grant(1, 0)], [1], C.MCS_MAX_AGE_NS + 2,
        )
        self.assertEqual(selected.mcs, 0)
        self.assertEqual(status, "STALE")
        self.assertGreater(age, 200.0)

    def test_mcs_zero_is_a_valid_measurement(self) -> None:
        selected, status, _ = A._select_prior([grant(100, 0)], [100], 101)
        self.assertEqual(status, "VALID")
        self.assertEqual(selected.mcs, 0)

    def test_grant_digest_binds_identity(self) -> None:
        self.assertNotEqual(A._grant_digest(grant(100, 7)),
                            A._grant_digest(grant(101, 7)))


class SenderRefusalTests(unittest.TestCase):
    def test_wrong_frame_count_refused_before_socket_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                frames=1, payload_bytes=C.PROBE_PAYLOAD_BYTES,
                period_ns=C.PERIOD_NS, profile_id=C.PROFILE_IDS[0],
                trace_id=C.PROFILE_IDENTITIES[C.PROFILE_IDS[0]]["trace_id"],
                payload_seed=1, send_buffer_bytes=1024, bind_host="127.0.0.1",
                remote_host="127.0.0.1", remote_port=9,
                start_monotonic_ns=1, log_csv=Path(temp) / "rows.csv",
                summary_json=Path(temp) / "summary.json",
            )
            with self.assertRaises(SystemExit):
                S.run(args)
            self.assertFalse(args.log_csv.exists())


class RunnerHelperTests(unittest.TestCase):
    def test_container_states_uses_each_exact_inspect_result(self) -> None:
        responses = (
            SimpleNamespace(returncode=0, stdout="true healthy\n"),
            SimpleNamespace(returncode=1, stdout="missing\n"),
        )
        with mock.patch.object(R.subprocess, "run", side_effect=responses) as run:
            states = R._container_states(("present", "absent"))
        self.assertEqual(
            states,
            {"present": "true healthy", "absent": "ABSENT"},
        )
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0:6], [
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
            ])


if __name__ == "__main__":
    unittest.main()
