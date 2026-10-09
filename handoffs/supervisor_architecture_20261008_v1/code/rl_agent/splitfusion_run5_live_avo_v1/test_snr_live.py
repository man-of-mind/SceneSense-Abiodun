"""Offline causal and real-local-IPC checks; no CARLA, OAI, CUDA or radio."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
import uuid
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_run5_live_avo_v1 import snr_live as S


CTRL = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"
OTHER = "33333333-3333-4333-8333-333333333333"


def boundary(commit=200, *, session=SESSION, ue="ue1", opened=None, clock=S.CLOCK_DOMAIN):
    return R4.DecisionBoundaryV1(R4.DecisionIdentityV1(session, ue, 0), commit,
                                 commit + 1 if opened is None else opened, clock)


def packet(seq=1, *, ack=100, beat=110, target=20.0, valid=True, reason=None,
           controller=CTRL):
    return S.SnrSnapshot(S.SCHEMA, controller, seq, S.CLOCK_DOMAIN,
                         f"command-{seq}", ack, beat, target, valid, reason).encode()


class CausalProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = S.LiveSnrProvider("/tmp/not-created-run5.sock", CTRL, "ue1",
                                          start_receiver=False, clock=lambda: 10)
        self.provider.adopt_session(SESSION)

    def test_original_type_and_scaling(self):
        self.assertTrue(self.provider.accept_packet(packet(), 120))
        obs = self.provider.observe(boundary())
        self.assertIsInstance(obs, S.R5.UlSnrLeaseObservationV1)
        self.assertTrue(obs.valid)
        self.assertEqual(obs.value_db, 20.)
        self.assertEqual(obs.effective_since_ns, 100)
        self.assertEqual(obs.heartbeat_ns, 110)
        self.assertEqual(S.R5.scale_snr_db(obs.value_db), (20 - 5.5) / 19)

    def test_source_and_receipt_must_be_strictly_before_commit(self):
        self.provider.accept_packet(packet(), 200)
        self.assertFalse(self.provider.observe(boundary(200)).valid)
        self.assertTrue(self.provider.observe(boundary(201)).valid)
        self.assertFalse(self.provider.accept_packet(packet(2, beat=301), 300))

    def test_selection_uses_latest_received_before_cutoff(self):
        self.provider.accept_packet(packet(), 120)
        self.provider.accept_packet(packet(2, ack=210, beat=210, target=18.), 220)
        self.assertEqual(self.provider.observe(boundary(200)).value_db, 20.)
        self.assertEqual(self.provider.observe(boundary(220)).value_db, 20.)
        self.assertEqual(self.provider.observe(boundary(221)).value_db, 18.)

    def test_latest_invalid_never_revives_older_valid_value(self):
        self.provider.accept_packet(packet(), 120)
        self.provider.accept_packet(packet(2, ack=130, beat=130, target=None,
                                           valid=False, reason="ACTIVE_COMMAND_CLAMPED"), 140)
        obs = self.provider.observe(boundary())
        self.assertFalse(obs.valid)
        self.assertIsNone(obs.value_db)
        self.assertEqual(obs.missing_reason, "ACTIVE_COMMAND_CLAMPED")

    def test_lease_inclusive_and_old_ack_can_be_held(self):
        self.provider.accept_packet(packet(ack=1, beat=1_000_000_000), 1_000_000_001)
        valid = self.provider.observe(boundary(1_199_999_999, opened=1_200_000_000))
        stale = self.provider.observe(boundary(1_200_000_000, opened=1_200_000_001))
        self.assertTrue(valid.valid)
        self.assertFalse(stale.valid)
        self.assertEqual(valid.effective_since_ns, 1)

    def test_no_zero_fill_or_support_clipping(self):
        self.assertIsNone(self.provider.observe(boundary()).value_db)
        for seq, value in enumerate((5.49, 24.51, 0.0), 1):
            self.provider.accept_packet(packet(seq, target=value), 120 + seq)
            obs = self.provider.observe(boundary())
            self.assertFalse(obs.valid)
            self.assertIsNone(obs.value_db)
        self.provider.accept_packet(packet(4, target=5.5), 130)
        self.assertTrue(self.provider.observe(boundary()).valid)
        self.provider.accept_packet(packet(5, target=24.5), 140)
        self.assertTrue(self.provider.observe(boundary()).valid)

    def test_controller_identity_cannot_change(self):
        self.assertFalse(self.provider.accept_packet(packet(controller=OTHER), 120))
        self.assertFalse(self.provider.observe(boundary()).valid)
        self.assertEqual(self.provider.stats["foreign_controller"], 1)

    def test_explicit_session_adoption_never_refreshes_lease(self):
        self.provider.accept_packet(packet(), 120)
        self.assertFalse(self.provider.observe(boundary(session=OTHER)).valid)
        self.provider.adopt_session(OTHER)
        obs = self.provider.observe(boundary(session=OTHER))
        self.assertTrue(obs.valid)
        self.assertEqual(obs.controller_session_uuid, OTHER)
        self.assertEqual(obs.effective_since_ns, 100)
        self.assertEqual(obs.heartbeat_ns, 110)
        self.assertEqual(self.provider.evidence()["controller_uuid"], CTRL)
        self.assertEqual(self.provider.adoptions[-1]["source_controller_uuid"], CTRL)
        self.assertFalse(self.provider.observe(boundary(200_000_111, session=OTHER)).valid)

    def test_foreign_ue_and_clock_are_invalid(self):
        self.provider.accept_packet(packet(), 120)
        self.assertFalse(self.provider.observe(boundary(ue="ue2")).valid)
        self.assertFalse(self.provider.observe(boundary(clock="CLOCK_MONOTONIC")).valid)

    def test_duplicate_reorder_and_clock_reversal_refused(self):
        self.assertTrue(self.provider.accept_packet(packet(3), 130))
        self.assertFalse(self.provider.accept_packet(packet(3), 131))
        self.assertFalse(self.provider.accept_packet(packet(2), 131))
        self.assertFalse(self.provider.accept_packet(packet(4, beat=109), 132))
        self.assertFalse(self.provider.accept_packet(packet(5), 129))
        self.assertEqual(self.provider.observe(boundary()).active_command_id, "command-3")

    def test_extra_profile_or_future_fields_rejected(self):
        for field in ("profile_id", "step_index", "reason", "future_snr", "noise_db", "gnb_snr"):
            body = json.loads(packet())
            body[field] = "must not enter actor input"
            self.assertFalse(self.provider.accept_packet(json.dumps(body).encode(), 120))
        self.assertFalse(self.provider.accept_packet(b"{" + b" " * 5000, 120))

    def test_malformed_field_types_do_not_escape_receiver(self):
        for field, value in (("controller_uuid", 42), ("heartbeat_raw_ns", True),
                             ("sequence", "1"), ("valid", 1)):
            body = json.loads(packet())
            body[field] = value
            self.assertFalse(self.provider.accept_packet(json.dumps(body).encode(), 120))

    def test_history_is_bounded(self):
        provider = S.LiveSnrProvider("/tmp/not-created-run5.sock", CTRL, "ue1",
                                    start_receiver=False, capacity=2)
        provider.adopt_session(SESSION)
        for seq in range(1, 5):
            provider.accept_packet(packet(seq, ack=seq, beat=seq), seq + 1)
        self.assertEqual(len(provider.evidence()["received_snapshots"]), 2)
        self.assertFalse(provider.observe(boundary(2)).valid)


class PublisherTests(unittest.TestCase):
    def publisher(self, clock=lambda: 100):
        result = S.ControllerPublisher("/tmp/nonexistent-run5-snr.sock", CTRL, clock=clock)
        self.addCleanup(result.close)
        return result

    def test_ack_late_is_current_not_future_and_missing_ipc_does_not_kill_actuator(self):
        publisher = self.publisher()
        publisher.on_row({"command_timing_status": "ACK_LATE", "target_snr_db": 20.,
                          "command_ack_monotonic_ns": 1234, "profile_id": "NEVER_PUBLISHED"})
        self.assertTrue(publisher.current.valid)
        self.assertEqual(publisher.current.effective_raw_ns, 100)
        self.assertEqual(publisher.stats["send_errors"], 1)
        self.assertNotIn("profile_id", publisher.current.encode().decode())

    def test_skip_tick_holds_old_ack_and_never_exposes_unsent_target(self):
        now = [100]
        publisher = self.publisher(lambda: now[0])
        publisher.on_row({"command_timing_status": "ACK_ON_TIME", "target_snr_db": 20.,
                          "command_ack_monotonic_ns": 1234})
        command_id = publisher.current.command_id
        now[0] = 100_000_100
        publisher.on_row({"command_timing_status": "SKIP_OBSOLETE_NEVER_BURST",
                          "target_snr_db": 6.})
        self.assertEqual(publisher.current.target_snr_db, 20.)
        self.assertEqual(publisher.current.command_id, command_id)
        self.assertEqual(publisher.current.effective_raw_ns, 100)
        self.assertEqual(publisher.current.heartbeat_raw_ns, 100_000_100)

    def test_initial_skip_and_unknown_status_are_invalid(self):
        publisher = self.publisher()
        publisher.on_row({"command_timing_status": "SKIP_OBSOLETE_NEVER_BURST", "target_snr_db": 20.})
        self.assertFalse(publisher.current.valid)
        publisher.on_row({"command_timing_status": "ERROR", "target_snr_db": 20.})
        self.assertFalse(publisher.current.valid)

    def test_missing_ack_refused(self):
        publisher = self.publisher()
        with self.assertRaises(ValueError):
            publisher.on_row({"command_timing_status": "ACK_ON_TIME", "target_snr_db": 20.})

    def test_audit_writer_failure_invalidates_lease_without_throwing(self):
        publisher = self.publisher()
        publisher.audit = mock.Mock()
        publisher.audit.write.side_effect = OSError("simulated full disk")
        publisher.on_row({"command_timing_status": "ACK_ON_TIME", "target_snr_db": 20.,
                          "command_ack_monotonic_ns": 1234})
        self.assertFalse(publisher.current.valid)
        self.assertEqual(publisher.current.missing_reason, "SNR_EVIDENCE_WRITER_FAILED")

    def test_controller_wrapper_preserves_original_args_and_rows(self):
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_target_snr_runtime as legacy
        order = []
        row = {"command_timing_status": "ACK_ON_TIME", "target_snr_db": 20.,
               "command_ack_monotonic_ns": 1234}
        publisher = mock.Mock()

        def original(*args):
            self.assertIs(args[2], row)
            order.append("original")

        def main(args):
            self.assertEqual(args, ["--profile-id", "FAVORABLE_STABLE"])
            legacy.write_row("handle", "writer", row)
            return 17

        publisher.on_row.side_effect = lambda r: order.append("raw-event")
        with mock.patch.object(legacy, "write_row", original), mock.patch.object(legacy, "main", main), \
                mock.patch.object(S, "ControllerPublisher", return_value=publisher):
            code = S.controller_main(["--controller", "--snr-socket", "/tmp/test.sock",
                                      "--snr-controller-id", CTRL,
                                      "--profile-id", "FAVORABLE_STABLE"])
            self.assertIs(legacy.write_row, original)
        self.assertEqual(code, 17)
        self.assertEqual(order, ["original", "raw-event"])
        publisher.close.assert_called_once()


class LocalIpcTests(unittest.TestCase):
    def test_real_unix_transport_raw_clock_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="r5snr-") as directory:
            path = Path(directory) / "state.sock"
            provider = S.LiveSnrProvider(path, CTRL, "ue1")
            publisher = S.ControllerPublisher(path, CTRL)
            try:
                provider.adopt_session(SESSION)
                publisher.on_row({"command_timing_status": "ACK_ON_TIME", "target_snr_db": 20.,
                                  "command_ack_monotonic_ns": time.monotonic_ns()})
                limit = time.monotonic() + 1.
                while provider.stats["accepted"] < 1 and time.monotonic() < limit:
                    time.sleep(.001)
                commit = S.raw_now_ns()
                obs = provider.observe(boundary(commit))
                self.assertTrue(obs.valid)
                row = provider.evidence()["received_snapshots"][-1]
                self.assertLess(row["heartbeat_raw_ns"], row["receipt_raw_ns"])
                self.assertLess(row["receipt_raw_ns"], commit)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(FileExistsError):
                    S.LiveSnrProvider(path, CTRL, "ue1")
            finally:
                publisher.close()
                provider.close()
            self.assertFalse(path.exists())

    def test_existing_regular_file_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="r5snr-") as directory:
            path = Path(directory) / "foreign"
            path.touch()
            with self.assertRaises(FileExistsError):
                S.LiveSnrProvider(path, CTRL, "ue1")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
