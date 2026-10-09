"""Offline adversarial tests for causal-SNR postflight; no hardware access."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import actor_binding as A
from .verify_snr_capture import verify_capture


class SnrCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="run5_capture_check_")
        self.addCleanup(self.tmp.cleanup)
        self.capture = Path(self.tmp.name)
        self.cell = self.capture / "cells/test"
        (self.cell / "run5_policy").mkdir(parents=True)
        (self.cell / "run4_phase6").mkdir()
        snapshot = {"receipt_raw_ns": 110, "heartbeat_raw_ns": 100, "effective_raw_ns": 90,
                    "sequence": 1, "command_id": "ack-1", "controller_uuid": "controller",
                    "target_snr_db": 15.0, "valid": True}
        observation = {"active_command_id": "ack-1", "heartbeat_command_id": "ack-1",
                       "value_db": 15.0, "effective_since_ns": 90, "heartbeat_ns": 100,
                       "clock_domain": "CLOCK_MONOTONIC_RAW", "valid": True,
                       "controller_session_uuid": "session",
                       "identity": {"ue_id": "ue", "session_uuid": "session"}}
        identity = {"session_uuid": "session", "decision_seq": 0, "ue_id": "ue"}
        row = {"frame_id": 1, "admitted": True, "snr_scaled": .5,
               "decision_identity": identity, "snr": observation,
               "boundary": {"state_commit_timestamp_ns": 120, "action_open_timestamp_ns": 125,
                            "clock_domain": "CLOCK_MONOTONIC_RAW", "identity": dict(identity)}}
        self.evidence = {"actor_tree_sha256": A.ACTOR_TREE_SHA256, "weights_sha256": A.WEIGHTS_SHA256,
                         "state_dim": 22, "controller_started": True, "runtime_bound": True,
                         "provider": {"received_snapshots": [snapshot], "controller_uuid": "controller"},
                         "decision_snr": [row]}
        self.write(self.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json",
                   {"decisions": [{"kind": "POLICY_DECISION", "frame_id": 1,
                       "tensor_seq": 2, "decision_session_uuid": "session", "features": [0.] * 21 + [.5],
                       "state_commit": {"domain": "CLOCK_MONOTONIC_RAW", "ns": 120},
                       "action_open": {"domain": "CLOCK_MONOTONIC_RAW", "ns": 125}}],
                    "transmitted_identities": [{"run4_identity": {"frame_kind": "POLICY_DECISION", "frame_id": 1,
                       "session_uuid": "session", "decision_seq": 0, "tensor_seq": 2}}]})
        self.write(self.capture / "PROFILER_ANALYSIS.json",
                   {"policy_runtime": {"observed_state_dim": 22},
                    "semantics": {"actor_replay": {"passed": True}}})

    @staticmethod
    def write(path, body):
        path.write_text(json.dumps(body))

    def verify(self):
        self.write(self.cell / "run5_policy/SNR_EVIDENCE.json", self.evidence)
        return verify_capture(self.capture)

    def test_exact_causal_receipt_passes(self):
        self.assertTrue(self.verify()["passed"])

    def test_future_received_new_target_does_not_invalidate_causal_previous(self):
        old = self.evidence["provider"]["received_snapshots"][0]
        self.evidence["provider"]["received_snapshots"].append({**old, "sequence": 2,
            "receipt_raw_ns": 126, "heartbeat_raw_ns": 123, "target_snr_db": 20.0})
        self.assertTrue(self.verify()["passed"])

    def test_future_used_as_current_refused(self):
        self.evidence["provider"]["received_snapshots"][0]["receipt_raw_ns"] = 120
        self.assertFalse(self.verify()["passed"])

    def test_newer_causal_target_not_selected_refused(self):
        old = self.evidence["provider"]["received_snapshots"][0]
        self.evidence["provider"]["received_snapshots"].append({**old, "sequence": 2,
            "receipt_raw_ns": 119, "heartbeat_raw_ns": 118, "target_snr_db": 20.0})
        self.assertFalse(self.verify()["passed"])

    def test_wrong_actor_and_stale_lease_refused(self):
        self.evidence["actor_tree_sha256"] = "wrong"
        self.evidence["decision_snr"][0]["boundary"]["action_open_timestamp_ns"] = 300_000_000
        report = self.verify()
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["original_run5_actor"])
        self.assertFalse(report["checks"]["all_admitted_snr_is_latest_received_causal_lease"])

    def test_valid_lease_from_other_decision_or_boundary_refused(self):
        self.evidence["decision_snr"][0]["decision_identity"]["decision_seq"] = 1
        self.assertFalse(self.verify()["checks"]["snr_is_bound_to_exact_executed_decision_and_state_feature"])

    def test_duplicate_observation_refused(self):
        self.evidence["decision_snr"].append(self.evidence["decision_snr"][0])
        self.assertFalse(self.verify()["checks"]["unique_exact_decision_population"])

    def test_feature_relabel_refused(self):
        path = self.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json"
        evidence = json.loads(path.read_text())
        evidence["decisions"][0]["features"][21] = .4
        self.write(path, evidence)
        self.assertFalse(self.verify()["checks"]["snr_is_bound_to_exact_executed_decision_and_state_feature"])


if __name__ == "__main__":
    unittest.main()
