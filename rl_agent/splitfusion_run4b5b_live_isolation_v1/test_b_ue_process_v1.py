"""CPU-only tests for the W10275 B process entrypoint."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_ue_process_v1 as U
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_ack_v1 as A
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_trace_v1 as T
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import postrun_artifact_v1 as G


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class BUEProcessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest_path = root / "actor.json"
        order = L.RUN4B_FEATURE_ORDER
        self.manifest = L.BActorManifestV1(
            variant=L.ActorVariant.RUN4B, feature_order=order,
            feature_count=len(order),
            feature_schema_sha256=L.feature_schema_sha256(
                L.ActorVariant.RUN4B, order),
            actor_boundary_sha256=sha("actor"),
            weights_file_sha256=sha("weights"), selected_seed=43,
            selected_update=10_000)
        self.manifest_path.write_text(json.dumps(self.manifest.as_dict()))
        self.output = root / "output"
        self.evidence = root / "evidence"
        self.raw = {
            "schema": U.REQUEST_SCHEMA, "role": U.ROLE,
            "run_id": "run4b_offline_300", "variant": self.manifest.variant.value,
            "config_binding_sha256": sha("config"),
            "actor_boundary_sha256": self.manifest.actor_boundary_sha256,
            "feature_schema_sha256": self.manifest.feature_schema_sha256,
            "transmitted_budget": 300, "deadline_ns": A.ACK_DEADLINE_NS,
            "ack_semantics": "TAIL_OUTPUT_READY__GT_FREE__BEFORE_MAP_AND_EVALUATION",
            "postrun_semantics": "CARLA_GT_AND_QPERC_POSTRUN_ONLY__NEVER_LIVE_FEEDBACK",
            "clock_domain": A.CLOCK_DOMAIN,
            "split_host": {
                "carla_host": "W10275.idcc.lab", "ue_host": "W10275.idcc.lab",
                "cn_host": "L10319.idcc.lab", "edge_host": "L10319.idcc.lab",
                "ext_dn_host": "L10319.idcc.lab",
                "ack_receiver_host": "W10275.idcc.lab", "ack_receiver_port": 41070},
            "output_root": str(self.output), "evidence_root": str(self.evidence),
            "actor_manifest_path": str(self.manifest_path),
            "remote_attempt_root": None,
            "required_authority_modules": list(U.REQUIRED_AUTHORITIES),
            "old_live_quality_runtime_permitted": False}

    def tearDown(self): self.temp.cleanup()

    def encoded(self):
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(payload).decode()

    def test_request_and_selected_manifest_are_exact(self):
        request = U.BUEProcessRequestV1.from_b64(self.encoded())
        self.assertEqual(request.variant, L.ActorVariant.RUN4B)
        self.assertEqual(U.load_selected_live_manifest(request), self.manifest)
        with self.assertRaisesRegex(U.FinalActorUnavailableError,
                                    "joint-channel"):
            U.production_preflight(request)

    def test_offline_fake_runs_exact_300_and_writes_postrun_only_gt(self):
        request = U.BUEProcessRequestV1.from_b64(self.encoded())
        result = U.offline_fake(request)
        self.assertEqual(result["transmitted_frames"], 300)
        self.assertEqual(result["terminal_status"], "COMPLETE")
        report = json.loads((self.output / U.REPORT_NAME).read_text())
        self.assertEqual(report["policy_decisions"], 300)
        self.assertEqual(report["operational_successes"], 270)
        self.assertEqual(report["operational_timeouts"], 30)
        self.assertEqual(report["ground_truth_records"], 300)
        self.assertFalse(report["live_qperc_computed"])
        self.assertFalse(report["live_reward_computed"])
        self.assertFalse(report["gt_used_for_ack_or_state"])
        outcomes = A.OperationalEvidenceStoreV1.open_existing(
            self.evidence / "operational_evidence").load_outcomes()
        traces = T.OperationalTraceStoreV1.open_existing(
            self.evidence / "operational_trace").verify_all()
        gt = G.GroundTruthEvidenceStoreV1.open_existing(
            self.evidence / "carla_gt").verify_all()
        self.assertEqual((len(outcomes), len(traces), len(gt)), (300, 300, 300))

    def test_offline_fake_cli_is_runnable_but_production_run_is_blocked(self):
        self.assertEqual(U.main(["--offline-fake", "--request-b64", self.encoded()]), 0)
        root = Path(self.temp.name)
        self.raw["output_root"] = str(root / "output2")
        self.raw["evidence_root"] = str(root / "evidence2")
        with self.assertRaises(U.FinalActorUnavailableError):
            U.main(["run", "--request-b64", self.encoded(),
                    "--execute", U.EXECUTE_TOKEN])

    def test_request_refuses_old_live_quality_or_wrong_feature_binding(self):
        self.raw["old_live_quality_runtime_permitted"] = True
        with self.assertRaises(U.BUEProcessError):
            U.BUEProcessRequestV1.from_b64(self.encoded())
        self.raw["old_live_quality_runtime_permitted"] = False
        self.raw["feature_schema_sha256"] = sha("foreign")
        with self.assertRaises(U.BUEProcessError):
            U.BUEProcessRequestV1.from_b64(self.encoded())


if __name__ == "__main__":
    unittest.main()
