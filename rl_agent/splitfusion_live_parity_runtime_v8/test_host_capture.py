"""Versioned proof gates must not accept old readiness or partial coverage."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import host_capture as H, contract as C
from rl_agent.splitfusion_live_parity_runtime_v7 import host_capture as V7


class HostGateTests(unittest.TestCase):
    @contextmanager
    def proof(self, kind):
        with tempfile.TemporaryDirectory(prefix="v8-proof-") as tmp:
            root = Path(tmp)
            sources = {"registered.py": "a" * 64}
            if kind == "offline":
                suites = {name: dict(returncode=0, tests=1, skips=[], skipped=0) for name in H.OFFLINE_SUITES}
                value = dict(schema="splitfusion.parity_runtime.offline.v8", passed=True,
                    live_started=False, training_started=False, sources=sources, fixtures={},
                    suites=suites, total_tests=len(suites), logs={})
                logs = [name + ".log" for name in suites]
            else:
                value = dict(schema="splitfusion.parity_runtime.edge_startup_scope.v8", passed=True,
                    live_started=False, cuda_initialized=False, sources=sources, image_id=C.IMAGE_ID,
                    edge_module=C.EDGE, checks=dict.fromkeys(H.STARTUP_CHECKS, True),
                    publisher=H.publisher_binding(), sources_unchanged=True, ready_exists=False, errors=[], logs={})
                logs = ["container_check.log"]
            for name in logs:
                path = root / name
                path.write_text("proof fixture\n")
                value["logs"][name] = C.sha(path)
            proof = root / "REPORT.json"
            proof.write_text(json.dumps(value))
            with mock.patch.object(C, "source_inventory", return_value=sources), \
                    mock.patch.object(H, "CPU_FIXTURE_FILES", {}):
                yield root, proof, value

    def test_full_offline_suite_set_is_additive_and_only_old_skip_is_allowed(self):
        self.assertTrue(V7.OFFLINE_SUITES < H.OFFLINE_SUITES)
        self.assertEqual(H.ALLOWED_SKIP, V7.ALLOWED_SKIP)
        self.assertTrue({"handoff_v1", "optimization_v5", "optimization_v6", "integration_v8"} <= H.OFFLINE_SUITES)
        with self.proof("offline") as (root, path, value):
            self.assertTrue(H.verify_offline(path, root)["passed"])
            value["suites"]["frozen_semantics"].update(skipped=1, skips=[H.ALLOWED_SKIP])
            path.write_text(json.dumps(value))
            self.assertTrue(H.verify_offline(path, root)["passed"])

    def test_offline_failure_omission_foreign_skip_and_old_version_fail_closed(self):
        mutations = [lambda x: x.update(schema="splitfusion.parity_runtime.offline.v7"),
            lambda x: x["suites"].pop("handoff_v1"),
            lambda x: x["suites"]["integration_v8"].update(returncode=1),
            lambda x: x["suites"]["handoff_v1"].update(tests=0),
            lambda x: x["suites"]["handoff_v1"].update(skipped=1, skips=[H.ALLOWED_SKIP]),
            lambda x: x.update(total_tests=1), lambda x: x.update(sources={}),
            lambda x: x["logs"].pop("handoff_v1.log")]
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.proof("offline") as (root, path, value):
                mutate(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    H.verify_offline(path, root)

    def test_new_startup_proof_requires_each_real_lifecycle_and_packet_check(self):
        with self.proof("startup") as (root, path, value):
            self.assertTrue(H.verify_startup_scope(path, root)["passed"])
            for name in H.STARTUP_CHECKS:
                changed = copy.deepcopy(value)
                changed["checks"][name] = False
                path.write_text(json.dumps(changed))
                with self.subTest(check=name), self.assertRaises(ValueError):
                    H.verify_startup_scope(path, root)

    def test_startup_old_image_version_identity_sources_or_live_ready_rejected(self):
        for key, value in (("schema", "splitfusion.parity_runtime.edge_startup_scope.v7"),
                ("image_id", "foreign"), ("publisher", {}), ("errors", ["failure"]),
                ("ready_exists", True), ("sources_unchanged", False), ("cuda_initialized", True)):
            with self.subTest(key=key), self.proof("startup") as (root, path, original):
                original[key] = value
                path.write_text(json.dumps(original))
                with self.assertRaises(ValueError):
                    H.verify_startup_scope(path, root)

    def test_startup_log_tamper_refused(self):
        with self.proof("startup") as (root, path, _value):
            (root / "container_check.log").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "log drift"):
                H.verify_startup_scope(path, root)

    def test_host_uid_is_explicit_and_root_is_not_a_host_publisher(self):
        with mock.patch.object(H.os, "getuid", return_value=1234), \
                mock.patch.object(H.os, "getgid", return_value=2345):
            value = H.publisher_binding()
            self.assertEqual((value["publisher_uid"], value["publisher_gid"]), (1234, 2345))
            self.assertEqual(value["socket_mode"], "0600")
        with mock.patch.object(H.os, "getuid", return_value=0), self.assertRaises(ValueError):
            H.publisher_binding()

    def test_handoff_proof_old_worker_report_is_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="v8-foreign-handoff-") as tmp:
            root = Path(tmp)
            report, manifest = root / "REPORT.json", root / "OUTPUT_SHA256.json"
            report.write_text(json.dumps(dict(schema="splitfusion.cpu_gt_transfer_lifecycle_result.v4")))
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "fixed final-code handoff"):
                H.verify_handoff_proof(report, root)

    def test_handoff_wrapped_manifest_binds_final_replay_bootstrap_and_retained_body(self):
        with tempfile.TemporaryDirectory(prefix="v8-sealed-fixture-") as tmp:
            root = Path(tmp)
            attempt = root / "experiments/handoff/attempt"
            design_path = attempt / "full48_final/DESIGN.json"
            design_path.parent.mkdir(parents=True)
            bootstrap, worker = {"candidate.py": "a" * 64}, {"retained.py": "b" * 64}
            binding = "c" * 64
            report = attempt / "REPORT.json"
            report.write_text(json.dumps(dict(schema="splitfusion.gt_scoring_before_archive.result.v1",
                status="OFFLINE_STRUCTURAL_HANDOFF_EXACT_PASS__NOT_LIVE_QUALIFIED",
                exact_full_replays=192, unique_frames=48, total_tests=325, live_runs=0,
                cuda_initialized=False, actor_retrained=False, deadline_success_improvement_established=False,
                source_binding=binding)))
            design_path.write_text(json.dumps(dict(binding=binding, sources=dict(candidate=bootstrap, body=worker))))
            manifest = attempt / "OUTPUT_SHA256.json"
            manifest.write_text(json.dumps(dict(files={str(path.relative_to(root)): C.sha(path)
                                                       for path in (report, design_path)})))
            with mock.patch.object(H, "HANDOFF_MANIFEST_SHA256", C.sha(manifest)), \
                    mock.patch.object(C, "handoff_binding", return_value=binding), \
                    mock.patch.object(C, "bootstrap_inventory", return_value=bootstrap), \
                    mock.patch.object(C, "worker_inventory", return_value=worker):
                result = H.verify_handoff_proof(report, root)
                self.assertTrue(result["passed"])
                self.assertEqual(result["checked_files"], 2)
                with mock.patch.object(C, "bootstrap_inventory", return_value={}):
                    with self.assertRaisesRegex(ValueError, "executable sources"):
                        H.verify_handoff_proof(report, root)


if __name__ == "__main__":
    unittest.main()
