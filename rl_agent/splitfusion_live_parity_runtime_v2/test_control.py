"""CPU-only proof gates and exact retained lifecycle substitution checks."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from . import contract as C, host_capture as H, partition_apply, runner
from rl_agent.splitfusion_live_parity_runtime_v1 import contract as V1


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parity-v2-control-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"sealed/source.py": "a" * 64}
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(C, "source_inventory", return_value=self.source).start()
        fixture = self.root / "fixture.bin"
        fixture.write_bytes(b"bounded original fixture")
        self.fixtures = {fixture.name: (fixture.stat().st_size, C.sha(fixture))}
        mock.patch.object(H, "CPU_FIXTURE_FILES", self.fixtures).start()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def offline(self):
        logs = {}
        for name in H.OFFLINE_SUITES:
            log = self.root / (name + ".log")
            log.write_text("one fresh CPU test passed\n")
            logs[log.name] = C.sha(log)
        result = dict(schema="splitfusion.parity_runtime.offline.v2", sources=self.source,
            passed=True, live_started=False, training_started=False, total_tests=len(H.OFFLINE_SUITES),
            fixtures={name: digest for name, (_, digest) in self.fixtures.items()}, logs=logs,
            suites={name: dict(returncode=0, tests=1, skipped=0, skips=[]) for name in H.OFFLINE_SUITES})
        return result

    def startup(self):
        log = self.root / "startup.log"
        log.write_text("pinned image real scopes and lazy scoring passed\n")
        return dict(schema="splitfusion.parity_runtime.edge_startup_scope.v2", sources=self.source,
            passed=True, live_started=False, cuda_initialized=False, image_id=C.IMAGE_ID,
            edge_module=C.EDGE, checks=dict.fromkeys(H.STARTUP_CHECKS, True), logs={log.name: C.sha(log)})

    def test_science_and_v1_identity_unchanged(self):
        self.assertIs(C.metrics, V1.metrics)
        self.assertEqual(C.metrics(), V1.metrics())
        self.assertEqual(C.BUDGET, 100)
        self.assertEqual(C.metrics()["inclusive_deadline_ms"], 170)
        self.assertEqual(C.metrics()["actor_state_dim"], 22)
        self.assertNotEqual(C.ID, V1.ID)
        self.assertNotEqual(C.TOKEN, V1.TOKEN)
        self.assertTrue(C.CHILD.endswith(".child_v1_run5_parity_v2"))
        self.assertTrue(C.EDGE.endswith(".edge_profiled_v1_run5_parity_v2"))

    def test_valid_fresh_offline_proof(self):
        path = self.write("OFFLINE.json", self.offline())
        self.assertEqual(H.verify_offline(path, self.root), dict(path=str(path), sha256=C.sha(path), passed=True))

    def test_only_exact_historical_skip_is_permitted(self):
        value = self.offline()
        value["suites"]["frozen_semantics"].update(skipped=1, skips=[dict(H.ALLOWED_SKIP)])
        path = self.write("OFFLINE.json", value)
        H.verify_offline(path, self.root)
        value["suites"]["frozen_semantics"]["skips"][0]["reason"] = "registered actor unavailable"
        self.write(path.name, value)
        with self.assertRaisesRegex(ValueError, "unapproved skip"):
            H.verify_offline(path, self.root)

    def test_missing_suite_failed_suite_zero_tests_and_new_skip_refused(self):
        original = self.offline()
        changes = [lambda value: value["suites"].pop("repair"),
            lambda value: value["suites"]["repair"].update(returncode=1),
            lambda value: value["suites"]["repair"].update(tests=0),
            lambda value: value["suites"]["repair"].update(skipped=1, skips=[dict(H.ALLOWED_SKIP)]),
            lambda value: value["suites"]["frozen_semantics"].update(skipped=2, skips=[dict(H.ALLOWED_SKIP)] * 2)]
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                H.verify_offline(self.write("OFFLINE.json", value), self.root)

    def test_old_schema_source_drift_and_live_flag_refused(self):
        for key, value in (("schema", "splitfusion.parity_runtime.offline.v1"), ("sources", {}),
                           ("live_started", True), ("training_started", True), ("total_tests", 0)):
            document = self.offline()
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                H.verify_offline(self.write("OFFLINE.json", document), self.root)

    def test_offline_log_and_fixture_drift_refused(self):
        path = self.write("OFFLINE.json", self.offline())
        (self.root / "repair.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "log drift"):
            H.verify_offline(path, self.root)
        path = self.write("OFFLINE.json", self.offline())
        (self.root / "fixture.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "fixture drift"):
            H.verify_offline(path, self.root)

    def test_valid_startup_scope(self):
        path = self.write("STARTUP.json", self.startup())
        result = H.verify_startup_scope(path, self.root)
        self.assertEqual(result["sha256"], C.sha(path))
        self.assertEqual(result["image_id"], C.IMAGE_ID)

    def test_startup_all_required_checks_fail_closed(self):
        original = self.startup()
        for check in H.STARTUP_CHECKS:
            value = copy.deepcopy(original)
            value["checks"][check] = False
            with self.subTest(check=check), self.assertRaises(ValueError):
                H.verify_startup_scope(self.write("STARTUP.json", value), self.root)

    def test_startup_wrong_image_command_sources_or_unsafe_flags_refused(self):
        for key, value in (("image_id", "mutable:latest"), ("edge_module", V1.EDGE), ("sources", {}),
                           ("cuda_initialized", True), ("live_started", True), ("checks", {})):
            document = self.startup()
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                H.verify_startup_scope(self.write("STARTUP.json", document), self.root)

    def test_missing_symlinked_proof_and_escaping_logs_refused(self):
        with self.assertRaises(ValueError):
            H.verify_startup_scope(self.root / "absent", self.root)
        path = self.write("STARTUP.json", self.startup())
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            H.verify_startup_scope(link, self.root)
        document = self.startup()
        document["logs"] = {"../escape.log": "a" * 64}
        with self.assertRaisesRegex(ValueError, "log drift"):
            H.verify_startup_scope(self.write("STARTUP.json", document), self.root)

    def test_prepare_requires_image_proof_before_output_creation(self):
        output = self.root / "new_attempt"
        with mock.patch.object(H, "verify_offline", return_value={}), \
                mock.patch.object(H, "verify_startup_scope", side_effect=ValueError("missing image proof")):
            with self.assertRaisesRegex(ValueError, "missing image proof"):
                H.prepare(output, Path(H.__file__).resolve().parents[2], self.root,
                          offline="offline", startup_scope="startup", sync_probe="sync")
        self.assertFalse(output.exists())

    def test_launch_has_only_two_exact_source_substitutions(self):
        expected = inspect.getsource(H.H.launch)
        for before, after in (("rl_agent.splitfusion_run5_live_avo_v1.runner", C.RUNNER),
                              ("rl_agent.splitfusion_run5_live_avo_v1.partition_apply", C.PARTITION)):
            self.assertEqual(expected.count(before), 1)
            expected = expected.replace(before, after)
        observed = {}
        def compile_source(source, filename, mode):
            observed["source"] = source
            return compile(source, filename, mode)
        def intercept(code, namespace):
            observed["namespace"] = namespace
            namespace["launch"] = lambda output, execute_token: (output, execute_token)
        with mock.patch.object(H, "compile", compile_source, create=True), mock.patch("builtins.exec", intercept):
            self.assertEqual(H.launch("attempt", execute_token="explicit"), ("attempt", "explicit"))
        self.assertEqual(observed["source"], expected)
        self.assertIs(observed["namespace"]["_verify_prospective"], H.verify_prospective)
        self.assertIs(observed["namespace"]["_verify_prerequisite"], H.verify_prerequisite)

    def test_postflight_private_globals_and_verifier_restored_on_failure(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as repaired
        from rl_agent.splitfusion_run5_live_avo_v1 import verify_live_gt_capture as V
        previous = H.H._verify_prospective
        original_verify = V.verify_capture
        namespace = {"__name__": "test_retained", "_verify_prospective": lambda _: self.fail("old verifier used"),
                     "V": V, "expected": repaired.verify_avo_capture}
        exec("def postflight(output):\n    assert V.verify_capture is expected\n    _verify_prospective(output)\n    raise RuntimeError('retained failure')\n", namespace)
        with mock.patch.object(H.H, "postflight", namespace["postflight"]), \
                mock.patch.object(H, "verify_prospective", return_value={}) as verify:
            with self.assertRaisesRegex(RuntimeError, "retained failure"):
                H.postflight("attempt")
            self.assertEqual(verify.call_count, 2)
        self.assertIs(V.verify_capture, original_verify)
        self.assertIs(H.H._verify_prospective, previous)
        self.assertEqual(V1.ID, "RUN5_COMPLETE_GT_PARITY_100_ENGINEERING_V1")

    def test_postflight_uses_retained_repair_without_global_contract_patch(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import postflight as repaired
        namespace = {}
        exec("def postflight(output):\n    return None\n", namespace)
        with mock.patch.object(H.H, "postflight", namespace["postflight"]), \
                mock.patch.object(H, "verify_prospective", return_value={}), \
                mock.patch.object(repaired, "postflight", return_value={"passed": False}) as post:
            self.assertEqual(H.postflight("attempt"), {"passed": False})
        post.assert_called_once_with("attempt")

    def test_partition_rebinds_matcher_without_mutating_old_module(self):
        from rl_agent.splitfusion_run5_live_avo_v1 import partition_apply as old
        namespace = {"EDGE_MODULE": old.EDGE_MODULE}
        exec("def apply(output, retained_code):\n    return EDGE_MODULE, output, retained_code\n", namespace)
        previous = old.EDGE_MODULE
        with mock.patch.object(old, "apply", namespace["apply"]):
            self.assertEqual(partition_apply.apply("attempt", "retained"), (C.EDGE, "attempt", "retained"))
        self.assertEqual(old.EDGE_MODULE, previous)
        self.assertEqual(namespace["EDGE_MODULE"], previous)

    def test_runner_requires_explicit_registration(self):
        for args in ([], ["--registration"], ["--registration", "a", "--registration", "b"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.main(args)

    def test_runner_binds_registered_original_actor_lineage_and_preflight(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as parent
        from rl_agent.splitfusion_run5_live_avo_v1 import audit_actor
        repository = Path(runner.__file__).resolve().parents[2]
        registration = self.write("registration.json", dict(id=C.ID, controller_lineage_sha256="newlineage"))
        actor = types.SimpleNamespace(boundary_sha256="original_actor_boundary")
        with mock.patch.object(H, "verify_prospective", return_value={"worktree": str(repository)}), \
                mock.patch.object(H, "verify_prerequisite", return_value={}) as proof, \
                mock.patch.object(audit_actor, "load_registered_actor", return_value=actor), \
                mock.patch.object(C, "lineage", return_value="newlineage") as lineage, \
                mock.patch.object(parent, "controller_lineage_sha256"), \
                mock.patch.object(parent, "offline_preflight", return_value=("config", ["cell"], {"legacy": True})):
            runner.install_parent(registration)
            self.assertEqual(parent.controller_lineage_sha256(), "newlineage")
            config, cells, preflight = parent.offline_preflight()
            self.assertEqual((config, cells), ("config", ["cell"]))
            self.assertEqual(preflight["actor_boundary_sha256"], actor.boundary_sha256)
            self.assertEqual(preflight["controller_lineage_sha256"], "newlineage")
            self.assertEqual(preflight["state_dim"], 22)
            self.assertFalse(preflight["live_command_executed"])
            self.assertFalse(preflight["cuda_initialized"])
            lineage.assert_called_once_with(repository, actor.boundary_sha256)
            proof.assert_called_once()

    def test_runner_refuses_wrong_repository_and_lineage_before_parent_mutation(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as parent
        from rl_agent.splitfusion_run5_live_avo_v1 import audit_actor
        registration = self.write("registration.json", dict(id=C.ID, controller_lineage_sha256="wrong"))
        prior = parent.offline_preflight, parent.controller_lineage_sha256
        for repository in (self.root, Path(runner.__file__).resolve().parents[2]):
            with self.subTest(repository=repository), \
                    mock.patch.object(H, "verify_prospective", return_value={"worktree": str(repository)}), \
                    mock.patch.object(H, "verify_prerequisite", return_value={}), \
                    mock.patch.object(audit_actor, "load_registered_actor", return_value=types.SimpleNamespace(boundary_sha256="original")), \
                    mock.patch.object(C, "lineage", return_value="right"):
                with self.assertRaises(ValueError):
                    runner.install_parent(registration)
                self.assertEqual((parent.offline_preflight, parent.controller_lineage_sha256), prior)


class InventoryTests(unittest.TestCase):
    def test_inventory_retains_v1_and_all_additive_files_and_shims(self):
        with tempfile.TemporaryDirectory(prefix="v2-source-inventory-") as temporary:
            root = Path(temporary)
            paths = [C.PACKAGE + "/__init__.py", C.PACKAGE + "/entries.py",
                     C.CHILD.replace(".", "/") + ".py", C.EDGE.replace(".", "/") + ".py"]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# additive\n")
            retained = {"rl_agent/splitfusion_live_parity_runtime_v1/entries.py": "a" * 64}
            with mock.patch.object(V1, "source_inventory", return_value=retained.copy()):
                result = C.source_inventory(root)
                self.assertEqual(result, {**retained, **{name: C.sha(root / name) for name in paths}})
                extra = root / C.PACKAGE / "new_test.py"
                extra.write_text("# bound too\n")
                self.assertIn(str(extra.relative_to(root)), C.source_inventory(root))
                extra.unlink()
                extra.symlink_to(root / paths[0])
                with self.assertRaisesRegex(ValueError, "nonregular"):
                    C.source_inventory(root)

    def test_lineage_binds_new_inventory_without_science_change(self):
        with mock.patch.object(C, "source_inventory", return_value={"new.py": "a" * 64}):
            actual = C.lineage("unused", "boundary")
        expected = C.canonical(dict(actor_boundary_sha256="boundary", sources={"new.py": "a" * 64},
            metrics=V1.metrics(), controller="RewardHoldControllerV2", engine="Run4DecisionEngineV2",
            state="ORIGINAL_RUN5_22D_QUALITY_AWARE", fallback="mode11_q9800_a71"))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
