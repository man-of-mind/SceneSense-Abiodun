"""Saved V8 receipts and adversarial process-scope classification checks."""
import copy
import json
from pathlib import Path
import unittest

from . import process_scope as P

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = ROOT / "experiments/splitfusion_live_parity_runtime_v8/20261004_handoff_live_v8/qualification_100"


class ProcessScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def load(name):
            return json.loads((QUALIFICATION / name).read_bytes())
        cell = "capture/cells/a71__favorable_stable/"
        cls.original = dict(registration=load("registration.json"), record=load("PROSPECTIVE.json"),
            scope=load(cell + "GT_WORKER_SCOPE.json"), timeline=load(cell + "CPU_GT_PROCESS_TIMELINE.json"),
            qualification=load("capture/QUALIFICATION_RESULT.json"),
            rows=[json.loads(line) for line in (QUALIFICATION /
                "capture/threadcap/PROCESS_ENV_INVENTORY.jsonl").read_text().splitlines()])

    def fixture(self):
        return copy.deepcopy(self.original)

    def verify(self, fixture):
        record = fixture["record"]
        return P.verify_registered_children(fixture["registration"], fixture["scope"], fixture["timeline"], fixture["rows"],
            baseline={key: None for key in P.CAP_VARIABLES}, expected_worker_sources=record["worker_sources"],
            expected_bootstrap_sources=record["bootstrap_sources"], expected_binding=record["handoff_binding"])

    def classify(self, fixture):
        children = self.verify(fixture)
        return P.classify_inventory(fixture["rows"], baseline={key: None for key in P.CAP_VARIABLES}, children=children)

    def test_actual_saved_binding_two_native_children_and_original_failure_remain_unchanged(self):
        paths = [QUALIFICATION / name for name in ("PROSPECTIVE.json", "registration.json",
            "capture/QUALIFICATION_RESULT.json", "capture/threadcap/PROCESS_ENV_INVENTORY.jsonl")]
        before = {str(path): P.sha(path) for path in paths}
        result = P.verify_saved_scope(QUALIFICATION, source_root=ROOT)
        self.assertTrue(result["passed"])
        self.assertTrue(result["registered_children_verified"])
        self.assertEqual(result["children"]["requests"], 48)
        self.assertEqual(result["children"]["native_contexts_checked"], 50)
        self.assertEqual(result["children"]["gt_pid"], 2400654)
        self.assertEqual(result["children"]["archive_pid"], 2400659)
        self.assertFalse(result["original_e3"])
        self.assertFalse(result["original_gate_relabelled"])
        self.assertFalse(result["live_qualified"])
        self.assertFalse(result["deployed"])
        self.assertEqual(before, {str(path): P.sha(path) for path in paths})

    def test_every_unreadable_process_is_retained_and_prevents_complete_coverage_claim(self):
        fixture = self.fixture()
        value = self.classify(fixture)
        original = fixture["qualification"]["threadcap"]["inventory"]
        self.assertEqual(value["unreadable"], original["unreadable"])
        self.assertEqual(value["rows"], original["rows"])
        self.assertEqual(value["ue_processes"], original["ue_processes"])
        self.assertEqual(len(value["unreadable"]), 3)
        self.assertTrue(value["observed_other_processes_baseline"])
        self.assertFalse(value["all_processes_verified"])

    def test_gt_exec_snapshot_is_not_rewritten_to_its_later_effective_caps(self):
        fixture = self.fixture()
        before = copy.deepcopy(fixture["rows"])
        result = self.verify(fixture)
        self.assertIsNone(result["gt_exec_snapshot"]["OMP_NUM_THREADS"])
        self.assertIsNone(result["gt_exec_snapshot"]["MKL_NUM_THREADS"])
        self.assertEqual(result["effective_environment"], P.ENVIRONMENT)
        self.assertEqual(fixture["rows"], before)

    def test_same_command_foreign_child_with_inherited_caps_gets_no_exception(self):
        fixture = self.fixture()
        source = next(row for row in fixture["rows"] if row.get("pid") == 2400659)
        foreign = copy.deepcopy(source)
        foreign["pid"] += 12345
        fixture["rows"].insert(-1, foreign)
        value = self.classify(fixture)
        self.assertFalse(value["observed_other_processes_baseline"])
        self.assertEqual({row["pid"] for row in value["scope_violations"]}, {foreign["pid"]})
        self.assertEqual({row["variable"] for row in value["scope_violations"]}, set(P.CAP_VARIABLES))

    def test_existing_other_role_with_inherited_caps_still_fails_baseline(self):
        fixture = self.fixture()
        other = next(row for row in fixture["rows"] if row.get("role") == "edge")
        other["variables"]["OMP_NUM_THREADS"] = "1"
        result = self.classify(fixture)
        self.assertFalse(result["observed_other_processes_baseline"])
        self.assertEqual(result["scope_violations"], [dict(pid=other["pid"], role="edge", comm=other["comm"],
            variable="OMP_NUM_THREADS", value="1", baseline=None)])

    def test_missing_registration_source_binding_or_factory_never_grants_exception(self):
        for key in ("worker_sources", "bootstrap_sources", "handoff_binding", "worker_client_factory"):
            fixture = self.fixture()
            fixture["registration"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_foreign_timeline_factory_or_source_map_is_rejected(self):
        for key, value in (("factory", "foreign.Client"), ("bootstrap_sources", {}),
                ("sources", {}), ("handoff_binding", "0" * 64), ("complete", False)):
            fixture = self.fixture()
            fixture["timeline"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_ready_every_run_and_close_contexts_require_cpu_caps_cuda_and_sources(self):
        for location in ("ready", "first_run", "last_run", "close"):
            for key, value in (("environment", {}), ("torch_threads", 2),
                    ("cuda_initialized", True), ("bootstrap_sources", {}), ("pid", 999999)):
                fixture = self.fixture()
                client, requests = fixture["timeline"]["clients"][0], fixture["timeline"]["requests"]
                context = (client[location]["process_context"] if location in ("ready", "close")
                           else requests[0 if location == "first_run" else -1]["process_context"])
                context[key] = value
                with self.subTest(location=location, key=key), self.assertRaises(ValueError):
                    self.verify(fixture)

    def test_archive_context_parent_credentials_or_caps_cannot_be_missing(self):
        for key, value in (("parent_pid", 999), ("pid", 999), ("environment", {}),
                ("uid", [0] * 4), ("inspection", "UNBOUND")):
            fixture = self.fixture()
            fixture["timeline"]["clients"][0]["archive_startup"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_inventory_pid_parent_command_caps_and_unreadability_fail_closed(self):
        mutations = (("ppid", 999), ("cmdline", "/usr/bin/python3 -m foreign"),
                     ("variables", None), ("descendant_of_runner", False), ("in_container", True))
        for key, value in mutations:
            fixture = self.fixture()
            row = next(row for row in fixture["rows"] if row.get("pid") == 2400659)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(fixture)
        fixture = self.fixture()
        row = next(row for row in fixture["rows"] if row.get("pid") == 2400654)
        row["variables"]["OPENBLAS_NUM_THREADS"] = "8"
        with self.assertRaises(ValueError):
            self.verify(fixture)

    def test_missing_duplicate_or_failed_native_evidence_is_rejected(self):
        mutations = [lambda f: f["rows"].append(copy.deepcopy(next(r for r in f["rows"] if r.get("pid") == 2400654))),
            lambda f: f["timeline"]["requests"].pop(),
            lambda f: f["timeline"]["clients"][0].update(exitcode=1),
            lambda f: f["timeline"]["requests"][0]["worker"].update(pid=999),
            lambda f: f["scope"].update(worker_close_succeeded=False)]
        for mutate in mutations:
            fixture = self.fixture()
            mutate(fixture)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_inventory_change_after_native_verification_is_not_implicitly_approved(self):
        fixture = self.fixture()
        children = self.verify(fixture)
        fixture["rows"][0]["variables"]["OMP_NUM_THREADS"] = "1"
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            P.classify_inventory(fixture["rows"], baseline={key: None for key in P.CAP_VARIABLES}, children=children)


if __name__ == "__main__":
    unittest.main()
