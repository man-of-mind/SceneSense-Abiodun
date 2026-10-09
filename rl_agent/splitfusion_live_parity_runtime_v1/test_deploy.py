"""Isolated deployment planning; no runtime imports, external commands or CUDA."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import deploy as D


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parity-deploy-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.target = self.base / "source", self.base / "target"
        self.source.mkdir(); self.target.mkdir()
        (self.target / D.PROFILER_PACKAGE).mkdir(parents=True)
        self.pin = "registered/reference.py"
        content = b"FROZEN = True\n"
        self.put(self.source, self.pin, content)
        self.put(self.target, self.pin, content)
        digest = hashlib.sha256(content).hexdigest()
        for package in D.PACKAGES:
            self.put(self.source, package + "/__init__.py", b"")
        for relative, collection in D.PIN_DECLARATIONS:
            text = (f"SOURCES = {{'test': ({self.pin!r}, {digest!r})}}\n" if collection
                    else f"TEST_SOURCE = {self.pin!r}\nTEST_SHA256 = {digest!r}\n")
            self.put(self.source, relative, text.encode())
        for relative in D.HISTORICAL_SCRIPTS:
            self.put(self.source, relative, b"# frozen historical function source\n")
        for name in ("child_v1_parity.py", "edge_profiled_v1_parity.py"):
            self.put(self.source, D.RUNTIME_PACKAGE + "/runtime_shims/" + name,
                     b"# sampler-visible trampoline\n")
        for root in (self.source, self.target):
            self.put(root, D.OLD_RUNTIME + "/__init__.py", b"# unchanged old runtime\n")
            self.put(root, D.OLD_RUNTIME + "/actor_export/actor.pt", b"unchanged actor bytes")

    @staticmethod
    def put(root, relative, data):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_default_is_read_only_complete_sealed_plan(self):
        before = sorted(str(p.relative_to(self.target)) for p in self.target.rglob("*"))
        result = D.deploy(self.target, source_root=self.source)
        after = sorted(str(p.relative_to(self.target)) for p in self.target.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "PLANNED_NOT_STAGED")
        plan = result["plan"]
        self.assertGreater(plan["create_count"], 0)
        self.assertEqual(plan["keep_identical_count"], 0)
        self.assertEqual(len(plan["inventory_sha256"]), 64)
        self.assertIn(self.pin, plan["verified_existing"])
        self.assertTrue(set(D.HISTORICAL_SCRIPTS).issubset(plan["files"]))
        self.assertIn(D.PROFILER_PACKAGE + "/child_v1_parity.py", plan["files"])
        self.assertFalse(plan["old_runtime_modified"])
        self.assertFalse(plan["retained_partition_copied"])
        self.assertFalse(any(p.startswith(D.OLD_RUNTIME + "/") for p in plan["files"]))

    def test_additive_execution_is_repeatable_for_identical_existing_bytes(self):
        first = D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertGreater(len(first["created"]), 0)
        self.assertEqual(first["verification"]["create_count"], 0)
        second = D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertEqual(second["created"], [])
        self.assertEqual(second["plan"]["create_count"], 0)
        for relative, row in first["plan"]["files"].items():
            self.assertEqual((self.target / relative).read_bytes(),
                             (self.source / row["source_relative"]).read_bytes())

    def test_foreign_late_destination_refused_before_any_copy(self):
        late = D.PROFILER_PACKAGE + "/edge_profiled_v1_parity.py"
        self.put(self.target, late, b"foreign")
        with self.assertRaisesRegex(D.DeploymentError, "foreign bytes"):
            D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())
        self.assertEqual((self.target / late).read_bytes(), b"foreign")

    def test_existing_dependency_drift_refused_before_any_copy(self):
        self.put(self.target, self.pin, b"changed frozen dependency")
        with self.assertRaisesRegex(D.DeploymentError, "dependency differs"):
            D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())

    def test_foreign_extra_package_file_refused_before_any_copy(self):
        extra = D.PACKAGES[2] + "/unregistered_module.py"
        self.put(self.target, extra, b"foreign integration source")
        with self.assertRaisesRegex(D.DeploymentError, "foreign destination package inventory"):
            D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())
        self.assertEqual((self.target / extra).read_bytes(), b"foreign integration source")

    def test_old_actor_drift_refused_before_any_copy(self):
        self.put(self.target, D.OLD_RUNTIME + "/actor_export/actor.pt", b"foreign actor")
        with self.assertRaisesRegex(D.DeploymentError, "dependency differs"):
            D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())

    def test_source_pin_drift_refused(self):
        self.put(self.source, self.pin, b"changed source")
        with self.assertRaisesRegex(D.DeploymentError, "source pin changed"):
            D.deployment_plan(self.target, source_root=self.source)

    def test_symbolic_roots_and_parent_components_refused(self):
        alias = self.base / "alias"
        alias.symlink_to(self.target, target_is_directory=True)
        with self.assertRaisesRegex(D.DeploymentError, "symbolic"):
            D.deployment_plan(alias, source_root=self.source)
        source_alias = self.base / "source_alias"
        source_alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(D.DeploymentError, "symbolic"):
            D.deployment_plan(self.target, source_root=source_alias)
        with self.assertRaisesRegex(D.DeploymentError, "symbolic"):
            D.deployment_plan(alias / ".." / "target", source_root=self.source)

    def test_symbolic_source_and_destination_entries_refused(self):
        source_file = self.source / D.PACKAGES[0] / "bad.py"
        source_file.symlink_to(self.source / self.pin)
        with self.assertRaisesRegex(D.DeploymentError, "symbolic"):
            D.deployment_plan(self.target, source_root=self.source)
        source_file.unlink()
        (self.target / D.PACKAGES[0]).symlink_to(self.source / D.PACKAGES[0], target_is_directory=True)
        with self.assertRaisesRegex(D.DeploymentError, "symbolic"):
            D.deploy(self.target, source_root=self.source, plan_only=False)

    def test_missing_shim_and_missing_existing_dependency_refused(self):
        shim = self.source / D.RUNTIME_PACKAGE / "runtime_shims/child_v1_parity.py"
        shim.unlink()
        with self.assertRaisesRegex(D.DeploymentError, "sampler-visible"):
            D.deployment_plan(self.target, source_root=self.source)
        self.put(self.source, shim.relative_to(self.source), b"# repaired\n")
        (self.target / self.pin).unlink()
        with self.assertRaisesRegex(D.DeploymentError, "missing"):
            D.deployment_plan(self.target, source_root=self.source)

    def test_source_change_after_plan_refused_before_copy(self):
        original = D.deployment_plan
        calls = []
        def plan_then_change(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            if len(calls) == 1:
                self.put(self.source, D.PACKAGES[0] + "/__init__.py", b"changed after plan")
            return result
        with mock.patch.object(D, "deployment_plan", side_effect=plan_then_change):
            with self.assertRaisesRegex(D.DeploymentError, "source changed after planning"):
                D.deploy(self.target, source_root=self.source, plan_only=False)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())

    def fixture_files(self):
        paths = {"experiments/registered_test/a.csv": b"first fixture\n",
                 "experiments/registered_test/b.png": b"second fixture\n"}
        for path, data in paths.items():
            self.put(self.source, path, data)
        return {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in paths.items()}

    def test_cpu_fixture_copy_is_explicit_and_complete(self):
        fixtures = self.fixture_files()
        with mock.patch.object(D, "CPU_FIXTURE_FILES", fixtures):
            normal = D.deploy(self.target, source_root=self.source, plan_only=False)
            self.assertIsNone(normal["plan"]["test_fixtures"])
            self.assertTrue(all(not (self.target / name).exists() for name in fixtures))
            extra = D.deploy(self.target, source_root=self.source, plan_only=False, include_test_fixtures=True)
            self.assertEqual(set(extra["created"]), set(fixtures))
            self.assertEqual(extra["plan"]["test_fixtures"]["count"], 2)
            for name, (_size, digest) in fixtures.items():
                self.assertEqual(hashlib.sha256((self.target / name).read_bytes()).hexdigest(), digest)

    def test_all_optional_fixture_hashes_checked_before_any_write(self):
        fixtures = self.fixture_files()
        last = sorted(fixtures)[-1]
        self.put(self.source, last, b"X" * fixtures[last][0])
        with mock.patch.object(D, "CPU_FIXTURE_FILES", fixtures):
            with self.assertRaisesRegex(D.DeploymentError, "CPU fixture SHA-256 changed"):
                D.deploy(self.target, source_root=self.source, plan_only=False, include_test_fixtures=True)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())
        self.assertTrue(all(not (self.target / name).exists() for name in fixtures))

    def test_foreign_optional_fixture_destination_prevents_all_copies(self):
        fixtures = self.fixture_files()
        last = sorted(fixtures)[-1]
        self.put(self.target, last, b"foreign")
        with mock.patch.object(D, "CPU_FIXTURE_FILES", fixtures):
            with self.assertRaisesRegex(D.DeploymentError, "foreign bytes"):
                D.deploy(self.target, source_root=self.source, plan_only=False, include_test_fixtures=True)
        self.assertFalse((self.target / D.PACKAGES[0]).exists())
        self.assertFalse((self.target / sorted(fixtures)[0]).exists())

    def test_cpu_fixture_source_override_copies_regular_files_without_aliases(self):
        fixtures = self.fixture_files()
        first = sorted(fixtures)[0]
        canonical = "data_collection/registered_test/a.csv"
        self.put(self.source, canonical, (self.source / first).read_bytes())
        (self.source / first).unlink()
        (self.source / first).symlink_to(self.source / canonical)
        with mock.patch.object(D, "CPU_FIXTURE_FILES", fixtures), \
                mock.patch.object(D, "CPU_FIXTURE_SOURCE_OVERRIDES", {first: canonical}):
            staged = D.deploy(self.target, source_root=self.source, plan_only=False, include_test_fixtures=True)
        self.assertEqual(staged["plan"]["files"][first]["source_relative"], canonical)
        self.assertFalse((self.target / first).is_symlink())
        self.assertEqual((self.target / first).read_bytes(), (self.source / canonical).read_bytes())


if __name__ == "__main__":
    unittest.main()
