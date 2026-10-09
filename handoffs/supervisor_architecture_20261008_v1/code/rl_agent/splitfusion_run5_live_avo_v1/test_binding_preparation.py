"""Artifact packaging tests; source exports and frozen code remain read-only."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from . import actor_binding as A
from . import audit_actor as AU
from . import prepare_binding as P
from .test_actor_binding import EXPORT


class DependencyTests(unittest.TestCase):
    def test_sources_are_pinned_and_current(self):
        self.assertEqual(P.verify_sources(), P.FROZEN_DEPENDENCIES)
        self.assertTrue(all(r["matches"] for r in P.source_inventory().values()))

    def test_missing_dependencies_reported_without_import_or_mutation(self):
        with tempfile.TemporaryDirectory(prefix="run5_dependencies_") as temp:
            self.assertTrue(all(not r["matches"] for r in P.source_inventory(Path(temp)).values()))
            with self.assertRaises(A.Run5LiveBindingError):
                P.verify_sources(Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])


@unittest.skipUnless(EXPORT.is_dir(), "original Run-5 export required")
class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="run5_packaging_")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "actor_export"

    def test_create_only_cold_load_and_audit_repeatability(self):
        before = {n: A.F4.sha256_file(EXPORT / n) for n in P.SOURCE_FILES}
        P.prepare(EXPORT, self.output)
        first, second = AU.audit(self.output), AU.audit(self.output)
        self.assertEqual(first, second)
        self.assertEqual(first["verdict"], "PASS")
        self.assertTrue(all(first["checks"].values()))
        self.assertEqual(before, {n: A.F4.sha256_file(EXPORT / n) for n in P.SOURCE_FILES})
        with self.assertRaises(A.Run5LiveBindingError):
            P.prepare(EXPORT, self.output)

    def test_binding_profiler_keys_preserve_all_identities(self):
        P.prepare(EXPORT, self.output)
        _actor, binding = P.verify_packaged_export(self.output)
        self.assertEqual(binding["actor"]["weights_file_sha256"], A.WEIGHTS_SHA256)
        self.assertEqual(binding["actor"]["boundary_sha256"], A.ACTOR_TREE_SHA256)
        self.assertEqual(binding["binding"]["policy_feature_count"], 22)
        self.assertEqual(binding["binding"]["feature_schema_sha256"], A.FEATURE_SCHEMA_SHA256)

    def test_relocated_export_does_not_discover_original_source(self):
        P.prepare(EXPORT, self.output)
        target = Path(self.temp.name) / "relocated"
        shutil.copytree(self.output, target)
        self.assertEqual(AU.audit(target)["verdict"], "PASS")

    def test_resealed_foreign_binding_refused(self):
        P.prepare(EXPORT, self.output)
        path = self.output / "ACTOR_BINDING.json"
        binding = json.loads(path.read_text())
        binding["binding"]["policy_feature_count"] = 21
        binding["manifest_sha256"] = A._sha({k: v for k, v in binding.items() if k != "manifest_sha256"})
        path.write_text(json.dumps(binding))
        with self.assertRaises(A.Run5LiveBindingError):
            P.verify_packaged_export(self.output)

    def test_partial_export_and_symlink_weights_refused(self):
        P.prepare(EXPORT, self.output)
        weights = self.output / "actor_state_dict.pt"
        weights.unlink()
        with self.assertRaises(A.Run5LiveBindingError):
            P.verify_packaged_export(self.output)
        weights.symlink_to(EXPORT / "actor_state_dict.pt")
        with self.assertRaises(A.Run5LiveBindingError):
            P.verify_packaged_export(self.output)

    def test_invalid_source_never_creates_output(self):
        with self.assertRaises(A.Run5LiveBindingError):
            P.prepare(Path(self.temp.name) / "absent", self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
