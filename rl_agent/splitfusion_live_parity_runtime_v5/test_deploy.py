"""Temporary-directory staging tests; no runtime worktree or service is touched."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import contract as C, deploy as D
from .startup import write_x


class DeployTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory(prefix="parity-v5-deployment-") as temporary:
            root = Path(temporary) / "main"
            runtime = Path(temporary) / "runtime"
            root.mkdir(); runtime.mkdir()
            sealed = "rl_agent/splitfusion_live_gt_optimization_v1/runtime.py"
            write_x(runtime / sealed, b"sealed v1 bytes\n")
            previous = root / D.PREVIOUS_PROSPECTIVE
            record = {"schema": "splitfusion.parity_runtime.prospective.v4", "budget": 100,
                      "worktree": str(runtime), "runtime_sources": {sealed: C.sha(runtime / sealed)}}
            write_x(previous, record)
            write_x(root / C.PACKAGE / "__init__.py", b"new v5 package\n")
            write_x(root / C.OPTIMIZATION_PACKAGE / "__init__.py", b"new v2 package\n")
            write_x(root / C.OPTIMIZATION_PACKAGE / "candidate.py", b"new v2 candidate\n")
            # Old packages deliberately differ: v5 must never enumerate them.
            write_x(root / sealed, b"must not overwrite old optimization\n")
            write_x(root / C.REPAIR_PACKAGE / "__init__.py", b"must not stage old repair\n")
            for name in (C.CHILD, C.EDGE):
                leaf = name.rsplit(".", 1)[1] + ".py"
                write_x(root / C.PACKAGE / "runtime_shims" / leaf, name.encode() + b"\n")
            with mock.patch.object(D, "ROOT", root):
                yield root, runtime, previous, record, Path(temporary) / "REPORT.json"

    def test_stages_only_new_packages_and_shims_preserving_sealed_source(self):
        with self.fixture() as (root, runtime, previous, record, report):
            result = D.deploy(runtime, previous, report, execute=True)
            exact_shims = {name.replace(".", "/") + ".py" for name in (C.CHILD, C.EDGE)}
            self.assertTrue(result["executed"])
            for name, digest in result["files"].items():
                self.assertTrue(name.startswith(C.PACKAGE + "/")
                    or name.startswith(C.OPTIMIZATION_PACKAGE + "/") or name in exact_shims, name)
                self.assertEqual(C.sha(runtime / name), digest)
            for name, digest in record["runtime_sources"].items():
                self.assertNotIn(name, result["files"])
                self.assertEqual(C.sha(runtime / name), digest)
            self.assertFalse((runtime / C.REPAIR_PACKAGE).exists())

    def test_prior_source_drift_rejected_before_any_new_source_write(self):
        with self.fixture() as (root, runtime, previous, record, report):
            sealed = runtime / next(iter(record["runtime_sources"]))
            with sealed.open("ab") as stream:
                stream.write(b"foreign mutation\n")
            with self.assertRaisesRegex(ValueError, "prior source changed"):
                D.deploy(runtime, previous, report, execute=True)
            self.assertFalse((runtime / C.PACKAGE).exists())
            self.assertFalse(report.exists())

    def test_foreign_prior_record_or_existing_candidate_refused(self):
        with self.fixture() as (root, runtime, previous, record, report):
            other = root / "OTHER_PROSPECTIVE.json"
            write_x(other, record)
            with self.assertRaisesRegex(ValueError, "exact sealed v4"):
                D.deploy(runtime, other, report)
            write_x(runtime / C.PACKAGE / "__init__.py", b"foreign existing candidate\n")
            with self.assertRaisesRegex(ValueError, "existing foreign target"):
                D.deploy(runtime, previous, report, execute=True)
            self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
