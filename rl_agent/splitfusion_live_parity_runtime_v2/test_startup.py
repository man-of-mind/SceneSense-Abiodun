import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import startup


class StartupTests(unittest.TestCase):
    def test_exception_evidence_before_cleanup_and_same_exception(self):
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "edge_launcher.log").write_bytes(b"launcher bytes")
            expected = RuntimeError("preload failed")
            original = mock.Mock(side_effect=expected)
            with mock.patch.object(D, "start_direct_live_edge", original), \
                    mock.patch.object(D, "_ENDPOINT", {}), \
                    mock.patch.object(startup.subprocess, "run", return_value=SimpleNamespace(stdout=b"traceback", returncode=0)) as logs:
                with startup.preserve_startup(root / "attempt"):
                    with self.assertRaises(RuntimeError) as got:
                        D.start_direct_live_edge({}, {}, root)
                    self.assertIs(got.exception, expected)
                    target = root / "attempt/parity_startup_diagnostics"
                    self.assertEqual((target / "edge_container.log").read_bytes(), b"traceback")
                    self.assertEqual((target / "edge_launcher.log").read_bytes(), b"launcher bytes")
                    self.assertFalse(json.loads((target / "HOST_STARTUP_RESULT.json").read_text())["edge_ready"])
                    logs.assert_called_once()
                self.assertIs(D.start_direct_live_edge, original)

    def test_diagnostic_failure_never_masks_original(self):
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        expected = ValueError("real failure")
        with mock.patch.object(D, "start_direct_live_edge", side_effect=expected), \
                mock.patch.object(startup, "_preserve", side_effect=OSError("full disk")):
            with startup.preserve_startup("unused"):
                with self.assertRaises(ValueError) as got:
                    D.start_direct_live_edge({}, {}, "unused")
                self.assertIs(got.exception, expected)

    def test_success_keeps_return_and_no_docker_log_call(self):
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(D, "start_direct_live_edge", return_value="scratch"), \
                mock.patch.object(D, "_ENDPOINT", {}), mock.patch.object(startup.subprocess, "run") as logs:
            with startup.preserve_startup(temp):
                self.assertEqual(D.start_direct_live_edge({}, {}, temp), "scratch")
            logs.assert_not_called()
            record = json.loads((Path(temp) / "parity_startup_diagnostics/HOST_STARTUP_RESULT.json").read_text())
            self.assertTrue(record["edge_ready"])

    def test_edge_dispatch_matches_original_except_imports(self):
        root = Path(__file__).resolve().parents[2]
        def body(path, name):
            tree = ast.parse(path.read_text())
            method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            return [ast.dump(n) for n in method.body if not isinstance(n, (ast.Import, ast.ImportFrom))
                    and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        self.assertEqual(body(root / "rl_agent/splitfusion_live_parity_repair_v1/composition.py", "run_edge"),
                         body(Path(__file__).with_name("entries.py"), "run_edge_only"))


if __name__ == "__main__":
    unittest.main()
