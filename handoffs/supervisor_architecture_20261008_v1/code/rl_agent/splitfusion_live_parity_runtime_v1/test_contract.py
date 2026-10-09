import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import contract as C, host_capture as H
from .entries import write_x, edge_close_evidence


class Contracts(unittest.TestCase):
    def test_science(self):
        value = C.metrics()
        self.assertEqual(value["inclusive_deadline_ms"], 170)
        self.assertTrue(value["previous_quality_in_state"])
        self.assertTrue(value["quality_evaluation_inside_deadline"])
        self.assertEqual(value["actor_state_dim"], 22)
        self.assertFalse(value["historical_manual_quarantine_reused"])
        self.assertFalse(value["live_qualification_claim"])
        self.assertEqual(json.loads(json.dumps(value)), value)

    def test_sampler_names(self):
        self.assertIn(".child_v1", C.CHILD)
        self.assertIn(".edge_profiled_v1", C.EDGE)
        self.assertEqual(C.BUDGET, 100)

    def test_launch_preserves_lifecycle(self):
        # Exact function globals seam is auditable; not a rewritten launcher.
        source = inspect.getsource(H.H.launch)
        self.assertEqual(source.count("rl_agent.splitfusion_run5_live_avo_v1.runner"), 1)
        self.assertEqual(source.count("rl_agent.splitfusion_run5_live_avo_v1.partition_apply"), 1)
        seen = {}
        def trap(code, namespace):
            seen["code"] = code
            seen["namespace"] = namespace
            namespace["launch"] = lambda output, execute_token: (output, execute_token)
        with patch("builtins.exec", trap):
            self.assertEqual(H.launch("root", execute_token="explicit"), ("root", "explicit"))
        self.assertIs(seen["namespace"]["_verify_prospective"], H.verify_prospective)
        constants = seen["code"].co_consts
        function = next(v for v in constants if inspect.iscode(v))
        self.assertIn(C.RUNNER, function.co_consts)
        self.assertIn(C.PARTITION, function.co_consts)

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            write_x(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_x(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_edge_close_records_actual_result_and_restores(self):
        from contextlib import contextmanager
        from rl_agent.splitfusion_live_parity_repair_v1 import runtime
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
        actual = {"worker_alive": False, "parity_archive_complete": True, "parity_archive_errors": []}
        @contextmanager
        def installed(**kwargs):
            yield {}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runtime, "install_edge_evaluation", installed), \
                patch.object(E.Run4EvaluatorV2, "close", lambda self, timeout_s=None: actual):
            old = E.Run4EvaluatorV2.close
            with edge_close_evidence(Path(directory)), runtime.install_edge_evaluation():
                observed = E.Run4EvaluatorV2.close(object())
                self.assertIs(observed, actual)
            self.assertIs(E.Run4EvaluatorV2.close, old)
            self.assertEqual(json.loads((Path(directory)/"evaluator_close_result.json").read_text()), actual)


if __name__ == "__main__":
    unittest.main()
