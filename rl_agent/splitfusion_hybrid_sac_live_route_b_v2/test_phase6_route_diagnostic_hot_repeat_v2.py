"""Addendum-8 offline tests: preserved route failure, CARLA log, hot-repeat warm-up."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from . import phase6_live_child_v2 as C
from . import phase6_live_runner_v2 as RUN
from . import phase6_prewarm_v2 as PW
from .test_phase3_continuous_execution_v2 import CONTRACT, FakeAE, FakeCodec, runtimes
from .test_phase6_live_integration_v2 import processor
from .test_phase6_prewarm_gt_priority_v2 import _Runtime

PKG = Path(__file__).resolve().parent
DETAIL = {"route_runner_returncode": 2, "density_status": "", "route_completed": False,
          "route_abort_reason": "", "error": "RuntimeError: time-out of 10000ms while "
          "waiting for the simulator", "intervention_policy": {"interventions_permitted_and_expected": True},
          "route_summary_persisted": False, "route_summary_retention_reason": "SUMMARY_ABSENT",
          "route_summary_sha256": "", "route_summary_bytes": 0,
          "route_summary_identity_ok": False, "route_summary_identity_mismatches": [],
          "route_summary_parse_error": "", "route_outcome": {"classification": "RUNNER_FAILED"}}


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class RouteDetailTest(unittest.TestCase):
    def test_collector_none_still_retains_original_route_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts, tmp_root = root / "artifacts", root / "tmp"
            artifacts.mkdir()
            tmp_root.mkdir()
            old = tmp_root / "ue_route_b_metrics_old_population_events.jsonl"
            old.write_text("{}\n")
            os.utime(old, (1.0, 1.0))
            since = time.time() - 1
            new = tmp_root / "ue_route_b_metrics_new_population_events.jsonl"
            new.write_text('{"event": "spawn"}\n')
            result = {}
            C.record_route_detail(result, DETAIL, artifacts, since_unix_s=since,
                                  tmp_root=tmp_root)
            saved = json.loads((artifacts / C.ROUTE_DETAIL_NAME).read_text())
            self.assertEqual(saved, DETAIL)                          # complete, unfiltered
            self.assertEqual(result["route_detail"]["error"], DETAIL["error"])
            self.assertEqual(result["route_population_artifacts"], [new.name])
            self.assertEqual((artifacts / new.name).read_text(), new.read_text())
            with self.assertRaises(FileExistsError):                 # create-only
                C.record_route_detail({}, DETAIL, artifacts, since_unix_s=since,
                                      tmp_root=tmp_root)

    def test_run_records_detail_before_asserting_collector(self) -> None:
        run = _function(ast.parse((PKG / "phase6_live_child_v2.py").read_text()), "run")
        record_line = min(n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
                          and getattr(n.func, "id", "") == "record_route_detail")
        assert_line = min(n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
                          and getattr(n.func, "id", "") == "require"
                          and "collector is not None" in ast.unparse(n))
        route_line = min(n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
                         and ast.unparse(n.func).endswith("run_route_b"))
        self.assertLess(route_line, record_line)
        self.assertLess(record_line, assert_line)


class CarlaLogTest(unittest.TestCase):
    def test_log_preserved_create_only_or_reported_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, attempt = Path(tmp) / "service", Path(tmp) / "attempt"
            service.mkdir()
            attempt.mkdir()
            self.assertEqual(RUN.preserve_service_log(service, attempt)["reason"], "absent")
            (service / "carla_server.log").write_text("LogCarla: world loaded\n")
            record = RUN.preserve_service_log(service, attempt)
            self.assertTrue(record["preserved"])
            self.assertEqual((attempt / "carla_server.log").read_text(),
                             "LogCarla: world loaded\n")
            (service / "carla_server.log").write_text("changed\n")
            again = RUN.preserve_service_log(service, attempt)          # never overwrite
            self.assertFalse(again["preserved"])
            self.assertEqual((attempt / "carla_server.log").read_text(),
                             "LogCarla: world loaded\n")

    def test_runner_preserves_log_before_deleting_service(self) -> None:
        cell = _function(ast.parse((PKG / "phase6_live_runner_v2.py").read_text()),
                         "run_one_cell")
        preserve = min(n.lineno for n in ast.walk(cell) if isinstance(n, ast.Call)
                       and getattr(n.func, "id", "") == "preserve_service_log")
        rmtree = min(n.lineno for n in ast.walk(cell) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == "shutil.rmtree" and "service" in ast.unparse(n))
        self.assertLess(preserve, rmtree)
        finally_bodies = [n for n in ast.walk(cell) if isinstance(n, ast.Try) and n.finalbody]
        self.assertTrue(any("preserve_service_log" in ast.unparse(t.finalbody[i])
                            for t in finally_bodies for i in range(len(t.finalbody))))


class HotRepeatTest(unittest.TestCase):
    def _reports(self):
        ue, _edge, _ = runtimes()
        ue_report = PW.warm_ue(ue, CONTRACT, prepare_input=lambda f, r: object(),
                               sync=lambda: None)
        edge_report = PW.warm_edge(processor(), _Runtime(), CONTRACT, device="cpu",
                                   encoders={f: FakeAE(f) for f in ("AE128", "AE64", "AE32")},
                                   codec=FakeCodec(), unguarded_tail=object(), sync=lambda: None)
        return ue_report, edge_report

    def test_every_path_has_first_and_hot_timings_with_identical_payload(self) -> None:
        n = len(PW.warm_paths(CONTRACT))
        for report in self._reports():
            self.assertTrue(report["completed"])
            self.assertEqual(len(report["paths"]), n)
            for row in report["paths"]:
                self.assertIn("first_pass_ms", row)
                self.assertIn("hot_repeat_ms", row)
                self.assertTrue(row["hot_identical_payload"], row)
            summary = report["timing_summary"]
            for label in PW.PASSES:
                self.assertEqual(summary["overall"][label]["n"], n)
                for key in ("p50", "p95", "max"):
                    self.assertIsNotNone(summary["overall"][label][key])
            self.assertEqual(sorted(summary["per_mode"], key=int), [str(m) for m in range(12)])
            from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
                MODELED_SMOKE_SUPPORT,
            )
            self.assertEqual(summary["mode11_highest_q"]["q_e4"],
                             int(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[11][1]))

    def test_edge_hot_repeat_uses_isolated_increasing_identities(self) -> None:
        runtime = _Runtime()
        before = (runtime._counters, runtime._context_session, runtime._detached_tail)
        report = PW.warm_edge(processor(), runtime, CONTRACT, device="cpu",
                              encoders={f: FakeAE(f) for f in ("AE128", "AE64", "AE32")},
                              codec=FakeCodec(), unguarded_tail=object(), sync=lambda: None)
        self.assertEqual(before, (runtime._counters, runtime._context_session,
                                  runtime._detached_tail))
        for row in report["paths"]:
            self.assertNotEqual(row["first_pass_wire_sha256"], row["hot_repeat_wire_sha256"])

    def test_ready_requires_both_passes(self) -> None:
        written = []
        with self.assertRaises(PW.PrewarmError):
            PW.publish_ready_after_warmup(lambda: {"completed": False,
                                                   "modes_warmed": list(range(12))},
                                          lambda: written.append(1))
        self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()
