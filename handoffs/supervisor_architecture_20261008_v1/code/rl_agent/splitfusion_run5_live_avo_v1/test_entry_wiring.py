"""Process-entry composition checks, with all launches replaced by fakes."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import rl_agent

# MAIN owns the additive sources; the retained profiler is in its worktree.
# This is a test-only explicit namespace, never a production path fallback.
PROFILER_ROOT = Path(__file__).resolve().parents[2].parent / "abiodun_run4_single_gpu_profiler_v1"
if importlib.util.find_spec("rl_agent.splitfusion_single_gpu_profiler_v1") is None:
    rl_agent.__path__.append(str(PROFILER_ROOT / "rl_agent"))

from . import child_entry as CHILD
from . import edge_entry as EDGE
from . import runner as RUNNER


class ChildEntryTests(unittest.TestCase):
    def test_real_entry_installs_gt_before_phase6_then_policy_before_profiler(self):
        from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
        from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_threadcap as CAP
        from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_cpu_partition_v1 as PART
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_nobuild_v2 as NB
        from . import live_gt, runtime_policy

        events = []
        handle = SimpleNamespace(close=lambda: events.append("policy_close"))
        argv = ["--attempt-dir", "/tmp/test-run5-owned", "--transmitted-budget", "300"]
        original_argv = sys.argv

        def phase6(*args, **kwargs):
            events.append("phase6")
            return {"original_seams": True}

        def policy(**kwargs):
            self.assertEqual(kwargs["attempt_dir"], Path("/tmp/test-run5-owned"))
            self.assertEqual(kwargs["export_dir"].name, "actor_export")
            events.append("run5_policy")
            return handle

        def ordering(*args, **kwargs):
            events.append("ordering")
            self.assertEqual(CAP.ORDERING_EDGE_MODULE, "-m " + CHILD.EDGE_MODULE)

        def partition_main():
            self.assertEqual(sys.argv[1:], argv)
            # The real profiler child first calls the existing no-build hook,
            # then applies the profiler around the installed policy methods.
            got = NB.install_run4_seams_nobuild({}, attempt_dir=Path("/tmp/test-run5-owned"))
            self.assertEqual(got, {"original_seams": True})
            events.append("profiler")
            CAP.ORDERING_EDGE_MODULE = "retained_partition_default"
            CAP.install_ordering_route("C2", "serial", object())
            return 0

        with mock.patch.object(live_gt, "install_ue_gt_seams",
                               side_effect=lambda **kwargs: events.append("GT") or {}), \
             mock.patch.object(NB, "install_run4_seams_nobuild", phase6), \
             mock.patch.object(runtime_policy, "install_policy_seams", policy), \
             mock.patch.object(CAP, "install_ordering_route", ordering), \
             mock.patch.object(PART, "main", partition_main):
            self.assertEqual(CHILD.main(argv), 0)
            self.assertIs(NB.install_run4_seams_nobuild, phase6)
            self.assertIs(CAP.install_ordering_route, ordering)
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(events, ["GT", "phase6", "run5_policy", "profiler", "ordering", "policy_close"])

    def test_exception_closes_policy_and_restores_argv(self):
        from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_cpu_partition_v1 as PART
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_nobuild_v2 as NB
        from . import live_gt, runtime_policy
        handle = mock.Mock()
        before = sys.argv
        def fail():
            NB.install_run4_seams_nobuild({}, attempt_dir=Path("/tmp/test-run5-owned"))
            raise RuntimeError("synthetic startup failure")
        with mock.patch.object(live_gt, "install_ue_gt_seams", return_value={}), \
             mock.patch.object(NB, "install_run4_seams_nobuild", return_value={}), \
             mock.patch.object(runtime_policy, "install_policy_seams", return_value=handle), \
             mock.patch.object(PART, "main", fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic startup"):
                CHILD.main(["--attempt-dir", "/tmp/test-run5-owned"])
        handle.close.assert_called_once()
        self.assertIs(before, sys.argv)

    def test_missing_attempt_refuses_before_gt_install(self):
        from . import live_gt
        with mock.patch.object(live_gt, "install_ue_gt_seams") as install:
            with self.assertRaises(ValueError):
                CHILD.main([])
        install.assert_not_called()


class ParentEntryTests(unittest.TestCase):
    def test_legacy_infrastructure_checks_remain_then_new_binding_is_reported(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
        from . import audit_actor, quality_adapter
        old = R.offline_preflight
        old_lineage = R.controller_lineage_sha256
        calls = []
        def infrastructure(*args, **kwargs):
            calls.append((args, kwargs, R.controller_lineage_sha256()))
            return {"config": 1}, [{"cell": 1}], {"actor_boundary_sha256": "run4", "radio": "verified"}
        try:
            with mock.patch.object(R, "offline_preflight", infrastructure), \
                 mock.patch.object(RUNNER, "lineage", return_value="new-controller-lineage"), \
                 mock.patch.object(audit_actor, "load_registered_actor",
                                   return_value=SimpleNamespace(boundary_sha256="run5")), \
                 mock.patch.object(quality_adapter, "live_metrics_contract", return_value={"avo": .65}):
                RUNNER.install_parent()
                config, cells, result = R.offline_preflight(Path("config"), output_root=None,
                                                          transmitted_budget=300)
            self.assertEqual(calls, [((Path("config"),), {"output_root": None, "transmitted_budget": 300},
                                     "new-controller-lineage")])
            self.assertEqual(config, {"config": 1})
            self.assertEqual(cells, [{"cell": 1}])
            self.assertEqual(result["actor_boundary_sha256"], "run5")
            self.assertEqual(result["legacy_infrastructure_preflight"]["radio"], "verified")
            self.assertEqual(result["state_dim"], 22)
            self.assertEqual(result["controller_lineage_sha256"], "new-controller-lineage")
        finally:
            R.offline_preflight, R.controller_lineage_sha256 = old, old_lineage

    def test_old_preflight_failure_is_not_suppressed(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
        from . import audit_actor, quality_adapter
        old_lineage = R.controller_lineage_sha256
        try:
            with mock.patch.object(R, "offline_preflight", side_effect=ValueError("radio drift")), \
                 mock.patch.object(RUNNER, "lineage", return_value="new"), \
                 mock.patch.object(audit_actor, "load_registered_actor") as actor:
                RUNNER.install_parent()
                with self.assertRaisesRegex(ValueError, "radio drift"):
                    R.offline_preflight(Path("config"), output_root=None, transmitted_budget=300)
                actor.assert_not_called()
        finally:
            R.controller_lineage_sha256 = old_lineage


class EdgeEntryTests(unittest.TestCase):
    def test_quality_and_partition_gate_precede_profiled_serial_entry(self):
        from . import edge_quality
        from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDER
        gate = ModuleType("edge_gate")
        events = []
        gate.install = lambda *args: events.append(("partition", args[0]))
        before = list(sys.path)
        try:
            with mock.patch.dict(sys.modules, {"edge_gate": gate}), \
                 mock.patch.object(edge_quality, "install_edge_quality_seams",
                                   side_effect=lambda: events.append(("quality",))), \
                 mock.patch.object(ORDER, "main", side_effect=lambda: events.append(("serial",)) or 0):
                self.assertEqual(EDGE.main(), 0)
            self.assertEqual(events, [("quality",), ("partition", "/partition_output"), ("serial",)])
        finally:
            sys.path[:] = before


if __name__ == "__main__":
    unittest.main()
