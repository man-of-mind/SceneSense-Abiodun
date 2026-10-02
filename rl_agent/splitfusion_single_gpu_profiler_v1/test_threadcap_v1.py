"""Focused tests for the UE thread-cap experiment seam (offline, CPU).

Tests that patch ``subprocess.Popen`` or set the cap variables run in fresh
interpreters so they cannot leak into each other or into the rest of the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import runner_v1 as RUN
from . import threadcap_v1 as TC

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
EXPERIMENT_PATH = PACKAGE / "experiment_threadcap_ab_v1a.json"


def fresh_python(script: str, *, env: dict | None = None, timeout: float = 300.0) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", *TC.CAP_VARIABLES)}
    base.update(env or {})
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=str(ROOT),
                               env=base, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    return completed.stdout.strip().splitlines()[-1]


class ArmEnvironmentTest(unittest.TestCase):
    def test_arm_a_adds_only_markers_and_keeps_baseline(self) -> None:
        base = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
        env = TC.arm_environment(base, "A")
        self.assertEqual({k: v for k, v in env.items() if k not in TC.MARKER_VARIABLES}, base)
        self.assertEqual(env[TC.ARM_ENV], "A")
        self.assertEqual(json.loads(env[TC.RESTORE_ENV]),
                         {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})

    def test_arm_b_adds_exactly_the_two_cap_variables(self) -> None:
        base = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
        env = TC.arm_environment(base, "B")
        added = {k: v for k, v in env.items() if k not in base and k not in TC.MARKER_VARIABLES}
        self.assertEqual(added, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        with self.assertRaises(TC.ThreadCapError):
            TC.arm_environment(base, "C")

    def test_lifecycle_proxy_changes_only_the_ue_child_env(self) -> None:
        module = SimpleNamespace(child_env=lambda: {"PATH": "/bin", "PYTHONPATH_REMOVED": "1"},
                                 start_carla=lambda *a: "carla", wait_for_rpc=lambda *a: 1)
        proxy = TC.LifecycleArmProxy(module, "B")
        self.assertEqual(proxy.child_env()["OMP_NUM_THREADS"], "1")
        self.assertIs(proxy.start_carla, module.start_carla)
        self.assertNotIn("OMP_NUM_THREADS", module.child_env())      # CARLA path untouched
        supervisor = SimpleNamespace(import_lifecycle_helper=lambda config: module)
        TC.install_lifecycle_arm(supervisor, "A")
        TC.install_lifecycle_arm(supervisor, "B")                      # idempotent re-wrap
        self.assertIs(supervisor.import_lifecycle_helper.__wrapped__(None), module)
        self.assertEqual(supervisor.import_lifecycle_helper({}).child_env()["MKL_NUM_THREADS"], "1")

    def test_restored_environment_is_pure(self) -> None:
        restore = {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": "4"}
        env, changed = TC.restored_environment(
            {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", TC.ARM_ENV: "B", "X": "y"}, restore)
        self.assertEqual(env, {"MKL_NUM_THREADS": "4", "X": "y"})
        self.assertEqual(changed, 2)
        env, changed = TC.restored_environment({"X": "y", "MKL_NUM_THREADS": "4"}, restore)
        self.assertEqual((env, changed), ({"X": "y", "MKL_NUM_THREADS": "4"}, 0))


class ChildProcessScopingTest(unittest.TestCase):
    def test_every_launch_from_the_capped_child_gets_baseline_values(self) -> None:
        line = fresh_python("""
            import json, os, subprocess, sys
            from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
            os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            os.environ[TC.ARM_ENV] = "B"
            TC.install_child_environment_restore({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
            probe = [sys.executable, "-c",
                     "import os,json;print(json.dumps([os.environ.get('OMP_NUM_THREADS'),"
                     "os.environ.get('MKL_NUM_THREADS'),os.environ.get('%s')]))" % TC.ARM_ENV]
            seen = []
            seen.append(subprocess.run(probe, capture_output=True, text=True).stdout.strip())
            seen.append(subprocess.run(probe, capture_output=True, text=True,
                                       env={**os.environ, "OMP_NUM_THREADS": "1"}).stdout.strip())
            p = subprocess.Popen(probe, -1, None, None, subprocess.PIPE, None, None, True, False,
                                 None, dict(os.environ))          # env passed positionally
            seen.append(p.communicate()[0].decode().strip())
            print(json.dumps({"seen": seen, "own": os.environ["OMP_NUM_THREADS"],
                              "counters": TC.restore_counters()}))
        """)
        out = json.loads(line)
        self.assertEqual(out["seen"], ['[null, null, null]'] * 3)
        self.assertEqual(out["own"], "1")                      # the child itself keeps the cap
        self.assertEqual(out["counters"]["launches"], 3)
        self.assertGreaterEqual(out["counters"]["variables_restored"], 6)

    def test_child_arm_validation(self) -> None:
        restore = json.dumps({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
        script = """
            import json
            from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
            try:
                arm, restore = TC.child_arm(); print(json.dumps(["ok", arm, restore]))
            except TC.ThreadCapError as exc:
                print(json.dumps(["refused", str(exc)[:60]]))
        """
        ok_b = json.loads(fresh_python(script, env={TC.ARM_ENV: "B", TC.RESTORE_ENV: restore,
                                                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))
        self.assertEqual(ok_b[:2], ["ok", "B"])
        missing_cap = json.loads(fresh_python(script, env={TC.ARM_ENV: "B", TC.RESTORE_ENV: restore}))
        self.assertEqual(missing_cap[0], "refused")
        leaked_into_a = json.loads(fresh_python(script, env={TC.ARM_ENV: "A", TC.RESTORE_ENV: restore,
                                                             "OMP_NUM_THREADS": "1"}))
        self.assertEqual(leaked_into_a[0], "refused")
        ok_a = json.loads(fresh_python(script, env={TC.ARM_ENV: "A", TC.RESTORE_ENV: restore}))
        self.assertEqual(ok_a[:2], ["ok", "A"])

    def test_effective_settings_reflect_the_cap(self) -> None:
        script = """
            import json, torch, numpy, cv2
            from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
            s = TC.settings_snapshot("post", arm=None)
            print(json.dumps({"intra": s["torch"]["intra_op_threads"], "inter": s["torch"]["inter_op_threads"],
                              "omp": s["openmp_max_threads"], "cap": s["cap_effective"],
                              "pools": [(p["internal_api"], p["num_threads"]) for p in s["threadpools"]
                                        if isinstance(p, dict)],
                              "cv2": s["cv2"]["threads"], "libs": len(s["numerical_libraries_loaded"])}))
        """
        capped = json.loads(fresh_python(script, env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                                      "CUDA_VISIBLE_DEVICES": ""}))
        baseline = json.loads(fresh_python(script, env={"CUDA_VISIBLE_DEVICES": ""}))
        self.assertEqual(capped["intra"], 1)
        self.assertTrue(capped["cap"])
        self.assertTrue(all(v == 1 for _api, v in capped["omp"].items() if isinstance(v, int)))
        self.assertGreater(baseline["intra"], 1)
        self.assertFalse(baseline["cap"])
        self.assertEqual(capped["inter"], baseline["inter"])          # inter-op untouched
        self.assertEqual(capped["cv2"], baseline["cv2"])              # OpenCV (TBB) untouched
        self.assertGreater(capped["libs"], 0)

    def test_pre_import_snapshot_uses_only_the_standard_library(self) -> None:
        line = fresh_python("""
            import json, sys
            from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
            s = TC.settings_snapshot("pre", arm="A")
            print(json.dumps({"torch": "torch" in sys.modules, "numpy": "numpy" in sys.modules,
                              "cap": s["cap_effective"], "present": s["modules_present"]}))
        """)
        out = json.loads(line)
        self.assertFalse(out["torch"])
        self.assertFalse(out["numpy"])
        self.assertIsNone(out["cap"])

    def test_threadcap_child_delegates_with_identical_argv_and_records_both_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "cells" / "c"
            attempt.mkdir(parents=True)
            restore = json.dumps({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
            line = fresh_python(f"""
                import json, os, sys
                from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as CH
                from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1_threadcap as TCC
                seen = []
                CH.main = lambda args: seen.append(list(args)) or 7
                argv = ["--attempt-dir", {str(attempt)!r}, "--carla-port", "2000"]
                rc = TCC.main(argv)
                print(json.dumps({{"rc": rc, "seen": seen, "omp": os.environ.get("OMP_NUM_THREADS")}}))
            """, env={TC.ARM_ENV: "B", TC.RESTORE_ENV: restore, "OMP_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1"})
            out = json.loads(line)
            self.assertEqual(out["rc"], 7)                                   # return code unchanged
            self.assertEqual(out["seen"], [["--attempt-dir", str(attempt), "--carla-port", "2000"]])
            pre = json.loads((attempt / "threadcap" / "UE_THREAD_SETTINGS_PRE.json").read_text())
            post = json.loads((attempt / "threadcap" / "UE_THREAD_SETTINGS_POST.json").read_text())
            self.assertEqual((pre["stage"], post["stage"], pre["arm"]), ("pre", "post", "B"))
            self.assertEqual(pre["environment"]["OMP_NUM_THREADS"], "1")
            self.assertTrue(post["launch_restore"]["installed"])


class InventoryTest(unittest.TestCase):
    def test_inventory_records_descendants_and_flags_scope_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory = TC.ProcessEnvironmentInventory(
                Path(tmp) / "inv.jsonl", root_pid=os.getpid(),
                baseline={"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None},
                classify=lambda comm, cmdline: "ue" if "THREADCAP_UE_MARKER" in cmdline else None,
                interval_s=0.2)
            env_leak = {**os.environ, "OMP_NUM_THREADS": "1"}
            env_leak.pop("PYTHONPATH", None)
            leaked = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], env=env_leak)
            clean = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
            ue = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5) # THREADCAP_UE_MARKER"],
                                  env={**env_leak, "MKL_NUM_THREADS": "1"})
            try:
                inventory.start()
                time.sleep(1.0)
                summary = inventory.stop()
            finally:
                for process in (leaked, clean, ue):
                    process.kill()
                    process.wait()
            pids = {row["pid"] for row in inventory._rows}
            self.assertTrue({leaked.pid, clean.pid, ue.pid} <= pids)
            self.assertEqual([v["pid"] for v in summary["scope_violations"]], [leaked.pid])
            self.assertEqual(summary["ue_processes"][0]["variables"],
                             {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertFalse(summary["scope_ok"])
            rows = [json.loads(l) for l in (Path(tmp) / "inv.jsonl").read_text().splitlines()]
            self.assertIn("summary", rows[-1])

    def test_read_environ_of_own_process(self) -> None:
        environ, source = TC.read_environ(os.getpid(), sudo=False)
        self.assertEqual(source, "direct")
        self.assertIn("PATH", environ)


class RunnerExperimentTest(unittest.TestCase):
    def _snapshot(self, cell: Path, stage: str, arm: str, *, cap: bool | None, env: dict) -> None:
        (cell / TC.EVIDENCE_LEAF).mkdir(parents=True, exist_ok=True)
        document = {"stage": stage, "arm": arm, "environment": env, "cap_effective": cap,
                    "torch": {"intra_op_threads": 1 if cap else 24, "inter_op_threads": 24},
                    "openmp_max_threads": {"libgomp.so.1": 1 if cap else 24}}
        (cell / TC.EVIDENCE_LEAF / TC.UE_SETTINGS_NAME.format(stage=stage.upper())).write_text(
            json.dumps(document))

    def test_experiment_checks_per_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            env_b = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            self._snapshot(cell, "pre", "B", cap=None, env=env_b)
            self._snapshot(cell, "post", "B", cap=True, env=env_b)
            ok = RUN.experiment_checks(cell=cell, arm="B", inventory={"scope_ok": True})
            self.assertTrue(ok["all_passed"], ok["checks"])
            bad = RUN.experiment_checks(cell=cell, arm="B", inventory={"scope_ok": False})
            self.assertFalse(bad["checks"]["E3_SCOPE_OTHER_PROCESSES_BASELINE"])
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            env_a = {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None}
            self._snapshot(cell, "pre", "A", cap=None, env=env_a)
            self._snapshot(cell, "post", "A", cap=False, env=env_a)
            ok = RUN.experiment_checks(cell=cell, arm="A", inventory={"scope_ok": True})
            self.assertTrue(ok["all_passed"], ok["checks"])
            self._snapshot(Path(tmp) / "x", "pre", "A", cap=None, env={"OMP_NUM_THREADS": "1",
                                                                       "MKL_NUM_THREADS": None})
            self._snapshot(Path(tmp) / "x", "post", "A", cap=False, env={"OMP_NUM_THREADS": "1",
                                                                         "MKL_NUM_THREADS": None})
            leaked = RUN.experiment_checks(cell=Path(tmp) / "x", arm="A", inventory={"scope_ok": True})
            self.assertFalse(leaked["checks"]["E2_UE_CAP_STATE_AS_REGISTERED"])

    def test_select_arm_without_arms_is_the_unchanged_capture_path(self) -> None:
        capture = RUN.load_qualification(RUN.CAPTURE_PATH)
        self.assertEqual(RUN.select_arm(capture, None)["execute_token"], capture["execute_token"])
        with self.assertRaises(RUN.ProfilerRunnerError):
            RUN.select_arm(capture, "A")

    @unittest.skipUnless(EXPERIMENT_PATH.is_file(), "experiment registration not sealed yet")
    def test_experiment_registration_is_sealed_and_bound(self) -> None:
        registration = RUN.load_qualification(EXPERIMENT_PATH)
        self.assertEqual(registration["kind"], "PROFILING_CAPTURE")
        arms = registration["arms"]
        self.assertEqual(set(arms), {"A", "B"})
        self.assertEqual(arms["A"]["env"], {})
        self.assertEqual(arms["B"]["env"], TC.TREATMENT)
        self.assertNotEqual(arms["A"]["execute_token"], arms["B"]["execute_token"])
        __import__("importlib").import_module(registration["child_module"])
        self.assertEqual(registration["policy"], RUN.load_qualification(RUN.CAPTURE_PATH)["policy"])
        self.assertEqual(registration["lifecycle"], RUN.load_qualification(RUN.CAPTURE_PATH)["lifecycle"])
        RUN.verify_capture_binding(registration)
        bound = RUN.verify_experiment_binding(registration)
        self.assertIn("offline_parity_result", bound)
        tampered = json.loads(EXPERIMENT_PATH.read_text())
        tampered["arms"]["B"]["env"]["OMP_WAIT_POLICY"] = "PASSIVE"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(tampered))
            with self.assertRaises(RUN.ProfilerRunnerError):
                RUN.load_qualification(path)
        for arm in ("A", "B"):
            selected = RUN.select_arm(registration, arm)
            self.assertEqual(selected["env"], TC.ARMS[arm])


class ComparatorTest(unittest.TestCase):
    def test_comparator_runs_on_two_existing_captures(self) -> None:
        from . import compare_threadcap_ab_v1 as CMP

        base = ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
        a = base / "20261001T230025Z_run4_instrumentation_qualification_300"
        b = base / "20261001T231343Z_run4_profiling_capture_1000"
        if not ((a / "export_v1").is_dir() and (b / "export_v1").is_dir()):
            self.skipTest("existing exports are off Git")
        report = CMP.compare(a, b)
        for arm in ("A", "B"):
            self.assertIn("timely_feedback_le_170ms", report[arm]["primary"])
            self.assertGreater(report[arm]["host"]["samples"], 0)
            # edge container rows depend on a first-sight classification race: reported, not assumed
            self.assertIsInstance(report[arm]["host"]["edge_rows_in_procs_csv"], int)
        self.assertEqual(report["B"]["primary"]["policy_decisions"], 420)
        markdown = CMP.render_markdown(report["A"], report["B"])
        self.assertIn("Feedback <= 170 ms / all decisions", markdown)


if __name__ == "__main__":
    unittest.main()


class FreshnessAnalysisTest(unittest.TestCase):
    def test_reason_classes(self) -> None:
        from . import freshness_fallback_analysis_v1 as FA

        rc = FA.Capture.reason_class
        self.assertEqual(rc(["camera_si is stale (119035738 ns)", "use external fallback"]), "SCENE_STALE")
        self.assertEqual(rc(["UE_PRIOR_NEW_DATA_UL_MCS_INDEX:STALE:100295329"]), "MCS_STALE")
        self.assertEqual(rc(["CLOCK_BRIDGE_INVALID_OR_WARMING", "UE_PRIOR_NEW_DATA_UL_MCS_INDEX:MISSING:X"]),
                         "STARTUP_CLOCK_BRIDGE_WARMING")
        self.assertEqual(rc(["UE_PRE_ACTION_RLC_BACKLOG_BYTES:MISSING:X"]), "RADIO_MISSING_STEADY_STATE")

    def test_radio_stale_classification_uses_only_rows_received_before_the_boundary(self) -> None:
        from . import freshness_fallback_analysis_v1 as FA

        boundary = 1_000_000_000
        late = {"receipt_raw": boundary + 5_000_000, "event_raw": boundary - 20_000_000, "direction": "1", "round": "0"}
        old = {"receipt_raw": boundary - 150_000_000, "event_raw": boundary - 151_000_000, "direction": "1", "round": "0"}
        out = FA.classify_radio_stale(boundary, 151_000_000, [old, late], FA.dci_eligible, "prior_ul_mcs")
        self.assertEqual(out["cause"], "SOURCE_TO_READER_DELAY")
        self.assertEqual(out["eligible_events_in_window_received_after_boundary"], 1)
        retx = dict(old, receipt_raw=boundary - 10_000_000, event_raw=boundary - 11_000_000, round="1")
        out = FA.classify_radio_stale(boundary, 151_000_000, [old, retx], FA.dci_eligible, "prior_ul_mcs")
        self.assertEqual(out["cause"], "NO_ELIGIBLE_SOURCE_EVENT__ONLY_INELIGIBLE_IN_WINDOW")

    def test_runs_on_the_frozen_capture(self) -> None:
        from . import freshness_fallback_analysis_v1 as FA

        root = (ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                "20261001T231343Z_run4_profiling_capture_1000")
        if not root.is_dir():
            self.skipTest("frozen capture is off Git")
        result = FA.Capture(root).analyze()
        self.assertEqual(result["fallback_classes"], {"STARTUP_CLOCK_BRIDGE_WARMING": 2,
                                                      "SCENE_STALE": 107, "MCS_STALE": 13})
        self.assertEqual(result["radio_stale"]["stream_agrees_with_provider"], 13)
        self.assertEqual(result["identity"]["rgb_receipt_equals_collector_trace"],
                         result["identity"]["frames_checked"])
