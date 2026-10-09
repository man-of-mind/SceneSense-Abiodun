"""Prepare/guard tests using a temporary runtime, never launching a process."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import host_capture as H
from . import prepare_binding as P


class HostCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="run5_host_config_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.worktree = self.root / "runtime"
        self.package = self.worktree / "rl_agent/splitfusion_run5_live_avo_v1"
        shutil.copytree(Path(__file__).parent, self.package, ignore=shutil.ignore_patterns("__pycache__"))
        for relative in P.FROZEN_DEPENDENCIES:
            dst = self.worktree / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(P.ROOT / relative, dst)
        profiler = self.worktree / "rl_agent/splitfusion_single_gpu_profiler_v1"
        profiler.mkdir()
        for source in (Path(__file__).parent / "runtime_shims").glob("*.py"):
            shutil.copyfile(source, profiler / source.name)
        template = self.worktree / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
            "20261002T074000Z_disjoint_cpu_partition_v2/treatment/registration.json")
        template.parent.mkdir(parents=True)
        template.write_text(json.dumps({"schema": "scenesense.single_gpu_profiler.experiment.v1",
            "semantics": {}, "profiler": {}, "experiment_binds": {},
            "arms": {"C2": {"env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}}}))
        candidates = (P.ROOT / "rl_agent/splitfusion_cpu_partition_v1",
                      P.ROOT.parent / "abiodun/rl_agent/splitfusion_cpu_partition_v1")
        self.retained = next(path for path in candidates if path.is_dir())
        self.probe = self.root / "SYNC_PROBE_RESULT.json"
        self.probe.write_text(json.dumps({"passed": True}))
        self.original_probe_verifier = H._probe_prerequisite
        self.probe_patch = mock.patch.object(H, "_probe_prerequisite", side_effect=self.probe_proof)
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)

    def probe_proof(self, path, worktree):
        if path is None:
            raise ValueError("300 frames require a successful exact-source CARLA sensor-sync probe")
        return {"kind": "EXACT_SOURCE_SENSOR_SYNC_PROBE", "path": str(Path(path).resolve()),
                "report_sha256": H.sha(path), "verification": {"passed": True}}

    def prepare(self, output):
        return H.prepare(output, self.worktree, self.retained, 300, sync_probe=self.probe)

    def test_prospective_config_pins_new_actor_quality_and_all_source_files(self):
        output = self.root / "qualification"
        result = self.prepare(output)
        self.assertEqual(result["status"], "PREPARED_NOT_RUN")
        config = json.loads((output / "registration.json").read_text())
        self.assertEqual(config["policy"]["expected_weights_sha256"], H.A.WEIGHTS_SHA256)
        self.assertEqual(config["live_metric_contract"]["avo_threshold"], .65)
        self.assertEqual(config["network_profile"], "FAVORABLE_STABLE")
        self.assertEqual(config["prerequisite_binding"]["report_sha256"], H.sha(self.probe))
        self.assertEqual(config["registration_sha256"], H.canonical(
            {k: v for k, v in config.items() if k != "registration_sha256"}))
        with self.assertRaises(FileExistsError):
            H.prepare(output, self.worktree, self.retained, 300)

    def test_launch_refuses_unqualified_1000_before_subprocess(self):
        output = self.root / "capture1000"
        with self.assertRaisesRegex(ValueError, "successful new Run5 qualification"):
            H.prepare(output, self.worktree, self.retained, 1000)
        self.assertFalse((output / "controller_events.jsonl").exists())
        self.assertFalse(output.exists())

    def test_deployed_source_change_refuses_before_subprocess(self):
        output = self.root / "qualification"
        self.prepare(output)
        with (self.package / "runner.py").open("a") as handle:
            handle.write("\n# deliberately changed test fixture\n")
        with self.assertRaisesRegex(ValueError, "source drift"):
            H.launch(output, execute_token="RUN5_LIVE_CORRECTED_METRICS_AVO_300_ONCE_EXECUTE")
        self.assertFalse((output / "controller_events.jsonl").exists())

    def test_missing_probe_refuses_before_output_creation(self):
        output = self.root / "qualification"
        with self.assertRaisesRegex(ValueError, "sensor-sync probe"):
            H.prepare(output, self.worktree, self.retained, 300)
        self.assertFalse(output.exists())

    def test_probe_gate_requires_200_ticks_and_calls_real_verifier_api(self):
        short = {"passed": True, "complete_ticks": 1, "scored_exact_frames": 1}
        with mock.patch("rl_agent.splitfusion_run5_live_avo_v1.sync_probe.verify_probe", return_value=short):
            with self.assertRaisesRegex(ValueError, "source-bound verification"):
                self.original_probe_verifier(self.probe, self.worktree)
        complete = {"passed": True, "complete_ticks": 200, "scored_exact_frames": 180}
        with mock.patch("rl_agent.splitfusion_run5_live_avo_v1.sync_probe.verify_probe", return_value=complete) as verify:
            proof = self.original_probe_verifier(self.probe, self.worktree)
            verify.assert_called_once_with(self.probe.resolve(), source_root=self.worktree)
            self.assertEqual(proof["verification"], complete)

    def test_new_source_after_sealing_is_refused(self):
        output = self.root / "qualification"
        self.prepare(output)
        (self.package / "extra_source.py").write_text("# added after sealing\n")
        with self.assertRaisesRegex(ValueError, "source set drift"):
            H.launch(output, execute_token="RUN5_LIVE_CORRECTED_METRICS_AVO_300_ONCE_EXECUTE")

    def test_probe_drift_refuses_before_launch(self):
        output = self.root / "qualification"
        self.prepare(output)
        self.probe.write_text(json.dumps({"passed": False}))
        with mock.patch.object(H.subprocess, "run", side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError, "prerequisite changed"):
                H.launch(output, execute_token="RUN5_LIVE_CORRECTED_METRICS_AVO_300_ONCE_EXECUTE")
        self.assertFalse((output / "controller_events.jsonl").exists())

    def fake_completed_capture(self):
        output = self.root / "qualification"
        self.prepare(output)
        capture = output / "capture"
        capture.mkdir()
        result = {"passed": True, "gates": {f"G{i}": True for i in range(1, 10)},
                  "threadcap": {"checks": {f"E{i}": True for i in range(1, 6)}},
                  "audits": {role: {"returncode": 0, "verdict": "PASS"} for role in ("before", "after")}}
        H.write(capture / "QUALIFICATION_RESULT.json", result)
        for name in ("AFFINITY_APPLIED.json", "monitor_footer.json", "RESTORATION_LATEST.json"):
            H.write(output / name, {"passed": True})
        for name in ("prestop_restore_command.json", "radio_prestop_restore_command.json"):
            H.write(output / name, {"returncode": 0})
        return output

    def run_postflight(self, output, *, gt_status="LIVE_GT_AVO_INTEGRATION_VERIFIED"):
        with mock.patch("rl_agent.splitfusion_run5_live_avo_v1.verify_snr_capture.verify_capture",
                        return_value={"passed": True}), mock.patch(
                "rl_agent.splitfusion_run5_live_avo_v1.verify_live_gt_capture.verify_capture",
                return_value={"status": gt_status}):
            return H.postflight(output)

    def test_all_postflight_evidence_required_for_1000(self):
        output = self.fake_completed_capture()
        report = self.run_postflight(output)
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["checks"]), H.POSTFLIGHT_CHECKS)
        proof = H._qualification_prerequisite(output, self.worktree)
        self.assertEqual(proof["report_sha256"], H.sha(output / "RUN5_POSTFLIGHT.json"))
        destination = self.root / "capture1000"
        result = H.prepare(destination, self.worktree, self.retained, 1000, qualification=output)
        self.assertEqual(result["status"], "PREPARED_NOT_RUN")
        with self.assertRaises(FileExistsError):
            H.postflight(output)

    def test_gt_status_not_boolean_pass_and_cannot_be_ignored(self):
        output = self.fake_completed_capture()
        report = self.run_postflight(output, gt_status="VERIFICATION_FAILED")
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["exact_live_gt_avo"])
        with self.assertRaisesRegex(ValueError, "no complete successful postflight"):
            H._qualification_prerequisite(output, self.worktree)

    def test_qualification_raw_evidence_drift_is_refused(self):
        output = self.fake_completed_capture()
        raw = output / "capture/frame.npy"
        raw.write_bytes(b"raw sensor evidence")
        self.run_postflight(output)
        raw.write_bytes(b"modified raw sensor evidence")
        with self.assertRaisesRegex(ValueError, "evidence drift"):
            H._qualification_prerequisite(output, self.worktree)

    def test_qualification_is_not_just_a_passed_flag(self):
        output = self.fake_completed_capture()
        H.write(output / "RUN5_POSTFLIGHT.json", {"passed": True, "source_result_sha256": "fake"})
        with self.assertRaisesRegex(ValueError, "no complete successful postflight"):
            H._qualification_prerequisite(output, self.worktree)

    def test_incomplete_or_failed_profiler_never_qualifies(self):
        output = self.fake_completed_capture()
        result_file = output / "capture/QUALIFICATION_RESULT.json"
        result = json.loads(result_file.read_text())
        result["gates"].pop("G8")
        result_file.write_text(json.dumps(result))
        report = self.run_postflight(output)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["profiler_g1_g9"])

    def test_prerequisite_tamper_is_detected(self):
        output = self.root / "qualification"
        self.prepare(output)
        (output / "PREREQUISITE.json").write_text('{"passed":true}')
        with self.assertRaisesRegex(ValueError, "PREREQUISITE.json drift"):
            H.launch(output, execute_token="RUN5_LIVE_CORRECTED_METRICS_AVO_300_ONCE_EXECUTE")


if __name__ == "__main__":
    unittest.main()
