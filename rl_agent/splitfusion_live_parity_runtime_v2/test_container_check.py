"""Fresh interpreter CPU tests; never execute Docker or any service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import container_check as C

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r'''
import importlib.util, json, sys
from pathlib import Path
from unittest import mock
import rl_agent
from rl_agent.splitfusion_live_parity_runtime_v2 import container_check as C
root, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
partition_code = root / "rl_agent/splitfusion_cpu_partition_v1"
if not partition_code.is_dir():
    partition_code = root.parent / "abiodun/rl_agent/splitfusion_cpu_partition_v1"
sys.path.append(str(partition_code))
if importlib.util.find_spec("rl_agent.splitfusion_single_gpu_profiler_v1") is None:
    rl_agent.__path__ = list(rl_agent.__path__) + [str(root.parent / "abiodun_run4_single_gpu_profiler_v1" / "rl_agent")]
registration = C.synthetic_registration()
if mode == "bad_registration":
    registration["registration_sha256"] = "0" * 64
reg = out / "registration.json"
C.write_x(reg, registration)
C.REGISTRATION = reg
manifest = out / "SOURCE_MANIFEST.json"
C.write_x(manifest, {"sources": {str(Path(C.__file__).relative_to(root)): C.sha(C.__file__)}})
old_read = Path.read_text
def read(path, *args, **kwargs):
    if str(path) == "/partition_output/registration.json":
        return old_read(reg, *args, **kwargs)
    return old_read(path, *args, **kwargs)
if mode == "lazy_forbidden":
    original = C._ticket_and_target
    def forbidden_target(*args):
        import scipy
        return original(*args)
    C._ticket_and_target = forbidden_target
if mode == "source_drift":
    value = json.loads(manifest.read_bytes())
    value["sources"][next(iter(value["sources"]))] = "0" * 64
    manifest.write_text(json.dumps(value))
with mock.patch.object(Path, "read_text", read):
    result = C.run_check(source_manifest=manifest, report=out/"REPORT.json", repo_root=root, image_id=C.IMAGE_ID)
print(json.dumps({"passed": result["passed"], "errors": result["errors"]}))
'''


class ContainerCheckTests(unittest.TestCase):
    def run_fresh(self, mode):
        temporary = tempfile.TemporaryDirectory(prefix="parity_container_check_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        done = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), str(root), mode],
            cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=45, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return root, json.loads((root/"REPORT.json").read_bytes())

    def test_real_entry_all_scopes_lazy_scoring_and_archive_flush(self):
        root, result = self.run_fresh("valid")
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(set(result["checks"]), set(C.CHECKS))
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["forbidden_import_attempts"], [])
        self.assertEqual(result["service_start_attempts"], [])
        self.assertFalse(result["ready_exists"])
        self.assertFalse(result["cuda_initialized"])
        self.assertTrue(result["sources_unchanged"])
        self.assertEqual(result["scoring"]["measurement"]["loc_vehicle_tp"], 1)
        self.assertEqual(result["scoring"]["measurement"]["loc_person_tp"], 1)
        for name, digest in result["logs"].items():
            self.assertEqual(C.sha(root/name), digest)
        self.assertTrue((root/"profiler"/"prediction_archive"/"MANIFEST.json").is_file())
        self.assertTrue((root/"profiler"/"pose_archive"/"MANIFEST.json").is_file())

    def test_real_registration_rejection_never_reaches_dispatch(self):
        root, result = self.run_fresh("bad_registration")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["registration_checked"])
        self.assertFalse(result["checks"]["retained_dispatch_reached"])
        self.assertIn("registration", result["errors"][0]["error"])
        self.assertFalse((root/"profiler"/"prediction_archive").exists())

    def test_forbidden_lazy_dependency_is_not_hidden_by_import_only_success(self):
        _, result = self.run_fresh("lazy_forbidden")
        self.assertTrue(result["checks"]["retained_dispatch_reached"])
        self.assertFalse(result["checks"]["target_roundtrip_scored"])
        self.assertFalse(result["passed"])
        self.assertIn("scipy", result["forbidden_import_attempts"])

    def test_source_drift_stops_before_importing_entry(self):
        _, result = self.run_fresh("source_drift")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["entry_imported"])
        self.assertIn("source changed", result["errors"][0]["error"])

    def test_registration_is_diagnostic_only_and_canonical(self):
        from . import contract
        value = C.synthetic_registration()
        self.assertEqual(value["id"], contract.ID)
        self.assertEqual(value["edge_module"], contract.EDGE)
        self.assertEqual(value["live_metric_contract"], contract.metrics())
        self.assertEqual(value["registration_sha256"], contract.canonical(
            {k: v for k, v in value.items() if k != "registration_sha256"}))
        self.assertTrue(value["diagnostic_only"])
        self.assertFalse(value["live_authorization"])
        self.assertNotIn("execute_token", value)

    def test_docker_command_is_cpu_isolated_pinned_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            dirs = []
            for name in ("repo", "binding", "code", "scratch"):
                directory = base/name; directory.mkdir(); dirs.append(directory)
            command = C.docker_command(repo_root=dirs[0], binding_directory=dirs[1],
                partition_code=dirs[2], scratch_directory=dirs[3])
            self.assertIn(C.IMAGE_ID, command)
            self.assertEqual(command[command.index("--network")+1], "none")
            self.assertEqual(command[command.index("--runtime")+1], "runc")
            self.assertEqual(command[command.index("--pull")+1], "never")
            self.assertNotIn("--gpus", command)
            self.assertNotIn("--privileged", command)
            mounts = [command[i+1] for i, value in enumerate(command) if value == "--mount"]
            self.assertEqual(sum(value.endswith(",readonly") for value in mounts), 3)
            self.assertIn("--read-only", command)
            self.assertEqual(command.count("--tmpfs"), 2)
            with self.assertRaises(ValueError):
                C.docker_command(repo_root=dirs[0], binding_directory=dirs[1],
                    partition_code=dirs[2], scratch_directory=dirs[0])

    def test_report_create_only_and_wrong_image_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp)/"REPORT.json"
            C.write_x(report, {"retained": True})
            with self.assertRaises(ValueError):
                C.run_check(source_manifest="unused", report=report, repo_root=ROOT, image_id=C.IMAGE_ID)
            self.assertEqual(json.loads(report.read_bytes()), {"retained": True})
            with self.assertRaises(ValueError):
                C.run_check(source_manifest="unused", report=Path(temp)/"other", repo_root=ROOT, image_id="mutable:tag")


if __name__ == "__main__":
    unittest.main()
