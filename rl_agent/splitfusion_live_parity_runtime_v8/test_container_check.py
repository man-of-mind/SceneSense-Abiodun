"""CPU proof safeguards and a real broker/async evaluator fixture lifecycle."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from . import container_check as D


class ContainerCheckTests(unittest.TestCase):
    def test_docker_argv_is_pinned_root_isolated_and_uses_existing_evidence_mount(self):
        with tempfile.TemporaryDirectory(prefix="v8-docker-argv-") as tmp:
            root = Path(tmp)
            repo, binding, code, scratch = [root / name for name in ("repo", "binding", "code", "scratch")]
            for path in (repo, binding, code, scratch, scratch / "segmentation_evidence"):
                path.mkdir()
            argv = D.docker_command(repo_root=repo, binding_directory=binding,
                partition_code=code, scratch_directory=scratch)
            self.assertEqual(argv[argv.index("--user") + 1], "0:0")
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertIn(D.IMAGE_ID, argv)
            self.assertIn("--read-only", argv)
            self.assertNotIn("--gpus", argv)
            self.assertNotIn("--privileged", argv)
            self.assertEqual([argv[i + 1] for i, value in enumerate(argv) if value == "--cap-add"],
                             ["CHOWN", "FOWNER", "DAC_OVERRIDE"])
            self.assertIn(f"type=bind,src={scratch / 'segmentation_evidence'},dst=/work/torch_cache/segmentation_evidence", argv)
            self.assertIn(f"type=bind,src={repo},dst=/work/abiodun,readonly", argv)
            self.assertIn("rl_agent.splitfusion_live_parity_runtime_v8.container_check", argv)

    def test_service_guard_denies_network_processes_and_arbitrary_threads(self):
        with D.deny_live_services() as state:
            with self.assertRaisesRegex(RuntimeError, "non-handoff"):
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with self.assertRaisesRegex(RuntimeError, "subprocess"):
                subprocess.Popen(["never-execute"])
            with self.assertRaisesRegex(RuntimeError, "thread"):
                threading.Thread(target=lambda: None, name="run4-evaluator").start()
            self.assertEqual(len(state["forbidden"]), 3)

    def test_registration_refuses_root_host(self):
        with mock.patch.object(D.os, "getuid", return_value=0), self.assertRaises(ValueError):
            D.synthetic_registration()

    def test_actual_cpu_packet_fixture_preserves_exact_quality_and_real_thread_drain(self):
        from rl_agent.splitfusion_live_gt_handoff_v1 import edge as H
        from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="v8-real-container-fixture-") as tmp:
            root = Path(tmp)
            (root / "segmentation_evidence").mkdir()
            profile = root / "profiler"
            profile.mkdir()
            registration = dict(handoff_binding="b" * 64, publisher_uid=os.getuid(), publisher_gid=os.getgid())
            checks = dict.fromkeys(D.CHECKS, False)
            with ThreadPoolExecutor(max_workers=1) as pool:
                publish = pool.submit(D.wait_and_publish, root, timeout_s=10.)
                with install_prediction_contract(), H.install_edge_handoff(
                        socket_path=root / "segmentation_evidence/gt_packet.sock", binding="b" * 64,
                        max_frames=1, archive_directory=profile / "prediction_archive",
                        publisher_uid=os.getuid(), publisher_gid=os.getgid()) as state:
                    result = D.exercise_scoring(profile, repo, checks, registration)
                self.assertTrue(publish.result(timeout=1.)["passed"])
            self.assertTrue(state["complete"])
            self.assertFalse(state["broker"]["worker_alive"])
            self.assertEqual(state["broker"]["admitted"], state["broker"]["consumed"])
            self.assertTrue(result["worker_thread_started"])
            self.assertFalse(result["real_ready_published"])
            self.assertNotEqual(result["packet_sha256"], result["durable_fixture_ready_sha256"])
            for key in ("no_disk_gt_read_during_scoring", "actual_evaluator_thread_drained",
                        "target_roundtrip_scored", "scoring_packet_distinct_from_durable_ready",
                        "actual_close_archives_flushed"):
                self.assertTrue(checks[key], key)


if __name__ == "__main__":
    unittest.main()
