"""Real-child CPU rehearsal test; run with socket permission in the test host."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from .live_clock import ASYNC_HANDOFF_ID, ASYNC_SCHEMA
from .replay import OUTPUT_ROOT, ROOT


class AsyncLiveRehearsalTests(unittest.TestCase):
    def test_four_frames_use_actual_async_handoff_and_durable_children(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="async-live-rehearsal-test-",
                                         dir=OUTPUT_ROOT) as temporary:
            output = Path(temporary) / "run"
            command = ["/usr/bin/python3", "-B", "-m",
                       "rl_agent.splitfusion_numeric_gt_runtime_v1.async_live_rehearsal",
                       "--output", str(output)]
            finished = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                      capture_output=True, timeout=180, check=False)
            self.assertEqual(finished.returncode, 0,
                             f"{finished.stdout}\n{finished.stderr}")
            result = json.loads((output / "RESULT.json").read_bytes())
            self.assertEqual(result["status"],
                             "FOUR_FRAME_ASYNC_UE_REAL_CHILD_CPU_REHEARSAL_PASS")
            self.assertEqual(result["handoff_id"], ASYNC_HANDOFF_ID)
            self.assertIs(result["actual_ue_handoff"], True)
            self.assertEqual(result["feedback_scope"], "HARNESS_CALLBACK_NOT_RADIO")
            self.assertIs(result["deadline_claim"], False)
            self.assertIs(result["performance_claim"], False)
            self.assertIs(result["live_services_started"], False)
            self.assertIs(result["producer_owner_thread_alive_after_close"], False)
            self.assertEqual(len(result["frames"]), 4)
            self.assertEqual(len({row["frame_id"] for row in result["frames"]}), 4)
            self.assertEqual(result["overlap"]["admitted"], 4)
            self.assertEqual(result["overlap"]["active_producer_jobs"], 2)
            self.assertEqual(result["overlap"]["pending_reward_jobs"], 2)
            self.assertEqual(result["overlap"]["producer_retained_jobs"], 2)
            self.assertIs(result["overlap"]["archive_receipt_gate_entered"], True)
            self.assertTrue(all(row["clock_schema"] == ASYNC_SCHEMA and row["files"] >= 9
                                for row in result["frames"]))
            self.assertTrue(all(row["numerical_q_perc"] is not None
                                for row in result["frames"]))
            self.assertEqual(result["closes"]["producer"]["errors"], [])
            self.assertEqual(result["closes"]["producer"]["published"], 4)
            self.assertEqual(result["closes"]["producer"]["handoff_id"], ASYNC_HANDOFF_ID)
            self.assertFalse((output / "FAILURE.json").exists())
            manifest = json.loads((output / "OUTPUT_SHA256.json").read_bytes())
            for relative, digest in manifest.items():
                self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                                 digest, relative)
            self.assertIn("RESULT.json", manifest)


if __name__ == "__main__":
    unittest.main()
