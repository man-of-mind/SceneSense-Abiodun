"""CPU integration test with actual children and real durable completion files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from .completion_rehearsal import STATUS
from .live_clock import ASYNC_HANDOFF_ID, ASYNC_SCHEMA, DEADLINE_NS
from .replay import OUTPUT_ROOT, ROOT


class CompletionRehearsalTests(unittest.TestCase):
    def test_edge_terminals_and_lost_uplink_release_real_producer_credits(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="completion-rehearsal-test-",
                                         dir=OUTPUT_ROOT) as temporary:
            output = Path(temporary) / "run"
            finished = subprocess.run(["/usr/bin/python3", "-B", "-m",
                "rl_agent.splitfusion_numeric_gt_runtime_v1.completion_rehearsal",
                "--output", str(output)], cwd=ROOT, env=env, text=True,
                capture_output=True, timeout=180, check=False)
            self.assertEqual(finished.returncode, 0, f"{finished.stdout}\n{finished.stderr}")
            result = json.loads((output / "RESULT.json").read_bytes())
            self.assertEqual(result["status"], STATUS)
            self.assertEqual(result["handoff_id"], ASYNC_HANDOFF_ID)
            self.assertIs(result["cpu_only"], True)
            self.assertIs(result["actual_ue_handoff"], True)
            for claim in ("live_services_started", "live_capture_claim", "live_qualified",
                          "deadline_claim", "performance_claim"):
                self.assertIs(result[claim], False, claim)
            self.assertEqual(result["feedback_scope"], "HARNESS_CALLBACK_NOT_RADIO")
            self.assertEqual(len(result["frames"]), 5)
            self.assertEqual(len({row["frame_id"] for row in result["frames"]}), 5)
            unscored = [row for row in result["frames"] if row["completion"] == "UNSCORED_DURABLE"]
            scored = [row for row in result["frames"] if row["completion"] == "SCORED_DURABLE"]
            self.assertEqual(len(unscored), 3)
            self.assertEqual(len(scored), 2)
            self.assertEqual({row["terminal_order"] for row in unscored},
                             {"BEFORE_TARGET", "AFTER_TARGET", "NO_EDGE_PROCESSING_OR_TERMINAL"})
            for row in unscored:
                self.assertIs(row["score_exists"], False)
                self.assertIsNone(row["numerical_q_perc"])
                self.assertEqual(row["files"], 2)
                self.assertEqual(row["terminal"], "FEEDBACK_TIMEOUT")
                self.assertGreater(row["timeout_elapsed_ns"], DEADLINE_NS)
                directory = output / "archive/numeric_gt" / f"frame_{row['frame_id']:010d}"
                terminal = json.loads((directory / "unscored_terminal.json").read_bytes())
                self.assertFalse(terminal["score_exists"])
                self.assertFalse(terminal["quality_feedback_claim"])
                self.assertFalse(terminal["feedback_receipt_claim"])
                self.assertFalse((directory / "numerical_score.json").exists())
                events = [item["event"] for item in result["observed_event_order"]
                          if item["frame_id"] == row["frame_id"]]
                if row["terminal_order"] == "NO_EDGE_PROCESSING_OR_TERMINAL":
                    self.assertEqual(events, ["TARGET_ADMITTED"])
                    self.assertEqual(row["durable_files"], 3)
                    admission = row["admission_expiry"]
                    raw = (output / "archive" / admission["relative"]).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), admission["sha256"])
                    proof = json.loads(raw)
                    self.assertFalse(proof["verified_processing_admitted"])
                    self.assertFalse(proof["evaluation_submitted"])
                    self.assertFalse(proof["score_exists"])
                    self.assertTrue(proof["future_processing_refused"])
                    self.assertEqual(proof["processing_horizon"]["ack_timeout_s"], .5)
                    self.assertGreater(proof["observed_wall_s"], proof["capture_deadline_wall_s"])
                else:
                    expected = ["EDGE_TERMINAL_OBSERVED", "TARGET_ADMITTED"]
                    self.assertEqual(events, expected if row["terminal_order"] == "BEFORE_TARGET"
                                     else list(reversed(expected)))
                    self.assertEqual(row["durable_files"], 2)
                    self.assertIsNone(row["admission_expiry"])
            self.assertTrue(all(row["score_exists"] and row["numerical_q_perc"] is not None
                                and row["files"] >= 9 for row in scored))
            self.assertTrue(all(row["clock_schema"] == ASYNC_SCHEMA for row in result["frames"]))
            self.assertEqual(result["overlap"]["active_producer_jobs"], 2)
            self.assertEqual(result["overlap"]["pending_reward_jobs"], 2)
            self.assertEqual(result["overlap"]["producer_retained_jobs"], 2)
            self.assertIs(result["overlap"]["archive_receipt_gate_entered"], True)
            self.assertEqual(result["credit_reuse"]["unscored_retired_before_scoring"], 2)
            self.assertEqual(result["credit_reuse"]["later_targets_published_before_scoring"], 2)
            self.assertEqual(result["child_exitcodes"], dict(producer=0, scorer=0, archive=0))
            self.assertIs(result["all_children_exit_zero"], True)
            self.assertIs(result["producer_owner_thread_alive_after_close"], False)
            self.assertEqual(result["closes"]["producer_handoff"]["errors"], [])
            for role in ("producer", "scorer"):
                self.assertTrue(result["closes"][role]["closed"])
                self.assertTrue(result["closes"][role]["valid"])
                self.assertIsNone(result["closes"][role]["failure"])
                self.assertEqual(result["closes"][role]["retained_jobs"], 0)
            self.assertEqual(result["closes"]["producer"]["issued"], 5)
            self.assertEqual(result["closes"]["producer"]["retired"], 5)
            self.assertEqual(result["closes"]["producer"]["retained_jobs"], 0)
            self.assertEqual(result["closes"]["scorer"]["scored"], 2)
            self.assertEqual(result["closes"]["scorer"]["released"], 2)
            self.assertEqual(len(result["closes"]["archive"]["completions"]), 5)
            self.assertEqual(result["closes"]["archive"]["submitted"], 5)
            self.assertEqual(result["closes"]["archive"]["completed"], 5)
            self.assertEqual(result["closes"]["archive"]["outstanding_jobs"], 0)
            self.assertFalse(result["closes"]["archive"]["failed"])
            self.assertIsNone(result["closes"]["archive"]["fault"])
            self.assertEqual(result["lost_frame_retained_jobs_after_completion"], 0)
            self.assertEqual(len(result["edge_scope"]["inbox"]["admission_expiries"]), 1)
            self.assertIsNone(result["edge_report"]["failure"])
            self.assertFalse((output / "FAILURE.json").exists())
            manifest = json.loads((output / "OUTPUT_SHA256.json").read_bytes())
            self.assertIn("RESULT.json", manifest)
            self.assertIn("EVENT_ORDER.json", manifest)
            self.assertIn("CREDIT_REUSE.json", manifest)
            self.assertIn("LOST_FRAME_DRAIN.json", manifest)
            self.assertEqual(set(manifest), {str(path.relative_to(output))
                for path in output.rglob("*") if path.is_file() and path.name != "OUTPUT_SHA256.json"})
            for relative, digest in manifest.items():
                self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                                 digest, relative)


if __name__ == "__main__":
    unittest.main()
