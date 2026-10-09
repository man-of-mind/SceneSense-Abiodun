"""CPU integration test: control vs candidate trampoline prep workers on the first 8 sealed v11 frames.

Run from abiodun/ with CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/python3 (inside the shared host lock). Checks: byte-identical per-frame outputs and producer
history, recorded q_perc reproduced, every request answered (no refusal/drop), clean worker exit, GC state
restored by uninstall, candidate idle maintenance ran and was never skipped for a pending request.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ENV_OK = (sys.executable == "/usr/bin/python3" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
          and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")))
FRAMES = 8
FIELDS = ("frame_id", "identity", "semantic", "commit", "q_perc", "measurement", "producer_files")


@unittest.skipUnless(ENV_OK, "CPU-only registered environment required")
class ReplayParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P, bench as B
        from . import replay as RY
        corpus = P.Corpus(49)
        corpus.cases = corpus.cases[:FRAMES]
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        original = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {B.CPU_PARENT})
        try:
            cls.sessions = [RY.session(corpus, out, i, m, int(240e6)) for i, m in enumerate(("C", "T"))]
        finally:
            os.sched_setaffinity(0, original)
        cls.traces = [json.loads((out / s["trace"]).read_text()) for s in cls.sessions]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_byte_identical_and_recorded(self):
        c, t = self.sessions
        self.assertEqual(c["history_sha256"], t["history_sha256"])
        self.assertEqual([[r[f] for f in FIELDS] for r in c["rows"]], [[r[f] for f in FIELDS] for r in t["rows"]])
        self.assertTrue(all(r["q_perc_matches_recorded"] for s in self.sessions for r in s["rows"]))

    def test_no_dropped_or_refused_requests_and_clean_exit(self):
        for s, tr in zip(self.sessions, self.traces):
            self.assertEqual(s["status"], "COMPLETE")
            self.assertEqual(len(s["rows"]), FRAMES)
            self.assertEqual(s["prep_returncode"], 0)
            self.assertEqual(s["engine"]["retired"], FRAMES)
            self.assertEqual(s["engine"]["retained_jobs"], 0)
            self.assertEqual(s["jobs_phase"], ["RETIRED"])
            ops = [o["op"] for o in tr["ops"]]
            self.assertEqual(ops.count("FREEZE"), FRAMES)
            self.assertEqual(ops.count("EVIDENCE_DURABLE"), FRAMES)

    def test_gc_state_restored_and_policy_active(self):
        c, t = self.traces
        self.assertEqual(c["policy_events"], [])
        self.assertEqual(c["gc_state_at_close"], c["gc_state_after_uninstall"])
        self.assertEqual(t["gc_state_after_uninstall"], c["gc_state_at_close"])
        self.assertGreater(t["gc_state_at_close"]["freeze_count"], 0)
        self.assertEqual(t["gc_state_after_uninstall"]["freeze_count"], 0)
        kinds = [e["kind"] for e in t["policy_events"]]
        self.assertEqual(kinds[0], "warmup_collect_freeze")
        self.assertIn("idle_full_collect", kinds)
        self.assertNotIn("skipped_pending", kinds)
        self.assertFalse(any(e.get("request_arrived_during") for e in t["policy_events"]))
        self.assertFalse([g for g in t["gc_events"] if g["gen"] == 2 and g["op_seq"] is not None],
                         "candidate ran a full collection inside a request")


if __name__ == "__main__":
    unittest.main()
