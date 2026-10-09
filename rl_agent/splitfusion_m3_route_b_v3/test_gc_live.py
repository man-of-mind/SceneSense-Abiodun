"""Focused integration tests for the paired GT GC live qualification (CPU only; no CARLA/OAI/GPU, no fixtures).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v rl_agent.splitfusion_m3_route_b_v3.test_gc_live
The real-service start at the exact M3 frame-ceiling limits is the separate proof ``gc_service_rehearsal``.
"""
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from . import contract as C
from . import gc_seams as S
from . import gc_trace as T
from . import service_entry as SE
from . import worker_entry as WE
from .gc_policy_live import SafeBoundedGcPolicy

ROOT = Path(__file__).resolve().parents[2]


def gc_state():
    return dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()), freeze_count=gc.get_freeze_count())


class Stubs:
    """Replace the heavy role-core constructors/handlers BEFORE worker_entry.main captures them."""

    def __enter__(self):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import worker as W, prep_core as PC
        self.W, self.PC = W, PC
        self.saved = (PC.PrepCore.__init__, PC.PrepCore.handle, W.ScoreCore.__init__, W.ScoreCore.handle)

        def init(core, config):
            core.config = config
            if isinstance(core, PC.PrepCore):        # what the guarded worker inspects: the constructed producer
                from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
                core.engine = type("E", (), {})()
                core.engine._owner = type("O", (), {})()
                core.engine._owner._producer = D.NS.NumericSourceProducer.__new__(D.NS.NumericSourceProducer)

        def handle(core, op, args, fd):
            garbage = [[] for _ in range(30000)]          # allocation pressure -> young collections, as real requests
            for item in garbage[:1000]:
                item.append(item)                          # cycles
            return dict(op=op), None
        PC.PrepCore.__init__, PC.PrepCore.handle = init, handle
        W.ScoreCore.__init__, W.ScoreCore.handle = init, handle
        return self

    def __exit__(self, *exc):
        PC, W = self.PC, self.W
        PC.PrepCore.__init__, PC.PrepCore.handle, W.ScoreCore.__init__, W.ScoreCore.handle = self.saved


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gc_live_test_"))
        self.initial = gc_state()

    def tearDown(self):
        self.assertEqual(gc_state(), self.initial, "test leaked GC state")

    def argv(self, role, policy, trace=None, producer="v1"):
        trace = trace or self.tmp
        return ["--fd", "0", "--parent", str(os.getpid()), "--role", role, "--gc-policy", policy,
                "--trace-dir", str(trace), "--max-frames", "1000", "--producer", producer]

    def trace(self, role, directory=None):
        return json.loads(((directory or self.tmp) / f"{role}_gc_trace.json").read_text())


class ContractTests(Base):
    def test_arms_tokens_registration_and_wording(self):
        self.assertNotEqual(C.token("control"), C.token("candidate"))
        self.assertNotEqual(C.token("candidate", producer="v1"), C.token("candidate", producer="v2"))
        self.assertEqual(C.gc_registration("candidate", 1000, "v1")["producer_substitutions"], [])
        self.assertEqual(len(C.gc_registration("candidate", 1000, "v2")["producer_substitutions"]), 2)
        with self.assertRaises(ValueError):
            C.gc_registration("candidate", 1000, "v3")
        self.assertIsNone(C.gc_registration("control", 1000)["policy"])
        self.assertEqual(C.gc_registration("candidate", 1000)["policy"], C.GC_POLICY)
        self.assertEqual(C.gc_registration("control", 1000)["safety"], C.gc_registration("candidate", 1000)["safety"])
        self.assertIn("NOT a memory cap", C.gc_registration("candidate", 1000)["threshold_wording"])
        self.assertEqual(C.trace_limits(1000), dict(max_ops=16000, max_gen2_events=4000, max_policy_events=4000))
        with self.assertRaises(ValueError):
            C.token("other")
        self.assertLess(C.SAFETY["idle_collection_wall_max_ms"], 100)
        self.assertGreater(C.SAFETY["prep_rss_max_bytes"], 720 * 2**20)


class WorkerRoleTests(Base):
    def test_candidate_refused_outside_prep_role_before_start(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        proc = subprocess.run([sys.executable, "-B", "-m", C.WORKER_ENTRY, *self.argv("score", "candidate")], cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("REFUSED", proc.stderr)
        self.assertFalse((self.tmp / "score_gc_trace.json").exists())

    def test_init_role_mismatch_is_refused_and_restored(self):
        with Stubs() as st:
            def serve(fd, parent, cpu):
                st.W.ScoreCore({})          # the service sent role=score to a worker registered as prep
            with self.assertRaises(RuntimeError):
                WE.main(self.argv("prep", "candidate"), serve=serve)
        doc = self.trace("prep")
        self.assertIn("differs from registered --role", doc["refusal"])
        self.assertFalse(doc["policy_installed"])
        self.assertTrue(doc["restored_equals_initial"])

    def test_candidate_prep_installs_and_restores_on_failure_path(self):
        seen = {}
        with Stubs() as st:
            def serve(fd, parent, cpu):
                st.PC.PrepCore({})
                seen["threshold"] = gc.get_threshold()
                raise ValueError("simulated worker failure")
            with self.assertRaises(ValueError):
                WE.main(self.argv("prep", "candidate"), serve=serve)
        self.assertEqual(seen["threshold"][2], C.GC_POLICY["gen2_threshold"])
        doc = self.trace("prep")
        self.assertTrue(doc["policy_installed"])
        self.assertIn("simulated worker failure", doc["exit"]["exception"])
        self.assertTrue(doc["restored_equals_initial"])

    def test_control_never_installs(self):
        seen = {}
        with Stubs() as st:
            def serve(fd, parent, cpu):
                st.PC.PrepCore({})
                seen["threshold"] = gc.get_threshold()
            WE.main(self.argv("prep", "control"), serve=serve)
        self.assertEqual(list(seen["threshold"]), self.initial["threshold"])
        doc = self.trace("prep")
        self.assertFalse(doc["policy_installed"])
        self.assertEqual(doc["policy_events"], [])

    def test_reply_hook_warmup_freeze_then_idle_collections(self):
        a, b = socket.socketpair()
        with Stubs() as st:
            def serve(fd, parent, cpu):
                core = st.PC.PrepCore({})
                for seq in range(1, 6):
                    for op in ("FREEZE", "BIND", "EVIDENCE_DURABLE"):
                        core.handle(op, {}, None)
                        st.W.CH.send(a, dict(seq=seq, status="OK"), deadline=time.monotonic() + 5)
                        b.recv(65536)
            WE.main(self.argv("prep", "candidate"), serve=serve)
        doc = self.trace("prep")
        kinds = [e["kind"] for e in doc["policy_events"]]
        self.assertEqual(kinds[0], "warmup_collect_freeze")
        self.assertEqual(doc["policy_events"][0]["decision"], 2)
        self.assertEqual(kinds.count("idle_full_collect"), 3)
        self.assertTrue(doc["restored_equals_initial"])
        s = T.summarize(doc, C.GC_POLICY["warmup_decisions"])
        self.assertEqual(s["decisions"], 5)
        self.assertEqual(s["idle_collections"], 3)


class ProducerTests(Base):
    def test_v2_refused_outside_prep_role_before_start(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        proc = subprocess.run([sys.executable, "-B", "-m", C.WORKER_ENTRY, *self.argv("score", "control", producer="v2")],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("producer v2 may only be selected in the GT prep worker role", proc.stderr)

    def test_v2_installed_in_prep_and_restored(self):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OW
        from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS
        seen = {}
        with Stubs() as st:
            def serve(fd, parent, cpu):
                seen["owner_type"] = OW.NumericSourceProducer.__name__
                st.PC.PrepCore({})
            WE.main(self.argv("prep", "candidate", producer="v2"), serve=serve)
        doc = self.trace("prep")
        self.assertEqual(seen["owner_type"], "NumericSourceProducerV2")
        self.assertEqual(doc["producer_constructed"], "NumericSourceProducerV2")
        self.assertTrue(doc["producer_switch_restored"])
        self.assertIs(D.NS, NS)
        self.assertIs(OW.NumericSourceProducer, NS.NumericSourceProducer)
        self.assertEqual(set(doc["producer_v2_sources"]), {"__init__.py", "exact_frame.py", "source_v2.py", "worker_switch.py",
                                                          "verify_parity.py", "replay_producer.py", "worker_entry.py"})

    def test_v1_rollback_installs_nothing(self):
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OW
        seen = {}
        with Stubs() as st:
            def serve(fd, parent, cpu):
                seen["owner_type"] = OW.NumericSourceProducer.__name__
                st.PC.PrepCore({})
            WE.main(self.argv("prep", "candidate", producer="v1"), serve=serve)
        doc = self.trace("prep")
        self.assertEqual(seen["owner_type"], "NumericSourceProducer")
        self.assertEqual(doc["producer_constructed"], "NumericSourceProducer")
        self.assertIsNone(doc["producer_switch_restored"])

    def test_constructed_class_mismatch_refused(self):
        with Stubs() as st:
            def serve(fd, parent, cpu):
                from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as SW
                SW.uninstall()                       # simulate the substitution not taking effect
                st.PC.PrepCore({})
            with self.assertRaises(RuntimeError):
                WE.main(self.argv("prep", "candidate", producer="v2"), serve=serve)
        self.assertIn("differs from registered", self.trace("prep")["refusal"])


class SafetyTests(Base):
    def make(self, rss_values):
        values = iter(rss_values)
        return SafeBoundedGcPolicy(policy=C.GC_POLICY, safety=C.SAFETY, max_events=4, rss=lambda: next(values))

    def test_rss_max_stop_restores_stock_gc(self):
        p = self.make([10**9, 10**9, 2 * 2**30]).install()   # decision-1 check, freeze baseline, decision-2 check
        p.after_reply("EVIDENCE_DURABLE")            # decision 1: no maintenance, rss ok
        p.after_reply("EVIDENCE_DURABLE")            # decision 2: freeze, then rss above the stop
        self.assertEqual(p.safety_stop["reason"], "PREP_RSS_MAX")
        self.assertFalse(p.installed)
        self.assertEqual(gc_state(), self.initial)
        self.assertIsNone(p.after_reply("EVIDENCE_DURABLE"))

    def test_growth_after_freeze_stop(self):
        p = self.make([10**9, 10**9, 10**9, 10**9 + 300 * 2**20]).install()
        for _ in range(3):
            p.after_reply("EVIDENCE_DURABLE")
        self.assertEqual(p.safety_stop["reason"], "PREP_RSS_GROWTH_AFTER_FREEZE")
        self.assertEqual(gc_state(), self.initial)

    def test_idle_wall_stop_and_event_bound(self):
        p = self.make([10**9] * 20).install()
        original = p._run
        p._run = lambda kind, op, freeze=False: dict(original(kind, op, freeze=freeze), wall_ms=60.0)
        p.after_reply("EVIDENCE_DURABLE"); p.after_reply("EVIDENCE_DURABLE")
        self.assertEqual(p.safety_stop["reason"], "IDLE_COLLECTION_WALL")
        self.assertEqual(gc_state(), self.initial)
        values = iter([10**9] * 50)
        q = SafeBoundedGcPolicy(policy=dict(C.GC_POLICY, idle_min_gen1=0), safety=C.SAFETY, max_events=4,
                                rss=lambda: next(values)).install()
        for _ in range(20):
            q.after_reply("EVIDENCE_DURABLE")
        self.assertLessEqual(len(q.events), 4)
        self.assertGreater(q.events_dropped, 0)
        q.uninstall()


class TracerTests(Base):
    def test_ops_bound_counts_overflow(self):
        tracer = WE.Tracer(dict(max_ops=3, max_gen2_events=1, max_policy_events=1))

        class Core:
            def handle(self, op, args, fd):
                return op
        tracer.wrap_handle(Core)
        core = Core()
        for _ in range(5):
            core.handle("FREEZE", {}, None)
        self.assertEqual(len(tracer.ops), 3)
        self.assertEqual(tracer.dropped["ops"], 2)
        self.assertEqual(tracer.last_op, "FREEZE")


class SubstitutionTests(Base):
    def test_service_worker_rewrite_roles_and_restore(self):
        from rl_agent.splitfusion_exact_gt_service_proto_v2 import service as SV
        before = (SV.WorkerClient.__init__, SV.subprocess)
        sub = SE.WorkerSubstitution(arm="candidate", trace_dir=self.tmp, max_frames=1000, producer="v2").install()
        try:
            base = [sys.executable, "-B", "-u", "-m", C.WORKER_MODULE, "--fd", "5", "--parent", "9", "--cpu", "4"]
            sub._role.value = "prep"
            prep = sub.rewrite(base)
            sub._role.value = "score"
            score = sub.rewrite(base)
            self.assertEqual(prep[prep.index("--gc-policy") + 1], "candidate")
            self.assertEqual(score[score.index("--gc-policy") + 1], "control")
            self.assertEqual(prep[prep.index("--producer") + 1], "v2")
            self.assertEqual(score[score.index("--producer") + 1], "v1")
            self.assertEqual(prep[prep.index("-m") + 1], C.WORKER_ENTRY)
            self.assertIn("--cpu", prep)
            with self.assertRaises(RuntimeError):
                sub.rewrite([sys.executable, "-m", "something.else"])
            sub._role.value = None
            with self.assertRaises(RuntimeError):
                sub.rewrite(base)
        finally:
            sub.uninstall()
        self.assertEqual((SV.WorkerClient.__init__, SV.subprocess), before)

    def test_ue_seam_rewrites_only_service_and_restores_on_failure(self):
        from rl_agent.splitfusion_exact_gt_live_v2 import ue as UE
        reg = dict(id=C.ID, transmitted_budget=1000, gt_gc=C.gc_registration("control", 1000, "v2"))
        calls = []
        real = subprocess.Popen
        subprocess.Popen = lambda argv, *a, **k: calls.append(list(argv))
        try:
            with self.assertRaises(KeyError):
                with S.install_ue_service_seam(registration=reg, output=self.tmp):
                    UE.subprocess.Popen([sys.executable, "-m", C.SERVICE_MODULE, "--fd", "3", "--parent", "1"])
                    UE.subprocess.Popen(["echo", "untouched"])
                    raise KeyError("simulated child failure")
        finally:
            subprocess.Popen = real
        self.assertIs(UE.subprocess, subprocess)
        self.assertEqual(calls[0][calls[0].index("-m") + 1], C.SERVICE_ENTRY)
        self.assertEqual(calls[0][calls[0].index("--gc-policy") + 1], "control")
        self.assertEqual(calls[0][calls[0].index("--producer") + 1], "v2")
        self.assertEqual(calls[1], ["echo", "untouched"])
        doc = json.loads((self.tmp / C.TRACE_DIR / "ue_substitution.json").read_text())
        self.assertTrue(doc["restored"])
        self.assertIn("simulated child failure", doc["error"])
        with self.assertRaises(ValueError):
            with S.install_ue_service_seam(registration=dict(reg, id="OTHER"), output=self.tmp / "x"):
                pass


if __name__ == "__main__":
    unittest.main()
