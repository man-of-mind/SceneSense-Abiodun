"""Step 4 lifecycle and capacity tests with real prep/score worker processes (CPU only).

Event times only order operations; no duration is measured or asserted.

    cd abiodun && env TMPDIR=<scratch> CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m unittest rl_agent.splitfusion_exact_gt_service_proto_v1.test_lifecycle -v
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO

from . import parity_common as P, service as SV, channel as CH
from .lifecycle import Coordinator, Frame, recorded_rows, expected_history, PREDICTION_FIELDS

_CORPUS = None
RESULTS = {}


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = P.Corpus(49)
    return _CORPUS


def configs(c, *, prep_retained, scorer_jobs, scorer_waiting=2):
    engine = c.engine_config(offline=False)
    engine.pop("live_process_contract")
    engine.update(max_retained_jobs=prep_retained + 1, max_retained_bytes=(prep_retained + 1) * PR.RESERVATION)
    score = dict(run_binding=c.binding, producer_binding=c.startup["binding_sha256"], max_jobs=scorer_jobs,
                 max_bytes=scorer_jobs * 8 * 1024 * 1024, reservation_bytes=8 * 1024 * 1024,
                 max_waiting=scorer_waiting, max_frames=100)
    return dict(engine=engine), score


def recorded_frames(decisions, *, holds=True):
    rows = recorded_rows()
    base = rows[0]["open_ns"]
    frames = []
    for k in range(decisions):
        open_ms = (rows[k]["open_ns"] - base) / 1e6
        frames.append(Frame(0, "DECISION", open_ms, send_ms=rows[k]["send"], prediction_ms=rows[k]["prediction"]))
        if holds and k + 1 < decisions:
            mid = (open_ms + (rows[k + 1]["open_ns"] - base) / 1e6) / 2
            frames.append(Frame(0, "HOLD", mid, send_ms=rows[k]["send"], prediction_ms=rows[k]["prediction"]))
    for index, f in enumerate(frames):
        f.index = index
    return frames


def spaced_frames(decisions, spacing_ms=200.0):
    rows = recorded_rows()
    return [Frame(k, "DECISION", k * spacing_ms, send_ms=rows[k]["send"], prediction_ms=rows[k]["prediction"])
            for k in range(decisions)]


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        P.environment_ok()
        cls.c = corpus()
        cls.tmp = tempfile.TemporaryDirectory(prefix="exact_gt_proto_lifecycle_")

    @classmethod
    def tearDownClass(cls):
        out = os.environ.get("PROTO_LIFECYCLE_RESULT")
        if out:
            with open(out, "x") as stream:
                json.dump(RESULTS, stream, indent=1, sort_keys=True, default=str)
        cls.tmp.cleanup()

    def harness(self, name, **kwargs):
        prep, score = configs(self.c, **kwargs)
        return SV.Harness(Path(self.tmp.name) / name, prep_config=prep, score_config=score)

    def run_scenario(self, name, frames, *, prep_retained=3, scorer_jobs=4, scorer_waiting=2, stalls=()):
        h = self.harness(name, prep_retained=prep_retained, scorer_jobs=scorer_jobs, scorer_waiting=scorer_waiting)
        try:
            coordinator = Coordinator(h, self.c, frames, max_prep_retained=prep_retained, archive_stalls=stalls)
            summary = coordinator.run()
            prep_snap, score_snap = h.snapshots()
            summary.update(final_prep_history=prep_snap["history_sha256"], prep_retained_final=prep_snap["engine"]["retained_jobs"],
                           scorer_final=dict(slots=score_snap["retained_slots"], waiting=score_snap["waiting_predictions"],
                                             counters=score_snap["counters"]),
                           prep_deferral=prep_snap["engine"]["deferral"])
            summary["close"] = h.close()["archive"]["completed"]
        except BaseException:
            h.abort()
            raise
        self.assertTrue(summary["exactly_once"] and summary["no_feedback_non_decisions"] and summary["all_retired"])
        self.assertEqual(summary["prep_retained_final"], 0)
        self.assertEqual((summary["scorer_final"]["slots"], summary["scorer_final"]["waiting"]), (0, 0))
        if summary["committed"]:
            self.assertEqual(summary["final_prep_history"], expected_history(self.c, summary["committed"]))
        RESULTS[name] = summary
        return coordinator, summary

    def test_1_recorded_spacing_with_speculative_holds(self):
        frames = recorded_frames(6, holds=True)
        co, s = self.run_scenario("recorded_holds", frames)
        decisions = [f.index for f in frames if f.kind == "DECISION"]
        self.assertEqual(s["feedback_frames"], decisions)
        holds = [f for f in frames if f.kind == "HOLD"]
        self.assertTrue(all(("PREDICTION_REFUSED", "NO_FEEDBACK_FOR_NON_DECISION", False) in f.notes for f in holds))
        self.assertEqual(s["prep_deferral"]["cancelled_unbound"], len(holds))
        self.assertEqual(s["committed"], decisions)

    def test_2_200ms_consecutive_decisions(self):
        frames = spaced_frames(10)
        overlap = sum(f.prediction_ms > 200 - 25 for f in frames)
        co, s = self.run_scenario("spacing_200ms", frames)
        self.assertGreater(overlap, 0)   # some predictions arrive after the next frame was frozen
        self.assertEqual(s["feedback_frames"], list(range(10)))
        for f in frames:   # every frame accepted: quality equals the sealed v11 outcome
            self.assertEqual(f.result["q_perc"], self.c.predictions[self.c.cases[f.index].frame_id]["q_perc"])
        RESULTS["spacing_200ms"]["predictions_after_next_freeze"] = overlap

    def test_3_delayed_send_queues_then_refuses_freeze(self):
        frames = spaced_frames(6)
        for index, send in ((1, 190.0), (3, 420.0)):   # a delayed send delays that frame's prediction too
            frames[index].prediction_ms += send - frames[index].send_ms
            frames[index].send_ms = send
        co, s = self.run_scenario("delayed_send", frames)
        self.assertEqual(frames[2].freeze, "ADMITTED")                     # was queued behind frame 1
        self.assertIn((5, "REFUSED_FREEZE_QUEUE_FULL"), s["refusals"])     # frame 4 held the only queue slot
        self.assertEqual(s["max_seen"]["queue"], 1)
        self.assertEqual(s["committed"], [0, 1, 2, 3, 4])
        self.assertEqual(s["feedback_frames"], [0, 1, 2, 4])
        self.assertEqual(frames[3].retire, "UNSCORED_NO_RESULT_BY_HORIZON")   # its prediction passed the horizon
        self.assertIn(("PREDICTION_REFUSED", "LATE_AFTER_TERMINAL", False), frames[3].notes)
        self.assertEqual(frames[5].terminal, "UNSCORED_WAITING_DROPPED")

    def test_4_missing_late_failed_and_refused(self):
        frames = spaced_frames(8)
        frames[2].prediction_ms, frames[2].late_prediction_ms = None, 600.0
        frames[4].send_ms, frames[4].prediction_ms = -42.0, None
        frames[6].refuse_deferral = True
        co, s = self.run_scenario("missing_late_failed_refused", frames)
        self.assertEqual(s["feedback_frames"], [0, 1, 3, 5, 7])
        self.assertEqual(frames[2].retire, "UNSCORED_NO_RESULT_BY_HORIZON")
        self.assertIn(("PREDICTION_REFUSED", "LATE_AFTER_TERMINAL", True), frames[2].notes)
        self.assertEqual(frames[4].phase, "CANCELLED")
        self.assertTrue(frames[6].phase.startswith("NO_GT_DEFERRAL_REFUSED"))
        self.assertEqual(s["committed"], [0, 1, 2, 3, 5, 7])

    def test_5_stalled_archive_stays_bounded(self):
        frames = spaced_frames(12)
        co, s = self.run_scenario("stalled_archive", frames, prep_retained=3, scorer_jobs=2, stalls=[(0, 1500)])
        reasons = {r for _, r in s["refusals"]}
        self.assertIn("REFUSED_PREP_CAPACITY", reasons)
        self.assertIn("UNSCORED_SCORER_CAPACITY", {f.retire for f in frames})   # published, refused by full cache
        self.assertLessEqual(s["max_seen"]["scorer_slots"], 2)
        self.assertLessEqual(s["max_seen"]["prep_retained"], 3)
        self.assertIn(0, s["feedback_frames"]); self.assertIn(1, s["feedback_frames"])   # feedback during the stall
        self.assertTrue(any(i >= 8 for i in s["feedback_frames"]))                       # recovery after unstall
        for f in frames:   # every sent decision is either scored or an explicit no-score terminal
            self.assertTrue(f.feedback == 1 or f.terminal or (f.retire or "").startswith("UNSCORED"))

    def test_6_exactly_once_protocol(self):
        h = self.harness("exactly_once", prep_retained=2, scorer_jobs=2)
        try:
            case = self.c.cases[0]
            pred = self.c.predictions[case.frame_id]
            context = PO.context_document(self.c.context(case))
            plan = self.c.plan(case).decode()
            h.freeze(self.c.sourcekwargs(case), case.kwargs["policy_timestamp_s"])
            job = h.finish_freeze()["job"]
            identity = SV.ok(h.prep.call("BIND", dict(job=job, plan=plan, frame_context=context)), "BIND")["identity"]
            self.assertEqual(h.prep.call("BIND", dict(job=job, plan=plan, frame_context=context)).status, "REFUSED")
            SV.ok(h.prep.call("CONFIRM_SEND", dict(job=job, plan=plan)), "CONFIRM")
            self.assertEqual(h.prep.call("CONFIRM_SEND", dict(job=job, plan=plan)).status, "REFUSED")
            reply = h.prep.call("PUBLISH", dict(job=job))
            published = SV.ok(reply, "PUBLISH")
            packet = CH.read_sealed(reply.fd, published["size"]); os.close(reply.fd)
            self.assertEqual(h.prep.call("PUBLISH", dict(job=job)).status, "REFUSED")
            for expected in ("OK", "REFUSED"):   # second admission of the same target is refused
                fd, size = CH.seal_bytes([packet])
                r = h.score.call("TARGET", dict(identity=identity, semantic_sha256=published["semantic_sha256"], size=size), fd=fd)
                os.close(fd)
                self.assertEqual(r.status, expected)
            self.assertEqual(r.reason, "DUPLICATE_TARGET")
            ident = {k: pred["identity"][k] for k in PREDICTION_FIELDS}
            labels = pred["labels"].tobytes(order="C")
            first = SV.ok(h.predict(ident, context, pred["records"], labels), "PREDICTION")["result"]
            self.assertEqual(h.predict(ident, context, pred["records"], labels).reason, "DUPLICATE_PREDICTION")
            again = SV.ok(h.score.call("GET_RESULT", dict(identity=identity)), "GET_RESULT")["result"]
            self.assertEqual(first, again)
            self.assertEqual(h.score.call("ARCHIVE_DURABLE", dict(identity=identity, receipt={}, relative_prefix="x")).reason,
                             "DURABILITY_PRECEDES_EVIDENCE")
            SV.ok(h.score.call("RESULT_DELIVERED", dict(identity=identity)), "DELIVERED")
            self.assertEqual(h.score.call("RESULT_DELIVERED", dict(identity=identity)).status, "REFUSED")
            SV.ok(h.prep.call("RESULT_DELIVERED", dict(job=job)), "prep DELIVERED")
            h.retire_scored(job, case.kwargs["identity"], identity, case.frame_id)
            self.assertEqual(h.predict(ident, context, pred["records"], labels).reason, "LATE_AFTER_TERMINAL")
            h.close()
        except BaseException:
            h.abort()
            raise

    def test_7_shutdown_with_retained_work_fails_closed(self):
        h = self.harness("shutdown", prep_retained=2, scorer_jobs=2)
        try:
            case = self.c.cases[0]
            context = PO.context_document(self.c.context(case))
            plan = self.c.plan(case).decode()
            h.freeze(self.c.sourcekwargs(case), case.kwargs["policy_timestamp_s"])
            job = h.finish_freeze()["job"]
            SV.ok(h.prep.call("BIND", dict(job=job, plan=plan, frame_context=context)), "BIND")
            SV.ok(h.prep.call("CONFIRM_SEND", dict(job=job, plan=plan)), "CONFIRM")
            h.publish_to_scorer(job)
            self.assertEqual(h.score.call("CLOSE", dict(abandon=False)).reason, "CLOSE_WITH_RETAINED_WORK")
            SV.ok(h.score.close(abandon=True), "score abandon close")
            with self.assertRaises(SV.WorkerFailed):   # producer refuses a silent close with committed work
                h.prep.call("CLOSE", dict(abandon=False))
            h.prep.process.wait(timeout=30)
            self.assertNotEqual(h.prep.process.returncode, 0)
            h.prep.abort()                             # releases the failed worker's socket and log
            h.archive.close()
        except BaseException:
            h.abort()
            raise

    def test_8_worker_exits_when_parent_dies(self):
        script = (
            "import os,sys,json\n"
            "from rl_agent.splitfusion_exact_gt_service_proto_v1 import service as SV\n"
            "c=json.loads(sys.argv[1])\n"
            "w=SV.WorkerClient('score', c, log_path=sys.argv[2])\n"
            "print(w.process.pid, flush=True)\n"
            "os._exit(0)\n")
        _, score = configs(self.c, prep_retained=1, scorer_jobs=1)
        log = Path(self.tmp.name) / "orphan_score.log"
        env = dict(os.environ); env.pop("PYTHONPATH", None)
        out = subprocess.run([sys.executable, "-c", script, json.dumps(score), str(log)], cwd=P.ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
        pid = int(out.stdout.split()[0])
        deadline = time.monotonic() + 30
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        state = Path(f"/proc/{pid}/stat").read_text().split()[2] if Path(f"/proc/{pid}").exists() else "gone"
        self.assertIn(state, ("gone", "Z"))


if __name__ == "__main__":
    unittest.main()
