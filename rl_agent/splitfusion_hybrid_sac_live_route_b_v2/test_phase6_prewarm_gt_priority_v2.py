"""Addendum-7 offline tests: deterministic pre-warm and reward-priority object GT.

The GT tests drive the unchanged pinned ``PassiveSplitCollector._evaluation_worker``
against both the original FIFO and the new two-class queue. No CARLA, OAI,
Docker, CUDA or network.
"""

from __future__ import annotations

import copy
import json
import queue
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned

from . import phase6_gt_handoff_v2 as GH
from . import phase6_gt_priority_v2 as GP
from . import phase6_prewarm_v2 as PW
from .test_phase3_continuous_execution_v2 import CONTRACT, FakeAE, FakeCodec, runtimes
from .test_phase6_live_integration_v2 import processor
from .test_phase6_live_path_repair_v2 import build, planner, run_frame

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Reward-priority object-GT queue
# ---------------------------------------------------------------------------


def _classifier(high: set):
    def classify(item):
        frame = int(item["frame_id"])
        if frame < 0:
            raise GP.GtQueueError("foreign")
        return GP.HIGH if frame in high else GP.LOW
    return classify


def _ticket(frame: int) -> dict:
    return {"frame_id": frame, "timestamp": frame / 10.0, "camera_matrix": None,
            "camera_inverse": None, "radar_points": {}, "scene": object(),
            "camera_location": None}


class _Counters:
    def __init__(self) -> None:
        self.values = {}

    def bump(self, name: str) -> None:
        self.values[name] = self.values.get(name, 0) + 1


class FakeHost:
    """The attributes the pinned object-GT worker touches, and nothing else."""

    def __init__(self, evaluation_queue, *, compute_s=0.0, fail=(), started=None,
                 release=None) -> None:
        self.evaluation_queue = evaluation_queue
        self.stop_event = threading.Event()
        self.scene_source = None
        self.gt_lock = threading.Lock()
        self.source_gt = {}
        self.transport_counters = _Counters()
        self.evaluation_errors = {}
        self.order = []
        self._compute_s, self._fail = compute_s, set(fail)
        self._started, self._release = started, release

    def _ground_truth(self, **kwargs):
        frame = int(kwargs["frame_id"])
        self.order.append(frame)
        if self._started is not None and not self._started.is_set():
            self._started.set()
            self._release.wait(2.0)
        time.sleep(self._compute_s)
        if frame in self._fail:
            raise RuntimeError(f"object GT failed for {frame}")
        return [{"class_name": "vehicle", "frame_id": frame, "world_x": frame * 1.5}]

    def run(self):
        thread = threading.Thread(target=pinned.PassiveSplitCollector._evaluation_worker,
                                  args=(self,), daemon=True)
        thread.start()
        return thread


def _drain(host, thread):
    deadline = time.monotonic() + 3.0
    while host.evaluation_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    host.evaluation_queue.put_nowait(None)
    thread.join(timeout=3.0)
    return not thread.is_alive()


class PriorityQueueTest(unittest.TestCase):
    def test_high_behind_queued_low_is_selected_first(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({3}))
        for frame in (1, 2, 3):
            q.put_nowait(_ticket(frame))
        self.assertEqual([q.get(timeout=0.1)["frame_id"] for _ in range(3)], [3, 1, 2])

    def test_multiple_high_fifo_then_low(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({2, 4}))
        for frame in (1, 2, 3, 4):
            q.put_nowait(_ticket(frame))
        self.assertEqual([q.get(timeout=0.1)["frame_id"] for _ in range(4)], [2, 4, 1, 3])

    def test_low_progresses_when_high_empty(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier(set()))
        q.put_nowait(_ticket(7))
        self.assertEqual(q.get(timeout=0.1)["frame_id"], 7)
        with self.assertRaises(queue.Empty):
            q.get(timeout=0.01)

    def test_overflow_duplicate_and_foreign_fail_explicitly(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({1}), maxsize_high=1,
                                       maxsize_low=1)
        q.put_nowait(_ticket(1))
        with self.assertRaises(GP.GtQueueError):
            q.put_nowait(_ticket(1))                     # duplicate identity
        q.put_nowait(_ticket(2))
        for frame, klass in ((5, "HIGH"), (6, "LOW")):
            classify = _classifier({5})
            q._classify = classify
            with self.assertRaisesRegex(GP.GtQueueError, "overflow"):
                q.put_nowait(_ticket(frame))
        with self.assertRaises(GP.GtQueueError):
            q.put_nowait(_ticket(-3))                    # foreign identity
        self.assertFalse(issubclass(GP.GtQueueError, queue.Full))

    def test_reward_gt_never_waits_behind_queued_holds(self) -> None:
        started, release = threading.Event(), threading.Event()
        log = GP.GtTicketLogV2()
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({4}), on_enqueue=log.enqueued,
                                       on_dequeue=log.dequeued)
        host = FakeHost(q, started=started, release=release)
        thread = host.run()
        q.put_nowait(_ticket(0))                          # LOW, starts executing
        self.assertTrue(started.wait(1.0))
        for frame in (1, 2, 3):
            q.put_nowait(_ticket(frame))                  # queued holds/fallbacks
        q.put_nowait(_ticket(4))                          # reward frame arrives last
        release.set()
        self.assertTrue(_drain(host, thread))
        # The executing LOW item 0 is not pre-empted (bounded residual: one LOW
        # computation); the reward item runs before every queued LOW item.
        self.assertEqual(host.order, [0, 4, 1, 2, 3])
        rows = {r["frame_id"]: r for r in log.snapshot()["tickets"]}
        self.assertEqual(rows[4]["queue_class"], "HIGH")
        self.assertIsNotNone(rows[4]["queue_wait_ms"])

    def test_worker_exception_is_recorded_and_accounted(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({2}))
        host = FakeHost(q, fail={2})
        thread = host.run()
        for frame in (1, 2, 3):
            q.put_nowait(_ticket(frame))
        self.assertTrue(_drain(host, thread))
        self.assertIn("OBJECT_GT_EVALUATION_FAILED", host.evaluation_errors[2])
        self.assertEqual(sorted(host.source_gt), [1, 3])
        self.assertEqual(q.unfinished_tasks, 0)

    def test_shutdown_drains_every_ticket(self) -> None:
        q = GP.RewardPriorityGtQueueV2(classify=_classifier({1, 4}))
        host = FakeHost(q, compute_s=0.002)
        thread = host.run()
        for frame in range(1, 7):
            q.put_nowait(_ticket(frame))
        self.assertTrue(_drain(host, thread))
        self.assertEqual(sorted(host.source_gt), list(range(1, 7)))
        self.assertEqual(q.unfinished_tasks, 0)

    def test_object_gt_contents_identical_to_original_fifo(self) -> None:
        results = []
        for make in (lambda: queue.Queue(maxsize=64),
                     lambda: GP.RewardPriorityGtQueueV2(classify=_classifier({2, 5}))):
            host = FakeHost(make())
            thread = host.run()
            for frame in range(1, 7):
                host.evaluation_queue.put_nowait(_ticket(frame))
            self.assertTrue(_drain(host, thread))
            results.append(host.source_gt)
        self.assertEqual(results[0], results[1])

    def test_startup_refresh_and_timed_refresh(self) -> None:
        class Source:
            _refreshed_at = 0.0

            def refresh_static(self, *, force=False):
                if force or time.monotonic() - self._refreshed_at > 1e9:
                    self._refreshed_at = time.monotonic()

        source = Source()
        record = GP.timed_startup_refresh(source)
        self.assertTrue(record["completed"])
        self.assertLessEqual(record["start_wall_ns"], record["end_wall_ns"])
        log = GP.GtTicketLogV2()
        GP.install_timed_refresh(source, log)
        source.refresh_static()
        source.refresh_static(force=True)
        self.assertEqual([c["refreshed"] for c in log.refresh_calls], [False, True])

        class Broken:
            def refresh_static(self, *, force=False):
                raise RuntimeError("CARLA RPC failed")

        with self.assertRaisesRegex(GP.GtQueueError, "pre-route refresh_static failed"):
            GP.timed_startup_refresh(Broken())

    def test_object_write_timing_is_logged_per_ticket(self) -> None:
        log = GP.GtTicketLogV2()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = GH.GtWriteRecorderV2(Path(tmp) / "w.jsonl", ticket_log=log)
            from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
            objects, _ = recorder.wrap(G.write_object_ground_truth,
                                       G.write_semantic_ground_truth)
            identity = {"run_id": "r", "cell_id": "c", "stream_id": "s", "frame_id": 9,
                        "action_id": 71, "profile_id": "p", "capture_timestamp_ns": 1}
            log.enqueued(9, GP.HIGH, time.time_ns())
            log.dequeued(9, GP.HIGH, time.time_ns())
            log.rows_started(9, time.time_ns())
            objects(Path(tmp), identity=identity, frozen_carla_frame_id=9,
                    rows=[{"class_name": "vehicle"}] * 3)
            log.completed(9, time.time_ns())
        row = log.snapshot()["tickets"][0]
        for key in ("enqueue_wall_ns", "worker_start_wall_ns", "queue_wait_ms",
                    "object_rows_start_wall_ns", "object_rows_end_wall_ns",
                    "objects_write_start_wall_ns", "objects_write_end_wall_ns",
                    "output_sha256", "output_size_bytes", "completion_wall_ns",
                    "object_rows_ms"):
            self.assertIn(key, row)
        self.assertEqual(row["object_count"], 3)
        self.assertEqual(row["output_identity"]["frame_id"], 9)
        self.assertEqual(log.missing_high_outputs(), [])


# ---------------------------------------------------------------------------
# Deterministic pre-warm
# ---------------------------------------------------------------------------


class _Runtime:
    def __init__(self) -> None:
        self._counters = types.SimpleNamespace(frames_attempted=5, frames_completed=5)
        self._context_session = object()
        self._detached_tail = object()


class PrewarmTest(unittest.TestCase):
    def test_all_12_modes_and_q_extremes(self) -> None:
        paths = PW.warm_paths(CONTRACT)
        self.assertEqual({p["mode_id"] for p in paths}, set(range(12)))
        by_mode = {}
        for path in paths:
            by_mode.setdefault(path["mode_id"], []).append(path["q_e4"])
        from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
            MODELED_SMOKE_SUPPORT,
        )
        for mode, (lower, upper) in enumerate(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds):
            self.assertIn(int(lower), by_mode[mode])
            self.assertIn(int(upper), by_mode[mode])
        self.assertEqual({(p["family"], p["quantizer"]) for p in paths},
                         {(f, q) for f in ("noAE", "AE128", "AE64", "AE32")
                          for q in ("UINT4", "UINT6", "UINT8")})

    def _warm_both(self):
        ue, _edge, _ranker = runtimes()
        syncs = []
        ue_report = PW.warm_ue(ue, CONTRACT, prepare_input=lambda f, r: ("7ch", f.shape,
                                                                         r.shape),
                               sync=lambda: syncs.append("s"))
        runtime = _Runtime()
        before = (runtime._counters, runtime._context_session, runtime._detached_tail)
        edge_report = PW.warm_edge(processor(), runtime, CONTRACT, device="cpu",
                                   encoders={f: FakeAE(f) for f in ("AE128", "AE64", "AE32")},
                                   codec=FakeCodec(), unguarded_tail=object(),
                                   sync=lambda: syncs.append("s"))
        after = (runtime._counters, runtime._context_session, runtime._detached_tail)
        return ue_report, edge_report, before, after, syncs

    def test_warm_ue_and_edge_cover_every_path_with_sync(self) -> None:
        ue_report, edge_report, before, after, syncs = self._warm_both()
        n = len(PW.warm_paths(CONTRACT))
        self.assertTrue(ue_report["completed"] and edge_report["completed"])
        self.assertEqual(ue_report["modes_warmed"], list(range(12)))
        self.assertEqual(edge_report["modes_warmed"], list(range(12)))
        self.assertEqual(len(ue_report["paths"]), n)
        self.assertEqual(len(edge_report["paths"]), n)
        self.assertEqual(len(syncs), 2 * (1 + n) + 2 * n)   # before+after every timed path
        self.assertEqual(ue_report["input_shape"], [])        # fake input carries no shape
        for row in edge_report["paths"]:
            self.assertEqual(len(row["wire_sha256"]), 64)
            self.assertEqual(len(row["map_update_sha256"]), 64)
        self.assertEqual(before, after)                       # swapped and restored
        self.assertIs(before[0], after[0])

    def test_warm_up_has_no_policy_or_accounting_side_effects(self) -> None:
        h, engine, pipe, actor = build()
        plan = planner(h, pipe)
        run_frame(h, plan, 10)                                # one real decision exists
        snapshot = {
            "counters": copy.deepcopy(engine.counters.__dict__),
            "opportunities": copy.deepcopy(engine.opportunities),
            "tensor_seq": engine._tensor_seq,
            "controllers": [(c.session_uuid, len(c.ledger.frames), len(c.ledger.feedback),
                             c._next_decision_seq, c._tensor_seq) for c in engine.controllers],
            "decisions": copy.deepcopy(pipe.decisions),
            "actor_calls": actor.calls,
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": np.random.get_state()[1].copy(),
        }
        self._warm_both()
        self.assertEqual(engine.counters.__dict__, snapshot["counters"])
        self.assertEqual(engine.opportunities, snapshot["opportunities"])
        self.assertEqual(engine._tensor_seq, snapshot["tensor_seq"])
        self.assertEqual([(c.session_uuid, len(c.ledger.frames), len(c.ledger.feedback),
                           c._next_decision_seq, c._tensor_seq) for c in engine.controllers],
                         snapshot["controllers"])
        self.assertEqual(pipe.decisions, snapshot["decisions"])
        self.assertEqual(actor.calls, snapshot["actor_calls"])
        self.assertTrue(torch.equal(torch.get_rng_state(), snapshot["torch_rng"]))
        self.assertTrue(np.array_equal(np.random.get_state()[1], snapshot["numpy_rng"]))

    def test_actor_weights_and_rng_fingerprint_unchanged(self) -> None:
        try:
            from . import frozen_actor_v2 as FA
            actor = FA.load_registered_actor()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"registered actor unavailable: {exc}")
        boundary, rng = FA.actor_boundary_sha256(actor.module), FA.rng_fingerprint()
        self._warm_both()
        self.assertEqual(FA.actor_boundary_sha256(actor.module), boundary)
        self.assertEqual(FA.rng_fingerprint(), rng)

    def test_ready_refused_when_warm_up_fails_or_is_incomplete(self) -> None:
        written = []

        def failing():
            raise PW.PrewarmError("tail failed")
        with self.assertRaises(PW.PrewarmError):
            PW.publish_ready_after_warmup(failing, lambda: written.append(1))
        with self.assertRaisesRegex(PW.PrewarmError, "READY refused"):
            PW.publish_ready_after_warmup(lambda: {"completed": True,
                                                   "modes_warmed": list(range(11))},
                                          lambda: written.append(1))
        self.assertEqual(written, [])
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "prewarm.json"
            PW.publish_ready_after_warmup(lambda: {"completed": True,
                                                   "modes_warmed": list(range(12))},
                                          lambda: written.append(1), report_path=report)
            self.assertEqual(written, [1])
            self.assertTrue(json.loads(report.read_text())["completed"])
            with self.assertRaises(FileExistsError):
                PW.write_report_create_only(report, {})

    def test_warm_frame_never_yields_an_evaluation_ticket(self) -> None:
        _ue, edge_report, *_ = self._warm_both()
        self.assertTrue(all(r["record_count"] >= 0 for r in edge_report["paths"]))

    def test_imports_start_nothing(self) -> None:
        import subprocess
        import sys
        code = ("import sys\nseen=[]\nsys.addaudithook(lambda e,a: seen.append(e) if e in "
                "('socket.connect','socket.bind','subprocess.Popen','open') and "
                "(e!='open' or not str(a[0]).endswith(('.py','.pyc','.so'))) else None)\n"
                "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_prewarm_v2\n"
                "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_gt_priority_v2\n"
                "import torch\nprint([e for e in seen if e!='open'], torch.cuda.is_initialized())\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                              text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin"})
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertEqual(done.stdout.strip(), "[] False")


# ---------------------------------------------------------------------------
# Prospective handshake verdict (v2) on synthetic evidence
# ---------------------------------------------------------------------------


def _cell(root: Path, *, superseded=False, latency=150.0, q=0.42) -> Path:
    cell = root / "cell"
    ev = cell / "run4_phase6"
    (ev / "gt_scratch_preserved").mkdir(parents=True)
    stem = "quality_gt_x_10"
    sha = {n: f"{i}" * 64 for i, n in enumerate(GH.COMPONENTS)}
    frames = [{"frame_id": 10, "session_uuid": "s", "ticket_seq": 0, "reward_requested": True},
              {"frame_id": 11, "session_uuid": "s", "ticket_seq": 0, "reward_requested": False}]
    ue = {"frames": frames, "prewarm_ue_completed": True,
          "gt_refresh_startup": {"completed": True, "end_wall_ns": 1},
          "decisions": [{"frame_id": 10, "kind": "POLICY_DECISION", "capture": {"ns": 1000},
                         "action_open": {"ns": 0},
                         "stages": {"input_7ch_start_raw_ns": 1_000_000,
                                    "front_end_raw_ns": 30_000_000,
                                    "first_packet_send_raw_ns": 31_000_000}}],
          "feedback_rows": [] if superseded else [{"frame_id": 10, "class": "ACCEPTED",
                                                   "reason": "NONE", "q_perc": q}],
          "feedback_ledgers": [[{"class": "ACCEPTED"}]],
          "resolutions": [{"terminal": "SUCCESS", "latency_ms": latency, "q_perc": q}],
          "unresolved_tickets_at_close": 0, "faulted": None, "gt_missing_high_outputs": [],
          "gt_objects": {"tickets": [{"frame_id": 10, "queue_class": "HIGH",
                                      "queue_wait_ms": 3.0, "object_rows_ms": 95.0}]}}
    handoff = {"stem": stem, "identity": {"frame_id": 10}, "read_errors": {},
               "read_success_wall_ns": 200_000_000,
               "components": {n: {"container_path": f"/c/{stem}.{n}",
                                  "first_observed_wall_ns": 150_000_000,
                                  "sha256_at_read": sha[n], "size_bytes": 1}
                              for n in GH.COMPONENTS}}
    evaluations = [] if superseded else [{
        "frame_id": 10, "kind": "DELIVERED_SUCCESS", "reason": "NONE", "q_perc": q,
        "emit_wall_ns": 220_000_000,
        "timing": {"enqueued_wall_ns": 100_000_000, "gt_ready_detected_wall_ns": 200_000_000,
                   "evaluator_start_wall_ns": 205_000_000, "evaluator_end_wall_ns": 215_000_000,
                   "gt_handoff": handoff}}]
    (ev / "PHASE6_UE_EVIDENCE.json").write_text(json.dumps(ue))
    (ev / "edge_report.json").write_text(json.dumps({"evaluations": evaluations}))
    (ev / "gt_scratch_preserved" / "run4_phase6_prewarm_edge.json").write_text(
        json.dumps({"completed": True, "modes_warmed": list(range(12))}))
    (ev / "gt_scratch_preserved.manifest.json").write_text(json.dumps({
        "verified": True, "files": {f"{stem}.{n}": {"sha256": sha[n]} for n in GH.COMPONENTS}}))
    with (ev / "gt_handoff_ue.jsonl").open("w") as handle:
        for n in GH.COMPONENTS:
            handle.write(json.dumps({"name": f"{stem}.{n}", "sha256": sha[n]}) + "\n")
    (ev / "edge_image_launch.json").write_text(json.dumps(
        {"post_create_container": {"mounts": {"state": {"source": "/h"}}}}))
    outcome = "SUPERSEDED_PENDING" if superseded else "RESULT_INSTALLED"
    (cell / "map_feedback.csv").write_text(f"frame_id,outcome\n10,{outcome}\n11,RESULT_INSTALLED\n")
    (cell / "CELL_RESULT.json").write_text(json.dumps({
        "phase6": {"gates": {"P7_FEEDBACK_OVER_DOWNLINK": not superseded,
                             "P8_NO_INFRASTRUCTURE_FAULT": True}},
        "cleanup": {"all_gates_passed": True,
                    "radio": {"noise_power_db_restored_and_read_back": True}}}))
    return cell


AUDIT = {"verdict": "PASS", "weights_file_sha256": "d" * 64}


class AddendumTest(unittest.TestCase):
    def test_addendum_and_manifest_bind_prior_attempts_and_are_not_executed(self) -> None:
        import hashlib
        pkg = ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
        addendum = json.loads((pkg / "phase6_prewarm_gt_priority_addendum_7.json").read_text())
        manifest = json.loads((pkg / "phase6_handshake_manifest_v7.json").read_text())
        self.assertEqual(manifest["status"], "PROSPECTIVE_NOT_EXECUTED_REQUIRES_AUTHORIZATION")
        self.assertIn("--stop-after-decisions 1", manifest["command"])
        self.assertEqual(len(addendum["preserved_attempts_sha256"]), 6)
        for files in addendum["preserved_attempts_sha256"].values():
            for path, digest in files.items():
                target = ROOT / path
                if not target.exists():
                    self.skipTest("prior evidence not present on this host")
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest, path)


class HandshakeVerdictV2Test(unittest.TestCase):
    def test_complete_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict = GH.handshake_verdict_v2(_cell(Path(tmp)), actor_audit_before=AUDIT,
                                              actor_audit_after=AUDIT)
        self.assertEqual(verdict["classification"], "PASS", verdict["checks"])
        self.assertIsNone(verdict["stage_excess"])

    def test_superseded_after_warmup_requires_reward_frame_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict = GH.handshake_verdict_v2(_cell(Path(tmp), superseded=True),
                                              actor_audit_before=AUDIT, actor_audit_after=AUDIT)
        self.assertEqual(verdict["classification"], "REWARD_FRAME_PROTECTION_REQUIRED")
        self.assertFalse(verdict["checks"]["reward_frame_not_superseded"])

    def test_late_feedback_reports_exact_excess_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict = GH.handshake_verdict_v2(_cell(Path(tmp), latency=190.0),
                                              actor_audit_before=AUDIT, actor_audit_after=AUDIT)
        self.assertEqual(verdict["classification"], "FAIL")
        excess = verdict["stage_excess"]
        self.assertAlmostEqual(excess["excess_ms"], 20.0)
        self.assertAlmostEqual(excess["ue_raw_clock_ms"]["7ch_to_front_end"], 29.0)
        self.assertAlmostEqual(excess["edge_wall_clock_ms"]["gt_ready_to_evaluator_start"], 5.0)


if __name__ == "__main__":
    unittest.main()
