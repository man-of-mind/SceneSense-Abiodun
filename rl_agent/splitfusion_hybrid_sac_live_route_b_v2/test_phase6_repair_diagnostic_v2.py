"""Addendum-6 offline tests: cycle-aware stopping, non-vacuous E5, GT handoff.

No CARLA, OAI, Docker, CUDA or network. The GT tests use the unchanged
``gt_evidence`` writers/readers on a temporary directory.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G

from . import phase6_decision_engine_v2 as E
from . import phase6_edge_runtime_v2 as ER
from . import phase6_engineering_gates_v2 as EG
from . import phase6_gt_handoff_v2 as GH
from . import phase6_live_child_nobuild_v2 as NB
from . import phase6_ue_runtime_v2 as U
from . import reward_hold_controller_v2 as R
from . import run4_live_wire_v2 as W
from .test_phase6_live_integration_v2 import MS, SPEC, processor
from .test_phase6_live_path_repair_v2 import build, planner, run_frame

ROOT = Path(__file__).resolve().parents[2]
FAILED_120 = (ROOT / "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
              "20260930T014900Z_phase6_engineering_qualification_120/cells/"
              "a71__favorable_stable")
VALID_MEASUREMENT = {
    **{f"seg_{c}_{k}": 10 for c in ("vehicle", "person")
       for k in ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")},
    **{f"loc_{c}_{k}": v for c in ("vehicle", "person")
       for k, v in (("eligible_gt", 1), ("tp", 1), ("fn", 0), ("fp", 0))},
    "loc_vehicle_matched_xy_errors_m": [0.1], "loc_person_matched_xy_errors_m": [0.1],
}


# ---------------------------------------------------------------------------
# Phase 3: stop only at a completed decision cycle
# ---------------------------------------------------------------------------


class CycleRun:
    """Mimics the runtime's submit: budget check, then sensor-first planning."""

    def __init__(self, *, frame_budget=None, decision_cap=None) -> None:
        self.h, self.engine, self.pipe, self.actor = build()
        self.plan = planner(self.h, self.pipe)
        self.budget = U.CycleBudgetV2(frame_budget=frame_budget, decision_cap=decision_cap)
        self.sent = 0
        self.frame = 10
        self.boundary = False

    def step(self, **kwargs):
        self.engine.new_opportunity_allowed = self.budget.allow_new_opportunity(
            sent=self.sent, policy_decisions=self.engine.counters.policy_decisions)
        self.frame += 1
        try:
            prepared = run_frame(self.h, self.plan, self.frame, **kwargs)
        except E.DecisionCapacityExhausted:
            self.boundary = True
            return None
        self.sent += 1
        return prepared.plan.kind

    def drain(self):
        current = self.engine.controller.current
        if current is not None and current.resolution is None:
            self.engine.controller.poll(current.action_open_ns + 300 * MS)

    def cycles(self):
        per = {}
        for f in self.engine.controller.ledger.frames:
            per.setdefault(f.ticket_seq, []).append(f)
        return per


class CycleBudgetTest(unittest.TestCase):
    def assert_closed_cycles(self, run: CycleRun) -> None:
        run.drain()
        for frames in run.cycles().values():
            self.assertGreaterEqual(len(frames), R.K_MIN)
            self.assertEqual(sum(1 for f in frames if f.reward_requested), 1)
        current = run.engine.controller.current
        self.assertTrue(current is None or current.resolution is not None)   # zero unresolved
        self.assertEqual(run.engine.counters.actor_calls, run.engine.counters.policy_decisions)

    def test_no_new_decision_when_its_k_min_group_cannot_fit(self) -> None:
        run = CycleRun(frame_budget=3)
        self.assertEqual(run.step(), E.FrameKind.POLICY_DECISION)
        self.assertEqual(run.step(), E.FrameKind.POLICY_HOLD)
        opportunities = len(run.engine.opportunities)
        self.assertIsNone(run.step())                  # remaining 1: refused, nothing assigned
        self.assertTrue(run.boundary)
        self.assertEqual(run.sent, 2)
        self.assertEqual(len(run.engine.opportunities), opportunities)
        self.assertEqual(run.actor.calls, 1)
        self.assert_closed_cycles(run)

    def test_hold_is_transmitted_when_only_one_frame_remains(self) -> None:
        run = CycleRun(frame_budget=2)
        self.assertEqual(run.step(), E.FrameKind.POLICY_DECISION)
        self.assertEqual(run.step(), E.FrameKind.POLICY_HOLD)   # remaining 1, hold allowed
        self.assertEqual(run.sent, 2)
        self.assert_closed_cycles(run)

    def test_stopping_at_a_fallback_opportunity(self) -> None:
        run = CycleRun(frame_budget=2)
        self.assertEqual(run.step(rgb_age_ns=150 * MS), E.FrameKind.FALLBACK)
        self.assertIsNone(run.step())                  # remaining 1: no new opportunity
        self.assertEqual(run.sent, 1)
        self.assertIsNone(run.engine.controller.current)
        self.assert_closed_cycles(run)

    def test_already_closed_cycle_allows_the_next_complete_cycle(self) -> None:
        run = CycleRun(frame_budget=5)
        self.assertEqual(run.step(), E.FrameKind.POLICY_DECISION)
        self.assertEqual(run.step(), E.FrameKind.POLICY_HOLD)
        run.drain()                                    # first ticket closed (timeout)
        self.assertEqual(run.step(), E.FrameKind.POLICY_DECISION)
        self.assertEqual(run.step(), E.FrameKind.POLICY_HOLD)
        self.assertIsNone(run.step())                  # remaining 1
        self.assertEqual((run.sent, run.engine.counters.policy_decisions), (4, 2))
        self.assert_closed_cycles(run)

    def test_decision_cap_stops_after_first_decision_and_its_hold(self) -> None:
        run = CycleRun(frame_budget=40, decision_cap=1)
        self.assertEqual(run.step(rgb_age_ns=150 * MS), E.FrameKind.FALLBACK)   # warm-up
        self.assertEqual(run.step(), E.FrameKind.POLICY_DECISION)
        self.assertEqual(run.step(), E.FrameKind.POLICY_HOLD)
        self.assertIsNone(run.step())
        self.assertEqual(run.sent, 3)
        self.assert_closed_cycles(run)

    def test_child_argv_split(self) -> None:
        self.assertEqual(NB.split_argv(["--a", "1", "--stop-after-decisions", "1", "--b"]),
                         (["--a", "1", "--b"], 1))
        self.assertEqual(NB.split_argv(["--a"]), (["--a"], None))
        with self.assertRaises(SystemExit):
            NB.split_argv(["--stop-after-decisions", "0"])


# ---------------------------------------------------------------------------
# Phase 2: non-vacuous E5
# ---------------------------------------------------------------------------


def _record(*, ready=True, wait_ms=20.0, reason="NONE"):
    timing = {"enqueued_wall_ns": 1_000}
    if ready:
        timing["gt_ready_detected_wall_ns"] = 2_000_000
        timing["evaluator_start_wall_ns"] = 2_000_000 + int(wait_ms * 1e6)
    else:
        timing["gt_expired_wall_ns"] = 2_000_000_000
    return {"timing": timing, "reason": reason if ready else "GROUND_TRUTH_UNAVAILABLE"}


class E5Test(unittest.TestCase):
    def test_all_gt_unavailable_is_never_pass(self) -> None:
        status, detail = EG.e5_status({"evaluations": [_record(ready=False)] * 5,
                                       "evaluator": {}})
        self.assertEqual(status, "INCONCLUSIVE_NO_GT_READY")
        self.assertEqual(detail["ready_wait_n"], 0)
        status, _ = EG.e5_status({"evaluations": [], "evaluator": {}})
        self.assertNotEqual(status, "PASS")

    def test_pass_and_each_failure_condition(self) -> None:
        good = {"evaluations": [_record(), _record(ready=False)], "evaluator": {}}
        self.assertEqual(EG.e5_status(good)[0], "PASS")
        self.assertEqual(EG.e5_status({"evaluations": [_record(wait_ms=251)],
                                       "evaluator": {}})[0], "FAIL")
        self.assertEqual(EG.e5_status({"evaluations": [_record(), _record(
            reason="EVALUATOR_EXCEPTION")], "evaluator": {}})[0], "FAIL")
        self.assertEqual(EG.e5_status({"evaluations": [_record()],
                                       "evaluator": {"queue_overflow": 1}})[0], "FAIL")
        self.assertEqual(EG.e5_status({"evaluations": [_record()],
                                       "evaluator": {"duplicate_emission_refused": 1}})[0],
                         "FAIL")

    def test_failed_120_frame_run_is_inconclusive_not_pass(self) -> None:
        if not FAILED_120.is_dir():
            self.skipTest("prior 120-frame evidence not present")
        report = EG.evaluate(FAILED_120)
        self.assertFalse(report["gates"]["E5_NON_VACUOUS_EVALUATOR"])
        self.assertEqual(report["evidence"]["e5_status"], "INCONCLUSIVE_NO_GT_READY")
        self.assertFalse(report["gates"]["E2_TIMING_BOUNDARY_EVERY_FRAME"])   # old ordering


# ---------------------------------------------------------------------------
# Phase 4: GT handoff diagnostics
# ---------------------------------------------------------------------------


def _ticket():
    h, engine, pipe, _ = build()
    prepared = run_frame(h, planner(h, pipe), 10)
    return processor().process(prepared.wire, edge_timing={}).evaluation


def _write_gt(directory: Path, ticket, recorder=None, *, semantic=True):
    objects, sem = G.write_object_ground_truth, G.write_semantic_ground_truth
    if recorder is not None:
        objects, sem = recorder.wrap(objects, sem)
    identity = dict(ticket.gt_identity)
    frame = int(identity["frame_id"])
    objects(directory, identity=identity, frozen_carla_frame_id=frame,
            rows=[{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}])
    if semantic:
        sem(directory, identity=identity, frozen_carla_frame_id=frame,
            mask=np.zeros((4, 4), np.uint8))


def _evaluator(directory: Path, sent: list, gt_timeout_s=0.3):
    return ER.Run4EvaluatorV2(
        spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=gt_timeout_s,
        read_ground_truth=lambda **kw: G.read_ground_truth(directory, **kw),
        probe_timeout_s=0.002, poll_interval_s=0.002, evidence_dir=directory)


class GtHandoffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "scratch" / "segmentation_evidence"
        self.evidence.mkdir(parents=True)
        self.cell = self.root / "cell"
        (self.cell / "run4_phase6").mkdir(parents=True)
        self.recorder = GH.GtWriteRecorderV2(self.cell / "run4_phase6" / "gt_handoff_ue.jsonl")
        self.ticket = _ticket()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recorder_logs_every_component_with_digest(self) -> None:
        _write_gt(self.evidence, self.ticket, self.recorder)
        rows = GH._jsonl(self.recorder.path)
        names = sorted(r["name"].split(".", 1)[1] for r in rows)
        self.assertEqual(names, sorted(GH.COMPONENTS))
        for row in rows:
            self.assertEqual(row["sha256"], GH._sha256(Path(row["host_path"])))
            self.assertEqual(row["identity"]["frame_id"], self.ticket.gt_identity["frame_id"])

    def test_preservation_is_create_only_and_verified(self) -> None:
        _write_gt(self.evidence, self.ticket)
        target = self.cell / "run4_phase6" / "gt_scratch_preserved"
        manifest = GH.preserve_directory(self.evidence, target)
        self.assertTrue(manifest["verified"])
        self.assertEqual(len(manifest["files"]), 3)
        self.assertTrue((target.parent / "gt_scratch_preserved.manifest.json").is_file())
        with self.assertRaises(FileExistsError):
            GH.preserve_directory(self.evidence, target)

    def test_evaluator_records_container_view_and_reads_real_gt(self) -> None:
        sent: list[bytes] = []
        evaluator = _evaluator(self.evidence, sent, gt_timeout_s=1.0)
        evaluator.start()
        evaluator.submit(self.ticket)
        time.sleep(0.05)                                 # GT arrives after submission
        _write_gt(self.evidence, self.ticket, self.recorder)
        evaluator.close()
        feedback, reason = W.decode_feedback(sent[0])
        self.assertEqual((feedback.kind, reason.name), ("DELIVERED_SUCCESS", "NONE"))
        timing = evaluator.records[0]["timing"]
        handoff = timing["gt_handoff"]
        self.assertTrue(handoff["read_errors"])          # exact misses before arrival
        self.assertTrue(all("ground truth missing" in k for k in handoff["read_errors"]))
        for name, entry in handoff["components"].items():
            self.assertIsNotNone(entry["first_observed_wall_ns"], name)
            self.assertIsNotNone(entry["sha256_at_read"], name)
        self.assertLessEqual((timing["evaluator_start_wall_ns"]
                              - timing["gt_ready_detected_wall_ns"]) / 1e6, 250.0)
        self._write_cell(evaluator)
        report = GH.handoff_report(self.cell)
        components = report["tickets"][0]["components"]
        self.assertTrue(all(c["host_container_match"] for c in components.values()))
        self.assertTrue(all(c["preserved_match"] for c in components.values()))

    def test_missing_semantic_component_is_named_exactly(self) -> None:
        sent: list[bytes] = []
        evaluator = _evaluator(self.evidence, sent, gt_timeout_s=0.1)
        _write_gt(self.evidence, self.ticket, self.recorder, semantic=False)
        evaluator.start()
        evaluator.submit(self.ticket)
        evaluator.close()
        record = evaluator.records[0]
        self.assertEqual(record["reason"], "GROUND_TRUTH_UNAVAILABLE")
        comps = record["timing"]["gt_handoff"]["components"]
        self.assertIsNotNone(comps["objects.json"]["first_observed_wall_ns"])
        self.assertIsNone(comps["semantic.npy"]["first_observed_wall_ns"])
        self.assertIsNone(comps["semantic.json"]["first_observed_wall_ns"])
        self._write_cell(evaluator)
        verdict = GH.handshake_verdict(self.cell, actor_audit_before={"verdict": "PASS"},
                                       actor_audit_after={"verdict": "PASS"})
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertEqual(sorted(verdict["components_never_written_on_host"]),
                         ["semantic.json", "semantic.npy"])

    def _write_cell(self, evaluator) -> None:
        evidence = self.cell / "run4_phase6"
        (evidence / "edge_report.json").write_text(json.dumps(
            {"evaluations": evaluator.records, "evaluator": evaluator.counters}, default=str))
        (evidence / "edge_image_launch.json").write_text(json.dumps({
            "post_create_container": {"mounts": {"state": {
                "source": str(self.evidence.parent), "destination": "/work/torch_cache",
                "rw": True}}}}))
        target = evidence / "gt_scratch_preserved"
        if not target.exists():
            GH.preserve_directory(self.evidence, target)


class BoundedChangeTest(unittest.TestCase):
    """Addendum 6 touches only the declared functions of frozen Phase-6 modules."""

    BASE = "09ba6e560297b5f17767986147b7f1d224006626"

    def _nodes(self, source: str) -> dict:
        import ast
        return {getattr(n, "name", f"stmt{i}"): ast.dump(n)
                for i, n in enumerate(ast.parse(source).body) if hasattr(n, "name")}

    def _committed(self, name: str) -> str:
        import subprocess
        return subprocess.run(
            ["git", "show", f"{self.BASE}:rl_agent/splitfusion_hybrid_sac_live_route_b_v2/{name}"],
            cwd=ROOT, capture_output=True, check=True, text=True).stdout

    def test_child_changes_only_run(self) -> None:
        name = "phase6_live_child_v2.py"
        old = self._nodes(self._committed(name))
        new_source = (ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2" / name).read_text()
        new = self._nodes(new_source)
        self.assertEqual(set(old), set(new))
        self.assertEqual({k for k in old if old[k] != new[k]}, {"run"})
        self.assertIn("DECISION_CYCLE_BOUNDARY", new_source)

    def test_addendum_binds_base_and_all_prior_attempts(self) -> None:
        import hashlib
        document = json.loads((ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/"
                               "phase6_repair_diagnostic_addendum_6.json").read_text())
        self.assertEqual(document["base_commit"], self.BASE)
        self.assertEqual(len(document["preserved_attempts_sha256"]), 5)
        for files in document["preserved_attempts_sha256"].values():
            for path, digest in files.items():
                target = ROOT / path
                if not target.exists():
                    self.skipTest("prior evidence not present on this host")
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest, path)

    def test_scientific_modules_unchanged_since_base(self) -> None:
        for name in ("frozen_actor_v2.py", "live_state_v2.py", "continuous_execution_v2.py",
                     "run4_live_wire_v2.py", "run4_map_protocol_v2.py",
                     "ue_telemetry_provider_v2.py", "reward_hold_controller_v2.py",
                     "phase6_result_reporting_v2.py", "live_qualification_300_v2.json",
                     "ACTOR_BINDING_V2.json"):
            self.assertEqual(
                (ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2" / name).read_text(),
                self._committed(name), name)


if __name__ == "__main__":
    unittest.main()
