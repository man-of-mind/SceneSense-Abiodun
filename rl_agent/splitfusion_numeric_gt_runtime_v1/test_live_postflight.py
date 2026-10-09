"""Pure numerical postflight checks; synthetic reports are not live evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from . import live_postflight as P
from .live_process_scope import WORKER_SCOPE_SCHEMA


def fixture():
    events = {
        "1": dict(frame_id=101, ready_published_raw_ns=10, outcome=P.SCORED),
        "2": dict(frame_id=102, ready_published_raw_ns=20, outcome=P.SCORED),
        "3": dict(frame_id=103, ready_published_raw_ns=30,
                  outcome="LIVE_UNSCORED_FAULT_DURABLE"),
    }
    p = dict(closed=True, valid=True, failure=None, retained_jobs=0,
             issued=3, retired=3, events=events)
    s = dict(closed=True, valid=True, failure=None, retained_jobs=0,
             scored=2, released=2)
    completions = [dict(status="COMPLETE", durable=True, error=None,
                        identity=dict(frame_id=frame)) for frame in (101, 102, 103)]
    a = dict(failed=False, fault=None, outstanding_jobs=0, submitted=3,
             completed=3, completions=completions)
    ue = dict(schema=WORKER_SCOPE_SCHEMA, workers=[dict(role="producer", close=p)])
    edge = dict(schema=WORKER_SCOPE_SCHEMA, workers=[dict(role="scorer", close=s),
        dict(role="archive", close=a)], state=dict(installed=True, errors=[],
        inbox=dict(targets=3, terminals=3, failure=None, worker_alive=False),
        evaluator_reports=[dict(archive_worker_alive=False, failure=None,
            archive_records=[dict(status="SCORED_DURABLE"),
                             dict(status="SCORED_DURABLE"),
                             dict(status="UNSCORED_DURABLE")])]))
    profiler = dict(counts=dict(transmitted=100),
        validity={role: dict(passed=True) for role in ("ue", "edge", "map")},
        coverage={name: dict(records=100) for name in P.PROFILE_STAGES},
        sampler=dict(present=True, footer_present=True, dropped_samples=0),
        identity={"ue": dict(conflicts=0), "edge": dict(conflicts=0)},
        semantics={name: dict(passed=True) for name in
                   ("actor_replay", "reward", "receipt_marks")})
    timing = dict(counts=dict(sent_frames=100))
    qualification = dict(gates={"G1_FRAME_ACCOUNTING": True,
        "G2_ACTOR_HASH_UNCHANGED": True, "G9_HOST_RESTORED_CLEAN": True,
        "G4_COVERAGE_REPORTED": False})
    return dict(ue_scope=ue, edge_scope=edge, profiler=profiler, timing=timing,
        qualification=qualification, process_scope=dict(scope_ok=True),
        transmitted_budget=100)


class NumericPostflightTests(unittest.TestCase):
    def test_exact_numeric_durable_counts_pass_without_old_gt_stage(self):
        result = P.evaluate_reports(**fixture())
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["old_gt_stage_coverage_claim"])
        self.assertFalse(result["retained_gates_report_only"]["G4_COVERAGE_REPORTED"])
        self.assertEqual(result["counts"]["scored"], 2)
        self.assertEqual(result["counts"]["unscored"], 1)

    def test_missing_durable_archive_or_retained_slot_fails(self):
        data = fixture()
        data["edge_scope"]["state"]["evaluator_reports"][0]["archive_records"].pop()
        self.assertFalse(P.evaluate_reports(**data)["checks"]["N4_EXACT_DURABLE_ARCHIVES"])
        data = fixture()
        data["ue_scope"]["workers"][0]["close"]["retained_jobs"] = 1
        self.assertFalse(P.evaluate_reports(**data)["checks"]["N2_PRODUCER_DRAIN_AND_TARGET_COUNT"])

    def test_profiler_and_identity_are_actual_separate_checks(self):
        data = fixture()
        data["profiler"]["coverage"]["edge.compute"]["records"] = 0
        self.assertFalse(P.evaluate_reports(**data)["checks"]["N6_REAL_PROFILER_COVERAGE"])
        data = fixture()
        data["profiler"]["semantics"]["reward"]["passed"] = False
        self.assertFalse(P.evaluate_reports(**data)["checks"]["N7_IDENTITIES_ACTOR_AND_REWARD"])

    def test_frame_boundary_needs_retained_g1_proof(self):
        data = fixture()
        data["timing"]["counts"]["sent_frames"] = 99
        data["profiler"]["counts"]["transmitted"] = 99
        self.assertTrue(P.evaluate_reports(**data)["checks"]["N1_RETAINED_FRAME_ACCOUNTING"])
        data["qualification"]["gates"]["G1_FRAME_ACCOUNTING"] = False
        self.assertFalse(P.evaluate_reports(**data)["checks"]["N1_RETAINED_FRAME_ACCOUNTING"])

    def test_archive_readback_checks_actual_copied_bytes(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temp:
            edge = Path(temp) / "profiler" / "edge"
            root = edge / "numeric_runtime" / "archive"
            root.mkdir(parents=True)
            scope = edge / "numeric_edge_scope.json"
            raw = b"actual archived bytes"
            target = root / "numeric_gt" / "frame_0000000101" / "live_clock.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(raw)
            completion = data["edge_scope"]["workers"][1]["close"]["completions"][0]
            completion["written"] = [dict(root=0,
                relative="numeric_gt/frame_0000000101/live_clock.json",
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), ready=False)]
            for frame in (102, 103):
                row = data["edge_scope"]["workers"][1]["close"]["completions"][frame-101]
                name = f"numeric_gt/frame_{frame:010d}/live_clock.json"
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(raw)
                row["written"] = [dict(root=0, relative=name,
                    sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), ready=False)]
            self.assertTrue(P.verify_archive_readback(edge_scope_path=scope,
                edge_scope=data["edge_scope"])["passed"])
            target.write_bytes(b"changed")
            result = P.verify_archive_readback(edge_scope_path=scope,
                edge_scope=data["edge_scope"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["errors"][0]["problem"], "BYTE_OR_HASH_DRIFT")


if __name__ == "__main__":
    unittest.main()
