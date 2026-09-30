"""Offline tests for Phase-6 addendum-2 (option c) result reporting.

The scenario runs the real decision engine, controller and R4FB codec:
an undefined-quality (no eligible GT) outcome, a success, a timeout and a
ticket still open at stop. It checks that undefined quality keeps its frozen
semantics and that the new fields report it exactly.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

from . import phase6_decision_engine_v2 as E
from . import phase6_live_runner_v2 as RUN
from . import phase6_result_reporting_v2 as REP
from . import reward_hold_controller_v2 as R
from . import run4_live_wire_v2 as W
from .test_phase6_live_integration_v2 import MS, SPEC, Pipeline, processor

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
BASE_COMMIT = "42bca445b993b095d5404cca38b6372eb3bc8704"
FROZEN = (
    "phase6_decision_engine_v2.py", "reward_hold_controller_v2.py", "live_state_v2.py",
    "run4_live_wire_v2.py", "continuous_execution_v2.py", "frozen_actor_v2.py",
    # phase6_edge_runtime_v2.py: addendum-4 ready-record repair (AST-bounded elsewhere).
    "phase6_ue_runtime_v2.py", "phase6_live_child_v2.py",
    "live_qualification_300_v2.json", "PHASE6_RUNNER_REPORT.md", "ACTOR_BINDING_V2.json",
)
VALID = {
    **{f"seg_{c}_{k}": 10 for c in ("vehicle", "person")
       for k in ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")},
    **{f"loc_{c}_{k}": v for c in ("vehicle", "person")
       for k, v in (("eligible_gt", 1), ("tp", 1), ("fn", 0), ("fp", 0))},
    "loc_vehicle_matched_xy_errors_m": [0.1], "loc_person_matched_xy_errors_m": [0.1],
}
NO_GT = {f"loc_{c}_eligible_gt": 0 for c in ("vehicle", "person")}


def _feedback_row(p: Pipeline, prepared, measurement, delay_ms: int) -> dict:
    """Edge evaluation -> R4FB wire -> UE decode -> engine, as the UE records it."""
    processed = processor().process(prepared.wire, edge_timing={})
    feedback, reason = W.quality_feedback(
        SPEC, {"frame_id": processed.envelope.frame_id, **measurement}, processed.envelope)
    decoded, decoded_reason = W.decode_feedback(W.encode_feedback(feedback, reason))
    opened = p.engine.controller.current.action_open_ns
    outcome = p.engine.on_feedback(decoded, receipt_raw_ns=opened + delay_ms * MS)
    return {"frame_id": decoded.frame_id, "class": outcome.value, "kind": decoded.kind,
            "reason": decoded_reason.name, "q_perc": decoded.q_perc}


def _ue_evidence(p: Pipeline, feedback_rows: list) -> dict:
    """The fields of PHASE6_UE_EVIDENCE.json that reporting reads (as close() writes)."""
    engine = p.engine
    return json.loads(json.dumps({
        "coverage": engine.coverage(),
        "counters": dataclasses.asdict(engine.counters),
        "faulted": engine.faulted,
        "sessions": [c.session_uuid for c in engine.controllers],
        "resolutions": [r.to_canonical_dict() for c in engine.controllers
                        for r in c.resolutions()],
        "frames": [{**{n: getattr(f, n) for n in f.__dataclass_fields__ if n != "action"},
                    "action": f.action.to_canonical_dict(), "session_uuid": c.session_uuid}
                   for c in engine.controllers for f in c.ledger.frames],
        "feedback_rows": feedback_rows,
    }, default=str))


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        p = Pipeline()
        rows = []
        d1 = p.step()                                   # decision 1: no eligible GT
        rows.append(_feedback_row(p, d1, NO_GT, 40))
        cls.after_excluded = p.engine.controller.current.resolution
        p.step()                                        # held tensor
        d2 = p.step()                                   # decision 2: genesis, success
        cls.d1, cls.d2 = d1, d2
        rows.append(_feedback_row(p, d2, VALID, 50))
        cls.success = p.engine.controller.current.resolution
        p.step()
        d3 = p.step()                                   # decision 3: never answered
        p.step()
        d4 = p.step()                                   # decision 4: open at stop
        cls.d3, cls.d4 = d3, d4
        cls.p, cls.rows = p, rows
        cls.ue = _ue_evidence(p, rows)
        cls.summary = REP.result_summary(cls.ue)

    def test_undefined_quality_keeps_frozen_semantics(self) -> None:
        excluded = self.after_excluded
        self.assertIs(excluded.terminal, contract.RewardTerminal.EVALUATOR_FAULT)
        self.assertFalse(excluded.learning_included)
        self.assertIsNone(excluded.reward)                 # no zero/neutral reward
        self.assertIsNone(excluded.q_perc)
        self.assertEqual(self.rows[0]["reason"], "QUALITY_UNDEFINED_NO_ELIGIBLE_GT")
        self.assertEqual(self.rows[0]["class"], "ACCEPTED")
        # No stale previous outcome: the next decision is genesis of a new session.
        self.assertIs(self.d2.plan.kind, E.FrameKind.POLICY_DECISION)
        self.assertNotEqual(self.d2.plan.decision_session_uuid,
                            self.d1.plan.decision_session_uuid)
        self.assertEqual(self.d2.plan.frame_identity.decision_seq, 0)
        self.assertEqual(self.p.engine.counters.session_rollovers, 1)
        # The timeout decision is a TIMEOUT, not relabelled, and stays in its session.
        terminals = [r["terminal"] for r in self.ue["resolutions"]]
        self.assertEqual(terminals, ["EVALUATOR_FAULT", "SUCCESS", "TIMEOUT"])

    def test_reporting_fields(self) -> None:
        s = self.summary
        self.assertEqual(s["reward_requested_decisions"], 4)
        self.assertEqual(s["resolved_decisions"], 3)
        self.assertEqual(s["open_at_stop"], 1)
        self.assertEqual(s["eligible_quality_denominator"], 2)
        self.assertEqual(s["excluded_count"], 1)
        self.assertAlmostEqual(s["excluded_rate"], 1 / 3)
        self.assertEqual(s["exclusion_reason_histogram"],
                         {"EVALUATOR_FAULT:QUALITY_UNDEFINED_NO_ELIGIBLE_GT": 1})
        self.assertEqual(s["session_rollovers"], 1)
        self.assertEqual(s["sessions"], 2)
        self.assertTrue(s["sessions_consistent_with_rollovers"])
        self.assertEqual(s["integrity"], {"extra_reward_frames": 0,
                                          "excluded_with_reward_value": 0})
        mean = s["conditional_reward_mean"]
        self.assertAlmostEqual(mean["value"], (self.success.reward - 1.0) / 2)
        self.assertEqual(mean["n"], 2)
        self.assertEqual((mean["label"], mean["gated"]), (REP.CONDITIONAL_LABEL, False))
        rate = s["conditional_success_rate"]
        self.assertEqual((rate["value"], rate["successes"], rate["n"]), (0.5, 1, 2))
        self.assertFalse(rate["gated"])
        self.assertEqual(s["claim_scope"], "SYSTEMS_INTEGRATION_QUALIFICATION_ONLY")
        self.assertFalse(s["policy_performance_claim"])
        self.assertIn("not a policy-performance PASS", s["statement"])

    def test_other_exclusion_reasons_are_histogrammed(self) -> None:
        ue = json.loads(json.dumps(self.ue))
        frame = self.rows[0]["frame_id"]
        for row in ue["feedback_rows"]:
            if row["frame_id"] == frame:
                row["reason"] = "GROUND_TRUTH_UNAVAILABLE"
        self.assertEqual(REP.result_summary(ue)["exclusion_reason_histogram"],
                         {"EVALUATOR_FAULT:GROUND_TRUTH_UNAVAILABLE": 1})
        ue["feedback_rows"] = []
        self.assertEqual(REP.result_summary(ue)["exclusion_reason_histogram"],
                         {"EVALUATOR_FAULT:UNATTRIBUTED": 1})

    def test_evaluate_phase6_adds_scope_without_changing_gates(self) -> None:
        evaluation = RUN.evaluate_phase6(ue=self.ue, edge={}, map_identity_rows=[],
                                         feedback_packets_on_ue_tunnel=[], cleanup_ok=True)
        self.assertEqual(sorted(evaluation["gates"]), sorted([
            "P0_POLICY_COVERAGE", "P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING",
            "P5_NO_ACTOR_AFTER_REFUSAL", "P6_RESTORE_COLD", "P7_FEEDBACK_OVER_DOWNLINK",
            "P8_NO_INFRASTRUCTURE_FAULT"]))
        self.assertEqual(evaluation["claim_scope"], REP.CLAIM_SCOPE)
        self.assertFalse(evaluation["policy_performance_claim"])
        self.assertEqual(evaluation["result_summary"], self.summary)
        self.assertEqual(evaluation["reported_not_gated"]["reward_mean_label"],
                         REP.CONDITIONAL_LABEL)
        self.assertEqual(evaluation["reported_not_gated"]["session_rollovers"], 1)
        # Coverage below 10 opportunities stays INCONCLUSIVE, exactly as before.
        self.assertEqual(evaluation["verdict"], "INCONCLUSIVE_OR_FAILED")

    def test_markdown_leads_with_systems_integration_scope(self) -> None:
        evaluation = RUN.evaluate_phase6(ue=self.ue, edge={}, map_identity_rows=[],
                                         feedback_packets_on_ue_tunnel=[], cleanup_ok=True)
        passed = {**evaluation, "verdict": "PASSED",
                  "gates": {k: True for k in evaluation["gates"]}}
        text = REP.render_markdown(passed)
        head = "\n".join(text.splitlines()[:6])
        self.assertIn("SYSTEMS-INTEGRATION PASS (not a policy-performance PASS)", head)
        self.assertIn("P0-P8 PASS is a systems-integration PASS, not a policy-performance "
                      "PASS", head)
        self.assertIn(REP.CONDITIONAL_LABEL, text)
        self.assertIn("EVALUATOR_FAULT:QUALITY_UNDEFINED_NO_ELIGIBLE_GT", text)
        with tempfile.TemporaryDirectory() as tmp:
            path = RUN.write_result_summary(Path(tmp), passed)
            self.assertEqual(path.read_text(encoding="utf-8"), text)
            with self.assertRaises(FileExistsError):
                RUN.write_result_summary(Path(tmp), passed)


class AddendumAndFreezeTest(unittest.TestCase):
    def test_addendum_binds_unchanged_plan_and_memo(self) -> None:
        verified = RUN.verify_addendum()
        self.assertEqual(verified["id"], REP.ADDENDUM_ID)
        self.assertFalse(verified["policy_performance_claim"])
        with tempfile.TemporaryDirectory() as tmp:
            forged = Path(tmp) / "addendum.json"
            document = json.loads(RUN.ADDENDUM_PATH.read_text(encoding="utf-8"))
            document["amends"]["plan_sha256"] = "0" * 64
            forged.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(RUN.Phase6RunnerError):
                RUN.verify_addendum(forged)
            document = json.loads(RUN.ADDENDUM_PATH.read_text(encoding="utf-8"))
            document["id"] = "SOMETHING_ELSE"
            forged.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(RUN.Phase6RunnerError):
                RUN.verify_addendum(forged)

    def test_frozen_semantics_and_earlier_report_are_byte_identical(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git unavailable")
        for name in FROZEN:
            committed = subprocess.run(
                ["git", "show", f"{BASE_COMMIT}:{PACKAGE}/{name}"], cwd=ROOT,
                capture_output=True, check=True).stdout
            self.assertEqual((ROOT / PACKAGE / name).read_bytes(), committed, name)

    def test_reporting_module_import_is_pure(self) -> None:
        text = Path(REP.__file__).read_text(encoding="utf-8")
        for token in ("open(", "Path(", "subprocess", "socket", "torch"):
            self.assertNotIn(token, text.split('"""', 2)[2])


if __name__ == "__main__":
    unittest.main()
