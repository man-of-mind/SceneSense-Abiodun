"""Separate forced-screen verdict implementing the prospective I9 amendment (amendment 01).

The frozen runner and its QUALIFICATION_RESULT are never modified: under forced actions the retained G8
is false by design (its actor_replay component compares the executed action with the actor's proposal),
so the retained runner exits 1 and the retained host lifecycle raises ``Capture runner exit 1``.

The retained verdict-related exit is accepted ONLY when every condition below holds; anything else
(missing results, unknown exit codes, child crash, other gate failures, incomplete teardown, unknown
route error) is a stop:

  R  route segment ended normally at the registered capture budget (child rc 0, cell PASSED with no
     error, collector stop reason == the passing G1 branch, no collector failures, child/cell cleanup
     passed, route-runner terminal error one of the retained budget-stop signatures);
  G  G8 is the SOLE false retained gate (G1-G7, G9 true);
  C  every other G8 component passes: reward arithmetic, receipt marks, P3/P4/P5;
  A  the retained actor_replay failure is only the intended proposal-vs-forced comparison: replayed ==
     span-recorded proposal and planned == forced action for every reported mismatch;
  F  independent replay reproduces every logged proposal and every executed decision/hold equals the
     forced action (integration I3 + independent forced replay with zero mismatches);
  I  every remaining integration criterion passes (I1-I8, I10, I11; I9 without its launch clause:
     service/prep/score/archive exit, edge, restoration, no leftovers);
  X  launch exit code is 0, or exactly 1 with the retained ``RuntimeError: Capture runner exit 1`` as the
     only exception in the launch log and the retained verdict CAPTURE_INVALID.
"""
import json
from pathlib import Path

ROUTE_BUDGET_STOP_SIGNATURES = frozenset((
    "RouteBudgetReached: DECISION_CYCLE_BOUNDARY", "RouteBudgetReached: TRANSMITTED_BUDGET_REACHED",
    "TypeError: 'bool' object does not support item assignment"))
EXPECTED_LAUNCH_EXCEPTION = "RuntimeError: Capture runner exit 1"
G_OTHER = ("G1_FRAME_ACCOUNTING", "G2_ACTOR_HASH_UNCHANGED", "G3_ZERO_IDENTITY_CONFLICTS", "G4_COVERAGE_REPORTED",
           "G5_SAMPLER_REPORTED", "G6_NO_PROFILER_INDUCED_FAULT", "G7_INSTRUMENTATION_EVIDENCE_VALID", "G9_HOST_RESTORED_CLEAN")
INTEGRATION = ("I1_retained_gates", "I2_g8_components_except_actor_replay", "I3_forced_action", "I4_terminal_accounting",
               "I5_exact_quality_offline", "I6_history", "I7_no_pre_action_gt_work", "I8_bounded", "I9_cleanup",
               "I10_retained_settings", "I11_anchor_provenance")


def evaluate(e):
    """Pure decision over collected evidence ``e``; returns dict(verdict, conditions, reasons)."""
    reasons = []

    def need(condition, name, ok):
        if not ok:
            reasons.append(f"{condition}:{name}")
        return ok

    q = e.get("qualification")
    if not isinstance(q, dict) or not isinstance(e.get("cell_result"), dict) or not isinstance(e.get("integration"), dict) \
            or not isinstance(e.get("semantics"), dict):
        return dict(verdict="STOP", conditions={}, reasons=["MISSING_RESULTS"])
    gates = q.get("gates") or {}
    g1 = q.get("g1_detail") or {}
    cell, child = e["cell_result"], (e["cell_result"].get("child") or {})
    collector, route = child.get("collector") or {}, child.get("route_detail") or {}
    cleanup = e["cell_result"].get("cleanup") if isinstance(e["cell_result"].get("cleanup"), dict) else {}
    conditions = {}
    conditions["R"] = all([
        need("R", "child_return_code", child.get("return_code") == 0),
        need("R", "cell_passed", cell.get("status") == "PASSED" and not cell.get("error") and not child.get("error")),
        need("R", "collector_stop_matches_g1", g1.get("passed") is True and collector.get("stop_reason")
             and collector.get("stop_reason") == g1.get("stop_reason")),
        need("R", "no_collector_failures", collector.get("collector_failures") == []),
        need("R", "child_cleanup", all((child.get("cleanup") or {}).get(k) is True
                                       for k in ("edge_stopped", "map_stopped", "target_snr_restored"))),
        need("R", "cell_cleanup", cleanup.get("all_gates_passed") is True),
        need("R", "route_error_is_retained_budget_stop", route.get("error") in ROUTE_BUDGET_STOP_SIGNATURES)])
    conditions["G"] = all([need("G", "other_gates_true", all(gates.get(k) is True for k in G_OTHER)),
                           need("G", "g8_false", gates.get("G8_REWARD_TIMEOUT_SEMANTICS") is False),
                           need("G", "no_unknown_gates", set(gates) == set(G_OTHER) | {"G8_REWARD_TIMEOUT_SEMANTICS"})])
    phase6 = ((q.get("phase6_report_only") or {}).get("gates")) or {}
    sem = e["semantics"]
    conditions["C"] = all([need("C", "reward_semantics", (sem.get("reward") or {}).get("passed") is True),
                           need("C", "receipt_marks", (sem.get("receipt_marks") or {}).get("passed") is True),
                           need("C", "p3_p4_p5", all(phase6.get(k) is True for k in
                                                    ("P3_HOLD_TICKET", "P4_ACCOUNTING", "P5_NO_ACTOR_AFTER_REFUSAL")))])
    replay = sem.get("actor_replay") or {}
    forced = e.get("forced_action") or {}
    examples = replay.get("mismatch_examples") or []
    conditions["A"] = all([
        need("A", "actor_replay_reported", replay.get("passed") is False and replay.get("replayed") == replay.get("policy_decisions")
             and isinstance(replay.get("mismatches"), int) and replay["mismatches"] >= 1),
        need("A", "only_forced_comparison", all(m.get("replayed") == m.get("recorded")
             and list(m.get("planned") or []) == [forced.get("mode_id"), forced.get("q_e4")] for m in examples)
             and len(examples) == min(10, replay.get("mismatches") or 0))])
    fr = e.get("forced_replay") or {}
    integ = e["integration"]
    conditions["F"] = all([need("F", "I3_forced_action", (integ.get("checks") or {}).get("I3_forced_action") is True),
                           need("F", "independent_replay_zero_mismatch", fr.get("mismatches") == 0
                                and fr.get("replayed") == integ.get("decisions") and integ.get("decisions", 0) > 0)])
    checks = integ.get("checks") or {}
    conditions["I"] = all([need("I", name, checks.get(name) is True) for name in INTEGRATION]
                          + [need("I", "integration_evaluated_without_launch_clause", integ.get("launch_returncode") is None)])
    rc, log = e.get("launch_returncode"), e.get("launch_log_tail") or ""
    exceptions = [line.strip() for line in log.splitlines() if line and not line.startswith((" ", "\t", "Traceback"))
                  and ("Error" in line.split(":")[0] or "Exception" in line.split(":")[0])]
    if rc == 0:
        conditions["X"] = need("X", "verdict_valid_on_rc0", q.get("verdict") == "CAPTURE_VALID")
    elif rc == 1:
        conditions["X"] = all([need("X", "expected_exception_only", exceptions == [EXPECTED_LAUNCH_EXCEPTION]),
                               need("X", "retained_verdict_invalid", q.get("verdict") == "CAPTURE_INVALID")])
    else:
        conditions["X"] = need("X", f"unknown_exit_code_{rc}", False)
    verdict = "PASS_FORCED_SCREEN" if all(conditions.values()) and not reasons else "STOP"
    return dict(verdict=verdict, conditions=conditions, reasons=reasons, retained_verdict=q.get("verdict"),
                retained_g8=gates.get("G8_REWARD_TIMEOUT_SEMANTICS"), launch_returncode=rc)


def collect(arm_dir, *, launch_returncode, launch_log, integration_result):
    """Load the evidence for ``evaluate`` from an attempt root; missing files stay None (-> STOP)."""
    arm = Path(arm_dir)

    def load(path):
        try:
            return json.loads(Path(path).read_bytes())
        except (OSError, ValueError):
            return None
    cells = sorted((arm / "capture/cells").glob("*")) if (arm / "capture/cells").is_dir() else []
    cell = cells[0] if len(cells) == 1 else None
    analysis = load(arm / "capture/PROFILER_ANALYSIS.json") or {}
    try:
        tail = Path(launch_log).read_text()[-4000:]
    except OSError:
        tail = ""
    return dict(qualification=load(arm / "capture/QUALIFICATION_RESULT.json"),
                cell_result=None if cell is None else load(cell / "CELL_RESULT.json"),
                semantics=analysis.get("semantics"), integration=integration_result,
                forced_replay=(integration_result or {}).get("forced_replay"),
                forced_action=(load(arm / "registration.json") or {}).get("forced_action"),
                launch_returncode=launch_returncode, launch_log_tail=tail)
