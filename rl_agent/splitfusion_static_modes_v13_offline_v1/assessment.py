"""Outcome-aware assessment of one static-mode arm (registered before any v13 frame; also qualified on M1).

Classes (exactly one):
  INVALID_IDENTITY        the executed action is not the registered one (fixed arm: any live compute span, decision
                          or hold, with another mode/q; actor called; fallback substituted; engine mismatch counter;
                          agent arm: actor calls != policy decisions)
  INFRASTRUCTURE_FAILURE  any non-actor integrity/accounting/terminal/cleanup check fails (G1-G7, G9; P1, P3, P4, P6,
                          P8; reward and receipt semantics; completion accounting; protocol faults; GC/producer; child
                          cleanup; the registered budget stop; v13 finish repair, drain and channel evidence; edge
                          evaluator queue overflow; a non-zero launch rc that is not explained by actor-only gates)
  VALID_MEASUREMENT       otherwise; ``ZERO_COMPLETION`` is a valid subtype when no prediction completed.
Actor-only checks (G8 actor replay, P0 coverage, P5 actor-after-refusal) are REQUIRED for the agent arm and
NOT_APPLICABLE for fixed arms.  Delivery-dependent checks (P7) are NOT_EXERCISED -- never PASS and never FAIL -- when
no prediction completed, and then the absence of quality feedback is itself checked.  GT-producer capacity refusals
are registered bounded behaviour and are REPORTED, never counted as completions or as infrastructure failure.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from . import completion as CP

INFRA_GATES = ("G1_FRAME_ACCOUNTING", "G2_ACTOR_HASH_UNCHANGED", "G3_ZERO_IDENTITY_CONFLICTS", "G4_COVERAGE_REPORTED",
               "G5_SAMPLER_REPORTED", "G6_NO_PROFILER_INDUCED_FAULT", "G7_INSTRUMENTATION_EVIDENCE_VALID",
               "G9_HOST_RESTORED_CLEAN")
INFRA_PHASE = ("P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING", "P6_RESTORE_COLD", "P8_NO_INFRASTRUCTURE_FAULT")
ACTOR_ONLY = ("G8_REWARD_TIMEOUT_SEMANTICS", "P0_POLICY_COVERAGE", "P5_NO_ACTOR_AFTER_REFUSAL")
ROUTE_STOP = {"m1": "TypeError: 'bool' object does not support item assignment",     # pre-repair M1 signature
              "v13": "RouteBudgetReached: TRANSMITTED_BUDGET_REACHED"}                # repaired (V3 FinishResult)
FIXED_DIRS = {"m1": "m1_fixed_action", "v13": "static_fixed_action"}
SCHEMA = "splitfusion.static_modes_v13_offline_v1.arm_assessment.v1"
PASS, FAIL, NA, NX = "PASS", "FAIL", "NOT_APPLICABLE", "NOT_EXERCISED"


def _load(path):
    return json.loads(Path(path).read_bytes())


def _one(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    if len(hits) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(hits)}")
    return hits[0]


def _status(value):
    return PASS if value is True else FAIL


def radio_trace_check(cell, prefix):
    """Every commanded row ACKed and equal to the registered FAVORABLE_STABLE prefix at its step index."""
    with (cell / "radio_trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    bad_status = [r["step_index"] for r in rows if r["command_timing_status"] not in
                  ("ACK_ON_TIME", "ACK_LATE", "SKIP_OBSOLETE_NEVER_BURST")]
    bad_value = [r["step_index"] for r in rows if int(r["step_index"]) < len(prefix)
                 and abs(float(r["target_snr_db"]) - prefix[int(r["step_index"])][1]) > 5.1e-7]
    ids = {(r["profile_id"], r["trace_id"], r["seed"]) for r in rows}
    return dict(rows=len(rows), bad_status=bad_status[:10], bad_value=bad_value[:10], identities=sorted(ids),
                passed=bool(rows) and not bad_status and not bad_value and ids == {
                    ("FAVORABLE_STABLE", "GM_V2_FAVORABLE_STABLE_SEED_2026082101", "2026082101")})


def assess(root, *, fmt, expected_fixed=None, launch_rc=None, budget=1000, trace_prefix=None):
    """``expected_fixed`` None => agent arm.  ``fmt`` in {"m1","v13"} selects only file locations and the registered
    budget-stop signature; every rule is the same."""
    root = Path(root)
    capture = root / "capture"
    cell = CP.find_cell(root)
    qualification = _load(capture / "QUALIFICATION_RESULT.json")
    gates = qualification.get("gates") or {}
    phase = (qualification.get("phase6_report_only") or {}).get("gates") or {}
    analysis = _load(capture / "PROFILER_ANALYSIS.json")
    semantics = analysis.get("semantics") or {}
    evidence = _load(_one(capture, "PHASE6_UE_EVIDENCE.json"))
    edge = _load(_one(capture, "edge_report.json"))
    cell_result = _load(cell / "CELL_RESULT.json")
    counters = evidence["counters"]
    acct = CP.account(root)
    child = cell_result.get("child") or {}
    route = child.get("route_detail") or {}
    fixed_arm = expected_fixed is not None
    checks, identity = {}, {}

    # ---- identity
    if fixed_arm:
        summary = _load(cell / FIXED_DIRS[fmt] / "FIXED_ACTION_SUMMARY.json")
        fc = summary["counters"]
        registered = {k: expected_fixed[k] for k in ("mode_id", "q_e4", "action_id", "profile_id",
                                                     "execution_bundle_sha256", "keep_count", "drop_count",
                                                     "family", "quantizer")}
        identity.update(
            summary_identity=_status(all(summary["fixed_action"].get(k) == v for k, v in registered.items())),
            actor_never_called=_status(counters["actor_calls"] == fc["actor_calls"] == 0),
            no_fallback_substitution=_status(counters["fallbacks"] == fc["fallbacks"] == 0),
            engine_identity_checks=_status(fc["executed_mismatch"] == 0 and fc["decisions"] == counters[
                "policy_decisions"] and fc["holds"] == counters["policy_holds"]),
            every_live_compute_span_is_the_fixed_action=_status(set(acct["executed_modes"]) <= {
                (expected_fixed["mode_id"], expected_fixed["q_e4"])}))
    else:
        identity.update(actor_called_once_per_decision=_status(counters["actor_calls"] == counters["policy_decisions"]))

    # ---- infrastructure
    for gate in INFRA_GATES:
        checks[gate] = _status(gates.get(gate))
    for gate in INFRA_PHASE:
        checks[gate] = _status(phase.get(gate))
    checks["reward_semantics"] = _status((semantics.get("reward") or {}).get("passed"))
    checks["receipt_marks"] = _status((semantics.get("receipt_marks") or {}).get("passed"))
    for gate in ACTOR_ONLY:
        value = gates.get(gate) if gate.startswith("G") else phase.get(gate)
        checks[gate] = NA if fixed_arm else _status(value)
    if acct["completed"] > 0:
        checks["P7_FEEDBACK_OVER_DOWNLINK"] = _status(phase.get("P7_FEEDBACK_OVER_DOWNLINK"))
    else:
        checks["P7_FEEDBACK_OVER_DOWNLINK"] = NX
        checks["no_quality_feedback_without_completion"] = _status(len(edge.get("evaluations") or []) == 0)
    checks["completion_accounting_one_to_one"] = _status(not acct["accounting_problems"])
    protocol = []
    if evidence.get("faulted") is not None:
        protocol.append(evidence["faulted"])
    protocol.extend(edge.get("failures") or [])
    early = list(capture.rglob("EDGE_EARLY_GT.json"))
    overflow = None
    for path in early:
        value = _load(path)
        if value.get("error") is not None:
            protocol.append(value["error"])
        counters_e = (value.get("state") or {}).get("evaluator_counters") or {}
        overflow = (overflow or 0) + int(counters_e.get("queue_overflow", 0)) + int(counters_e.get("control_overflow", 0))
    checks["no_protocol_faults"] = _status(not protocol)
    checks["edge_evaluator_no_queue_overflow"] = _status(overflow == 0)
    from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check as gc_check
    checks["gc_and_producer"] = _status(gc_check(root)["passed"])
    cleanup = cell_result.get("cleanup") or {}
    checks["child_cleanup"] = _status(
        child.get("return_code") == 0
        and all((child.get("cleanup") or {}).get(k) is True for k in ("edge_stopped", "map_stopped",
                                                                      "target_snr_restored"))
        and cleanup.get("all_gates_passed") is True and not qualification.get("leftover_processes"))
    stop = (child.get("collector") or {}).get("stop_reason") or child.get("stop_reason")
    transmitted = len(evidence["transmitted_identities"])
    checks["registered_budget_stop"] = _status(
        (stop == "TRANSMITTED_BUDGET_REACHED" and transmitted == budget
         or stop == "DECISION_CYCLE_BOUNDARY" and 1 <= transmitted <= budget)
        and route.get("error") == ROUTE_STOP[fmt])
    holds_ok = counters["policy_holds"] in (counters["policy_decisions"], counters["policy_decisions"] - 1)
    checks["decision_hold_structure"] = _status(holds_ok)
    if fmt == "v13":
        finish = _load(cell / "m3_route" / "FINISH_RESULT.json")
        drain = _load(cell / "m3_route" / "END_OF_ROUTE_DRAIN.json")
        seam = _load(cell / "m3_route" / "CONTROLLER_MODULE_SEAM.json")
        checks["finish_repair"] = _status(finish.get("inner_type") == "bool" and finish.get("cleanup_ok") is True)
        checks["end_of_run_drain_closed_equals_sent"] = _status(
            drain.get("closed_equals_sent") is True and drain.get("missing_terminal_declared_successful") is False
            and drain.get("deadline_extended") is False)
        checks["retained_controller_unsubstituted"] = _status(seam.get("substituted") is None
                                                              and seam.get("unchanged") is True)
        checks["no_retention_artifacts"] = _status(not (root / "paired_retention_v2").exists()
                                                   and not (root / "RETENTION_RESULT.json").exists())
        trace = radio_trace_check(cell, trace_prefix or [])
        checks["commanded_channel_is_registered_trace"] = _status(trace["passed"])
    if launch_rc is None:
        checks["launch_rc_explained"] = NX
    else:
        false_gates = sorted(g for g, v in gates.items() if v is not True)
        explained = (launch_rc == 0 and qualification.get("verdict") == "CAPTURE_VALID") or (
            fixed_arm and launch_rc != 0 and false_gates == ["G8_REWARD_TIMEOUT_SEMANTICS"])
        checks["launch_rc_explained"] = _status(explained)

    identity_failed = any(v == FAIL for v in identity.values())
    infra_failed = any(v == FAIL for v in checks.values())
    verdict = ("INVALID_IDENTITY" if identity_failed else "INFRASTRUCTURE_FAILURE" if infra_failed
               else "VALID_MEASUREMENT")
    latencies = acct["latency_ms"]

    def pct(p):
        if not latencies:
            return None
        import numpy as np
        return float(np.percentile(latencies, p))
    gt_refusals = collections.Counter(d["service_outcome"] for d in acct["per_decision"]
                                      if d["service_outcome"] and "NO_GT" in d["service_outcome"])
    return dict(
        schema=SCHEMA, verdict=verdict,
        subtype=("ZERO_COMPLETION" if verdict == "VALID_MEASUREMENT" and acct["completed"] == 0 else None),
        arm_kind="EXTERNAL_FIXED" if fixed_arm else "DEPLOYED_ACTOR", format=fmt, root=str(root),
        identity=identity, checks=checks,
        frames=dict(offered_budget=budget, transmitted=transmitted, policy_decisions=counters["policy_decisions"],
                    policy_holds=counters["policy_holds"], fallbacks=counters["fallbacks"],
                    actor_calls=counters["actor_calls"], collector_stop=stop, route_error=route.get("error")),
        prediction_completion=dict(
            rule="successful live edge.compute (ok, reward_requested) + edge:prediction_materialized, one-to-one",
            decisions=acct["decisions"], completed=acct["completed"], completed_by_170=acct["completed_by_170"],
            completed_later=acct["completed_later"], no_completed_prediction=acct["no_completed_prediction"],
            no_completion_subcategories=acct["no_completion_subcategories"],
            failed_compute_span_frames=acct["failed_compute_span_frames"],
            latency_ms=dict(p50=pct(50), p95=pct(95), p99=pct(99), n=len(latencies)),
            never_used=["TIMEOUT/STALE timestamps", "failed spans", "map publication", "feedback receipt"]),
        gt_capacity_refusals_reported=dict(gt_refusals),
        controller_outcomes=dict(collections.Counter(r.get("terminal") for r in evidence.get("resolutions", []))),
        retained_generic_qualification=dict(verdict=qualification.get("verdict"), gates=gates, phase6=phase),
        launch_rc=launch_rc, payload_bytes=_payload(cell))


def _payload(cell):
    path = cell / "per_frame_metrics.csv"
    with path.open(newline="") as stream:
        values = sorted(int(r["payload_bytes"]) for r in csv.DictReader(stream)
                        if r.get("prepare_status") == "SENT" and r.get("payload_bytes"))
    if not values:
        return dict(n=0)
    import numpy as np
    return dict(n=len(values), p50=float(np.percentile(values, 50)), p99=float(np.percentile(values, 99)),
                layer="SFD4 application-frame bytes, all transmitted frames")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch-rc", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from . import binding
    expectations = binding.resolve(args.root)
    result = assess(args.root, fmt="v13", expected_fixed=expectations["external_fixed_action"],
                    launch_rc=args.launch_rc, budget=expectations["budget"], trace_prefix=expectations["trace_prefix"])
    result["binding"] = expectations["binding"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(dict(verdict=result["verdict"], subtype=result["subtype"],
                          completion={k: result["prediction_completion"][k] for k in
                                      ("decisions", "completed", "completed_by_170", "no_completed_prediction")}),
                     sort_keys=True))
    return 0 if result["verdict"] == "VALID_MEASUREMENT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
