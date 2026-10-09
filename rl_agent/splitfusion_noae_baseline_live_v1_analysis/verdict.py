"""Registered validity verdict for the ONE action-0 baseline capture (pure over durable evidence; CPU only).

VALID_BASELINE_MEASUREMENT iff every condition below holds; otherwise STOP_INVALID with explicit reasons.

Base = the amendment-02 forced-screen verdict carried forward by v3 (``splitfusion_fixed_action_screen_analysis_v3.verdict``)
over the v3 integration (I1-I12), i.e. route/lifecycle (R), only G8 false among retained gates (G), reward/receipt/P3-P5
(C), actor-replay failure only the intended proposal-vs-forced comparison (A), forced action on every decision/hold and
independent proposal replay (F), integration I1-I12 (I), launch exit 0 or the retained "Capture runner exit 1" (X);
P0 policy coverage report-only. Two registered adaptations, both declared before launch:

  1. Independent proposal replay uses the FULL Run-5 actor that actually runs in shadow
     (``splitfusion_run5_full_live_v1.actor.load_registered_actor``) instead of the original Run-5 audit actor that the
     v2 helper hard-codes; the v2 helper is substituted for this analysis process only, and restored.
  2. I8_bounded is split. I8a (binding, resource safety): every sampled GT-service occupancy within the configured limits.
     I8b (REPORTED OUTCOME, not validity): GT-service refusal count and edge evaluator queue-overflow count. Under a
     ~3.6 MB/frame uplink these are delivery outcomes of the baseline, like timeliness, missing replies and coverage.
Plus the gc_v2 GC-trace/producer check (adopted policy only in prep, producer v2 constructed, no safety stop).

Never inferred from the exit code alone. Low timeliness, missing/late replies, reassembly expiry, STALE terminals,
fallbacks and policy coverage are reported, not invalidating. A post-assignment send exception (retained
INFRASTRUCTURE_FAULT -> P8 false), identity conflicts, executed-action mismatch, inexact rescore, leftovers or failed
restoration are genuine faults -> STOP_INVALID.
"""
import contextlib
import json
from pathlib import Path

from rl_agent.splitfusion_fixed_action_screen_analysis_v2 import analyze as A2
from rl_agent.splitfusion_fixed_action_screen_analysis_v3 import integration as I3, verdict as V3

VERDICT_ID = "NOAE_BASELINE_A0_VERDICT_V1"


def forced_replay_full(arm, decisions_by_frame):
    """``A2.forced_replay`` with the full Run-5 shadow actor (identical comparison logic)."""
    import collections
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    actor = load_registered_actor()
    _, _, spans = A2.profile(arm.cell / "profiler/ue.profile.jsonl")
    logged = {r["frame_id"]: r for r in A2.load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")}
    mismatches, replayed = [], 0
    for span in spans:
        fid = int(span["frame_id"])
        decision = actor.act_on_vector(tuple(float(v) for v in span["features"]))
        replayed += 1
        got = (decision.mode_id, decision.q_e4, decision.state_sha256, decision.heads_sha256)
        recorded = (span.get("decided_mode_id"), span.get("decided_q_e4"), span.get("state_sha256"), span.get("heads_sha256"))
        p = (logged.get(fid) or {}).get("proposed") or {}
        proposal = (p.get("proposed_mode_id"), p.get("proposed_q_e4"), p.get("state_sha256"), p.get("heads_sha256"))
        executed = (decisions_by_frame.get(fid) or {})
        if got != recorded or got != proposal or (executed.get("mode_id"), executed.get("q_e4")) != (arm.forced["mode_id"], arm.forced["q_e4"]):
            mismatches.append(dict(frame_id=fid, replay=got, span=recorded, logged=proposal,
                                   executed=(executed.get("mode_id"), executed.get("q_e4"))))
    return dict(replayed=replayed, mismatches=len(mismatches), examples=mismatches[:5], actor="FULL_RUN5_SEED43_U10000",
                proposal_distribution=dict(collections.Counter(f"{r['proposed']['proposed_mode_id']}/{r['proposed']['proposed_q_e4']}"
                                                               for r in logged.values() if r.get("proposed"))))


@contextlib.contextmanager
def full_actor_replay():
    original = A2.forced_replay
    A2.forced_replay = forced_replay_full
    try:
        yield
    finally:
        A2.forced_replay = original


def split_i8(arm):
    service = arm.j("numeric_gt/service/SERVICE_RESULT.json") or {}
    samples = arm.j("numeric_gt/service/QUEUE_SAMPLES.json") or []
    edge_doc = arm.j("profiler/edge/numeric_runtime/EDGE_EARLY_GT.json") or {}
    limits = service.get("limits") or {}
    i8a = bool(limits) and all(s["prep_retained"] <= limits.get("prep_jobs", 0) and s["scorer_slots"] <= limits.get("scorer_jobs", 0)
                               and s["freeze_queue"] <= limits.get("freeze_queue", 0) for s in samples)
    counters = (edge_doc.get("state") or {}).get("evaluator_counters") or {}
    return i8a, dict(gt_service_refusals=len(service.get("refusals") or []), refusals=service.get("refusals"),
                     edge_evaluator_queue_overflow=counters.get("queue_overflow"), queue_samples=len(samples))


def integration(arm_dir, launch_returncode=None, do_rescore=True):
    arm = A2.Arm(arm_dir)
    with full_actor_replay():
        d = I3.integration(arm, launch_returncode, do_rescore)
    if arm.cell is None:
        return d
    i8a, i8b = split_i8(arm)
    checks = dict(d["checks"])
    checks.pop("I8_bounded", None)
    checks["I8a_gt_service_within_limits"] = i8a
    return dict(d, checks=checks, passed=all(checks.values()), i8b_reported_outcomes=i8b,
                retained_i8_bounded=d["checks"].get("I8_bounded"))


def gc_check(arm_dir):
    """gc_v2 GC-trace + producer checks against this launcher's contract (same rules)."""
    from rl_agent.splitfusion_run5_full_live_gc_analysis_v1.gc_check import check as base
    from rl_agent.splitfusion_noae_baseline_live_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_producer_v2.source_v2 import package_sources
    arm_dir = Path(arm_dir)
    result = base(arm_dir)
    reg = json.loads((arm_dir / "registration.json").read_bytes())
    producer = reg["gt_gc"]["producer"]
    trace_dir = arm_dir / C.TRACE_DIR

    def load(name):
        path = trace_dir / name
        return json.loads(path.read_bytes()) if path.exists() else {}
    prep, sub = load("prep_gc_trace.json"), load("service_substitution.json")
    pc = dict(constructed_registered_class=prep.get("producer_constructed") == C.PRODUCER_CLASS[producer],
              switch_restored=prep.get("producer_switch_restored") is True,
              prep_received_registered_producer=any(s.get("role") == "prep" and s.get("producer") == producer
                                                    for s in sub.get("substitutions", [])),
              score_received_v1=any(s.get("role") == "score" and s.get("producer") == "v1" for s in sub.get("substitutions", [])),
              v2_sources_equal_main=prep.get("producer_v2_sources") == package_sources())
    result["producer"] = dict(producer=producer, checks=pc, passed=all(pc.values()), constructed=prep.get("producer_constructed"))
    result["passed"] = bool(result["passed"] and result["producer"]["passed"])
    return result


def evaluate(evidence, integ, gc):
    """Pure decision: v3/amendment-02 forced verdict with the registered I8 split, plus the GC/producer check."""
    base = V3.evaluate(evidence)
    reasons = [r for r in base["reasons"] if r != "I:I8_bounded"]
    if base["reasons"] == ["MISSING_RESULTS"]:
        reasons = ["MISSING_RESULTS"]
    checks = (integ or {}).get("checks") or {}
    if checks.get("I8a_gt_service_within_limits") is not True:
        reasons.append("I:I8a_gt_service_within_limits")
    conditions = dict(base.get("conditions") or {})
    if "I" in conditions:
        conditions["I"] = not any(r.startswith("I:") for r in reasons)
    conditions["GC"] = bool((gc or {}).get("passed"))
    if not conditions["GC"]:
        reasons.append("GC:trace_or_producer")
    valid = bool(conditions) and all(conditions.values()) and not reasons
    return dict(verdict_id=VERDICT_ID, verdict="VALID_BASELINE_MEASUREMENT" if valid else "STOP_INVALID",
                conditions=conditions, reasons=reasons, forced_screen_base=base,
                reported_outcomes=dict(i8b=(integ or {}).get("i8b_reported_outcomes"),
                                       p0_policy_coverage=base.get("p0_policy_coverage"),
                                       retained_verdict=base.get("retained_verdict"), retained_g8=base.get("retained_g8")))


def run(arm_dir, launch_returncode, launch_log):
    integ = integration(arm_dir, launch_returncode=None)          # integration evaluated without the launch clause
    evidence = V3.collect(arm_dir, launch_returncode=launch_returncode, launch_log=launch_log, integration_result=integ)
    gc = gc_check(arm_dir)
    out = evaluate(evidence, integ, gc)
    out.update(integration=integ, gc_trace=gc, launch_returncode=launch_returncode)
    return out


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--launch-returncode", type=int, required=True)
    parser.add_argument("--launch-log", required=True)
    args = parser.parse_args()
    value = run(args.arm, args.launch_returncode, args.launch_log)
    print(json.dumps(value, indent=1, default=str))
    sys.exit(0 if value["verdict"] == "VALID_BASELINE_MEASUREMENT" else 3)
