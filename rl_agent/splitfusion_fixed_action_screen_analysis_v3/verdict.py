"""Forced-screen verdict for the v3 frontier measurement = amendment-02 verdict, plus the strict v3 checks.

``forced_verdict_v2.evaluate`` (unchanged) is applied to the collected evidence; it already requires every
registered v2 integration check (I1-I11, I9 without its launch clause) and permits exactly two retained gate
failures: G8's intentional forced-vs-proposal replay mismatch and P0 policy coverage (report-only). The v3 check
I12_executed_identity (and the strict I11 computed by ``integration``) must additionally be true. Any other
failure, crash, unknown exit or missing result is a STOP.
"""
from rl_agent.splitfusion_fixed_action_screen_analysis_v2 import forced_verdict_v2 as V2

collect = V2.collect


def evaluate(e):
    base = V2.evaluate(e)
    if base["reasons"] == ["MISSING_RESULTS"]:
        return dict(base, amendment="AMENDMENT_02_CARRIED_FORWARD_V3")
    checks = ((e.get("integration") or {}).get("checks")) or {}
    extra = [f"I:{name}" for name in ("I11_anchor_provenance", "I12_executed_identity") if checks.get(name) is not True]
    reasons = list(base["reasons"]) + [r for r in extra if r not in base["reasons"]]
    conditions = dict(base["conditions"], V3=not extra)
    verdict = "PASS_FORCED_SCREEN" if all(conditions.values()) and not reasons else "STOP"
    return dict(base, verdict=verdict, conditions=conditions, reasons=reasons, amendment="AMENDMENT_02_CARRIED_FORWARD_V3")
