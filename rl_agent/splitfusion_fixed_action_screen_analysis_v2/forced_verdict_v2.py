"""Forced-screen verdict for amendment 02: P0 policy coverage report-only (diagnostic screen only).

Identical to amendment 01 (``forced_verdict.py``, unchanged) except condition R: instead of requiring the
aggregate cell status PASSED, every individual cell condition is checked, and the ONLY Phase-6 gate allowed
to be false is P0_POLICY_COVERAGE. The recorded cell status must equal what the retained rule derives from
the gates (PASSED when all true; INCONCLUSIVE_OR_FAILED when only P0 is false), so any other failure source
still stops. The only permitted retained gate failures are therefore: (1) G8's intentional forced-action vs
actor-proposal replay mismatch (conditions G/C/A unchanged) and (2) P0 policy coverage.
"""
from . import forced_verdict as V1

PHASE6_BINDING = ("P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING", "P5_NO_ACTOR_AFTER_REFUSAL",
                  "P6_RESTORE_COLD", "P7_FEEDBACK_OVER_DOWNLINK", "P8_NO_INFRASTRUCTURE_FAULT")
PHASE6_ALL = PHASE6_BINDING + ("P0_POLICY_COVERAGE",)
collect = V1.collect


def evaluate(e):
    base = V1.evaluate(e)
    if base["reasons"] == ["MISSING_RESULTS"]:
        return dict(base, amendment="AMENDMENT_02")
    reasons = [r for r in base["reasons"] if r != "R:cell_passed"]
    q, cell = e["qualification"], e["cell_result"]
    child = cell.get("child") or {}
    phase6 = ((q.get("phase6_report_only") or {}).get("gates")) or {}
    extra = []

    def need(name, ok):
        if not ok:
            extra.append("R:" + name)
        return ok
    expected_status = ("PASSED" if all(phase6.get(k) is True for k in PHASE6_ALL)
                       else "INCONCLUSIVE_OR_FAILED" if all(phase6.get(k) is True for k in PHASE6_BINDING)
                       else "FAILED")
    r_individual = all([
        need("no_cell_error", not cell.get("error")),
        need("no_child_error", not child.get("error")),
        need("phase6_gate_set_known", set(phase6) == set(PHASE6_ALL)),
        need("phase6_binding_gates_true", all(phase6.get(k) is True for k in PHASE6_BINDING)),
        need("cell_status_consistent_with_gates", cell.get("status") == expected_status
             and (q.get("phase6_report_only") or {}).get("status") == expected_status)])
    r_other = not any(r.startswith("R:") for r in reasons)
    conditions = dict(base["conditions"], R=r_individual and r_other)
    reasons += extra
    verdict = "PASS_FORCED_SCREEN" if all(conditions.values()) and not reasons else "STOP"
    return dict(base, verdict=verdict, conditions=conditions, reasons=reasons, amendment="AMENDMENT_02",
                p0_policy_coverage=phase6.get("P0_POLICY_COVERAGE"), cell_status=cell.get("status"),
                p0_report_only=True)
