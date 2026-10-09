"""Phase 6 result reporting under prospective addendum 2 (option c).

Pure and read-only: derives reporting fields from the UE evidence document
that ``phase6_ue_runtime_v2`` already writes. It changes no gate, verdict,
reward, timeout, state or session semantics.

* Every reward-requested decision is counted.
* Excluded outcomes (``learning_included = False``: no eligible GT, GT
  unavailable, evaluator exception/overflow, infrastructure fault) keep their
  frozen semantics: no reward value, no zero/neutral substitute, no timeout
  substitution, no stale previous outcome. They are counted, rated and
  histogrammed by terminal and evaluator reason, and each forces the genesis
  session rollover that is reported.
* Reward mean and success rate are computed only over non-excluded
  resolutions and are labelled conditional and non-gated.
* A P0-P8 PASS is a systems-integration PASS, never a policy-performance PASS.

Importing this module performs no I/O.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

ADDENDUM_ID = "PHASE6_PROSPECTIVE_ADDENDUM_2_OPTION_C"
CLAIM_SCOPE = "SYSTEMS_INTEGRATION_QUALIFICATION_ONLY"
CONDITIONAL_LABEL = (
    "CONDITIONAL_ON_NON_EXCLUDED_OUTCOMES__EXCLUDES_NO_ELIGIBLE_GT_GT_UNAVAILABLE_"
    "EVALUATOR_AND_INFRASTRUCTURE_FAULTS__REPORTED_NOT_GATED"
)
PASS_STATEMENT = (
    "P0-P8 PASS is a systems-integration PASS, not a policy-performance PASS. "
    "Reward mean and success rate are conditional on eligible GT (non-excluded "
    "outcomes) and are not gated."
)
NO_POLICY_CLAIM = "No policy-performance claim is made by this run."
UNATTRIBUTED = "UNATTRIBUTED"


def _decision_key(session_uuid: Any, decision_seq: Any) -> tuple[str, int]:
    return str(session_uuid), int(decision_seq)


def _reward_frames(
    frames: Iterable[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], int], int]:
    """First reward frame per decision, plus the count of extra ones (P3's job)."""
    out: dict[tuple[str, int], int] = {}
    extra = 0
    for frame in frames:
        if frame.get("reward_requested"):
            key = _decision_key(frame.get("session_uuid", ""), frame["decision_seq"])
            if key in out:
                extra += 1
                continue
            out[key] = int(frame["frame_id"])
    return out, extra


def _accepted_reasons(rows: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    reasons: dict[int, str] = {}
    for row in rows:
        if row.get("class") == "ACCEPTED":
            reasons[int(row["frame_id"])] = str(row.get("reason") or UNATTRIBUTED)
    return reasons


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def result_summary(ue: Mapping[str, Any]) -> dict[str, Any]:
    """Option-(c) reporting fields from one ``PHASE6_UE_EVIDENCE.json``."""
    reward_frames, extra_reward_frames = _reward_frames(ue.get("frames") or ())
    reasons = _accepted_reasons(ue.get("feedback_rows") or ())
    resolutions = list(ue.get("resolutions") or ())
    counters = dict(ue.get("counters") or {})
    sessions = list(ue.get("sessions") or ())

    included, excluded = [], []
    for resolution in resolutions:
        (included if resolution.get("learning_included") else excluded).append(resolution)
    histogram: dict[str, int] = {}
    for resolution in excluded:
        identity = resolution.get("identity") or {}
        frame = reward_frames.get(_decision_key(identity.get("session_uuid", ""),
                                                identity.get("decision_seq", -1)))
        reason = reasons.get(frame, UNATTRIBUTED) if frame is not None else UNATTRIBUTED
        key = f"{resolution.get('terminal')}:{reason}"
        histogram[key] = histogram.get(key, 0) + 1
    rewards = [float(r["reward"]) for r in included]
    successes = sum(1 for r in included if r.get("terminal") == "SUCCESS")
    rollovers = counters.get("session_rollovers")
    return {
        "addendum_id": ADDENDUM_ID,
        "claim_scope": CLAIM_SCOPE,
        "policy_performance_claim": False,
        "statement": PASS_STATEMENT,
        "reward_requested_decisions": len(reward_frames),
        "resolved_decisions": len(resolutions),
        "open_at_stop": len(reward_frames) - len(resolutions),
        "eligible_quality_denominator": len(included),
        "eligible_quality_denominator_definition": (
            "resolutions with learning_included=True (SUCCESS, TIMEOUT, registered "
            "delivery/service failures); the denominator of the conditional reward "
            "mean and success rate"
        ),
        "excluded_count": len(excluded),
        "excluded_rate": _rate(len(excluded), len(resolutions)),
        "excluded_rate_definition": "excluded_count / resolved_decisions",
        "exclusion_reason_histogram": dict(sorted(histogram.items())),
        "integrity": {
            # Reporting never changes a verdict; these are expected to be 0.
            "extra_reward_frames": extra_reward_frames,
            "excluded_with_reward_value": sum(
                1 for r in excluded if r.get("reward") is not None),
        },
        "session_rollovers": rollovers,
        "sessions": len(sessions),
        "sessions_consistent_with_rollovers": (
            rollovers is not None and len(sessions) == int(rollovers) + 1),
        "conditional_reward_mean": {
            "value": sum(rewards) / len(rewards) if rewards else None,
            "n": len(rewards),
            "label": CONDITIONAL_LABEL,
            "gated": False,
        },
        "conditional_success_rate": {
            "value": _rate(successes, len(included)),
            "successes": successes,
            "n": len(included),
            "label": CONDITIONAL_LABEL,
            "gated": False,
        },
    }


def render_markdown(evaluation: Mapping[str, Any]) -> str:
    """Human-readable result summary; the scope statement always leads."""
    summary = evaluation["result_summary"]
    gates = evaluation["gates"]
    verdict = evaluation["verdict"]
    scope = ("SYSTEMS-INTEGRATION PASS (not a policy-performance PASS)"
             if verdict == "PASSED" else f"{verdict} (systems-integration qualification)")
    lines = [
        "# Run-4 Phase-6 result summary",
        "",
        f"**Verdict:** {scope}",
        "",
        f"> {PASS_STATEMENT} {NO_POLICY_CLAIM}",
        f"> Governing decision: `{ADDENDUM_ID}`.",
        "",
        "## Gates (P0-P8)",
        "",
        "| Gate | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in sorted(gates.items())],
        "",
        "## Decisions and exclusions (reported, not gated)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Reward-requested decisions | {summary['reward_requested_decisions']} |",
        f"| Resolved decisions | {summary['resolved_decisions']} |",
        f"| Open at stop | {summary['open_at_stop']} |",
        f"| Eligible-quality denominator | {summary['eligible_quality_denominator']} |",
        f"| Excluded count | {summary['excluded_count']} |",
        f"| Excluded rate | {_fmt(summary['excluded_rate'])} |",
        f"| Session rollovers | {summary['session_rollovers']} |",
        f"| Sessions | {summary['sessions']} |",
        "",
        "Exclusion reasons (terminal:evaluator reason):",
        "",
        *([f"- `{k}`: {v}" for k, v in summary["exclusion_reason_histogram"].items()]
          or ["- none"]),
        "",
        "## Conditional reward statistics (conditional on eligible GT; NOT gated)",
        "",
        f"- Conditional reward mean: {_fmt(summary['conditional_reward_mean']['value'])} "
        f"(n = {summary['conditional_reward_mean']['n']})",
        f"- Conditional success rate: {_fmt(summary['conditional_success_rate']['value'])} "
        f"({summary['conditional_success_rate']['successes']}/"
        f"{summary['conditional_success_rate']['n']})",
        f"- Label: `{CONDITIONAL_LABEL}`",
        "",
        "These statistics exclude every excluded outcome and are not a policy-level "
        "expectation.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


__all__ = [
    "ADDENDUM_ID",
    "CLAIM_SCOPE",
    "CONDITIONAL_LABEL",
    "PASS_STATEMENT",
    "result_summary",
    "render_markdown",
]
