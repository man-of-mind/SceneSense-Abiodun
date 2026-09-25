"""Preregistered analysis specification for the Run-4 near-capacity sweep.

Everything an analyst could otherwise choose *after* seeing the data is pinned
here, before collection: the feature bins, the estimator, the back-off order,
the minimum support, the exclusion rules, the gate thresholds, and what to do
when an arm turns out to be degenerate.

The estimator is deliberately a binned conditional median / rate -- assumption
light, directly interpretable, and the form that makes the monotonicity gate
checkable by construction rather than by inspecting fitted coefficients.  No
neural policy is trained here and no agent is modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C

# --------------------------------------------------------------------------
# Feature binning. Predeclared; never re-cut after seeing outcomes.
# --------------------------------------------------------------------------

#: Payload is categorical and ordered: exactly the three registered tiers.
PAYLOAD_LEVELS: tuple[int, ...] = tuple(
    sorted(spec["payload_bytes"] for spec in C.EXPECTED_TIERS.values()))

#: Backlog bin edges in bytes. Bin 0 is exactly zero, which is a real and
#: meaningful state (an empty queue), never merged with "small".
BACKLOG_EDGES: tuple[float, ...] = (
    0.0, 1.0, 1e3, 1e4, 1e5, 1e6, 1e7, math.inf)
BACKLOG_BIN_LABELS: tuple[str, ...] = (
    "zero", "lt_1KB", "1KB_10KB", "10KB_100KB", "100KB_1MB", "1MB_10MB", "ge_10MB")

#: MCS bins over table-0 indices 0..28. Width 4, last bin absorbs 28.
MCS_EDGES: tuple[float, ...] = (0, 4, 8, 12, 16, 20, 24, math.inf)
MCS_BIN_LABELS: tuple[str, ...] = (
    "0_3", "4_7", "8_11", "12_15", "16_19", "20_23", "24_28")

#: A bin must hold at least this many FIT observations to be used directly.
MIN_BIN_SUPPORT = 20

#: Back-off order when a bin is under-supported. Strictly coarser each step;
#: the final step is the global FIT marginal. Never an extrapolation, never a
#: fit borrowed from the VALIDATION cells.
BACKOFF_ORDER: tuple[tuple[str, ...], ...] = (
    ("payload", "backlog", "mcs"),
    ("payload", "backlog"),
    ("payload",),
    (),
)
#: For the backlog-only feature set the MCS term is simply absent.
BACKOFF_ORDER_NO_MCS: tuple[tuple[str, ...], ...] = (
    ("payload", "backlog"),
    ("payload",),
    (),
)

FEATURE_SET_BACKLOG_ONLY = "payload_backlog"
FEATURE_SET_WITH_MCS = "payload_backlog_mcs"


def backlog_bin(value: float) -> int:
    if value < 0:
        raise ValueError(f"negative backlog {value}")
    if value == 0:
        return 0
    for index in range(1, len(BACKLOG_EDGES) - 1):
        if value < BACKLOG_EDGES[index + 1]:
            return index
    return len(BACKLOG_BIN_LABELS) - 1


def mcs_bin(value: int) -> int:
    for index in range(len(MCS_BIN_LABELS)):
        if value < MCS_EDGES[index + 1]:
            return index
    return len(MCS_BIN_LABELS) - 1


# --------------------------------------------------------------------------
# Exclusion rules. Missing stays missing.
# --------------------------------------------------------------------------

EXCLUSION_RULES: Mapping[str, str] = {
    "verifier_unmatched": (
        "Decisions whose verifier-only gNB provenance row did not match are "
        "EXCLUDED from model fitting and from gate 7's MCS contrasts. Their "
        "identities and their distribution over cell/tier/block are reported. "
        "They are never imputed, forward-filled, or recorded as mismatches."),
    "missing_mcs": (
        "A decision with no prior round-0 grant, or a stale one, carries an "
        "explicit missing marker and is excluded from every MCS-conditioned "
        "estimate. It is never coerced to 0, which is a real modulation index. "
        "UE-side MCS coverage is separately gated at 100%, so a nonempty "
        "missing set is itself a gate-2 failure, not a silent filter."),
    "missing_backlog": (
        "A decision with no RLC buffer-status tick strictly before its enqueue "
        "is excluded and counted. Never forward-filled, never zero-filled."),
    "boundary_inversion": (
        "A decision whose observation timestamp is not strictly before its own "
        "enqueue is a stop condition, not an exclusion."),
    "clock_bridge": (
        "If the same-event wall/monotonic bridge residual P95 exceeds 1 us the "
        "run stops; the join is not attempted with a degraded bridge."),
}

#: Observed backlog is censored once the RLC buffer hits its ceiling: the
#: recurrence is no longer identifiable there. Run 3 measured the ceiling at
#: 49,984,583 B with 23.3% of decisions within 5% of it, which is exactly the
#: regime this run is designed to avoid.
CEILING_PROXIMITY_FRACTION = 0.95
RUN3_OBSERVED_CEILING_BYTES = 49_984_583


# --------------------------------------------------------------------------
# Gate thresholds. Frozen.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    key: str
    number: int
    statement: str
    thresholds: Mapping[str, Any] = field(default_factory=dict)


GATES: tuple[Gate, ...] = (
    Gate("COMPLETE_CAPTURE", 1,
         "All 12 cells captured and exactly 5,400 terminal outcomes, with "
         "exact decision, terminal and chunk accounting.",
         {"cells": C.EXPECTED_CELLS, "decisions": C.EXPECTED_DECISIONS}),
    Gate("MCS_COVERAGE_AND_PROVENANCE", 2,
         "UE-side MCS coverage is 100%. The verifier-only gNB join reaches at "
         "least 99% unique match coverage in EVERY cell, with zero ambiguity "
         "and zero UE-versus-gNB final-MCS mismatch among matched rows. "
         "Unmatched verifier rows are excluded from fitting and reported.",
         {"ue_mcs_coverage": 1.0, "gnb_match_coverage_per_cell": 0.99,
          "ambiguous": 0, "ue_final_mismatches": 0}),
    Gate("CENSORING_BELOW_1_PERCENT", 3,
         "Queue-ceiling censoring is below 1% of decisions. Otherwise stop and "
         "redesign; do not fit through a censored region.",
         {"max_censored_fraction": 0.01,
          "ceiling_proximity_fraction": CEILING_PROXIMITY_FRACTION}),
    Gate("VALIDATION_NEXT_BACKLOG_ERROR", 4,
         "On blocked VALIDATION cells, next-backlog normalized median absolute "
         "error is at most 10% AND at least 20% better than a "
         "backlog-persistence baseline.",
         {"max_nmae": 0.10, "min_improvement_over_persistence": 0.20}),
    Gate("VALIDATION_LATENCY_ERROR", 5,
         "Per-cell VALIDATION uplink-latency P50 error at most 17 ms and P95 "
         "error at most 34 ms.",
         {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0}),
    Gate("VALIDATION_TRANSPORT_OUTCOME", 6,
         "VALIDATION 170-ms false-success rate at most 5% and Brier score at "
         "most 0.15.",
         {"max_false_success_rate": 0.05, "max_brier": 0.15,
          "budget_ms": C.AGENT_PATH_BUDGET_MS}),
    Gate("MCS_DOES_NOT_HURT_AND_POINTS_THE_RIGHT_WAY", 7,
         "Adding MCS does not worsen the backlog-only VALIDATION Brier by more "
         "than 0.01, and matched near-boundary MCS contrasts have the "
         "physically correct direction.",
         {"max_brier_degradation": 0.01,
          "required_contrast_direction": "higher MCS -> not worse delivery"}),
    Gate("MONOTONICITY", 8,
         "Inside measured support and at fixed other inputs: increasing "
         "payload or backlog never improves predicted delivery or latency, and "
         "increasing MCS never worsens it.",
         {"max_violations": 0}),
)

GATES_BY_KEY: Mapping[str, Gate] = {gate.key: gate for gate in GATES}


# --------------------------------------------------------------------------
# Degenerate-arm rule, declared BEFORE collection.
#
# The registered tiers are 2.25 / 6.49 / 10.38 Mbps. Against the capacity
# re-derived from Run-3 evidence (ADVERSE P50 12.05 Mbps, FAVORABLE P50
# 42.13 Mbps) the ADVERSE arm spans 0.19 / 0.54 / 0.86 of capacity and does
# bracket the knee, while the FAVORABLE arm spans 0.05 / 0.15 / 0.25 and is
# expected to drain with backlog at or near zero throughout.
#
# If an arm is degenerate its persistence baseline error is ~0, so "at least
# 20% better than persistence" is not merely hard but arithmetically
# unreachable. Declaring this now prevents two dishonest outcomes: silently
# pooling the arms so the informative one carries a vacuous one, and relaxing
# the threshold after the fact.
# --------------------------------------------------------------------------

#: An arm is degenerate for a target if the baseline it must beat has
#: essentially no error to beat.
DEGENERATE_BASELINE_EPSILON = 1e-9

DEGENERATE_ARM_RULE = (
    "Gates 4-7 are computed per channel arm and reported per arm, plus pooled. "
    "An arm whose persistence-baseline error is below DEGENERATE_BASELINE_"
    "EPSILON, or whose VALIDATION outcome is constant, is reported as "
    "NON_INFORMATIVE for the affected criterion. NON_INFORMATIVE is neither a "
    "pass nor a fail: it is reported explicitly, it never counts toward a pass, "
    "and the gate verdict rests on the informative arm(s). If NO arm is "
    "informative the gate is INDETERMINATE and the run cannot be called a "
    "kernel acceptance."
)

PASS = "PASS"
FAIL = "FAIL"
NON_INFORMATIVE = "NON_INFORMATIVE"
INDETERMINATE = "INDETERMINATE"


def combine_arm_verdicts(arm_verdicts: Mapping[str, str]) -> str:
    """Apply DEGENERATE_ARM_RULE to per-arm verdicts."""
    values = list(arm_verdicts.values())
    if any(v == FAIL for v in values):
        return FAIL
    informative = [v for v in values if v in (PASS, FAIL)]
    if not informative:
        return INDETERMINATE
    return PASS


# --------------------------------------------------------------------------
# Metrics. Pure functions, unit-tested on synthetic data before collection.
# --------------------------------------------------------------------------

def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def normalized_median_absolute_error(
    predicted: Sequence[float], actual: Sequence[float], *, scale: float
) -> float:
    """Median |error| divided by a predeclared scale.

    The scale is the VALIDATION arm's median observed next backlog, so the
    figure is a fraction of typical queue occupancy rather than of a
    per-observation denominator that explodes near zero.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted/actual length mismatch")
    if scale <= 0:
        raise ValueError("scale must be positive; a zero scale means the arm "
                         "is degenerate and must be reported NON_INFORMATIVE")
    return median([abs(p - a) for p, a in zip(predicted, actual)]) / scale


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("probability/outcome length mismatch")
    if not probabilities:
        raise ValueError("brier score of empty sequence")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(outcomes)


def false_success_rate(
    probabilities: Sequence[float], outcomes: Sequence[int], *, threshold: float = 0.5
) -> float:
    """Fraction of predicted successes that did not succeed.

    This is the safety-relevant direction: promising the controller a delivery
    that never arrived. Defined over predicted successes, and reported as
    undefined rather than 0 when nothing was predicted to succeed.
    """
    predicted_success = [(p, o) for p, o in zip(probabilities, outcomes)
                         if p >= threshold]
    if not predicted_success:
        return float("nan")
    return sum(1 for _, o in predicted_success if not o) / len(predicted_success)


def improvement_over_baseline(model_error: float, baseline_error: float) -> float:
    """Fractional reduction in error. Undefined when the baseline is perfect."""
    if baseline_error <= DEGENERATE_BASELINE_EPSILON:
        return float("nan")
    return (baseline_error - model_error) / baseline_error


def monotonicity_violations(
    predict: Callable[[int, float, int], float],
    *,
    payloads: Sequence[int],
    backlogs: Sequence[float],
    mcs_values: Sequence[int],
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    """Enumerate gate-8 violations over the predeclared support grid.

    ``higher_is_better`` describes the prediction: True for delivery
    probability, False for latency. Increasing payload or backlog must not
    improve it; increasing MCS must not worsen it.
    """
    violations: list[dict[str, Any]] = []

    def worse(new: float, old: float) -> bool:
        return new > old if higher_is_better else new < old

    for mcs in mcs_values:
        for backlog in backlogs:
            for a, b in zip(payloads, payloads[1:]):
                if worse(predict(b, backlog, mcs), predict(a, backlog, mcs)):
                    violations.append({"axis": "payload", "from": a, "to": b,
                                       "backlog": backlog, "mcs": mcs})
        for payload in payloads:
            for a, b in zip(backlogs, backlogs[1:]):
                if worse(predict(payload, b, mcs), predict(payload, a, mcs)):
                    violations.append({"axis": "backlog", "from": a, "to": b,
                                       "payload": payload, "mcs": mcs})
    for payload in payloads:
        for backlog in backlogs:
            for a, b in zip(mcs_values, mcs_values[1:]):
                new, old = predict(payload, backlog, b), predict(payload, backlog, a)
                if (new < old) if higher_is_better else (new > old):
                    violations.append({"axis": "mcs", "from": a, "to": b,
                                       "payload": payload, "backlog": backlog})
    return violations


# --------------------------------------------------------------------------
# Queue recurrence. Stated here so Phase B cannot restate it differently.
# --------------------------------------------------------------------------

def next_backlog(backlog: float, payload: int, service: float,
                 *, backlog_max: float) -> tuple[float, bool]:
    """B_{t+1} = min(Bmax, max(0, B_t + P_t - S_t)), with explicit overflow.

    Returns the next backlog and whether the queue overflowed. Overflow is
    surfaced, never silently clipped away.
    """
    raw = backlog + payload - service
    overflowed = raw > backlog_max
    return min(backlog_max, max(0.0, raw)), overflowed


#: The reward semantics this run will LATER inform. Recorded for continuity and
#: explicitly NOT exercised here: no reward is computed, no agent is fitted, and
#: admission probability is never multiplied into a reward.
INTENDED_REWARD_SEMANTICS = (
    "r = Q_perc - 0.25 * L / 170 on success; r = -1 on delivery failure, "
    "service failure or timeout, where success requires complete UDP "
    "reassembly AND full action-open-to-quality-feedback latency L <= 170 ms. "
    "Held tensors reuse the exact action, request no reward, and still enter "
    "the queue recurrence. NOT trained, computed or calibrated in this task."
)
