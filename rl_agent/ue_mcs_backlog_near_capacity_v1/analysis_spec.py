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

#: Payload is categorical and ordered: exactly the three tiers the
#: deterministic rule selected against the MEASURED adverse capacity. The
#: levels are therefore resolved from the run's own plan, not hardcoded -- the
#: actions are no longer frozen in the contract.
def payload_levels(tiers: Sequence[Any]) -> tuple[int, ...]:
    """The three payload levels, ascending, from the adopted tiers."""
    levels = tuple(sorted(int(t.payload_bytes) for t in tiers))
    if len(levels) != 3 or len(set(levels)) != 3:
        raise ValueError(f"expected three distinct payload levels, got {levels}")
    return levels

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
# The tiers are selected by the deterministic rule against the ADVERSE channel's
# measured capacity, so the ADVERSE arm brackets its boundary by construction.
# The FAVORABLE arm sees the SAME offered loads on a faster channel, so it is
# expected to drain with backlog at or near zero.
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


# --------------------------------------------------------------------------
# Completed estimators and rules (registered; no post-hoc freedom).
# --------------------------------------------------------------------------

#: EXACT MCS freshness limit. One value, not a candidate list.
#:
#: A round-0 grant older than this is recorded MISSING_STALE and never used.
#: 200 ms is two decision periods at 10 fps, and is the bound under which Run 3
#: measured 100% UE-side MCS coverage in all twelve cells. It is pinned here so
#: the limit cannot be tuned after seeing coverage: if coverage at 200 ms is
#: below 100%, gate 2 FAILS. The limit is not relaxed to rescue it.
MCS_MAX_AGE_MS: float = 200.0

#: Reported as a sensitivity diagnostic only. Never selects the operative limit.
MCS_AGE_SENSITIVITY_REPORT_MS: tuple[float, ...] = (100.0, 150.0, 200.0, 250.0)

#: Clock-bridge refusal threshold. Above this the join is not attempted.
CLOCK_BRIDGE_MAX_RESIDUAL_P95_US: float = 1.0


class ClockBridgeError(RuntimeError):
    """Raised when the same-event wall/monotonic bridge is too loose to join on."""


def require_clock_bridge(residual_p95_us: float, *, cell_id: str) -> float:
    """Refuse a degraded bridge instead of joining through it."""
    if not (residual_p95_us == residual_p95_us):  # NaN
        raise ClockBridgeError(
            f"{cell_id}: clock-bridge residual P95 is undefined; no same-event "
            f"NR_PDCP_TX_SDU rows were available to build the bridge")
    if residual_p95_us > CLOCK_BRIDGE_MAX_RESIDUAL_P95_US:
        raise ClockBridgeError(
            f"{cell_id}: clock-bridge residual P95 {residual_p95_us:.4f} us "
            f"exceeds the registered {CLOCK_BRIDGE_MAX_RESIDUAL_P95_US} us limit; "
            f"refusing to join. Do not widen the limit.")
    return residual_p95_us


# --- Gate 5: the P95 latency estimator, completed ------------------------
#
# A binned conditional MEDIAN cannot produce a P95, so gate 5's P95 arm was
# previously unspecified. It is completed here as a mixture estimator: each FIT
# bin retains its full empirical latency sample, and a validation cell's
# predicted distribution is the mixture of those samples weighted by how often
# that cell actually visits each bin. The predicted P95 is the P95 of the
# mixture. This uses exactly the same bins, support floor and back-off order as
# the median estimator, so the two arms of gate 5 cannot disagree about what the
# model is.

MIXTURE_MIN_SAMPLES = 50


def mixture_percentile(
    bin_samples: Mapping[Any, Sequence[float]],
    bin_weights: Mapping[Any, float],
    q: float,
) -> float:
    """Percentile of the weight-mixture of per-bin empirical samples.

    Uses the *weighted* form of the same linear-interpolation convention as
    :func:`percentile`, so the predicted and observed arms of gate 5 cannot
    disagree about what a percentile means. With equal weights this reduces
    exactly to ``percentile`` on the pooled sample.
    """
    total_weight = sum(w for k, w in bin_weights.items()
                       if w > 0 and bin_samples.get(k))
    if total_weight <= 0:
        raise ValueError("no occupied bin has FIT samples; back off further")

    pairs: list[tuple[float, float]] = []
    for key, weight in bin_weights.items():
        samples = bin_samples.get(key)
        if not samples or weight <= 0:
            continue
        share = (weight / total_weight) / len(samples)
        pairs.extend((float(value), share) for value in samples)
    pairs.sort(key=lambda item: item[0])
    if len(pairs) == 1:
        return pairs[0][0]

    # Type-7 weighted plotting positions: p_i = (C_i - w_i) / (1 - w_i), which
    # for n equal weights gives i/(n-1), i.e. numpy's default.
    positions: list[float] = []
    cumulative = 0.0
    for _, share in pairs:
        cumulative += share
        denominator = 1.0 - share
        positions.append(0.0 if denominator <= 0
                         else (cumulative - share) / denominator)

    target = q / 100.0
    if target <= positions[0]:
        return pairs[0][0]
    if target >= positions[-1]:
        return pairs[-1][0]
    for index in range(1, len(positions)):
        if target <= positions[index]:
            low_pos, high_pos = positions[index - 1], positions[index]
            low_val, high_val = pairs[index - 1][0], pairs[index][0]
            if high_pos == low_pos:
                return high_val
            ratio = (target - low_pos) / (high_pos - low_pos)
            return low_val + ratio * (high_val - low_val)
    return pairs[-1][0]


def latency_errors(
    predicted_p50: float, predicted_p95: float,
    observed: Sequence[float],
) -> dict[str, float]:
    """Per-cell gate-5 errors, in milliseconds."""
    return {
        "p50_error_ms": abs(predicted_p50 - percentile(observed, 50)),
        "p95_error_ms": abs(predicted_p95 - percentile(observed, 95)),
        "observed_p50_ms": percentile(observed, 50),
        "observed_p95_ms": percentile(observed, 95),
        "n": float(len(observed)),
    }


# --- Gate 7: the matching and effect rule, completed ---------------------
#
# "Matched near-boundary MCS contrasts must have the physically correct
# direction" was previously a sentence, not a procedure. Completed:
#
#   MATCH   exactly on (payload level, backlog bin), within one validation
#           channel arm. Exact matching, never a propensity score: there are
#           only three payloads and seven backlog bins, so exact strata exist.
#   NEAR-   only strata whose payload sits within NEAR_BOUNDARY_RATIO of the
#   BOUNDARY measured adverse capacity, because that is the only region where
#           the channel is supposed to decide feasibility.
#   CONTRAST within a stratum, compare the LOW MCS group (bin <= low) against
#           the HIGH MCS group (bin >= high), requiring at least
#           MIN_CONTRAST_SUPPORT observations on each side and a gap of at
#           least MIN_MCS_BIN_GAP bins.
#   EFFECT  observed success-rate difference (high minus low).
#   RULE    the effect must not be negative beyond CONTRAST_NOISE_TOLERANCE.
#           A small negative effect inside the tolerance is reported as NULL,
#           not as a pass and not as a violation.

NEAR_BOUNDARY_RATIO = 0.25
MIN_CONTRAST_SUPPORT = 30
MIN_MCS_BIN_GAP = 2
CONTRAST_NOISE_TOLERANCE = 0.05

CONTRAST_PASS = "CORRECT_DIRECTION"
CONTRAST_NULL = "NULL_WITHIN_TOLERANCE"
CONTRAST_VIOLATION = "WRONG_DIRECTION"


def is_near_boundary(offered_mbps: float, adverse_capacity_mbps: float) -> bool:
    if adverse_capacity_mbps <= 0:
        raise ValueError("adverse capacity must be positive")
    return abs(offered_mbps / adverse_capacity_mbps - 1.0) <= NEAR_BOUNDARY_RATIO


def mcs_contrast(
    low_group: Sequence[int], high_group: Sequence[int],
    *, low_bin: int, high_bin: int,
) -> dict[str, Any]:
    """One matched near-boundary MCS contrast, with its verdict."""
    if high_bin - low_bin < MIN_MCS_BIN_GAP:
        return {"verdict": None, "reason": "MCS_BIN_GAP_TOO_SMALL",
                "low_bin": low_bin, "high_bin": high_bin}
    if len(low_group) < MIN_CONTRAST_SUPPORT or len(high_group) < MIN_CONTRAST_SUPPORT:
        return {"verdict": None, "reason": "INSUFFICIENT_SUPPORT",
                "n_low": len(low_group), "n_high": len(high_group)}
    low_rate = sum(low_group) / len(low_group)
    high_rate = sum(high_group) / len(high_group)
    effect = high_rate - low_rate
    if effect < -CONTRAST_NOISE_TOLERANCE:
        verdict = CONTRAST_VIOLATION
    elif effect <= CONTRAST_NOISE_TOLERANCE:
        verdict = CONTRAST_NULL
    else:
        verdict = CONTRAST_PASS
    return {"verdict": verdict, "effect": effect, "low_rate": low_rate,
            "high_rate": high_rate, "n_low": len(low_group),
            "n_high": len(high_group), "low_bin": low_bin, "high_bin": high_bin}


def gate7_direction_verdict(contrasts: Sequence[Mapping[str, Any]]) -> str:
    """Any wrong-direction contrast fails; no evaluable contrast is not a pass."""
    evaluated = [c for c in contrasts if c.get("verdict")]
    if any(c["verdict"] == CONTRAST_VIOLATION for c in evaluated):
        return FAIL
    if not evaluated:
        return INDETERMINATE
    return PASS
