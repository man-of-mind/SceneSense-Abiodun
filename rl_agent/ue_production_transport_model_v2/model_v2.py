#!/usr/bin/env python3
"""Phase C/D: one small monotone two-part transport model, grouped-CV selected.

Two heads, fitted on strictly causal decision-frame rows only.

**Deadline head** - the full sent population::

    p_on_time = sigmoid(w0 - w1*backlog_scaled - w2*bytes_mb + w3*mcs_norm)

with ``w1, w2, w3 >= 0`` enforced by projection, so the probability is
non-increasing in backlog, non-increasing in bytes and non-decreasing in MCS.

**Conditional-latency head** - only rows that completed within 170 ms::

    latency_ms = c0 + c_backlog*backlog_bytes + c_bytes*wire_bytes
                    - c_mcs*mcs_norm

with every coefficient sign-constrained.  ``c_backlog`` and ``c_bytes`` are
**separate**; no shared ``(backlog + bytes)`` coefficient is used.

This is not survivor bias: the deadline head carries the complete population,
and every row the deadline head assigns to failure still receives exactly the
registered ``-1``.  The conditional head is only ever consulted for rows the
deadline head has already placed on the on-time branch.

The model family and seed are fixed.  Selection is grouped
leave-one-whole-FIT-cell-out cross-validation; no open-ended search is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import contract_v2 as C2


MODEL_SCHEMA = "scenesense.production_transport_model.v2"

# Fixed optimizer settings; part of the frozen family, never tuned per fold.
DEADLINE_ITERATIONS = 4000
DEADLINE_LEARNING_RATE = 0.5
LATENCY_ITERATIONS = 20000
LATENCY_LEARNING_RATE = 0.5
BYTES_SCALE = 1e6            # wire bytes -> megabytes, for conditioning only
DEADLINE_MS = 170.0          # bounded latency head range
LOGIT_EPSILON_MS = 1e-3      # keeps the logit finite at the endpoints
MIN_QUEUE_BIN_SUPPORT = 10
MCS_SCALE = 28.0


# The MCS binning is imported unchanged from the already-frozen registration;
# it is not re-chosen here.
from rl_agent.ue_mcs_backlog_run4_analysis_v1 import contract as FROZEN

MCS_EDGES = FROZEN.MCS_EDGES


def mcs_bin(value: float) -> int:
    import bisect
    return max(0, bisect.bisect_right(list(MCS_EDGES), float(value)) - 1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class ModelV2Error(RuntimeError):
    """A model invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelV2Error(message)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class Features:
    backlog_scaled: np.ndarray
    backlog_bytes: np.ndarray
    bytes_mb: np.ndarray
    wire_bytes: np.ndarray
    mcs_norm: np.ndarray


def build_features(rows: Sequence[Mapping[str, Any]]) -> Features:
    backlog_bytes = np.array(
        [float(row["pre_enqueue_backlog_bytes"]) for row in rows])
    wire_bytes = np.array([float(row["action_wire_bytes"]) for row in rows])
    mcs = np.array([float(row["prior_ul_mcs"]) for row in rows])
    backlog_scaled = np.array(
        [C2.backlog_scaled(float(row["pre_enqueue_backlog_bytes"]))
         for row in rows])
    return Features(
        backlog_scaled=backlog_scaled, backlog_bytes=backlog_bytes,
        bytes_mb=wire_bytes / BYTES_SCALE, wire_bytes=wire_bytes,
        mcs_norm=mcs / MCS_SCALE)


@dataclass(frozen=True, slots=True)
class DeadlineHead:
    """Sign-constrained logistic head over the full sent population."""

    w0: float
    w_backlog: float          # >= 0, enters negatively
    w_bytes: float            # >= 0, enters negatively
    w_mcs: float              # >= 0, enters positively

    def probability(self, features: Features) -> np.ndarray:
        z = (self.w0
             - self.w_backlog * features.backlog_scaled
             - self.w_bytes * features.bytes_mb
             + self.w_mcs * features.mcs_norm)
        return _sigmoid(z)

    def to_dict(self) -> dict[str, float]:
        return {"w0": self.w0, "w_backlog": self.w_backlog,
                "w_bytes": self.w_bytes, "w_mcs": self.w_mcs}


@dataclass(frozen=True, slots=True)
class LatencyHead:
    """Bounded monotone conditional on-time latency head.

    Generatively coherent: the output is a *bounded* transform, so it lies
    strictly inside ``(0, DEADLINE_MS]`` by construction rather than being
    clipped after the fact.

        latency_ms = 170 * sigmoid(z)
        z = z0 + a*backlog_scaled + b*bytes_mb - c*mcs_norm

    ``sigmoid`` is strictly increasing and maps to ``(0, 1)``, so latency is
    strictly positive and strictly below the deadline.  With ``a, b, c >= 0``
    it is non-decreasing in backlog, non-decreasing in bytes and
    non-increasing in MCS.  ``a`` and ``b`` are separate coefficients; no
    shared ``(backlog + bytes)`` term exists.
    """

    z0: float
    a_backlog: float          # >= 0
    b_bytes: float            # >= 0
    c_mcs: float              # >= 0

    def _z(self, features: "Features") -> np.ndarray:
        return (self.z0
                + self.a_backlog * features.backlog_scaled
                + self.b_bytes * features.bytes_mb
                - self.c_mcs * features.mcs_norm)

    def latency_ms(self, features: "Features") -> np.ndarray:
        return DEADLINE_MS * _sigmoid(self._z(features))

    def to_dict(self) -> dict[str, float]:
        return {"z0": self.z0, "a_backlog": self.a_backlog,
                "b_bytes": self.b_bytes, "c_mcs": self.c_mcs}


@dataclass(frozen=True, slots=True)
class QueueTransitionHead:
    """Causal queue transition: B_next = max(0, B + ingress - service(MCS)).

    ``service(MCS)`` is the radio's per-cycle *capacity*, estimated only from
    FIT transitions whose successor backlog is strictly positive.  Those are
    exactly the intervals in which the queue did not drain, so they are the
    ones that reveal capacity; a drained interval only shows that capacity was
    at least the offered bytes, which is not binding.  The estimate is made
    non-decreasing in the MCS bin by weighted pool-adjacent-violators.

    The measured successor backlog is the fit *target*.  No future
    measurement, realized ingress or realized service is ever a predictor.
    """

    service_by_mcs_bin: Mapping[int, float]
    global_service_bytes: float
    support: Mapping[str, Any]

    def service_bytes(self, mcs: float) -> tuple[float, str]:
        key = mcs_bin(mcs)
        value = self.service_by_mcs_bin.get(key)
        if value is not None:
            return value, f"MCS_BIN{key}"
        return self.global_service_bytes, "GLOBAL"

    def next_backlog_bytes(
        self, *, backlog_bytes: float, ingress_bytes: float, mcs: float,
    ) -> tuple[float, str]:
        service, source = self.service_bytes(mcs)
        return max(0.0, backlog_bytes + ingress_bytes - service), source

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_by_mcs_bin": {str(k): v for k, v
                                   in sorted(self.service_by_mcs_bin.items())},
            "global_service_bytes": self.global_service_bytes,
            "support": dict(self.support),
            "form": "B_next = max(0, B + deterministic_ingress - service(MCS))",
        }


def fit_queue_transition_head(
    rows: Sequence[Mapping[str, Any]],
) -> QueueTransitionHead:
    buckets: dict[int, list[float]] = {}
    uncensored: list[float] = []
    censored = 0
    used = 0
    for row in rows:
        if not row.get("has_successor"):
            continue
        successor = row["successor_backlog_bytes"]
        if successor is None:
            continue
        backlog = float(row["pre_enqueue_backlog_bytes"])
        ingress = float(row["deterministic_action_ingress_bytes"])
        if successor <= 0:
            censored += 1                     # drained; capacity not revealed
            continue
        service = backlog + ingress - float(successor)
        if service <= 0:
            continue
        used += 1
        uncensored.append(service)
        buckets.setdefault(mcs_bin(row["prior_ul_mcs"]), []).append(service)
    require(bool(uncensored),
            "no uncensored FIT transitions reveal the queue service capacity")

    ordered = sorted(bucket for bucket, values in buckets.items()
                     if len(values) >= MIN_QUEUE_BIN_SUPPORT)
    medians = [statistics.median(buckets[bucket]) for bucket in ordered]
    weights = [len(buckets[bucket]) for bucket in ordered]
    projected = _pava_non_decreasing(medians, weights)
    table = {bucket: value for bucket, value in zip(ordered, projected)}
    return QueueTransitionHead(
        service_by_mcs_bin=table,
        global_service_bytes=statistics.median(uncensored),
        support={"transitions_used": used, "censored_drained": censored,
                 "bins": {str(k): len(v) for k, v in sorted(buckets.items())}})


def _pava(values: Sequence[float], weights: Sequence[float],
          *, non_decreasing: bool) -> list[float]:
    """Weighted pool-adjacent-violators returning one value per input."""
    blocks: list[list[float]] = []      # [value, weight, count]
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight), 1])
        while len(blocks) > 1 and (
            (non_decreasing and blocks[-2][0] > blocks[-1][0])
            or (not non_decreasing and blocks[-2][0] < blocks[-1][0])
        ):
            vb, wb, cb = blocks.pop()
            va, wa, ca = blocks.pop()
            total = wa + wb
            blocks.append([(va * wa + vb * wb) / total, total, ca + cb])
    out: list[float] = []
    for value, _weight, count in blocks:
        out.extend([value] * count)
    assert len(out) == len(values)
    return out


def _pava_non_decreasing(values, weights):
    return _pava(values, weights, non_decreasing=True)


def _pava_non_increasing(values, weights):
    return _pava(values, weights, non_decreasing=False)


def fit_deadline_head(rows: Sequence[Mapping[str, Any]]) -> DeadlineHead:
    features = build_features(rows)
    y = np.array([1.0 if row["completed_within_deadline"] in (True, "True")
                  else 0.0 for row in rows])
    x = np.stack([np.ones_like(y), -features.backlog_scaled,
                  -features.bytes_mb, features.mcs_norm], axis=1)
    rng = np.random.default_rng(C2.MODEL_SEED)
    weights = rng.normal(0.0, 0.01, size=4)
    weights[1:] = np.abs(weights[1:])
    n = len(y)
    for _ in range(DEADLINE_ITERATIONS):
        z = x @ weights
        p = _sigmoid(z)
        gradient = x.T @ (p - y) / n
        weights -= DEADLINE_LEARNING_RATE * gradient
        weights[1:] = np.maximum(weights[1:], 0.0)      # projection
    return DeadlineHead(w0=float(weights[0]), w_backlog=float(weights[1]),
                        w_bytes=float(weights[2]), w_mcs=float(weights[3]))


def fit_latency_head(rows: Sequence[Mapping[str, Any]]) -> LatencyHead:
    """Exact bounded monotone fit, solved in logit space.

    The bounded head is ``latency = 170 * sigmoid(z)``.  Rather than running a
    non-convex search on the saturating sigmoid, the on-time target is mapped
    into logit space once,

        y_logit = log(y / (170 - y)),

    where the model is linear.  The sign-constrained linear problem is convex
    and is solved exactly by bounded least squares, so the fit is
    deterministic and cannot diverge.  ``sigmoid`` is strictly increasing, so
    monotonicity in logit space is monotonicity in milliseconds.
    """
    from scipy.optimize import lsq_linear

    on_time = [row for row in rows
               if row["completed_within_deadline"] in (True, "True")]
    require(len(on_time) >= 50,
            f"conditional head needs >=50 on-time rows, got {len(on_time)}")
    features = build_features(on_time)
    y = np.array([float(row["transport_latency_ns"]) / 1e6 for row in on_time])
    require(bool(np.all((y > 0) & (y <= DEADLINE_MS))),
            "on-time targets must lie inside (0, deadline]")
    # Keep the logit finite at the closed upper endpoint.
    y_clamped = np.clip(y, LOGIT_EPSILON_MS, DEADLINE_MS - LOGIT_EPSILON_MS)
    y_logit = np.log(y_clamped / (DEADLINE_MS - y_clamped))

    design = np.stack([np.ones_like(y), features.backlog_scaled,
                       features.bytes_mb, -features.mcs_norm], axis=1)
    lower = np.array([-np.inf, 0.0, 0.0, 0.0])
    upper = np.array([np.inf, np.inf, np.inf, np.inf])
    solution = lsq_linear(design, y_logit, bounds=(lower, upper),
                          method="trf", tol=1e-12, max_iter=500)
    weights = solution.x
    return LatencyHead(z0=float(weights[0]), a_backlog=float(weights[1]),
                       b_bytes=float(weights[2]), c_mcs=float(weights[3]))


@dataclass(frozen=True, slots=True)
class TwoPartModel:
    deadline: DeadlineHead
    latency: LatencyHead
    queue: "QueueTransitionHead"
    support: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"deadline_head": self.deadline.to_dict(),
                "latency_head": self.latency.to_dict(),
                "queue_head": self.queue.to_dict(),
                "support": dict(self.support)}


def fit_model(rows: Sequence[Mapping[str, Any]]) -> TwoPartModel:
    backlogs = [float(row["pre_enqueue_backlog_bytes"]) for row in rows]
    wires = [float(row["action_wire_bytes"]) for row in rows]
    mcs = [float(row["prior_ul_mcs"]) for row in rows]
    return TwoPartModel(
        deadline=fit_deadline_head(rows), latency=fit_latency_head(rows),
        queue=fit_queue_transition_head(rows),
        support={
            "rows": len(rows),
            "min_backlog_bytes": min(backlogs),
            "max_backlog_bytes": max(backlogs),
            "min_wire_bytes": min(wires), "max_wire_bytes": max(wires),
            "min_prior_ul_mcs": min(mcs), "max_prior_ul_mcs": max(mcs),
        })


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(model: TwoPartModel,
          rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = build_features(rows)
    probability = model.deadline.probability(features)
    predicted_latency = model.latency.latency_ms(features)
    actual = np.array([1.0 if row["completed_within_deadline"] in (True, "True")
                       else 0.0 for row in rows])

    brier = float(np.mean((probability - actual) ** 2))
    predicted_positive = probability >= 0.5
    true_positive = int(np.sum(predicted_positive & (actual == 1.0)))
    false_positive = int(np.sum(predicted_positive & (actual == 0.0)))
    true_negative = int(np.sum(~predicted_positive & (actual == 0.0)))
    false_negative = int(np.sum(~predicted_positive & (actual == 1.0)))
    predicted_count = true_positive + false_positive
    false_success_rate = (false_positive / predicted_count
                          if predicted_count else 0.0)

    on_time_mask = actual == 1.0
    errors = [abs(float(predicted_latency[i])
                  - float(rows[i]["transport_latency_ns"]) / 1e6)
              for i in range(len(rows)) if on_time_mask[i]]
    reward_errors = [C2.reward_error_for_latency_error_ms(value)
                     for value in errors]
    return {
        "n": len(rows), "n_on_time": int(np.sum(on_time_mask)),
        "brier": brier,
        "confusion": {"true_positive": true_positive,
                      "false_positive": false_positive,
                      "true_negative": true_negative,
                      "false_negative": false_negative},
        "predicted_success": predicted_count,
        "false_success_rate": false_success_rate,
        "latency_abs_error_p50_ms": _percentile(errors, 0.50),
        "latency_abs_error_p95_ms": _percentile(errors, 0.95),
        "reward_error_p50": _percentile(reward_errors, 0.50),
        "reward_error_p95": _percentile(reward_errors, 0.95),
    }


def score_queue(model: TwoPartModel,
                rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Held-out next-backlog accuracy for the causal queue-transition head."""
    predicted: list[float] = []
    observed: list[float] = []
    persistence: list[float] = []
    for row in rows:
        if not row.get("has_successor") or row["successor_backlog_bytes"] is None:
            continue
        backlog = float(row["pre_enqueue_backlog_bytes"])
        value, _ = model.queue.next_backlog_bytes(
            backlog_bytes=backlog,
            ingress_bytes=float(row["deterministic_action_ingress_bytes"]),
            mcs=float(row["prior_ul_mcs"]))
        predicted.append(value)
        observed.append(float(row["successor_backlog_bytes"]))
        persistence.append(backlog)
    denominator = sum(abs(v) for v in observed)
    def nmae(values: Sequence[float]) -> float:
        if denominator <= 0:
            return math.inf
        return sum(abs(p - o) for p, o in zip(values, observed)) / denominator
    model_nmae = nmae(predicted)
    persistence_nmae = nmae(persistence)
    distinct = len({round(v, 3) for v in predicted})
    return {
        "n": len(predicted), "nmae": model_nmae,
        "persistence_nmae": persistence_nmae,
        "improvement_over_persistence":
            ((persistence_nmae - model_nmae) / persistence_nmae)
            if persistence_nmae not in (0.0, math.inf) else 0.0,
        "distinct_predicted_values": distinct,
        "predicted_zero_fraction":
            sum(1 for v in predicted if v == 0.0) / len(predicted)
            if predicted else math.nan,
        "predicted_p50": _percentile(predicted, 0.5),
        "predicted_p95": _percentile(predicted, 0.95),
        "observed_p50": _percentile(observed, 0.5),
        "observed_p95": _percentile(observed, 0.95),
    }


def monotonicity_violations(model: TwoPartModel) -> dict[str, Any]:
    """Structural check of both heads inside the fitted support."""
    support = model.support
    backlogs = np.linspace(support["min_backlog_bytes"],
                           support["max_backlog_bytes"], 9)
    wires = np.linspace(support["min_wire_bytes"],
                        support["max_wire_bytes"], 9)
    mcs_values = np.linspace(support["min_prior_ul_mcs"],
                             support["max_prior_ul_mcs"], 9)
    violations = 0
    checks = 0

    def probe(backlog: float, wire: float, mcs: float) -> Features:
        return Features(
            backlog_scaled=np.array([C2.backlog_scaled(backlog)]),
            backlog_bytes=np.array([backlog]),
            bytes_mb=np.array([wire / BYTES_SCALE]),
            wire_bytes=np.array([wire]),
            mcs_norm=np.array([mcs / MCS_SCALE]))

    for wire in wires:
        for mcs in mcs_values:
            previous_p = previous_l = None
            for backlog in backlogs:
                f = probe(backlog, wire, mcs)
                p = float(model.deadline.probability(f)[0])
                l = float(model.latency.latency_ms(f)[0])
                checks += 1
                if previous_p is not None and p > previous_p + 1e-12:
                    violations += 1
                if previous_l is not None and l < previous_l - 1e-9:
                    violations += 1
                previous_p, previous_l = p, l
    for backlog in backlogs:
        for mcs in mcs_values:
            previous_p = previous_l = None
            for wire in wires:
                f = probe(backlog, wire, mcs)
                p = float(model.deadline.probability(f)[0])
                l = float(model.latency.latency_ms(f)[0])
                checks += 1
                if previous_p is not None and p > previous_p + 1e-12:
                    violations += 1
                if previous_l is not None and l < previous_l - 1e-9:
                    violations += 1
                previous_p, previous_l = p, l
    for backlog in backlogs:
        for wire in wires:
            previous_p = previous_l = None
            for mcs in mcs_values:
                f = probe(backlog, wire, mcs)
                p = float(model.deadline.probability(f)[0])
                l = float(model.latency.latency_ms(f)[0])
                checks += 1
                if previous_p is not None and p < previous_p - 1e-12:
                    violations += 1
                if previous_l is not None and l > previous_l + 1e-9:
                    violations += 1
                previous_p, previous_l = p, l
    return {"violations": violations, "checks": checks,
            "axes": "backlog up / bytes up -> p down, latency up; "
                    "mcs up -> p up, latency down"}


def action_ranking_sensitivity(
    model: TwoPartModel, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Does the model rank the three byte roles the way the evidence does?

    For each observed (backlog, mcs) state the model scores the three measured
    wire-byte levels by expected reward with Q_perc held fixed, so only the
    transport term can move the ranking.  The empirical ranking is the mean
    realized on-time rate of that byte role in the same MCS bin.
    """
    levels = sorted({float(row["action_wire_bytes"]) for row in rows})
    representative = [levels[0], levels[len(levels) // 2], levels[-1]]
    empirical: dict[float, list[float]] = {value: [] for value in representative}
    for row in rows:
        wire = float(row["action_wire_bytes"])
        nearest = min(representative, key=lambda value: abs(value - wire))
        empirical[nearest].append(
            1.0 if row["completed_within_deadline"] in (True, "True") else 0.0)
    empirical_rank = sorted(
        representative,
        key=lambda value: -(statistics.fmean(empirical[value])
                            if empirical[value] else 0.0))

    agree = 0
    total = 0
    for row in rows:
        backlog = float(row["pre_enqueue_backlog_bytes"])
        mcs = float(row["prior_ul_mcs"])
        scored = []
        for wire in representative:
            f = Features(
                backlog_scaled=np.array([C2.backlog_scaled(backlog)]),
                backlog_bytes=np.array([backlog]),
                bytes_mb=np.array([wire / BYTES_SCALE]),
                wire_bytes=np.array([wire]),
                mcs_norm=np.array([mcs / MCS_SCALE]))
            p = float(model.deadline.probability(f)[0])
            latency = float(model.latency.latency_ms(f)[0])
            # Q_perc held fixed at 1.0 so only transport moves the ranking.
            on_time_reward = 1.0 - C2.REWARD_LATENCY_WEIGHT * latency / C2.REWARD_DEADLINE_MS
            scored.append((wire, p * on_time_reward + (1 - p) * -1.0))
        model_rank = [value for value, _ in sorted(scored, key=lambda i: -i[1])]
        total += 1
        agree += int(model_rank == empirical_rank)
    return {
        "representative_wire_bytes": representative,
        "empirical_on_time_rate": {
            str(int(value)): (statistics.fmean(empirical[value])
                              if empirical[value] else None)
            for value in representative},
        "empirical_ranking": [int(value) for value in empirical_rank],
        "model_agrees_with_empirical_ranking_fraction":
            agree / total if total else math.nan,
        "states_scored": total,
    }


# ---------------------------------------------------------------------------
# Grouped leave-one-whole-FIT-cell-out cross-validation
# ---------------------------------------------------------------------------
def grouped_cross_validation(
    fit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = sorted({row["cell_id"] for row in fit_rows})
    require(len(cells) >= 3, "grouped CV needs at least three FIT cells")
    folds: list[dict[str, Any]] = []
    pooled_errors: list[float] = []
    pooled_probability: list[float] = []
    pooled_actual: list[float] = []
    pooled_predicted_positive = 0
    pooled_false_positive = 0
    for held_out in cells:
        train = [row for row in fit_rows if row["cell_id"] != held_out]
        test = [row for row in fit_rows if row["cell_id"] == held_out]
        model = fit_model(train)
        result = score(model, test)
        queue = score_queue(model, test)
        folds.append({"held_out_cell": held_out, **result,
                      "queue_nmae": queue["nmae"],
                      "queue_improvement_over_persistence":
                          queue["improvement_over_persistence"],
                      "queue_distinct_predicted": queue["distinct_predicted_values"]})
        features = build_features(test)
        probability = model.deadline.probability(features)
        predicted_latency = model.latency.latency_ms(features)
        for index, row in enumerate(test):
            actual = (1.0 if row["completed_within_deadline"] in (True, "True")
                      else 0.0)
            pooled_probability.append(float(probability[index]))
            pooled_actual.append(actual)
            if probability[index] >= 0.5:
                pooled_predicted_positive += 1
                if actual == 0.0:
                    pooled_false_positive += 1
            if actual == 1.0:
                pooled_errors.append(abs(
                    float(predicted_latency[index])
                    - float(row["transport_latency_ns"]) / 1e6))
    pooled_brier = float(np.mean(
        (np.array(pooled_probability) - np.array(pooled_actual)) ** 2))
    pooled_reward = [C2.reward_error_for_latency_error_ms(value)
                     for value in pooled_errors]
    return {
        "protocol": C2.SELECTION_PROTOCOL,
        "folds": folds, "fold_count": len(folds),
        "pooled": {
            "n": len(pooled_actual), "n_on_time": len(pooled_errors),
            "brier": pooled_brier,
            "predicted_success": pooled_predicted_positive,
            "false_success": pooled_false_positive,
            "false_success_rate": (pooled_false_positive
                                   / pooled_predicted_positive
                                   if pooled_predicted_positive else 0.0),
            "latency_abs_error_p50_ms": _percentile(pooled_errors, 0.50),
            "latency_abs_error_p95_ms": _percentile(pooled_errors, 0.95),
            "reward_error_p50": _percentile(pooled_reward, 0.50),
            "reward_error_p95": _percentile(pooled_reward, 0.95),
        },
    }


def load_decisions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "cell_id": row["cell_id"], "partition": row["partition"],
                "frame_index": int(row["frame_index"]),
                "action_id": int(row["action_id"]),
                "mode_id": int(row["mode_id"]), "q_e4": int(row["q_e4"]),
                "pre_enqueue_backlog_bytes":
                    int(row["pre_enqueue_backlog_bytes"]),
                "prior_ul_mcs": int(row["prior_ul_mcs"]),
                "action_wire_bytes": int(row["action_wire_bytes"]),
                "held_action_wire_bytes": int(row["held_action_wire_bytes"]),
                "deterministic_action_ingress_bytes":
                    int(row["deterministic_action_ingress_bytes"]),
                "transport_latency_ns":
                    (int(row["transport_latency_ns"])
                     if row["transport_latency_ns"] not in ("", "None")
                     else None),
                "completed_within_deadline":
                    row["completed_within_deadline"] == "True",
                "terminal_outcome": row["terminal_outcome"],
                "has_successor": row["has_successor"] == "True",
                "successor_backlog_bytes":
                    (int(row["successor_backlog_bytes"])
                     if row["successor_backlog_bytes"] not in ("", "None")
                     else None),
            })
    return rows
