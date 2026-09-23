"""Train-only analytic preflight for the empirical Run-3 reward.

The preflight answers a deliberately bounded question before another SAC run:
does the registered 391-scene training partition contain useful executable
actions under the realized-outcome Run-3 reward/kernel?  Initialization
re-verifies the hash-pinned full quality surface and the pre-existing legacy D1
qualification, including registered legacy ``held_scene`` evidence.  It also
loads reward-blind partition metadata containing ``fit_validation`` identities.
After that inherited integrity gate, neither legacy ``held_scene`` nor
``fit_validation`` identities enter Run-3 aggregation, action selection, or
reward evaluation: only the 391 registered training identities do.  No Run-3
validation or checkpoint selection is performed until that evaluation is
frozen.  This module never samples a terminal event, trains a model, or
initializes CUDA.  Probabilities are privileged simulator-kernel fields and
never become policy features.

The full executable wire support contains 52,240 ``(mode_id, q_e4)`` pairs.
The implementation evaluates one scene/profile/mode vector at a time and
retains only aggregate vectors and compact summaries; it never materializes a
``scene x profile x action`` table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import fixed_stage_latency_ms
from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
)
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalOneStepEnvironmentV1,
)
from .empirical_contextual_run3_reward import (
    RUN3_KERNEL_SPEC_SHA256,
    RUN3_REWARD_SPEC,
    RUN3_REWARD_SPEC_SHA256,
    QuantileLatencyProxyV1,
    expected_run3_reward,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .offline_quality_grid.contract import Q_E4_GRID
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "ActionVectorBatchV1",
    "PREFLIGHT_SCHEMA",
    "RegisteredRun3TrainSourceV1",
    "Run3PreflightError",
    "Run3PreflightResultV1",
    "Run3TrainOnlyPreflightV1",
    "TrainContextV1",
    "render_run3_train_only_preflight",
    "run_registered_train_only_preflight",
]


PREFLIGHT_SCHEMA = "splitfusion.run3_train_only_preflight.v1"
EXPECTED_TRAIN_SCENES = 391
EXPECTED_PROFILES = tuple(NETWORK_PROFILE_ORDER)
EXPECTED_ACTIONS_PER_SCENE = sum(
    upper - lower + 1 for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS
)
EXPECTED_CONTEXT_PROFILE_ACTION_EVALUATIONS = (
    EXPECTED_TRAIN_SCENES * len(EXPECTED_PROFILES) * EXPECTED_ACTIONS_PER_SCENE
)
QUALITY_FIELDS: Tuple[str, ...] = (
    "q_loc",
    "q_seg",
    "q_perc",
    "vehicle_recall",
    "person_recall",
    "vehicle_xy_error_m",
    "person_xy_error_m",
    "seg_vehicle_iou",
    "seg_person_iou",
)
RAW_QUALITY_FIELDS: Tuple[str, ...] = (
    "vehicle_recall",
    "person_recall",
    "vehicle_xy_error_m",
    "person_xy_error_m",
    "seg_vehicle_iou",
    "seg_person_iou",
)
VALIDITY_FIELDS: Tuple[str, ...] = tuple(
    f"{name}_valid_fraction" for name in RAW_QUALITY_FIELDS
)
SUMMARY_FIELDS: Tuple[str, ...] = QUALITY_FIELDS + VALIDITY_FIELDS + (
    "payload_bytes",
    "datagram_count",
    "p_complete_reassembly_given_sent",
    "p_edge_admission_given_reassembled",
    "p_reassembly_failure",
    "p_admission_failure",
    "p_admitted",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "p_timeout_given_admitted",
    "p_success_within_deadline_given_admitted",
    "p_service_timeout",
    "p_timely_feedback",
    "p_total_failure",
    "conditional_admitted_expected_reward",
    "conditional_timely_success_reward",
    "conditional_timely_success_reward_valid_fraction",
    "expected_run3_return",
    "tail_stress_1p25_p_timeout_given_admitted",
    "tail_stress_1p25_p_success_within_deadline_given_admitted",
    "tail_stress_1p25_p_service_timeout",
    "tail_stress_1p25_p_timely_feedback",
    "tail_stress_1p25_p_total_failure",
    "tail_stress_1p25_conditional_admitted_expected_reward",
    "tail_stress_1p25_conditional_timely_success_reward",
    "tail_stress_1p25_conditional_timely_success_reward_valid_fraction",
    "tail_stress_1p25_expected_run3_return",
    "tail_stress_1p50_p_timeout_given_admitted",
    "tail_stress_1p50_p_success_within_deadline_given_admitted",
    "tail_stress_1p50_p_service_timeout",
    "tail_stress_1p50_p_timely_feedback",
    "tail_stress_1p50_p_total_failure",
    "tail_stress_1p50_conditional_admitted_expected_reward",
    "tail_stress_1p50_conditional_timely_success_reward",
    "tail_stress_1p50_conditional_timely_success_reward_valid_fraction",
    "tail_stress_1p50_expected_run3_return",
)
OPTIONAL_METRIC_FIELDS: Tuple[str, ...] = (
    "conditional_timely_success_reward",
    "tail_stress_1p25_conditional_timely_success_reward",
    "tail_stress_1p50_conditional_timely_success_reward",
)
EXPLICIT_Q_E4: Tuple[int, ...] = (7000, 7500, 8000)
Q_BINS: Tuple[Tuple[str, int, int], ...] = (
    ("q_lt_0p70", 0, 6999),
    ("q_0p70_to_lt_0p80", 7000, 7999),
    ("q_ge_0p80", 8000, 9800),
)
SCALAR_VECTOR_TOLERANCE = 5e-11
TIMELY_REGION_TOLERANCE = 1e-12
FUTURE_DECISION_KEY_CONTRACT = (
    "UNIQUE_PER_STOCHASTIC_VISIT_AND_EXOGENOUS_TO_ACTION:"
    "RUN_SEED_OR_SESSION_PLUS_EPISODE_OR_VISIT_PLUS_DECISION_SEQ;"
    "NEVER_ACTION_ID;NEVER_SAMPLE_PROFILE_ALONE"
)
TAIL_SENSITIVITY_SPEC = {
    "base": "REGISTERED_KERNEL_CAPS_UNKNOWN_TOP_1_PERCENT_AT_P99",
    "scope": "TRAIN_ONLY_ANALYSIS_NOT_TRAINING_REWARD_NOT_WEIGHT_TUNING",
    "stress_alternatives": [
        {"name": "tail_stress_1p25", "u_0p99_latency_ms": "p99", "u_1p00_latency_ms": "1.25*p99"},
        {"name": "tail_stress_1p50", "u_0p99_latency_ms": "p99", "u_1p00_latency_ms": "1.50*p99"},
    ],
}
TAIL_SENSITIVITY_SPEC_SHA256 = canonical_sha256(TAIL_SENSITIVITY_SPEC)


class Run3PreflightError(RuntimeError):
    """A train-only boundary, model invariant, or analytic check failed."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise Run3PreflightError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Run3PreflightError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TrainContextV1:
    """One registered training scene; no evaluation split is representable."""

    sample_id: str
    sampling_weight: float
    split: str = TRAIN_SPLIT

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise Run3PreflightError("training sample_id must be non-empty")
        if self.split != TRAIN_SPLIT:
            raise Run3PreflightError("preflight categorically accepts only split='train'")
        weight = _finite(self.sampling_weight, "sampling_weight")
        if weight <= 0.0:
            raise Run3PreflightError("sampling_weight must be positive")


@dataclass(frozen=True, slots=True)
class ActionVectorBatchV1:
    """All executable q values for one train scene/profile/mode."""

    sample_id: str
    network_profile: str
    mode_id: int
    q_e4: np.ndarray
    metrics: Mapping[str, np.ndarray]

    def revalidate(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise Run3PreflightError("action vector sample_id must be non-empty")
        if self.network_profile not in EXPECTED_PROFILES:
            raise Run3PreflightError("action vector has an unknown network profile")
        if type(self.mode_id) is not int or not 0 <= self.mode_id < 12:
            raise Run3PreflightError("mode_id must be an exact integer in [0,11]")
        lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[self.mode_id]
        expected_q = np.arange(lower, upper + 1, dtype=np.int64)
        if self.q_e4.dtype != np.int64 or not np.array_equal(self.q_e4, expected_q):
            raise Run3PreflightError("action vector does not cover exact executable support")
        if set(self.metrics) != set(SUMMARY_FIELDS):
            missing = sorted(set(SUMMARY_FIELDS).difference(self.metrics))
            foreign = sorted(set(self.metrics).difference(SUMMARY_FIELDS))
            raise Run3PreflightError(
                f"action metric inventory drift: missing={missing}, foreign={foreign}"
            )
        for name, values in self.metrics.items():
            if type(values) is not np.ndarray or values.shape != self.q_e4.shape:
                raise Run3PreflightError(f"{name} vector shape/type drift")
            if values.dtype != np.float64:
                raise Run3PreflightError(f"{name} must be float64")
            if name in (*RAW_QUALITY_FIELDS, *OPTIONAL_METRIC_FIELDS):
                if np.any(np.isinf(values)):
                    raise Run3PreflightError(f"{name} contains infinity")
            elif not np.all(np.isfinite(values)):
                raise Run3PreflightError(f"{name} contains non-finite values")
        for name in (
            "q_loc",
            "q_seg",
            "q_perc",
            "p_complete_reassembly_given_sent",
            "p_edge_admission_given_reassembled",
            "p_reassembly_failure",
            "p_admission_failure",
            "p_admitted",
            "p_timeout_given_admitted",
            "p_success_within_deadline_given_admitted",
            "p_service_timeout",
            "p_timely_feedback",
            "p_total_failure",
            "tail_stress_1p25_p_timeout_given_admitted",
            "tail_stress_1p25_p_success_within_deadline_given_admitted",
            "tail_stress_1p25_p_service_timeout",
            "tail_stress_1p25_p_timely_feedback",
            "tail_stress_1p25_p_total_failure",
            "tail_stress_1p50_p_timeout_given_admitted",
            "tail_stress_1p50_p_success_within_deadline_given_admitted",
            "tail_stress_1p50_p_service_timeout",
            "tail_stress_1p50_p_timely_feedback",
            "tail_stress_1p50_p_total_failure",
            "conditional_timely_success_reward_valid_fraction",
            "tail_stress_1p25_conditional_timely_success_reward_valid_fraction",
            "tail_stress_1p50_conditional_timely_success_reward_valid_fraction",
            *VALIDITY_FIELDS,
        ):
            values = self.metrics[name]
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise Run3PreflightError(f"{name} escaped [0,1]")
        for name in RAW_QUALITY_FIELDS:
            observed_validity = self.metrics[f"{name}_valid_fraction"]
            expected_validity = np.isfinite(self.metrics[name]).astype(np.float64)
            if not np.array_equal(observed_validity, expected_validity):
                raise Run3PreflightError(
                    f"{name} validity mask does not match finite-value support"
                )
        for name in (
            "vehicle_recall",
            "person_recall",
            "seg_vehicle_iou",
            "seg_person_iou",
        ):
            finite = self.metrics[name][np.isfinite(self.metrics[name])]
            if np.any(finite < 0.0) or np.any(finite > 1.0):
                raise Run3PreflightError(f"finite {name} escaped [0,1]")
        for name in ("vehicle_xy_error_m", "person_xy_error_m"):
            finite = self.metrics[name][np.isfinite(self.metrics[name])]
            if np.any(finite < 0.0):
                raise Run3PreflightError(f"finite {name} must be non-negative")
        p50 = self.metrics["latency_p50_ms"]
        p95 = self.metrics["latency_p95_ms"]
        p99 = self.metrics["latency_p99_ms"]
        if np.any(p50 < 0.0) or np.any(p50 > p95) or np.any(p95 > p99):
            raise Run3PreflightError("latency quantiles require 0 <= p50 <= p95 <= p99")
        if np.any(self.metrics["payload_bytes"] <= 0.0):
            raise Run3PreflightError("payload must be positive")
        expected_datagrams = np.ceil(
            self.metrics["payload_bytes"] / UDP_PAYLOAD_CAPACITY_BYTES
        )
        if not np.array_equal(self.metrics["datagram_count"], expected_datagrams):
            raise Run3PreflightError("datagram count does not reconcile with payload")
        p_reassembly = self.metrics["p_complete_reassembly_given_sent"]
        p_admission = self.metrics["p_edge_admission_given_reassembled"]
        p_admitted = p_reassembly * p_admission
        p_reassembly_failure = 1.0 - p_reassembly
        p_admission_failure = p_reassembly * (1.0 - p_admission)

        def require_exact(name: str, expected: np.ndarray) -> None:
            if not np.array_equal(self.metrics[name], expected):
                raise Run3PreflightError(f"{name} exact reconciliation failed")

        require_exact("p_reassembly_failure", p_reassembly_failure)
        require_exact("p_admission_failure", p_admission_failure)
        require_exact("p_admitted", p_admitted)
        branch_sum = p_reassembly_failure + p_admission_failure + p_admitted
        if not np.allclose(
            branch_sum, np.ones_like(branch_sum), rtol=0.0, atol=2e-15
        ):
            raise Run3PreflightError("kernel probability branches do not sum to one")
        conditional_reward, timeout = _integrate_latency_proxy_vectors(
            self.metrics["q_perc"], p50, p95, p99
        )
        success_given_admitted = 1.0 - timeout
        require_exact("p_timeout_given_admitted", timeout)
        require_exact(
            "p_success_within_deadline_given_admitted", success_given_admitted
        )
        p_service_timeout = p_admitted * timeout
        p_timely = p_admitted * success_given_admitted
        p_total_failure = (
            p_reassembly_failure + p_admission_failure + p_service_timeout
        )
        require_exact("p_service_timeout", p_service_timeout)
        require_exact("p_timely_feedback", p_timely)
        require_exact("p_total_failure", p_total_failure)
        if not np.allclose(
            p_total_failure + p_timely,
            np.ones_like(p_timely),
            rtol=0.0,
            atol=2e-15,
        ):
            raise Run3PreflightError("four terminal probability masses do not sum to one")
        require_exact(
            "conditional_admitted_expected_reward", conditional_reward
        )
        success_reward, success_valid = _conditional_timely_success_reward(
            conditional_reward, timeout
        )
        require_exact(
            "conditional_timely_success_reward_valid_fraction", success_valid
        )
        observed_success_reward = self.metrics[
            "conditional_timely_success_reward"
        ]
        if not np.array_equal(
            np.isnan(observed_success_reward), np.isnan(success_reward)
        ) or not np.array_equal(
            observed_success_reward[success_valid.astype(bool)],
            success_reward[success_valid.astype(bool)],
        ):
            raise Run3PreflightError(
                "conditional_timely_success_reward exact reconciliation failed"
            )
        reconstructed_conditional = np.where(
            success_valid.astype(bool),
            success_given_admitted * success_reward - timeout,
            -timeout,
        )
        if not np.allclose(
            reconstructed_conditional,
            conditional_reward,
            rtol=0.0,
            atol=2e-15,
        ):
            raise Run3PreflightError("conditional reward branch reconstruction failed")
        require_exact(
            "expected_run3_return",
            -p_reassembly_failure
            - p_admission_failure
            + p_admitted * conditional_reward,
        )
        for label, multiplier in (
            ("tail_stress_1p25", 1.25),
            ("tail_stress_1p50", 1.5),
        ):
            stress_reward, stress_timeout = _integrate_latency_proxy_vectors(
                self.metrics["q_perc"],
                p50,
                p95,
                p99,
                top_endpoint_multiplier=multiplier,
            )
            stress_success = 1.0 - stress_timeout
            require_exact(f"{label}_p_timeout_given_admitted", stress_timeout)
            require_exact(
                f"{label}_p_success_within_deadline_given_admitted",
                stress_success,
            )
            stress_service_timeout = p_admitted * stress_timeout
            stress_timely = p_admitted * stress_success
            stress_total_failure = (
                p_reassembly_failure
                + p_admission_failure
                + stress_service_timeout
            )
            require_exact(f"{label}_p_service_timeout", stress_service_timeout)
            require_exact(f"{label}_p_timely_feedback", stress_timely)
            require_exact(f"{label}_p_total_failure", stress_total_failure)
            if not np.allclose(
                stress_total_failure + stress_timely,
                np.ones_like(stress_timely),
                rtol=0.0,
                atol=2e-15,
            ):
                raise Run3PreflightError(
                    f"{label} four terminal probability masses do not sum to one"
                )
            require_exact(
                f"{label}_conditional_admitted_expected_reward", stress_reward
            )
            stress_success_reward, stress_success_valid = (
                _conditional_timely_success_reward(stress_reward, stress_timeout)
            )
            require_exact(
                f"{label}_conditional_timely_success_reward_valid_fraction",
                stress_success_valid,
            )
            observed_stress_success = self.metrics[
                f"{label}_conditional_timely_success_reward"
            ]
            stress_valid_mask = stress_success_valid.astype(bool)
            if not np.array_equal(
                np.isnan(observed_stress_success),
                np.isnan(stress_success_reward),
            ) or not np.array_equal(
                observed_stress_success[stress_valid_mask],
                stress_success_reward[stress_valid_mask],
            ):
                raise Run3PreflightError(
                    f"{label} conditional timely-success reward reconciliation failed"
                )
            reconstructed_stress = np.where(
                stress_valid_mask,
                stress_success * stress_success_reward - stress_timeout,
                -stress_timeout,
            )
            if not np.allclose(
                reconstructed_stress,
                stress_reward,
                rtol=0.0,
                atol=2e-15,
            ):
                raise Run3PreflightError(
                    f"{label} conditional reward branch reconstruction failed"
                )
            require_exact(
                f"{label}_expected_run3_return",
                -p_reassembly_failure
                - p_admission_failure
                + p_admitted * stress_reward,
            )


class Run3TrainVectorSource(Protocol):
    """Small seam allowing fast synthetic tests and one registered real source."""

    split: str
    reward_spec_sha256: str
    kernel_spec_sha256: str
    source_bindings: Mapping[str, str]

    @property
    def contexts(self) -> Tuple[TrainContextV1, ...]: ...

    def action_vectors(
        self, sample_id: str, network_profile: str, mode_id: int
    ) -> ActionVectorBatchV1: ...

    def close(self) -> None: ...


def _curve_vector(curve: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray([block.x_center for block in curve.blocks], dtype=np.float64)
    values = np.asarray([block.value for block in curve.blocks], dtype=np.float64)
    weights = np.asarray([block.weight for block in curve.blocks], dtype=np.float64)
    if len(centers) == 0 or not np.all(np.diff(centers) > 0.0):
        raise Run3PreflightError("network curve centers are not strictly increasing")
    if np.any(x < curve.raw_x_min) or np.any(x > curve.raw_x_max):
        raise Run3PreflightError("network query would extrapolate")
    result = np.empty_like(x)
    support = np.empty_like(x)
    low = x <= centers[0]
    high = x >= centers[-1]
    middle = ~(low | high)
    result[low], support[low] = values[0], weights[0]
    result[high], support[high] = values[-1], weights[-1]
    if np.any(middle):
        xi = x[middle]
        right = np.searchsorted(centers, xi, side="left")
        left = right - 1
        fraction = (xi - centers[left]) / (centers[right] - centers[left])
        result[middle] = values[left] + fraction * (values[right] - values[left])
        support[middle] = np.minimum(weights[left], weights[right])
    return result, support


def _local_latency_support(model: Any, x: np.ndarray) -> np.ndarray:
    knot_x = np.asarray([item[0] for item in model.latency_support_knots], dtype=np.float64)
    knot_n = np.asarray([item[1] for item in model.latency_support_knots], dtype=np.float64)
    if len(knot_x) == 0 or not np.all(np.diff(knot_x) > 0.0):
        raise Run3PreflightError("latency support knots are not strictly increasing")
    if np.any(x < knot_x[0]) or np.any(x > knot_x[-1]):
        raise Run3PreflightError("latency support query escaped its envelope")
    right = np.searchsorted(knot_x, x, side="left")
    exact = (right < len(knot_x)) & (
        x == knot_x[np.minimum(right, len(knot_x) - 1)]
    )
    result = np.empty_like(x)
    result[exact] = knot_n[right[exact]]
    if np.any(~exact):
        r = right[~exact]
        if np.any(r <= 0) or np.any(r >= len(knot_x)):
            raise Run3PreflightError("latency support was not bracketed")
        result[~exact] = np.minimum(knot_n[r - 1], knot_n[r])
    return result


def _integrate_latency_proxy_vectors(
    q_perc: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    p99: np.ndarray,
    *,
    top_endpoint_multiplier: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vector equivalent of the registered scalar quantile-proxy integral."""

    lower = np.maximum(0.0, 2.0 * p50 - p95)
    if top_endpoint_multiplier not in (1.0, 1.25, 1.5):
        raise Run3PreflightError("unregistered tail-sensitivity multiplier")
    ys = (lower, p50, p95, p99, p99 * top_endpoint_multiplier)
    us = (0.0, 0.5, 0.95, 0.99, 1.0)
    reward = np.zeros_like(q_perc)
    timeout = np.zeros_like(q_perc)
    deadline = RUN3_REWARD_SPEC.deadline_ms
    for u0, u1, y0, y1 in zip(us, us[1:], ys, ys[1:]):
        width = u1 - u0
        all_late = y0 > deadline
        all_timely = y1 <= deadline
        crossing = ~(all_late | all_timely)
        reward[all_late] -= width
        timeout[all_late] += width
        reward[all_timely] += width * (
            q_perc[all_timely]
            - RUN3_REWARD_SPEC.latency_weight
            * ((y0[all_timely] + y1[all_timely]) / 2.0)
            / deadline
        )
        if np.any(crossing):
            timely_fraction = (deadline - y0[crossing]) / (
                y1[crossing] - y0[crossing]
            )
            timely_width = width * timely_fraction
            late_width = width - timely_width
            reward[crossing] += timely_width * (
                q_perc[crossing]
                - RUN3_REWARD_SPEC.latency_weight
                * ((y0[crossing] + deadline) / 2.0)
                / deadline
            )
            reward[crossing] -= late_width
            timeout[crossing] += late_width
    return reward, timeout


def _conditional_timely_success_reward(
    conditional_admitted_reward: np.ndarray,
    timeout_given_admitted: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover E[success reward | admitted and timely] without inventing zeros.

    ``conditional_admitted_reward`` includes the registered ``-1`` timeout
    branch.  Adding timeout mass removes that branch, leaving timely-reward
    mass conditional on admission.  Division by timely mass is defined only
    where that mass is nonzero; undefined rows remain NaN with an explicit
    validity vector.
    """

    timely = 1.0 - timeout_given_admitted
    valid = timely > 0.0
    result = np.full(conditional_admitted_reward.shape, np.nan, dtype=np.float64)
    np.divide(
        conditional_admitted_reward + timeout_given_admitted,
        timely,
        out=result,
        where=valid,
    )
    return result, valid.astype(np.float64)


def _scalar_tail_analysis(
    *,
    q_perc: float,
    p_reassembly: float,
    p_admission: float,
    p50: float,
    p95: float,
    p99: float,
    top_endpoint_multiplier: float,
) -> Dict[str, float]:
    """Independent scalar integral for counterfactual tail-stress auditing."""

    if top_endpoint_multiplier not in (1.25, 1.5):
        raise Run3PreflightError("unregistered scalar tail-sensitivity multiplier")
    lower = max(0.0, 2.0 * p50 - p95)
    points = (
        (0.0, lower),
        (0.5, p50),
        (0.95, p95),
        (0.99, p99),
        (1.0, p99 * top_endpoint_multiplier),
    )
    conditional_reward = 0.0
    timeout = 0.0
    deadline = RUN3_REWARD_SPEC.deadline_ms
    for (u0, y0), (u1, y1) in zip(points, points[1:]):
        width = u1 - u0
        if y0 > deadline:
            conditional_reward -= width
            timeout += width
        elif y1 <= deadline:
            conditional_reward += width * (
                q_perc
                - RUN3_REWARD_SPEC.latency_weight
                * ((y0 + y1) / 2.0)
                / deadline
            )
        else:
            timely_fraction = (deadline - y0) / (y1 - y0)
            timely_width = width * timely_fraction
            timeout_width = width - timely_width
            conditional_reward += timely_width * (
                q_perc
                - RUN3_REWARD_SPEC.latency_weight
                * ((y0 + deadline) / 2.0)
                / deadline
            )
            conditional_reward -= timeout_width
            timeout += timeout_width
    success = 1.0 - timeout
    success_reward = (
        (conditional_reward + timeout) / success if success > 0.0 else math.nan
    )
    p_reassembly_failure = 1.0 - p_reassembly
    p_admission_failure = p_reassembly * (1.0 - p_admission)
    p_admitted = p_reassembly * p_admission
    p_service_timeout = p_admitted * timeout
    p_timely = p_admitted * success
    p_total_failure = (
        p_reassembly_failure + p_admission_failure + p_service_timeout
    )
    return {
        "p_timeout_given_admitted": timeout,
        "p_success_within_deadline_given_admitted": success,
        "p_service_timeout": p_service_timeout,
        "p_timely_feedback": p_timely,
        "p_total_failure": p_total_failure,
        "conditional_admitted_expected_reward": conditional_reward,
        "conditional_timely_success_reward": success_reward,
        "conditional_timely_success_reward_valid_fraction": float(success > 0.0),
        "expected_run3_return": (
            -p_reassembly_failure
            - p_admission_failure
            + p_admitted * conditional_reward
        ),
    }


def _max_contiguous_true(mask: np.ndarray) -> int:
    """Return the longest positive-width run in an ordered executable support."""

    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


class RegisteredRun3TrainSourceV1:
    """Read-only adapter over the registered train partition and surrogates."""

    split = TRAIN_SPLIT
    reward_spec_sha256 = RUN3_REWARD_SPEC_SHA256
    kernel_spec_sha256 = RUN3_KERNEL_SPEC_SHA256

    def __init__(self, environment: PartitionedEmpiricalOneStepEnvironmentV1) -> None:
        if type(environment) is not PartitionedEmpiricalOneStepEnvironmentV1:
            raise Run3PreflightError("registered source needs exact partitioned environment")
        if environment.sampling_split != TRAIN_SPLIT:
            raise Run3PreflightError("registered source refuses a non-training environment")
        if environment.fit_partition_sha256 != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
            raise Run3PreflightError("fit partition binding drift")
        if environment.scene_population_count != EXPECTED_TRAIN_SCENES:
            raise Run3PreflightError("registered training scene count drift")
        partition = environment._fit_partition
        allowed = {
            row.sample_id for row in partition.scene_assignments if row.split == TRAIN_SPLIT
        }
        forbidden = {
            row.sample_id for row in partition.scene_assignments if row.split != TRAIN_SPLIT
        }
        contexts = tuple(
            TrainContextV1(item.sample_id, item.sampling_weight)
            for item in environment._split_contexts
        )
        ids = {item.sample_id for item in contexts}
        if ids != allowed or ids.intersection(forbidden):
            raise Run3PreflightError("source scene inventory crossed the training boundary")
        self._environment = environment
        self._contexts = contexts
        self._allowed_sample_ids = frozenset(ids)
        self.source_bindings = {
            "d1_environment_binding_sha256": environment.binding.canonical_sha256(),
            "fit_partition_sha256": environment.fit_partition_sha256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
            "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        }

    @classmethod
    def load_registered(
        cls, *, project_root: Optional[Path] = None
    ) -> "RegisteredRun3TrainSourceV1":
        environment = PartitionedEmpiricalOneStepEnvironmentV1.load_registered(
            seed=0, split=TRAIN_SPLIT, project_root=project_root
        )
        try:
            return cls(environment)
        except Exception:
            environment.close()
            raise

    @property
    def contexts(self) -> Tuple[TrainContextV1, ...]:
        return self._contexts

    def close(self) -> None:
        self._environment.close()

    def _quality_payload_vectors(self, sample_id: str, mode_id: int) -> Dict[str, np.ndarray]:
        if sample_id not in self._allowed_sample_ids:
            raise Run3PreflightError("query refused: sample is not in training partition")
        lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
        q = np.arange(lower, upper + 1, dtype=np.int64)
        rows = self._environment._surface._rows_for(sample_id, mode_id)
        if any(row.grid_split != "fit" for row in rows):
            raise Run3PreflightError("surface returned a non-fit quality row")
        anchor_q = np.asarray([row.q_e4 for row in rows], dtype=np.float64)
        if tuple(int(value) for value in anchor_q) != Q_E4_GRID:
            raise Run3PreflightError("surface anchor q grid drift")
        result: Dict[str, np.ndarray] = {"q_e4": q}
        payload = np.asarray(
            [row.total_transmitted_bytes for row in rows], dtype=np.float64
        )
        if not np.all(np.diff(payload) < 0.0):
            raise Run3PreflightError("payload anchors are not strictly decreasing")
        result["payload_bytes"] = np.interp(q, anchor_q, payload)
        result["datagram_count"] = np.ceil(
            result["payload_bytes"] / UDP_PAYLOAD_CAPACITY_BYTES
        ).astype(np.float64)
        anchor_components: Dict[str, np.ndarray] = {}
        for name in QUALITY_FIELDS:
            values: list[float] = []
            valid_flags: list[bool] = []
            for row in rows:
                component = row.component(name)
                if not component.valid or component.value is None:
                    values.append(math.nan)
                    valid_flags.append(False)
                else:
                    values.append(float(component.value))
                    valid_flags.append(True)
            if all(valid_flags):
                anchors = np.asarray(values, dtype=np.float64)
                anchor_components[name] = anchors
                result[name] = np.interp(q, anchor_q, anchors)
            elif name in RAW_QUALITY_FIELDS:
                anchors = np.asarray(values, dtype=np.float64)
                modeled = np.full(q.shape, np.nan, dtype=np.float64)
                right = np.searchsorted(anchor_q, q, side="left")
                for index, q_value in enumerate(q):
                    r = int(right[index])
                    if r < len(anchor_q) and q_value == anchor_q[r]:
                        if math.isfinite(anchors[r]):
                            modeled[index] = anchors[r]
                    elif 0 < r < len(anchor_q):
                        left = r - 1
                        if math.isfinite(anchors[left]) and math.isfinite(anchors[r]):
                            alpha = (q_value - anchor_q[left]) / (
                                anchor_q[r] - anchor_q[left]
                            )
                            modeled[index] = anchors[left] + alpha * (
                                anchors[r] - anchors[left]
                            )
                result[name] = modeled
            else:
                raise Run3PreflightError(f"required quality component {name} is undefined")
            if name in RAW_QUALITY_FIELDS:
                result[f"{name}_valid_fraction"] = np.isfinite(
                    result[name]
                ).astype(np.float64)

        # Exact nodes obey the nonlinear registered identity.  Interior q uses
        # the qualified same-frame linear interpolation of q_perc itself; it is
        # intentionally not recomputed from separately interpolated factors.
        exact_q_perc = anchor_components["q_loc"] * (
            (1.0 - 0.3) + 0.3 * anchor_components["q_seg"]
        )
        if not np.allclose(
            exact_q_perc, anchor_components["q_perc"], rtol=0.0, atol=2e-12
        ):
            raise Run3PreflightError("exact-anchor Qperc identity drift")
        direct_interp = np.interp(q, anchor_q, anchor_components["q_perc"])
        if not np.array_equal(direct_interp, result["q_perc"]):
            raise Run3PreflightError("Qperc interpolation path drift")
        return result

    def _network_vectors(
        self, network_profile: str, payload: np.ndarray, datagrams: np.ndarray
    ) -> Dict[str, np.ndarray]:
        network = self._environment._network
        model = network.profile_models.get(network_profile)
        if model is None or network_profile not in EXPECTED_PROFILES:
            raise Run3PreflightError("unknown registered network profile")
        expected_datagrams = np.ceil(
            payload / network.contract.udp_payload_capacity_bytes
        ).astype(np.float64)
        if not np.array_equal(datagrams, expected_datagrams):
            raise Run3PreflightError("network datagram relation drift")
        x = np.log(payload)
        reassembly, _ = _curve_vector(model.reassembly_curve, x)
        admission, _ = _curve_vector(model.admission_curve, x)
        reassembly = np.clip(reassembly, 0.0, 1.0)
        admission = np.clip(admission, 0.0, 1.0)
        support = _local_latency_support(model, x)
        latencies: Dict[str, np.ndarray] = {}
        for percentile in ("p50", "p95", "p99"):
            values, curve_support = _curve_vector(model.latency_curves[percentile], x)
            support = np.minimum(support, curve_support)
            latencies[percentile] = values
        if np.any(support < network.contract.latency_min_support):
            raise Run3PreflightError("action lacks qualified latency support")
        p50 = fixed_stage_latency_ms() + latencies["p50"]
        p95 = fixed_stage_latency_ms() + np.maximum(latencies["p50"], latencies["p95"])
        p99 = fixed_stage_latency_ms() + np.maximum(
            np.maximum(latencies["p50"], latencies["p95"]), latencies["p99"]
        )
        return {
            "p_complete_reassembly_given_sent": reassembly,
            "p_edge_admission_given_reassembled": admission,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
        }

    def action_vectors(
        self, sample_id: str, network_profile: str, mode_id: int
    ) -> ActionVectorBatchV1:
        if type(mode_id) is not int or not 0 <= mode_id < 12:
            raise Run3PreflightError("mode_id must be in [0,11]")
        values = self._quality_payload_vectors(sample_id, mode_id)
        network = self._network_vectors(
            network_profile, values["payload_bytes"], values["datagram_count"]
        )
        values.update(network)
        conditional_reward, timeout = _integrate_latency_proxy_vectors(
            values["q_perc"],
            values["latency_p50_ms"],
            values["latency_p95_ms"],
            values["latency_p99_ms"],
        )
        p_admitted = (
            values["p_complete_reassembly_given_sent"]
            * values["p_edge_admission_given_reassembled"]
        )
        p_reassembly_failure = 1.0 - values["p_complete_reassembly_given_sent"]
        p_admission_failure = values["p_complete_reassembly_given_sent"] * (
            1.0 - values["p_edge_admission_given_reassembled"]
        )
        values["p_reassembly_failure"] = p_reassembly_failure
        values["p_admission_failure"] = p_admission_failure
        values["p_admitted"] = p_admitted
        values["p_timeout_given_admitted"] = timeout
        values["p_success_within_deadline_given_admitted"] = 1.0 - timeout
        values["p_service_timeout"] = p_admitted * timeout
        values["p_timely_feedback"] = (
            p_admitted * values["p_success_within_deadline_given_admitted"]
        )
        values["p_total_failure"] = (
            p_reassembly_failure
            + p_admission_failure
            + values["p_service_timeout"]
        )
        values["conditional_admitted_expected_reward"] = conditional_reward
        (
            values["conditional_timely_success_reward"],
            values["conditional_timely_success_reward_valid_fraction"],
        ) = _conditional_timely_success_reward(conditional_reward, timeout)
        values["expected_run3_return"] = (
            -p_reassembly_failure - p_admission_failure + p_admitted * conditional_reward
        )
        for label, multiplier in (("tail_stress_1p25", 1.25), ("tail_stress_1p50", 1.5)):
            stress_reward, stress_timeout = _integrate_latency_proxy_vectors(
                values["q_perc"],
                values["latency_p50_ms"],
                values["latency_p95_ms"],
                values["latency_p99_ms"],
                top_endpoint_multiplier=multiplier,
            )
            values[f"{label}_p_timeout_given_admitted"] = stress_timeout
            values[f"{label}_p_success_within_deadline_given_admitted"] = (
                1.0 - stress_timeout
            )
            values[f"{label}_p_service_timeout"] = p_admitted * stress_timeout
            values[f"{label}_p_timely_feedback"] = p_admitted * values[
                f"{label}_p_success_within_deadline_given_admitted"
            ]
            values[f"{label}_p_total_failure"] = (
                p_reassembly_failure
                + p_admission_failure
                + values[f"{label}_p_service_timeout"]
            )
            values[f"{label}_conditional_admitted_expected_reward"] = (
                stress_reward
            )
            (
                values[f"{label}_conditional_timely_success_reward"],
                values[
                    f"{label}_conditional_timely_success_reward_valid_fraction"
                ],
            ) = _conditional_timely_success_reward(stress_reward, stress_timeout)
            values[f"{label}_expected_run3_return"] = (
                -p_reassembly_failure
                - p_admission_failure
                + p_admitted * stress_reward
            )
        batch = ActionVectorBatchV1(
            sample_id=sample_id,
            network_profile=network_profile,
            mode_id=mode_id,
            q_e4=values.pop("q_e4"),
            metrics=values,
        )
        batch.revalidate()
        return batch


@dataclass(frozen=True, slots=True)
class Run3PreflightResultV1:
    """Canonical compact result independent of an output directory."""

    summary: Mapping[str, Any]
    q_bin_rows: Tuple[Mapping[str, Any], ...]
    explicit_q_rows: Tuple[Mapping[str, Any], ...]
    pareto_rows: Tuple[Mapping[str, Any], ...]
    baseline_rows: Tuple[Mapping[str, Any], ...]
    contextual_winner_rows: Tuple[Mapping[str, Any], ...]

    def canonical_document(self) -> Dict[str, Any]:
        return {
            "baseline_rows": [dict(row) for row in self.baseline_rows],
            "contextual_winner_rows": [
                dict(row) for row in self.contextual_winner_rows
            ],
            "explicit_q_rows": [dict(row) for row in self.explicit_q_rows],
            "pareto_rows": [dict(row) for row in self.pareto_rows],
            "q_bin_rows": [dict(row) for row in self.q_bin_rows],
            "summary": dict(self.summary),
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.canonical_document())


class _Aggregate:
    def __init__(self, shape: Tuple[int, ...] = ()) -> None:
        self.weight = np.zeros(shape, dtype=np.float64)
        self.sums = {
            name: np.zeros(shape, dtype=np.float64) for name in SUMMARY_FIELDS
        }
        self.valid_weight = {
            name: np.zeros(shape, dtype=np.float64)
            for name in (*RAW_QUALITY_FIELDS, *OPTIONAL_METRIC_FIELDS)
        }

    def add(self, metrics: Mapping[str, np.ndarray], weight: float) -> None:
        lengths = {np.asarray(values).size for values in metrics.values()}
        if len(lengths) != 1:
            raise Run3PreflightError("aggregate metric vector lengths disagree")
        count = lengths.pop()
        if self.weight.shape == ():
            self.weight += weight * count
        else:
            if any(np.asarray(values).shape != self.weight.shape for values in metrics.values()):
                raise Run3PreflightError("aggregate metric shape mismatch")
            self.weight += weight
        for name in SUMMARY_FIELDS:
            values = metrics[name]
            if name in self.valid_weight:
                valid = np.isfinite(values)
                contribution = np.where(valid, values, 0.0) * weight
                valid_contribution = valid.astype(np.float64) * weight
                if self.weight.shape == ():
                    contribution = np.sum(contribution)
                    valid_contribution = np.sum(valid_contribution)
                self.sums[name] += contribution
                self.valid_weight[name] += valid_contribution
            else:
                contribution = values * weight
                if self.weight.shape == ():
                    contribution = np.sum(contribution)
                self.sums[name] += contribution

    def means(self) -> Dict[str, np.ndarray]:
        if np.any(self.weight <= 0.0):
            raise Run3PreflightError("aggregate contains an empty support point")
        result = {
            name: self.sums[name] / self.weight for name in SUMMARY_FIELDS
        }
        for name, valid_weight in self.valid_weight.items():
            result[name] = np.divide(
                self.sums[name],
                valid_weight,
                out=np.full_like(self.sums[name], np.nan),
                where=valid_weight > 0.0,
            )
        return result


def _candidate_key_for(
    metrics: Mapping[str, np.ndarray],
    index: int,
    mode_id: int,
    q_e4: int,
    *,
    reward_field: str,
    timely_field: str,
) -> Tuple[float, float, float, int, int]:
    return (
        float(metrics[reward_field][index]),
        float(metrics[timely_field][index]),
        -float(metrics["latency_p50_ms"][index]),
        -mode_id,
        -q_e4,
    )


def _candidate_key(
    metrics: Mapping[str, np.ndarray], index: int, mode_id: int, q_e4: int
) -> Tuple[float, float, float, int, int]:
    return _candidate_key_for(
        metrics,
        index,
        mode_id,
        q_e4,
        reward_field="expected_run3_return",
        timely_field="p_timely_feedback",
    )


def _best_index(batch: ActionVectorBatchV1) -> int:
    best = max(
        range(len(batch.q_e4)),
        key=lambda index: _candidate_key(
            batch.metrics, index, batch.mode_id, int(batch.q_e4[index])
        ),
    )
    return int(best)


def _best_index_for(
    batch: ActionVectorBatchV1, *, reward_field: str, timely_field: str
) -> int:
    return int(
        max(
            range(len(batch.q_e4)),
            key=lambda index: _candidate_key_for(
                batch.metrics,
                index,
                batch.mode_id,
                int(batch.q_e4[index]),
                reward_field=reward_field,
                timely_field=timely_field,
            ),
        )
    )


def _scalar_record(batch: ActionVectorBatchV1, index: int) -> Dict[str, float | int | str | None]:
    row: Dict[str, float | int | str | None] = {
        "sample_id": batch.sample_id,
        "network_profile": batch.network_profile,
        "mode_id": batch.mode_id,
        "q_e4": int(batch.q_e4[index]),
    }
    for name in SUMMARY_FIELDS:
        value = float(batch.metrics[name][index])
        row[name] = None if math.isnan(value) else value
    return row


class Run3TrainOnlyPreflightV1:
    """Streaming analytic engine over a train-only vector source."""

    def __init__(
        self,
        source: Run3TrainVectorSource,
        *,
        expected_scene_count: int = EXPECTED_TRAIN_SCENES,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if source.split != TRAIN_SPLIT:
            raise Run3PreflightError("preflight source must be train-only")
        if source.reward_spec_sha256 != RUN3_REWARD_SPEC_SHA256:
            raise Run3PreflightError("Run-3 reward binding drift")
        if source.kernel_spec_sha256 != RUN3_KERNEL_SPEC_SHA256:
            raise Run3PreflightError("Run-3 kernel binding drift")
        if type(expected_scene_count) is not int or expected_scene_count <= 0:
            raise Run3PreflightError("expected_scene_count must be positive")
        contexts = source.contexts
        if len(contexts) != expected_scene_count:
            raise Run3PreflightError(
                f"training scene count {len(contexts)} != {expected_scene_count}"
            )
        if len({item.sample_id for item in contexts}) != len(contexts):
            raise Run3PreflightError("training contexts contain duplicate sample IDs")
        if any(item.split != TRAIN_SPLIT for item in contexts):
            raise Run3PreflightError("non-training context reached preflight")
        self.source = source
        self.contexts = contexts
        self.progress = progress

    @staticmethod
    def estimate_full_run() -> Dict[str, Any]:
        return {
            "action_evaluations": EXPECTED_CONTEXT_PROFILE_ACTION_EVALUATIONS,
            "materialized_context_action_rows": 0,
            "peak_vector_elements": max(
                upper - lower + 1
                for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS
            ),
            "estimated_output_bytes_upper_bound": 2_000_000,
            "estimated_cpu_runtime_minutes": [5, 20],
            "estimate_status": "ENGINEERING_ESTIMATE_VALIDATE_WITH_FIRST_FULL_RUN",
        }

    def _scalar_probe(self, batch: ActionVectorBatchV1, index: int) -> None:
        proxy = QuantileLatencyProxyV1(
            p50_ms=float(batch.metrics["latency_p50_ms"][index]),
            p95_ms=float(batch.metrics["latency_p95_ms"][index]),
            p99_ms=float(batch.metrics["latency_p99_ms"][index]),
        )
        scalar = expected_run3_reward(
            q_perc=float(batch.metrics["q_perc"][index]),
            p_complete_reassembly_given_sent=float(
                batch.metrics["p_complete_reassembly_given_sent"][index]
            ),
            p_edge_admission_given_reassembled=float(
                batch.metrics["p_edge_admission_given_reassembled"][index]
            ),
            latency_proxy=proxy,
        )
        scalar_success_probability = 1.0 - scalar.p_timeout_given_admitted
        scalar_success_reward = (
            (
                scalar.conditional_admitted_expected_reward
                + scalar.p_timeout_given_admitted
            )
            / scalar_success_probability
            if scalar_success_probability > 0.0
            else math.nan
        )
        scalar_service_timeout = (
            scalar.p_admitted * scalar.p_timeout_given_admitted
        )
        checks = {
            "expected_run3_return": scalar.expected_reward,
            "p_reassembly_failure": scalar.p_reassembly_failure,
            "p_admission_failure": scalar.p_admission_failure,
            "p_admitted": scalar.p_admitted,
            "p_timeout_given_admitted": scalar.p_timeout_given_admitted,
            "p_success_within_deadline_given_admitted": (
                scalar_success_probability
            ),
            "p_service_timeout": scalar_service_timeout,
            "p_timely_feedback": scalar.p_timely_feedback,
            "p_total_failure": (
                scalar.p_reassembly_failure
                + scalar.p_admission_failure
                + scalar_service_timeout
            ),
            "conditional_admitted_expected_reward": (
                scalar.conditional_admitted_expected_reward
            ),
            "conditional_timely_success_reward": scalar_success_reward,
        }
        for name, expected in checks.items():
            observed = float(batch.metrics[name][index])
            agrees = (
                math.isnan(observed) and math.isnan(expected)
            ) or math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=SCALAR_VECTOR_TOLERANCE,
            )
            if not agrees:
                raise Run3PreflightError(
                    f"scalar/vector disagreement for {name}: {observed} vs {expected}"
                )

    def _scalar_stress_probe(
        self,
        batch: ActionVectorBatchV1,
        index: int,
        *,
        label: str,
        multiplier: float,
    ) -> None:
        expected = _scalar_tail_analysis(
            q_perc=float(batch.metrics["q_perc"][index]),
            p_reassembly=float(
                batch.metrics["p_complete_reassembly_given_sent"][index]
            ),
            p_admission=float(
                batch.metrics["p_edge_admission_given_reassembled"][index]
            ),
            p50=float(batch.metrics["latency_p50_ms"][index]),
            p95=float(batch.metrics["latency_p95_ms"][index]),
            p99=float(batch.metrics["latency_p99_ms"][index]),
            top_endpoint_multiplier=multiplier,
        )
        for suffix, scalar in expected.items():
            name = f"{label}_{suffix}"
            observed = float(batch.metrics[name][index])
            agrees = (
                math.isnan(observed) and math.isnan(scalar)
            ) or math.isclose(
                observed,
                scalar,
                rel_tol=0.0,
                abs_tol=SCALAR_VECTOR_TOLERANCE,
            )
            if not agrees:
                raise Run3PreflightError(
                    f"scalar/vector stress disagreement for {name}: "
                    f"{observed} vs {scalar}"
                )

    def run(self) -> Run3PreflightResultV1:
        profiles = EXPECTED_PROFILES
        global_actions: Dict[int, _Aggregate] = {}
        for mode_id, (lower, upper) in enumerate(MODELED_SMOKE_MODE_Q_E4_BOUNDS):
            global_actions[mode_id] = _Aggregate((upper - lower + 1,))
        mode_contextual = [_Aggregate() for _ in range(12)]
        oracle = _Aggregate()
        stress_specs = {"tail_stress_1p25": 1.25, "tail_stress_1p50": 1.5}
        stress_labels = tuple(stress_specs)
        stress_oracles = {label: _Aggregate() for label in stress_labels}
        stress_base_winners = {label: _Aggregate() for label in stress_labels}
        stress_mode_contextual = {
            label: [_Aggregate() for _ in range(12)] for label in stress_labels
        }
        stress_mode_q_weighted_sum = {
            label: np.zeros(12, dtype=np.float64) for label in stress_labels
        }
        stress_mode_weight = {
            label: np.zeros(12, dtype=np.float64) for label in stress_labels
        }
        stress_agreement_counts = {label: 0 for label in stress_labels}
        stress_agreement_weights = {label: 0.0 for label in stress_labels}
        stress_total_weight = 0.0
        q_bins = {name: _Aggregate() for name, _lower, _upper in Q_BINS}
        explicit = {
            (profile, mode_id, q_e4): _Aggregate()
            for profile in profiles
            for mode_id, (lower, upper) in enumerate(MODELED_SMOKE_MODE_Q_E4_BOUNDS)
            for q_e4 in EXPLICIT_Q_E4
            if lower <= q_e4 <= upper
        }
        support_count = 0
        scalar_probe_count = 0
        mode_best_scalar_probe_count = 0
        stress_mode_best_scalar_probe_count = {label: 0 for label in stress_labels}
        stress_contextual_winner_scalar_probe_count = {
            label: 0 for label in stress_labels
        }
        stress_fixed_action_scalar_probe_count = {
            label: 0 for label in stress_labels
        }
        fixed_action_second_pass_scalar_probe_count = 0
        profile_timely_region = {
            profile: {"count": 0, "q_e4_width": 0, "mode_id": None}
            for profile in profiles
        }
        winner_frequency: Dict[Tuple[str, int, int, str], Dict[str, float]] = {}

        for context_index, context in enumerate(self.contexts):
            for profile in profiles:
                contextual_candidates: list[Tuple[ActionVectorBatchV1, int]] = []
                stress_contextual_candidates: Dict[
                    str, list[Tuple[ActionVectorBatchV1, int]]
                ] = {label: [] for label in stress_labels}
                for mode_id in range(12):
                    batch = self.source.action_vectors(
                        context.sample_id, profile, mode_id
                    )
                    batch.revalidate()
                    contiguous_count = _max_contiguous_true(
                        batch.metrics["p_timely_feedback"]
                        > TIMELY_REGION_TOLERANCE
                    )
                    if contiguous_count > profile_timely_region[profile]["count"]:
                        profile_timely_region[profile] = {
                            "count": contiguous_count,
                            "q_e4_width": max(0, contiguous_count - 1),
                            "mode_id": mode_id,
                        }
                    weight = float(context.sampling_weight) / len(profiles)
                    global_actions[mode_id].add(batch.metrics, weight)
                    support_count += len(batch.q_e4)
                    for name, lower, upper in Q_BINS:
                        mask = (batch.q_e4 >= lower) & (batch.q_e4 <= upper)
                        if np.any(mask):
                            q_bins[name].add(
                                {key: values[mask] for key, values in batch.metrics.items()},
                                weight,
                            )
                    for q_e4 in EXPLICIT_Q_E4:
                        matches = np.flatnonzero(batch.q_e4 == q_e4)
                        if len(matches) == 1:
                            explicit[(profile, mode_id, q_e4)].add(
                                {
                                    key: np.asarray([values[int(matches[0])]], dtype=np.float64)
                                    for key, values in batch.metrics.items()
                                },
                                float(context.sampling_weight),
                            )
                    best_index = _best_index(batch)
                    self._scalar_probe(batch, best_index)
                    scalar_probe_count += 1
                    mode_best_scalar_probe_count += 1
                    mode_contextual[mode_id].add(
                        {
                            key: np.asarray([values[best_index]], dtype=np.float64)
                            for key, values in batch.metrics.items()
                        },
                        weight,
                    )
                    contextual_candidates.append((batch, best_index))
                    for label in stress_labels:
                        stress_best_index = _best_index_for(
                            batch,
                            reward_field=f"{label}_expected_run3_return",
                            timely_field=f"{label}_p_timely_feedback",
                        )
                        self._scalar_stress_probe(
                            batch,
                            stress_best_index,
                            label=label,
                            multiplier=stress_specs[label],
                        )
                        stress_mode_best_scalar_probe_count[label] += 1
                        stress_mode_contextual[label][mode_id].add(
                            {
                                key: np.asarray(
                                    [values[stress_best_index]], dtype=np.float64
                                )
                                for key, values in batch.metrics.items()
                            },
                            weight,
                        )
                        stress_mode_q_weighted_sum[label][mode_id] += (
                            weight * int(batch.q_e4[stress_best_index])
                        )
                        stress_mode_weight[label][mode_id] += weight
                        stress_contextual_candidates[label].append(
                            (
                                batch,
                                stress_best_index,
                            )
                        )

                    # Deterministic, bounded probes cover endpoints, midpoint,
                    # and each explicit high-q point when supported.
                    probe_indices = {0, len(batch.q_e4) // 2, len(batch.q_e4) - 1}
                    for q_e4 in EXPLICIT_Q_E4:
                        found = np.flatnonzero(batch.q_e4 == q_e4)
                        if len(found) == 1:
                            probe_indices.add(int(found[0]))
                    if context_index in (0, len(self.contexts) - 1):
                        for probe in sorted(probe_indices.difference({best_index})):
                            self._scalar_probe(batch, probe)
                            scalar_probe_count += 1

                winner_batch, winner_index = max(
                    contextual_candidates,
                    key=lambda item: _candidate_key(
                        item[0].metrics,
                        item[1],
                        item[0].mode_id,
                        int(item[0].q_e4[item[1]]),
                    ),
                )
                oracle.add(
                    {
                        key: np.asarray([values[winner_index]], dtype=np.float64)
                        for key, values in winner_batch.metrics.items()
                    },
                    float(context.sampling_weight) / len(profiles),
                )
                stress_total_weight += float(context.sampling_weight)
                for label in stress_labels:
                    stress_winner_batch, stress_winner_index = max(
                        stress_contextual_candidates[label],
                        key=lambda item, stress_label=label: _candidate_key_for(
                            item[0].metrics,
                            item[1],
                            item[0].mode_id,
                            int(item[0].q_e4[item[1]]),
                            reward_field=f"{stress_label}_expected_run3_return",
                            timely_field=f"{stress_label}_p_timely_feedback",
                        ),
                    )
                    stress_oracles[label].add(
                        {
                            key: np.asarray(
                                [values[stress_winner_index]], dtype=np.float64
                            )
                            for key, values in stress_winner_batch.metrics.items()
                        },
                        float(context.sampling_weight) / len(profiles),
                    )
                    self._scalar_stress_probe(
                        stress_winner_batch,
                        stress_winner_index,
                        label=label,
                        multiplier=stress_specs[label],
                    )
                    stress_contextual_winner_scalar_probe_count[label] += 1
                    stress_base_winners[label].add(
                        {
                            key: np.asarray(
                                [values[winner_index]], dtype=np.float64
                            )
                            for key, values in winner_batch.metrics.items()
                        },
                        float(context.sampling_weight) / len(profiles),
                    )
                    if (
                        stress_winner_batch.mode_id == winner_batch.mode_id
                        and int(stress_winner_batch.q_e4[stress_winner_index])
                        == int(winner_batch.q_e4[winner_index])
                    ):
                        stress_agreement_counts[label] += 1
                        stress_agreement_weights[label] += float(
                            context.sampling_weight
                        )
                winner_q = int(winner_batch.q_e4[winner_index])
                winner_bin = next(
                    name
                    for name, lower, upper in Q_BINS
                    if lower <= winner_q <= upper
                )
                frequency = winner_frequency.setdefault(
                    (profile, winner_batch.mode_id, winner_q, winner_bin),
                    {"count": 0.0, "weight": 0.0},
                )
                frequency["count"] += 1.0
                frequency["weight"] += float(context.sampling_weight)
            if self.progress is not None:
                self.progress(context_index + 1, len(self.contexts))

        expected_support = len(self.contexts) * len(profiles) * EXPECTED_ACTIONS_PER_SCENE
        if support_count != expected_support:
            raise Run3PreflightError(
                f"streamed support count {support_count} != {expected_support}"
            )

        # Choose the fixed action from train aggregates only.
        fixed_candidates = []
        global_means: Dict[int, Dict[str, np.ndarray]] = {}
        for mode_id, aggregate in global_actions.items():
            means = aggregate.means()
            global_means[mode_id] = means
            lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
            q = np.arange(lower, upper + 1, dtype=np.int64)
            index = max(
                range(len(q)),
                key=lambda i: _candidate_key(means, i, mode_id, int(q[i])),
            )
            fixed_candidates.append((mode_id, int(index), q, means))
        fixed_mode, fixed_index, fixed_q_support, fixed_means = max(
            fixed_candidates,
            key=lambda item: _candidate_key(
                item[3], item[1], item[0], int(item[2][item[1]])
            ),
        )
        stress_fixed_optima: Dict[str, Tuple[int, int, int, Dict[str, np.ndarray]]] = {}
        for label in stress_labels:
            candidates = []
            for mode_id, means in global_means.items():
                lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
                q_support = np.arange(lower, upper + 1, dtype=np.int64)
                stress_index = max(
                    range(len(q_support)),
                    key=lambda index, candidate_mode=mode_id, candidate_q=q_support, candidate_means=means, stress_label=label: _candidate_key_for(
                        candidate_means,
                        index,
                        candidate_mode,
                        int(candidate_q[index]),
                        reward_field=f"{stress_label}_expected_run3_return",
                        timely_field=f"{stress_label}_p_timely_feedback",
                    ),
                )
                candidates.append((mode_id, int(stress_index), q_support, means))
            stress_mode, stress_index, stress_q_support, stress_means = max(
                candidates,
                key=lambda item, stress_label=label: _candidate_key_for(
                    item[3],
                    item[1],
                    item[0],
                    int(item[2][item[1]]),
                    reward_field=f"{stress_label}_expected_run3_return",
                    timely_field=f"{stress_label}_p_timely_feedback",
                ),
            )
            stress_fixed_optima[label] = (
                stress_mode,
                stress_index,
                int(stress_q_support[stress_index]),
                stress_means,
            )

        # Select one fixed mode using its train-only contextual-best-q mean.
        mode_means = [item.means() for item in mode_contextual]
        selected_mode = max(
            range(12),
            key=lambda mode: (
                float(mode_means[mode]["expected_run3_return"]),
                float(mode_means[mode]["p_timely_feedback"]),
                -float(mode_means[mode]["latency_p50_ms"]),
                -mode,
            ),
        )

        # Bounded independent pass: make 391 x 4 selected-mode vector queries,
        # consume only the finally selected fixed q, scalar-probe it, and
        # reconcile its aggregate against the vector first pass.  This is not a
        # second all-mode/all-q enumeration.
        fixed_q_e4 = int(fixed_q_support[fixed_index])
        fixed_action_recheck = _Aggregate()
        fixed_lower, _fixed_upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[fixed_mode]
        fixed_recheck_index = fixed_q_e4 - fixed_lower
        for context in self.contexts:
            for profile in profiles:
                batch = self.source.action_vectors(
                    context.sample_id, profile, fixed_mode
                )
                batch.revalidate()
                if int(batch.q_e4[fixed_recheck_index]) != fixed_q_e4:
                    raise Run3PreflightError(
                        "fixed-action second-pass q identity drift"
                    )
                self._scalar_probe(batch, fixed_recheck_index)
                scalar_probe_count += 1
                fixed_action_second_pass_scalar_probe_count += 1
                for label, multiplier in stress_specs.items():
                    self._scalar_stress_probe(
                        batch,
                        fixed_recheck_index,
                        label=label,
                        multiplier=multiplier,
                    )
                    stress_fixed_action_scalar_probe_count[label] += 1
                fixed_action_recheck.add(
                    {
                        name: np.asarray(
                            [values[fixed_recheck_index]], dtype=np.float64
                        )
                        for name, values in batch.metrics.items()
                    },
                    float(context.sampling_weight) / len(profiles),
                )
        fixed_recheck_means = fixed_action_recheck.means()
        fixed_action_aggregate_max_abs_diff = 0.0
        for name in SUMMARY_FIELDS:
            first = float(fixed_means[name][fixed_index])
            second = float(fixed_recheck_means[name])
            if math.isnan(first) and math.isnan(second):
                continue
            if not math.isfinite(first) or not math.isfinite(second):
                raise Run3PreflightError(
                    f"fixed-action aggregate {name} became non-finite"
                )
            difference = abs(first - second)
            fixed_action_aggregate_max_abs_diff = max(
                fixed_action_aggregate_max_abs_diff, difference
            )
            if difference > 1e-13:
                raise Run3PreflightError(
                    f"fixed-action second-pass aggregate mismatch for {name}: "
                    f"{first} vs {second}"
                )

        stress_fixed_recheck_max_abs_diff: Dict[str, float] = {}
        for label, multiplier in stress_specs.items():
            stress_mode, stress_index, stress_q_e4, stress_means = (
                stress_fixed_optima[label]
            )
            stress_lower, _stress_upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[
                stress_mode
            ]
            stress_recheck_index = stress_q_e4 - stress_lower
            stress_recheck = _Aggregate()
            for context in self.contexts:
                for profile in profiles:
                    batch = self.source.action_vectors(
                        context.sample_id, profile, stress_mode
                    )
                    batch.revalidate()
                    if int(batch.q_e4[stress_recheck_index]) != stress_q_e4:
                        raise Run3PreflightError(
                            f"{label} fixed-action q identity drift"
                        )
                    self._scalar_stress_probe(
                        batch,
                        stress_recheck_index,
                        label=label,
                        multiplier=multiplier,
                    )
                    stress_fixed_action_scalar_probe_count[label] += 1
                    stress_recheck.add(
                        {
                            name: np.asarray(
                                [values[stress_recheck_index]], dtype=np.float64
                            )
                            for name, values in batch.metrics.items()
                        },
                        float(context.sampling_weight) / len(profiles),
                    )
            rechecked = stress_recheck.means()
            maximum = 0.0
            for name in SUMMARY_FIELDS:
                first = float(stress_means[name][stress_index])
                second = float(rechecked[name])
                if math.isnan(first) and math.isnan(second):
                    continue
                if not math.isfinite(first) or not math.isfinite(second):
                    raise Run3PreflightError(
                        f"{label} fixed-action aggregate {name} became non-finite"
                    )
                difference = abs(first - second)
                maximum = max(maximum, difference)
                if difference > 1e-13:
                    raise Run3PreflightError(
                        f"{label} fixed-action second-pass mismatch for {name}: "
                        f"{first} vs {second}"
                    )
            stress_fixed_recheck_max_abs_diff[label] = maximum

        baseline_rows: list[Mapping[str, Any]] = []
        baseline_rows.append(
            self._summary_row(
                "TRAIN_SELECTED_FIXED_ACTION",
                fixed_means,
                fixed_index,
                mode_id=fixed_mode,
                q_e4=int(fixed_q_support[fixed_index]),
            )
        )
        baseline_rows.append(
            self._summary_row(
                "TRAIN_SELECTED_FIXED_MODE_CONTEXTUAL_BEST_Q",
                mode_means[selected_mode],
                None,
                mode_id=selected_mode,
                q_e4=None,
            )
        )
        baseline_rows.append(
            self._summary_row(
                "TRAIN_CONTEXTUAL_MODE_AND_Q_ORACLE",
                oracle.means(),
                None,
                mode_id=None,
                q_e4=None,
            )
        )

        q_bin_rows = []
        for name, lower, upper in Q_BINS:
            aggregate = q_bins[name]
            means = aggregate.means()
            q_bin_rows.append(
                self._summary_row(
                    name,
                    means,
                    None,
                    q_e4_lower=lower,
                    q_e4_upper=upper,
                )
            )

        explicit_rows = []
        for key in sorted(explicit):
            profile, mode_id, q_e4 = key
            means = explicit[key].means()
            explicit_rows.append(
                self._summary_row(
                    "EXPLICIT_Q_POINT",
                    means,
                    None,
                    network_profile=profile,
                    mode_id=mode_id,
                    q_e4=q_e4,
                )
            )

        pareto_rows = self._pareto_rows(global_means)
        missing_timely_profiles = sorted(
            profile
            for profile, region in profile_timely_region.items()
            if int(region["count"]) < 2
        )
        if missing_timely_profiles:
            raise Run3PreflightError(
                "no adjacent executable q pair with p_timely_feedback > "
                f"{TIMELY_REGION_TOLERANCE} in profiles "
                + ",".join(missing_timely_profiles)
            )

        profile_weight = {
            profile: sum(
                item["weight"]
                for (candidate, _mode, _q_e4, _bin), item in winner_frequency.items()
                if candidate == profile
            )
            for profile in profiles
        }
        contextual_winner_rows = []
        for (profile, mode_id, q_e4, q_bin), item in sorted(
            winner_frequency.items()
        ):
            contextual_winner_rows.append(
                {
                    "network_profile": profile,
                    "mode_id": mode_id,
                    "q_e4": q_e4,
                    "q_bin": q_bin,
                    "scene_count": int(item["count"]),
                    "weighted_fraction_within_profile": (
                        item["weight"] / profile_weight[profile]
                    ),
                }
            )

        tail_sensitivity_results: Dict[str, Mapping[str, Any]] = {}
        contextual_decision_count = len(self.contexts) * len(profiles)
        for label in stress_labels:
            stress_oracle_means = stress_oracles[label].means()
            stress_base_winner_means = stress_base_winners[label].means()
            stress_mode, stress_index, stress_q_e4, stress_means = (
                stress_fixed_optima[label]
            )
            contextual_oracle_return = float(
                stress_oracle_means[f"{label}_expected_run3_return"]
            )
            base_winner_return = float(
                stress_base_winner_means[f"{label}_expected_run3_return"]
            )
            stress_fixed_return = float(
                stress_means[f"{label}_expected_run3_return"][stress_index]
            )
            base_fixed_return = float(
                fixed_means[f"{label}_expected_run3_return"][fixed_index]
            )
            stress_mode_means = [
                aggregate.means() for aggregate in stress_mode_contextual[label]
            ]
            stress_optimal_mode = max(
                range(12),
                key=lambda mode: (
                    float(
                        stress_mode_means[mode][
                            f"{label}_expected_run3_return"
                        ]
                    ),
                    float(
                        stress_mode_means[mode][f"{label}_p_timely_feedback"]
                    ),
                    -float(stress_mode_means[mode]["latency_p50_ms"]),
                    -mode,
                ),
            )
            base_mode_stress_return = float(
                stress_mode_means[selected_mode][
                    f"{label}_expected_run3_return"
                ]
            )
            stress_optimal_mode_return = float(
                stress_mode_means[stress_optimal_mode][
                    f"{label}_expected_run3_return"
                ]
            )
            per_mode_contextual_best_q = [
                {
                    "mode_id": mode,
                    "weighted_mean_selected_q_e4": float(
                        stress_mode_q_weighted_sum[label][mode]
                        / stress_mode_weight[label][mode]
                    ),
                    "expected_return_under_stress": float(
                        stress_mode_means[mode][
                            f"{label}_expected_run3_return"
                        ]
                    ),
                }
                for mode in range(12)
            ]
            tail_sensitivity_results[label] = {
                "analysis_role": (
                    "COUNTERFACTUAL_TRAIN_ONLY_ANALYSIS_DOES_NOT_SELECT_"
                    "TRAINING_REWARD_COMPARATOR_OR_GO_NO_GO"
                ),
                "contextual_base_winner_agreement_fraction": (
                    stress_agreement_counts[label] / contextual_decision_count
                ),
                "contextual_base_winner_weighted_agreement_fraction": (
                    stress_agreement_weights[label] / stress_total_weight
                ),
                "contextual_base_winner_expected_return_under_stress": (
                    base_winner_return
                ),
                "contextual_stress_oracle_expected_return": (
                    contextual_oracle_return
                ),
                "contextual_stress_oracle_gain_over_base_winner": (
                    contextual_oracle_return - base_winner_return
                ),
                "fixed_action_matches_stress_optimum": (
                    fixed_mode == stress_mode
                    and fixed_q_e4 == stress_q_e4
                ),
                "base_fixed_action_mode_id": fixed_mode,
                "base_fixed_action_q_e4": fixed_q_e4,
                "base_fixed_action_expected_return_under_stress": (
                    base_fixed_return
                ),
                "stress_optimal_fixed_action_mode_id": stress_mode,
                "stress_optimal_fixed_action_q_e4": stress_q_e4,
                "stress_optimal_fixed_action_expected_return": (
                    stress_fixed_return
                ),
                "stress_optimal_fixed_action_gain_over_base_fixed": (
                    stress_fixed_return - base_fixed_return
                ),
                "base_selected_fixed_mode_id": selected_mode,
                "base_selected_fixed_mode_remains_stress_optimal": (
                    selected_mode == stress_optimal_mode
                ),
                "base_selected_mode_contextual_best_q_return_under_stress": (
                    base_mode_stress_return
                ),
                "stress_optimal_contextual_best_q_mode_id": (
                    stress_optimal_mode
                ),
                "stress_optimal_contextual_best_q_mode_return": (
                    stress_optimal_mode_return
                ),
                "base_selected_mode_regret_under_stress": (
                    stress_optimal_mode_return - base_mode_stress_return
                ),
                "per_mode_contextual_best_q_under_stress": (
                    per_mode_contextual_best_q
                ),
            }

        summary = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "PASS",
            "scope": "REGISTERED_391_SCENE_TRAIN_PARTITION_ONLY",
            "profiles": list(profiles),
            "scene_count": len(self.contexts),
            "mode_count": 12,
            "wire_action_count_per_scene": EXPECTED_ACTIONS_PER_SCENE,
            "streamed_context_profile_action_evaluations": support_count,
            "materialized_context_action_rows": 0,
            "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
            "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
            "source_bindings": dict(sorted(self.source.source_bindings.items())),
            "scalar_vector_probe_count": scalar_probe_count,
            "mode_best_scalar_probe_count": mode_best_scalar_probe_count,
            "fixed_action_second_pass_scalar_probe_count": (
                fixed_action_second_pass_scalar_probe_count
            ),
            "fixed_action_second_pass_expected_probe_count": (
                len(self.contexts) * len(profiles)
            ),
            "fixed_action_second_pass_mode_vector_queries": (
                len(self.contexts) * len(profiles)
            ),
            "fixed_action_second_pass_mode_vector_action_values_generated": (
                len(self.contexts)
                * len(profiles)
                * len(fixed_q_support)
            ),
            "fixed_action_aggregate_max_abs_diff": (
                fixed_action_aggregate_max_abs_diff
            ),
            "stress_mode_best_scalar_probe_count": (
                stress_mode_best_scalar_probe_count
            ),
            "stress_contextual_winner_scalar_probe_count": (
                stress_contextual_winner_scalar_probe_count
            ),
            "stress_fixed_action_scalar_probe_count": (
                stress_fixed_action_scalar_probe_count
            ),
            "stress_fixed_action_recheck_max_abs_diff": (
                stress_fixed_recheck_max_abs_diff
            ),
            "scalar_vector_abs_tolerance": SCALAR_VECTOR_TOLERANCE,
            "q_bins": [name for name, _lower, _upper in Q_BINS],
            "explicit_q_e4": list(EXPLICIT_Q_E4),
            "fixed_action_selection": "TRAIN_ONLY_EXPECTED_RUN3_RETURN",
            "fixed_mode_selection": "TRAIN_ONLY_CONTEXTUAL_BEST_Q_EXPECTED_RETURN",
            "contextual_oracle_scope": "TRAIN_ONLY_NOT_ACHIEVABLE_POLICY",
            "tie_rule": (
                "MAX_EXPECTED_RETURN_THEN_MAX_TIMELY_PROBABILITY_THEN_"
                "MIN_P50_THEN_MIN_MODE_ID_THEN_MIN_Q_E4"
            ),
            "quality_interpolation": (
                "EXACT_ANCHORS_ASSERT_QPERC_EQUALS_QLOC_TIMES_0P7_PLUS_0P3_QSEG;"
                "INTERIOR_Q_USES_QUALIFIED_DIRECT_LINEAR_QPERC_INTERPOLATION"
            ),
            "probability_role": (
                "SIMULATOR_KERNEL_FIELDS_ONLY_NOT_REWARD_INPUT_NOT_POLICY_STATE"
            ),
            "conditional_admitted_expected_reward_semantics": (
                "EXPECTED_REWARD_CONDITIONAL_ON_EDGE_ADMISSION_INCLUDES_"
                "MINUS_ONE_SERVICE_TIMEOUT_BRANCH"
            ),
            "conditional_timely_success_reward_semantics": (
                "MEAN_QPERC_MINUS_LATENCY_PENALTY_CONDITIONAL_ON_ADMITTED_"
                "AND_TIMELY;UNDEFINED_NULL_WHEN_TIMELY_MASS_IS_ZERO"
            ),
            "initialization_evidence_scope": (
                "REVERIFIES_HASH_PINNED_FULL_SURFACE_AND_LEGACY_D1_"
                "QUALIFICATION_INCLUDING_LEGACY_HELD_SCENE_EVIDENCE;LOADS_"
                "REWARD_BLIND_FIT_VALIDATION_METADATA"
            ),
            "run3_aggregation_split": "TRAIN_ONLY_391_IDENTITIES",
            "legacy_held_scene_enters_run3_aggregation": False,
            "fit_validation_enters_run3_aggregation": False,
            "stochastic_preflight_sampling": False,
            "future_training_decision_key_contract": FUTURE_DECISION_KEY_CONTRACT,
            "tail_sensitivity_spec": TAIL_SENSITIVITY_SPEC,
            "tail_sensitivity_spec_sha256": TAIL_SENSITIVITY_SPEC_SHA256,
            "tail_sensitivity_decision_role": (
                "ANALYSIS_ONLY_DOES_NOT_AFFECT_GO_NO_GO_TRAINING_REWARD_"
                "ACTION_SELECTION_OR_COMPARATOR_SELECTION"
            ),
            "tail_sensitivity_results": tail_sensitivity_results,
            "timely_region_gate": {
                "threshold_exclusive": TIMELY_REGION_TOLERANCE,
                "minimum_adjacent_executable_q_count": 2,
                "minimum_q_e4_width": 1,
                "profile_maxima": profile_timely_region,
            },
            "run2_winner_or_checkpoint_reused": False,
            "high_q_required_to_win": False,
            "nonzero_timely_action_in_every_profile": True,
            "full_run_estimate": self.estimate_full_run(),
        }
        return Run3PreflightResultV1(
            summary=summary,
            q_bin_rows=tuple(q_bin_rows),
            explicit_q_rows=tuple(explicit_rows),
            pareto_rows=tuple(pareto_rows),
            baseline_rows=tuple(baseline_rows),
            contextual_winner_rows=tuple(contextual_winner_rows),
        )

    @staticmethod
    def _summary_row(
        label: str,
        metrics: Mapping[str, np.ndarray],
        index: Optional[int],
        **identity: Any,
    ) -> Mapping[str, Any]:
        row: Dict[str, Any] = {"label": label, **identity}
        for name in SUMMARY_FIELDS:
            value = metrics[name] if index is None else metrics[name][index]
            scalar = float(np.asarray(value))
            row[name] = None if math.isnan(scalar) else scalar
        return row

    @staticmethod
    def _pareto_rows(
        global_means: Mapping[int, Mapping[str, np.ndarray]]
    ) -> list[Mapping[str, Any]]:
        rows: list[Mapping[str, Any]] = []
        for mode_id, metrics in sorted(global_means.items()):
            lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
            q = np.arange(lower, upper + 1, dtype=np.int64)
            # Scientific intent: expose compression points that preserve
            # localization.  A point is Pareto only when no action in this
            # mode has both at least as much Qloc and no greater payload, with
            # one strict improvement.  Qseg, Qperc and P50 remain explanatory
            # columns; none can change this frontier membership.
            order = np.lexsort((-metrics["q_loc"], metrics["payload_bytes"]))
            best_localization = -math.inf
            pareto = np.zeros(len(q), dtype=bool)
            for index in order:
                localization = float(metrics["q_loc"][index])
                if localization > best_localization:
                    pareto[index] = True
                    best_localization = localization
            for q_e4 in EXPLICIT_Q_E4:
                match = np.flatnonzero(q == q_e4)
                if len(match) != 1:
                    continue
                index = int(match[0])
                row = dict(
                    Run3TrainOnlyPreflightV1._summary_row(
                        "HIGH_Q_LOCALIZATION_PAYLOAD_PARETO_VISIBILITY",
                        metrics,
                        index,
                        mode_id=mode_id,
                        q_e4=q_e4,
                    )
                )
                row["pareto_qloc_vs_payload_within_mode"] = bool(pareto[index])
                rows.append(row)
        return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise Run3PreflightError("refusing to render an empty CSV")
    columns = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    return stream.getvalue().encode("utf-8")


def render_run3_train_only_preflight(
    result: Run3PreflightResultV1, output_dir: Path
) -> Mapping[str, str]:
    """Write deterministic compact artifacts; refuse overwrite."""

    if type(result) is not Run3PreflightResultV1:
        raise Run3PreflightError("result must be exact Run3PreflightResultV1")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    payloads = {
        "preflight.json": canonical_json_bytes(result.canonical_document()),
        "q_bin_summary.csv": _csv_bytes(result.q_bin_rows),
        "explicit_q_summary.csv": _csv_bytes(result.explicit_q_rows),
        "high_q_pareto.csv": _csv_bytes(result.pareto_rows),
        "train_only_baselines.csv": _csv_bytes(result.baseline_rows),
        "contextual_winner_frequency.csv": _csv_bytes(
            result.contextual_winner_rows
        ),
    }
    hashes: Dict[str, str] = {}
    for name in sorted(payloads):
        path = destination / name
        path.write_bytes(payloads[name])
        hashes[name] = hashlib.sha256(payloads[name]).hexdigest()
    manifest = {
        "artifact_sha256": hashes,
        "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        "preflight_result_sha256": result.canonical_sha256(),
        "record": "splitfusion.run3_train_only_preflight_manifest.v1",
        "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
        "status": "PASS",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (destination / "manifest.json").write_bytes(manifest_bytes)
    hashes = dict(hashes)
    hashes["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    return hashes


def run_registered_train_only_preflight(
    *,
    output_dir: Path,
    project_root: Optional[Path] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Run3PreflightResultV1:
    source = RegisteredRun3TrainSourceV1.load_registered(project_root=project_root)
    try:
        result = Run3TrainOnlyPreflightV1(source, progress=progress).run()
        render_run3_train_only_preflight(result, output_dir)
        return result
    finally:
        source.close()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    def progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 25 == 0:
            print(f"run3 train-only preflight: {done}/{total} scenes", flush=True)

    run_registered_train_only_preflight(
        output_dir=args.output,
        project_root=args.project_root,
        progress=progress,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
