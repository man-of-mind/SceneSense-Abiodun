"""Measured-anchor proxy environment for provisional Hybrid-SAC qualification.

This module is an intentionally narrow bridge between the frozen 72-action /
288-cell aggregate evidence and a future training runner.  It is **not** a
scientific simulator, a per-frame reconstruction of the campaign, or a source
of production replay transitions.

Evidence boundary
-----------------

Every object emitted here is labelled :data:`PROXY_EVIDENCE_CLASS` and
:data:`PROXY_USE_RESTRICTION`.  The environment samples from campaign-cell
aggregate terminal frequencies and a piecewise-linear quantile proxy.  It
therefore cannot claim that a sampled outcome was measured on one real frame.
It never constructs ``ReplayTransitionV1`` and must never enter
``ReplayBufferV1``.

The authored network-profile identity remains privileged environment context.
It is absent from :class:`ProxyPolicyObservation`.  The frozen target-SNR
trace is used only to deterministically order proxy draws; it is explicitly a
*design target*, not achieved UE SNR.  No BSR or MCS value is fabricated.

Continuous-q gate
-----------------

The six measured q anchors of each mode are the default action surface.  A
non-anchor q is admitted only if a preregistered leave-one-anchor-out (LOAO)
linear interpolation check passes for quality, payload, delivery/admission
rates and conditional feedback-boundary latency.  Missing conditional latency
support is a validation failure, not a zero-latency observation.  Extrapolation
is always forbidden.  On the current evidence the gate is expected to fail;
that is a useful result, because it leaves an honest 72-anchor baseline rather
than manufacturing continuous-q evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
    EXPECTED_Q_ANCHOR_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)
from .anchor_store import (
    ACTION_SUMMARY_SHA256,
    EXPECTED_CELL_COUNT,
    NETWORK_PROFILE_ORDER,
    PROFILE_LATENCY_SHA256,
    REGISTERED_Q_ANCHORS_E4,
    AnchorEvidenceStore,
    MeasuredAnchorRecord,
    default_anchor_store,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "ANCHOR_PROXY_SCHEMA",
    "CONFIRMED_FAILURE_TERMINALS",
    "ContinuousProxyBlockedError",
    "HiddenProxyContext",
    "InterpolationEstimate",
    "LOAO_METHOD",
    "LOAO_SPEC_SHA256",
    "LOAO_THRESHOLDS",
    "LoaoErrorRecord",
    "LoaoQualificationReport",
    "NondegeneracyReport",
    "PROFILE_DESIGN_RELATIVE_PATH",
    "PROFILE_DESIGN_SHA256",
    "PROXY_EVIDENCE_CLASS",
    "PROXY_USE_RESTRICTION",
    "ProxyActionEstimate",
    "ProxyEnvironmentError",
    "ProxyEvidenceBinding",
    "ProxyExtrapolationError",
    "ProxyOutcome",
    "ProxyOutcomeGroup",
    "ProxyPolicyObservation",
    "ProxyReplayForbiddenError",
    "ProxyRewardConfig",
    "ProxyTransition",
    "SUPERSEDED_CENSORED_TERMINALS",
    "TARGET_SNR_TRACE_RELATIVE_PATH",
    "TARGET_SNR_TRACE_SHA256",
    "TARGET_SNR_VALUE_SEMANTICS",
    "TargetSnrDesignPoint",
    "TargetSnrTraceStore",
    "MeasuredAnchorProxy",
    "MeasuredAnchorProxyEnvironment",
    "load_default_anchor_proxy",
]


ANCHOR_PROXY_SCHEMA = "splitfusion_anchor_proxy_environment_v1"
PROXY_EVIDENCE_CLASS = "PROXY_COUNTERFACTUAL_TRAINING_ONLY"
PROXY_USE_RESTRICTION = (
    "Aggregate-evidence proxy for provisional baseline qualification only. "
    "NOT a measured per-frame transition, NOT production replay, NOT a "
    "testbed-training result, and NOT evidence for an unmeasured continuous-q "
    "action."
)

PROFILE_DESIGN_RELATIVE_PATH = "rl_agent/configs/network_profile_design_v2.json"
PROFILE_DESIGN_SHA256 = (
    "056247e5731c1ae9ac281432034e1b79d1e6da24ab4a2d579a7fe2d85917e483"
)
TARGET_SNR_TRACE_RELATIVE_PATH = (
    "rl_agent/experiments/network_profile_design_v2/"
    "20260822_route_b_v2/traces.csv"
)
TARGET_SNR_TRACE_SHA256 = (
    "32f1be66e976cba322803c128eefdeb81a8d31ed64bd253ff85ea1ab0583303d"
)
TARGET_SNR_VALUE_SEMANTICS = "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR"

# These thresholds are declared in code before any admission decision.  They
# are deliberately interpretable in the units of each metric.  Passing means
# *all* held-out interior anchors satisfy *all* applicable limits.
LOAO_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "combined_quality_abs": 0.05,
        "payload_log_ratio_abs": math.log(1.25),
        "reassembly_rate_abs": 0.10,
        "admission_rate_abs": 0.10,
        "sensor_model_ready_p50_relative": 0.20,
    }
)
LOAO_METHOD = (
    "leave each interior q anchor out; linearly interpolate between the nearest "
    "retained lower/upper anchors within the same mode; require every "
    "metric/profile case to meet its preregistered limit"
)
LOAO_SPEC_SHA256 = canonical_sha256(
    {
        "interior_q_e4": list(REGISTERED_Q_ANCHORS_E4[1:-1]),
        "method": LOAO_METHOD,
        "missing_support_rule": "FAIL_CLOSED_NOT_ZERO",
        "schema": "splitfusion_anchor_proxy_loao_spec_v1",
        "thresholds": dict(LOAO_THRESHOLDS),
    }
)

_INTERIOR_Q_E4 = REGISTERED_Q_ANCHORS_E4[1:-1]
_PUBLISHED_TERMINAL = "terminal_result_published"

CONFIRMED_FAILURE_TERMINALS = frozenset(
    {
        "terminal_measured_pre_queue_rejection",
        "terminal_predicted_map_install_horizon_exceeded",
        "terminal_processing_horizon_expired_after_compute",
        "terminal_processing_horizon_expired_after_publication",
        "terminal_processing_horizon_expired_at_arrival",
        "terminal_processing_horizon_expired_before_compute",
        "terminal_processing_horizon_expired_before_publication",
        "terminal_queue_wait_budget_exceeded",
        "terminal_transport_incomplete",
    }
)
SUPERSEDED_CENSORED_TERMINALS = frozenset(
    {
        "terminal_superseded_pending",
        "terminal_superseded_publication_pending",
    }
)


class ProxyEnvironmentError(ValueError):
    """Base error for a violated proxy boundary."""


class ProxyEvidenceError(ProxyEnvironmentError):
    """Pinned aggregate or target-trace evidence is missing or altered."""


class ContinuousProxyBlockedError(ProxyEnvironmentError):
    """A non-anchor q was requested although LOAO qualification failed."""


class ProxyExtrapolationError(ProxyEnvironmentError):
    """A request lies outside the measured q support."""


class ProxyReplayForbiddenError(ProxyEnvironmentError):
    """A proxy transition was offered as production replay evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProxyEvidenceError(message)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_pinned(path: Path, expected_sha256: str, label: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProxyEvidenceError(f"cannot read {label} at {path}: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    _require(
        actual == expected_sha256,
        f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}",
    )
    return raw


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProxyEnvironmentError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ProxyEnvironmentError(f"{name} must be finite")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ProxyEnvironmentError(f"{name} must be in [0, 1], got {result}")
    return result


def _counter_uniform(seed: str, stream: str, *parts: Any) -> float:
    if not isinstance(seed, str) or not seed:
        raise ProxyEnvironmentError("seed must be a non-empty string")
    payload = json.dumps(
        {"parts": parts, "seed": seed, "stream": stream},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)


@dataclass(frozen=True, slots=True)
class ProxyEvidenceBinding:
    """Hashes that make one proxy construction reproducible."""

    catalog_sha256: str
    action_summary_sha256: str
    profile_latency_sha256: str
    anchor_store_sha256: str
    profile_design_sha256: str
    target_snr_trace_sha256: str
    loao_spec_sha256: str
    evidence_class: str = PROXY_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        expected = {
            "catalog_sha256": CATALOG_SHA256,
            "action_summary_sha256": ACTION_SUMMARY_SHA256,
            "profile_latency_sha256": PROFILE_LATENCY_SHA256,
            "profile_design_sha256": PROFILE_DESIGN_SHA256,
            "target_snr_trace_sha256": TARGET_SNR_TRACE_SHA256,
            "loao_spec_sha256": LOAO_SPEC_SHA256,
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise ProxyEvidenceError(
                    f"proxy binding {name} drift: expected {wanted}, "
                    f"got {getattr(self, name)}"
                )
        if self.evidence_class != PROXY_EVIDENCE_CLASS:
            raise ProxyEvidenceError("proxy binding cannot claim measured evidence")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "schema": ANCHOR_PROXY_SCHEMA,
            "catalog_sha256": self.catalog_sha256,
            "action_summary_sha256": self.action_summary_sha256,
            "profile_latency_sha256": self.profile_latency_sha256,
            "anchor_store_sha256": self.anchor_store_sha256,
            "profile_design_sha256": self.profile_design_sha256,
            "target_snr_trace_sha256": self.target_snr_trace_sha256,
            "loao_spec_sha256": self.loao_spec_sha256,
            "evidence_class": self.evidence_class,
        }


@dataclass(frozen=True, slots=True)
class TargetSnrDesignPoint:
    """One hidden design-target trace point (never achieved UE SNR)."""

    step_index: int
    target_snr_db: float
    state: str
    state_index: int
    trace_id: str
    value_semantics: str = TARGET_SNR_VALUE_SEMANTICS

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ProxyEvidenceError("target-SNR step_index must be non-negative int")
        _finite(self.target_snr_db, "target_snr_db")
        if self.state not in ("ADVERSE", "INTERMEDIATE", "FAVORABLE"):
            raise ProxyEvidenceError(f"unknown target-SNR state {self.state!r}")
        if self.state_index != ("ADVERSE", "INTERMEDIATE", "FAVORABLE").index(
            self.state
        ):
            raise ProxyEvidenceError("target-SNR state index disagrees with state")
        if self.value_semantics != TARGET_SNR_VALUE_SEMANTICS:
            raise ProxyEvidenceError("target trace is not explicitly design-target SNR")

    @property
    def hidden_driver_token(self) -> str:
        return canonical_sha256(
            {
                "state": self.state,
                "step_index": self.step_index,
                "target_snr_db": self.target_snr_db,
                "trace_id": self.trace_id,
                "value_semantics": self.value_semantics,
            }
        )


@dataclass(frozen=True, slots=True)
class TargetSnrTraceStore:
    """Pinned privileged target-SNR traces for the four authored profiles."""

    points_by_profile: Mapping[str, Tuple[TargetSnrDesignPoint, ...]]
    design_sha256: str
    trace_sha256: str
    claim_boundary: str

    @classmethod
    def load_default(cls) -> "TargetSnrTraceStore":
        root = _project_root()
        design_raw = _read_pinned(
            root / PROFILE_DESIGN_RELATIVE_PATH,
            PROFILE_DESIGN_SHA256,
            "network profile design",
        )
        trace_raw = _read_pinned(
            root / TARGET_SNR_TRACE_RELATIVE_PATH,
            TARGET_SNR_TRACE_SHA256,
            "target-SNR traces",
        )
        try:
            document = json.loads(design_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProxyEvidenceError(f"invalid network profile design JSON: {exc}") from exc
        _require(document.get("schema") == "scenesense.network_profile_design.v2", "wrong profile-design schema")
        claim = document.get("claim_boundary")
        _require(
            claim
            == "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR_RFSIM_ACTUATION_REQUIRES_CALIBRATION",
            "profile design lost its target-not-achieved claim boundary",
        )
        profiles = document.get("profiles")
        _require(isinstance(profiles, list), "profile design has no profile list")
        definitions = {item["profile_id"]: item for item in profiles}
        _require(tuple(definitions) == NETWORK_PROFILE_ORDER, "profile order disagrees with anchor evidence")

        decoded = trace_raw.decode("utf-8").splitlines()
        reader = csv.DictReader(decoded)
        required = {
            "profile_id",
            "trace_id",
            "seed",
            "value_semantics",
            "step_index",
            "interval_start_s",
            "interval_end_s",
            "state_index",
            "state",
            "target_snr_db",
        }
        _require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "target trace columns are incomplete")
        grouped: Dict[str, list[TargetSnrDesignPoint]] = {
            profile: [] for profile in NETWORK_PROFILE_ORDER
        }
        for row_index, row in enumerate(reader, start=2):
            profile = row["profile_id"]
            _require(profile in grouped, f"trace row {row_index}: foreign profile {profile!r}")
            definition = definitions[profile]
            _require(row["trace_id"] == definition["trace_id"], f"trace row {row_index}: trace_id drift")
            _require(int(row["seed"]) == definition["seed"], f"trace row {row_index}: seed drift")
            step = int(row["step_index"])
            _require(step == len(grouped[profile]), f"trace row {row_index}: non-contiguous step")
            start = float(row["interval_start_s"])
            end = float(row["interval_end_s"])
            _require(abs(start - step * 0.1) < 1e-12, f"trace row {row_index}: wrong interval start")
            _require(abs(end - (step + 1) * 0.1) < 1e-12, f"trace row {row_index}: wrong interval end")
            grouped[profile].append(
                TargetSnrDesignPoint(
                    step_index=step,
                    target_snr_db=float(row["target_snr_db"]),
                    state=row["state"],
                    state_index=int(row["state_index"]),
                    trace_id=row["trace_id"],
                    value_semantics=row["value_semantics"],
                )
            )
        expected_count = int(document["route"]["sample_count"])
        for profile, points in grouped.items():
            _require(len(points) == expected_count, f"{profile}: expected {expected_count} trace points, got {len(points)}")
        return cls(
            points_by_profile=MappingProxyType(
                {profile: tuple(grouped[profile]) for profile in NETWORK_PROFILE_ORDER}
            ),
            design_sha256=PROFILE_DESIGN_SHA256,
            trace_sha256=TARGET_SNR_TRACE_SHA256,
            claim_boundary=str(claim),
        )

    def point(self, hidden_profile: str, step_index: int) -> TargetSnrDesignPoint:
        if hidden_profile not in self.points_by_profile:
            raise ProxyEnvironmentError(f"unknown hidden profile {hidden_profile!r}")
        if type(step_index) is not int or step_index < 0:
            raise ProxyEnvironmentError("step_index must be non-negative int")
        points = self.points_by_profile[hidden_profile]
        return points[step_index % len(points)]


@dataclass(frozen=True, slots=True)
class LoaoErrorRecord:
    mode_id: int
    held_out_q_e4: int
    metric: str
    profile: Optional[str]
    actual: Optional[float]
    predicted: Optional[float]
    normalized_error: Optional[float]
    threshold: float
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class LoaoQualificationReport:
    method: str
    thresholds: Mapping[str, float]
    records: Tuple[LoaoErrorRecord, ...]
    continuous_q_qualified: bool
    missing_support_count: int
    failed_count: int
    max_error_by_metric: Mapping[str, Optional[float]]
    failure_count_by_metric: Mapping[str, int]
    spec_sha256: str = LOAO_SPEC_SHA256
    evidence_class: str = PROXY_EVIDENCE_CLASS

    @property
    def evaluated_count(self) -> int:
        return len(self.records)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "thresholds": dict(self.thresholds),
            "continuous_q_qualified": self.continuous_q_qualified,
            "evaluated_count": self.evaluated_count,
            "missing_support_count": self.missing_support_count,
            "failed_count": self.failed_count,
            "max_error_by_metric": dict(self.max_error_by_metric),
            "failure_count_by_metric": dict(self.failure_count_by_metric),
            "spec_sha256": self.spec_sha256,
            "evidence_class": self.evidence_class,
            "records": [
                {
                    "mode_id": item.mode_id,
                    "held_out_q_e4": item.held_out_q_e4,
                    "metric": item.metric,
                    "profile": item.profile,
                    "actual": item.actual,
                    "predicted": item.predicted,
                    "normalized_error": item.normalized_error,
                    "threshold": item.threshold,
                    "passed": item.passed,
                    "reason": item.reason,
                }
                for item in self.records
            ],
        }


def _linear(x: int, x0: int, y0: float, x1: int, y1: float) -> float:
    if not x0 < x < x1:
        raise ProxyExtrapolationError(
            f"linear interpolation requires {x0} < {x} < {x1}"
        )
    return y0 + (x - x0) / float(x1 - x0) * (y1 - y0)


def _quality(record: MeasuredAnchorRecord) -> Optional[float]:
    return record.quality.derived_presentation_quality["combined_quality"]


def _payload(record: MeasuredAnchorRecord, profile: str) -> Optional[float]:
    return record.outcome(profile).measured_payload_bytes[
        "replay_v3__median_payload_bytes"
    ]


def _rate(record: MeasuredAnchorRecord, profile: str, rate: str) -> Optional[float]:
    return record.outcome(profile).rates[rate]


def _feedback_p50(record: MeasuredAnchorRecord, profile: str) -> Optional[float]:
    return record.outcome(profile).latency_stat("sensor_model_ready").p50_ms


def _metric_error(metric: str, actual: float, predicted: float) -> float:
    if metric == "payload_log_ratio_abs":
        if actual <= 0.0 or predicted <= 0.0:
            return math.inf
        return abs(math.log(predicted / actual))
    if metric == "sensor_model_ready_p50_relative":
        return abs(predicted - actual) / max(abs(actual), 1e-12)
    return abs(predicted - actual)


def _loao_metric(
    mode_id: int,
    q: int,
    metric: str,
    profile: Optional[str],
    lower: MeasuredAnchorRecord,
    held: MeasuredAnchorRecord,
    upper: MeasuredAnchorRecord,
    getter: Any,
) -> LoaoErrorRecord:
    threshold = LOAO_THRESHOLDS[metric]
    values = (
        getter(lower, profile) if profile is not None else getter(lower),
        getter(held, profile) if profile is not None else getter(held),
        getter(upper, profile) if profile is not None else getter(upper),
    )
    if any(value is None for value in values):
        return LoaoErrorRecord(
            mode_id,
            q,
            metric,
            profile,
            values[1],
            None,
            None,
            threshold,
            False,
            "MISSING_CONDITIONAL_SUPPORT",
        )
    y0, actual, y1 = (float(value) for value in values)
    predicted = _linear(q, lower.quality.q_e4, y0, upper.quality.q_e4, y1)
    error = _metric_error(metric, actual, predicted)
    return LoaoErrorRecord(
        mode_id,
        q,
        metric,
        profile,
        actual,
        predicted,
        error,
        threshold,
        error <= threshold,
        "WITHIN_THRESHOLD" if error <= threshold else "ERROR_EXCEEDS_THRESHOLD",
    )


def _build_loao_report(store: AnchorEvidenceStore) -> LoaoQualificationReport:
    errors: list[LoaoErrorRecord] = []
    for mode_id in range(EXPECTED_MODE_COUNT):
        anchors = store.records_for_mode(mode_id)
        by_q = {record.quality.q_e4: record for record in anchors}
        for q in _INTERIOR_Q_E4:
            retained = [candidate for candidate in REGISTERED_Q_ANCHORS_E4 if candidate != q]
            lower_q = max(candidate for candidate in retained if candidate < q)
            upper_q = min(candidate for candidate in retained if candidate > q)
            lower, held, upper = by_q[lower_q], by_q[q], by_q[upper_q]
            errors.append(
                _loao_metric(
                    mode_id,
                    q,
                    "combined_quality_abs",
                    None,
                    lower,
                    held,
                    upper,
                    _quality,
                )
            )
            for profile in NETWORK_PROFILE_ORDER:
                specifications = (
                    (
                        "payload_log_ratio_abs",
                        lambda record, selected: _payload(record, selected),
                    ),
                    (
                        "reassembly_rate_abs",
                        lambda record, selected: _rate(
                            record,
                            selected,
                            "replay_v3__rate_reassembled_per_sent",
                        ),
                    ),
                    (
                        "admission_rate_abs",
                        lambda record, selected: _rate(
                            record,
                            selected,
                            "replay_v3__rate_admitted_per_sent",
                        ),
                    ),
                    (
                        "sensor_model_ready_p50_relative",
                        lambda record, selected: _feedback_p50(record, selected),
                    ),
                )
                for metric, getter in specifications:
                    errors.append(
                        _loao_metric(
                            mode_id,
                            q,
                            metric,
                            profile,
                            lower,
                            held,
                            upper,
                            getter,
                        )
                    )
    maxima: Dict[str, Optional[float]] = {}
    for metric in LOAO_THRESHOLDS:
        finite_errors = [
            item.normalized_error
            for item in errors
            if item.metric == metric and item.normalized_error is not None
        ]
        maxima[metric] = max(finite_errors) if finite_errors else None
    missing = sum(item.reason == "MISSING_CONDITIONAL_SUPPORT" for item in errors)
    failed = sum(not item.passed for item in errors)
    failures_by_metric = {
        metric: sum(not item.passed and item.metric == metric for item in errors)
        for metric in LOAO_THRESHOLDS
    }
    return LoaoQualificationReport(
        method=LOAO_METHOD,
        thresholds=LOAO_THRESHOLDS,
        records=tuple(errors),
        continuous_q_qualified=failed == 0,
        missing_support_count=missing,
        failed_count=failed,
        max_error_by_metric=MappingProxyType(maxima),
        failure_count_by_metric=MappingProxyType(failures_by_metric),
    )


@dataclass(frozen=True, slots=True)
class ProxyActionEstimate:
    """Exact measured anchor projected into the proxy's outcome coordinates."""

    action_id: int
    mode_id: int
    q_e4: int
    combined_quality: float
    payload_bytes: float
    reassembly_rate: float
    admission_rate: float
    sensor_model_ready_p50_ms: Optional[float]
    anchor_record_sha256: str
    evidence_class: str = PROXY_EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class InterpolationEstimate:
    """A gated non-anchor estimate and its explicit LOAO uncertainty."""

    mode_id: int
    q_e4: int
    lower_q_e4: int
    upper_q_e4: int
    values: Mapping[str, float]
    max_loao_error_by_metric: Mapping[str, Optional[float]]
    evidence_class: str = PROXY_EVIDENCE_CLASS


@dataclass(frozen=True, slots=True)
class MeasuredAnchorProxy:
    """Bound anchor surface plus its independently computed LOAO gate."""

    store: AnchorEvidenceStore
    traces: TargetSnrTraceStore
    binding: ProxyEvidenceBinding
    loao: LoaoQualificationReport

    @classmethod
    def bind(
        cls,
        store: Optional[AnchorEvidenceStore] = None,
        traces: Optional[TargetSnrTraceStore] = None,
    ) -> "MeasuredAnchorProxy":
        bound_store = store if store is not None else default_anchor_store()
        bound_traces = traces if traces is not None else TargetSnrTraceStore.load_default()
        _require(bound_store.anchor_count == 72, "anchor proxy requires exactly 72 anchors")
        _require(bound_store.cell_count == EXPECTED_CELL_COUNT, "anchor proxy requires exactly 288 cells")
        _require(bound_store.contract.mode_count == EXPECTED_MODE_COUNT, "anchor proxy requires exactly 12 modes")
        for mode_id in range(EXPECTED_MODE_COUNT):
            anchors = bound_store.records_for_mode(mode_id)
            _require(len(anchors) == EXPECTED_Q_ANCHOR_COUNT, f"mode {mode_id}: expected six anchors")
            _require(tuple(item.quality.q_e4 for item in anchors) == REGISTERED_Q_ANCHORS_E4, f"mode {mode_id}: q-anchor drift")
        loao = _build_loao_report(bound_store)
        binding = ProxyEvidenceBinding(
            catalog_sha256=bound_store.contract.catalog_sha256,
            action_summary_sha256=bound_store.action_summary_sha256,
            profile_latency_sha256=bound_store.profile_latency_sha256,
            anchor_store_sha256=bound_store.canonical_sha256(),
            profile_design_sha256=bound_traces.design_sha256,
            target_snr_trace_sha256=bound_traces.trace_sha256,
            loao_spec_sha256=loao.spec_sha256,
        )
        return cls(
            store=bound_store,
            traces=bound_traces,
            binding=binding,
            loao=loao,
        )

    @property
    def continuous_q_enabled(self) -> bool:
        return self.loao.continuous_q_qualified

    def exact_action(self, action_id: int, hidden_profile: str) -> ProxyActionEstimate:
        if hidden_profile not in NETWORK_PROFILE_ORDER:
            raise ProxyEnvironmentError(f"unknown hidden profile {hidden_profile!r}")
        record = self.store.by_action_id(action_id)
        outcome = record.outcome(hidden_profile)
        quality = _quality(record)
        payload = _payload(record, hidden_profile)
        reassembly = _rate(
            record, hidden_profile, "replay_v3__rate_reassembled_per_sent"
        )
        admission = _rate(
            record, hidden_profile, "replay_v3__rate_admitted_per_sent"
        )
        if quality is None or payload is None or reassembly is None or admission is None:
            raise ProxyEvidenceError(f"action {action_id}: required anchor evidence missing")
        return ProxyActionEstimate(
            action_id=action_id,
            mode_id=record.quality.mode_id,
            q_e4=record.quality.q_e4,
            combined_quality=_unit(quality, "combined_quality"),
            payload_bytes=_finite(payload, "payload_bytes"),
            reassembly_rate=_unit(reassembly, "reassembly_rate"),
            admission_rate=_unit(admission, "admission_rate"),
            sensor_model_ready_p50_ms=_feedback_p50(record, hidden_profile),
            anchor_record_sha256=record.canonical_sha256(),
        )

    def interpolate(
        self, mode_id: int, q_e4: int, hidden_profile: str
    ) -> InterpolationEstimate:
        if type(q_e4) is not int:
            raise ProxyEnvironmentError("q_e4 must be exact int")
        if q_e4 < Q_E4_MIN or q_e4 > Q_E4_MAX:
            raise ProxyExtrapolationError(
                f"q_e4={q_e4} is outside measured mechanical support"
            )
        if q_e4 in REGISTERED_Q_ANCHORS_E4:
            raise ProxyEnvironmentError("exact anchors must use exact_action")
        if not self.continuous_q_enabled:
            raise ContinuousProxyBlockedError(
                "continuous-q proxy is blocked: LOAO failed "
                f"{self.loao.failed_count}/{self.loao.evaluated_count} cases "
                f"({self.loao.missing_support_count} missing-support cases)"
            )
        anchors = self.store.records_for_mode(mode_id)
        lower = max((r for r in anchors if r.quality.q_e4 < q_e4), key=lambda r: r.quality.q_e4, default=None)
        upper = min((r for r in anchors if r.quality.q_e4 > q_e4), key=lambda r: r.quality.q_e4, default=None)
        if lower is None or upper is None:
            raise ProxyExtrapolationError("non-anchor request lacks two measured brackets")
        metrics = {
            "combined_quality": (_quality(lower), _quality(upper)),
            "payload_bytes": (_payload(lower, hidden_profile), _payload(upper, hidden_profile)),
            "reassembly_rate": (
                _rate(lower, hidden_profile, "replay_v3__rate_reassembled_per_sent"),
                _rate(upper, hidden_profile, "replay_v3__rate_reassembled_per_sent"),
            ),
            "admission_rate": (
                _rate(lower, hidden_profile, "replay_v3__rate_admitted_per_sent"),
                _rate(upper, hidden_profile, "replay_v3__rate_admitted_per_sent"),
            ),
            "sensor_model_ready_p50_ms": (
                _feedback_p50(lower, hidden_profile),
                _feedback_p50(upper, hidden_profile),
            ),
        }
        if any(value is None for pair in metrics.values() for value in pair):
            raise ContinuousProxyBlockedError("bracketing anchors lack conditional support")
        values = {
            name: _linear(
                q_e4,
                lower.quality.q_e4,
                float(pair[0]),
                upper.quality.q_e4,
                float(pair[1]),
            )
            for name, pair in metrics.items()
        }
        return InterpolationEstimate(
            mode_id=mode_id,
            q_e4=q_e4,
            lower_q_e4=lower.quality.q_e4,
            upper_q_e4=upper.quality.q_e4,
            values=MappingProxyType(values),
            max_loao_error_by_metric=self.loao.max_error_by_metric,
        )


class ProxyOutcomeGroup(str, Enum):
    PUBLISHED_WITH_LATENCY = "PUBLISHED_WITH_LATENCY"
    PUBLISHED_LATENCY_UNOBSERVED = "PUBLISHED_LATENCY_UNOBSERVED"
    CONFIRMED_FAILURE_RIGHT_CENSORED = "CONFIRMED_FAILURE_RIGHT_CENSORED"
    SUPERSEDED_RIGHT_CENSORED = "SUPERSEDED_RIGHT_CENSORED"


@dataclass(frozen=True, slots=True)
class ProxyRewardConfig:
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    deadline_budget_ms: float = 200.0
    timeout_or_failure_penalty: float = -1.0
    superseded_censored_penalty: float = -0.5

    def __post_init__(self) -> None:
        if _finite(self.quality_weight, "quality_weight") < 0.0:
            raise ProxyEnvironmentError("quality_weight must be non-negative")
        if _finite(self.latency_weight, "latency_weight") < 0.0:
            raise ProxyEnvironmentError("latency_weight must be non-negative")
        if _finite(self.deadline_budget_ms, "deadline_budget_ms") <= 0.0:
            raise ProxyEnvironmentError("deadline_budget_ms must be positive")
        if _finite(self.timeout_or_failure_penalty, "timeout_or_failure_penalty") >= 0.0:
            raise ProxyEnvironmentError("failure penalty must be negative")
        if _finite(self.superseded_censored_penalty, "superseded_censored_penalty") >= 0.0:
            raise ProxyEnvironmentError("superseded penalty must be negative")

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the complete reward rule in a deterministic representation."""
        return {
            "schema": "splitfusion_anchor_proxy_reward_config_v1",
            "quality_weight": self.quality_weight,
            "latency_weight": self.latency_weight,
            "deadline_budget_ms": self.deadline_budget_ms,
            "timeout_or_failure_penalty": self.timeout_or_failure_penalty,
            "superseded_censored_penalty": self.superseded_censored_penalty,
        }

    def canonical_sha256(self) -> str:
        """Hash used to distinguish proxy environments with different rewards."""
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class ProxyPolicyObservation:
    """Policy-safe proxy observation with no authored-profile leakage.

    The proxy has no measured per-decision SNR, BSR or MCS.  Those values are
    represented as unavailable, not as zeros.  The only causal information is
    the prior proxy outcome and action.  This is intentionally not the frozen
    31-feature production vector.
    """

    step_index: int
    previous_action_id: Optional[int]
    previous_reward: Optional[float]
    previous_published: Optional[bool]
    previous_confirmed_failure: Optional[bool]
    previous_right_censored: Optional[bool]
    achieved_snr_db: Optional[float] = None
    bsr_bytes: Optional[int] = None
    mcs_index: Optional[int] = None
    evidence_class: str = PROXY_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ProxyEnvironmentError("observation step_index must be non-negative int")
        if self.achieved_snr_db is not None or self.bsr_bytes is not None or self.mcs_index is not None:
            raise ProxyEnvironmentError("proxy must not fabricate achieved SNR, BSR or MCS")
        if self.evidence_class != PROXY_EVIDENCE_CLASS:
            raise ProxyEnvironmentError("proxy observation cannot claim measured evidence")

    def policy_dict(self) -> Dict[str, Any]:
        """Return the complete policy view; no hidden profile/trace is present."""
        return {
            "step_index": self.step_index,
            "previous_action_id": self.previous_action_id,
            "previous_reward": self.previous_reward,
            "previous_published": self.previous_published,
            "previous_confirmed_failure": self.previous_confirmed_failure,
            "previous_right_censored": self.previous_right_censored,
            "achieved_snr_db": None,
            "achieved_snr_available": False,
            "bsr_bytes": None,
            "bsr_available": False,
            "mcs_index": None,
            "mcs_available": False,
            "evidence_class": self.evidence_class,
        }


@dataclass(frozen=True, slots=True)
class HiddenProxyContext:
    """Privileged driver; never hand this object to a policy."""

    network_profile: str
    target_snr_design: TargetSnrDesignPoint
    target_snr_is_achieved_measurement: bool = False

    def __post_init__(self) -> None:
        if self.network_profile not in NETWORK_PROFILE_ORDER:
            raise ProxyEnvironmentError("invalid hidden network profile")
        if self.target_snr_is_achieved_measurement:
            raise ProxyEnvironmentError("target-SNR design must not claim achieved SNR")


@dataclass(frozen=True, slots=True)
class ProxyOutcome:
    source_terminal: str
    group: ProxyOutcomeGroup
    latency_ms: Optional[float]
    right_censored: bool
    confirmed_failure: bool
    published: bool
    deadline_met: Optional[bool]
    reward: float
    quality_anchor: float
    evidence_class: str = PROXY_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        _finite(self.reward, "reward")
        _unit(self.quality_anchor, "quality_anchor")
        if self.latency_ms is not None and _finite(self.latency_ms, "latency_ms") < 0.0:
            raise ProxyEnvironmentError("latency cannot be negative")
        if self.right_censored and self.latency_ms is not None:
            raise ProxyEnvironmentError("right-censored outcome cannot carry invented latency")


@dataclass(frozen=True, slots=True)
class ProxyTransition:
    observation: ProxyPolicyObservation
    action_id: int
    outcome: ProxyOutcome
    next_observation: ProxyPolicyObservation
    done: bool
    proxy_sequence_id: str
    step_index: int
    binding_sha256: str
    evidence_class: str = PROXY_EVIDENCE_CLASS
    use_restriction: str = PROXY_USE_RESTRICTION
    replay_admissible: bool = False

    def as_replay_transition(self) -> None:
        raise ProxyReplayForbiddenError(
            "PROXY_COUNTERFACTUAL_TRAINING_ONLY is not a measured per-frame "
            "transition and may not enter ReplayTransitionV1/ReplayBufferV1"
        )


@dataclass(frozen=True, slots=True)
class NondegeneracyReport:
    profile: str
    action_count: int
    unique_expected_rewards: int
    reward_min: float
    reward_max: float
    best_action_id: int
    best_expected_reward: float
    evidence_class: str = PROXY_EVIDENCE_CLASS

    @property
    def nondegenerate(self) -> bool:
        return self.action_count == 72 and self.unique_expected_rewards > 1 and self.reward_max > self.reward_min


def _sample_quantile(stat: Any, u: float) -> Optional[float]:
    """Piecewise-linear inverse-CDF proxy through p50/p95/p99.

    The lower endpoint is ``max(0, 2*p50-p95)`` and the upper endpoint is
    p99.  This is a transparent proxy distribution, not reconstructed raw data.
    """
    if not stat.observed:
        return None
    p50, p95, p99 = stat.p50_ms, stat.p95_ms, stat.p99_ms
    if p50 is None or p95 is None or p99 is None:
        return None
    points = (
        (0.0, max(0.0, 2.0 * p50 - p95)),
        (0.5, p50),
        (0.95, p95),
        (0.99, p99),
        (1.0, p99),
    )
    for (u0, y0), (u1, y1) in zip(points, points[1:]):
        if u <= u1:
            if u1 == u0:
                return float(y1)
            return float(y0 + (u - u0) / (u1 - u0) * (y1 - y0))
    return float(p99)


def _terminal_from_counts(terminal_counts: Mapping[str, int], frames: int, u: float) -> str:
    ordered = tuple(sorted(terminal_counts))
    threshold = u * frames
    cumulative = 0
    for terminal in ordered:
        cumulative += terminal_counts[terminal]
        if threshold < cumulative:
            return terminal
    # u is in [0,1), so exact accounting should make this unreachable.
    raise ProxyEvidenceError("terminal counts failed to cover the proxy draw")


@dataclass(slots=True)
class MeasuredAnchorProxyEnvironment:
    """Deterministic sequential 72-anchor qualification environment."""

    proxy: MeasuredAnchorProxy
    hidden_profile: str
    seed: str
    horizon_steps: int = 512
    reward_config: ProxyRewardConfig = ProxyRewardConfig()
    _step_index: int = field(init=False, repr=False)
    _previous_action_id: Optional[int] = field(init=False, repr=False)
    _previous_outcome: Optional[ProxyOutcome] = field(init=False, repr=False)
    _sequence_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hidden_profile not in NETWORK_PROFILE_ORDER:
            raise ProxyEnvironmentError(f"unknown hidden profile {self.hidden_profile!r}")
        if not isinstance(self.seed, str) or not self.seed:
            raise ProxyEnvironmentError("seed must be non-empty string")
        if type(self.horizon_steps) is not int or self.horizon_steps <= 0:
            raise ProxyEnvironmentError("horizon_steps must be positive int")
        self._step_index = 0
        self._previous_action_id: Optional[int] = None
        self._previous_outcome: Optional[ProxyOutcome] = None
        self._sequence_id = canonical_sha256(
            {
                "binding": self.proxy.binding.to_canonical_dict(),
                "hidden_profile": self.hidden_profile,
                "horizon_steps": self.horizon_steps,
                "loao_spec_sha256": self.proxy.loao.spec_sha256,
                "reward_config": self.reward_config.to_canonical_dict(),
                "reward_config_sha256": self.reward_config.canonical_sha256(),
                "schema": "splitfusion_anchor_proxy_environment_identity_v1",
                "seed": self.seed,
            }
        )

    @property
    def environment_identity_sha256(self) -> str:
        """Identity of evidence, LOAO rule, reward rule and episode controls."""
        return self._sequence_id

    def _observation(self) -> ProxyPolicyObservation:
        outcome = self._previous_outcome
        return ProxyPolicyObservation(
            step_index=self._step_index,
            previous_action_id=self._previous_action_id,
            previous_reward=None if outcome is None else outcome.reward,
            previous_published=None if outcome is None else outcome.published,
            previous_confirmed_failure=None if outcome is None else outcome.confirmed_failure,
            previous_right_censored=None if outcome is None else outcome.right_censored,
        )

    def reset(self) -> ProxyPolicyObservation:
        self._step_index = 0
        self._previous_action_id = None
        self._previous_outcome = None
        return self._observation()

    def hidden_context(self) -> HiddenProxyContext:
        return HiddenProxyContext(
            network_profile=self.hidden_profile,
            target_snr_design=self.proxy.traces.point(
                self.hidden_profile, self._step_index
            ),
        )

    def _outcome(self, action_id: int, context: HiddenProxyContext) -> ProxyOutcome:
        record = self.proxy.store.by_action_id(action_id)
        cell = record.outcome(self.hidden_profile)
        token = context.target_snr_design.hidden_driver_token
        terminal_u = _counter_uniform(
            self.seed, "terminal", self._step_index, action_id, token
        )
        terminal = _terminal_from_counts(cell.terminal_counts, cell.frames_sent, terminal_u)
        quality = _quality(record)
        if quality is None:
            raise ProxyEvidenceError(f"action {action_id}: combined quality absent")
        quality = _unit(quality, "combined_quality")
        config = self.reward_config
        if terminal == _PUBLISHED_TERMINAL:
            latency_u = _counter_uniform(
                self.seed, "latency", self._step_index, action_id, token
            )
            latency = _sample_quantile(
                cell.latency_stat("sensor_model_ready"), latency_u
            )
            if latency is None:
                return ProxyOutcome(
                    source_terminal=terminal,
                    group=ProxyOutcomeGroup.PUBLISHED_LATENCY_UNOBSERVED,
                    latency_ms=None,
                    right_censored=True,
                    confirmed_failure=False,
                    published=True,
                    deadline_met=None,
                    reward=config.superseded_censored_penalty,
                    quality_anchor=quality,
                )
            deadline_met = latency <= config.deadline_budget_ms
            reward = (
                config.quality_weight * quality
                - config.latency_weight * latency / config.deadline_budget_ms
                if deadline_met
                else config.timeout_or_failure_penalty
            )
            return ProxyOutcome(
                source_terminal=terminal,
                group=ProxyOutcomeGroup.PUBLISHED_WITH_LATENCY,
                latency_ms=latency,
                right_censored=False,
                confirmed_failure=False,
                published=True,
                deadline_met=deadline_met,
                reward=reward,
                quality_anchor=quality,
            )
        if terminal in SUPERSEDED_CENSORED_TERMINALS:
            return ProxyOutcome(
                source_terminal=terminal,
                group=ProxyOutcomeGroup.SUPERSEDED_RIGHT_CENSORED,
                latency_ms=None,
                right_censored=True,
                confirmed_failure=False,
                published=False,
                deadline_met=None,
                reward=config.superseded_censored_penalty,
                quality_anchor=quality,
            )
        if terminal not in CONFIRMED_FAILURE_TERMINALS:
            raise ProxyEvidenceError(f"unclassified terminal {terminal!r}")
        return ProxyOutcome(
            source_terminal=terminal,
            group=ProxyOutcomeGroup.CONFIRMED_FAILURE_RIGHT_CENSORED,
            latency_ms=None,
            right_censored=True,
            confirmed_failure=True,
            published=False,
            deadline_met=False,
            reward=config.timeout_or_failure_penalty,
            quality_anchor=quality,
        )

    def step(self, action_id: int) -> ProxyTransition:
        if self._step_index >= self.horizon_steps:
            raise ProxyEnvironmentError("episode is complete; call reset")
        # This lookup is the discrete 72-anchor action gate.
        self.proxy.store.by_action_id(action_id)
        observation = self._observation()
        context = self.hidden_context()
        outcome = self._outcome(action_id, context)
        current_step = self._step_index
        self._previous_action_id = action_id
        self._previous_outcome = outcome
        self._step_index += 1
        done = self._step_index >= self.horizon_steps
        return ProxyTransition(
            observation=observation,
            action_id=action_id,
            outcome=outcome,
            next_observation=self._observation(),
            done=done,
            proxy_sequence_id=self._sequence_id,
            step_index=current_step,
            binding_sha256=canonical_sha256(self.proxy.binding.to_canonical_dict()),
        )

    def _expected_reward(self, action_id: int, quantile_grid: int = 401) -> float:
        record = self.proxy.store.by_action_id(action_id)
        cell = record.outcome(self.hidden_profile)
        quality = _quality(record)
        if quality is None:
            raise ProxyEvidenceError("quality missing")
        cfg = self.reward_config
        stat = cell.latency_stat("sensor_model_ready")
        published_count = cell.terminal_counts[_PUBLISHED_TERMINAL]
        if published_count and stat.observed:
            published_rewards = []
            for index in range(quantile_grid):
                latency = _sample_quantile(stat, (index + 0.5) / quantile_grid)
                if latency is None:
                    published_rewards.append(cfg.superseded_censored_penalty)
                elif latency <= cfg.deadline_budget_ms:
                    published_rewards.append(
                        cfg.quality_weight * quality
                        - cfg.latency_weight * latency / cfg.deadline_budget_ms
                    )
                else:
                    published_rewards.append(cfg.timeout_or_failure_penalty)
            published_reward = sum(published_rewards) / len(published_rewards)
        else:
            published_reward = cfg.superseded_censored_penalty
        superseded = sum(
            cell.terminal_counts[name] for name in SUPERSEDED_CENSORED_TERMINALS
        )
        failures = cell.frames_sent - published_count - superseded
        return (
            published_count * published_reward
            + superseded * cfg.superseded_censored_penalty
            + failures * cfg.timeout_or_failure_penalty
        ) / cell.frames_sent

    def nondegeneracy_report(self) -> NondegeneracyReport:
        rewards = [
            (self._expected_reward(record.action_id), record.action_id)
            for record in self.proxy.store.records
        ]
        best_reward, best_action = max(rewards)
        values = [item[0] for item in rewards]
        return NondegeneracyReport(
            profile=self.hidden_profile,
            action_count=len(rewards),
            unique_expected_rewards=len({round(value, 12) for value in values}),
            reward_min=min(values),
            reward_max=max(values),
            best_action_id=best_action,
            best_expected_reward=best_reward,
        )


def load_default_anchor_proxy() -> MeasuredAnchorProxy:
    """Load all pinned evidence and evaluate the continuous-q gate."""
    return MeasuredAnchorProxy.bind()
