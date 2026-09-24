"""Run-4 exploration schedule, feedback coverage gate, and actor diagnostics.

This module is deliberately independent of the environment and trainer.  It
does not load evidence, initialize CUDA, import a live service, or take a
gradient.  A caller supplies the exact per-mode q support and every gate
threshold explicitly.

The warm-up schedule is action-stratified but state-agnostic: the environment
must first produce its next naturally evolving causal state, then consume the
next scheduled action.  No radio or scene state is synthesized, selected, or
relabelled to make the gate pass.  The final scheduled action must appear as
the previous action in one additional natural feedback state before gradient
start is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
    Q_E4_SCALE,
)


REQUIRED_STATE_FEATURES: Tuple[str, ...] = (
    "scene_si",
    "scene_p40",
    "ue_dl_snr_db",
    "rlc_backlog_bytes",
)
REQUIRED_OUTCOME_METRICS: Tuple[str, ...] = (
    "previous_quality",
    "previous_latency_ms",
)
RUN4_FEEDBACK_DEADLINE_MS = 170.0

_OBSERVATION_KEYS = frozenset((*REQUIRED_STATE_FEATURES, "previous"))
_PREVIOUS_KEYS = frozenset(
    ("mode_id", "q_e4", "success", "quality", "latency_ms")
)
_FORBIDDEN_POLICY_KEY_PARTS = (
    "network_profile",
    "profile_id",
    "profile_label",
    "frame_id",
    "session",
    "uuid",
    "decision_id",
    "reward",
    "future",
    "next_",
)


class ExplorationError(ValueError):
    """Base class for exploration-contract failures."""


class ScheduleExhausted(ExplorationError):
    """The finite preregistered warm-up schedule has no remaining action."""


class CoverageRecordError(ExplorationError):
    """A coverage record is malformed, duplicated, or out of causal order."""


class GradientStartRefused(RuntimeError):
    """The preregistered exploration and state-evidence gate did not pass."""


class ActorPathError(ExplorationError):
    """An actor sampling path does not match its training/evaluation phase."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ExplorationError("value is not canonical-JSON encodable") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ExplorationError(f"{name} must be a positive exact int")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ExplorationError(f"{name} must be a non-negative exact int")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ExplorationError(f"{name} must be a real number, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ExplorationError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class QualityBin:
    """One inclusive equal-width stratum of an integer q support."""

    index: int
    lower_e4: int
    upper_e4: int

    @property
    def point_count(self) -> int:
        return self.upper_e4 - self.lower_e4 + 1

    def contains(self, q_e4: int) -> bool:
        return self.lower_e4 <= q_e4 <= self.upper_e4


def partition_quality_support(
    lower_e4: int, upper_e4: int, bin_count: int
) -> Tuple[QualityBin, ...]:
    """Partition an inclusive integer support without gaps or overlap.

    The bins contain either ``floor(N / K)`` or ``ceil(N / K)`` wire points,
    where ``N = upper - lower + 1``.  This is derived from the supplied exact
    bounds; measured catalog anchor locations are never consulted.
    """

    if type(lower_e4) is not int or type(upper_e4) is not int:
        raise ExplorationError("q bounds must be exact ints")
    _positive_int(bin_count, "bin_count")
    if not Q_E4_MIN <= lower_e4 <= upper_e4 <= Q_E4_MAX:
        raise ExplorationError(
            f"q bounds [{lower_e4}, {upper_e4}] escape "
            f"[{Q_E4_MIN}, {Q_E4_MAX}]"
        )
    point_count = upper_e4 - lower_e4 + 1
    if bin_count > point_count:
        raise ExplorationError(
            f"{bin_count} bins cannot partition only {point_count} q points"
        )
    result = []
    for index in range(bin_count):
        start = lower_e4 + (index * point_count) // bin_count
        stop = lower_e4 + ((index + 1) * point_count) // bin_count - 1
        result.append(QualityBin(index=index, lower_e4=start, upper_e4=stop))
    if (
        result[0].lower_e4 != lower_e4
        or result[-1].upper_e4 != upper_e4
        or any(left.upper_e4 + 1 != right.lower_e4 for left, right in zip(result, result[1:]))
        or max(item.point_count for item in result)
        - min(item.point_count for item in result)
        > 1
    ):
        raise ExplorationError("internal q-stratum partition failure")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class WarmupScheduleConfig:
    """Complete constructor-bound warm-up schedule contract."""

    mode_q_e4_bounds: Tuple[Tuple[int, int], ...]
    q_bin_count: int
    samples_per_q_bin: int
    master_seed: int
    support_contract_id: str

    def __post_init__(self) -> None:
        if type(self.mode_q_e4_bounds) is not tuple or len(
            self.mode_q_e4_bounds
        ) != EXPECTED_MODE_COUNT:
            raise ExplorationError(
                f"mode_q_e4_bounds must contain exactly {EXPECTED_MODE_COUNT} pairs"
            )
        _positive_int(self.q_bin_count, "q_bin_count")
        _positive_int(self.samples_per_q_bin, "samples_per_q_bin")
        _nonnegative_int(self.master_seed, "master_seed")
        if type(self.support_contract_id) is not str or not self.support_contract_id:
            raise ExplorationError("support_contract_id must be a non-empty string")
        for mode_id, pair in enumerate(self.mode_q_e4_bounds):
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not int
                or type(pair[1]) is not int
            ):
                raise ExplorationError(f"mode {mode_id} q support is not an int pair")
            partition_quality_support(pair[0], pair[1], self.q_bin_count)

    @property
    def action_count(self) -> int:
        return (
            EXPECTED_MODE_COUNT * self.q_bin_count * self.samples_per_q_bin
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "master_seed": self.master_seed,
            "mode_q_e4_bounds": [list(pair) for pair in self.mode_q_e4_bounds],
            "q_bin_count": self.q_bin_count,
            "record": "run4_stratified_warmup_schedule_v1",
            "samples_per_q_bin": self.samples_per_q_bin,
            "support_contract_id": self.support_contract_id,
        }

    @property
    def schedule_id(self) -> str:
        return _sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class WarmupAction:
    """One counter-addressable warm-up action."""

    warmup_ordinal: int
    mode_id: int
    q_e4: int
    q_bin_index: int
    q_bin_lower_e4: int
    q_bin_upper_e4: int
    schedule_id: str
    counter_identity: str

    @property
    def requested_q(self) -> float:
        return self.q_e4 / float(Q_E4_SCALE)


def _derived_seed(config: WarmupScheduleConfig, *parts: object) -> int:
    material = {
        "domain": "RUN4_STRATIFIED_WARMUP_LOCAL_RNG_V1",
        "master_seed": config.master_seed,
        "parts": list(parts),
        "schedule_id": config.schedule_id,
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(material)).digest()[:8], "big")


def _representative_q(
    q_bin: QualityBin, repetition: int, repetition_count: int
) -> int:
    """Evenly cover a bin, including both ends when repetitions permit.

    Integer ties use half-up rounding.  One sample uses the half-up midpoint;
    two or more samples include the exact inclusive boundaries.
    """

    if repetition_count == 1:
        return (q_bin.lower_e4 + q_bin.upper_e4 + 1) // 2
    denominator = repetition_count - 1
    span = q_bin.upper_e4 - q_bin.lower_e4
    offset = (2 * repetition * span + denominator) // (2 * denominator)
    return q_bin.lower_e4 + offset


class StratifiedWarmupSchedule:
    """Balanced, locally permuted schedule over every mode/q stratum.

    Each consecutive block of 12 actions contains every mode exactly once.
    Within a repetition, every mode visits every q bin exactly once.  Ordering
    uses only private ``random.Random`` instances derived from the config and
    never advances Python's module-level RNG.
    """

    def __init__(self, config: WarmupScheduleConfig) -> None:
        if type(config) is not WarmupScheduleConfig:
            raise ExplorationError("config must be an exact WarmupScheduleConfig")
        config.__post_init__()
        self.config = config
        self._bins = tuple(
            partition_quality_support(lower, upper, config.q_bin_count)
            for lower, upper in config.mode_q_e4_bounds
        )
        self._actions = self._build_actions()

    def _build_actions(self) -> Tuple[WarmupAction, ...]:
        cells: list[tuple[int, int, int]] = []
        for repetition in range(self.config.samples_per_q_bin):
            per_mode_bins: list[list[int]] = []
            for mode_id in range(EXPECTED_MODE_COUNT):
                order = list(range(self.config.q_bin_count))
                random.Random(
                    _derived_seed(self.config, "q-bin-order", repetition, mode_id)
                ).shuffle(order)
                per_mode_bins.append(order)
            for layer in range(self.config.q_bin_count):
                mode_order = list(range(EXPECTED_MODE_COUNT))
                random.Random(
                    _derived_seed(self.config, "mode-order", repetition, layer)
                ).shuffle(mode_order)
                for mode_id in mode_order:
                    cells.append((mode_id, per_mode_bins[mode_id][layer], repetition))

        actions = []
        for ordinal, (mode_id, bin_index, repetition) in enumerate(cells):
            q_bin = self._bins[mode_id][bin_index]
            q_e4 = _representative_q(
                q_bin, repetition, self.config.samples_per_q_bin
            )
            identity_document = {
                "mode_id": mode_id,
                "q_bin_index": bin_index,
                "q_e4": q_e4,
                "record": "run4_warmup_counter_identity_v1",
                "schedule_id": self.config.schedule_id,
                "warmup_ordinal": ordinal,
            }
            actions.append(
                WarmupAction(
                    warmup_ordinal=ordinal,
                    mode_id=mode_id,
                    q_e4=q_e4,
                    q_bin_index=bin_index,
                    q_bin_lower_e4=q_bin.lower_e4,
                    q_bin_upper_e4=q_bin.upper_e4,
                    schedule_id=self.config.schedule_id,
                    counter_identity=_sha256(identity_document),
                )
            )
        if len(actions) != self.config.action_count:
            raise ExplorationError("warm-up action-count construction drift")
        if len({action.counter_identity for action in actions}) != len(actions):
            raise ExplorationError("warm-up counter identities are not unique")
        return tuple(actions)

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> Tuple[WarmupAction, ...]:
        return self._actions

    def action_at(self, warmup_ordinal: int) -> WarmupAction:
        if type(warmup_ordinal) is not int or warmup_ordinal < 0:
            raise ExplorationError("warmup_ordinal must be a non-negative exact int")
        if warmup_ordinal >= len(self._actions):
            raise ScheduleExhausted(
                f"warm-up ordinal {warmup_ordinal} is outside [0, {len(self._actions)})"
            )
        return self._actions[warmup_ordinal]

    def bins_for_mode(self, mode_id: int) -> Tuple[QualityBin, ...]:
        if type(mode_id) is not int or not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise ExplorationError("mode_id is outside the registered inventory")
        return self._bins[mode_id]

    def classify_q(self, mode_id: int, q_e4: int) -> int:
        if type(q_e4) is not int:
            raise ExplorationError("q_e4 must be an exact int")
        for q_bin in self.bins_for_mode(mode_id):
            if q_bin.contains(q_e4):
                return q_bin.index
        lower, upper = self.config.mode_q_e4_bounds[mode_id]
        raise ExplorationError(
            f"q_e4 {q_e4} is outside mode {mode_id} support [{lower}, {upper}]"
        )


@dataclass(frozen=True, slots=True)
class PreviousDecisionObservation:
    """Policy-visible result of the immediately preceding decision.

    Raw reward is intentionally absent.  Success/failure, quality, and latency
    are the concise causal outcome signals needed by the next decision.
    """

    mode_id: int
    q_e4: int
    success: bool
    quality: Optional[float]
    latency_ms: Optional[float]

    def __post_init__(self) -> None:
        if type(self.mode_id) is not int or not 0 <= self.mode_id < EXPECTED_MODE_COUNT:
            raise CoverageRecordError("previous mode_id is outside [0, 12)")
        if type(self.q_e4) is not int or not Q_E4_MIN <= self.q_e4 <= Q_E4_MAX:
            raise CoverageRecordError("previous q_e4 is outside wire bounds")
        if type(self.success) is not bool:
            raise CoverageRecordError("previous success must be an exact bool")
        for name in ("quality", "latency_ms"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, numbers.Real)
            ):
                raise CoverageRecordError(f"previous {name} must be real or None")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.success:
            if self.quality is None or not math.isfinite(self.quality):
                raise CoverageRecordError(
                    "successful previous outcome requires finite quality"
                )
            if not 0.0 <= self.quality <= 1.0:
                raise CoverageRecordError(
                    "successful previous quality must lie in [0,1]"
                )
            if self.latency_ms is None or not math.isfinite(self.latency_ms):
                raise CoverageRecordError(
                    "successful previous outcome requires finite latency"
                )
            if not 0.0 <= self.latency_ms <= RUN4_FEEDBACK_DEADLINE_MS:
                raise CoverageRecordError(
                    "successful previous latency must lie in [0,170] ms"
                )
        elif self.quality is not None or self.latency_ms is not None:
            raise CoverageRecordError(
                "failed/timeout previous outcome requires absent quality and latency"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreviousDecisionObservation":
        if not isinstance(value, Mapping):
            raise CoverageRecordError("previous must be a mapping or None")
        keys = set(value)
        if keys != _PREVIOUS_KEYS:
            raise CoverageRecordError(
                f"previous fields must be exactly {sorted(_PREVIOUS_KEYS)}, got {sorted(keys)}"
            )
        return cls(
            mode_id=value["mode_id"],
            q_e4=value["q_e4"],
            success=value["success"],
            quality=value["quality"],
            latency_ms=value["latency_ms"],
        )


@dataclass(frozen=True, slots=True)
class CoverageObservation:
    """Only the causal inputs whose warm-up coverage is audited."""

    scene_si: float
    scene_p40: float
    ue_dl_snr_db: float
    rlc_backlog_bytes: float
    previous: Optional[PreviousDecisionObservation]

    def __post_init__(self) -> None:
        # Non-finite values are retained so the gate can report and refuse
        # them, rather than making missing/invalid telemetry disappear.
        for name in REQUIRED_STATE_FEATURES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise CoverageRecordError(f"{name} must be a real number")
            object.__setattr__(self, name, float(value))
        if self.previous is not None and type(self.previous) is not PreviousDecisionObservation:
            raise CoverageRecordError(
                "previous must be an exact PreviousDecisionObservation or None"
            )

    @property
    def previous_present(self) -> bool:
        return self.previous is not None

    def state_values(self) -> Tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in REQUIRED_STATE_FEATURES)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CoverageObservation":
        if not isinstance(value, Mapping):
            raise CoverageRecordError("coverage observation must be a mapping")
        keys = set(value)
        forbidden = sorted(
            key
            for key in keys
            if not isinstance(key, str)
            or any(part in key.lower() for part in _FORBIDDEN_POLICY_KEY_PARTS)
        )
        if forbidden:
            raise CoverageRecordError(
                f"identifiers/profile/reward fields are not policy observations: {forbidden}"
            )
        if keys != _OBSERVATION_KEYS:
            raise CoverageRecordError(
                f"observation fields must be exactly {sorted(_OBSERVATION_KEYS)}, got {sorted(keys)}"
            )
        previous_raw = value["previous"]
        previous = (
            None
            if previous_raw is None
            else PreviousDecisionObservation.from_mapping(previous_raw)
        )
        return cls(
            scene_si=value["scene_si"],
            scene_p40=value["scene_p40"],
            ue_dl_snr_db=value["ue_dl_snr_db"],
            rlc_backlog_bytes=value["rlc_backlog_bytes"],
            previous=previous,
        )


@dataclass(frozen=True, slots=True)
class StateFeatureThreshold:
    """Caller-preregistered variation/saturation limits for one state input."""

    name: str
    min_finite_count: int
    min_unique_values: int
    min_span: float
    min_per_mode_finite_count: int
    min_per_mode_unique_values: int
    min_per_mode_span: float
    saturation_lower: float
    saturation_upper: float
    saturation_tolerance: float
    max_boundary_saturation_fraction: float

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_STATE_FEATURES:
            raise ExplorationError(f"unknown required state feature {self.name!r}")
        _positive_int(self.min_finite_count, "min_finite_count")
        if _positive_int(self.min_unique_values, "min_unique_values") < 2:
            raise ExplorationError("min_unique_values must enforce nonconstant evidence")
        span = _finite_float(self.min_span, "min_span")
        if span <= 0.0:
            raise ExplorationError("min_span must be positive")
        _positive_int(self.min_per_mode_finite_count, "min_per_mode_finite_count")
        if _positive_int(
            self.min_per_mode_unique_values, "min_per_mode_unique_values"
        ) < 2:
            raise ExplorationError(
                "min_per_mode_unique_values must enforce nonconstant evidence"
            )
        per_mode_span = _finite_float(self.min_per_mode_span, "min_per_mode_span")
        if per_mode_span <= 0.0:
            raise ExplorationError("min_per_mode_span must be positive")
        lower = _finite_float(self.saturation_lower, "saturation_lower")
        upper = _finite_float(self.saturation_upper, "saturation_upper")
        tolerance = _finite_float(self.saturation_tolerance, "saturation_tolerance")
        if upper <= lower:
            raise ExplorationError("saturation_upper must exceed saturation_lower")
        if tolerance < 0.0 or 2.0 * tolerance >= upper - lower:
            raise ExplorationError("saturation_tolerance is invalid for its interval")
        fraction = _finite_float(
            self.max_boundary_saturation_fraction,
            "max_boundary_saturation_fraction",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ExplorationError("max saturation fraction must lie in [0,1]")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OutcomeMetricThreshold:
    """Caller-preregistered availability/variation requirement for feedback."""

    name: str
    min_finite_count: int
    min_unique_values: int
    min_span: float

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_OUTCOME_METRICS:
            raise ExplorationError(f"unknown outcome metric {self.name!r}")
        _positive_int(self.min_finite_count, "min_finite_count")
        if _positive_int(self.min_unique_values, "min_unique_values") < 2:
            raise ExplorationError("outcome min_unique_values must be at least two")
        if _finite_float(self.min_span, "min_span") <= 0.0:
            raise ExplorationError("outcome min_span must be positive")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CoverageGateConfig:
    """No defaults: every provisional Run-4 gate value is caller-bound."""

    preregistration_id: str
    min_current_actions_per_mode: int
    min_current_actions_per_q_bin: int
    min_previous_actions_per_mode: int
    min_previous_actions_per_q_bin: int
    min_previous_present: int
    min_previous_success: int
    min_previous_failure: int
    state_thresholds: Tuple[StateFeatureThreshold, ...]
    outcome_thresholds: Tuple[OutcomeMetricThreshold, ...]

    def __post_init__(self) -> None:
        if type(self.preregistration_id) is not str or not self.preregistration_id:
            raise ExplorationError("preregistration_id must be a non-empty string")
        for name in (
            "min_current_actions_per_mode",
            "min_current_actions_per_q_bin",
            "min_previous_actions_per_mode",
            "min_previous_actions_per_q_bin",
            "min_previous_present",
            "min_previous_success",
            "min_previous_failure",
        ):
            _positive_int(getattr(self, name), name)
        if type(self.state_thresholds) is not tuple or {
            item.name for item in self.state_thresholds if type(item) is StateFeatureThreshold
        } != set(REQUIRED_STATE_FEATURES) or len(self.state_thresholds) != len(
            REQUIRED_STATE_FEATURES
        ):
            raise ExplorationError(
                "state_thresholds must contain each required feature exactly once"
            )
        if any(type(item) is not StateFeatureThreshold for item in self.state_thresholds):
            raise ExplorationError("state threshold has a foreign type")
        if type(self.outcome_thresholds) is not tuple or {
            item.name for item in self.outcome_thresholds if type(item) is OutcomeMetricThreshold
        } != set(REQUIRED_OUTCOME_METRICS) or len(self.outcome_thresholds) != len(
            REQUIRED_OUTCOME_METRICS
        ):
            raise ExplorationError(
                "outcome_thresholds must contain quality and latency exactly once"
            )
        if any(type(item) is not OutcomeMetricThreshold for item in self.outcome_thresholds):
            raise ExplorationError("outcome threshold has a foreign type")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "min_current_actions_per_mode": self.min_current_actions_per_mode,
            "min_current_actions_per_q_bin": self.min_current_actions_per_q_bin,
            "min_previous_actions_per_mode": self.min_previous_actions_per_mode,
            "min_previous_actions_per_q_bin": self.min_previous_actions_per_q_bin,
            "min_previous_failure": self.min_previous_failure,
            "min_previous_present": self.min_previous_present,
            "min_previous_success": self.min_previous_success,
            "outcome_thresholds": [
                item.to_canonical_dict()
                for item in sorted(self.outcome_thresholds, key=lambda item: item.name)
            ],
            "preregistration_id": self.preregistration_id,
            "record": "run4_exploration_coverage_gate_v1",
            "state_thresholds": [
                item.to_canonical_dict()
                for item in sorted(self.state_thresholds, key=lambda item: item.name)
            ],
        }

    @property
    def config_sha256(self) -> str:
        return _sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class MetricStats:
    observation_count: int
    finite_count: int
    missing_count: int
    nonfinite_count: int
    unique_finite_count: int
    minimum: Optional[float]
    maximum: Optional[float]
    span: Optional[float]
    lower_saturation_count: Optional[int]
    upper_saturation_count: Optional[int]
    boundary_saturation_fraction: Optional[float]


def _metric_stats(
    values: Sequence[Optional[float]],
    threshold: Optional[StateFeatureThreshold] = None,
) -> MetricStats:
    missing = sum(value is None for value in values)
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    nonfinite = sum(
        value is not None and not math.isfinite(float(value)) for value in values
    )
    minimum = min(finite) if finite else None
    maximum = max(finite) if finite else None
    span = maximum - minimum if finite else None
    lower_count: Optional[int] = None
    upper_count: Optional[int] = None
    boundary_fraction: Optional[float] = None
    if threshold is not None:
        lower_count = sum(
            value <= threshold.saturation_lower + threshold.saturation_tolerance
            for value in finite
        )
        upper_count = sum(
            value >= threshold.saturation_upper - threshold.saturation_tolerance
            for value in finite
        )
        boundary_fraction = (
            (lower_count + upper_count) / len(finite) if finite else None
        )
    return MetricStats(
        observation_count=len(values),
        finite_count=len(finite),
        missing_count=missing,
        nonfinite_count=nonfinite,
        unique_finite_count=len(set(finite)),
        minimum=minimum,
        maximum=maximum,
        span=span,
        lower_saturation_count=lower_count,
        upper_saturation_count=upper_count,
        boundary_saturation_fraction=boundary_fraction,
    )


class DecisionPhase(Enum):
    WARMUP = "WARMUP_NO_GRADIENT"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"


class SelectionKind(Enum):
    STRATIFIED_WARMUP = "STRATIFIED_WARMUP"
    STOCHASTIC_ACTOR = "STOCHASTIC_SAC_ACTOR"
    DETERMINISTIC_ACTOR = "DETERMINISTIC_ACTOR"


def require_selection_path(
    phase: DecisionPhase, selection_kind: SelectionKind
) -> None:
    """Fail closed on deterministic training or stochastic evaluation."""

    if not isinstance(phase, DecisionPhase) or not isinstance(
        selection_kind, SelectionKind
    ):
        raise ActorPathError("phase and selection_kind must be typed enums")
    expected = {
        DecisionPhase.WARMUP: SelectionKind.STRATIFIED_WARMUP,
        DecisionPhase.TRAINING: SelectionKind.STOCHASTIC_ACTOR,
        DecisionPhase.EVALUATION: SelectionKind.DETERMINISTIC_ACTOR,
    }[phase]
    if selection_kind is not expected:
        raise ActorPathError(
            f"{phase.value} requires {expected.value}, got {selection_kind.value}"
        )


@dataclass(frozen=True, slots=True)
class ActionTraceSample:
    """One identifier-free sample for actor behavior diagnostics."""

    phase: DecisionPhase
    selection_kind: SelectionKind
    mode_id: int
    q_e4: int
    observation: CoverageObservation

    def __post_init__(self) -> None:
        require_selection_path(self.phase, self.selection_kind)
        if type(self.mode_id) is not int or not 0 <= self.mode_id < EXPECTED_MODE_COUNT:
            raise CoverageRecordError("trace mode_id is outside [0,12)")
        if type(self.q_e4) is not int:
            raise CoverageRecordError("trace q_e4 must be an exact int")
        if type(self.observation) is not CoverageObservation:
            raise CoverageRecordError("trace observation has a foreign type")


@dataclass(frozen=True, slots=True)
class FeatureActionAssociation:
    feature_name: str
    q_support_fraction_pearson: Optional[float]
    mode_eta_squared: Optional[float]


@dataclass(frozen=True, slots=True)
class ActionTraceDiagnostics:
    sample_count: int
    mode_counts: Tuple[int, ...]
    normalized_mode_entropy: Optional[float]
    dominant_mode_fraction: Optional[float]
    per_mode_q_bin_counts: Tuple[Tuple[int, ...], ...]
    per_mode_distinct_q_count: Tuple[int, ...]
    per_mode_lower_boundary_count: Tuple[int, ...]
    per_mode_upper_boundary_count: Tuple[int, ...]
    lower_q_bin_fraction: Optional[float]
    upper_q_bin_fraction: Optional[float]
    outer_q_bin_fraction: Optional[float]
    feature_associations: Tuple[FeatureActionAssociation, ...]


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    ss_left = sum(value * value for value in centered_left)
    ss_right = sum(value * value for value in centered_right)
    if ss_left == 0.0 or ss_right == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / math.sqrt(
        ss_left * ss_right
    )


def _eta_squared(values: Sequence[float], groups: Sequence[int]) -> Optional[float]:
    if len(values) < 2 or len(values) != len(groups):
        return None
    overall = sum(values) / len(values)
    total = sum((value - overall) ** 2 for value in values)
    if total == 0.0:
        return None
    between = 0.0
    for group in sorted(set(groups)):
        selected = [value for value, observed_group in zip(values, groups) if observed_group == group]
        group_mean = sum(selected) / len(selected)
        between += len(selected) * (group_mean - overall) ** 2
    return min(max(between / total, 0.0), 1.0)


def summarize_action_trace(
    samples: Iterable[ActionTraceSample], schedule: StratifiedWarmupSchedule
) -> ActionTraceDiagnostics:
    """Report mode/q collapse and observational state-action associations.

    Associations are diagnostics, not causal proof that a feature is used.
    Actor-input intervention tests remain the correct later test for feature
    sensitivity.  No collapse threshold is invented here.
    """

    if type(schedule) is not StratifiedWarmupSchedule:
        raise ExplorationError("schedule must be an exact StratifiedWarmupSchedule")
    rows = tuple(samples)
    if any(type(row) is not ActionTraceSample for row in rows):
        raise ExplorationError("action trace contains a foreign sample type")
    mode_counts = [0] * EXPECTED_MODE_COUNT
    q_bins = [
        [0] * schedule.config.q_bin_count for _ in range(EXPECTED_MODE_COUNT)
    ]
    q_values: list[list[int]] = [[] for _ in range(EXPECTED_MODE_COUNT)]
    lower_counts = [0] * EXPECTED_MODE_COUNT
    upper_counts = [0] * EXPECTED_MODE_COUNT
    q_fractions: list[float] = []
    valid_rows: list[ActionTraceSample] = []
    lower_bin_count = 0
    upper_bin_count = 0
    for row in rows:
        bin_index = schedule.classify_q(row.mode_id, row.q_e4)
        lower, upper = schedule.config.mode_q_e4_bounds[row.mode_id]
        mode_counts[row.mode_id] += 1
        q_bins[row.mode_id][bin_index] += 1
        q_values[row.mode_id].append(row.q_e4)
        lower_counts[row.mode_id] += int(row.q_e4 == lower)
        upper_counts[row.mode_id] += int(row.q_e4 == upper)
        lower_bin_count += int(bin_index == 0)
        upper_bin_count += int(bin_index == schedule.config.q_bin_count - 1)
        q_fractions.append(
            0.0 if upper == lower else (row.q_e4 - lower) / (upper - lower)
        )
        valid_rows.append(row)

    total = len(rows)
    entropy = None
    dominant = None
    if total:
        probabilities = [count / total for count in mode_counts if count]
        entropy = -sum(value * math.log(value) for value in probabilities) / math.log(
            EXPECTED_MODE_COUNT
        )
        dominant = max(mode_counts) / total

    associations = []
    for feature_name in REQUIRED_STATE_FEATURES:
        feature_values = [
            float(getattr(row.observation, feature_name)) for row in valid_rows
        ]
        finite_indices = [
            index for index, value in enumerate(feature_values) if math.isfinite(value)
        ]
        associations.append(
            FeatureActionAssociation(
                feature_name=feature_name,
                q_support_fraction_pearson=_pearson(
                    [feature_values[index] for index in finite_indices],
                    [q_fractions[index] for index in finite_indices],
                ),
                mode_eta_squared=_eta_squared(
                    [feature_values[index] for index in finite_indices],
                    [valid_rows[index].mode_id for index in finite_indices],
                ),
            )
        )
    return ActionTraceDiagnostics(
        sample_count=total,
        mode_counts=tuple(mode_counts),
        normalized_mode_entropy=entropy,
        dominant_mode_fraction=dominant,
        per_mode_q_bin_counts=tuple(tuple(row) for row in q_bins),
        per_mode_distinct_q_count=tuple(len(set(row)) for row in q_values),
        per_mode_lower_boundary_count=tuple(lower_counts),
        per_mode_upper_boundary_count=tuple(upper_counts),
        lower_q_bin_fraction=(lower_bin_count / total if total else None),
        upper_q_bin_fraction=(upper_bin_count / total if total else None),
        outer_q_bin_fraction=(
            (lower_bin_count + upper_bin_count) / total if total else None
        ),
        feature_associations=tuple(associations),
    )


@dataclass(frozen=True, slots=True)
class ExplorationCoverageReport:
    schedule_id: str
    gate_config_sha256: str
    decision_count: int
    expected_decision_count: int
    final_feedback_state_recorded: bool
    current_action_diagnostics: ActionTraceDiagnostics
    previous_mode_counts: Tuple[int, ...]
    previous_q_bin_counts: Tuple[Tuple[int, ...], ...]
    previous_present_count: int
    previous_absent_count: int
    previous_success_count: int
    previous_failure_count: int
    state_stats: Tuple[Tuple[str, MetricStats], ...]
    per_mode_state_stats: Tuple[Tuple[Tuple[str, MetricStats], ...], ...]
    outcome_stats: Tuple[Tuple[str, MetricStats], ...]
    failures: Tuple[str, ...]

    @property
    def gradient_start_allowed(self) -> bool:
        return not self.failures

    def state_stat(self, name: str) -> MetricStats:
        try:
            return dict(self.state_stats)[name]
        except KeyError as exc:
            raise ExplorationError(f"unknown state statistic {name!r}") from exc

    def outcome_stat(self, name: str) -> MetricStats:
        try:
            return dict(self.outcome_stats)[name]
        except KeyError as exc:
            raise ExplorationError(f"unknown outcome statistic {name!r}") from exc


class ExplorationCoverageLedger:
    """Causal pre-gradient ledger for one finite stratified warm-up.

    ``record_decision`` accepts only the next exact scheduled action and checks
    that the state exposes the immediately previous scheduled action/outcome.
    Held transmissions must never be recorded here: they are queue evolution,
    not policy decisions and not reward-bearing transitions.
    """

    def __init__(
        self,
        schedule: StratifiedWarmupSchedule,
        gate_config: CoverageGateConfig,
    ) -> None:
        if type(schedule) is not StratifiedWarmupSchedule:
            raise ExplorationError("schedule has a foreign type")
        if type(gate_config) is not CoverageGateConfig:
            raise ExplorationError("gate_config has a foreign type")
        gate_config.__post_init__()
        self.schedule = schedule
        self.gate_config = gate_config
        self._decision_ids: set[str] = set()
        self._samples: list[ActionTraceSample] = []
        self._observations: list[CoverageObservation] = []
        self._final_feedback_recorded = False

    @property
    def decision_count(self) -> int:
        return len(self._samples)

    def _validate_previous_link(
        self, observation: CoverageObservation, expected_action: Optional[WarmupAction]
    ) -> None:
        previous = observation.previous
        if expected_action is None:
            if previous is not None:
                raise CoverageRecordError(
                    "first warm-up state must have no policy predecessor"
                )
            return
        if previous is None:
            raise CoverageRecordError(
                "feedback-gated sequence is missing the preceding action outcome"
            )
        if (previous.mode_id, previous.q_e4) != (
            expected_action.mode_id,
            expected_action.q_e4,
        ):
            raise CoverageRecordError(
                "previous action does not match the immediately preceding "
                "scheduled decision"
            )

    def record_decision(
        self,
        *,
        decision_identity: str,
        action: WarmupAction,
        observation: CoverageObservation,
    ) -> None:
        if self._final_feedback_recorded:
            raise CoverageRecordError("cannot append a decision after final feedback")
        if type(decision_identity) is not str or not decision_identity:
            raise CoverageRecordError("decision_identity must be a non-empty string")
        if decision_identity in self._decision_ids:
            raise CoverageRecordError("duplicate decision_identity")
        if type(action) is not WarmupAction:
            raise CoverageRecordError("warm-up action has a foreign type")
        if type(observation) is not CoverageObservation:
            raise CoverageRecordError("coverage observation has a foreign type")
        expected = self.schedule.action_at(self.decision_count)
        if action != expected:
            raise CoverageRecordError(
                "action differs from the exact counter-addressed schedule entry"
            )
        predecessor = (
            None
            if self.decision_count == 0
            else self.schedule.action_at(self.decision_count - 1)
        )
        self._validate_previous_link(observation, predecessor)
        self._decision_ids.add(decision_identity)
        self._observations.append(observation)
        self._samples.append(
            ActionTraceSample(
                phase=DecisionPhase.WARMUP,
                selection_kind=SelectionKind.STRATIFIED_WARMUP,
                mode_id=action.mode_id,
                q_e4=action.q_e4,
                observation=observation,
            )
        )

    def record_final_feedback_state(self, observation: CoverageObservation) -> None:
        if self.decision_count != len(self.schedule):
            raise CoverageRecordError(
                "final feedback state is allowed only after the complete schedule"
            )
        if self._final_feedback_recorded:
            raise CoverageRecordError("final feedback state is already recorded")
        if type(observation) is not CoverageObservation:
            raise CoverageRecordError("coverage observation has a foreign type")
        self._validate_previous_link(observation, self.schedule.action_at(len(self.schedule) - 1))
        self._observations.append(observation)
        self._final_feedback_recorded = True

    def _previous_inventory(self) -> tuple[
        Tuple[int, ...], Tuple[Tuple[int, ...], ...], int, int, int, int
    ]:
        mode_counts = [0] * EXPECTED_MODE_COUNT
        bin_counts = [
            [0] * self.schedule.config.q_bin_count
            for _ in range(EXPECTED_MODE_COUNT)
        ]
        present = success = failure = 0
        for observation in self._observations:
            previous = observation.previous
            if previous is None:
                continue
            present += 1
            success += int(previous.success)
            failure += int(not previous.success)
            bin_index = self.schedule.classify_q(previous.mode_id, previous.q_e4)
            mode_counts[previous.mode_id] += 1
            bin_counts[previous.mode_id][bin_index] += 1
        return (
            tuple(mode_counts),
            tuple(tuple(row) for row in bin_counts),
            present,
            len(self._observations) - present,
            success,
            failure,
        )

    def report(self) -> ExplorationCoverageReport:
        state_thresholds = {item.name: item for item in self.gate_config.state_thresholds}
        outcome_thresholds = {
            item.name: item for item in self.gate_config.outcome_thresholds
        }
        state_stats = tuple(
            (
                name,
                _metric_stats(
                    [getattr(observation, name) for observation in self._observations],
                    state_thresholds[name],
                ),
            )
            for name in REQUIRED_STATE_FEATURES
        )
        per_mode_state_stats = []
        for mode_id in range(EXPECTED_MODE_COUNT):
            observations = [
                sample.observation
                for sample in self._samples
                if sample.mode_id == mode_id
            ]
            per_mode_state_stats.append(
                tuple(
                    (
                        name,
                        _metric_stats(
                            [getattr(observation, name) for observation in observations],
                            state_thresholds[name],
                        ),
                    )
                    for name in REQUIRED_STATE_FEATURES
                )
            )
        quality_values = [
            None if item.previous is None else item.previous.quality
            for item in self._observations
        ]
        latency_values = [
            None if item.previous is None else item.previous.latency_ms
            for item in self._observations
        ]
        outcome_stats = (
            ("previous_quality", _metric_stats(quality_values)),
            ("previous_latency_ms", _metric_stats(latency_values)),
        )
        diagnostics = summarize_action_trace(self._samples, self.schedule)
        (
            previous_modes,
            previous_bins,
            previous_present,
            previous_absent,
            previous_success,
            previous_failure,
        ) = self._previous_inventory()

        failures: list[str] = []
        if self.decision_count != len(self.schedule):
            failures.append(
                f"warm-up incomplete: {self.decision_count}/{len(self.schedule)} decisions"
            )
        if not self._final_feedback_recorded:
            failures.append("final scheduled action has no successor feedback state")
        for mode_id, count in enumerate(diagnostics.mode_counts):
            if count < self.gate_config.min_current_actions_per_mode:
                failures.append(
                    f"current mode {mode_id} count {count} < "
                    f"{self.gate_config.min_current_actions_per_mode}"
                )
            for bin_index, bin_count in enumerate(
                diagnostics.per_mode_q_bin_counts[mode_id]
            ):
                if bin_count < self.gate_config.min_current_actions_per_q_bin:
                    failures.append(
                        f"current mode {mode_id} q bin {bin_index} count {bin_count} < "
                        f"{self.gate_config.min_current_actions_per_q_bin}"
                    )
        for mode_id, count in enumerate(previous_modes):
            if count < self.gate_config.min_previous_actions_per_mode:
                failures.append(
                    f"previous mode {mode_id} count {count} < "
                    f"{self.gate_config.min_previous_actions_per_mode}"
                )
            for bin_index, bin_count in enumerate(previous_bins[mode_id]):
                if bin_count < self.gate_config.min_previous_actions_per_q_bin:
                    failures.append(
                        f"previous mode {mode_id} q bin {bin_index} count {bin_count} < "
                        f"{self.gate_config.min_previous_actions_per_q_bin}"
                    )
        for label, observed, required in (
            ("previous present", previous_present, self.gate_config.min_previous_present),
            ("previous success", previous_success, self.gate_config.min_previous_success),
            ("previous failure", previous_failure, self.gate_config.min_previous_failure),
        ):
            if observed < required:
                failures.append(f"{label} count {observed} < {required}")

        for name, stats in state_stats:
            threshold = state_thresholds[name]
            if stats.nonfinite_count:
                failures.append(f"{name} has {stats.nonfinite_count} non-finite observations")
            if stats.finite_count < threshold.min_finite_count:
                failures.append(
                    f"{name} finite count {stats.finite_count} < {threshold.min_finite_count}"
                )
            if stats.unique_finite_count < threshold.min_unique_values:
                failures.append(
                    f"{name} unique count {stats.unique_finite_count} < {threshold.min_unique_values}"
                )
            if stats.span is None or stats.span < threshold.min_span:
                failures.append(f"{name} span {stats.span} < {threshold.min_span}")
            if (
                stats.boundary_saturation_fraction is None
                or stats.boundary_saturation_fraction
                > threshold.max_boundary_saturation_fraction
            ):
                failures.append(
                    f"{name} boundary saturation {stats.boundary_saturation_fraction} > "
                    f"{threshold.max_boundary_saturation_fraction}"
                )
            feature_index = REQUIRED_STATE_FEATURES.index(name)
            for mode_id in range(EXPECTED_MODE_COUNT):
                mode_stats = per_mode_state_stats[mode_id][feature_index][1]
                if mode_stats.nonfinite_count:
                    failures.append(
                        f"mode {mode_id} {name} has {mode_stats.nonfinite_count} non-finite observations"
                    )
                if mode_stats.finite_count < threshold.min_per_mode_finite_count:
                    failures.append(
                        f"mode {mode_id} {name} finite count {mode_stats.finite_count} < "
                        f"{threshold.min_per_mode_finite_count}"
                    )
                if mode_stats.unique_finite_count < threshold.min_per_mode_unique_values:
                    failures.append(
                        f"mode {mode_id} {name} unique count {mode_stats.unique_finite_count} < "
                        f"{threshold.min_per_mode_unique_values}"
                    )
                if mode_stats.span is None or mode_stats.span < threshold.min_per_mode_span:
                    failures.append(
                        f"mode {mode_id} {name} span {mode_stats.span} < "
                        f"{threshold.min_per_mode_span}"
                    )
                if (
                    mode_stats.boundary_saturation_fraction is None
                    or mode_stats.boundary_saturation_fraction
                    > threshold.max_boundary_saturation_fraction
                ):
                    failures.append(
                        f"mode {mode_id} {name} boundary saturation "
                        f"{mode_stats.boundary_saturation_fraction} > "
                        f"{threshold.max_boundary_saturation_fraction}"
                    )

        for name, stats in outcome_stats:
            threshold = outcome_thresholds[name]
            if stats.nonfinite_count:
                failures.append(f"{name} has {stats.nonfinite_count} non-finite observations")
            if stats.finite_count < threshold.min_finite_count:
                failures.append(
                    f"{name} finite count {stats.finite_count} < {threshold.min_finite_count}"
                )
            if stats.unique_finite_count < threshold.min_unique_values:
                failures.append(
                    f"{name} unique count {stats.unique_finite_count} < {threshold.min_unique_values}"
                )
            if stats.span is None or stats.span < threshold.min_span:
                failures.append(f"{name} span {stats.span} < {threshold.min_span}")

        return ExplorationCoverageReport(
            schedule_id=self.schedule.config.schedule_id,
            gate_config_sha256=self.gate_config.config_sha256,
            decision_count=self.decision_count,
            expected_decision_count=len(self.schedule),
            final_feedback_state_recorded=self._final_feedback_recorded,
            current_action_diagnostics=diagnostics,
            previous_mode_counts=previous_modes,
            previous_q_bin_counts=previous_bins,
            previous_present_count=previous_present,
            previous_absent_count=previous_absent,
            previous_success_count=previous_success,
            previous_failure_count=previous_failure,
            state_stats=state_stats,
            per_mode_state_stats=tuple(per_mode_state_stats),
            outcome_stats=outcome_stats,
            failures=tuple(failures),
        )

    def require_gradient_start(self) -> ExplorationCoverageReport:
        report = self.report()
        if not report.gradient_start_allowed:
            preview = "; ".join(report.failures[:8])
            remainder = len(report.failures) - min(len(report.failures), 8)
            if remainder:
                preview += f"; and {remainder} more"
            raise GradientStartRefused(
                f"Run-4 gradient start refused by {report.gate_config_sha256}: {preview}"
            )
        return report


def registered_modeled_support_config(
    *,
    q_bin_count: int,
    samples_per_q_bin: int,
    master_seed: int,
) -> WarmupScheduleConfig:
    """Explicitly opt into the existing hash-bound modeled support.

    The import is lazy so merely importing this module never loads even a
    support module.  This helper is optional: a future Run-4 registered support
    can be passed directly through :class:`WarmupScheduleConfig` instead.
    """

    from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
        MODELED_SMOKE_SUPPORT,
        MODELED_SMOKE_SUPPORT_SHA256,
        require_registered_modeled_smoke_support,
    )

    support = require_registered_modeled_smoke_support(MODELED_SMOKE_SUPPORT)
    return WarmupScheduleConfig(
        mode_q_e4_bounds=support.mode_q_e4_bounds,
        q_bin_count=q_bin_count,
        samples_per_q_bin=samples_per_q_bin,
        master_seed=master_seed,
        support_contract_id=MODELED_SMOKE_SUPPORT_SHA256,
    )


__all__ = [
    "ActionTraceDiagnostics",
    "ActionTraceSample",
    "ActorPathError",
    "CoverageGateConfig",
    "CoverageObservation",
    "CoverageRecordError",
    "DecisionPhase",
    "ExplorationCoverageLedger",
    "ExplorationCoverageReport",
    "ExplorationError",
    "FeatureActionAssociation",
    "GradientStartRefused",
    "MetricStats",
    "OutcomeMetricThreshold",
    "PreviousDecisionObservation",
    "QualityBin",
    "REQUIRED_OUTCOME_METRICS",
    "REQUIRED_STATE_FEATURES",
    "RUN4_FEEDBACK_DEADLINE_MS",
    "ScheduleExhausted",
    "SelectionKind",
    "StateFeatureThreshold",
    "StratifiedWarmupSchedule",
    "WarmupAction",
    "WarmupScheduleConfig",
    "partition_quality_support",
    "registered_modeled_support_config",
    "require_selection_path",
    "summarize_action_trace",
]
