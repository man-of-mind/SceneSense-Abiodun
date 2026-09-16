"""Versioned, causal observation assembly for ``tail_only_v1``.

The policy never consumes raw absolute frame IDs or wall-clock timestamps.
Exact IDs remain in the feedback ledger for attribution; the stationary policy
features are bounded lags, ages, availability flags, and rolling outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from rl_agent.rl_policy_study_v1.delayed_feedback import (
    RECENT_OUTCOME_WINDOW,
    TAIL_ONLY_V1_DEADLINE_MS,
)


OBSERVATION_SCHEMA = "splitfusion.tail_only_observation_candidate.v1"

_PREVIOUS_ACTION_FEATURES = tuple(
    f"previous_action_{action_id:02d}" for action_id in range(72)
)
_LATEST_TAIL_ACTION_FEATURES = tuple(
    f"latest_tail_action_{action_id:02d}" for action_id in range(72)
)
_LAST_TERMINAL_ACTION_FEATURES = tuple(
    f"last_terminal_action_{action_id:02d}" for action_id in range(72)
)

OBSERVATION_FEATURES_V1 = (
    "snr_normalized",
    "snr_available",
    "mcs_normalized",
    "mcs_available",
    "bsr_log_normalized",
    "bsr_available",
    "uplink_throughput_normalized",
    "uplink_throughput_available",
    "sensor_elapsed_ratio",
    "remaining_budget_ratio",
    "previous_action_available",
    *_PREVIOUS_ACTION_FEATURES,
    "previous_payload_normalized",
    "tail_feedback_available",
    "latest_tail_frame_lag_normalized",
    "latest_tail_feedback_age_ratio",
    *_LATEST_TAIL_ACTION_FEATURES,
    "latest_feedback_latency_ratio",
    "latest_segmentation_quality",
    "latest_localization_quality",
    "latest_quality_available",
    "quality_source_frozen",
    "quality_source_live_proxy",
    "latest_deadline_met",
    "terminal_event_available",
    "last_terminal_frame_lag_normalized",
    *_LAST_TERMINAL_ACTION_FEATURES,
    "last_terminal_age_ratio",
    "last_terminal_on_time_tail",
    "last_terminal_late_tail",
    "last_terminal_superseded",
    "last_terminal_reassembly_failure",
    "last_terminal_stale_before_edge",
    "last_terminal_action_failure",
    "last_terminal_reconciled_expiry",
    "pending_count_normalized",
    "deadline_missed_pending_count_normalized",
    "oldest_pending_frame_lag_normalized",
    "oldest_pending_age_ratio",
    "recent_attempt_count_normalized",
    "recent_on_time_tail_rate",
    "recent_late_tail_rate",
    "recent_superseded_rate",
    "recent_service_failure_rate",
    "recent_deadline_missed_pending_rate",
    "recent_deadline_miss_rate",
)


class ObservationContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObservationLimitsV1:
    """Normalization constants that must be frozen with a training run."""

    deadline_ms: float
    snr_min_db: float
    snr_max_db: float
    maximum_mcs: int
    maximum_bsr_bytes: int
    maximum_uplink_mbps: float
    maximum_payload_bytes: int
    maximum_frame_lag: int
    maximum_pending_count: int
    maximum_feedback_age_ms: float
    recent_outcome_window: int = RECENT_OUTCOME_WINDOW

    def validate(self) -> None:
        if not (
            math.isfinite(self.deadline_ms)
            and self.deadline_ms > 0.0
            and math.isfinite(self.snr_min_db)
            and math.isfinite(self.snr_max_db)
            and self.snr_max_db > self.snr_min_db
            and self.maximum_mcs > 0
            and self.maximum_bsr_bytes > 0
            and math.isfinite(self.maximum_uplink_mbps)
            and self.maximum_uplink_mbps > 0.0
            and self.maximum_payload_bytes > 0
            and self.maximum_frame_lag > 0
            and self.maximum_pending_count > 0
            and math.isfinite(self.maximum_feedback_age_ms)
            and self.maximum_feedback_age_ms > 0.0
            and self.recent_outcome_window == RECENT_OUTCOME_WINDOW
            and math.isclose(
                self.deadline_ms,
                TAIL_ONLY_V1_DEADLINE_MS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ObservationContractError("invalid observation normalization limits")


@dataclass(frozen=True)
class CurrentSignalsV1:
    """Signals causally available at the current action-selection cutoff."""

    sensor_elapsed_ms: float
    snr_db: float | None
    mcs: int | None
    bsr_bytes: int | None
    uplink_throughput_mbps: float | None
    previous_action_id: int | None
    previous_payload_bytes: int


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _optional_finite(value: float | int | None, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ObservationContractError(f"{name} must be finite when available")


def assemble_observation_v1(
    signals: CurrentSignalsV1,
    ledger: Mapping[str, float | int | bool],
    limits: ObservationLimitsV1,
) -> tuple[float, ...]:
    """Return features in the immutable ``OBSERVATION_FEATURES_V1`` order."""

    limits.validate()
    _optional_finite(signals.sensor_elapsed_ms, "sensor elapsed")
    _optional_finite(signals.snr_db, "SNR")
    _optional_finite(signals.mcs, "MCS")
    _optional_finite(signals.bsr_bytes, "BSR")
    _optional_finite(signals.uplink_throughput_mbps, "uplink throughput")
    if signals.sensor_elapsed_ms < 0.0:
        raise ObservationContractError("sensor elapsed cannot be negative")
    if signals.previous_action_id is not None and not (
        0 <= signals.previous_action_id < 72
    ):
        raise ObservationContractError("invalid previous action")
    if signals.previous_payload_bytes < 0:
        raise ObservationContractError("negative previous payload")

    snr_available = signals.snr_db is not None
    snr = 0.0 if signals.snr_db is None else float(signals.snr_db)
    snr_normalized = _clip(
        (snr - limits.snr_min_db) / (limits.snr_max_db - limits.snr_min_db),
        0.0,
        1.0,
    ) if snr_available else 0.0

    mcs_available = signals.mcs is not None
    mcs_normalized = (
        _clip(float(signals.mcs) / limits.maximum_mcs, 0.0, 1.0)
        if mcs_available
        else 0.0
    )
    bsr_available = signals.bsr_bytes is not None
    if bsr_available and int(signals.bsr_bytes) < 0:
        raise ObservationContractError("negative BSR")
    bsr_normalized = (
        _clip(
            math.log1p(int(signals.bsr_bytes))
            / math.log1p(limits.maximum_bsr_bytes),
            0.0,
            1.0,
        )
        if bsr_available
        else 0.0
    )
    throughput_available = signals.uplink_throughput_mbps is not None
    throughput_normalized = (
        _clip(
            float(signals.uplink_throughput_mbps) / limits.maximum_uplink_mbps,
            0.0,
            1.0,
        )
        if throughput_available
        else 0.0
    )
    sensor_ratio = _clip(signals.sensor_elapsed_ms / limits.deadline_ms, 0.0, 2.0)
    remaining_ratio = _clip(
        (limits.deadline_ms - signals.sensor_elapsed_ms) / limits.deadline_ms,
        -1.0,
        1.0,
    )
    previous_action_available = signals.previous_action_id is not None
    previous_action_one_hot = tuple(
        float(signals.previous_action_id == action_id)
        if previous_action_available
        else 0.0
        for action_id in range(72)
    )
    has_tail = bool(ledger["has_tail_feedback"])
    latest_action_id = int(ledger["latest_tail_action_id"])
    if has_tail and not (0 <= latest_action_id < 72):
        raise ObservationContractError("invalid latest-tail action")
    latest_tail_action_one_hot = tuple(
        float(has_tail and latest_action_id == action_id)
        for action_id in range(72)
    )
    has_terminal = bool(ledger["has_terminal_event"])
    last_terminal_action_id = int(ledger["last_terminal_action_id"])
    if has_terminal and not (0 <= last_terminal_action_id < 72):
        raise ObservationContractError("invalid last-terminal action")
    last_terminal_action_one_hot = tuple(
        float(has_terminal and last_terminal_action_id == action_id)
        for action_id in range(72)
    )

    values = (
        snr_normalized,
        float(snr_available),
        mcs_normalized,
        float(mcs_available),
        bsr_normalized,
        float(bsr_available),
        throughput_normalized,
        float(throughput_available),
        sensor_ratio,
        remaining_ratio,
        float(previous_action_available),
        *previous_action_one_hot,
        _clip(signals.previous_payload_bytes / limits.maximum_payload_bytes, 0.0, 1.0),
        float(has_tail),
        _clip(float(ledger["latest_tail_frame_lag"]) / limits.maximum_frame_lag, 0.0, 1.0),
        _clip(float(ledger["time_since_latest_tail_feedback_ms"]) / limits.maximum_feedback_age_ms, 0.0, 1.0),
        *latest_tail_action_one_hot,
        _clip(float(ledger["latest_feedback_latency_ratio"]), 0.0, 2.0),
        _clip(float(ledger["latest_segmentation_quality"]), 0.0, 1.0),
        _clip(float(ledger["latest_localization_quality"]), 0.0, 1.0),
        float(bool(ledger["latest_quality_available"])),
        float(bool(ledger["latest_quality_is_frozen_anchor"])),
        float(bool(ledger["latest_quality_is_live_proxy"])),
        float(bool(ledger["latest_deadline_met"])),
        float(has_terminal),
        _clip(float(ledger["last_terminal_frame_lag"]) / limits.maximum_frame_lag, 0.0, 1.0),
        *last_terminal_action_one_hot,
        _clip(float(ledger["time_since_last_terminal_event_ms"]) / limits.maximum_feedback_age_ms, 0.0, 1.0),
        float(bool(ledger["last_terminal_is_on_time_tail"])),
        float(bool(ledger["last_terminal_is_late_tail"])),
        float(bool(ledger["last_terminal_is_superseded"])),
        float(bool(ledger["last_terminal_is_reassembly_failure"])),
        float(bool(ledger["last_terminal_is_stale_before_edge"])),
        float(bool(ledger["last_terminal_is_action_failure"])),
        float(bool(ledger["last_terminal_is_reconciled_expiry"])),
        _clip(float(ledger["pending_count"]) / limits.maximum_pending_count, 0.0, 1.0),
        _clip(float(ledger["deadline_missed_pending_count"]) / limits.maximum_pending_count, 0.0, 1.0),
        _clip(float(ledger["oldest_pending_frame_lag"]) / limits.maximum_frame_lag, 0.0, 1.0),
        _clip(float(ledger["oldest_pending_age_ms"]) / limits.maximum_feedback_age_ms, 0.0, 1.0),
        _clip(float(ledger["recent_attempt_count"]) / limits.recent_outcome_window, 0.0, 1.0),
        _clip(float(ledger["recent_on_time_tail_rate"]), 0.0, 1.0),
        _clip(float(ledger["recent_late_tail_rate"]), 0.0, 1.0),
        _clip(float(ledger["recent_superseded_rate"]), 0.0, 1.0),
        _clip(float(ledger["recent_service_failure_rate"]), 0.0, 1.0),
        _clip(float(ledger["recent_deadline_missed_pending_rate"]), 0.0, 1.0),
        _clip(float(ledger["recent_deadline_miss_rate"]), 0.0, 1.0),
    )
    if len(values) != len(OBSERVATION_FEATURES_V1):
        raise AssertionError("observation schema length drift")
    if not all(math.isfinite(value) for value in values):
        raise ObservationContractError("assembled observation is non-finite")
    return values
