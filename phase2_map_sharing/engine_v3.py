"""Order-invariant equal-time fusion for the Phase-2 recipient map.

``RecipientMapEngineV2`` is intentionally left unchanged because it defines the
provenance of the completed calibration replay.  V3 keeps the v2 contribution
wire schema, association gate, propagation model, and warning geometry, but it
removes v2's equal-measurement-time latest-writer dependence after observations
have associated to the same canonical track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .engine_v2 import (
    RecipientMapEngineV2,
    _TrackV2,
    _largest_position_sigma,
    _normalize_class,
    propagate_cv,
)
from .schemas_v2 import MapContributionV2, MapObjectObservationV2, RecipientStateV2


FUSION_RULE_ID_V3 = "quality_weighted_moment_equal_time_v1"
_EQUAL_TIME_TOLERANCE_S = 1e-9
_QUALITY_EPSILON = 1e-12


@dataclass(frozen=True)
class WarningConfirmationContextV3:
    """Causal metadata exposed to an optional warning-confirmation policy.

    The v2 wire schema does not carry source-tracker hit streak, confirmation
    age, or miss streak.  Consequently this context deliberately exposes only
    metadata already available at the recipient; it does not pretend that a
    temporal-persistence rule can be calibrated from unavailable fields.
    """

    canonical_track_id: str
    class_name: str
    warning_at_s: float
    fused_measurement_at_s: float
    fused_confidence: float
    active_fusion_sources: Tuple[str, ...]
    active_source_track_ids: Tuple[Tuple[str, str], ...]
    active_source_measurement_at_s: Tuple[Tuple[str, float], ...]
    active_source_confidences: Tuple[Tuple[str, float], ...]
    fusion_rule_id: str = FUSION_RULE_ID_V3


WarningConfirmationPolicyV3 = Callable[[WarningConfirmationContextV3], bool]


@dataclass(frozen=True)
class _SourceEstimateV3:
    source_ue_id: str
    source_track_id: str
    state: Tuple[float, ...]
    covariance: Tuple[float, ...]
    confidence: float
    measured_at_s: float
    captured_at_s: float
    published_at_s: float
    motion_model_id: str
    process_noise_model_id: str
    process_noise_covariance_per_s: Tuple[float, ...]
    validity_horizon_s: float


@dataclass
class _TrackV3(_TrackV2):
    source_estimates: Dict[str, _SourceEstimateV3] = field(default_factory=dict)
    fusion_source_ids: Tuple[str, ...] = ()
    fusion_rule_id: str = FUSION_RULE_ID_V3


def _quality_weight(estimate: _SourceEstimateV3) -> float:
    """Return a deterministic scalar quality without assuming independence."""

    position_variance = max(
        _QUALITY_EPSILON,
        max(0.0, float(estimate.covariance[0]))
        + max(0.0, float(estimate.covariance[5])),
    )
    return max(_QUALITY_EPSILON, float(estimate.confidence)) / position_variance


def _normalized_weights(
    estimates: Sequence[_SourceEstimateV3],
) -> Tuple[float, ...]:
    raw = tuple(_quality_weight(estimate) for estimate in estimates)
    total = math.fsum(raw)
    if total <= _QUALITY_EPSILON:
        # Zero-confidence observations remain map-admissible under the frozen
        # v2 contract.  Covariance-only weights keep that behavior deterministic.
        raw = tuple(
            1.0
            / max(
                _QUALITY_EPSILON,
                max(0.0, float(estimate.covariance[0]))
                + max(0.0, float(estimate.covariance[5])),
            )
            for estimate in estimates
        )
        total = math.fsum(raw)
    return tuple(value / total for value in raw)


def _weighted_vector(
    weights: Sequence[float], vectors: Sequence[Sequence[float]]
) -> Tuple[float, ...]:
    return tuple(
        math.fsum(
            weight * float(vector[index])
            for weight, vector in zip(weights, vectors)
        )
        for index in range(4)
    )


def _moment_covariance(
    weights: Sequence[float],
    states: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[float]],
    mean: Sequence[float],
) -> Tuple[float, ...]:
    """Fuse as a Gaussian-mixture moment, retaining inter-source disagreement.

    This is deliberately more conservative than adding information matrices:
    the current contract provides no cross-source correlation, so v3 must not
    claim an uncertainty reduction that assumes independent sensor errors.
    """

    values = []
    for row in range(4):
        for column in range(4):
            values.append(
                math.fsum(
                    weight
                    * (
                        float(covariance[4 * row + column])
                        + (float(state[row]) - float(mean[row]))
                        * (float(state[column]) - float(mean[column]))
                    )
                    for weight, state, covariance in zip(
                        weights, states, covariances
                    )
                )
            )
    # Eliminate insignificant platform-dependent asymmetry before propagation.
    for row in range(4):
        for column in range(row):
            value = 0.5 * (values[4 * row + column] + values[4 * column + row])
            values[4 * row + column] = value
            values[4 * column + row] = value
    return tuple(values)


class RecipientMapEngineV3(RecipientMapEngineV2):
    """V2-compatible engine with deterministic equal-time state fusion.

    Order invariance is guaranteed for source observations that have associated
    to the same canonical track and share its newest measurement timestamp.
    Streaming greedy association itself is intentionally unchanged; ambiguous
    multi-object association requires a later timestamp-batched/global matcher.
    """

    def __init__(
        self,
        recipient_ue_id: str,
        *,
        association_gate_m: float = 3.0,
        association_sigma_multiplier: float = 2.0,
        warning_sigma_multiplier: float = 2.0,
        track_ttl_s: float = 1.0,
        max_transport_age_s: float = 1.0,
        warning_horizon_s: float = 5.0,
        warning_emission_confidence_floor: float = 0.15,
        safety_radius_m_by_class: Optional[Mapping[str, float]] = None,
        warning_confirmation_policy: Optional[WarningConfirmationPolicyV3] = None,
    ) -> None:
        super().__init__(
            recipient_ue_id,
            association_gate_m=association_gate_m,
            association_sigma_multiplier=association_sigma_multiplier,
            warning_sigma_multiplier=warning_sigma_multiplier,
            track_ttl_s=track_ttl_s,
            max_transport_age_s=max_transport_age_s,
            warning_horizon_s=warning_horizon_s,
            warning_emission_confidence_floor=warning_emission_confidence_floor,
            safety_radius_m_by_class=safety_radius_m_by_class,
        )
        if warning_confirmation_policy is not None and not callable(
            warning_confirmation_policy
        ):
            raise TypeError("warning_confirmation_policy must be callable")
        self.warning_confirmation_policy = warning_confirmation_policy
        self.counters.update(
            {
                "equal_time_multi_source_fusions": 0,
                "older_source_estimates_retained": 0,
                "confirmation_policy_rejections": 0,
            }
        )

    def _new_track(
        self, obj: MapObjectObservationV2, contribution: MapContributionV2
    ) -> _TrackV3:
        track_id = f"map_track_v3_{self.next_track_number:05d}"
        self.next_track_number += 1
        return _TrackV3(
            canonical_track_id=track_id,
            class_name=_normalize_class(obj.class_name),
            state=(obj.x_m, obj.y_m, obj.vx_mps, obj.vy_mps),
            covariance=obj.state_covariance,
            confidence=obj.confidence,
            latest_measurement_at_s=obj.measured_at_s,
            latest_capture_at_s=contribution.captured_at_s,
            latest_publish_at_s=contribution.published_at_s,
            motion_model_id=obj.motion_model_id,
            process_noise_model_id=obj.process_noise_model_id,
            process_noise_covariance_per_s=obj.process_noise_covariance_per_s,
            validity_horizon_s=obj.validity_horizon_s,
            clock_id=contribution.clock_id,
        )

    @staticmethod
    def _estimate(
        obj: MapObjectObservationV2, contribution: MapContributionV2
    ) -> _SourceEstimateV3:
        return _SourceEstimateV3(
            source_ue_id=contribution.source_ue_id,
            source_track_id=obj.source_track_id,
            state=(obj.x_m, obj.y_m, obj.vx_mps, obj.vy_mps),
            covariance=obj.state_covariance,
            confidence=obj.confidence,
            measured_at_s=obj.measured_at_s,
            captured_at_s=contribution.captured_at_s,
            published_at_s=contribution.published_at_s,
            motion_model_id=obj.motion_model_id,
            process_noise_model_id=obj.process_noise_model_id,
            process_noise_covariance_per_s=obj.process_noise_covariance_per_s,
            validity_horizon_s=obj.validity_horizon_s,
        )

    def _recompute_fused_state(self, track: _TrackV3) -> None:
        newest_measurement = max(
            estimate.measured_at_s for estimate in track.source_estimates.values()
        )
        source_ids = tuple(
            sorted(
                source
                for source, estimate in track.source_estimates.items()
                if abs(estimate.measured_at_s - newest_measurement)
                <= _EQUAL_TIME_TOLERANCE_S
            )
        )
        estimates = tuple(track.source_estimates[source] for source in source_ids)
        propagated = tuple(
            propagate_cv(
                estimate.state,
                estimate.covariance,
                estimate.process_noise_covariance_per_s,
                newest_measurement - estimate.measured_at_s,
            )
            for estimate in estimates
        )
        states = tuple(item[0] for item in propagated)
        covariances = tuple(item[1] for item in propagated)
        weights = _normalized_weights(estimates)
        fused_state = _weighted_vector(weights, states)
        fused_covariance = _moment_covariance(
            weights, states, covariances, fused_state
        )
        fused_process_noise = tuple(
            math.fsum(
                weight * estimate.process_noise_covariance_per_s[index]
                for weight, estimate in zip(weights, estimates)
            )
            for index in range(16)
        )
        track.state = fused_state
        track.covariance = fused_covariance
        track.confidence = math.fsum(
            weight * estimate.confidence
            for weight, estimate in zip(weights, estimates)
        )
        track.latest_measurement_at_s = newest_measurement
        track.latest_capture_at_s = max(
            estimate.captured_at_s for estimate in estimates
        )
        track.latest_publish_at_s = max(
            estimate.published_at_s for estimate in estimates
        )
        track.motion_model_id = "CV"
        process_noise_ids = {
            estimate.process_noise_model_id for estimate in estimates
        }
        track.process_noise_model_id = (
            next(iter(process_noise_ids))
            if len(process_noise_ids) == 1
            else "quality_weighted_moment_q_v1"
        )
        track.process_noise_covariance_per_s = fused_process_noise
        # The complete fused estimate is valid only for the common declared
        # horizon.  This avoids silently retaining an expired component.
        track.validity_horizon_s = min(
            estimate.validity_horizon_s for estimate in estimates
        )
        track.fusion_source_ids = source_ids
        if len(source_ids) > 1:
            self.counters["equal_time_multi_source_fusions"] += 1

    def install(
        self,
        contribution: MapContributionV2,
        received_at_s: float,
        received_clock_id: str,
    ) -> str:
        contribution.validate()
        received = float(received_at_s)
        if not math.isfinite(received):
            raise ValueError("received_at_s must be finite")
        if str(received_clock_id) != contribution.clock_id:
            self.counters["clock_mismatch_rejections"] += 1
            return "rejected_clock_mismatch"
        if received + 1e-12 < contribution.published_at_s:
            raise ValueError("received_at_s cannot precede published_at_s")
        if contribution.recipient_ue_id != self.recipient_ue_id:
            self.counters["wrong_recipient_rejections"] += 1
            return "rejected_wrong_recipient"
        if any(obj.motion_model_id != "CV" for obj in contribution.objects):
            self.counters["unsupported_motion_model_rejections"] += 1
            return "rejected_unsupported_motion_model"
        if self.clock_id is not None and contribution.clock_id != self.clock_id:
            self.counters["clock_mismatch_rejections"] += 1
            return "rejected_clock_mismatch"
        previous = self.last_sequence_by_source.get(contribution.source_ue_id, -1)
        if contribution.sequence_number <= previous:
            self.counters["sequence_rejections"] += 1
            return "rejected_sequence"
        if received - contribution.captured_at_s > self.max_transport_age_s + 1e-12:
            self.counters["transport_stale_rejections"] += 1
            return "rejected_transport_stale"
        if self.clock_id is None:
            self.clock_id = contribution.clock_id
        self.last_sequence_by_source[contribution.source_ue_id] = (
            contribution.sequence_number
        )
        self._expire(received)
        reserved: set[str] = set()
        for obj in contribution.objects:
            track = self._associate(obj, reserved, contribution.clock_id)
            if track is None:
                track = self._new_track(obj, contribution)
                self.tracks[track.canonical_track_id] = track
            if not isinstance(track, _TrackV3):
                raise TypeError("v3 engine contains a non-v3 track")
            prior = track.source_estimates.get(contribution.source_ue_id)
            if prior is not None and (
                obj.measured_at_s + _EQUAL_TIME_TOLERANCE_S
                < prior.measured_at_s
            ):
                self.counters["older_track_updates_ignored"] += 1
                continue
            if (
                track.source_estimates
                and obj.measured_at_s + _EQUAL_TIME_TOLERANCE_S
                < track.latest_measurement_at_s
            ):
                self.counters["older_source_estimates_retained"] += 1
            estimate = self._estimate(obj, contribution)
            track.source_estimates[contribution.source_ue_id] = estimate
            track.source_track_ids[contribution.source_ue_id] = obj.source_track_id
            track.source_capture_at_s[contribution.source_ue_id] = (
                contribution.captured_at_s
            )
            track.source_measurement_at_s[contribution.source_ue_id] = (
                obj.measured_at_s
            )
            self._recompute_fused_state(track)
            reserved.add(track.canonical_track_id)
        self.counters["accepted_contributions"] += 1
        return "accepted"

    def _active_sources(
        self, track: _TrackV2, timestamp_s: float
    ) -> Tuple[str, ...]:
        if not isinstance(track, _TrackV3):
            return super()._active_sources(track, timestamp_s)
        return tuple(
            source
            for source in track.fusion_source_ids
            if float(timestamp_s) - track.source_measurement_at_s[source]
            <= self.track_ttl_s + 1e-12
        )

    def _confirmation_context(
        self, track: _TrackV3, warning_at_s: float
    ) -> WarningConfirmationContextV3:
        active_sources = self._active_sources(track, warning_at_s)
        return WarningConfirmationContextV3(
            canonical_track_id=track.canonical_track_id,
            class_name=track.class_name,
            warning_at_s=float(warning_at_s),
            fused_measurement_at_s=track.latest_measurement_at_s,
            fused_confidence=track.confidence,
            active_fusion_sources=active_sources,
            active_source_track_ids=tuple(
                (source, track.source_track_ids[source]) for source in active_sources
            ),
            active_source_measurement_at_s=tuple(
                (source, track.source_measurement_at_s[source])
                for source in active_sources
            ),
            active_source_confidences=tuple(
                (source, track.source_estimates[source].confidence)
                for source in active_sources
            ),
        )

    def warnings(self, recipient: RecipientStateV2):
        candidates = super().warnings(recipient)
        if self.warning_confirmation_policy is None:
            return candidates
        accepted = []
        for warning in candidates:
            track = self.tracks.get(warning.canonical_track_id)
            if not isinstance(track, _TrackV3):
                raise TypeError("v3 warning references a non-v3 track")
            decision = self.warning_confirmation_policy(
                self._confirmation_context(track, warning.warning_at_s)
            )
            if not isinstance(decision, bool):
                raise TypeError("warning_confirmation_policy must return bool")
            if decision:
                accepted.append(warning)
            else:
                self.counters["confirmation_policy_rejections"] += 1
        return accepted

    def snapshot(self, timestamp_s: float, clock_id: str) -> dict:
        result = super().snapshot(timestamp_s, clock_id)
        result["engine_version"] = "v3"
        result["fusion_rule_id"] = FUSION_RULE_ID_V3
        for row in result["tracks"]:
            track = self.tracks[row["canonical_track_id"]]
            if not isinstance(track, _TrackV3):
                raise TypeError("v3 snapshot contains a non-v3 track")
            active_sources = self._active_sources(track, float(timestamp_s))
            row.update(
                {
                    "confidence": track.confidence,
                    "fusion_rule_id": track.fusion_rule_id,
                    "active_fusion_sources": list(active_sources),
                    "active_fusion_source_count": len(active_sources),
                    "temporal_confirmation_metadata_status": (
                        "source_tracker_confirmation_fields_absent_from_v2_wire_schema"
                    ),
                }
            )
        return result
