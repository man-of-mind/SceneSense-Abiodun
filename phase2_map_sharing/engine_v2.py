"""Uncertainty-propagating recipient map for the Phase-2 v2 contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas_v2 import (
    MapContributionV2,
    MapObjectObservationV2,
    RecipientStateV2,
    WarningEventV2,
)


def _normalize_class(value: str) -> str:
    name = str(value).strip().lower()
    if name in {"person", "walker"}:
        return "pedestrian"
    if name in {"bike", "bicycle"}:
        return "cyclist"
    return name


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(4)] for row in range(4)]


def _as_matrix(values: Sequence[float]) -> list[list[float]]:
    return [[float(values[4 * row + column]) for column in range(4)] for row in range(4)]


def _as_tuple(matrix: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def propagate_cv(
    state: Sequence[float],
    covariance: Sequence[float],
    process_noise_covariance_per_s: Sequence[float],
    dt_s: float,
) -> tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Propagate [x,y,vx,vy] and covariance with a declared CV model."""

    dt = max(0.0, float(dt_s))
    transition = [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    vector = tuple(float(item) for item in state)
    if len(vector) != 4:
        raise ValueError("state must be [x,y,vx,vy]")
    predicted = tuple(sum(transition[row][k] * vector[k] for k in range(4)) for row in range(4))
    prior = _as_matrix(covariance)
    noise = _as_matrix(process_noise_covariance_per_s)
    propagated = _matmul(_matmul(transition, prior), _transpose(transition))
    for row in range(4):
        for column in range(4):
            propagated[row][column] += noise[row][column] * dt
    return predicted, _as_tuple(propagated)


def _largest_position_sigma(covariance: Sequence[float]) -> float:
    pxx = max(0.0, float(covariance[0]))
    pxy = float(covariance[1])
    pyy = max(0.0, float(covariance[5]))
    discriminant = max(0.0, (pxx - pyy) ** 2 + 4.0 * pxy * pxy)
    largest_eigenvalue = 0.5 * (pxx + pyy + math.sqrt(discriminant))
    return math.sqrt(max(0.0, largest_eigenvalue))


@dataclass
class _TrackV2:
    canonical_track_id: str
    class_name: str
    state: Tuple[float, ...]
    covariance: Tuple[float, ...]
    confidence: float
    latest_measurement_at_s: float
    latest_capture_at_s: float
    latest_publish_at_s: float
    motion_model_id: str
    process_noise_model_id: str
    process_noise_covariance_per_s: Tuple[float, ...]
    validity_horizon_s: float
    clock_id: str
    source_track_ids: Dict[str, str] = field(default_factory=dict)
    source_capture_at_s: Dict[str, float] = field(default_factory=dict)

    def predicted(self, timestamp_s: float) -> tuple[Tuple[float, ...], Tuple[float, ...]]:
        return propagate_cv(
            self.state,
            self.covariance,
            self.process_noise_covariance_per_s,
            max(0.0, float(timestamp_s) - self.latest_measurement_at_s),
        )


class RecipientMapEngineV2:
    """Deterministic baseline; calibration is intentionally outside this class."""

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
        confidence_floor: float = 0.15,
        safety_radius_m_by_class: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.recipient_ue_id = str(recipient_ue_id)
        self.association_gate_m = float(association_gate_m)
        self.association_sigma_multiplier = float(association_sigma_multiplier)
        self.warning_sigma_multiplier = float(warning_sigma_multiplier)
        self.track_ttl_s = float(track_ttl_s)
        self.max_transport_age_s = float(max_transport_age_s)
        self.warning_horizon_s = float(warning_horizon_s)
        self.confidence_floor = float(confidence_floor)
        self.safety_radius = {
            "pedestrian": 2.5,
            "cyclist": 3.0,
            "vehicle": 3.0,
            **dict(safety_radius_m_by_class or {}),
        }
        numeric_config = (
            self.association_gate_m,
            self.association_sigma_multiplier,
            self.warning_sigma_multiplier,
            self.track_ttl_s,
            self.max_transport_age_s,
            self.warning_horizon_s,
            self.confidence_floor,
            *self.safety_radius.values(),
        )
        if not all(math.isfinite(value) for value in numeric_config):
            raise ValueError("v2 engine configuration values must be finite")
        if min(
            self.association_gate_m,
            self.track_ttl_s,
            self.max_transport_age_s,
            self.warning_horizon_s,
            *self.safety_radius.values(),
        ) <= 0.0:
            raise ValueError("v2 gates/horizons must be positive")
        if min(self.association_sigma_multiplier, self.warning_sigma_multiplier) < 0.0:
            raise ValueError("uncertainty multipliers must be nonnegative")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        self.tracks: Dict[str, _TrackV2] = {}
        self.clock_id: Optional[str] = None
        self.last_sequence_by_source: Dict[str, int] = {}
        self.next_track_number = 1
        self.counters = {
            "accepted_contributions": 0,
            "wrong_recipient_rejections": 0,
            "sequence_rejections": 0,
            "transport_stale_rejections": 0,
            "clock_mismatch_rejections": 0,
            "unsupported_motion_model_rejections": 0,
            "older_track_updates_ignored": 0,
            "expired_tracks": 0,
        }

    def _expire(self, now_s: float) -> None:
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if float(now_s) - track.latest_measurement_at_s
            > min(self.track_ttl_s, track.validity_horizon_s) + 1e-12
        ]
        for track_id in stale:
            del self.tracks[track_id]
            self.counters["expired_tracks"] += 1

    def _associate(
        self,
        obj: MapObjectObservationV2,
        reserved: set[str],
        clock_id: str,
    ) -> Optional[_TrackV2]:
        candidates = []
        object_sigma = _largest_position_sigma(obj.state_covariance)
        for track in self.tracks.values():
            if (
                track.canonical_track_id in reserved
                or track.class_name != _normalize_class(obj.class_name)
                or track.clock_id != clock_id
            ):
                continue
            state, covariance = track.predicted(obj.measured_at_s)
            distance = math.hypot(obj.x_m - state[0], obj.y_m - state[1])
            combined_sigma = math.hypot(_largest_position_sigma(covariance), object_sigma)
            gate = self.association_gate_m + self.association_sigma_multiplier * combined_sigma
            if distance <= gate:
                candidates.append((distance, track.canonical_track_id, track))
        return min(candidates, default=(0.0, "", None))[2]

    def _new_track(
        self, obj: MapObjectObservationV2, contribution: MapContributionV2
    ) -> _TrackV2:
        track_id = f"map_track_v2_{self.next_track_number:05d}"
        self.next_track_number += 1
        return _TrackV2(
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
        self.last_sequence_by_source[contribution.source_ue_id] = contribution.sequence_number
        self._expire(received)
        reserved: set[str] = set()
        for obj in contribution.objects:
            track = self._associate(obj, reserved, contribution.clock_id)
            if track is None:
                track = self._new_track(obj, contribution)
                self.tracks[track.canonical_track_id] = track
            elif obj.measured_at_s + 1e-12 < track.latest_measurement_at_s:
                self.counters["older_track_updates_ignored"] += 1
                continue
            track.state = (obj.x_m, obj.y_m, obj.vx_mps, obj.vy_mps)
            track.covariance = obj.state_covariance
            track.confidence = obj.confidence
            track.latest_measurement_at_s = obj.measured_at_s
            track.latest_capture_at_s = contribution.captured_at_s
            track.latest_publish_at_s = contribution.published_at_s
            track.motion_model_id = obj.motion_model_id
            track.process_noise_model_id = obj.process_noise_model_id
            track.process_noise_covariance_per_s = obj.process_noise_covariance_per_s
            track.validity_horizon_s = obj.validity_horizon_s
            track.source_track_ids[contribution.source_ue_id] = obj.source_track_id
            track.source_capture_at_s[contribution.source_ue_id] = contribution.captured_at_s
            reserved.add(track.canonical_track_id)
        self.counters["accepted_contributions"] += 1
        return "accepted"

    def warnings(self, recipient: RecipientStateV2) -> List[WarningEventV2]:
        recipient.validate()
        if recipient.recipient_ue_id != self.recipient_ue_id:
            raise ValueError("recipient state belongs to a different recipient")
        if self.clock_id is not None and recipient.clock_id != self.clock_id:
            raise ValueError("recipient state and map tracks use different clock domains")
        if recipient.motion_model_id != "CV":
            raise ValueError("recipient map baseline supports only CV recipient motion")
        now = recipient.available_at_s
        recipient_dt = max(0.0, now - recipient.observed_at_s)
        recipient_state, recipient_covariance = propagate_cv(
            (recipient.x_m, recipient.y_m, recipient.vx_mps, recipient.vy_mps),
            recipient.state_covariance,
            recipient.process_noise_covariance_per_s,
            recipient_dt,
        )
        recipient_x = recipient_state[0]
        recipient_y = recipient_state[1]
        self._expire(now)
        warnings = []
        for track in self.tracks.values():
            if track.confidence < self.confidence_floor:
                continue
            current_state, current_covariance = track.predicted(now)
            rx = current_state[0] - recipient_x
            ry = current_state[1] - recipient_y
            rvx = current_state[2] - recipient.vx_mps
            rvy = current_state[3] - recipient.vy_mps
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq <= 1e-12:
                tca = 0.0
            else:
                tca = max(0.0, min(self.warning_horizon_s, -(rx * rvx + ry * rvy) / speed_sq))
            closest_x = rx + rvx * tca
            closest_y = ry + rvy * tca
            closest = math.hypot(closest_x, closest_y)
            _, closest_covariance = propagate_cv(
                current_state,
                current_covariance,
                track.process_noise_covariance_per_s,
                tca,
            )
            _, recipient_closest_covariance = propagate_cv(
                recipient_state,
                recipient_covariance,
                recipient.process_noise_covariance_per_s,
                tca,
            )
            # Baseline assumes recipient/object state errors are independent.
            # A correlated estimator must transmit cross-covariance and replace
            # this sum before any stronger safety claim is made.
            relative_covariance = tuple(
                object_value + recipient_value
                for object_value, recipient_value in zip(
                    closest_covariance, recipient_closest_covariance
                )
            )
            sigma = _largest_position_sigma(relative_covariance)
            expanded_closest = max(0.0, closest - self.warning_sigma_multiplier * sigma)
            safety_radius = float(self.safety_radius.get(track.class_name, 3.0))
            if expanded_closest > safety_radius:
                continue
            active_sources = tuple(
                sorted(
                    source
                    for source, captured_at_s in track.source_capture_at_s.items()
                    if now - captured_at_s <= self.track_ttl_s + 1e-12
                )
            )
            active_track_ids = tuple(track.source_track_ids[source] for source in active_sources)
            evidence_scope = (
                "multi_source"
                if len(active_sources) > 1
                else "ego_only"
                if active_sources == (self.recipient_ue_id,)
                else "helper_only"
            )
            warnings.append(
                WarningEventV2(
                    recipient_ue_id=self.recipient_ue_id,
                    canonical_track_id=track.canonical_track_id,
                    class_name=track.class_name,
                    warning_at_s=now,
                    time_to_closest_approach_s=tca,
                    closest_approach_m=closest,
                    uncertainty_expanded_closest_approach_m=expanded_closest,
                    position_sigma_at_closest_approach_m=sigma,
                    map_aoi_s=max(0.0, now - track.latest_capture_at_s),
                    evidence_sources=active_sources,
                    evidence_track_ids=active_track_ids,
                    evidence_scope=evidence_scope,
                    latest_capture_at_s=track.latest_capture_at_s,
                    latest_publish_at_s=track.latest_publish_at_s,
                    motion_model_id=track.motion_model_id,
                    process_noise_model_id=track.process_noise_model_id,
                )
            )
        return sorted(warnings, key=lambda item: (item.time_to_closest_approach_s, item.canonical_track_id))

    def snapshot(self, timestamp_s: float, clock_id: str) -> dict:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("snapshot timestamp must be finite")
        if self.clock_id is not None and str(clock_id) != self.clock_id:
            raise ValueError("snapshot and map tracks use different clock domains")
        self._expire(timestamp)
        rows = []
        for track in sorted(self.tracks.values(), key=lambda item: item.canonical_track_id):
            state, covariance = track.predicted(timestamp)
            rows.append(
                {
                    "canonical_track_id": track.canonical_track_id,
                    "class_name": track.class_name,
                    "x_m": state[0],
                    "y_m": state[1],
                    "vx_mps": state[2],
                    "vy_mps": state[3],
                    "position_sigma_m": _largest_position_sigma(covariance),
                    "map_aoi_s": max(0.0, timestamp - track.latest_capture_at_s),
                    "evidence_sources": sorted(track.source_track_ids),
                    "motion_model_id": track.motion_model_id,
                    "process_noise_model_id": track.process_noise_model_id,
                }
            )
        return {
            "recipient_ue_id": self.recipient_ue_id,
            "timestamp_s": timestamp,
            "clock_id": str(clock_id),
            "tracks": rows,
        }
