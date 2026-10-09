"""Deterministic recipient map, association, freshness, and warning baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .schemas import EgoState, MapContribution, MapObjectObservation, WarningEvent


def _normalize_class(value: str) -> str:
    name = str(value).strip().lower()
    if name in {"person", "walker"}:
        return "pedestrian"
    if name in {"bike", "bicycle"}:
        return "cyclist"
    return name


@dataclass
class _Track:
    canonical_track_id: str
    class_name: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    confidence: float
    latest_capture_at_s: float
    latest_publish_at_s: float
    source_track_ids: Dict[str, str] = field(default_factory=dict)
    source_capture_at_s: Dict[str, float] = field(default_factory=dict)

    def predicted_xy(self, timestamp_s: float) -> tuple[float, float]:
        dt = max(0.0, float(timestamp_s) - self.latest_capture_at_s)
        return self.x_m + self.vx_mps * dt, self.y_m + self.vy_mps * dt


class RecipientMapEngine:
    def __init__(
        self,
        recipient_ue_id: str,
        *,
        association_gate_m: float = 3.0,
        track_ttl_s: float = 1.0,
        max_transport_age_s: float = 1.0,
        warning_horizon_s: float = 5.0,
        confidence_floor: float = 0.15,
        safety_radius_m_by_class: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.recipient_ue_id = str(recipient_ue_id)
        self.association_gate_m = float(association_gate_m)
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
        if min(
            self.association_gate_m,
            self.track_ttl_s,
            self.max_transport_age_s,
            self.warning_horizon_s,
        ) <= 0.0:
            raise ValueError("Phase-2 gates/horizons must be positive")
        self.tracks: Dict[str, _Track] = {}
        self.last_sequence_by_source: Dict[str, int] = {}
        self.next_track_number = 1
        self.counters = {
            "accepted_contributions": 0,
            "wrong_recipient_rejections": 0,
            "sequence_rejections": 0,
            "transport_stale_rejections": 0,
            "older_track_updates_ignored": 0,
            "expired_tracks": 0,
        }

    def _expire(self, now_s: float) -> None:
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if now_s - track.latest_capture_at_s > self.track_ttl_s + 1e-12
        ]
        for track_id in stale:
            del self.tracks[track_id]
            self.counters["expired_tracks"] += 1

    def _associate(
        self,
        obj: MapObjectObservation,
        capture_s: float,
        reserved: set[str],
    ) -> Optional[_Track]:
        candidates = []
        for track in self.tracks.values():
            if track.canonical_track_id in reserved or track.class_name != _normalize_class(obj.class_name):
                continue
            px, py = track.predicted_xy(capture_s)
            distance = math.hypot(obj.x_m - px, obj.y_m - py)
            if distance <= self.association_gate_m:
                candidates.append((distance, track.canonical_track_id, track))
        return min(candidates, default=(0.0, "", None))[2]

    def _new_track(self, obj: MapObjectObservation, contribution: MapContribution) -> _Track:
        track_id = f"map_track_{self.next_track_number:05d}"
        self.next_track_number += 1
        return _Track(
            canonical_track_id=track_id,
            class_name=_normalize_class(obj.class_name),
            x_m=obj.x_m,
            y_m=obj.y_m,
            vx_mps=obj.vx_mps,
            vy_mps=obj.vy_mps,
            confidence=obj.confidence,
            latest_capture_at_s=contribution.captured_at_s,
            latest_publish_at_s=contribution.published_at_s,
        )

    def install(self, contribution: MapContribution, received_at_s: float) -> str:
        contribution.validate()
        received = float(received_at_s)
        if received + 1e-12 < contribution.published_at_s:
            raise ValueError("received_at_s cannot precede published_at_s")
        if contribution.recipient_ue_id != self.recipient_ue_id:
            self.counters["wrong_recipient_rejections"] += 1
            return "rejected_wrong_recipient"
        previous = self.last_sequence_by_source.get(contribution.source_ue_id, -1)
        if contribution.sequence_number <= previous:
            self.counters["sequence_rejections"] += 1
            return "rejected_sequence"
        if received - contribution.captured_at_s > self.max_transport_age_s + 1e-12:
            self.counters["transport_stale_rejections"] += 1
            return "rejected_transport_stale"
        self.last_sequence_by_source[contribution.source_ue_id] = contribution.sequence_number
        self._expire(received)
        reserved: set[str] = set()
        for obj in contribution.objects:
            track = self._associate(obj, contribution.captured_at_s, reserved)
            if track is None:
                track = self._new_track(obj, contribution)
                self.tracks[track.canonical_track_id] = track
            elif contribution.captured_at_s + 1e-12 < track.latest_capture_at_s:
                self.counters["older_track_updates_ignored"] += 1
                continue
            track.x_m = obj.x_m
            track.y_m = obj.y_m
            track.vx_mps = obj.vx_mps
            track.vy_mps = obj.vy_mps
            track.confidence = obj.confidence
            track.latest_capture_at_s = contribution.captured_at_s
            track.latest_publish_at_s = contribution.published_at_s
            track.source_track_ids[contribution.source_ue_id] = obj.source_track_id
            track.source_capture_at_s[contribution.source_ue_id] = contribution.captured_at_s
            reserved.add(track.canonical_track_id)
        self.counters["accepted_contributions"] += 1
        return "accepted"

    def warnings(self, ego: EgoState) -> List[WarningEvent]:
        if ego.recipient_ue_id != self.recipient_ue_id:
            raise ValueError("ego state belongs to a different recipient")
        self._expire(ego.timestamp_s)
        warnings = []
        for track in self.tracks.values():
            if track.confidence < self.confidence_floor:
                continue
            ox, oy = track.predicted_xy(ego.timestamp_s)
            rx, ry = ox - ego.x_m, oy - ego.y_m
            rvx, rvy = track.vx_mps - ego.vx_mps, track.vy_mps - ego.vy_mps
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq <= 1e-12:
                t_closest = 0.0
            else:
                t_closest = max(0.0, min(self.warning_horizon_s, -(rx * rvx + ry * rvy) / speed_sq))
            closest = math.hypot(rx + rvx * t_closest, ry + rvy * t_closest)
            threshold = float(self.safety_radius.get(track.class_name, 2.5))
            if t_closest > self.warning_horizon_s or closest > threshold:
                continue
            # Provenance is actionable only while the corresponding evidence is
            # live.  A once-seen helper must not receive permanent credit after
            # its observation has aged out while a newer ego update keeps the
            # canonical track alive.
            sources = tuple(
                sorted(
                    source
                    for source, capture_s in track.source_capture_at_s.items()
                    if ego.timestamp_s - capture_s <= self.track_ttl_s + 1e-12
                )
            )
            if not sources:
                continue
            if sources == (self.recipient_ue_id,):
                scope = "ego_only"
            elif self.recipient_ue_id not in sources:
                scope = "helper_only"
            else:
                scope = "multi_source"
            warnings.append(
                WarningEvent(
                    recipient_ue_id=self.recipient_ue_id,
                    canonical_track_id=track.canonical_track_id,
                    class_name=track.class_name,
                    warning_at_s=ego.timestamp_s,
                    time_to_closest_approach_s=t_closest,
                    closest_approach_m=closest,
                    object_x_m=ox,
                    object_y_m=oy,
                    map_aoi_s=max(0.0, ego.timestamp_s - track.latest_capture_at_s),
                    evidence_sources=sources,
                    evidence_track_ids=tuple(track.source_track_ids[source] for source in sources),
                    evidence_scope=scope,
                    latest_capture_at_s=track.latest_capture_at_s,
                    latest_publish_at_s=track.latest_publish_at_s,
                )
            )
        return sorted(warnings, key=lambda item: (item.time_to_closest_approach_s, item.canonical_track_id))

    def snapshot(self, now_s: float) -> dict:
        self._expire(now_s)
        return {
            "recipient_ue_id": self.recipient_ue_id,
            "timestamp_s": float(now_s),
            "tracks": [
                {
                    "canonical_track_id": track.canonical_track_id,
                    "class_name": track.class_name,
                    "x_m": track.x_m,
                    "y_m": track.y_m,
                    "vx_mps": track.vx_mps,
                    "vy_mps": track.vy_mps,
                    "confidence": track.confidence,
                    "latest_capture_at_s": track.latest_capture_at_s,
                    "latest_publish_at_s": track.latest_publish_at_s,
                    "source_stream_ids": sorted(
                        source
                        for source, capture_s in track.source_capture_at_s.items()
                        if now_s - capture_s <= self.track_ttl_s + 1e-12
                    ),
                }
                for track in sorted(self.tracks.values(), key=lambda item: item.canonical_track_id)
            ],
            "counters": dict(self.counters),
        }
