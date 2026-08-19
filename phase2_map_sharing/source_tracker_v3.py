"""Causal source-local tracker v3 for bounded offline Phase-2 evaluation.

The capture-time v1 tracker in :mod:`data_collection.phase2_causal_runtime`
is deliberately unchanged for provenance.  This sibling implementation keeps
the same ``update`` shape while adding three pre-publication safeguards:

* a track is returned only after consecutive causal detections confirm it;
* velocity is exponentially smoothed and bounded by class plausibility; and
* same-class detections within a small world-space radius are suppressed by
  score before association.

The tracker has no truth input and rejects out-of-order updates.  It therefore
uses only the current detection set and state retained from earlier calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple


TRACKER_V3_VERSION = "source_local_confirmed_cv.v3"
ROLE_NAMES = frozenset({"helper", "recipient"})

DEFAULT_MAXIMUM_SPEED_MPS_BY_CLASS = {
    "person": 12.0,
    "pedestrian": 12.0,
    "walker": 12.0,
    "cyclist": 25.0,
    "bicycle": 25.0,
    "vehicle": 60.0,
    "car": 60.0,
    "truck": 60.0,
    "bus": 60.0,
    "object": 40.0,
}


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


def _class_name(value: object) -> str:
    name = str(value if value is not None else "object").strip().lower()
    return name or "object"


def _limit_planar_velocity(
    vx: float,
    vy: float,
    maximum_speed_mps: float,
) -> tuple[float, float, bool]:
    speed = math.hypot(vx, vy)
    if speed <= maximum_speed_mps + 1e-12 or speed <= 1e-12:
        return float(vx), float(vy), False
    scale = maximum_speed_mps / speed
    return float(vx * scale), float(vy * scale), True


@dataclass
class _TrackV3:
    track_id: str
    class_name: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    score: float
    last_timestamp_s: float
    last_frame_id: int
    hit_count: int
    consecutive_hits: int
    confirmed: bool
    confirmed_at_frame_id: Optional[int]
    missed_frames: int = 0
    velocity_limited: bool = False


class SourceLocalCausalTrackerV3:
    """Confirmed nearest-CV tracker using present and past observations only.

    Defaults are intentionally conservative rather than learned.  Two
    consecutive hits add 0.1 seconds of confirmation delay at 10 Hz.  A
    tentative track dies on its first miss, while a confirmed track retains the
    configured missed-frame grace.  Duplicate suppression is class-exact and
    keeps the highest-scoring detection (then the lowest input index on a tie).
    """

    def __init__(
        self,
        source_role: str,
        *,
        association_gate_m: float = 5.0,
        maximum_missed_frames: int = 3,
        minimum_confirmation_hits: int = 2,
        duplicate_suppression_radius_m: float = 0.75,
        velocity_smoothing_alpha: float = 0.5,
        speed_plausibility_slack_m: float = 0.75,
        maximum_speed_mps_by_class: Optional[Mapping[str, float]] = None,
        maximum_vertical_speed_mps: float = 8.0,
    ) -> None:
        if source_role not in ROLE_NAMES:
            raise ValueError("source_role must be helper or recipient")
        if (
            not math.isfinite(float(association_gate_m))
            or association_gate_m <= 0.0
        ):
            raise ValueError("association_gate_m must be finite and positive")
        if type(maximum_missed_frames) is not int or maximum_missed_frames < 0:
            raise ValueError("maximum_missed_frames must be a non-negative integer")
        if (
            type(minimum_confirmation_hits) is not int
            or minimum_confirmation_hits < 1
        ):
            raise ValueError("minimum_confirmation_hits must be a positive integer")
        if (
            not math.isfinite(float(duplicate_suppression_radius_m))
            or duplicate_suppression_radius_m < 0.0
        ):
            raise ValueError(
                "duplicate_suppression_radius_m must be finite and non-negative"
            )
        if (
            not math.isfinite(float(velocity_smoothing_alpha))
            or not 0.0 < velocity_smoothing_alpha <= 1.0
        ):
            raise ValueError("velocity_smoothing_alpha must be within (0, 1]")
        if (
            not math.isfinite(float(speed_plausibility_slack_m))
            or speed_plausibility_slack_m < 0.0
        ):
            raise ValueError(
                "speed_plausibility_slack_m must be finite and non-negative"
            )
        if (
            not math.isfinite(float(maximum_vertical_speed_mps))
            or maximum_vertical_speed_mps <= 0.0
        ):
            raise ValueError("maximum_vertical_speed_mps must be finite and positive")

        speed_limits = dict(DEFAULT_MAXIMUM_SPEED_MPS_BY_CLASS)
        for name, value in (maximum_speed_mps_by_class or {}).items():
            limit = float(value)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("class maximum speeds must be finite and positive")
            speed_limits[_class_name(name)] = limit

        self.source_role = source_role
        self.association_gate_m = float(association_gate_m)
        self.maximum_missed_frames = maximum_missed_frames
        self.minimum_confirmation_hits = minimum_confirmation_hits
        self.duplicate_suppression_radius_m = float(
            duplicate_suppression_radius_m
        )
        self.velocity_smoothing_alpha = float(velocity_smoothing_alpha)
        self.speed_plausibility_slack_m = float(speed_plausibility_slack_m)
        self.maximum_speed_mps_by_class = speed_limits
        self.maximum_vertical_speed_mps = float(maximum_vertical_speed_mps)
        self._next_id = 1
        self._tracks: Dict[str, _TrackV3] = {}
        self._last_frame_id: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None

    def _maximum_speed(self, class_name: str) -> float:
        return float(
            self.maximum_speed_mps_by_class.get(
                class_name,
                self.maximum_speed_mps_by_class["object"],
            )
        )

    def _normalize_and_suppress(
        self,
        detections: Sequence[Mapping[str, object]],
    ) -> tuple[list[dict], list[dict]]:
        normalized: list[dict] = []
        for index, detection in enumerate(detections):
            x = _finite(detection.get("world_x"), float("nan"))
            y = _finite(detection.get("world_y"), float("nan"))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            normalized.append(
                {
                    "detection_index": int(index),
                    "class_name": _class_name(detection.get("class_name")),
                    "x": x,
                    "y": y,
                    "z": _finite(detection.get("world_z"), 0.0),
                    "score": _finite(detection.get("score"), 0.0),
                }
            )

        kept: list[dict] = []
        suppressed: list[dict] = []
        ranked = sorted(
            normalized,
            key=lambda row: (-float(row["score"]), int(row["detection_index"])),
        )
        for detection in ranked:
            duplicate_candidates: list[tuple[float, int, dict]] = []
            if self.duplicate_suppression_radius_m > 0.0:
                for candidate in kept:
                    if candidate["class_name"] != detection["class_name"]:
                        continue
                    distance = math.hypot(
                        float(candidate["x"]) - float(detection["x"]),
                        float(candidate["y"]) - float(detection["y"]),
                    )
                    if distance <= self.duplicate_suppression_radius_m + 1e-12:
                        duplicate_candidates.append(
                            (distance, int(candidate["detection_index"]), candidate)
                        )
            if duplicate_candidates:
                distance, _, duplicate_of = min(duplicate_candidates)
                suppressed.append(
                    {
                        "detection": detection,
                        "duplicate_of_detection_index": int(
                            duplicate_of["detection_index"]
                        ),
                        "distance_m": float(distance),
                    }
                )
            else:
                kept.append(detection)
        kept.sort(key=lambda row: int(row["detection_index"]))
        suppressed.sort(
            key=lambda row: int(row["detection"]["detection_index"])
        )
        return kept, suppressed

    @staticmethod
    def _association_row(
        *,
        frame_id: int,
        timestamp_s: float,
        detection_index: object,
        source_track_id: str,
        association: str,
        association_distance_m: object,
        class_name: str,
        duplicate_of_detection_index: object = "",
        observed_speed_mps: object = "",
        velocity_limited: object = "",
    ) -> dict:
        return {
            "frame_id": int(frame_id),
            "timestamp_s": float(timestamp_s),
            "detection_index": detection_index,
            "source_track_id": source_track_id,
            "association": association,
            "association_distance_m": association_distance_m,
            "class_name": class_name,
            "duplicate_of_detection_index": duplicate_of_detection_index,
            "observed_speed_mps": observed_speed_mps,
            "velocity_limited": velocity_limited,
        }

    def update(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        detections: Sequence[Mapping[str, object]],
    ) -> Tuple[list[dict], list[dict]]:
        frame = int(frame_id)
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if (
            self._last_timestamp_s is not None
            and timestamp < self._last_timestamp_s - 1e-12
        ):
            raise ValueError("timestamp_s cannot move backwards")

        normalized, suppressed = self._normalize_and_suppress(detections)
        detection_associations = [
            self._association_row(
                frame_id=frame,
                timestamp_s=timestamp,
                detection_index=int(row["detection"]["detection_index"]),
                source_track_id="",
                association="duplicate_suppressed",
                association_distance_m=float(row["distance_m"]),
                class_name=str(row["detection"]["class_name"]),
                duplicate_of_detection_index=int(
                    row["duplicate_of_detection_index"]
                ),
            )
            for row in suppressed
        ]

        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            dt = max(0.0, timestamp - track.last_timestamp_s)
            predicted_x = track.x + track.vx * dt
            predicted_y = track.y + track.vy * dt
            maximum_displacement = (
                self._maximum_speed(track.class_name) * dt
                + self.speed_plausibility_slack_m
            )
            for detection_index, detection in enumerate(normalized):
                if detection["class_name"] != track.class_name:
                    continue
                observed_displacement = math.hypot(
                    float(detection["x"]) - track.x,
                    float(detection["y"]) - track.y,
                )
                if observed_displacement > maximum_displacement + 1e-12:
                    continue
                innovation = math.hypot(
                    float(detection["x"]) - predicted_x,
                    float(detection["y"]) - predicted_y,
                )
                if innovation <= self.association_gate_m + 1e-12:
                    candidates.append((innovation, track_id, detection_index))

        assignments: Dict[int, tuple[str, float]] = {}
        used_tracks: set[str] = set()
        for distance, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in assignments:
                continue
            used_tracks.add(track_id)
            assignments[detection_index] = (track_id, float(distance))

        updated: Dict[str, _TrackV3] = {}
        for detection_index, detection in enumerate(normalized):
            if detection_index in assignments:
                track_id, association_distance = assignments[detection_index]
                previous = self._tracks[track_id]
                dt = timestamp - previous.last_timestamp_s
                if dt > 1e-9:
                    raw_vx = (float(detection["x"]) - previous.x) / dt
                    raw_vy = (float(detection["y"]) - previous.y) / dt
                    raw_vz = (float(detection["z"]) - previous.z) / dt
                    observed_speed = math.hypot(raw_vx, raw_vy)
                    bounded_vx, bounded_vy, planar_limited = _limit_planar_velocity(
                        raw_vx,
                        raw_vy,
                        self._maximum_speed(previous.class_name),
                    )
                    bounded_vz = max(
                        -self.maximum_vertical_speed_mps,
                        min(self.maximum_vertical_speed_mps, raw_vz),
                    )
                    vertical_limited = not math.isclose(
                        bounded_vz, raw_vz, rel_tol=0.0, abs_tol=1e-12
                    )
                    alpha = self.velocity_smoothing_alpha
                    vx = alpha * bounded_vx + (1.0 - alpha) * previous.vx
                    vy = alpha * bounded_vy + (1.0 - alpha) * previous.vy
                    vz = alpha * bounded_vz + (1.0 - alpha) * previous.vz
                    velocity_limited = planar_limited or vertical_limited
                else:
                    vx, vy, vz = previous.vx, previous.vy, previous.vz
                    observed_speed = 0.0
                    velocity_limited = False
                consecutive_hits = (
                    previous.consecutive_hits + 1
                    if previous.missed_frames == 0
                    else 1
                )
                confirmed = bool(
                    previous.confirmed
                    or consecutive_hits >= self.minimum_confirmation_hits
                )
                confirmed_at = previous.confirmed_at_frame_id
                if confirmed and confirmed_at is None:
                    confirmed_at = frame
                if not previous.confirmed and confirmed:
                    lifecycle = "confirmed"
                elif confirmed:
                    lifecycle = "matched_confirmed"
                else:
                    lifecycle = "matched_tentative"
                track = _TrackV3(
                    track_id=track_id,
                    class_name=str(detection["class_name"]),
                    x=float(detection["x"]),
                    y=float(detection["y"]),
                    z=float(detection["z"]),
                    vx=float(vx),
                    vy=float(vy),
                    vz=float(vz),
                    score=float(detection["score"]),
                    last_timestamp_s=timestamp,
                    last_frame_id=frame,
                    hit_count=previous.hit_count + 1,
                    consecutive_hits=consecutive_hits,
                    confirmed=confirmed,
                    confirmed_at_frame_id=confirmed_at,
                    missed_frames=0,
                    velocity_limited=velocity_limited,
                )
            else:
                track_id = f"{self.source_role}:track:{self._next_id:06d}"
                self._next_id += 1
                association_distance = float("nan")
                observed_speed = ""
                velocity_limited = False
                confirmed = self.minimum_confirmation_hits == 1
                lifecycle = "birth_confirmed" if confirmed else "birth_tentative"
                track = _TrackV3(
                    track_id=track_id,
                    class_name=str(detection["class_name"]),
                    x=float(detection["x"]),
                    y=float(detection["y"]),
                    z=float(detection["z"]),
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                    score=float(detection["score"]),
                    last_timestamp_s=timestamp,
                    last_frame_id=frame,
                    hit_count=1,
                    consecutive_hits=1,
                    confirmed=confirmed,
                    confirmed_at_frame_id=frame if confirmed else None,
                )
            updated[track_id] = track
            detection_associations.append(
                self._association_row(
                    frame_id=frame,
                    timestamp_s=timestamp,
                    detection_index=int(detection["detection_index"]),
                    source_track_id=track_id,
                    association=lifecycle,
                    association_distance_m=association_distance,
                    class_name=track.class_name,
                    observed_speed_mps=observed_speed,
                    velocity_limited=velocity_limited,
                )
            )

        track_state_associations: list[dict] = []
        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            if not track.confirmed:
                lifecycle = "death_tentative"
            else:
                missed_frames = track.missed_frames + 1
                if missed_frames <= self.maximum_missed_frames:
                    updated[track_id] = replace(
                        track,
                        missed_frames=missed_frames,
                        consecutive_hits=0,
                    )
                    lifecycle = "missed_confirmed"
                else:
                    lifecycle = "death_confirmed"
            track_state_associations.append(
                self._association_row(
                    frame_id=frame,
                    timestamp_s=timestamp,
                    detection_index="",
                    source_track_id=track_id,
                    association=lifecycle,
                    association_distance_m="",
                    class_name=track.class_name,
                )
            )

        self._tracks = updated
        self._last_frame_id = frame
        self._last_timestamp_s = timestamp
        detection_associations.sort(
            key=lambda row: (int(row["detection_index"]), str(row["association"]))
        )
        associations = detection_associations + track_state_associations
        outputs = [
            {
                "source_track_id": track.track_id,
                "source_role": self.source_role,
                "tracker_version": TRACKER_V3_VERSION,
                "class_name": track.class_name,
                "world_x": track.x,
                "world_y": track.y,
                "world_z": track.z,
                "velocity_x": track.vx,
                "velocity_y": track.vy,
                "velocity_z": track.vz,
                "score": track.score,
                "last_observed_timestamp_s": track.last_timestamp_s,
                "last_observed_frame_id": track.last_frame_id,
                "missed_frames": track.missed_frames,
                "confirmed": True,
                "confirmation_hits": track.hit_count,
                "consecutive_hits": track.consecutive_hits,
                "confirmed_at_frame_id": track.confirmed_at_frame_id,
                "velocity_limited": track.velocity_limited,
            }
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            if track.confirmed
        ]
        return outputs, associations
