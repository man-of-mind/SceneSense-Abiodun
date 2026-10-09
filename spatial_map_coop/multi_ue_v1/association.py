"""Conservative filtering, association, and track selection for multi-UE maps.

The implementation deliberately does not average positions.  Detector
confidence is useful for deciding which of two otherwise equally fresh
observations to retain, but it is not a positional covariance.  Until a
validated uncertainty model exists, the selected observation is therefore the
freshest member of an associated group, with confidence used only as a
tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .buffer import MultiUESnapshot, SourceFrame
from .contracts import ObjectObservation


ASSOCIATION_ALGORITHM = "sequential_class_gated_hungarian.v1"
SELECTION_RULE = "freshest_capture_then_confidence_no_position_averaging.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_integer(value: int, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    result = value
    _require(result > 0, f"{label} must be positive")
    return result


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class AssociationPolicy:
    """Explicit engineering policy; thresholds are not scientific claims."""

    maximum_observation_age_ns: int
    alignment_tolerance_ns: int
    maximum_pair_time_delta_ns: int
    maximum_xy_distance_m: float
    maximum_relative_size_difference: float
    track_match_distance_m: float
    track_stale_after_ns: int
    minimum_confidence: float = 0.0
    confidence_provenance: str = "UPSTREAM_P025_OUTPUT_NO_ADDITIONAL_SCORE_GATE"

    def __post_init__(self) -> None:
        _positive_integer(self.maximum_observation_age_ns, "maximum_observation_age_ns")
        _positive_integer(self.alignment_tolerance_ns, "alignment_tolerance_ns")
        _positive_integer(self.maximum_pair_time_delta_ns, "maximum_pair_time_delta_ns")
        _positive_integer(self.track_stale_after_ns, "track_stale_after_ns")
        _require(
            self.maximum_pair_time_delta_ns <= self.alignment_tolerance_ns,
            "pair time delta must not exceed the alignment tolerance",
        )
        _require(
            _finite_nonnegative(self.maximum_xy_distance_m, "maximum_xy_distance_m") > 0.0,
            "maximum_xy_distance_m must be positive",
        )
        _require(
            _finite_nonnegative(self.track_match_distance_m, "track_match_distance_m") > 0.0,
            "track_match_distance_m must be positive",
        )
        relative = _finite_nonnegative(
            self.maximum_relative_size_difference,
            "maximum_relative_size_difference",
        )
        _require(relative <= 1.0, "maximum_relative_size_difference must not exceed one")
        confidence = _finite_nonnegative(self.minimum_confidence, "minimum_confidence")
        _require(confidence <= 1.0, "minimum_confidence must not exceed one")
        _require(bool(str(self.confidence_provenance).strip()), "confidence provenance is required")

    def as_dict(self) -> dict[str, object]:
        return {
            "maximum_observation_age_ns": self.maximum_observation_age_ns,
            "alignment_tolerance_ns": self.alignment_tolerance_ns,
            "maximum_pair_time_delta_ns": self.maximum_pair_time_delta_ns,
            "maximum_xy_distance_m": self.maximum_xy_distance_m,
            "maximum_relative_size_difference": self.maximum_relative_size_difference,
            "track_match_distance_m": self.track_match_distance_m,
            "track_stale_after_ns": self.track_stale_after_ns,
            "minimum_confidence": self.minimum_confidence,
            "confidence_provenance": self.confidence_provenance,
            "association_algorithm": ASSOCIATION_ALGORITHM,
            "selection_rule": SELECTION_RULE,
        }


@dataclass(frozen=True)
class CandidateObservation:
    source_key: tuple[str, str, str]
    frame_id: int
    capture_timestamp_ns: int
    age_ns: int
    observation: ObjectObservation

    @property
    def identity(self) -> tuple[str, str, str, int, str]:
        return (*self.source_key, self.frame_id, self.observation.observation_id)


@dataclass(frozen=True)
class RejectedObservation:
    identity: tuple[str, str, str, int, str]
    reason: str
    score: float
    age_ns: int


@dataclass(frozen=True)
class AssociatedObject:
    association_id: str
    class_name: str
    members: tuple[CandidateObservation, ...]
    selected: CandidateObservation

    @property
    def source_count(self) -> int:
        return len({member.source_key for member in self.members})


@dataclass(frozen=True)
class AssociationResult:
    snapshot_timestamp_ns: int
    clock_domain: str
    accepted: tuple[CandidateObservation, ...]
    rejected: tuple[RejectedObservation, ...]
    associations: tuple[AssociatedObject, ...]
    input_source_count: int
    aligned_source_count: int


@dataclass(frozen=True)
class MapTrack:
    track_id: str
    class_name: str
    world_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_deg: float
    score: float
    selected_ue_id: str
    selected_session_id: str
    selected_stream_id: str
    selected_frame_id: int
    selected_observation_id: str
    selected_raw_record_sha256: str
    capture_timestamp_ns: int
    age_ns: int
    contributing_sources: tuple[tuple[str, str, str], ...]
    contributing_observation_hashes: tuple[str, ...]
    association_id: str
    update_count: int


@dataclass
class _TrackState:
    track: MapTrack
    last_observed_timestamp_ns: int


def canonical_class_name(value: str) -> str | None:
    lowered = str(value or "").strip().lower().replace("_", "")
    aliases = {
        "vehicle": "Vehicle",
        "movingvehicle": "Vehicle",
        "parkedvehicle": "Vehicle",
        "car": "Vehicle",
        "person": "Pedestrian",
        "pedestrian": "Pedestrian",
        "walker": "Pedestrian",
        "cyclist": "Cyclist",
        "bicycle": "Cyclist",
    }
    return aliases.get(lowered)


def _source_candidates(source: SourceFrame) -> Iterable[CandidateObservation]:
    for observation in source.batch.objects:
        yield CandidateObservation(
            source_key=source.batch.source_key,
            frame_id=source.batch.frame_id,
            capture_timestamp_ns=source.batch.capture_timestamp_ns,
            age_ns=source.age_ns,
            observation=observation,
        )


def _candidate_order(candidate: CandidateObservation) -> tuple[object, ...]:
    return (*candidate.source_key, candidate.frame_id, candidate.observation.observation_id)


def _selection_key(candidate: CandidateObservation) -> tuple[object, ...]:
    return (
        candidate.capture_timestamp_ns,
        candidate.observation.score,
        tuple(reversed(_candidate_order(candidate))),
    )


def _selected(members: Sequence[CandidateObservation]) -> CandidateObservation:
    return max(members, key=_selection_key)


def _relative_size_difference(first: ObjectObservation, second: ObjectObservation) -> float:
    a = np.asarray(first.size_lwh, dtype=np.float64)
    b = np.asarray(second.size_lwh, dtype=np.float64)
    denominator = np.maximum(np.maximum(a, b), 0.05)
    return float(np.max(np.abs(a - b) / denominator))


def _pair_cost(
    first: CandidateObservation,
    second: CandidateObservation,
    policy: AssociationPolicy,
) -> float | None:
    first_class = canonical_class_name(first.observation.class_name)
    second_class = canonical_class_name(second.observation.class_name)
    if first_class is None or first_class != second_class:
        return None
    time_delta = abs(first.capture_timestamp_ns - second.capture_timestamp_ns)
    if time_delta > policy.maximum_pair_time_delta_ns:
        return None
    xy_a = np.asarray(first.observation.world_xyz[:2], dtype=np.float64)
    xy_b = np.asarray(second.observation.world_xyz[:2], dtype=np.float64)
    xy_distance = float(np.linalg.norm(xy_a - xy_b))
    if xy_distance > policy.maximum_xy_distance_m:
        return None
    size_difference = _relative_size_difference(first.observation, second.observation)
    if size_difference > policy.maximum_relative_size_difference:
        return None
    return (
        xy_distance / policy.maximum_xy_distance_m
        + size_difference / max(policy.maximum_relative_size_difference, 1.0e-12)
        + time_delta / policy.maximum_pair_time_delta_ns
    )


def _group_cost(
    members: Sequence[CandidateObservation],
    candidate: CandidateObservation,
    policy: AssociationPolicy,
) -> float | None:
    costs = [_pair_cost(member, candidate, policy) for member in members]
    if any(cost is None for cost in costs):
        return None
    return float(sum(float(cost) for cost in costs) / len(costs))


def _association_id(members: Sequence[CandidateObservation]) -> str:
    identity = [
        {
            "source_key": list(member.source_key),
            "frame_id": member.frame_id,
            "observation_id": member.observation.observation_id,
            "raw_record_sha256": member.observation.raw_record_sha256,
        }
        for member in sorted(members, key=_candidate_order)
    ]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def associate_snapshot(
    snapshot: MultiUESnapshot,
    policy: AssociationPolicy,
) -> AssociationResult:
    """Filter observations and associate different UE sources deterministically."""

    if snapshot.alignment_tolerance_ns != policy.alignment_tolerance_ns:
        raise ValueError("snapshot and association policy use different alignment tolerances")
    sources = sorted(snapshot.sources, key=lambda item: item.batch.source_key)
    latest_capture = max(
        (source.batch.capture_timestamp_ns for source in sources),
        default=snapshot.snapshot_timestamp_ns,
    )
    aligned_sources: list[SourceFrame] = []
    rejected: list[RejectedObservation] = []
    for source in sources:
        outside_alignment = (
            latest_capture - source.batch.capture_timestamp_ns > policy.alignment_tolerance_ns
        )
        too_old = source.age_ns > policy.maximum_observation_age_ns
        if not outside_alignment and not too_old:
            aligned_sources.append(source)
            continue
        reason = "SOURCE_OUTSIDE_ALIGNMENT_WINDOW" if outside_alignment else "OBSERVATION_TOO_OLD"
        for candidate in _source_candidates(source):
            rejected.append(
                RejectedObservation(
                    identity=candidate.identity,
                    reason=reason,
                    score=candidate.observation.score,
                    age_ns=candidate.age_ns,
                )
            )

    accepted: list[CandidateObservation] = []
    candidates_by_source: list[tuple[tuple[str, str, str], list[CandidateObservation]]] = []
    for source in aligned_sources:
        source_accepted: list[CandidateObservation] = []
        for candidate in _source_candidates(source):
            class_name = canonical_class_name(candidate.observation.class_name)
            if class_name is None:
                reason = "UNSUPPORTED_OBJECT_CLASS"
            elif candidate.observation.score < policy.minimum_confidence:
                reason = "BELOW_REGISTERED_CONFIDENCE"
            else:
                source_accepted.append(candidate)
                accepted.append(candidate)
                continue
            rejected.append(
                RejectedObservation(
                    identity=candidate.identity,
                    reason=reason,
                    score=candidate.observation.score,
                    age_ns=candidate.age_ns,
                )
            )
        candidates_by_source.append((source.batch.source_key, source_accepted))

    # Process the freshest source first, then use stable source identity as the
    # tie-breaker. Hungarian remains one-to-one for each new UE against the
    # groups already formed from earlier UEs.
    capture_by_key = {
        source.batch.source_key: source.batch.capture_timestamp_ns for source in aligned_sources
    }
    candidates_by_source.sort(key=lambda item: (-capture_by_key[item[0]], item[0]))
    groups: list[list[CandidateObservation]] = []
    for _source_key, candidates in candidates_by_source:
        candidates = sorted(candidates, key=_candidate_order)
        if not groups:
            groups.extend([[candidate] for candidate in candidates])
            continue
        if not candidates:
            continue
        sentinel = 1.0e9
        cost_matrix = np.full((len(groups), len(candidates)), sentinel, dtype=np.float64)
        for group_index, group in enumerate(groups):
            for candidate_index, candidate in enumerate(candidates):
                cost = _group_cost(group, candidate, policy)
                if cost is not None:
                    cost_matrix[group_index, candidate_index] = cost
        row_indices, column_indices = linear_sum_assignment(cost_matrix)
        matched_candidates: set[int] = set()
        for group_index, candidate_index in zip(row_indices.tolist(), column_indices.tolist()):
            if cost_matrix[group_index, candidate_index] >= sentinel:
                continue
            groups[group_index].append(candidates[candidate_index])
            matched_candidates.add(candidate_index)
        groups.extend(
            [candidate]
            for index, candidate in enumerate(candidates)
            if index not in matched_candidates
        )

    associations = []
    for members in groups:
        ordered_members = tuple(sorted(members, key=_candidate_order))
        chosen = _selected(ordered_members)
        associations.append(
            AssociatedObject(
                association_id=_association_id(ordered_members),
                class_name=canonical_class_name(chosen.observation.class_name) or "Unknown",
                members=ordered_members,
                selected=chosen,
            )
        )
    associations.sort(key=lambda group: (group.class_name, group.association_id))
    return AssociationResult(
        snapshot_timestamp_ns=snapshot.snapshot_timestamp_ns,
        clock_domain=snapshot.clock_domain,
        accepted=tuple(sorted(accepted, key=_candidate_order)),
        rejected=tuple(sorted(rejected, key=lambda item: item.identity)),
        associations=tuple(associations),
        input_source_count=len(sources),
        aligned_source_count=len(aligned_sources),
    )


class ConservativeTrackStore:
    """Maintain map identities while retaining selected-source measurements."""

    def __init__(self, policy: AssociationPolicy) -> None:
        self._policy = policy
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1

    def _new_track_id(self) -> str:
        track_id = f"coop_track_{self._next_track_number:06d}"
        self._next_track_number += 1
        return track_id

    @staticmethod
    def _track_cost(track: MapTrack, association: AssociatedObject, maximum: float) -> float | None:
        if track.class_name != association.class_name:
            return None
        first = np.asarray(track.world_xyz[:2], dtype=np.float64)
        second = np.asarray(association.selected.observation.world_xyz[:2], dtype=np.float64)
        distance = float(np.linalg.norm(first - second))
        return distance if distance <= maximum else None

    @staticmethod
    def _materialize(
        track_id: str,
        association: AssociatedObject,
        snapshot_timestamp_ns: int,
        update_count: int,
    ) -> MapTrack:
        selected = association.selected
        observation = selected.observation
        return MapTrack(
            track_id=track_id,
            class_name=association.class_name,
            world_xyz=observation.world_xyz,
            size_lwh=observation.size_lwh,
            yaw_deg=observation.yaw_deg,
            score=observation.score,
            selected_ue_id=selected.source_key[0],
            selected_session_id=selected.source_key[1],
            selected_stream_id=selected.source_key[2],
            selected_frame_id=selected.frame_id,
            selected_observation_id=observation.observation_id,
            selected_raw_record_sha256=observation.raw_record_sha256,
            capture_timestamp_ns=selected.capture_timestamp_ns,
            age_ns=snapshot_timestamp_ns - selected.capture_timestamp_ns,
            contributing_sources=tuple(sorted({member.source_key for member in association.members})),
            contributing_observation_hashes=tuple(
                sorted(member.observation.raw_record_sha256 for member in association.members)
            ),
            association_id=association.association_id,
            update_count=update_count,
        )

    def update(self, result: AssociationResult) -> tuple[MapTrack, ...]:
        now_ns = result.snapshot_timestamp_ns
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if now_ns - state.last_observed_timestamp_ns <= self._policy.track_stale_after_ns
        }
        track_ids = sorted(self._tracks)
        associations = list(result.associations)
        matched_associations: set[int] = set()
        if track_ids and associations:
            sentinel = 1.0e9
            costs = np.full((len(track_ids), len(associations)), sentinel, dtype=np.float64)
            for track_index, track_id in enumerate(track_ids):
                for association_index, association in enumerate(associations):
                    cost = self._track_cost(
                        self._tracks[track_id].track,
                        association,
                        self._policy.track_match_distance_m,
                    )
                    if cost is not None:
                        costs[track_index, association_index] = cost
            rows, columns = linear_sum_assignment(costs)
            for track_index, association_index in zip(rows.tolist(), columns.tolist()):
                if costs[track_index, association_index] >= sentinel:
                    continue
                track_id = track_ids[track_index]
                state = self._tracks[track_id]
                association = associations[association_index]
                repeated_measurement = state.track.association_id == association.association_id
                updated = self._materialize(
                    track_id,
                    association,
                    now_ns,
                    state.track.update_count + (0 if repeated_measurement else 1),
                )
                self._tracks[track_id] = _TrackState(
                    updated,
                    association.selected.capture_timestamp_ns,
                )
                matched_associations.add(association_index)
        for association_index, association in enumerate(associations):
            if association_index in matched_associations:
                continue
            track_id = self._new_track_id()
            track = self._materialize(track_id, association, now_ns, 1)
            self._tracks[track_id] = _TrackState(
                track,
                association.selected.capture_timestamp_ns,
            )
        return tuple(
            replace(
                self._tracks[track_id].track,
                age_ns=now_ns - self._tracks[track_id].track.capture_timestamp_ns,
            )
            for track_id in sorted(self._tracks)
        )
