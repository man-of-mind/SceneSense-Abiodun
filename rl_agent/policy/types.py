"""Shared, dependency-light data structures for the Track A surrogate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


TrackKey = Tuple[str, int]


@dataclass(frozen=True)
class SceneObject:
    track_key: TrackKey
    class_name: str
    world_x: float
    world_y: float
    range_m: float
    speed_mps: float
    speed_sigma_mps: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class SceneFrame:
    episode_id: str
    step_index: int
    timestamp_s: float
    truth_objects: Tuple[SceneObject, ...]
    observed_objects: Tuple[SceneObject, ...]


@dataclass(frozen=True)
class QualitySnapshot:
    profile_id: str
    miou: float
    pedestrian_recall: float
    vehicle_recall: float
    object_recall: float
    normalized_utility: float
    base_loc_m: float


@dataclass(frozen=True)
class Contribution:
    source_ue_id: str
    capture_timestamp_s: float
    publish_timestamp_s: float
    confidence: float
    profile_id: str
    quality: QualitySnapshot


@dataclass(order=True)
class PendingFrame:
    publish_timestamp_s: float
    sequence_id: int
    capture_timestamp_s: float = field(compare=False)
    captured_track_keys: Tuple[TrackKey, ...] = field(compare=False)
    contribution: Contribution = field(compare=False)
    action_id: str = field(compare=False)
    delivered: bool = field(compare=False, default=True)


@dataclass
class MapObjectState:
    track_key: TrackKey
    contributions: List[Contribution] = field(default_factory=list)

    @property
    def newest(self) -> Optional[Contribution]:
        if not self.contributions:
            return None
        return max(self.contributions, key=lambda value: value.capture_timestamp_s)

    def install(self, contribution: Contribution) -> bool:
        newest = self.newest
        if newest is not None and contribution.capture_timestamp_s <= newest.capture_timestamp_s:
            return False
        self.contributions.append(contribution)
        return True


@dataclass(frozen=True)
class ChannelSnapshot:
    rung: str
    mcs: int
    observed_rung: str
    observed_mcs: int
    true_capacity_mbps: float
    estimated_capacity_mbps: float
    estimate_sigma_mbps: float
    representative_snr_db: float


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    objects: Tuple[SceneObject, ...]
    estimated_capacity_mbps: float
    capacity_sigma_mbps: float
    observed_channel_rung: str
    previous_action_id: str
    previous_delivery: Optional[bool]
    previous_latency_ms: Optional[float]
    scheduler_credit: float
    active_schedule_id: Optional[str]
    inflight_count: int
    newest_pending_capture_age_s: Optional[float]
    next_expected_arrival_s: Optional[float]
    map_capture_times: Dict[TrackKey, float]
    map_quality: Dict[TrackKey, QualitySnapshot]
