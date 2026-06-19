from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class SensorPose2D:
    """Top-down sensor pose in world coordinates."""

    x: float
    y: float
    yaw_deg: float
    z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SpatialObject:
    """Object estimate expressed in world coordinates."""

    object_id: str
    class_name: str
    x: float
    y: float
    z: float = 0.0
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    yaw_deg: float = 0.0
    confidence: float = 1.0
    source_stream_id: str = ""
    frame_id: Optional[int] = None
    timestamp_s: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def xy(self) -> Point2D:
        return (float(self.x), float(self.y))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class LocalSensorMap:
    """Map contribution from one camera/radar/vehicle/pole stream."""

    stream_id: str
    pose: SensorPose2D
    fov_polygon: List[Point2D]
    objects: List[SpatialObject] = field(default_factory=list)
    timestamp_s: Optional[float] = None
    frame_id: Optional[int] = None
    sensor_type: str = "rgb_radar"
    fov_deg: float = 90.0
    range_m: float = 60.0
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["pose"] = self.pose.to_dict()
        payload["objects"] = [obj.to_dict() for obj in self.objects]
        return payload


@dataclass
class AssociationResult:
    """One cross-stream object association cluster."""

    canonical_id: str
    class_name: str
    members: List[SpatialObject]
    centroid_x: float
    centroid_y: float
    max_pairwise_distance_m: float

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["members"] = [obj.to_dict() for obj in self.members]
        return payload


@dataclass
class OcclusionHypothesis:
    """Conservative explanation for an object missing in an overlapping FoV."""

    source_stream_id: str
    missing_from_stream_id: str
    object_id: str
    class_name: str
    x: float
    y: float
    reason: str
    confidence: float
    overlap_area_m2: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
