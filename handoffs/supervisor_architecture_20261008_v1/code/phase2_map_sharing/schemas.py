"""Wire-safe schemas for recipient-specific map contributions and warnings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Tuple


SCHEMA_VERSION = "scenesense.map_contribution.v1"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class MapObjectObservation:
    source_track_id: str
    class_name: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    confidence: float
    observed_at_s: float
    occlusion_state: str = "unknown"
    hazard_score: float = 0.0

    def validate(self) -> None:
        if not self.source_track_id or not self.class_name:
            raise ValueError("object source_track_id and class_name are required")
        for name in ("x_m", "y_m", "vx_mps", "vy_mps", "observed_at_s"):
            _finite(name, getattr(self, name))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("object confidence must be in [0, 1]")
        if not 0.0 <= float(self.hazard_score) <= 1.0:
            raise ValueError("object hazard_score must be in [0, 1]")


@dataclass(frozen=True)
class MapContribution:
    contribution_id: str
    source_ue_id: str
    recipient_ue_id: str
    sequence_number: int
    captured_at_s: float
    published_at_s: float
    profile_id: str
    payload_bytes: int
    objects: Tuple[MapObjectObservation, ...]
    schema: str = SCHEMA_VERSION
    operation: str = "Update"

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION or self.operation != "Update":
            raise ValueError("unsupported map-contribution schema or operation")
        if not self.contribution_id or not self.source_ue_id or not self.recipient_ue_id:
            raise ValueError("contribution/source/recipient IDs are required")
        if int(self.sequence_number) < 0 or int(self.payload_bytes) < 0:
            raise ValueError("sequence_number and payload_bytes must be nonnegative")
        capture = _finite("captured_at_s", self.captured_at_s)
        publish = _finite("published_at_s", self.published_at_s)
        if publish + 1e-12 < capture:
            raise ValueError("published_at_s cannot precede captured_at_s")
        source_track_ids = set()
        for obj in self.objects:
            obj.validate()
            if obj.observed_at_s > capture + 1e-9:
                raise ValueError("object observation cannot occur after contribution capture")
            if obj.source_track_id in source_track_ids:
                raise ValueError("source track IDs must be unique within a contribution")
            source_track_ids.add(obj.source_track_id)

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["objects"] = [asdict(obj) for obj in self.objects]
        payload["resource_uri"] = (
            f"/ss-sm-management/v1/spatial-maps/{self.recipient_ue_id}"
        )
        return payload

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "MapContribution":
        contribution = cls.from_dict(json.loads(payload.decode("utf-8")))
        if contribution.payload_bytes != len(payload):
            raise ValueError("declared payload_bytes does not match serialized application bytes")
        return contribution

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MapContribution":
        forbidden = {"actor_id", "carla_actor_id", "ground_truth_id"}
        if forbidden & set(payload):
            raise ValueError("runtime contribution contains evaluation-only identity")
        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError("objects must be a list")
        objects = []
        for raw in raw_objects:
            if not isinstance(raw, Mapping):
                raise ValueError("each object must be a mapping")
            if forbidden & set(raw):
                raise ValueError("runtime object contains evaluation-only identity")
            objects.append(
                MapObjectObservation(
                    source_track_id=str(raw["source_track_id"]),
                    class_name=str(raw["class_name"]).lower(),
                    x_m=float(raw["x_m"]),
                    y_m=float(raw["y_m"]),
                    vx_mps=float(raw["vx_mps"]),
                    vy_mps=float(raw["vy_mps"]),
                    confidence=float(raw["confidence"]),
                    observed_at_s=float(raw["observed_at_s"]),
                    occlusion_state=str(raw.get("occlusion_state", "unknown")),
                    hazard_score=float(raw.get("hazard_score", 0.0)),
                )
            )
        contribution = cls(
            contribution_id=str(payload["contribution_id"]),
            source_ue_id=str(payload["source_ue_id"]),
            recipient_ue_id=str(payload["recipient_ue_id"]),
            sequence_number=int(payload["sequence_number"]),
            captured_at_s=float(payload["captured_at_s"]),
            published_at_s=float(payload["published_at_s"]),
            profile_id=str(payload.get("profile_id", "unknown")),
            payload_bytes=int(payload.get("payload_bytes", 0)),
            objects=tuple(objects),
            schema=str(payload.get("schema", "")),
            operation=str(payload.get("operation", "")),
        )
        contribution.validate()
        resource_uri = payload.get("resource_uri")
        expected_uri = f"/ss-sm-management/v1/spatial-maps/{contribution.recipient_ue_id}"
        if resource_uri is not None and str(resource_uri) != expected_uri:
            raise ValueError("resource_uri does not match the named recipient")
        return contribution


def with_exact_payload_bytes(contribution: MapContribution) -> MapContribution:
    """Resolve the self-described application-byte count to a fixed point."""

    current = replace(contribution, payload_bytes=0)
    for _ in range(8):
        encoded_bytes = len(current.to_json_bytes())
        if current.payload_bytes == encoded_bytes:
            return current
        current = replace(current, payload_bytes=encoded_bytes)
    raise RuntimeError("map-contribution payload byte count did not converge")


@dataclass(frozen=True)
class EgoState:
    recipient_ue_id: str
    timestamp_s: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float


@dataclass(frozen=True)
class WarningEvent:
    recipient_ue_id: str
    canonical_track_id: str
    class_name: str
    warning_at_s: float
    time_to_closest_approach_s: float
    closest_approach_m: float
    object_x_m: float
    object_y_m: float
    map_aoi_s: float
    evidence_sources: Tuple[str, ...]
    evidence_track_ids: Tuple[str, ...]
    evidence_scope: str
    latest_capture_at_s: float
    latest_publish_at_s: float
