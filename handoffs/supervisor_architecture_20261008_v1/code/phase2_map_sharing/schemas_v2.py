"""Versioned causal wire schema for Phase-2 map contributions.

The v1 schema remains frozen for its checked-in plumbing artifacts.  This module
adds the uncertainty, decision-timing, and provenance required before the paired
causal pilot without changing v1 serialization.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION_V2 = "scenesense.map_contribution.v2"
RECIPIENT_STATE_SCHEMA_V2 = "scenesense.recipient_state.v2"
STATE_ORDER = ("x_m", "y_m", "vx_mps", "vy_mps")
PLACEMENT_ACTIONS = frozenset({"SPLIT_FEATURE", "LOCAL_INFER", "SKIP_INFERENCE"})
PUBLICATION_ACTIONS = frozenset(
    {"PUBLISH_ALL", "PUBLISH_HAZARD_SUBSET", "SKIP_PUBLICATION"}
)
CONTRIBUTION_PUBLICATION_ACTIONS = PUBLICATION_ACTIONS - {"SKIP_PUBLICATION"}
FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "actor_id",
        "carla_actor_id",
        "gt_actor_id",
        "gt_id",
        "ground_truth_id",
        "matched_gt_id",
        "oracle_association",
        "evaluation_truth",
        "shadow_inference",
        "truth_id",
        "future_trajectory",
        "collision_label",
    }
)
PRODUCTION_CHUNK_HEADER_BYTES = 8


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _sha256(name: str, value: str) -> str:
    result = _nonempty(name, value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return result


def _matrix4(name: str, values: Sequence[float]) -> Tuple[float, ...]:
    result = tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))
    if len(result) != 16:
        raise ValueError(f"{name} must contain 16 row-major values for {STATE_ORDER}")
    tolerance = 1e-9
    for row in range(4):
        for column in range(4):
            if abs(result[4 * row + column] - result[4 * column + row]) > tolerance:
                raise ValueError(f"{name} must be symmetric")

    # Cholesky-like positive-semidefinite check that also permits zero-variance
    # dimensions when their row/column contains no incompatible covariance.
    lower = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for column in range(row + 1):
            residual = result[4 * row + column] - sum(
                lower[row][k] * lower[column][k] for k in range(column)
            )
            if row == column:
                if residual < -tolerance:
                    raise ValueError(f"{name} must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, residual))
            elif lower[column][column] <= tolerance:
                if abs(residual) > tolerance:
                    raise ValueError(f"{name} must be positive semidefinite")
            else:
                lower[row][column] = residual / lower[column][column]
    return result


def _reject_forbidden_runtime_keys(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        normalized_keys = {str(key).strip().lower() for key in value}
        forbidden = set(FORBIDDEN_RUNTIME_KEYS & normalized_keys)
        forbidden.update(
            key
            for key in normalized_keys
            if key.startswith("gt_")
            or key.startswith("truth_")
            or key.endswith("_truth_id")
            or "carla_actor" in key
        )
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"{path} contains evaluation-only runtime keys: {names}")
        for key, item in value.items():
            _reject_forbidden_runtime_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_runtime_keys(item, f"{path}[{index}]")


def _reject_unknown_keys(
    payload: Mapping[str, object], allowed: set[str], path: str
) -> None:
    unknown = {str(key) for key in payload} - allowed
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class MapObjectObservationV2:
    source_track_id: str
    tracker_id: str
    tracker_version: str
    class_name: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    confidence: float
    measured_at_s: float
    state_covariance: Tuple[float, ...]
    motion_model_id: str
    process_noise_model_id: str
    process_noise_covariance_per_s: Tuple[float, ...]
    validity_horizon_s: float
    occlusion_state: str = "unknown"
    occlusion_source: str = "unknown"
    hazard_score: float = 0.0
    hazard_source: str = "none"
    recipient_state_observed_at_s: Optional[float] = None
    recipient_state_available_at_s: Optional[float] = None

    def validate(self) -> None:
        for name in (
            "source_track_id",
            "tracker_id",
            "tracker_version",
            "class_name",
            "motion_model_id",
            "process_noise_model_id",
            "occlusion_state",
            "occlusion_source",
            "hazard_source",
        ):
            _nonempty(name, getattr(self, name))
        for name in ("x_m", "y_m", "vx_mps", "vy_mps", "measured_at_s"):
            _finite(name, getattr(self, name))
        _matrix4("state_covariance", self.state_covariance)
        _matrix4("process_noise_covariance_per_s", self.process_noise_covariance_per_s)
        if _finite("validity_horizon_s", self.validity_horizon_s) <= 0.0:
            raise ValueError("validity_horizon_s must be positive")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= float(self.hazard_score) <= 1.0:
            raise ValueError("hazard_score must be in [0, 1]")
        observed = self.recipient_state_observed_at_s
        available = self.recipient_state_available_at_s
        if (observed is None) != (available is None):
            raise ValueError("recipient-state observed/available timestamps must appear together")
        if observed is not None and available is not None:
            observed_value = _finite("recipient_state_observed_at_s", observed)
            available_value = _finite("recipient_state_available_at_s", available)
            if available_value + 1e-12 < observed_value:
                raise ValueError("recipient state cannot be available before it was observed")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["state_order"] = list(STATE_ORDER)
        payload["state_covariance"] = list(self.state_covariance)
        payload["process_noise_covariance_per_s"] = list(
            self.process_noise_covariance_per_s
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MapObjectObservationV2":
        _reject_forbidden_runtime_keys(payload, "object")
        _reject_unknown_keys(
            payload,
            {item.name for item in fields(cls)} | {"state_order"},
            "object",
        )
        state_order = tuple(str(item) for item in payload.get("state_order", ()))
        if state_order != STATE_ORDER:
            raise ValueError(f"state_order must be {STATE_ORDER}")
        result = cls(
            source_track_id=str(payload["source_track_id"]),
            tracker_id=str(payload["tracker_id"]),
            tracker_version=str(payload["tracker_version"]),
            class_name=str(payload["class_name"]).lower(),
            x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]),
            vx_mps=float(payload["vx_mps"]),
            vy_mps=float(payload["vy_mps"]),
            confidence=float(payload["confidence"]),
            measured_at_s=float(payload["measured_at_s"]),
            state_covariance=tuple(float(item) for item in payload["state_covariance"]),
            motion_model_id=str(payload["motion_model_id"]),
            process_noise_model_id=str(payload["process_noise_model_id"]),
            process_noise_covariance_per_s=tuple(
                float(item) for item in payload["process_noise_covariance_per_s"]
            ),
            validity_horizon_s=float(payload["validity_horizon_s"]),
            occlusion_state=str(payload.get("occlusion_state", "unknown")),
            occlusion_source=str(payload.get("occlusion_source", "unknown")),
            hazard_score=float(payload.get("hazard_score", 0.0)),
            hazard_source=str(payload.get("hazard_source", "none")),
            recipient_state_observed_at_s=(
                None
                if payload.get("recipient_state_observed_at_s") is None
                else float(payload["recipient_state_observed_at_s"])
            ),
            recipient_state_available_at_s=(
                None
                if payload.get("recipient_state_available_at_s") is None
                else float(payload["recipient_state_available_at_s"])
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class MapContributionV2:
    contribution_id: str
    source_ue_id: str
    recipient_ue_id: str
    sequence_number: int
    captured_at_s: float
    placement_decision_id: str
    placement_decision_at_s: float
    inference_completed_at_s: float
    publication_decision_id: str
    publication_decision_at_s: float
    published_at_s: float
    clock_id: str
    publication_decision_locus: str
    inference_placement: str
    publication_action: str
    profile_id: str
    target_fps: float
    model_id: str
    model_sha256: str
    config_sha256: str
    code_revision: str
    source_sensor_ids: Tuple[str, ...]
    calibration_ids: Tuple[str, ...]
    transport_chunk_bytes: int
    chunk_count: int
    application_payload_bytes: int
    objects: Tuple[MapObjectObservationV2, ...]
    schema: str = SCHEMA_VERSION_V2
    operation: str = "Update"

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION_V2 or self.operation != "Update":
            raise ValueError("unsupported v2 map-contribution schema or operation")
        for name in (
            "contribution_id",
            "source_ue_id",
            "recipient_ue_id",
            "placement_decision_id",
            "publication_decision_id",
            "clock_id",
            "publication_decision_locus",
            "profile_id",
            "model_id",
            "code_revision",
        ):
            _nonempty(name, getattr(self, name))
        _sha256("model_sha256", self.model_sha256)
        _sha256("config_sha256", self.config_sha256)
        if self.inference_placement not in PLACEMENT_ACTIONS - {"SKIP_INFERENCE"}:
            raise ValueError("a contribution requires SPLIT_FEATURE or LOCAL_INFER placement")
        if self.publication_action not in CONTRIBUTION_PUBLICATION_ACTIONS:
            raise ValueError("a contribution requires a publishing action")
        if self.publication_decision_locus not in {"helper", "edge", "recipient"}:
            raise ValueError("publication_decision_locus must be helper, edge, or recipient")
        if int(self.sequence_number) < 0:
            raise ValueError("sequence_number must be nonnegative")
        if _finite("target_fps", self.target_fps) <= 0.0:
            raise ValueError("target_fps must be positive")
        if int(self.application_payload_bytes) < 0 or int(self.chunk_count) <= 0:
            raise ValueError("payload bytes must be nonnegative and chunk_count positive")
        capacity = int(self.transport_chunk_bytes) - PRODUCTION_CHUNK_HEADER_BYTES
        if capacity <= 0:
            raise ValueError("transport_chunk_bytes must exceed the production header")
        expected_chunks = max(
            1, (int(self.application_payload_bytes) + capacity - 1) // capacity
        )
        if int(self.chunk_count) != expected_chunks:
            raise ValueError("chunk_count does not match application payload and chunk capacity")
        if not self.source_sensor_ids or not self.calibration_ids:
            raise ValueError("source_sensor_ids and calibration_ids must be nonempty")
        for sensor_id in self.source_sensor_ids:
            _nonempty("source_sensor_id", sensor_id)
        for calibration_id in self.calibration_ids:
            _nonempty("calibration_id", calibration_id)
        if len(set(self.source_sensor_ids)) != len(self.source_sensor_ids):
            raise ValueError("source_sensor_ids must be unique")
        if len(set(self.calibration_ids)) != len(self.calibration_ids):
            raise ValueError("calibration_ids must be unique")

        timestamps = [
            _finite("placement_decision_at_s", self.placement_decision_at_s),
            _finite("captured_at_s", self.captured_at_s),
            _finite("inference_completed_at_s", self.inference_completed_at_s),
            _finite("publication_decision_at_s", self.publication_decision_at_s),
            _finite("published_at_s", self.published_at_s),
        ]
        if any(later + 1e-12 < earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("v2 contribution timestamps violate causal ordering")

        source_track_ids = set()
        for obj in self.objects:
            obj.validate()
            if obj.measured_at_s > self.captured_at_s + 1e-9:
                raise ValueError("object measurement cannot occur after contribution capture")
            if obj.source_track_id in source_track_ids:
                raise ValueError("source track IDs must be unique within a contribution")
            source_track_ids.add(obj.source_track_id)
            if self.publication_action == "PUBLISH_HAZARD_SUBSET":
                if obj.recipient_state_available_at_s is None:
                    raise ValueError("hazard-subset objects require causal recipient-state provenance")
                if obj.recipient_state_available_at_s > self.publication_decision_at_s + 1e-12:
                    raise ValueError("recipient state arrived after the publication decision")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["source_sensor_ids"] = list(self.source_sensor_ids)
        payload["calibration_ids"] = list(self.calibration_ids)
        payload["objects"] = [item.to_dict() for item in self.objects]
        payload["resource_uri"] = (
            f"/ss-sm-management/v2/spatial-maps/{self.recipient_ue_id}"
        )
        return payload

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "MapContributionV2":
        result = cls.from_dict(json.loads(payload.decode("utf-8")))
        if result.application_payload_bytes != len(payload):
            raise ValueError("declared application_payload_bytes does not match serialization")
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MapContributionV2":
        _reject_forbidden_runtime_keys(payload)
        _reject_unknown_keys(
            payload,
            {item.name for item in fields(cls)} | {"resource_uri"},
            "contribution",
        )
        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError("objects must be a list")
        result = cls(
            contribution_id=str(payload["contribution_id"]),
            source_ue_id=str(payload["source_ue_id"]),
            recipient_ue_id=str(payload["recipient_ue_id"]),
            sequence_number=int(payload["sequence_number"]),
            captured_at_s=float(payload["captured_at_s"]),
            placement_decision_id=str(payload["placement_decision_id"]),
            placement_decision_at_s=float(payload["placement_decision_at_s"]),
            inference_completed_at_s=float(payload["inference_completed_at_s"]),
            publication_decision_id=str(payload["publication_decision_id"]),
            publication_decision_at_s=float(payload["publication_decision_at_s"]),
            published_at_s=float(payload["published_at_s"]),
            clock_id=str(payload["clock_id"]),
            publication_decision_locus=str(payload["publication_decision_locus"]),
            inference_placement=str(payload["inference_placement"]),
            publication_action=str(payload["publication_action"]),
            profile_id=str(payload["profile_id"]),
            target_fps=float(payload["target_fps"]),
            model_id=str(payload["model_id"]),
            model_sha256=str(payload["model_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            code_revision=str(payload["code_revision"]),
            source_sensor_ids=tuple(str(item) for item in payload["source_sensor_ids"]),
            calibration_ids=tuple(str(item) for item in payload["calibration_ids"]),
            transport_chunk_bytes=int(payload["transport_chunk_bytes"]),
            chunk_count=int(payload["chunk_count"]),
            application_payload_bytes=int(payload["application_payload_bytes"]),
            objects=tuple(MapObjectObservationV2.from_dict(item) for item in raw_objects),
            schema=str(payload.get("schema", "")),
            operation=str(payload.get("operation", "")),
        )
        result.validate()
        resource_uri = payload.get("resource_uri")
        expected_uri = f"/ss-sm-management/v2/spatial-maps/{result.recipient_ue_id}"
        if resource_uri is not None and str(resource_uri) != expected_uri:
            raise ValueError("resource_uri does not match the named recipient")
        return result


@dataclass(frozen=True)
class RecipientStateV2:
    recipient_ue_id: str
    observed_at_s: float
    available_at_s: float
    clock_id: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    state_covariance: Tuple[float, ...]
    motion_model_id: str
    process_noise_model_id: str
    process_noise_covariance_per_s: Tuple[float, ...]

    def validate(self) -> None:
        _nonempty("recipient_ue_id", self.recipient_ue_id)
        _nonempty("clock_id", self.clock_id)
        _nonempty("motion_model_id", self.motion_model_id)
        _nonempty("process_noise_model_id", self.process_noise_model_id)
        observed = _finite("observed_at_s", self.observed_at_s)
        available = _finite("available_at_s", self.available_at_s)
        if available + 1e-12 < observed:
            raise ValueError("recipient state cannot be available before observation")
        for name in ("x_m", "y_m", "vx_mps", "vy_mps"):
            _finite(name, getattr(self, name))
        _matrix4("state_covariance", self.state_covariance)
        _matrix4(
            "process_noise_covariance_per_s",
            self.process_noise_covariance_per_s,
        )

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["schema"] = RECIPIENT_STATE_SCHEMA_V2
        payload["state_order"] = list(STATE_ORDER)
        payload["state_covariance"] = list(self.state_covariance)
        payload["process_noise_covariance_per_s"] = list(
            self.process_noise_covariance_per_s
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RecipientStateV2":
        _reject_forbidden_runtime_keys(payload, "recipient_state")
        _reject_unknown_keys(
            payload,
            {item.name for item in fields(cls)} | {"schema", "state_order"},
            "recipient_state",
        )
        if payload.get("schema") != RECIPIENT_STATE_SCHEMA_V2:
            raise ValueError("unsupported recipient-state schema")
        if tuple(str(item) for item in payload.get("state_order", ())) != STATE_ORDER:
            raise ValueError(f"state_order must be {STATE_ORDER}")
        result = cls(
            recipient_ue_id=str(payload["recipient_ue_id"]),
            observed_at_s=float(payload["observed_at_s"]),
            available_at_s=float(payload["available_at_s"]),
            clock_id=str(payload["clock_id"]),
            x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]),
            vx_mps=float(payload["vx_mps"]),
            vy_mps=float(payload["vy_mps"]),
            state_covariance=tuple(float(item) for item in payload["state_covariance"]),
            motion_model_id=str(payload["motion_model_id"]),
            process_noise_model_id=str(payload["process_noise_model_id"]),
            process_noise_covariance_per_s=tuple(
                float(item) for item in payload["process_noise_covariance_per_s"]
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class WarningEventV2:
    recipient_ue_id: str
    canonical_track_id: str
    class_name: str
    warning_at_s: float
    time_to_closest_approach_s: float
    closest_approach_m: float
    uncertainty_expanded_closest_approach_m: float
    position_sigma_at_closest_approach_m: float
    map_aoi_s: float
    evidence_sources: Tuple[str, ...]
    evidence_track_ids: Tuple[str, ...]
    evidence_scope: str
    latest_capture_at_s: float
    latest_publish_at_s: float
    motion_model_id: str
    process_noise_model_id: str


def with_exact_payload_bytes_v2(contribution: MapContributionV2) -> MapContributionV2:
    """Resolve application bytes and production-header chunk count together."""

    current = replace(contribution, application_payload_bytes=0, chunk_count=1)
    capacity = int(current.transport_chunk_bytes) - PRODUCTION_CHUNK_HEADER_BYTES
    if capacity <= 0:
        raise ValueError("transport_chunk_bytes must exceed the production header")
    for _ in range(12):
        encoded_bytes = len(current.to_json_bytes())
        chunks = max(1, (encoded_bytes + capacity - 1) // capacity)
        if (
            current.application_payload_bytes == encoded_bytes
            and current.chunk_count == chunks
        ):
            return current
        current = replace(
            current,
            application_payload_bytes=encoded_bytes,
            chunk_count=chunks,
        )
    raise RuntimeError("v2 contribution byte/chunk counts did not converge")
