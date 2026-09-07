"""Versioned SplitFusion frame context and immutable edge camera registry."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .registry import DispatchContractError


FRAME_CONTEXT_VERSION = 1
MAX_STREAM_ID_BYTES = 255
STATIC_CAMERA_MODEL_SHA256 = (
    "c4a1349b416448169a8ced2808389797dae3822a75d62c0c2f066b6b13657009"
)
STATIC_CAMERA_MOUNT_SHA256 = (
    "d54a19cabb31623d0fdc41956c529a6af12694dd0c174ebe3084481f8f8080d8"
)
STATIC_INTRINSIC_TENSOR_SHA256 = (
    "5de4959514a8c858facd02eedacc8226ed27583284465badd0636ca21b23e031"
)

CAMERA_MODEL_NUMERIC_FIELDS = (
    ("camera_width", "1280"),
    ("camera_height", "720"),
    ("camera_fx", "369.5041722813606"),
    ("camera_fy", "369.5041722813606"),
    ("camera_cx", "640.0"),
    ("camera_cy", "360.0"),
)
CAMERA_MOUNT_NUMERIC_FIELDS = (
    ("ego_camera_relative_x", "1.8"),
    ("ego_camera_relative_y", "0"),
    ("ego_camera_relative_z", "1.55"),
    ("ego_camera_relative_pitch_deg", "-4"),
    ("ego_camera_relative_yaw_deg", "0"),
    ("ego_camera_relative_roll_deg", "0"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchContractError(message)


def _decimal_text(value: object) -> str:
    number = Decimal(str(value))
    if number == 0:
        number = Decimal(0)
    return format(number.normalize(), "f")


def canonical_numeric_sha256(
    schema: str, fields: Sequence[tuple[str, object]]
) -> str:
    """Hash explicitly named numeric values without hashing source row text."""
    document = {
        "schema": str(schema),
        "numeric_fields": [
            [str(name), _decimal_text(value)] for name, value in fields
        ],
    }
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    prefix = f"{str(tensor.dtype)}:{list(tensor.shape)}:".encode("ascii")
    return hashlib.sha256(prefix + tensor.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    def values(self) -> tuple[float, ...]:
        return (self.x, self.y, self.z, self.pitch, self.yaw, self.roll)


@dataclass(frozen=True)
class FrameContextV1:
    stream_id: str
    frame_id: int
    sequence_id: int
    capture_timestamp_ns: int
    ego_world: Pose6D
    camera_model_sha256: str
    camera_mount_sha256: str


def build_frame_context_v1(
    *,
    stream_id: str,
    frame_id: int,
    sequence_id: int,
    capture_timestamp_ns: int,
    ego_world_x: float,
    ego_world_y: float,
    ego_world_z: float,
    ego_world_pitch: float,
    ego_world_yaw: float,
    ego_world_roll: float,
    camera_model_sha256: str = STATIC_CAMERA_MODEL_SHA256,
    camera_mount_sha256: str = STATIC_CAMERA_MOUNT_SHA256,
) -> FrameContextV1:
    """Build the UE context directly from synchronized CARLA transform fields."""
    context = FrameContextV1(
        stream_id=stream_id,
        frame_id=frame_id,
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        ego_world=Pose6D(
            ego_world_x,
            ego_world_y,
            ego_world_z,
            ego_world_pitch,
            ego_world_yaw,
            ego_world_roll,
        ),
        camera_model_sha256=camera_model_sha256,
        camera_mount_sha256=camera_mount_sha256,
    )
    validate_frame_context(context)
    return context


def validate_frame_context(context: FrameContextV1) -> bytes:
    _require(isinstance(context, FrameContextV1), "frame context type is unsupported")
    try:
        stream = context.stream_id.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise DispatchContractError("frame-context stream identity is invalid UTF-8") from exc
    _require(bool(stream), "frame-context stream identity is empty")
    _require(
        len(stream) <= MAX_STREAM_ID_BYTES,
        "frame-context stream identity is oversized",
    )
    for name, value in (
        ("frame_id", context.frame_id),
        ("sequence_id", context.sequence_id),
        ("capture_timestamp_ns", context.capture_timestamp_ns),
    ):
        _require(
            not isinstance(value, bool)
            and isinstance(value, int)
            and 0 <= value < (1 << 64),
            f"frame-context {name} is outside uint64",
        )
    _require(
        all(math.isfinite(value) for value in context.ego_world.values()),
        "frame-context pose is non-finite",
    )
    for name, digest in (
        ("camera hash", context.camera_model_sha256),
        ("mount hash", context.camera_mount_sha256),
    ):
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and digest == digest.lower(),
            f"frame-context {name} is not a full lowercase SHA-256",
        )
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise DispatchContractError(
                f"frame-context {name} is not hexadecimal"
            ) from exc
    return stream


def carla_transform_matrix(pose: Pose6D | Iterable[float]) -> np.ndarray:
    """Pure CARLA Transform.get_matrix convention: x,y,z,pitch,yaw,roll."""
    values = pose.values() if isinstance(pose, Pose6D) else tuple(pose)
    _require(len(values) == 6, "CARLA transform requires six pose values")
    x, y, z, pitch_deg, yaw_deg, roll_deg = (float(value) for value in values)
    _require(
        all(math.isfinite(value) for value in (x, y, z, pitch_deg, yaw_deg, roll_deg)),
        "CARLA transform pose is non-finite",
    )
    pitch, yaw, roll = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
            [sp, -cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class StaticCameraCalibration:
    camera_model_sha256: str
    camera_mount_sha256: str
    source_width: int
    source_height: int
    intrinsic_values: tuple[tuple[float, float, float], ...]
    camera_to_ego_values: tuple[tuple[float, float, float, float], ...]

    def intrinsic(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.intrinsic_values, dtype=torch.float32, device=device)

    def camera_to_ego(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            self.camera_to_ego_values, dtype=torch.float64, device=device
        )


class StaticCameraRegistry:
    """Immutable lookup for edge-resident static camera and mounting data."""

    def __init__(self, entries: Sequence[StaticCameraCalibration]) -> None:
        _require(bool(entries), "static camera registry is empty")
        by_pair = {
            (entry.camera_model_sha256, entry.camera_mount_sha256): entry
            for entry in entries
        }
        _require(len(by_pair) == len(entries), "static camera registry has duplicates")
        self._by_pair: Mapping[
            tuple[str, str], StaticCameraCalibration
        ] = MappingProxyType(by_pair)

    @classmethod
    def audited(cls) -> "StaticCameraRegistry":
        model_hash = canonical_numeric_sha256(
            "scenesense.static_camera_model.v1", CAMERA_MODEL_NUMERIC_FIELDS
        )
        mount_hash = canonical_numeric_sha256(
            "scenesense.static_camera_mount.v1", CAMERA_MOUNT_NUMERIC_FIELDS
        )
        _require(model_hash == STATIC_CAMERA_MODEL_SHA256, "audited camera hash drift")
        _require(mount_hash == STATIC_CAMERA_MOUNT_SHA256, "audited mount hash drift")
        values = {name: float(value) for name, value in CAMERA_MODEL_NUMERIC_FIELDS}
        sx = 768.0 / values["camera_width"]
        sy = 432.0 / values["camera_height"]
        intrinsic = (
            (values["camera_fx"] * sx, 0.0, values["camera_cx"] * sx),
            (0.0, values["camera_fy"] * sy, values["camera_cy"] * sy),
            (0.0, 0.0, 1.0),
        )
        mount = carla_transform_matrix(Pose6D(1.8, 0.0, 1.55, -4.0, 0.0, 0.0))
        entry = StaticCameraCalibration(
            camera_model_sha256=model_hash,
            camera_mount_sha256=mount_hash,
            source_width=1280,
            source_height=720,
            intrinsic_values=tuple(tuple(float(value) for value in row) for row in intrinsic),
            camera_to_ego_values=tuple(tuple(float(value) for value in row) for row in mount),
        )
        _require(
            tensor_sha256(entry.intrinsic(torch.device("cpu")))
            == STATIC_INTRINSIC_TENSOR_SHA256,
            "audited intrinsic tensor hash drift",
        )
        return cls((entry,))

    def resolve(
        self, camera_model_sha256: str, camera_mount_sha256: str
    ) -> StaticCameraCalibration:
        camera_known = any(
            camera_hash == camera_model_sha256
            for camera_hash, _mount_hash in self._by_pair
        )
        _require(camera_known, "unknown static-camera hash")
        mount_known = any(
            mount_hash == camera_mount_sha256
            for _camera_hash, mount_hash in self._by_pair
        )
        _require(mount_known, "unknown static mount hash")
        key = (camera_model_sha256, camera_mount_sha256)
        _require(key in self._by_pair, "static camera/mount pairing is unregistered")
        return self._by_pair[key]

    @property
    def identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._by_pair)


@dataclass
class _StreamState:
    last_sequence_id: int
    last_capture_timestamp_ns: int
    frame_ids: set[int]


class FrameContextSessionValidator:
    """Fail-closed monotonicity and duplicate state for live stream sessions."""

    def __init__(self) -> None:
        self._streams: dict[str, _StreamState] = {}
        self._lock = threading.Lock()

    def accept(self, context: FrameContextV1) -> None:
        validate_frame_context(context)
        with self._lock:
            state = self._streams.get(context.stream_id)
            if state is not None:
                _require(
                    context.frame_id not in state.frame_ids,
                    "duplicate frame context within stream session",
                )
                _require(
                    context.sequence_id > state.last_sequence_id,
                    "non-increasing frame-context sequence",
                )
                _require(
                    context.capture_timestamp_ns > state.last_capture_timestamp_ns,
                    "non-increasing frame-context timestamp",
                )
                state.last_sequence_id = context.sequence_id
                state.last_capture_timestamp_ns = context.capture_timestamp_ns
                state.frame_ids.add(context.frame_id)
            else:
                self._streams[context.stream_id] = _StreamState(
                    last_sequence_id=context.sequence_id,
                    last_capture_timestamp_ns=context.capture_timestamp_ns,
                    frame_ids={context.frame_id},
                )

    def reset(self, stream_id: str | None = None) -> None:
        with self._lock:
            if stream_id is None:
                self._streams.clear()
            else:
                self._streams.pop(stream_id, None)

    @property
    def stream_count(self) -> int:
        with self._lock:
            return len(self._streams)


def camera_world_matrix(
    context: FrameContextV1, calibration: StaticCameraCalibration
) -> np.ndarray:
    validate_frame_context(context)
    ego_world = carla_transform_matrix(context.ego_world)
    camera_to_ego = np.asarray(calibration.camera_to_ego_values, dtype=np.float64)
    result = ego_world @ camera_to_ego
    _require(
        result.shape == (4, 4) and bool(np.isfinite(result).all()),
        "reconstructed camera-to-world matrix is invalid",
    )
    return result
