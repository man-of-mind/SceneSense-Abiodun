"""V4 chunked transport for the byte-identical, exact V3 GT snapshot.

``pack_chunks(kwargs, radar_world_xyz)`` returns immutable ordered wire chunks;
``pack_snapshot`` flattens them for compatibility. ``unpack_snapshot`` accepts
bytes or a read-only contiguous memoryview (e.g. a sealed read-only mmap).
The caller must retain the mapped owner until all decoded image views expire.
Mapped inputs require an externally verified immutable owner, not merely a
read-only view over mutable storage. All functions perform no file I/O,
RPC, model import, or CUDA operation. ``sync_evidence`` is deliberately omitted:
it is a persistent child-owned object, not a snapshot. The worker must enforce
request ordering, one outstanding request, ownership, and joint shutdown; this
stateless codec grants no live authority and cannot supply those guarantees.

Wire format: magic, big-endian uint64 JSON length, SHA256(JSON + binary), strict
JSON metadata, and contiguous numeric buffers. No pickle/object arrays/base64.
Limits apply before binary allocation/reconstruction. Camera/radar inputs must
have the retained native C layout and 16-byte alignment. Decoded numeric arrays
receive individual immutable bytes backing with an explicit 16-byte alignment
check; packet-offset views can change NumPy matvec rounding even when their
values, bytes, and dtype-level ALIGNED flag are identical (saved frame 1470).
These copies are child-owned and must be included in the worker memory budget.
All actor matrices AND native inverse matrices are
transported, never recomputed. Reconstructed CARLA values use the original
float components: in particular ``get_location().distance(...)`` remains the
original C++ float arithmetic, not a NumPy/Python substitute. Only local value
constructors are used; CARLA is imported lazily at decode, never at import/pack.

Geometry binding describes the unchanged consumer, not an executable payload.
The worker must verify these source pins independently before starting. JSON
float roundtrip is binary64 exact; the decoder rejects any component that a
CARLA value constructor would round, rather than silently relaxing parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import math
import struct

import numpy as np

SCHEMA = "splitfusion.gt_v3.primitive_snapshot.v1"
MAGIC = b"SFGTSNP3"
HEADER_SIZE = 48
SOURCE_PINS = {
    "carla_collect_parked_ego_fusion_training_data.py":
        "d9cb8c8357269c9cbb2ad86d492fec9fb2966c838135889ce7946b926c11d836",
    "rl_agent/splitfusion_live_parity_repair_v1/source_geometry.py":
        "2a9d2251fd9c2173fa85825f302cdb731e921b086f4aae969bce014d8a6d906e",
}
_FIELDS = {"identity", "rgb", "semantic_image", "depth_image", "frozen_world",
           "camera_matrix", "camera_inverse", "policy_timestamp_s"}


class SnapshotError(ValueError):
    pass


def _require(value, message):
    if not value:
        raise SnapshotError(message)


@dataclass(frozen=True)
class Limits:
    max_payload_bytes: int = 64 * 1024 * 1024
    max_metadata_bytes: int = 16 * 1024 * 1024
    max_actors: int = 10000
    max_radar_points: int = 100000
    max_width: int = 1280
    max_height: int = 720

    def __post_init__(self):
        hard = (64 * 1024 * 1024, 16 * 1024 * 1024, 10000, 100000, 1280, 720)
        for value, maximum in zip(self.__dict__.values(), hard):
            _require(type(value) is int and 0 < value <= maximum, "invalid snapshot limit")


DEFAULT_LIMITS = Limits()


def _primitive(value, depth=0):
    _require(depth <= 24, "metadata nesting limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        _require(-(2**63) <= value < 2**64, "metadata integer limit")
        return value
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite metadata")
        return value
    if type(value) is str:
        _require(len(value) <= 1024 * 1024, "metadata string limit")
        return value
    if type(value) is list:
        _require(len(value) <= 100000, "metadata list limit")
        return [_primitive(item, depth + 1) for item in value]
    if type(value) is dict:
        _require(len(value) <= 100000 and all(type(k) is str for k in value), "metadata dictionary keys")
        return {k: _primitive(v, depth + 1) for k, v in value.items()}
    raise SnapshotError("nonprimitive metadata value: " + type(value).__name__)


def _canonical(value):
    return json.dumps(_primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _integer(value, name):
    _require(type(value) is int and 0 <= value < 2**63, "invalid " + name)
    return value


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), "invalid " + name)
    return float(value)


def _components(value, axes=("x", "y", "z")):
    result = [float(getattr(value, axis)) for axis in axes]
    _require(all(math.isfinite(v) for v in result), "nonfinite geometry component")
    return result


def _matrix(value):
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == (4, 4) and np.isfinite(result).all(), "invalid geometry matrix")
    return result


def _numeric(value, shape=None):
    result = np.asarray(value)
    _require(result.dtype.kind in "fiu" and result.dtype.itemsize in (1, 2, 4, 8), "invalid numeric dtype")
    _require(result.ndim <= 4 and (shape is None or result.shape == shape), "invalid numeric shape")
    _require(np.isfinite(result).all(), "nonfinite numeric array")
    return result


def _native_layout(array):
    # np.flags.aligned alone permits address%16 == 8 for float64. That is not
    # the retained np.asarray(list)/np.linalg.inv/owned-radar allocation path.
    _require(array.flags.c_contiguous and array.ctypes.data % 16 == 0,
             "unsupported native numeric layout/alignment")
    return array


def _immutable_numeric_copy(raw, dtype, shape):
    # Always materialize from a memoryview, not bytes(existing_bytes), which
    # could reuse the packet allocation. This CPython bytes allocation is
    # checked, never assumed, to have the retained 16-byte alignment. Fail
    # closed on another allocator/platform rather than use changed arithmetic.
    owned = bytes(memoryview(raw))
    array = np.frombuffer(owned, dtype=dtype).reshape(shape)
    _native_layout(array)
    _require(not array.flags.writeable, "decoded numeric buffer is writable")
    return array


def _pose(transform):
    # Ownership FrozenTransform carries only a matrix; native actor transforms
    # additionally carry location/rotation. Do not infer missing pose components.
    return {"location": _components(transform.location) if getattr(transform, "location", None) is not None else None,
            "rotation": _components(transform.rotation, ("pitch", "yaw", "roll"))
            if getattr(transform, "rotation", None) is not None else None}


def _check_metadata(meta, limits):
    _require(type(meta) is dict and set(meta) == {"schema", "source_pins", "identity", "rgb", "images",
        "world", "actors", "policy_timestamp_s", "buffers"}, "snapshot metadata fields")
    _require(meta["schema"] == SCHEMA and meta["source_pins"] == SOURCE_PINS, "snapshot schema/source pin mismatch")
    identity = meta["identity"]
    _require(type(identity) is dict and "frame_id" in identity, "missing frame identity")
    frame = _integer(identity["frame_id"], "identity frame")
    rgb = meta["rgb"]
    _require(type(rgb) is dict and type(rgb.get("frame_id")) is int and rgb["frame_id"] == frame, "RGB frame mismatch")
    stamp = _number(rgb.get("timestamp"), "RGB timestamp")
    _require(_number(meta["policy_timestamp_s"], "policy timestamp") == stamp, "policy timestamp mismatch")
    width, height = rgb.get("width"), rgb.get("height")
    _require(type(width) is int and type(height) is int and 0 < width <= limits.max_width
             and 0 < height <= limits.max_height, "image dimensions exceed bound")
    camera = _matrix(rgb["camera_matrix"])
    images = meta["images"]
    _require(type(images) is list and len(images) == 3, "exactly three native images required")
    for image in images:
        _require(type(image) is dict and set(image) == {"frame", "timestamp", "width", "height", "pose"}, "image metadata fields")
        _require(type(image["frame"]) is int and image["frame"] == frame, "image frame mismatch")
        _require(_number(image["timestamp"], "image timestamp") == stamp, "image timestamp mismatch")
        _require((image["width"], image["height"]) == (width, height), "native image geometry mismatch")
    world = meta["world"]
    _require(type(world) is dict and set(world) == {"frame_id", "timestamp_s", "observed_raw_ns", "population_evidence"}, "world metadata fields")
    _require(type(world["frame_id"]) is int and world["frame_id"] == frame, "world frame mismatch")
    _require(_number(world["timestamp_s"], "world timestamp") == stamp, "world timestamp mismatch")
    _integer(world["observed_raw_ns"], "snapshot observation")
    population = world["population_evidence"]
    _require(type(population) is dict and population.get("frame_id") == frame, "population frame mismatch")
    actors = meta["actors"]
    _require(type(actors) is list and len(actors) <= limits.max_actors, "actor bound exceeded")
    ids = []
    for actor in actors:
        _require(type(actor) is dict and set(actor) == {"id", "type_id", "pose", "location", "velocity", "bbox"}, "actor metadata fields")
        ids.append(_integer(actor["id"], "actor ID"))
        _require(type(actor["type_id"]) is str and 0 < len(actor["type_id"]) <= 512, "invalid actor type")
        _require(type(actor["bbox"]) is dict and set(actor["bbox"]) == {"location", "extent", "rotation"}, "bbox metadata fields")
        _require(all(v >= 0 for v in _vector(actor["bbox"]["extent"])), "negative bbox extent")
        for vector in (actor["location"], actor["velocity"], actor["bbox"]["location"], actor["bbox"]["rotation"]):
            _vector(vector)
        _check_pose(actor["pose"], required=True)
    _require(len(ids) == len(set(ids)), "duplicate actor ID")
    # Keep original order: never sort actors or historical population records.
    active, registered = population.get("active_population_ids"), population.get("registered_population_ids")
    _require(type(active) is list and active == ids, "active actor inventory mismatch")
    _require(type(registered) is list and len(registered) <= limits.max_actors
             and all(type(v) is int and v >= 0 for v in registered)
             and len(registered) == len(set(registered)) and set(ids) <= set(registered), "registered population mismatch")
    for image in images:
        _check_pose(image["pose"])
    for role in ("semantic_callback", "depth_callback", "radar_callback"):
        callback = rgb.get(role)
        _require(type(callback) is dict and type(callback.get("frame_id")) is int and callback["frame_id"] == frame
                 and _number(callback.get("timestamp"), role + " timestamp") == stamp, "callback frame/timestamp mismatch")
        callback_matrix = _matrix(callback["camera_matrix"])
        if role != "radar_callback":
            _require((callback.get("width"), callback.get("height")) == (width, height), "callback image geometry mismatch")
            _require(np.allclose(callback_matrix, camera, rtol=0, atol=1e-5), "callback image pose mismatch")
    return width, height, len(actors)


def _vector(value):
    _require(type(value) is list and len(value) == 3, "invalid geometry vector")
    return [_number(v, "geometry component") for v in value]


def _check_pose(pose, required=False):
    _require(type(pose) is dict and set(pose) == {"location", "rotation"}, "pose metadata fields")
    for vector in pose.values():
        if vector is not None or required:
            _vector(vector)


def pack_chunks(kwargs, radar_world_xyz, *, limits=DEFAULT_LIMITS):
    """Copy one immutable job; ignore only the intentionally child-owned sync object."""
    try:
        _require(type(limits) is Limits, "invalid limits object")
        _require(type(kwargs) is dict and _FIELDS <= set(kwargs)
                 and set(kwargs) <= _FIELDS | {"sync_evidence"}, "complete-GT keyword inventory mismatch")
        world = kwargs["frozen_world"]
        # Require a snapshot header before invoking the frozen local accessor.
        _integer(world.frozen_frame_id, "frozen frame")
        actor_view = world.get_actors()
        _require(len(actor_view) <= limits.max_actors, "actor bound exceeded")
        actors = list(actor_view)
        rgb = dict(kwargs["rgb"])
        images = [rgb.pop("_rgb_image"), kwargs["semantic_image"], kwargs["depth_image"]]
        metadata = {"schema": SCHEMA, "source_pins": SOURCE_PINS, "identity": kwargs["identity"], "rgb": rgb,
            "policy_timestamp_s": float(kwargs["policy_timestamp_s"]), "buffers": [], "images": [], "actors": [],
            "world": {"frame_id": world.frozen_frame_id, "timestamp_s": world.frozen_timestamp_s,
                "observed_raw_ns": world.frozen_observed_raw_ns, "population_evidence": world.population_evidence}}
        buffers = []
        offset = 0

        def add(name, value):
            nonlocal offset
            array = _numeric(value)
            _require(offset + array.nbytes <= limits.max_payload_bytes - HEADER_SIZE, "snapshot binary bound exceeded")
            metadata["buffers"].append(dict(name=name, dtype=array.dtype.str, shape=list(array.shape), offset=offset, nbytes=array.nbytes))
            # Only share buffers with a proven immutable bytes owner. A
            # read-only view of a bytearray/ndarray is not an immutable owner.
            owner = array
            seen = set()
            while isinstance(owner, (np.ndarray, memoryview)) and id(owner) not in seen:
                seen.add(id(owner))
                owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
            if type(owner) is bytes and array.flags.c_contiguous:
                raw = memoryview(array).cast("B")
                _require(raw.readonly, "immutable owner exposed a writable view")
            else:
                raw = memoryview(array.tobytes(order="C"))
            buffers.append(raw)
            offset += array.nbytes

        for index, image in enumerate(images):
            width, height = image.width, image.height
            _require(type(width) is int and type(height) is int and 0 < width <= limits.max_width
                     and 0 < height <= limits.max_height, "image dimensions exceed bound")
            _require(isinstance(image.raw_data, (bytes, bytearray, memoryview)) and len(image.raw_data) == width * height * 4, "native BGRA byte size mismatch")
            metadata["images"].append(dict(frame=image.frame, timestamp=image.timestamp, width=width, height=height, pose=_pose(image.transform)))
            add("image_" + str(index), np.frombuffer(image.raw_data, dtype=np.uint8).reshape(height, width, 4))
            add("image_pose_" + str(index), np.stack([_matrix(image.transform.get_matrix()), _matrix(image.transform.get_inverse_matrix())]))
        for key in ("camera_matrix", "camera_inverse"):
            add(key, _native_layout(_numeric(kwargs[key], (4, 4))))
        radar = _numeric(radar_world_xyz)
        _require(radar.ndim == 2 and radar.shape[1] == 3 and radar.shape[0] <= limits.max_radar_points, "radar shape/bound mismatch")
        add("radar", _native_layout(radar))
        matrices = []
        for actor in actors:
            transform, bbox = actor.get_transform(), actor.bounding_box
            metadata["actors"].append(dict(id=actor.id, type_id=actor.type_id, pose=_pose(transform),
                location=_components(actor.get_location()), velocity=_components(actor.get_velocity()),
                bbox=dict(location=_components(bbox.location), extent=_components(bbox.extent),
                          rotation=_components(bbox.rotation, ("pitch", "yaw", "roll")))))
            matrices.append([_matrix(transform.get_matrix()), _matrix(transform.get_inverse_matrix())])
        add("actor_poses", np.asarray(matrices, dtype=np.float64).reshape(len(actors), 2, 4, 4))
        _check_metadata(metadata, limits)
        camera, inverse = _numeric(kwargs["camera_matrix"], (4, 4)), _numeric(kwargs["camera_inverse"], (4, 4))
        _require(np.allclose(camera, _matrix(rgb["camera_matrix"]), rtol=0, atol=1e-5), "RGB camera mismatch")
        _require(np.allclose(camera @ inverse, np.eye(4), rtol=0, atol=1e-5), "camera inverse mismatch")
        raw_json = _canonical(metadata)
        _require(len(raw_json) <= limits.max_metadata_bytes and HEADER_SIZE + len(raw_json) + offset <= limits.max_payload_bytes, "snapshot metadata/payload bound exceeded")
        digest = hashlib.sha256(raw_json)
        for raw in buffers:
            digest.update(raw)
        header = MAGIC + struct.pack(">Q", len(raw_json)) + digest.digest()
        return (header, raw_json, *buffers)
    except SnapshotError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise SnapshotError("invalid snapshot source: " + str(exc)) from exc


def pack_snapshot(kwargs, radar_world_xyz, *, limits=DEFAULT_LIMITS):
    """Compatibility helper; the descriptor transport consumes chunks directly."""
    return b"".join(pack_chunks(kwargs, radar_world_xyz, limits=limits))


@dataclass(frozen=True)
class FrozenTransform:
    matrix: np.ndarray
    inverse: np.ndarray
    location: object
    rotation: object

    def get_matrix(self):
        return self.matrix.tolist()

    def get_inverse_matrix(self):
        return self.inverse.tolist()


@dataclass(frozen=True)
class FrozenImage:
    frame: int
    timestamp: float
    width: int
    height: int
    raw_data: memoryview
    transform: FrozenTransform


@dataclass(frozen=True)
class FrozenActor:
    id: int
    type_id: str
    bounding_box: object
    _transform: FrozenTransform
    _location: object
    _velocity: object

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity


class FrozenActorList(tuple):
    def filter(self, pattern):
        return tuple(actor for actor in self if fnmatch.fnmatch(actor.type_id, str(pattern)))

    def find(self, actor_id):
        return next((actor for actor in self if actor.id == actor_id), None)


@dataclass(frozen=True)
class FrozenWorld:
    actors: FrozenActorList
    frozen_frame_id: int
    frozen_timestamp_s: float
    frozen_observed_raw_ns: int
    population_evidence: dict

    def get_actors(self):
        return self.actors


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _local_value(constructor, values, axes=("x", "y", "z")):
    components = _vector(values)
    value = constructor(**dict(zip(axes, components)))
    # Detect rounding including the sign of zero, rather than silently changing
    # a historical fake's binary64 coordinate into CARLA's binary32 coordinate.
    _require(all(float(getattr(value, axis)).hex() == component.hex()
                 for axis, component in zip(axes, components)), "CARLA reconstruction rounded geometry")
    return value


def unpack_snapshot(payload, *, limits=DEFAULT_LIMITS):
    """Validate the entire envelope before invoking local CARLA constructors."""
    try:
        _require(type(limits) is Limits, "invalid limits object")
        _require(type(payload) in (bytes, memoryview), "snapshot requires bytes or read-only memoryview")
        payload = memoryview(payload)
        _require(payload.readonly and payload.c_contiguous, "snapshot view is writable or noncontiguous")
        payload = payload.cast("B")
        _require(HEADER_SIZE <= len(payload) <= limits.max_payload_bytes, "snapshot payload byte bound")
        _require(payload[:8] == MAGIC, "snapshot magic mismatch")
        size = struct.unpack(">Q", payload[8:16])[0]
        _require(0 < size <= limits.max_metadata_bytes and HEADER_SIZE + size <= len(payload), "snapshot metadata byte bound")
        body = memoryview(payload)[HEADER_SIZE:]
        _require(hashlib.sha256(body).digest() == payload[16:48], "snapshot digest mismatch")
        meta = json.loads(bytes(body[:size]), object_pairs_hook=_json_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(SnapshotError("nonfinite JSON constant")))
        _primitive(meta)
        width, height, actor_count = _check_metadata(meta, limits)
        binary = body[size:]
        shapes = {**{"image_" + str(i): (height, width, 4) for i in range(3)},
            **{"image_pose_" + str(i): (2, 4, 4) for i in range(3)},
            "camera_matrix": (4, 4), "camera_inverse": (4, 4), "actor_poses": (actor_count, 2, 4, 4)}
        arrays = {}
        offset = 0
        _require(type(meta["buffers"]) is list and len(meta["buffers"]) == 10, "binary buffer inventory")
        for spec in meta["buffers"]:
            _require(type(spec) is dict and set(spec) == {"name", "dtype", "shape", "offset", "nbytes"}, "buffer metadata fields")
            name = spec["name"]
            _require(type(name) is str and name in set(shapes) | {"radar"} and name not in arrays, "duplicate/unknown binary buffer")
            _require(type(spec["dtype"]) is str and len(spec["dtype"]) <= 4, "invalid buffer dtype")
            dtype = np.dtype(spec["dtype"])
            _require(dtype.kind in "fiu" and dtype.itemsize in (1, 2, 4, 8), "invalid buffer dtype")
            shape = spec["shape"]
            _require(type(shape) is list and len(shape) <= 4 and all(type(v) is int and v >= 0 for v in shape), "invalid buffer shape")
            if name == "radar":
                _require(len(shape) == 2 and shape[1] == 3 and shape[0] <= limits.max_radar_points, "radar shape/bound mismatch")
            else:
                _require(tuple(shape) == shapes[name], "binary buffer shape mismatch")
            if name.startswith("image_") and not name.startswith("image_pose_"):
                _require(dtype == np.dtype("uint8"), "image dtype mismatch")
            if name.startswith("image_pose_") or name == "actor_poses":
                _require(dtype == np.dtype("float64"), "geometry matrix dtype mismatch")
            nbytes = math.prod(shape) * dtype.itemsize
            _require(type(spec["offset"]) is int and type(spec["nbytes"]) is int
                     and spec["offset"] == offset and spec["nbytes"] == nbytes
                     and offset + nbytes <= len(binary), "binary buffer extent mismatch")
            raw = binary[offset:offset+nbytes]
            is_image = name.startswith("image_") and not name.startswith("image_pose_")
            array = (np.frombuffer(raw, dtype=dtype).reshape(shape) if is_image else
                     _immutable_numeric_copy(raw, dtype, shape))
            _require(np.isfinite(array).all(), "nonfinite binary array")
            arrays[name] = array
            offset += nbytes
        _require(offset == len(binary) and set(arrays) == set(shapes) | {"radar"}, "trailing/missing binary payload")
        camera = arrays["camera_matrix"]
        _require(np.allclose(camera, _matrix(meta["rgb"]["camera_matrix"]), rtol=0, atol=1e-5), "RGB camera mismatch")
        _require(np.allclose(camera @ arrays["camera_inverse"], np.eye(4), rtol=0, atol=1e-5), "camera inverse mismatch")
        for index in range(3):
            _require(np.allclose(arrays["image_pose_" + str(index)][0], camera, rtol=0, atol=1e-5), "native image pose mismatch")
        import carla  # local value constructors only; never Client/World/Actor RPC

        def pose(spec, matrices):
            location = None if spec["location"] is None else _local_value(carla.Location, spec["location"])
            rotation = None if spec["rotation"] is None else _local_value(carla.Rotation, spec["rotation"], ("pitch", "yaw", "roll"))
            return FrozenTransform(matrices[0], matrices[1], location, rotation)

        images = []
        for index, spec in enumerate(meta["images"]):
            images.append(FrozenImage(spec["frame"], spec["timestamp"], width, height,
                memoryview(arrays["image_" + str(index)]).cast("B"), pose(spec["pose"], arrays["image_pose_" + str(index)])))
        actors = []
        for index, spec in enumerate(meta["actors"]):
            box = spec["bbox"]
            bbox = carla.BoundingBox(_local_value(carla.Location, box["location"]), _local_value(carla.Vector3D, box["extent"]))
            bbox.rotation = _local_value(carla.Rotation, box["rotation"], ("pitch", "yaw", "roll"))
            actors.append(FrozenActor(spec["id"], spec["type_id"], bbox, pose(spec["pose"], arrays["actor_poses"][index]),
                _local_value(carla.Location, spec["location"]), _local_value(carla.Vector3D, spec["velocity"])))
        world = meta["world"]
        frozen = FrozenWorld(FrozenActorList(actors), world["frame_id"], world["timestamp_s"], world["observed_raw_ns"], world["population_evidence"])
        kwargs = dict(identity=meta["identity"], rgb={**meta["rgb"], "_rgb_image": images[0]}, semantic_image=images[1],
            depth_image=images[2], frozen_world=frozen, camera_matrix=camera, camera_inverse=arrays["camera_inverse"],
            policy_timestamp_s=meta["policy_timestamp_s"])
        return kwargs, arrays["radar"]
    except SnapshotError:
        raise
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SnapshotError("invalid snapshot payload: " + str(exc)) from exc
