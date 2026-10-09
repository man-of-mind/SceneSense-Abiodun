"""Binary source handoff: one sealed memfd of aligned raw buffers plus a small descriptor.

Writer (UE side, ``freeze``): copies the already-owned native images, matrices,
radar points and a fixed-layout actor table into aligned buffers and declares
their layout and SHA-256 in a bounded canonical descriptor. No canonical JSON of
the actor population, no JSON float round trip for geometry.

Reader (service side, ``FreezeGate.admit``):
* descriptor: bounded size, strict canonical JSON, exact field set and schema;
* session and ordering: matching session id, sequence exactly last+1 (rejects
  replay, gaps and reordering), frame id strictly greater than the last admitted
  (rejects stale frames); any rejection fails the gate closed;
* memfd: regular file, exact size, all four kernel seals (immutability only);
* layout: known names, exact dtype/shape per name, 64-byte aligned, ordered,
  non-overlapping, in bounds;
* content: SHA-256 of every mapped buffer equals the descriptor (binds content to
  the declared frame; seals alone do not);
* semantics: the unchanged V4 ``_check_metadata`` validation (frame, timestamps,
  geometry, callbacks, actors, population), the V4 camera/inverse/image-pose
  checks, the source-codec population checks and the action-neutral template.
Objects are reconstructed with the unchanged V4 frozen value classes and its
rounding-rejecting CARLA value constructor, so decoded values equal the V4 codec.

Ownership: image buffers are read-only views of the sealed mapping; numeric arrays
are immutable aligned copies (V4 rule). Views must be released before the context
exits; the unchanged V4 ``mapped`` helper fails closed on a leaked view.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import re

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M, snapshot as S
from rl_agent.splitfusion_numeric_gt_runtime_v1 import staging as ST, source_codec as SC

SCHEMA = "splitfusion.exact_gt_service_proto.freeze_descriptor.v1"
ALIGN = 64
MAX_DESCRIPTOR_BYTES = 256 * 1024
FIELDS = frozenset(("schema", "session_id", "sequence", "frame_id", "capture_template", "policy_timestamp_s",
                    "rgb", "images", "world", "dynamic_snapshot_evidence", "actor_type_ids", "buffers", "total_bytes"))
BUFFER_FIELDS = frozenset(("name", "dtype", "shape", "offset", "nbytes", "sha256"))
ACTOR_VECTORS = ("actor_pose_location", "actor_pose_rotation", "actor_location", "actor_velocity",
                 "actor_bbox_location", "actor_bbox_extent", "actor_bbox_rotation")
NAMES = (*(f"image_{i}" for i in range(3)), *(f"image_pose_{i}" for i in range(3)), "camera_matrix",
         "camera_inverse", "radar", "actor_ids", *ACTOR_VECTORS, "actor_poses")
SESSION_RE = re.compile(r"[0-9a-f]{64}")


class FreezeError(ValueError):
    pass


def require(value, message):
    if not value:
        raise FreezeError(message)


def _pad(offset):
    return (-offset) % ALIGN


def freeze(sourcekwargs, policy_timestamp_s, *, session_id, sequence, limits=S.DEFAULT_LIMITS):
    """UE-side: owned source -> (descriptor bytes, ordered chunks for one sealed memfd)."""
    require(type(sourcekwargs) is dict and set(sourcekwargs) == set(SC.SOURCE_KEYS), "owned source inventory differs")
    require(type(session_id) is str and SESSION_RE.fullmatch(session_id) and type(sequence) is int and sequence > 0,
            "session id and positive sequence required")
    stamp = SC._timestamp(policy_timestamp_s)
    template = C._json_owned(sourcekwargs["identity"])
    C._capture(template)
    require(template == ST.neutral_identity(template), "freeze carries only the action-neutral capture template")
    population = C._json_owned(sourcekwargs["dynamic_snapshot_evidence"])
    world = sourcekwargs["frozen_world"]
    require(C.canonical(population) == C.canonical(world.population_evidence), "source/world population differs")
    S._integer(world.frozen_frame_id, "frozen frame")
    actors = list(world.get_actors())
    require(len(actors) <= limits.max_actors, "actor bound exceeded")
    rgb = dict(sourcekwargs["rgb"])
    images = [rgb.pop("_rgb_image"), sourcekwargs["semantic_image"], sourcekwargs["depth_image"]]
    arrays, image_meta = [], []
    for index, image in enumerate(images):
        width, height = image.width, image.height
        require(type(width) is int and type(height) is int and 0 < width <= limits.max_width
                and 0 < height <= limits.max_height, "image dimensions exceed bound")
        require(isinstance(image.raw_data, (bytes, bytearray, memoryview)) and len(image.raw_data) == width * height * 4,
                "native BGRA byte size mismatch")
        image_meta.append(dict(frame=image.frame, timestamp=image.timestamp, width=width, height=height,
                               pose=S._pose(image.transform)))
        arrays.append((f"image_{index}", np.frombuffer(image.raw_data, dtype=np.uint8).reshape(height, width, 4)))
    for index, image in enumerate(images):
        arrays.append((f"image_pose_{index}", np.stack([S._matrix(image.transform.get_matrix()),
                                                        S._matrix(image.transform.get_inverse_matrix())])))
    for key in ("camera_matrix", "camera_inverse"):
        arrays.append((key, S._native_layout(S._numeric(sourcekwargs[key], (4, 4)))))
    radar = S._numeric(sourcekwargs["radar_world_xyz"])
    require(radar.ndim == 2 and radar.shape[1] == 3 and radar.shape[0] <= limits.max_radar_points, "radar shape/bound")
    arrays.append(("radar", S._native_layout(radar)))
    ids, type_ids, vectors, matrices = [], [], {name: [] for name in ACTOR_VECTORS}, []
    for actor in actors:
        transform, bbox = actor.get_transform(), actor.bounding_box
        pose = S._pose(transform)
        require(pose["location"] is not None and pose["rotation"] is not None, "actor pose components required")
        ids.append(S._integer(actor.id, "actor ID"))
        type_ids.append(actor.type_id)
        vectors["actor_pose_location"].append(pose["location"])
        vectors["actor_pose_rotation"].append(pose["rotation"])
        vectors["actor_location"].append(S._components(actor.get_location()))
        vectors["actor_velocity"].append(S._components(actor.get_velocity()))
        vectors["actor_bbox_location"].append(S._components(bbox.location))
        vectors["actor_bbox_extent"].append(S._components(bbox.extent))
        vectors["actor_bbox_rotation"].append(S._components(bbox.rotation, ("pitch", "yaw", "roll")))
        matrices.append([S._matrix(transform.get_matrix()), S._matrix(transform.get_inverse_matrix())])
    count = len(actors)
    arrays.append(("actor_ids", np.asarray(ids, dtype=np.int64).reshape(count)))
    for name in ACTOR_VECTORS:
        arrays.append((name, np.asarray(vectors[name], dtype=np.float64).reshape(count, 3)))
    arrays.append(("actor_poses", np.asarray(matrices, dtype=np.float64).reshape(count, 2, 4, 4)))
    chunks, table, offset = [], [], 0
    for name, array in arrays:
        array = np.ascontiguousarray(array)
        require(array.dtype.kind in "fiu" and np.isfinite(array).all(), "finite numeric buffer required: " + name)
        pad = _pad(offset)
        if pad:
            chunks.append(bytes(pad))
            offset += pad
        raw = memoryview(array).cast("B")
        table.append(dict(name=name, dtype=array.dtype.str, shape=list(array.shape), offset=offset,
                          nbytes=array.nbytes, sha256=hashlib.sha256(raw).hexdigest()))
        chunks.append(raw)
        offset += array.nbytes
    descriptor = dict(schema=SCHEMA, session_id=session_id, sequence=sequence, frame_id=template["frame_id"],
        capture_template=template, policy_timestamp_s=stamp, rgb=rgb, images=image_meta,
        world=dict(frame_id=world.frozen_frame_id, timestamp_s=world.frozen_timestamp_s,
                   observed_raw_ns=world.frozen_observed_raw_ns),
        dynamic_snapshot_evidence=population, actor_type_ids=type_ids, buffers=table, total_bytes=offset)
    encoded = C.canonical(descriptor)
    require(len(encoded) <= MAX_DESCRIPTOR_BYTES and 0 < offset <= M.MAX_BYTES and len(chunks) <= 64,
            "descriptor or sealed source bound exceeded")
    return encoded, chunks


def seal(chunks):
    """Create the sealed memfd (unchanged V4 transport). Caller owns and closes the fd."""
    return M.create(chunks)


def _strict_json(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_DESCRIPTOR_BYTES, "bounded descriptor bytes required")
    try:
        value = json.loads(raw, object_pairs_hook=S._json_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(FreezeError("nonfinite JSON")))
    except (ValueError, S.SnapshotError) as exc:
        raise FreezeError("malformed descriptor JSON: " + str(exc)) from exc
    require(type(value) is dict and C.canonical(value) == raw, "canonical descriptor JSON required")
    return value


def _shapes(descriptor):
    rgb = descriptor["rgb"]
    width, height = rgb.get("width"), rgb.get("height")
    count = len(descriptor["actor_type_ids"])
    shapes = {**{f"image_{i}": ((height, width, 4), "|u1") for i in range(3)},
              **{f"image_pose_{i}": ((2, 4, 4), "<f8") for i in range(3)},
              "camera_matrix": ((4, 4), "<f8"), "camera_inverse": ((4, 4), "<f8"),
              "actor_ids": ((count,), "<i8"), "actor_poses": ((count, 2, 4, 4), "<f8"),
              **{name: ((count, 3), "<f8") for name in ACTOR_VECTORS}}
    return shapes, count


class Admitted:
    """Borrowed decoded source; valid only inside the admit() context."""

    __slots__ = ("descriptor", "kwargs")

    def __init__(self, descriptor, kwargs):
        self.descriptor, self.kwargs = descriptor, kwargs


class FreezeGate:
    """Service-side admission for one UE session; fails closed on any rejection."""

    def __init__(self, session_id, *, limits=S.DEFAULT_LIMITS):
        require(type(session_id) is str and SESSION_RE.fullmatch(session_id), "session id required")
        self.session_id, self.limits = session_id, limits
        self.last_sequence, self.last_frame = 0, -1
        self.failure = None
        self.admitted = 0

    def _fail(self, error):
        self.failure = self.failure or str(error)[:512]

    @contextmanager
    def admit(self, descriptor_bytes, fd):
        require(self.failure is None, "freeze gate failed closed: " + str(self.failure))
        try:
            descriptor = self._descriptor(descriptor_bytes)
            M.verify(fd, descriptor["total_bytes"])
        except BaseException as error:
            self._fail(error)
            raise
        with M.mapped(fd, descriptor["total_bytes"]) as view:
            try:
                kwargs = self._decode(descriptor, view)
            except BaseException as error:
                self._fail(error)
                raise
            self.last_sequence, self.last_frame = descriptor["sequence"], descriptor["frame_id"]
            self.admitted += 1
            holder = Admitted(descriptor, kwargs)
            del kwargs
            try:
                yield holder
            finally:
                # The consumer must not retain decoded views; the mapping release
                # below raises (fails closed) if any image view is still exported.
                holder.kwargs = None

    def void(self, descriptor_bytes):
        """Coordinator-refused freeze: the same descriptor checks, no memfd, explicit sequence advance.

        Keeps the strict +1 ordering (replay/gap/stale detection) intact when admission
        control refuses a frame. Any malformed or foreign void fails the gate closed.
        """
        require(self.failure is None, "freeze gate failed closed: " + str(self.failure))
        try:
            descriptor = self._descriptor(descriptor_bytes)
        except BaseException as error:
            self._fail(error)
            raise
        self.last_sequence, self.last_frame = descriptor["sequence"], descriptor["frame_id"]
        self.voided = getattr(self, "voided", 0) + 1
        return dict(sequence=descriptor["sequence"], frame_id=descriptor["frame_id"])

    def _descriptor(self, raw):
        d = _strict_json(raw)
        require(set(d) == FIELDS and d["schema"] == SCHEMA, "descriptor fields/schema differ")
        require(d["session_id"] == self.session_id, "foreign session descriptor")
        require(type(d["sequence"]) is int and d["sequence"] == self.last_sequence + 1,
                "replayed, reordered or missing descriptor sequence")
        require(type(d["frame_id"]) is int and d["frame_id"] > self.last_frame, "stale or repeated frame")
        template = C._json_owned(d["capture_template"])
        C._capture(template)
        require(template == ST.neutral_identity(template) and template["frame_id"] == d["frame_id"],
                "capture template is action-bound or belongs to another frame")
        SC._timestamp(d["policy_timestamp_s"])
        require(type(d["rgb"]) is dict and type(d["images"]) is list and type(d["world"]) is dict
                and set(d["world"]) == {"frame_id", "timestamp_s", "observed_raw_ns"}, "metadata structure differs")
        types = d["actor_type_ids"]
        require(type(types) is list and len(types) <= self.limits.max_actors, "actor bound exceeded")
        total = d["total_bytes"]
        require(type(total) is int and 0 < total <= M.MAX_BYTES, "sealed source byte bound")
        shapes, count = _shapes(d)
        table = d["buffers"]
        require(type(table) is list and [b.get("name") if type(b) is dict else None for b in table] == list(NAMES),
                "buffer inventory/order differs")
        offset = 0
        for spec in table:
            require(set(spec) == BUFFER_FIELDS, "buffer descriptor fields differ")
            name, dtype = spec["name"], spec["dtype"]
            require(type(dtype) is str and type(spec["shape"]) is list
                    and all(type(v) is int and v >= 0 for v in spec["shape"]), "buffer dtype/shape type")
            if name == "radar":
                require(dtype in ("<f4", "<f8") and len(spec["shape"]) == 2 and spec["shape"][1] == 3
                        and spec["shape"][0] <= self.limits.max_radar_points, "radar shape/dtype/bound")
            else:
                shape, expected = shapes[name]
                require(tuple(spec["shape"]) == shape and dtype == expected, "buffer shape/dtype differs: " + name)
            nbytes = math.prod(spec["shape"]) * np.dtype(dtype).itemsize
            require(type(spec["offset"]) is int and spec["offset"] % ALIGN == 0
                    and offset <= spec["offset"] < offset + ALIGN and spec["nbytes"] == nbytes
                    and spec["offset"] + nbytes <= total, "buffer extent/alignment/overlap differs: " + name)
            require(type(spec["sha256"]) is str and C.is_digest(spec["sha256"]), "buffer digest required")
            offset = spec["offset"] + nbytes
        require(offset == total, "trailing or missing sealed bytes")
        require(count == len(types), "actor table count differs")
        return d

    def _decode(self, d, view):
        arrays = {}
        for spec in d["buffers"]:
            raw = view[spec["offset"]:spec["offset"] + spec["nbytes"]]
            require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "sealed buffer content differs: " + spec["name"])
            dtype, shape = np.dtype(spec["dtype"]), tuple(spec["shape"])
            if spec["name"].startswith("image_") and not spec["name"].startswith("image_pose_"):
                array = np.frombuffer(raw, dtype=dtype).reshape(shape)
            else:
                array = S._immutable_numeric_copy(raw, dtype, shape)
            require(np.isfinite(array).all(), "nonfinite sealed buffer: " + spec["name"])
            arrays[spec["name"]] = array
        count = len(d["actor_type_ids"])
        ids = arrays["actor_ids"].tolist()
        vec = {name: arrays[name].tolist() for name in ACTOR_VECTORS}
        population = d["dynamic_snapshot_evidence"]
        # Reassemble the exact V4 metadata view and run its unchanged validator.
        meta = dict(schema=S.SCHEMA, source_pins=S.SOURCE_PINS, identity=d["capture_template"], rgb=d["rgb"],
            images=d["images"], policy_timestamp_s=d["policy_timestamp_s"], buffers=[],
            world=dict(d["world"], population_evidence=population),
            actors=[dict(id=ids[i], type_id=d["actor_type_ids"][i],
                         pose=dict(location=vec["actor_pose_location"][i], rotation=vec["actor_pose_rotation"][i]),
                         location=vec["actor_location"][i], velocity=vec["actor_velocity"][i],
                         bbox=dict(location=vec["actor_bbox_location"][i], extent=vec["actor_bbox_extent"][i],
                                   rotation=vec["actor_bbox_rotation"][i])) for i in range(count)])
        S._primitive(meta)
        width, height, _ = S._check_metadata(meta, self.limits)
        camera = arrays["camera_matrix"]
        require(np.allclose(camera, S._matrix(d["rgb"]["camera_matrix"]), rtol=0, atol=1e-5), "RGB camera mismatch")
        require(np.allclose(camera @ arrays["camera_inverse"], np.eye(4), rtol=0, atol=1e-5), "camera inverse mismatch")
        for index in range(3):
            require(np.allclose(arrays[f"image_pose_{index}"][0], camera, rtol=0, atol=1e-5), "native image pose mismatch")
        # Source-codec population and staging template checks (unchanged rules).
        require(type(population) is dict and population.get("frame_id") == d["frame_id"], "population frame differs")
        for key in ("registry_complete", "owned_snapshot_complete", "no_unregistered_spawn"):
            require(population.get(key) is True, "incomplete dynamic population: " + key)
        for key in ("active_population_ids", "registered_population_ids"):
            values = population.get(key)
            require(type(values) is list and len(values) <= self.limits.max_actors
                    and all(type(i) is int and 0 <= i < 2**63 for i in values) and len(set(values)) == len(values),
                    "invalid dynamic population IDs")
        require(set(population["active_population_ids"]) <= set(population["registered_population_ids"]),
                "active actor absent from registered population")
        require(C.is_digest(population.get("population_sha256")), "dynamic population binding missing")
        require(SC._timestamp(d["policy_timestamp_s"]).hex() == SC._timestamp(d["rgb"]["timestamp"]).hex(),
                "policy/RGB timestamp differs")
        import carla  # local value constructors only (same as V4)

        def pose(spec, matrices):
            location = None if spec["location"] is None else S._local_value(carla.Location, spec["location"])
            rotation = None if spec["rotation"] is None else S._local_value(carla.Rotation, spec["rotation"],
                                                                            ("pitch", "yaw", "roll"))
            return S.FrozenTransform(matrices[0], matrices[1], location, rotation)

        images = [S.FrozenImage(spec["frame"], spec["timestamp"], width, height,
                                memoryview(arrays[f"image_{i}"]).cast("B"), pose(spec["pose"], arrays[f"image_pose_{i}"]))
                  for i, spec in enumerate(d["images"])]
        actors = []
        for index, spec in enumerate(meta["actors"]):
            box = spec["bbox"]
            bbox = carla.BoundingBox(S._local_value(carla.Location, box["location"]),
                                     S._local_value(carla.Vector3D, box["extent"]))
            bbox.rotation = S._local_value(carla.Rotation, box["rotation"], ("pitch", "yaw", "roll"))
            actors.append(S.FrozenActor(spec["id"], spec["type_id"], bbox, pose(spec["pose"], arrays["actor_poses"][index]),
                S._local_value(carla.Location, spec["location"]), S._local_value(carla.Vector3D, spec["velocity"])))
        world = d["world"]
        frozen = S.FrozenWorld(S.FrozenActorList(actors), world["frame_id"], world["timestamp_s"],
                               world["observed_raw_ns"], population)
        return dict(identity=C._json_owned(d["capture_template"]), rgb={**d["rgb"], "_rgb_image": images[0]},
                    semantic_image=images[1], depth_image=images[2], frozen_world=frozen, camera_matrix=camera,
                    camera_inverse=arrays["camera_inverse"], radar_world_xyz=arrays["radar"],
                    dynamic_snapshot_evidence=copy.deepcopy(population))
