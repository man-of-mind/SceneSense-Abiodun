"""Field-by-field exact comparison of two decoded producer sources.

Floats are compared by their binary64 hex (sign of zero included), arrays by
dtype, shape, raw bytes, C-contiguity, 16-byte alignment and writeability, and
objects by their concrete frozen/CARLA value types.
"""
import json

import numpy as np

from rl_agent.splitfusion_numeric_gt_runtime_v1 import source_codec as SC


def _hex(value):
    return None if value is None else float(value).hex()


def _vector(value, axes=("x", "y", "z")):
    return None if value is None else (type(value).__name__, tuple(_hex(getattr(value, a)) for a in axes))


def _array(value):
    a = np.asarray(value)
    return (a.dtype.str, a.shape, a.tobytes(order="C"), a.flags.c_contiguous, a.ctypes.data % 16 == 0,
            a.flags.writeable)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _transform(t):
    return (type(t).__name__, _array(t.matrix), _array(t.inverse), _vector(t.location),
            _vector(t.rotation, ("pitch", "yaw", "roll")))


def _image(image):
    return (type(image).__name__, image.frame, _hex(image.timestamp), image.width, image.height,
            bytes(image.raw_data), _transform(image.transform))


def _actor(actor):
    box = actor.bounding_box
    return (type(actor).__name__, actor.id, actor.type_id, _transform(actor.get_transform()),
            _vector(actor.get_location()), _vector(actor.get_velocity()),
            (type(box).__name__, _vector(box.location), _vector(box.extent),
             _vector(box.rotation, ("pitch", "yaw", "roll"))))


def _world(world):
    return (type(world).__name__, type(world.actors).__name__, world.frozen_frame_id,
            _hex(world.frozen_timestamp_s), world.frozen_observed_raw_ns, _json(world.population_evidence))


def compare(a, b):
    """Return {field: bool}; every field must be True for exact equality."""
    out = dict(keys=set(a) == set(b) == set(SC.SOURCE_KEYS))
    out["identity"] = _json(a["identity"]) == _json(b["identity"])
    meta_a = {k: v for k, v in a["rgb"].items() if k != "_rgb_image"}
    meta_b = {k: v for k, v in b["rgb"].items() if k != "_rgb_image"}
    out["rgb_metadata"] = _json(meta_a) == _json(meta_b) and set(a["rgb"]) == set(b["rgb"])
    out["rgb_image"] = _image(a["rgb"]["_rgb_image"]) == _image(b["rgb"]["_rgb_image"])
    out["semantic_image"] = _image(a["semantic_image"]) == _image(b["semantic_image"])
    out["depth_image"] = _image(a["depth_image"]) == _image(b["depth_image"])
    out["camera_matrix"] = _array(a["camera_matrix"]) == _array(b["camera_matrix"])
    out["camera_inverse"] = _array(a["camera_inverse"]) == _array(b["camera_inverse"])
    out["radar_world_xyz"] = _array(a["radar_world_xyz"]) == _array(b["radar_world_xyz"])
    out["dynamic_snapshot_evidence"] = _json(a["dynamic_snapshot_evidence"]) == _json(b["dynamic_snapshot_evidence"])
    out["frozen_world"] = _world(a["frozen_world"]) == _world(b["frozen_world"])
    actors_a, actors_b = list(a["frozen_world"].get_actors()), list(b["frozen_world"].get_actors())
    out["actor_count"] = len(actors_a) == len(actors_b)
    out["actors_in_order"] = out["actor_count"] and all(_actor(x) == _actor(y) for x, y in zip(actors_a, actors_b))
    return out
