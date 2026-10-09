"""Coherent high-coordinate fixtures for the real GT prep and score workers."""
from __future__ import annotations

import dataclasses
import hashlib
import math
from types import SimpleNamespace as NS

import numpy as np

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as S
from rl_agent.splitfusion_live_parity_runtime_v1.pose import context_camera


ARCHIVED_V2_EGO = (-92.44287109375, 129.01821899414062, 0.05807912349700928,
                   0.01369752548635006, 42.05836868286133, -0.12117782235145569)
ARCHIVED_V2_CAMERA = np.asarray([
    [0.740567684173584, -0.6698862314224243, 0.053027842193841934, -91.10452270507812],
    [0.6683762073516846, 0.7424607872962952, 0.0450027696788311, 130.22134399414062],
    [-0.0695178285241127, 0.0021149504464119673, 0.9975784420967102, 1.6085058450698853],
    [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
ARCHIVED_V2_DELTA_M = 1.0009130875232586e-05
PLANNED_TRANSLATION_Y_M = 10.8


def _f32(value):
    return float(np.float32(value))


def _planned_camera():
    value = np.array(ARCHIVED_V2_CAMERA, copy=True)
    value[1, 3] = _f32(value[1, 3] + PLANNED_TRANSLATION_Y_M)
    return value


def _pose(matrix):
    import carla
    matrix = np.asarray(matrix, dtype=np.float64)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(matrix[2, 0])))))
    yaw = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    roll = math.degrees(math.atan2(float(-matrix[2, 1]), float(matrix[2, 2])))
    location = carla.Location(x=_f32(matrix[0, 3]), y=_f32(matrix[1, 3]), z=_f32(matrix[2, 3]))
    rotation = carla.Rotation(pitch=_f32(pitch), yaw=_f32(yaw), roll=_f32(roll))
    return location, rotation


def _transform(matrix, world_change):
    value = np.asarray(world_change @ np.asarray(matrix, dtype=np.float64), dtype=np.float64)
    # Match the CARLA world-coordinate storage used by the captured source.
    value[:3, 3] = np.asarray([_f32(x) for x in value[:3, 3]], dtype=np.float64)
    value[3] = (0.0, 0.0, 0.0, 1.0)
    return np.ascontiguousarray(value)


def _frozen_transform(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    return S.FrozenTransform(matrix=matrix, inverse=np.ascontiguousarray(np.linalg.inv(matrix)),
                             location=_pose(matrix)[0], rotation=_pose(matrix)[1])


def transform_kwargs(kwargs, desired_camera):
    """Apply one world transform to sensors, actors and radar; images remain camera-relative."""
    import carla
    base = np.asarray(kwargs["camera_matrix"], dtype=np.float64)
    desired = np.ascontiguousarray(desired_camera, dtype=np.float64)
    world_change = desired @ np.linalg.inv(base)

    def image(value, matrix=None):
        matrix = desired if matrix is None else np.asarray(matrix, dtype=np.float64)
        return S.FrozenImage(frame=int(value.frame), timestamp=float(value.timestamp), width=int(value.width),
                             height=int(value.height), raw_data=memoryview(value.raw_data),
                             transform=_frozen_transform(matrix))

    rgb = {k: v for k, v in kwargs["rgb"].items() if k != "_rgb_image"}
    rgb["camera_matrix"] = desired.tolist()
    for role in ("semantic_callback", "depth_callback", "radar_callback"):
        callback = dict(rgb[role])
        callback["camera_matrix"] = _transform(callback["camera_matrix"], world_change).tolist()
        rgb[role] = callback
    rgb["_rgb_image"] = image(kwargs["rgb"]["_rgb_image"])

    actors = []
    for actor in kwargs["frozen_world"].get_actors():
        matrix = _transform(actor.get_transform().get_matrix(), world_change)
        transform = _frozen_transform(matrix)
        velocity = np.asarray([float(getattr(actor.get_velocity(), k)) for k in ("x", "y", "z")])
        velocity = world_change[:3, :3] @ velocity
        actors.append(S.FrozenActor(id=int(actor.id), type_id=str(actor.type_id), bounding_box=actor.bounding_box,
            _transform=transform, _location=transform.location,
            _velocity=carla.Vector3D(x=_f32(velocity[0]), y=_f32(velocity[1]), z=_f32(velocity[2]))))
    original_world = kwargs["frozen_world"]
    actor_view = S.FrozenActorList(actors)
    world = NS(actors=actor_view, frozen_frame_id=int(original_world.frozen_frame_id),
        frozen_timestamp_s=float(original_world.frozen_timestamp_s),
        frozen_observed_raw_ns=int(original_world.frozen_observed_raw_ns),
        population_evidence=original_world.population_evidence,
        get_actors=lambda values=actor_view: values)
    radar = np.asarray(kwargs["radar_world_xyz"], dtype=np.float64)
    radar = (world_change[:3, :3] @ radar.T).T + world_change[:3, 3]
    inverse = np.ascontiguousarray(np.linalg.inv(desired))
    return dict(kwargs, rgb=rgb, semantic_image=image(kwargs["semantic_image"]),
                depth_image=image(kwargs["depth_image"]), frozen_world=world,
                camera_matrix=desired, camera_inverse=inverse,
                radar_world_xyz=np.ascontiguousarray(radar))


def _context(base, ego):
    names = ("ego_world_x", "ego_world_y", "ego_world_z", "ego_world_pitch", "ego_world_yaw", "ego_world_roll")
    return build_frame_context_v1(stream_id=base.stream_id, frame_id=base.frame_id,
        sequence_id=base.sequence_id, capture_timestamp_ns=base.capture_timestamp_ns,
        camera_model_sha256=base.camera_model_sha256, camera_mount_sha256=base.camera_mount_sha256,
        **dict(zip(names, ego)))


class FixtureCorpus(P.Corpus):
    """The retained corpus with two source-owned, explicitly labelled rigid transforms."""
    def __init__(self, frames=49):
        super().__init__(frames)
        type(self).LAST = self
        self.fixture_evidence = []
        specifications = (
            (0, "ARCHIVED_V2_FRAME_3785_GEOMETRY", ARCHIVED_V2_EGO, ARCHIVED_V2_CAMERA, False),
            (1, "SYNTHETIC_RIGID_TRANSLATION_TO_141M", tuple(
                value + (PLANNED_TRANSLATION_Y_M if index == 1 else 0.0)
                for index, value in enumerate(ARCHIVED_V2_EGO)),
             _planned_camera(), True),
        )
        for index, label, ego, camera, synthetic in specifications:
            old = self.cases[index]
            context = _context(self.data["contexts"][old.frame_id]["context"], ego)
            kwargs = transform_kwargs(old.kwargs, camera)
            case = dataclasses.replace(old, kwargs=kwargs)
            self.cases[index] = case
            self.data["contexts"][case.frame_id]["context"] = context
            packet, extras = self.staged_packet(case)
            reconstructed = context_camera(context)
            delta = float(np.max(np.abs(camera - reconstructed)))
            self.fixture_evidence.append(dict(label=label, synthetic=synthetic, source_case_frame_id=case.frame_id,
                archived_origin_frame_id=3785, archived_source_sha256=(
                    "51547623a067b860c2c41bda4ce2fe3ca224b5ce64bdf4cb92f504b2c8dad442"),
                source_camera_matrix=camera.tolist(), reconstructed_camera_matrix=reconstructed.tolist(),
                max_abs_matrix_delta_m=delta, max_abs_translation_m=float(np.max(np.abs(camera[:3, 3]))),
                camera_inverse_max_identity_error=float(np.max(np.abs(camera @ kwargs["camera_inverse"] - np.eye(4)))),
                staged_packet_sha256=hashlib.sha256(packet).hexdigest(), stage_extras=extras))
        if self.fixture_evidence[0]["max_abs_matrix_delta_m"] != ARCHIVED_V2_DELTA_M:
            raise ValueError("archived v2 discrepancy was not reproduced exactly")
        if not all(row["max_abs_matrix_delta_m"] > 1e-5 for row in self.fixture_evidence):
            raise ValueError("high-coordinate fixtures do not cross the retained 1e-5 bound")
