"""Explicitly labelled SYNTHETIC tilted-pedestrian fixtures on retained real frames (V4 ``fixture_cases`` pattern).

One scoreable pedestrian's frozen transform is replaced by ``carla.Transform(location, Rotation(pitch, yaw, roll))``
with its own ``get_matrix()`` / ``get_inverse_matrix()`` -- how a live frozen actor is captured.  Location, yaw,
bounding box, velocity, every other actor, images, radar and population evidence are unchanged.  The depth image is
NOT re-rendered, so a fixture exercises the geometry/service path; it is not a physically re-simulated scene.
"""
import dataclasses
from types import SimpleNamespace as NS

import numpy as np

from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as S

LIVE_V7_TILT = (-0.031540337949991226, -0.010556723922491074)      # frame 3502 actor 117 (pitch, roll)


def tilt_case(case, actor_id, pitch, roll):
    import carla
    world = case.kwargs["frozen_world"]
    actors, found = [], None
    for actor in world.get_actors():
        if str(actor.id) == str(actor_id):
            old = actor.get_transform()
            new = carla.Transform(carla.Location(x=old.location.x, y=old.location.y, z=old.location.z),
                                  carla.Rotation(pitch=float(pitch), yaw=float(old.rotation.yaw), roll=float(roll)))
            transform = S.FrozenTransform(matrix=np.ascontiguousarray(new.get_matrix(), dtype=np.float64),
                                          inverse=np.ascontiguousarray(new.get_inverse_matrix(), dtype=np.float64),
                                          location=new.location, rotation=new.rotation)
            actor = S.FrozenActor(id=int(actor.id), type_id=str(actor.type_id), bounding_box=actor.bounding_box,
                                  _transform=transform, _location=transform.location, _velocity=actor.get_velocity())
            found = dict(actor_id=str(actor_id), pitch_deg=float(new.rotation.pitch), yaw_deg=float(new.rotation.yaw),
                         roll_deg=float(new.rotation.roll),
                         translation_unchanged=bool(np.array_equal(np.asarray(old.get_matrix())[:3, 3],
                                                                   np.asarray(new.get_matrix())[:3, 3])))
        actors.append(actor)
    if found is None:
        raise ValueError(f"actor {actor_id} not in frame {case.frame_id}")
    view = S.FrozenActorList(actors)
    frozen = NS(actors=view, frozen_frame_id=int(world.frozen_frame_id), frozen_timestamp_s=float(world.frozen_timestamp_s),
                frozen_observed_raw_ns=int(world.frozen_observed_raw_ns), population_evidence=world.population_evidence,
                get_actors=lambda values=view: values)
    return dataclasses.replace(case, kwargs=dict(case.kwargs, frozen_world=frozen)), found


class TiltCorpus(P.Corpus):
    """The retained 49-frame corpus with labelled synthetic tilts at the given case indices."""

    def __init__(self, tilts, frames=49):
        super().__init__(frames)
        self.fixture_evidence = []
        for index, (actor_id, pitch, roll, label) in sorted(tilts.items()):
            case, found = tilt_case(self.cases[index], actor_id, pitch, roll)
            self.cases[index] = case
            self.fixture_evidence.append(dict(found, case_index=index, frame_id=case.frame_id, label=label,
                                              synthetic=True, depth_rerendered=False))
