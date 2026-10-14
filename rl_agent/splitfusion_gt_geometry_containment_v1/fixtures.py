"""Explicitly labelled SYNTHETIC fixtures on retained real frames (the avo_tilt ``fixtures.tilt_case`` pattern).

* ``TILT`` (``fixtures.tilt_case``, unchanged): a scoreable pedestrian's frozen rotation is replaced -> TARGET cases.
* ``RELOCATE_TILT``: a pedestrian that is not a target in that frame is moved 1.0 m BEHIND a target along the
  camera ray (so it is beyond 40 m -> structural ``distance_gt_40m``, yet inside the bounding-sphere reach of the
  target, i.e. not provably disjoint) and tilted -> POSSIBLE_COMPETITOR.  The retained corpus has no natural pair
  (scan recorded in the service proof), so relocation is required and labelled.
* ``BBOX_ROTATION``: a pedestrian's bounding-box rotation is made non-zero -> the unrelated, fatal
  ``non-zero pedestrian bounding-box rotation`` contract error (must NOT be contained).

Depth/semantic images are NOT re-rendered: fixtures exercise geometry/service paths; they are not re-simulated scenes.
"""
import dataclasses
from types import SimpleNamespace as NS

import numpy as np

from rl_agent.splitfusion_avo_tilt_v1.fixtures import tilt_case
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as S

S1_TILT = (3.7189927101135254, 7.646248817443848)      # S1 frame 6146 actor 106 (pitch, roll), ERROR.json


def _replace_actor(case, actor_id, make):
    world = case.kwargs["frozen_world"]
    actors, found = [], None
    for actor in world.get_actors():
        if str(actor.id) == str(actor_id):
            actor, found = make(actor)
        actors.append(actor)
    if found is None:
        raise ValueError(f"actor {actor_id} not in frame {case.frame_id}")
    view = S.FrozenActorList(actors)
    frozen = NS(actors=view, frozen_frame_id=int(world.frozen_frame_id), frozen_timestamp_s=float(world.frozen_timestamp_s),
                frozen_observed_raw_ns=int(world.frozen_observed_raw_ns), population_evidence=world.population_evidence,
                get_actors=lambda values=view: values)
    return dataclasses.replace(case, kwargs=dict(case.kwargs, frozen_world=frozen)), found


def _frozen(actor, transform, bounding_box=None):
    frozen = S.FrozenTransform(matrix=np.ascontiguousarray(transform.get_matrix(), dtype=np.float64),
                               inverse=np.ascontiguousarray(transform.get_inverse_matrix(), dtype=np.float64),
                               location=transform.location, rotation=transform.rotation)
    return S.FrozenActor(id=int(actor.id), type_id=str(actor.type_id),
                         bounding_box=actor.bounding_box if bounding_box is None else bounding_box,
                         _transform=frozen, _location=frozen.location, _velocity=actor.get_velocity())


def relocate_tilt_case(case, mover_id, target_id, pitch, roll, behind_m=1.0):
    import carla
    world = case.kwargs["frozen_world"]
    target = next(a for a in world.get_actors() if str(a.id) == str(target_id))
    camera = np.asarray(case.kwargs["camera_matrix"])[:3, 3]
    t_loc = target.get_transform().location
    ray = np.array([t_loc.x - camera[0], t_loc.y - camera[1], 0.0])
    ray /= np.linalg.norm(ray)

    def make(actor):
        old = actor.get_transform()
        new = carla.Transform(carla.Location(x=float(t_loc.x + behind_m * ray[0]), y=float(t_loc.y + behind_m * ray[1]),
                                             z=float(t_loc.z)),
                              carla.Rotation(pitch=float(pitch), yaw=float(old.rotation.yaw), roll=float(roll)))
        return _frozen(actor, new), dict(actor_id=str(actor.id), target_id=str(target_id), behind_m=behind_m,
                                         pitch_deg=float(new.rotation.pitch), roll_deg=float(new.rotation.roll),
                                         yaw_deg=float(new.rotation.yaw))
    return _replace_actor(case, mover_id, make)


def bbox_rotation_case(case, actor_id, yaw_deg=5.0):
    import carla

    def make(actor):
        box = actor.bounding_box
        rotated = carla.BoundingBox(box.location, box.extent)
        rotated.rotation = carla.Rotation(pitch=0.0, yaw=float(yaw_deg), roll=0.0)
        return _frozen(actor, actor.get_transform(), rotated), dict(actor_id=str(actor_id), bbox_yaw_deg=float(yaw_deg))
    return _replace_actor(case, actor_id, make)


def other_pedestrian(case, exclude):
    """First pedestrian of the frame not in ``exclude`` (deterministic: frozen actor order)."""
    for actor in case.kwargs["frozen_world"].get_actors():
        if str(actor.type_id).startswith("walker.pedestrian.") and str(actor.id) not in exclude:
            return str(actor.id)
    raise ValueError(f"no other pedestrian in frame {case.frame_id}")


def _people(case):
    """Per-pedestrian geometry of the fixture frame as the tilt-aware AVO sees it (footprint tilt, role reasons)."""
    from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, quality as Q
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG, quality_adapter as QA
    kw = case.kwargs
    try:
        rows = Q.frozen_people_rows(kw["frozen_world"], frame_id=case.frame_id, timestamp=float(kw["rgb"]["timestamp"]),
                                    camera_matrix=np.asarray(kw["camera_matrix"]),
                                    camera_inverse=np.asarray(kw["camera_inverse"]),
                                    intrinsics=LG.native_intrinsics(1280, 720), width=1280, height=720)
    except Exception as error:                      # e.g. the bbox-rotation fixture is refused by the row builder
        return dict(error=f"{type(error).__name__}: {error}"[:200])
    out = {}
    for row in rows:
        rotation, path = Q._row_rotation(row)
        extent = [row[f"gt_extent_{a}_m"] for a in "xyz"]
        if path == G.PATH_TILTED or not QA._structural_reasons(row, 1280, 720):
            out[row["gt_actor_id"]] = dict(path=path, distance_m=round(row["gt_distance_m"], 3),
                                           structural=QA._structural_reasons(row, 1280, 720),
                                           footprint_tilt_m=G.footprint_tilt_m(rotation, extent))
    return out


def targets(corpus):
    """Per case: sorted pedestrian target ids (no structural reason) of the unmodified frame."""
    return [sorted(k for k, v in _people(case).items() if not v["structural"]) for case in corpus.cases]


class FixtureCorpus(P.Corpus):
    """The retained 49-frame corpus with labelled synthetic fixtures at the given case indices.

    ``plan`` maps case index -> ("TILT", actor, pitch, roll, label) | ("RELOCATE_TILT", target, pitch, roll, label)
    | ("BBOX_ROTATION", actor, yaw, label).
    """

    def __init__(self, plan, frames=49):
        super().__init__(frames)
        self.fixture_evidence = []
        for index, spec in sorted(plan.items()):
            kind, label = spec[0], spec[-1]
            case = self.cases[index]
            if kind == "TILT":
                case, found = tilt_case(case, spec[1], spec[2], spec[3])
            elif kind == "RELOCATE_TILT":
                mover = other_pedestrian(case, {"101", "122", "146", str(spec[1])})     # never a corpus target
                case, found = relocate_tilt_case(case, mover, spec[1], spec[2], spec[3])
            elif kind == "BBOX_ROTATION":
                case, found = bbox_rotation_case(case, spec[1], spec[2])
            else:
                raise ValueError(kind)
            self.cases[index] = case
            self.fixture_evidence.append(dict(found, kind=kind, case_index=index, frame_id=case.frame_id, label=label,
                                              synthetic=True, depth_rerendered=False, people=_people(case)))
