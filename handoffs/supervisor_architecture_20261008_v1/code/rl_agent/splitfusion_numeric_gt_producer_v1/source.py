"""Exact corrected-GT source numerics with staged (commit/cancel) history.

Every validator of the retained ``LiveCompleteTargetProducer.build`` is kept
verbatim and raises the same ``SourceGeometryError``. The numerical functions
are the pinned originals. What changes is orchestration only:

* native depth is decoded once and shared by source rows, AVO and the target;
* full-image class masks are computed once per frame and sliced, never recomputed
  per static object or per box (identical booleans);
* the radar homogeneous matrix is formed once per frame instead of once per actor;
* the catalogue digest is computed once at construction (immutable catalogue);
* the complete-input digest is computed once and reused by the unchanged validator;
* evidence/provenance canonicalisation is explicit and optionally deferred;
* tracker/conflict history advances only on ``commit``; ``cancel`` restores it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_gt_optimization_v2 import candidate as V2C
from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as C
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY

from . import originals as O

SourceGeometryError = S.SourceGeometryError
HISTORY_SCHEMA = "splitfusion.numeric_gt_producer.history_digest.v1"


class StagingError(RuntimeError):
    """Exactly-once staging protocol violated (double commit, compute while pending...)."""


def _sha(value):
    """Exact ``S._sha``: canonical JSON digest with the proven fast normaliser."""
    return O.bound().adapter.input_content_sha256.__globals__["hashlib"].sha256(
        A.canonical(V1S.json_value(value))).hexdigest()


# --------------------------------------------------------------------------- shared frame work

def hoisted_transform(points_world: np.ndarray):
    """Original ``transform_world_to_actor_local`` with one homogeneous matrix per frame.

    The original rebuilds ``concatenate([points.astype(float64), ones])`` for every
    actor. The matrix is a pure function of the radar array, so one fresh C-order
    allocation per frame yields identical operands and the identical matmul.
    """
    if points_world.size == 0:
        homo = None
    else:
        homo = np.concatenate([points_world.astype(np.float64), np.ones((points_world.shape[0], 1))], axis=1)

    def transform_world_to_actor_local(points, actor):
        if points is not points_world:
            raise SourceGeometryError("hoisted radar transform received a foreign point array")
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
        return (inverse @ homo.T).T[:, :3]
    return transform_world_to_actor_local


def frame_masks(semantic: np.ndarray, static_ns: Mapping[str, Any]):
    """Full-image boolean masks the original recomputes per object; identical values."""
    vehicle = np.isin(semantic, list(static_ns["VEHICLE_SEMANTIC_TAGS"]))
    same_class = {}

    def same(tag: int) -> np.ndarray:
        mask = same_class.get(tag)
        if mask is None:
            mask = same_class[tag] = semantic == tag
        return mask
    return vehicle, same


def static_rows(bound: O.Bound, *, frame, boxes, geometry, depth_m, semantic, vehicle_mask, same_class, radar):
    """Literal ``audit_visibility`` for one frame with owned in-memory inputs.

    Row dictionaries, order, numeric validation, failure counters and the summary
    are the original's. Only the per-object full-image ``isin``/``==`` are hoisted
    and the depth is the already decoded array (same float32 values).
    """
    ns = bound.static
    carla = ns["carla"]
    width, height = int(frame["camera_width"]), int(frame["camera_height"])
    if (width, height) != (1280, 720):
        raise RuntimeError(f"unexpected retained camera size: {(width, height)}")
    episode, sample_id = frame["experiment_id"], frame["sample_id"]
    if not ns["depth_is_plausible"](depth_m):
        raise RuntimeError(f"implausible retained depth: {sample_id}")
    if semantic is None or semantic.shape != (height, width):
        raise RuntimeError(f"invalid retained semantic tags: {sample_id}")
    radar_world = np.asarray(radar, dtype=np.float64)
    if radar_world.ndim != 2 or radar_world.shape[1] != 3 or not np.all(np.isfinite(radar_world)):
        raise RuntimeError(f"invalid retained radar world points: {sample_id}")
    radar_tree = cKDTree(radar_world) if len(radar_world) else None
    camera_inverse = np.asarray(json.loads(frame["camera_inverse_matrix_json"]), dtype=np.float64)
    intrinsics = ns["intrinsics_from_manifest"](frame)
    camera_center = np.asarray([float(frame[f"camera_{axis}"]) for axis in "xyz"], dtype=np.float64)
    class_tag = {name: int(getattr(carla.CityObjectLabel, name)) for name in ns["SEMANTIC_NAMES"]}
    actor_rows_by_sample = {sample_id: boxes}
    rows, ambiguity_rows = [], []
    projection_failures = Counter()
    frame_counts = {"sample_id": sample_id, "in_range": 0, "projected_in_frame": 0, "area_qualified": 0,
                    "eligible_v010": 0, "eligible_v025": 0, "unobservable_area_qualified": 0, "ambiguous": 0}
    for item in geometry:
        catalog, center, extent = item["row"], item["center"], item["extent"]
        distance = float(np.linalg.norm(center - camera_center))
        if distance > ns["MAX_RANGE_M"]:
            continue
        frame_counts["in_range"] += 1
        try:
            metrics, visible_mask = ns["visibility_from_corners"](item["corners"], camera_inverse, intrinsics, depth_m,
                distance_m=distance, width=width, height=height, model_width=ns["MODEL_WIDTH"], model_height=ns["MODEL_HEIGHT"])
        except ValueError as exc:
            projection_failures[str(exc)] += 1
            continue
        frame_counts["projected_in_frame"] += 1
        sensor_center = (camera_inverse @ np.asarray([center[0], center[1], center[2], 1.0]))[:3]
        semantic_vehicle_px = int(np.count_nonzero(visible_mask & vehicle_mask))
        semantic_same_px = int(np.count_nonzero(visible_mask & same_class(class_tag[catalog["semantic_class"]])))
        native_visible_px = int(metrics["native_visible_px"])
        static_size = extent * 2.0
        ambiguities = ns["actor_ambiguities"](center, static_size, actor_rows_by_sample.get(sample_id, []))
        nearest = ambiguities[0] if ambiguities else None
        row = {
            "experiment_id": episode, "sample_id": sample_id, "frame_id": int(frame["frame_id"]),
            "timestamp": float(frame["timestamp"]),
            **{key: catalog[key] for key in (
                "environment_object_id", "carla_environment_object_id", "object_name", "semantic_class", "enabled",
                "map_name", "map_sha256", "bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m",
                "bbox_extent_x_m", "bbox_extent_y_m", "bbox_extent_z_m", "bbox_rotation_roll_deg",
                "bbox_rotation_pitch_deg", "bbox_rotation_yaw_deg")},
            "size_x_m": float(static_size[0]), "size_y_m": float(static_size[1]), "size_z_m": float(static_size[2]),
            "object_sensor_x_m": float(sensor_center[0]), "object_sensor_y_m": float(sensor_center[1]),
            "object_sensor_z_m": float(sensor_center[2]), "range_m": distance, **metrics,
            "semantic_vehicle_visible_px": semantic_vehicle_px, "semantic_same_class_visible_px": semantic_same_px,
            "semantic_vehicle_visible_fraction": (semantic_vehicle_px / native_visible_px if native_visible_px else 0.0),
            "semantic_same_class_visible_fraction": (semantic_same_px / native_visible_px if native_visible_px else 0.0),
            "radar_support_points": ns["radar_support_count"](radar_tree, radar_world, center, extent, item["rotation"]),
            "radar_support_margin_m": ns["RADAR_MARGIN_M"],
            "possible_dynamic_static_duplicate": bool(ambiguities), "ambiguous_actor_count": len(ambiguities),
            "ambiguous_actor_ids": ";".join(str(match["gt_actor_id"]) for match in ambiguities),
            "nearest_ambiguous_actor_id": (nearest["gt_actor_id"] if nearest else ""),
            "nearest_ambiguous_center_distance_m": (nearest["center_distance_m"] if nearest else ""),
            "nearest_ambiguous_size_x_delta_m": (nearest["size_deltas_m"][0] if nearest else ""),
            "nearest_ambiguous_size_y_delta_m": (nearest["size_deltas_m"][1] if nearest else ""),
            "nearest_ambiguous_size_z_delta_m": (nearest["size_deltas_m"][2] if nearest else ""),
        }
        rows.append(row)
        if ambiguities:
            ambiguity_rows.append(row)
            frame_counts["ambiguous"] += 1
        if bool(metrics["geometry_qualified_v2"]):
            frame_counts["area_qualified"] += 1
            if metrics["visibility_tier"] == "unobservable":
                frame_counts["unobservable_area_qualified"] += 1
        if bool(metrics["eligible_visible_v010"]):
            frame_counts["eligible_v010"] += 1
        if bool(metrics["eligible_clear_v025"]):
            frame_counts["eligible_v025"] += 1
    numeric_fields = ("bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m", "bbox_extent_x_m", "bbox_extent_y_m",
        "bbox_extent_z_m", "size_x_m", "size_y_m", "size_z_m", "bbox_rotation_roll_deg", "bbox_rotation_pitch_deg",
        "bbox_rotation_yaw_deg", "object_sensor_x_m", "object_sensor_y_m", "object_sensor_z_m", "range_m",
        "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h", "clipped_projected_area_px",
        "visible_fraction", "model_input_visible_px", "semantic_vehicle_visible_px", "radar_support_points")
    for row in rows:
        for name in numeric_fields:
            ns["finite_float"](row[name], name)
        if row["visibility_algorithm_version"] != ns["ALGORITHM_VERSION"]:
            raise RuntimeError("visibility helper version drift")
    keys = ("in_range", "projected_in_frame", "area_qualified", "eligible_v010", "eligible_v025",
            "unobservable_area_qualified", "ambiguous")
    summary = {"projected_row_count": len(rows), "ambiguity_row_count": len(ambiguity_rows),
        "ambiguity_unique_static_objects": len({row["environment_object_id"] for row in ambiguity_rows}),
        "ambiguity_frames": len({row["sample_id"] for row in ambiguity_rows}),
        "projection_failures": dict(sorted(projection_failures.items())),
        "totals": {key: sum(int(r[key]) for r in [frame_counts]) for key in keys},
        "per_frame": {key: ns["numeric_summary"]([int(r[key]) for r in [frame_counts]]) for key in keys}}
    return rows, ambiguity_rows, summary


# --------------------------------------------------------------------------- staged result

@dataclass
class Staged:
    """One computed frame awaiting exactly-once commit or cancel.

    Numerical outputs are complete; ``evidence``/``provenance`` are present when
    requested at compute time (``bind_owned_source``) or built later by
    ``artifacts``. Mutating this object never affects producer history.
    """
    frame_id: int
    identity: Mapping[str, Any]
    frame: dict
    sid: str
    stamp: float
    k: np.ndarray
    camera_matrix: np.ndarray
    camera_inverse: np.ndarray
    depth_raw: np.ndarray
    semantic: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    vehicle_mask: np.ndarray
    same_class: Any
    radar: np.ndarray
    boxes: list
    visibility: list
    static: list
    static_summary: dict
    conflict: dict
    sync: dict
    corners_evidence: list | None
    registration: Any
    start_timestamp: float
    population: frozenset
    population_evidence_sha256: str
    dynamic_snapshot_evidence: dict
    evidence: dict | None
    inputs: Any
    history_before_sha256: str
    history_after_sha256: str
    _restore: dict = field(default_factory=dict, repr=False)
    _state: str = "PENDING"


class NumericSourceProducer:
    """Same constructor contract and validators as the retained producer.

    ``compute`` performs every original check, advances the tracker/conflict
    state tentatively and returns a ``Staged`` frame. Only ``commit`` makes the
    frame part of the accepted history; ``cancel`` restores the previous state
    exactly. A second ``compute`` while one frame is pending is refused.
    """
    def __init__(self, *, static_catalogue_dir: Path, experiment_id: str, ego_id: int,
                 dynamic_population_ids: Sequence[int], population_evidence_sha256: str, max_frames: int):
        self.bound = O.bound()
        self.sources = {name: dict(value) for name, value in self.bound.source_pins.items()}
        if not experiment_id or "canonical_v3_" in experiment_id:
            raise SourceGeometryError("new-run experiment identity required")
        self.experiment_id, self.ego_id = str(experiment_id), int(ego_id)
        self.population = frozenset(int(i) for i in dynamic_population_ids)
        if len(self.population) != len(dynamic_population_ids) or self.ego_id in self.population:
            raise SourceGeometryError("invalid complete dynamic population registry")
        if len(population_evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in population_evidence_sha256):
            raise SourceGeometryError("population evidence SHA-256 required")
        self.population_evidence_sha256 = population_evidence_sha256
        if type(max_frames) is not int or not 1 <= max_frames <= 1000:
            raise SourceGeometryError("registered max_frames must be an integer in [1,1000]")
        self.max_frames, self.processed_frames = max_frames, 0
        self.pending_registration = None
        self.catalogue, self.catalogue_snapshot, self.catalogue_evidence = S._load_catalogue(Path(static_catalogue_dir))
        self.geometry, _ = self.bound.static["build_catalog_geometry"](self.catalogue)
        self.tracker = self.bound.objects["ActorStationaryTracker"](.35, 5.)
        self.conflict_state = C.IncrementalConflictState(self.catalogue, max_frames=max_frames)
        # The catalogue is immutable after construction; its digest is a constant.
        self.catalogue_digest = C.catalogue_content_sha256(self.catalogue)
        if self.catalogue_digest != self.conflict_state.catalogue_digest:
            raise SourceGeometryError("catalogue digest differs between producer and conflict state")
        self.start_timestamp = None
        self.previous_frame = -1
        self.previous_timestamp = -1.
        self.last_evidence = None
        self.staged: Staged | None = None

    # ------------------------------------------------------------------ registration (unchanged)
    def register_population(self, ids: Sequence[int], evidence_sha256: str, before_frame_id: int):
        if self.staged is not None:
            raise StagingError("cannot register a population while a frame is pending")
        if self.pending_registration is not None:
            raise SourceGeometryError("an unconsumed population registration already exists")
        if type(before_frame_id) is not int or before_frame_id <= self.previous_frame:
            raise SourceGeometryError("population registration must precede its new owned frame")
        if (not isinstance(ids, (tuple, list)) or any(type(i) is not int for i in ids)
                or len(set(ids)) != len(ids) or not self.population.issubset(ids)
                or self.ego_id in ids or any(i < 0 for i in ids)):
            raise SourceGeometryError("complete monotonic cumulative population required")
        if (not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64
                or any(c not in "0123456789abcdef" for c in evidence_sha256)):
            raise SourceGeometryError("owned population registration SHA-256 required")
        self.pending_registration = {"population_ids": list(ids), "evidence_sha256": evidence_sha256,
                                     "before_frame_id": before_frame_id}

    # ------------------------------------------------------------------ history digest
    def _history_digest(self, *, processed_frames, previous_frame, previous_timestamp, start_timestamp,
                        population_sha256, population) -> str:
        state = self.conflict_state
        return _sha({"schema": HISTORY_SCHEMA, "processed_frames": processed_frames,
            "previous_frame": previous_frame, "previous_timestamp": previous_timestamp,
            "start_timestamp": start_timestamp, "population_sha256": population_sha256,
            "population": sorted(population),
            "tracker_ages": sorted(self.tracker._ages.items()), "tracker_last": sorted(self.tracker._last_time.items()),
            "conflict_chain": state.chain, "conflict_frames": state.frame_count, "conflict_rows": state.row_count})

    def history_sha256(self) -> str:
        """Cheap digest of everything that advances with an accepted frame."""
        return self._history_digest(processed_frames=self.processed_frames, previous_frame=self.previous_frame,
            previous_timestamp=self.previous_timestamp, start_timestamp=self.start_timestamp,
            population_sha256=self.population_evidence_sha256, population=self.population)

    def _projected_history_sha256(self, staged: "Staged") -> str:
        """Digest the committed state would have, computed before commit."""
        registration = staged.registration
        return self._history_digest(processed_frames=self.processed_frames + 1, previous_frame=staged.frame_id,
            previous_timestamp=staged.stamp, start_timestamp=staged.start_timestamp,
            population_sha256=(self.population_evidence_sha256 if registration is None else registration["evidence_sha256"]),
            population=staged.population)

    # ------------------------------------------------------------------ compute
    def compute(self, *, identity: Mapping[str, Any], rgb: Mapping[str, Any], semantic_image: Any, depth_image: Any,
                frozen_world: Any, camera_matrix: np.ndarray, camera_inverse: np.ndarray, radar_world_xyz: np.ndarray,
                dynamic_snapshot_evidence: Mapping[str, Any], bind_owned_source: bool = True) -> Staged:
        if self.staged is not None:
            raise StagingError("a computed frame is pending commit or cancel")
        bound = self.bound
        frame_id, stamp = int(identity["frame_id"]), float(rgb["timestamp"])
        # ---- validators: verbatim from the retained producer ----
        if self.processed_frames >= self.max_frames:
            raise SourceGeometryError("registered complete-target frame budget exhausted")
        registration = self.pending_registration
        if registration is not None and registration["before_frame_id"] != frame_id:
            raise SourceGeometryError("population registration belongs to a different owned frame")
        population = self.population if registration is None else frozenset(registration["population_ids"])
        sync = SY.source_alignment_record(identity=identity, rgb=rgb, semantic_image=semantic_image,
            depth_image=depth_image, frozen_world=frozen_world, policy_timestamp_s=stamp)
        SY.validate_source_alignment(sync)
        if frame_id <= self.previous_frame or not math.isfinite(stamp) or stamp <= self.previous_timestamp:
            raise SourceGeometryError("source frame order must be strictly increasing")
        for key in ("registry_complete", "no_unregistered_spawn", "owned_snapshot_complete"):
            if dynamic_snapshot_evidence.get(key) is not True:
                raise SourceGeometryError(f"dynamic population evidence incomplete: {key}")
        if int(dynamic_snapshot_evidence.get("frame_id", -1)) != frame_id:
            raise SourceGeometryError("dynamic population evidence has wrong frame")
        active_list = dynamic_snapshot_evidence.get("active_population_ids")
        if not isinstance(active_list, (list, tuple)) or any(type(i) is not int for i in active_list):
            raise SourceGeometryError("exact owned-snapshot active population IDs required")
        active = frozenset(active_list)
        if len(active) != len(active_list) or not active.issubset(population):
            raise SourceGeometryError("active population includes unknown or duplicate actors")
        actors = list(frozen_world.get_actors())
        if {int(a.id) for a in actors} != active or len(actors) != len(active):
            raise SourceGeometryError("frozen actor registry differs from exact active population")
        if any(not (str(a.type_id).startswith("vehicle.") or str(a.type_id).startswith("walker.pedestrian.")) for a in actors):
            raise SourceGeometryError("non-vehicle/person actor in dynamic geometry population")
        if int(self.catalogue_snapshot["capture"]["frame_id"]) > frame_id:
            raise SourceGeometryError("static catalogue was captured after this frame")
        if (int(depth_image.width), int(depth_image.height)) != (1280, 720):
            raise SourceGeometryError("native 1280x720 source images required")
        for image in (semantic_image, depth_image):
            if (int(image.frame), float(image.timestamp), int(image.width), int(image.height)) != (frame_id, stamp, 1280, 720):
                raise SourceGeometryError("actual source image differs from callback identity")
            if not np.allclose(np.asarray(image.transform.get_matrix()), np.asarray(rgb["camera_matrix"]), rtol=0, atol=1e-5):
                raise SourceGeometryError("actual source image pose differs from callback")
        camera_matrix, camera_inverse = np.asarray(camera_matrix), np.asarray(camera_inverse)
        for matrix in (camera_matrix, camera_inverse):
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise SourceGeometryError("invalid owned camera matrix")
        if not np.allclose(camera_matrix @ camera_inverse, np.eye(4), rtol=0, atol=1e-8):
            raise SourceGeometryError("camera inverse does not invert owned camera pose")
        if not np.allclose(camera_matrix, np.asarray(rgb["camera_matrix"]), rtol=0, atol=1e-5):
            raise SourceGeometryError("camera matrix differs from owned RGB pose")
        radar = np.asarray(radar_world_xyz)
        if radar.ndim != 2 or radar.shape[1] != 3 or not np.isfinite(radar).all():
            raise SourceGeometryError("same-frame radar world points required")
        # ---- native arrays: one decode, one plausibility check, shared downstream ----
        depth_raw = np.frombuffer(depth_image.raw_data, np.uint8).reshape(720, 1280, 4).copy()
        semantic = np.frombuffer(semantic_image.raw_data, np.uint8).reshape(720, 1280, 4)[..., 2].copy()
        depth = bound.visibility.decode_depth_bgra(depth_raw)
        if not bound.visibility.depth_is_plausible(depth):
            raise SourceGeometryError("implausible native encoded depth")
        native_f = (1280. / 2) / math.tan(math.radians(120.) / 2)
        k = np.array([[native_f, 0, 640.], [0, native_f, 360.], [0, 0, 1.]])
        sid = f"{self.experiment_id}_frame{frame_id}"
        frame = dict(experiment_id=self.experiment_id, sample_id=sid, frame_id=frame_id,
            timestamp=stamp, camera_width=1280, camera_height=720, camera_fx=k[0, 0], camera_fy=k[1, 1],
            camera_cx=640., camera_cy=360., camera_x=camera_matrix[0, 3], camera_y=camera_matrix[1, 3],
            camera_z=camera_matrix[2, 3], camera_inverse_matrix_json=json.dumps(camera_inverse.tolist()),
            mask_path=f"masks/{sid}.png", instance_raw_path=f"semantic/{sid}.png", radar_points_path=f"radar/{sid}.npz")
        history_before = self.history_sha256()
        restore = {"ages": dict(self.tracker._ages), "last": dict(self.tracker._last_time),
                   "conflict": self._conflict_snapshot()}
        try:
            carla = bound.static["carla"]
            objects = bound.objects
            objects["transform_world_to_actor_local"] = hoisted_transform(radar)
            try:
                boxes = objects["build_object_rows"](world=frozen_world, ego_vehicle=SimpleNamespace(id=self.ego_id),
                    sample_base={key: frame[key] for key in ("experiment_id", "sample_id", "frame_id", "timestamp")},
                    camera_location=carla.Location(*map(float, camera_matrix[:3, 3])),
                    camera_matrix=camera_matrix, camera_inverse_matrix=camera_inverse, intrinsics=k,
                    width=1280, height=720, max_distance_m=140., radar_world_xyz=radar,
                    stationary_tracker=self.tracker, include_pedestrians=True, radar_support_margin_m=1.,
                    radar_person_support_mode="radius", radar_person_support_radius_m=1.5,
                    radar_person_support_z_down_m=.5, radar_person_support_z_up_m=2.)
            finally:
                objects["transform_world_to_actor_local"] = _ORIGINAL_TRANSFORM
            by_id = {int(a.id): a for a in actors}
            visibility, person_masks = [], []
            for row in boxes:
                actor = by_id[int(row["gt_actor_id"])]
                _center, corners = objects["actor_bbox_world_points"](actor)
                metrics, support = bound.visibility.visibility_from_corners(corners, camera_inverse, k, depth,
                    distance_m=float(row["gt_distance_m"]), width=1280, height=720, model_width=768, model_height=432)
                for a, b in (("gt_bbox_x", "clipped_bbox_x"), ("gt_bbox_y", "clipped_bbox_y"),
                             ("gt_bbox_w", "clipped_bbox_w"), ("gt_bbox_h", "clipped_bbox_h")):
                    if abs(float(row[a]) - float(metrics[b])) > 1e-6:
                        raise SourceGeometryError("original actor box/visibility projection differs")
                vis = {key: row[key] for key in ("experiment_id", "sample_id", "frame_id", "timestamp",
                    "gt_actor_id", "label", "gt_actor_type_id", "gt_distance_m")}
                vis.update(metrics, depth_path=f"{self.experiment_id}/depth/{sid}.png", depth_frame_id=frame_id,
                    depth_timestamp_s=stamp, person_mask_candidate_px=0, person_mask_painted_px=0,
                    frame_person_mask_px=0, low_support_fallback_used=False)
                visibility.append(vis)
                if row["label"] == "person":
                    person_masks.append((vis, support))
            mask = bound.semantic["carla_semantic_tags_to_training_mask"](semantic)
            counters = SimpleNamespace(person_mask_subset_failures=0, ineligible_person_paint_events=0,
                                       vehicle_mask_preservation_failures=0)
            scope = dict(np=np, mask=mask, person_masks=person_masks, frame_visibility=visibility,
                         frame_id=frame_id, self=counters, v2=SimpleNamespace(PilotError=SourceGeometryError))
            exec(bound.paint, scope)
            vehicle_mask, same_class = frame_masks(semantic, bound.static)
            static, _ambiguities, static_summary = static_rows(bound, frame=frame, boxes=boxes, geometry=self.geometry,
                depth_m=depth, semantic=semantic, vehicle_mask=vehicle_mask, same_class=same_class, radar=radar)
            start = stamp if self.start_timestamp is None else self.start_timestamp
            conflict = self.conflict_state.derive(frame=frame, actor_boxes=boxes, actor_visibility=visibility,
                static_candidates=static,
                provenance={"scope": C.SCOPE, "capture_identity": dict(identity),
                    "complete": dict.fromkeys(C.COMPLETENESS, True),
                    "catalogue_content_sha256": self.catalogue_digest,
                    "history_population": "EVERY_ACCEPTED_REWARD_FRAME_OF_THIS_PRODUCER",
                    "history_window": {"start_timestamp": start, "end_timestamp": stamp}})
            staged = Staged(frame_id=frame_id, identity=dict(identity), frame=frame, sid=sid, stamp=stamp, k=k,
                camera_matrix=camera_matrix, camera_inverse=camera_inverse, depth_raw=depth_raw, semantic=semantic,
                depth=depth, mask=mask, vehicle_mask=vehicle_mask, same_class=same_class, radar=radar, boxes=boxes,
                visibility=visibility, static=static, static_summary=static_summary, conflict=conflict, sync=sync,
                corners_evidence=None, registration=registration, start_timestamp=start, population=population,
                population_evidence_sha256=self.population_evidence_sha256,
                dynamic_snapshot_evidence=dict(dynamic_snapshot_evidence), evidence=None, inputs=None,
                history_before_sha256=history_before, history_after_sha256="", _restore=restore)
            staged.corners_evidence = self.dynamic_corners(actors)
            if bind_owned_source:
                staged.evidence = self.source_evidence(staged)
            staged.inputs = self.complete_inputs(staged)
            staged.history_after_sha256 = self._projected_history_sha256(staged)
        except BaseException:
            self._restore_history(restore)
            raise
        self.staged = staged
        return staged

    # ------------------------------------------------------------------ evidence / inputs (constructors)
    def dynamic_corners(self, actors):
        objects = self.bound.objects
        corners_evidence = []
        for actor in actors:
            center, corners = objects["actor_bbox_world_points"](actor)
            bbox = actor.bounding_box
            corners_evidence.append({"actor_id": int(actor.id), "type_id": str(actor.type_id),
                "center": center.tolist(), "corners": corners.tolist(),
                "matrix": actor.get_transform().get_matrix(),
                "bbox_location": [float(getattr(bbox.location, axis)) for axis in "xyz"],
                "bbox_extent": [float(getattr(bbox.extent, axis)) for axis in "xyz"],
                "velocity": [float(getattr(actor.get_velocity(), axis)) for axis in "xyz"],
                "corner_convention": "ORIGINAL_ACTOR_MATRIX_TIMES_BBOX_LOCATION_PLUS_EXTENT_OFFSETS"})
        return corners_evidence

    def source_evidence(self, staged: Staged) -> dict:
        """Literal retained evidence document (deferred unless bound at compute)."""
        return {"identity": dict(staged.identity), "source_synchronization": staged.sync,
            "dynamic_snapshot_evidence": dict(staged.dynamic_snapshot_evidence),
            "population_evidence_sha256": staged.population_evidence_sha256,
            "new_population_registration": staged.registration,
            "dynamic_corners": staged.corners_evidence, "static_catalogue": self.catalogue_evidence,
            "static_summary": staged.static_summary, "conflicts": staged.conflict["evidence"], "sources": self.sources,
            "non_target_metadata_note": "Original stationary metadata maintained by a separate GT producer tracker; no policy input is modified."}

    def complete_inputs(self, staged: Staged):
        """Exact ``CompleteTargetInputs`` with the unchanged validator, digest computed once."""
        if staged.evidence is None:
            staged.evidence = self.source_evidence(staged)
        adapter = self.bound.adapter
        provenance = {"schema": A.SCHEMA, "scope": C.SCOPE, "sample_id": staged.sid,
            "capture_identity": dict(staged.identity),
            "complete": dict.fromkeys(A.COMPLETENESS_KEYS, True),
            "evidence_sha256": {"owned_source": _sha(staged.evidence), "population": staged.population_evidence_sha256,
                "catalogue": self.catalogue_digest, "conflicts": staged.conflict["evidence_sha256"]}}
        inputs = A.CompleteTargetInputs(frame=staged.frame, split="val", depth_bgra=staged.depth_raw,
            raw_semantic=staged.semantic, original_v3_mask=staged.mask, actor_boxes=staged.boxes,
            actor_visibility=staged.visibility, static_candidates=staged.static,
            quarantine=staged.conflict["quarantine"], provenance=provenance)
        digest = adapter.input_content_sha256(inputs)
        inputs = replace(inputs, provenance=dict(provenance, input_content_sha256=digest))
        # The unchanged validator recomputes the content digest; serve the value just
        # computed for this exact object instead of hashing 5.5 MB a second time.
        namespace = dict(vars(adapter))
        def memoized(value, _inputs=inputs, _digest=digest):
            if value is not _inputs:
                return adapter.input_content_sha256(value)
            return _digest
        namespace["input_content_sha256"] = memoized
        # The validator's full-mask label scan uses the qualified v2 exact uint8
        # label set (bincount) instead of a 0.9 MB sort; all pixels are checked.
        namespace["np"] = V2C.ExactArrayOperations()
        # Clone the validator captured at startup (pristine code object), never the
        # live module attribute, so instrumentation or foreign wrappers cannot
        # route the check back through a second full digest.
        V1S._function(adapter._validate, namespace)(inputs)
        return inputs

    # ------------------------------------------------------------------ history staging
    def _conflict_snapshot(self):
        s = self.conflict_state
        return dict(templates=dict(s.templates), pairs=dict(s.pairs), decisions=dict(s.decisions),
            frame_count=s.frame_count, row_count=s.row_count, previous_frame=s.previous_frame,
            previous_timestamp=s.previous_timestamp, first_timestamp=s.first_timestamp, episode=s.episode,
            seen_samples=set(s.seen_samples), chain=s.chain)

    def _restore_history(self, restore):
        self.tracker._ages = dict(restore["ages"])
        self.tracker._last_time = dict(restore["last"])
        s, c = self.conflict_state, restore["conflict"]
        s.templates, s.pairs, s.decisions = dict(c["templates"]), dict(c["pairs"]), dict(c["decisions"])
        s.frame_count, s.row_count = c["frame_count"], c["row_count"]
        s.previous_frame, s.previous_timestamp, s.first_timestamp = c["previous_frame"], c["previous_timestamp"], c["first_timestamp"]
        s.episode, s.seen_samples, s.chain = c["episode"], set(c["seen_samples"]), c["chain"]

    def _take(self, staged: Staged):
        if self.staged is None or staged is not self.staged or staged._state != "PENDING":
            raise StagingError("staged frame is not the pending frame of this producer")
        self.staged = None

    def commit(self, staged: Staged) -> dict:
        """Exactly-once acceptance: the frame now belongs to the accepted history."""
        self._take(staged)
        self.previous_frame, self.start_timestamp = staged.frame_id, staged.start_timestamp
        self.last_evidence = staged.evidence if staged.evidence is not None else self.source_evidence(staged)
        self.previous_timestamp = staged.stamp
        if staged.registration is not None:
            self.population = staged.population
            self.population_evidence_sha256 = staged.registration["evidence_sha256"]
            self.pending_registration = None
        self.processed_frames += 1
        staged._state = "COMMITTED"
        staged._restore = {}
        after = self.history_sha256()
        if after != staged.history_after_sha256:
            raise StagingError("committed history differs from the staged after-digest")
        return {"frame_id": staged.frame_id, "history_before_sha256": staged.history_before_sha256,
                "history_after_sha256": after, "processed_frames": self.processed_frames}

    def cancel(self, staged: Staged) -> dict:
        """Exactly-once rejection: tracker/conflict history return to the pre-frame state."""
        self._take(staged)
        self._restore_history(staged._restore)
        # A registration bound to a frame that was not accepted is discarded; the
        # owner must re-register from that later frame's own evidence.
        self.pending_registration = None
        staged._state = "CANCELLED"
        staged._restore = {}
        after = self.history_sha256()
        if after != staged.history_before_sha256:
            raise StagingError("cancelled history differs from the staged before-digest")
        return {"frame_id": staged.frame_id, "history_sha256": after, "processed_frames": self.processed_frames}


_ORIGINAL_TRANSFORM = O.bound().objects["transform_world_to_actor_local"]
