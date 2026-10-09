"""Complete prospective v010 inputs from an owned live sensor/world snapshot.

No simulator RPC occurs per frame. The only RPC-capable operation is the
explicit pre-route catalogue capture. Original object projection, depth
visibility, v3 person painting, static projection and conflict functions are
executed from hash-pinned source. No historical frame/sample data is reused.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import conflicts

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "objects": ("carla_collect_parked_ego_fusion_training_data.py", "d9cb8c8357269c9cbb2ad86d492fec9fb2966c838135889ce7946b926c11d836"),
    "semantic": ("pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/common.py", "b98196081b5bfd7488eefc3ea152aa4235866c9aeea6436f09d9505b44e6007b"),
    "catalogue": ("data_collection/phase2_static_environment_truth_v1.py", "b279f2ba3d23e3ca764d9486f0deeb2a95dee251167cb99b8ac21da92bc50d54"),
    "static_projection": ("experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706/scripts/run_static_vehicle_audit_v1.py", "864dd63b3be0bc2f4bb2271d6135fca25d21cc3d0eb988cd05655bb97861f546"),
}


class SourceGeometryError(ValueError):
    pass


def _sha(value):
    return hashlib.sha256(A.canonical(A._json_value(value))).hexdigest()


def verify_sources():
    result = A.verify_sources(ROOT)
    for name, (path, expected) in SOURCES.items():
        actual = A.sha256(ROOT / path)
        if actual != expected:
            raise SourceGeometryError(f"original {name} source changed")
        result[name] = {"path": path, "sha256": actual}
    return result


def _functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names):
        raise SourceGeometryError(f"original function inventory changed: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@lru_cache(maxsize=1)
def _original():
    verify_sources()
    # CARLA types are used for matrix construction only, never a Client/World.
    import carla
    _, visibility, _ = A._original_modules(ROOT)
    base = dict(np=np, math=math, Any=Any, Dict=Dict, List=List, Optional=Optional,
                Sequence=Sequence, Tuple=Tuple, Mapping=Mapping, carla=carla)
    objects = _functions(ROOT / SOURCES["objects"][0], {
        "bbox_corner_offsets", "actor_bbox_world_points", "project_world_points_to_bbox",
        "transform_world_to_actor_local", "radar_support_count", "ActorStationaryTracker",
        "project_actor_to_object_row", "build_object_rows"}, dict(base))
    semantic = dict(base, VEHICLE_TAGS={14, 15, 16, 17, 18, 19}, PERSON_TAGS={12, 13},
                    CLASS_VEHICLE=1, CLASS_PERSON=2)
    _functions(ROOT / SOURCES["semantic"][0], {"carla_semantic_tags_to_training_mask"}, semantic)
    static = dict(base, cv2=cv2, json=json, Counter=Counter, cKDTree=cKDTree,
        VIEW=Path("/owned_live_snapshot"), SEMANTIC_NAMES=("Car", "Truck", "Bus"),
        VEHICLE_SEMANTIC_TAGS=frozenset((14, 15, 16)), MAX_RANGE_M=40., MODEL_WIDTH=768,
        MODEL_HEIGHT=432, AMBIGUITY_CENTER_M=1., AMBIGUITY_DIMENSION_M=1., RADAR_MARGIN_M=1.,
        ALGORITHM_VERSION=visibility.ALGORITHM_VERSION,
        decode_depth_bgra=visibility.decode_depth_bgra, depth_is_plausible=visibility.depth_is_plausible,
        visibility_from_corners=visibility.visibility_from_corners)
    _functions(ROOT / SOURCES["static_projection"][0], {
        "finite_float", "numeric_summary", "corners_from_pose", "build_catalog_geometry",
        "actor_ambiguities", "radar_support_count", "intrinsics_from_manifest", "audit_visibility"}, static)
    # Execute the literal original paint block, including its invariants.
    source = ROOT / A.SOURCES["collector"][0]
    tree = ast.parse(source.read_text())
    persist = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_persist")
    start = next(i for i, n in enumerate(persist.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "vehicle_pixels_before" for t in n.targets))
    end = next(i for i in range(start + 1, len(persist.body))
               if isinstance(persist.body[i], ast.If) and "cv2.imwrite" in ast.unparse(persist.body[i].test))
    paint = compile(ast.Module(body=persist.body[start:end], type_ignores=[]), str(source), "exec")
    return objects, semantic, static, visibility, paint


def capture_catalogue(world, output_dir: Path, *, semantic_labels, required_semantic_classes,
                      enabled_state_by_id: Mapping, selection_contract: str):
    """Explicit pre-route operation; caller must supply actual enabled states."""
    verify_sources()
    from data_collection.phase2_static_environment_truth_v1 import capture_static_environment_truth_v1
    return capture_static_environment_truth_v1(world, output_dir, semantic_labels=semantic_labels,
        required_semantic_classes=required_semantic_classes, enabled_state_by_id=enabled_state_by_id,
        selection_contract=selection_contract)


def _load_catalogue(directory: Path):
    from data_collection.phase2_static_environment_truth_v1 import (
        verify_static_environment_truth_v1, OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME, MANIFEST_JSON_NAME)
    verification = verify_static_environment_truth_v1(directory)
    with (directory / OBJECTS_CSV_NAME).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    snapshot = json.loads((directory / SNAPSHOT_JSON_NAME).read_text())
    if set(snapshot["requested_semantic_labels"]) != {"Car", "Truck", "Bus"}:
        raise SourceGeometryError("complete Car/Truck/Bus catalogue required")
    # The original static audit assumes all admitted map objects are enabled.
    # Reject disabled objects rather than silently adapting that frozen rule.
    if any(row["enabled"] != "true" for row in rows):
        raise SourceGeometryError("original static audit requires an all-enabled catalogue")
    hashes = {name: A.sha256(directory / name)
              for name in (OBJECTS_CSV_NAME, SNAPSHOT_JSON_NAME, MANIFEST_JSON_NAME)}
    return rows, snapshot, {"verification": verification, "sha256": hashes}


def _static_rows(original, frame, boxes, geometry, depth_bgra, semantic, radar_world):
    """Execute original audit_visibility unchanged with owned in-memory inputs."""
    namespace = dict(original)
    reads = {str(namespace["VIEW"] / frame["experiment_id"] / "depth" / f"{frame['sample_id']}.png"): depth_bgra,
             str(namespace["VIEW"] / frame["instance_raw_path"]): semantic}
    radar_path = str(namespace["VIEW"] / frame["radar_points_path"])

    class Images:
        IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED
        def imread(self, path, flags):
            if str(path) not in reads:
                raise SourceGeometryError("unregistered original static image read")
            return reads[str(path)]

    class Radar:
        def __enter__(self): return {"world_xyz": radar_world}
        def __exit__(self, *args): return False

    class Arrays:
        def __getattr__(self, name): return getattr(np, name)
        def load(self, path):
            if str(path) != radar_path:
                raise SourceGeometryError("unregistered original static radar read")
            return Radar()

    namespace.update(cv2=Images(), np=Arrays())
    # Each call gets its own globals; no source/module monkey-patch or shared IO.
    import types
    for name, value in list(namespace.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is original:
            namespace[name] = types.FunctionType(value.__code__, namespace, name,
                                                  value.__defaults__, value.__closure__)
    return namespace["audit_visibility"]([frame], {frame["sample_id"]: boxes}, geometry)


class LiveCompleteTargetProducer:
    """Single existing GT worker owns this object and its causal history.

    ``dynamic_population_ids`` is a complete pre-route registered vehicle/person
    population excluding ego. Per-frame evidence supplies active population IDs
    from the exact owned world snapshot (registered actors may despawn) and must
    attest that no unregistered actor was spawned. Missing geometry fails closed.
    """
    def __init__(self, *, static_catalogue_dir: Path, experiment_id: str, ego_id: int,
                 dynamic_population_ids: Sequence[int], population_evidence_sha256: str,
                 max_frames: int):
        self.sources = verify_sources()
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
        self.catalogue, self.catalogue_snapshot, self.catalogue_evidence = _load_catalogue(Path(static_catalogue_dir))
        self.objects, self.semantic, self.static, self.visibility, self.paint = _original()
        self.geometry, _ = self.static["build_catalog_geometry"](self.catalogue)
        self.tracker = self.objects["ActorStationaryTracker"](.35, 5.)
        self.conflict_state = conflicts.IncrementalConflictState(self.catalogue, max_frames=max_frames)
        self.start_timestamp = None
        self.previous_frame = -1
        self.previous_timestamp = -1.
        self.last_evidence = None

    def register_population(self, ids: Sequence[int], evidence_sha256: str, before_frame_id: int):
        """Stage a complete cumulative registry for exactly the next built frame.

        Called on the existing GT worker from immutable owned-tick registry
        evidence, never concurrently from a sensor callback. Despawns remain in
        this cumulative registry but not in the frame's active ID set. The
        outside owner persists the registration document bound by its digest.
        """
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

    def build(self, *, identity: Mapping[str, Any], rgb: Mapping[str, Any],
              semantic_image: Any, depth_image: Any, frozen_world: Any,
              camera_matrix: np.ndarray, camera_inverse: np.ndarray,
              radar_world_xyz: np.ndarray, dynamic_snapshot_evidence: Mapping[str, Any]):
        from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import source_alignment_record, validate_source_alignment
        frame_id, stamp = int(identity["frame_id"]), float(rgb["timestamp"])
        if self.processed_frames >= self.max_frames:
            raise SourceGeometryError("registered complete-target frame budget exhausted")
        registration = self.pending_registration
        if registration is not None and registration["before_frame_id"] != frame_id:
            raise SourceGeometryError("population registration belongs to a different owned frame")
        population = self.population if registration is None else frozenset(registration["population_ids"])
        sync = source_alignment_record(identity=identity, rgb=rgb, semantic_image=semantic_image,
            depth_image=depth_image, frozen_world=frozen_world, policy_timestamp_s=stamp)
        validate_source_alignment(sync)
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
        depth_raw = np.frombuffer(depth_image.raw_data, np.uint8).reshape(720, 1280, 4).copy()
        semantic = np.frombuffer(semantic_image.raw_data, np.uint8).reshape(720, 1280, 4)[..., 2].copy()
        depth = self.visibility.decode_depth_bgra(depth_raw)
        if not self.visibility.depth_is_plausible(depth):
            raise SourceGeometryError("implausible native encoded depth")
        native_f = (1280. / 2) / math.tan(math.radians(120.) / 2)
        k = np.array([[native_f, 0, 640.], [0, native_f, 360.], [0, 0, 1.]])
        sid = f"{self.experiment_id}_frame{frame_id}"
        frame = dict(experiment_id=self.experiment_id, sample_id=sid, frame_id=frame_id,
            timestamp=stamp, camera_width=1280, camera_height=720, camera_fx=k[0, 0], camera_fy=k[1, 1],
            camera_cx=640., camera_cy=360., camera_x=camera_matrix[0, 3], camera_y=camera_matrix[1, 3],
            camera_z=camera_matrix[2, 3], camera_inverse_matrix_json=json.dumps(camera_inverse.tolist()),
            mask_path=f"masks/{sid}.png", instance_raw_path=f"semantic/{sid}.png", radar_points_path=f"radar/{sid}.npz")
        import carla
        boxes = self.objects["build_object_rows"](world=frozen_world, ego_vehicle=SimpleNamespace(id=self.ego_id),
            sample_base={k: frame[k] for k in ("experiment_id", "sample_id", "frame_id", "timestamp")},
            camera_location=carla.Location(*map(float, camera_matrix[:3, 3])),
            camera_matrix=camera_matrix, camera_inverse_matrix=camera_inverse, intrinsics=k,
            width=1280, height=720, max_distance_m=140., radar_world_xyz=radar,
            stationary_tracker=self.tracker, include_pedestrians=True, radar_support_margin_m=1.,
            radar_person_support_mode="radius", radar_person_support_radius_m=1.5,
            radar_person_support_z_down_m=.5, radar_person_support_z_up_m=2.)
        by_id = {int(a.id): a for a in actors}
        visibility, person_masks, corners_evidence = [], [], []
        # Keep all exact owned geometry, including actors the original projector
        # excludes as distant/behind camera. Never retain only localization GT.
        for actor in actors:
            center, corners = self.objects["actor_bbox_world_points"](actor)
            bbox = actor.bounding_box
            corners_evidence.append({"actor_id": int(actor.id), "type_id": str(actor.type_id),
                "center": center.tolist(), "corners": corners.tolist(),
                "matrix": actor.get_transform().get_matrix(),
                "bbox_location": [float(getattr(bbox.location, axis)) for axis in "xyz"],
                "bbox_extent": [float(getattr(bbox.extent, axis)) for axis in "xyz"],
                "velocity": [float(getattr(actor.get_velocity(), axis)) for axis in "xyz"],
                "corner_convention": "ORIGINAL_ACTOR_MATRIX_TIMES_BBOX_LOCATION_PLUS_EXTENT_OFFSETS"})
        for row in boxes:
            actor = by_id[int(row["gt_actor_id"])]
            center, corners = self.objects["actor_bbox_world_points"](actor)
            metrics, support = self.visibility.visibility_from_corners(corners, camera_inverse, k, depth,
                distance_m=float(row["gt_distance_m"]), width=1280, height=720, model_width=768, model_height=432)
            for a, b in (("gt_bbox_x", "clipped_bbox_x"), ("gt_bbox_y", "clipped_bbox_y"),
                         ("gt_bbox_w", "clipped_bbox_w"), ("gt_bbox_h", "clipped_bbox_h")):
                if abs(float(row[a])-float(metrics[b])) > 1e-6:
                    raise SourceGeometryError("original actor box/visibility projection differs")
            vis = {key: row[key] for key in ("experiment_id", "sample_id", "frame_id", "timestamp",
                "gt_actor_id", "label", "gt_actor_type_id", "gt_distance_m")}
            vis.update(metrics, depth_path=f"{self.experiment_id}/depth/{sid}.png", depth_frame_id=frame_id,
                depth_timestamp_s=stamp, person_mask_candidate_px=0, person_mask_painted_px=0,
                frame_person_mask_px=0, low_support_fallback_used=False)
            visibility.append(vis)
            if row["label"] == "person": person_masks.append((vis, support))
        mask = self.semantic["carla_semantic_tags_to_training_mask"](semantic)
        counters = SimpleNamespace(person_mask_subset_failures=0, ineligible_person_paint_events=0,
                                   vehicle_mask_preservation_failures=0)
        scope = dict(np=np, mask=mask, person_masks=person_masks, frame_visibility=visibility,
                     frame_id=frame_id, self=counters, v2=SimpleNamespace(PilotError=SourceGeometryError))
        exec(self.paint, scope)
        static, _ambiguities, static_summary = _static_rows(self.static, frame, boxes, self.geometry, depth_raw, semantic, radar)
        start = stamp if self.start_timestamp is None else self.start_timestamp
        conflict = self.conflict_state.derive(frame=frame, actor_boxes=boxes, actor_visibility=visibility,
            static_candidates=static,
            provenance={"scope": conflicts.SCOPE, "capture_identity": dict(identity),
                "complete": dict.fromkeys(conflicts.COMPLETENESS, True),
                "catalogue_content_sha256": conflicts.catalogue_content_sha256(self.catalogue),
                "history_population": "EVERY_ACCEPTED_REWARD_FRAME_OF_THIS_PRODUCER",
                "history_window": {"start_timestamp": start, "end_timestamp": stamp}})
        evidence = {"identity": dict(identity), "source_synchronization": sync,
            "dynamic_snapshot_evidence": dict(dynamic_snapshot_evidence),
            "population_evidence_sha256": self.population_evidence_sha256,
            "new_population_registration": registration,
            "dynamic_corners": corners_evidence, "static_catalogue": self.catalogue_evidence,
            "static_summary": static_summary, "conflicts": conflict["evidence"], "sources": self.sources,
            "non_target_metadata_note": "Original stationary metadata maintained by a separate GT producer tracker; no policy input is modified."}
        provenance = {"schema": A.SCHEMA, "scope": conflicts.SCOPE, "sample_id": sid,
            "capture_identity": dict(identity),
            "complete": dict.fromkeys(A.COMPLETENESS_KEYS, True),
            "evidence_sha256": {"owned_source": _sha(evidence), "population": self.population_evidence_sha256,
                "catalogue": conflicts.catalogue_content_sha256(self.catalogue), "conflicts": conflict["evidence_sha256"]}}
        inputs = A.CompleteTargetInputs(frame=frame, split="val", depth_bgra=depth_raw, raw_semantic=semantic,
            original_v3_mask=mask, actor_boxes=boxes, actor_visibility=visibility,
            static_candidates=static, quarantine=conflict["quarantine"], provenance=provenance)
        inputs = replace(inputs, provenance=dict(provenance, input_content_sha256=A.input_content_sha256(inputs)))
        A._validate(inputs)
        self.previous_frame, self.start_timestamp, self.last_evidence = frame_id, start, evidence
        self.previous_timestamp = stamp
        if registration is not None:
            self.population = population
            self.population_evidence_sha256 = registration["evidence_sha256"]
            self.pending_registration = None
        self.processed_frames += 1
        return inputs
