"""Strictly post-route materialization of raw CARLA evidence.

The live Route-B bridge writes primitive actor/camera/radar state and raw
semantic bytes only.  This module verifies that sealed spool and, after the
route has stopped, calls the authoritative object and semantic GT helpers to
create :class:`GroundTruthEvidenceStoreV1` records.  It computes neither
Q_perc nor reward; the existing post-run evaluator owns those operations.

Missing scene or semantic evidence is reported rather than fabricated.  The
post-run evaluator uses the durable operational ledger as its population, so
missing GT never removes a success or timeout row.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .operational_ack_v1 import FrameActionIdentityV1
from .postrun_artifact_v1 import GroundTruthEvidenceStoreV1


RAW_SPOOL_SCHEMA = "scenesense.splitfusion.b.raw_gt_spool.v3"
REPORT_SCHEMA = "scenesense.splitfusion.run4b5b.postroute_gt_materialization.v1"
CLAIM_SCOPE = "OFFLINE_CARLA_RESEARCH_EVALUATION_NOT_LIVE_POLICY_FEEDBACK"

MODEL_WIDTH, MODEL_HEIGHT = 768, 448
CAMERA_FOV_DEG = 120.0
BUILDER_DISTANCE_M, ELIGIBLE_DISTANCE_M = 140.0, 40.0
MIN_ELIGIBLE_AREA_PX = 12.0
STATIONARY_VELOCITY_MPS, PARKED_THRESHOLD_S = 0.35, 5.0
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class PostRouteMaterializationError(RuntimeError):
    pass


class RawSpoolIntegrityError(PostRouteMaterializationError):
    pass


class MaterializationCreateOnlyError(PostRouteMaterializationError):
    pass


def _require(value: bool, message: str,
             error: type[PostRouteMaterializationError] =
             RawSpoolIntegrityError) -> None:
    if not value:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RawSpoolIntegrityError("value is not canonicalizable") from exc


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, name: str) -> str:
    _require(type(value) is str and bool(_SHA_RE.fullmatch(value)),
             f"{name} is not a lowercase SHA-256")
    return value


def _fields(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    _require(type(raw) is dict and set(raw) == expected,
             f"{name} fields are incomplete or foreign")


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(),
             f"JSON is missing or a symlink: {path.name}")
    payload = path.read_bytes()
    try:
        raw = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawSpoolIntegrityError(f"invalid JSON: {path.name}") from exc
    _require(type(raw) is dict, f"JSON root is not a mapping: {path.name}")
    _require(payload == _canonical(raw) + b"\n",
             f"JSON is not canonical: {path.name}")
    return raw, payload


def _vector(raw: Any, name: str) -> dict[str, float]:
    _require(type(raw) is dict and set(raw) == {"x", "y", "z"},
             f"{name} is not xyz")
    result = {key: float(raw[key]) for key in ("x", "y", "z")}
    _require(all(math.isfinite(value) for value in result.values()),
             f"{name} contains a non-finite value")
    return result


def _rotation(raw: Any, name: str) -> dict[str, float]:
    _require(type(raw) is dict and set(raw) == {"pitch", "yaw", "roll"},
             f"{name} is not a rotation")
    result = {key: float(raw[key]) for key in ("pitch", "yaw", "roll")}
    _require(all(math.isfinite(value) for value in result.values()),
             f"{name} contains a non-finite value")
    return result


def _name(frame: int, suffix: str) -> str:
    return f"{frame:010d}.{suffix}"


@dataclass(frozen=True, slots=True)
class _Spool:
    root: Path
    manifest_sha256: str
    frames: Mapping[str, tuple[int, ...]]
    inventoried: frozenset[str]


@dataclass(frozen=True, slots=True)
class _SceneFrame:
    identity: FrameActionIdentityV1
    scene: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PostRouteMaterializationResultV1:
    ground_truth_root: Path
    report_path: Path
    identity_frame_count: int
    record_count: int
    missing_count: int
    orphan_scene_count: int
    orphan_semantic_count: int


def _verify_spool(root: Path) -> _Spool:
    root = Path(root)
    _require(root.is_dir() and not root.is_symlink(),
             "raw spool root is missing or a symlink")
    for part in ("identity", "scene", "semantic"):
        directory = root / part
        _require(directory.is_dir() and not directory.is_symlink(),
                 f"raw {part} directory is missing or a symlink")
    manifest, manifest_bytes = _json(root / "MANIFEST.json")
    _fields(manifest, {"schema", "status", "counts", "frames", "files"},
            "raw manifest")
    _require(manifest["schema"] == RAW_SPOOL_SCHEMA, "raw schema drift")
    _require(manifest["status"] == "SEALED", "raw spool is not sealed")
    kinds = {"identity", "scene", "semantic"}
    _require(type(manifest["counts"]) is dict
             and set(manifest["counts"]) == kinds,
             "raw counts are incomplete or foreign")
    _require(type(manifest["frames"]) is dict
             and set(manifest["frames"]) == kinds,
             "raw frame sets are incomplete or foreign")
    frames: dict[str, tuple[int, ...]] = {}
    for kind in sorted(kinds):
        values = manifest["frames"][kind]
        _require(type(values) is list
                 and all(type(item) is int and item >= 0 for item in values)
                 and values == sorted(set(values)),
                 f"raw {kind} frame set is invalid")
        _require(manifest["counts"][kind] == len(values),
                 f"raw {kind} count differs from frame set")
        frames[kind] = tuple(values)

    entries = manifest["files"]
    _require(type(entries) is list, "raw file inventory is not a list")
    inventoried: set[str] = set()
    for entry in entries:
        _fields(entry, {"path", "bytes", "sha256"}, "raw file entry")
        relative = entry["path"]
        _require(type(relative) is str, "raw file path is not text")
        pure = PurePosixPath(relative)
        _require(not pure.is_absolute() and ".." not in pure.parts
                 and len(pure.parts) == 2 and pure.parts[0] in kinds,
                 "raw file path is unsafe or foreign")
        _require(relative not in inventoried, "duplicate raw file inventory path")
        inventoried.add(relative)
        path = root / relative
        _require(path.is_file() and not path.is_symlink(),
                 f"inventoried raw file is missing or a symlink: {relative}")
        payload = path.read_bytes()
        _require(type(entry["bytes"]) is int and entry["bytes"] == len(payload),
                 f"raw file byte count mismatch: {relative}")
        _require(_digest(entry["sha256"], "raw file sha256") == _sha(payload),
                 f"raw file digest mismatch: {relative}")
    actual: set[str] = set()
    for part in kinds:
        for path in (root / part).iterdir():
            _require(path.is_file() and not path.is_symlink(),
                     f"foreign raw entry: {part}/{path.name}")
            actual.add(path.relative_to(root).as_posix())
    _require(actual == inventoried,
             "raw spool has an unlisted or missing inventoried file")
    return _Spool(root, _sha(manifest_bytes), frames, frozenset(inventoried))


def _relative(part: str, frame: int, suffix: str) -> str:
    return f"{part}/{_name(frame, suffix)}"


def _missing_component(spool: _Spool, frame: int, part: str) -> tuple[str, ...]:
    suffixes = {
        "identity": ("json", "sha256"),
        "scene": ("json", "sha256", "npz"),
        "semantic": ("json", "bgra"),
    }[part]
    return tuple(path for path in
                 (_relative(part, frame, suffix) for suffix in suffixes)
                 if path not in spool.inventoried)


def _sidecar(path: Path, expected: str) -> None:
    _require(path.read_bytes() == (expected + "\n").encode("ascii"),
             f"digest sidecar mismatch: {path.name}")


def _load_identity(spool: _Spool, frame: int) -> FrameActionIdentityV1:
    path = spool.root / "identity" / _name(frame, "json")
    raw, payload = _json(path)
    _fields(raw, {"schema", "kind", "frame_id", "identity",
                  "identity_sha256"}, "raw identity")
    _require(raw["schema"] == RAW_SPOOL_SCHEMA and raw["kind"] == "identity"
             and raw["frame_id"] == frame, "raw identity envelope drift")
    identity = FrameActionIdentityV1.from_mapping(raw["identity"])
    _require(identity.frame_id == frame, "identity frame differs from filename")
    _require(raw["identity_sha256"] == identity.exact_sha256(),
             "identity digest mismatch")
    _sidecar(spool.root / "identity" / _name(frame, "sha256"), _sha(payload))
    return identity


def _load_scene(spool: _Spool, frame: int,
                identity: FrameActionIdentityV1) -> _SceneFrame:
    path = spool.root / "scene" / _name(frame, "json")
    raw, payload = _json(path)
    _fields(raw, {"schema", "kind", "frame_id", "carla_timestamp",
                  "camera_location", "actors", "array_file", "array_sha256"},
            "raw scene")
    _require(raw["schema"] == RAW_SPOOL_SCHEMA and raw["kind"] == "scene"
             and raw["frame_id"] == frame, "raw scene envelope drift")
    _require(isinstance(raw["carla_timestamp"], (int, float))
             and math.isfinite(float(raw["carla_timestamp"])),
             "raw scene timestamp is invalid")
    _vector(raw["camera_location"], "camera_location")
    _require(type(raw["actors"]) is list, "raw actors are not a list")
    actor_ids: list[int] = []
    for actor in raw["actors"]:
        _fields(actor, {"actor_id", "type_id", "bbox_location", "bbox_extent",
                        "bbox_rotation", "transform", "velocity"}, "raw actor")
        _require(type(actor["actor_id"]) is int and actor["actor_id"] >= 0,
                 "raw actor id is invalid")
        _require(type(actor["type_id"]) is str
                 and (actor["type_id"].startswith("vehicle.")
                      or actor["type_id"].startswith("walker.pedestrian.")),
                 "raw actor type is foreign")
        _vector(actor["bbox_location"], "bbox_location")
        extent = _vector(actor["bbox_extent"], "bbox_extent")
        _require(all(value >= 0.0 for value in extent.values()),
                 "raw bbox extent is negative")
        _rotation(actor["bbox_rotation"], "bbox_rotation")
        transform = actor["transform"]
        _require(type(transform) is dict
                 and set(transform) == {"location", "rotation"},
                 "raw actor transform is invalid")
        _vector(transform["location"], "actor location")
        _rotation(transform["rotation"], "actor rotation")
        _vector(actor["velocity"], "actor velocity")
        actor_ids.append(actor["actor_id"])
    _require(actor_ids == sorted(set(actor_ids)),
             "raw actors are duplicate or out of captured order")
    npz_name = _name(frame, "npz")
    _require(raw["array_file"] == npz_name,
             "scene array filename differs from frame")
    npz_path = spool.root / "scene" / npz_name
    _require(_digest(raw["array_sha256"], "scene array sha256")
             == _sha(npz_path.read_bytes()), "scene array digest mismatch")
    _sidecar(spool.root / "scene" / _name(frame, "sha256"), _sha(payload))
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            _require(set(archive.files) ==
                     {"camera_matrix", "camera_inverse", "radar_world_xyz"},
                     "scene arrays are incomplete or foreign")
            arrays = {key: np.asarray(archive[key]).copy()
                      for key in archive.files}
    except (OSError, ValueError) as exc:
        raise RawSpoolIntegrityError("scene array archive is invalid") from exc
    _require(arrays["camera_matrix"].shape == (4, 4)
             and arrays["camera_matrix"].dtype == np.float64,
             "camera_matrix dtype or shape drift")
    _require(arrays["camera_inverse"].shape == (4, 4)
             and arrays["camera_inverse"].dtype == np.float64,
             "camera_inverse dtype or shape drift")
    radar = arrays["radar_world_xyz"]
    _require(radar.dtype == np.float32 and radar.ndim == 2
             and radar.shape[1] == 3,
             "radar_world_xyz dtype or shape drift")
    _require(all(np.all(np.isfinite(value)) for value in arrays.values()),
             "scene arrays contain a non-finite value")
    return _SceneFrame(identity, raw, arrays)


class _SemanticImage:
    def __init__(self, raw: bytes, metadata: Mapping[str, Any]) -> None:
        self.raw_data = raw
        self.frame = int(metadata["sensor_frame"])
        self.timestamp = float(metadata["carla_timestamp"])
        self.width = int(metadata["width"])
        self.height = int(metadata["height"])


def _load_semantic(spool: _Spool, frame: int) -> _SemanticImage:
    raw, _payload = _json(spool.root / "semantic" / _name(frame, "json"))
    _fields(raw, {"schema", "kind", "frame_id", "sensor_frame",
                  "carla_timestamp", "width", "height", "raw_file",
                  "raw_sha256"}, "raw semantic")
    _require(raw["schema"] == RAW_SPOOL_SCHEMA
             and raw["kind"] == "semantic_raw_bgra"
             and raw["frame_id"] == frame and raw["sensor_frame"] == frame,
             "raw semantic envelope drift")
    _require(type(raw["width"]) is int and raw["width"] > 0
             and type(raw["height"]) is int and raw["height"] > 0,
             "semantic dimensions are invalid")
    _require(isinstance(raw["carla_timestamp"], (int, float))
             and math.isfinite(float(raw["carla_timestamp"])),
             "semantic timestamp is invalid")
    filename = _name(frame, "bgra")
    _require(raw["raw_file"] == filename,
             "semantic filename differs from frame")
    payload = (spool.root / "semantic" / filename).read_bytes()
    _require(len(payload) == raw["width"] * raw["height"] * 4,
             "semantic byte count differs from dimensions")
    _require(_digest(raw["raw_sha256"], "semantic sha256") == _sha(payload),
             "semantic digest mismatch")
    return _SemanticImage(payload, raw)


class _Ego:
    id = -1


def _load_authoritative_modules() -> tuple[Any, Any, Any, Any]:
    """Use the exact namespace seam installed by the pinned Route-B tests."""

    import carla
    import pole_lraspp_multimodal_fusion as fusion_namespace

    legacy = (Path(__file__).resolve().parents[2]
              / "pole_lraspp_multimodal_fusion"
              / "pole_lraspp_multimodal_fusion").resolve()
    paths = {str(Path(value).resolve()) for value in fusion_namespace.__path__}
    if str(legacy) not in paths:
        fusion_namespace.__path__.append(str(legacy))
    import carla_collect_parked_ego_fusion_training_data as parked
    from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import (
        valid_localization_objects,
    )
    from rl_agent import ue_route_b_split_cell_adapter_v1 as route
    return carla, parked, valid_localization_objects, route


def _world(scene: Mapping[str, Any], carla: Any, route: Any) -> Any:
    actors = []
    for raw in scene["actors"]:
        bbox = carla.BoundingBox(
            carla.Location(**_vector(raw["bbox_location"], "bbox_location")),
            carla.Vector3D(**_vector(raw["bbox_extent"], "bbox_extent")),
        )
        bbox.rotation = carla.Rotation(
            **_rotation(raw["bbox_rotation"], "bbox_rotation"))
        actors.append(route.FrozenActor(
            actor_id=raw["actor_id"], type_id=raw["type_id"],
            bounding_box=bbox,
            transform=carla.Transform(
                carla.Location(**_vector(raw["transform"]["location"],
                                         "actor location")),
                carla.Rotation(**_rotation(raw["transform"]["rotation"],
                                           "actor rotation")),
            ),
            velocity=carla.Vector3D(
                **_vector(raw["velocity"], "actor velocity")),
        ))
    return route.FrozenWorld(route.FrozenActorList(actors))


def _exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise MaterializationCreateOnlyError(
            f"create-only output exists: {path}") from exc


class PostRouteGroundTruthMaterializerV1:
    """Verify a sealed raw spool and create its post-run GT subset."""

    def __init__(self) -> None:
        self.last_result: PostRouteMaterializationResultV1 | None = None

    @staticmethod
    def report_path(evidence_root: Path) -> Path:
        root = Path(evidence_root)
        return root.with_name(f"{root.name}.materialization.json")

    def __call__(self, spool_root: Path, evidence_root: Path) -> int:
        return self.materialize(spool_root=spool_root,
                                evidence_root=evidence_root).record_count

    def materialize(self, *, spool_root: Path,
                    evidence_root: Path) -> PostRouteMaterializationResultV1:
        spool = _verify_spool(Path(spool_root))
        target = Path(evidence_root)
        report_path = self.report_path(target)
        if target.exists() or report_path.exists():
            raise MaterializationCreateOnlyError(
                "post-route GT store or report already exists")
        _require(target.parent.is_dir(), "post-route GT parent is absent")

        identity_frames = spool.frames["identity"]
        scene_set, semantic_set = (set(spool.frames["scene"]),
                                   set(spool.frames["semantic"]))
        identities: dict[int, FrameActionIdentityV1] = {}
        missing: list[dict[str, Any]] = []
        for frame in identity_frames:
            absent_identity = _missing_component(spool, frame, "identity")
            _require(not absent_identity,
                     f"claimed identity files are incomplete for frame {frame}")
            identities[frame] = _load_identity(spool, frame)
            reasons: list[str] = []
            if frame not in scene_set:
                reasons.append("SCENE_NOT_CAPTURED")
            elif _missing_component(spool, frame, "scene"):
                reasons.append("SCENE_FILES_INCOMPLETE")
            if frame not in semantic_set:
                reasons.append("SEMANTIC_NOT_CAPTURED")
            elif _missing_component(spool, frame, "semantic"):
                reasons.append("SEMANTIC_FILES_INCOMPLETE")
            if reasons:
                missing.append({"frame_id": frame, "reasons": reasons})

        store = GroundTruthEvidenceStoreV1.create(target)
        records = []
        complete_frames: list[int] = []
        # Actor stationary state advances over every transmitted frame with a
        # scene, even if that frame lacks semantic evidence.  This matches the
        # authoritative object's independent evaluation worker.
        scene_frames = [frame for frame in identity_frames if frame in scene_set
                        and not _missing_component(spool, frame, "scene")]
        if scene_frames:
            carla, parked, valid_objects, route = _load_authoritative_modules()
            tracker = parked.ActorStationaryTracker(
                STATIONARY_VELOCITY_MPS, PARKED_THRESHOLD_S)
            intrinsics = route.camera_intrinsics(
                MODEL_WIDTH, MODEL_HEIGHT, CAMERA_FOV_DEG)
            for frame in scene_frames:
                value = _load_scene(spool, frame, identities[frame])
                rows = parked.build_object_rows(
                    world=_world(value.scene, carla, route),
                    ego_vehicle=_Ego(),
                    sample_base={"timestamp": float(value.scene["carla_timestamp"]),
                                 "frame_id": frame},
                    camera_location=carla.Location(
                        **_vector(value.scene["camera_location"],
                                  "camera_location")),
                    camera_matrix=value.arrays["camera_matrix"],
                    camera_inverse_matrix=value.arrays["camera_inverse"],
                    intrinsics=intrinsics,
                    width=MODEL_WIDTH, height=MODEL_HEIGHT,
                    max_distance_m=BUILDER_DISTANCE_M,
                    radar_world_xyz=value.arrays["radar_world_xyz"],
                    stationary_tracker=tracker,
                    include_pedestrians=True,
                    radar_support_margin_m=1.0,
                    radar_person_support_mode="radius",
                    radar_person_support_radius_m=1.5,
                    radar_person_support_z_down_m=0.5,
                    radar_person_support_z_up_m=2.0,
                )
                eligible = valid_objects(
                    rows, image_width=MODEL_WIDTH, image_height=MODEL_HEIGHT,
                    min_area_px=MIN_ELIGIBLE_AREA_PX,
                    max_distance_m=ELIGIBLE_DISTANCE_M,
                )
                if (frame not in semantic_set
                        or _missing_component(spool, frame, "semantic")):
                    continue
                semantic = _load_semantic(spool, frame)
                mask = route.semantic_gt_3class(semantic)
                records.append(store.write(
                    identity=identities[frame], eligible_objects=eligible,
                    semantic_mask=mask,
                    recorded_monotonic_raw_ns=time.clock_gettime_ns(
                        time.CLOCK_MONOTONIC_RAW),
                ))
                complete_frames.append(frame)
        verified = store.verify_all()
        _require(tuple(records) == verified,
                 "materialized GT store did not verify exactly")

        identity_set = set(identity_frames)
        report = {
            "schema": REPORT_SCHEMA,
            "status": ("COMPLETE_WITH_ALL_GT" if not missing else
                       "COMPLETE_WITH_MISSING_GT_REPORTED"),
            "claim_scope": CLAIM_SCOPE,
            "materialization_phase": "AFTER_ROUTE_STOP_ONLY",
            "live_policy_feedback": False,
            "live_qperc_computed": False,
            "raw_spool_schema": RAW_SPOOL_SCHEMA,
            "raw_spool_manifest_sha256": spool.manifest_sha256,
            "ground_truth_store_name": target.name,
            "identity_frame_count": len(identity_frames),
            "scene_frame_count": len(scene_set),
            "semantic_frame_count": len(semantic_set),
            "materialized_record_count": len(verified),
            "complete_frames": complete_frames,
            "missing_identity_frames": missing,
            "orphan_scene_frames": sorted(scene_set - identity_set),
            "orphan_semantic_frames": sorted(semantic_set - identity_set),
            "authoritative_sources": {
                "objects": "carla_collect_parked_ego_fusion_training_data.build_object_rows",
                "object_eligibility": "pole_lraspp_multimodal_fusion.object_targets.valid_localization_objects",
                "semantic": "rl_agent.ue_route_b_split_cell_adapter_v1.semantic_gt_3class",
                "qperc": "postrun_operational_population_v1:score_serial/evaluate_exact_quality",
            },
            "fixed_parameters": {
                "model_width": MODEL_WIDTH, "model_height": MODEL_HEIGHT,
                "camera_fov_deg": CAMERA_FOV_DEG,
                "builder_distance_m": BUILDER_DISTANCE_M,
                "eligible_distance_m": ELIGIBLE_DISTANCE_M,
                "min_eligible_area_px": MIN_ELIGIBLE_AREA_PX,
                "stationary_velocity_mps": STATIONARY_VELOCITY_MPS,
                "parked_threshold_s": PARKED_THRESHOLD_S,
            },
            "population_rule": (
                "GT enriches durable operational outcomes; missing GT never "
                "removes a success or timeout row"
            ),
        }
        _exclusive(report_path, _canonical(report) + b"\n")
        result = PostRouteMaterializationResultV1(
            target, report_path, len(identity_frames), len(verified), len(missing),
            len(scene_set - identity_set), len(semantic_set - identity_set))
        self.last_result = result
        return result


def materialize_postroute_ground_truth(spool_root: Path,
                                       evidence_root: Path) -> int:
    return PostRouteGroundTruthMaterializerV1()(spool_root, evidence_root)


__all__ = [
    "RAW_SPOOL_SCHEMA", "REPORT_SCHEMA", "CLAIM_SCOPE",
    "PostRouteMaterializationError", "RawSpoolIntegrityError",
    "MaterializationCreateOnlyError", "PostRouteMaterializationResultV1",
    "PostRouteGroundTruthMaterializerV1", "materialize_postroute_ground_truth",
]
