"""Complete-target handoff on the existing post-send GT worker.

No new thread, tick owner, GT deadline, or policy input. Installation is local
to one fresh child process and reversible. Static enablement is established
only at the retained route's fresh-load boundary, before dynamic actors exist.
This wiring is CPU-tested, not live-qualified.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import evaluation, source_geometry as S, target_transport as T


def canonical(value):
    return A.canonical(A._json_value(value))


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def dynamic(kind):
    return str(kind).startswith(("vehicle.", "walker.pedestrian."))


def bind_legacy_namespace():
    # The frozen sources intentionally use both the outer model directory and
    # the inner legacy library. Their import order must not select one only.
    import pole_lraspp_multimodal_fusion as package
    outer = Path(__file__).resolve().parents[2] / "pole_lraspp_multimodal_fusion"
    for path in (outer, outer / "pole_lraspp_multimodal_fusion"):
        if not path.is_dir():
            raise ImportError("registered fusion source directory missing")
        if str(path) not in package.__path__:
            package.__path__.append(str(path))


def make_complete_scene_source(base):
    """Refresh on population change BEFORE freezing, not after a stale 2s cache.

    Route B can replenish and delete actors. Bounding boxes/types are cached;
    transforms and velocities always come from this exact owned world tick.
    A population-refresh RPC, if needed, is before preparation/action-open and
    is recorded, not treated as free. No live RPC occurs in the GT worker.
    """
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q

    class OwnedWorld(Q.QualityFrozenWorld):
        __slots__ = ("frozen_timestamp_s", "frozen_observed_raw_ns", "population_evidence")

    class CompleteSource(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._parity_all_ids = set()
            self._parity_population = {}
            self._parity_initialized = False
            self._parity_registry = {}

        def refresh_static(self, *, force=False):
            # Legacy evaluation worker refreshes are redundant. Only startup
            # or owned-tick capture may change our authoritative registry.
            if self._parity_initialized:
                return
            self._refresh_owned()

        def _refresh_owned(self):
            actors = list(self.world.get_actors())
            ids = {int(actor.id) for actor in actors}
            if len(ids) != len(actors) or len(ids | self._parity_all_ids) > 10000:
                raise S.SourceGeometryError("invalid or oversized live actor registry")
            for actor in actors:
                kind, ident = str(actor.type_id), int(actor.id)
                if not dynamic(kind) or ident == self.ego_id:
                    continue
                bbox = actor.bounding_box
                record = {"actor_id": ident, "type_id": kind,
                    "bbox_location": [float(getattr(bbox.location, a)) for a in "xyz"],
                    "bbox_extent": [float(getattr(bbox.extent, a)) for a in "xyz"],
                    "bbox_rotation": [float(getattr(bbox.rotation, a)) for a in ("roll", "pitch", "yaw")]}
                if ident in self._parity_population and self._parity_population[ident] != record:
                    raise S.SourceGeometryError("actor identity/immutable geometry was reused")
                self._parity_population[ident] = record
                self._parity_registry[ident] = (kind, bbox)
            self._parity_all_ids.update(ids)
            self._parity_initialized = True

        def capture(self, snapshot):
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            active_all = {int(item.id) for item in snapshot}
            refreshed = not active_all.issubset(self._parity_all_ids)
            if refreshed:
                self._refresh_owned()
            if not active_all.issubset(self._parity_all_ids):
                raise S.SourceGeometryError("owned world contains unregistered actors")
            if int(self.world.get_snapshot().frame) != int(snapshot.frame):
                raise S.SourceGeometryError("world advanced during population registration")
            actors = []
            for ident in sorted(active_all.intersection(self._parity_registry)):
                kind, bbox = self._parity_registry[ident]
                item = snapshot.find(ident)
                if item is None:
                    raise S.SourceGeometryError("owned snapshot inventory changed")
                actors.append(P.FrozenActor(actor_id=ident, type_id=kind, bounding_box=bbox,
                    transform=item.get_transform(), velocity=item.get_velocity()))
            population = {"schema": "splitfusion.live_exact_population.v1",
                "ego_id": self.ego_id, "registered_actors": list(self._parity_population.values())}
            evidence = {"frame_id": int(snapshot.frame), "registry_complete": True,
                "owned_snapshot_complete": True, "no_unregistered_spawn": True,
                "active_population_ids": [int(a.id) for a in actors],
                "registered_population_ids": sorted(self._parity_registry),
                "owned_snapshot_actor_ids": sorted(active_all),
                "population": population, "population_sha256": sha(population),
                "registry_refreshed": refreshed, "started_raw_ns": start,
                "completed_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
            frozen = OwnedWorld(P.FrozenActorList(actors), frozen_frame_id=int(snapshot.frame))
            frozen.frozen_timestamp_s = float(snapshot.timestamp.elapsed_seconds)
            frozen.frozen_observed_raw_ns = start
            frozen.population_evidence = evidence
            return frozen

    return CompleteSource


class FreshCatalogue:
    def __init__(self, directory):
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError("fresh catalogue must not already exist")
        self.world = None

    def after_load(self, world):
        if self.world is not None:
            raise S.SourceGeometryError("one fresh world/catalogue per attempt")
        actors = list(world.get_actors())
        if any(dynamic(actor.type_id) for actor in actors):
            raise S.SourceGeometryError("static enablement proof requires fresh world before dynamic actors")
        import carla
        labels = [getattr(carla.CityObjectLabel, name) for name in ("Car", "Truck", "Bus")]
        enabled = {}
        for label in labels:
            items = world.get_environment_objects(label)
            if items is None:
                raise S.SourceGeometryError("incomplete fresh static catalogue")
            for item in items:
                if int(item.id) in enabled:
                    raise S.SourceGeometryError("duplicate static object across classes")
                enabled[int(item.id)] = True
        S.capture_catalogue(world, self.directory, semantic_labels=labels,
            required_semantic_classes=["Car"], enabled_state_by_id=enabled,
            selection_contract="FRESH_ROUTE_OWNED_LOAD_BEFORE_DYNAMIC_SPAWN_OR_ENVIRONMENT_TOGGLE_V1")
        self.world = world
        return world

    def require_world(self, world):
        if self.world is not world:
            raise S.SourceGeometryError("collector world not owned by this fresh catalogue")


def write_source_archive(directory, *, inputs, evidence, rgb, radar):
    """Keep actual reconstruction sources, not unverifiable hash-only claims."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    if sha(evidence) != inputs.provenance["evidence_sha256"]["owned_source"]:
        raise S.SourceGeometryError("producer source evidence digest differs")
    image = rgb.get("_rgb_image")
    if image is None or int(image.frame) != int(inputs.frame["frame_id"]):
        raise S.SourceGeometryError("owned native RGB needed for reproducible perception")
    pixels = np.frombuffer(image.raw_data, np.uint8).reshape(int(image.height), int(image.width), 4)
    payloads = {"SOURCE_EVIDENCE.json": canonical(evidence),
        "rgb_bgra.npy": T._array_bytes(pixels), "radar_world_xyz.npy": T._array_bytes(radar)}
    for name, data in payloads.items():
        T._write_x(directory / name, data)
    manifest = {"schema": "splitfusion.live_parity_source_archive.v1",
        "identity": inputs.provenance["capture_identity"],
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    encoded = canonical(manifest)
    T._write_x(directory / "MANIFEST.json", encoded)
    T._fsync_directory(directory)
    return {"directory": str(directory), "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def preserve_complete_targets(records, destination):
    """After worker drain, retain exact bundles and auxiliary source evidence."""
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    kept = []
    for record in records:
        if "complete_target" not in record:
            raise S.SourceGeometryError("AVO-only frame is not a complete repair target")
        desc, source_desc = record["complete_target"], record["complete_sources"]
        root = Path(desc["directory"])
        identity = record["complete_identity"]
        T.verify_full_inputs(root.parent, expected_identity=identity,
                             expected_manifest_sha256=desc["manifest_sha256"])
        source = Path(source_desc["directory"])
        payload = (source / "MANIFEST.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != source_desc["manifest_sha256"]:
            raise S.SourceGeometryError("source archive manifest changed")
        document = json.loads(payload)
        if document["identity"] != identity:
            raise S.SourceGeometryError("source archive identity changed")
        for name, digest in document["sha256"].items():
            if Path(name).name != name or A.sha256(source / name) != digest:
                raise S.SourceGeometryError("source archive payload changed")
        for folder in (root, source):
            # No deletion/move: preserve the owned scratch originals until normal teardown.
            shutil.copytree(folder, destination / folder.name)
        T.verify_full_inputs(destination, expected_identity=identity,
                             expected_manifest_sha256=desc["manifest_sha256"])
        kept.append({"identity": identity, "manifest_sha256": desc["manifest_sha256"],
                     "source_manifest_sha256": source_desc["manifest_sha256"]})
    T._write_x(destination / "MANIFEST.json", canonical({"schema": "splitfusion.complete_target_preservation.v1",
                                                       "frames": kept}))
    return {"directory": str(destination), "frames": len(kept), "complete": True}


@contextmanager
def install_complete_gt(*, attempt_dir: Path, max_frames: int):
    """Install before retained Run5 child installs its old AVO collector.

    No live launch authority is implied. The new entry must still receive a
    separately sealed qualification configuration before executing a route.
    """
    if type(max_frames) is not int or not 1 <= max_frames <= 1000:
        raise ValueError("registered 1..1000 frame bound required")
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    bind_legacy_namespace()
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from data_collection.run_route_b_perception_collection_v2 import ClientProxy
    if LG._INSTALLED:
        raise RuntimeError("complete GT must install in a fresh child before legacy AVO")
    original = (LG.build_live_gt_collector, LG.build_timestamped_scene_source,
                LG.write_avo_bundle, LG.preserve_avo_evidence)
    sentinel = object()
    old_load = ClientProxy.__dict__.get("load_world", sentinel)
    catalogue = FreshCatalogue(Path(attempt_dir) / "parity_static_catalogue")
    local = threading.local()

    def load_world(client, *args, **kwargs):
        return catalogue.after_load(client._client.load_world(*args, **kwargs))

    def source(base, frozen):
        return make_complete_scene_source(original[1](base, frozen))

    def collector(base):
        inherited = original[0](base)
        class CompleteCollector(inherited):
            def _spawn_sensors(self):
                catalogue.require_world(self.world)
                self._complete_producer = None
                return super()._spawn_sensors()

            def _ground_truth(self, **kwargs):
                if getattr(local, "collector", None) is not None:
                    raise S.SourceGeometryError("nested GT worker ownership")
                local.collector, local.kwargs = self, kwargs
                try:
                    return super()._ground_truth(**kwargs)
                finally:
                    local.collector = local.kwargs = None
        return CompleteCollector

    def write(directory, **kwargs):
        owner = getattr(local, "collector", None)
        if owner is None:
            raise S.SourceGeometryError("complete target outside existing GT worker")
        frame = int(kwargs["identity"]["frame_id"])
        if owner.live.last_datagram.get(frame) is None:
            raise S.SourceGeometryError("complete GT started before send completion")
        frozen = kwargs["frozen_world"]
        population = frozen.population_evidence
        if frame != population["frame_id"]:
            raise S.SourceGeometryError("population evidence belongs to another frame")
        radar = local.kwargs.get("radar_points", {}).get("world_xyz")
        if radar is None:
            raise S.SourceGeometryError("complete GT requires owned world radar points")
        if owner._complete_producer is None:
            owner._complete_producer = S.LiveCompleteTargetProducer(
                static_catalogue_dir=catalogue.directory,
                experiment_id="live_parity_" + Path(attempt_dir).name,
                ego_id=int(owner.ego.id), dynamic_population_ids=population["registered_population_ids"],
                population_evidence_sha256=population["population_sha256"], max_frames=max_frames)
        producer = owner._complete_producer
        if producer.population != frozenset(population["registered_population_ids"]):
            producer.register_population(population["registered_population_ids"],
                population["population_sha256"], before_frame_id=frame)
        # Original AVO threshold/normalization and causal sources are unchanged.
        record = original[2](directory, **kwargs)
        stage = "complete_source_build"
        target_dir = T.bundle_path(directory, kwargs["identity"])
        source_dir = target_dir.with_suffix(".sources")
        try:
            inputs = producer.build(**{k: kwargs[k] for k in ("identity", "rgb", "semantic_image", "depth_image",
                "frozen_world", "camera_matrix", "camera_inverse")},
                radar_world_xyz=radar, dynamic_snapshot_evidence=population)
            stage = "prepare_target"
            prepared = evaluation.prepare_target(inputs)
            stage = "publish_target"
            descriptor = T.write_bundle(directory, identity=kwargs["identity"], inputs=inputs, prepared=prepared)
            # Auxiliary RGB/radar archival is not needed by the evaluator.
            # Publish the complete scoring target before these extra writes.
            stage = "archive_sources"
            sources = write_source_archive(source_dir, inputs=inputs, evidence=producer.last_evidence,
                                           rgb=kwargs["rgb"], radar=radar)
        except Exception as exc:
            # LG's existing finish now retains the successful AVO inputs even
            # if a later target stage failed. Partial target files stay marked
            # incomplete, and remain inspectable after scratch teardown.
            exc.avo_evidence_record = {**record, "error": f"{type(exc).__name__}: {exc}"}
            failed = Path(attempt_dir) / "parity_rejected_targets" / target_dir.name
            failed.mkdir(parents=True, exist_ok=False)
            for folder in (target_dir, source_dir):
                if folder.exists():
                    shutil.copytree(folder, failed / folder.name)
            native = {"rgb_bgra.npy": np.frombuffer(kwargs["rgb"]["_rgb_image"].raw_data, np.uint8).reshape(720,1280,4),
                "depth_bgra.npy": np.frombuffer(kwargs["depth_image"].raw_data, np.uint8).reshape(720,1280,4),
                "raw_semantic_bgra.npy": np.frombuffer(kwargs["semantic_image"].raw_data, np.uint8).reshape(720,1280,4),
                "radar_world_xyz.npy": radar}
            for name, array in native.items():
                T._write_x(failed / name, T._array_bytes(array))
            source_evidence = producer.last_evidence
            if not source_evidence or source_evidence.get("identity") != kwargs["identity"]:
                source_evidence = {
                "identity": kwargs["identity"], "population": population,
                "camera_matrix": np.asarray(kwargs["camera_matrix"]).tolist(),
                "camera_inverse": np.asarray(kwargs["camera_inverse"]).tolist(),
                "incomplete_current_source": True, "failed_stage": stage}
            T._write_x(failed / "SOURCE_EVIDENCE.json", canonical(source_evidence))
            files = {str(path.relative_to(failed)): A.sha256(path) for path in sorted(failed.rglob("*")) if path.is_file()}
            T._write_x(failed / "REJECTION.json", canonical({"identity": kwargs["identity"],
                "stage": stage, "error": str(exc), "avo_record": record, "sha256": files,
                "target_ready": (target_dir / T.READY_NAME).is_file(), "live_qualified": False}))
            raise
        return {**record, "complete_identity": dict(kwargs["identity"]),
                "complete_target": descriptor, "complete_sources": sources}

    def preserve(records, directory, **kwargs):
        result = original[3](records, directory, **kwargs)
        targets = preserve_complete_targets(records, Path(directory) / "complete_targets")
        return {**result, "complete_targets": targets}

    ClientProxy.load_world = load_world
    LG.build_live_gt_collector, LG.build_timestamped_scene_source = collector, source
    LG.write_avo_bundle, LG.preserve_avo_evidence = write, preserve
    try:
        yield {"catalogue": catalogue, "new_threads": 0, "live_qualified": False}
    finally:
        (LG.build_live_gt_collector, LG.build_timestamped_scene_source,
         LG.write_avo_bundle, LG.preserve_avo_evidence) = original
        if old_load is sentinel:
            del ClientProxy.load_world
        else:
            ClientProxy.load_world = old_load
