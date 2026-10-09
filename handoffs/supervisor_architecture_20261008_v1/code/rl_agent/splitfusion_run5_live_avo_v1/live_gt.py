"""Exact-frame privileged live depth/AVO producer; no action-path GT thread.

The frozen collector already snapshots actor geometry at each owned world tick.
This seam consumes that snapshot on its EXISTING object-GT worker, after submit
has sent the last datagram. Depth is a new evaluation-only sensor; it must be
qualified live before a policy performance capture. Masks are corrected raw
CARLA labels, explicitly NOT the offline FCOS-v010 target-mask construction.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .quality_adapter import LiveGtContractError, corrected_raw_semantic_3class, score_live_avo_people
from .gt_source_sync import (LiveSyncEvidence, sensor_metadata, source_alignment_record,
                             validate_source_alignment, SourceSynchronizationError, SYNC_SCHEMA)

SCHEMA = "splitfusion.run5.live_corrected_semantic_original_avo.v1"
MASK_CONTRACT = "LIVE_CORRECTED_RAW_CARLA_3CLASS_NOT_OFFLINE_V010"
MEASUREMENT_SEMANTICS = (MASK_CONTRACT + "__ORIGINAL_UNNORMALIZED_AVO_GE_0_65__"
                         "LIVE_VEHICLE_LOCALIZATION_UNCHANGED")
_INSTALLED = False


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_depth(image: Any) -> np.ndarray:
    """CARLA native BGRA depth: (R + 256G + 65536B)/(2^24-1)*1000 m."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    values = bgra.astype(np.float32, copy=False)
    blue, green, red = values[..., 0], values[..., 1], values[..., 2]
    normalized = (red + green * 256.0 + blue * 256.0 * 256.0) / (256.0**3 - 1.0)
    return (1000. * normalized).astype(np.float32, copy=False)


def raw_semantic(image: Any) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[..., 2].copy()


def corrected_semantic_image(image: Any) -> np.ndarray:
    return corrected_raw_semantic_3class(raw_semantic(image))


def native_intrinsics(width: int, height: int, fov: float = 120.) -> np.ndarray:
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])


def frozen_people_rows(world: Any, *, frame_id: int, timestamp: float,
                       camera_matrix: np.ndarray, camera_inverse: np.ndarray,
                       intrinsics: np.ndarray, width: int, height: int) -> list[dict]:
    """All frozen pedestrian volumes, before localization/visibility filtering.

    Construction is geometry-only (no live CARLA RPC, no tracker or radar work).
    Nonprojected volumes remain structural ignores and competing volumes.
    """
    from data_collection.route_b_publication_actor_volume_visibility_v1 import core

    if int(world.frozen_frame_id) != frame_id:
        raise LiveGtContractError("frozen world frame differs from depth")
    out = []
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        transform, bbox = actor.get_transform(), actor.bounding_box
        matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
        centre = (matrix @ np.array([bbox.location.x, bbox.location.y, bbox.location.z, 1.]))[:3]
        extent = (float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z))
        sensor = (camera_inverse @ np.r_[centre, 1.])[:3]
        row = dict(label="person", gt_actor_id=str(actor.id), frame_id=frame_id,
                   timestamp=timestamp, object_yaw_deg=float(transform.rotation.yaw),
                   object_pitch_deg=float(transform.rotation.pitch),
                   object_roll_deg=float(transform.rotation.roll),
                   gt_distance_m=float(np.linalg.norm(matrix[:3, 3] - camera_matrix[:3, 3])),
                   object_sensor_x=float(sensor[0]), gt_center_x=-1., gt_center_y=-1.,
                   gt_bbox_area_px=0.)
        row.update({f"object_world_{axis}": float(centre[i]) for i, axis in enumerate("xyz")})
        row.update({f"gt_extent_{axis}_m": extent[i] for i, axis in enumerate("xyz")})
        try:
            box = core.projected_boxes(core.oriented_box_corners(centre, extent, row["object_yaw_deg"]),
                                       camera_inverse, intrinsics, width=width, height=height)
        except ValueError:
            box = None
        if box is not None:
            row.update(gt_center_x=box["clipped_bbox_x"] + box["clipped_bbox_w"] / 2,
                       gt_center_y=box["clipped_bbox_y"] + box["clipped_bbox_h"] / 2,
                       gt_bbox_area_px=box["clipped_projected_area_px"])
        out.append(row)
    return out


def sidecar_path(directory: Path, identity: Mapping[str, Any]) -> Path:
    from rl_agent.splitfusion_quality_feedback_probe_v1.gt_evidence import _stem
    return Path(directory) / (_stem(str(identity["stream_id"]), int(identity["frame_id"])) + ".avo.json")


def write_avo_bundle(directory: Path, *, identity: Mapping[str, Any], rgb: Mapping[str, Any],
                     semantic_image: Any, depth_image: Any, frozen_world: Any,
                     camera_matrix: np.ndarray, camera_inverse: np.ndarray,
                     policy_timestamp_s: float, sync_evidence: LiveSyncEvidence | None = None) -> dict:
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G

    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    frame_id = int(identity["frame_id"])
    synchronization = source_alignment_record(identity=identity, rgb=rgb,
        semantic_image=semantic_image, depth_image=depth_image, frozen_world=frozen_world,
        policy_timestamp_s=policy_timestamp_s)
    if sync_evidence is None:
        validate_source_alignment(synchronization)
    else:
        sync_evidence.check_and_record(synchronization, raw_sources={
            "rgb": rgb.get("_rgb_image"), "semantic": semantic_image, "depth": depth_image})
    for image in (semantic_image, depth_image):
        if int(image.frame) != frame_id or float(image.timestamp) != float(rgb["timestamp"]):
            raise LiveGtContractError("RGB/semantic/depth frame or timestamp mismatch")
        if (int(image.width), int(image.height)) != (int(rgb["width"]), int(rgb["height"])):
            raise LiveGtContractError("RGB/semantic/depth image geometry mismatch")
        if not np.allclose(np.asarray(image.transform.get_matrix()), camera_matrix, rtol=0, atol=1e-5):
            raise LiveGtContractError("RGB/semantic/depth pose mismatch")
    if int(rgb["frame_id"]) != frame_id or not np.allclose(rgb["camera_matrix"], camera_matrix, rtol=0, atol=1e-5):
        raise LiveGtContractError("RGB/ticket identity or pose mismatch")
    depth = decode_depth(depth_image)
    raw = raw_semantic(semantic_image)
    width, height = int(depth_image.width), int(depth_image.height)
    k = native_intrinsics(width, height)
    rows = frozen_people_rows(frozen_world, frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                             camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                             intrinsics=k, width=width, height=height)
    path = sidecar_path(directory, identity)
    payloads = {}
    # Preserve encoded depth, not float32 rounding, for exact offline replay.
    encoded_depth = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(height, width, 4)
    for name, value in (("depth_bgra", encoded_depth), ("rawsemantic", raw)):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        payload = buffer.getvalue()
        destination = path.with_name(path.name.replace(".avo.json", f".{name}.npy"))
        G._create_identical_or_fail(destination, payload)
        payloads[name] = {"filename": destination.name, "sha256": hashlib.sha256(payload).hexdigest()}
    prescore = {"schema": SCHEMA, "record_kind": "AVO_PRE_SCORE_INPUTS", "mask_contract": MASK_CONTRACT, "identity": dict(identity),
                "source_frame_id": frame_id, "sensor_timestamp": float(rgb["timestamp"]),
                "native_shape": [height, width], "camera_matrix": camera_matrix.tolist(),
                "camera_inverse": camera_inverse.tolist(), "intrinsics": k.tolist(),
                "source_synchronization": synchronization,
                "raw_pedestrian_rows": rows, "payloads": payloads,
                "start_raw_ns": start, "write_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
    inputs_path = path.with_name(path.name.replace(".avo.json", ".avo_inputs.json"))
    inputs_encoded = _canonical(prescore)
    G._create_identical_or_fail(inputs_path, inputs_encoded)
    input_binding = {"filename": inputs_path.name, "sha256": hashlib.sha256(inputs_encoded).hexdigest()}
    inputs_ready = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    scoring_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    try:
        avo = score_live_avo_people(frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                                   depth_frame_id=int(depth_image.frame), depth_timestamp=float(depth_image.timestamp),
                                   depth_m=depth, camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                                   intrinsics=k, raw_actor_rows=rows, width=width, height=height)
    except Exception as exc:
        rejection = {"schema": SCHEMA, "record_kind": "AVO_SCORING_REJECTED", "identity": dict(identity),
                     "prescore_input": input_binding, "prescore_input_ready_raw_ns": inputs_ready,
                     "scoring_start_raw_ns": scoring_start, "error": f"{type(exc).__name__}: {exc}",
                     "geometry_diagnostics": getattr(exc, "geometry_diagnostics", None),
                     "rejected_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
        rejection_path = path.with_name(path.name.replace(".avo.json", ".avo_rejected.json"))
        encoded = _canonical(rejection)
        G._create_identical_or_fail(rejection_path, encoded)
        exc.avo_evidence_record = {"frame_id": frame_id, "path": str(rejection_path),
                                   "sha256": hashlib.sha256(encoded).hexdigest(), "error": rejection["error"]}
        raise
    document = {**prescore, "record_kind": "AVO_SCORED_FRAME", "avo": avo,
                "prescore_input": input_binding, "prescore_input_ready_raw_ns": inputs_ready,
                "scoring_start_raw_ns": scoring_start, "end_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "ready_wall_ns": time.time_ns()}
    encoded = _canonical(document)
    G._create_identical_or_fail(path, encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(),
            "frame_id": frame_id, "pedestrians": len(rows),
            "eligible": sum(row["avo"] >= .65 for row in avo["qualified"]),
            "duration_ms": (document["end_raw_ns"] - start) / 1e6}


def read_avo_bundle(directory: Path, *, expected_identity: Mapping[str, Any], timeout_s: float) -> dict:
    path = sidecar_path(directory, expected_identity)
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise LiveGtContractError("synchronized live AVO sidecar not ready")
        time.sleep(min(.001, max(0., deadline - time.monotonic())))
    encoded = path.read_bytes()
    document = json.loads(encoded)
    if document.get("schema") != SCHEMA or document.get("mask_contract") != MASK_CONTRACT:
        raise LiveGtContractError("live AVO schema mismatch")
    if document["identity"] != dict(expected_identity) or document["source_frame_id"] != int(expected_identity["frame_id"]):
        raise LiveGtContractError("live AVO sidecar identity mismatch")
    if document["avo"]["frame_id"] != document["source_frame_id"] or document["avo"]["threshold"] != .65:
        raise LiveGtContractError("live AVO result identity/threshold mismatch")
    return {"avo": document["avo"], "avo_sidecar_sha256": hashlib.sha256(encoded).hexdigest(),
            "avo_ready_wall_ns": document["ready_wall_ns"], "avo_measurement_semantics": MEASUREMENT_SEMANTICS}


def preserve_avo_evidence(records, directory: Path, *, rejected_records=()) -> dict:
    """After workers drain, retain all exact AVO inputs before scratch teardown."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for record in [*records, *rejected_records]:
        source = Path(record["path"])
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise LiveGtContractError("AVO sidecar changed before preservation")
        document = json.loads(payload)
        paths = [source]
        bound = document["prescore_input"]
        input_path = source.parent / bound["filename"]
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != bound["sha256"]:
            raise LiveGtContractError("AVO pre-score input changed before preservation")
        inputs = json.loads(input_path.read_bytes())
        paths.append(input_path)
        for item in inputs["payloads"].values():
            path = source.parent / item["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise LiveGtContractError("AVO raw sensor payload changed before preservation")
            paths.append(path)
        for suffix in ("objects.json", "semantic.json", "semantic.npy"):
            stem_name = input_path.name.replace("avo_inputs.json", suffix)
            candidate = source.with_name(stem_name)
            if candidate.is_file():
                paths.append(candidate)
        for path in paths:
            target = directory / path.name
            try:
                os.link(path, target)
            except OSError as exc:
                import errno
                if exc.errno != errno.EXDEV:
                    raise
                with path.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            manifest[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
    G._create_identical_or_fail(directory / "MANIFEST.json", _canonical({"schema": SCHEMA, "sha256": manifest}))
    return {"directory": str(directory), "files": len(manifest), "all_avo_inputs_preserved": True}


def build_live_gt_collector(base: type) -> type:
    class LiveAvoCollector(base):
        def __init__(self, **kwargs):
            self._avo_condition = threading.Condition()
            self._avo_depth = OrderedDict()
            self._avo_semantic = OrderedDict()
            self._avo_rgb = OrderedDict()
            self._avo_radar = OrderedDict()
            self._avo_callbacks_complete = OrderedDict()
            self._avo_tick_barrier_records = []
            self._avo_tick_barrier_limit = 8192
            self._avo_results = []
            self._avo_rejected_results = []
            self._avo_sync = (None if "attempt_dir" not in kwargs else
                LiveSyncEvidence(Path(kwargs["attempt_dir"]) / "run5_live_sync_evidence"))
            self._avo_barrier_rejections = (None if "attempt_dir" not in kwargs else
                LiveSyncEvidence(Path(kwargs["attempt_dir"]) / "run5_live_barrier_rejections"))
            super().__init__(**kwargs)

        def _avo_put(self, bucket, frame_id, value):
            with self._avo_condition:
                bucket[int(frame_id)] = value
                while len(bucket) > 24:
                    bucket.popitem(last=False)
                self._avo_condition.notify_all()

        def _avo_complete_callback(self, role, frame_id):
            # A received header alone is insufficient: the original collector
            # must also finish installing that callback before work is queued.
            with self._avo_condition:
                self._avo_callbacks_complete.setdefault(int(frame_id), set()).add(role)
                while len(self._avo_callbacks_complete) > 24:
                    self._avo_callbacks_complete.popitem(last=False)
                self._avo_condition.notify_all()

        def _on_rgb(self, image):
            metadata = sensor_metadata(image)
            metadata["_rgb_image"] = image
            self._avo_put(self._avo_rgb, image.frame, metadata)
            result = super()._on_rgb(image)
            self._avo_complete_callback("rgb", image.frame)
            return result

        def _on_semantic(self, image):
            self._avo_put(self._avo_semantic, image.frame, (image, sensor_metadata(image)))
            result = super()._on_semantic(image)
            self._avo_complete_callback("semantic", image.frame)
            return result

        def _on_depth(self, image):
            self._avo_put(self._avo_depth, image.frame, (image, sensor_metadata(image)))
            self._avo_complete_callback("depth", image.frame)

        def _on_radar(self, measurement):
            self._avo_put(self._avo_radar, measurement.frame, sensor_metadata(measurement))
            result = super()._on_radar(measurement)
            self._avo_complete_callback("radar", measurement.frame)
            return result

        def wait_for_sensor_frame(self, frame_id, *, snapshot, timeout_s=2.0):
            """All-tick barrier BEFORE preparation/action-open, shared with probe.

            No tick is advanced here and no buffer is consumed. Every callback,
            including unselected20Hz ticks, finishes before SamplingWorld may
            advance again. Waiting cannot repair an already mis-stamped frame.
            """
            frame_id = int(frame_id)
            if not 0 < float(timeout_s) <= 3.:
                raise ValueError("source barrier requires a bounded0..3s wait")
            start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            deadline = time.monotonic() + float(timeout_s)
            expected = {"rgb", "semantic", "depth", "radar"}
            with self._avo_condition:
                while not expected.issubset(self._avo_callbacks_complete.get(frame_id, set())):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._avo_condition.wait(left)
                rgb = self._avo_rgb.get(frame_id)
                semantic = self._avo_semantic.get(frame_id)
                depth = self._avo_depth.get(frame_id)
                radar = self._avo_radar.get(frame_id)
                complete = set(self._avo_callbacks_complete.get(frame_id, set()))
            world_timestamp = float(snapshot.timestamp.elapsed_seconds)
            world_meta = {"frame_id": int(snapshot.frame), "timestamp": world_timestamp,
                          "timestamp_repr": repr(world_timestamp), "timestamp_hex": world_timestamp.hex()}
            if rgb is not None and semantic is not None and depth is not None:
                rgb_copy = dict(rgb, semantic_callback=semantic[1], depth_callback=depth[1], radar_callback=radar)
                record = source_alignment_record(identity={"frame_id": frame_id}, rgb=rgb_copy,
                    semantic_image=semantic[0], depth_image=depth[0], policy_timestamp_s=world_timestamp,
                    frozen_world=SimpleNamespace(frozen_frame_id=int(snapshot.frame), frozen_timestamp_s=world_timestamp,
                                                 frozen_observed_raw_ns=start_ns))
            else:
                # Preserve incomplete-source metadata, rather than failing
                # before there is evidence of which callback did not arrive.
                record = {"schema": SYNC_SCHEMA, "identity": {"frame_id": frame_id},
                    "sources": {"rgb": None if rgb is None else {k: v for k, v in rgb.items() if not k.startswith("_")},
                                "semantic": None if semantic is None else semantic[1],
                                "depth": None if depth is None else depth[1], "radar": radar,
                                "policy_radar": dict(world_meta), "frozen_world": dict(world_meta)},
                    "observed_raw_ns": start_ns}
            end_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            record.update(start_raw_ns=start_ns, end_raw_ns=end_ns, wait_ms=(end_ns-start_ns)/1e6,
                          outside_action_open=True, completed_roles=sorted(complete),
                          timestamp_tolerance_s=0., pose_matrix_atol=1e-5)
            if len(self._avo_tick_barrier_records) >= self._avo_tick_barrier_limit:
                raise LiveGtContractError("all-tick synchronization evidence bound exceeded")
            self._avo_tick_barrier_records.append(record)  # BEFORE validation
            raw_sources = {"rgb": None if rgb is None else rgb.get("_rgb_image"),
                           "semantic": None if semantic is None else semantic[0],
                           "depth": None if depth is None else depth[0]}
            try:
                validate_source_alignment(record)
                if complete != expected:
                    raise SourceSynchronizationError("SOURCE_CALLBACK_NOT_COMPLETED", str(sorted(expected-complete)))
                camera = snapshot.find(int(self.camera.id))
                if camera is None:
                    raise SourceSynchronizationError("SOURCE_OWNED_CAMERA_POSE_MISSING", str(frame_id))
                owned = np.asarray(camera.get_transform().get_matrix(), dtype=np.float64)
                record["owned_camera_matrix"] = owned.tolist()
                deltas = {role: float(np.max(np.abs(np.asarray(record["sources"][role]["camera_matrix"])-owned)))
                          for role in ("rgb", "semantic", "depth")}
                record["camera_pose_to_owned_tick_max_abs_delta"] = deltas
                if owned.shape != (4, 4) or not np.all(np.isfinite(owned)) or any(delta > 1e-5 for delta in deltas.values()):
                    raise SourceSynchronizationError("SOURCE_OWNED_CAMERA_POSE_MISMATCH", str(deltas))
            except SourceSynchronizationError as exc:
                record["status"] = exc.status
                record["error"] = str(exc)
                if self._avo_barrier_rejections is not None:
                    self._avo_barrier_rejections.record_rejection(record, exc, raw_sources=raw_sources)
                exc.record = record
                raise
            record["status"] = "EXACT_ALL_SENSOR_TICK_ACCEPTED"
            return record

        def on_world_tick(self, frame_id, route_tick):
            snapshot = self.world.get_snapshot()
            self.wait_for_sensor_frame(frame_id, snapshot=snapshot)
            # Parent freezes the same owned snapshot and only THEN offers the
            # alternate-tick preparation token; no policy timer is moved.
            return super().on_world_tick(frame_id, route_tick)

        def _spawn_sensors(self):
            from rl_agent import ue_route_b_split_cell_adapter_v1 as P
            super()._spawn_sensors()
            bp = self.world.get_blueprint_library().find("sensor.camera.depth")
            for key, value in {"image_size_x": "1280", "image_size_y": "720", "fov": "120", "sensor_tick": "0.0"}.items():
                bp.set_attribute(key, value)
            depth = self.world.spawn_actor(bp, P.sensor_transform(P.CAMERA_MOUNT), attach_to=self.ego)
            self.depth_camera = depth
            self.sensors.append(depth)  # unchanged collector.finish owns destruction
            depth.listen(self._on_depth)

        def _avo_sources(self, frame_id):
            deadline = time.monotonic() + .25
            with self._avo_condition:
                while any(frame_id not in bucket for bucket in (self._avo_rgb, self._avo_semantic, self._avo_depth)):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LiveGtContractError(f"synchronized RGB/semantic/depth missing for {frame_id}")
                    self._avo_condition.wait(remaining)
                rgb, semantic, depth = tuple(bucket.pop(frame_id) for bucket in
                                             (self._avo_rgb, self._avo_semantic, self._avo_depth))
                # Header metadata was copied at callback entry, not queried from
                # a later live sensor pose. Keep actual images for encoded data.
                semantic_image, semantic_meta = semantic
                depth_image, depth_meta = depth
                rgb["semantic_callback"] = semantic_meta
                rgb["depth_callback"] = depth_meta
                rgb["radar_callback"] = self._avo_radar.pop(frame_id, None)
                return rgb, semantic_image, depth_image

        def _ground_truth(self, **kwargs):
            # The baseline has a separate current/aligned map diagnostic. It
            # must neither pop source-frame sensors nor attempt AVO a second time.
            if (kwargs.get("world") is None and kwargs.get("stationary_tracker")
                    is getattr(self, "aligned_actor_tracker", object())):
                return super()._ground_truth(**kwargs)
            frame_id = int(kwargs["frame_id"])
            identity = getattr(self.live, "_run4_identity", {}).get(frame_id, {})
            reward = identity.get("reward_requested") is True
            with self._quality_scene_lock:
                frozen = self._quality_scenes.get(frame_id)
            rows = super()._ground_truth(**kwargs)
            if reward:
                if self.live.last_datagram.get(frame_id) is None:
                    raise LiveGtContractError("AVO work admitted before last datagram")
                if frozen is None:
                    raise LiveGtContractError("AVO lacks owned-tick frozen geometry")
                rgb, semantic, depth = self._avo_sources(frame_id)
                # The inherited ticket's matrices are queried in preparation,
                # potentially after capture. AVO must instead use Image.transform
                # from the synchronized RGB/depth snapshot. Vehicle localization
                # and model preparation retain their existing path unchanged.
                exact_camera = np.asarray(rgb["camera_matrix"], dtype=np.float64)
                try:
                    record = write_avo_bundle(self.edge_evidence_dir,
                        identity=self.live.identity_for_frame(frame_id), rgb=rgb,
                        semantic_image=semantic, depth_image=depth, frozen_world=frozen,
                        camera_matrix=exact_camera, camera_inverse=np.linalg.inv(exact_camera),
                        policy_timestamp_s=float(kwargs["timestamp"]), sync_evidence=self._avo_sync)
                except Exception as exc:
                    if hasattr(exc, "avo_evidence_record"):
                        self._avo_rejected_results.append(exc.avo_evidence_record)
                    raise
                record["legacy_ticket_camera_max_abs_delta"] = float(np.max(np.abs(
                    exact_camera - np.asarray(kwargs["camera_matrix"]))))
                self._avo_results.append(record)
            return rows

        def finish(self):
            result = super().finish()
            from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
            preservation = preserve_avo_evidence(self._avo_results, Path(self.attempt_dir) / "run5_live_avo_inputs",
                                                 rejected_records=self._avo_rejected_results)
            G._create_identical_or_fail(Path(self.attempt_dir) / "run5_live_avo_gt_summary.json",
                _canonical({"schema": SCHEMA, "measurement_semantics": MEASUREMENT_SEMANTICS,
                            "records": self._avo_results, "rejected_records": self._avo_rejected_results, "new_gt_helper_threads": 0,
                            "preservation": preservation,
                            "source_synchronization": None if self._avo_sync is None else self._avo_sync.summary(),
                            "all_tick_sensor_barrier": {"schema": SYNC_SCHEMA,
                                "outside_action_open": True, "records": self._avo_tick_barrier_records,
                                "rejections": None if self._avo_barrier_rejections is None else self._avo_barrier_rejections.summary()}}))
            return result

    return LiveAvoCollector


def build_timestamped_scene_source(base_source: type, base_world: type) -> type:
    """Preserve source-world time on the SAME immutable owned-tick geometry."""
    class TimestampedFrozenWorld(base_world):
        __slots__ = ("frozen_timestamp_s", "frozen_observed_raw_ns")

        def __init__(self, frozen, snapshot):
            super().__init__(frozen._actors, frozen_frame_id=int(snapshot.frame))
            self.frozen_timestamp_s = float(snapshot.timestamp.elapsed_seconds)
            self.frozen_observed_raw_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    class TimestampedSceneSource(base_source):
        def capture(self, world_snapshot):
            frozen = super().capture(world_snapshot)
            return TimestampedFrozenWorld(frozen, world_snapshot)

    return TimestampedSceneSource


def install_ue_gt_seams(*, attempt_dir: Path | None = None) -> dict:
    """Call before Phase6 builds its collector class; applies only in this process."""
    global _INSTALLED
    del attempt_dir  # the constructed collector owns its exact attempt path
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    if not _INSTALLED:
        P.semantic_gt_3class = corrected_semantic_image
        Q.QualitySceneSnapshotSource = build_timestamped_scene_source(
            Q.QualitySceneSnapshotSource, Q.QualityFrozenWorld)
        Q.QualityPassiveSplitCollector = build_live_gt_collector(Q.QualityPassiveSplitCollector)
        _INSTALLED = True
    return {"schema": SCHEMA, "measurement_semantics": MEASUREMENT_SEMANTICS,
            "depth_sensor": "1280x720_FOV120_SAME_POSE_SAME_FRAME", "new_gt_helper_threads": 0}
