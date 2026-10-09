"""Exact owned-tick model poses, with durable UE/wire/target reconciliation.

Install OUTSIDE the sealed repair composition. No wire, actor, SI/P40, reward,
deadline, thread, device or sensor contract changes. The existing callback
barrier supplies immutable poses; scoped preparation helpers consume them.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    StaticCameraRegistry, build_frame_context_v1, camera_world_matrix,
    carla_transform_matrix, validate_frame_context,
)

SCHEMA = "splitfusion.live_parity_runtime.pose_evidence.v1"
ATOL = 1e-5
RADAR_MOUNT = (2., 0., 1., 0., 0., 0.)
_LOCAL = threading.local()


class PoseParityError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PoseParityError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _matrix(value, name):
    result = np.array(value, dtype=np.float64, copy=True)
    _require(result.shape == (4, 4) and np.isfinite(result).all(), f"invalid {name} pose")
    _require(np.allclose(result[3], (0., 0., 0., 1.), rtol=0, atol=1e-8),
             f"invalid {name} homogeneous row")
    _require(np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), rtol=0, atol=1e-5)
             and abs(np.linalg.det(result[:3, :3]) - 1.) <= 1e-5,
             f"invalid {name} rigid rotation")
    result.flags.writeable = False
    return result


def _agree(actual, expected, name):
    actual, expected = _matrix(actual, name), _matrix(expected, name)
    delta = float(np.max(np.abs(actual - expected)))
    _require(delta <= ATOL, f"{name} pose mismatch: max_abs_matrix_delta={delta}")
    return delta


def context_document(context):
    validate_frame_context(context)
    return {"stream_id": context.stream_id, "frame_id": context.frame_id,
            "sequence_id": context.sequence_id,
            "capture_timestamp_ns": context.capture_timestamp_ns,
            "ego_world": list(context.ego_world.values()),
            "camera_model_sha256": context.camera_model_sha256,
            "camera_mount_sha256": context.camera_mount_sha256}


def context_camera(context, registry=None):
    registry = registry or StaticCameraRegistry.audited()
    calibration = registry.resolve(context.camera_model_sha256, context.camera_mount_sha256)
    return camera_world_matrix(context, calibration)


@dataclass(frozen=True)
class OwnedPose:
    """All mutable CARLA data is copied into primitive tuples at the barrier."""
    source: bytes

    @property
    def document(self):
        return json.loads(self.source)

    @property
    def sha256(self):
        return _sha(self.source)

    def matrix(self, role):
        return _matrix(self.document[role + "_matrix"], role)


def capture_owned_pose(*, collector, frame_id, snapshot, barrier):
    frame = int(frame_id)
    _require(int(snapshot.frame) == frame == int(barrier["identity"]["frame_id"]),
             "owned snapshot/frame mismatch")
    _require(barrier.get("status") == "EXACT_ALL_SENSOR_TICK_ACCEPTED",
             "owned pose requires accepted all-sensor barrier")
    stamp = float(snapshot.timestamp.elapsed_seconds)
    _require(math.isfinite(stamp), "nonfinite owned timestamp")
    sources = barrier["sources"]
    for role in ("rgb", "radar", "semantic", "depth", "frozen_world"):
        value = sources[role]
        _require(value["frame_id"] == frame and value["timestamp"] == stamp,
                 f"{role} source frame/timestamp mismatch")
    owned = {}
    ids = {}
    for role, actor in (("camera", collector.camera), ("radar", collector.radar), ("ego", collector.ego)):
        ids[role] = int(actor.id)
        item = snapshot.find(int(actor.id))
        _require(item is not None, f"owned snapshot missing {role}")
        owned[role] = item.get_transform()
    camera = _matrix(sources["rgb"]["camera_matrix"], "rgb")
    radar = _matrix(sources["radar"]["camera_matrix"], "radar")
    _agree(camera, owned["camera"].get_matrix(), "owned RGB camera")
    _agree(radar, owned["radar"].get_matrix(), "owned radar")
    location, rotation = owned["ego"].location, owned["ego"].rotation
    ego_pose = tuple(float(getattr(location, key)) for key in ("x", "y", "z")) + tuple(
        float(getattr(rotation, key)) for key in ("pitch", "yaw", "roll"))
    ego = carla_transform_matrix(ego_pose)
    _agree(ego, owned["ego"].get_matrix(), "owned ego")
    registry = StaticCameraRegistry.audited()
    calibration = registry.resolve(*registry.identities[0])
    _agree(ego @ np.asarray(calibration.camera_to_ego_values), camera, "ego/mount RGB camera")
    _agree(ego @ carla_transform_matrix(RADAR_MOUNT), radar, "ego/mount radar")
    return OwnedPose(canonical({"frame_id": frame, "stream_id": str(collector.stream_id),
        "timestamp": stamp, "timestamp_hex": stamp.hex(), "actor_ids": ids,
        "camera_matrix": camera.tolist(), "radar_matrix": radar.tolist(),
        "ego_matrix": ego.tolist(), "ego_world": list(ego_pose),
        "camera_model_sha256": calibration.camera_model_sha256,
        "camera_mount_sha256": calibration.camera_mount_sha256,
        "source_schema": barrier["schema"],
        "source_callback_receipt_raw_ns": {role: sources[role].get("callback_receipt_raw_ns")
                                           for role in ("rgb", "radar", "semantic", "depth")},
        "barrier_end_raw_ns": barrier.get("end_raw_ns")}))


class PoseArchive:
    """Bounded primitive evidence; fsync only at shutdown after worker drain."""
    def __init__(self, directory, *, side, max_frames):
        _require(side in ("ue", "edge"), "invalid pose archive side")
        _require(type(max_frames) is int and 1 <= max_frames <= 1000, "registered pose frame bound required")
        self.directory, self.side, self.max_frames = Path(directory), side, max_frames
        if self.directory.exists():
            raise FileExistsError("pose archive is create-only")
        self._lock = threading.Lock()
        self._events, self._bytes = [], 0
        self._counts = Counter()
        self.errors = []
        self.manifest = None

    def add(self, kind, **value):
        payload = canonical({"kind": kind, "observed_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), **value})
        with self._lock:
            _require(self.manifest is None, "pose archive already closed")
            _require(len(self._events) < 32768 and self._bytes + len(payload) <= 64 * 1024 * 1024,
                     "pose evidence capacity exhausted")
            if kind in ("UE_WIRE", "EDGE_WIRE"):
                _require(self._counts[kind] < self.max_frames, "pose wire frame budget exhausted")
            self._events.append(payload)
            self._bytes += len(payload)
            self._counts[kind] += 1

    def error(self, exc, **identity):
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.errors.append(message)
        self.add("ERROR", error=message, **identity)

    def flush(self):
        with self._lock:
            if self.manifest is not None:
                return self.manifest
            self.directory.mkdir(parents=True, exist_ok=False)
            payload = b"".join(row + b"\n" for row in self._events)
            result = {"schema": SCHEMA, "side": self.side, "max_frames": self.max_frames,
                "counts": dict(self._counts), "errors": list(self.errors), "complete": not self.errors,
                "frames": self._counts["UE_WIRE" if self.side == "ue" else "EDGE_WIRE"],
                "sha256": {"pose_evidence.jsonl": _sha(payload)}}
            for name, body in (("pose_evidence.jsonl", payload), ("MANIFEST.json", canonical(result))):
                with (self.directory / name).open("xb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.manifest = result
            return result


def read_archive(directory):
    directory = Path(directory)
    _require(directory.is_dir() and not directory.is_symlink(), "missing/unsafe pose archive")
    _require({p.name for p in directory.iterdir()} == {"MANIFEST.json", "pose_evidence.jsonl"},
             "pose archive inventory mismatch")
    for name in ("MANIFEST.json", "pose_evidence.jsonl"):
        path = directory / name
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024 * 1024,
                 "unsafe pose evidence file")
    manifest = json.loads((directory / "MANIFEST.json").read_bytes())
    payload = (directory / "pose_evidence.jsonl").read_bytes()
    _require(manifest["schema"] == SCHEMA and manifest["side"] in ("ue", "edge"), "pose archive schema")
    _require(manifest["sha256"] == {"pose_evidence.jsonl": _sha(payload)}, "pose archive hash mismatch")
    events = [json.loads(line) for line in payload.splitlines()]
    counts = dict(Counter(row["kind"] for row in events))
    _require(counts == manifest["counts"], "pose archive count mismatch")
    _require(manifest["frames"] == counts.get("UE_WIRE" if manifest["side"] == "ue" else "EDGE_WIRE", 0),
             "pose wire count mismatch")
    _require(manifest["complete"] is (not manifest["errors"]), "pose archive status mismatch")
    return {"manifest": manifest, "events": events}


def verify_archives(ue_directory, edge_directory, *, expected_transmitted, expected_evaluated):
    """Independent pose joins against authoritative transmitted/evaluated rows.

    Expected rows use ``frame_id`` and ``run4_identity``. A returned PASS never
    substitutes for target/source completeness or the latency qualification.
    """
    ue, edge = read_archive(ue_directory), read_archive(edge_directory)
    _require(ue["manifest"]["side"] == "ue" and edge["manifest"]["side"] == "edge",
             "pose archive sides reversed")
    _require(ue["manifest"]["complete"] and edge["manifest"]["complete"], "pose archive recorded errors")

    def indexed(rows, kind):
        selected = [r for r in rows if r["kind"] == kind]
        key = lambda r: int(r.get("frame_id", r.get("context", {}).get("frame_id", -1)))
        result = {key(r): r for r in selected}
        _require(len(result) == len(selected) and -1 not in result, f"duplicate/invalid {kind} frame")
        return result

    wires = indexed(ue["events"], "UE_WIRE")
    sent = {f: r for f, r in indexed(ue["events"], "UE_PREPARATION_DONE").items() if r["sent"] is True}
    received = indexed(edge["events"], "EDGE_WIRE")
    reconciled = indexed(edge["events"], "EDGE_TARGET_RECONCILED")
    expected = {int(row["frame_id"]): row["run4_identity"] for row in expected_transmitted}
    evaluated = {int(row["frame_id"]): row["run4_identity"] for row in expected_evaluated}
    _require(len(expected) == len(expected_transmitted) and len(evaluated) == len(expected_evaluated),
             "duplicate authoritative pose frame")
    _require(bool(expected) and set(wires) == set(sent) == set(received) == set(expected),
             "UE/edge/authoritative transmitted pose populations differ")
    _require(set(reconciled) == set(evaluated) and set(evaluated) <= set(expected),
             "reconciled/evaluated pose populations differ")
    sources = {}
    for event in ue["events"]:
        if event["kind"] == "OWNED_SOURCE":
            digest = _sha(canonical(event["source"]))
            _require(digest == event["source_sha256"] and digest not in sources, "owned source digest/duplicate")
            sources[digest] = event["source"]
    maximum = 0.
    registry = StaticCameraRegistry.audited()
    for frame, action in expected.items():
        a, b = wires[frame], received[frame]
        _require(a["action_identity"] == b["action_identity"] == action,
                 "pose wire/action identity differs")
        _require(a["gt_identity"] == b["gt_identity"] and a["context"] == b["context"],
                 "UE/edge context identity differs")
        source = sources.get(a["source_sha256"])
        _require(source is not None and source["frame_id"] == frame
                 and source["stream_id"] == a["context"]["stream_id"], "wire lacks owned source")
        context = dict(a["context"])
        ego = context.pop("ego_world")
        value = build_frame_context_v1(**context, **dict(zip(
            ("ego_world_x", "ego_world_y", "ego_world_z", "ego_world_pitch", "ego_world_yaw", "ego_world_roll"), ego)))
        _require(list(ego) == source["ego_world"], "wire ego differs from owned snapshot")
        camera = context_camera(value, registry)
        maximum = max(maximum, _agree(camera, source["camera_matrix"], "archive source/wire camera"))
        _agree(camera, b["camera_matrix"], "archive reconstructed edge camera")
        if frame in reconciled:
            target = reconciled[frame]
            _require(target["action_identity"] == evaluated[frame] == action
                     and target["context"] == a["context"] and target["gt_identity"] == a["gt_identity"],
                     "target reconciliation identity differs")
            delta = _agree(camera, target["target_camera_matrix"], "archive wire/target camera")
            _require(abs(delta - target["max_abs_matrix_delta"]) <= 1e-12, "recorded pose delta differs")
            maximum = max(maximum, delta)
    return {"status": "PASS", "passed": True, "schema": SCHEMA, "transmitted_frames": len(expected),
            "evaluated_frames": len(evaluated), "owned_source_frames": len(sources),
            "maximum_camera_matrix_delta": maximum, "pose_matrix_atol": ATOL}


@contextmanager
def _bind_preparation(owner, source, archive):
    _require(getattr(_LOCAL, "binding", None) is None, "nested owned-pose preparation")
    binding = {"owner": owner, "source": source, "used": set(), "archive": archive}
    _LOCAL.binding = binding
    try:
        yield binding
    finally:
        _LOCAL.binding = None


@contextmanager
def install_ue_pose(*, attempt_dir: Path, max_frames: int):
    """Outer scope for composition.run_ue_child; no historical file is edited."""
    from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
    bind_legacy_namespace()
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    _require(not LG._INSTALLED, "pose scope requires fresh pre-collector child")
    archive = PoseArchive(Path(attempt_dir) / "pose_archive", side="ue", max_frames=max_frames)
    originals = (LG.build_live_gt_collector, P.actor_world_matrix, P.actor_world_inverse_matrix,
                 U.Run4FramePipelineV2.materialize)

    def collector(base):
        inherited = originals[0](base)
        class OwnedPoseCollector(inherited):
            def __init__(self, **kwargs):
                self._pose_lock = threading.Lock()
                self._pose_sources = OrderedDict()
                self._pose_fault = None
                super().__init__(**kwargs)

            def wait_for_sensor_frame(self, frame_id, *, snapshot, timeout_s=2.):
                try:
                    barrier = super().wait_for_sensor_frame(frame_id, snapshot=snapshot, timeout_s=timeout_s)
                    source = capture_owned_pose(collector=self, frame_id=frame_id, snapshot=snapshot, barrier=barrier)
                    with self._pose_lock:
                        _require(frame_id not in self._pose_sources, "duplicate owned pose frame")
                        self._pose_sources[frame_id] = source
                        while len(self._pose_sources) > 512:
                            self._pose_sources.popitem(last=False)
                    archive.add("OWNED_SOURCE", source=source.document, source_sha256=source.sha256)
                    return barrier
                except Exception as exc:
                    self._pose_fault = str(exc)
                    archive.error(exc, frame_id=int(frame_id))
                    raise

            def on_world_tick(self, frame_id, route_tick):
                _require(self._pose_fault is None, f"owned pose preparation failed: {self._pose_fault}")
                return super().on_world_tick(frame_id, route_tick)

            def _records_for(self, frame_id, timeout_s=.25):
                result = super()._records_for(frame_id, timeout_s=timeout_s)
                if result is not None:
                    binding = getattr(_LOCAL, "binding", None)
                    _require(binding is not None and binding["owner"] is self,
                             "sensor records outside owned-pose preparation")
                    source = binding["source"]
                    for image, role in ((result[0], "camera"), (result[3], "radar")):
                        _require(int(image.frame) == int(frame_id) == source.document["frame_id"]
                                 and float(image.timestamp) == source.document["timestamp"],
                                 "preparation sensor frame/timestamp mismatch")
                        _agree(image.transform.get_matrix(), source.matrix(role), f"preparation {role}")
                    binding["used"].add("sensor_records")
                return result

            def _process_token(self, token):
                frame = int(token["frame_id"])
                try:
                    with self._pose_lock:
                        source = self._pose_sources.pop(frame, None)
                    _require(source is not None, f"missing owned pose for preparation frame {frame}")
                    with _bind_preparation(self, source, archive):
                        result = super()._process_token(token)
                    archive.add("UE_PREPARATION_DONE", frame_id=frame, stream_id=self.stream_id,
                                sent=frame in self.sent_frames)
                    return result
                except Exception as exc:
                    self._pose_fault = str(exc)
                    archive.error(exc, frame_id=frame)
                    raise
        return OwnedPoseCollector

    def matrix(actor, inverse=False):
        binding = getattr(_LOCAL, "binding", None)
        if binding is None:
            return originals[2 if inverse else 1](actor)
        doc = binding["source"].document
        for role in ("camera", "radar"):
            if actor is getattr(binding["owner"], role):
                value = binding["source"].matrix(role)
                binding["used"].add(role + ("_inverse" if inverse else "_matrix"))
                return np.linalg.inv(value) if inverse else value
        raise PoseParityError(f"unknown actor in owned preparation: {getattr(actor, 'id', None)}")

    def materialize(pipeline, planned, **kwargs):
        binding = getattr(_LOCAL, "binding", None)
        _require(binding is not None, "FrameContext outside owned-pose preparation")
        source, used = binding["source"], binding["used"]
        doc = source.document
        _require(kwargs["frame_id"] == doc["frame_id"] and pipeline.stream_id == doc["stream_id"]
                 and kwargs.get("carla_timestamp") == doc["timestamp"], "materialize source identity mismatch")
        _require({"sensor_records", "camera_matrix", "camera_inverse", "radar_matrix", "radar_inverse"} <= used,
                 "materialize lacks exact sensor/radar projection poses")
        legacy_ego = list(kwargs["ego_pose"])
        kwargs = {**kwargs, "ego_pose": tuple(doc["ego_world"])}
        result = originals[3](pipeline, planned, **kwargs)
        envelope, context = W.unpack_sfd4(result.wire)
        _require(envelope == result.envelope and context.frame_id == doc["frame_id"]
                 and context.stream_id == doc["stream_id"]
                 and tuple(context.ego_world.values()) == tuple(doc["ego_world"]),
                 "serialized FrameContext differs from owned source")
        delta = _agree(context_camera(context), source.matrix("camera"), "wire/source camera")
        archive.add("UE_WIRE", gt_identity=dict(result.gt_identity), action_identity=dict(result.identity),
            context=context_document(context), source_sha256=source.sha256,
            inherited_ego_world=legacy_ego, max_abs_matrix_delta=delta,
            preparation_pose_reads=sorted(used))
        return result

    LG.build_live_gt_collector = collector
    P.actor_world_matrix = lambda actor: matrix(actor)
    P.actor_world_inverse_matrix = lambda actor: matrix(actor, True)
    U.Run4FramePipelineV2.materialize = materialize
    state = {"archive": archive, "errors": archive.errors, "flushed": False}
    try:
        yield state
    except BaseException as exc:
        archive.error(exc)
        raise
    finally:
        (LG.build_live_gt_collector, P.actor_world_matrix, P.actor_world_inverse_matrix,
         U.Run4FramePipelineV2.materialize) = originals
        state["manifest"] = archive.flush()
        state["flushed"] = True


def _verified_target_camera(bundle):
    """Small source metadata only, independently hash checked after read_bundle."""
    from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
    item = bundle.manifest["payloads"]["inputs.json"]
    payload = T._read(bundle.directory / "inputs.json")
    _require(len(payload) == item["bytes"] and _sha(payload) == item["sha256"],
             "target pose metadata hash mismatch")
    metadata = json.loads(payload)
    T._validate_metadata(metadata, dict(bundle.identity))
    frame = metadata["frame"]
    inverse = _matrix(json.loads(frame["camera_inverse_matrix_json"]), "target inverse")
    return _matrix(np.linalg.inv(inverse), "target camera")


@contextmanager
def install_edge_pose(*, archive_directory: Path, max_frames: int):
    """Outer scope for composition.run_edge; enter evaluator seam AFTER repair."""
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime as R, target_transport as T
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    archive = PoseArchive(archive_directory, side="edge", max_frames=max_frames)
    registry = StaticCameraRegistry.audited()
    old_scope, old_read, old_verify = R.install_edge_evaluation, T.read_bundle, E.Run4EdgeProcessorV2.verify
    old_publish = PW.publish_ready_after_warmup
    sources, lock = {}, threading.Lock()
    wire_records = {}
    state = {"archive": archive, "errors": archive.errors, "flushed": False,
             "ready": False, "warming": False, "warm_verified": 0}

    def publish(warm, write_ready, **kwargs):
        _require(not state["ready"] and not state["warming"], "repeated pose READY publication")
        def warm_owned():
            state["warming"] = True
            try:
                return warm()
            finally:
                state["warming"] = False
        def written():
            write_ready()
            state["ready"] = True
            archive.add("EDGE_READY", warm_verified=state["warm_verified"])
        return old_publish(warm_owned, written, **kwargs)

    def verify(processor, frame):
        try:
            envelope, context, profile, identity = old_verify(processor, frame)
            synthetic = (context.stream_id == PW.WARM_STREAM
                         and envelope.session_uuid == PW.WARM_SESSION
                         and envelope.controller_lineage_sha256 == PW.WARM_LINEAGE
                         and envelope.reward_requested is False)
            if state["warming"]:
                _require(synthetic, "non-synthetic frame inside owned edge warmup")
                state["warm_verified"] += 1
                return envelope, context, profile, identity
            _require(state["ready"] and not synthetic, "live pose wire before READY or synthetic wire after READY")
            camera = context_camera(context, registry)
            gt = MP.gt_identity(run_id=processor.run_id, cell_id=processor.cell_id,
                stream_id=context.stream_id, frame_id=context.frame_id,
                anchor_action_id=profile.action_id, anchor_profile_id=profile.profile_id,
                capture_timestamp_ns=context.capture_timestamp_ns)
            event = {"gt_identity": gt, "action_identity": dict(identity),
                     "context": context_document(context), "camera_matrix": camera.tolist(),
                     "wire_sha256": _sha(bytes(frame))}
            key, fingerprint = (context.stream_id, context.frame_id), _sha(canonical(event))
            with lock:
                prior = wire_records.get(key)
                if prior is None:
                    archive.add("EDGE_WIRE", **event)
                    wire_records[key] = fingerprint
                else:
                    _require(prior == fingerprint, "conflicting repeated edge wire/context")
                    archive.add("EDGE_VERIFY_REPEAT", frame_id=context.frame_id,
                                stream_id=context.stream_id, wire_sha256=event["wire_sha256"])
            return envelope, context, profile, identity
        except Exception as exc:
            archive.error(exc)
            raise

    def read(*args, **kwargs):
        result = old_read(*args, **kwargs)
        camera = _verified_target_camera(result)
        key = canonical(dict(result.identity))
        with lock:
            _require(key in sources or len(sources) < max_frames, "target pose evidence budget exhausted")
            if key in sources:
                _require(sources[key][1] == result.manifest_sha256, "target pose manifest changed")
            sources[key] = (camera, result.manifest_sha256)
        return result

    @contextmanager
    def evaluation_scope(**kwargs):
        with old_scope(**kwargs) as repaired:
            klass = E.Run4EvaluatorV2
            old_evaluate, old_close = klass._evaluate_with, klass.close

            def evaluate(evaluator, ticket, gt, timing):
                try:
                    key = canonical(dict(ticket.gt_identity))
                    with lock:
                        value = sources.get(key)
                    _require(value is not None, "evaluator lacks verified target camera")
                    target_camera, digest = value
                    _require(gt["manifest_sha256"] == digest, "evaluator target pose digest mismatch")
                    delta = _agree(context_camera(ticket.context, registry), target_camera, "edge wire/target camera")
                    archive.add("EDGE_TARGET_RECONCILED", gt_identity=dict(ticket.gt_identity),
                        action_identity=MP.run4_identity(ticket.envelope), context=context_document(ticket.context),
                        target_manifest_sha256=digest, target_camera_matrix=target_camera.tolist(),
                        max_abs_matrix_delta=delta)
                except Exception as exc:
                    archive.error(exc, frame_id=ticket.context.frame_id)
                    raise
                return old_evaluate(evaluator, ticket, gt, timing)

            def close(evaluator, timeout_s=None):
                result = old_close(evaluator, timeout_s)
                if result["worker_alive"]:
                    archive.error(PoseParityError("evaluator did not drain before pose preservation"))
                    return {**result, "pose_archive_complete": False}
                state["manifest"] = archive.flush()
                state["flushed"] = True
                return {**result, "pose_archive_complete": not archive.errors,
                        "pose_archive": state["manifest"]}

            klass._evaluate_with, klass.close = evaluate, close
            try:
                yield repaired
            finally:
                klass._evaluate_with, klass.close = old_evaluate, old_close

    R.install_edge_evaluation, T.read_bundle, E.Run4EdgeProcessorV2.verify = evaluation_scope, read, verify
    PW.publish_ready_after_warmup = publish
    try:
        yield state
    except BaseException as exc:
        if archive.manifest is None:
            archive.error(exc)
        raise
    finally:
        R.install_edge_evaluation, T.read_bundle, E.Run4EdgeProcessorV2.verify = old_scope, old_read, old_verify
        PW.publish_ready_after_warmup = old_publish
        if not state["flushed"]:
            state["manifest"] = archive.flush()
            state["flushed"] = True
