"""Bounded exact-source ownership from reward GT admission through completion.

Install before the retained LG factory, outside pose and complete-GT scopes.
The original queue classifier still enforces reward identity and last-send
causality. Only then are immutable sensor bytes retained, before queue append.
No sensor, actor, reward, scheduling, timeout, thread or numerical rule changes.
"""
from __future__ import annotations

from collections import Counter
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

SCHEMA = "splitfusion.parity_repair.exact_source_ownership.v2"
DEFAULT_MAX_BYTES = 768 * 1024 * 1024
DEFAULT_MAX_PENDING = 65  # retained queue: 64 waiting plus its one GT worker


class SourceOwnershipError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SourceOwnershipError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def now():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _matrix(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (4, 4) and np.isfinite(result).all(), "invalid source pose")
    return result


@dataclass(frozen=True)
class FrozenTransform:
    matrix: tuple

    def get_matrix(self):
        return [list(row) for row in self.matrix]

    def get_inverse_matrix(self):
        return np.linalg.inv(self.matrix).tolist()


@dataclass(frozen=True)
class FrozenImage:
    frame: int
    timestamp: float
    width: int
    height: int
    raw_data: bytes
    transform: FrozenTransform


def freeze_image(image, metadata):
    """Retain bytes directly; copy only a mutable buffer, never alias it."""
    frame, stamp = int(image.frame), float(image.timestamp)
    width, height = int(image.width), int(image.height)
    require(frame == metadata["frame_id"] and stamp == metadata["timestamp"] and math.isfinite(stamp),
            "image/callback frame or timestamp differs")
    require((width, height) == (metadata["width"], metadata["height"]) and width > 0 and height > 0,
            "image/callback geometry differs")
    matrix = _matrix(image.transform.get_matrix())
    require(np.allclose(matrix, _matrix(metadata["camera_matrix"]), rtol=0, atol=1e-5), "image/callback pose differs")
    raw = image.raw_data
    require(isinstance(raw, (bytes, bytearray, memoryview)) and len(raw) == width*height*4,
            "source image is not bounded BGRA bytes")
    copied = 0 if type(raw) is bytes else len(raw)
    raw = raw if type(raw) is bytes else bytes(raw)
    return FrozenImage(frame, stamp, width, height, raw,
        FrozenTransform(tuple(tuple(float(v) for v in row) for row in matrix))), copied


def freeze_array(value, *, matrix=False):
    """Immutable bytes backing is stronger than ndarray writeable=False."""
    array = np.asarray(value)
    require(array.dtype.kind in "fiu" and np.isfinite(array).all(), "invalid owned numeric array")
    require(array.shape == (4, 4) if matrix else array.ndim == 2 and array.shape[1] == 3,
            "invalid owned matrix/radar layout")
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    reuse = isinstance(root, bytes) and array.flags.c_contiguous and not array.flags.writeable
    if reuse:
        return array, 0
    raw = array.tobytes(order="C")
    return np.frombuffer(raw, dtype=array.dtype).reshape(array.shape), len(raw)


@dataclass(frozen=True)
class OwnedSources:
    frame_id: int
    identity: bytes
    action: bytes
    metadata: bytes
    rgb: FrozenImage
    semantic: FrozenImage
    depth: FrozenImage
    frozen_world: object
    byte_size: int
    copied_bytes: int

    def sensors(self):
        rgb = json.loads(self.metadata)
        rgb["_rgb_image"] = self.rgb
        return rgb, self.semantic, self.depth


class OwnershipStore:
    def __init__(self, *, max_frames, max_pending=DEFAULT_MAX_PENDING, max_bytes=DEFAULT_MAX_BYTES):
        require(type(max_frames) is int and 1 <= max_frames <= 1000, "registered frame bound required")
        require(type(max_pending) is int and 1 <= max_pending <= DEFAULT_MAX_PENDING,
                "pending bound exceeds retained waiting-plus-worker capacity")
        require(type(max_bytes) is int and max_bytes > 0, "positive source byte bound required")
        self.max_frames, self.max_pending, self.max_bytes = max_frames, max_pending, max_bytes
        self.lock = threading.RLock()
        self.pending = {}; self.seen = set(); self.consumed = set()
        self.bytes = self.peak_bytes = self.peak_pending = 0
        self.events = []; self.errors = []; self.manifest = None

    def event(self, kind, **fields):
        with self.lock:
            require(self.manifest is None, "source ownership evidence already sealed")
            require(len(self.events) < self.max_frames*12+64, "source ownership event bound exceeded")
            self.events.append({"kind": kind, "observed_raw_ns": now(), **fields})

    def failure(self, kind, frame, error):
        with self.lock:
            self.errors.append({"kind": kind, "frame_id": int(frame), "error": f"{type(error).__name__}: {error}"})
            self.event(kind, frame_id=int(frame), error=self.errors[-1]["error"])

    def admit(self, owner, item):
        started = now(); frame = int(item["frame_id"])
        with self.lock:
            require(frame not in self.seen and frame not in self.pending, "duplicate source ownership admission")
            require(len(self.seen) < self.max_frames and len(self.pending) < self.max_pending, "source ticket bound exceeded")
            action = owner.live._run4_identity.get(frame)
            require(action is not None and action.get("reward_requested") is True, "source admission is not a reward ticket")
            require(owner.live.last_datagram.get(frame) is not None, "source admission before last send")
            identity = owner.live.identity_for_frame(frame)
            require(identity["frame_id"] == frame, "source GT identity frame differs")
            with owner._quality_scene_lock:
                frozen = owner._quality_scenes.get(frame)
            require(frozen is not None and int(frozen.frozen_frame_id) == frame,
                    "source admission lacks exact frozen world")
            with owner._avo_condition:
                values = (owner._avo_rgb.get(frame), owner._avo_semantic.get(frame),
                          owner._avo_depth.get(frame), owner._avo_radar.get(frame))
                require(all(value is not None for value in values), "source admission missing exact sensor callbacks")
                rgb_meta, semantic_pair, depth_pair, radar_meta = values
                rgb_metadata = {k: v for k, v in rgb_meta.items() if k != "_rgb_image"}
                rgb_metadata = json.loads(canonical(rgb_metadata))
                rgb_metadata["semantic_callback"] = json.loads(canonical(semantic_pair[1]))
                rgb_metadata["depth_callback"] = json.loads(canonical(depth_pair[1]))
                rgb_metadata["radar_callback"] = json.loads(canonical(radar_meta))
                images = (rgb_meta.get("_rgb_image"), semantic_pair[0], depth_pair[0])
                require(images[0] is not None, "RGB callback lacks retained native image")
                # Validate capacity BEFORE copying any mutable image payload.
                needed = sum(int(image.width)*int(image.height)*4 for image in images)
                radar = np.asarray(item.get("radar_points", {}).get("world_xyz"))
                needed += radar.nbytes + np.asarray(item["camera_matrix"]).nbytes + np.asarray(item["camera_inverse"]).nbytes
                require(self.bytes+needed <= self.max_bytes, "source byte bound exceeded")
                frozen_images = [freeze_image(image, metadata) for image, metadata in zip(images,
                    (rgb_metadata, rgb_metadata["semantic_callback"], rgb_metadata["depth_callback"]))]
            from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import source_alignment_record, validate_source_alignment
            rgb, semantic, depth = [pair[0] for pair in frozen_images]
            alignment = source_alignment_record(identity=identity, rgb=rgb_metadata,
                semantic_image=semantic, depth_image=depth, frozen_world=frozen, policy_timestamp_s=float(item["timestamp"]))
            validate_source_alignment(alignment)
            camera, camera_copy = freeze_array(item["camera_matrix"], matrix=True)
            inverse, inverse_copy = freeze_array(item["camera_inverse"], matrix=True)
            require(np.allclose(camera, _matrix(rgb_metadata["camera_matrix"]), rtol=0, atol=1e-5)
                    and np.allclose(camera @ inverse, np.eye(4), rtol=0, atol=1e-5), "ticket/source camera or inverse differs")
            radar, radar_copy = freeze_array(radar)
            copies = sum(pair[1] for pair in frozen_images)+camera_copy+inverse_copy+radar_copy
            owned = OwnedSources(frame, canonical(identity), canonical(action), canonical(rgb_metadata),
                rgb, semantic, depth, frozen, needed, copies)
            self.pending[frame] = owned; self.seen.add(frame); self.bytes += needed
            self.peak_bytes = max(self.peak_bytes, self.bytes); self.peak_pending = max(self.peak_pending, len(self.pending))
            self.event("SOURCE_ADMITTED", frame_id=frame, identity=json.loads(owned.identity), action_identity=json.loads(owned.action),
                source_metadata_sha256=hashlib.sha256(owned.metadata).hexdigest(), owned_bytes=needed,
                copied_bytes=copies, immutable_image_bytes_reused=sum(len(i.raw_data) for i, count in frozen_images if count == 0),
                admission_ms=(now()-started)/1e6)
            return {**item, "camera_matrix": camera, "camera_inverse": inverse,
                    "radar_points": {**item["radar_points"], "world_xyz": radar}, "scene": frozen}

    def get(self, frame, *, consume=False):
        with self.lock:
            require(frame in self.pending, "reward GT lost admitted source ownership")
            if consume:
                require(frame not in self.consumed, "source consumed twice")
                self.consumed.add(frame); self.event("SOURCE_CONSUMED", frame_id=frame)
            return self.pending[frame]

    def release(self, frame, *, reason):
        with self.lock:
            owned = self.pending.pop(frame, None)
            if owned is not None:
                self.bytes -= owned.byte_size
                self.consumed.discard(frame)
                self.event("SOURCE_RELEASED", frame_id=frame, reason=reason, owned_bytes=owned.byte_size)
            return owned is not None

    def flush(self, directory, *, worker_drained=True, worker_alive=False):
        with self.lock:
            if self.manifest is not None:
                return self.manifest
            require(type(worker_drained) is bool and worker_alive in (True, False, None), "invalid worker drain observation")
            require(not worker_drained or worker_alive is False, "drained worker cannot be alive/unknown")
            if not worker_drained:
                self.failure("WORKER_DRAIN_UNVERIFIED", -1,
                    SourceOwnershipError("GT worker is alive or its drain could not be inspected"))
                # This is only an incomplete evidence snapshot. A still-running
                # worker keeps its references and may consume/release them;
                # do not seal the store or clear its pending objects beneath it.
            else:
                for frame in list(self.pending):
                    self.failure("UNDRAINED_SOURCE", frame, SourceOwnershipError("source remained at shutdown"))
                    self.release(frame, reason="SHUTDOWN_UNDRAINED")
            directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
            payload = b"".join(canonical(event)+b"\n" for event in self.events)
            with (directory/"events.jsonl").open("xb") as stream:
                stream.write(payload)
                stream.flush(); os.fsync(stream.fileno())
            manifest = {"schema": SCHEMA, "complete": worker_drained and not self.errors, "errors": list(self.errors),
                "worker_drained": worker_drained, "worker_alive": worker_alive,
                "counts": dict(Counter(event["kind"] for event in self.events)), "admitted_frames": len(self.seen),
                "pending_at_close": len(self.pending), "owned_bytes_at_close": self.bytes,
                "peak_pending": self.peak_pending, "peak_owned_bytes": self.peak_bytes,
                "max_frames": self.max_frames, "max_pending": self.max_pending, "max_bytes": self.max_bytes,
                "new_threads": 0, "sha256": {"events.jsonl": hashlib.sha256(payload).hexdigest()}}
            with (directory/"MANIFEST.json").open("xb") as stream:
                stream.write(canonical(manifest)+b"\n")
                stream.flush(); os.fsync(stream.fileno())
            if worker_drained:
                self.manifest = manifest
            return manifest


def verify_ownership_archive(directory, *, expected_identities, expected_action_identities=None):
    """Qualify complete reward-source ownership, never legacy object output."""
    from rl_agent.splitfusion_live_parity_repair_v1.target_transport import validate_identity
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "unsafe ownership archive")
    require({p.name for p in directory.iterdir()} == {"MANIFEST.json", "events.jsonl"}
            and not any(p.is_symlink() for p in directory.iterdir()), "ownership archive inventory differs")
    manifest = json.loads((directory/"MANIFEST.json").read_bytes())
    payload = (directory/"events.jsonl").read_bytes()
    require(manifest.get("schema") == SCHEMA and manifest.get("complete") is True and manifest.get("errors") == [],
            "ownership archive is incomplete or has source failures")
    require(manifest.get("worker_drained") is True and manifest.get("worker_alive") is False,
            "ownership worker drain was not established")
    require(manifest["sha256"] == {"events.jsonl": hashlib.sha256(payload).hexdigest()}, "ownership archive digest differs")
    events = [json.loads(line) for line in payload.splitlines()]
    counts = dict(Counter(event["kind"] for event in events))
    require(counts == manifest["counts"] and set(counts) == {"SOURCE_ADMITTED", "SOURCE_CONSUMED", "SOURCE_RELEASED"},
            "ownership event population contains missing/failed/foreign kinds")
    identities = list(expected_identities.values()) if isinstance(expected_identities, dict) else list(expected_identities)
    expected = {value["frame_id"]: validate_identity(value) for value in identities}
    require(bool(expected) and len(expected) == len(identities), "empty/duplicate expected ownership identities")
    actions = None
    if expected_action_identities is not None:
        rows = list(expected_action_identities.values()) if isinstance(expected_action_identities, dict) else list(expected_action_identities)
        actions = {row["frame_id"]: row for row in rows}
        require(len(actions) == len(rows) and set(actions) == set(expected), "ownership expected action population differs")
    active = {}; consumed = set(); admitted = set(); released = set()
    current_bytes = peak_bytes = peak_pending = 0
    for event in events:
        frame = event["frame_id"]
        require(frame in expected, "foreign/hold ownership frame")
        if event["kind"] == "SOURCE_ADMITTED":
            require(frame not in admitted and event["identity"] == expected[frame], "ownership admitted identity/duplicate differs")
            action = event["action_identity"]
            require(action.get("reward_requested") is True and action["frame_id"] == frame,
                    "held/nonreward source was admitted")
            require(actions is None or action == actions[frame], "admitted scientific action identity differs")
            size = event["owned_bytes"]
            require(type(size) is int and size > 0 and 0 <= event["copied_bytes"] <= size,
                    "invalid ownership byte accounting")
            admitted.add(frame); active[frame] = size; current_bytes += size
            peak_bytes = max(peak_bytes, current_bytes); peak_pending = max(peak_pending, len(active))
            require(len(active) <= manifest["max_pending"] and current_bytes <= manifest["max_bytes"],
                    "ownership bound exceeded")
        elif event["kind"] == "SOURCE_CONSUMED":
            require(frame in active and frame not in consumed, "source consumption precedes admission or repeats")
            consumed.add(frame)
        else:
            require(frame in active and frame in consumed and frame not in released
                    and event["reason"] == "GT_COMPLETE" and event["owned_bytes"] == active[frame],
                    "source released before successful composed completion")
            current_bytes -= active.pop(frame); released.add(frame)
    require(admitted == consumed == released == set(expected) and not active and current_bytes == 0,
            "incomplete reward-source admission/consumption/release population")
    require(manifest["admitted_frames"] == len(expected) <= manifest["max_frames"]
            and manifest["pending_at_close"] == manifest["owned_bytes_at_close"] == 0
            and manifest["peak_pending"] == peak_pending and manifest["peak_owned_bytes"] == peak_bytes
            and manifest["new_threads"] == 0, "ownership manifest accounting differs")
    return {"schema": SCHEMA, "passed": True, "reward_frames": len(expected),
            "admitted": len(admitted), "consumed": len(consumed), "released": len(released),
            "pending_at_close": 0, "owned_bytes_at_close": 0, "peak_pending": peak_pending,
            "peak_owned_bytes": peak_bytes, "manifest_sha256": hashlib.sha256((directory/"MANIFEST.json").read_bytes()).hexdigest()}


def owned_collector(base, *, stores, max_frames, max_pending, max_bytes, collectors=None):
    """Wrap the retained LG class; final Phase6 collector may subclass this."""
    class OwnedSourceCollector(base):
        def __init__(self, **kwargs):
            self._source_ownership = OwnershipStore(max_frames=max_frames, max_pending=max_pending, max_bytes=max_bytes)
            stores.append(self._source_ownership)
            if collectors is not None:
                collectors.append(self)
            super().__init__(**kwargs)

        def _avo_sources(self, frame_id):
            try:
                return self._source_ownership.get(int(frame_id), consume=True).sensors()
            except Exception as exc:
                self._source_ownership.failure("SOURCE_CONSUMPTION_FAILED", frame_id, exc)
                raise

        def _ground_truth(self, **kwargs):
            frame = int(kwargs["frame_id"])
            if kwargs.get("world") is None and kwargs.get("stationary_tracker") is getattr(self, "aligned_actor_tracker", object()):
                return super()._ground_truth(**kwargs)
            if self.live._run4_identity.get(frame, {}).get("reward_requested") is not True:
                return super()._ground_truth(**kwargs)
            reason = "GT_COMPLETE"
            try:
                source = self._source_ownership.get(frame)
                require(source.identity == canonical(self.live.identity_for_frame(frame))
                        and source.action == canonical(self.live._run4_identity[frame]), "admitted GT scientific identity mutated")
                with self._quality_scene_lock:
                    prior = self._quality_scenes.get(frame)
                    require(prior is None or prior is source.frozen_world, "owned frozen world replaced")
                    self._quality_scenes[frame] = source.frozen_world
                result = super()._ground_truth(**kwargs)
                require(frame in self._source_ownership.consumed, "reward GT returned without consuming its source")
                return result
            except BaseException as exc:
                reason = "GT_FAILED"
                self._source_ownership.failure(reason, frame, exc)
                raise
            finally:
                self._source_ownership.release(frame, reason=reason)
    return OwnedSourceCollector


def owned_queue(base):
    """Wrap real queue classification and task_done; do not change scheduling."""
    class OwnedSourceQueue(base):
        def __init__(self, *, classify, **kwargs):
            owner = getattr(classify, "__self__", None)
            require(owner is not None and hasattr(owner, "_source_ownership"), "owned source queue requires composed collector")
            self._ownership_owner = owner; self._ownership_store = owner._source_ownership
            self._ownership_admission = threading.local(); self._ownership_worker = threading.local()
            def guarded(item):
                klass = classify(item)
                state = self._ownership_admission.state
                if klass == "HIGH" and not state["acquired"]:
                    # Preserve the retained queue's own overflow decision. In
                    # its guarded/classifying path this lock is re-entrant;
                    # no transient 66th source is needed just to be rejected.
                    with self._cond:
                        if len(self._queues[klass]) >= self._max[klass]:
                            return klass
                    item.update(self._ownership_store.admit(owner, item))
                    state["acquired"] = True
                return klass
            super().__init__(classify=guarded, **kwargs)

        def put_nowait(self, item):
            if item is None:
                return super().put_nowait(None)
            frame = int(item["frame_id"]); state = {"acquired": False}
            require(getattr(self._ownership_admission, "state", None) is None, "nested source queue admission")
            self._ownership_admission.state = state
            try:
                return super().put_nowait(dict(item))
            except BaseException as exc:
                self._ownership_store.failure("SOURCE_ADMISSION_FAILED", frame, exc)
                if state["acquired"]:
                    self._ownership_store.release(frame, reason="QUEUE_ADMISSION_FAILED")
                raise
            finally:
                self._ownership_admission.state = None

        put = put_nowait

        def get(self, *args, **kwargs):
            item = super().get(*args, **kwargs)
            self._ownership_worker.frame = None if item is None else int(item["frame_id"])
            return item

        def task_done(self):
            frame = getattr(self._ownership_worker, "frame", None)
            try:
                if frame is not None and frame in self._ownership_store.pending:
                    self._ownership_store.failure("WORKER_SOURCE_NOT_COMPLETED", frame,
                        SourceOwnershipError("worker ended before composed source consumption/release"))
                    self._ownership_store.release(frame, reason="WORKER_FINALLY")
            finally:
                self._ownership_worker.frame = None
                super().task_done()
    return OwnedSourceQueue


@contextmanager
def install_source_ownership(*, attempt_dir, max_frames, max_pending=DEFAULT_MAX_PENDING, max_bytes=DEFAULT_MAX_BYTES):
    """Outer UE-only scope; flush after retained collector has joined GT worker."""
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_object_gt_v2 as OG
    require(not LG._INSTALLED, "source ownership must precede retained collector installation")
    stores = []; collectors = []
    def inspect_drain():
        if len(collectors) != 1:
            return {"worker_drained": False, "worker_alive": None, "reason": "NO_UNIQUE_COLLECTOR"}
        worker = getattr(collectors[0], "evaluation_worker", None)
        if worker is None or not callable(getattr(worker, "is_alive", None)):
            return {"worker_drained": False, "worker_alive": None, "reason": "NO_INSPECTABLE_GT_WORKER"}
        try:
            alive = worker.is_alive()
            require(type(alive) is bool, "GT worker liveness must be bool")
        except Exception as exc:
            return {"worker_drained": False, "worker_alive": None, "reason": f"LIVENESS_ERROR:{type(exc).__name__}:{exc}"}
        return {"worker_drained": not alive, "worker_alive": alive, "reason": "GT_WORKER_OBSERVED"}
    state = {"stores": stores, "collectors": collectors, "manifest": None,
             "drained": False, "worker_alive": None, "inspect_drain": inspect_drain}
    old_factory, old_queue = LG.build_live_gt_collector, OG.DeferringRewardPriorityGtQueueV2
    def factory(base):
        return owned_collector(old_factory(base), stores=stores, max_frames=max_frames,
                               max_pending=max_pending, max_bytes=max_bytes, collectors=collectors)
    LG.build_live_gt_collector = factory
    OG.DeferringRewardPriorityGtQueueV2 = owned_queue(old_queue)
    try:
        yield state
    finally:
        LG.build_live_gt_collector, OG.DeferringRewardPriorityGtQueueV2 = old_factory, old_queue
        require(len(stores) <= 1, "multiple UE collectors in one source ownership scope")
        store = stores[0] if stores else OwnershipStore(max_frames=max_frames, max_pending=max_pending, max_bytes=max_bytes)
        status = inspect_drain()
        state["drained"], state["worker_alive"] = status["worker_drained"], status["worker_alive"]
        state["manifest"] = store.flush(Path(attempt_dir)/"source_ownership",
            worker_drained=status["worker_drained"], worker_alive=status["worker_alive"])
