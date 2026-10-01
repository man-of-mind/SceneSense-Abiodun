"""GT-free Route-B callback bridge for the B live runtime.

The live route does only three things here: freeze primitive CARLA state,
copy raw sensor evidence to a create-only disk spool, and hand the newest
prepared opportunity to the sequential B UE process.  Object projection,
semantic conversion, matching, scoring and Q_perc are deliberately absent.
An injected materializer may consume the sealed spool only after the route
thread has joined.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from . import b_ue_process_v1 as U
from . import operational_ack_v1 as A


class BRouteBridgeError(RuntimeError): pass
class OpportunitySuperseded(BRouteBridgeError): pass
class RouteStopped(BRouteBridgeError): pass
class RouteFailed(BRouteBridgeError): pass
class RouteBudgetReached(BRouteBridgeError): pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise BRouteBridgeError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BRouteBridgeError("raw evidence is not canonical") from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _primitive_vector(value: Any) -> dict[str, float]:
    return {axis: float(getattr(value, axis)) for axis in ("x", "y", "z")}


def _primitive_rotation(value: Any) -> dict[str, float]:
    return {axis: float(getattr(value, axis))
            for axis in ("pitch", "yaw", "roll")}


def _primitive_transform(value: Any) -> dict[str, Any]:
    return {"location": _primitive_vector(value.location),
            "rotation": _primitive_rotation(value.rotation)}


@dataclasses.dataclass(frozen=True, slots=True)
class PrimitiveActorV3:
    actor_id: int
    type_id: str
    bbox_location: Mapping[str, float]
    bbox_extent: Mapping[str, float]
    bbox_rotation: Mapping[str, float]
    transform: Mapping[str, Any]
    velocity: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class PrimitiveSceneV3:
    actors: tuple[PrimitiveActorV3, ...]


class PrimitiveSceneSnapshotSourceV3:
    """CARLA snapshot adapter which retains primitives, never CARLA objects."""

    def __init__(self, world: Any, *, ego_id: int,
                 refresh_interval_s: float = 2.0) -> None:
        self.world, self.ego_id = world, int(ego_id)
        self.refresh_interval_s = float(refresh_interval_s)
        self._lock = threading.Lock()
        self._static: dict[int, tuple[str, dict[str, Any]]] = {}
        self._refreshed_at = 0.0

    def refresh_static(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._refreshed_at < self.refresh_interval_s:
                return
        registry: dict[int, tuple[str, dict[str, Any]]] = {}
        for actor in self.world.get_actors():
            kind = str(getattr(actor, "type_id", ""))
            if not (kind.startswith("vehicle.") or
                    kind.startswith("walker.pedestrian.")):
                continue
            try:
                bbox = actor.bounding_box
                registry[int(actor.id)] = (kind, {
                    "location": _primitive_vector(bbox.location),
                    "extent": _primitive_vector(bbox.extent),
                    "rotation": _primitive_rotation(bbox.rotation),
                })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        with self._lock:
            self._static, self._refreshed_at = registry, now

    def capture(self, world_snapshot: Any) -> PrimitiveSceneV3:
        with self._lock:
            static = dict(self._static)
        values: list[PrimitiveActorV3] = []
        for actor_id in sorted(static):
            if actor_id == self.ego_id:
                continue
            item = world_snapshot.find(actor_id)
            if item is None:
                continue
            kind, bbox = static[actor_id]
            values.append(PrimitiveActorV3(
                actor_id=actor_id, type_id=kind,
                bbox_location=bbox["location"], bbox_extent=bbox["extent"],
                bbox_rotation=bbox["rotation"],
                transform=_primitive_transform(item.get_transform()),
                velocity=_primitive_vector(item.get_velocity())))
        return PrimitiveSceneV3(tuple(values))


class RawGroundTruthSpoolV3:
    """Create-only per-frame raw evidence; safe across abrupt termination."""

    SCHEMA = "scenesense.splitfusion.b.raw_gt_spool.v3"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        _require(not self.root.exists(), "raw GT spool root already exists")
        self.root.mkdir(parents=True)
        for name in ("identity", "scene", "semantic"):
            (self.root / name).mkdir()
        self._lock = threading.Lock()
        self._seen: dict[str, set[int]] = {
            "identity": set(), "scene": set(), "semantic": set()}

    @staticmethod
    def _name(frame_id: int, suffix: str) -> str:
        _require(type(frame_id) is int and frame_id >= 0, "invalid frame id")
        return f"{frame_id:010d}.{suffix}"

    def _write(self, relative: Path, payload: bytes) -> str:
        path = self.root / relative
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return _sha(payload)

    def _claim(self, kind: str, frame_id: int) -> None:
        with self._lock:
            _require(frame_id not in self._seen[kind],
                     f"duplicate {kind} evidence for frame {frame_id}")
            self._seen[kind].add(frame_id)

    def write_identity(self, identity: A.FrameActionIdentityV1) -> None:
        _require(type(identity) is A.FrameActionIdentityV1,
                 "foreign action identity")
        frame = identity.frame_id
        self._claim("identity", frame)
        body = _canonical({"schema": self.SCHEMA, "kind": "identity",
                           "frame_id": frame, "identity": identity.as_dict(),
                           "identity_sha256": identity.exact_sha256()})
        digest = self._write(Path("identity") / self._name(frame, "json"),
                             body + b"\n")
        self._write(Path("identity") / self._name(frame, "sha256"),
                    (digest + "\n").encode("ascii"))

    def write_scene(self, *, frame_id: int, timestamp: float,
                    scene: PrimitiveSceneV3, camera_matrix: Any,
                    camera_inverse: Any, camera_location: Any,
                    radar_world_xyz: Any) -> None:
        _require(type(scene) is PrimitiveSceneV3, "scene is not primitive")
        self._claim("scene", frame_id)
        arrays = {
            "camera_matrix": np.asarray(camera_matrix, dtype=np.float64),
            "camera_inverse": np.asarray(camera_inverse, dtype=np.float64),
            "radar_world_xyz": np.asarray(radar_world_xyz, dtype=np.float32),
        }
        for name, value in arrays.items():
            _require(value.flags.c_contiguous or value.size == 0,
                     f"{name} must be contiguous")
            arrays[name] = np.ascontiguousarray(value)
        npz_path = self.root / "scene" / self._name(frame_id, "npz")
        with npz_path.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        npz_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        document = {
            "schema": self.SCHEMA, "kind": "scene", "frame_id": frame_id,
            "carla_timestamp": float(timestamp),
            "camera_location": _primitive_vector(camera_location),
            "actors": [actor.to_dict() for actor in scene.actors],
            "array_file": npz_path.name, "array_sha256": npz_digest,
        }
        body = _canonical(document) + b"\n"
        digest = self._write(Path("scene") / self._name(frame_id, "json"), body)
        self._write(Path("scene") / self._name(frame_id, "sha256"),
                    (digest + "\n").encode("ascii"))

    def write_semantic(self, *, frame_id: int, image: Any) -> None:
        self._claim("semantic", frame_id)
        raw = bytes(image.raw_data)
        _require(raw, "semantic sensor raw bytes are empty")
        raw_name = self._name(frame_id, "bgra")
        raw_digest = self._write(Path("semantic") / raw_name, raw)
        body = _canonical({
            "schema": self.SCHEMA, "kind": "semantic_raw_bgra",
            "frame_id": frame_id, "sensor_frame": int(image.frame),
            "carla_timestamp": float(image.timestamp),
            "width": int(image.width), "height": int(image.height),
            "raw_file": raw_name, "raw_sha256": raw_digest,
        }) + b"\n"
        self._write(Path("semantic") / self._name(frame_id, "json"), body)

    def seal(self) -> Mapping[str, Any]:
        with self._lock:
            counts = {key: len(value) for key, value in self._seen.items()}
            frames = {key: sorted(value) for key, value in self._seen.items()}
        files = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                files.append({"path": str(path.relative_to(self.root)),
                              "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {"schema": self.SCHEMA, "status": "SEALED",
                    "counts": counts, "frames": frames, "files": files}
        self._write(Path("MANIFEST.json"), _canonical(manifest) + b"\n")
        return manifest


@dataclasses.dataclass(frozen=True, slots=True)
class RouteOpportunityV3:
    sequence: int
    frame_id: int
    capture_timestamp_ns: int
    action_open_monotonic_raw_ns: int
    submit_kwargs: Mapping[str, Any]


class OpportunityProcessorV3(Protocol):
    def __call__(self, opportunity: RouteOpportunityV3,
                 previous: Optional[A.OperationalOutcomeV1]) -> U.BTransmissionV1: ...


class PostrunMaterializerV3(Protocol):
    def __call__(self, spool_root: Path, evidence_root: Path) -> int: ...


@dataclasses.dataclass(slots=True)
class _Submission:
    opportunity: RouteOpportunityV3
    done: threading.Event = dataclasses.field(default_factory=threading.Event)
    result: Optional[U.BTransmissionV1] = None
    error: Optional[BaseException] = None

    def resolve(self, result: U.BTransmissionV1) -> None:
        self.result = result; self.done.set()

    def reject(self, error: BaseException) -> None:
        self.error = error; self.done.set()

    def wait(self) -> U.BTransmissionV1:
        self.done.wait()
        if self.error is not None: raise self.error
        _require(type(self.result) is U.BTransmissionV1, "missing transmission")
        return self.result


class LatestOnlyOpportunitySlotV3:
    def __init__(self) -> None:
        self._cv = threading.Condition(); self._pending = None; self._closed = False

    def publish(self, item: _Submission) -> None:
        with self._cv:
            if self._closed: raise RouteStopped("opportunity slot closed")
            displaced, self._pending = self._pending, item
            self._cv.notify_all()
        if displaced is not None:
            displaced.reject(OpportunitySuperseded(
                f"frame {displaced.opportunity.frame_id} superseded by "
                f"{item.opportunity.frame_id}"))

    def take(self, failure: Callable[[], Optional[BaseException]]) -> _Submission:
        with self._cv:
            while self._pending is None and not self._closed:
                error = failure()
                if error is not None: raise error
                self._cv.wait(0.05)
            error = failure()
            if error is not None: raise error
            if self._pending is None: raise RouteStopped("opportunity stream ended")
            item, self._pending = self._pending, None
            return item

    def close(self, error: BaseException) -> None:
        with self._cv:
            self._closed = True; item, self._pending = self._pending, None
            self._cv.notify_all()
        if item is not None: item.reject(error)

    def wake(self) -> None:
        with self._cv: self._cv.notify_all()


class BRouteBridgeV3:
    def __init__(self, *, variant: Any, feature_schema_sha256: str,
                 actor_boundary_sha256: str, processor: OpportunityProcessorV3,
                 route_driver: Callable[["BRouteBridgeV3"], Any],
                 raw_spool_root: Path,
                 postrun_materializer: Optional[PostrunMaterializerV3] = None,
                 transmitted_budget: int = U.TRANSMITTED_BUDGET) -> None:
        _require(transmitted_budget == U.TRANSMITTED_BUDGET,
                 "production bridge budget must be exactly 300")
        self.variant, self.feature_schema_sha256 = variant, feature_schema_sha256
        self.actor_boundary_sha256 = actor_boundary_sha256
        self.processor, self.route_driver = processor, route_driver
        self.transmitted_budget = transmitted_budget
        self.spool = RawGroundTruthSpoolV3(raw_spool_root)
        self.postrun_materializer = postrun_materializer
        self.slot = LatestOnlyOpportunitySlotV3()
        self.stop_requested = threading.Event(); self._lock = threading.Lock()
        self._route_started = False; self._route_thread = None
        self._route_error: Optional[BaseException] = None; self._sent = 0
        self._sealed: Optional[Mapping[str, Any]] = None

    @property
    def transmitted(self) -> int:
        with self._lock: return self._sent

    def _start(self) -> None:
        with self._lock:
            if self._route_started: return
            self._route_started = True
            self._route_thread = threading.Thread(
                target=self._route_main, name="b-route-v3", daemon=True)
            self._route_thread.start()

    def _route_main(self) -> None:
        try:
            self.route_driver(self)
            if not self.stop_requested.is_set():
                self._route_error = RouteFailed("route exited before budget")
        except RouteBudgetReached:
            if self.transmitted != self.transmitted_budget:
                self._route_error = RouteFailed("budget stop at wrong count")
        except BaseException as exc:
            self._route_error = RouteFailed(
                f"route failed: {type(exc).__name__}: {exc}")
        finally:
            self.slot.wake()

    def route_failure(self) -> Optional[BaseException]: return self._route_error

    def offer_prepared(self, opportunity: RouteOpportunityV3) -> U.BTransmissionV1:
        if self.stop_requested.is_set(): raise RouteBudgetReached("budget complete")
        item = _Submission(opportunity); self.slot.publish(item); return item.wait()

    def transmit_next(self, frame_index: int,
                      previous: Optional[A.OperationalOutcomeV1]
                      ) -> U.BTransmissionV1:
        self._start()
        _require(frame_index == self.transmitted, "non-contiguous consumer index")
        item = self.slot.take(self.route_failure)
        try:
            result = self.processor(item.opportunity, previous)
            _require(type(result) is U.BTransmissionV1, "foreign transmission")
            identity = result.identity
            _require(identity.frame_id == item.opportunity.frame_id,
                     "frame identity changed")
            _require(identity.capture_timestamp_ns
                     == item.opportunity.capture_timestamp_ns,
                     "capture identity changed")
            _require(result.action_open_monotonic_raw_ns
                     == item.opportunity.action_open_monotonic_raw_ns,
                     "action-open changed")
            self.spool.write_identity(identity)
            # Stop admission before waking the route worker.  This closes the
            # 300th-frame race in which a 301st callback could otherwise enter.
            with self._lock:
                self._sent += 1
                _require(self._sent <= self.transmitted_budget, "budget exceeded")
                reached = self._sent == self.transmitted_budget
                if reached: self.stop_requested.set()
            item.resolve(result)
            return result
        except BaseException as exc:
            item.reject(exc); self._route_error = exc
            self.stop_requested.set(); raise

    def close(self) -> None:
        self.stop_requested.set(); self.slot.close(RouteStopped("bridge closed"))
        if self._route_thread is not None:
            self._route_thread.join(15)
            _require(not self._route_thread.is_alive(), "route did not stop")
        if self._route_error is not None: raise self._route_error
        _require(self.transmitted == self.transmitted_budget,
                 "closed before transmitted budget")
        self._sealed = self.spool.seal()

    def materialize_postroute(self, evidence_root: Path) -> int:
        _require(self._sealed is not None, "raw spool is not sealed")
        _require(self._route_thread is None or not self._route_thread.is_alive(),
                 "postrun materialization attempted while route is live")
        _require(self.postrun_materializer is not None,
                 "postrun GT materializer is not bound")
        return int(self.postrun_materializer(self.spool.root, Path(evidence_root)))


class _Counters:
    def snapshot(self) -> dict[str, int]: return {}


class BridgeLiveRuntimeV3:
    def __init__(self, bridge: BRouteBridgeV3, **_kwargs: Any) -> None:
        self.bridge, self.counters, self._metrics = bridge, _Counters(), {}

    def submit(self, **kwargs: Any) -> Mapping[str, Any]:
        kwargs.pop("on_commit", None)  # the only legacy feedback ticket opener
        frame = int(kwargs["frame_id"])
        sent = self.bridge.offer_prepared(RouteOpportunityV3(
            sequence=self.bridge.transmitted, frame_id=frame,
            capture_timestamp_ns=int(kwargs["capture_timestamp_ns"]),
            action_open_monotonic_raw_ns=time.clock_gettime_ns(
                time.CLOCK_MONOTONIC_RAW), submit_kwargs=dict(kwargs)))
        self._metrics[frame] = {"payload_bytes": sent.payload_bytes,
                                "identity_sha256": sent.identity.exact_sha256()}
        return {"sent": True, "front_ms": "",
                "payload_bytes": sent.payload_bytes,
                "payload_bytes_uncompressed": "", "payload_chunks": ""}

    def take_metric(self, frame: int) -> Optional[Mapping[str, Any]]:
        row = self._metrics.get(int(frame)); return None if row is None else dict(row)

    def close(self) -> Mapping[str, Any]:
        return {"errors": [], "transport_counters": {}}


class NoLegacyFeedbackLedgerV3:
    def __init__(self, **_kwargs: Any) -> None: self.pending = {}
    def register_capture(self, **_kwargs: Any) -> None:
        raise BRouteBridgeError("legacy feedback ticket is forbidden")
    def receive_once(self) -> None: return None
    def record_expired(self, *_args: Any, **_kwargs: Any) -> None: return None
    def close(self) -> None: return None


def build_b_collector_class(base: type, bridge: BRouteBridgeV3) -> type:
    class BCollector(base):
        def on_world_tick(self, frame_id: int, route_tick: Any) -> None:
            if bridge.stop_requested.is_set():
                raise RouteBudgetReached("TRANSMITTED_BUDGET_REACHED")
            super().on_world_tick(frame_id, route_tick)

        def _feedback_worker(self) -> None:
            while not self.stop_event.wait(0.05): pass

        def _exact_record_worker(self) -> None:
            while not self.exact_retrieval_stop_event.wait(0.05): pass

        def _evaluation_worker(self) -> None:
            while True:
                try: ticket = self.evaluation_queue.get(timeout=0.05)
                except queue.Empty:
                    if self.stop_event.is_set() and self.evaluation_queue.empty(): return
                    continue
                try:
                    if ticket is None: return
                    bridge.spool.write_scene(
                        frame_id=int(ticket["frame_id"]),
                        timestamp=float(ticket["timestamp"]),
                        scene=ticket["scene"], camera_matrix=ticket["camera_matrix"],
                        camera_inverse=ticket["camera_inverse"],
                        camera_location=ticket["camera_location"],
                        radar_world_xyz=ticket["radar_points"]["world_xyz"])
                finally:
                    self.evaluation_queue.task_done()

        def _segmentation_worker(self) -> None:
            while True:
                try: frame = self.segmentation_queue.get(timeout=0.05)
                except queue.Empty:
                    if self.segmentation_stop_event.is_set(): return
                    continue
                try:
                    if frame is None: return
                    image = self._semantic_for(int(frame))
                    if image is not None:
                        bridge.spool.write_semantic(frame_id=int(frame), image=image)
                finally:
                    self.segmentation_queue.task_done()
    BCollector.__name__ = "BCollectorV3"
    return BCollector


_SEAM_LOCK = threading.Lock()


@contextlib.contextmanager
def installed_b_route_seams(bridge: BRouteBridgeV3):
    from rl_agent import ue_map_install_feedback_v1 as feedback
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    _require(_SEAM_LOCK.acquire(blocking=False), "Route-B seams already installed")
    prior = (pinned.LivePilotCellRuntime, pinned.PassiveSplitCollector,
             pinned.SceneSnapshotSource, feedback.InstallFeedbackLedger)
    try:
        pinned.LivePilotCellRuntime = lambda **kw: BridgeLiveRuntimeV3(bridge, **kw)
        pinned.PassiveSplitCollector = build_b_collector_class(prior[1], bridge)
        pinned.SceneSnapshotSource = PrimitiveSceneSnapshotSourceV3
        feedback.InstallFeedbackLedger = NoLegacyFeedbackLedgerV3
        yield pinned
    finally:
        (pinned.LivePilotCellRuntime, pinned.PassiveSplitCollector,
         pinned.SceneSnapshotSource, feedback.InstallFeedbackLedger) = prior
        _SEAM_LOCK.release()


def pinned_route_driver(route_kwargs: Mapping[str, Any]) -> Callable[[BRouteBridgeV3], Any]:
    frozen = dict(route_kwargs)
    def run(bridge: BRouteBridgeV3) -> Any:
        with installed_b_route_seams(bridge) as pinned:
            return pinned.run_route_b(**frozen)
    return run


def execute_300_with_postrun_gt(request: U.BUEProcessRequestV1,
                                bridge: BRouteBridgeV3,
                                receiver: U.AckReceiverV1) -> Mapping[str, Any]:
    """Operational ACK/state loop first; GT reconstruction strictly post-route."""
    result = U.execute_300(request, bridge, receiver)
    records = bridge.materialize_postroute(request.evidence_root / "carla_gt_postrun")
    return {**result, "postrun_ground_truth_records": records,
            "raw_ground_truth_spool": str(bridge.spool.root),
            "live_qperc_computed": False}
