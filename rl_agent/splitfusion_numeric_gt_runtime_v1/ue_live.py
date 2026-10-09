"""Process-local UE handoff for prospective live numerical ground truth.

The retained Run5 planner, actor, hold/fallback rules, deadlines and datagram
sender remain the owners of those operations.  This seam only snapshots the
already owned source before planning. Only a sent reward plan enters the
background CPU producer; encoding and IPC never block state admission.
No historical AVO prescore event or READY file is asserted here.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import os
import threading
import time

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_parity_runtime_v1 import pose
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge
from rl_agent.splitfusion_live_parity_repair_v2 import source_ownership as SO
from . import staging
from .live_clock import ASYNC_HANDOFF_ID, read_boot_id
from .process import LIVE_PROCESS_CONTRACT
from .worker_owner import OwnedClient


SCHEMA = "splitfusion.numeric_gt.edge_observed_events.v2"
HANDOFF_ID = ASYNC_HANDOFF_ID
MAX_TICKETS = 1000


class NumericUeError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise NumericUeError(message)


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def event(role, name, instant, pid, boot_id):
    require(role in ("collector", "producer", "ue") and type(name) is str,
            "unsupported UE handoff event")
    require(type(instant) is int and instant > 0 and type(pid) is int and pid > 0,
            "actual owner clock and PID required")
    require(type(boot_id) is str and bool(boot_id), "actual boot identity required")
    return dict(role=role, name=name, raw_ns=instant, pid=pid, boot_id=boot_id)


def observed_events_bytes(events, *, feedback=None):
    names = [(item["role"], item["name"]) for item in events]
    required = {("collector", "owned_tick_snapshot"), ("ue", "source_owned"),
                ("producer", "source_stage_received"),
                ("ue", "action_open"), ("ue", "last_datagram_sent")}
    require(required <= set(names) and len(names) == len(set(names)), "missing or duplicate owned UE event")
    require((names.count(("ue", "feedback_received")) + names.count(("ue", "feedback_timeout"))) == 1,
            "exactly one observed UE terminal required")
    document = dict(schema=SCHEMA, handoff_id=HANDOFF_ID, events=list(events))
    if feedback is not None:
        require(type(feedback) is dict and set(feedback) ==
                {"packet_sha256", "kind", "reason", "q_perc"}
                and C.is_digest(feedback["packet_sha256"])
                and type(feedback["kind"]) is str and bool(feedback["kind"])
                and type(feedback["reason"]) is str and bool(feedback["reason"]),
                "exact observed feedback packet metadata required")
        require(("ue", "feedback_received") in names,
                "feedback packet metadata without observed timely receipt")
        document["feedback"] = dict(feedback)
    encoded = C.canonical(document)
    require(len(encoded) <= C.MAX_METADATA_BYTES, "bounded observed event document required")
    return encoded


def _planned_action(planned, lineage):
    plan = planned.plan
    frame = plan.frame_identity
    action = dict(session_uuid=frame.session_uuid, controller_lineage_sha256=lineage,
        decision_seq=int(frame.decision_seq), ticket_seq=int(frame.ticket_seq),
        frame_id=int(plan.frame_id), tensor_seq=int(plan.tensor_seq),
        mode_id=int(plan.profile.mode_id), q_e4=int(plan.profile.q_e4),
        execution_bundle_sha256=plan.profile.execution_bundle_sha256,
        anchor_action_id=plan.profile.action_id,
        reward_requested=bool(frame.reward_requested), frame_kind=plan.kind.value)
    MP.validate_run4_identity(action)
    return action


def full_plan_bytes(planned, *, lineage, capture_template, ego_pose, stream_id):
    """Encode exactly the selected full plan; never relabel a hold as reward."""
    require(planned.plan.kind.value == "POLICY_DECISION"
            and planned.plan.frame_identity.reward_requested is True,
            "only accepted reward policy plans enter numerical producer")
    action = _planned_action(planned, lineage)
    capture = dict(capture_template, action_id=planned.plan.profile.action_id,
                   profile_id=planned.plan.profile.profile_id)
    C._capture(capture)
    require(staging.neutral_identity(capture) == capture_template
            and capture["frame_id"] == planned.frame_id
            and capture["capture_timestamp_ns"] == planned.capture_wall_ns,
            "planned capture differs from staged owned source")
    context = build_frame_context_v1(stream_id=stream_id, frame_id=planned.frame_id,
        sequence_id=planned.tensor_seq, capture_timestamp_ns=planned.capture_wall_ns,
        ego_world_x=ego_pose[0], ego_world_y=ego_pose[1], ego_world_z=ego_pose[2],
        ego_world_pitch=ego_pose[3], ego_world_yaw=ego_pose[4], ego_world_roll=ego_pose[5])
    context_doc = pose.context_document(context)
    plan_bytes = C.canonical(dict(scope=C.LIVE_SCOPE, capture=capture,
                                  action=action, frame_context=context_doc))
    return plan_bytes, capture, context_doc, action


def _strict_camera_inverse(camera):
    """Invert the frozen owned RGB matrix, as retained Run5 live GT does.

    CARLA's separately rounded native inverse has a coordinate-dependent
    residual.  It is neither the source-ownership proof nor the inverse used
    by the original live GT path.  Frame/timestamp/camera ownership is checked
    at the sensor barrier and again when the RGB image is frozen below.
    """
    require(camera.shape == (4, 4) and np.isfinite(camera).all(),
            "finite owned camera matrix required")
    try:
        inverse = np.linalg.inv(camera)
    except np.linalg.LinAlgError as exc:
        raise NumericUeError("owned native camera matrix is singular") from exc
    require(np.allclose(camera @ inverse, np.eye(4), rtol=0, atol=1e-8),
            "owned camera cannot provide strict producer inverse")
    return inverse


def capture_owned_source(collector, *, frame_id, carla_timestamp, capture_identity):
    """Freeze the source already owned by this Run5 tick, before action-open."""
    frame_id = int(frame_id)
    with collector._quality_scene_lock:
        world = collector._quality_scenes.get(frame_id)
    require(world is not None and int(world.frozen_frame_id) == frame_id,
            "exact owned-tick frozen world missing")
    population = world.population_evidence
    require(population["frame_id"] == frame_id and population["registry_complete"] is True,
            "unregistered owned dynamic population")
    with collector._avo_condition:
        rgb_meta = collector._avo_rgb.get(frame_id)
        sem_pair = collector._avo_semantic.get(frame_id)
        depth_pair = collector._avo_depth.get(frame_id)
        radar_meta = collector._avo_radar.get(frame_id)
        completed = set(collector._avo_callbacks_complete.get(frame_id, ()))
        require(all(value is not None for value in (rgb_meta, sem_pair, depth_pair, radar_meta))
                and completed == {"rgb", "semantic", "depth", "radar"},
                "source callbacks not actually completed for owned tick")
        rgb_meta = dict(rgb_meta)
        rgb_image = rgb_meta.pop("_rgb_image", None)
        require(rgb_image is not None, "native RGB callback image missing")
        sem_image, sem_meta = sem_pair
        depth_image, depth_meta = depth_pair
        rgb_meta.update(semantic_callback=dict(sem_meta), depth_callback=dict(depth_meta),
                        radar_callback=dict(radar_meta))
        rgb = SO.freeze_image(rgb_image, rgb_meta)[0]
        semantic = SO.freeze_image(sem_image, sem_meta)[0]
        depth = SO.freeze_image(depth_image, depth_meta)[0]
        camera = np.asarray(rgb.transform.get_matrix(), dtype=np.float64)
    with collector._numeric_radar_lock:
        radar_points = collector._numeric_radar.pop(float(carla_timestamp), None)
    require(radar_points is not None and "world_xyz" in radar_points,
            "exact prepared radar world points missing")
    radar, _ = SO.freeze_array(radar_points["world_xyz"])
    inverse = _strict_camera_inverse(camera)
    C._capture(capture_identity)
    require(capture_identity["frame_id"] == frame_id
            and float(rgb.timestamp).hex() == float(carla_timestamp).hex()
            and float(world.frozen_timestamp_s).hex() == float(carla_timestamp).hex(),
            "source/capture/RGB/radar/world tick differs")
    return dict(identity=capture_identity, rgb={**rgb_meta, "_rgb_image": rgb},
        semantic_image=semantic, depth_image=depth, frozen_world=world,
        camera_matrix=camera, camera_inverse=inverse, radar_world_xyz=radar,
        dynamic_snapshot_evidence=population)


@dataclass
class _Ticket:
    job: int
    frame_id: int
    capture_template: dict
    population: dict
    collector_event: dict
    stage_event: dict | None = None
    owned_source: dict | None = None
    source_owned: dict | None = None
    context_doc: dict | None = None
    carla_timestamp: float | None = None
    source_bytes: int = 0
    producer_admitted: bool = False
    retired: bool = False
    plan_bytes: bytes | None = None
    capture_identity: dict | None = None
    action: dict | None = None
    action_open: dict | None = None
    last_send: dict | None = None
    terminal: dict | None = None
    terminal_feedback: dict | None = None
    prepared: dict | None = None
    published: bool = False
    fault: str | None = None
    thread: threading.Thread | None = None
    coordinator_started: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)


class NumericUeHandoff:
    """One bounded UE-owned bridge; policy and radio state are never rewritten."""

    def __init__(self, *, attempt_dir, registration, publisher=None, client_factory=OwnedClient):
        self.attempt_dir = Path(attempt_dir).absolute()
        self.registration = dict(registration)
        self.publisher = publisher
        self.client_factory = client_factory
        self.client = None
        self.collector = None
        self.catalogue = gt_bridge.FreshCatalogue(self.attempt_dir / "parity_static_catalogue")
        self.directory = self.attempt_dir / "numeric_gt"
        require(not self.directory.exists() and self.attempt_dir.is_dir(),
                "fresh owned numeric UE evidence directory required")
        self.directory.mkdir(exist_ok=False)
        self.boot_id = read_boot_id()
        self.pid = os.getpid()
        self.socket_path = self.registration.get("edge_evidence_dir")
        self.socket_name = self.registration.get("socket_name", "numeric_gt.sock")
        if self.socket_path is not None:
            self.socket_path = Path(self.socket_path) / self.socket_name
        self._tickets = {}
        self._next_job = 0
        self._lock = threading.RLock()
        self._producer_lock = threading.RLock()
        self._producer_turn = threading.Condition(self._lock)
        self._producer_order = deque()
        self.errors = []
        self._closed_result = None
        self._aborted = False
        self._peak_owned_bytes = 0

    def _check_fault(self):
        with self._lock:
            require(not self.errors, self.errors[0] if self.errors else "numeric handoff failed")
            require(not getattr(self, "_aborted", False), "numeric handoff aborted")

    def attach(self, collector):
        require(self.collector is None, "one collector per numeric UE handoff")
        self.collector = collector
        evidence_dir = Path(collector.edge_evidence_dir)
        require(evidence_dir.is_absolute() and evidence_dir.is_dir(),
                "actual mounted edge evidence directory required")
        if self.socket_path is None:
            self.socket_path = evidence_dir / self.socket_name
        else:
            require(self.socket_path == evidence_dir / self.socket_name,
                    "registration edge socket differs from collector evidence mount")
        collector.live._numeric_handoff = self
        original = collector.live.pipeline.plan
        def observed_plan(*args, **kwargs):
            planned = original(*args, **kwargs)
            self.on_plan(collector.live, planned)
            return planned
        collector.live.pipeline.plan = observed_plan
        original_feedback = collector.live.engine.on_feedback
        def observed_feedback(feedback, *, receipt_raw_ns):
            result = original_feedback(feedback, receipt_raw_ns=receipt_raw_ns)
            if result.value == "ACCEPTED":
                self.on_feedback(int(feedback.frame_id), int(receipt_raw_ns))
            return result
        collector.live.engine.on_feedback = observed_feedback

    def _start_client(self, source):
        self._check_fault()
        if self.client is not None:
            return self.client
        from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O
        limits = self.registration["numeric_limits"]
        require(self.catalogue.directory.is_dir() and self.catalogue.world is not None,
                "fresh static catalogue not established before source stage")
        config = dict(producer_args=dict(static_catalogue_dir=str(self.catalogue.directory),
                experiment_id=str(self.registration.get("numeric_experiment_id", "numeric_live_" + self.attempt_dir.name)),
                ego_id=int(self.collector.ego.id),
                dynamic_population_ids=source["dynamic_snapshot_evidence"]["registered_population_ids"],
                population_evidence_sha256=source["dynamic_snapshot_evidence"]["population_sha256"],
                max_frames=int(limits["max_frames"])),
            run_binding=self.registration["numeric_run_binding"],
            producer_binding=self.registration["numeric_producer_binding"],
            prediction_contract=self.registration["numeric_prediction_contract"],
            max_frames=int(limits["max_frames"]),
            max_retained_jobs=int(limits["retained_jobs"]),
            max_retained_bytes=int(limits["retained_bytes"]),
            offline_replay=False, live_process_contract=LIVE_PROCESS_CONTRACT)
        require(config["producer_binding"] == O.bound().binding_sha256,
                "registered original producer source binding differs")
        self.client = self.client_factory("producer", config,
            directory=self.directory / "producer_process",
            source_sha256=self.registration["numeric_worker_sources"],
            cpu=int(limits["producer_cpu"]),
            timeout_s=float(limits["process_timeout_s"]),
            writable_attempt_root=self.attempt_dir)
        if getattr(self, "_aborted", False):
            self.client.abort()
            raise NumericUeError("producer startup completed after handoff abort")
        self._write_created("PROCESS_READY.json", C.canonical(dict(producer=self.client.ready)))
        return self.client

    def prestart_from_scene(self):
        """Preload from the real startup registry, before any frame is admitted.

        No sensor frame or source-stage event is invented here. The retained
        scene source has already refreshed its cumulative actor registry; a
        later owned tick must still validate its exact population and pose.
        """
        self._check_fault()
        require(self.collector is not None and not self._tickets and self.client is None,
                "producer prestart requires attached pre-frame collector")
        scene = self.collector.scene_source
        require(getattr(scene, "_parity_initialized", False)
                and type(scene._parity_registry) is dict
                and type(scene._parity_population) is dict,
                "complete startup scene registry required before producer preload")
        ids = sorted(scene._parity_registry)
        require(set(ids) == set(scene._parity_population)
                and int(scene.ego_id) == int(self.collector.ego.id)
                and all(type(i) is int and i >= 0 and i != scene.ego_id
                        and scene._parity_population[i]["actor_id"] == i for i in ids),
                "startup registry and population identities disagree")
        population = dict(schema="splitfusion.live_exact_population.v1",
            ego_id=scene.ego_id, registered_actors=list(scene._parity_population.values()))
        started = raw()
        with self._producer_lock:
            self._start_client(dict(dynamic_snapshot_evidence=dict(
                registered_population_ids=ids, population_sha256=gt_bridge.sha(population))))
        self._write_created("PRODUCER_PRESTART.json", C.canonical(dict(
            schema="splitfusion.numeric_gt.producer_prestart.v1",
            started_raw_ns=started, completed_raw_ns=raw(),
            registered_population_ids=ids, population_sha256=gt_bridge.sha(population),
            no_frame_admitted=True, no_stage_source_sent=True,
            owner_pid=self.pid, boot_id=self.boot_id)))

    def _write_created(self, name, payload, *, at_attempt_root=False):
        directory = self.attempt_dir if at_attempt_root else self.directory
        path = directory / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def stage_before_action(self, runtime, *, frame_id, carla_timestamp,
                            capture_timestamp_ns, ego_pose):
        """Own immutable tick data only; no encode, child startup, or RPC here."""
        self._check_fault()
        frame_id = int(frame_id)
        with self._lock:
            require(len(self._tickets) < MAX_TICKETS and frame_id not in self._tickets,
                    "bounded nonduplicate numeric UE frame required")
            self._next_job += 1
            job = self._next_job
        capture = MP.gt_identity(run_id=runtime.pipeline.run_id, cell_id=runtime.pipeline.cell_id,
            stream_id=runtime.pipeline.stream_id, frame_id=frame_id,
            anchor_action_id=None, anchor_profile_id=None,
            capture_timestamp_ns=int(capture_timestamp_ns))
        source = capture_owned_source(self.collector, frame_id=frame_id,
            carla_timestamp=carla_timestamp, capture_identity=capture)
        collector_event = event("collector", "owned_tick_snapshot",
            int(source["frozen_world"].frozen_observed_raw_ns), self.pid, self.boot_id)
        owned_event = event("ue", "source_owned", raw(), self.pid, self.boot_id)
        require(collector_event["raw_ns"] <= owned_event["raw_ns"],
                "source ownership precedes actual owned tick")
        size = sum(len(image.raw_data) for image in
                   (source["rgb"]["_rgb_image"], source["semantic_image"], source["depth_image"]))
        size += source["radar_world_xyz"].nbytes + source["camera_matrix"].nbytes + source["camera_inverse"].nbytes
        ticket = _Ticket(job, frame_id, capture, source["dynamic_snapshot_evidence"],
                         collector_event, owned_source=source, source_owned=owned_event,
                         carla_timestamp=float(carla_timestamp), source_bytes=size)
        with self._lock:
            owned_bytes = sum(t.source_bytes for t in self._tickets.values() if t.owned_source is not None) + size
            require(owned_bytes <= self.registration["numeric_limits"]["retained_bytes"],
                    "bounded immutable GT source capacity exceeded")
            self._peak_owned_bytes = max(self._peak_owned_bytes, owned_bytes)
            self._tickets[frame_id] = ticket
        return ticket

    def on_plan(self, runtime, planned):
        self._check_fault()
        frame = int(planned.frame_id)
        with self._lock:
            ticket = self._tickets.get(frame)
        require(ticket is not None, "plan lacks pre-action staged owned source")
        action = _planned_action(planned, runtime.engine.lineage)
        if not (action["reward_requested"] and action["frame_kind"] == "POLICY_DECISION"):
            return
        ego_pose = runtime._numeric_current_ego_pose
        plan_bytes, capture, context_doc, action = full_plan_bytes(planned,
            lineage=runtime.engine.lineage, capture_template=ticket.capture_template,
            ego_pose=ego_pose, stream_id=runtime.pipeline.stream_id)
        record = runtime.pipeline.decisions[-1]
        require(record["frame_id"] == frame and record["action_open"]["domain"] == "CLOCK_MONOTONIC_RAW",
                "retained action-open record missing")
        opened = event("ue", "action_open", int(record["action_open"]["ns"]), self.pid, self.boot_id)
        require(ticket.source_owned is not None
                and ticket.source_owned["raw_ns"] <= opened["raw_ns"],
                "source was not owned before real action-open")
        with self._lock:
            active = sum(t.plan_bytes is not None and not t.retired for t in self._tickets.values())
            require(active < self._outstanding_limit(),
                    "bounded outstanding GT reward capacity exceeded")
        ticket.plan_bytes, ticket.capture_identity, ticket.action = plan_bytes, capture, action
        ticket.action_open, ticket.context_doc = opened, context_doc

    def _outstanding_limit(self):
        # This queue owns immutable source snapshots, not producer reservations.
        # The producer still retains at most two jobs until durable retirement.
        limits = self.registration["numeric_limits"]
        pending = limits.get("pending_reward_jobs", 0)
        require(type(pending) is int and 0 <= pending <= 2,
                "bounded registered pending reward capacity required")
        return limits["retained_jobs"] + pending

    def after_submit(self, runtime, frame_id, result):
        with self._lock:
            ticket = self._tickets.get(int(frame_id))
        require(ticket is not None, "submitted frame lacks staged source")
        sent_raw = runtime.last_datagram.get(int(frame_id))
        if ticket.plan_bytes is None:
            # Holds and fallbacks never enter the GT worker at all.
            ticket.owned_source = None
            ticket.retired = True
            return
        if sent_raw is None or result.get("sent") is not True:
            # No producer transaction was opened. Retained Run5 still owns
            # classification of this post-assignment transmission failure.
            ticket.owned_source = None
            ticket.retired = True
            return
        ticket.last_send = event("ue", "last_datagram_sent", int(sent_raw), self.pid, self.boot_id)
        require(ticket.action_open["raw_ns"] <= ticket.last_send["raw_ns"],
                "actual last send precedes action-open")
        # The UE submission order, not background thread scheduling, owns the
        # numerical producer's strictly increasing frame/history sequence.
        with self._producer_turn:
            require(len(self._producer_order) < self._outstanding_limit()
                    and ticket.job not in self._producer_order,
                    "bounded nonduplicate sent producer admission required")
            self._producer_order.append(ticket.job)
            self._producer_turn.notify_all()
        thread = threading.Thread(target=self._finish_sent, args=(ticket,),
                                  name=f"numeric_ready_{frame_id}", daemon=True)
        ticket.thread = thread
        try:
            thread.start()
        except BaseException as exc:
            with self._producer_turn:
                self._producer_order.remove(ticket.job)
                self._producer_turn.notify_all()
            ticket.thread = None
            self._fault(ticket, exc)
            raise

    def _await_prepare_turn(self, ticket):
        deadline = time.monotonic() + 5 * float(self.registration["numeric_limits"]["process_timeout_s"]) + 5.
        with self._producer_turn:
            while True:
                require(not self._aborted and not self.errors,
                        "ordered producer admission aborted or failed")
                require(ticket.job in self._producer_order,
                        "sent producer ticket lacks ordered admission")
                admitted = sum(t.producer_admitted and not t.retired
                               for t in self._tickets.values())
                if (self._producer_order[0] == ticket.job
                        and admitted < self.registration["numeric_limits"]["retained_jobs"]):
                    require(not ticket.producer_admitted, "producer admission repeated")
                    ticket.producer_admitted = True
                    return
                remaining = deadline - time.monotonic()
                require(remaining > 0, "ordered producer admission timed out")
                self._producer_turn.wait(timeout=min(.25, remaining))

    def _release_prepare_turn(self, ticket):
        with self._producer_turn:
            require(self._producer_order and self._producer_order[0] == ticket.job,
                    "producer history turn released out of send order")
            self._producer_order.popleft()
            self._producer_turn.notify_all()

    def _prepare_sent(self, ticket):
        """Expensive packing/IPC after send, on a bounded background worker."""
        self._check_fault()
        require(ticket.last_send is not None and ticket.owned_source is not None,
                "actual sent frame and immutable owned source required")
        source = ticket.owned_source
        chunks, extras = staging.encode_stage(source, ticket.carla_timestamp)
        packet = b"".join(chunks)
        client = self._start_client(source)
        result = client.call("STAGE_SOURCE", dict(job=ticket.job, source_packet=packet,
                                                  stage_extras=extras))
        require(result["job"] == ticket.job and result["capture_template"] == ticket.capture_template,
                "producer did not accept exact owned neutral source")
        ticket.stage_event = event("producer", "source_stage_received",
            int(result["source_stage_received_raw_ns"]),
            int(result["source_stage_owner_pid"]), result["source_stage_boot_id"])
        require(ticket.last_send["raw_ns"] <= ticket.stage_event["raw_ns"],
                "producer received source before actual last send")
        # The child now owns the decoded source; don't retain native images.
        ticket.owned_source = None
        return client.call("PREPARE_STAGED", dict(job=ticket.job, plan_bytes=ticket.plan_bytes,
            capture_identity=ticket.capture_identity, frame_context=ticket.context_doc,
            population=ticket.population, archive_clocks=None, payload_filenames=None,
            prescore_filename=None))

    def _finish_sent(self, ticket):
        ticket.coordinator_started.set()
        try:
            self._await_prepare_turn(ticket)
            with self._producer_lock:
                self._check_fault()
                prepared = self._prepare_sent(ticket)
                require(prepared["job"] == ticket.job, "foreign prepared producer job")
                ticket.prepared = prepared
                self.client.call("CONFIRM_SEND", dict(job=ticket.job, plan_bytes=ticket.plan_bytes))
                packet = self.client.call("PUBLISH_READY", dict(job=ticket.job))
                source_export = self.client.call("EXPORT_LIVE_SOURCE", dict(job=ticket.job,
                    identity_bytes=prepared["identity_bytes"], semantic_sha256=prepared["semantic_sha256"]))
            require(type(packet) is bytes and type(source_export) is dict,
                    "published packet and exact producer source export required")
            require(self.socket_path is not None and self.publisher is not None,
                    "numeric edge inbox was not bound")
            self._check_fault()
            self.publisher.publish_target(self.socket_path,
                identity_bytes=prepared["identity_bytes"], packet=packet,
                semantic_sha256=prepared["semantic_sha256"])
            ticket.published = True
            self._release_prepare_turn(ticket)
            self._check_fault()
            with ticket.condition:
                while ticket.terminal is None and ticket.fault is None:
                    ticket.condition.wait(timeout=0.25)
                require(ticket.fault is None and ticket.terminal is not None,
                        "UE terminal observer failed")
            if ticket.terminal["name"] == "feedback_received":
                ticket.terminal_feedback = self._feedback_row(ticket)
            events = observed_events_bytes((ticket.collector_event, ticket.source_owned,
                                             ticket.action_open, ticket.last_send, ticket.stage_event,
                                             ticket.terminal),
                                           feedback=ticket.terminal_feedback)
            self.publisher.publish_terminal(self.socket_path,
                identity_bytes=prepared["identity_bytes"], ue_events_bytes=events,
                source_export=source_export)
            completion = self.publisher.await_completion(self.socket_path,
                identity_bytes=prepared["identity_bytes"], timeout_s=120.)
            if completion["status"] == "SCORED_DURABLE":
                proof = completion["scorer_durable_proof"]
                self._call_when_idle("SCORE_COMPLETED", dict(job=ticket.job,
                    identity_bytes=prepared["identity_bytes"],
                    semantic_sha256=prepared["semantic_sha256"]))
                self._call_when_idle("COMPLETE_LIVE_SCORED", dict(job=ticket.job,
                    identity_bytes=prepared["identity_bytes"],
                    semantic_sha256=prepared["semantic_sha256"], scorer_proof=proof))
            elif completion["status"] == "UNSCORED_REQUIRED":
                fault = completion.get("fault_feedback")
                fault_packet = None if fault is None else fault["packet_bytes"]
                received_sha = None if ticket.terminal_feedback is None else ticket.terminal_feedback["packet_sha256"]
                prepared_terminal = self._call_when_idle("PREPARE_LIVE_UNSCORED_TERMINAL",
                    dict(job=ticket.job, identity_bytes=prepared["identity_bytes"],
                         semantic_sha256=prepared["semantic_sha256"],
                         clock_descriptor_bytes=completion["clock_descriptor_bytes"],
                         reason=completion["reason"],
                         fault_feedback_packet_bytes=fault_packet,
                         ue_received_packet_sha256=received_sha))
                require(prepared_terminal["identity_bytes"] == prepared["identity_bytes"]
                        and prepared_terminal["semantic_sha256"] == prepared["semantic_sha256"]
                        and set(prepared_terminal["files"]) ==
                            {"live_clock.json", "unscored_terminal.json"},
                        "foreign unscored terminal file inventory")
                self.publisher.publish_unscored_files(self.socket_path,
                    identity_bytes=prepared["identity_bytes"], files=prepared_terminal["files"])
                deadline = time.monotonic() + 120.
                while True:
                    require(time.monotonic() < deadline, "unscored durable completion timeout")
                    completion = self.publisher.await_completion(self.socket_path,
                        identity_bytes=prepared["identity_bytes"], timeout_s=30.)
                    if completion["status"] != "UNSCORED_REQUIRED":
                        break
                    time.sleep(.02)
                require(completion["status"] == "UNSCORED_DURABLE"
                        and completion["unscored_terminal_sha256"] ==
                            C.sha256(prepared_terminal["files"]["unscored_terminal.json"]),
                        "unscored archive did not durably complete exact terminal")
                self._call_when_idle("COMPLETE_LIVE_UNSCORED", dict(job=ticket.job,
                    identity_bytes=prepared["identity_bytes"],
                    semantic_sha256=prepared["semantic_sha256"],
                    terminal_sha256=prepared_terminal["terminal_sha256"],
                    receipt=completion["receipt"],
                    relative_prefix=completion["relative_prefix"]))
            else:
                raise NumericUeError("edge did not return durable scored/unscored result")
            with self._producer_turn:
                ticket.retired = True
                self._producer_turn.notify_all()
        except BaseException as exc:
            self._fault(ticket, exc)
        finally:
            ticket.owned_source = None

    def _call_when_idle(self, operation, value):
        deadline = time.monotonic() + float(self.registration["numeric_limits"]["process_timeout_s"]) + 5.
        while True:
            with self._producer_lock:
                if self.client.pending is None:
                    return self.client.call(operation, value)
            require(time.monotonic() < deadline, "producer remained busy during durable retirement")
            time.sleep(.005)

    def _feedback_row(self, ticket):
        """Read retained UE receiver evidence after its append, never infer bytes."""
        deadline = time.monotonic() + 3.
        while time.monotonic() < deadline:
            rows = list(self.collector.live._feedback_rows)
            matches = [row for row in rows if row.get("frame_id") == ticket.frame_id
                       and row.get("receipt", {}).get("ns") == ticket.terminal["raw_ns"]]
            require(len(matches) <= 1, "duplicate exact UE receiver evidence")
            if matches:
                row = matches[0]
                require(row.get("class") == "ACCEPTED", "retained UE feedback was not accepted")
                return dict(packet_sha256=row["sha256"], kind=row["kind"],
                            reason=row["reason"], q_perc=row["q_perc"])
            time.sleep(.005)
        raise NumericUeError("retained UE feedback packet evidence did not appear")

    def _fault(self, ticket, exc):
        message = f"frame {ticket.frame_id}: {type(exc).__name__}: {exc}"
        with ticket.condition:
            ticket.fault = message
            ticket.condition.notify_all()
        with self._producer_turn:
            self.errors.append(message)
            self._producer_turn.notify_all()

    def on_feedback(self, frame_id, receipt_raw_ns):
        with self._lock:
            ticket = self._tickets.get(int(frame_id))
        if ticket is None or ticket.action_open is None or ticket.terminal is not None:
            return
        if receipt_raw_ns - ticket.action_open["raw_ns"] > 170_000_000:
            return  # retained controller classifies this as late, not timely
        with ticket.condition:
            require(ticket.terminal is None, "duplicate UE terminal")
            ticket.terminal = event("ue", "feedback_received", int(receipt_raw_ns), self.pid, self.boot_id)
            ticket.condition.notify_all()

    def on_controller_poll(self, ticket_before, resolution_before, now_ns, ticket_after):
        if ticket_before is None or resolution_before is not None or ticket_after is None:
            return
        if ticket_after.resolution is None or ticket_after.resolved_by != "TIMEOUT":
            return
        frame = ticket_after.reward_frame
        if frame is None:
            return
        with self._lock:
            handoff = self._tickets.get(int(frame.frame_id))
        if handoff is None or handoff.action_open is None:
            return
        with handoff.condition:
            if handoff.terminal is None:
                handoff.terminal = event("ue", "feedback_timeout", int(now_ns), self.pid, self.boot_id)
                handoff.condition.notify_all()

    def _write_abort_fault(self, exc):
        # A failed child cannot answer CLOSE. Preserve its already observed
        # request/response timeline and UE ticket faults before owned abort;
        # this sidecar deliberately does not claim a clean worker CLOSE.
        rows = [dict(job=ticket.job, frame_id=ticket.frame_id,
                     fault=ticket.fault, published=ticket.published,
                     action_open=ticket.action_open, last_send=ticket.last_send,
                     terminal=ticket.terminal)
                for ticket in self._tickets.values()]
        path = self.directory / "UE_ABORT_FAULT.json"
        if path.exists():
            return  # First create-only failure evidence is never overwritten.
        self._write_created("UE_ABORT_FAULT.json", C.canonical(dict(
            schema="splitfusion.numeric_gt.ue_abort_fault.v1",
            status="PRE_ABORT_FAILURE_EVIDENCE",
            cause=f"{type(exc).__name__}: {exc}",
            errors=list(self.errors), tickets=rows,
            producer_ready=getattr(self.client, "ready", None),
            producer_pending=getattr(self.client, "pending", None),
            producer_events=list(getattr(self.client, "events", ())))))

    def abort(self, exc):
        """Preserve the first failure even if retained collector.finish raises."""
        with self._producer_turn:
            self._aborted = True
            self._producer_turn.notify_all()
        for ticket in list(self._tickets.values()):
            with ticket.condition:
                if ticket.fault is None and not ticket.retired:
                    ticket.fault = f"aborted: {type(exc).__name__}: {exc}"
                ticket.condition.notify_all()
        try:
            self._write_abort_fault(exc)
        except BaseException as evidence_exc:
            self.errors.append("UE abort evidence write failed: " + str(evidence_exc))
        try:
            if self.client is not None and not getattr(self.client, "closed", False):
                self.client.abort()
        except BaseException as cleanup_exc:
            self.errors.append("UE owned producer abort failed: " + str(cleanup_exc))
        deadline = time.monotonic() + 3.
        for ticket in list(self._tickets.values()):
            if ticket.thread is not None and ticket.thread is not threading.current_thread():
                ticket.thread.join(timeout=max(0., deadline - time.monotonic()))
            if ticket.thread is None or not ticket.thread.is_alive():
                ticket.owned_source = None
            else:
                self.errors.append(f"UE owned handoff still draining at abort: frame {ticket.frame_id}")

    def close(self):
        if getattr(self, "_closed_result", None) is not None:
            return self._closed_result
        try:
            tickets = list(self._tickets.values())
            # A scored wait can run to its bounded edge horizon; an unscored
            # branch then needs a second bounded archive/readback horizon.
            deadline = time.monotonic() + 250.
            for ticket in tickets:
                if ticket.thread is not None:
                    ticket.thread.join(timeout=max(0., deadline - time.monotonic()))
                    require(not ticket.thread.is_alive(), "numeric UE handoff thread did not drain")
            worker_scope = []
            if self.client is not None:
                closed = self.client.close()
                self._write_created("PRODUCER_PROCESS_TIMELINE.json",
                                    C.canonical(dict(schema="splitfusion.numeric_gt.ue_process_timeline.v1",
                                                     events=list(self.client.events))))
                self._write_created("PROCESS_CLOSE.json", C.canonical(dict(producer=closed)))
                worker_scope.append(dict(role="producer", ready=self.client.ready,
                    close=closed, exitcode=self.client.process.returncode,
                    source_sha256=self.registration["numeric_worker_sources"]))
            self._write_created("numeric_ue_scope.json", C.canonical(dict(
                schema="splitfusion.numeric_gt.worker_scope.v1", workers=worker_scope)),
                at_attempt_root=True)
            require(not self.errors, "; ".join(self.errors))
            self._closed_result = dict(schema="splitfusion.numeric_gt.ue_handoff_result.v2",
                handoff_id=HANDOFF_ID, source_owned=len(tickets),
                staged=sum(ticket.stage_event is not None for ticket in tickets),
                published=sum(ticket.published for ticket in tickets),
                peak_owned_bytes=getattr(self, "_peak_owned_bytes", 0), errors=list(self.errors))
            return self._closed_result
        except BaseException as exc:
            self.abort(exc)
            raise


@contextmanager
def install_numeric_ue(*, attempt_dir, registration, publisher=None, client_factory=OwnedClient):
    """Install only in a fresh Run5 UE child, before its collector factory runs."""
    gt_bridge.bind_legacy_namespace()
    from data_collection.run_route_b_perception_collection_v2 import ClientProxy
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as R
    require(not LG._INSTALLED, "numeric UE installation requires fresh Run5 child")
    if publisher is None:
        from .edge_live import NumericEdgeInbox
        publisher = NumericEdgeInbox
    handoff = NumericUeHandoff(attempt_dir=attempt_dir, registration=registration,
                               publisher=publisher, client_factory=client_factory)
    absent = object()
    old_load = ClientProxy.__dict__.get("load_world", absent)
    old = (LG.build_live_gt_collector, LG.build_timestamped_scene_source,
           old_load, R.RewardHoldControllerV2.poll,
           U.build_run4_runtime_class, U.build_run4_collector_class,
           LG._INSTALLED, Q.QualitySceneSnapshotSource, Q.QualityPassiveSplitCollector,
           P.semantic_gt_3class)

    def load_world(client, *args, **kwargs):
        return handoff.catalogue.after_load(client._client.load_world(*args, **kwargs))

    def source(base, frozen):
        return gt_bridge.make_complete_scene_source(old[1](base, frozen))

    def collector(base):
        inherited = old[0](base)
        class NumericCollector(inherited):
            def __init__(self, **kwargs):
                self._numeric_radar_lock = threading.Lock()
                self._numeric_radar = {}
                super().__init__(**kwargs)
                handoff.catalogue.require_world(self.world)

            def _ground_truth(self, **kwargs):
                # Numerical target owns reward GT; no second legacy AVO build.
                if kwargs.get("world") is not None:
                    frame_id = int(kwargs["frame_id"])
                    with self._quality_scene_lock:
                        self._quality_scenes.pop(frame_id, None)
                    # Retained Run4 opened this LOW-work deferral at decision
                    # open.  The old GT worker has now explicitly bypassed
                    # its ticket; do not leave later LOW work blocked.
                    self._run4_gate.close(frame_id, "NUMERIC_GT_LEGACY_BYPASS")
                    return []
                return super()._ground_truth(**kwargs)

            def finish(self):
                # Retained pose/Run5 finish still drains and preserves its own
                # diagnostics; no old reward AVO bundle was admitted above.
                try:
                    result = super().finish()
                except BaseException as exc:
                    handoff.abort(exc)
                    raise
                result["numeric_gt"] = handoff.close()
                return result
        return NumericCollector

    def run4_collector_class(base):
        inherited = old[5](base)
        class NumericFinalCollector(inherited):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Run4 has now installed its timed radar builder, planner,
                # gate, and receiver.  No world tick is admitted before this
                # constructor returns to the retained route.
                handoff.attach(self)
                handoff.prestart_from_scene()

            def _run4_timed_build(self, real_build, **kwargs):
                result = super()._run4_timed_build(real_build, **kwargs)
                require(type(result) in (tuple, list) and len(result) == 3
                        and type(result[1]) is dict, "actual radar preparation result required")
                with self._numeric_radar_lock:
                    self._numeric_radar[float(kwargs["frame_time_s"])] = result[1]
                    require(len(self._numeric_radar) <= 64, "bounded prepared radar sources required")
                return result
        return NumericFinalCollector

    def poll(controller, now_ns):
        before = controller.current
        resolution = None if before is None else before.resolution
        result = old[3](controller, now_ns)
        handoff.on_controller_poll(before, resolution, now_ns, controller.current)
        return result

    def runtime_class(bindings):
        inherited = old[4](bindings)
        class NumericRun4Runtime(inherited):
            def submit(self, *, frame_bgr, radar_tensor, frame_id, capture_timestamp_ns,
                       ego_pose, stream_id, carla_timestamp, capture_id, action_id=None,
                       on_commit=None):
                require(self._numeric_handoff is handoff,
                        "numeric runtime lacks exact owned collector")
                binding = getattr(pose._LOCAL, "binding", None)
                require(binding is not None and binding["owner"] is handoff.collector,
                        "numeric plan lacks retained owned-pose preparation")
                pose_doc = binding["source"].document
                require(pose_doc["frame_id"] == int(frame_id)
                        and pose_doc["stream_id"] == stream_id
                        and float(pose_doc["timestamp"]).hex() == float(carla_timestamp).hex(),
                        "numeric plan and exact owned UE pose differ")
                handoff.stage_before_action(self, frame_id=frame_id,
                    carla_timestamp=carla_timestamp,
                    capture_timestamp_ns=capture_timestamp_ns, ego_pose=ego_pose)
                self._numeric_current_ego_pose = tuple(pose_doc["ego_world"])
                try:
                    result = super().submit(frame_bgr=frame_bgr, radar_tensor=radar_tensor,
                        frame_id=frame_id, capture_timestamp_ns=capture_timestamp_ns,
                        ego_pose=ego_pose, stream_id=stream_id,
                        carla_timestamp=carla_timestamp, capture_id=capture_id,
                        action_id=action_id, on_commit=on_commit)
                except BaseException:
                    handoff.after_submit(self, frame_id, {"sent": False})
                    raise
                finally:
                    self._numeric_current_ego_pose = None
                handoff.after_submit(self, frame_id, result)
                return result
        return NumericRun4Runtime

    LG.build_live_gt_collector = collector
    LG.build_timestamped_scene_source = source
    ClientProxy.load_world = load_world
    R.RewardHoldControllerV2.poll = poll
    U.build_run4_runtime_class = runtime_class
    U.build_run4_collector_class = run4_collector_class
    try:
        yield handoff
    except BaseException as exc:
        handoff.abort(exc)
        raise
    finally:
        (LG.build_live_gt_collector, LG.build_timestamped_scene_source,
         _, R.RewardHoldControllerV2.poll,
         U.build_run4_runtime_class, U.build_run4_collector_class,
         LG._INSTALLED, Q.QualitySceneSnapshotSource, Q.QualityPassiveSplitCollector,
         P.semantic_gt_3class) = old
        if old_load is absent:
            del ClientProxy.load_world
        else:
            ClientProxy.load_world = old_load
        # A retained child exception may bypass collector.finish.  Reap only
        # this explicitly owned producer, never a process-name match.
        if handoff.client is not None and not getattr(handoff.client, "closed", False):
            handoff.abort(NumericUeError("numeric context exited without producer close"))
