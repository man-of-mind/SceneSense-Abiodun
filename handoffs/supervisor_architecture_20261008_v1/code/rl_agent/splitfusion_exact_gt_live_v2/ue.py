"""UE-side integration: early freeze handoff replacing the last-send STAGE/PREPARE RPC path.

Unchanged and reused from the numeric runtime: pre-action ownership of the exact same-frame
source (``capture_owned_source`` inside ``stage_before_action``), full plan/identity
construction (``full_plan_bytes``), the collector/runtime/controller seams and the action-open
record. Nothing GT-related is added before action-open.

New: at ``on_plan`` (after the retained action-open stamp) a reward POLICY_DECISION enqueues
references to its owned source on ONE persistent ``gt-freeze`` thread, which freezes into a
sealed memfd and sends the descriptor to the host service. The UE worker thread additionally
sends three small messages: OPEN/PLAN (at plan), SEND (after the actual last datagram or a
send failure) and TERMINAL (actual UE receipt within 170 ms, or controller TIMEOUT).
Holds/fallbacks do no GT work. The old producer child is not started (no dual path).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time

from rl_agent.splitfusion_numeric_gt_runtime_v1 import ue_live as NG
from rl_agent.splitfusion_exact_gt_service_proto_v2 import channel as CH, freeze_codec as FZ

SCHEMA = "splitfusion.exact_gt_live.ue_result.v1"
ROOT = Path(__file__).resolve().parents[2]


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class EarlyGtUeHandoff:
    """One bounded UE-owned bridge to the host service; policy/radio state never rewritten."""

    def __init__(self, *, attempt_dir, registration, service_config):
        self.attempt_dir = Path(attempt_dir).absolute()
        self.registration = dict(registration)
        self.service_config = dict(service_config)
        self.directory = self.attempt_dir / "numeric_gt"
        NG.require(not self.directory.exists() and self.attempt_dir.is_dir(), "fresh owned UE evidence directory required")
        self.directory.mkdir(exist_ok=False)
        self.catalogue = NG.gt_bridge.FreshCatalogue(self.attempt_dir / "parity_static_catalogue")
        self.collector = None
        self.pid, self.boot_id = os.getpid(), NG.read_boot_id()
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._tickets = {}
        self._sequence = 0
        self.errors = []
        self.records = []
        self.process = self.sock = None
        self.ready = None
        self._queue = queue.Queue(maxsize=int(self.registration["exact_gt_limits"]["ue_freeze_queue"]))
        self._freeze_thread = None
        self._closed_result = None
        self.counters = dict(staged=0, decisions=0, holds_or_fallbacks=0, freeze_sent=0, freeze_refused_full=0,
                             sends_ok=0, sends_failed=0, terminals=0)

    # ------------------------------------------------------------------ plumbing
    def _send(self, value, fd=None):
        with self._send_lock:
            CH.send(self.sock, value, fd=fd, deadline=time.monotonic() + 5)

    def _fault(self, message):
        with self._lock:
            self.errors.append(str(message)[:2000])

    def _check(self):
        with self._lock:
            NG.require(not self.errors, self.errors[0] if self.errors else "early GT handoff failed")

    def attach(self, collector):
        NG.require(self.collector is None, "one collector per handoff")
        self.collector = collector
        evidence = Path(collector.edge_evidence_dir)
        NG.require(evidence.is_absolute() and evidence.is_dir(), "actual mounted edge evidence directory required")
        self.evidence_dir = evidence
        collector.live._numeric_handoff = self
        original_plan = collector.live.pipeline.plan

        def observed_plan(*args, **kwargs):
            planned = original_plan(*args, **kwargs)
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

    def prestart_from_scene(self):
        """Start the host service before any frame; it also binds the edge socket."""
        NG.require(self.collector is not None and self.process is None, "service prestart requires attached collector")
        scene = self.collector.scene_source
        NG.require(getattr(scene, "_parity_initialized", False) and type(scene._parity_registry) is dict,
                   "complete startup scene registry required before service start")
        ids = sorted(scene._parity_registry)
        population = dict(schema="splitfusion.live_exact_population.v1", ego_id=scene.ego_id,
                          registered_actors=list(scene._parity_population.values()))
        NG.require(self.catalogue.directory.is_dir() and self.catalogue.world is not None,
                   "fresh static catalogue before service start")
        reg, limits = self.registration, self.registration["exact_gt_limits"]
        from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O
        NG.require(reg["numeric_producer_binding"] == O.bound().binding_sha256, "producer source binding differs")
        engine = dict(producer_args=dict(static_catalogue_dir=str(self.catalogue.directory),
                experiment_id="exact_gt_live_" + self.attempt_dir.name, ego_id=int(self.collector.ego.id),
                dynamic_population_ids=ids, population_evidence_sha256=NG.gt_bridge.sha(population),
                max_frames=int(limits["max_frames"])),
            run_binding=reg["numeric_run_binding"], producer_binding=reg["numeric_producer_binding"],
            prediction_contract=reg["numeric_prediction_contract"], max_frames=int(limits["max_frames"]),
            max_retained_jobs=int(limits["prep_jobs"]), max_retained_bytes=int(limits["prep_jobs"]) * 64 * 1024 * 1024,
            offline_replay=False)
        self.start_service(engine)

    def start_service(self, engine):
        """Spawn the host service with an explicit producer configuration (also used by the rehearsal)."""
        reg, limits = self.registration, self.registration["exact_gt_limits"]
        score = dict(run_binding=reg["numeric_run_binding"], producer_binding=reg["numeric_producer_binding"],
                     max_jobs=int(limits["scorer_jobs"]), max_bytes=int(limits["scorer_jobs"]) * 8 * 1024 * 1024,
                     reservation_bytes=8 * 1024 * 1024, max_waiting=int(limits["max_waiting"]),
                     max_frames=4 * int(limits["max_frames"]))
        config = dict(engine=engine, score=score, limits=limits, sources=reg["exact_gt_sources"],
                      evidence_dir=str(self.directory / "service"),
                      socket_path=str(self.evidence_dir / reg["exact_gt_socket_name"]),
                      capability_path=str(self.evidence_dir / reg["exact_gt_capability_name"]),
                      binding=dict(run=reg["numeric_run_binding"], scope="exact_gt_live_v2"))
        parent, child = CH.pair()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        log = open(self.directory / "service.log", "xb")
        started = raw()
        self.process = subprocess.Popen([sys.executable, "-B", "-u", "-m", "rl_agent.splitfusion_exact_gt_live_v2.service",
            "--fd", str(child.fileno()), "--parent", str(os.getpid())], pass_fds=(child.fileno(),), cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        child.close()
        self.sock, self.log = parent, log
        CH.send(parent, dict(nonce="ue", role="service", config=config), deadline=time.monotonic() + 30)
        ready, _ = CH.receive(parent, deadline=time.monotonic() + 120)
        NG.require(ready.get("op") == "READY" and ready.get("pid") == self.process.pid, "service READY differs: " + str(ready))
        self.ready = dict(ready, started_raw_ns=started, ready_raw_ns=raw())
        self.session = ready["session"]
        self._freeze_thread = threading.Thread(target=self._freeze_worker, name="gt-freeze", daemon=True)
        self._freeze_thread.start()
        self._reader = threading.Thread(target=self._read_service, name="gt-service-reader", daemon=True)
        self._reader.start()
        self._write("SERVICE_READY.json", self.ready)

    def _write(self, name, value):
        with open(self.directory / name, "x") as stream:
            json.dump(value, stream, indent=1, sort_keys=True, default=str)

    def _read_service(self):
        while True:
            try:
                message, fd = CH.receive(self.sock, deadline=time.monotonic() + 3600)
            except BaseException as error:
                if self._closed_result is None and not getattr(self, "_closing", False):
                    self._fault(f"service channel: {type(error).__name__}: {error}")
                return
            if fd is not None:
                os.close(fd)
            if message.get("op") == "CLOSED":
                self._service_closed = message
                return
            if message.get("op") == "FAILED":
                self._fault("service failed: " + str(message.get("error")))
                return

    # ------------------------------------------------------------------ freeze thread
    def _freeze_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame_id, sequence, source, stamp, enqueued = item
            try:
                start, c0 = raw(), time.thread_time_ns()
                descriptor, chunks = FZ.freeze(source, stamp, session_id=self.session, sequence=sequence)
                frozen = raw()
                fd, size = FZ.seal(chunks)
                sealed = raw()
                del chunks, source
                try:
                    self._send(dict(op="FREEZE", frame_id=frame_id, descriptor=descriptor.decode(),
                        events=dict(freeze_enqueued=enqueued, freeze_start=start, freeze_frozen=frozen, freeze_sealed=sealed),
                        stats=dict(sealed_bytes=size, descriptor_bytes=len(descriptor),
                                   thread_cpu_ms=(time.thread_time_ns() - c0) / 1e6)), fd)
                finally:
                    os.close(fd)
                with self._lock:
                    self.counters["freeze_sent"] += 1
            except BaseException as error:
                self._fault(f"freeze frame {frame_id}: {type(error).__name__}: {error}")

    # ------------------------------------------------------------------ retained seams
    def stage_before_action(self, runtime, *, frame_id, carla_timestamp, capture_timestamp_ns, ego_pose):
        """Unchanged pre-action ownership (same function, same place, same cost)."""
        self._check()
        frame_id = int(frame_id)
        with self._lock:
            NG.require(frame_id not in self._tickets and len(self._tickets) < 1000, "bounded nonduplicate frame required")
        capture = NG.MP.gt_identity(run_id=runtime.pipeline.run_id, cell_id=runtime.pipeline.cell_id,
            stream_id=runtime.pipeline.stream_id, frame_id=frame_id, anchor_action_id=None, anchor_profile_id=None,
            capture_timestamp_ns=int(capture_timestamp_ns))
        source = NG.capture_owned_source(self.collector, frame_id=frame_id, carla_timestamp=carla_timestamp,
                                         capture_identity=capture)
        ticket = dict(frame_id=frame_id, template=capture, source=source, carla_timestamp=float(carla_timestamp),
                      collector_tick_raw_ns=int(source["frozen_world"].frozen_observed_raw_ns), source_owned_raw_ns=raw())
        with self._lock:
            self._tickets[frame_id] = ticket
            self.counters["staged"] += 1

    def on_plan(self, runtime, planned):
        self._check()
        frame_id = int(planned.frame_id)
        with self._lock:
            ticket = self._tickets.get(frame_id)
        NG.require(ticket is not None, "plan lacks pre-action owned source")
        action = NG._planned_action(planned, runtime.engine.lineage)
        if not (action["reward_requested"] and action["frame_kind"] == "POLICY_DECISION"):
            ticket["source"] = None
            ticket["decision"] = False
            with self._lock:
                self.counters["holds_or_fallbacks"] += 1
            return
        record = runtime.pipeline.decisions[-1]
        NG.require(record["frame_id"] == frame_id and record["action_open"]["domain"] == "CLOCK_MONOTONIC_RAW",
                   "retained action-open record missing")
        opened = int(record["action_open"]["ns"])
        NG.require(ticket["source_owned_raw_ns"] <= opened, "source not owned before action-open")
        plan_bytes, _, context_doc, _ = NG.full_plan_bytes(planned, lineage=runtime.engine.lineage,
            capture_template=ticket["template"], ego_pose=runtime._numeric_current_ego_pose,
            stream_id=runtime.pipeline.stream_id)
        ticket.update(decision=True, action_open_raw_ns=opened)
        self._sequence += 1
        enqueued = raw()
        try:
            self._queue.put_nowait((frame_id, self._sequence, ticket["source"], ticket["carla_timestamp"], enqueued))
        except queue.Full:
            # Never block the UE thread: the frame proceeds without GT (explicit service outcome).
            self._sequence -= 1
            with self._lock:
                self.counters["freeze_refused_full"] += 1
            ticket["freeze_refused"] = True
        ticket["source"] = None
        ticket["handoff_put_ns"] = raw() - enqueued
        self._send(dict(op="OPEN", frame_id=frame_id, decision=True, action_open_raw_ns=opened,
                        freeze_refused=bool(ticket.get("freeze_refused")),
                        events=dict(collector_tick=ticket["collector_tick_raw_ns"], source_owned=ticket["source_owned_raw_ns"],
                                    action_open=opened, freeze_enqueued_on_ue_thread=enqueued)))
        self._send(dict(op="PLAN", frame_id=frame_id, plan=plan_bytes.decode(), frame_context=context_doc))
        with self._lock:
            self.counters["decisions"] += 1

    def after_submit(self, runtime, frame_id, result):
        with self._lock:
            ticket = self._tickets.get(int(frame_id))
        NG.require(ticket is not None, "submitted frame lacks staged source")
        ticket["source"] = None
        if not ticket.get("decision"):
            return
        sent_raw = runtime.last_datagram.get(int(frame_id))
        ok = sent_raw is not None and result.get("sent") is True
        self._send(dict(op="SEND", frame_id=int(frame_id), ok=ok, raw_ns=int(sent_raw) if ok else raw()))
        with self._lock:
            self.counters["sends_ok" if ok else "sends_failed"] += 1

    def _terminal(self, frame_id, name, raw_ns):
        with self._lock:
            ticket = self._tickets.get(int(frame_id))
            if ticket is None or not ticket.get("decision") or ticket.get("terminal") is not None:
                return
            ticket["terminal"] = (name, raw_ns)
            self.counters["terminals"] += 1
        self._send(dict(op="TERMINAL", frame_id=int(frame_id), name=name, raw_ns=int(raw_ns)))

    def on_feedback(self, frame_id, receipt_raw_ns):
        with self._lock:
            ticket = self._tickets.get(int(frame_id))
        if ticket is None or not ticket.get("decision"):
            return
        if receipt_raw_ns - ticket["action_open_raw_ns"] > 170_000_000:
            return     # retained controller classifies this as late; its TIMEOUT is the terminal
        self._terminal(frame_id, "feedback_received", receipt_raw_ns)

    def on_controller_poll(self, ticket_before, resolution_before, now_ns, ticket_after):
        if ticket_before is None or resolution_before is not None or ticket_after is None:
            return
        if ticket_after.resolution is None or ticket_after.resolved_by != "TIMEOUT":
            return
        frame = ticket_after.reward_frame
        if frame is not None:
            self._terminal(frame.frame_id, "feedback_timeout", now_ns)

    # ------------------------------------------------------------------ shutdown
    def close(self):
        if self._closed_result is not None:
            return self._closed_result
        self._closing = True
        try:
            self._queue.put(None, timeout=30)
            self._freeze_thread.join(timeout=60)
            NG.require(not self._freeze_thread.is_alive(), "gt-freeze thread did not drain")
            self._send(dict(op="CLOSE"))
            self._reader.join(timeout=300)
            self.process.wait(timeout=60)
            closed = getattr(self, "_service_closed", None)
            NG.require(closed is not None, "service CLOSED reply missing")
            tickets = [dict({k: v for k, v in t.items() if k not in ("source", "template")}) for t in self._tickets.values()]
            result = dict(schema=SCHEMA, counters=dict(self.counters), service=closed["result"],
                          service_returncode=self.process.returncode, errors=list(self.errors), service_ready=self.ready)
            self._write("UE_TICKETS.json", tickets)
            self._write("EXACT_GT_UE_RESULT.json", result)
            self.sock.close(); self.log.close()
            NG.require(not self.errors and self.process.returncode == 0 and not closed["result"]["errors"],
                       "early GT service close failed: " + "; ".join(self.errors + closed["result"]["errors"]))
            self._closed_result = result
            return result
        except BaseException as exc:
            self.abort(exc)
            raise

    def abort(self, exc):
        self._fault(f"aborted: {type(exc).__name__}: {exc}")
        try:
            self._queue.put_nowait(None)
        except BaseException:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        try:
            self._write("UE_ABORT.json", dict(errors=list(self.errors), counters=dict(self.counters)))
        except BaseException:
            pass


@contextmanager
def install_early_gt_ue(*, attempt_dir, registration):
    """Same retained seams as ``install_numeric_ue``; only the handoff class differs."""
    NG.gt_bridge.bind_legacy_namespace()
    from data_collection.run_route_b_perception_collection_v2 import ClientProxy
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as RC
    NG.require(not LG._INSTALLED, "early GT installation requires fresh Run5 child")
    handoff = EarlyGtUeHandoff(attempt_dir=attempt_dir, registration=registration, service_config={})
    absent = object()
    old_load = ClientProxy.__dict__.get("load_world", absent)
    old = (LG.build_live_gt_collector, LG.build_timestamped_scene_source, old_load, RC.RewardHoldControllerV2.poll,
           U.build_run4_runtime_class, U.build_run4_collector_class, LG._INSTALLED, Q.QualitySceneSnapshotSource,
           Q.QualityPassiveSplitCollector, P.semantic_gt_3class)

    def load_world(client, *args, **kwargs):
        return handoff.catalogue.after_load(client._client.load_world(*args, **kwargs))

    def source(base, frozen):
        return NG.gt_bridge.make_complete_scene_source(old[1](base, frozen))

    def collector(base):
        inherited = old[0](base)

        class EarlyGtCollector(inherited):
            def __init__(self, **kwargs):
                self._numeric_radar_lock = threading.Lock()
                self._numeric_radar = {}
                super().__init__(**kwargs)
                handoff.catalogue.require_world(self.world)

            def _ground_truth(self, **kwargs):
                if kwargs.get("world") is not None:
                    frame_id = int(kwargs["frame_id"])
                    with self._quality_scene_lock:
                        self._quality_scenes.pop(frame_id, None)
                    self._run4_gate.close(frame_id, "NUMERIC_GT_LEGACY_BYPASS")
                    return []
                return super()._ground_truth(**kwargs)

            def finish(self):
                try:
                    result = super().finish()
                except BaseException as exc:
                    handoff.abort(exc)
                    raise
                result["numeric_gt"] = handoff.close()
                return result
        return EarlyGtCollector

    def run4_collector_class(base):
        inherited = old[5](base)

        class EarlyGtFinalCollector(inherited):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                handoff.attach(self)
                handoff.prestart_from_scene()

            def _run4_timed_build(self, real_build, **kwargs):
                result = super()._run4_timed_build(real_build, **kwargs)
                NG.require(type(result) in (tuple, list) and len(result) == 3 and type(result[1]) is dict,
                           "actual radar preparation result required")
                with self._numeric_radar_lock:
                    self._numeric_radar[float(kwargs["frame_time_s"])] = result[1]
                    NG.require(len(self._numeric_radar) <= 64, "bounded prepared radar sources required")
                return result
        return EarlyGtFinalCollector

    def poll(controller, now_ns):
        before = controller.current
        resolution = None if before is None else before.resolution
        result = old[3](controller, now_ns)
        handoff.on_controller_poll(before, resolution, now_ns, controller.current)
        return result

    def runtime_class(bindings):
        inherited = old[4](bindings)

        class EarlyGtRun4Runtime(inherited):
            def submit(self, *, frame_bgr, radar_tensor, frame_id, capture_timestamp_ns, ego_pose, stream_id,
                       carla_timestamp, capture_id, action_id=None, on_commit=None):
                NG.require(self._numeric_handoff is handoff, "runtime lacks exact owned collector")
                binding = getattr(NG.pose._LOCAL, "binding", None)
                NG.require(binding is not None and binding["owner"] is handoff.collector, "plan lacks owned-pose preparation")
                pose_doc = binding["source"].document
                NG.require(pose_doc["frame_id"] == int(frame_id) and pose_doc["stream_id"] == stream_id
                           and float(pose_doc["timestamp"]).hex() == float(carla_timestamp).hex(),
                           "plan and exact owned UE pose differ")
                handoff.stage_before_action(self, frame_id=frame_id, carla_timestamp=carla_timestamp,
                                            capture_timestamp_ns=capture_timestamp_ns, ego_pose=ego_pose)
                self._numeric_current_ego_pose = tuple(pose_doc["ego_world"])
                try:
                    result = super().submit(frame_bgr=frame_bgr, radar_tensor=radar_tensor, frame_id=frame_id,
                        capture_timestamp_ns=capture_timestamp_ns, ego_pose=ego_pose, stream_id=stream_id,
                        carla_timestamp=carla_timestamp, capture_id=capture_id, action_id=action_id, on_commit=on_commit)
                except BaseException:
                    handoff.after_submit(self, frame_id, {"sent": False})
                    raise
                finally:
                    self._numeric_current_ego_pose = None
                handoff.after_submit(self, frame_id, result)
                return result
        return EarlyGtRun4Runtime

    LG.build_live_gt_collector = collector
    LG.build_timestamped_scene_source = source
    ClientProxy.load_world = load_world
    RC.RewardHoldControllerV2.poll = poll
    U.build_run4_runtime_class = runtime_class
    U.build_run4_collector_class = run4_collector_class
    try:
        yield handoff
    except BaseException as exc:
        handoff.abort(exc)
        raise
    finally:
        (LG.build_live_gt_collector, LG.build_timestamped_scene_source, _, RC.RewardHoldControllerV2.poll,
         U.build_run4_runtime_class, U.build_run4_collector_class, LG._INSTALLED, Q.QualitySceneSnapshotSource,
         Q.QualityPassiveSplitCollector, P.semantic_gt_3class) = old
        if old_load is absent:
            del ClientProxy.load_world
        else:
            ClientProxy.load_world = old_load
        if handoff.process is not None and handoff.process.poll() is None:
            handoff.abort(RuntimeError("early GT context exited without service close"))
