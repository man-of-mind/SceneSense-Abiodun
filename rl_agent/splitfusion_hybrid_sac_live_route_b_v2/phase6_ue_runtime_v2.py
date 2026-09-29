"""Phase 6 UE side: sensor SI/P40 -> telemetry -> state -> actor -> SFD4 -> radio.

:class:`Run4FramePipelineV2` is the pure per-frame core (unit-testable with
fakes).  :class:`Run4LiveRuntimeV2` is thin glue over the unchanged direct UE
runtime: it reuses the hash-verified, once-preloaded front/ranker/AE/codec
(``live_pilot_runtime._preload_ue``), the SFD1 chunking and socket, the deadline
stages and the direct control receiver, replacing only ``submit`` and the
control loop.  :func:`build_run4_collector_class` adds the two sensor facts the
pinned collector does not expose -- the RGB receipt instant on
``CLOCK_MONOTONIC_RAW`` and the exact radar window used for P40.

Clock discipline: physical capture wall time names the CARLA frame and map
AoI; ``CLOCK_MONOTONIC_RAW`` times state commit, action open, feedback receipt
and the 170-ms deadline.  The two never meet in one subtraction.

Importing this module starts nothing and does not initialize CUDA.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

from . import continuous_execution_v2 as X
from . import live_state_v2 as LS
from . import phase6_decision_engine_v2 as E
from . import run4_live_wire_v2 as W
from . import run4_map_protocol_v2 as MP
from . import ue_telemetry_provider_v2 as T

__all__ = [
    "SceneDescriptorsV2",
    "compute_scene_descriptors",
    "PreparedRun4FrameV2",
    "Run4FramePipelineV2",
    "Run4LiveBindingsV2",
    "build_run4_runtime_class",
    "build_run4_collector_class",
]

SCENE_SOURCE = "phase6_live_sensor_preparation:SI_P40_V1"


class Phase6UeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase6UeError(message)


@dataclasses.dataclass(frozen=True)
class SceneDescriptorsV2:
    camera_si: Optional[float]
    radar_p40: Optional[float]
    camera_status: str
    radar_status: str
    source_raw_ns: int
    available_raw_ns: int


def compute_scene_descriptors(frame_bgr: Any, window_meta: Any, *, source_raw_ns: int,
                              clock: Callable[[], int] = T.raw_now_ns) -> SceneDescriptorsV2:
    """The frozen live SI/P40 recipe (splitfusion_scene_descriptor_live_v1)."""
    import cv2

    from rl_agent.splitfusion_hybrid_sac_v1.scene_descriptors import (
        SceneDescriptorError,
        camera_spatial_information,
    )
    from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (
        profile_current_sweep_p40,
    )

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (768, 448), interpolation=cv2.INTER_LINEAR)
        camera = float(camera_spatial_information(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
        camera_status = "VALID"
    except (SceneDescriptorError, cv2.error, TypeError, ValueError) as exc:
        camera, camera_status = None, type(exc).__name__
    radar = profile_current_sweep_p40(window_meta, enabled=True)
    p40 = radar["radar_p40"]
    radar_value = float(p40) if radar["scene_descriptor_radar_status"] == "VALID" else None
    return SceneDescriptorsV2(camera_si=camera, radar_p40=radar_value,
                              camera_status=camera_status,
                              radar_status=str(radar["scene_descriptor_radar_status"]),
                              source_raw_ns=int(source_raw_ns), available_raw_ns=int(clock()))


@dataclasses.dataclass(frozen=True)
class PreparedRun4FrameV2:
    plan: E.FramePlanV2
    envelope: X.ExecutionEnvelopeV3
    wire: bytes
    identity: Mapping[str, Any]
    gt_identity: Mapping[str, Any]
    anchor_profile_id: Optional[str]


class _AdoptingProvider:
    def __init__(self, provider: T.UeTelemetryProviderV2, decision_session: str) -> None:
        self._provider = provider
        self._session = decision_session

    def snapshot(self) -> T.TelemetrySnapshotV2:
        return E.adopt_snapshot(self._provider.snapshot(),
                                root_session=self._provider.session_uuid,
                                decision_session=self._session)


class Run4FramePipelineV2:
    """Pure per-frame composition; all telemetry parsing stays in the provider."""

    def __init__(self, *, engine: E.Run4DecisionEngineV2,
                 continuous_ue: X.ContinuousUERuntimeV2,
                 provider: T.UeTelemetryProviderV2, stream_id: str, run_id: str,
                 cell_id: str, clock: Callable[[], int] = T.raw_now_ns) -> None:
        self.engine = engine
        self._ue = continuous_ue
        self._provider = provider
        self.stream_id = str(stream_id)
        self.run_id, self.cell_id = str(run_id), str(cell_id)
        self._clock = clock
        self.decisions: list[dict[str, Any]] = []

    def process(self, *, frame_id: int, capture_wall_ns: int,
                ego_pose: Sequence[float], scene: SceneDescriptorsV2,
                input_7ch: Callable[[], Any]) -> PreparedRun4FrameV2:
        record: dict[str, Any] = {"frame_id": int(frame_id),
                                  "capture": W.wall(int(capture_wall_ns)).to_dict()}

        def build_state(identity, previous):
            snapshot, boundary, latency = T.open_decision(
                _AdoptingProvider(self._provider, identity.session_uuid), identity,
                clock=self._clock)
            evidence = T.assemble_radio_evidence(
                snapshot, identity=identity, boundary=boundary,
                payload_enqueue_timestamp_ns=boundary.action_open_timestamp_ns + 1)
            reasons = list(evidence.fallback_reasons)
            if scene.camera_si is None:
                reasons.append(f"CAMERA_SI:{scene.camera_status}")
            if scene.radar_p40 is None:
                reasons.append(f"RADAR_P40:{scene.radar_status}")
            record.update({
                "snapshot_latency_ns": latency,
                "state_commit": W.raw(boundary.state_commit_timestamp_ns).to_dict(),
                "action_open": W.raw(boundary.action_open_timestamp_ns).to_dict(),
                "mcs_age_ns": evidence.mcs_age_ns, "backlog_age_ns": evidence.backlog_age_ns,
                "radio_reasons": list(evidence.fallback_reasons)})
            if reasons:
                raise contract.ExternalFallbackRequired("; ".join(reasons))
            camera, radar = LS.scene_observations(
                identity, sample_seq=int(frame_id), camera_si=scene.camera_si,
                radar_p40=scene.radar_p40, source=SCENE_SOURCE,
                source_timestamp_ns=scene.source_raw_ns,
                available_timestamp_ns=scene.available_raw_ns, clock_domain=T.CLOCK_DOMAIN)
            guarded, features = LS.build_live_state(
                identity=identity, boundary=boundary, camera_si=camera, radar_p40=radar,
                prior_ul_mcs=evidence.prior_ul_mcs,
                pre_action_rlc_backlog=evidence.pre_action_rlc_backlog, previous=previous)
            record["features"] = list(features.as_tuple())
            return guarded, features

        plan = self.engine.plan_frame(frame_id=int(frame_id),
                                      capture_wall_ns=int(capture_wall_ns),
                                      now_raw_ns=self._clock(), build_state=build_state)
        record.update({"kind": plan.kind.value, "tensor_seq": plan.tensor_seq,
                       "mode_id": plan.profile.mode_id, "q_e4": plan.profile.q_e4,
                       "anchor_action_id": plan.profile.action_id,
                       "fallback_reasons": list(plan.fallback_reasons),
                       "decision_session_uuid": plan.decision_session_uuid})
        self.decisions.append(record)
        # From here the tensor is assigned: any failure is an infrastructure fault.
        try:
            context = build_frame_context_v1(
                stream_id=self.stream_id, frame_id=int(frame_id),
                sequence_id=int(plan.tensor_seq), capture_timestamp_ns=int(capture_wall_ns),
                ego_world_x=ego_pose[0], ego_world_y=ego_pose[1], ego_world_z=ego_pose[2],
                ego_world_pitch=ego_pose[3], ego_world_yaw=ego_pose[4],
                ego_world_roll=ego_pose[5])
            prepared = self._ue.prepare(plan.profile, input_7ch(), plan.frame_identity)
            wire = W.pack_sfd4(prepared.envelope, context)
            identity = MP.run4_identity(prepared.envelope)
            MP.validate_run4_identity(identity)
        except Exception as exc:  # noqa: BLE001 - classified, never a timeout
            self.engine.transport_failed(plan, f"{type(exc).__name__}: {exc}")
            raise  # pragma: no cover - transport_failed always raises
        return PreparedRun4FrameV2(
            plan=plan, envelope=prepared.envelope, wire=wire, identity=identity,
            gt_identity=MP.gt_identity(
                run_id=self.run_id, cell_id=self.cell_id, stream_id=self.stream_id,
                frame_id=int(frame_id), anchor_action_id=plan.profile.action_id,
                anchor_profile_id=plan.profile.profile_id,
                capture_timestamp_ns=int(capture_wall_ns)),
            anchor_profile_id=plan.profile.profile_id)


@dataclasses.dataclass(frozen=True)
class Run4LiveBindingsV2:
    controller_lineage_sha256: str
    tracer_dir: Path
    t_messages: Path
    ue_relay_port: int
    evidence_dir: Path


class _SqueezedFront:
    """The UE front with the qualified single-frame C2 normalisation."""

    def __init__(self, front: Any, device: Any) -> None:
        self._front, self._device = front, device

    def __call__(self, input_7ch: Any) -> Any:
        import torch

        c2 = self._front(input_7ch)
        if isinstance(c2, torch.Tensor):
            if c2.ndim == 4 and int(c2.shape[0]) == 1:
                c2 = c2[0]
            _require(c2.ndim == 3, "UE front must return one C2 frame")
            _require(c2.device == self._device, "UE front C2 device drift")
        return c2


def build_run4_runtime_class(bindings: Run4LiveBindingsV2):  # pragma: no cover - live
    """Return the live runtime class (needs CUDA; constructed only live)."""
    from rl_agent.splitfusion_direct_edge_map_v1 import live_pilot_runtime_direct_v1 as DR
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
    from phase2_map_sharing.transport import chunk_payload

    from . import frozen_actor_v2 as FA
    from . import reward_hold_controller_v2 as R

    class Run4LiveRuntimeV2(DR.DirectLivePilotCellRuntime):
        def __init__(self, *, ledger: Any, **kwargs: Any) -> None:
            self._run4_ready = threading.Event()
            self._engine_lock = threading.Lock()
            self._gt_identity: dict[int, dict[str, Any]] = {}
            self._run4_identity: dict[int, dict[str, Any]] = {}
            self._feedback_rows: list[dict[str, Any]] = []
            self.scene_hooks: Optional[Callable[[int, float], tuple]] = None
            super().__init__(ledger=ledger, **kwargs)     # preloads models once
            self.contract = dec.load_dynamic_execution_contract()
            self.actor = FA.load_registered_actor()
            self.provider = T.UeTelemetryProviderV2(bridge=T.CausalClockBridgeV2())
            handlers = {"NRUE_MAC_DCI_GRANT": self.provider.on_dci,
                        "NRUE_MAC_RLC_BUFFER_STATUS": self.provider.on_rlc,
                        "NR_PDCP_TX_SDU": self.provider.on_pdcp}
            self.readers = []
            for event, handler in handlers.items():
                reader = T.LiveEventReaderV2(event, handler, self.provider)
                reader.start(T.csv_reader_argv(bindings.tracer_dir, bindings.t_messages,
                                               bindings.ue_relay_port, event),
                             cwd=Path(bindings.tracer_dir))
                self.readers.append(reader)
            self.audit = T.AuditWriterV2(self.readers, bindings.evidence_dir / "telemetry_live")
            self.audit.start()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not (
                    self.provider.snapshot().all_readers_alive
                    and len(self.provider.unbound_ue_candidates) == 1):
                time.sleep(0.05)
            _require(len(self.provider.unbound_ue_candidates) == 1,
                     "exactly one UE must be visible in RLC telemetry")
            rnti, ue_id = sorted(self.provider.unbound_ue_candidates)[0]
            self.provider.bind_ue(rnti=rnti, oai_ue_id=ue_id)
            self.engine = E.Run4DecisionEngineV2(
                contract_=self.contract, actor=self.actor, ue_id=self.provider.ue_label,
                controller_lineage_sha256=bindings.controller_lineage_sha256)
            continuous = X.ContinuousUERuntimeV2(
                self.contract, front=_SqueezedFront(self.ue._front, self.device),
                ranker=self.ue._ranker, ae_encoders=dict(self.ue._ae_encoders),
                codec=self.ue._codec)
            self.pipeline = Run4FramePipelineV2(
                engine=self.engine, continuous_ue=continuous, provider=self.provider,
                stream_id=f"ue288_{self.cell['cell_id']}",
                run_id=str(self.campaign["campaign_id"]), cell_id=str(self.cell["cell_id"]))
            self._chunk_payload = chunk_payload
            self._run4_ready.set()

        @property
        def infrastructure_fault(self) -> Optional[str]:
            return self.engine.faulted if hasattr(self, "engine") else None

        def register_sensor_ready(self, frame_id: int, *, observed_wall_ns: int) -> None:
            return None

        def identity_for_frame(self, frame_id: int) -> dict[str, Any]:
            return dict(self._gt_identity[int(frame_id)])

        def submit(self, *, frame_bgr, radar_tensor, frame_id, capture_timestamp_ns,
                   ego_pose, stream_id, carla_timestamp, capture_id, action_id=None,
                   on_commit=None) -> dict[str, Any]:
            del action_id  # the Run-4 engine owns the per-frame action
            base._require(not self.errors, self.errors[0] if self.errors else "failed")
            base._require(self.thread.is_alive(), "control receiver exited")
            _require(self.engine.faulted is None, str(self.engine.faulted))
            try:
                base.check_deadline(base.UE_STAGE_AFTER_PREPARATION,
                                    capture_timestamp_ns, self.deadline_s)
            except base.DeadlineExpired as expired:   # before any assignment
                return self._record_stale(expired, frame_id=frame_id,
                                          capture_id=capture_id, stream_id=stream_id,
                                          profile=self.profile)
            window_meta, rgb_raw_ns = self.scene_hooks(int(frame_id), float(carla_timestamp))
            scene = compute_scene_descriptors(frame_bgr, window_meta,
                                              source_raw_ns=rgb_raw_ns)
            started = time.perf_counter_ns()
            with self._engine_lock:
                prepared = self.pipeline.process(
                    frame_id=int(frame_id), capture_wall_ns=int(capture_timestamp_ns),
                    ego_pose=ego_pose, scene=scene,
                    input_7ch=lambda: base._prepare_live_input(frame_bgr, radar_tensor,
                                                               self.device))
            try:
                chunks = self._chunk_payload(prepared.wire, message_id=int(frame_id),
                                             chunk_bytes=self.chunk_bytes)
                self.ledger.stage(capture_id, identity=prepared.identity,
                                  anchor_profile_id=prepared.anchor_profile_id)
                self._gt_identity[int(frame_id)] = dict(prepared.gt_identity)
                self._run4_identity[int(frame_id)] = dict(prepared.identity)
                if on_commit is not None:
                    on_commit()
                with self.lock:
                    self.metrics[int(frame_id)] = {
                        "capture_id": str(capture_id), "frame_id": int(frame_id),
                        "stream_id": str(stream_id),
                        "action_id": ("" if prepared.plan.profile.action_id is None
                                      else prepared.plan.profile.action_id),
                        "profile_id": prepared.plan.profile.profile_id or "",
                        "model_family": prepared.plan.profile.family,
                        "quantizer": prepared.plan.profile.quantizer,
                        "q_e4": prepared.plan.profile.q_e4,
                        "routing_tag": prepared.plan.profile.routing_tag,
                        "run4_identity": dict(prepared.identity),
                        "run4_frame_kind": prepared.plan.kind.value,
                        "carla_timestamp": float(carla_timestamp),
                        "capture_started_ns": started, "sfd4_bytes": len(prepared.wire),
                        "datagrams": len(chunks),
                    }
                for chunk in chunks:
                    self.sender.sendto(chunk, self.remote)
            except Exception as exc:  # noqa: BLE001
                with self._engine_lock:
                    self.engine.transport_failed(prepared.plan, f"{type(exc).__name__}: {exc}")
            with self.lock:
                self.metrics[int(frame_id)]["service_deadline_at"] = base.deadline_at_s(
                    capture_timestamp_ns, self.service_deadline_s)
                self.metrics[int(frame_id)]["ack_timeout_at"] = base.deadline_at_s(
                    capture_timestamp_ns, self.ack_timeout_s)
                self.sent += 1
            return {"sent": True, "front_ms": (time.perf_counter_ns() - started) / 1e6,
                    "payload_bytes": len(prepared.wire), "payload_bytes_uncompressed": "",
                    "payload_chunks": len(chunks)}

        def _result_loop(self) -> None:
            import socket as _socket

            self._run4_ready.wait()
            while not self.stop_event.is_set():
                try:
                    datagram, address = self.receiver.recvfrom(65535)
                except _socket.timeout:
                    continue
                except OSError:
                    return
                receipt_raw = T.raw_now_ns()
                received_at = time.time()
                try:
                    if datagram[:4] == W.MAGIC_FB:
                        feedback, reason = W.decode_feedback(datagram)
                        with self._engine_lock:
                            outcome = self.engine.on_feedback(feedback,
                                                              receipt_raw_ns=receipt_raw)
                        self._feedback_rows.append({
                            "frame_id": feedback.frame_id, "class": outcome.value,
                            "kind": feedback.kind, "reason": reason.name,
                            "q_perc": feedback.q_perc,
                            "receipt": W.raw(receipt_raw).to_dict(),
                            "source_address": f"{address[0]}:{address[1]}",
                            "bytes": len(datagram),
                            "sha256": hashlib.sha256(datagram).hexdigest()})
                        continue
                    message = DR.protocol.decode(datagram)
                    DR.protocol.assert_no_object_records(message)
                    schema = str(message.get("schema") or "")
                    _require(schema in (MP.RUN4_FEEDBACK_SCHEMA, MP.RUN4_TERMINAL_SCHEMA),
                             f"unknown control schema {schema!r}")
                    frame_id = int(message["frame_id"])
                    _require(self._run4_identity.get(frame_id)
                             == dict(message.get("run4_identity") or {}),
                             "control message Run-4 identity drift")
                    message["_feedback_bytes"] = len(datagram)
                    self.ledger.enqueue(message, received_at)
                    with self.lock:
                        if bool(message.get("terminal")) and frame_id not in self._terminal_frames:
                            self._terminal_frames[frame_id] = str(message.get("outcome") or "")
                            self._published_frames.add(frame_id)
                            self.completed += 1
                except R.ConflictingFeedbackError as exc:
                    with self.lock:
                        self.errors.append(f"conflicting reward feedback: {exc}")
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self.errors.append(f"{type(exc).__name__}: {exc}")
                    return

        def close(self) -> dict[str, Any]:
            with self._engine_lock:
                for controller in self.engine.controllers:
                    controller.poll(T.raw_now_ns())
            for reader in getattr(self, "readers", []):
                reader.stop()
            if getattr(self, "audit", None) is not None:
                self.audit.stop()
            evidence = bindings.evidence_dir
            evidence.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema": "scenesense.run4_live_v2.phase6_ue_evidence.v1",
                "coverage": self.engine.coverage(),
                "counters": dataclasses.asdict(self.engine.counters),
                "faulted": self.engine.faulted,
                "sessions": [c.session_uuid for c in self.engine.controllers],
                "resolutions": [r.to_canonical_dict()
                                for c in self.engine.controllers for r in c.resolutions()],
                "feedback_ledgers": [c.ledger.feedback for c in self.engine.controllers],
                "frames": [{**{name: getattr(f, name) for name in f.__dataclass_fields__
                               if name != "action"},
                            "action": f.action.to_canonical_dict(),
                            "session_uuid": c.session_uuid}
                           for c in self.engine.controllers for f in c.ledger.frames],
                "transmitted_identities": [
                    {"frame_id": frame_id, "run4_identity": identity}
                    for frame_id, identity in sorted(self._run4_identity.items())],
                "opportunities": list(self.engine.opportunities),
                "fallback_log": list(self.engine.fallback_log),
                "decisions": list(self.pipeline.decisions),
                "feedback_rows": list(self._feedback_rows),
                "telemetry_counters": dict(self.provider.counters),
                "bridge_counters": dict(self.provider.bridge.counters),
                "reader_counters": {r.event: dict(r.counters) for r in self.readers},
            }
            (evidence / "PHASE6_UE_EVIDENCE.json").write_text(
                json.dumps(summary, sort_keys=True, default=str), encoding="utf-8")
            report = super().close()
            report["run4_phase6"] = {"coverage": summary["coverage"],
                                     "faulted": summary["faulted"],
                                     "counters": summary["counters"]}
            return report

    return Run4LiveRuntimeV2


def build_run4_collector_class(base_collector: type) -> type:  # pragma: no cover - live
    """Collector that records the RGB receipt instant and the P40 radar window."""

    class Run4PassiveSplitCollectorV2(base_collector):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            self._run4_lock = threading.Lock()
            self._run4_rgb_raw: "collections.OrderedDict[int, int]" = collections.OrderedDict()
            self._run4_window: "collections.OrderedDict[float, Any]" = collections.OrderedDict()
            super().__init__(**kwargs)
            original = self.aggregator.window_detections

            def recorded(*args: Any, **keywords: Any):
                detections, meta = original(*args, **keywords)
                with self._run4_lock:
                    self._run4_window[float(keywords["reference_timestamp_s"])] = meta
                    while len(self._run4_window) > 64:
                        self._run4_window.popitem(last=False)
                return detections, meta

            self.aggregator.window_detections = recorded
            self.live.scene_hooks = self._run4_scene_inputs

        def _on_rgb(self, image: Any) -> None:
            with self._run4_lock:
                self._run4_rgb_raw[int(image.frame)] = T.raw_now_ns()
                while len(self._run4_rgb_raw) > 64:
                    self._run4_rgb_raw.popitem(last=False)
            super()._on_rgb(image)

        def _run4_scene_inputs(self, frame_id: int, carla_timestamp: float):
            with self._run4_lock:
                meta = self._run4_window.pop(float(carla_timestamp), None)
                rgb_raw = self._run4_rgb_raw.pop(int(frame_id), None)
            _require(meta is not None and rgb_raw is not None,
                     f"sensor inputs missing for frame {frame_id}")
            return meta, rgb_raw

        def on_world_tick(self, frame_id: int, route_tick: int) -> None:
            fault = getattr(self.live, "infrastructure_fault", None)
            if fault:
                self.failures.append(fault)
                raise RuntimeError(f"RUN4_INFRASTRUCTURE_FAULT: {fault}")
            super().on_world_tick(frame_id, route_tick)

    return Run4PassiveSplitCollectorV2
