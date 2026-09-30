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
    "PlannedRun4FrameV2",
    "SensorFirstPlannerV2",
    "CycleBudgetV2",
    "boundary_violations",
    "radar_window_sha256",
    "registered_terminal_from_message",
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
        self._records: dict[int, dict[str, Any]] = {}
        self.stage_clock: Callable[[], int] = T.raw_now_ns

    def process(self, *, frame_id: int, capture_wall_ns: int,
                ego_pose: Sequence[float], scene: SceneDescriptorsV2,
                input_7ch: Callable[[], Any]) -> PreparedRun4FrameV2:
        """Legacy single-step path: exactly plan followed by materialize."""
        planned = self.plan(frame_id=frame_id, capture_wall_ns=capture_wall_ns, scene=scene)
        return self.materialize(planned, frame_id=frame_id, capture_wall_ns=capture_wall_ns,
                                ego_pose=ego_pose, input_7ch=input_7ch)

    def plan(self, *, frame_id: int, capture_wall_ns: int, scene: SceneDescriptorsV2,
             carla_timestamp: Optional[float] = None,
             radar_window_sha256: Optional[str] = None,
             stages: Optional[Mapping[str, Any]] = None) -> "PlannedRun4FrameV2":
        """Guard, state, actor (at most once), hold/fallback: the immutable plan."""
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
        if stages is not None:
            record["stages"] = dict(stages)
        if carla_timestamp is not None:
            record["carla_timestamp"] = float(carla_timestamp)
            record["radar_window_sha256"] = radar_window_sha256
        self.decisions.append(record)
        planned = PlannedRun4FrameV2.bind(
            plan, carla_timestamp=carla_timestamp, radar_window_sha256=radar_window_sha256)
        self._records[planned.tensor_seq] = record
        return planned

    def materialize(self, planned: "PlannedRun4FrameV2", *, frame_id: int,
                    capture_wall_ns: int, ego_pose: Sequence[float],
                    input_7ch: Callable[[], Any], carla_timestamp: Optional[float] = None,
                    radar_window_sha256: Optional[str] = None) -> PreparedRun4FrameV2:
        """Front/codec/SFD4 for exactly the planned frame; fails closed on drift."""
        plan = planned.plan
        stages = self._records.get(planned.tensor_seq, {}).setdefault("stages", {})
        # From here the tensor is assigned: any failure is an infrastructure fault.
        try:
            planned.verify(frame_id=frame_id, capture_wall_ns=capture_wall_ns,
                           carla_timestamp=carla_timestamp,
                           radar_window_sha256=radar_window_sha256)
            context = build_frame_context_v1(
                stream_id=self.stream_id, frame_id=int(frame_id),
                sequence_id=int(plan.tensor_seq), capture_timestamp_ns=int(capture_wall_ns),
                ego_world_x=ego_pose[0], ego_world_y=ego_pose[1], ego_world_z=ego_pose[2],
                ego_world_pitch=ego_pose[3], ego_world_yaw=ego_pose[4],
                ego_world_roll=ego_pose[5])
            stages["input_7ch_start_raw_ns"] = self.stage_clock()
            tensor = input_7ch()
            stages["front_start_raw_ns"] = self.stage_clock()
            prepared = self._ue.prepare(plan.profile, tensor, plan.frame_identity)
            stages["front_end_raw_ns"] = self.stage_clock()
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

    def stages_for(self, tensor_seq: int) -> dict[str, Any]:
        return self._records.get(int(tensor_seq), {}).setdefault("stages", {})


def radar_window_sha256(window_meta: Mapping[str, Any], reference_timestamp_s: float) -> str:
    """Identity of one complete radar window (the exact window P40 uses)."""
    document = {"reference_timestamp_s": float(reference_timestamp_s),
                "sweep_indices": [int(v) for v in window_meta["sweep_indices"]],
                "callbacks": int(window_meta["callbacks"]),
                "returns": int(window_meta["returns"]),
                "window_span_s": float(window_meta["window_span_s"])}
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class PlannedRun4FrameV2:
    """Immutable action plan made before expensive tensor preparation.

    Binds frame, capture and CARLA timestamps, radar-window identity, tensor
    sequence, session/decision/ticket identity, mode, q_e4 and execution bundle;
    :meth:`verify` refuses any drift at materialization.
    """

    plan: E.FramePlanV2
    frame_id: int
    capture_wall_ns: int
    carla_timestamp: Optional[float]
    radar_window_sha256: Optional[str]
    tensor_seq: int
    session_uuid: str
    decision_seq: int
    ticket_seq: int
    reward_requested: bool
    mode_id: int
    q_e4: int
    execution_bundle_sha256: str
    anchor_action_id: Optional[int]
    digest: str

    @staticmethod
    def _binding(fields: Mapping[str, Any]) -> str:
        return hashlib.sha256(json.dumps(dict(fields), sort_keys=True,
                                         default=str).encode()).hexdigest()

    @classmethod
    def bind(cls, plan: E.FramePlanV2, *, carla_timestamp: Optional[float],
             radar_window_sha256: Optional[str]) -> "PlannedRun4FrameV2":
        ident = plan.frame_identity
        fields = {
            "frame_id": int(plan.frame_id), "capture_wall_ns": int(plan.capture_wall_ns),
            "carla_timestamp": None if carla_timestamp is None else float(carla_timestamp),
            "radar_window_sha256": radar_window_sha256, "tensor_seq": int(plan.tensor_seq),
            "session_uuid": ident.session_uuid, "decision_seq": int(ident.decision_seq),
            "ticket_seq": int(ident.ticket_seq),
            "reward_requested": bool(ident.reward_requested),
            "mode_id": int(plan.profile.mode_id), "q_e4": int(plan.profile.q_e4),
            "execution_bundle_sha256": plan.profile.execution_bundle_sha256,
            "anchor_action_id": plan.profile.action_id}
        return cls(plan=plan, digest=cls._binding(fields), **fields)

    def binding_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "frame_id", "capture_wall_ns", "carla_timestamp", "radar_window_sha256",
            "tensor_seq", "session_uuid", "decision_seq", "ticket_seq", "reward_requested",
            "mode_id", "q_e4", "execution_bundle_sha256", "anchor_action_id")}

    def verify(self, *, frame_id: int, capture_wall_ns: int,
               carla_timestamp: Optional[float], radar_window_sha256: Optional[str]) -> None:
        _require(self._binding(self.binding_fields()) == self.digest,
                 "planned frame binding digest drift")
        plan, ident = self.plan, self.plan.frame_identity
        _require((plan.frame_id, plan.tensor_seq, plan.capture_wall_ns,
                  plan.profile.mode_id, plan.profile.q_e4,
                  plan.profile.execution_bundle_sha256, ident.session_uuid,
                  ident.decision_seq, ident.ticket_seq)
                 == (self.frame_id, self.tensor_seq, self.capture_wall_ns, self.mode_id,
                     self.q_e4, self.execution_bundle_sha256, self.session_uuid,
                     self.decision_seq, self.ticket_seq),
                 "planned frame disagrees with its engine plan")
        _require(int(frame_id) == self.frame_id, "materialized frame ID differs from plan")
        _require(int(capture_wall_ns) == self.capture_wall_ns,
                 "materialized capture timestamp differs from plan")
        if self.carla_timestamp is not None or carla_timestamp is not None:
            _require(carla_timestamp is not None and self.carla_timestamp is not None
                     and float(carla_timestamp) == self.carla_timestamp,
                     "materialized CARLA timestamp differs from plan")
        if self.radar_window_sha256 is not None or radar_window_sha256 is not None:
            _require(radar_window_sha256 == self.radar_window_sha256,
                     "materialized radar window differs from plan")


class CycleBudgetV2:
    """Addendum 6: stop only at a completed k_min decision cycle.

    A new opportunity (decision or fallback) is admitted only if a complete
    k_min group still fits in the frame budget and the optional decision cap is
    not yet reached; a hold for an already-open decision is never refused.
    """

    def __init__(self, *, frame_budget: Optional[int] = None,
                 decision_cap: Optional[int] = None) -> None:
        self.frame_budget = None if frame_budget is None else int(frame_budget)
        self.decision_cap = None if decision_cap is None else int(decision_cap)

    def allow_new_opportunity(self, *, sent: int, policy_decisions: int) -> bool:
        from . import reward_hold_controller_v2 as R

        if self.frame_budget is not None and self.frame_budget - int(sent) < R.K_MIN:
            return False
        if self.decision_cap is not None and int(policy_decisions) >= self.decision_cap:
            return False
        return True


class SensorFirstPlannerV2:
    """Addendum 6: the RUN4_CONTRACT latency boundary.

    Sensor preparation (RGB conversion, radar rasterization, camera SI, radar
    P40) completes and the causal state is committed before action-open; the
    actor decision, 7-channel construction, front, compression and send run
    inside the unchanged 170-ms action-open clock, which is never re-stamped.
    The immutable plan binds the exact rasterized radar window.
    """

    def __init__(self, *, pipeline: Run4FramePipelineV2, lock: Any,
                 scene_fn: Callable[..., SceneDescriptorsV2] = compute_scene_descriptors,
                 clock: Callable[[], int] = T.raw_now_ns) -> None:
        self._pipeline = pipeline
        self._lock = lock
        self._scene_fn = scene_fn
        self._clock = clock

    def plan_and_materialize(self, *, frame_id: int, capture_wall_ns: int,
                             carla_timestamp: float, frame_bgr: Any, window_meta: Any,
                             rgb_raw_ns: int, preparation: Mapping[str, Any],
                             ego_pose: Sequence[float], input_7ch: Callable[[], Any],
                             on_planned: Optional[Callable[["PlannedRun4FrameV2"], None]] = None
                             ) -> PreparedRun4FrameV2:
        _require(int(window_meta["callbacks"]) == 4,
                 "refusing to plan from an incomplete radar window")
        window = radar_window_sha256(window_meta, carla_timestamp)
        _require(preparation.get("radar_window_sha256") == window,
                 "rasterized radar window differs from the P40 window")
        _require(preparation.get("radar_tensor_end_raw_ns") is not None,
                 "radar rasterization did not complete before planning")
        stages: dict[str, Any] = {
            "rgb_receipt_raw_ns": int(rgb_raw_ns),
            "radar_window_ready_raw_ns": preparation.get("radar_window_ready_raw_ns"),
            "radar_tensor_start_raw_ns": preparation.get("radar_tensor_start_raw_ns"),
            "radar_tensor_end_raw_ns": preparation.get("radar_tensor_end_raw_ns"),
            "sensor_prepared_raw_ns": preparation.get("sensor_prepared_raw_ns"),
            "si_p40_start_raw_ns": self._clock()}
        scene = self._scene_fn(frame_bgr, window_meta, source_raw_ns=int(rgb_raw_ns))
        stages["si_p40_end_raw_ns"] = self._clock()
        with self._lock:
            planned = self._pipeline.plan(
                frame_id=frame_id, capture_wall_ns=capture_wall_ns, scene=scene,
                carla_timestamp=carla_timestamp, radar_window_sha256=window, stages=stages)
            if on_planned is not None:           # addendum 9: reward-GT early start
                on_planned(planned)
            return self._pipeline.materialize(
                planned, frame_id=frame_id, capture_wall_ns=capture_wall_ns,
                ego_pose=ego_pose, input_7ch=input_7ch, carla_timestamp=carla_timestamp,
                radar_window_sha256=window)


BOUNDARY_ORDER = ("radar_tensor_end_raw_ns", "sensor_prepared_raw_ns", "si_p40_start_raw_ns",
                  "si_p40_end_raw_ns", "state_commit", "action_open", "input_7ch_start_raw_ns",
                  "front_start_raw_ns", "front_end_raw_ns", "first_packet_send_raw_ns")


def boundary_violations(record: Mapping[str, Any]) -> list[str]:
    """Out-of-order stage pairs for one frame record (empty = boundary holds).

    Holds have no state commit/action-open of their own; their ordering is
    checked on the remaining stages.
    """
    stages = dict(record.get("stages") or {})
    for name in ("state_commit", "action_open"):
        if isinstance(record.get(name), Mapping):
            stages[name] = int(record[name]["ns"])
    present = [(name, stages[name]) for name in BOUNDARY_ORDER
               if stages.get(name) is not None]
    return [f"{a}>{b}" for (a, x), (b, y) in zip(present, present[1:]) if x > y]


def registered_terminal_from_message(message: Mapping[str, Any]
                                     ) -> Optional["R.RegisteredTerminalV2"]:
    """RUN4 edge terminal -> exact registered terminal, only for reward frames.

    Hold/fallback terminals (``reward_requested`` false) and non-service
    outcomes stay ledger-only (``None``).
    """
    from . import reward_hold_controller_v2 as R

    if str(message.get("schema")) != MP.RUN4_TERMINAL_SCHEMA:
        return None
    identity = dict(message.get("run4_identity") or {})
    if identity.get("reward_requested") is not True:
        return None
    if str(message.get("outcome")) not in R.SERVICE_FAILURE_EDGE_OUTCOMES:
        return None
    return R.RegisteredTerminalV2(
        session_uuid=str(identity["session_uuid"]),
        controller_lineage_sha256=str(identity["controller_lineage_sha256"]),
        decision_seq=int(identity["decision_seq"]), ticket_seq=int(identity["ticket_seq"]),
        frame_id=int(identity["frame_id"]), tensor_seq=int(identity["tensor_seq"]),
        capture_timestamp_ns=int(message["capture_timestamp_ns"]),
        mode_id=int(identity["mode_id"]), q_e4=int(identity["q_e4"]),
        execution_bundle_sha256=str(identity["execution_bundle_sha256"]),
        anchor_action_id=identity["anchor_action_id"], reward_requested=True,
        outcome=str(message["outcome"]), agent_credit=str(message.get("agent_credit")),
        stage=str(message.get("stage") or ""))


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
            # Addendum 9: called at decision open with (frame, carla ts, reward?).
            self.reward_planned_hook: Optional[Callable[[int, float, bool], None]] = None
            from . import phase6_object_gt_v2 as OG

            # Addendum 10: reward GT may be enqueued only after this mark exists.
            self.last_datagram = OG.LastDatagramMarksV2()
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
            self._terminal_rows: list[dict[str, Any]] = []
            # Addendum 6: sensor-first planning and cycle-aware stopping.
            self.sensor_first = SensorFirstPlannerV2(pipeline=self.pipeline,
                                                     lock=self._engine_lock)
            self.cycle_budget = CycleBudgetV2()      # configured by the child entry
            self.sensor_stages: "collections.OrderedDict[float, dict]" = (
                collections.OrderedDict())
            self.cycle_boundary_reached = False
            # Addendum 7: warm every registered UE path before any frame exists.
            from . import phase6_prewarm_v2 as PW

            self.prewarm_report = PW.warm_ue(
                continuous, self.contract,
                prepare_input=lambda frame, radar: base._prepare_live_input(
                    frame, radar, self.device))
            _require(self.prewarm_report["completed"]
                     and self.prewarm_report["modes_warmed"] == list(range(12)),
                     "UE pre-warm incomplete; no scientific frame may be admitted")
            PW.write_report_create_only(bindings.evidence_dir / "prewarm_ue.json",
                                        self.prewarm_report)
            if not hasattr(self, "gt_log"):
                from . import phase6_object_gt_v2 as OG
                self.gt_log = OG.GtTicketLogV3()
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
            started = time.perf_counter_ns()
            try:
                base.check_deadline(base.UE_STAGE_AFTER_PREPARATION,
                                    capture_timestamp_ns, self.deadline_s)
            except base.DeadlineExpired as expired:   # before any assignment
                return self._record_stale(expired, frame_id=frame_id,
                                          capture_id=capture_id, stream_id=stream_id,
                                          profile=self.profile)
            # RGB conversion and radar rasterization are complete here (the
            # adapter performed both before calling submit).
            preparation = dict(self.sensor_stages.pop(float(carla_timestamp), {}))
            preparation["sensor_prepared_raw_ns"] = T.raw_now_ns()
            window_meta, rgb_raw_ns = self.scene_hooks(int(frame_id), float(carla_timestamp))
            with self._engine_lock:
                self.engine.new_opportunity_allowed = self.cycle_budget.allow_new_opportunity(
                    sent=self.sent, policy_decisions=self.engine.counters.policy_decisions)
            hook = self.reward_planned_hook
            on_planned = (None if hook is None else
                          lambda planned: hook(int(frame_id), float(carla_timestamp),
                                               bool(planned.reward_requested)))
            try:
                prepared = self.sensor_first.plan_and_materialize(
                    frame_id=int(frame_id), capture_wall_ns=int(capture_timestamp_ns),
                    carla_timestamp=float(carla_timestamp), frame_bgr=frame_bgr,
                    window_meta=window_meta, rgb_raw_ns=int(rgb_raw_ns),
                    preparation=preparation, ego_pose=ego_pose,
                    input_7ch=lambda: base._prepare_live_input(frame_bgr, radar_tensor,
                                                               self.device),
                    on_planned=on_planned)
            except E.DecisionCapacityExhausted:
                # Nothing was assigned: stop at this closed decision cycle.
                self.cycle_boundary_reached = True
                return {"sent": False, "prepare_status": "DECISION_CYCLE_BOUNDARY",
                        "stale_stage": "", "stale_age_ms": ""}
            stages = self.pipeline.stages_for(prepared.plan.tensor_seq)
            try:
                chunks = self._chunk_payload(prepared.wire, message_id=int(frame_id),
                                             chunk_bytes=self.chunk_bytes)
                self.ledger.stage(capture_id, identity=prepared.identity,
                                  anchor_profile_id=prepared.anchor_profile_id)
                stages["ledger_staged_raw_ns"] = T.raw_now_ns()
                self._gt_identity[int(frame_id)] = dict(prepared.gt_identity)
                self._run4_identity[int(frame_id)] = dict(prepared.identity)
                if on_commit is not None:
                    on_commit()
                stages["commit_registered_raw_ns"] = T.raw_now_ns()
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
                stages["first_packet_send_raw_ns"] = T.raw_now_ns()
                for chunk in chunks:
                    self.sender.sendto(chunk, self.remote)
                stages["last_packet_send_raw_ns"] = T.raw_now_ns()
                # Addendum 10: record-only mark; the pinned worker enqueues the
                # object-GT ticket after submit returns, never from here.
                self.last_datagram.mark(int(frame_id), stages["last_packet_send_raw_ns"])
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
                    terminal = registered_terminal_from_message(message)
                    if terminal is not None:      # addendum 5: exact service terminal
                        with self._engine_lock:
                            klass = self.engine.on_registered_terminal(
                                terminal, receipt_raw_ns=receipt_raw)
                        self._terminal_rows.append({
                            "frame_id": terminal.frame_id, "class": klass.value,
                            "outcome": terminal.outcome,
                            "agent_credit": terminal.agent_credit,
                            "stage": terminal.stage, "request_key": list(terminal.request_key),
                            "receipt": W.raw(receipt_raw).to_dict()})
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
            if hasattr(self, "engine"):
                # Addendum 6: drain the active ticket (feedback, terminal or the
                # registered timeout) before accounting; nothing new is admitted.
                deadline = time.monotonic() + R.TIMEOUT_RESOLUTION_ELAPSED_NS / 1e9 + 0.3
                while time.monotonic() < deadline:
                    with self._engine_lock:
                        current = self.engine.controller.current
                        self.engine.controller.poll(T.raw_now_ns())
                        if current is None or current.resolution is not None:
                            break
                    time.sleep(0.01)
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
                "terminal_rows": list(self._terminal_rows),
                "cycle_budget": {"frame_budget": self.cycle_budget.frame_budget,
                                 "decision_cap": self.cycle_budget.decision_cap},
                "cycle_boundary_reached": bool(self.cycle_boundary_reached),
                "prewarm_ue_completed": bool(self.prewarm_report.get("completed")),
                "gt_refresh_startup": getattr(self, "gt_refresh_startup", None),
                "gt_objects": self.gt_log.snapshot(),
                **self._run4_object_gt_evidence(),
                "gt_missing_high_outputs": self.gt_log.missing_high_outputs(),
                "gt_queue": ({"counters": dict(self.gt_queue.counters),
                              "depth_at_close": self.gt_queue.depth(),
                              "unfinished_tasks": self.gt_queue.unfinished_tasks}
                             if getattr(self, "gt_queue", None) is not None else None),
                "unresolved_tickets_at_close": sum(
                    1 for c in self.engine.controllers
                    if c.current is not None and c.current.resolution is None),
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

        def _run4_object_gt_evidence(self) -> dict[str, Any]:
            """Addenda 9/10: reward gate, last-datagram marks, builder/front overlap."""
            from . import phase6_object_gt_v2 as OG

            gate = getattr(self, "gt_gate", None)
            profiles = {int(row["frame_id"]): row["object_builder"]
                        for row in self.gt_log.snapshot()["tickets"]
                        if row.get("object_builder")}
            return {"gt_reward_gate": None if gate is None else {
                        "events": list(gate.events), "pending_at_close": gate.pending()},
                    "gt_last_datagram": self.last_datagram.snapshot(),
                    "gt_builder_overlap": OG.overlap_report(profiles, self.pipeline.decisions)}

    return Run4LiveRuntimeV2


class _PreparationTimer:
    """Instance-local proxy: ``build_radar_sample`` gains a timing hook and,
    from addendum 9, ``build_object_rows`` is routed to the collector."""

    def __init__(self, parked: Any, hook: Callable[..., Any],
                 object_hook: Optional[Callable[..., Any]] = None) -> None:
        self._parked = parked
        self._hook = hook
        self._object_hook = object_hook

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parked, name)

    def build_radar_sample(self, **kwargs: Any) -> Any:
        return self._hook(self._parked.build_radar_sample, **kwargs)

    def build_object_rows(self, **kwargs: Any) -> Any:
        if self._object_hook is None:
            return self._parked.build_object_rows(**kwargs)
        return self._object_hook(self._parked.build_object_rows, **kwargs)


def build_run4_collector_class(base_collector: type) -> type:  # pragma: no cover - live
    """Collector that records the RGB receipt instant and the P40 radar window."""

    class Run4PassiveSplitCollectorV2(base_collector):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            self._run4_lock = threading.Lock()
            self._run4_rgb_raw: "collections.OrderedDict[int, int]" = collections.OrderedDict()
            self._run4_window: "collections.OrderedDict[float, Any]" = collections.OrderedDict()
            self._run4_window_ready: "collections.OrderedDict[float, int]" = (
                collections.OrderedDict())
            super().__init__(**kwargs)
            original = self.aggregator.window_detections

            def recorded(*args: Any, **keywords: Any):
                detections, meta = original(*args, **keywords)
                with self._run4_lock:
                    self._run4_window[float(keywords["reference_timestamp_s"])] = meta
                    self._run4_window_ready[float(keywords["reference_timestamp_s"])] = (
                        T.raw_now_ns())
                    while len(self._run4_window) > 64:
                        self._run4_window.popitem(last=False)
                    while len(self._run4_window_ready) > 64:
                        self._run4_window_ready.popitem(last=False)
                return detections, meta

            self.aggregator.window_detections = recorded
            self.live.scene_hooks = self._run4_scene_inputs
            self.parked = _PreparationTimer(self.parked, self._run4_timed_build,
                                            self._run4_object_rows)
            self._run4_install_gt_priority()

        def _run4_install_gt_priority(self) -> None:
            """Addendum 7: reward-priority object GT and a timed pre-route refresh."""
            from . import phase6_gt_priority_v2 as GP

            _require(self.scene_source is not None, "object GT requires a scene source")
            log = getattr(self.live, "gt_log", None) or GP.GtTicketLogV2()
            self.live.gt_log = log
            # completes before any frame; a failure raises and admits nothing
            self.live.gt_refresh_startup = GP.timed_startup_refresh(self.scene_source)
            GP.install_timed_refresh(self.scene_source, log)
            # Addendum 9: the reward gate opens at decision open and closes when
            # the HIGH ticket is written (or the frame is never sent); LOW
            # never starts, and yields, while it is open. Addendum 10: no
            # reward-GT work starts at decision open (the gate only).
            from . import phase6_object_gt_v2 as OG

            self._run4_gate = OG.RewardPendingGateV2()
            self.live.gt_gate = self._run4_gate
            self.live.reward_planned_hook = self._run4_on_planned
            queue = OG.DeferringRewardPriorityGtQueueV2(
                classify=self._run4_gt_class, maxsize_high=64, maxsize_low=64,
                on_enqueue=log.enqueued, on_dequeue=log.dequeued,
                low_blocked=self._run4_gate.blocked, on_skip=self._run4_low_skipped)
            self.evaluation_queue = queue            # worker reads it per iteration
            self.live.gt_queue = queue

        def _run4_low_skipped(self, frame_id: int, reason: str) -> None:
            from . import phase6_object_gt_v2 as OG

            self.live.gt_log.low_skipped(frame_id, reason, time.time_ns())
            with self.gt_lock:
                self.evaluation_errors[int(frame_id)] = f"{OG.LOW_SKIPPED_STATUS}:{reason}"
            with self._quality_scene_lock:           # never consumed; release it
                self._quality_scenes.pop(int(frame_id), None)

        def _run4_on_planned(self, frame_id: int, carla_timestamp: float,
                             reward_requested: bool) -> None:
            """Decision open: raise the reward gate only (LOW deferral)."""
            del carla_timestamp
            if reward_requested:
                self._run4_gate.open(frame_id)

        @staticmethod
        def _run4_support_kwargs() -> dict[str, Any]:
            # the literal radar-support arguments of the pinned _ground_truth
            return {"radar_support_margin_m": 1.0, "radar_person_support_mode": "radius",
                    "radar_person_support_radius_m": 1.5,
                    "radar_person_support_z_down_m": 0.5,
                    "radar_person_support_z_up_m": 2.0}

        def _run4_object_rows(self, real_build: Callable[..., Any], **kwargs: Any):
            """Addendum 9: instrumented, range-limited build for exact-scene tickets."""
            from . import phase6_gt_priority_v2 as GP
            from . import phase6_object_gt_v2 as OG

            if kwargs.get("world") is self.world:    # live-world diagnostic: unchanged
                return real_build(**kwargs)
            frame_id = int(kwargs["sample_base"]["frame_id"])
            identity = self.live._run4_identity.get(frame_id) or {}
            high = identity.get("reward_requested") is True
            klass = GP.HIGH if high else GP.LOW
            profile = OG.ObjectGtProfileV2(frame_id=frame_id, queue_class=klass)
            profile.begin()
            try:
                rows = OG.build_object_rows_v2(
                    self.parked._parked, **kwargs,
                    eligibility_distance_m=float(self.max_gt_distance_m),
                    profile=profile,
                    preempt=None if high else self._run4_gate.blocked)
            except OG.LowObjectGtPreempted:
                self.live.gt_log.object_profile(frame_id, profile.end(outcome="LOW_PREEMPTED"))
                raise
            self.live.gt_log.object_profile(frame_id, profile.end())
            return rows

        def _run4_gt_class(self, item: Mapping[str, Any]) -> str:
            from . import phase6_gt_priority_v2 as GP

            frame_id = int(item["frame_id"])
            identity = self.live._run4_identity.get(frame_id)
            if identity is None or frame_id not in self.live._gt_identity:
                raise GP.GtQueueError(f"object-GT ticket for foreign frame {frame_id}")
            if identity.get("reward_requested") is not True:
                return GP.LOW
            # Addendum 10: reward GT is admitted only after the exact
            # last-datagram-sent mark of the same frame exists.
            sent_raw = self.live.last_datagram.get(frame_id)
            if sent_raw is None:
                raise GP.GtQueueError(
                    f"reward object-GT for frame {frame_id} before last-datagram-sent")
            self.live.gt_log.note(frame_id, last_datagram_raw_ns=sent_raw,
                                  gt_enqueue_raw_ns=T.raw_now_ns())
            return GP.HIGH

        def _ground_truth(self, **kwargs: Any):
            exact = kwargs.get("world") is not None and hasattr(self.live, "gt_log")
            frame_id = int(kwargs["frame_id"])
            if exact:
                self.live.gt_log.rows_started(frame_id, time.time_ns())
            try:
                rows = super()._ground_truth(**kwargs)
            except Exception as exc:
                if exact:
                    self.live.gt_log.completed(frame_id, time.time_ns(),
                                               error=f"{type(exc).__name__}: {exc}"[:200])
                raise
            finally:
                if exact:                            # addendum 9: HIGH written or failed
                    self._run4_gate.close(frame_id, "OBJECT_GT_DONE")
            if exact:
                self.live.gt_log.completed(frame_id, time.time_ns())
            return rows

        def _run4_timed_build(self, real_build: Callable[..., Any], **kwargs: Any):
            """Time the radar rasterization and bind it to its exact window."""
            timestamp = float(kwargs["frame_time_s"])
            with self._run4_lock:
                meta = self._run4_window.get(timestamp)
                ready_raw = self._run4_window_ready.get(timestamp)
            started = T.raw_now_ns()
            result = real_build(**kwargs)
            ended = T.raw_now_ns()
            stages = self.live.sensor_stages
            stages[timestamp] = {
                "radar_window_ready_raw_ns": ready_raw,
                "radar_tensor_start_raw_ns": started, "radar_tensor_end_raw_ns": ended,
                "radar_window_sha256": (None if meta is None
                                        else radar_window_sha256(meta, timestamp))}
            while len(stages) > 64:
                stages.popitem(last=False)
            return result

        def _process_token(self, token: Mapping[str, Any]) -> None:
            try:
                super()._process_token(token)
            finally:
                # addendum 9: a planned reward frame that was never sent has no
                # GT ticket; release the gate so LOW is not blocked forever.
                frame_id = int(token["frame_id"])
                if frame_id not in self.sent_frames:
                    self._run4_gate.close(frame_id, "NOT_SENT")
            # Addendum 6: a refused opportunity means no complete cycle fits;
            # stop admitting work at this closed decision cycle.
            if getattr(self.live, "cycle_boundary_reached", False) and hasattr(
                    self, "_request_probe_stop"):
                self._request_probe_stop("DECISION_CYCLE_BOUNDARY")

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
