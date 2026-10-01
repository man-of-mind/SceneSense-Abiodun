"""Profiler seams for the Phase-6 UE child process (record-only).

Installed after the unchanged Phase-6 seams (``phase6_live_child_v2`` plus the
no-build/addendum-6 layer), in the child process only.  Every seam is a
pass-through wrapper: it calls the original with the same arguments, returns
its result unchanged and re-raises its exceptions unchanged.  No pinned file is
edited; the Phase-6 package stays byte-identical to 6199d49.

What is added, on ``CLOCK_MONOTONIC_RAW``:

* CARLA: the ``world.tick()`` call itself (server step + render) and each RGB
  receipt;
* the per-frame worker: sensor wait, radar window, radar rasterization, RGB
  conversion, scene snapshot and ``submit``;
* inside ``submit``: SI/P40 and the datagram burst (re-emitted from the exact
  Phase-6 stamps, never re-measured), planning, telemetry snapshot, state
  build, the actor call (with its decision), seven-channel preparation, the
  registered UE codec stages with CUDA-event execution time, SFD4 packing;
* reward feedback and edge-terminal receipt at their exact receipt stamp;
* queue gauges (preparation slot, GT queue, open ticket, sensor stages).

The edge and map processes are redirected to profiled entry points that wrap
the same unchanged Phase-6 modules (``edge_profiled_v1``, ``map_profiled_v1``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import identity_v1 as IV
from . import recorder_v1 as RV
from . import stages_v1 as SV

EDGE_MODULE = "-m rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1"
MAP_SERVER = "rl_agent/splitfusion_single_gpu_profiler_v1/map_profiled_v1.py"
PROFILER_DIR_ENV = "SCENESENSE_PROFILER_DIR"
EDGE_SCRATCH_LEAF = "profiler"


def planned_identity(planned: Any) -> dict[str, Any]:
    return {"frame_id": int(planned.frame_id), "tensor_seq": int(planned.tensor_seq),
            "session_uuid": str(planned.session_uuid),
            "decision_seq": int(planned.decision_seq), "ticket_seq": int(planned.ticket_seq),
            "reward_requested": bool(planned.reward_requested),
            "frame_kind": str(planned.plan.kind.value), "mode_id": int(planned.mode_id),
            "q_e4": int(planned.q_e4)}


def wire_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The run identity dict carried on the wire/ledgers, in profiler field names."""
    keys = ("frame_id", "tensor_seq", "session_uuid", "decision_seq", "ticket_seq",
            "reward_requested", "mode_id", "q_e4")
    out = {k: identity.get(k) for k in keys if identity.get(k) is not None}
    if identity.get("frame_kind") is not None:
        out["frame_kind"] = identity["frame_kind"]
    return out


def feedback_identity(fb: Any) -> dict[str, Any]:
    return {"frame_id": int(fb.frame_id), "tensor_seq": int(fb.tensor_seq),
            "session_uuid": str(fb.session_uuid), "decision_seq": int(fb.decision_seq),
            "ticket_seq": int(fb.ticket_seq), "reward_requested": bool(fb.reward_requested),
            "mode_id": int(fb.mode_id), "q_e4": int(fb.q_e4)}


def bound_span(rec_fn: Callable[[], Any], stage: str, fn: Callable[..., Any], *,
               cuda: bool = False) -> Callable[..., Any]:
    """Wrap ``fn`` in ``stage`` only while a live frame context is bound."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        rec = rec_fn()
        if not rec.enabled or not rec.bound():
            return fn(*args, **kwargs)
        with rec.span(stage, cuda=cuda):
            return fn(*args, **kwargs)

    wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


class _ActorProxy:
    """Forwards everything; ``act`` gains the ``agent.actor`` span and its decision."""

    def __init__(self, actor: Any, rec: Any) -> None:
        self._profiled_actor = actor
        self._profiled_rec = rec
        self._profiled_vector_noted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._profiled_actor, name)

    def act(self, features: Any) -> Any:
        rec = self._profiled_rec
        if not self._profiled_vector_noted:
            self._profiled_vector_noted = True
            rec.note("policy_state_vector", **IV.decision_vector_identity(features))
        with rec.span("agent.actor") as span:
            decision = self._profiled_actor.act(features)
            span.annotate(decided_mode_id=getattr(decision, "mode_id", None),
                          decided_q_e4=getattr(decision, "q_e4", None),
                          state_sha256=getattr(decision, "state_sha256", None),
                          heads_sha256=getattr(decision, "heads_sha256", None),
                          features=list(features.as_tuple())
                          if hasattr(features, "as_tuple") else None)
        return decision


class _TickTimer:
    """World facade whose ``tick`` is bracketed; every other attribute forwards."""

    def __init__(self, world: Any, rec_fn: Callable[[], Any]) -> None:
        self._profiled_world = world
        self._profiled_rec_fn = rec_fn

    def tick(self, *args: Any, **kwargs: Any) -> Any:
        with self._profiled_rec_fn().span("carla.world_tick") as span:
            frame = self._profiled_world.tick(*args, **kwargs)
            span.annotate(carla_frame=int(frame))
        return frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self._profiled_world, name)


class _MethodSpanProxy:
    """Forwards everything; the named methods are wrapped in spans."""

    def __init__(self, target: Any, spans: Mapping[str, str], rec_fn: Callable[[], Any]) -> None:
        self._profiled_target = target
        self._profiled_spans = dict(spans)
        self._profiled_rec_fn = rec_fn

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._profiled_target, name)
        stage = self._profiled_spans.get(name)
        if stage is None or not callable(value):
            return value
        return bound_span(self._profiled_rec_fn, stage, value)


def emit_submit_derivatives(rec: Any, record: Mapping[str, Any]) -> None:
    """Spans re-emitted from the exact Phase-6 stage stamps of one frame."""
    stages = dict(record.get("stages") or {})
    rec.emit_span("ue.si_p40", stages.get("si_p40_start_raw_ns"),
                  stages.get("si_p40_end_raw_ns"))
    rec.emit_span("ue.udp_send", stages.get("first_packet_send_raw_ns"),
                  stages.get("last_packet_send_raw_ns"))


def instrument_runtime(runtime: Any, rec: Any, *, policy_identity: Mapping[str, Any]) -> None:
    """Instance-level wrappers on one constructed Phase-6 UE runtime."""
    rec.note("policy_runtime_identity", binding=dict(policy_identity),
             runtime=IV.runtime_actor_identity(runtime.engine._actor))
    runtime.engine._actor = _ActorProxy(runtime.engine._actor, rec)

    pipeline = runtime.pipeline
    original_plan = pipeline.plan

    def plan(**kwargs: Any) -> Any:
        with rec.span("ue.plan"):
            planned = original_plan(**kwargs)
            rec.bind(**planned_identity(planned))
        return planned

    pipeline.plan = plan

    original_submit = runtime.submit

    def submit(*args: Any, **kwargs: Any) -> Any:
        frame_id = int(kwargs["frame_id"])
        rec.bind(frame_id=frame_id)
        before = len(pipeline.decisions)
        with rec.span("ue.submit") as span:
            result = original_submit(*args, **kwargs)
            try:
                if len(pipeline.decisions) > before and \
                        int(pipeline.decisions[-1]["frame_id"]) == frame_id:
                    record = pipeline.decisions[-1]
                    emit_submit_derivatives(rec, record)
                    rec.note("frame", frame_id=frame_id, kind=record.get("kind"),
                             stages=dict(record.get("stages") or {}),
                             action_open=record.get("action_open"),
                             state_commit=record.get("state_commit"),
                             transmitted=runtime._run4_identity.get(frame_id),
                             sent=bool(result.get("sent")) if isinstance(result, dict)
                             else None)
                span.annotate(sent=bool(result.get("sent")) if isinstance(result, dict)
                              else None,
                              prepare_status=result.get("prepare_status")
                              if isinstance(result, dict) else None)
            except Exception:  # noqa: BLE001 - evidence only
                rec.note("submit_evidence_error", frame_id=frame_id)
        return result

    runtime.submit = submit

    engine = runtime.engine
    original_feedback = engine.on_feedback
    original_terminal = engine.on_registered_terminal

    def on_feedback(fb: Any, *, receipt_raw_ns: int) -> Any:
        outcome = original_feedback(fb, receipt_raw_ns=receipt_raw_ns)
        rec.mark("ue.feedback_receipt", t_raw_ns=receipt_raw_ns, **feedback_identity(fb),
                 feedback_class=getattr(outcome, "value", str(outcome)),
                 feedback_kind=str(fb.kind))
        return outcome

    def on_registered_terminal(terminal: Any, *, receipt_raw_ns: int) -> Any:
        outcome = original_terminal(terminal, receipt_raw_ns=receipt_raw_ns)
        rec.mark("ue.terminal_receipt", t_raw_ns=receipt_raw_ns,
                 **feedback_identity(terminal),
                 terminal_class=getattr(outcome, "value", str(outcome)),
                 outcome=str(terminal.outcome))
        return outcome

    engine.on_feedback = on_feedback
    engine.on_registered_terminal = on_registered_terminal

    def open_ticket() -> int:
        current = engine.controller.current
        return int(current is not None and current.resolution is None)

    rec.register_gauge("ue.open_ticket", open_ticket)
    rec.register_gauge("ue.sent", lambda: int(runtime.sent))
    rec.register_gauge("ue.sensor_stages", lambda: len(runtime.sensor_stages))
    rec.register_gauge("ue.gt_queue_depth",
                       lambda: runtime.gt_queue.depth()
                       if getattr(runtime, "gt_queue", None) is not None else None)


def build_collector_class(base: type, rec_fn: Callable[[], Any]) -> type:
    class ProfiledCollectorV1(base):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rec = rec_fn()
            original_window = self.aggregator.window_detections
            self.aggregator.window_detections = bound_span(rec_fn, "ue.radar_window",
                                                           original_window)
            self.parked = _MethodSpanProxy(self.parked,
                                           {"build_radar_sample": "ue.radar_rasterize"}, rec_fn)
            if self.scene_source is not None:
                try:
                    self.scene_source.capture = bound_span(
                        rec_fn, "ue.scene_snapshot", self.scene_source.capture)
                except Exception:  # noqa: BLE001 - slotted source: span omitted, counted
                    rec.note("seam_unavailable", seam="scene_source.capture")
            rec.register_gauge("ue.prepared_queue_depth", self.prepared_queue.depth)

        def _process_token(self, token: Mapping[str, Any]) -> None:
            rec = rec_fn()
            rec.bind(frame_id=int(token["frame_id"]), route_tick=token.get("route_tick"))
            try:
                with rec.span("ue.worker_frame"):
                    return super()._process_token(token)
            finally:
                rec.unbind()

        def _records_for(self, frame_id: int, *args: Any, **kwargs: Any) -> Any:
            with rec_fn().span("ue.sensor_wait"):
                return super()._records_for(frame_id, *args, **kwargs)

        def _on_rgb(self, image: Any) -> None:
            rec_fn().mark("carla.rgb_receipt", carla_frame=int(image.frame))
            super()._on_rgb(image)

        def _ground_truth(self, **kwargs: Any) -> Any:
            rec = rec_fn()
            frame_id = int(kwargs["frame_id"])
            identity = (getattr(self.live, "_run4_identity", {}) or {}).get(frame_id) or {}
            rec.bind(frame_id=frame_id, **wire_identity(identity))
            try:
                with rec.span("ue.gt_build"):
                    return super()._ground_truth(**kwargs)
            finally:
                rec.unbind()

    return ProfiledCollectorV1


def ensure_legacy_fusion_namespace(root: Path) -> None:
    """The exact namespace repair ``run_route_b`` performs before importing the
    route module (append the legacy package directory once; idempotent)."""
    import pole_lraspp_multimodal_fusion as fusion_namespace

    legacy_package = Path(root) / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion"
    namespace_paths = {Path(value).resolve() for value in fusion_namespace.__path__}
    if legacy_package.resolve() not in namespace_paths:
        fusion_namespace.__path__.append(str(legacy_package.resolve()))


def install_ue_seams(rec: Any, *, attempt_dir: Path,
                     policy_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Install every UE-process seam.  Call after the Phase-6 seams."""
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned

    ensure_legacy_fusion_namespace(pinned.ROOT)
    from data_collection import run_route_b_perception_collection_v2 as RB
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ue_telemetry_provider_v2 as T
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
    from rl_agent.splitfusion_live_dispatch_v1 import timing as TM

    rec_fn = RV.active
    installed: dict[str, Any] = {}

    # Module-level, record only while a live frame context is bound.
    X.StageRecorder = SV.make_profiled_stage_recorder(TM.StageRecorder, "ue.codec", rec_fn)
    base._prepare_live_input = bound_span(rec_fn, "ue.input_7ch", base._prepare_live_input,
                                          cuda=True)
    W.pack_sfd4 = bound_span(rec_fn, "ue.sfd4_pack", W.pack_sfd4)
    LS.build_live_state = bound_span(rec_fn, "agent.state_build", LS.build_live_state)
    T.open_decision = bound_span(rec_fn, "agent.telemetry_snapshot", T.open_decision)
    pinned.carla_image_to_bgr = bound_span(rec_fn, "ue.rgb_convert", pinned.carla_image_to_bgr)

    # CARLA tick (route thread).
    tick_base = RB.SamplingWorld

    class ProfiledSamplingWorldV1(tick_base):  # type: ignore[misc, valid-type]
        def __init__(self, world: Any, collector: Any, population: Any = None) -> None:
            super().__init__(_TickTimer(world, rec_fn), collector, population)

    RB.SamplingWorld = ProfiledSamplingWorldV1

    # Collector and runtime.
    pinned.PassiveSplitCollector = build_collector_class(pinned.PassiveSplitCollector, rec_fn)
    inner_factory = pinned.LivePilotCellRuntime

    def profiled_factory(**kwargs: Any) -> Any:
        runtime = inner_factory(**kwargs)
        instrument_runtime(runtime, rec, policy_identity=policy_identity)
        rec.set_phase("live")
        return runtime

    pinned.LivePilotCellRuntime = profiled_factory

    # Map and edge processes: profiled entry points over the same modules.
    profiler_dir = Path(attempt_dir) / "profiler"
    os.environ[PROFILER_DIR_ENV] = str(profiler_dir)
    D.DIRECT_EDGE_MODULE = EDGE_MODULE
    D.DIRECT_MAP_SERVER = MAP_SERVER
    original_seed = pinned.seed_cell_edge_state

    def seed(campaign: Mapping[str, Any], edge_scratch: Path) -> None:
        original_seed(campaign, edge_scratch)
        leaf = Path(edge_scratch) / EDGE_SCRATCH_LEAF
        leaf.mkdir(parents=False, exist_ok=False, mode=0o777)
        os.chmod(leaf, 0o777)          # the root container writes here; host can clean up
        installed["edge_scratch"] = str(edge_scratch)

    pinned.seed_cell_edge_state = seed
    inner_stop = pinned.stop_tail

    def stop_tail_preserving_profile() -> bool:
        stopped = inner_stop()
        installed["edge_profile_copied"] = copy_edge_profile(
            installed.get("edge_scratch"), profiler_dir / "edge")
        return stopped

    pinned.stop_tail = stop_tail_preserving_profile
    installed.update({"edge_module": EDGE_MODULE, "map_server": MAP_SERVER,
                      "profiler_dir": str(profiler_dir)})
    rec.note("seams_installed", **{k: v for k, v in installed.items()})
    return installed


def copy_edge_profile(edge_scratch: Optional[str], target: Path) -> dict[str, Any]:
    """Create-only copy of the container's profile files out of the scratch dir."""
    if not edge_scratch:
        return {"copied": [], "reason": "no edge scratch"}
    source = Path(edge_scratch) / EDGE_SCRATCH_LEAF
    if not source.is_dir():
        return {"copied": [], "reason": "no profiler leaf"}
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = target / path.name
        if destination.exists():
            continue                          # create-only: never overwrite
        with path.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        copied.append({"name": path.name, "bytes": destination.stat().st_size})
    return {"copied": copied}
