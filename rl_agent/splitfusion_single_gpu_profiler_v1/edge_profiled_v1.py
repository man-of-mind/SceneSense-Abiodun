"""Edge-container entry point: the unchanged Phase-6 edge service, profiled.

Runs ``phase6_edge_runtime_v2.main`` with the same arguments.  Before it runs,
the module attributes that the service resolves at call time are replaced by
pass-through subclasses/wrappers (the source files are not edited):

* ``ChunkReassembler``: first-datagram and reassembly-complete marks;
* ``LatestFramePendingSlot``: queue offer/take marks and the pending depth;
* ``Run4EdgeProcessorV2``: ``edge.process`` (identity bound after verify),
  ``edge.verify``, ``edge.compute`` (CUDA events) and terminal marks;
* ``StageRecorder`` (phase-6 edge and detached runtime): the registered decode,
  dequantize, AE, tail and serialization stages with CUDA events;
* ``publish_cpu`` and the tail snapshot (output construction), the map update
  build and the map publication;
* ``Run4EvaluatorV2``: submission, Q_perc evaluation and feedback emission.

The container shares the host kernel, so ``CLOCK_MONOTONIC_RAW`` here is the
same clock as on the host; the analyzer verifies this from clock probes.
Records go to ``/work/torch_cache/profiler/edge.profile.jsonl`` (the per-cell
edge scratch, created 0777 by the host) and are copied out before teardown.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import recorder_v1 as RV
from . import stages_v1 as SV

DEFAULT_RUN4_CONFIG = "/work/torch_cache/run4_phase6_edge_config.json"


def envelope_identity(envelope: Any) -> dict[str, Any]:
    return {"frame_id": int(envelope.frame_id), "tensor_seq": int(envelope.tensor_seq),
            "session_uuid": str(envelope.session_uuid),
            "decision_seq": int(envelope.decision_seq), "ticket_seq": int(envelope.ticket_seq),
            "reward_requested": bool(envelope.reward_requested),
            "mode_id": int(envelope.mode_id), "q_e4": int(envelope.q_e4)}


def document_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(document.get("run4_identity") or {})
    out = {k: identity.get(k) for k in ("frame_id", "tensor_seq", "session_uuid",
                                        "decision_seq", "ticket_seq", "reward_requested",
                                        "mode_id", "q_e4") if identity.get(k) is not None}
    if "frame_id" not in out and document.get("frame_id") is not None:
        out["frame_id"] = int(document["frame_id"])
    return out


def _name_thread(name: str) -> None:
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode()[:15], 0, 0, 0)   # PR_SET_NAME
    except Exception:  # noqa: BLE001 - naming is cosmetic
        pass


def install_edge_seams(rec: Any) -> dict[str, Any]:
    # The Phase-6 edge module first: its top-level imports run in their original order.
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E

    from phase2_map_sharing.transport import CHUNK_HEADER

    from rl_agent.splitfusion_edge_optimization_v1 import detached_edge_preload_v3 as PL
    from rl_agent.splitfusion_edge_optimization_v1 import detached_runtime as DRT
    from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import (
        DetachedOptimizedTailAdapterV3 as TA,
    )
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
    from rl_agent.splitfusion_live_dispatch_v1 import timing as TM

    from .seams_ue_v1 import bound_span

    rec_fn = RV.active
    notes: dict[str, Any] = {}

    profiled_recorder = SV.make_profiled_stage_recorder(TM.StageRecorder, "edge.codec", rec_fn)
    E.StageRecorder = profiled_recorder
    DRT.StageRecorder = profiled_recorder
    MP.build_run4_map_update = bound_span(rec_fn, "edge.map_update_build",
                                          MP.build_run4_map_update)

    class ProfiledReassemblerV1(E.ChunkReassembler):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rec.register_gauge("edge.reassembly_partial", lambda: len(self.pending))

        def ingest(self, source_endpoint: str, datagram: bytes, *, received_at_s: float):
            first = None
            try:
                message_id, _index, total = CHUNK_HEADER.unpack_from(datagram)
                if (str(source_endpoint), message_id) not in self.pending:
                    first = (message_id, total)
            except Exception:  # noqa: BLE001 - the base raises its own error below
                pass
            if first is not None:
                rec.mark("edge.datagram_first", frame_id=first[0], chunks=first[1])
            complete = super().ingest(source_endpoint, datagram, received_at_s=received_at_s)
            if complete is not None:
                rec.mark("edge.reassembly_complete", **RV.safe_fields(lambda: {
                    "frame_id": int(complete.message_id),
                    "chunks": int(complete.chunk_count),
                    "duplicates": int(complete.duplicate_chunks)}))
            return complete

    E.ChunkReassembler = ProfiledReassemblerV1

    class ProfiledPendingSlotV1(base.LatestFramePendingSlot):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rec.register_gauge("edge.pending_depth", self.depth)

        def offer(self, stream_id: str, item: Any, *, sequence: int):
            admitted, displaced = super().offer(stream_id, item, sequence=sequence)
            rec.mark("edge.queue_offer", **RV.safe_fields(lambda: {
                "frame_id": _frame(item), "admitted": bool(admitted),
                "displaced_frame_id": _frame(displaced)}))
            return admitted, displaced

        def take(self, timeout: float):
            taken = super().take(timeout)
            if taken is not None:
                rec.mark("edge.queue_take", **RV.safe_fields(lambda: {
                    "frame_id": _frame(taken[1])}))
            return taken

    base.LatestFramePendingSlot = ProfiledPendingSlotV1

    class ProfiledProcessorV1(E.Run4EdgeProcessorV2):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            compute = kwargs["compute"]

            def profiled_compute(**inner: Any) -> Any:
                with rec.span("edge.compute", cuda=True):
                    return compute(**inner)

            kwargs["compute"] = profiled_compute
            super().__init__(**kwargs)

        def verify(self, frame: bytes):
            with rec.span("edge.verify"):
                result = super().verify(frame)
                rec.bind(**RV.safe_fields(envelope_identity, result[0]))
            return result

        def process(self, frame: bytes, *, edge_timing: Mapping[str, Any],
                    segmentation: Optional[Mapping[str, Any]] = None):
            try:
                with rec.span("edge.process"):
                    return super().process(frame, edge_timing=edge_timing,
                                           segmentation=segmentation)
            finally:
                rec.unbind()

        def terminal(self, frame: bytes, *, outcome: str, stage: str,
                     superseded_by_frame_id: Optional[int] = None):
            saved = rec.context()
            try:
                document = super().terminal(frame, outcome=outcome, stage=stage,
                                            superseded_by_frame_id=superseded_by_frame_id)
            finally:
                rec.unbind()
                rec.bind(**saved)
            rec.mark("edge.terminal_built", **RV.safe_fields(lambda: {
                **document_identity(document), "outcome": str(outcome),
                "terminal_stage": str(stage),
                "superseded_by_frame_id": superseded_by_frame_id}))
            return document

    E.Run4EdgeProcessorV2 = ProfiledProcessorV1

    class ProfiledPublisherV1(E.Run4MapPublisherV2):  # type: ignore[misc, valid-type]
        def publish(self, document: Mapping[str, Any]):
            with rec.span("edge.map_publish", **RV.safe_fields(document_identity, document)):
                return super().publish(document)

    E.Run4MapPublisherV2 = ProfiledPublisherV1

    class ProfiledEvaluatorV1(E.Run4EvaluatorV2):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            rec.register_gauge("edge.evaluator_queue", self._queue.qsize)
            rec.register_gauge("edge.evaluator_outstanding",
                               lambda: int(self.counters["submitted"])
                               - int(self.counters["emitted"]))

        def submit(self, ticket: Any) -> None:
            rec.mark("edge.evaluator_submit",
                     **RV.safe_fields(lambda: envelope_identity(ticket.envelope)))
            return super().submit(ticket)

        def _evaluate_with(self, ticket: Any, gt: Mapping[str, Any], timing: dict) -> None:
            with rec.span("edge.qperc_evaluate",
                          **RV.safe_fields(lambda: envelope_identity(ticket.envelope))):
                return super()._evaluate_with(ticket, gt, timing)

        def _emit(self, ticket: Any, feedback: Any, reason: Any, **kwargs: Any) -> None:
            with rec.span("edge.feedback_emit", **RV.safe_fields(lambda: {
                    **envelope_identity(ticket.envelope),
                    "reason": getattr(reason, "name", str(reason))})):
                return super()._emit(ticket, feedback, reason, **kwargs)

        def _run(self) -> None:
            _name_thread("run4-evaluator")
            return super()._run()

    E.Run4EvaluatorV2 = ProfiledEvaluatorV1

    original_preload = PL.preload_detached_optimized_edge_v3

    def preload(*args: Any, **kwargs: Any) -> Any:
        edge = original_preload(*args, **kwargs)
        runtime = edge.runtime
        try:
            runtime.publish_cpu = bound_span(rec_fn, "edge.publish_cpu", runtime.publish_cpu,
                                             cuda=True)
        except Exception:  # noqa: BLE001
            notes["publish_cpu_seam"] = "unavailable"
        return edge

    PL.preload_detached_optimized_edge_v3 = preload
    original_snapshot = TA.__dict__["snapshot"]
    snapshot_fn = getattr(original_snapshot, "__func__", original_snapshot)
    TA.snapshot = staticmethod(bound_span(rec_fn, "edge.output_snapshot", snapshot_fn,
                                          cuda=True))

    original_ready = PW.publish_ready_after_warmup

    def publish_ready_after_warmup(*args: Any, **kwargs: Any) -> Any:
        rec.set_phase("prewarm")
        result = original_ready(*args, **kwargs)
        rec.set_phase("live")
        return result

    PW.publish_ready_after_warmup = publish_ready_after_warmup
    rec.note("seams_installed", **notes)
    return notes


def _frame(item: Any) -> Optional[int]:
    if isinstance(item, Mapping) and item.get("frame_id") is not None:
        return int(item["frame_id"])
    return None


def profile_path(argv: Sequence[str]) -> Path:
    config = DEFAULT_RUN4_CONFIG
    items = list(argv)
    for index, item in enumerate(items):
        if item == "--run4-config" and index + 1 < len(items):
            config = items[index + 1]
        elif item.startswith("--run4-config="):
            config = item.split("=", 1)[1]
    return Path(config).parent / "profiler" / "edge.profile.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E

    args = list(sys.argv[1:] if argv is None else argv)
    rec = RV.ProfileRecorderV1(profile_path(args), role="edge").start()
    RV.activate(rec)
    try:
        install_edge_seams(rec)
        return E.main(args)
    finally:
        rec.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
