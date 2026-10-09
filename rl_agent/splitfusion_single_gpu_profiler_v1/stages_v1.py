"""Registered stage vocabulary: names, required nesting and coverage scope.

``parents`` is the exact set of stages a span may be directly nested in on its
own thread (``None`` = a thread root).  ``scope`` names the population the
analyzer uses as the coverage denominator:

* ``sent``      every transmitted frame;
* ``decision``  every policy decision (actor called);
* ``planned``   every frame that reached planning (decision, hold or fallback);
* ``edge``      every frame the edge reassembled;
* ``processed`` every frame the edge computed (admitted and fresh);
* ``reward``    every reward-requested frame the edge computed;
* ``installed`` every frame the map installed;
* ``tick``      every CARLA world tick of the route;
* ``event``     no fixed denominator (counted only).

``cuda`` marks spans that also carry CUDA-event execution time.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

UE_CODEC = ("total_ue_preparation", "front_backbone", "ranker_selection", "ae_encode",
            "quantize_pack", "zstd_compression")
EDGE_CODEC = ("total_edge_processing", "zstd_decompression", "unpack_dequantize", "ae_decode",
              "frozen_tail", "output_serialization")


def _s(role: str, parents: Optional[tuple], scope: str, cuda: bool = False,
       kind: str = "span") -> dict[str, Any]:
    return {"role": role, "parents": None if parents is None else frozenset(parents),
            "scope": scope, "cuda": cuda, "kind": kind}


STAGES: dict[str, dict[str, Any]] = {
    # --- UE process: CARLA client side -------------------------------------------
    "carla.world_tick": _s("ue", None, "tick"),
    "carla.rgb_receipt": _s("ue", None, "event", kind="mark"),
    # --- UE process: per-frame worker ---------------------------------------------
    "ue.worker_frame": _s("ue", None, "event"),
    "ue.sensor_wait": _s("ue", ("ue.worker_frame",), "event"),
    "ue.radar_window": _s("ue", ("ue.worker_frame",), "event"),
    "ue.radar_rasterize": _s("ue", ("ue.worker_frame",), "sent"),
    "ue.rgb_convert": _s("ue", ("ue.worker_frame",), "sent"),
    "ue.scene_snapshot": _s("ue", ("ue.worker_frame",), "sent"),
    "ue.submit": _s("ue", ("ue.worker_frame",), "sent"),
    "ue.si_p40": _s("ue", ("ue.submit",), "sent"),
    "ue.plan": _s("ue", ("ue.submit",), "planned"),
    "agent.telemetry_snapshot": _s("ue", ("ue.plan",), "event"),
    "agent.state_build": _s("ue", ("ue.plan",), "decision"),
    "agent.actor": _s("ue", ("ue.plan",), "decision"),
    "ue.input_7ch": _s("ue", ("ue.submit",), "sent", cuda=True),
    "ue.codec.total_ue_preparation": _s("ue", ("ue.submit",), "sent", cuda=True),
    "ue.codec.front_backbone": _s("ue", ("ue.codec.total_ue_preparation",), "sent", cuda=True),
    "ue.codec.ranker_selection": _s("ue", ("ue.codec.total_ue_preparation",), "event",
                                    cuda=True),
    "ue.codec.ae_encode": _s("ue", ("ue.codec.total_ue_preparation",), "event", cuda=True),
    "ue.codec.quantize_pack": _s("ue", ("ue.codec.total_ue_preparation",), "sent", cuda=True),
    "ue.codec.zstd_compression": _s("ue", ("ue.codec.total_ue_preparation",), "sent",
                                    cuda=True),
    "ue.sfd4_pack": _s("ue", ("ue.submit",), "sent"),
    "ue.udp_send": _s("ue", ("ue.submit",), "sent"),
    "ue.gt_build": _s("ue", None, "event"),
    "ue.feedback_receipt": _s("ue", None, "event", kind="mark"),
    "ue.terminal_receipt": _s("ue", None, "event", kind="mark"),
    # --- edge process (container) ---------------------------------------------------
    "edge.datagram_first": _s("edge", None, "edge", kind="mark"),
    "edge.reassembly_complete": _s("edge", None, "edge", kind="mark"),
    "edge.queue_offer": _s("edge", None, "edge", kind="mark"),
    "edge.queue_take": _s("edge", None, "event", kind="mark"),
    "edge.process": _s("edge", None, "processed"),
    "edge.verify": _s("edge", (None, "edge.process"), "processed"),
    "edge.compute": _s("edge", ("edge.process",), "processed", cuda=True),
    "edge.codec.total_edge_processing": _s("edge", ("edge.compute",), "processed", cuda=True),
    "edge.codec.zstd_decompression": _s("edge", ("edge.codec.total_edge_processing",),
                                        "processed", cuda=True),
    "edge.codec.unpack_dequantize": _s("edge", ("edge.codec.total_edge_processing",),
                                       "processed", cuda=True),
    "edge.codec.ae_decode": _s("edge", ("edge.codec.total_edge_processing",), "event",
                               cuda=True),
    "edge.codec.frozen_tail": _s("edge", ("edge.codec.total_edge_processing",), "processed",
                                 cuda=True),
    "edge.codec.output_serialization": _s("edge", ("edge.publish_cpu",), "event", cuda=True),
    "edge.publish_cpu": _s("edge", ("edge.compute",), "processed", cuda=True),
    "edge.output_snapshot": _s("edge", ("edge.compute",), "processed", cuda=True),
    "edge.map_update_build": _s("edge", ("edge.process",), "processed"),
    "edge.map_publish": _s("edge", None, "processed"),
    "edge.evaluator_submit": _s("edge", None, "reward", kind="mark"),
    "edge.qperc_evaluate": _s("edge", None, "event"),
    "edge.feedback_emit": _s("edge", (None, "edge.qperc_evaluate"), "reward"),
    "edge.terminal_built": _s("edge", None, "event", kind="mark"),
    # --- map process (host) -----------------------------------------------------------
    "map.ingest_handle": _s("map", None, "event"),
    "map.install": _s("map", ("map.ingest_handle",), "installed"),
    "map.respond": _s("map", ("map.ingest_handle",), "event"),
}


def stage(name: str) -> Mapping[str, Any]:
    return STAGES[name]


def make_profiled_stage_recorder(base: type, prefix: str, recorder_fn: Any) -> type:
    """A ``StageRecorder`` subclass that also emits raw-clock/CUDA spans.

    Behaviour of the base recorder is unchanged: the same names are validated
    (an unregistered name raises before any span opens), the same boundaries
    are recorded on ``time.monotonic_ns``, and the same snapshot is returned.
    A span is emitted only while a frame context is bound on the thread, so
    pre-warm and fixture work is never mistaken for a live frame.
    """

    class ProfiledStageRecorderV1(base):  # type: ignore[misc, valid-type]
        _profiler_prefix = prefix

        def stage(self, name: str):  # noqa: D401 - same contract as the base
            inner = super().stage(name)
            rec = recorder_fn()
            if not rec.enabled or not rec.bound():
                return inner
            return _Both(inner, rec.span(f"{prefix}.{name}", cuda=True))

    ProfiledStageRecorderV1.__qualname__ = f"ProfiledStageRecorderV1[{base.__qualname__}]"
    return ProfiledStageRecorderV1


class _Both:
    """Enter the base boundary first (it validates), then the span; exit reversed."""

    __slots__ = ("_inner", "_span")

    def __init__(self, inner: Any, span: Any) -> None:
        self._inner, self._span = inner, span

    def __enter__(self) -> None:
        self._inner.__enter__()
        self._span.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._span.__exit__(exc_type, exc, tb)
        return bool(self._inner.__exit__(exc_type, exc, tb))
