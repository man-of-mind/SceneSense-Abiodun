"""Run-4B GT-free edge: the unchanged Phase-6 edge service plus two seams.

``phase6_edge_runtime_v2.run_run4_edge_service`` is reused as-is: SFD4
reassembly, identity verification, deadline stages, latest-only admission,
registered edge terminals, tail compute, map publication and prewarm are its
code.  Inside this container process only:

* the edge processor's injected tail ``compute`` emits the operational ACK
  to the UE the moment usable tail output exists (reward-requested decision
  tensors only), before the map update is built or published;
* the live Q_perc evaluator is replaced by a prediction sink: the decision
  frame's records and predicted mask are queued after map publication and
  written create-only by a background thread for post-run evaluation.
  No ground truth is read and no quality is computed on the edge.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    phase6_edge_runtime_v2 as E6,
)

from . import ack_wire_v1 as AW

PREDICTION_LEAF = "run4b_predictions"
PREDICTION_SCHEMA = "scenesense.splitfusion.run4b_single_host.edge_prediction.v1"


class Run4BEdgeError(RuntimeError):
    pass


class _AckLink:
    """The control-socket sender, captured from the evaluator construction."""

    def __init__(self) -> None:
        self.send: Optional[Callable[[bytes], Any]] = None
        self.lock = threading.Lock()
        self.counters: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []

    def bump(self, name: str) -> None:
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + 1


LINK = _AckLink()


def build_ack(*, envelope: Any, context: Any, records: Sequence[Any], labels: Any,
              edge_timing: dict[str, Any], tail_ready_wall_ns: int
              ) -> AW.OperationalAckV1:
    return AW.OperationalAckV1(
        session_uuid=str(envelope.session_uuid),
        controller_lineage_sha256=str(envelope.controller_lineage_sha256),
        decision_seq=int(envelope.decision_seq), ticket_seq=int(envelope.ticket_seq),
        frame_id=int(envelope.frame_id), tensor_seq=int(envelope.tensor_seq),
        capture_timestamp_ns=int(envelope.capture_timestamp_ns),
        mode_id=int(envelope.mode_id), q_e4=int(envelope.q_e4),
        keep_count=int(envelope.keep_count),
        execution_bundle_sha256=str(envelope.execution_bundle_sha256),
        anchor_action_id=envelope.anchor_action_id, reward_requested=True,
        tail_output_sha256=AW.tail_output_sha256(records, labels),
        record_count=len(records),
        reassembly_complete_wall_s=float(edge_timing.get("reassembly_complete_wall_s", 0.0)),
        compute_start_wall_s=float(edge_timing.get("compute_start_wall_s", 0.0)),
        tail_ready_wall_ns=int(tail_ready_wall_ns),
        ack_send_wall_ns=time.time_ns())


class OperationalAckEdgeProcessorV1(E6.Run4EdgeProcessorV2):
    """Phase-6 processor whose tail compute acknowledges usable output first."""

    def __init__(self, *, compute: Callable[..., Any], link: _AckLink = LINK,
                 **kwargs: Any) -> None:
        super().__init__(compute=self._compute_then_ack, **kwargs)
        self._inner_compute = compute
        self._link = link
        self._timing: dict[str, Any] = {}

    def process(self, frame: bytes, *, edge_timing, segmentation=None):
        self._timing = dict(edge_timing)
        return super().process(frame, edge_timing=edge_timing,
                               segmentation=segmentation)

    def _compute_then_ack(self, *, envelope, context, profile, identity):
        records, labels = self._inner_compute(
            envelope=envelope, context=context, profile=profile, identity=identity)
        tail_ready = time.time_ns()
        if envelope.reward_requested:
            send = self._link.send
            if send is None:
                self._link.bump("ack_without_sender")
            else:
                ack = build_ack(envelope=envelope, context=context, records=records,
                                labels=labels, edge_timing=self._timing,
                                tail_ready_wall_ns=tail_ready)
                payload = AW.encode(ack)
                try:
                    send(payload)
                    self._link.bump("acks_sent")
                except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                    self._link.bump("ack_send_failed")
                    with self._link.lock:
                        self._link.records.append({"frame_id": ack.frame_id,
                                                   "send_error": str(exc)})
                else:
                    with self._link.lock:
                        self._link.records.append({
                            "frame_id": ack.frame_id, "tensor_seq": ack.tensor_seq,
                            "decision_seq": ack.decision_seq,
                            "session_uuid": ack.session_uuid,
                            "tail_output_sha256": ack.tail_output_sha256,
                            "record_count": ack.record_count,
                            "reassembly_complete_wall_s": ack.reassembly_complete_wall_s,
                            "compute_start_wall_s": ack.compute_start_wall_s,
                            "tail_ready_wall_ns": ack.tail_ready_wall_ns,
                            "ack_send_wall_ns": ack.ack_send_wall_ns,
                            "ack_sha256": hashlib.sha256(payload).hexdigest()})
        return records, labels


class PredictionSinkV1:
    """Drop-in for ``Run4EvaluatorV2``: retains predictions, computes nothing.

    ``submit`` runs after map publication; the bounded queue is drained by a
    background thread that writes ``<stem>.records.json`` and
    ``<stem>.mask.npy`` create-only under ``<evidence_dir>/run4b_predictions``.
    """

    def __init__(self, *, spec: Any, send: Callable[[bytes], Any],
                 read_ground_truth: Any, match_distance_m: float, gt_timeout_s: float,
                 queue_depth: int, evidence_dir: Path, link: _AckLink = LINK) -> None:
        del spec, read_ground_truth, match_distance_m, gt_timeout_s  # no live quality
        link.send = send
        self._link = link
        self.root = Path(evidence_dir) / PREDICTION_LEAF
        self.root.mkdir(parents=True, exist_ok=False)
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(queue_depth)))
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="run4b-prediction-sink")
        self.written = 0
        self.dropped = 0
        self.failures: list[str] = []

    @property
    def counters(self) -> dict[str, int]:
        counters = dict(self._link.counters)
        counters.update({"predictions_written": self.written,
                         "predictions_dropped_queue_full": self.dropped,
                         "live_quality_evaluations": 0})
        return counters

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._link.lock:
            return list(self._link.records)

    def start(self) -> None:
        self._thread.start()

    def submit(self, ticket: Any) -> None:
        try:
            self._queue.put_nowait(ticket)
        except queue.Full:
            self.dropped += 1

    def _write(self, ticket: Any) -> None:
        import numpy as np

        envelope, context = ticket.envelope, ticket.context
        stem = f"{context.stream_id}_{int(context.frame_id):010d}"
        mask = np.ascontiguousarray(ticket.predicted_mask)
        records = list(ticket.records)
        mask_path = self.root / f"{stem}.mask.npy"
        with mask_path.open("xb") as handle:
            np.save(handle, mask, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        document = {
            "schema": PREDICTION_SCHEMA, "gt_identity": dict(ticket.gt_identity),
            "frame_id": int(envelope.frame_id), "tensor_seq": int(envelope.tensor_seq),
            "decision_seq": int(envelope.decision_seq),
            "session_uuid": str(envelope.session_uuid),
            "mode_id": int(envelope.mode_id), "q_e4": int(envelope.q_e4),
            "tail_output_sha256": AW.tail_output_sha256(records, mask),
            "mask_file": mask_path.name, "mask_dtype": str(mask.dtype),
            "mask_shape": list(mask.shape),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "records": records}
        with (self.root / f"{stem}.records.json").open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, default=float)
            handle.flush()
            os.fsync(handle.fileno())
        self.written += 1

    def _run(self) -> None:
        while True:
            ticket = self._queue.get()
            try:
                if ticket is None:
                    return
                self._write(ticket)
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"{type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30.0)


def install_edge_seams() -> None:
    E6.Run4EdgeProcessorV2 = OperationalAckEdgeProcessorV1
    E6.Run4EvaluatorV2 = PredictionSinkV1


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    install_edge_seams()
    return E6.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
