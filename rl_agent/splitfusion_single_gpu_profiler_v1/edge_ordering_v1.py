"""Edge ordering seam: submit the evaluation ticket before the map update is built.

One ordering change on the edge compute thread, nothing else:

* **serial** (control, today's order): tail compute -> build map update ->
  (return) -> publish map -> ``evaluator.submit``.
* **early** (treatment): tail compute -> **``evaluator.submit``** -> build map
  update -> (return) -> publish map.  The map branch is unchanged in content
  and order; it starts a queue-put later.

Guardrails implemented here:

1. The ticket is created only after the real compute callable has returned,
   i.e. after the tail product was published to CPU and the labels were
   materialized (``.to(cpu).numpy()``), and only after ``verify`` has
   established the identities the compute was called with.
2. Exactly once: the ticket is submitted from inside the compute wrapper for
   reward-requested frames only; the base ``process`` still builds its own
   ticket object afterwards, and that object is removed from the returned
   frame (``evaluation=None``) so the unchanged process loop never submits it.
   Synthetic prewarm frames are not reward-requested; a live gate (set once
   the service has published READY) refuses them regardless.
3. Ownership: the ticket carries bounded copies of the record dicts
   (``tuple(dict(r) for r in records)``); the map builder copies them again;
   labels are read only by the evaluator.  Tests assert neither branch mutates.
4. The compute/map path never waits on GT or evaluation: ``submit`` is a
   ``put_nowait`` (its queue-full path emits an infrastructure-fault feedback
   synchronously, exactly as today, just earlier); an exception from that path
   is counted, noted in the profiler and does not stop the map update.
5. Failure accounting is unchanged: map-build or publish failures still raise
   or count where they do today; no terminal path is touched.

Counters are written create-only to ``EDGE_ORDERING.json`` in the edge
profiler leaf when the evaluator closes (so both arms record their ordering).
"""

from __future__ import annotations

import collections
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

SERIAL, EARLY = "serial", "early"
ORDERING_FILE = "edge_ordering.json"        # written by the UE child into the edge scratch leaf
COUNTERS_NAME = "EDGE_ORDERING.json"        # written by the edge at evaluator close
_STATE: dict[str, Any] = {"evaluator": None, "live": False, "ordering": SERIAL,
                          "counters": collections.Counter(), "errors": [], "dir": None,
                          "submitted_keys": set()}


class EdgeOrderingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EdgeOrderingError(message)


def read_ordering(path: Path) -> str:
    """The registered ordering for this cell; absent file means serial (control)."""
    if not Path(path).is_file():
        return SERIAL
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    ordering = str(document.get("ordering", SERIAL))
    require(ordering in (SERIAL, EARLY), f"unknown edge ordering {ordering!r}")
    return ordering


def write_ordering_config(directory: Path, ordering: str, arm: str) -> Path:
    require(ordering in (SERIAL, EARLY), f"unknown edge ordering {ordering!r}")
    path = Path(directory) / ORDERING_FILE
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"ordering": ordering, "arm": arm, "written_wall_ns": time.time_ns()}, handle)
    return path


def mark_live() -> None:
    _STATE["live"] = True


def reset_for_tests() -> None:
    _STATE.update({"evaluator": None, "live": False, "ordering": SERIAL,
                   "counters": collections.Counter(), "errors": [], "dir": None,
                   "submitted_keys": set()})


def ticket_key(envelope: Any) -> tuple:
    return (envelope.session_uuid, envelope.controller_lineage_sha256, envelope.decision_seq,
            envelope.ticket_seq, envelope.frame_id, envelope.tensor_seq)


def counters_snapshot() -> dict[str, Any]:
    return {"ordering": _STATE["ordering"], "live": _STATE["live"],
            "counters": dict(_STATE["counters"]), "errors": list(_STATE["errors"])[:32]}


def make_early_submit_processor(base: type, edge_module: Any, map_protocol: Any,
                                recorder_fn: Any) -> type:
    """Processor subclass that submits the ticket from inside the compute wrapper."""

    class EarlySubmitProcessorV1(base):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            inner = self._compute

            def compute(**kw: Any) -> Any:
                records, labels = inner(**kw)          # materialized: CPU records + numpy labels
                self._early_submit(records, labels, **kw)
                return records, labels

            self._compute = compute
            self._early_keys: set = set()

        def _early_submit(self, records: Any, labels: Any, *, envelope: Any, context: Any,
                          profile: Any, identity: Any) -> None:
            counters = _STATE["counters"]
            if not envelope.reward_requested:
                counters["non_reward_frames"] += 1
                return
            counters["reward_frames_seen"] += 1
            if not _STATE["live"]:                       # prewarm / pre-READY: never evaluate
                counters["live_gate_rejected"] += 1
                return
            evaluator = _STATE["evaluator"]
            require(evaluator is not None, "early submission needs the live evaluator")
            key = ticket_key(envelope)
            require(key not in _STATE["submitted_keys"], f"ticket {key} already submitted")
            ticket = edge_module.EvaluationTicketV2(
                envelope=envelope, context=context,
                records=tuple(dict(record) for record in records),      # bounded copies
                predicted_mask=labels,
                gt_identity=map_protocol.gt_identity(
                    run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
                    frame_id=context.frame_id, anchor_action_id=profile.action_id,
                    anchor_profile_id=profile.profile_id,
                    capture_timestamp_ns=context.capture_timestamp_ns),
                enqueued_wall_ns=time.time_ns())
            _STATE["submitted_keys"].add(key)
            self._early_keys.add(key)
            try:
                evaluator.submit(ticket)
                counters["early_submissions"] += 1
            except Exception as exc:  # noqa: BLE001 - visible, never stops the map path
                counters["early_submit_errors"] += 1
                _STATE["errors"].append(f"frame {envelope.frame_id}: {type(exc).__name__}: {exc}")
                rec = recorder_fn()
                if getattr(rec, "enabled", False):
                    rec.note("edge_early_submit_error", frame_id=int(envelope.frame_id),
                             error=f"{type(exc).__name__}: {exc}"[:160])

        def process(self, frame: bytes, *, edge_timing: Mapping[str, Any],
                    segmentation: Optional[Mapping[str, Any]] = None) -> Any:
            processed = super().process(frame, edge_timing=edge_timing, segmentation=segmentation)
            if processed.evaluation is not None and ticket_key(processed.envelope) in self._early_keys:
                _STATE["counters"]["loop_submission_suppressed"] += 1
                return dataclasses.replace(processed, evaluation=None)
            return processed

    return EarlySubmitProcessorV1


def make_evaluator(base: type) -> type:
    """Evaluator subclass that registers itself for early submission and writes
    the ordering counters when the service closes it (both arms)."""

    class OrderingEvaluatorV1(base):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            _STATE["evaluator"] = self

        def close(self, timeout_s: Optional[float] = None) -> dict[str, Any]:
            try:
                return super().close(timeout_s=timeout_s)
            finally:
                write_counters()

    return OrderingEvaluatorV1


def write_counters() -> Optional[Path]:
    directory = _STATE.get("dir")
    if directory is None:
        return None
    path = Path(directory) / COUNTERS_NAME
    if path.exists():
        return path
    with path.open("x", encoding="utf-8") as handle:
        json.dump({**counters_snapshot(), "written_wall_ns": time.time_ns()}, handle, indent=1,
                  sort_keys=True)
    return path


def install(*, edge_module: Any, prewarm_module: Any, map_protocol: Any, ordering: str,
            counters_dir: Path, recorder_fn: Any) -> dict[str, Any]:
    """Wrap the (already profiled) edge classes.  Serial arms only get the
    counters/live-gate plumbing so both arms record the same evidence."""
    require(ordering in (SERIAL, EARLY), f"unknown edge ordering {ordering!r}")
    _STATE["ordering"] = ordering
    _STATE["dir"] = Path(counters_dir)
    edge_module.Run4EvaluatorV2 = make_evaluator(edge_module.Run4EvaluatorV2)
    original_ready = prewarm_module.publish_ready_after_warmup

    def publish_ready_after_warmup(*args: Any, **kwargs: Any) -> Any:
        result = original_ready(*args, **kwargs)
        mark_live()
        return result

    prewarm_module.publish_ready_after_warmup = publish_ready_after_warmup
    if ordering == EARLY:
        edge_module.Run4EdgeProcessorV2 = make_early_submit_processor(
            edge_module.Run4EdgeProcessorV2, edge_module, map_protocol, recorder_fn)
    return {"ordering": ordering, "counters_dir": str(counters_dir)}
