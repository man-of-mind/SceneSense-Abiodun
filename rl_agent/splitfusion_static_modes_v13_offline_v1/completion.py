"""Prediction-completion accounting for one live cell (format shared by the M1 and v13 captures).

Definition (registered): a policy decision's prediction is COMPLETED iff its live ``edge.compute`` span
(``ph == "live"``, ``reward_requested is True``) ended with ``ok is True`` AND the GT service recorded the edge's
``edge:prediction_materialized`` stamp for that frame (set by the edge immediately after the compute returns: it
must lie at/after the span end and at/before the service's receipt of the prediction).  Latency = span end - action open (both
CLOCK_MONOTONIC_RAW).  The two signals must agree one-to-one; any disagreement is an accounting failure, never a
silent choice of one of them.

Never used as completion: TIMEOUT / STALE timestamps, failed spans, map publication, feedback receipt, service receipt.
Decisions without a completed prediction keep their service outcome and edge terminal as subcategories; a
missing compute span is reported as such (not as packet loss).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

DEADLINE_MS = 170.0


def _jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def find_cell(root):
    cells = sorted(Path(root).glob("capture/cells/*/CELL_RESULT.json"))
    if len(cells) != 1:
        raise ValueError(f"expected exactly one cell below {root}, found {len(cells)}")
    return cells[0].parent


def edge_terminals(cell):
    """frame_id -> terminal outcome marks emitted by the edge (if any)."""
    out = collections.defaultdict(list)
    for row in _jsonl(cell / "profiler/edge/edge.profile.jsonl"):
        if row.get("r") == "mark" and row.get("stage") == "edge.terminal_built":
            out[int(row.get("frame_id", -1))].append(dict(outcome=row.get("outcome"),
                                                          stage=row.get("terminal_stage")))
    return out


def account(root):
    cell = find_cell(root)
    frames = [f for f in _jsonl(cell / "numeric_gt/service/FRAMES.jsonl") if f.get("decision")]
    spans = collections.defaultdict(list)
    all_live = []
    for row in _jsonl(cell / "profiler/edge/edge.profile.jsonl"):
        if row.get("r") == "span" and row.get("stage") == "edge.compute" and row.get("ph") == "live":
            all_live.append(row)
            if row.get("reward_requested") is True:
                spans[int(row["frame_id"])].append(row)
    terminals = edge_terminals(cell)
    decisions, problems = [], []
    for f in frames:
        fid, opened = int(f["frame_id"]), int(f["action_open_raw_ns"])
        events = {}
        for e in f.get("events") or []:
            events.setdefault(f"{e.get('role')}:{e.get('name')}", e.get("raw_ns"))
        materialized = events.get("edge:prediction_materialized")
        rows = spans.get(fid, [])
        if len(rows) > 1:
            problems.append(f"frame {fid}: {len(rows)} live reward compute spans")
        span = rows[0] if rows else None
        ok_span = span is not None and span.get("ok") is True
        if ok_span != (materialized is not None):
            problems.append(f"frame {fid}: span ok={None if span is None else span.get('ok')} but materialized="
                            f"{materialized is not None}")
        received = events.get("service:prediction_received")
        if ok_span and materialized is not None and not (
                int(span["end_raw_ns"]) <= int(materialized)
                and (received is None or int(materialized) <= int(received))):
            problems.append(f"frame {fid}: materialization stamp not between compute end and service receipt")
        completed = ok_span and materialized is not None
        latency = (int(span["end_raw_ns"]) - opened) / 1e6 if completed else None
        state = "NONE" if not completed else ("TIMELY" if latency <= DEADLINE_MS else "LATE")
        decisions.append(dict(
            frame_id=fid, action_open_raw_ns=opened, state=state, latency_ms=latency,
            compute_span=None if span is None else ("OK" if span.get("ok") is True else "FAILED"),
            service_outcome=f.get("outcome"), edge_terminals=terminals.get(fid, []),
            mode_id=None if span is None else span.get("mode_id"), q_e4=None if span is None else span.get("q_e4")))
    states = collections.Counter(d["state"] for d in decisions)
    none = [d for d in decisions if d["state"] == "NONE"]
    latencies = sorted(d["latency_ms"] for d in decisions if d["latency_ms"] is not None)
    return dict(
        cell=str(cell), decisions=len(decisions), completed=states["TIMELY"] + states["LATE"],
        completed_by_170=states["TIMELY"], completed_later=states["LATE"], no_completed_prediction=states["NONE"],
        no_completion_subcategories=dict(collections.Counter(
            (d["compute_span"] or "NO_COMPUTE_SPAN") + "|" + str(d["service_outcome"]) for d in none)),
        failed_compute_span_frames=[d["frame_id"] for d in decisions if d["compute_span"] == "FAILED"],
        latency_ms=latencies, live_compute_spans=len(all_live),
        live_compute_ok=sum(r.get("ok") is True for r in all_live),
        executed_modes=dict(collections.Counter((r.get("mode_id"), r.get("q_e4")) for r in all_live)),
        accounting_problems=problems, per_decision=decisions)
