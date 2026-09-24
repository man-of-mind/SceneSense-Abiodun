#!/usr/bin/env python3
"""Build causal per-decision records for one cell.

Every state value attached to decision *t* is an observation that already
existed strictly before *t*. Nothing is forward-filled past its validity
window, and "no prior grant" stays missing rather than becoming MCS 0.

Clock handling is the crux. The T-tracer renders its ``CLOCK_REALTIME`` header
as a date-less local ``HH:MM:SS.ffffff``, while the sender and the production
receiver both stamp ``CLOCK_MONOTONIC``. ``NR_PDCP_TX_SDU`` carries *both* --
the tracer header and an in-payload ``CLOCK_MONOTONIC`` pair taken at the same
call site (``nr_pdcp_oai_api.c:941-944``). Those rows are therefore a
**measured same-event clock bridge**, not an assumed one, and its residual
spread is reported so the join precision is visible rather than trusted.
"""

from __future__ import annotations

import bisect
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    DCI_GRANT_HEADER, MCS_MAX_AGE_MS, NEW_DATA_HARQ_ROUND, PDCP_TX_SDU_HEADER,
    RLC_BUFFER_HEADER, UL_DIRECTION,
)


class JoinError(RuntimeError):
    """A join cannot be performed safely and is refused."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JoinError(message)


def read_exact_csv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        actual = next(reader, [])
        require(tuple(actual) == tuple(header),
                f"{path}: unexpected header\n expected {list(header)}\n found {actual}")
        return [dict(zip(header, record)) for record in reader
                if len(record) == len(header)]


def tracer_time_of_day_ns(text: str) -> int:
    """Date-less ``HH:MM:SS.ffffff`` to nanoseconds-of-day."""
    parsed = datetime.strptime(text.strip(), "%H:%M:%S.%f").time()
    return (((parsed.hour * 60 + parsed.minute) * 60 + parsed.second) * 1_000_000
            + parsed.microsecond) * 1000


NS_PER_DAY = 86_400 * 1_000_000_000


# --------------------------------------------------------------------------
# Clock bridge
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockBridge:
    """Measured mapping from tracer time-of-day to CLOCK_MONOTONIC ns."""

    offset_ns: int
    samples: int
    residual_p50_ns: float
    residual_p95_ns: float
    residual_max_ns: float
    day_wraps: int

    def to_monotonic(self, time_of_day_ns: int) -> int:
        return time_of_day_ns + self.offset_ns

    def to_json(self) -> dict[str, Any]:
        return {
            "method": "SAME_EVENT_BRIDGE_FROM_NR_PDCP_TX_SDU",
            "offset_ns": self.offset_ns, "samples": self.samples,
            "residual_ns": {
                "p50": self.residual_p50_ns, "p95": self.residual_p95_ns,
                "max": self.residual_max_ns,
            },
            "day_wraps": self.day_wraps,
            "note": ("tracer header CLOCK_REALTIME and in-payload CLOCK_MONOTONIC "
                     "are captured at the same call site, so this is a measured "
                     "bridge rather than a reconstructed date"),
        }


def build_clock_bridge(pdcp_rows: Sequence[Mapping[str, str]]) -> ClockBridge:
    """Fit tracer-time -> monotonic from same-event NR_PDCP_TX_SDU pairs."""
    require(bool(pdcp_rows),
            "no NR_PDCP_TX_SDU rows: without them there is no measured clock "
            "bridge and the join is refused rather than assumed")
    offsets: list[int] = []
    for row in pdcp_rows:
        tod = tracer_time_of_day_ns(row["time"])
        mono = int(row["mono_sec"]) * 1_000_000_000 + int(row["mono_nsec"])
        offsets.append(mono - tod)
    offsets.sort()
    median = offsets[len(offsets) // 2]
    # A run that straddles midnight shows a one-day step in the raw offsets.
    wraps = sum(1 for value in offsets if abs(value - median) > NS_PER_DAY // 2)
    kept = [value for value in offsets if abs(value - median) <= NS_PER_DAY // 2]
    median = sorted(kept)[len(kept) // 2]
    residuals = sorted(abs(value - median) for value in kept)
    return ClockBridge(
        offset_ns=median, samples=len(kept),
        residual_p50_ns=float(residuals[len(residuals) // 2]),
        residual_p95_ns=float(residuals[min(len(residuals) - 1,
                                            int(0.95 * len(residuals)))]),
        residual_max_ns=float(residuals[-1]),
        day_wraps=wraps,
    )


# --------------------------------------------------------------------------
# UE-local observations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BacklogTick:
    """One UL MAC tick's total RLC occupancy, raw bytes."""

    monotonic_ns: int
    total_bytes: int
    per_lcid: tuple[tuple[int, int], ...]


def build_backlog_ticks(
    rows: Sequence[Mapping[str, str]], bridge: ClockBridge
) -> list[BacklogTick]:
    """Group per-LCID rows into per-tick raw totals.

    ``nr_update_rlc_buffers_status`` emits one row per active LCID inside a
    single call sharing ``(frame, slot)``. A tick ends when ``(frame, slot)``
    changes or an LCID repeats. Zero rows are kept, so an empty buffer stays a
    measured zero rather than a gap.
    """
    ticks: list[BacklogTick] = []
    current: list[tuple[int, int, int, int]] = []  # tod, frame, slot, lcid/bytes

    def flush() -> None:
        if not current:
            return
        ticks.append(BacklogTick(
            monotonic_ns=bridge.to_monotonic(current[0][0]),
            total_bytes=sum(item[4] for item in current),  # type: ignore[misc]
            per_lcid=tuple(sorted((item[3], item[4]) for item in current)),  # type: ignore[misc]
        ))
        current.clear()

    packed: list[tuple[int, int, int, int, int]] = []
    for row in rows:
        packed.append((
            tracer_time_of_day_ns(row["time"]), int(row["frame"]), int(row["slot"]),
            int(row["lcid"]), int(row["bytes_in_buffer"]),
        ))
    for item in packed:
        if current and ((item[1], item[2]) != (current[0][1], current[0][2])
                        or any(member[3] == item[3] for member in current)):
            flush()
        current.append(item)  # type: ignore[arg-type]
    flush()
    ticks.sort(key=lambda tick: tick.monotonic_ns)
    return ticks


@dataclass(frozen=True)
class UlGrant:
    """One UE-decoded uplink DCI grant."""

    monotonic_ns: int
    mcs: int
    mcs_table: int
    rb_size: int
    tbs: int
    harq_pid: int
    ndi: int
    rv: int
    harq_round: int


def build_ul_grants(
    rows: Sequence[Mapping[str, str]], bridge: ClockBridge
) -> tuple[list[UlGrant], dict[str, int]]:
    """UE-decoded uplink grants, split into new-data and retransmission.

    Only HARQ round 0 grants carry a freshly selected MCS. A retransmission
    repeats the original transmission's MCS and would smear a stale scheduler
    decision into the feature, so it is excluded and counted.
    """
    new_data: list[UlGrant] = []
    counts = {"ul_rows": 0, "new_data": 0, "retransmission": 0, "non_ul": 0}
    for row in rows:
        if row["direction"] != UL_DIRECTION:
            counts["non_ul"] += 1
            continue
        counts["ul_rows"] += 1
        harq_round = int(row["round"])
        if harq_round != NEW_DATA_HARQ_ROUND:
            counts["retransmission"] += 1
            continue
        counts["new_data"] += 1
        new_data.append(UlGrant(
            monotonic_ns=bridge.to_monotonic(tracer_time_of_day_ns(row["time"])),
            mcs=int(row["mcs"]), mcs_table=int(row["mcs_table"]),
            rb_size=int(row["rb_size"]), tbs=int(row["tbs"]),
            harq_pid=int(row["harq_pid"]), ndi=int(row["ndi"]),
            rv=int(row["rv"]), harq_round=harq_round,
        ))
    new_data.sort(key=lambda grant: grant.monotonic_ns)
    return new_data, counts


# --------------------------------------------------------------------------
# Receiver outcomes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameArrival:
    first_monotonic_ns: int
    last_monotonic_ns: int
    unique_chunks: int
    complete: bool


def build_arrivals(events_jsonl: Path, expected_chunks: int) -> dict[int, FrameArrival]:
    """Per-frame first/last datagram arrival and completeness.

    Keyed by the receiver's ``frame_index``, which restarts at 0 in every block
    because each block has its own receiver instance. Duplicates are not
    arrivals, so a retransmitted chunk cannot shorten a frame's measured span.
    """
    seen: dict[int, set[int]] = {}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    with events_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            index = event.get("frame_index")
            if index is None or event.get("status") == "MALFORMED":
                continue
            chunk = event.get("chunk_index")
            # The production receiver names this field receiver_monotonic_ns.
            stamp = int(event["receiver_monotonic_ns"])
            bucket = seen.setdefault(int(index), set())
            if chunk in bucket:
                continue                      # duplicate: not a new arrival
            bucket.add(int(chunk))
            first.setdefault(int(index), stamp)
            last[int(index)] = stamp
    return {
        index: FrameArrival(
            first_monotonic_ns=first[index], last_monotonic_ns=last[index],
            unique_chunks=len(chunks), complete=len(chunks) == expected_chunks,
        )
        for index, chunks in seen.items()
    }


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------

DECISION_FIELDS = (
    "cell_id", "profile_id", "order_index", "repetition", "sequence",
    "block_index", "tier", "action_id", "decision_index",
    "frame_index_in_block", "decisions_since_transition", "previous_tier",
    "is_first_frame_of_block",
    "decision_monotonic_ns", "schedule_lag_ms",
    "payload_bytes", "chunks_per_frame", "chunks_sent", "chunks_dropped",
    "send_terminal_reason",
    "pre_enqueue_backlog_bytes", "backlog_age_ms", "backlog_status",
    "previous_ul_mcs", "mcs_age_ms", "mcs_status",
    "mcs_table", "prev_grant_rb_size", "prev_grant_tbs", "prev_grant_harq_pid",
    "prev_grant_ndi", "prev_grant_rv", "prev_grant_round",
    "first_arrival_monotonic_ns", "last_arrival_monotonic_ns",
    "unique_chunks_received", "complete_reassembly",
    "uplink_latency_ms", "within_transport_budget", "terminal_outcome",
)


def join_cell(
    *,
    cell_meta: Mapping[str, Any],
    sender_rows: Sequence[Mapping[str, str]],
    ticks: Sequence[BacklogTick],
    grants: Sequence[UlGrant],
    arrivals_by_block: Mapping[int, Mapping[int, FrameArrival]],
    budget_ms: float,
) -> list[dict[str, Any]]:
    """One causal record per application decision, across every block."""
    tick_times = [tick.monotonic_ns for tick in ticks]
    grant_times = [grant.monotonic_ns for grant in grants]
    out: list[dict[str, Any]] = []

    for row in sender_rows:
        decision = int(row["decision_monotonic_ns"])
        block_index = int(row["block_index"])
        frame_index = int(row["frame_index_in_block"])

        # --- backlog: last tick strictly before the decision ---------
        position = bisect.bisect_left(tick_times, decision) - 1
        if position >= 0:
            tick = ticks[position]
            backlog = tick.total_bytes
            backlog_age = (decision - tick.monotonic_ns) / 1e6
            backlog_status = "OBSERVED"
        else:
            backlog, backlog_age, backlog_status = None, None, "MISSING_NO_PRIOR_TICK"

        # --- previous new-data UL MCS: strictly before, bounded age ---
        position = bisect.bisect_left(grant_times, decision) - 1
        if position < 0:
            mcs, mcs_age, mcs_status, grant = None, None, "MISSING_NO_PRIOR_GRANT", None
        else:
            grant = grants[position]
            mcs_age = (decision - grant.monotonic_ns) / 1e6
            if mcs_age > MCS_MAX_AGE_MS:
                # Never forward-filled past its validity window, and never
                # coerced to 0, which is a real modulation index.
                mcs, mcs_status, grant = None, "MISSING_STALE", grant
            else:
                mcs, mcs_status = grant.mcs, "OBSERVED"

        arrival = arrivals_by_block.get(block_index, {}).get(frame_index)
        if arrival is None:
            first = last = None
            unique = 0
            complete = False
            latency = None
        else:
            first = arrival.first_monotonic_ns
            last = arrival.last_monotonic_ns
            unique = arrival.unique_chunks
            complete = arrival.complete
            latency = (last - decision) / 1e6 if complete else None

        if complete:
            terminal = "COMPLETE_REASSEMBLY"
        elif unique > 0:
            terminal = "INCOMPLETE_REASSEMBLY"
        elif row["terminal_reason"] == "SOCKET_BACKPRESSURE_ALL_CHUNKS_DROPPED":
            terminal = "NEVER_SENT_SOCKET_BACKPRESSURE"
        else:
            terminal = "NO_DATAGRAM_ARRIVED"

        out.append({
            "cell_id": cell_meta["cell_id"],
            "profile_id": cell_meta["profile_id"],
            "order_index": cell_meta["order_index"],
            "repetition": cell_meta["repetition"],
            "sequence": "->".join(cell_meta["sequence"]),
            "block_index": block_index,
            "tier": row["tier"],
            "action_id": int(row["action_id"]),
            "decision_index": int(row["decision_index"]),
            "frame_index_in_block": frame_index,
            "decisions_since_transition": int(row["decisions_since_transition"]),
            "previous_tier": row["previous_tier"],
            "is_first_frame_of_block": row["is_first_frame_of_block"] in ("True", "true", "1"),
            "decision_monotonic_ns": decision,
            "schedule_lag_ms": float(row["schedule_lag_ms"]),
            "payload_bytes": int(row["payload_bytes"]),
            "chunks_per_frame": int(row["chunks_per_frame"]),
            "chunks_sent": int(row["chunks_sent"]),
            "chunks_dropped": int(row["chunks_dropped"]),
            "send_terminal_reason": row["terminal_reason"],
            "pre_enqueue_backlog_bytes": backlog,
            "backlog_age_ms": backlog_age,
            "backlog_status": backlog_status,
            "previous_ul_mcs": mcs,
            "mcs_age_ms": mcs_age,
            "mcs_status": mcs_status,
            "mcs_table": grant.mcs_table if grant is not None else None,
            "prev_grant_rb_size": grant.rb_size if grant is not None else None,
            "prev_grant_tbs": grant.tbs if grant is not None else None,
            "prev_grant_harq_pid": grant.harq_pid if grant is not None else None,
            "prev_grant_ndi": grant.ndi if grant is not None else None,
            "prev_grant_rv": grant.rv if grant is not None else None,
            "prev_grant_round": grant.harq_round if grant is not None else None,
            "first_arrival_monotonic_ns": first,
            "last_arrival_monotonic_ns": last,
            "unique_chunks_received": unique,
            "complete_reassembly": complete,
            "uplink_latency_ms": latency,
            "within_transport_budget": (
                None if latency is None else bool(latency <= budget_ms)),
            "terminal_outcome": terminal,
        })
    return out


def causal_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove the ordering and missingness rules actually held."""
    negative_latency = sum(
        1 for row in records
        if row["uplink_latency_ms"] is not None and row["uplink_latency_ms"] < 0)
    negative_age = sum(
        1 for row in records
        for key in ("backlog_age_ms", "mcs_age_ms")
        if row[key] is not None and row[key] < 0)
    stale_used = sum(
        1 for row in records
        if row["mcs_status"] == "OBSERVED" and (row["mcs_age_ms"] or 0) > MCS_MAX_AGE_MS)
    zero_coerced = sum(
        1 for row in records
        if row["mcs_status"] != "OBSERVED" and row["previous_ul_mcs"] == 0)
    tables = {row["mcs_table"] for row in records if row["mcs_table"] is not None}
    rounds = {row["prev_grant_round"] for row in records
              if row["prev_grant_round"] is not None}
    return {
        "records": len(records),
        "negative_uplink_latency": negative_latency,
        "negative_observation_age": negative_age,
        "stale_mcs_used_as_observed": stale_used,
        "missing_mcs_coerced_to_zero": zero_coerced,
        "mcs_observed": sum(1 for r in records if r["mcs_status"] == "OBSERVED"),
        "mcs_missing_stale": sum(1 for r in records if r["mcs_status"] == "MISSING_STALE"),
        "mcs_missing_no_prior": sum(
            1 for r in records if r["mcs_status"] == "MISSING_NO_PRIOR_GRANT"),
        "backlog_observed": sum(1 for r in records if r["backlog_status"] == "OBSERVED"),
        "distinct_mcs_tables": sorted(tables),
        "distinct_prev_grant_rounds": sorted(rounds),
        "all_joined_observations_precede_decision": (
            negative_age == 0),
    }
