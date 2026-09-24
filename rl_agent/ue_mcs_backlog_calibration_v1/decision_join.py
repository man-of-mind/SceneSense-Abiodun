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
    DCI_GRANT_HEADER, GNB_MCS_DECISION_HEADER, NEW_DATA_HARQ_ROUND,
    PDCP_TX_SDU_HEADER, RLC_BUFFER_HEADER, UL_DIRECTION,
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


def build_clock_bridge_from_sender(
    sender_rows: Sequence[Mapping[str, str]]
) -> ClockBridge:
    """Fallback bridge, from the sender's own same-instant clock pairs.

    The preferred bridge is ``NR_PDCP_TX_SDU``, which carries the tracer's
    CLOCK_REALTIME header and an in-payload CLOCK_MONOTONIC stamp from one call
    site. When that event is not emitted, this is an equally *measured*
    substitute: for every decision the sender reads ``CLOCK_REALTIME`` and
    ``CLOCK_MONOTONIC`` microseconds apart in one process, giving hundreds of
    same-instant pairs spanning the whole cell. Both clocks are system-wide on
    Linux, so the pairs bridge the tracer's realtime domain to the monotonic
    domain the sender and receiver share.

    The only reconstructed quantity is the *date*, which the tracer drops. It is
    recovered from the same wall timestamps and is unambiguous for a run of this
    length; :func:`audit_bridge_window` checks the result actually lands inside
    the cell's own wall-clock window instead of trusting it.
    """
    pairs: list[tuple[int, int]] = []
    for row in sender_rows:
        wall = row.get("decision_wall_ns")
        mono = row.get("decision_monotonic_ns")
        if not wall or not mono:
            continue
        pairs.append((int(wall), int(mono)))
    require(bool(pairs),
            "sender recorded no wall/monotonic pairs; no measured clock bridge")

    reference = datetime.fromtimestamp(pairs[0][0] / 1e9).astimezone()
    utc_offset_ns = int(reference.utcoffset().total_seconds()) * 1_000_000_000

    offsets = [mono - ((wall + utc_offset_ns) % NS_PER_DAY) for wall, mono in pairs]
    offsets.sort()
    median = offsets[len(offsets) // 2]
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


def audit_bridge_window(
    bridge: ClockBridge, tracer_rows: Sequence[Mapping[str, str]],
    sender_rows: Sequence[Mapping[str, str]], *, slack_s: float = 120.0
) -> dict[str, Any]:
    """Check converted tracer times land inside the cell's own wall window.

    A wrong date reconstruction would shift every tracer sample by whole days,
    which this catches immediately rather than letting it silently poison the
    join.
    """
    monos = [int(r["decision_monotonic_ns"]) for r in sender_rows
             if r.get("decision_monotonic_ns")]
    require(bool(monos), "sender rows carry no monotonic timestamps")
    low = min(monos) - int(slack_s * 1e9)
    high = max(monos) + int(slack_s * 1e9)
    converted = [bridge.to_monotonic(tracer_time_of_day_ns(r["time"]))
                 for r in tracer_rows[:5000]]
    inside = sum(1 for value in converted if low <= value <= high)
    return {
        "checked_rows": len(converted),
        "inside_cell_window": inside,
        "fraction_inside": (inside / len(converted)) if converted else None,
        "window_slack_s": slack_s,
        "verified": bool(converted) and inside / len(converted) >= 0.95,
    }


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
class UeUlGrant:
    """A UE-decoded round-0 UL grant with its exact scheduling identity."""

    monotonic_ns: int
    rnti: int
    dci_frame: int
    dci_slot: int
    sched_frame: int
    sched_slot: int
    mcs: int
    mcs_table: int
    rb_size: int
    tbs: int
    harq_pid: int
    ndi: int
    rv: int
    harq_round: int

    @property
    def schedule_identity(self) -> tuple[int, int, int, int, int, int]:
        return (self.rnti, self.dci_frame, self.dci_slot, self.sched_frame,
                self.sched_slot, self.mcs_table)

    @property
    def provenance_identity(self) -> tuple[int, int, int, int, int, int, int]:
        return (self.monotonic_ns, *self.schedule_identity)


@dataclass(frozen=True)
class GnbMcsDecision:
    """gNB-side provenance for one UL grant; never a policy input."""

    monotonic_ns: int
    rnti: int
    frame: int
    slot: int
    sched_frame: int
    sched_slot: int
    mcs_table: int
    selected_mcs: int
    final_mcs: int

    @property
    def schedule_identity(self) -> tuple[int, int, int, int, int, int]:
        return (self.rnti, self.frame, self.slot, self.sched_frame,
                self.sched_slot, self.mcs_table)


def build_ul_grants(
    rows: Sequence[Mapping[str, str]], bridge: ClockBridge
) -> tuple[list[UeUlGrant], dict[str, int]]:
    """UE-decoded uplink grants, split into new-data and retransmission.

    Only HARQ round 0 grants carry a freshly selected MCS. A retransmission
    repeats the original transmission's MCS and would smear a stale scheduler
    decision into the feature, so it is excluded and counted.
    """
    new_data: list[UeUlGrant] = []
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
        new_data.append(UeUlGrant(
            monotonic_ns=bridge.to_monotonic(tracer_time_of_day_ns(row["time"])),
            rnti=int(row["rnti"]), dci_frame=int(row["dci_frame"]),
            dci_slot=int(row["dci_slot"]), sched_frame=int(row["sched_frame"]),
            sched_slot=int(row["sched_slot"]),
            mcs=int(row["mcs"]), mcs_table=int(row["mcs_table"]),
            rb_size=int(row["rb_size"]), tbs=int(row["tbs"]),
            harq_pid=int(row["harq_pid"]), ndi=int(row["ndi"]),
            rv=int(row["rv"]), harq_round=harq_round,
        ))
    new_data.sort(key=lambda grant: grant.monotonic_ns)
    return new_data, counts


def build_gnb_mcs_decisions(
    rows: Sequence[Mapping[str, str]], bridge: ClockBridge
) -> list[GnbMcsDecision]:
    """Parse exact-schema gNB MCS decisions for provenance only."""
    decisions = [GnbMcsDecision(
        monotonic_ns=bridge.to_monotonic(tracer_time_of_day_ns(row["time"])),
        rnti=int(row["rnti"]), frame=int(row["frame"]), slot=int(row["slot"]),
        sched_frame=int(row["sched_frame"]), sched_slot=int(row["sched_slot"]),
        mcs_table=int(row["mcs_table"]), selected_mcs=int(row["selected_mcs"]),
        final_mcs=int(row["final_mcs"]),
    ) for row in rows]
    decisions.sort(key=lambda item: item.monotonic_ns)
    return decisions


def audit_ue_gnb_mcs_provenance(
    ue_grants: Sequence[UeUlGrant],
    gnb_decisions: Sequence[GnbMcsDecision],
    *,
    max_delta_ms: float = 10.0,
) -> dict[str, Any]:
    """One-to-one reconciliation using exact schedule identity plus time.

    NR frame numbers wrap, so schedule identity alone is intentionally not
    treated as globally unique.  Within an identity bucket a UE grant is paired
    only with one unused gNB record inside the explicit same-event time bound.
    Equal-nearest candidates are ambiguous and remain unmatched.
    """
    require(max_delta_ms > 0, "provenance max_delta_ms must be positive")
    by_identity: dict[tuple[int, int, int, int, int, int], list[GnbMcsDecision]] = {}
    for item in gnb_decisions:
        by_identity.setdefault(item.schedule_identity, []).append(item)
    used: set[tuple[tuple[int, int, int, int, int, int], int]] = set()
    max_delta_ns = int(max_delta_ms * 1e6)
    deltas_ms: list[float] = []
    selected_final_adjustments: list[dict[str, Any]] = []
    ue_final_mismatches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for ue in ue_grants:
        identity = ue.schedule_identity
        candidates = []
        for index, gnb in enumerate(by_identity.get(identity, ())):
            token = (identity, index)
            delta = abs(ue.monotonic_ns - gnb.monotonic_ns)
            if token not in used and delta <= max_delta_ns:
                candidates.append((delta, index, gnb))
        candidates.sort(key=lambda item: item[0])
        base = {"schedule_identity": list(identity), "ue_mcs": ue.mcs}
        if not candidates:
            unmatched.append(base)
            continue
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            ambiguous.append({**base, "nearest_delta_ms": candidates[0][0] / 1e6})
            continue
        delta, index, gnb = candidates[0]
        used.add((identity, index))
        deltas_ms.append(delta / 1e6)
        detail = {**base, "selected_mcs": gnb.selected_mcs,
                  "final_mcs": gnb.final_mcs, "delta_ms": delta / 1e6}
        if gnb.selected_mcs != gnb.final_mcs:
            selected_final_adjustments.append(detail)
        if ue.mcs != gnb.final_mcs:
            ue_final_mismatches.append(detail)

    matched = len(deltas_ms)
    return {
        "ue_round0_grants": len(ue_grants),
        "gnb_mcs_decisions": len(gnb_decisions),
        "matched": matched,
        "coverage": matched / len(ue_grants) if ue_grants else None,
        "max_match_delta_ms": max_delta_ms,
        "observed_delta_ms": {
            "min": min(deltas_ms) if deltas_ms else None,
            "p50": statistics.median(deltas_ms) if deltas_ms else None,
            "max": max(deltas_ms) if deltas_ms else None,
        },
        "unmatched": len(unmatched), "ambiguous": len(ambiguous),
        "selected_final_adjustments": len(selected_final_adjustments),
        "selected_final_adjustment_rate": (
            len(selected_final_adjustments) / matched if matched else None),
        "ue_final_mismatches": len(ue_final_mismatches),
        "examples": {
            "unmatched": unmatched[:5], "ambiguous": ambiguous[:5],
            "selected_final_adjustments": selected_final_adjustments[:5],
            "ue_final": ue_final_mismatches[:5],
        },
        "provenance_verified": (
            bool(ue_grants) and matched == len(ue_grants)
            and not ambiguous and not ue_final_mismatches),
    }


def grants_used_by_records(
    records: Sequence[Mapping[str, Any]], grants: Sequence[UeUlGrant]
) -> list[UeUlGrant]:
    """Resolve exactly the unique prior grants selected by decision rows."""
    by_identity: dict[tuple[int, int, int, int, int, int, int], UeUlGrant] = {}
    for grant in grants:
        require(grant.provenance_identity not in by_identity,
                f"duplicate UE grant provenance identity: {grant.provenance_identity}")
        by_identity[grant.provenance_identity] = grant
    used: set[tuple[int, int, int, int, int, int, int]] = set()
    for row in records:
        if not row["has_prior_ul_grant"]:
            continue
        identity = (
            int(row["prior_grant_monotonic_ns"]), int(row["prior_grant_rnti"]),
            int(row["prior_grant_dci_frame"]), int(row["prior_grant_dci_slot"]),
            int(row["prior_grant_sched_frame"]), int(row["prior_grant_sched_slot"]),
            int(row["mcs_table"]),
        )
        require(identity in by_identity,
                f"decision references an unknown UE grant identity: {identity}")
        used.add(identity)
    return [by_identity[identity] for identity in sorted(used)]


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
    "prior_ul_mcs_raw", "has_prior_ul_grant", "mcs_age_ms", "mcs_status",
    "prior_grant_monotonic_ns", "prior_grant_rnti",
    "prior_grant_dci_frame", "prior_grant_dci_slot",
    "prior_grant_sched_frame", "prior_grant_sched_slot",
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
    grants: Sequence[UeUlGrant],
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

        # --- previous new-data UL MCS: strictly before, lossless raw value ---
        position = bisect.bisect_left(grant_times, decision) - 1
        if position < 0:
            mcs, mcs_age, mcs_status, grant = None, None, "MISSING_NO_PRIOR_GRANT", None
        else:
            grant = grants[position]
            mcs_age = (decision - grant.monotonic_ns) / 1e6
            mcs, mcs_status = grant.mcs, "OBSERVED_PRIOR"

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
            # The frame was handed to the socket but nothing arrived while the
            # receiver was still listening. Under a saturated uplink a datagram
            # can still be queued when the window closes, so this is stated as
            # a bounded observation rather than as proven loss.
            terminal = "NO_ARRIVAL_WITHIN_OBSERVATION_WINDOW"

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
            "prior_ul_mcs_raw": mcs,
            "has_prior_ul_grant": grant is not None,
            "mcs_age_ms": mcs_age,
            "mcs_status": mcs_status,
            "prior_grant_monotonic_ns": (
                grant.monotonic_ns if grant is not None else None),
            "prior_grant_rnti": grant.rnti if grant is not None else None,
            "prior_grant_dci_frame": (
                grant.dci_frame if grant is not None else None),
            "prior_grant_dci_slot": grant.dci_slot if grant is not None else None,
            "prior_grant_sched_frame": (
                grant.sched_frame if grant is not None else None),
            "prior_grant_sched_slot": (
                grant.sched_slot if grant is not None else None),
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
    zero_coerced = sum(
        1 for row in records
        if not row["has_prior_ul_grant"] and row["prior_ul_mcs_raw"] == 0)
    tables = {row["mcs_table"] for row in records if row["mcs_table"] is not None}
    rounds = {row["prev_grant_round"] for row in records
              if row["prev_grant_round"] is not None}
    return {
        "records": len(records),
        "negative_uplink_latency": negative_latency,
        "negative_observation_age": negative_age,
        "missing_mcs_coerced_to_zero": zero_coerced,
        "mcs_observed": sum(1 for r in records if r["has_prior_ul_grant"]),
        "mcs_missing_no_prior": sum(
            1 for r in records if r["mcs_status"] == "MISSING_NO_PRIOR_GRANT"),
        "backlog_observed": sum(1 for r in records if r["backlog_status"] == "OBSERVED"),
        "distinct_mcs_tables": sorted(tables),
        "distinct_prev_grant_rounds": sorted(rounds),
        "all_joined_observations_precede_decision": (
            negative_age == 0),
    }
