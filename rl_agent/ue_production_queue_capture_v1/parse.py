#!/usr/bin/env python3
"""Raw-evidence parser: sender + receiver + UE T-tracer -> canonical rows.

Produces two create-only tables per campaign:

``frames``  one row per sent frame, with its causal inputs, its measured
            queue quantities and exactly one terminal outcome;
``cycles``  one row per closed d=2 controller cycle
            (decision t, held t+1, successor t+2).

Nothing here fits a model.  Timestamps are never clamped, rows are never
silently dropped, and every exclusion is counted with a reason.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import contract as C


PARSE_SCHEMA = "scenesense.production_queue_capture_parsed.v1"

UE_TRACE_FILES = {
    "pdcp_tx_sdu": "NR_PDCP_TX_SDU.csv",
    "rlc_tx_sdu": "NR_RLC_TX_SDU.csv",
    "rlc_tx_dequeue": "NR_RLC_TX_DEQUEUE.csv",
    "rlc_buffer_status": "NRUE_MAC_RLC_BUFFER_STATUS.csv",
    "dci_grant": "NRUE_MAC_DCI_GRANT.csv",
}

UL_DIRECTION = 1


class ParseError(RuntimeError):
    """A structural expectation about the raw evidence failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParseError(message)


def _wall_ns(value: str) -> int:
    """`HH:MM:SS.ffffff` -> nanoseconds since midnight."""
    hours, minutes, rest = value.split(":")
    seconds, _, fraction = rest.partition(".")
    fraction = (fraction + "000000")[:6]
    return (((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1_000_000
            + int(fraction)) * 1_000


def _mono_ns(row: Mapping[str, str]) -> int:
    return int(row["mono_sec"]) * 1_000_000_000 + int(row["mono_nsec"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing trace: {path}")
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


@dataclass
class ClockBridge:
    """Affine wall->monotonic map built from same-event dual-stamped rows.

    `NR_PDCP_TX_SDU`, `NR_RLC_TX_SDU` and `NR_RLC_TX_DEQUEUE` carry both a wall
    stamp and a monotonic stamp for the *same* event, so the offset is
    measured, never assumed.  Wall-only traces (RLC buffer status, DCI grant)
    are converted with it.
    """

    offset_ns: int
    residual_p95_ns: int
    residual_max_ns: int
    samples: int

    @classmethod
    def build(cls, dual: Sequence[tuple[int, int]]) -> "ClockBridge":
        require(len(dual) >= 100,
                f"clock bridge needs >=100 dual-stamped events, got {len(dual)}")
        offsets = sorted(mono - wall for wall, mono in dual)
        offset = offsets[len(offsets) // 2]
        residuals = sorted(abs(value - offset) for value in offsets)
        index = max(0, int(0.95 * len(residuals)) - 1)
        return cls(offset_ns=offset, residual_p95_ns=residuals[index],
                   residual_max_ns=residuals[-1], samples=len(dual))

    def to_mono(self, wall: int) -> int:
        return wall + self.offset_ns

    def to_json(self) -> dict[str, Any]:
        return {"offset_ns": self.offset_ns,
                "residual_p95_ns": self.residual_p95_ns,
                "residual_max_ns": self.residual_max_ns,
                "samples": self.samples}


@dataclass
class _Series:
    """Sorted monotonic-stamped scalar samples with causal lookup."""

    times: list[int] = field(default_factory=list)
    values: list[int] = field(default_factory=list)

    def add(self, time_ns: int, value: int) -> None:
        self.times.append(time_ns)
        self.values.append(value)

    def finalize(self) -> None:
        order = sorted(range(len(self.times)), key=self.times.__getitem__)
        self.times = [self.times[i] for i in order]
        self.values = [self.values[i] for i in order]

    def latest_strictly_before(
        self, when_ns: int, *, max_age_ns: int,
    ) -> tuple[int | None, int | None]:
        """Most recent sample strictly earlier than `when_ns` and fresh enough."""
        index = bisect.bisect_left(self.times, when_ns) - 1
        if index < 0:
            return None, None
        age = when_ns - self.times[index]
        if age > max_age_ns:
            return None, age
        return self.values[index], age

    def sum_in(self, start_ns: int, end_ns: int) -> int:
        """Sum of values with `start_ns <= t < end_ns`."""
        low = bisect.bisect_left(self.times, start_ns)
        high = bisect.bisect_left(self.times, end_ns)
        return sum(self.values[low:high])

    def count_in(self, start_ns: int, end_ns: int) -> int:
        return (bisect.bisect_left(self.times, end_ns)
                - bisect.bisect_left(self.times, start_ns))


def load_ue_traces(cell_dir: Path) -> dict[str, Any]:
    root = cell_dir / "ttracer" / "ue" / "csv"
    raw = {key: _read_csv(root / name) for key, name in UE_TRACE_FILES.items()}

    dual: list[tuple[int, int]] = []
    for key in ("pdcp_tx_sdu", "rlc_tx_sdu", "rlc_tx_dequeue"):
        for row in raw[key]:
            dual.append((_wall_ns(row["time"]), _mono_ns(row)))
    bridge = ClockBridge.build(dual)

    pdcp = _Series()
    for row in raw["pdcp_tx_sdu"]:
        pdcp.add(_mono_ns(row), int(row["sdu_bytes"]))
    rlc_in = _Series()
    for row in raw["rlc_tx_sdu"]:
        rlc_in.add(_mono_ns(row), int(row["sdu_bytes"]))
    rlc_out = _Series()
    for row in raw["rlc_tx_dequeue"]:
        rlc_out.add(_mono_ns(row), int(row["pdu_bytes"]))

    backlog = _Series()
    for row in raw["rlc_buffer_status"]:
        backlog.add(bridge.to_mono(_wall_ns(row["time"])),
                    int(row["bytes_in_buffer"]))

    mcs = _Series()
    mcs_rejected = 0
    for row in raw["dci_grant"]:
        if (int(row["direction"]) != UL_DIRECTION
                or int(row["round"]) != C.UE_MCS_ROUND
                or int(row["mcs_table"]) != C.UE_MCS_TABLE
                or int(row["ndi"]) != 1):
            mcs_rejected += 1
            continue
        index = int(row["mcs"])
        if not C.UE_MCS_MIN <= index <= C.UE_MCS_MAX:
            mcs_rejected += 1
            continue
        mcs.add(bridge.to_mono(_wall_ns(row["time"])), index)

    grant_bytes = _Series()
    for row in raw["dci_grant"]:
        if int(row["direction"]) == UL_DIRECTION and int(row["round"]) == 0:
            grant_bytes.add(bridge.to_mono(_wall_ns(row["time"])),
                            int(row["tbs"]))

    for series in (pdcp, rlc_in, rlc_out, backlog, mcs, grant_bytes):
        series.finalize()

    return {
        "bridge": bridge, "pdcp": pdcp, "rlc_in": rlc_in, "rlc_out": rlc_out,
        "backlog": backlog, "mcs": mcs, "grant_bytes": grant_bytes,
        "counts": {key: len(value) for key, value in raw.items()},
        "mcs_rows_rejected_non_causal_class": mcs_rejected,
    }


def classify_terminal(
    *, complete: bool, datagrams_received: int, transport_ns: int | None,
) -> str:
    if transport_ns is not None and transport_ns <= 0:
        return "EXCLUDED_INFRASTRUCTURE_FAULT"
    if complete:
        assert transport_ns is not None
        return ("COMPLETE_WITHIN_DEADLINE" if transport_ns <= C.REWARD_DEADLINE_NS
                else "COMPLETE_AFTER_DEADLINE")
    return ("INCOMPLETE_AT_DEADLINE" if datagrams_received > 0
            else "NEVER_COMPLETED")


def closure_seal(
    cell_dir: Path, frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Same-domain sender -> PDCP -> RLC reconciliation for one cell.

    The requirement is a *zero unexplained* residual, not a zero residual: each
    layer adds a constant header.  The constant is measured per event and any
    event that deviates from it is an unexplained residual.
    """
    root = cell_dir / "ttracer" / "ue" / "csv"
    pdcp = [int(row["sdu_bytes"])
            for row in _read_csv(root / UE_TRACE_FILES["pdcp_tx_sdu"])]
    rlc = [int(row["sdu_bytes"])
           for row in _read_csv(root / UE_TRACE_FILES["rlc_tx_sdu"])]
    require(len(pdcp) == len(rlc),
            "PDCP and RLC SDU event counts differ")
    deltas = {b - a for a, b in zip(pdcp, rlc)}
    require(len(deltas) == 1,
            f"RLC-PDCP per-SDU overhead is not constant: {sorted(deltas)[:5]}")
    rlc_overhead = deltas.pop()

    sender = json.loads((cell_dir / "sender_summary.json").read_text())
    receiver_payload = 0
    receiver_wire = 0
    fragmentation_observed = False
    for tier in C.TIER_NAMES:
        value = json.loads(
            (cell_dir / f"receiver_{tier}_summary.json").read_text())
        receiver_payload += int(
            value["total_application_payload_bytes_received"])
        receiver_wire += int(value["total_udp_application_bytes_received"])
        fragmentation_observed = (fragmentation_observed
                                  or bool(value["observed_ip_fragmentation"]))
    totals = sender["totals"]
    outcomes = [row["terminal_outcome"] for row in frames]

    return {
        "sender_udp_application_bytes":
            totals["udp_application_bytes_handed_to_socket"],
        "sender_payload_bytes":
            totals["application_payload_bytes_handed_to_socket"],
        "receiver_udp_application_bytes": receiver_wire,
        "receiver_payload_bytes": receiver_payload,
        "wire_bytes_conserved":
            totals["udp_application_bytes_handed_to_socket"] == receiver_wire,
        "payload_bytes_conserved":
            totals["application_payload_bytes_handed_to_socket"]
            == receiver_payload,
        "datagrams_dropped_at_socket":
            totals["datagrams_dropped_at_socket"],
        "pdcp_sdu_events": len(pdcp), "rlc_sdu_events": len(rlc),
        "pdcp_sdu_bytes": sum(pdcp), "rlc_sdu_bytes": sum(rlc),
        "rlc_minus_pdcp_bytes_per_sdu": rlc_overhead,
        "unexplained_layer_residual_bytes":
            sum(rlc) - sum(pdcp) - rlc_overhead * len(pdcp),
        "observed_ip_fragmentation": fragmentation_observed,
        "frames_with_exactly_one_terminal": len(outcomes),
        "distinct_terminal_outcomes": sorted(set(outcomes)),
        "terminal_outcomes_registered":
            all(value in C.TERMINAL_OUTCOMES for value in outcomes),
    }


def parse_cell(cell_dir: Path) -> dict[str, Any]:
    record = json.loads((cell_dir / "cell_record.json").read_text())
    require(record["status"] == "CAPTURED",
            f"{cell_dir.name}: cell status is {record['status']}")
    sender = {int(row["frame_index"]): row
              for row in _read_csv(cell_dir / "sender_frames.csv")}
    require(len(sender) == C.FRAMES_PER_CELL,
            f"{cell_dir.name}: sender has {len(sender)} frames")

    receiver: dict[int, dict[str, str]] = {}
    for tier in C.TIER_NAMES:
        for row in _read_csv(cell_dir / f"receiver_{tier}_frames.csv"):
            key = int(row["message_id"])
            require(key not in receiver,
                    f"{cell_dir.name}: duplicate message id {key}")
            receiver[key] = row

    traces = load_ue_traces(cell_dir)
    backlog, mcs = traces["backlog"], traces["mcs"]
    pdcp, rlc_in, rlc_out = traces["pdcp"], traces["rlc_in"], traces["rlc_out"]

    order = sorted(sender)
    frames: list[dict[str, Any]] = []
    for position, frame_index in enumerate(order):
        tx = sender[frame_index]
        first_send = int(tx["first_send_monotonic_ns"])
        last_send = int(tx["last_send_monotonic_ns"])
        # The ingress window runs from this frame's first handoff to the next
        # frame's first handoff; the final frame closes on its own last handoff
        # plus one nominal period.
        if position + 1 < len(order):
            window_end = int(sender[order[position + 1]]
                             ["first_send_monotonic_ns"])
        else:
            window_end = last_send + C.STEP_PERIOD_NS

        pre_backlog, backlog_age = backlog.latest_strictly_before(
            first_send, max_age_ns=C.FRESHNESS_MAX_AGE_NS)
        prior_mcs, mcs_age = mcs.latest_strictly_before(
            first_send, max_age_ns=C.FRESHNESS_MAX_AGE_NS)
        post_backlog, _ = backlog.latest_strictly_before(
            window_end, max_age_ns=C.FRESHNESS_MAX_AGE_NS)

        rx = receiver.get(frame_index)
        if rx is None:
            complete = False
            datagrams_received = 0
            transport_ns: int | None = None
            first_arrival = last_arrival = None
        else:
            complete = rx["complete"] == "True"
            datagrams_received = int(rx["datagrams_received"])
            first_arrival = int(rx["first_datagram_monotonic_ns"])
            last_arrival = int(rx["last_datagram_monotonic_ns"])
            transport_ns = (
                int(rx["complete_reassembly_monotonic_ns"]) - last_send
                if complete else None)

        frames.append({
            "cell_id": record["cell_id"], "cell_tag": record["cell_tag"],
            "partition": record["partition"],
            "profile_id_audit_only": record["profile_id"],
            "frame_index": frame_index, "block_index": int(tx["block_index"]),
            "tier_audit_only": tx["tier"], "action_id": int(tx["action_id"]),
            "mode_id": int(tx["mode_id"]), "q_e4": int(tx["q_e4"]),
            "row_sha256": tx["row_sha256"],
            "total_transmitted_bytes": int(tx["total_transmitted_bytes"]),
            "datagram_count": int(tx["datagram_count"]),
            "udp_application_bytes":
                int(tx["udp_application_bytes_handed_to_socket"]),
            "first_send_monotonic_ns": first_send,
            "last_send_monotonic_ns": last_send,
            "send_span_ns": last_send - first_send,
            "schedule_lag_ms": float(tx["schedule_lag_ms"]),
            "pre_action_rlc_backlog_bytes": pre_backlog,
            "pre_action_rlc_backlog_age_ns": backlog_age,
            "prior_ul_mcs": prior_mcs,
            "prior_ul_mcs_age_ns": mcs_age,
            "successor_rlc_backlog_bytes": post_backlog,
            "pdcp_ingress_bytes": pdcp.sum_in(first_send, window_end),
            "pdcp_ingress_events": pdcp.count_in(first_send, window_end),
            "rlc_ingress_bytes": rlc_in.sum_in(first_send, window_end),
            "rlc_service_bytes": rlc_out.sum_in(first_send, window_end),
            "datagrams_received": datagrams_received,
            "complete": complete,
            "first_arrival_monotonic_ns": first_arrival,
            "last_arrival_monotonic_ns": last_arrival,
            "transport_latency_ns": transport_ns,
            "terminal_outcome": classify_terminal(
                complete=complete, datagrams_received=datagrams_received,
                transport_ns=transport_ns),
            "window_end_monotonic_ns": window_end,
        })

    # Every sent frame carries exactly one terminal outcome.
    require(len(frames) == C.FRAMES_PER_CELL, "frame table row count drifted")
    require(all(row["terminal_outcome"] in C.TERMINAL_OUTCOMES
                for row in frames), "unregistered terminal outcome")

    cycles = []
    by_index = {row["frame_index"]: row for row in frames}
    for start, held, successor in C.primary_cycle_indices():
        decision, hold, nxt = by_index[start], by_index[held], by_index[successor]
        require(decision["action_id"] == hold["action_id"]
                and decision["q_e4"] == hold["q_e4"],
                f"held frame {held} does not reuse the decision action")
        service = decision["rlc_service_bytes"] + hold["rlc_service_bytes"]
        ingress = decision["rlc_ingress_bytes"] + hold["rlc_ingress_bytes"]
        cycles.append({
            "cell_id": decision["cell_id"], "cell_tag": decision["cell_tag"],
            "partition": decision["partition"],
            "profile_id_audit_only": decision["profile_id_audit_only"],
            "cycle_start_index": start,
            "block_index": decision["block_index"],
            "tier_audit_only": decision["tier_audit_only"],
            "action_id": decision["action_id"], "mode_id": decision["mode_id"],
            "q_e4": decision["q_e4"],
            "pre_action_rlc_backlog_bytes":
                decision["pre_action_rlc_backlog_bytes"],
            "prior_ul_mcs": decision["prior_ul_mcs"],
            "decision_frame_total_transmitted_bytes":
                decision["total_transmitted_bytes"],
            "held_frame_total_transmitted_bytes":
                hold["total_transmitted_bytes"],
            "pair_total_transmitted_bytes":
                decision["total_transmitted_bytes"]
                + hold["total_transmitted_bytes"],
            "measured_rlc_ingress_bytes": ingress,
            "measured_rlc_service_bytes": service,
            "observed_next_backlog_bytes":
                nxt["pre_action_rlc_backlog_bytes"],
            "conserved_next_backlog_bytes": (
                C.queue_next_backlog(
                    decision["pre_action_rlc_backlog_bytes"], ingress, service)
                if decision["pre_action_rlc_backlog_bytes"] is not None
                else None),
            "decision_transport_latency_ns": decision["transport_latency_ns"],
            "decision_terminal_outcome": decision["terminal_outcome"],
            "held_terminal_outcome": hold["terminal_outcome"],
        })

    unclosed = [index for index in C.unclosed_decision_indices()]
    return {
        "closure_seal": closure_seal(cell_dir, frames),
        "cell_record": {
            "cell_id": record["cell_id"], "cell_tag": record["cell_tag"],
            "partition": record["partition"],
            "profile_id_audit_only": record["profile_id"],
            "scene_split": record["scene_split"],
        },
        "clock_bridge": traces["bridge"].to_json(),
        "trace_counts": traces["counts"],
        "mcs_rows_rejected_non_causal_class":
            traces["mcs_rows_rejected_non_causal_class"],
        "frames": frames, "cycles": cycles,
        "unclosed_decision_indices": unclosed,
    }


FRAME_FIELDS = (
    "cell_id", "cell_tag", "partition", "profile_id_audit_only", "frame_index",
    "block_index", "tier_audit_only", "action_id", "mode_id", "q_e4",
    "row_sha256", "total_transmitted_bytes", "datagram_count",
    "udp_application_bytes", "first_send_monotonic_ns",
    "last_send_monotonic_ns", "send_span_ns", "schedule_lag_ms",
    "pre_action_rlc_backlog_bytes", "pre_action_rlc_backlog_age_ns",
    "prior_ul_mcs", "prior_ul_mcs_age_ns", "successor_rlc_backlog_bytes",
    "pdcp_ingress_bytes", "pdcp_ingress_events", "rlc_ingress_bytes",
    "rlc_service_bytes", "datagrams_received", "complete",
    "first_arrival_monotonic_ns", "last_arrival_monotonic_ns",
    "transport_latency_ns", "terminal_outcome", "window_end_monotonic_ns",
)

CYCLE_FIELDS = (
    "cell_id", "cell_tag", "partition", "profile_id_audit_only",
    "cycle_start_index", "block_index", "tier_audit_only", "action_id",
    "mode_id", "q_e4", "pre_action_rlc_backlog_bytes", "prior_ul_mcs",
    "decision_frame_total_transmitted_bytes",
    "held_frame_total_transmitted_bytes", "pair_total_transmitted_bytes",
    "measured_rlc_ingress_bytes", "measured_rlc_service_bytes",
    "observed_next_backlog_bytes", "conserved_next_backlog_bytes",
    "decision_transport_latency_ns", "decision_terminal_outcome",
    "held_terminal_outcome",
)


def parse_campaign(capture_root: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), "parser output is create-only")
    cell_dirs = sorted((capture_root / "cells").iterdir())
    require(len(cell_dirs) == C.EXPECTED_CELLS,
            f"expected {C.EXPECTED_CELLS} cells, found {len(cell_dirs)}")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_frames: list[dict[str, Any]] = []
    all_cycles: list[dict[str, Any]] = []
    cell_reports: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        parsed = parse_cell(cell_dir)
        all_frames.extend(parsed["frames"])
        all_cycles.extend(parsed["cycles"])
        bridge = parsed["clock_bridge"]
        outcomes: dict[str, int] = {}
        for row in parsed["frames"]:
            outcomes[row["terminal_outcome"]] = (
                outcomes.get(row["terminal_outcome"], 0) + 1)
        cell_reports.append({
            **parsed["cell_record"], "clock_bridge": bridge,
            "trace_counts": parsed["trace_counts"],
            "frames": len(parsed["frames"]), "cycles": len(parsed["cycles"]),
            "terminal_outcomes": outcomes,
            "unclosed_decision_indices": parsed["unclosed_decision_indices"],
            "mcs_rows_rejected_non_causal_class":
                parsed["mcs_rows_rejected_non_causal_class"],
            "closure_seal": parsed["closure_seal"],
        })

    frames_path = output_dir / "frames.csv"
    with frames_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        writer.writerows(all_frames)
    cycles_path = output_dir / "cycles.csv"
    with cycles_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CYCLE_FIELDS))
        writer.writeheader()
        writer.writerows(all_cycles)

    outcome_totals: dict[str, int] = {}
    for row in all_frames:
        outcome_totals[row["terminal_outcome"]] = (
            outcome_totals.get(row["terminal_outcome"], 0) + 1)
    report = {
        "schema": PARSE_SCHEMA,
        "contract_sha256": C.CONTRACT_SHA256,
        "capture_root": str(capture_root),
        "cells": cell_reports,
        "totals": {
            "frames": len(all_frames), "cycles": len(all_cycles),
            "expected_frames": C.EXPECTED_RAW_FRAMES,
            "expected_cycles": C.EXPECTED_PRIMARY_CYCLES,
            "terminal_outcomes": outcome_totals,
        },
        "frames_csv_sha256": C.sha256_file(frames_path),
        "cycles_csv_sha256": C.sha256_file(cycles_path),
    }
    with (output_dir / "PARSE_REPORT.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = parse_campaign(Path(args.capture_root), Path(args.output_dir))
    json.dump(report["totals"], sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
