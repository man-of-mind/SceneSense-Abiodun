#!/usr/bin/env python3
"""Durable, denominator-explicit analysis of the bounded quality probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class QualityAnalysisError(RuntimeError):
    pass


BOUNDARIES = (
    ("capture", "capture_timestamp_ns"),
    ("sensor_ready", "sensor_ready_wall_ns"),
    ("model_action_start", "model_action_start_wall_ns"),
    ("model_prepare_start", "model_prepare_start_wall_ns"),
    ("first_feature_datagram_send", "first_feature_datagram_send_wall_ns"),
    ("final_prediction_ready", "final_prediction_ready_wall_ns"),
    ("evaluation_enqueued", "evaluation_enqueued_wall_ns"),
    ("evaluation_started", "evaluation_started_wall_ns"),
    ("evaluation_completed", "evaluation_completed_wall_ns"),
    ("ack_emit_start", "ack_emit_start_wall_ns"),
    ("edge_socket_send_call", "edge_socket_send_call_wall_ns"),
)
STAGE_INTERVALS = (
    (
        "model_ready_to_final_prediction",
        "model_ready_wall_ns",
        "final_prediction_ready_wall_ns",
    ),
    (
        "evaluation_queue_wait",
        "evaluation_enqueued_wall_ns",
        "evaluation_started_wall_ns",
    ),
    (
        "quality_evaluation_compute",
        "evaluation_started_wall_ns",
        "evaluation_completed_wall_ns",
    ),
    (
        "final_prediction_to_quality_complete",
        "final_prediction_ready_wall_ns",
        "evaluation_completed_wall_ns",
    ),
    (
        "quality_complete_to_socket_send",
        "evaluation_completed_wall_ns",
        "edge_socket_send_call_wall_ns",
    ),
    (
        "quality_ack_downlink",
        "edge_socket_send_call_wall_ns",
        "quality_ack_received_wall_ns",
    ),
)
FPS_PERIOD_MS = {"8_fps": 125.0, "9_fps": 1000.0 / 9.0, "10_fps": 100.0}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualityAnalysisError(message)


def _integer(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "quantile requested for empty population")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float], *, denominator: int) -> dict[str, Any]:
    population = [float(value) for value in values]
    if not population:
        return {
            "available": 0,
            "denominator_sent": int(denominator),
            "coverage": 0.0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "within_140ms_count": 0,
            "within_140ms_fraction_of_available": None,
            "within_140ms_fraction_of_sent": 0.0,
        }
    within = sum(value <= 140.0 for value in population)
    return {
        "available": len(population),
        "denominator_sent": int(denominator),
        "coverage": len(population) / int(denominator),
        "p50_ms": _quantile(population, 0.50),
        "p95_ms": _quantile(population, 0.95),
        "p99_ms": _quantile(population, 0.99),
        "max_ms": max(population),
        "within_140ms_count": within,
        "within_140ms_fraction_of_available": within / len(population),
        "within_140ms_fraction_of_sent": within / int(denominator),
    }


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["run_id"]), str(row["cell_id"]), str(row["stream_id"]),
        int(row["frame_id"]), int(row["action_id"]), str(row["profile_id"]),
        int(row["capture_timestamp_ns"]),
    )


def analyze_attempt(attempt_dir: Path) -> dict[str, Any]:
    attempt = Path(attempt_dir)
    timing_path = attempt / "quality_policy_timing.csv"
    ledger_path = attempt / "quality_feedback.csv"
    edge_path = attempt / "direct_edge_map/quality_edge_report.json"
    edge_counters_path = attempt / "direct_edge_map/direct_edge_counters.json"
    packet_path = attempt / "quality_ack_packet_evidence.json"
    for path in (
        timing_path,
        ledger_path,
        edge_path,
        edge_counters_path,
        packet_path,
    ):
        _require(path.is_file(), f"analysis input is absent: {path}")

    with timing_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        ledger = list(csv.DictReader(handle))
    edge = json.loads(edge_path.read_text(encoding="utf-8"))
    edge_counters = json.loads(edge_counters_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    sent = len(rows)
    _require(sent > 0, "quality timing population is empty")
    _require(len({_identity(row) for row in rows}) == sent, "duplicate timing identity")
    edge_message_by_identity = {
        tuple(message["identity"]): message for message in edge.get("messages", [])
    }
    _require(
        len(edge_message_by_identity) == len(edge.get("messages", [])),
        "duplicate edge quality-message identity",
    )
    for row in rows:
        message = edge_message_by_identity.get(_identity(row))
        if message is not None:
            row["edge_socket_send_call_wall_ns"] = message[
                "socket_send_call_wall_ns"
            ]

    outcomes = {
        "sent": sent,
        "feature_messages_reassembled": int(
            (edge_counters.get("counters") or {}).get(
                "feature_messages_reassembled", 0
            )
        ),
        "final_prediction_eligible": sum(
            _integer(row.get("final_prediction_ready_wall_ns")) is not None
            for row in rows
        ),
        "quality_evaluated": sum(
            row.get("quality_feedback_event") == "QUALITY_EVALUATED" for row in rows
        ),
        "quality_evaluation_failed": sum(
            row.get("quality_feedback_event") == "QUALITY_EVALUATION_FAILED"
            for row in rows
        ),
        "not_eligible_no_final_prediction": sum(
            row.get("quality_feedback_event") in ("", None) for row in rows
        ),
        "quality_ack_received": sum(
            _integer(row.get("quality_ack_received_wall_ns")) is not None
            for row in rows
        ),
        "ue_ledger_rows_all_dispositions": len(ledger),
        "ue_quality_ack_rows": sum(bool(row.get("message_sha256")) for row in ledger),
        "edge_quality_messages": len(edge.get("messages", [])),
        "oaitun_quality_packets": int(packet["quality_ack_packets"]),
    }
    _require(
        outcomes["quality_evaluated"] + outcomes["quality_evaluation_failed"]
        == outcomes["quality_ack_received"],
        "quality ACK outcome accounting does not reconcile",
    )
    _require(
        outcomes["quality_ack_received"]
        + outcomes["not_eligible_no_final_prediction"] == sent
        and outcomes["final_prediction_eligible"]
        == outcomes["quality_ack_received"]
        and outcomes["ue_ledger_rows_all_dispositions"] == sent,
        "sent/eligible/not-eligible quality denominators do not reconcile",
    )
    _require(
        outcomes["quality_ack_received"] == outcomes["ue_quality_ack_rows"]
        == outcomes["edge_quality_messages"] == outcomes["oaitun_quality_packets"],
        "edge/packet/UE quality populations do not reconcile",
    )

    intervals: dict[str, dict[str, Any]] = {}
    per_frame: list[dict[str, Any]] = []
    for row in rows:
        ack = _integer(row.get("quality_ack_received_wall_ns"))
        joined: dict[str, Any] = {
            "frame_id": int(row["frame_id"]),
            "action_id": int(row["action_id"]),
            "profile_id": row["profile_id"],
            "quality_feedback_event": row.get("quality_feedback_event") or "NOT_ELIGIBLE",
        }
        for label, field in BOUNDARIES:
            start = _integer(row.get(field))
            duration = None if start is None or ack is None else (ack - start) / 1e6
            _require(duration is None or duration >= 0.0, f"negative {label}->ACK interval")
            joined[f"{label}_to_ue_receive_ms"] = duration
        per_frame.append(joined)

    stages: dict[str, dict[str, Any]] = {}
    for label, start_field, end_field in STAGE_INTERVALS:
        values: list[float] = []
        for row in rows:
            start, end = _integer(row.get(start_field)), _integer(row.get(end_field))
            if start is None or end is None:
                continue
            duration = (end - start) / 1e6
            _require(duration >= 0.0, f"negative {label} interval")
            values.append(duration)
        stages[label] = summarize(values, denominator=sent)

    for label, _ in BOUNDARIES:
        field = f"{label}_to_ue_receive_ms"
        values = [float(row[field]) for row in per_frame if row[field] is not None]
        summary = summarize(values, denominator=sent)
        summary["feedback_before_next_decision"] = {
            name: {
                "period_ms": period,
                "count": sum(value <= period for value in values),
                "fraction_of_available": (
                    sum(value <= period for value in values) / len(values)
                    if values else None
                ),
                "fraction_of_sent": sum(value <= period for value in values) / sent,
            }
            for name, period in FPS_PERIOD_MS.items()
        }
        intervals[f"{label}_to_ue_receive"] = summary

    joined_path = attempt / "quality_feedback_timing_join.csv"
    fields = list(per_frame[0])
    with joined_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)
    result = {
        "schema": "scenesense.quality_feedback_timing_analysis.v1",
        "status": "PASS",
        "clock_domain": "time.time_ns shared host wall clock",
        "population_denominators": outcomes,
        "intervals": intervals,
        "stage_intervals": stages,
        "fps_periods_ms": FPS_PERIOD_MS,
        "claims": {
            "full_route_b_completed": False,
            "quality_is_privileged_carla_ground_truth": True,
            "quality_feedback_is_deployable": False,
            "packet_path_proven_on_oaitun_ue1": True,
        },
        "joined_csv_sha256": hashlib.sha256(joined_path.read_bytes()).hexdigest(),
    }
    result_path = attempt / "QUALITY_FEEDBACK_ANALYSIS.json"
    temporary = result_path.with_name(result_path.name + ".partial")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.link(temporary, result_path)
    temporary.unlink()

    report = attempt / "QUALITY_FEEDBACK_REPORT.md"
    lines = [
        "# Bounded exact-quality feedback probe",
        "",
        f"Sent frames: **{sent}**. Quality ACK received: "
        f"**{outcomes['quality_ack_received']}**. The denominators are not conflated.",
        "",
        "| Boundary to UE receipt | n | p50 (ms) | p95 (ms) | p99 (ms) | <=140 ms / sent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in intervals.items():
        lines.append(
            f"| {name.replace('_', ' ')} | {item['available']} | "
            f"{item['p50_ms'] if item['p50_ms'] is not None else 'NA'} | "
            f"{item['p95_ms'] if item['p95_ms'] is not None else 'NA'} | "
            f"{item['p99_ms'] if item['p99_ms'] is not None else 'NA'} | "
            f"{item['within_140ms_count']}/{sent} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
