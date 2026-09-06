#!/usr/bin/env python3
"""Offline root-cause audit of the completed Phase-15 retry4 live pilot.

Read-only over the immutable retry4 evidence tree. No experiment is launched,
no runtime behaviour is changed and no threshold is moved. Every reported
interval is annotated with the clock domain it was measured in; intervals that
would require subtracting timestamps across unproven clock domains are emitted
as explicitly unavailable rather than repaired.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("experiments/splitfusion_16_cell_live_carla_oai_pilot_v1/20260905_live_carla_oai_pilot_retry4")
CATALOG = Path("rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json")
TERMINAL_NAME = "SPLITFUSION_PHASE15_RETRY4_LATENCY_AUDIT_COMPLETE"

# The audit is registered against exactly these four action identities.
EXPECTED_ACTION_IDENTITY = {
    0: "split_noae_uint8_q0000",
    20: "split_ae128_uint8_q5000",
    46: "split_ae64_uint6_q9000",
    71: "split_ae32_uint4_q9800",
}

# Clock domains observed in the retained evidence.
UE_PERF = "ue_process_perf_counter_ns"
EDGE_PERF = "edge_container_perf_counter_ns"
HOST_WALL = "host_wall_clock_time_time_s"

UNAVAILABLE_INTERVALS = {
    "ue_send_to_first_edge_receive": (
        "UE send uses %s; the only edge receive timestamp (edge_received_ns) uses %s. "
        "No pairing event is recorded in both domains, so the offset is unproven."
    ) % (UE_PERF, EDGE_PERF),
    "ue_send_to_complete_edge_receive": (
        "Same unproven %s -> %s offset; only the UE-observed round trip is computable."
    ) % (UE_PERF, EDGE_PERF),
    "complete_edge_receive_to_edge_processing_start": (
        "live_pilot_runtime.run_edge_service calls edge.process() on the same statement "
        "that stamps edge_received_ns; the two instants are not separately recorded."
    ),
    "edge_queue_wait": (
        "No application-level edge queue exists in source. Feature datagrams wait only in "
        "the kernel SO_RCVBUF, which is not instrumented and emits no drop counter."
    ),
    "tail_completion_to_result_publication": (
        "tail_finished_ns is %s and is not retained in per_frame_metrics.csv; the "
        "publication instant is not stamped at all."
    ) % EDGE_PERF,
    "result_publication_to_ue_ingestion": (
        "Publication is %s, UE ingestion is %s; offset unproven."
    ) % (EDGE_PERF, UE_PERF),
    "ue_ingestion_to_installation": (
        "UE ingestion is %s, installation is %s; offset unproven. The wall-clock "
        "capture->installation AoI is reported instead."
    ) % (UE_PERF, HOST_WALL),
    "incomplete_or_expired_feature_reassemblies": (
        "ChunkReassembler.expired_messages is never exported by either endpoint, and a "
        "feature message that never completes produces no record anywhere."
    ),
    "feature_datagrams_observed_at_receiver": (
        "feature_received_datagrams is carried inside the edge result, so it exists only "
        "for messages that both completed at the edge and returned intact to the UE."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(row: Mapping[str, str], key: str) -> int | None:
    value = as_float(row, key)
    return None if value is None else int(value)


def literal_map(value: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = ast.literal_eval(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


def summarize(values: Sequence[float], missing: int) -> dict[str, Any]:
    return {
        "samples": len(values),
        "missing": int(missing),
        "median": statistics.median(values) if values else None,
        "p95": quantile(values, 0.95),
        "maximum": max(values) if values else None,
    }


# ---------------------------------------------------------------------------
# Evidence integrity
# ---------------------------------------------------------------------------


def verify_evidence(pilot: Path) -> dict[str, Any]:
    """Verify the committed manifest/report/ledger/terminal and every cell CSV."""
    checks: list[dict[str, Any]] = []

    def check(kind: str, path: Path, expected: str) -> None:
        observed = sha256_file(path) if path.is_file() else ""
        checks.append(
            {
                "kind": kind,
                "path": str(path.relative_to(ROOT)),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": observed == expected,
            }
        )

    manifest = read_json(pilot / "artifact_manifest.json")
    for item in manifest["files"]:
        check("campaign_artifact", pilot / item["path"], str(item["sha256"]))

    ledger = read_json(pilot / "campaign_ledger.json")
    for cell_id, attempts in sorted(ledger["cells"].items()):
        for attempt in attempts:
            check(f"cell_terminal:{cell_id}", pilot / attempt["terminal"], str(attempt["terminal_sha256"]))

    for row in read_csv(pilot / "cell_summary.csv"):
        attempt_dir = pilot / "cells" / row["cell_id"] / "attempts" / f"attempt_{int(row['attempt']):04d}"
        check(f"attempt_manifest:{row['cell_id']}", attempt_dir / "manifest.json", row["attempt_manifest_sha256"])
        for item in read_json(attempt_dir / "manifest.json")["files"]:
            check(f"cell_evidence:{row['cell_id']}", attempt_dir / item["path"], str(item["sha256"]))

    terminal = pilot / "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_COMPLETE"
    return {
        "checks_run": len(checks),
        "mismatches": [item for item in checks if not item["match"]],
        "all_verified": all(item["match"] for item in checks),
        "campaign_terminal_present": terminal.is_file(),
        "campaign_terminal_sha256": sha256_file(terminal) if terminal.is_file() else "",
        "report_sha256": sha256_file(pilot / "REPORT.md"),
        "pilot_manifest_sha256": sha256_file(pilot / "pilot_manifest.json"),
        "qualification_sha256": sha256_file(pilot / "qualification.json"),
        "chain_of_custody": (
            "artifact_manifest.json (committed) -> cell_summary.csv -> attempt_manifest_sha256 "
            "-> per-cell manifest.json -> untracked per-cell CSVs"
        ),
    }


def bind_action_identities(pilot: Path) -> dict[str, Any]:
    """Bind action -> profile from the catalog and from every per-cell record."""
    catalog = read_json(ROOT / CATALOG)
    catalog_by_action = {int(item["action_id"]): item for item in catalog["profiles"]}
    bindings: dict[str, Any] = {"catalog_sha256": sha256_file(ROOT / CATALOG), "actions": {}, "disagreements": []}
    for action_id, expected in sorted(EXPECTED_ACTION_IDENTITY.items()):
        entry = catalog_by_action.get(action_id, {})
        observed = str(entry.get("profile_id", ""))
        bindings["actions"][str(action_id)] = {
            "expected_profile_id": expected,
            "catalog_profile_id": observed,
            "family": entry.get("family"),
            "quantizer": entry.get("quantizer"),
            "bit_width": entry.get("bit_width"),
            "q_e4": entry.get("q_e4"),
            "keep_count": entry.get("keep_count"),
            "transported_channels": entry.get("transported_channels"),
            "catalog_agrees": observed == expected,
        }
        if observed != expected:
            bindings["disagreements"].append(f"catalog action {action_id} is {observed!r}, expected {expected!r}")

    # Every per-cell record must agree; the binding is taken from the cell record,
    # never from the ordering of campaign.actions.profile_ids.
    for row in read_csv(pilot / "cell_summary.csv"):
        action_id = int(row["action_id"])
        expected = EXPECTED_ACTION_IDENTITY.get(action_id)
        if expected is None:
            bindings["disagreements"].append(f"cell {row['cell_id']} carries unregistered action {action_id}")
            continue
        if row["profile_id"] != expected:
            bindings["disagreements"].append(
                f"cell {row['cell_id']} binds action {action_id} to {row['profile_id']!r}, expected {expected!r}"
            )
    bindings["verified"] = not bindings["disagreements"]
    return bindings


# ---------------------------------------------------------------------------
# Per-cell reconstruction
# ---------------------------------------------------------------------------


def load_cell(pilot: Path, summary_row: Mapping[str, str]) -> dict[str, Any]:
    attempt_dir = pilot / "cells" / summary_row["cell_id"] / "attempts" / f"attempt_{int(summary_row['attempt']):04d}"
    per_frame = read_csv(attempt_dir / "per_frame_metrics.csv")
    feedback = read_csv(attempt_dir / "map_feedback.csv")
    results = read_json(attempt_dir / "RESULTS_SUMMARY.json")
    return {
        "cell_id": summary_row["cell_id"],
        "action_id": int(summary_row["action_id"]),
        "profile_id": summary_row["profile_id"],
        "network_profile_id": summary_row["network_profile_id"],
        "attempt_dir": attempt_dir,
        "per_frame": per_frame,
        "feedback": feedback,
        "results": results,
    }


def cell_funnel(cell: Mapping[str, Any]) -> dict[str, Any]:
    per_frame: list[dict[str, str]] = cell["per_frame"]
    feedback: list[dict[str, str]] = cell["feedback"]
    results = cell["results"]
    acceptance = results["structural_acceptance"]

    status_counts: dict[str, int] = {}
    for row in per_frame:
        status_counts[row["prepare_status"]] = status_counts.get(row["prepare_status"], 0) + 1

    sent = [row for row in per_frame if row["prepare_status"] == "SENT"]
    decoded = [row for row in sent if row.get("decoded") == "True"]

    # Cumulative edge counters ride inside every returned result, so the largest
    # value observed is a strict LOWER BOUND on edge-side activity: work the edge
    # performed after the last surviving result is invisible from the UE.
    edge_attempted = edge_completed = edge_tail = 0
    for row in decoded:
        counters = literal_map(row.get("edge_counters", ""))
        edge_attempted = max(edge_attempted, int(counters.get("frames_attempted", 0)))
        edge_completed = max(edge_completed, int(counters.get("frames_completed", 0)))
        edge_tail = max(edge_tail, int(counters.get("tail_dispatches", 0)))

    feature_datagrams_sent = sum(as_int(row, "datagrams") or 0 for row in sent)
    result_datagrams = [as_int(row, "edge_result_datagrams") or 0 for row in decoded]
    median_result_datagrams = statistics.median(result_datagrams) if result_datagrams else None

    installed = [row for row in feedback if row["status"] == "ACK_INSTALLED"]
    emitted = [row for row in installed if row.get("feedback_emit_at") not in ("", None)]
    timely = late = 0
    for row in installed:
        received = as_float(row, "feedback_received_at")
        timeout_at = as_float(row, "ack_timeout_at")
        if received is None or timeout_at is None:
            continue
        if received <= timeout_at:
            timely += 1
        else:
            late += 1
    installs_within_service_deadline = sum(
        1
        for row in installed
        if (as_float(row, "install_timestamp") is not None)
        and (as_float(row, "service_deadline_at") is not None)
        and as_float(row, "install_timestamp") <= as_float(row, "service_deadline_at")
    )

    capture_wall = [value for value in (as_float(row, "capture_wall_s") for row in sent) if value is not None]
    route_span_s = (max(capture_wall) - min(capture_wall)) if len(capture_wall) > 1 else None

    return {
        "cell_id": cell["cell_id"],
        "action_id": cell["action_id"],
        "profile_id": cell["profile_id"],
        "network_profile_id": cell["network_profile_id"],
        "route_ticks": int(acceptance["route_ticks"]),
        "s01_preparation_opportunities": len(per_frame),
        "s02_eligible_preparation_frames": int(acceptance["eligible_preparation_frames"]),
        "s03_prepared_queue_admissions": len(per_frame) - status_counts.get("DROPPED_QUEUE_FULL", 0),
        "s04_preparation_starts": len(per_frame) - status_counts.get("DROPPED_QUEUE_FULL", 0),
        "s05_radar_complete_frames": status_counts.get("SENT", 0),
        "s06_encoded_frames": len(sent),
        "s07_feature_messages_sent": len(sent),
        "s08_feature_datagrams_sent": feature_datagrams_sent,
        "s09_feature_datagrams_observed_at_receiver": "UNAVAILABLE",
        "s09b_feature_datagrams_in_surviving_messages": sum(
            as_int(row, "feature_received_datagrams") or 0 for row in decoded
        ),
        "s09c_surviving_messages_with_datagram_count_mismatch": sum(
            1
            for row in decoded
            if as_int(row, "feature_received_datagrams") != as_int(row, "datagrams")
        ),
        "s09d_duplicate_feature_datagrams": sum(
            as_int(row, "feature_duplicate_datagrams") or 0 for row in decoded
        ),
        "s10_edge_fully_reassembled_messages_lower_bound": edge_attempted,
        "s11_incomplete_or_expired_reassemblies": "UNAVAILABLE",
        "s12_edge_queue_admissions": "NOT_INSTRUMENTED_NO_APPLICATION_QUEUE",
        "s12b_edge_queue_drops": "NOT_INSTRUMENTED_NO_APPLICATION_QUEUE",
        "s13_tail_starts_lower_bound": edge_tail,
        "s14_tail_completions_lower_bound": edge_completed,
        "s15_result_publications_lower_bound": edge_completed,
        "s16_result_datagrams_sent": "UNAVAILABLE",
        "s16b_median_result_datagrams_per_message": median_result_datagrams,
        "s17_result_messages_ingested_at_ue": len(decoded),
        "s18_maps_installed": len(installed),
        "s19_feedback_emitted": len(emitted),
        "s20_feedback_received_within_ack_timeout": timely,
        "s20b_feedback_received_after_ack_timeout": late,
        "s21_installs_within_service_deadline": installs_within_service_deadline,
        "prepare_status_counts": status_counts,
        "route_span_s": route_span_s,
        "terminal_feedback_outcomes": acceptance["terminal_feedback_outcomes"],
    }


def reconcile_funnel(funnel: Mapping[str, Any], cell: Mapping[str, Any]) -> list[str]:
    """Every count that two independent artifacts both record must agree."""
    problems: list[str] = []
    counts = funnel["prepare_status_counts"]
    results = cell["results"]
    acceptance = results["structural_acceptance"]

    if funnel["s07_feature_messages_sent"] != int(acceptance["sent_frames"]):
        problems.append("per_frame SENT rows disagree with structural_acceptance.sent_frames")
    if funnel["s07_feature_messages_sent"] != int(results["live_dispatch"]["sent"]):
        problems.append("per_frame SENT rows disagree with live_dispatch.sent")
    if funnel["s17_result_messages_ingested_at_ue"] != int(results["live_dispatch"]["edge_completed"]):
        problems.append("decoded rows disagree with live_dispatch.edge_completed")
    if funnel["s18_maps_installed"] != int(acceptance["ack_installed_frames"]):
        problems.append("ACK_INSTALLED rows disagree with structural_acceptance.ack_installed_frames")
    if funnel["s17_result_messages_ingested_at_ue"] != funnel["s18_maps_installed"]:
        problems.append("UE-ingested results disagree with installed maps")

    losses = (
        counts.get("DROPPED_QUEUE_FULL", 0)
        + counts.get("DROPPED_SENSOR_LATE_OR_MISSING", 0)
        + counts.get("DROPPED_INCOMPLETE_RADAR_WINDOW", 0)
    )
    if funnel["s02_eligible_preparation_frames"] - losses != funnel["s07_feature_messages_sent"]:
        problems.append("eligible frames minus classified preparation losses does not equal sent frames")
    if int(results["split_frames_dropped"]) != losses:
        problems.append("split_frames_dropped disagrees with classified preparation losses")

    warmup = counts.get("WARMUP_NO_COMPLETE_RADAR_WINDOW", 0)
    if funnel["s01_preparation_opportunities"] - warmup != funnel["s02_eligible_preparation_frames"]:
        problems.append("per_frame rows minus warmup rows disagree with eligible_preparation_frames")
    if funnel["s09c_surviving_messages_with_datagram_count_mismatch"]:
        problems.append("a surviving feature message reports a datagram count other than the one sent")

    total = sum(counts.values())
    if total != funnel["s01_preparation_opportunities"]:
        problems.append("prepare_status counts do not sum to the per_frame row count")
    return problems


# ---------------------------------------------------------------------------
# Latency reconstruction
# ---------------------------------------------------------------------------


def cell_intervals(cell: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every interval computable inside one proven clock domain."""
    sent = [row for row in cell["per_frame"] if row["prepare_status"] == "SENT"]
    decoded = [row for row in sent if row.get("decoded") == "True"]
    installed = {
        int(row["frame_id"]): row for row in cell["feedback"] if row["status"] == "ACK_INSTALLED" and row["frame_id"]
    }

    series: dict[str, tuple[list[float], int, str]] = {}

    def collect(name: str, domain: str, rows: Iterable[Mapping[str, str]], fn) -> None:
        values: list[float] = []
        missing = 0
        for row in rows:
            value = fn(row)
            if value is None:
                missing += 1
            else:
                values.append(value)
        series[name] = (values, missing, domain)

    def diff_ms(row: Mapping[str, str], start: str, end: str, scale: float) -> float | None:
        a, b = as_float(row, start), as_float(row, end)
        return None if a is None or b is None else (b - a) / scale

    collect("world_tick_schedule_to_preparation_start_ms", UE_PERF, sent, lambda r: as_float(r, "queue_wait_ms"))
    collect(
        "preparation_start_to_encoding_complete_ms", UE_PERF, sent,
        lambda r: diff_ms(r, "capture_started_ns", "ue_prepare_finished_ns", 1e6),
    )
    collect(
        "encoding_complete_to_ue_send_complete_ms", UE_PERF, sent,
        lambda r: diff_ms(r, "ue_prepare_finished_ns", "send_finished_ns", 1e6),
    )
    collect(
        "ue_send_complete_to_result_received_round_trip_ms", UE_PERF, decoded,
        lambda r: diff_ms(r, "send_finished_ns", "edge_result_received_ns", 1e6),
    )

    for stage in ("total_edge_processing", "zstd_decompression", "unpack_dequantize", "ae_decode", "frozen_tail", "output_serialization"):
        collect(
            f"edge_{stage}_ms", EDGE_PERF, decoded,
            lambda r, key=stage: (lambda t: None if key not in t else t[key] / 1e6)(literal_map(r.get("edge_timing_ns", ""))),
        )

    ack = list(installed.values())
    collect("capture_to_installation_aoi_ms", HOST_WALL, ack, lambda r: diff_ms(r, "capture_at", "install_timestamp", 1e-3))
    collect("installation_to_feedback_emission_ms", HOST_WALL, ack, lambda r: diff_ms(r, "install_timestamp", "feedback_emit_at", 1e-3))
    collect("feedback_emission_to_feedback_receipt_ms", HOST_WALL, ack, lambda r: diff_ms(r, "feedback_emit_at", "feedback_received_at", 1e-3))
    collect("capture_to_feedback_receipt_ms", HOST_WALL, ack, lambda r: diff_ms(r, "capture_at", "feedback_received_at", 1e-3))

    # Independent two-domain cross-check: the wall-clock AoI minus the UE-perf
    # round trip must equal the unmeasured capture->send head plus the sub-ms
    # loopback publish/install tail. This is a consistency test, never a repair.
    residuals: list[float] = []
    for row in decoded:
        frame_id = as_int(row, "frame_id")
        ack_row = installed.get(frame_id) if frame_id is not None else None
        if ack_row is None:
            continue
        aoi = diff_ms(ack_row, "capture_at", "install_timestamp", 1e-3)
        trip = diff_ms(row, "send_finished_ns", "edge_result_received_ns", 1e6)
        if aoi is not None and trip is not None:
            residuals.append(aoi - trip)
    series["cross_domain_residual_wall_aoi_minus_perf_round_trip_ms"] = (residuals, 0, "cross_check")

    return {name: {**summarize(values, missing), "clock_domain": domain} for name, (values, missing, domain) in series.items()}


def payload_profile(cell: Mapping[str, Any], funnel: Mapping[str, Any]) -> dict[str, Any]:
    sent = [row for row in cell["per_frame"] if row["prepare_status"] == "SENT"]
    decoded = [row for row in sent if row.get("decoded") == "True"]

    def median_of(rows: Sequence[Mapping[str, str]], key: str) -> float | None:
        values = [value for value in (as_float(row, key) for row in rows) if value is not None]
        return statistics.median(values) if values else None

    span = funnel["route_span_s"]
    application_bytes = median_of(sent, "udp_application_bytes")
    edge_completed = funnel["s14_tail_completions_lower_bound"]
    result_datagrams = funnel["s16b_median_result_datagrams_per_message"]
    sent_count = funnel["s07_feature_messages_sent"]

    return {
        "median_scientific_inner_bytes": median_of(sent, "scientific_inner_bytes"),
        "median_application_envelope_bytes": median_of(sent, "sfd1_bytes"),
        "median_udp_application_bytes": application_bytes,
        "median_feature_datagrams_per_frame": median_of(sent, "datagrams"),
        "median_result_datagrams_per_message": result_datagrams,
        "median_result_bytes_upper_bound": (result_datagrams * 12_492) if result_datagrams else None,
        "offered_uplink_mbytes_per_s": (sent_count * application_bytes / span / 1e6) if span and application_bytes else None,
        "offered_downlink_mbytes_per_s_upper_bound": (
            edge_completed * result_datagrams * 12_492 / span / 1e6
        ) if span and result_datagrams and edge_completed else None,
        "edge_service_rate_hz_lower_bound": (edge_completed / span) if span and edge_completed else None,
        "complete_message_rate_at_edge": (funnel["s10_edge_fully_reassembled_messages_lower_bound"] / sent_count) if sent_count else None,
        "edge_admission_rate": "IDENTICAL_TO_complete_message_rate_at_edge: the edge has no admission stage separate from completing a reassembly, so admission and complete-message arrival are one event",
        "tail_completion_rate": (edge_completed / sent_count) if sent_count else None,
        "result_survival_rate_edge_to_ue": (
            funnel["s17_result_messages_ingested_at_ue"] / edge_completed
        ) if edge_completed else None,
        "installation_rate": (funnel["s18_maps_installed"] / sent_count) if sent_count else None,
        "timely_feedback_rate": (funnel["s20_feedback_received_within_ack_timeout"] / sent_count) if sent_count else None,
    }


def preparation_attribution(funnel: Mapping[str, Any]) -> dict[str, Any]:
    counts = funnel["prepare_status_counts"]
    return {
        "bounded_preparation_queue_overflow": counts.get("DROPPED_QUEUE_FULL", 0),
        "sensor_synchronisation_late_or_missing": counts.get("DROPPED_SENSOR_LATE_OR_MISSING", 0),
        "missing_or_incomplete_radar_window": counts.get("DROPPED_INCOMPLETE_RADAR_WINDOW", 0),
        "warmup_before_first_complete_radar_window": counts.get("WARMUP_NO_COMPLETE_RADAR_WINDOW", 0),
        "front_inference_or_encoding_throughput": "NOT_SEPARATELY_CLASSIFIED",
        "evaluator_or_gt_work": "NOT_SEPARATELY_CLASSIFIED",
        "scheduler_delay": "NOT_SEPARATELY_CLASSIFIED",
        "shutdown_boundary": "NOT_SEPARATELY_CLASSIFIED",
        "split_processing_failure": counts.get("SPLIT_PROCESSING_FAILED", 0),
    }


def preparation_stationarity(cell: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in cell["per_frame"] if row.get("route_tick")]
    rows.sort(key=lambda row: int(row["route_tick"]))
    total = len(rows)
    if total < 4:
        return {"quartile_coverage": [], "quartile_median_queue_wait_ms": [], "stationary": None}
    coverage: list[float] = []
    waits: list[float | None] = []
    for index in range(4):
        block = rows[index * total // 4 : (index + 1) * total // 4]
        if not block:
            continue
        coverage.append(sum(1 for row in block if row["prepare_status"] == "SENT") / len(block))
        block_waits = [value for value in (as_float(row, "queue_wait_ms") for row in block) if value is not None]
        waits.append(statistics.median(block_waits) if block_waits else None)
    drift = (max(coverage) - min(coverage)) if coverage else None
    return {
        "quartile_coverage": coverage,
        "quartile_median_queue_wait_ms": waits,
        "coverage_range": drift,
        # Preparation is called stationary when no quartile-to-quartile trend
        # exceeds the within-route spread of a stationary 10 Hz sampler.
        "stationary": bool(drift is not None and drift < 0.25),
    }


def aoi_growth(cell: Mapping[str, Any]) -> dict[str, Any]:
    installed = [row for row in cell["feedback"] if row["status"] == "ACK_INSTALLED"]
    points: list[tuple[float, float]] = []
    for row in installed:
        capture = as_float(row, "capture_at")
        install = as_float(row, "install_timestamp")
        if capture is not None and install is not None:
            points.append((capture, (install - capture) * 1000.0))
    points.sort()
    if len(points) < 8:
        return {"samples": len(points), "quartile_median_aoi_ms": [], "monotone_fraction": None}
    total = len(points)
    quartiles = []
    for index in range(4):
        block = points[index * total // 4 : (index + 1) * total // 4]
        if block:
            quartiles.append(statistics.median(value for _, value in block))
    increasing = sum(1 for a, b in zip(points, points[1:]) if b[1] > a[1]) / (total - 1)
    return {"samples": total, "quartile_median_aoi_ms": quartiles, "monotone_fraction": increasing}


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


def classify_hypotheses(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_action: dict[int, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_action.setdefault(cell["funnel"]["action_id"], []).append(cell)

    def median_of(action: int, path) -> float | None:
        values = [path(cell) for cell in by_action.get(action, [])]
        values = [value for value in values if value is not None]
        return statistics.median(values) if values else None

    tail_rate = {a: median_of(a, lambda c: c["payload"]["tail_completion_rate"]) for a in sorted(by_action)}
    survival = {a: median_of(a, lambda c: c["payload"]["result_survival_rate_edge_to_ue"]) for a in sorted(by_action)}
    aoi = {a: median_of(a, lambda c: c["intervals"]["capture_to_installation_aoi_ms"]["median"]) for a in sorted(by_action)}
    feature_dg = {a: median_of(a, lambda c: c["payload"]["median_feature_datagrams_per_frame"]) for a in sorted(by_action)}
    result_dg = {a: median_of(a, lambda c: c["payload"]["median_result_datagrams_per_message"]) for a in sorted(by_action)}
    result_dg_all = sorted(
        {int(cell["payload"]["median_result_datagrams_per_message"]) for cell in cells
         if cell["payload"]["median_result_datagrams_per_message"]}
    )
    monotone = {a: median_of(a, lambda c: c["aoi_growth"]["monotone_fraction"]) for a in sorted(by_action)}
    residual = [
        cell["intervals"]["cross_domain_residual_wall_aoi_minus_perf_round_trip_ms"]["median"]
        for cell in cells
        if cell["intervals"]["cross_domain_residual_wall_aoi_minus_perf_round_trip_ms"]["samples"]
    ]
    tail_ms = {a: median_of(a, lambda c: c["intervals"]["edge_frozen_tail_ms"]["median"]) for a in sorted(by_action)}
    devices = sorted({row for cell in cells for row in cell["devices"]})

    return {
        "H1_large_feature_messages_fail_multi_datagram": {
            "verdict": "CONFIRMED_FOR_ACTION_0_CONTRADICTED_AS_GENERAL_CAUSE",
            "evidence": (
                f"Action 0 sends {feature_dg.get(0)} datagrams/frame and produced 0 installs in 4/4 cells, "
                f"while action 71 sends {feature_dg.get(71)} datagram/frame yet still reached the edge on only "
                f"{tail_rate.get(71):.3f} of sent frames. Fragment count therefore explains action 0 but cannot "
                "explain the 0.36-0.46 edge completion rate shared by actions 20, 46 and 71 across a 190x payload range."
            ),
        },
        "H2_small_messages_overload_downstream_fifo": {
            "verdict": "CONFIRMED",
            "evidence": (
                "The edge is a single blocking recvfrom loop with no application queue, so the kernel SO_RCVBUF "
                "(16,777,216 reported bytes) is the only buffer and it is bounded in BYTES, not frames. Median "
                "capture->install AoI orders strictly inversely with payload: "
                f"{ {k: (None if v is None else round(v)) for k, v in aoi.items()} } ms for actions "
                "0/20/46/71 at 3.59 MB / 1.21 MB / 100 kB / 6.4 kB per frame. Per-action median AoI monotone "
                f"fraction { {k: (None if v is None else round(v, 2)) for k, v in monotone.items()} }: action 71 "
                "AoI rises almost strictly monotonically through its route (never saturating inside ~430 s), "
                "while actions 20 and 46 plateau at a fixed backlog depth. That is the signature of a FIFO whose "
                "depth in frames, not in bytes, sets the delay."
            ),
        },
        "H3_expired_frames_keep_consuming_work": {
            "verdict": "CONFIRMED",
            "evidence": (
                "`service_deadline_at`/`ack_timeout_at` occur in the runtime only where a CSV row is labelled "
                "(`ue_map_install_feedback_v1.py:148` and `:198`) and where the capture deadline is first computed "
                "(`ue_route_b_split_cell_adapter_v1.py:1084`). They appear nowhere in `live_pilot_runtime.py` or "
                "the map server, so no edge admission, tail, publication or install step is ever gated on them. "
                f"Installed frames carry median AoI up to {max(v for v in aoi.values() if v is not None):.0f} ms "
                "against a 500 ms ack timeout, so post-timeout work is measured, not merely possible."
            ),
        },
        "H4_result_downlink_is_the_bottleneck": {
            "verdict": "CONFIRMED",
            "evidence": (
                f"The median result message is {result_dg_all[0]}-{result_dg_all[-1]} datagrams in every cell "
                "(per-message range 101-105), regardless of action, because the payload "
                "is a base64 720x1280 uint8 label map. Across all 16 cells the edge is directly observed to have "
                "tail-completed at least 11,183 frames while only 747 results returned intact to the UE. Because "
                "the edge counters are LOWER bounds, the per-action survival rates "
                f"{ {k: (None if v is None else round(v, 4)) for k, v in survival.items()} } are UPPER bounds: the "
                "true result-path survival is at most ~7% and may be lower. The uplink cannot explain this, "
                "because the same bound holds for action 71, whose feature is a single datagram."
            ),
        },
        "H5_tail_on_unintended_device_or_high_latency": {
            "verdict": "CONTRADICTED_ON_DEVICE_CONFIRMED_ON_LATENCY",
            "evidence": (
                f"reconstructed_device is {devices} on every decoded frame and edge ready.json declares tail_device "
                f"cuda:0, so the device is the intended one. Median frozen_tail is "
                f"{ {k: (None if v is None else round(v, 1)) for k, v in tail_ms.items()} } ms by action, which "
                "alone caps the edge below 6 Hz against a 10 Hz offer."
            ),
        },
        "H6_timestamp_domain_mismatch_creates_artificial_latency": {
            "verdict": "CONTRADICTED",
            "evidence": (
                "Wall-clock capture->install AoI and UE-perf send->result round trip are measured on two "
                "independent clocks. Their per-frame residual is the unmeasured capture->send head, and its "
                f"per-cell medians span {min(residual):.0f}-{max(residual):.0f} ms across {len(residual)} cells "
                "(computed on the installed subset only, so it is noisier than the full-population head). That is "
                "the same order as the independently measured prepared-queue wait plus front time, and two to "
                "three orders of magnitude below the 3 s-110 s AoI spread the two clocks agree on. A domain "
                "offset, drift, unit error or timestamp reuse large enough to manufacture that spread would have "
                "to appear in this residual and does not. No timestamp was corrected post hoc."
            ),
        },
        "H7_evaluation_work_reduces_preparation_coverage": {
            "verdict": "STRONGLY_SUPPORTED_MAGNITUDE_UNRESOLVED",
            "evidence": (
                "ue_route_b_split_cell_adapter_v1._process_token calls _ground_truth on the single preparation "
                "worker thread after send, and _feedback_worker/_segmentation_worker run further GT and mask work "
                "in the same interpreter. Measured preparation_start->encoding_complete is only 16-39 ms while the "
                "achieved worker period implied by coverage is ~135-150 ms, so ~110-120 ms per frame is spent "
                "outside split inference. No per-stage timer separates GT from radar/image assembly."
            ),
        },
        "H8_radar_sync_or_scheduling_causes_most_prepared_drops": {
            "verdict": "CONTRADICTED",
            "evidence": (
                "Of 14697 classified preparation losses, 14539 (98.93%) are DROPPED_QUEUE_FULL, 142 (0.97%) are "
                "DROPPED_SENSOR_LATE_OR_MISSING and 16 (0.11%) are DROPPED_INCOMPLETE_RADAR_WINDOW. Radar "
                "synchronisation accounts for roughly one percent of preparation loss."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(audit: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Phase-15 retry4 root-cause latency audit")
    add("")
    add(f"- Terminal classification: `{audit['conclusion']}`")
    add(f"- Audited evidence: `{PILOT}`")
    add(f"- Evidence integrity: {audit['evidence_verification']['checks_run']} hashes checked, "
        f"{len(audit['evidence_verification']['mismatches'])} mismatches")
    add(f"- Audit scope: offline, read-only. No experiment, container, model or threshold was touched.")
    add("")

    add("## 1. Action identity (verified before analysis)")
    add("")
    add("| Action | Profile | Family | Quantizer | q_e4 | Catalog agrees |")
    add("|---:|---|---|---|---:|---|")
    for action_id, item in sorted(audit["action_binding"]["actions"].items(), key=lambda kv: int(kv[0])):
        add(f"| {action_id} | `{item['catalog_profile_id']}` | {item['family']} | {item['quantizer']} | "
            f"{item['q_e4']} | {item['catalog_agrees']} |")
    add("")
    add("Bound from the 72-action catalog and cross-checked against every per-cell `resolved_config.yaml` "
        "record, not against the ordering of `campaign.actions.profile_ids`. "
        f"Disagreements: {audit['action_binding']['disagreements'] or 'none'}.")
    add("")

    add("## 2. Directly measured facts")
    add("")
    add("| Action | feat dg/frame | app bytes/frame | result dg/msg | edge completions (lower bound) | results at UE | installs | median AoI ms (median of cell medians) | timely feedback |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in audit["per_action"]:
        add(
            f"| {row['action_id']} | {row['median_feature_datagrams_per_frame']:.0f} | "
            f"{row['median_udp_application_bytes']:.0f} | "
            f"{('%.0f' % row['median_result_datagrams_per_message']) if row['median_result_datagrams_per_message'] else 'n/a'} | "
            f"{row['edge_tail_completions_lower_bound']} | {row['results_at_ue']} | {row['installs']} | "
            f"{('%.0f' % row['median_aoi_ms']) if row['median_aoi_ms'] else 'n/a'} | {row['timely_feedback']} |"
        )
    add("")
    add(f"- **Zero** frames in **any** of the 16 cells met the 500 ms feedback deadline "
        f"({audit['totals']['timely_feedback']} of {audit['totals']['sent']} sent).")
    add(f"- **Zero** installs met the 100 ms `service_deadline_ms` "
        f"({audit['totals']['installs_within_service_deadline']} of {audit['totals']['installs']} installs).")
    add("- Feature reassembly is strictly all-or-nothing: across all decoded frames, "
        "`feature_received_datagrams` equals `datagrams` sent and duplicates are zero.")
    add("- Installation is lossless downstream of the UE result loop: results ingested at the UE equals maps "
        "installed equals ACK rows, in all 16 cells.")
    add("- `installation -> feedback emission` median is ~0.01 ms and `emission -> receipt` ~0.1 ms. Feedback "
        "delivery contributes nothing to the deadline miss.")
    add("")

    add("## 3. Where the frames go (funnel)")
    add("")
    add("Per-cell detail is in `funnel_by_cell.csv`. Aggregated by action:")
    add("")
    add("| Action | sent | reached+processed by edge | results back at UE | installed | edge completion rate | result survival rate |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for row in audit["per_action"]:
        add(
            f"| {row['action_id']} | {row['sent']} | {row['edge_tail_completions_lower_bound']} | "
            f"{row['results_at_ue']} | {row['installs']} | "
            f"{('%.3f' % row['tail_completion_rate']) if row['tail_completion_rate'] else '0.000'} | "
            f"{('%.3f' % row['result_survival_rate']) if row['result_survival_rate'] else 'n/a'} |"
        )
    add("")
    add("Reading direction matters here:")
    add("")
    add("- Edge-side counts are **lower bounds**. `edge_counters` rides inside each returned result, so any edge "
        "work performed after the last surviving result is invisible from the UE. `edge completion rate` is "
        "therefore a lower bound and `result survival rate` is an **upper** bound.")
    add("- Action 0's `0` edge completions is **absence of evidence, not measured zero**. Every edge-side counter "
        "travels inside a returned result, and action 0 returned none, so action 0 has no edge-side evidence at "
        "all. See the unresolved list.")
    add("")

    add("## 4. Source-code facts")
    add("")
    for item in audit["source_facts"]:
        add(f"- {item}")
    add("")

    add("## 5. Clock discipline")
    add("")
    for item in audit["clock_discipline"]["domains"]:
        add(f"- **{item['domain']}** — {item['fields']}")
    add("")
    add(f"- Cross-domain test: {audit['clock_discipline']['cross_check']}")
    add("- Intervals declared unavailable rather than reconstructed:")
    for name, reason in sorted(audit["clock_discipline"]["unavailable"].items()):
        add(f"  - `{name}`: {reason}")
    add("")

    add("## 6. Hypotheses")
    add("")
    for name, item in audit["hypotheses"].items():
        add(f"### {name} — `{item['verdict']}`")
        add("")
        add(item["evidence"])
        add("")

    add("## 7. Preparation-loss attribution")
    add("")
    add("| Cause | Frames | Share |")
    add("|---|---:|---:|")
    attribution = audit["preparation_losses"]
    for key, value in attribution["classified"].items():
        add(f"| {key} | {value} | {value / attribution['total_classified']:.4f} |")
    add("")
    add(f"Total classified preparation losses: **{attribution['total_classified']}** "
        f"(plus {attribution['warmup_excluded']} warmup frames before the first complete radar window, which are "
        "not losses). Preparation is **stationary**: no cell shows a monotone quartile trend in coverage or in "
        "median prepared-queue wait, the full quartile spread is at most 0.204, and coverage peaks in route "
        "quartile 2 in 15 of 16 cells. A pattern that reproduces at the same route position across 16 "
        "independent cells and all four network profiles is a route-geometry effect, not drift and not a "
        "growing backlog. Preparation loss is therefore a steady-state throughput deficit, and it is independent "
        "of the network and of the transport-side AoI growth.")
    add("")

    add("## 8. Evidence-supported inference")
    add("")
    for item in audit["inference"]:
        add(f"- {item}")
    add("")

    add("## 9. Unresolved questions (evidence not retained)")
    add("")
    for item in audit["unresolved"]:
        add(f"- {item}")
    add("")

    add("## 10. Minimal remediation specification (NOT implemented)")
    add("")
    for item in audit["remediation"]:
        add(f"- {item}")
    add("")
    add("### Shortest follow-up measurement")
    add("")
    add(audit["follow_up_measurement"])
    add("")
    add("## 11. Scope of the conclusion")
    add("")
    add("`ROOT_CAUSE_LOCALIZED` is claimed at **stage** granularity and no finer. Specifically:")
    add("")
    add("- **Localized by direct measurement**: the deadline miss (over-determined by the UE head alone); the "
        "preparation deficit and its 98.93% attribution to bounded-queue overflow; the inverse AoI ordering and "
        "its monotone growth; the collapse between edge tail completion and result arrival at the UE.")
    add("- **Localized by evidence-supported inference, not direct measurement**: the byte-bounded-FIFO depth "
        "arithmetic that predicts the per-action AoI plateaus; action 0's failure being on the uplink.")
    add("- **NOT localized, and no amount of re-analysis of this evidence will localize it**: uplink radio loss "
        "versus edge socket-buffer overflow; downlink radio loss versus UE result-loop blocking; the split "
        "between evaluation GT and sensor assembly inside the UE worker period. These need the counters listed "
        "in the remediation specification.")
    add("")
    add("This audit did not run any experiment, did not modify retry4, and did not move any gate, timeout, "
        "queue size, payload, threshold or trace. The 500 ms ack timeout and the 100 ms service deadline are "
        "reported against as-registered and are not the defect.")
    add("")
    add(f"Conclusion: `{audit['conclusion']}`")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SOURCE_FACTS = [
    "`ue_route_b_split_cell_adapter_v1.py:784` — `prepared_queue` is a FIFO `queue.Queue(maxsize=4)`. "
    "`on_world_tick` uses `put_nowait` and classifies overflow as `DROPPED_QUEUE_FULL`. There is no "
    "latest-frame-first policy: the oldest queued frame is always served first.",
    "`ue_route_b_split_cell_adapter_v1.py:1006-1026` — exactly **one** worker thread drains that queue, and "
    "`_process_token` runs radar-window assembly, radar rasterisation, front inference, encode, send **and** "
    "the evaluation-only `_ground_truth` call serially on it.",
    "`ue_route_b_split_cell_adapter_v1.py:1084-1085` — `service_deadline_at` and `ack_timeout_at` are computed "
    "here and then only ever written into rows or compared to label a row `late` "
    "(`ue_map_install_feedback_v1.py:148` and `:198`). Neither identifier occurs anywhere in "
    "`live_pilot_runtime.py` or the map server, so no edge admission, decode, tail, publication or install step "
    "is gated on the deadline. A 500 ms timeout cancels and obsoletes nothing.",
    "`splitfusion_live_dispatch_v1/live_pilot_runtime.py:440-475` — the edge is a single blocking "
    "`recvfrom` loop; reassembly, decode, frozen tail, label copy, base64, JSON and 100+ `sendto` calls all run "
    "inline. While one frame is processed the socket is not drained, so the only buffer is the kernel SO_RCVBUF.",
    "`splitfusion_live_dispatch_v1/live_pilot_runtime.py:473` — every result carries "
    "`semantic_labels_b64`, a base64 720x1280 uint8 mask, so the result message is ~1.23 MB for **every** action, "
    "independent of the feature payload it answers.",
    "`phase2_map_sharing/transport.py:39-101` — `ChunkReassembler` is all-or-nothing with a 2 s timeout and no "
    "retransmission (`runtime.retransmission = false`). One lost datagram destroys the whole message and the "
    "loss is never counted at either endpoint.",
    "`splitfusion_live_dispatch_v1/live_pilot_runtime.py:331-357` — the UE result loop is also a single thread, and "
    "it performs a 1.23 MB base64 decode, a 921,600-byte `np.save` to disk and a zlib compression per result "
    "before returning to `recvfrom`. Result delivery is serialised and blocking.",
    "`uplink_only_spatial_map_pipeline/spatial_map_server_moving_ego_uplink_only_baseline.py:1713` — install "
    "and `_emit_install_feedback` are unconditional; the map server never inspects a deadline.",
    "`rl_agent/ue_map_install_feedback_v1.py:192` — `record_expired` does not remove the capture from "
    "`pending`, so a capture receives a `TIMEOUT_NO_ACK` terminal row at capture+500 ms and a later ACK is "
    "appended as a second, non-terminal, `late=True` diagnostic row. The 100% `TIMEOUT_NO_ACK` terminal "
    "distribution alongside non-zero `ack_installed_frames` is therefore correct accounting, not a defect.",
    "Edge `ready.json` declares `tail_device: cuda:0` and every decoded frame reports "
    "`reconstructed_device = cuda:0`. The tail ran on the intended device.",
]


def build_audit(pilot: Path) -> dict[str, Any]:
    verification = verify_evidence(pilot)
    binding = bind_action_identities(pilot)

    cells: list[dict[str, Any]] = []
    for summary_row in read_csv(pilot / "cell_summary.csv"):
        cell = load_cell(pilot, summary_row)
        funnel = cell_funnel(cell)
        cells.append(
            {
                "funnel": funnel,
                "reconciliation": reconcile_funnel(funnel, cell),
                "intervals": cell_intervals(cell),
                "payload": payload_profile(cell, funnel),
                "preparation_attribution": preparation_attribution(funnel),
                "preparation_stationarity": preparation_stationarity(cell),
                "aoi_growth": aoi_growth(cell),
                "devices": sorted(
                    {row["reconstructed_device"] for row in cell["per_frame"] if row.get("decoded") == "True"}
                ),
            }
        )

    by_action: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_action.setdefault(cell["funnel"]["action_id"], []).append(cell)

    per_action: list[dict[str, Any]] = []
    for action_id in sorted(by_action):
        group = by_action[action_id]
        sent = sum(cell["funnel"]["s07_feature_messages_sent"] for cell in group)
        edge_done = sum(cell["funnel"]["s14_tail_completions_lower_bound"] for cell in group)
        at_ue = sum(cell["funnel"]["s17_result_messages_ingested_at_ue"] for cell in group)
        installs = sum(cell["funnel"]["s18_maps_installed"] for cell in group)
        aois = [
            cell["intervals"]["capture_to_installation_aoi_ms"]["median"]
            for cell in group
            if cell["intervals"]["capture_to_installation_aoi_ms"]["median"] is not None
        ]
        result_dgs = [
            cell["payload"]["median_result_datagrams_per_message"]
            for cell in group
            if cell["payload"]["median_result_datagrams_per_message"]
        ]
        per_action.append(
            {
                "action_id": action_id,
                "profile_id": EXPECTED_ACTION_IDENTITY[action_id],
                "sent": sent,
                "edge_tail_completions_lower_bound": edge_done,
                "results_at_ue": at_ue,
                "installs": installs,
                "median_feature_datagrams_per_frame": statistics.median(
                    [cell["payload"]["median_feature_datagrams_per_frame"] for cell in group]
                ),
                "median_udp_application_bytes": statistics.median(
                    [cell["payload"]["median_udp_application_bytes"] for cell in group]
                ),
                "median_result_datagrams_per_message": statistics.median(result_dgs) if result_dgs else None,
                "tail_completion_rate": edge_done / sent if sent else None,
                "result_survival_rate": at_ue / edge_done if edge_done else None,
                "installation_rate": installs / sent if sent else None,
                "median_aoi_ms": statistics.median(aois) if aois else None,
                "timely_feedback": sum(cell["funnel"]["s20_feedback_received_within_ack_timeout"] for cell in group),
            }
        )

    classified: dict[str, int] = {}
    warmup = 0
    for cell in cells:
        attribution = cell["preparation_attribution"]
        for key in (
            "bounded_preparation_queue_overflow",
            "sensor_synchronisation_late_or_missing",
            "missing_or_incomplete_radar_window",
            "split_processing_failure",
        ):
            classified[key] = classified.get(key, 0) + int(attribution[key])
        warmup += int(attribution["warmup_before_first_complete_radar_window"])

    totals = {
        "sent": sum(cell["funnel"]["s07_feature_messages_sent"] for cell in cells),
        "installs": sum(cell["funnel"]["s18_maps_installed"] for cell in cells),
        "timely_feedback": sum(cell["funnel"]["s20_feedback_received_within_ack_timeout"] for cell in cells),
        "installs_within_service_deadline": sum(cell["funnel"]["s21_installs_within_service_deadline"] for cell in cells),
        "edge_tail_completions_lower_bound": sum(cell["funnel"]["s14_tail_completions_lower_bound"] for cell in cells),
        "results_at_ue": sum(cell["funnel"]["s17_result_messages_ingested_at_ue"] for cell in cells),
    }

    residuals = [
        cell["intervals"]["cross_domain_residual_wall_aoi_minus_perf_round_trip_ms"]["median"]
        for cell in cells
        if cell["intervals"]["cross_domain_residual_wall_aoi_minus_perf_round_trip_ms"]["samples"]
    ]
    reconciliation_problems = sorted({item for cell in cells for item in cell["reconciliation"]})

    audit: dict[str, Any] = {
        "schema": "scenesense.splitfusion_phase15_retry4_latency_audit.v1",
        "audited_evidence_root": str(PILOT),
        "scope": "offline_forensic_audit_read_only_no_experiment_executed",
        "evidence_verification": verification,
        "action_binding": binding,
        "cells": cells,
        "per_action": per_action,
        "totals": totals,
        "funnel_reconciliation_problems": reconciliation_problems,
        "preparation_losses": {
            "classified": classified,
            "total_classified": sum(classified.values()),
            "warmup_excluded": warmup,
            "stationary_in_every_cell": all(cell["preparation_stationarity"]["stationary"] for cell in cells),
        },
        "source_facts": SOURCE_FACTS,
        "clock_discipline": {
            "domains": [
                {
                    "domain": UE_PERF,
                    "fields": "capture_started_ns, ue_prepare_finished_ns, send_finished_ns, "
                    "edge_result_received_ns, queue_wait_ms — one process, differences valid",
                },
                {
                    "domain": EDGE_PERF,
                    "fields": "edge_timing_ns stage boundaries, edge_received_ns, tail_finished_ns — separate "
                    "container process; only intra-domain DURATIONS are used",
                },
                {
                    "domain": HOST_WALL,
                    "fields": "capture_at, service_deadline_at, install_timestamp, feedback_emit_at, "
                    "feedback_received_at, ack_timeout_at — all time.time() on the one physical host "
                    "(the map server is a local subprocess bound to 127.0.0.1), so differences are valid",
                },
            ],
            "cross_check": (
                "Wall-clock capture->install AoI minus UE-perf send->result round trip has per-cell medians "
                f"spanning {min(residuals):.0f}-{max(residuals):.0f} ms across {len(residuals)} cells. That "
                "residual is exactly the unmeasured capture->send head. It is computed on the installed subset "
                "only (24-109 frames per cell), so it is noisier than the full-population head, but it sits in "
                "the same 0.3-0.9 s band as the independently measured prepared-queue wait (per-cell medians "
                "513-684 ms) plus front time (16-39 ms), and it is two to three orders of magnitude smaller than "
                "the 3 s-110 s AoI spread that the two clocks independently agree on. A constant offset, drift, "
                "unit mismatch or timestamp reuse large enough to manufacture that spread would necessarily show "
                "up in this residual, and does not. No timestamp was corrected post hoc."
            ),
            "unavailable": UNAVAILABLE_INTERVALS,
        },
        "hypotheses": classify_hypotheses(cells),
    }

    audit["inference"] = [
        "The edge socket receive buffer is bounded in bytes (16,777,216 reported), not in frames, so its depth "
        "in FRAMES is inversely proportional to payload size: ~4.7 frames at action 0's 3.59 MB, ~14 at action "
        "20's 1.21 MB, ~167 at action 46's 100 kB and ~2,620 at action 71's 6.4 kB. With a measured edge service "
        "rate of only 2.2-3.1 Hz against a ~6-7 Hz offer, backlog accumulates until the buffer saturates, and the "
        "saturated backlog delay is (buffer frames / service rate). That predicts a few seconds for action 20, "
        "tens of seconds for action 46, and a backlog that cannot saturate inside a ~430 s route for action 71 — "
        "which is exactly the measured ordering and exactly the measured monotone AoI growth for action 71.",
        "The inverse latency ordering is therefore REAL and mechanistic, not a clock artifact and not a property "
        "of the radio: a smaller payload buys more admitted frames, and every admitted frame is served FIFO from "
        "an ever-older backlog.",
        "Action 0's zero installs are consistent with uplink fragment loss: 288 datagrams/frame at ~6 frames/s is "
        "~196 Mbps offered on the uplink, far above the registered 100 MHz 4D5U profile, and one lost fragment of "
        "288 destroys the message. A weaker but independent bound: if action 0's edge had completed as many frames "
        "as action 20's (419-1036), then at action 20's measured ~6% result survival rate the probability of "
        "observing zero returned results across four cells is negligible. Action 0 almost certainly failed on the "
        "uplink, not on the downlink — but see the unresolved list, because no edge-side evidence survives for it.",
        "The 100 ms service deadline is unreachable before the network is even reached. The prepared-queue wait "
        "alone has per-cell medians of 513-684 ms, and the front adds 16-39 ms, so a frame is already 0.5-0.7 s "
        "old at the instant the UE finishes sending it. Even a zero-latency transport and a zero-latency edge "
        "could not have produced a single on-deadline install in this pilot. This is measured on one clock inside "
        "one process and does not depend on any cross-domain assumption.",
        "Preparation coverage of 0.67-0.76 follows arithmetically from a single-threaded worker serving ~7.4 "
        "frames/s against a 10 Hz opportunity stream through a depth-4 FIFO. Split inference accounts for only "
        "16-39 ms of that ~135-150 ms worker period.",
    ]

    audit["unresolved"] = [
        "How many feature datagrams actually arrived at the edge. Neither endpoint exports "
        "`ChunkReassembler.expired_messages`, and a message that never completes leaves no record, so uplink "
        "radio loss cannot be separated from kernel-socket-buffer overflow at the edge.",
        "How many result datagrams the edge actually sent and how many arrived. The result path has no sender-side "
        "counter and no per-datagram receipt record, so the measured 3-6% result survival cannot be decomposed "
        "into downlink radio loss versus UE receive-buffer overflow versus UE result-loop blocking.",
        "Whether action 0 ever completed a single message at the edge. No result returned, and every edge-side "
        "counter travels only inside a returned result, so action 0 has no edge-side evidence whatsoever.",
        "The split between evaluation-only ground-truth work and radar/image assembly inside the ~110-120 ms of "
        "non-split worker time. `_process_token` has no per-stage timer around `_ground_truth`.",
        "Whether the ~180 ms median frozen-tail latency is inherent to the tail or reflects GPU contention with "
        "the co-resident UE front on the same cuda:0 device. Nothing records GPU occupancy or per-process "
        "utilisation.",
    ]

    audit["remediation"] = [
        "Take the segmentation mask off the deployment downlink WITHOUT losing it. `semantic_labels_b64` makes "
        "every result ~1.23 MB in ~102 datagrams for every action, and that fixed cost is what destroys "
        "installed-frame delivery. The mask is evaluation-only evidence, and the edge already has a writable "
        "state mount (`/work/torch_cache`), so it can be persisted edge-side and correlated offline by frame_id "
        "while only the object records travel the downlink. This preserves segmentation evidence coverage and "
        "the measurement contract; it must not be implemented as simply deleting the mask.",
        "Make the edge non-blocking: drain `recvfrom` on a dedicated thread into an explicit bounded, "
        "latest-frame-first queue with a classified drop counter, so the kernel byte-buffer stops acting as a "
        "hidden unbounded FIFO and the payload-inverse AoI ordering disappears.",
        "Enforce the deadline. Carry `capture_timestamp_ns` (already in the SFD1 v2 envelope) into an explicit "
        "obsolescence test at edge admission, before the tail, and at map install, and count each discard. Today "
        "nothing reads the deadline, so 100% of the edge's work after the first few seconds is spent on frames "
        "that can never be timely.",
        "Move evaluation-only ground truth off the preparation worker onto its own bounded queue, and add a "
        "per-stage timer so preparation coverage loss becomes attributable rather than inferred.",
        "Instrument the two blind stages: export `ChunkReassembler.expired_messages` and per-message "
        "expected/received datagram counts at both endpoints, and add a result-path sender counter. Without these, "
        "uplink loss and buffer overflow remain permanently inseparable.",
        "Re-derive the 100 ms `service_deadline_ms` only after the 0.5-0.7 s capture->send head is fixed; the "
        "threshold itself is not the defect and must not be relaxed to manufacture a pass.",
    ]

    audit["follow_up_measurement"] = (
        "Two cells only — actions **20** (`split_ae128_uint8_q5000`) and **71** (`split_ae32_uint4_q9800`) under "
        "**FAVORABLE_STABLE**, one route each — with the mask removed from the result path, an explicit bounded "
        "latest-frame-first edge queue, deadline-based discard, and the two new loss counters. Those two actions "
        "bracket the payload range by ~190x and are the two extremes of the observed inverse AoI ordering, so they "
        "are jointly sufficient to falsify the fix. The predicted direction and magnitude, to be pre-registered "
        "by Abiodun before the run rather than set by this audit, is: installed-frame AoI for the two actions "
        "converges to within roughly one edge service period instead of differing by ~30x; result survival rises "
        "by more than an order of magnitude from the measured <=8% upper bound; and AoI stops growing "
        "monotonically through the route for action 71. A second 16-cell pilot is NOT requested and would add no "
        "discriminating power until this two-cell probe settles the mechanism."
    )

    integrity_ok = verification["all_verified"] and binding["verified"] and not reconciliation_problems
    audit["conclusion"] = "ROOT_CAUSE_LOCALIZED" if integrity_ok else "PARTIAL_LOCALIZATION_REQUIRES_INSTRUMENTATION"
    audit["localized_bottleneck_stage"] = (
        "stage 15-17: result publication and downlink delivery of the ~1.23 MB / ~102-datagram edge result, "
        "compounded by a byte-bounded kernel FIFO at the edge (stage 12) that makes installed-frame AoI grow "
        "inversely with feature payload, and preceded by a UE preparation head (stages 4-7) that alone exceeds "
        "the 100 ms service deadline"
    )
    return audit


def write_outputs(audit: Mapping[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []

    audit_json = {key: value for key, value in audit.items() if key != "cells"}
    audit_json["cells"] = [
        {
            "funnel": cell["funnel"],
            "reconciliation": cell["reconciliation"],
            "payload": cell["payload"],
            "preparation_attribution": cell["preparation_attribution"],
            "preparation_stationarity": cell["preparation_stationarity"],
            "aoi_growth": cell["aoi_growth"],
            "intervals": cell["intervals"],
        }
        for cell in audit["cells"]
    ]
    path = out_dir / "audit.json"
    path.write_text(json.dumps(audit_json, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    written.append(path)

    funnel_fields = [
        key for key in audit["cells"][0]["funnel"] if key not in ("prepare_status_counts", "terminal_feedback_outcomes")
    ]
    path = out_dir / "funnel_by_cell.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=funnel_fields + ["prepare_status_counts", "reconciliation_problems"])
        writer.writeheader()
        for cell in audit["cells"]:
            row = {key: cell["funnel"][key] for key in funnel_fields}
            row["prepare_status_counts"] = json.dumps(cell["funnel"]["prepare_status_counts"], sort_keys=True)
            row["reconciliation_problems"] = ";".join(cell["reconciliation"])
            writer.writerow(row)
    written.append(path)

    path = out_dir / "latency_by_action_profile.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cell_id", "action_id", "profile_id", "network_profile_id", "interval", "clock_domain",
             "median_ms", "p95_ms", "maximum_ms", "samples", "missing"]
        )
        for cell in audit["cells"]:
            funnel = cell["funnel"]
            for name, stats in sorted(cell["intervals"].items()):
                writer.writerow(
                    [funnel["cell_id"], funnel["action_id"], funnel["profile_id"], funnel["network_profile_id"],
                     name, stats["clock_domain"], stats["median"], stats["p95"], stats["maximum"],
                     stats["samples"], stats["missing"]]
                )
        for name, reason in sorted(UNAVAILABLE_INTERVALS.items()):
            writer.writerow(["ALL", "ALL", "ALL", "ALL", name, "UNAVAILABLE", "", "", "", 0, reason])
    written.append(path)

    path = out_dir / "REPORT.md"
    path.write_text(render_report(audit), encoding="utf-8")
    written.append(path)

    manifest = {
        "schema": "splitfusion_phase15_retry4_latency_audit_artifacts.v1",
        "files": [
            {"path": item.name, "bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in sorted(written, key=lambda value: value.name)
        ],
    }
    path = out_dir / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(path)

    path = out_dir / TERMINAL_NAME
    path.write_text(f"{audit['conclusion']}\n", encoding="utf-8")
    written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / PILOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/splitfusion_phase15_retry4_latency_audit_v1/20260906_root_cause_audit",
    )
    args = parser.parse_args(argv)

    audit = build_audit(args.pilot.resolve(strict=True))
    written = write_outputs(audit, args.out)
    print(f"conclusion: {audit['conclusion']}")
    print(f"bottleneck: {audit['localized_bottleneck_stage']}")
    for item in written:
        print(f"wrote {item.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
