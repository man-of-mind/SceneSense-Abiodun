#!/usr/bin/env python3
"""Short live validation of the direct edge-to-map SplitFusion deployment.

Four representative actions under FAVORABLE_STABLE -- 15 (noAE), 30 (AE128),
50 (AE64), 71 (AE32) -- each on a fresh CARLA world and a fresh OAI radio, with
a full teardown between actions. The run proves the corrected architecture:
object-map updates travel edge -> map on the edge-local CN5G bridge, the map
installs under its own state lock before any ACK is emitted, and the UE receives
only compact, record-free control messages.

This is not a Route-B loop completion run and never claims one. It is also not
an authorization for another 288-cell live campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_288_campaign_supervisor as supervisor  # noqa: E402
from rl_agent.splitfusion_direct_edge_map_v1 import protocol  # noqa: E402

SCHEMA = "scenesense.splitfusion.direct_edge_map_live_validation.v1"
TOKEN = "SPLITFUSION_DIRECT_EDGE_MAP_LIVE_VALIDATION"
TERMINAL = "SPLITFUSION_DIRECT_EDGE_MAP_LIVE_VALIDATION_COMPLETE"
DEFAULT_CONFIG = ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
DIRECT_ADAPTER = ROOT / "rl_agent/splitfusion_direct_edge_map_v1/adapter_direct_v1.py"
ACTION_ORDER = (15, 30, 50, 71)
ACTION_FAMILY = {15: "noAE", 30: "AE128", 50: "AE64", 71: "AE32"}
NETWORK_PROFILE = "FAVORABLE_STABLE"
FRAMES_PER_ACTION = 300
# The UE addresses that an object-map update must never reach.
UE_TUNNEL_HOSTS = ("10.0.0.2",)
UE_RESULT_PORTS = (51004, 51104)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise supervisor.CampaignError(message)


def _f(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": min(finite),
        "p05": _quantile(finite, 0.05),
        "p25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.50),
        "p75": _quantile(finite, 0.75),
        "p95": _quantile(finite, 0.95),
        "p99": _quantile(finite, 0.99),
        "max": max(finite),
        "mean": statistics.fmean(finite),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_cells(cells: Sequence[supervisor.Cell]) -> list[supervisor.Cell]:
    """The four FAVORABLE_STABLE cells, in the registered action order."""

    by_action = {
        cell.action_id: cell
        for cell in cells
        if cell.network_profile_id == NETWORK_PROFILE
    }
    missing = [action for action in ACTION_ORDER if action not in by_action]
    require(not missing, f"configuration lacks FAVORABLE_STABLE cells for {missing}")
    return [by_action[action] for action in ACTION_ORDER]


def evaluate_cell(
    attempt_dir: Path, cell: supervisor.Cell, frames_per_action: int
) -> dict[str, Any]:
    """Derive the direct-architecture evidence for one action."""

    direct = attempt_dir / "direct_edge_map"
    per_frame = _read_csv(attempt_dir / "per_frame_metrics.csv")
    feedback = _read_csv(attempt_dir / "map_feedback.csv")
    ingest = _read_csv(direct / "direct_map_ingest.csv")
    sent = [row for row in per_frame if row.get("prepare_status") == "SENT"]
    sent.sort(key=lambda row: _f(row.get("capture_wall_s")) or 0.0)
    window = sent[:frames_per_action]
    window_frames = {str(row.get("frame_id")) for row in window}

    edge_counters: dict[str, Any] = {}
    counters_path = direct / "direct_edge_counters.json"
    if counters_path.is_file():
        try:
            edge_counters = json.loads(counters_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            edge_counters = {}
    map_report: dict[str, Any] = {}
    report_path = direct / "direct_map_report.json"
    if report_path.is_file():
        try:
            map_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            map_report = {}
    map_ready: dict[str, Any] = {}
    ready_path = direct / "direct_map_ready.json"
    if ready_path.is_file():
        try:
            map_ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            map_ready = {}
    edge_ready: dict[str, Any] = {}
    edge_ready_path = direct / "direct_edge_ready.json"
    if edge_ready_path.is_file():
        try:
            edge_ready = json.loads(edge_ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            edge_ready = {}

    # -- direct publication -> installation latency (the Phase-E input) -----
    installed = [row for row in ingest if row.get("outcome") == "RESULT_INSTALLED"]
    installed_window = [
        row for row in installed if str(row.get("frame_id")) in window_frames
    ] or installed
    publish_to_install_ms = [
        value
        for value in (
            _f(row.get("install_latency_from_publish_ms")) for row in installed_window
        )
        if value is not None
    ]
    tail_to_install_ms = [
        value
        for value in (
            _f(row.get("install_latency_from_tail_ms")) for row in installed_window
        )
        if value is not None
    ]
    ingest_to_install_ms = []
    publish_to_ingest_ms = []
    for row in installed_window:
        ingest_at = _f(row.get("map_ingest_at"))
        install_at = _f(row.get("map_install_at"))
        publish_at = _f(row.get("edge_publish_start_wall_s"))
        if ingest_at is not None and install_at is not None:
            ingest_to_install_ms.append((install_at - ingest_at) * 1000.0)
        if publish_at is not None and ingest_at is not None:
            publish_to_ingest_ms.append((ingest_at - publish_at) * 1000.0)
    map_age_ms = [
        value
        for value in (_f(row.get("map_age_at_install_ms")) for row in installed_window)
        if value is not None
    ]

    # -- ACK arrival, measured separately from physical freshness ----------
    ack_delay_ms = [
        value
        for value in (
            _f(row.get("feedback_observation_delay_ms"))
            for row in feedback
            if str(row.get("terminal", "")).lower() in {"1", "true"}
        )
        if value is not None
    ]

    # -- ordering, duplication and identity -------------------------------
    install_before_ack = 0
    ack_before_install = 0
    for row in installed:
        install_at = _f(row.get("map_install_at"))
        emit_at = _f(row.get("feedback_emit_at"))
        if install_at is None or emit_at is None:
            continue
        if emit_at >= install_at:
            install_before_ack += 1
        else:
            ack_before_install += 1

    install_keys = [
        (str(row.get("stream_id")), str(row.get("frame_id"))) for row in installed
    ]
    duplicate_installations = len(install_keys) - len(set(install_keys))

    identity_mismatches: list[str] = []
    for row in ingest:
        if str(row.get("cell_id")) != cell.cell_id:
            identity_mismatches.append(f"cell:{row.get('cell_id')}")
        if str(row.get("action_id")) not in {str(cell.action_id), ""}:
            identity_mismatches.append(f"action:{row.get('action_id')}")
        if str(row.get("profile_id")) not in {cell.profile_id, ""}:
            identity_mismatches.append(f"profile:{row.get('profile_id')}")

    # -- terminal accounting ----------------------------------------------
    terminal_rows = [
        row for row in feedback if str(row.get("terminal", "")).lower() in {"1", "true"}
    ]
    terminal_by_capture: dict[str, int] = {}
    for row in terminal_rows:
        key = str(row.get("capture_id"))
        terminal_by_capture[key] = terminal_by_capture.get(key, 0) + 1
    multi_terminal = {key: n for key, n in terminal_by_capture.items() if n != 1}
    outcome_counts: dict[str, int] = {}
    credit_counts: dict[str, int] = {}
    for row in terminal_rows:
        outcome = str(row.get("outcome") or "")
        credit = str(row.get("agent_credit") or "")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        credit_counts[credit] = credit_counts.get(credit, 0) + 1

    sent_captures = {str(row.get("capture_id")) for row in sent}
    unexpected_terminals = sorted(set(terminal_by_capture) - sent_captures)
    missing_terminals = sorted(sent_captures - set(terminal_by_capture))

    # -- record-free UE path ----------------------------------------------
    ue_record_bearing_messages = 0
    for row in feedback:
        for key in protocol.FORBIDDEN_UE_KEYS:
            if str(row.get(key) or ""):
                ue_record_bearing_messages += 1
    control_bytes = [
        value for value in (_f(row.get("feedback_bytes")) for row in ingest)
        if value is not None
    ]
    direct_bytes = [
        value for value in (_f(row.get("direct_update_bytes")) for row in ingest)
        if value is not None
    ]

    summary_path = attempt_dir / "RESULTS_SUMMARY.json"
    adapter_summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            adapter_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            adapter_summary = {}

    route_summary: dict[str, Any] = {}
    route_path = attempt_dir / "route_metrics_summary.json"
    if route_path.is_file():
        try:
            route_summary = json.loads(route_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            route_summary = {}

    counters = dict(edge_counters.get("counters") or {})
    publisher = dict(edge_counters.get("direct_map_publisher") or {})
    ue_control = dict(edge_counters.get("ue_control") or {})

    return {
        "cell_id": cell.cell_id,
        "action_id": cell.action_id,
        "model_family": ACTION_FAMILY.get(cell.action_id, cell.model_family),
        "profile_id": cell.profile_id,
        "network_profile_id": cell.network_profile_id,
        "attempt_dir": str(attempt_dir.relative_to(ROOT)),
        "adapter_status": str(adapter_summary.get("terminal_status") or ""),
        "route_loops_completed": int(route_summary.get("loops_completed", 0) or 0),
        "complete_route_b_loop_claimed": False,
        # transmission
        "prepared_rows": len(per_frame),
        "captures_sent": len(sent),
        "analysis_window_frames": len(window),
        "frames_per_action_target": frames_per_action,
        "window_is_complete": len(window) == frames_per_action,
        # edge
        "edge_feature_datagrams_received": int(
            counters.get("feature_datagrams_received", 0)
        ),
        "edge_feature_messages_reassembled": int(
            counters.get("feature_messages_reassembled", 0)
        ),
        "edge_incomplete_reassemblies_expired": int(
            counters.get("incomplete_reassemblies_expired", 0)
        ),
        "edge_queue_admissions": int(counters.get("edge_queue_admissions", 0)),
        "edge_pending_replacements": int(counters.get("edge_pending_replacements", 0)),
        "edge_admission_refused_not_freshest": int(
            counters.get("edge_admission_refused_not_freshest", 0)
        ),
        "edge_process_starts": int(counters.get("edge_process_starts", 0)),
        "edge_tail_completions": int(counters.get("tail_completions", 0)),
        "edge_direct_map_publications": int(
            counters.get("direct_map_publications", 0)
        ),
        "edge_direct_map_datagrams": int(
            publisher.get("direct_map_datagrams_transmitted", 0)
        ),
        "edge_direct_map_payload_bytes": int(
            publisher.get("direct_map_payload_bytes", 0)
        ),
        "edge_ue_control_messages": int(ue_control.get("ue_control_messages_sent", 0)),
        "edge_ue_control_bytes": int(ue_control.get("ue_control_bytes_sent", 0)),
        "edge_object_records_on_radio": bool(
            edge_ready.get("object_records_on_radio", True)
        ),
        "edge_dense_label_map_on_radio": bool(
            edge_ready.get("dense_label_map_on_radio", True)
        ),
        "edge_direct_map_host": str(edge_ready.get("direct_map_host") or ""),
        "edge_direct_map_port": int(edge_ready.get("direct_map_port", 0) or 0),
        # map
        "map_bind_host": str(map_ready.get("direct_map_host") or ""),
        "map_bind_port": int(map_ready.get("direct_map_port", 0) or 0),
        "map_legacy_loopback_listener": str(
            map_ready.get("legacy_loopback_ingest_listener") or ""
        ),
        "map_updates_reassembled": int(
            (map_report.get("direct_ingest") or {}).get("counters", {}).get(
                "direct_updates_reassembled", 0
            )
        ),
        "map_updates_installed": len(installed),
        "map_updates_duplicate": int(
            (map_report.get("direct_ingest") or {}).get("counters", {}).get(
                "direct_updates_duplicate", 0
            )
        ),
        "map_updates_superseded": sum(
            1 for row in ingest if row.get("outcome") == "SUPERSEDED_PENDING"
        ),
        "map_updates_rejected": sum(
            1 for row in ingest if row.get("outcome") == "MAP_REJECTED"
        ),
        "map_updates_stale": sum(
            1 for row in ingest if row.get("outcome") == "STALE_BEFORE_MAP"
        ),
        "duplicate_installations": duplicate_installations,
        "install_before_ack_frames": install_before_ack,
        "ack_before_install_frames": ack_before_install,
        "identity_mismatches": len(identity_mismatches),
        "identity_mismatch_examples": sorted(set(identity_mismatches))[:5],
        # latency
        "direct_publish_to_install_ms": _distribution(publish_to_install_ms),
        "direct_tail_to_install_ms": _distribution(tail_to_install_ms),
        "direct_publish_to_ingest_ms": _distribution(publish_to_ingest_ms),
        "direct_ingest_to_install_ms": _distribution(ingest_to_install_ms),
        "map_age_at_install_ms": _distribution(map_age_ms),
        "ack_observation_delay_ms": _distribution(ack_delay_ms),
        "direct_update_bytes": _distribution(direct_bytes),
        "compact_feedback_bytes": _distribution(control_bytes),
        # terminal accounting
        "terminal_rows": len(terminal_rows),
        "terminal_outcomes": dict(sorted(outcome_counts.items())),
        "agent_credits": dict(sorted(credit_counts.items())),
        "captures_without_terminal": len(missing_terminals),
        "unexpected_terminals": len(unexpected_terminals),
        "captures_with_multiple_terminals": len(multi_terminal),
        "ue_record_bearing_messages": ue_record_bearing_messages,
        "cleanup": dict(adapter_summary.get("cleanup") or {}),
    }


def gate(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The registered structural gates of the direct-architecture validation."""

    checks: list[dict[str, Any]] = []

    def add(name: str, holds: bool, detail: Any) -> None:
        checks.append({"gate": name, "holds": bool(holds), "detail": detail})

    add(
        "every_action_produced_a_cell",
        len(cells) == len(ACTION_ORDER)
        and {int(row["action_id"]) for row in cells} == set(ACTION_ORDER),
        sorted(int(row["action_id"]) for row in cells),
    )
    add(
        "real_ue_feature_uplink_reached_the_edge",
        all(int(row["edge_feature_messages_reassembled"]) > 0 for row in cells),
        {row["cell_id"]: row["edge_feature_messages_reassembled"] for row in cells},
    )
    add(
        "object_map_updates_were_published_directly_to_the_map",
        all(int(row["edge_direct_map_publications"]) > 0 for row in cells)
        and all(int(row["map_updates_reassembled"]) > 0 for row in cells),
        {
            row["cell_id"]: [
                row["edge_direct_map_publications"],
                row["map_updates_reassembled"],
            ]
            for row in cells
        },
    )
    add(
        "no_object_update_targeted_the_ue_result_address",
        all(row["edge_object_records_on_radio"] is False for row in cells)
        and all(
            row["edge_direct_map_host"] not in UE_TUNNEL_HOSTS for row in cells
        )
        and all(
            int(row["edge_direct_map_port"]) not in UE_RESULT_PORTS for row in cells
        ),
        {
            row["cell_id"]: f"{row['edge_direct_map_host']}:{row['edge_direct_map_port']}"
            for row in cells
        },
    )
    add(
        "legacy_loopback_map_listener_was_never_started",
        all(row["map_legacy_loopback_listener"] == "NOT_STARTED" for row in cells),
        {row["cell_id"]: row["map_legacy_loopback_listener"] for row in cells},
    )
    add(
        "map_installation_preceded_every_feedback_emission",
        all(int(row["ack_before_install_frames"]) == 0 for row in cells)
        and all(int(row["install_before_ack_frames"]) > 0 for row in cells),
        {
            row["cell_id"]: [
                row["install_before_ack_frames"],
                row["ack_before_install_frames"],
            ]
            for row in cells
        },
    )
    add(
        "ue_received_compact_record_free_feedback",
        all(int(row["ue_record_bearing_messages"]) == 0 for row in cells)
        and all(int(row["terminal_rows"]) > 0 for row in cells),
        {
            row["cell_id"]: [
                row["terminal_rows"],
                row["ue_record_bearing_messages"],
                (row["compact_feedback_bytes"] or {}).get("median"),
            ]
            for row in cells
        },
    )
    add(
        "exact_frame_action_stream_identity",
        all(int(row["identity_mismatches"]) == 0 for row in cells),
        {row["cell_id"]: row["identity_mismatch_examples"] for row in cells},
    )
    add(
        "no_duplicate_installation",
        all(int(row["duplicate_installations"]) == 0 for row in cells),
        {row["cell_id"]: row["duplicate_installations"] for row in cells},
    )
    add(
        "dense_masks_remained_edge_only",
        all(row["edge_dense_label_map_on_radio"] is False for row in cells),
        {row["cell_id"]: row["edge_dense_label_map_on_radio"] for row in cells},
    )
    add(
        "direct_publication_install_latency_is_measured",
        all(
            int((row["direct_publish_to_install_ms"] or {}).get("count", 0)) > 0
            for row in cells
        ),
        {
            row["cell_id"]: (row["direct_publish_to_install_ms"] or {}).get("median")
            for row in cells
        },
    )
    add(
        "ack_latency_is_measured_separately_from_installation",
        all(
            int((row["ack_observation_delay_ms"] or {}).get("count", 0)) > 0
            for row in cells
        ),
        {
            row["cell_id"]: (row["ack_observation_delay_ms"] or {}).get("median")
            for row in cells
        },
    )
    add(
        "exactly_one_terminal_per_transmission_obligation",
        all(int(row["captures_with_multiple_terminals"]) == 0 for row in cells)
        and all(int(row["unexpected_terminals"]) == 0 for row in cells)
        and all(int(row["captures_without_terminal"]) == 0 for row in cells),
        {
            row["cell_id"]: [
                row["captures_with_multiple_terminals"],
                row["unexpected_terminals"],
                row["captures_without_terminal"],
            ]
            for row in cells
        },
    )
    add(
        "superseded_frames_are_credited_as_replaced_not_lost",
        all(
            row["agent_credits"].get(protocol.CREDIT_NETWORK_INCOMPLETE, 0)
            == 0
            or row["terminal_outcomes"].get("TRANSPORT_INCOMPLETE", 0) > 0
            for row in cells
        ),
        {row["cell_id"]: row["agent_credits"] for row in cells},
    )
    add(
        "at_least_the_requested_frames_were_transmitted",
        all(int(row["captures_sent"]) >= FRAMES_PER_ACTION for row in cells),
        {row["cell_id"]: row["captures_sent"] for row in cells},
    )
    add(
        "clean_teardown",
        all(
            all(
                bool(value)
                for key, value in (row["cleanup"] or {}).items()
                if isinstance(value, bool)
            )
            for row in cells
        ),
        {row["cell_id"]: row["cleanup"] for row in cells},
    )
    holds = all(check["holds"] for check in checks)
    return {
        "status": "PASS" if holds else "FAIL",
        "checks": checks,
        "passed": sum(1 for check in checks if check["holds"]),
        "total": len(checks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--maximum-loop-sim-s", type=float, default=120.0)
    parser.add_argument("--frames-per-action", type=int, default=FRAMES_PER_ACTION)
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=ROOT
        / "experiments/splitfusion_phase15_live_deployment_qualification_v1",
    )
    parser.add_argument("--reevaluate-from", type=Path, default=None)
    return parser


def _evaluate_all(
    campaign_root: Path, cells: Sequence[supervisor.Cell], frames_per_action: int
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for cell in cells:
        attempts = sorted((campaign_root / "cells" / cell.cell_id / "attempts").glob("attempt_*"))
        if not attempts:
            evaluated.append(
                {
                    "cell_id": cell.cell_id,
                    "action_id": cell.action_id,
                    "error": "no attempt directory",
                }
            )
            continue
        try:
            evaluated.append(
                evaluate_cell(attempts[-1], cell, frames_per_action)
            )
        except Exception as exc:
            evaluated.append(
                {
                    "cell_id": cell.cell_id,
                    "action_id": cell.action_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return evaluated


def _report(document: Mapping[str, Any]) -> str:
    lines = [
        "# Direct edge-to-map live validation",
        "",
        "Four actions under FAVORABLE_STABLE, each on a fresh CARLA world and a",
        "fresh OAI radio, torn down completely between actions.",
        "",
        "**This is a short live validation, not a Route-B loop completion run,**",
        "**and not an authorization for another 288-cell live campaign.**",
        "",
        "## Per-action evidence",
        "",
        "| Action | Family | Sent | Published | Installed | Publish->install p50 (ms) | Map age p50 (ms) | ACK delay p50 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def show(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    for row in document["cells"]:
        if row.get("error"):
            lines.append(
                f"| {row.get('action_id')} | - | - | - | - | - | - | ERROR: {row['error']} |"
            )
            continue
        lines.append(
            f"| {row['action_id']} | {row['model_family']} | {row['captures_sent']} | "
            f"{row['edge_direct_map_publications']} | {row['map_updates_installed']} | "
            f"{show((row['direct_publish_to_install_ms'] or {}).get('median'))} | "
            f"{show((row['map_age_at_install_ms'] or {}).get('median'), 1)} | "
            f"{show((row['ack_observation_delay_ms'] or {}).get('median'))} |"
        )
    lines.extend(["", "## Gates", "", "| Gate | Holds |", "|---|---|"])
    for check in document["gates"]["checks"]:
        lines.append(f"| {check['gate']} | {'yes' if check['holds'] else 'NO'} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Physical map freshness ends at the map install timestamp; the ACK",
            "  arrival at the UE is a separate controller-observation delay and is",
            "  never included in map-installation age.",
            "- `10.0.0.2` is a local address on this host (`oaitun_ue1`), so the",
            "  map -> UE feedback is delivered locally by the kernel and does not",
            "  traverse the radio. The ACK delay is therefore a lower bound on an",
            "  over-the-air control-plane delay.",
            "- The Route-B loop is deliberately budget-bounded; loop completion is",
            "  neither required nor claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _finalize(campaign_root: Path, document: dict[str, Any]) -> None:
    supervisor.atomic_json(campaign_root / "DIRECT_VALIDATION_RESULTS.json", document)
    (campaign_root / "REPORT.md").write_text(_report(document), encoding="utf-8")
    rows = [row for row in document["cells"] if not row.get("error")]
    if rows:
        flat_fields = [
            key
            for key, value in rows[0].items()
            if not isinstance(value, (dict, list))
        ]
        with (campaign_root / "direct_validation_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in flat_fields})
    names = [
        "DIRECT_VALIDATION_RESULTS.json",
        "REPORT.md",
        "direct_validation_summary.csv",
    ]
    hashes = {
        name: supervisor.sha256_file(campaign_root / name)
        for name in names
        if (campaign_root / name).is_file()
    }
    supervisor.atomic_json(
        campaign_root / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "sha256": hashes},
    )
    if document["gates"]["status"] == "PASS":
        (campaign_root / TERMINAL).write_text(
            json.dumps(
                {"schema": f"{SCHEMA}.terminal", "status": "PASS", "sha256": hashes},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config, cells, _hashes = supervisor.validate_static(config_path)
    selected = select_cells(cells)

    if args.reevaluate_from is not None:
        source = args.reevaluate_from.resolve(strict=True)
        destination = args.output_root.resolve(strict=False)
        require(not destination.exists(), f"create-only output exists: {destination}")
        destination.mkdir(parents=True, exist_ok=False)
        evaluated = _evaluate_all(source, selected, int(args.frames_per_action))
        document = {
            "schema": SCHEMA,
            "status": "REEVALUATED",
            "source_root": str(source.relative_to(ROOT)),
            "cells": evaluated,
            "gates": gate([row for row in evaluated if not row.get("error")]),
        }
        _finalize(destination, document)
        print(json.dumps(document["gates"], indent=2, sort_keys=True))
        return 0 if document["gates"]["status"] == "PASS" else 1

    require(args.execute == TOKEN, f"exact execution token is required: {TOKEN}")
    require(
        config.get("authorization", {}).get("campaign_288_authorized") is False
        and config.get("authorization", {}).get("another_288_live_campaign_authorized")
        is False,
        "another 288-cell live campaign must remain unauthorized",
    )
    require(DIRECT_ADAPTER.is_file(), f"direct adapter missing: {DIRECT_ADAPTER}")

    worktree = supervisor.verify_live_pilot_worktree()
    supervisor._phase15_gpu_audit()
    supervisor._require_phase15_application_cold(config)
    catalog = supervisor.read_catalog(config)
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_resolved_models(config, catalog)
    config["_maximum_loop_sim_s_override"] = float(args.maximum_loop_sim_s)

    campaign_root = args.output_root.resolve(strict=False)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        campaign_root.relative_to(experiments)
    except ValueError as exc:
        raise supervisor.CampaignError(
            "validation output must remain beneath experiments"
        ) from exc
    require(not campaign_root.exists(), f"create-only output exists: {campaign_root}")
    campaign_root.parent.mkdir(parents=True, exist_ok=True)
    campaign_root.mkdir(parents=False, exist_ok=False)

    manifest = {
        "schema": SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": supervisor.sha256_file(config_path),
        "direct_adapter_sha256": supervisor.sha256_file(DIRECT_ADAPTER),
        "git": worktree,
        "architecture": "DIRECT_EDGE_TO_MAP_V1",
        "matrix": [
            {
                "action_id": cell.action_id,
                "family": ACTION_FAMILY[cell.action_id],
                "cell_id": cell.cell_id,
                "profile_id": cell.profile_id,
                "network_profile_id": cell.network_profile_id,
            }
            for cell in selected
        ],
        "frames_per_action": int(args.frames_per_action),
        "maximum_loop_sim_s": float(args.maximum_loop_sim_s),
        "complete_route_b_loop_required": False,
        "another_288_live_campaign_authorized": False,
        "started_at_unix_s": time.time(),
    }
    supervisor.write_create_only(
        campaign_root / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    ledger_path = campaign_root / str(config["cell"]["resume_ledger"])
    ledger = supervisor.load_ledger(
        ledger_path, str(config["campaign_id"]), manifest["config_sha256"]
    )
    for cell in selected:
        rows = ledger["cells"].setdefault(cell.cell_id, [])
        print(f"[DIRECT] === cell {cell.cell_id} (action {cell.action_id}) ===", flush=True)
        result = supervisor.run_one_cell(
            config=config,
            cell=cell,
            adapter=DIRECT_ADAPTER,
            campaign_root=campaign_root,
            ledger_rows=rows,
            port=int(args.carla_port),
        )
        rows.append(result)
        ledger["updated_at_unix_s"] = time.time()
        supervisor.atomic_json(ledger_path, ledger)
        print(
            f"[DIRECT] cell {cell.cell_id} -> {result.get('status')}",
            flush=True,
        )
        # Every action runs so the evidence is complete; a failed action is
        # reported, not used to abandon the remaining actions.

    evaluated = _evaluate_all(campaign_root, selected, int(args.frames_per_action))
    document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "LIVE_MEASUREMENT",
        "architecture": "DIRECT_EDGE_TO_MAP_V1",
        "manifest": manifest,
        "cells": evaluated,
        "gates": gate([row for row in evaluated if not row.get("error")]),
        "limitations": [
            "short live validation; no complete Route-B loop is claimed",
            "map->UE feedback is host-local because 10.0.0.2 is local to this host",
            "does not authorize another 288-cell live campaign",
        ],
        "finished_at_unix_s": time.time(),
    }
    _finalize(campaign_root, document)
    print(json.dumps(document["gates"], indent=2, sort_keys=True))
    if document["gates"]["status"] == "PASS":
        print(TERMINAL)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
