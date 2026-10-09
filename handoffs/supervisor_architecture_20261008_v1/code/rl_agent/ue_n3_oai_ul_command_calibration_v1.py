#!/usr/bin/env python3
"""Prepare or execute the bounded UE-N3 RFsim command-calibration campaign.

The default mode is ``PREPARE_ONLY`` and the versioned config has no live OAI
or socket authority.  Live execution, when separately reviewed, runs exactly
one candidate command in each fresh RAN epoch.  Outputs can propose empirical
target-to-command pairs, but this phase can never promote a mapping or an SNR
bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3_oai_ul_command_calibration_v1.json"
PREPARE_ONLY = "PREPARE_ONLY"
EXECUTE_LIVE = "EXECUTE_LIVE"
PLAN_FROZEN = "UE_N3_COMMAND_CALIBRATION_PLAN_FROZEN_REVIEW_REQUIRED"
RUNG_CAPTURED = "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY"
RUNG_DETACHED = "DETACHED_BEFORE_TARGET_CONFIRMATION"
RUNG_IDENTITY_DISCONTINUITY = "RNTI_IDENTITY_DISCONTINUITY_BEFORE_TARGET_CONFIRMATION"
RUNG_HARD_LOSS = "HARD_SERVICE_LOSS_BEFORE_TARGET_CONFIRMATION"
RUNG_UNCONFIRMED = "UE_N3_COMMAND_RUNG_TARGET_WINDOW_UNCONFIRMED"
RUNG_RECOVERY_UNCONFIRMED = "UE_N3_CLEAN_RECOVERY_UNCONFIRMED"
CAMPAIGN_CAPTURED = "UE_N3_COMMAND_CALIBRATION_SEARCH_CAPTURED_PROPOSALS_ONLY"
CAMPAIGN_UNRESOLVED = "UE_N3_TARGET_CALIBRATION_UNRESOLVED"
RESTORE_FAILED = "UE_N3_FAILED_RESTORE"


class CalibrationFailure(RuntimeError):
    """Authority, integrity, instrumentation, or infrastructure failure."""


class HardServiceLoss(RuntimeError):
    """Expected command-side loss before an achieved-SNR tail is confirmed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationFailure(message)


def classify_service_loss_reason(reason: str) -> str:
    if reason == "UE_TUNNEL_IDENTITY_LOST":
        return RUNG_DETACHED
    if reason == "RNTI_CHANGED":
        return RUNG_IDENTITY_DISCONTINUITY
    return RUNG_HARD_LOSS


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise CalibrationFailure(f"path escapes repository root: {relative}") from exc
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    require(
        config.get("schema") == "scenesense.ue_n3_oai_ul_command_calibration_config.v1",
        "unexpected command-calibration schema",
    )
    authority = config["authority"]
    require(authority.get("offline_plan_authorized") is True,
            "offline plan authority is absent")
    for key in (
        "carla_run_authorized",
        "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    require(isinstance(authority.get("live_oai_run_authorized"), bool),
            "live OAI authority must be boolean")
    require(isinstance(authority.get("live_socket_execution_authorized"), bool),
            "live socket authority must be boolean")
    live_enabled = (
        authority["live_oai_run_authorized"]
        and authority["live_socket_execution_authorized"]
    )
    require(
        authority["live_oai_run_authorized"]
        == authority["live_socket_execution_authorized"],
        "live OAI and socket authority must change together",
    )
    expected_basis = (
        "USER_REQUEST_2026-08-21_CONTINUE_LOWER_OAI_SNR_SEARCH_"
        "AFTER_CLEAN_CONTROL_PASS"
        if live_enabled else "NOT_AUTHORIZED_PREPARE_ONLY"
    )
    require(authority.get("live_authority_basis") == expected_basis,
            "live authority basis is absent or unexpected")
    live = config["live_prerequisites"]
    require(
        live.get("clean_control_terminal")
        == "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED.json",
        "clean-control terminal name drift",
    )
    require(
        live.get("clean_control_required_status")
        == "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
        "clean-control required status drift",
    )
    require(live.get("clean_control_manifest") == "manifest.json",
            "clean-control manifest name drift")
    require(live.get("clean_control_required_mode") == "CLEAN_RECEIVER_CONTROL",
            "clean-control required mode drift")
    require(live.get("clean_control_runner_path") == "rl_agent/ue_n3_oai_ul_live_stage.py",
            "clean-control runner path drift")
    require(
        live.get("clean_control_config_path")
        == "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json",
        "clean-control config path drift",
    )
    require(
        re.fullmatch(
            r"[0-9a-f]{64}", str(live.get("clean_control_config_sha256", ""))
        ) is not None,
        "clean-control config seal is malformed",
    )
    require(live.get("clean_control_resolved_config") == "resolved_config.json",
            "clean-control resolved-config name drift")
    require(live.get("clean_control_required_restore_command_db") == "-50",
            "clean-control restore command drift")
    clean_config_seals = [
        str(seal["sha256"]) for seal in config["runtime_seals"]
        if seal["path"] == live["clean_control_config_path"]
    ]
    require(
        clean_config_seals == [live["clean_control_config_sha256"]],
        "clean-control config is not uniquely pinned by runtime seals",
    )

    campaign = config["campaign"]
    commands = [float(value) for value in campaign["commanded_noise_power_db"]]
    require(commands == [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
            "candidate command ladder drift")
    require(campaign["one_fresh_ran_per_rung"] is True,
            "each command requires a fresh RAN")
    require(campaign["one_candidate_application_per_rung"] is True,
            "each rung permits exactly one candidate application")
    require(campaign["stop_after_hard_loss"] is True,
            "campaign must stop after hard service loss")
    require([float(value) for value in campaign["desired_achieved_pusch_snr_db"]]
            == [6.0, 4.0, 3.0, 2.0], "desired achieved-SNR targets drift")

    rung = config["rung"]
    require(math.isclose(float(rung["clean_commanded_noise_power_db"]), -50.0),
            "clean command must be -50")
    require(math.isclose(float(rung["clean_lead_s"]), 5.0), "clean lead must be 5 s")
    require(math.isclose(float(rung["settle_s"]), 10.0), "settle must be 10 s")
    require(math.isclose(float(rung["measured_tail_s"]), 5.0),
            "measured tail must be 5 s")
    require(math.isclose(float(rung["clean_recovery_s"]), 5.0),
            "clean recovery must be 5 s")
    component_duration = sum(float(rung[key]) for key in (
        "clean_lead_s", "settle_s", "measured_tail_s", "clean_recovery_s"
    ))
    require(math.isclose(float(rung["service_duration_s"]), component_duration),
            "rung service duration must equal lead+settle+tail+recovery")
    traffic = config["traffic"]
    require(math.isclose(float(traffic["fps"]), 10.0), "traffic must be 10 Hz")
    require(int(rung["sender_frames"]) == round(component_duration * float(traffic["fps"])),
            "rung frame count does not match its 10-Hz duration")
    require(
        int(rung["expected_tail_frames"])
        == round(float(rung["measured_tail_s"]) * float(traffic["fps"]))
        == 50,
        "measured tail must contain exactly 50 scheduled frames",
    )
    require(int(rung["minimum_recovery_receiver_frames"]) >= 1,
            "post-restore application-delivery gate must require at least one frame")
    require(float(rung["receiver_capture_duration_s"]) > component_duration,
            "receiver requires a terminal capture margin")
    require(int(traffic["frame_bytes"]) == 12_500
            and int(traffic["chunk_bytes"]) == 12_500,
            "matched probe must use one 12500-byte datagram per frame")
    require(int(traffic["expected_chunks_per_frame"]) == 1,
            "matched probe must use one chunk per frame")
    require(int(traffic["remote_port"]) == int(config["preflight"]["structured_udp_port"]),
            "UDP port mismatch")
    gates = config["transport_gates"]
    require(math.isclose(float(gates["primary_complete_frame_ratio"]), 0.99),
            "primary transport threshold must be 99 percent")
    require([float(value) for value in gates["sensitivity_complete_frame_ratios"]]
            == [0.95, 0.90], "sensitivity thresholds drift")
    require(int(gates["required_valid_streams"]) == 1,
            "exactly one source stream is required")
    require(gates["expected_source_ip"] == "192.168.70.134",
            "frozen UPF-SNAT source must be 192.168.70.134")
    require(gates["required_stop_reason"] == "DURATION_COMPLETE",
            "receiver must complete its bounded duration")
    require(int(gates["maximum_stream_limit_exceeded_datagrams"]) == 0,
            "stream-limit exceptions are forbidden")
    require(config["analysis"]["direct_ul_bler_zero_fill_authorized"] is False,
            "UL BLER zero-fill is forbidden")
    if verify_hashes:
        for seal in config["runtime_seals"]:
            path = resolve_repo_path(str(seal["path"]))
            require(path.is_file(), f"sealed runtime file missing: {path}")
            require(n2.sha256(path) == seal["sha256"], f"runtime seal drift: {path}")


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rung = config["rung"]
    return [
        {
            "rung_index": index,
            "commanded_noise_power_db": float(command),
            "fresh_ran_epoch_required": True,
            "candidate_application_count": 1,
            "clean_start_command_db": float(rung["clean_commanded_noise_power_db"]),
            "clean_lead_s": float(rung["clean_lead_s"]),
            "settle_s": float(rung["settle_s"]),
            "measured_tail_s": float(rung["measured_tail_s"]),
            "clean_recovery_s": float(rung["clean_recovery_s"]),
            "sender_frames": int(rung["sender_frames"]),
            "status": "BLOCKED_PENDING_LIVE_AUTHORITY",
        }
        for index, command in enumerate(config["campaign"]["commanded_noise_power_db"])
    ]


def parse_live_pusch(row: tuple[int, int, str]) -> dict[str, Any] | None:
    wall_ns, mono_ns, line = row
    try:
        fields = next(csv.reader([line]))
        if len(fields) < 5:
            return None
        snr = float(fields[4]) / 10.0
        if not math.isfinite(snr):
            return None
        return {
            "wall_ns": int(wall_ns), "mono_ns": int(mono_ns), "time": fields[0],
            "rnti": int(fields[1]), "frame": int(fields[2]), "slot": int(fields[3]),
            "snr_db": snr,
        }
    except (ValueError, csv.Error, StopIteration):
        return None


def parse_live_mcs(row: tuple[int, int, str]) -> dict[str, Any] | None:
    wall_ns, mono_ns, line = row
    try:
        fields = next(csv.reader([line]))
        if len(fields) < 24:
            return None
        return {
            "wall_ns": int(wall_ns), "mono_ns": int(mono_ns), "time": fields[0],
            "rnti": int(fields[1]), "frame": int(fields[2]), "slot": int(fields[3]),
            "scheduler_ema_snr_db": float(fields[6]) / 10.0,
            "mcs_table": int(fields[7]), "selected_mcs": int(fields[9]),
            "final_mcs": int(fields[12]), "force_ul_mcs": int(fields[23]),
        }
    except (ValueError, csv.Error, StopIteration):
        return None


def summarize_tail(
    pusch_rows: Iterable[tuple[int, int, str]],
    mcs_rows: Iterable[tuple[int, int, str]],
    *,
    start_ns: int,
    end_ns: int,
    expected_rnti: int,
    minimum_pusch: int,
    minimum_mcs: int,
    required_mcs_table: int,
    required_force_mcs: int,
) -> dict[str, Any]:
    pusch_all = [row for row in (parse_live_pusch(item) for item in pusch_rows) if row]
    mcs_all = [row for row in (parse_live_mcs(item) for item in mcs_rows) if row]
    pusch = [row for row in pusch_all if start_ns <= row["mono_ns"] < end_ns]
    mcs = [row for row in mcs_all if start_ns <= row["mono_ns"] < end_ns]
    observed_rntis = sorted({row["rnti"] for row in [*pusch, *mcs]})
    pusch = [row for row in pusch if row["rnti"] == expected_rnti]
    mcs = [row for row in mcs if row["rnti"] == expected_rnti]
    pusch_values = [float(row["snr_db"]) for row in pusch]
    ema_values = [float(row["scheduler_ema_snr_db"]) for row in mcs]
    selected = [int(row["selected_mcs"]) for row in mcs]
    final = [int(row["final_mcs"]) for row in mcs]
    seals_ok = all(
        row["mcs_table"] == required_mcs_table
        and row["force_ul_mcs"] == required_force_mcs
        for row in mcs
    )
    accepted = (
        observed_rntis == [expected_rnti]
        and len(pusch) >= minimum_pusch
        and len(mcs) >= minimum_mcs
        and seals_ok
    )
    return {
        "status": "TAIL_ACCEPTED" if accepted else "TAIL_UNCONFIRMED",
        "timestamp_semantics": "COLLECTOR_INGEST_WINDOW_NOT_RF_APPLICATION_TIMESTAMP",
        "start_monotonic_ns": start_ns, "end_monotonic_ns": end_ns,
        "expected_rnti": expected_rnti, "observed_rntis": observed_rntis,
        "pusch_samples": len(pusch), "minimum_pusch_samples": minimum_pusch,
        "mcs_samples": len(mcs), "minimum_mcs_samples": minimum_mcs,
        "mcs_seals_ok": seals_ok, "mcs_table": required_mcs_table,
        "force_ul_mcs": required_force_mcs,
        "achieved_pusch_snr_db_median": statistics.median(pusch_values) if pusch_values else None,
        "achieved_pusch_snr_db_p05": n2.percentile(pusch_values, 0.05),
        "achieved_pusch_snr_db_p25": n2.percentile(pusch_values, 0.25),
        "achieved_pusch_snr_db_p75": n2.percentile(pusch_values, 0.75),
        "achieved_pusch_snr_db_p95": n2.percentile(pusch_values, 0.95),
        "scheduler_ema_snr_db_median": statistics.median(ema_values) if ema_values else None,
        "selected_mcs_median": statistics.median(selected) if selected else None,
        "final_mcs_median": statistics.median(final) if final else None,
        "selected_mcs_histogram": {str(value): selected.count(value) for value in sorted(set(selected))},
        "final_mcs_histogram": {str(value): final.count(value) for value in sorted(set(final))},
    }


def read_sender_completion(path: Path, expected_frames: int) -> dict[str, Any]:
    if not path.is_file():
        return {"complete": False, "reason": "SENDER_CSV_MISSING"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames: dict[int, set[int]] = {}
    malformed = 0
    for row in rows:
        try:
            frames.setdefault(int(row["frame_index"]), set()).add(int(row["chunk_index"]))
        except (KeyError, TypeError, ValueError):
            malformed += 1
    expected = set(range(expected_frames))
    observed = set(frames)
    return {
        "complete": (
            observed == expected and len(rows) == expected_frames and malformed == 0
            and all(chunks == {0} for chunks in frames.values())
        ),
        "rows": len(rows), "unique_frames": len(observed),
        "missing_frames": sorted(expected - observed),
        "outside_expected_frames": sorted(observed - expected),
        "malformed_rows": malformed,
    }


def evaluate_transport(
    receiver: Mapping[str, Any], sender: Mapping[str, Any], *,
    expected_frames: int, gates: Mapping[str, Any],
) -> dict[str, Any]:
    streams = list(receiver.get("streams", []))
    stream = streams[0] if len(streams) == 1 else {}
    ratio = int(stream.get("complete_frames", 0)) / expected_frames
    expected_source_ip = str(gates["expected_source_ip"])
    source_ok = str(stream.get("stream_id", "")).startswith(f"{expected_source_ip}:")
    observed_outage_gaps = int(stream.get("interarrival_gaps_over_one_second", -1))
    outage_free = (
        0 <= observed_outage_gaps
        <= int(gates["maximum_interarrival_gaps_gte_1s"])
    )
    integrity = (
        receiver.get("schema") == "scenesense.ue_n3_structured_udp_receiver_summary.v1"
        and receiver.get("status") == "CAPTURED"
        and bool(receiver.get("clean_shutdown"))
        and receiver.get("stop_reason") == gates["required_stop_reason"]
        and int(receiver.get("valid_stream_count", -1)) == int(gates["required_valid_streams"])
        and len(streams) == int(gates["required_valid_streams"])
        and source_ok and bool(sender.get("complete"))
        and int(receiver.get("malformed_datagrams", -1))
        <= int(gates["maximum_malformed_datagrams"])
        and int(receiver.get("stream_limit_exceeded_datagrams", -1))
        <= int(gates["maximum_stream_limit_exceeded_datagrams"])
        and int(stream.get("contract_mismatch_datagrams", -1))
        <= int(gates["maximum_contract_mismatch_datagrams"])
        and int(stream.get("outside_expected_range_datagrams", -1))
        <= int(gates["maximum_outside_expected_range_datagrams"])
    )
    thresholds = [
        float(gates["primary_complete_frame_ratio"]),
        *[float(value) for value in gates["sensitivity_complete_frame_ratios"]],
    ]
    return {
        "integrity_gate": integrity, "source_isolated": source_ok,
        "expected_source_ip": expected_source_ip,
        "receiver_stop_reason": receiver.get("stop_reason"),
        "complete_frame_ratio": ratio,
        "no_one_second_outage_pass": outage_free,
        "primary_99_pass": integrity and outage_free and ratio >= thresholds[0],
        "sensitivity_95_pass": integrity and outage_free and ratio >= thresholds[1],
        "sensitivity_90_pass": integrity and outage_free and ratio >= thresholds[2],
        "interarrival_gaps_gte_1s": observed_outage_gaps,
        "maximum_interarrival_gap_s": stream.get("max_interarrival_gap_s"),
        "sender_completion": dict(sender), "goodput_gate_applied": False,
    }


def evaluate_tail_service(
    sender_csv: Path,
    receiver_events_jsonl: Path,
    *,
    start_wall_ns: int,
    end_wall_ns: int,
    fps: float,
    expected_tail_frames: int,
    expected_source_ip: str,
    structural_integrity: bool,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify exactly the frozen set of command-tail frames.

    The health-poll loop may finish a few milliseconds late.  That scheduling
    overrun must not silently enlarge the five-second, 10-Hz statistical unit
    from 50 to 51 frames, so the classification end is derived from the frozen
    frame count rather than the observed function-return time.
    """

    require(end_wall_ns > start_wall_ns, "tail service window is empty")
    require(fps > 0 and expected_tail_frames > 0,
            "tail service requires positive FPS and frame count")
    with sender_csv.open(newline="", encoding="utf-8") as handle:
        sender_rows = list(csv.DictReader(handle))
    require(sender_rows, "tail service sender CSV is empty")
    sender_epoch_s = statistics.median(
        float(row["wall_time_s"]) - float(row["elapsed_s"])
        for row in sender_rows
    )
    nominal_duration_ns = int(round(expected_tail_frames / fps * 1e9))
    classification_end_wall_ns = start_wall_ns + nominal_duration_ns
    scheduled_rows = [
        (
            int(row["frame_index"]),
            int(round(
                (sender_epoch_s + float(row["scheduled_frame_time_s"])) * 1e9
            )),
        )
        for row in sender_rows
    ]
    expected_frame_rows = [
        (frame, scheduled_ns)
        for frame, scheduled_ns in scheduled_rows
        if start_wall_ns <= scheduled_ns < classification_end_wall_ns
    ]
    expected_frames = {frame for frame, _ in expected_frame_rows}
    exact_schedule = (
        len(expected_frame_rows) == expected_tail_frames
        and len(expected_frames) == expected_tail_frames
    )
    full_nominal_window_observed = end_wall_ns >= classification_end_wall_ns

    received_times: dict[int, int] = {}
    malformed_event_rows = 0
    with receiver_events_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed_event_rows += 1
                continue
            if (
                event.get("event_type") == "datagram"
                and event.get("status") == "ACCEPTED_UNIQUE"
                and event.get("source_ip") == expected_source_ip
            ):
                try:
                    frame = int(event["frame_index"])
                    observed_wall_ns = int(event["receiver_wall_time_ns"])
                except (KeyError, TypeError, ValueError):
                    malformed_event_rows += 1
                    continue
                if (
                    frame in expected_frames
                    and start_wall_ns
                    <= observed_wall_ns
                    < classification_end_wall_ns
                ):
                    received_times.setdefault(frame, observed_wall_ns)

    received_frames = set(received_times)
    missing_frames = sorted(expected_frames - received_frames)
    ratio = len(received_frames) / len(expected_frames) if expected_frames else 0.0
    clipped = sorted(
        max(start_wall_ns, min(classification_end_wall_ns, value))
        for value in received_times.values()
    )
    boundaries = [start_wall_ns, *clipped, classification_end_wall_ns]
    maximum_gap_s = max(
        (right - left) / 1e9 for left, right in zip(boundaries, boundaries[1:])
    )
    outage_free = maximum_gap_s < 1.0
    integrity = (
        structural_integrity
        and exact_schedule
        and full_nominal_window_observed
        and malformed_event_rows == 0
    )
    primary = float(gates["primary_complete_frame_ratio"])
    sensitivity = [float(value) for value in gates["sensitivity_complete_frame_ratios"]]
    return {
        "window_role": "COMMAND_CONDITION_MEASURED_TAIL",
        "start_wall_time_ns": start_wall_ns,
        "observed_end_wall_time_ns": end_wall_ns,
        "classification_end_wall_time_ns": classification_end_wall_ns,
        "observed_window_overrun_ns": max(
            0, end_wall_ns - classification_end_wall_ns
        ),
        "full_nominal_window_observed": full_nominal_window_observed,
        "expected_frames": len(expected_frames),
        "required_expected_frames": expected_tail_frames,
        "exact_frozen_frame_set_pass": exact_schedule,
        "expected_frame_indices": sorted(expected_frames),
        "received_frames": len(received_frames),
        "missing_frame_indices": missing_frames,
        "complete_frame_ratio": ratio,
        "maximum_interarrival_or_boundary_gap_s": maximum_gap_s,
        "no_one_second_outage_pass": outage_free,
        "event_parse_errors": malformed_event_rows,
        "integrity_gate": integrity,
        "primary_99_pass": integrity and outage_free and ratio >= primary,
        "sensitivity_95_pass": integrity and outage_free and ratio >= sensitivity[0],
        "sensitivity_90_pass": integrity and outage_free and ratio >= sensitivity[1],
    }


def verify_rung_evidence(
    rung_dir: Path,
    *,
    expected_status: str,
    expected_rung_index: int,
    expected_command_db: float,
    expected_candidate_applications: int,
    expected_config_sha256: str,
    expected_runner_sha256: str,
    require_clean_restore: bool,
) -> dict[str, Any]:
    """Verify one independently sealed rung before campaign aggregation."""

    manifest_path = rung_dir / "manifest.json"
    terminal_path = rung_dir / f"{expected_status}.json"
    summary_path = rung_dir / "rung_summary.json"
    require(manifest_path.is_file() and terminal_path.is_file() and summary_path.is_file(),
            f"rung evidence is incomplete: {rung_dir}")
    manifest = load_json(manifest_path)
    terminal = load_json(terminal_path)
    summary = load_json(summary_path)
    manifest_hash = n2.sha256(manifest_path)
    require(manifest.get("status") == expected_status,
            "rung manifest status mismatch")
    require(terminal.get("status") == expected_status,
            "rung terminal status mismatch")
    require(summary.get("status") == expected_status,
            "rung summary status mismatch")
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"), (summary, "summary")):
        require(int(payload.get("rung_index", -1)) == expected_rung_index,
                f"rung {label} index is not bound to the campaign plan")
        require(
            math.isclose(
                float(payload.get("commanded_noise_power_db", math.nan)),
                expected_command_db,
                abs_tol=1e-9,
            ),
            f"rung {label} command is not bound to the campaign plan",
        )
        require(
            int(payload.get("candidate_application_count", -1))
            == expected_candidate_applications,
            f"rung {label} candidate-application count mismatch",
        )
    require(terminal.get("manifest_sha256") == manifest_hash,
            "rung terminal/manifest hash mismatch")
    require(manifest.get("config_sha256") == expected_config_sha256,
            "rung config hash mismatch")
    require(manifest.get("runner_sha256") == expected_runner_sha256,
            "rung runner hash mismatch")
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"), (summary, "summary")):
        require(payload.get("target_mapping_promoted") is False,
                f"rung {label} unexpectedly promoted a mapping")
        require(payload.get("numeric_bound_promoted") is False,
                f"rung {label} unexpectedly promoted a bound")
    if require_clean_restore:
        require(terminal.get("clean_restore_verified") is True,
                "successful rung lacks verified -50 restoration")

    ran_epoch_id = str(manifest.get("ran_epoch_id", ""))
    control_session_id = str(manifest.get("control_session_id", ""))
    require(ran_epoch_id and control_session_id and ran_epoch_id != control_session_id,
            "rung lacks distinct fresh-RAN/control-session identities")
    for payload, label in ((terminal, "terminal"), (summary, "summary")):
        require(payload.get("ran_epoch_id") == ran_epoch_id,
                f"rung {label} RAN epoch does not match its manifest")
        require(payload.get("control_session_id") == control_session_id,
                f"rung {label} control session does not match its manifest")

    output_rows = list(manifest.get("outputs", []))
    require(output_rows, "rung manifest has no output inventory")
    seen: set[str] = set()
    for row in output_rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in seen,
                "rung manifest has a blank or duplicate output path")
        seen.add(relative)
        artifact = (rung_dir / relative).resolve()
        try:
            artifact.relative_to(rung_dir.resolve())
        except ValueError as exc:
            raise CalibrationFailure(f"rung output escapes its directory: {relative}") from exc
        require(artifact.is_file(), f"rung output missing: {relative}")
        require(artifact.stat().st_size == int(row.get("bytes", -1)),
                f"rung output byte-count drift: {relative}")
        require(n2.sha256(artifact) == row.get("sha256"),
                f"rung output hash drift: {relative}")
    require("rung_summary.json" in seen, "rung summary is absent from its manifest")
    require("cleanup_report.json" in seen,
            "rung cleanup report is absent from its manifest")
    cleanup = load_json(rung_dir / "cleanup_report.json")
    require(cleanup.get("clean") is True and not cleanup.get("errors"),
            "rung cleanup evidence is not clean")
    return {
        "status": "VERIFIED_RUNG_EVIDENCE",
        "rung_directory": str(rung_dir),
        "rung_status": expected_status,
        "rung_index": expected_rung_index,
        "commanded_noise_power_db": expected_command_db,
        "candidate_application_count": expected_candidate_applications,
        "ran_epoch_id": ran_epoch_id,
        "control_session_id": control_session_id,
        "config_sha256": expected_config_sha256,
        "runner_sha256": expected_runner_sha256,
        "manifest_sha256": manifest_hash,
        "terminal": terminal_path.name,
        "terminal_sha256": n2.sha256(terminal_path),
        "rung_summary_sha256": n2.sha256(summary_path),
    }


def evaluate_monotonicity(rows: Sequence[Mapping[str, Any]], tolerance_db: float) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("achieved_pusch_snr_db_median") is not None]
    comparisons = []
    consecutive = maximum = 0
    for previous, current in zip(accepted, accepted[1:]):
        prior = float(previous["achieved_pusch_snr_db_median"])
        value = float(current["achieved_pusch_snr_db_median"])
        violation = value > prior + tolerance_db
        consecutive = consecutive + 1 if violation else 0
        maximum = max(maximum, consecutive)
        comparisons.append({
            "previous_command_db": previous["commanded_noise_power_db"],
            "current_command_db": current["commanded_noise_power_db"],
            "previous_snr_db": prior, "current_snr_db": value,
            "violation": violation,
        })
    return {
        "status": "MONOTONE_WITHIN_TOLERANCE" if not any(row["violation"] for row in comparisons)
        else "NON_MONOTONIC_REVIEW_REQUIRED",
        "comparisons": comparisons,
        "maximum_consecutive_violations": maximum,
    }


def propose_mappings(
    rows: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    tolerance_db: float,
    *,
    config_sha256: str,
    runner_sha256: str,
    monotonicity: Mapping[str, Any],
) -> dict[str, Any]:
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    require(bool(sha_pattern.fullmatch(config_sha256)),
            "proposal config seal is malformed")
    require(bool(sha_pattern.fullmatch(runner_sha256)),
            "proposal runner seal is malformed")
    measured = sorted(
        [
            row for row in rows
            if row.get("achieved_pusch_snr_db_median") is not None
            and math.isfinite(float(row["achieved_pusch_snr_db_median"]))
        ],
        key=lambda row: float(row["commanded_noise_power_db"]),
    )

    def source(row: Mapping[str, Any]) -> dict[str, Any]:
        evidence = dict(row.get("rung_evidence", {}))
        tail = dict(row.get("tail") or {})
        service = dict(row.get("tail_service") or {})
        recovery = dict(row.get("clean_recovery") or {})
        for key in (
            "manifest_sha256", "terminal_sha256", "rung_summary_sha256",
        ):
            require(bool(sha_pattern.fullmatch(str(evidence.get(key, "")))),
                    f"proposal source has malformed {key}")
        require(evidence.get("config_sha256") == config_sha256,
                "proposal source config seal mismatch")
        require(evidence.get("runner_sha256") == runner_sha256,
                "proposal source runner seal mismatch")
        require(evidence.get("ran_epoch_id") and evidence.get("control_session_id"),
                "proposal source lacks RAN/control identities")
        require(tail.get("status") == "TAIL_ACCEPTED",
                "proposal source lacks an accepted radio tail")
        require(
            int(tail.get("pusch_samples", 0)) > 0
            and int(tail.get("mcs_samples", 0)) > 0
            and tail.get("start_monotonic_ns") is not None
            and tail.get("end_monotonic_ns") is not None,
            "proposal source radio-tail window is incomplete",
        )
        require(recovery.get("passed") is True,
                "proposal source lacks verified clean recovery")
        require(
            service.get("exact_frozen_frame_set_pass") is True
            and int(service.get("required_expected_frames", -1)) == 50,
            "proposal source lacks the exact 50-frame service unit",
        )
        return {
            "rung_index": int(row["rung_index"]),
            "commanded_noise_power_db": float(row["commanded_noise_power_db"]),
            "achieved_pusch_snr_db_median": float(
                row["achieved_pusch_snr_db_median"]
            ),
            "rung_manifest_sha256": evidence.get("manifest_sha256"),
            "rung_terminal_sha256": evidence.get("terminal_sha256"),
            "rung_summary_sha256": evidence.get("rung_summary_sha256"),
            "ran_epoch_id": evidence.get("ran_epoch_id"),
            "control_session_id": evidence.get("control_session_id"),
            "radio_tail_start_monotonic_ns": tail.get("start_monotonic_ns"),
            "radio_tail_end_monotonic_ns": tail.get("end_monotonic_ns"),
            "radio_tail_pusch_samples": tail.get("pusch_samples"),
            "radio_tail_mcs_samples": tail.get("mcs_samples"),
            "achieved_pusch_snr_db_p05": tail.get(
                "achieved_pusch_snr_db_p05"
            ),
            "achieved_pusch_snr_db_p95": tail.get(
                "achieved_pusch_snr_db_p95"
            ),
            "clean_recovery_status": recovery.get("status"),
            "tail_service_complete_frame_ratio": service.get("complete_frame_ratio"),
            "tail_service_primary_99_pass": service.get("primary_99_pass"),
            "tail_service_sensitivity_95_pass": service.get("sensitivity_95_pass"),
            "tail_service_sensitivity_90_pass": service.get("sensitivity_90_pass"),
        }

    proposals = []
    for target in targets:
        direct = sorted(
            measured,
            key=lambda row: (
                abs(float(row["achieved_pusch_snr_db_median"]) - target),
                float(row["commanded_noise_power_db"]),
            ),
        )
        direct = [
            row for row in direct
            if abs(float(row["achieved_pusch_snr_db_median"]) - target)
            <= tolerance_db
        ]
        if len(direct) == 1:
            selected = direct[0]
            error = abs(
                float(selected["achieved_pusch_snr_db_median"]) - target
            )
            proposals.append({
                "desired_achieved_pusch_snr_db": target,
                "status": (
                    "DIRECT_MEASURED_WITHIN_TOLERANCE_REPLICATION_REQUIRED"
                ),
                "candidate_commanded_noise_power_db": float(
                    selected["commanded_noise_power_db"]
                ),
                "proposed_commanded_noise_power_db": None,
                "observed_achieved_pusch_snr_db": float(
                    selected["achieved_pusch_snr_db_median"]
                ),
                "absolute_error_db": error,
                "within_tolerance": True,
                "requires_independent_replication": True,
                "sources": [source(selected)],
            })
            continue
        if len(direct) > 1:
            proposals.append({
                "desired_achieved_pusch_snr_db": target,
                "status": "AMBIGUOUS_MULTIPLE_DIRECT_MATCHES_REVIEW_REQUIRED",
                "candidate_commanded_noise_power_db": None,
                "proposed_commanded_noise_power_db": None,
                "within_tolerance": True,
                "requires_independent_replication": True,
                "sources": [source(row) for row in direct],
            })
            continue

        brackets = []
        for left, right in zip(measured, measured[1:]):
            if int(right["rung_index"]) - int(left["rung_index"]) != 1:
                continue
            left_snr = float(left["achieved_pusch_snr_db_median"])
            right_snr = float(right["achieved_pusch_snr_db_median"])
            if left_snr == right_snr:
                continue
            if min(left_snr, right_snr) <= target <= max(left_snr, right_snr):
                left_command = float(left["commanded_noise_power_db"])
                right_command = float(right["commanded_noise_power_db"])
                brackets.append({
                    "commanded_noise_power_db_interval": sorted([
                        left_command, right_command,
                    ]),
                    "achieved_pusch_snr_db_interval": sorted([
                        left_snr, right_snr,
                    ]),
                    "sources": [source(left), source(right)],
                })
        if len(brackets) == 1:
            proposals.append({
                "desired_achieved_pusch_snr_db": target,
                "status": "ADJACENT_MEASURED_BRACKET_REPLICATION_REQUIRED",
                "candidate_commanded_noise_power_db": None,
                "proposed_commanded_noise_power_db": None,
                "observed_achieved_pusch_snr_db": None,
                "within_tolerance": False,
                "requires_independent_replication": True,
                **brackets[0],
            })
        elif len(brackets) > 1:
            proposals.append({
                "desired_achieved_pusch_snr_db": target,
                "status": "AMBIGUOUS_MULTIPLE_BRACKETS_REVIEW_REQUIRED",
                "candidate_commanded_noise_power_db": None,
                "proposed_commanded_noise_power_db": None,
                "observed_achieved_pusch_snr_db": None,
                "within_tolerance": False,
                "requires_independent_replication": True,
                "bracket_candidates": brackets,
            })
        else:
            nearest = min(
                measured,
                key=lambda row: abs(
                    float(row["achieved_pusch_snr_db_median"]) - target
                ),
                default=None,
            )
            proposals.append({
                "desired_achieved_pusch_snr_db": target,
                "status": "UNBRACKETED_NO_MEASURED_CANDIDATE",
                "candidate_commanded_noise_power_db": None,
                "proposed_commanded_noise_power_db": None,
                "observed_achieved_pusch_snr_db": None,
                "within_tolerance": False,
                "requires_independent_replication": True,
                "nearest_source": source(nearest) if nearest is not None else None,
            })
    monotonicity_status = str(monotonicity.get("status", "MISSING"))
    return {
        "schema": "scenesense.ue_n3_command_mapping_proposals.v2",
        "status": (
            "PROPOSALS_ONLY_NOT_PROMOTED"
            if monotonicity_status == "MONOTONE_WITHIN_TOLERANCE"
            else "PROPOSALS_ONLY_MONOTONICITY_REVIEW_REQUIRED"
        ),
        "config_sha256": config_sha256,
        "runner_sha256": runner_sha256,
        "monotonicity_status": monotonicity_status,
        "target_mapping_promoted": False, "numeric_bound_promoted": False,
        "mapping_service_relationship": (
            "ORTHOGONAL_PUSCH_MAPPING_WITH_TAIL_SERVICE_REPORTED_NOT_USED_AS_A_GATE"
        ),
        "proposals": proposals,
    }


@dataclass
class EventTail:
    path: Path
    expected_source_ip: str
    offset: int = 0
    last_unique_ns: int | None = None
    accepted_count: int = 0

    def poll(self) -> int | None:
        if not self.path.exists():
            return self.last_unique_ns
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(start)
                    break
                self.offset = handle.tell()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    event.get("event_type") == "datagram"
                    and event.get("status") == "ACCEPTED_UNIQUE"
                    and event.get("source_ip") == self.expected_source_ip
                ):
                    self.last_unique_ns = int(event["receiver_monotonic_ns"])
                    self.accepted_count += 1
        return self.last_unique_ns


class RungRunner(n2.Runner):
    """Execute one candidate in one fresh RAN epoch and one output directory."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        *,
        rung_index: int,
        command_db: float,
        clean_control_proof: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config_path, output_dir)
        self.rung_index = int(rung_index)
        self.command_db = float(command_db)
        expected = [float(value) for value in self.config["campaign"]["commanded_noise_power_db"]]
        require(0 <= self.rung_index < len(expected) and expected[self.rung_index] == self.command_db,
                "rung index/command is not the frozen campaign pair")
        self.live_mcs: n2.LiveCsv | None = None
        self.ext_dn_pid: int | None = None
        self.sender: n2.ManagedProcess | None = None
        self.receiver: n2.ManagedProcess | None = None
        self.event_tail: EventTail | None = None
        self.traffic_start_ns: int | None = None
        self.application_count = 0
        self.nonclean_applied = False
        self.control_validated = False
        self.hard_loss_reason: str | None = None
        self.receiver_service_outage_detected = False
        self.receiver_service_outage_first_monotonic_ns: int | None = None
        self.last_carla_check_monotonic_ns: int | None = None
        self.clean_control_proof = (
            dict(clean_control_proof) if clean_control_proof is not None else None
        )

    def verify_dependencies(self) -> None:
        validate_config(self.config, verify_hashes=True)
        require(self.clean_control_proof is not None,
                "rung execution requires a verified clean-control proof")
        proof_dir = Path(str(self.clean_control_proof.get("directory", "")))
        # Re-run the full predecessor validator inside every fresh rung. This
        # prevents direct RungRunner use from bypassing the campaign gate and
        # catches any evidence drift between independent RAN epochs.
        validator = object.__new__(CampaignRunner)
        validator.config = self.config
        validator.output_dir = self.output_dir
        validator.clean_control_evidence = proof_dir
        observed_proof = CampaignRunner.verify_clean_control_predecessor(validator)
        require(
            observed_proof.get("terminal_sha256")
            == self.clean_control_proof.get("terminal_sha256")
            and observed_proof.get("manifest_sha256")
            == self.clean_control_proof.get("manifest_sha256"),
            "clean-control proof changed after campaign validation",
        )
        for key in ("authoritative_n3_plan", "ue_n2_evidence"):
            predecessor = self.config["predecessors"][key]
            directory = self.path(predecessor["directory"])
            manifest = directory / predecessor["manifest"]
            terminal = directory / predecessor["terminal"]
            require(manifest.is_file() and terminal.is_file(), f"missing predecessor {key}")
            require(n2.sha256(manifest) == predecessor["manifest_sha256"],
                    f"predecessor manifest drift: {key}")
            require(n2.sha256(terminal) == predecessor["terminal_sha256"],
                    f"predecessor terminal drift: {key}")
            require(load_json(terminal).get("status") == predecessor["required_status"],
                    f"predecessor status mismatch: {key}")
        # Keep this outside runtime/: N2 materialize_configs() creates that
        # directory atomically and intentionally refuses a pre-existing one.
        n2.atomic_json(self.output_dir / "runtime_seals.json", {
            "status": "MATCHED", "observed_at": n2.utc_now(),
            "files": [
                {"path": seal["path"], "expected_sha256": seal["sha256"],
                 "observed_sha256": n2.sha256(self.path(seal["path"]))}
                for seal in self.config["runtime_seals"]
            ],
        })

    @staticmethod
    def strict_port_free(port: int, kind: int) -> bool:
        sock = socket.socket(socket.AF_INET, kind)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def assert_carla_absent(self) -> None:
        require(self.config["preflight"]["fail_if_carla_active"] is True,
                "CARLA fail-closed gate disabled")
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,comm=,args="], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CalibrationFailure(f"CARLA detector failed closed: {exc}") from exc
        require(result.returncode == 0, "CARLA process detector failed closed")
        markers = [str(value).lower() for value in self.config["preflight"]["carla_process_markers"]]
        matches = []
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=2)
            comm = fields[1].lower() if len(fields) >= 2 else ""
            if any(marker in comm for marker in markers):
                matches.append(line.strip())
        busy = [
            int(port) for port in self.config["preflight"]["carla_ports"]
            if not self.strict_port_free(int(port), socket.SOCK_STREAM)
            or not self.strict_port_free(int(port), socket.SOCK_DGRAM)
        ]
        evidence = {"process_matches": matches, "busy_ports": busy, "checked_at": n2.utc_now()}
        # N2 materialize_configs() owns creation of runtime/ and intentionally
        # refuses a pre-existing directory.  Keep pre-materialization gates at
        # the rung root so the fail-closed CARLA check cannot poison startup.
        n2.atomic_json(self.output_dir / "carla_absent_gate.json", evidence)
        require(not matches and not busy, f"CARLA_ACTIVE_FAIL_CLOSED: {evidence}")
        self.last_carla_check_monotonic_ns = time.monotonic_ns()

    def namespace_udp_busy(self) -> bool:
        require(self.ext_dn_pid is not None, "ext-DN PID unavailable")
        output = n2.run_checked([
            "sudo", "-n", "nsenter", "-t", str(self.ext_dn_pid), "-n", "ss", "-H", "-lun",
        ], timeout=10).stdout
        port = int(self.config["traffic"]["remote_port"])
        return any(re.search(rf":{port}\b", line) for line in output.splitlines())

    def preflight(self) -> None:
        self.verify_dependencies()
        authority = self.config["authority"]
        require(authority["live_oai_run_authorized"] is True, "live OAI authority absent")
        require(authority["live_socket_execution_authorized"] is True,
                "live socket authority absent")
        self.assert_carla_absent()
        n2.run_checked([
            str(self.path(self.config["paths"]["python"])), "-m",
            "rl_agent.ue_n1_freeze_oai_ul_actuator_v2", "--validate",
            str(self.path(self.config["predecessors"]["ue_n1_bundle"])),
        ])
        n2.run_checked(["sudo", "-n", "true"])
        for container in self.config["radio"]["core_containers"]:
            state = n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}", container,
            ]).stdout.strip()
            require(state.startswith("true") and "unhealthy" not in state,
                    f"core container not ready: {container}={state!r}")
        for process_name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(
                ["sudo", "-n", "pgrep", "-a", "-x", process_name], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(found.returncode != 0 or not found.stdout.strip(),
                    f"cold-RAN gate failed: {found.stdout.strip()}")
        require(not n2.oai_tunnel_interfaces(), "cold-RAN gate found stale tunnel")
        busy = [
            int(port) for port in self.config["preflight"]["required_host_tcp_ports"]
            if not n2.port_is_free(int(port))
        ]
        require(not busy, f"required host ports busy: {busy}")
        self.ext_dn_pid = int(n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
            self.config["radio"]["ext_dn_container"],
        ]).stdout.strip())
        require(not self.namespace_udp_busy(), "ext-DN structured UDP port busy")

    def start_telemetry(self) -> None:
        super().start_telemetry()
        telemetry = self.config["telemetry"]
        troot = self.path("OAI/openairinterface5g/common/utils/T/tracer")
        fields = (
            "time", "rnti", "frame", "slot", "sched_frame", "sched_slot",
            "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
            "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer",
            "sched_ul_bytes", "B", "min_rb", "available_rb_before",
            "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
            "force_ul_mcs",
        )
        command = [
            str(troot / "csv"), "-d", str(self.path(self.config["paths"]["t_messages"])),
            "-ip", "127.0.0.1", "-p", str(telemetry["gnb_relay_port"]),
            "-f", "-s", ",", "-t", "time", "GNB_MAC_UL_MCS_DECISION", *fields,
        ]
        self.live_mcs = n2.LiveCsv(command, self.output_dir / "ttracer/gnb/live_mcs.csv")

    def tunnel_ip(self) -> str | None:
        result = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "dev", self.config["radio"]["ue_interface"]],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        if result.returncode != 0:
            return None
        try:
            rows = json.loads(result.stdout)
            addresses = [
                str(info["local"]) for row in rows for info in row.get("addr_info", [])
                if info.get("family") == "inet" and info.get("local")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        return addresses[0] if len(addresses) == 1 else None

    def observed_rntis(self) -> set[int]:
        if self.live_csv is None:
            return set()
        return {
            int(row["rnti"]) for row in
            (parse_live_pusch(item) for item in self.live_csv.snapshot()) if row is not None
        }

    def latest_pusch_ns(self) -> int | None:
        if self.live_csv is None or self.current_rnti is None:
            return None
        values = [
            int(row["mono_ns"]) for row in
            (parse_live_pusch(item) for item in self.live_csv.snapshot())
            if row is not None and row["rnti"] == self.current_rnti
        ]
        return max(values) if values else None

    def check_health(self, *, enforce_silence: bool, sender_required: bool = True) -> None:
        now = time.monotonic_ns()
        if (
            self.last_carla_check_monotonic_ns is None
            or now - self.last_carla_check_monotonic_ns >= 1_000_000_000
        ):
            self.assert_carla_absent()
        for name in ("gnb", "ue"):
            process = next((item for item in self.processes if item.name == name), None)
            require(process is not None and process.process.poll() is None,
                    f"RAN process exited: {name}")
        require(self.live_csv is not None and self.live_csv.process.poll() is None,
                "live PUSCH collector exited")
        require(self.live_csv.thread.is_alive(),
                "live PUSCH collector drain thread exited")
        require(self.live_mcs is not None and self.live_mcs.process.poll() is None,
                "live MCS collector exited")
        require(self.live_mcs.thread.is_alive(),
                "live MCS collector drain thread exited")
        if self.tunnel_ip() != self.ue_ip:
            raise HardServiceLoss("UE_TUNNEL_IDENTITY_LOST")
        if self.current_rnti is not None and any(
            value != self.current_rnti for value in self.observed_rntis()
        ):
            raise HardServiceLoss("RNTI_CHANGED")
        require(self.receiver is not None and self.receiver.process.poll() is None,
                "structured receiver exited early")
        if sender_required:
            require(self.sender is not None and self.sender.process.poll() is None,
                    "structured sender exited early")
        if enforce_silence and self.traffic_start_ns is not None:
            threshold = int(float(self.config["rung"]["hard_loss_silence_s"]) * 1e9)
            now = time.monotonic_ns()
            receiver_last = self.event_tail.poll() if self.event_tail else None
            if now - (receiver_last or self.traffic_start_ns) >= threshold:
                self.receiver_service_outage_detected = True
                if self.receiver_service_outage_first_monotonic_ns is None:
                    self.receiver_service_outage_first_monotonic_ns = now
            if now - (self.latest_pusch_ns() or self.traffic_start_ns) >= threshold:
                if receiver_last is not None and now - receiver_last < threshold:
                    raise CalibrationFailure(
                        "PUSCH telemetry stale while expected-source receiver "
                        "delivery remains fresh"
                    )
                raise HardServiceLoss("CURRENT_RNTI_PUSCH_SILENCE")

    def wait_for(self, duration_s: float, *, enforce_silence: bool) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.check_health(enforce_silence=enforce_silence)
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def wait_ready(self, path: Path) -> dict[str, Any]:
        require(self.receiver is not None, "receiver process unavailable")
        deadline = time.monotonic() + float(self.config["traffic"]["receiver_ready_timeout_s"])
        while time.monotonic() < deadline:
            require(self.receiver.process.poll() is None, "receiver exited before READY")
            if path.is_file():
                ready = load_json(path)
                require(ready.get("schema") == "scenesense.ue_n3_structured_udp_receiver_ready.v1",
                        "receiver READY schema mismatch")
                require(ready.get("status") == "READY", "receiver READY status mismatch")
                require(ready.get("bind_host") == self.config["radio"]["ext_dn_ip"],
                        "receiver bind-host mismatch")
                require(int(ready.get("port", -1)) == int(self.config["traffic"]["remote_port"]),
                        "receiver port mismatch")
                return ready
            time.sleep(0.05)
        raise CalibrationFailure("receiver READY timeout")

    def start_probe(self) -> None:
        require(self.ue_ip is not None and self.ext_dn_pid is not None,
                "probe requires UE IP and ext-DN PID")
        rung, traffic = self.config["rung"], self.config["traffic"]
        directory = self.output_dir / "traffic"
        directory.mkdir(parents=True, exist_ok=True)
        events, summary, ready = (
            directory / "receiver_events.jsonl",
            directory / "receiver_summary.json",
            directory / "receiver_ready.json",
        )
        receiver = [
            "sudo", "-n", "nsenter", "-t", str(self.ext_dn_pid), "-n", "/usr/bin/python3",
            str(self.path(self.config["paths"]["receiver"])),
            "--bind-host", self.config["radio"]["ext_dn_ip"],
            "--port", str(traffic["remote_port"]), "--events-jsonl", str(events),
            "--summary-json", str(summary), "--ready-json", str(ready),
            "--duration-s", str(rung["receiver_capture_duration_s"]),
            "--expected-first-frame", "0", "--expected-frames", str(rung["sender_frames"]),
            "--expected-chunks-per-frame", "1", "--max-streams", "1",
            "--reorder-window-frames", str(traffic["receiver_reorder_window_frames"]),
            "--max-chunks-per-frame", "1", "--socket-receive-buffer-bytes",
            str(traffic["receiver_socket_buffer_bytes"]),
        ]
        self.receiver = self.spawn("structured_receiver", receiver, "logs/receiver.log", root_owned=True)
        ready_payload = self.wait_ready(ready)
        self.assert_carla_absent()
        sender = [
            str(self.path(self.config["paths"]["python"])),
            str(self.path(self.config["paths"]["sender"])),
            "--bind-host", self.ue_ip, "--remote-host", self.config["radio"]["ext_dn_ip"],
            "--remote-port", str(traffic["remote_port"]), "--fps", str(traffic["fps"]),
            "--frames", str(rung["sender_frames"]), "--frame-bytes", str(traffic["frame_bytes"]),
            "--chunk-bytes", str(traffic["chunk_bytes"]), "--idle-before-s", "0",
            "--cooldown-s", "0", "--log-csv", str(directory / "sender.csv"),
        ]
        self.traffic_start_ns = time.monotonic_ns()
        self.sender = self.spawn("structured_sender", sender, "logs/sender.log")
        self.event_tail = EventTail(
            events,
            expected_source_ip=str(
                self.config["transport_gates"]["expected_source_ip"]
            ),
        )
        n2.atomic_json(directory / "launch.json", {
            "receiver_ready": ready_payload, "sender_launch_monotonic_ns": self.traffic_start_ns,
            "commanded_noise_power_db": self.command_db,
        })

    def establish_clean_lead(self) -> None:
        require(self.live_csv is not None and self.traffic_start_ns is not None,
                "clean-lead anchors unavailable")
        lead_end = self.traffic_start_ns + int(float(self.config["rung"]["clean_lead_s"]) * 1e9)
        while time.monotonic_ns() < lead_end:
            self.check_health(enforce_silence=False)
            parsed = [
                row for row in (parse_live_pusch(item) for item in self.live_csv.snapshot())
                if row is not None and row["mono_ns"] >= self.traffic_start_ns
            ]
            rntis = {row["rnti"] for row in parsed}
            if len(parsed) >= 5 and len(rntis) == 1 and self.event_tail.poll() is not None:
                self.current_rnti = next(iter(rntis))
            time.sleep(0.05)
        require(self.current_rnti is not None, "clean lead lacked current-RNTI PUSCH")

    def apply_candidate_once(self, model_index: int) -> dict[str, Any]:
        require(self.telnet is not None, "control session unavailable")
        require(self.application_count == 0, "candidate command may be applied only once per rung")
        self.assert_carla_absent()
        target = f"{self.command_db:.1f}"
        # Treat entry into the control send as a conservative application
        # attempt. TelnetSession may raise only after sendall() succeeds, so an
        # ACK timeout must still force a -50 restoration attempt and must never
        # be reported as zero candidate applications.
        self.application_count = 1
        self.nonclean_applied = True
        row = {
            "rung_index": self.rung_index, "commanded_noise_power_db": self.command_db,
            "candidate_application_index": 0,
            "candidate_application_attempted_once": True,
            "attempt_started_monotonic_ns": time.monotonic_ns(),
            "attempt_started_wall_time_ns": time.time_ns(),
            "status": "SEND_ATTEMPT_STARTED_ACK_UNCONFIRMED",
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
        }
        self.command_rows.append(row)
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(
            f"channelmod modify {model_index} noise_power_dB {target}"
        )
        row.update({
            "send_monotonic_ns": sent_mono,
            "send_wall_time_ns": sent_wall,
            "response_received_monotonic_ns": ack_mono,
            "response_received_wall_time_ns": ack_wall,
            "handler_bracket_ms": (ack_mono - sent_mono) / 1e6,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "status": "ACK_RECEIVED_VALIDATION_PENDING",
        })
        self.validate_modify_response(response, target)
        row["status"] = "ACK_VALIDATED_POST_STATE_PENDING"
        _, _, _, _, state = self.telnet.command("channelmod show current")
        model = n2.parse_channel_models(state).get(
            self.config["actuator"]["channel_model_name"], {}
        )
        require(int(model.get("model_index", -1)) == model_index,
                "post-command model index mismatch")
        require(model.get("model_type") == self.config["actuator"]["channel_model_type"],
                "post-command model type mismatch")
        require(model.get("owner") == self.config["actuator"]["channel_model_owner"],
                "post-command model owner mismatch")
        require(abs(float(model.get("path_loss_db", math.nan))
                    - float(self.config["actuator"]["path_loss_db"])) <= 1e-6,
                "post-command path-loss mismatch")
        require(abs(float(model.get("noise_power_db", math.nan)) - self.command_db) <= 1e-6,
                "post-command noise state mismatch")
        n2.atomic_text(self.output_dir / "channel_state_candidate_applied.txt", state)
        row["post_apply_state_sha256"] = hashlib.sha256(state.encode()).hexdigest()
        row["candidate_applied_once"] = True
        row["status"] = "ACK_AND_POST_STATE_VALIDATED_ONCE"
        return row

    def verify_recovery(
        self,
        baseline_count: int,
        receiver_baseline_count: int,
        *,
        required: bool = True,
    ) -> dict[str, Any]:
        require(
            self.live_csv is not None
            and self.live_mcs is not None
            and self.current_rnti is not None
            and self.event_tail is not None,
            "recovery evidence anchors unavailable",
        )
        pre_restore_rnti = self.current_rnti
        minimum_pusch = int(self.config["rung"]["minimum_recovery_pusch_samples"])
        minimum_receiver = int(
            self.config["rung"]["minimum_recovery_receiver_frames"]
        )
        deadline = time.monotonic() + float(self.config["rung"]["clean_recovery_s"])
        parsed: list[dict[str, Any]] = []
        tunnel_recovered = False
        observed_rntis: set[int] = set()
        while time.monotonic() < deadline:
            now = time.monotonic_ns()
            if (
                self.last_carla_check_monotonic_ns is None
                or now - self.last_carla_check_monotonic_ns >= 1_000_000_000
            ):
                self.assert_carla_absent()
            for name in ("gnb", "ue"):
                process = next((item for item in self.processes if item.name == name), None)
                require(process is not None and process.process.poll() is None,
                        f"RAN process exited during recovery: {name}")
            require(self.live_csv.process.poll() is None,
                    "live PUSCH collector exited during recovery")
            require(self.live_csv.thread.is_alive(),
                    "live PUSCH collector drain thread exited during recovery")
            require(self.live_mcs.process.poll() is None,
                    "live MCS collector exited during recovery")
            require(self.live_mcs.thread.is_alive(),
                    "live MCS collector drain thread exited during recovery")
            require(self.receiver is not None and self.receiver.process.poll() is None,
                    "structured receiver exited during recovery")
            parsed = [
                row for row in
                (parse_live_pusch(item) for item in self.live_csv.snapshot()[baseline_count:])
                if row is not None
            ]
            observed_rntis = {int(row["rnti"]) for row in parsed}
            tunnel_recovered = self.tunnel_ip() == self.ue_ip
            self.event_tail.poll()
            time.sleep(0.05)
        stable_tail = parsed[-minimum_pusch:]
        stable_rntis = {int(row["rnti"]) for row in stable_tail}
        recovered_rnti = (
            next(iter(stable_rntis))
            if len(stable_tail) == minimum_pusch and len(stable_rntis) == 1
            else None
        )
        fresh = (
            [row for row in parsed if int(row["rnti"]) == recovered_rnti]
            if recovered_rnti is not None else []
        )
        receiver_frames = self.event_tail.accepted_count - receiver_baseline_count
        radio_recovery_passed = (
            tunnel_recovered
            and recovered_rnti is not None
            and len(fresh) >= minimum_pusch
        )
        application_delivery_passed = receiver_frames >= minimum_receiver
        passed = (
            radio_recovery_passed and application_delivery_passed
        )
        if passed:
            self.current_rnti = recovered_rnti
        result = {
            "status": "CLEAN_RECOVERY_PASSED" if passed else "CLEAN_RECOVERY_UNCONFIRMED",
            "required": required,
            "passed": passed,
            "radio_recovery_passed": radio_recovery_passed,
            "application_delivery_passed": application_delivery_passed,
            "pusch_samples": len(fresh),
            "minimum_pusch_samples": minimum_pusch,
            "post_restore_receiver_frames": receiver_frames,
            "minimum_post_restore_receiver_frames": minimum_receiver,
            "pre_restore_rnti": pre_restore_rnti,
            "recovered_rnti": recovered_rnti,
            "rnti_replaced_after_restore": (
                recovered_rnti is not None and recovered_rnti != pre_restore_rnti
            ),
            "observed_rntis": sorted(observed_rntis),
            "stable_tail_rntis": sorted(stable_rntis),
            "tunnel_recovered": tunnel_recovered,
            "median_snr_db": (
                statistics.median(row["snr_db"] for row in fresh) if fresh else None
            ),
        }
        n2.atomic_json(self.output_dir / "clean_recovery.json", result)
        if required:
            require(passed, f"clean -50 recovery PUSCH gate failed: {result}")
        return result

    def stop_probe_early(self) -> None:
        for process in (self.sender, self.receiver):
            if process is not None and process in self.processes:
                process.stop()
                self.processes.remove(process)

    def finish_probe(self, *, allow_partial_sender: bool = False) -> dict[str, Any]:
        require(self.sender is not None and self.receiver is not None, "probe processes unavailable")
        sender_timed_out = False
        if self.sender.process.poll() is None:
            elapsed_s = (
                (time.monotonic_ns() - self.traffic_start_ns) / 1e9
                if self.traffic_start_ns is not None else 0.0
            )
            sender_timeout_s = max(
                5.0,
                float(self.config["rung"]["service_duration_s"]) - elapsed_s + 5.0,
            )
            try:
                self.sender.process.wait(timeout=sender_timeout_s)
            except subprocess.TimeoutExpired as exc:
                if not allow_partial_sender:
                    raise CalibrationFailure("sender exceeded bounded rung") from exc
                sender_timed_out = True
                self.sender.stop()
        sender_returncode = self.sender.process.returncode
        if not allow_partial_sender:
            require(sender_returncode == 0, f"sender exited rc={sender_returncode}")
        if self.receiver.process.poll() is None:
            elapsed_s = (
                (time.monotonic_ns() - self.traffic_start_ns) / 1e9
                if self.traffic_start_ns is not None else 0.0
            )
            receiver_timeout_s = max(
                5.0,
                float(self.config["rung"]["receiver_capture_duration_s"])
                - elapsed_s + 5.0,
            )
            try:
                self.receiver.process.wait(timeout=receiver_timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise CalibrationFailure("receiver exceeded capture margin") from exc
        require(self.receiver.process.returncode == 0, f"receiver exited rc={self.receiver.process.returncode}")
        sender = read_sender_completion(
            self.output_dir / "traffic/sender.csv", int(self.config["rung"]["sender_frames"])
        )
        sender.update({
            "partial_sender_allowed_after_service_loss": allow_partial_sender,
            "process_returncode": sender_returncode,
            "bounded_wait_timed_out": sender_timed_out,
        })
        if not allow_partial_sender:
            require(sender["complete"], f"sender frame contract failed: {sender}")
        summary_path = self.output_dir / "traffic/receiver_summary.json"
        require(summary_path.is_file(), "receiver summary missing")
        transport = evaluate_transport(
            load_json(summary_path), sender,
            expected_frames=int(self.config["rung"]["sender_frames"]),
            gates=self.config["transport_gates"],
        )
        n2.atomic_json(self.output_dir / "transport_summary.json", transport)
        return transport

    def cleanup(self, *, strict: bool = False) -> list[str]:
        report_path = self.output_dir / "cleanup_report.json"
        prior_report = load_json(report_path) if report_path.exists() else {}
        prior_clean = prior_report.get("clean") is not False
        prior_errors = [str(value) for value in prior_report.get("errors", [])]
        errors: list[str] = []
        if self.live_mcs is not None:
            try:
                self.live_mcs.stop()
            except Exception as exc:
                errors.append(f"live MCS stop: {exc}")
            self.live_mcs = None
        errors.extend(super().cleanup(strict=False))
        if self.ext_dn_pid is not None:
            try:
                if self.namespace_udp_busy():
                    errors.append("ext-DN structured UDP port remains busy")
            except Exception as exc:
                errors.append(f"namespace UDP cleanup gate: {exc}")
        report = load_json(report_path) if report_path.exists() else {}
        current_clean = report.get("clean") is not False
        combined_errors = list(dict.fromkeys([
            *prior_errors,
            *[str(value) for value in report.get("errors", [])],
            *errors,
        ]))
        report.update({
            "clean": prior_clean and current_clean and not combined_errors,
            "errors": combined_errors,
            "checked_at": n2.utc_now(),
            "prior_dirty_state_preserved": not prior_clean or bool(prior_errors),
        })
        n2.atomic_json(report_path, report)
        if strict and combined_errors:
            raise CalibrationFailure(
                "cleanup failed: " + "; ".join(combined_errors)
            )
        return combined_errors

    def write_manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(self.output_dir / self.config["output"]["rung_summary"], summary)
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "FAILED.json"} \
                    and not path.name.startswith("UE_N3_") and path.name != RUNG_HARD_LOSS + ".json":
                files.append({"path": str(path.relative_to(self.output_dir)),
                              "bytes": path.stat().st_size, "sha256": n2.sha256(path)})
        manifest = {
            "schema": "scenesense.ue_n3_command_calibration_rung_manifest.v1",
            "status": status, "rung_index": self.rung_index,
            "commanded_noise_power_db": self.command_db,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "ran_epoch_id": self.ran_epoch_id, "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "target_mapping_promoted": False, "numeric_bound_promoted": False,
            "outputs": files,
        }
        n2.atomic_json(self.output_dir / "manifest.json", manifest)
        terminal = {
            **dict(summary),
            "status": status, "rung_index": self.rung_index,
            "commanded_noise_power_db": self.command_db,
            "clean_restore_verified": self.restored,
            "candidate_application_count": self.application_count,
            "target_mapping_promoted": False, "numeric_bound_promoted": False,
            "manifest_sha256": n2.sha256(self.output_dir / "manifest.json"),
        }
        n2.atomic_json(self.output_dir / f"{status}.json", terminal)

    def run(self) -> int:
        n2.atomic_json(self.output_dir / "resolved_config.json", {
            **self.config,
            "resolved_rung": {"rung_index": self.rung_index,
                              "commanded_noise_power_db": self.command_db},
        })
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise CalibrationFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught] = signal.getsignal(caught)
            signal.signal(caught, terminate)
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            self.wait_attach()
            n2.wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            self.start_telemetry()
            time.sleep(0.75)
            model_index = self.open_and_validate_telnet()
            self.control_validated = True
            self.start_probe()
            self.establish_clean_lead()
            command = self.apply_candidate_once(model_index)
            status = RUNG_CAPTURED
            service_loss_status: str | None = None
            tail: dict[str, Any] | None = None
            tail_start_wall_ns: int | None = None
            tail_end_wall_ns: int | None = None
            try:
                self.wait_for(float(self.config["rung"]["settle_s"]), enforce_silence=True)
                tail_start = time.monotonic_ns()
                tail_start_wall_ns = time.time_ns()
                frozen_tail_duration_ns = int(round(
                    float(self.config["rung"]["measured_tail_s"]) * 1e9
                ))
                frozen_tail_end = tail_start + frozen_tail_duration_ns
                frozen_tail_end_wall_ns = (
                    tail_start_wall_ns + frozen_tail_duration_ns
                )
                self.wait_for(float(self.config["rung"]["measured_tail_s"]), enforce_silence=True)
                observed_tail_end = time.monotonic_ns()
                observed_tail_end_wall_ns = time.time_ns()
                tail_end_wall_ns = frozen_tail_end_wall_ns
                tail = summarize_tail(
                    self.live_csv.snapshot() if self.live_csv else [],
                    self.live_mcs.snapshot() if self.live_mcs else [],
                    start_ns=tail_start, end_ns=frozen_tail_end,
                    expected_rnti=int(self.current_rnti),
                    minimum_pusch=int(self.config["rung"]["minimum_tail_pusch_samples"]),
                    minimum_mcs=int(self.config["rung"]["minimum_tail_mcs_samples"]),
                    required_mcs_table=int(self.config["analysis"]["scheduler_required_mcs_table"]),
                    required_force_mcs=int(self.config["analysis"]["scheduler_required_force_ul_mcs"]),
                )
                tail["start_wall_time_ns"] = tail_start_wall_ns
                tail["end_wall_time_ns"] = tail_end_wall_ns
                tail["observed_wait_end_monotonic_ns"] = observed_tail_end
                tail["observed_wait_end_wall_time_ns"] = observed_tail_end_wall_ns
                tail["wait_overrun_ns"] = max(
                    0, observed_tail_end - frozen_tail_end
                )
                if tail["status"] != "TAIL_ACCEPTED":
                    status = RUNG_UNCONFIRMED
            except HardServiceLoss as exc:
                self.hard_loss_reason = str(exc)
                service_loss_status = classify_service_loss_reason(
                    self.hard_loss_reason
                )
                status = service_loss_status

            self.restore(model_index)
            recovery_baseline = self.live_csv.count() if self.live_csv else 0
            if self.event_tail is not None:
                self.event_tail.poll()
            receiver_recovery_baseline = (
                self.event_tail.accepted_count if self.event_tail is not None else 0
            )
            recovery = self.verify_recovery(
                recovery_baseline,
                receiver_recovery_baseline,
                required=False,
            )
            status_before_recovery_gate = status
            if not recovery["passed"]:
                status = RUNG_RECOVERY_UNCONFIRMED
            transport = self.finish_probe(
                allow_partial_sender=service_loss_status is not None,
            )
            tail_service = None
            if tail_start_wall_ns is not None and tail_end_wall_ns is not None:
                tail_service = evaluate_tail_service(
                    self.output_dir / "traffic/sender.csv",
                    self.output_dir / "traffic/receiver_events.jsonl",
                    start_wall_ns=tail_start_wall_ns,
                    end_wall_ns=tail_end_wall_ns,
                    fps=float(self.config["traffic"]["fps"]),
                    expected_tail_frames=int(
                        self.config["rung"]["expected_tail_frames"]
                    ),
                    expected_source_ip=str(
                        self.config["transport_gates"]["expected_source_ip"]
                    ),
                    structural_integrity=bool(transport["integrity_gate"]),
                    gates=self.config["transport_gates"],
                )
                n2.atomic_json(self.output_dir / "tail_service_summary.json", tail_service)
            if status == RUNG_CAPTURED:
                require(transport["integrity_gate"],
                        "captured rung failed structured transport integrity")
            self.write_command_log()
            self.cleanup(strict=True)
            self.extract_ttracer()
            self.write_raw_limit_record()
            summary = {
                "status": status, "rung_index": self.rung_index,
                "commanded_noise_power_db": self.command_db,
                "command": command, "tail": tail, "transport": transport,
                "tail_service": tail_service,
                "clean_recovery": recovery, "hard_loss_reason": self.hard_loss_reason,
                "status_before_recovery_gate": status_before_recovery_gate,
                "ran_epoch_id": self.ran_epoch_id,
                "control_session_id": self.control_session_id,
                "receiver_service_outage_detected": self.receiver_service_outage_detected,
                "receiver_service_outage_first_monotonic_ns": (
                    self.receiver_service_outage_first_monotonic_ns
                ),
                "clean_restore_verified": self.restored,
                "candidate_application_count": self.application_count,
                "achieved_pusch_snr_db_median": (
                    tail.get("achieved_pusch_snr_db_median") if tail else None
                ),
                "target_mapping_promoted": False, "numeric_bound_promoted": False,
                "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED",
            }
            self.write_manifest_terminal(status, summary)
            return 1 if status == RUNG_RECOVERY_UNCONFIRMED else 0
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            try:
                self.write_command_log()
            except Exception:
                pass
            cleanup_errors = self.cleanup(strict=False)
            status = RESTORE_FAILED if self.nonclean_applied and not self.restored else "FAILED"
            failure = {
                "status": status, "rung_index": self.rung_index,
                "commanded_noise_power_db": self.command_db,
                "error_type": type(exc).__name__, "error": str(exc),
                "hard_loss_reason": self.hard_loss_reason,
                "ran_epoch_id": self.ran_epoch_id,
                "control_session_id": self.control_session_id,
                "clean_restore_verified": self.restored,
                "candidate_application_count": self.application_count,
                "cleanup_errors": cleanup_errors,
                "target_mapping_promoted": False, "numeric_bound_promoted": False,
            }
            self.write_manifest_terminal(status, failure)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


class CampaignRunner:
    def __init__(
        self, config_path: Path, output_dir: Path,
        *, clean_control_evidence: Path | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise CalibrationFailure(f"create-only output already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.clean_control_evidence = (
            clean_control_evidence.resolve() if clean_control_evidence is not None else None
        )

    def verify_clean_control_predecessor(self) -> dict[str, Any]:
        require(self.clean_control_evidence is not None,
                "live execution requires --clean-control-evidence")
        directory = self.clean_control_evidence
        require(directory.is_dir(), f"clean-control evidence directory missing: {directory}")
        frozen = self.config["live_prerequisites"]
        terminal_path = directory / frozen["clean_control_terminal"]
        manifest_path = directory / frozen["clean_control_manifest"]
        require(terminal_path.is_file() and manifest_path.is_file(),
                "clean-control terminal/manifest missing")
        terminal = load_json(terminal_path)
        manifest = load_json(manifest_path)
        manifest_hash = n2.sha256(manifest_path)
        require(terminal.get("status") == frozen["clean_control_required_status"],
                "clean-control predecessor did not pass")
        require(terminal.get("manifest_sha256") == manifest_hash,
                "clean-control terminal/manifest hash mismatch")
        require(terminal.get("primary_usable_service_pass") is True,
                "clean-control terminal lacks the primary service pass")
        require(terminal.get("clean_restore_verified") is True,
                "clean-control terminal lacks verified -50 restoration")
        require(terminal.get("mapping_promoted") is False,
                "clean-control terminal unexpectedly promoted a mapping")
        require(terminal.get("numeric_bound_promoted") is False,
                "clean-control terminal unexpectedly promoted a bound")
        require(manifest.get("status") == frozen["clean_control_required_status"],
                "clean-control manifest status mismatch")
        require(manifest.get("mode") == frozen["clean_control_required_mode"],
                "clean-control manifest mode mismatch")
        approved_clean_config = resolve_repo_path(
            str(frozen["clean_control_config_path"])
        )
        require(
            n2.sha256(approved_clean_config)
            == frozen["clean_control_config_sha256"],
            "approved clean-control config seal drift",
        )
        require(
            manifest.get("config_sha256")
            == frozen["clean_control_config_sha256"],
            "clean-control predecessor used an unexpected config",
        )
        require(
            Path(str(manifest.get("config_path", ""))).resolve()
            == approved_clean_config,
            "clean-control predecessor config path mismatch",
        )
        runner_seal = next(
            (
                seal["sha256"] for seal in self.config["runtime_seals"]
                if seal["path"] == frozen["clean_control_runner_path"]
            ),
            None,
        )
        require(runner_seal is not None, "clean-control runner seal is absent")
        require(manifest.get("runner_sha256") == runner_seal,
                "clean-control predecessor used an unexpected runner")
        output_rows = list(manifest.get("outputs", []))
        require(output_rows, "clean-control manifest has no output inventory")
        seen: set[str] = set()
        for row in output_rows:
            relative = str(row.get("path", ""))
            require(relative and relative not in seen,
                    "clean-control manifest has a blank or duplicate output path")
            seen.add(relative)
            artifact = (directory / relative).resolve()
            try:
                artifact.relative_to(directory.resolve())
            except ValueError as exc:
                raise CalibrationFailure(
                    f"clean-control output escapes evidence directory: {relative}"
                ) from exc
            require(artifact.is_file(), f"clean-control output missing: {relative}")
            require(artifact.stat().st_size == int(row.get("bytes", -1)),
                    f"clean-control output byte-count drift: {relative}")
            require(n2.sha256(artifact) == row.get("sha256"),
                    f"clean-control output hash drift: {relative}")
        required_outputs = {
            "summary.json", "receiver_gate.json", "cleanup_report.json",
            "preflight.json", "connectivity_gate.json", "mcs_summary.json",
            str(frozen["clean_control_resolved_config"]),
        }
        require(required_outputs.issubset(seen),
                f"clean-control manifest lacks required outputs: {sorted(required_outputs - seen)}")
        summary = load_json(directory / "summary.json")
        require(summary.get("status") == frozen["clean_control_required_status"],
                "clean-control summary status mismatch")
        require(summary.get("receiver_gate", {}).get("primary_usable_service_pass") is True,
                "clean-control summary lacks primary receiver pass")
        require(summary.get("restored_to_clean_minus50") is True,
                "clean-control summary lacks -50 restoration")
        require(summary.get("cleanup_clean") is True,
                "clean-control summary lacks clean teardown")
        require(summary.get("connectivity_gate", {}).get("status") == "PASSED",
                "clean-control summary lacks connectivity pass")
        cleanup = load_json(directory / "cleanup_report.json")
        require(cleanup.get("clean") is True,
                "clean-control cleanup report is not clean")
        require(cleanup.get("ext_dn_structured_udp_port_busy") is False,
                "clean-control cleanup left the ext-DN receiver port busy")
        resolved_clean_config = load_json(
            directory / str(frozen["clean_control_resolved_config"])
        )
        require(
            resolved_clean_config == load_json(approved_clean_config),
            "clean-control resolved config differs from the pinned config",
        )
        require(
            str(resolved_clean_config.get("actuator", {}).get(
                "clean_and_restore_commanded_noise_power_db", ""
            )) == frozen["clean_control_required_restore_command_db"],
            "clean-control resolved config does not restore to -50",
        )
        evidence = {
            "status": "VERIFIED_READ_ONLY_PREDECESSOR",
            "directory": str(directory),
            "terminal": terminal_path.name,
            "terminal_sha256": n2.sha256(terminal_path),
            "manifest": manifest_path.name,
            "manifest_sha256": manifest_hash,
        }
        n2.atomic_json(self.output_dir / "clean_control_predecessor.json", evidence)
        return evidence

    def write_plan(self) -> list[dict[str, Any]]:
        validate_config(self.config, verify_hashes=True)
        n2.atomic_json(self.output_dir / "resolved_config.json", self.config)
        rows = campaign_plan_rows(self.config)
        write_csv(self.output_dir / self.config["output"]["campaign_plan"], rows)
        return rows

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(self.output_dir / self.config["output"]["campaign_summary"], summary)
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path("manifest.json"):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3_") or path.name == "FAILED.json"
            ):
                continue
            files.append({"path": str(relative), "bytes": path.stat().st_size,
                          "sha256": n2.sha256(path)})
        n2.atomic_json(self.output_dir / "manifest.json", {
            "schema": "scenesense.ue_n3_command_calibration_campaign_manifest.v1",
            "status": status, "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "target_mapping_promoted": False, "numeric_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
            "outputs": files,
        })
        n2.atomic_json(self.output_dir / f"{status}.json", {
            **dict(summary),
            "status": status, "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
            "manifest_sha256": n2.sha256(self.output_dir / "manifest.json"),
        })

    def prepare(self) -> int:
        rows = self.write_plan()
        summary = {
            "status": PLAN_FROZEN, "runtime_executed": False,
            "socket_executed": False, "rungs": rows,
            "target_mapping_promoted": False, "numeric_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
            "next": "REVIEW_LIVE_AUTHORITY_AFTER_CARLA_IS_STOPPED",
        }
        self.manifest_terminal(PLAN_FROZEN, summary)
        print(json.dumps({"output_dir": str(self.output_dir), "status": PLAN_FROZEN}, sort_keys=True))
        return 0

    def execute(self) -> int:
        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        rung_evidence: list[dict[str, Any]] = []
        observed_ran_epochs: set[str] = set()
        observed_control_sessions: set[str] = set()
        failed_rung: dict[str, Any] | None = None
        clean_control: dict[str, Any] | None = None
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise CalibrationFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught] = signal.getsignal(caught)
            signal.signal(caught, terminate)
        try:
            rows = self.write_plan()
            require(self.config["authority"]["live_oai_run_authorized"] is True,
                    "campaign lacks live OAI authority")
            require(self.config["authority"]["live_socket_execution_authorized"] is True,
                    "campaign lacks live socket authority")
            clean_control = self.verify_clean_control_predecessor()
            campaign_failed = False
            for row in rows:
                label = str(row["commanded_noise_power_db"]).replace(
                    "-", "minus"
                ).replace(".", "p")
                rung_dir = (
                    self.output_dir / "rungs"
                    / f"rung_{int(row['rung_index']):02d}_{label}"
                )
                runner = RungRunner(
                    self.config_path, rung_dir,
                    rung_index=int(row["rung_index"]),
                    command_db=float(row["commanded_noise_power_db"]),
                    clean_control_proof=clean_control,
                )
                rc = runner.run()
                summary_path = rung_dir / self.config["output"]["rung_summary"]
                require(summary_path.is_file(),
                        f"rung summary is absent: {summary_path}")
                rung_summary = load_json(summary_path)
                planned_applications = int(row["candidate_application_count"])
                observed_applications = int(
                    rung_summary.get("candidate_application_count", -1)
                )
                require(
                    0 <= observed_applications <= planned_applications,
                    "failed rung has an impossible candidate-application count",
                )
                proof = verify_rung_evidence(
                    rung_dir,
                    expected_status=str(rung_summary.get("status")),
                    expected_rung_index=int(row["rung_index"]),
                    expected_command_db=float(row["commanded_noise_power_db"]),
                    expected_candidate_applications=(
                        planned_applications if rc == 0 else observed_applications
                    ),
                    expected_config_sha256=n2.sha256(self.config_path),
                    expected_runner_sha256=n2.sha256(Path(__file__).resolve()),
                    require_clean_restore=rc == 0,
                )
                require(proof["ran_epoch_id"] not in observed_ran_epochs,
                        "fresh-RAN epoch identity was reused across rungs")
                require(
                    proof["control_session_id"] not in observed_control_sessions,
                    "control-session identity was reused across rungs",
                )
                observed_ran_epochs.add(str(proof["ran_epoch_id"]))
                observed_control_sessions.add(str(proof["control_session_id"]))
                aggregated_rung = {**rung_summary, "rung_evidence": proof}
                results.append(aggregated_rung)
                rung_evidence.append(proof)
                if rc != 0:
                    campaign_failed = True
                    failed_rung = aggregated_rung
                    break
                if aggregated_rung["status"] in {
                    RUNG_DETACHED, RUNG_IDENTITY_DISCONTINUITY,
                    RUNG_HARD_LOSS, RUNG_UNCONFIRMED,
                    RUNG_RECOVERY_UNCONFIRMED,
                }:
                    break

            accepted = [row for row in results if row.get("status") == RUNG_CAPTURED]
            monotonicity = evaluate_monotonicity(
                accepted, float(self.config["campaign"]["monotonic_tolerance_db"])
            )
            proposals = propose_mappings(
                accepted,
                [
                    float(value) for value in
                    self.config["campaign"]["desired_achieved_pusch_snr_db"]
                ],
                float(self.config["campaign"]["target_tolerance_db"]),
                config_sha256=n2.sha256(self.config_path),
                runner_sha256=n2.sha256(Path(__file__).resolve()),
                monotonicity=monotonicity,
            )
            n2.atomic_json(
                self.output_dir / self.config["output"]["mapping_proposals"],
                proposals,
            )
            complete = (
                len(results) == len(rows)
                and all(row.get("status") == RUNG_CAPTURED for row in results)
            )
            stable = (
                monotonicity["maximum_consecutive_violations"]
                < int(self.config["campaign"]["maximum_consecutive_monotonicity_violations"])
            )
            status = (
                "FAILED" if campaign_failed
                else CAMPAIGN_CAPTURED if complete and stable
                else CAMPAIGN_UNRESOLVED
            )
            summary = {
                "status": status, "rungs_planned": len(rows),
                "rungs_executed": len(results), "rung_results": results,
                "rung_evidence": rung_evidence,
                "failed_rung": failed_rung,
                "monotonicity": monotonicity, "mapping_proposals": proposals,
                "clean_control_predecessor": clean_control,
                "stopped_after_hard_loss": any(
                    row.get("status") in {
                        RUNG_DETACHED, RUNG_IDENTITY_DISCONTINUITY,
                        RUNG_HARD_LOSS,
                    }
                    or row.get("status_before_recovery_gate")
                    in {
                        RUNG_DETACHED, RUNG_IDENTITY_DISCONTINUITY,
                        RUNG_HARD_LOSS,
                    }
                    for row in results
                ),
                "stopped_after_detachment": any(
                    row.get("status") == RUNG_DETACHED
                    or row.get("status_before_recovery_gate") == RUNG_DETACHED
                    for row in results
                ),
                "stopped_after_identity_discontinuity": any(
                    row.get("status") == RUNG_IDENTITY_DISCONTINUITY
                    or row.get("status_before_recovery_gate")
                    == RUNG_IDENTITY_DISCONTINUITY
                    for row in results
                ),
                "target_mapping_promoted": False, "numeric_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
            }
            self.manifest_terminal(status, summary)
            return 0 if not campaign_failed else 1
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "rungs_planned": len(rows),
                "rungs_executed": len(results),
                "rung_results": results,
                "rung_evidence": rung_evidence,
                "clean_control_predecessor": clean_control,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
            }
            self.manifest_terminal("FAILED", failure)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=(PREPARE_ONLY, EXECUTE_LIVE), default=PREPARE_ONLY)
    parser.add_argument("--clean-control-evidence")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = CampaignRunner(
        Path(args.config), Path(args.output_dir),
        clean_control_evidence=(
            Path(args.clean_control_evidence) if args.clean_control_evidence else None
        ),
    )
    return runner.prepare() if args.mode == PREPARE_ONLY else runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
