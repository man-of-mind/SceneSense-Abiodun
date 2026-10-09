#!/usr/bin/env python3
"""Single-attempt 12-cell production-domain UE queue/transport capture runner.

Importing this module is side-effect free.  Live execution requires the ``run``
subcommand and a durably consumed, expiring, exact-path authorization.

The RAN lifecycle (cold launcher start, edge binding, radio-path proof,
T-tracer, telnet actuator, RF restore, teardown, cold verification) is
inherited unchanged from the committed near-capacity runner chain.  Only the
traffic layer is new: it is the deployed production `!IHH` packetization at
12,500 bytes, not a 1,200-byte calibration chunker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.ue_mcs_backlog_calibration_v1 import contract as V3C
from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_runner as CR
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

from . import authorization as AUTH
from . import config as CFG
from . import contract as C
from . import payload_schedule as PS
from . import production_receiver as PR
from . import production_sender as PSND


ROOT = C.ROOT
STATUS_CAPTURED = "PRODUCTION_QUEUE_CAPTURE_CAPTURED"
STATUS_FAILED = "PRODUCTION_QUEUE_CAPTURE_FAILED"
MANIFEST_SCHEMA = "scenesense.production_queue_capture_manifest.v1"
TERMINAL_SCHEMA = "scenesense.production_queue_capture_terminal.v1"
RESULT_SCHEMA = "scenesense.production_queue_capture_result.v1"
VERIFY_STAGES = {"before_preflight", "before_cell", "final_sealing"}


class RunError(RuntimeError):
    """A live-capture invariant failed.  Evidence is preserved, never retried."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_inventory(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    skip = set(excluded)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in skip:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size,
                     "sha256": C.sha256_file(path)})
    return rows


class Runner(CR.Runner):
    """Inherited RAN lifecycle plus the production-domain traffic layer."""

    def __init__(
        self, config_path: Path, output_dir: Path, *,
        initial_inventory: Mapping[str, Any],
    ) -> None:
        # Deliberately do not call the inherited constructors: they read the
        # legacy capacity/scientific config keys directly.  Every lifecycle
        # field they rely on is initialised explicitly here.
        self.config_path = config_path
        self.registration_config = CFG.load_config(config_path)
        self.config = CFG.effective_runtime_config(config_path)
        self.output_dir = output_dir
        self.processes: list[V3R.n2.ManagedProcess] = []
        self.telnet = None
        self.live_pusch = None
        self.model_index: int | None = None
        self.ue_ip: str | None = None
        self.anchors = RB.anchors_for_interpolation()
        self.edge_host: str | None = None
        self.edge_pid: int | None = None
        self.aborted = False
        self.notes: list[str] = []
        self.radio_state_dir: Path | None = None
        self.verifications: list[dict[str, Any]] = []
        self.tiers = ()
        self.initial_inventory = dict(initial_inventory)
        self.container_images: dict[str, Any] = {}
        self.schedules: dict[str, dict[str, Any]] = {}

    # -- identity ------------------------------------------------------
    def verify_identities(self, stage: str) -> dict[str, Any]:
        require(stage in VERIFY_STAGES, f"unregistered verification stage {stage}")
        current = CFG.source_inventory(ROOT)
        require(current == self.initial_inventory,
                f"source inventory drifted at {stage}")
        report = {
            "stage": stage, "utc": utc_now(),
            "source_inventory_sha256": current["inventory_sha256"],
            "contract_sha256": C.CONTRACT_SHA256,
            "authorities": C.verify_authorities(ROOT),
            "radio_binding": RB.verify(stage, ROOT),
            "protected_evidence": PE.require_unchanged(stage, ROOT),
        }
        self.verifications.append(report)
        write_json_create(
            self.out(f"verification/{stage}_{len(self.verifications):02d}.json"),
            report)
        return report

    # -- preflight -----------------------------------------------------
    def preflight_capture(self) -> dict[str, Any]:
        RB.assert_no_forbidden_env(os.environ)
        privileged = self._run_external(
            ["sudo", "-n", "true"], timeout_name="process_probe")
        require(privileged.returncode == 0, "passwordless sudo preflight failed")
        self.assert_cold_ran("production queue capture preflight")
        core = self._container_states()
        require(not any(value.startswith("true") for value in core.values()),
                f"capture requires a cold core, found {core}")
        carla = self._run_external(
            ["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
            timeout_name="process_probe", stderr=subprocess.DEVNULL)
        require(not carla.stdout.strip(),
                "CARLA is running; this capture is network-only")
        record = {
            "utc": utc_now(), "ran_cold": True, "core_before": core,
            "carla_absent": True, "cuda_or_model_started": False,
            "contract_sha256": C.CONTRACT_SHA256,
            "packetization": {
                "identity": C.PACKETIZATION_IDENTITY,
                "chunk_bytes": C.UDP_CHUNK_BYTES_INCLUDING_HEADER,
                "payload_bytes_per_datagram": C.UDP_PAYLOAD_BYTES_PER_DATAGRAM,
                "full_ipv4_packet_bytes": C.FULL_IPV4_PACKET_BYTES,
                "path_mtu_bytes": C.PATH_MTU_BYTES,
                "ipv4_fragmentation_expected": C.IPV4_FRAGMENTATION_EXPECTED,
                "expected_fragments_per_full_datagram":
                    C.FRAGMENTS_PER_FULL_DATAGRAM,
            },
            "plan": [cell.to_json() for cell in C.planned_cells()],
        }
        write_json_create(self.out("preflight.json"), record)
        return record

    def materialize_schedules(self) -> dict[str, Any]:
        """Freeze every cell's byte schedule before any live mutation."""
        ports = self.config["traffic"]["ports"]
        connection = PS._open_authority(ROOT)
        try:
            documents = []
            for cell in C.planned_cells():
                frames = PS.build_cell_schedule(
                    cell, repo_root=ROOT, connection=connection)
                document = PS.schedule_document(cell, frames)
                for row in document["frames"]:
                    row["port"] = int(ports[row["tier"]])
                document["ports"] = dict(ports)
                document["schedule_sha256"] = C.canonical_sha256(
                    document["frames"])
                self.schedules[cell.cell_id] = document
                path = self.out(f"schedules/{cell.run_index:02d}__"
                                f"{cell.cell_id}.json")
                write_json_create(path, document)
                documents.append({
                    "cell_id": cell.cell_id, "run_index": cell.run_index,
                    "partition": cell.partition,
                    "schedule_sha256": document["schedule_sha256"],
                    "file_sha256": C.sha256_file(path),
                    "summary": document["summary"],
                })
        finally:
            connection.close()
        record = {
            "utc": utc_now(), "cells": documents,
            "campaign_schedule_sha256": C.canonical_sha256(documents),
            "payload_coordinate": C.PAYLOAD_COORDINATE,
            "payload_schedule_seed": C.PAYLOAD_SCHEDULE_SEED,
        }
        write_json_create(self.out("payload_schedules.json"), record)
        return record

    # -- traffic -------------------------------------------------------
    def _launch_receivers(
        self, *, cell_tag: str, cell_dir: Path,
    ) -> list[dict[str, Any]]:
        traffic = self.config["traffic"]
        assert self.edge_pid is not None
        receivers: list[dict[str, Any]] = []
        for tier in C.TIER_NAMES:
            port = int(traffic["ports"][tier])
            log_csv = cell_dir / f"receiver_{tier}_frames.csv"
            summary = cell_dir / f"receiver_{tier}_summary.json"
            ready = cell_dir / f"receiver_{tier}_ready.json"
            process = self.spawn(
                f"receiver_{cell_tag}_{tier}",
                ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
                 sys.executable, "-m",
                 "rl_agent.ue_production_queue_capture_v1.production_receiver",
                 "--cell-id", cell_tag, "--bind-host", "0.0.0.0",
                 "--bind-port", str(port),
                 "--socket-recvbuf", str(traffic["receive_buffer_bytes"]),
                 "--idle-timeout-s", str(traffic["receiver_idle_timeout_s"]),
                 "--initial-idle-timeout-s", "300",
                 "--log-csv", str(log_csv), "--summary-json", str(summary),
                 "--ready-json", str(ready)],
                f"cells/{cell_tag}/logs/receiver_{tier}.log", root_owned=True)
            receivers.append({
                "process": process, "ready": ready, "tier": tier,
                "port": port, "log_csv": log_csv, "summary": summary,
            })
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if all(item["ready"].is_file() for item in receivers):
                break
            for item in receivers:
                require(item["process"].process.poll() is None,
                        f"receiver {item['tier']} exited before READY")
            time.sleep(0.2)
        require(all(item["ready"].is_file() for item in receivers),
                "not every block receiver reported READY")
        return receivers

    def launch_traffic(
        self, *, cell_tag: str, cell: C.Cell, cell_dir: Path,
    ) -> dict[str, Any]:
        assert self.ue_ip is not None and self.edge_host is not None
        receivers = self._launch_receivers(cell_tag=cell_tag, cell_dir=cell_dir)
        period_ns = int(1e9 / C.FPS)
        design = self.registration_config["design"]
        schedule_path = cell_dir / "payload_schedule.json"
        write_json_create(schedule_path, self.schedules[cell.cell_id])
        sender_csv = cell_dir / "sender_frames.csv"
        sender_summary = cell_dir / "sender_summary.json"
        sender_ready = cell_dir / "sender_ready.json"
        epoch_path = cell_dir / "shared_epoch.json"
        arm_timeout_s = float(design["sender_arm_timeout_s"])
        sender = self.spawn(
            f"sender_{cell_tag}",
            [sys.executable, "-m",
             "rl_agent.ue_production_queue_capture_v1.production_sender",
             "--cell-id", cell.cell_id, "--bind-host", self.ue_ip,
             "--remote-host", self.edge_host,
             "--schedule-json", str(schedule_path),
             "--chunk-bytes", str(C.UDP_CHUNK_BYTES_INCLUDING_HEADER),
             "--socket-sendbuf", str(self.config["traffic"]["send_buffer_bytes"]),
             "--ready-json", str(sender_ready),
             "--epoch-contract", str(epoch_path),
             "--epoch-contract-timeout-s", f"{arm_timeout_s:.3f}",
             "--cell-wall-clock-guard-s",
             str(design["cell_wall_clock_guard_s"]),
             "--log-csv", str(sender_csv),
             "--summary-json", str(sender_summary)],
            f"cells/{cell_tag}/logs/sender.log")

        deadline = time.monotonic() + arm_timeout_s
        ready: Any = None
        while time.monotonic() < deadline:
            require(sender.process.poll() is None,
                    "production sender exited before reporting READY")
            if sender_ready.is_file():
                try:
                    ready = json.loads(sender_ready.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.005)
        require(type(ready) is dict, "production sender did not report READY")
        require(ready.get("schema") == PSND.READY_SCHEMA
                and ready.get("cell_id") == cell.cell_id
                and ready.get("bind_host") == self.ue_ip
                and ready.get("remote_host") == self.edge_host
                and ready.get("period_ns") == period_ns
                and ready.get("chunk_bytes")
                    == C.UDP_CHUNK_BYTES_INCLUDING_HEADER
                and ready.get("packetization_identity")
                    == C.PACKETIZATION_IDENTITY
                and ready.get("schedule_sha256")
                    == self.schedules[cell.cell_id]["schedule_sha256"],
                "production sender READY identity drifted")

        primer = self.target_channel_primer(cell_tag, cell_dir)
        require(int(ready["ready_monotonic_ns"])
                <= int(primer["send_start_monotonic_ns"]),
                "target-channel primer preceded sender READY")
        created_ns = time.monotonic_ns()
        lead_ns = int(float(design["future_epoch_lead_s"]) * 1e9)
        epoch_ns = created_ns + lead_ns
        grant_ns = int(primer["latest_grant"]["receipt_monotonic_ns"])
        completed_ns = int(primer["completed_monotonic_ns"])
        max_gap_ns = int(float(self.config["traffic"]["target_channel_primer"][
            "max_grant_receipt_to_first_decision_ms"]) * 1e6)
        require(created_ns >= completed_ns,
                "shared epoch publication preceded primer completion")
        require(0 <= epoch_ns - grant_ns <= max_gap_ns,
                "shared epoch would violate the primer freshness gate")
        epoch = {
            "schema": PSND.EPOCH_SCHEMA, "cell_id": cell.cell_id,
            "epoch_monotonic_ns": epoch_ns, "period_ns": period_ns,
            "created_monotonic_ns": created_ns,
            "sender_ready_sha256": C.sha256_file(sender_ready),
        }
        write_json_create(epoch_path, epoch)
        return {
            "receivers": receivers, "sender": sender,
            "frames": C.FRAMES_PER_CELL, "sender_csv": sender_csv,
            "sender_summary": sender_summary, "sender_ready": sender_ready,
            "epoch_contract": epoch_path, "target_channel_primer": primer,
            "epoch": {**epoch, "shared_epoch_monotonic_ns": epoch_ns},
        }

    def replay_profile(
        self, profile: Any, *, epoch_ns: int, first_command: float,
        command_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replay the registered 450-sample target-SNR grid on the shared epoch.

        The grid is nominal wall-clock, not frame-indexed.  If the sender
        overruns it (expected once the guard role saturates the queue), the
        channel holds the final commanded value and every later frame is
        joined offline to the command actually in force at its send instant.
        Commanded noise power and profile identity are audit-only, so the hold
        never reaches the model.
        """
        period_ns = int(float(self.config["campaign"]["sample_period_s"]) * 1e9)
        require(period_ns == int(1e9 / C.FPS),
                "profile replay and sender periods differ")
        granularity = float(self.config["actuator"]["command_granularity_db"])
        last_command: float | None = first_command
        sent = skipped = clamped = 0
        for sample in profile.samples:
            due = epoch_ns + int(sample["step_index"]) * period_ns
            now = time.monotonic_ns()
            if now > due + period_ns:
                skipped += 1
                continue
            if now < due:
                time.sleep((due - now) / 1e9)
            command, was_clamped = V3R.inverse_interpolate(
                sample["target_snr_db"], self.anchors)
            command = V3R.round_to_granularity(command, granularity)
            clamped += int(was_clamped)
            if command != last_command:
                self.send_noise(command, reason="PROFILE_REPLAY",
                                log=command_log,
                                profile_id=profile.profile_id,
                                step_index=sample["step_index"],
                                target_snr_db=sample["target_snr_db"],
                                clamped=was_clamped)
                last_command = command
                sent += 1
        require(skipped == 0, f"profile replay skipped {skipped} step(s)")
        require(clamped == 0, f"profile replay clamped {clamped} target(s)")
        return {
            "shared_epoch_monotonic_ns": epoch_ns,
            "samples": len(profile.samples), "commands_sent": sent,
            "commands_skipped": skipped, "targets_clamped": clamped,
            "replay_end_monotonic_ns": time.monotonic_ns(),
            "final_commanded_noise_power_db": last_command,
            "post_replay_policy": "HOLD_FINAL_COMMAND_AUDIT_ONLY",
        }

    def audit_primer_first_decision(
        self, sessions: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Receipt-time primer gate against this package's sender schema."""
        with Path(sessions["sender_csv"]).open(
                newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle), None)
        require(first is not None, "sender frame CSV is empty")
        first_ns = int(first["frame_open_monotonic_ns"])
        primer = sessions["target_channel_primer"]
        grant_receipt = int(primer["latest_grant"]["receipt_monotonic_ns"])
        completed = int(primer["completed_monotonic_ns"])
        gap_ms = (first_ns - grant_receipt) / 1e6
        completion_gap_ms = (first_ns - completed) / 1e6
        maximum = float(self.config["traffic"]["target_channel_primer"][
            "max_grant_receipt_to_first_decision_ms"])
        require(completion_gap_ms >= 0.0,
                "first scientific frame preceded the primer queue-drain proof")
        require(0.0 <= gap_ms <= maximum,
                f"primer grant receipt-to-first-frame gap {gap_ms:.3f} ms "
                f"is outside [0,{maximum}] ms")
        return {
            "first_frame_monotonic_ns": first_ns,
            "latest_grant_receipt_monotonic_ns": grant_receipt,
            "grant_receipt_to_first_frame_ms": gap_ms,
            "primer_complete_to_first_frame_ms": completion_gap_ms,
            "receipt_time_gate_passed": True,
            "authoritative_source_time_gate":
                "DEFERRED_TO_POST_EXTRACTION_CAUSAL_DECISION_JOIN",
        }

    def finish_traffic(self, sessions: Mapping[str, Any]) -> None:
        design = self.registration_config["design"]
        budget = float(design["cell_wall_clock_guard_s"]) + 60.0
        try:
            sessions["sender"].process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            self.notes.append("sender overran its budget")
        tail = float(self.config["traffic"]["receiver_tail_s"])
        for item in sessions["receivers"]:
            try:
                item["process"].process.wait(timeout=tail + 180.0)
            except subprocess.TimeoutExpired:
                self.notes.append(f"receiver {item['tier']} overran its budget")

    def audit_traffic(
        self, sessions: Mapping[str, Any], cell: C.Cell, cell_dir: Path,
    ) -> dict[str, Any]:
        summary_path = Path(sessions["sender_summary"])
        csv_path = Path(sessions["sender_csv"])
        require(summary_path.is_file() and csv_path.is_file(),
                "sender did not emit both durable outputs")
        sender = json.loads(summary_path.read_text(encoding="utf-8"))
        require(sender.get("schema") == PSND.SUMMARY_SCHEMA,
                "sender summary schema drifted")
        require(sender["frames"] == C.FRAMES_PER_CELL,
                f"sender emitted {sender['frames']} frames, "
                f"expected {C.FRAMES_PER_CELL}")
        totals = sender["totals"]
        require(totals["datagrams_dropped_at_socket"] == 0,
                "sender dropped datagrams at the socket")
        require(totals["datagrams_handed_to_socket"]
                == totals["planned_datagrams"],
                "sender datagram count disagrees with the frozen schedule")
        require(totals["application_payload_bytes_handed_to_socket"]
                == totals["planned_total_transmitted_bytes"],
                "sender payload bytes disagree with the frozen schedule")
        require(sender["packetization"]["identity"] == C.PACKETIZATION_IDENTITY
                and sender["packetization"]["chunk_bytes_including_header"]
                    == C.UDP_CHUNK_BYTES_INCLUDING_HEADER
                and sender["packetization"]["retransmission"] is False,
                "sender packetization drifted from the live production binding")

        with csv_path.open(newline="", encoding="utf-8") as handle:
            sender_rows = list(csv.DictReader(handle))
        require(len(sender_rows) == C.FRAMES_PER_CELL,
                "sender CSV row count drifted")

        receivers: list[dict[str, Any]] = []
        received_complete = 0
        received_observed = 0
        received_payload_bytes = 0
        fragmentation_observed = False
        for item in sessions["receivers"]:
            path = Path(item["summary"])
            require(path.is_file(),
                    f"receiver {item['tier']} did not emit a summary")
            value = json.loads(path.read_text(encoding="utf-8"))
            require(value.get("schema") == PR.SUMMARY_SCHEMA,
                    "receiver summary schema drifted")
            require(value["malformed_datagrams"] == 0,
                    f"receiver {item['tier']} saw malformed datagrams")
            received_complete += int(value["messages_complete"])
            received_observed += int(value["messages_observed"])
            received_payload_bytes += int(
                value["total_application_payload_bytes_received"])
            fragmentation_observed = (
                fragmentation_observed or bool(value["observed_ip_fragmentation"]))
            receivers.append({
                "tier": item["tier"], "port": item["port"],
                "messages_observed": value["messages_observed"],
                "messages_complete": value["messages_complete"],
                "messages_incomplete": value["messages_incomplete"],
                "duplicate_datagrams": value["duplicate_datagrams"],
                "total_datagrams_received": value["total_datagrams_received"],
                "application_payload_bytes_received":
                    value["total_application_payload_bytes_received"],
                "ip_counters_delta": value["ip_counters_delta"],
                "observed_ip_fragmentation":
                    value["observed_ip_fragmentation"],
                "summary_sha256": C.sha256_file(path),
                "frames_csv_sha256": C.sha256_file(Path(item["log_csv"])),
            })

        record = {
            "sender_summary_sha256": C.sha256_file(summary_path),
            "sender_csv_sha256": C.sha256_file(csv_path),
            "frames_sent": sender["frames"],
            "datagrams_sent": totals["datagrams_handed_to_socket"],
            "application_payload_bytes_sent":
                totals["application_payload_bytes_handed_to_socket"],
            "udp_application_bytes_sent":
                totals["udp_application_bytes_handed_to_socket"],
            "schedule_lag_ms": sender["schedule_lag_ms"],
            "receivers": receivers,
            "messages_observed_total": received_observed,
            "messages_complete_total": received_complete,
            "application_payload_bytes_received_total": received_payload_bytes,
            "observed_ip_fragmentation": fragmentation_observed,
            "delivery_is_not_gated_here": (
                "Incomplete delivery is the measurement, not a defect. "
                "Terminal classification happens in the offline parser."),
        }
        write_json_create(cell_dir / "traffic_audit.json", record)
        return record

    # -- one cell ------------------------------------------------------
    def run_cell(self, cell: C.Cell, profile: Any) -> dict[str, Any]:
        cell_tag = f"{cell.run_index:02d}__{cell.cell_id}"
        cell_dir = self.output_dir / "cells" / cell_tag
        PE.assert_outside_protected_run(cell_dir, ROOT)
        cell_dir.mkdir(parents=True, exist_ok=False)
        command_log: list[dict[str, Any]] = []
        notes_at_start = len(self.notes)
        record: dict[str, Any] = {
            "cell_tag": cell_tag, **cell.to_json(),
            "trace_id": profile.trace_id,
            "registered_trace_sha256": profile.trace_sha256,
            "started_utc": utc_now(), "status": "FAILED",
        }
        try:
            self.verify_identities("before_cell")
            self.assert_cold_ran(f"cell {cell_tag}")
            require(not any(value.startswith("true")
                            for value in self._container_states().values()),
                    f"cell {cell_tag} did not start from a cold core")
            record["radio_attach"] = self.start_ran_via_launcher(cell_tag, cell_dir)
            record["edge_context"] = self.bind_edge_context()
            record["radio_path"] = self.verify_radio_path(cell_dir)
            self.start_telemetry(cell_tag)
            self.open_telnet(cell_dir)
            record["noise_before_cell_db"] = self.read_back_noise()
            record["udp_probe"] = self.udp_probe(cell_tag, cell_dir)
            granularity = float(self.config["actuator"]["command_granularity_db"])
            first_command, clamped = V3R.inverse_interpolate(
                profile.samples[0]["target_snr_db"], self.anchors)
            require(not clamped, "profile prime would clamp")
            first_command = V3R.round_to_granularity(first_command, granularity)
            self.send_noise(first_command, reason="PROFILE_PRIME", log=command_log,
                            profile_id=profile.profile_id, step_index=0,
                            target_snr_db=profile.samples[0]["target_snr_db"],
                            clamped=False)
            require(abs(self.read_back_noise() - first_command) <= 1e-6,
                    "profile prime read-back mismatch")
            time.sleep(float(self.config["campaign"]["warmup_s"]))
            sessions = self.launch_traffic(
                cell_tag=cell_tag, cell=cell, cell_dir=cell_dir)
            record["target_channel_primer"] = sessions["target_channel_primer"]
            record["shared_epoch"] = sessions["epoch"]
            record["profile_replay"] = self.replay_profile(
                profile, epoch_ns=sessions["epoch"]["shared_epoch_monotonic_ns"],
                first_command=first_command, command_log=command_log)
            self.finish_traffic(sessions)
            record["primer_first_decision"] = self.audit_primer_first_decision(
                {**sessions, "sender_csv": sessions["sender_csv"]})
            record["traffic"] = self.audit_traffic(sessions, cell, cell_dir)
            record["restored"] = self.restore_clean(cell_dir, command_log)
            require(record["restored"], "RF restore/read-back failed")
            record["status"] = "CAPTURED_PENDING_TEARDOWN"
        except Exception as exc:  # noqa: BLE001 - preserve evidence and clean up
            record["failure"] = f"{type(exc).__name__}: {exc}"
            if self.telnet is not None:
                try:
                    record["restored"] = self.restore_clean(cell_dir, command_log)
                except Exception as inner:  # noqa: BLE001
                    record["restore_failure"] = f"{type(inner).__name__}: {inner}"
        finally:
            write_json_create(cell_dir / "command_log.json", command_log)
            ran_notes = self.teardown_ran()
            cell_notes = list(self.notes[notes_at_start:]) + list(ran_notes)
            record["lifecycle_notes"] = cell_notes
            try:
                self.extract_ttracer(cell_tag, cell_dir)
                record["ttracer"] = self.audit_ttracer_nonempty(cell_dir)
            except Exception as exc:  # noqa: BLE001
                record["ttracer_failure"] = f"{type(exc).__name__}: {exc}"
            record["core_teardown"] = self.stop_core()
            try:
                record["cell_cold_state"] = self._cell_cold_snapshot()
            except Exception as exc:  # noqa: BLE001
                record["cell_cold_state"] = {
                    "cold": False,
                    "probe_failure": f"{type(exc).__name__}: {exc}"}
            if (record["status"] == "CAPTURED_PENDING_TEARDOWN"
                    and not cell_notes and "ttracer" in record
                    and record["core_teardown"].get("stopped") is True
                    and record["cell_cold_state"].get("cold") is True):
                record["status"] = "CAPTURED"
            else:
                record["status"] = "FAILED"
                record.setdefault("failure", "cleanup/evidence gate failed")
            record["finished_utc"] = utc_now()
            write_json_create(cell_dir / "cell_record.json", record)
        return record

    def audit_ttracer_nonempty(self, cell_dir: Path) -> dict[str, Any]:
        root = cell_dir / "ttracer"
        require(root.is_dir(), "T-tracer output directory is missing")
        files = [path for path in root.rglob("*.csv") if path.is_file()]
        require(bool(files), "T-tracer produced no CSV output")
        nonempty = {}
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                rows = sum(1 for _ in handle)
            nonempty[str(path.relative_to(root))] = rows
        require(any(value > 1 for value in nonempty.values()),
                "every T-tracer CSV is header-only")
        return {"files": nonempty, "file_count": len(files)}

    def _cell_cold_snapshot(self) -> dict[str, Any]:
        states = self._container_states()
        running = [name for name, value in states.items()
                   if value.startswith("true")]
        processes = self._run_external(
            ["pgrep", "-af", "nr-softmodem|nr-uesoftmodem"],
            timeout_name="process_probe", stderr=subprocess.DEVNULL)
        return {"cold": not running and not processes.stdout.strip(),
                "containers": states,
                "ran_processes": processes.stdout.strip()}

    # -- sealing -------------------------------------------------------
    def _write_seals(
        self, *, status: str, failure: str | None,
        cells: Sequence[Mapping[str, Any]], final_cold: Mapping[str, Any],
        schedules: Mapping[str, Any], authorization: Mapping[str, Any],
    ) -> None:
        excluded = {"CAPTURE_RESULT.json", "manifest.json", "TERMINAL.json"}
        result = {
            "schema": RESULT_SCHEMA, "status": status, "failure": failure,
            "package_id": C.PACKAGE_ID,
            "contract_sha256": C.CONTRACT_SHA256,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "created_utc": utc_now(),
            "cells": list(cells),
            "cells_captured": sum(1 for row in cells
                                  if row.get("status") == "CAPTURED"),
            "expected_cells": C.EXPECTED_CELLS,
            "payload_schedules": dict(schedules),
            "authorization": dict(authorization),
            "final_cold_state": dict(final_cold),
            "verifications": list(self.verifications),
            "notes": list(self.notes),
            "packetization": {
                "identity": C.PACKETIZATION_IDENTITY,
                "chunk_bytes": C.UDP_CHUNK_BYTES_INCLUDING_HEADER,
                "retransmission": C.RETRANSMISSION,
            },
            "disclosures": {
                "catalogue_contract_tier": C.CATALOGUE_CONTRACT_TIER,
                "perception_endorsement": C.PERCEPTION_ENDORSEMENT,
                "byte_only_queue_design": C.BYTE_ONLY_QUEUE_DESIGN,
                "single_ue_radio_configuration_only": True,
                "profile_identity_role": C.PROFILE_IDENTITY_ROLE,
            },
        }
        write_json_create(self.out("CAPTURE_RESULT.json"), result)
        inventory = file_inventory(self.output_dir, excluded=excluded)
        manifest = {
            "schema": MANIFEST_SCHEMA, "created_utc": utc_now(),
            "output_root": str(self.output_dir.relative_to(ROOT)),
            "files": inventory,
            "inventory_sha256": C.canonical_sha256(inventory),
        }
        write_json_create(self.out("manifest.json"), manifest)
        terminal = {
            "schema": TERMINAL_SCHEMA, "status": status,
            "created_utc": utc_now(),
            "contract_sha256": C.CONTRACT_SHA256,
            "result_sha256": C.sha256_file(self.out("CAPTURE_RESULT.json")),
            "manifest_sha256": C.sha256_file(self.out("manifest.json")),
        }
        write_json_create(self.out("TERMINAL.json"), terminal)

    # -- campaign ------------------------------------------------------
    def run(self, authorization: Mapping[str, Any]) -> int:
        status = STATUS_FAILED
        failure: str | None = None
        cells: list[dict[str, Any]] = []
        schedules: dict[str, Any] = {}
        final_cold: dict[str, Any] = {}
        try:
            self.verify_identities("before_preflight")
            self.preflight_capture()
            schedules = self.materialize_schedules()
            profiles = {
                item.profile_id: item
                for item in V3C.resolve_profiles(ROOT, C.FRAMES_PER_CELL)
            }
            for name in C.PROFILES:
                require(name in profiles, f"registered profile {name} missing")
            for cell in C.planned_cells():
                record = self.run_cell(cell, profiles[cell.profile_id])
                cells.append(record)
                if record["status"] != "CAPTURED":
                    failure = (f"cell {record['cell_tag']} failed: "
                               f"{record.get('failure')}")
                    break
            else:
                status = STATUS_CAPTURED
        except Exception as exc:  # noqa: BLE001 - preserve and seal
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                final_cold = self.final_cold_with_core()
            except Exception as exc:  # noqa: BLE001
                final_cold = {"cold": False,
                              "probe_failure": f"{type(exc).__name__}: {exc}"}
            if status == STATUS_CAPTURED and not final_cold.get("cold", False):
                status = STATUS_FAILED
                failure = failure or "final cold-state verification failed"
            try:
                self.verify_identities("final_sealing")
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"final identity verification failed: {exc}")
                status = STATUS_FAILED
                failure = failure or "final identity verification failed"
            self._write_seals(
                status=status, failure=failure, cells=cells,
                final_cold=final_cold, schedules=schedules,
                authorization=dict(authorization))
        return 0 if status == STATUS_CAPTURED else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="print the frozen plan; no live effect")
    plan.add_argument("--config", default=str(CFG.DEFAULT_CONFIG))
    run_cmd = sub.add_parser("run", help="execute the single authorized capture")
    run_cmd.add_argument("--config", default=str(CFG.DEFAULT_CONFIG))
    run_cmd.add_argument("--output-dir", required=True)
    run_cmd.add_argument("--authorization", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        document = {
            "contract_sha256": C.CONTRACT_SHA256,
            "config_sha256": C.sha256_file(Path(args.config)),
            "source_inventory": CFG.source_inventory(ROOT)["inventory_sha256"],
            "cells": [cell.to_json() for cell in C.planned_cells()],
        }
        json.dump(document, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    output_dir = Path(args.output_dir).resolve()
    require(not output_dir.exists(), "output directory must be create-only")
    inventory = CFG.source_inventory(ROOT)
    consumption = AUTH.consume(
        Path(args.authorization), output_path=output_dir, repo_root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=False)
    runner = Runner(Path(args.config), output_dir,
                    initial_inventory=inventory)
    return runner.run(consumption)


if __name__ == "__main__":
    sys.exit(main())
