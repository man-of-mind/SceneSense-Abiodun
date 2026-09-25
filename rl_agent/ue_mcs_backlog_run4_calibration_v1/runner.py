#!/usr/bin/env python3
"""Registered 12-cell Run-4 physical queue-calibration runner.

Importing this module is side-effect free.  Live execution requires the
``run`` subcommand and a durably consumed, expiring, exact-path authorization.
The decision analyzer is deliberately outside this package.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_runner as CR
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as NR
from rl_agent import ue_n3_structured_udp_receiver as U3

from . import authorization as AUTH
from . import contract as C
from . import tagged_sender as TS


ROOT = C.ROOT
DEFAULT_CONFIG = C.DEFAULT_CONFIG
STATUS_CAPTURED = "RUN4_PHYSICAL_QUEUE_CALIBRATION_CAPTURED"
STATUS_FAILED = "RUN4_PHYSICAL_QUEUE_CALIBRATION_FAILED"
MANIFEST_SCHEMA = "scenesense.run4_physical_queue_calibration_manifest.v1"
TERMINAL_SCHEMA = "scenesense.run4_physical_queue_calibration_terminal.v1"
RESULT_SCHEMA = "scenesense.run4_physical_queue_calibration_result.v1"
VERIFY_STAGES = {"before_preflight", "before_cell", "final_sealing"}


class RunError(RuntimeError):
    """A lifecycle, source, accounting or registered-design gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_inventory(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    return [{"relative_path": str(path.relative_to(root)),
             "size_bytes": path.stat().st_size,
             "sha256": C.sha256_file(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and str(path.relative_to(root)) not in excluded_set]


EXPECTED_INHERITED_METHODS: Mapping[str, str] = {
    # Bounded process/core/RAN lifecycle from the capacity retry.
    "_timeout": CR.__name__, "_run_external": CR.__name__,
    "_tunnel_interfaces": CR.__name__, "assert_cold_ran": CR.__name__,
    "start_ran_via_launcher": CR.__name__, "teardown_ran": CR.__name__,
    "extract_ttracer": CR.__name__, "_container_states": CR.__name__,
    "_container_image_bindings": CR.__name__, "bind_edge_context": CR.__name__,
    "verify_radio_path": CR.__name__, "stop_core": CR.__name__,
    # Target-channel path proof and primer from the scientific runner.
    "udp_probe": NR.__name__, "target_channel_primer": NR.__name__,
    "audit_primer_first_decision": NR.__name__,
    # Generic managed process, telemetry, actuation and restore primitives.
    "path": V3R.__name__, "out": V3R.__name__, "spawn": V3R.__name__,
    "start_telemetry": V3R.__name__, "open_telnet": V3R.__name__,
    "send_noise": V3R.__name__, "read_back_noise": V3R.__name__,
    "restore_clean": V3R.__name__, "finish_traffic": V3R.__name__,
}


def inherited_lifecycle_audit(cls: type) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for name, expected_module in EXPECTED_INHERITED_METHODS.items():
        method = getattr(cls, name)
        owner = next(base for base in cls.__mro__ if name in base.__dict__)
        observed_module = owner.__module__
        require(observed_module == expected_module,
                f"inherited method {name} owner {observed_module} != "
                f"registered {expected_module}")
        methods[name] = {"owner_class": f"{owner.__module__}.{owner.__qualname__}",
                         "owner_module": observed_module,
                         "source_sha256": C.canonical_sha256(
                             inspect.getsource(method))}
    return {"verified": True, "methods": methods,
            "source_pins": C.verify_inherited_sources(ROOT)}


class Runner(CR.Runner):
    """Bounded inherited lifecycle plus Run-4 design, sender and evidence."""

    def __init__(
        self, config_path: Path, output_dir: Path, *,
        initial_inventory: Mapping[str, Any], lineage: Mapping[str, Any],
        amendment: Mapping[str, Any], lifecycle_audit: Mapping[str, Any],
    ) -> None:
        # Deliberately do not call the inherited constructors: they read the
        # old capacity/scientific config directly and would re-select refused
        # tiers.  Every lifecycle field they require is initialized explicitly.
        self.config_path = config_path
        self.registration_config = C.load_config(config_path)
        self.config = C.effective_runtime_config(config_path)
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
        self.tiers = C.registered_tiers()
        self.initial_inventory = dict(initial_inventory)
        self.lineage = dict(lineage)
        self.amendment = dict(amendment)
        self.lifecycle_audit = dict(lifecycle_audit)
        self.container_images: dict[str, Any] = {}

    def verify_identities(self, stage: str) -> dict[str, Any]:
        require(stage in VERIFY_STAGES, f"unregistered verification stage {stage}")
        current = C.source_inventory(ROOT)
        require(current == self.initial_inventory,
                f"Run-4 source inventory drifted at {stage}")
        amendment_path = ROOT / C.AMENDMENT_RELPATH
        require(C.sha256_file(amendment_path) == C.AMENDMENT_SHA256,
                f"sealed amendment drifted at {stage}")
        report = {
            "stage": stage, "utc": utc_now(),
            "source_inventory_sha256": current["inventory_sha256"],
            "inherited_sources": C.verify_inherited_sources(ROOT),
            "radio_binding": RB.verify(stage, ROOT),
            "protected_evidence": PE.require_unchanged(stage, ROOT),
            "amendment_sha256": C.AMENDMENT_SHA256,
        }
        self.verifications.append(report)
        write_json_create(self.out(f"verification/{stage}_{len(self.verifications):02d}.json"),
                          report)
        return report

    def preflight_run4(self) -> dict[str, Any]:
        RB.assert_no_forbidden_env(os.environ)
        privileged = self._run_external(
            ["sudo", "-n", "true"], timeout_name="process_probe")
        require(privileged.returncode == 0, "passwordless sudo preflight failed")
        self.assert_cold_ran("Run-4 calibration preflight")
        core = self._container_states()
        require(not any(value.startswith("true") for value in core.values()),
                f"Run-4 calibration requires a cold core, found {core}")
        carla = self._run_external(
            ["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
            timeout_name="process_probe", stderr=subprocess.DEVNULL)
        require(not carla.stdout.strip(), "CARLA is running; calibration is network-only")
        audit = C.audit_cell_plan(C.build_cell_plan(
            ports=self.config["traffic"]["ports"], seed=C.CELL_ORDER_SEED))
        require(audit["registered_design"], f"registered plan audit failed: {audit}")
        record = {"utc": utc_now(), "ran_cold": True, "core_before": core,
                  "carla_absent": True, "cuda_or_model_started": False,
                  "plan_audit": audit, "packetization": {
                      "chunk_bytes": C.CHUNK_BYTES,
                      "full_ipv4_packet_bytes": C.FULL_IPV4_PACKET_BYTES,
                      "path_mtu_bytes": C.PATH_MTU_BYTES,
                      "mtu_safe": True,
                  }}
        write_json_create(self.out("preflight.json"), record)
        return record

    def _launch_receivers(
        self, *, cell_tag: str, blocks: Sequence[Mapping[str, Any]],
        cell_dir: Path, total_frames: int,
    ) -> list[dict[str, Any]]:
        traffic = self.config["traffic"]
        assert self.edge_pid is not None
        duration = (total_frames / C.FPS
                    + float(traffic["receiver_tail_s"])
                    + float(traffic["target_channel_primer"]["timeout_s"])
                    + float(self.registration_config["design"]["sender_arm_timeout_s"])
                    + float(self.registration_config["design"]["future_epoch_lead_s"]))
        receivers: list[dict[str, Any]] = []
        for block in blocks:
            tag = f"block{block['block_index']}_{block['tier']}"
            events = cell_dir / f"receiver_{tag}_events.jsonl"
            summary = cell_dir / f"receiver_{tag}_summary.json"
            ready = cell_dir / f"receiver_{tag}_ready.json"
            expected_chunks = int(block["chunks_per_frame"])
            max_chunks = min(U3.MAX_CHUNKS_PER_FRAME_LIMIT,
                             max(expected_chunks, expected_chunks * 2))
            require(expected_chunks <= max_chunks <= U3.MAX_CHUNKS_PER_FRAME_LIMIT,
                    f"receiver chunk bound invalid for {block['tier']}")
            process = self.spawn(
                f"receiver_{cell_tag}_{tag}",
                ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
                 sys.executable, str(self.path(NC_PRODUCTION_RECEIVER)),
                 "--bind-host", "0.0.0.0", "--port", str(block["port"]),
                 "--events-jsonl", str(events), "--summary-json", str(summary),
                 "--ready-json", str(ready), "--duration-s", f"{duration:.3f}",
                 "--expected-frames", str(block["frames"]),
                 "--expected-chunks-per-frame", str(expected_chunks),
                 "--max-chunks-per-frame", str(max_chunks),
                 "--socket-receive-buffer-bytes",
                 str(traffic["receive_buffer_bytes"])],
                f"cells/{cell_tag}/logs/receiver_{tag}.log", root_owned=True)
            receivers.append({"process": process, "ready": ready,
                              "block_index": int(block["block_index"]),
                              "tier": block["tier"], "events": events,
                              "summary": summary})
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if all(row["ready"].is_file() for row in receivers):
                break
            for row in receivers:
                require(row["process"].process.poll() is None,
                        f"receiver {row['tier']} exited before READY")
            time.sleep(0.1)
        require(all(row["ready"].is_file() for row in receivers),
                "not every receiver reported READY")
        return receivers

    def launch_traffic(
        self, *, cell_tag: str, cell_id: str,
        blocks: Sequence[Mapping[str, Any]], cell_dir: Path, total_frames: int,
    ) -> dict[str, Any]:
        """Arm the sender, prove primer drain, then publish one shared epoch."""
        assert self.ue_ip is not None and self.edge_host is not None
        receivers = self._launch_receivers(
            cell_tag=cell_tag, blocks=blocks, cell_dir=cell_dir,
            total_frames=total_frames)
        period_ns = int(1e9 / C.FPS)
        plan_path = cell_dir / "block_plan.json"
        write_json_create(plan_path, list(blocks))
        sender_csv = cell_dir / "sender_decisions.csv"
        sender_summary = cell_dir / "sender_summary.json"
        sender_ready = cell_dir / "sender_ready.json"
        epoch_path = cell_dir / "shared_epoch.json"
        arm_timeout_s = float(
            self.registration_config["design"]["sender_arm_timeout_s"])
        sender = self.spawn(
            f"sender_{cell_tag}",
            [sys.executable, "-m",
             "rl_agent.ue_mcs_backlog_run4_calibration_v1.tagged_sender",
             "--cell-id", cell_id, "--bind-host", self.ue_ip,
             "--remote-host", self.edge_host, "--block-plan", str(plan_path),
             "--payload-seed", str(self.config["campaign"]["payload_seed"]),
             "--chunk-bytes", str(C.CHUNK_BYTES), "--fps", str(C.FPS),
             "--ready-json", str(sender_ready),
             "--epoch-contract", str(epoch_path),
             "--epoch-contract-timeout-s", f"{arm_timeout_s:.3f}",
             "--socket-sendbuf", str(self.config["traffic"]["send_buffer_bytes"]),
             "--log-csv", str(sender_csv), "--summary-json", str(sender_summary)],
            f"cells/{cell_tag}/logs/sender.log")

        deadline = time.monotonic() + arm_timeout_s
        ready: Any = None
        while time.monotonic() < deadline:
            require(sender.process.poll() is None,
                    "tagged sender exited before reporting READY")
            if sender_ready.is_file():
                try:
                    ready = json.loads(sender_ready.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.005)
        require(type(ready) is dict, "tagged sender did not report READY")
        require(set(ready) == {
            "schema", "cell_id", "ready_monotonic_ns", "bind_host",
            "remote_host", "socket_sendbuf", "period_ns", "block_plan_sha256",
        }, "tagged sender READY fields drifted")
        require(ready.get("schema") == TS.READY_SCHEMA
                and ready.get("cell_id") == cell_id
                and type(ready.get("ready_monotonic_ns")) is int
                and ready.get("bind_host") == self.ue_ip
                and ready.get("remote_host") == self.edge_host
                and ready.get("socket_sendbuf") ==
                    self.config["traffic"]["send_buffer_bytes"]
                and ready.get("period_ns") == period_ns
                and ready.get("block_plan_sha256") == C.sha256_file(plan_path),
                "tagged sender READY identity drifted")

        primer = self.target_channel_primer(cell_tag, cell_dir)
        require(int(ready["ready_monotonic_ns"]) <=
                int(primer["send_start_monotonic_ns"]),
                "target-channel primer preceded sender READY")
        created_ns = time.monotonic_ns()
        lead_ns = int(float(self.registration_config["design"][
            "future_epoch_lead_s"]) * 1e9)
        epoch_ns = created_ns + lead_ns
        grant_ns = int(primer["latest_grant"]["receipt_monotonic_ns"])
        completed_ns = int(primer["completed_monotonic_ns"])
        max_gap_ns = int(float(self.config["traffic"]["target_channel_primer"][
            "max_grant_receipt_to_first_decision_ms"]) * 1e6)
        require(created_ns >= completed_ns,
                "shared epoch publication preceded primer completion")
        require(0 <= epoch_ns - grant_ns <= max_gap_ns,
                "registered shared epoch would violate primer freshness gate")
        epoch = {
            "schema": TS.EPOCH_SCHEMA, "cell_id": cell_id,
            "epoch_monotonic_ns": epoch_ns, "period_ns": period_ns,
            "created_monotonic_ns": created_ns,
            "sender_ready_sha256": C.sha256_file(sender_ready),
        }
        write_json_create(epoch_path, epoch)
        return {"receivers": receivers, "sender": sender,
                "frames": total_frames, "sender_csv": sender_csv,
                "sender_summary": sender_summary, "sender_ready": sender_ready,
                "epoch_contract": epoch_path,
                "target_channel_primer": primer, "epoch": {
                    **epoch, "shared_epoch_monotonic_ns": epoch_ns}}

    def replay_profile(
        self, profile: Any, *, epoch_ns: int,
        first_command: float, command_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
        return {"shared_epoch_monotonic_ns": epoch_ns,
                "samples": len(profile.samples), "commands_sent": sent,
                "commands_skipped": skipped, "targets_clamped": clamped}

    def audit_traffic(
        self, sessions: Mapping[str, Any], cell: C.Cell, cell_dir: Path,
    ) -> dict[str, Any]:
        summary_path = Path(sessions["sender_summary"])
        csv_path = Path(sessions["sender_csv"])
        require(summary_path.is_file() and csv_path.is_file(),
                "sender did not emit both durable outputs")
        sender = json.loads(summary_path.read_text(encoding="utf-8"))
        require(sender.get("schema") == TS.SUMMARY_SCHEMA,
                "sender summary schema mismatch")
        require(sender.get("decisions") == C.FRAMES_PER_CELL,
                "sender decision count mismatch")
        require(sender.get("epoch_monotonic_ns") ==
                sessions["epoch"]["shared_epoch_monotonic_ns"],
                "sender did not use the shared epoch")
        require(sender.get("sender_ready_sha256") ==
                C.sha256_file(Path(sessions["sender_ready"])),
                "sender summary is not bound to its READY record")
        require(sender.get("epoch_contract_sha256") ==
                C.sha256_file(Path(sessions["epoch_contract"])),
                "sender summary is not bound to its epoch contract")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(tuple(reader.fieldnames or ()) == TS.FRAME_FIELDS,
                    "sender frame CSV field order/schema mismatch")
            rows = list(reader)
        require(len(rows) == C.FRAMES_PER_CELL, "sender frame-row count mismatch")
        schedule_lags = [float(row["schedule_lag_ms"]) for row in rows]
        observed_max_lag = max(schedule_lags)
        reported_max_lag = sender.get("max_schedule_lag_ms")
        require(type(reported_max_lag) in (int, float)
                and float(reported_max_lag) == observed_max_lag,
                "sender maximum schedule lag does not reproduce from frame rows")
        require(0.0 <= observed_max_lag < 1_000.0 / C.FPS,
                "sender schedule lag reached or exceeded one decision period")
        expected_chunks = expected_payload = expected_socket_bytes = 0
        for index, row in enumerate(rows):
            block = cell.blocks[index // C.FRAMES_PER_BLOCK]
            require(row["schema"] == TS.FRAME_SCHEMA
                    and int(row["decision_index"]) == index
                    and row["tier"] == block.tier
                    and int(row["action_id"]) == block.action_id
                    and int(row["payload_bytes"]) == block.payload_bytes
                    and int(row["chunks_per_frame"]) == block.chunks_per_frame,
                    f"sender frame identity drift at decision {index}")
            require(int(row["epoch_monotonic_ns"]) ==
                    sessions["epoch"]["shared_epoch_monotonic_ns"],
                    f"sender epoch drift at decision {index}")
            require(int(row["datagrams_dropped_at_socket"]) == 0
                    and int(row["datagrams_handed_to_socket"]) ==
                        block.chunks_per_frame,
                    f"sender socket loss at decision {index}")
            require(int(row["application_payload_bytes_handed_to_socket"]) ==
                    block.payload_bytes,
                    f"sender payload accounting drift at decision {index}")
            wire = block.payload_bytes + block.chunks_per_frame * U3.HEADER.size
            require(int(row["bytes_handed_to_socket"]) == wire,
                    f"sender socket-byte accounting drift at decision {index}")
            expected_chunks += block.chunks_per_frame
            expected_payload += block.payload_bytes
            expected_socket_bytes += wire
        require(sender.get("datagrams_handed_to_socket") == expected_chunks
                and sender.get("datagrams_dropped_at_socket") == 0
                and sender.get("application_payload_bytes_handed_to_socket") ==
                    expected_payload
                and sender.get("bytes_handed_to_socket") == expected_socket_bytes,
                "sender summary does not exactly reproduce from frame rows")

        receivers: list[dict[str, Any]] = []
        for item in sessions["receivers"]:
            require(item["process"].process.poll() in (0, None),
                    f"receiver {item['tier']} exited nonzero")
            require(item["summary"].is_file(),
                    f"receiver {item['tier']} emitted no summary")
            value = json.loads(item["summary"].read_text(encoding="utf-8"))
            block = cell.blocks[int(item["block_index"])]
            require(value.get("schema") ==
                    "scenesense.ue_n3_structured_udp_receiver_summary.v1"
                    and value.get("clean_shutdown") is True
                    and value.get("malformed_datagrams") == 0
                    and value.get("stream_limit_exceeded_datagrams") == 0,
                    f"receiver {item['tier']} structural evidence failed")
            measurement = value.get("measurement", {})
            require(measurement.get("expected_frames_per_stream") ==
                    C.FRAMES_PER_BLOCK
                    and measurement.get("expected_chunks_per_frame") ==
                    block.chunks_per_frame,
                    f"receiver {item['tier']} expected identity drifted")
            streams = value.get("streams")
            require(type(streams) is list and len(streams) <= 1,
                    f"receiver {item['tier']} observed ambiguous streams")
            for stream in streams:
                require(stream.get("contract_mismatch_datagrams") == 0
                        and stream.get("outside_expected_range_datagrams") == 0
                        and stream.get("expected_chunks") ==
                            block.chunks_per_frame * C.FRAMES_PER_BLOCK
                        and stream.get("received_unique_chunks", 0) <=
                            block.chunks_per_frame * C.FRAMES_PER_BLOCK,
                        f"receiver {item['tier']} accounting is invalid")
            receivers.append({"tier": item["tier"],
                              "block_index": item["block_index"], **value})
        outcome = {"schema": "scenesense.run4_exact_traffic_audit.v1",
                   "sender": sender, "sender_csv_rows": len(rows),
                   "expected_datagrams": expected_chunks,
                   "expected_application_payload_bytes": expected_payload,
                   "expected_socket_bytes": expected_socket_bytes,
                   "sender_accounting_exact": True,
                   "receivers": receivers}
        write_json_create(cell_dir / "traffic_audit.json", outcome)
        return outcome

    def audit_ttracer_nonempty(self, cell_dir: Path) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        for source in ("gnb", "ue"):
            raw = cell_dir / "ttracer" / source / f"{source}.raw"
            require(raw.is_file() and raw.stat().st_size > 0,
                    f"{source} T-tracer raw output is empty")
            for event in self.config["telemetry"]["events"][source]:
                path = cell_dir / "ttracer" / source / "csv" / f"{event}.csv"
                require(path.is_file() and path.stat().st_size > 0,
                        f"required T-tracer output missing/empty: {source}/{event}")
                with path.open(encoding="utf-8", errors="replace") as handle:
                    count = sum(1 for _ in handle)
                require(count >= 2,
                        f"required T-tracer output has no evidence row: {source}/{event}")
                rows[f"{source}/{event}"] = {
                    "relative_path": str(path.relative_to(cell_dir)),
                    "rows_including_header": count,
                    "size_bytes": path.stat().st_size,
                    "sha256": C.sha256_file(path),
                }
        return {"verified": True, "required_outputs": rows}

    def _cell_cold_snapshot(self) -> dict[str, Any]:
        states = self._container_states()
        tunnels = self._tunnel_interfaces()
        orphans: dict[str, list[str]] = {}
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = self._run_external(
                ["sudo", "-n", "pgrep", "-a", "-x", name],
                timeout_name="process_probe")
            if found.returncode == 0 and found.stdout.strip():
                orphans[name] = found.stdout.strip().splitlines()
        for label, pattern in (
            ("tracer", "T/tracer/(record|multi|csv|replay)"),
            ("sender", "ue_mcs_backlog_run4_calibration_v1.tagged_sender"),
            ("receiver", "ue_n3_structured_udp_receiver"),
        ):
            found = self._run_external(
                ["pgrep", "-af", pattern], timeout_name="process_probe",
                stderr=subprocess.DEVNULL)
            values = [row for row in found.stdout.splitlines()
                      if "pgrep" not in row and str(os.getpid()) not in row]
            if values:
                orphans[label] = values
        cold = (not orphans and not tunnels
                and not any(value.startswith("true") for value in states.values()))
        return {"orphan_processes": orphans, "residual_ue_tunnels": tunnels,
                "core_containers": states, "cold": cold}

    def final_cold_state(self) -> dict[str, Any]:
        snapshot = self._cell_cold_snapshot()
        carla = self._run_external(
            ["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
            timeout_name="process_probe", stderr=subprocess.DEVNULL)
        carla_rows = [row for row in carla.stdout.splitlines() if "pgrep" not in row]
        snapshot.update({
            "schema": "scenesense.run4_calibration_final_cold_state.v1",
            "utc": utc_now(), "carla_processes": carla_rows,
            "cold": snapshot["cold"] and not carla_rows,
        })
        write_json_create(self.output_dir / "final_cold_state.json", snapshot)
        return snapshot

    def run_cell(self, cell: C.Cell, profile: Any) -> dict[str, Any]:
        cell_tag = f"{cell.run_index:02d}__{cell.cell_id}"
        cell_dir = self.output_dir / "cells" / cell_tag
        PE.assert_outside_protected_run(cell_dir, ROOT)
        cell_dir.mkdir(parents=True, exist_ok=False)
        command_log: list[dict[str, Any]] = []
        notes_at_start = len(self.notes)
        record: dict[str, Any] = {
            "cell_tag": cell_tag, **cell.to_json(), "trace_id": profile.trace_id,
            "registered_trace_sha256": profile.trace_sha256,
            "started_utc": utc_now(), "status": "FAILED",
        }
        try:
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
            blocks = [block.to_json() for block in cell.blocks]
            sessions = self.launch_traffic(
                cell_tag=cell_tag, cell_id=cell.cell_id, blocks=blocks,
                cell_dir=cell_dir, total_frames=C.FRAMES_PER_CELL)
            record["target_channel_primer"] = sessions["target_channel_primer"]
            record["shared_epoch"] = sessions["epoch"]
            record["profile_replay"] = self.replay_profile(
                profile,
                epoch_ns=sessions["epoch"]["shared_epoch_monotonic_ns"],
                first_command=first_command, command_log=command_log)
            self.finish_traffic(sessions)
            record["primer_first_decision"] = self.audit_primer_first_decision(sessions)
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
                record["cell_cold_state"] = {"cold": False,
                    "probe_failure": f"{type(exc).__name__}: {exc}"}
            if (record["status"] == "CAPTURED_PENDING_TEARDOWN"
                    and not cell_notes
                    and "ttracer" in record
                    and record["core_teardown"].get("stopped") is True
                    and record["cell_cold_state"].get("cold") is True):
                record["status"] = "CAPTURED"
            else:
                record["status"] = "FAILED"
                record.setdefault("failure", "cleanup/evidence gate failed")
            record["finished_utc"] = utc_now()
            write_json_create(cell_dir / "cell_record.json", record)
        return record

    def _write_seals(
        self, *, status: str, failure: str | None,
        plan_audit: Mapping[str, Any], cells: Sequence[Mapping[str, Any]],
        final_cold: Mapping[str, Any],
    ) -> None:
        excluded = {"RUN4_CALIBRATION_RESULT.json", "manifest.json",
                    f"{status}.json"}
        evidence = file_inventory(self.output_dir, excluded=excluded)
        result = {
            "schema": RESULT_SCHEMA, "status": status,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "failure": failure, "qualified": status == STATUS_CAPTURED,
            "original_capacity_qualification_overturned": False,
            "original_tier_selection_qualified": False,
            "perception_endorsement": False,
            "byte_only_queue_calibration": True,
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "packetization": self.registration_config["packetization"],
            "tiers": [tier.to_json() for tier in self.tiers],
            "plan_audit": dict(plan_audit), "cells": list(cells),
            "source_inventory": self.initial_inventory,
            "source_verifications": self.verifications,
            "inherited_lifecycle_audit": self.lifecycle_audit,
            "robust_amendment_binding": self.amendment["binding"],
            "lineage": self.lineage, "final_cold_state": dict(final_cold),
            "evidence_files": evidence, "created_utc": utc_now(),
        }
        result_path = self.output_dir / "RUN4_CALIBRATION_RESULT.json"
        write_json_create(result_path, result)
        manifest = {
            "schema": MANIFEST_SCHEMA, "status": status,
            "result_sha256": C.sha256_file(result_path),
            "source_inventory_sha256": self.initial_inventory["inventory_sha256"],
            "files": file_inventory(
                self.output_dir,
                excluded={"manifest.json", f"{status}.json"}),
        }
        manifest_path = self.output_dir / "manifest.json"
        write_json_create(manifest_path, manifest)
        write_json_create(self.output_dir / f"{status}.json", {
            "schema": TERMINAL_SCHEMA, "status": status,
            "result_sha256": C.sha256_file(result_path),
            "manifest_sha256": C.sha256_file(manifest_path),
            "source_inventory_sha256": self.initial_inventory["inventory_sha256"],
            "created_utc": utc_now(),
        })

    def run(self) -> int:
        status = STATUS_FAILED
        failure: str | None = None
        cells: list[dict[str, Any]] = []
        final_cold: dict[str, Any] = {}
        plan = C.build_cell_plan(
            ports=self.config["traffic"]["ports"], seed=C.CELL_ORDER_SEED)
        plan_audit = C.audit_cell_plan(plan)
        profiles = {row.profile_id: row for row in C.resolve_profiles()}

        def terminate(signum: int, _frame: Any) -> None:
            self.aborted = True
            raise RunError(f"received signal {signum}")

        for caught in (signal.SIGINT, signal.SIGTERM):
            signal.signal(caught, terminate)
        try:
            self.verify_identities("before_preflight")
            self.preflight_run4()
            write_json_create(self.output_dir / "plan.json", {
                "schema": "scenesense.run4_calibration_plan.v1",
                "plan_audit": plan_audit, "cells": [row.to_json() for row in plan],
                "tiers": [tier.to_json() for tier in self.tiers],
                "cell_order_seed": C.CELL_ORDER_SEED,
                "fit_validation_split": "WHOLE_CELL_DISJOINT_TRANSITION_DIRECTIONS",
                "validation_limitation": (
                    "Validation reverses the three FIT orders, so it extrapolates "
                    "across transition direction rather than providing an independent "
                    "RF replication."
                ),
            })
            for cell in plan:
                require(not self.aborted, "campaign aborted")
                self.verify_identities("before_cell")
                record = self.run_cell(cell, profiles[cell.profile_id])
                cells.append(record)
                require(record["status"] == "CAPTURED",
                        f"cell {cell.cell_id} failed: {record.get('failure')}")
            require(len(cells) == C.EXPECTED_CELLS,
                    f"captured {len(cells)}/{C.EXPECTED_CELLS} cells")
            status = STATUS_CAPTURED
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            # Mandatory belt-and-suspenders cleanup even though every cell has
            # its own finally block.  No exception can bypass these attempts.
            cleanup_notes = self.teardown_ran()
            core = self.stop_core()
            all_cleanup_notes = list(self.notes) + list(cleanup_notes)
            if all_cleanup_notes or not core.get("stopped"):
                failure = failure or (
                    f"final cleanup failed: {all_cleanup_notes}; {core}")
                status = STATUS_FAILED
            try:
                final_cold = self.final_cold_state()
            except Exception as exc:  # noqa: BLE001
                final_cold = {"cold": False,
                              "probe_failure": f"{type(exc).__name__}: {exc}"}
            if not final_cold.get("cold"):
                failure = failure or f"final host is not cold: {final_cold}"
                status = STATUS_FAILED
            try:
                # Deep re-open at final sealing detects any preserved evidence
                # drift during the long live campaign.
                C.verify_amendment(ROOT)
                self.verify_identities("final_sealing")
            except Exception as exc:  # noqa: BLE001
                failure = failure or f"final source/amendment gate failed: {exc}"
                status = STATUS_FAILED
            if status == STATUS_CAPTURED and failure is not None:
                status = STATUS_FAILED
            self._write_seals(status=status, failure=failure,
                              plan_audit=plan_audit, cells=cells,
                              final_cold=final_cold)
        print(json.dumps({"status": status, "failure": failure,
                          "cells_captured": sum(row.get("status") == "CAPTURED"
                                                for row in cells),
                          "cells_planned": C.EXPECTED_CELLS,
                          "output_dir": str(self.output_dir),
                          "cold": final_cold.get("cold")}, indent=2))
        return 0 if status == STATUS_CAPTURED else 1


NC_PRODUCTION_RECEIVER = "rl_agent/ue_n3_structured_udp_receiver.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("authorization-template")
    template.add_argument("--output-dir", type=Path, required=True)
    template.add_argument("--granted-by", required=True)
    template.add_argument("--granted-utc", required=True)
    template.add_argument("--expires-utc", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "authorization-template":
        value = AUTH.authorization_template(
            args.output_dir, granted_by=args.granted_by,
            granted_utc=args.granted_utc, expires_utc=args.expires_utc,
            repo_root=ROOT)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0

    # All expensive/offline identity checks happen before grant consumption;
    # no live service exists yet, and a failed check leaves the grant unspent.
    C.load_config(args.config)
    require(args.config.resolve() == DEFAULT_CONFIG.resolve(),
            "alternate config paths are not authorized; use the pinned default")
    C.verify_inherited_sources(ROOT)
    amendment = C.verify_amendment(ROOT)
    lifecycle = inherited_lifecycle_audit(Runner)
    inventory = C.source_inventory(ROOT)
    expected_root = (ROOT / C.load_config(args.config)["paths"]["output_root"]).resolve()
    require(args.output_dir.resolve().parent == expected_root,
            f"output must be a direct child of registered root {expected_root}")
    consumed = AUTH.consume_authorization(
        args.authorization, args.output_dir, output_root=expected_root,
        repo_root=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    lineage = AUTH.lineage_record(consumed)
    write_json_create(args.output_dir / "lineage.json", lineage)
    write_json_create(args.output_dir / "initial_source_inventory.json", inventory)
    write_json_create(args.output_dir / "inherited_lifecycle_audit.json", lifecycle)
    return Runner(args.config, args.output_dir,
                  initial_inventory=inventory, lineage=lineage,
                  amendment=amendment, lifecycle_audit=lifecycle).run()


if __name__ == "__main__":
    raise SystemExit(main())
