#!/usr/bin/env python3
"""Run bounded live UE-N3 controls without promoting an SNR bound.

The first executable mode is ``CLEAN_RECEIVER_CONTROL``.  It launches one
fresh single-UE RFsim RAN at the clean ``-50`` command, sends exactly 600
matched SSBURST frames over 60 seconds, verifies transport/radio evidence,
rechecks the clean actuator state, and tears the owned RAN down.  The future
command-search design is present in the config but is deliberately not
executable in this version.
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import rl_agent.ue_n2_oai_ul_calibration_smoke as n2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3_oai_ul_live_stage_v1.json"
CONFIG_SCHEMA = "scenesense.ue_n3_oai_ul_live_stage_config.v1"
MODE = "CLEAN_RECEIVER_CONTROL"
SUCCESS_STATUS = "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED"


class LiveStageFailure(n2.SmokeFailure):
    """Fail-closed UE-N3 live-stage error."""


def resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise LiveStageFailure(f"path escapes repository root: {relative}") from exc
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveStageFailure(message)


def validate_config(config: Mapping[str, Any], *, require_live_authority: bool) -> None:
    require(config.get("schema") == CONFIG_SCHEMA, "unexpected live-stage config schema")
    require(
        config.get("claim_boundary")
        == "CONTROL_AND_COMMAND_CALIBRATION_EVIDENCE_ONLY_NO_MAPPING_OR_BOUND_PROMOTION",
        "claim boundary mismatch",
    )
    authority = config["authority"]
    require(authority.get("implementation_and_offline_tests_authorized") is True,
            "implementation authority is absent")
    require(authority.get("carla_run_authorized") is False, "CARLA authority must remain false")
    require(authority.get("target_mapping_promotion_authorized") is False,
            "target mapping promotion must remain forbidden")
    require(authority.get("numeric_bound_promotion_authorized") is False,
            "numeric bound promotion must remain forbidden")
    require(authority.get("policy_training_authorized") is False,
            "policy training must remain forbidden")
    if require_live_authority:
        require(authority.get("oai_run_authorized") is True, "live OAI authority is absent")
        require(authority.get("socket_execution_authorized") is True,
                "live socket authority is absent")
        require(
            authority.get("live_authority_basis")
            == "USER_REQUEST_2026-08-21_CONTINUE_LOWER_OAI_SNR_SEARCH",
            "live authority basis is absent or unexpected",
        )

    traffic = config["traffic"]
    clean = config["modes"][MODE]
    require(float(traffic["fps"]) == 10.0, "clean control must use 10 Hz")
    require(int(clean["sender_frames"]) == 600, "clean control must send 600 frames")
    require(float(clean["service_duration_s"]) == 60.0,
            "clean control must span 60 seconds")
    require(int(traffic["frame_bytes"]) == 12_500, "frame size must be 12500 bytes")
    require(int(traffic["chunk_bytes"]) == 12_500, "chunk size must be 12500 bytes")
    require(int(traffic["expected_chunks_per_frame"]) == 1,
            "clean control must use one chunk per frame")
    gates = config["gates"]
    require(math.isclose(float(gates["primary_complete_frame_ratio"]), 0.99),
            "primary delivery gate must be 99 percent")
    require([float(value) for value in gates["sensitivity_complete_frame_ratios"]]
            == [0.95, 0.90], "sensitivity gates must be 95 and 90 percent")
    require(int(gates["maximum_interarrival_gaps_gte_1s"]) == 0,
            "one-second outage gate must be zero")
    require(int(gates["required_valid_streams"]) == 1,
            "exactly one receiver stream is required")
    require(int(gates["maximum_stream_limit_exceeded_datagrams"]) == 0,
            "second-source datagrams must be rejected visibly")
    require(gates["expected_receiver_source_ipv4"] == ["192.168.70.134"],
            "expected ext-DN NAT source must be frozen")
    require(config["analysis"]["direct_ul_bler_zero_fill_authorized"] is False,
            "direct UL BLER must not be zero-filled")


def validate_predecessor(block: Mapping[str, Any]) -> None:
    directory = resolve(str(block["directory"]))
    manifest = directory / str(block["manifest"])
    terminal = directory / str(block["terminal"])
    require(manifest.is_file() and terminal.is_file(),
            f"predecessor evidence missing: {directory}")
    require(sha256(manifest) == block["manifest_sha256"],
            f"predecessor manifest hash drift: {directory}")
    require(sha256(terminal) == block["terminal_sha256"],
            f"predecessor terminal hash drift: {directory}")
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    require(payload.get("status") == block["required_status"],
            f"predecessor terminal status mismatch: {directory}")


def classify_receiver_gate(
    *,
    receiver_summary: Mapping[str, Any],
    sender_frames: int,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    streams = list(receiver_summary.get("streams", []))
    stream = streams[0] if len(streams) == 1 else {}
    complete_frames = int(stream.get("complete_frames", 0) or 0)
    expected_frames = int(stream.get("expected_frames", 0) or 0)
    ratio = complete_frames / expected_frames if expected_frames else 0.0
    observed_stream_id = str(stream.get("stream_id", ""))
    observed_source_ipv4 = observed_stream_id.rsplit(":", 1)[0]
    structural_pass = all([
        receiver_summary.get("status") == "CAPTURED",
        bool(receiver_summary.get("clean_shutdown")),
        receiver_summary.get("stop_reason") == "DURATION_COMPLETE",
        int(receiver_summary.get("valid_stream_count", 0) or 0)
        == int(gates["required_valid_streams"]),
        int(receiver_summary.get("stream_limit_exceeded_datagrams", 0) or 0)
        <= int(gates["maximum_stream_limit_exceeded_datagrams"]),
        observed_source_ipv4 in gates["expected_receiver_source_ipv4"],
        int(receiver_summary.get("malformed_datagrams", 0) or 0)
        <= int(gates["maximum_malformed_datagrams"]),
        int(stream.get("contract_mismatch_datagrams", 0) or 0)
        <= int(gates["maximum_contract_mismatch_datagrams"]),
        int(stream.get("outside_expected_range_datagrams", 0) or 0)
        <= int(gates["maximum_outside_expected_range_datagrams"]),
        int(stream.get("interarrival_gaps_over_one_second", 0) or 0)
        <= int(gates["maximum_interarrival_gaps_gte_1s"]),
        sender_frames == expected_frames == 600,
    ])
    primary_threshold = float(gates["primary_complete_frame_ratio"])
    sensitivity = {
        f"delivery_{int(round(float(threshold) * 100))}_pass": (
            structural_pass and ratio >= float(threshold)
        )
        for threshold in gates["sensitivity_complete_frame_ratios"]
    }
    return {
        "structural_pass": structural_pass,
        "sender_frames": sender_frames,
        "expected_frames": expected_frames,
        "complete_frames": complete_frames,
        "complete_frame_ratio": ratio,
        "observed_stream_id": observed_stream_id,
        "observed_source_ipv4": observed_source_ipv4,
        "expected_receiver_source_ipv4": gates["expected_receiver_source_ipv4"],
        "primary_threshold": primary_threshold,
        "primary_usable_service_pass": structural_pass and ratio >= primary_threshold,
        **sensitivity,
        "goodput_role": gates["goodput_role"],
        "receiver_unique_datagram_goodput_mbps": stream.get(
            "unique_datagram_goodput_mbps"
        ),
        "receiver_unique_payload_goodput_mbps": stream.get(
            "unique_payload_goodput_mbps"
        ),
    }


class CleanControlRunner(n2.Runner):
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
        validate_config(config, require_live_authority=False)
        super().__init__(config_path, output_dir)
        self.mode = MODE
        self.receiver_ready_observed = False
        self.receiver_process: n2.ManagedProcess | None = None
        self.sender_process: n2.ManagedProcess | None = None

    def _named_process(self, name: str) -> n2.ManagedProcess | None:
        return next((item for item in self.processes if item.name == name), None)

    def _carla_process_rows(self) -> list[str]:
        result = subprocess.run(
            ["ps", "-eo", "pid=,comm="], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return [
            line.strip() for line in result.stdout.splitlines()
            if re.search(r"(?:carla|unreal)", line, flags=re.IGNORECASE)
        ]

    def preflight(self) -> None:
        validate_config(self.config, require_live_authority=True)
        predecessors = self.config["predecessors"]
        validate_predecessor(predecessors["authoritative_n3_plan"])
        validate_predecessor(predecessors["ue_n2_evidence"])
        n2.run_checked([
            str(self.path(self.config["paths"]["python"])), "-m",
            "rl_agent.ue_n1_freeze_oai_ul_actuator_v2", "--validate",
            str(self.path(predecessors["ue_n1_bundle"])),
        ], timeout=30)

        checked_seals: list[dict[str, Any]] = []
        for entry in self.config["runtime_seals"]:
            path = self.path(str(entry["path"]))
            require(path.is_file(), f"sealed runtime file missing: {entry['path']}")
            observed = sha256(path)
            require(observed == entry["sha256"], f"runtime hash drift: {entry['path']}")
            checked_seals.append({"path": entry["path"], "sha256": observed})

        n2.run_checked(["sudo", "-n", "true"])
        for container in self.config["radio"]["core_containers"]:
            state = n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container,
            ]).stdout.strip()
            require(state.startswith("true") and "unhealthy" not in state,
                    f"core container is not ready: {container}={state!r}")

        for process_name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(
                ["sudo", "-n", "pgrep", "-a", "-x", process_name],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            require(found.returncode != 0 or not found.stdout.strip(),
                    f"cold-RAN gate failed: {found.stdout.strip()}")
        require(not n2.oai_tunnel_interfaces(), "cold-RAN gate found an OAI UE tunnel")

        preflight = self.config["preflight"]
        carla_ports_busy = [
            int(port) for port in preflight["carla_ports"]
            if not n2.port_is_free(int(port))
        ]
        carla_processes = self._carla_process_rows()
        if preflight["fail_if_carla_active"]:
            require(not carla_ports_busy and not carla_processes,
                    f"CARLA must be stopped: ports={carla_ports_busy} processes={carla_processes}")
        busy = [
            int(port) for port in preflight["required_host_tcp_ports"]
            if not n2.port_is_free(int(port))
        ]
        require(not busy, f"required host TCP ports are busy: {busy}")

        ext_dn = self.config["radio"]["ext_dn_container"]
        pid = int(n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", ext_dn,
        ]).stdout.strip())
        addresses = n2.run_checked([
            "sudo", "-n", "nsenter", "-t", str(pid), "-n",
            "ip", "-j", "-4", "addr", "show",
        ]).stdout
        require(self.config["radio"]["ext_dn_ip"] in addresses,
                "configured ext-DN address is absent from its network namespace")
        udp_state = n2.run_checked([
            "sudo", "-n", "nsenter", "-t", str(pid), "-n",
            "ss", "-H", "-lun",
        ]).stdout
        udp_port = int(preflight["structured_udp_port"])
        require(not re.search(rf":{udp_port}(?:\s|$)", udp_state),
                f"structured UDP port is busy in ext-DN namespace: {udp_port}")
        n2.atomic_json(self.output_dir / "preflight.json", {
            "status": "PASSED",
            "mode": self.mode,
            "checked_at": n2.utc_now(),
            "runtime_seals": checked_seals,
            "carla_ports_busy": carla_ports_busy,
            "carla_processes": carla_processes,
            "ext_dn_pid": pid,
            "structured_udp_port": udp_port,
        })

    def start_traffic(self) -> None:
        require(self.ue_ip is not None, "cannot start traffic before UE IP discovery")
        traffic = self.config["traffic"]
        mode = self.config["modes"][self.mode]
        ext_dn_pid = int(n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
            self.config["radio"]["ext_dn_container"],
        ]).stdout.strip())
        traffic_root = self.output_dir / "traffic"
        traffic_root.mkdir(parents=True, exist_ok=True)
        ready = traffic_root / "receiver_ready.json"
        receiver = [
            "sudo", "-n", "nsenter", "-t", str(ext_dn_pid), "-n",
            "/usr/bin/python3", str(self.path(self.config["paths"]["receiver"])),
            "--bind-host", self.config["radio"]["ext_dn_ip"],
            "--port", str(traffic["remote_port"]),
            "--events-jsonl", str(traffic_root / "receiver_events.jsonl"),
            "--summary-json", str(traffic_root / "receiver_summary.json"),
            "--ready-json", str(ready),
            "--duration-s", str(mode["receiver_capture_duration_s"]),
            "--expected-first-frame", "0",
            "--expected-frames", str(mode["sender_frames"]),
            "--expected-chunks-per-frame", str(traffic["expected_chunks_per_frame"]),
            "--max-streams", str(traffic["receiver_max_streams"]),
            "--reorder-window-frames", str(traffic["receiver_reorder_window_frames"]),
            "--max-chunks-per-frame", str(traffic["receiver_max_chunks_per_frame"]),
            "--socket-receive-buffer-bytes", str(traffic["receiver_socket_buffer_bytes"]),
        ]
        self.receiver_process = self.spawn(
            "structured_receiver", receiver, "logs/structured_receiver.log",
            root_owned=True,
        )
        deadline = time.monotonic() + float(traffic["receiver_ready_timeout_s"])
        while time.monotonic() < deadline:
            if self.receiver_process.process.poll() is not None:
                raise LiveStageFailure("structured receiver exited before READY")
            if ready.is_file():
                payload = json.loads(ready.read_text(encoding="utf-8"))
                require(payload.get("status") == "READY", "receiver READY record is invalid")
                require(int(payload.get("port", -1)) == int(traffic["remote_port"]),
                        "receiver READY port mismatch")
                self.receiver_ready_observed = True
                break
            time.sleep(0.05)
        require(self.receiver_ready_observed, "structured receiver READY timeout")

        sender = [
            str(self.path(self.config["paths"]["python"])),
            str(self.path(self.config["paths"]["sender"])),
            "--bind-host", str(self.ue_ip),
            "--remote-host", self.config["radio"]["ext_dn_ip"],
            "--remote-port", str(traffic["remote_port"]),
            "--fps", str(traffic["fps"]),
            "--frames", str(mode["sender_frames"]),
            "--frame-bytes", str(traffic["frame_bytes"]),
            "--chunk-bytes", str(traffic["chunk_bytes"]),
            "--idle-before-s", str(traffic["sender_idle_before_s"]),
            "--cooldown-s", str(traffic["sender_cooldown_s"]),
            "--log-csv", str(traffic_root / "sender.csv"),
        ]
        self.sender_process = self.spawn("structured_sender", sender, "logs/structured_sender.log")

    def wait_fresh_control_traffic(self, baseline_count: int) -> None:
        require(self.live_csv is not None, "live PUSCH collector is absent")
        mode = self.config["modes"][self.mode]
        deadline = time.monotonic() + float(self.config["traffic"]["fresh_pusch_timeout_s"])
        events = self.output_dir / "traffic/receiver_events.jsonl"
        while time.monotonic() < deadline:
            fresh = self.live_csv.snapshot()[baseline_count:]
            rntis: set[int] = set()
            for _wall_ns, _mono_ns, line in fresh:
                parts = line.split(",")
                try:
                    rntis.add(int(parts[1]))
                except (IndexError, ValueError):
                    continue
            if self.sender_process is not None and self.sender_process.process.poll() is not None:
                raise LiveStageFailure("sender exited before fresh-PUSCH gate")
            if self.receiver_process is not None and self.receiver_process.process.poll() is not None:
                raise LiveStageFailure("receiver exited before fresh-PUSCH gate")
            if (
                len(fresh) >= int(mode["minimum_clean_pusch_samples"])
                and len(rntis) == 1
                and events.is_file()
                and events.stat().st_size > 0
            ):
                self.current_rnti = next(iter(rntis))
                n2.atomic_json(self.output_dir / "fresh_traffic_gate.json", {
                    "status": "PASSED",
                    "baseline_pusch_rows": baseline_count,
                    "fresh_pusch_rows": len(fresh),
                    "current_rnti": self.current_rnti,
                    "receiver_ready_observed": self.receiver_ready_observed,
                    "matched_receiver_events_observed": True,
                })
                return
            time.sleep(0.1)
        raise LiveStageFailure("fresh matched traffic/PUSCH gate timed out")

    def wait_traffic_completion(self) -> None:
        require(self.sender_process is not None and self.receiver_process is not None,
                "traffic processes were not started")
        mode = self.config["modes"][self.mode]
        sender_timeout = float(mode["service_duration_s"]) + 10.0
        try:
            sender_rc = self.sender_process.process.wait(timeout=sender_timeout)
        except subprocess.TimeoutExpired as exc:
            raise LiveStageFailure("sender did not finish the complete 600-frame schedule") from exc
        require(sender_rc == 0, f"structured sender failed rc={sender_rc}")
        remaining = max(
            10.0,
            float(mode["receiver_capture_duration_s"]) - float(mode["service_duration_s"]) + 5.0,
        )
        try:
            receiver_rc = self.receiver_process.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise LiveStageFailure("structured receiver did not close its capture") from exc
        require(receiver_rc == 0, f"structured receiver failed rc={receiver_rc}")
        summary = self.output_dir / "traffic/receiver_summary.json"
        require(summary.is_file(), "structured receiver summary is missing")

    def connectivity_snapshot(self) -> dict[str, Any]:
        gnb = self._named_process("gnb")
        ue = self._named_process("ue")
        require(gnb is not None and ue is not None, "owned RAN process records are missing")
        processes_alive = gnb.process.poll() is None and ue.process.poll() is None
        interface = self.config["radio"]["ue_interface"]
        address = subprocess.run(
            ["ip", "-j", "-4", "addr", "show", "dev", interface],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        observed_ips: list[str] = []
        if address.returncode == 0:
            try:
                observed_ips = [
                    str(info["local"])
                    for row in json.loads(address.stdout)
                    for info in row.get("addr_info", [])
                    if info.get("family") == "inet" and info.get("local")
                ]
            except (json.JSONDecodeError, KeyError, TypeError):
                observed_ips = []
        ping = subprocess.run(
            ["ping", "-I", interface, "-c", "3", "-W", "2",
             self.config["radio"]["ext_dn_ip"]],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        snapshot = {
            "processes_alive": processes_alive,
            "observed_tunnel_ipv4": observed_ips,
            "same_tunnel_ipv4": observed_ips == [self.ue_ip],
            "ext_dn_ping_pass": ping.returncode == 0,
            "current_rnti": self.current_rnti,
            "status": "PASSED" if (
                processes_alive and observed_ips == [self.ue_ip] and ping.returncode == 0
                and self.current_rnti is not None
            ) else "FAILED",
        }
        n2.atomic_json(self.output_dir / "connectivity_gate.json", snapshot)
        require(snapshot["status"] == "PASSED", f"connectivity gate failed: {snapshot}")
        return snapshot

    def cleanup(self, *, strict: bool = False) -> list[str]:
        """Extend N2 cleanup with the ext-DN UDP receiver-port gate."""

        errors = super().cleanup(strict=False)
        port = int(self.config["preflight"]["structured_udp_port"])
        busy = False
        check_error: str | None = None
        try:
            pid = int(n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
                self.config["radio"]["ext_dn_container"],
            ]).stdout.strip())
            udp_state = n2.run_checked([
                "sudo", "-n", "nsenter", "-t", str(pid), "-n",
                "ss", "-H", "-lun",
            ]).stdout
            busy = bool(re.search(rf":{port}(?:\s|$)", udp_state))
            if busy:
                errors.append(f"ext-DN structured UDP port survived cleanup: {port}")
        except Exception as exc:
            check_error = str(exc)
            errors.append(f"ext-DN structured UDP cleanup check failed: {exc}")
        report_path = self.output_dir / "cleanup_report.json"
        report: dict[str, Any] = {}
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}
        report.update({
            "clean": not errors,
            "errors": errors,
            "ext_dn_structured_udp_port": port,
            "ext_dn_structured_udp_port_busy": busy,
            "ext_dn_structured_udp_check_error": check_error,
            "n3_cleanup_checked_at": n2.utc_now(),
        })
        n2.atomic_json(report_path, report)
        if strict and errors:
            raise LiveStageFailure("cleanup verification failed: " + "; ".join(errors))
        return errors

    def transport_and_radio_summary(self) -> dict[str, Any]:
        sender_path = self.output_dir / "traffic/sender.csv"
        receiver_path = self.output_dir / "traffic/receiver_summary.json"
        require(sender_path.is_file() and receiver_path.is_file(),
                "traffic output is incomplete")
        with sender_path.open(newline="", encoding="utf-8") as handle:
            sender_rows = list(csv.DictReader(handle))
        sender_frames = len({int(row["frame_index"]) for row in sender_rows})
        receiver_summary = json.loads(receiver_path.read_text(encoding="utf-8"))
        receiver_gate = classify_receiver_gate(
            receiver_summary=receiver_summary,
            sender_frames=sender_frames,
            gates=self.config["gates"],
        )
        n2.atomic_json(self.output_dir / self.config["output"]["receiver_gate"], receiver_gate)
        require(receiver_gate["primary_usable_service_pass"],
                f"clean receiver control missed the 99-percent gate: {receiver_gate}")

        require(self.live_csv is not None and self.current_rnti is not None,
                "live radio evidence is unavailable")
        snr: list[float] = []
        rntis: set[int] = set()
        for _wall_ns, _mono_ns, line in self.live_csv.snapshot():
            parts = line.split(",")
            try:
                rnti = int(parts[1])
                value = float(parts[4]) / 10.0
            except (IndexError, ValueError):
                continue
            rntis.add(rnti)
            if rnti == self.current_rnti and math.isfinite(value):
                snr.append(value)
        require(rntis == {self.current_rnti},
                f"single-current-RNTI gate failed: {rntis}")
        require(snr, "no finite live PUSCH SNR observations")
        return {
            "schema": "scenesense.ue_n3_clean_receiver_control_summary.v1",
            "status": SUCCESS_STATUS,
            "mode": self.mode,
            "claim_boundary": self.config["claim_boundary"],
            "receiver_gate": receiver_gate,
            "live_pusch_observation_count": len(snr),
            "instantaneous_pusch_snr_db_p50": n2.percentile(snr, 0.50),
            "instantaneous_pusch_snr_db_p05": n2.percentile(snr, 0.05),
            "instantaneous_pusch_snr_db_p95": n2.percentile(snr, 0.95),
            "current_rnti": self.current_rnti,
            "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED",
            "mapping_promoted": False,
            "numeric_bound_promoted": False,
        }

    def add_extracted_mcs_summary(self, summary: dict[str, Any]) -> None:
        path = self.output_dir / "ttracer/gnb/csv/GNB_MAC_UL_MCS_DECISION.csv"
        require(path.is_file(), "extracted scheduler MCS evidence is absent")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row.get("rnti") and int(row["rnti"]) == self.current_rnti
            ]
        require(rows, "scheduler MCS evidence has no current-RNTI rows")
        require(all(
            int(row.get("mcs_table", -999))
            == int(self.config["analysis"]["scheduler_required_mcs_table"])
            and int(row.get("force_ul_mcs", -999))
            == int(self.config["analysis"]["scheduler_required_force_ul_mcs"])
            for row in rows
        ), "scheduler policy seal failed in extracted evidence")
        selected = [float(row["selected_mcs"]) for row in rows]
        final = [float(row["final_mcs"]) for row in rows]
        ema = [float(row["avg_snr_x10"]) / 10.0 for row in rows]
        mcs_summary = {
            "row_count": len(rows),
            "selected_mcs_p50": n2.percentile(selected, 0.50),
            "final_mcs_p50": n2.percentile(final, 0.50),
            "scheduler_ema_snr_db_p50": n2.percentile(ema, 0.50),
            "mcs_table": self.config["analysis"]["scheduler_required_mcs_table"],
            "force_ul_mcs": self.config["analysis"]["scheduler_required_force_ul_mcs"],
        }
        n2.atomic_json(self.output_dir / self.config["output"]["mcs_summary"], mcs_summary)
        summary["scheduler"] = mcs_summary

    def manifest(self, status: str) -> None:
        excluded = {
            self.config["output"]["manifest"],
            self.config["output"]["failure"],
            self.config["output"].get("terminal_clean", ""),
        }
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name not in excluded:
                files.append({
                    "path": str(path.relative_to(self.output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
        n2.atomic_json(self.output_dir / self.config["output"]["manifest"], {
            "schema": "scenesense.ue_n3_oai_ul_live_stage_manifest.v1",
            "status": status,
            "mode": self.mode,
            "created_at": self.started_at,
            "completed_at": n2.utc_now(),
            "config_path": str(self.config_path),
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "outputs": files,
        })

    def run(self) -> int:
        n2.atomic_json(self.output_dir / self.config["output"]["resolved_config"], self.config)
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise LiveStageFailure(f"received termination signal {signal.Signals(signum).name}")

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
            time.sleep(float(self.config["telemetry"]["collector_tail_s"]))
            require(self.live_csv is not None, "live PUSCH collector did not start")
            baseline = self.live_csv.count()
            self.start_traffic()
            self.wait_fresh_control_traffic(baseline)
            model_index = self.open_and_validate_telnet()
            self.wait_traffic_completion()
            connectivity = self.connectivity_snapshot()
            summary = self.transport_and_radio_summary()
            summary["connectivity_gate"] = connectivity
            self.restore(model_index)
            self.write_command_log()
            self.cleanup(strict=True)
            self.extract_ttracer()
            self.write_raw_limit_record()
            self.add_extracted_mcs_summary(summary)
            summary["restored_to_clean_minus50"] = self.restored
            summary["cleanup_clean"] = True
            n2.atomic_json(self.output_dir / self.config["output"]["summary"], summary)
            self.manifest(SUCCESS_STATUS)
            terminal_name = self.config["output"]["terminal_clean"]
            terminal = {
                "status": SUCCESS_STATUS,
                "mode": self.mode,
                "primary_usable_service_pass": True,
                "mapping_promoted": False,
                "numeric_bound_promoted": False,
                "clean_restore_verified": self.restored,
                "manifest_sha256": sha256(
                    self.output_dir / self.config["output"]["manifest"]
                ),
                "next": "COMMAND_CALIBRATION_SEARCH",
            }
            n2.atomic_json(self.output_dir / terminal_name, terminal)
            print(json.dumps({"output_dir": str(self.output_dir), **terminal}, sort_keys=True))
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            try:
                self.write_command_log()
            except Exception:
                pass
            cleanup_errors = self.cleanup(strict=False)
            failure = {
                "status": "FAILED",
                "mode": self.mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "clean_restore_verified": self.restored,
                "cleanup_errors": cleanup_errors,
                "mapping_promoted": False,
                "numeric_bound_promoted": False,
                "failed_at": n2.utc_now(),
            }
            n2.atomic_json(self.output_dir / self.config["output"]["failure"], failure)
            self.manifest("FAILED")
            print(json.dumps({"output_dir": str(self.output_dir), **failure}, sort_keys=True),
                  file=sys.stderr)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default=MODE, choices=[MODE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return CleanControlRunner(Path(args.config), Path(args.output_dir)).run()


if __name__ == "__main__":
    raise SystemExit(main())
