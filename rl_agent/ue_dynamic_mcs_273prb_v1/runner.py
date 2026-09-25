#!/usr/bin/env python3
"""Network-only live capture for target-radio dynamic prior UL-MCS evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_dynamic_mcs_273prb_v1 import analysis as A
from rl_agent.ue_dynamic_mcs_273prb_v1 import contract as C
from rl_agent.ue_mcs_backlog_calibration_v1 import contract as BC
from rl_agent.ue_mcs_backlog_calibration_v1 import runner as BR
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as NR


ROOT = C.ROOT
BASE_CONFIG = ROOT / C.SOURCE_PINS["target_radio_config"][0]
OUTPUT_ROOT = ROOT / "rl_agent/experiments/ue_dynamic_mcs_273prb_v1"


class CaptureError(RuntimeError):
    """A live qualification gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.01))


def _container_states(names: Sequence[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    for name in names:
        inspect_result = subprocess.run(
            ["sudo", "-n", "docker", "inspect", "-f",
             "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
             name],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        states[name] = (
            inspect_result.stdout.strip()
            if inspect_result.returncode == 0
            else "ABSENT"
        )
    return states


class Runner(NR.Runner):
    """Reuse the hash-bound target-radio lifecycle, not its scientific plan."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(BASE_CONFIG, output_dir)
        self.profile_records: list[dict[str, Any]] = []

    def preflight_before_launcher(self, cell_tag: str) -> dict[str, Any]:
        C.verify_sources(ROOT)
        radio_binding = RB.verify(f"{cell_tag}:prelaunch", ROOT)
        radio = self.config["radio"]
        BR.n2.run_checked(["sudo", "-n", "true"])
        self.assert_cold_ran(f"{cell_tag}:prelaunch")
        carla = subprocess.run(
            ["pgrep", "-af", "CarlaUE4|CarlaUnreal"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        require(not carla.stdout.strip(), "CARLA is running; capture is network-only")
        states = _container_states(tuple(radio["core_containers"]))
        require(
            not any(value.startswith("true") for value in states.values()),
            "CN5G is already active; exact pre/post state cannot be restored: "
            f"{states}",
        )
        ports = [
            4043, int(self.config["actuator"]["telnet_port"]),
            int(self.config["telemetry"]["gnb_port"]),
            int(self.config["telemetry"]["ue_port"]),
            int(self.config["telemetry"]["gnb_relay_port"]),
            int(self.config["telemetry"]["ue_relay_port"]),
        ]
        busy = [port for port in ports if not BR.n2.port_is_free(port)]
        require(not busy, f"required ports are busy: {busy}")
        return {
            "utc": utc_now(), "cell_tag": cell_tag,
            "radio_profile_id": C.RADIO_PROFILE_ID,
            "core_before": states, "cold_ran": True,
            "cold_tunnels": True, "carla_absent": True,
            "cuda_or_model_process_started_by_this_capture": False,
            "free_ports": ports,
            "radio_binding": radio_binding,
        }

    def bind_post_launcher_context(self) -> dict[str, Any]:
        radio = self.config["radio"]
        states = _container_states(tuple(radio["core_containers"]))
        require(
            all(value.startswith("true") and "unhealthy" not in value
                for value in states.values()),
            f"launcher did not leave a healthy core: {states}",
        )
        container = str(radio["edge_container"])
        edge_host = BR.n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container,
        ]).stdout.strip()
        require(bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", edge_host)),
                f"invalid ext-DN address {edge_host!r}")
        local_table = BR.n2.run_checked(
            ["ip", "route", "show", "table", "local"]
        ).stdout
        require(f"local {edge_host} " not in local_table,
                "ext-DN address is host-local and would bypass the radio")
        self.edge_host = edge_host
        self.edge_pid = int(BR.n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", container,
        ]).stdout.strip())
        return {
            "core_after_launcher": states,
            "edge_container": container,
            "edge_host": edge_host,
            "edge_pid": self.edge_pid,
        }

    def stop_core_and_assert_cold(self) -> dict[str, Any]:
        cn_dir = ROOT / "OAI/oai-cn5g"
        compose_result = subprocess.run(
            ["sudo", "-n", "docker", "compose", "down", "--remove-orphans"],
            cwd=str(cn_dir), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        states = _container_states(tuple(self.config["radio"]["core_containers"]))
        softmodem_lines: list[str] = []
        for process_name in ("nr-softmodem", "nr-uesoftmodem"):
            pgrep_result = subprocess.run(
                ["pgrep", "-a", "-x", process_name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                softmodem_lines.append(pgrep_result.stdout.strip())
            elif pgrep_result.returncode not in (0, 1):
                softmodem_lines.append(
                    f"{process_name}: pgrep returncode={pgrep_result.returncode}"
                )
        softmodems = "\n".join(softmodem_lines)
        tunnels = BR.n2.oai_tunnel_interfaces()
        ports = [
            int(self.config["actuator"]["telnet_port"]),
            int(self.config["telemetry"]["gnb_port"]),
            int(self.config["telemetry"]["ue_port"]),
            int(self.config["telemetry"]["gnb_relay_port"]),
            int(self.config["telemetry"]["ue_relay_port"]),
        ]
        busy = [port for port in ports if not BR.n2.port_is_free(port)]
        cold = (
            compose_result.returncode == 0
            and not any(value.startswith("true") for value in states.values())
            and not softmodems and not tunnels and not busy
        )
        return {
            "docker_compose_down_returncode": compose_result.returncode,
            "docker_compose_down_tail": compose_result.stdout[-1000:],
            "core_after": states,
            "softmodems_after": softmodems,
            "tunnels_after": tunnels,
            "busy_ports_after": busy,
            "host_cold": cold,
        }

    def launch_receiver_and_sender(
        self, *, plan: C.ProfilePlan, cell_tag: str, cell_dir: Path,
        start_monotonic_ns: int,
    ) -> dict[str, Any]:
        assert self.edge_pid is not None and self.edge_host is not None
        assert self.ue_ip is not None
        receiver_events = cell_dir / "receiver_events.jsonl"
        receiver_summary = cell_dir / "receiver_summary.json"
        receiver_ready = cell_dir / "receiver_ready.json"
        duration_s = C.FRAMES_PER_PROFILE * C.PERIOD_NS / 1e9 + C.RECEIVER_TAIL_S
        receiver = self.spawn(
            f"receiver_{cell_tag}",
            [
                "sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
                sys.executable, str(ROOT / C.SOURCE_PINS["production_receiver"][0]),
                "--bind-host", "0.0.0.0", "--port", str(plan.port),
                "--events-jsonl", str(receiver_events),
                "--summary-json", str(receiver_summary),
                "--ready-json", str(receiver_ready),
                "--duration-s", f"{duration_s:.3f}",
                "--expected-frames", str(C.FRAMES_PER_PROFILE),
                "--expected-chunks-per-frame", str(C.PROBE_CHUNKS_PER_FRAME),
                "--max-chunks-per-frame", "16",
                "--socket-receive-buffer-bytes",
                str(self.config["traffic"]["receive_buffer_bytes"]),
            ],
            f"cells/{cell_tag}/logs/receiver.log", root_owned=True,
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not receiver_ready.is_file():
            require(receiver.process.poll() is None, "receiver exited before READY")
            time.sleep(0.1)
        require(receiver_ready.is_file(), "receiver did not report READY")

        sender_csv = cell_dir / "sender_decisions.csv"
        sender_summary = cell_dir / "sender_summary.json"
        sender = self.spawn(
            f"sender_{cell_tag}",
            [
                sys.executable, "-m",
                "rl_agent.ue_dynamic_mcs_273prb_v1.probe_sender",
                "--profile-id", plan.profile_id,
                "--trace-id", plan.trace_id,
                "--bind-host", self.ue_ip,
                "--remote-host", self.edge_host,
                "--remote-port", str(plan.port),
                "--start-monotonic-ns", str(start_monotonic_ns),
                "--period-ns", str(C.PERIOD_NS),
                "--frames", str(C.FRAMES_PER_PROFILE),
                "--payload-bytes", str(C.PROBE_PAYLOAD_BYTES),
                "--payload-seed", str(2026092402 + plan.run_index),
                "--send-buffer-bytes", str(self.config["traffic"]["send_buffer_bytes"]),
                "--log-csv", str(sender_csv),
                "--summary-json", str(sender_summary),
            ],
            f"cells/{cell_tag}/logs/sender.log",
        )
        return {
            "receiver": receiver, "sender": sender,
            "receiver_events": receiver_events,
            "receiver_summary": receiver_summary,
            "sender_csv": sender_csv, "sender_summary": sender_summary,
        }

    def replay_profile(
        self, *, plan: C.ProfilePlan, samples: Sequence[Mapping[str, Any]],
        start_monotonic_ns: int, command_log: list[dict[str, Any]],
        schedule_path: Path,
    ) -> None:
        fields = list(A.PROFILE_SCHEDULE_FIELDS)
        with schedule_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            last_command: float | None = None

            # Prime sample zero before the sender is allowed to reach its first
            # action boundary. The same mapped value is recorded on row zero.
            target0 = float(samples[0]["target_snr_db"])
            mapped0, clamped0 = BR.inverse_interpolate(target0, self.anchors)
            require(not clamped0, "sample-zero target would be clamped")
            mapped0 = BR.round_to_granularity(
                mapped0, float(self.config["actuator"]["command_granularity_db"])
            )
            self.send_noise(
                mapped0, reason="PROFILE_PRIME", log=command_log,
                profile_id=plan.profile_id, step_index=0, target_snr_db=target0,
                clamped=False,
            )
            prime = command_log[-1]
            last_command = mapped0

            for index, sample in enumerate(samples):
                action_open = start_monotonic_ns + index * C.PERIOD_NS
                due = action_open - C.COMMAND_GUARD_NS
                target = float(sample["target_snr_db"])
                mapped, clamped = BR.inverse_interpolate(target, self.anchors)
                require(not clamped, f"profile target {target} would be clamped")
                mapped = BR.round_to_granularity(
                    mapped,
                    float(self.config["actuator"]["command_granularity_db"]),
                )
                send_ns: int | None = None
                ack_ns: int | None = None
                ack_ms: float | None = None
                if index == 0:
                    send_ns = int(prime["send_monotonic_ns"])
                    ack_ns = int(prime["ack_monotonic_ns"])
                    ack_ms = float(prime["ack_latency_ms"])
                    status = "PRIMED"
                elif mapped == last_command:
                    _wait_until(due)
                    status = "HOLD"
                else:
                    _wait_until(due)
                    if time.monotonic_ns() >= action_open:
                        status = "MISSED_NO_COMMAND"
                    else:
                        self.send_noise(
                            mapped, reason="PROFILE_REPLAY", log=command_log,
                            profile_id=plan.profile_id, step_index=index,
                            target_snr_db=target, clamped=False,
                        )
                        event = command_log[-1]
                        send_ns = int(event["send_monotonic_ns"])
                        ack_ns = int(event["ack_monotonic_ns"])
                        ack_ms = float(event["ack_latency_ms"])
                        status = "ACK_ON_TIME" if ack_ns < action_open else "ACK_LATE"
                        if status == "ACK_ON_TIME":
                            last_command = mapped
                precedes = (
                    status == "HOLD" or (ack_ns is not None and ack_ns < action_open)
                )
                writer.writerow({
                    "profile_id": plan.profile_id,
                    "trace_id": plan.trace_id,
                    "step_index": index,
                    "target_snr_db": target,
                    "mapped_noise_power_db": mapped,
                    "scheduled_action_open_monotonic_ns": action_open,
                    "command_due_monotonic_ns": due,
                    "command_send_monotonic_ns": send_ns,
                    "command_ack_monotonic_ns": ack_ns,
                    "command_ack_latency_ms": ack_ms,
                    "command_status": status,
                    "command_precedes_action_open": precedes,
                    "clamped": clamped,
                })
                handle.flush()

    def audit_traffic(self, session: Mapping[str, Any]) -> dict[str, Any]:
        sender = json.loads(Path(session["sender_summary"]).read_text())
        receiver = json.loads(Path(session["receiver_summary"]).read_text())
        expected_chunks = C.FRAMES_PER_PROFILE * C.PROBE_CHUNKS_PER_FRAME
        outcome = {
            "sender": sender, "receiver": receiver,
            "expected_frames": C.FRAMES_PER_PROFILE,
            "expected_chunks": expected_chunks,
            "sender_exact": (
                int(sender["frames"]) == C.FRAMES_PER_PROFILE
                and int(sender["chunks_sent"]) == expected_chunks
                and int(sender["chunks_dropped"]) == 0
                and int(sender["schedule_misses"]) == 0
            ),
        }
        require(outcome["sender_exact"], f"sender accounting failed: {outcome}")
        return outcome

    def run_profile(self, plan: C.ProfilePlan, profile: Any) -> dict[str, Any]:
        cell_tag = f"{plan.run_index:02d}__{plan.profile_id.lower()}"
        cell_dir = self.output_dir / "cells" / cell_tag
        cell_dir.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            **plan.to_json(), "cell_tag": cell_tag,
            "radio_profile_id": C.RADIO_PROFILE_ID,
            "started_utc": utc_now(), "status": "FAILED",
        }
        command_log: list[dict[str, Any]] = []
        session: dict[str, Any] | None = None
        try:
            record["preflight"] = self.preflight_before_launcher(cell_tag)
            record["radio_attach"] = self.start_ran_via_launcher(cell_tag, cell_dir)
            record["post_launcher"] = self.bind_post_launcher_context()
            record["radio_path"] = self.verify_radio_path(cell_dir)
            self.start_telemetry(cell_tag)
            self.open_telnet(cell_dir)
            record["noise_before_db"] = self.read_back_noise()
            record["udp_probe"] = self.udp_probe(cell_tag, cell_dir)
            time.sleep(C.PROFILE_WARMUP_S)

            start_ns = time.monotonic_ns() + C.SENDER_START_LEAD_NS
            session = self.launch_receiver_and_sender(
                plan=plan, cell_tag=cell_tag, cell_dir=cell_dir,
                start_monotonic_ns=start_ns,
            )
            self.replay_profile(
                plan=plan, samples=profile.samples,
                start_monotonic_ns=start_ns, command_log=command_log,
                schedule_path=cell_dir / "profile_schedule.csv",
            )
            session["sender"].process.wait(timeout=15.0)
            session["receiver"].process.wait(timeout=C.RECEIVER_TAIL_S + 10.0)
            require(session["sender"].process.returncode == 0, "sender exited nonzero")
            require(session["receiver"].process.returncode == 0, "receiver exited nonzero")
            record["traffic"] = self.audit_traffic(session)
            record["noise_after_profile_db"] = self.read_back_noise()
            record["rf_restore_verified"] = self.restore_clean(cell_dir, command_log)
            require(record["rf_restore_verified"], "RFsim -50 dB restore failed")
            record["status"] = "CAPTURED"
        except Exception as exc:  # evidence is preserved, never overwritten
            record["failure"] = f"{type(exc).__name__}: {exc}"
            if self.telnet is not None:
                try:
                    record["rf_restore_verified"] = self.restore_clean(
                        cell_dir, command_log
                    )
                except Exception as restore_exc:
                    record["restore_failure"] = (
                        f"{type(restore_exc).__name__}: {restore_exc}"
                    )
        finally:
            notes = self.teardown_ran()
            record["teardown_notes"] = notes
            with (cell_dir / "command_log.json").open("x", encoding="utf-8") as handle:
                json.dump(command_log, handle, indent=2)
                handle.write("\n")
            try:
                self.extract_ttracer(cell_tag, cell_dir)
                record["ttracer_extraction_ok"] = True
            except Exception as exc:
                record["ttracer_extraction_ok"] = False
                record["ttracer_extraction_failure"] = f"{type(exc).__name__}: {exc}"
            record["cold_restore"] = self.stop_core_and_assert_cold()
            if record["status"] == "CAPTURED" and (
                notes or not record["ttracer_extraction_ok"]
                or not record["cold_restore"]["host_cold"]
            ):
                record["status"] = "FAILED"
                record["failure"] = "teardown/extraction/cold-state gate failed"
            record["finished_utc"] = utc_now()
            with (cell_dir / "cell_record.json").open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
        return record

    def write_manifest(self, status: str, analysis: Mapping[str, Any] | None,
                       failure: str | None) -> None:
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({
                    "relative_path": str(path.relative_to(self.output_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": C.sha256_file(path),
                })
        value = {
            "schema": C.SCHEMA, "contract_id": C.CONTRACT_ID,
            "contract_version": C.CONTRACT_VERSION,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "status": status, "failure": failure, "utc": utc_now(),
            "radio_profile_id": C.RADIO_PROFILE_ID,
            "design": C.design_record(),
            "design_sha256": C.canonical_json_sha256(C.design_record()),
            "source_verification": C.verify_sources(ROOT),
            "radio_binding_final": RB.verify("final_sealing", ROOT),
            "analysis": analysis, "files": files,
        }
        with (self.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    def run_capture(self) -> int:
        profiles = {
            profile.profile_id: profile
            for profile in BC.resolve_profiles(ROOT, C.FRAMES_PER_PROFILE)
        }
        for plan in C.build_plan():
            resolved = profiles[plan.profile_id]
            require(resolved.trace_id == plan.trace_id, "profile trace identity drift")
            require(resolved.seed == plan.seed, "profile seed drift")
            require(resolved.trace_sha256 == plan.trace_sha256, "profile hash drift")
            self.profile_records.append(self.run_profile(plan, resolved))
            if self.profile_records[-1]["status"] != "CAPTURED":
                failure = self.profile_records[-1].get("failure", "profile failed")
                self.write_manifest("FAILED", None, str(failure))
                return 1

        try:
            result = A.analyze_run(self.output_dir)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            self.write_manifest("FAILED_ANALYSIS_EXCEPTION", None, failure)
            return 1
        status = C.SUCCESS_TERMINAL if result["passed"] else "FAILED_ANALYSIS_GATES"
        self.write_manifest(status, result, None if result["passed"] else status)
        terminal = {
            "terminal": status,
            "manifest_sha256": C.sha256_file(self.output_dir / "manifest.json"),
            "utc": utc_now(),
        }
        with (self.output_dir / f"{status}.json").open("x", encoding="utf-8") as handle:
            json.dump(terminal, handle, indent=2)
            handle.write("\n")
        return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verification = C.verify_sources(ROOT)
        if args.validate_only:
            print(json.dumps({
                "source_verification": verification,
                "design": C.design_record(),
                "design_sha256": C.canonical_json_sha256(C.design_record()),
            }, indent=2, sort_keys=True))
            return 0
        require(args.execute == C.EXECUTION_TOKEN, "explicit execution token required")
        require(args.output is not None, "--output is required")
        output = args.output.resolve()
        require(str(output).startswith(str(OUTPUT_ROOT.resolve()) + os.sep),
                f"output must be a new child of {OUTPUT_ROOT}")
        require(not output.exists(), f"create-only output exists: {output}")
        output.mkdir(parents=True, exist_ok=False)
        return Runner(output).run_capture()
    except (CaptureError, C.ContractError, BR.RunFailure) as exc:
        print(f"dynamic-MCS capture refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
