#!/usr/bin/env python3
"""UE-N3E: lowest sustainable achieved SNR for the degraded/fallback route.

This is a *runtime sustain* investigation, not a cold-attachment test.  Each
run brings the RAN up at the known-good clean condition (-50 dB commanded
RFsim noise), attaches the UE, proves the PDU tunnel and external-DN
reachability, and only then applies one weak-channel candidate.  Under that
candidate the UE sends one 2 KB application payload every 100 ms to the DN and
the DN returns a small ACK, for an exact 60-second / 600-message window.  The
good condition is restored before the next candidate.

The workload is deliberately the ~164 kbps fallback payload, not the 1 Mbps
workload used by UE-N3/UE-N3A.  Those 1 Mbps results remain valid as
higher-load evidence and are not superseded here.

Search: descend the measured 0.5 dB command ladder from the first candidate
until a run fails, then repeat the lowest passing candidate to three runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json"

RUN_FIELDS = [
    "run_index", "run_id", "condition_id", "repetition_index",
    "rfsim_command", "commanded_noise_power_db",
    "achieved_pusch_snr_db_p05", "achieved_pusch_snr_db_median",
    "achieved_pusch_snr_db_p95", "tail_pusch_samples", "tail_status",
    "messages_attempted", "messages_delivered_and_acked",
    "delivery_ack_percent",
    "ack_latency_ms_p50", "ack_latency_ms_p95", "ack_latency_ms_max",
    "deadline_misses_over_100ms", "longest_outage_s",
    "ue_or_pdu_disconnected", "disconnect_reason",
    "recovered_after_restore", "clean_restore_verified",
    "verdict", "fail_reasons", "output_dir",
]


def format_command_db(value: float) -> str:
    """Render a commanded noise value at the precision the ladder actually uses.

    ``:.1f`` would silently truncate a 0.25 dB refinement step to -2.2 and then
    fail the post-apply state comparison, so keep two decimals and trim only a
    redundant trailing zero.
    """
    text = f"{float(value):.2f}"
    return text[:-1] if text.endswith("0") else text


class FloorFailure(RuntimeError):
    """Fail-closed infrastructure or evidence failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FloorFailure(message)


def percentile(values: Sequence[float], q: float) -> float | None:
    return n2.percentile(list(values), q)


class FallbackRunRunner(n2.Runner):
    """One candidate condition inside one fresh RAN epoch."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        *,
        run_index: int,
        command_db: float,
        condition_id: str,
        repetition_index: int,
    ) -> None:
        super().__init__(config_path, output_dir)
        self.run_index = int(run_index)
        self.command_db = float(command_db)
        self.condition_id = condition_id
        self.repetition_index = int(repetition_index)
        self.live_mcs: n2.LiveCsv | None = None
        self.ext_dn_pid: int | None = None
        self.client: n2.ManagedProcess | None = None
        self.responder: n2.ManagedProcess | None = None
        self.traffic_start_ns: int | None = None
        self.disconnect_reason: str | None = None
        self.applied = False

    # ---------------------------------------------------------------- gates

    def assert_carla_absent(self) -> None:
        result = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=5,
        )
        require(result.returncode == 0, "CARLA process detector failed closed")
        markers = [str(v).lower() for v in self.config["preflight"]["carla_process_markers"]]
        matches = [
            line.strip() for line in result.stdout.splitlines()
            if len(line.strip().split(maxsplit=2)) >= 2
            and any(m in line.strip().split(maxsplit=2)[1].lower() for m in markers)
        ]
        require(not matches, f"CARLA_ACTIVE_FAIL_CLOSED: {matches}")

    def namespace_udp_busy(self) -> bool:
        require(self.ext_dn_pid is not None, "ext-DN PID unavailable")
        output = n2.run_checked([
            "sudo", "-n", "nsenter", "-t", str(self.ext_dn_pid), "-n", "ss", "-H", "-lun",
        ], timeout=10).stdout
        port = int(self.config["workload"]["remote_port"])
        return any(f":{port} " in line or line.rstrip().endswith(f":{port}") for line in output.splitlines())

    def preflight(self) -> None:
        self.assert_carla_absent()
        n2.run_checked(["sudo", "-n", "true"])
        for container in self.config["radio"]["core_containers"]:
            state = n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container,
            ]).stdout.strip()
            require(state.startswith("true") and "unhealthy" not in state,
                    f"core container not ready: {container}={state!r}")
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(
                ["sudo", "-n", "pgrep", "-a", "-x", name], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            require(found.returncode != 0 or not found.stdout.strip(),
                    f"cold-RAN gate failed: {found.stdout.strip()}")
        require(not n2.oai_tunnel_interfaces(), "cold-RAN gate found a stale UE tunnel")
        busy = [
            int(p) for p in self.config["preflight"]["required_host_tcp_ports"]
            if not n2.port_is_free(int(p))
        ]
        require(not busy, f"required host TCP ports busy: {busy}")
        self.ext_dn_pid = int(n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
            self.config["radio"]["ext_dn_container"],
        ]).stdout.strip())
        require(not self.namespace_udp_busy(), "ext-DN fallback UDP port already bound")

    # ------------------------------------------------------------ telemetry

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
            (calibration.parse_live_pusch(item) for item in self.live_csv.snapshot())
            if row is not None
        }

    # -------------------------------------------------------------- traffic

    def start_probe(self) -> None:
        require(self.ue_ip is not None and self.ext_dn_pid is not None,
                "probe requires the UE tunnel IPv4 and the ext-DN PID")
        workload, run = self.config["workload"], self.config["run"]
        directory = self.output_dir / "traffic"
        directory.mkdir(parents=True, exist_ok=True)
        ready = directory / "responder_ready.json"
        responder = [
            "sudo", "-n", "nsenter", "-t", str(self.ext_dn_pid), "-n", "/usr/bin/python3",
            str(self.path(self.config["paths"]["ack_responder"])),
            "--bind-host", self.config["radio"]["ext_dn_ip"],
            "--port", str(workload["remote_port"]),
            "--duration-s", str(run["responder_duration_s"]),
            "--ready-json", str(ready),
            "--summary-json", str(directory / "responder_summary.json"),
            "--events-csv", str(directory / "responder_events.csv"),
        ]
        self.responder = self.spawn("ack_responder", responder, "logs/responder.log", root_owned=True)
        deadline = time.monotonic() + float(workload["responder_ready_timeout_s"])
        while time.monotonic() < deadline:
            require(self.responder.process.poll() is None, "responder exited before READY")
            if ready.is_file():
                payload = json.loads(ready.read_text(encoding="utf-8"))
                require(payload.get("status") == "READY", "responder READY status mismatch")
                require(int(payload.get("port", -1)) == int(workload["remote_port"]),
                        "responder port mismatch")
                break
            time.sleep(0.05)
        else:
            raise FloorFailure("responder READY timeout")

        client = [
            str(self.path(self.config["paths"]["python"])),
            str(self.path(self.config["paths"]["ack_client"])),
            "--bind-host", self.ue_ip,
            "--remote-host", self.config["radio"]["ext_dn_ip"],
            "--remote-port", str(workload["remote_port"]),
            "--payload-bytes", str(workload["payload_bytes"]),
            "--interval-ms", str(workload["interval_ms"]),
            "--count", str(run["client_total_messages"]),
            "--log-csv", str(directory / "client_messages.csv"),
            "--summary-json", str(directory / "client_summary.json"),
        ]
        self.traffic_start_ns = time.monotonic_ns()
        self.client = self.spawn("ack_client", client, "logs/client.log")

    def establish_clean_lead(self) -> None:
        require(self.live_csv is not None and self.traffic_start_ns is not None,
                "clean-lead anchors unavailable")
        lead_end = self.traffic_start_ns + int(float(self.config["run"]["clean_lead_s"]) * 1e9)
        while time.monotonic_ns() < lead_end:
            self.check_alive()
            parsed = [
                row for row in
                (calibration.parse_live_pusch(item) for item in self.live_csv.snapshot())
                if row is not None and row["mono_ns"] >= self.traffic_start_ns
            ]
            rntis = {row["rnti"] for row in parsed}
            if len(parsed) >= 5 and len(rntis) == 1:
                self.current_rnti = next(iter(rntis))
            time.sleep(0.05)
        require(self.current_rnti is not None,
                "clean lead produced no single-RNTI PUSCH observation")

    def check_alive(self) -> None:
        for name in ("gnb", "ue"):
            process = next((p for p in self.processes if p.name == name), None)
            require(process is not None and process.process.poll() is None,
                    f"RAN process exited: {name}")
        require(self.live_csv is not None and self.live_csv.process.poll() is None,
                "live PUSCH collector exited")
        require(self.live_mcs is not None and self.live_mcs.process.poll() is None,
                "live MCS collector exited")
        require(self.responder is not None and self.responder.process.poll() is None,
                "ACK responder exited early")
        require(self.client is not None and self.client.process.poll() is None,
                "ACK client exited early")

    def watch(self, until_ns: int) -> None:
        """Hold until ``until_ns``, recording (not raising on) service loss."""
        while time.monotonic_ns() < until_ns:
            self.check_alive()
            if self.disconnect_reason is None:
                if self.tunnel_ip() != self.ue_ip:
                    self.disconnect_reason = "UE_TUNNEL_IDENTITY_LOST"
                elif self.current_rnti is not None and any(
                    value != self.current_rnti for value in self.observed_rntis()
                ):
                    self.disconnect_reason = "RNTI_CHANGED"
            time.sleep(0.1)

    def apply_candidate(self, model_index: int) -> dict[str, Any]:
        require(self.telnet is not None, "control session unavailable")
        require(not self.applied, "candidate may be applied only once per run")
        self.applied = True
        target = format_command_db(self.command_db)
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(
            f"channelmod modify {model_index} noise_power_dB {target}"
        )
        self.validate_modify_response(response, target)
        _, _, _, _, state = self.telnet.command("channelmod show current")
        model = n2.parse_channel_models(state).get(
            self.config["actuator"]["channel_model_name"], {}
        )
        require(abs(float(model.get("noise_power_db", math.nan)) - self.command_db) <= 1e-6,
                f"post-command noise state mismatch: {model}")
        n2.atomic_text(self.output_dir / "channel_state_candidate_applied.txt", state)
        row = {
            "rfsim_command": f"channelmod modify {model_index} noise_power_dB {target}",
            "commanded_noise_power_db": self.command_db,
            "send_monotonic_ns": sent_mono, "send_wall_time_ns": sent_wall,
            "response_received_monotonic_ns": ack_mono,
            "response_received_wall_time_ns": ack_wall,
            "handler_bracket_ms": (ack_mono - sent_mono) / 1e6,
            "status": "ACK_AND_POST_STATE_VALIDATED_ONCE",
            "control_session_id": self.control_session_id,
            "ran_epoch_id": self.ran_epoch_id,
        }
        self.command_rows.append(row)
        return row

    # ------------------------------------------------------------- analysis

    def read_client_rows(self) -> list[dict[str, Any]]:
        path = self.output_dir / "traffic/client_messages.csv"
        require(path.is_file(), "client message log is missing")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def score_window(
        self, rows: Sequence[Mapping[str, Any]], start_ns: int, end_ns: int
    ) -> dict[str, Any]:
        criteria = self.config["pass_criteria"]
        window = [
            row for row in rows
            if start_ns <= int(row["send_monotonic_ns"]) < end_ns
        ]
        attempted = len(window)
        acked = [row for row in window if row["acked"] == "1"]
        foreign = [row for row in acked if row["ack_source_expected"] != "1"]
        latencies = sorted(float(row["ack_latency_ms"]) for row in acked)
        delivered = len(acked)
        ratio = delivered / attempted if attempted else 0.0
        misses = sum(
            1 for value in latencies
            if value > float(criteria["maximum_ack_latency_p95_ms"])
        )
        # An outage is the longest stretch of the measured window containing no
        # acknowledged delivery, bounded by the window edges.
        marks = [start_ns] + [int(row["ack_monotonic_ns"]) for row in acked] + [end_ns]
        marks.sort()
        longest_outage_s = max(
            (marks[i + 1] - marks[i]) / 1e9 for i in range(len(marks) - 1)
        ) if len(marks) > 1 else float(end_ns - start_ns) / 1e9
        p95 = percentile(latencies, 0.95)
        fail_reasons: list[str] = []
        if attempted != int(self.config["run"]["expected_tail_messages"]):
            fail_reasons.append(f"ATTEMPTED_{attempted}_NOT_600")
        if delivered < int(criteria["minimum_delivered_and_acked"]):
            fail_reasons.append(f"DELIVERY_{delivered}_BELOW_{criteria['minimum_delivered_and_acked']}")
        if p95 is None or p95 > float(criteria["maximum_ack_latency_p95_ms"]):
            fail_reasons.append(f"ACK_P95_{p95}")
        if longest_outage_s >= float(criteria["maximum_outage_s"]):
            fail_reasons.append(f"OUTAGE_{longest_outage_s:.3f}S")
        if self.disconnect_reason is not None:
            fail_reasons.append(f"DISCONNECT_{self.disconnect_reason}")
        if foreign:
            fail_reasons.append(f"FOREIGN_ACK_SOURCE_{len(foreign)}")
        return {
            "messages_attempted": attempted,
            "messages_delivered_and_acked": delivered,
            "delivery_ack_percent": round(100.0 * ratio, 4),
            "ack_latency_ms_p50": percentile(latencies, 0.50),
            "ack_latency_ms_p95": p95,
            "ack_latency_ms_max": latencies[-1] if latencies else None,
            "deadline_misses_over_100ms": misses,
            "longest_outage_s": round(longest_outage_s, 4),
            "verdict": "PASS" if not fail_reasons else "FAIL",
            "fail_reasons": ";".join(fail_reasons),
        }

    def score_recovery(self, rows: Sequence[Mapping[str, Any]], start_ns: int) -> bool:
        minimum = int(self.config["run"]["minimum_recovery_ack_messages"])
        acked = [
            row for row in rows
            if int(row["send_monotonic_ns"]) >= start_ns and row["acked"] == "1"
        ]
        return len(acked) >= minimum and self.tunnel_ip() == self.ue_ip

    # ------------------------------------------------------------------ run

    def execute(self) -> dict[str, Any]:
        n2.atomic_json(self.output_dir / "resolved_config.json", self.config)
        run = self.config["run"]
        summary: dict[str, Any] = {
            "run_index": self.run_index,
            "run_id": self.output_dir.name,
            "condition_id": self.condition_id,
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": self.command_db,
            "output_dir": self.output_dir.name,
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
        }
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            self.wait_attach()
            n2.wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            self.start_telemetry()
            time.sleep(2.0)
            model_index = self.open_and_validate_telnet()
            self.start_probe()
            self.establish_clean_lead()

            command_row = self.apply_candidate(model_index)
            summary["rfsim_command"] = command_row["rfsim_command"]
            settle_end = command_row["response_received_monotonic_ns"] + int(
                float(run["settle_s"]) * 1e9
            )
            self.watch(settle_end)
            tail_start = settle_end
            tail_end = tail_start + int(float(run["measured_tail_s"]) * 1e9)
            self.watch(tail_end)

            tail = calibration.summarize_tail(
                self.live_csv.snapshot() if self.live_csv else [],
                self.live_mcs.snapshot() if self.live_mcs else [],
                start_ns=tail_start, end_ns=tail_end,
                expected_rnti=int(self.current_rnti),
                minimum_pusch=int(self.config["analysis"]["minimum_tail_pusch_samples"]),
                minimum_mcs=int(self.config["analysis"]["minimum_tail_mcs_samples"]),
                required_mcs_table=int(self.config["analysis"]["scheduler_required_mcs_table"]),
                required_force_mcs=int(self.config["analysis"]["scheduler_required_force_ul_mcs"]),
            ) if self.live_csv else {}
            n2.atomic_json(self.output_dir / "radio_tail.json", tail)

            self.restore(model_index)
            recovery_start = time.monotonic_ns()
            self.watch(recovery_start + int(float(run["clean_recovery_s"]) * 1e9))

            # Let the bounded client finish so its CSV is complete.
            if self.client is not None and self.client.process.poll() is None:
                try:
                    self.client.process.wait(timeout=40.0)
                except subprocess.TimeoutExpired:
                    self.client.stop()
            self.write_command_log()
            rows = self.read_client_rows()
            scored = self.score_window(rows, tail_start, tail_end)
            summary.update(scored)
            summary.update({
                "achieved_pusch_snr_db_p05": tail.get("achieved_pusch_snr_db_p05"),
                "achieved_pusch_snr_db_median": tail.get("achieved_pusch_snr_db_median"),
                "achieved_pusch_snr_db_p95": tail.get("achieved_pusch_snr_db_p95"),
                "tail_pusch_samples": tail.get("pusch_samples"),
                "tail_status": tail.get("status"),
                "ue_or_pdu_disconnected": self.disconnect_reason is not None,
                "disconnect_reason": self.disconnect_reason or "",
                "recovered_after_restore": self.score_recovery(rows, recovery_start),
                "clean_restore_verified": self.restored,
                "status": "RUN_COMPLETE",
            })
            n2.atomic_json(self.output_dir / "run_summary.json", summary)
            self.cleanup(strict=False)
            return summary
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            try:
                self.write_command_log()
            except Exception:
                pass
            self.cleanup(strict=False)
            summary.update({
                "status": "RUN_FAILED",
                "verdict": "ERROR",
                "fail_reasons": f"{type(exc).__name__}: {exc}",
                "clean_restore_verified": self.restored,
                "ue_or_pdu_disconnected": self.disconnect_reason is not None,
                "disconnect_reason": self.disconnect_reason or "",
            })
            n2.atomic_json(self.output_dir / "run_summary.json", summary)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return summary


def write_runs_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RUN_FIELDS})


def condition_id(command_db: float) -> str:
    return "CMD_" + format_command_db(command_db).replace("-", "MINUS").replace(".", "P")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-db", type=float, default=None)
    parser.add_argument("--stop-db", type=float, default=None)
    parser.add_argument("--step-db", type=float, default=None)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    search = config["search"]
    start_db = args.start_db if args.start_db is not None else float(search["start_commanded_noise_power_db"])
    stop_db = args.stop_db if args.stop_db is not None else float(search["stop_commanded_noise_power_db"])
    step_db = args.step_db if args.step_db is not None else float(search["step_db"])
    repetitions = int(search["repetitions_for_lowest_pass"])

    campaign_dir = Path(args.output_dir).resolve()
    require(not campaign_dir.exists(), f"create-only output already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)
    runs_csv = campaign_dir / config["output"]["runs_csv"]

    interrupted = {"value": False}

    def terminate(signum: int, _frame: Any) -> None:
        interrupted["value"] = True
        raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")

    for caught in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(caught, terminate)

    rows: list[dict[str, Any]] = []
    ladder: list[float] = []
    value = start_db
    while value <= stop_db + 1e-9:
        ladder.append(round(value, 2))
        value += step_db

    lowest_pass: float | None = None
    status = "NO_CANDIDATE_PASSED"
    run_index = 0
    settle_s = float(config["run"].get("inter_run_settle_s", 5.0))

    def settle() -> None:
        """Let the previous RAN epoch release its tunnel and ports."""
        if run_index > 0:
            time.sleep(settle_s)
    try:
        for command_db in ladder:
            settle()
            run_index += 1
            print(f"[n3e] descent run {run_index}: commanded {format_command_db(command_db)} dB", flush=True)
            runner = FallbackRunRunner(
                config_path, campaign_dir / f"run_{run_index:02d}_{condition_id(command_db)}_rep01",
                run_index=run_index, command_db=command_db,
                condition_id=condition_id(command_db), repetition_index=1,
            )
            row = runner.execute()
            rows.append(row)
            write_runs_csv(runs_csv, rows)
            print(f"[n3e]   verdict={row.get('verdict')} "
                  f"snr_med={row.get('achieved_pusch_snr_db_median')} "
                  f"delivered={row.get('messages_delivered_and_acked')} "
                  f"reasons={row.get('fail_reasons')}", flush=True)
            if row.get("verdict") == "ERROR":
                status = "INFRASTRUCTURE_ERROR_STOP"
                break
            if row.get("verdict") == "PASS":
                lowest_pass = command_db
                continue
            status = "BOUNDARY_BRACKETED"
            break
        else:
            status = "LADDER_EXHAUSTED_WITHOUT_FAILURE"

        if lowest_pass is not None and status != "INFRASTRUCTURE_ERROR_STOP":
            for repetition in range(2, repetitions + 1):
                settle()
                run_index += 1
                print(f"[n3e] replication run {run_index}: commanded {format_command_db(lowest_pass)} dB "
                      f"rep {repetition}", flush=True)
                runner = FallbackRunRunner(
                    config_path,
                    campaign_dir / f"run_{run_index:02d}_{condition_id(lowest_pass)}_rep{repetition:02d}",
                    run_index=run_index, command_db=lowest_pass,
                    condition_id=condition_id(lowest_pass), repetition_index=repetition,
                )
                row = runner.execute()
                rows.append(row)
                write_runs_csv(runs_csv, rows)
                print(f"[n3e]   verdict={row.get('verdict')} "
                      f"snr_med={row.get('achieved_pusch_snr_db_median')} "
                      f"delivered={row.get('messages_delivered_and_acked')} "
                      f"reasons={row.get('fail_reasons')}", flush=True)
                if row.get("verdict") != "PASS":
                    status = "LOWEST_CANDIDATE_DID_NOT_REPLICATE"
                    break
    except KeyboardInterrupt:
        status = "INTERRUPTED"

    write_runs_csv(runs_csv, rows)
    confirmed = [
        row for row in rows
        if lowest_pass is not None
        and abs(float(row["commanded_noise_power_db"]) - lowest_pass) < 1e-9
        and row.get("verdict") == "PASS"
    ]
    replicated = lowest_pass is not None and len(confirmed) >= repetitions
    medians = [
        float(row["achieved_pusch_snr_db_median"]) for row in confirmed
        if row.get("achieved_pusch_snr_db_median") is not None
    ]
    campaign = {
        "schema": "scenesense.ue_n3e_fallback_snr_floor_campaign.v1",
        "status": status if not replicated else "PROVISIONAL_FALLBACK_FLOOR_REPLICATED",
        "claim_boundary": config["claim_boundary"],
        "workload": config["workload"],
        "pass_criteria": config["pass_criteria"],
        "tested_commands_db": [row["commanded_noise_power_db"] for row in rows],
        "lowest_passing_commanded_noise_power_db": lowest_pass if replicated else None,
        "lowest_passing_achieved_pusch_snr_db_median": (
            statistics.median(medians) if replicated and medians else None
        ),
        "replications_passed": len(confirmed),
        "replications_required": repetitions,
        "runs": rows,
        "completed_at": n2.utc_now(),
    }
    n2.atomic_json(campaign_dir / config["output"]["campaign_summary"], campaign)
    print(json.dumps({
        "output_dir": str(campaign_dir),
        "status": campaign["status"],
        "lowest_passing_commanded_noise_power_db": campaign["lowest_passing_commanded_noise_power_db"],
        "lowest_passing_achieved_pusch_snr_db_median": campaign["lowest_passing_achieved_pusch_snr_db_median"],
    }, sort_keys=True))
    return 0 if status not in {"INFRASTRUCTURE_ERROR_STOP", "INTERRUPTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
