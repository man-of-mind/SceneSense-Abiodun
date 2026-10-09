#!/usr/bin/env python3
"""Run the bounded UE-N2 single-UE OAI actuator calibration smoke.

This runner deliberately emits PARTIAL_EVIDENCE.  The stock OAI T-tracer raw
file is retained, but its stock CSV exporter exposes only local time-of-day at
microsecond precision and no collector-ingest clock.  Consequently the output
may describe plateau response and command timing, but never claims an RF
application timestamp or a causal first-effect lag.
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
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n2_oai_ul_calibration_smoke_v1.json"
SUCCESS_STATUS = "UE_N2_SMOKE_CAPTURED_PARTIAL_EVIDENCE"
PROMPT = b"softmodem_gnb> "


class SmokeFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[low]
    return clean[low] * (high - position) + clean[high] * (position - low)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def run_checked(argv: Sequence[str], *, cwd: Path = ROOT, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv), cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise SmokeFailure(f"command failed rc={result.returncode}: {argv!r}\n{result.stdout[-2000:]}")
    return result


def port_is_free(port: int, *, kind: int = socket.SOCK_STREAM) -> bool:
    sock = socket.socket(socket.AF_INET, kind)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_tcp(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise SmokeFailure(f"TCP port {port} did not become ready: {last_error}")


def oai_tunnel_interfaces() -> list[str]:
    result = subprocess.run(
        ["ip", "-o", "link", "show"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return sorted(set(re.findall(r"\b(oaitun_ue\d+)\b", result.stdout)))


def parse_channel_models(payload: str) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        header = re.match(r"^model\s+(\d+)\s+(\S+)\s+type\s+(\S+):$", line)
        if header:
            current = {
                "model_index": int(header.group(1)),
                "model_name": header.group(2),
                "model_type": header.group(3),
            }
            if str(current["model_name"]) in models:
                raise SmokeFailure(
                    f"duplicate active channel model name: {current['model_name']}"
                )
            models[str(current["model_name"])] = current
            continue
        if current is None:
            continue
        values = re.search(r"path loss:\s*([-+0-9.eE]+)\s+noise:\s*([-+0-9.eE]+)", line)
        if values:
            current["path_loss_db"] = float(values.group(1))
            current["noise_power_db"] = float(values.group(2))
        owner = re.search(r"model owner:\s*(\S+)", line)
        if owner:
            current["owner"] = owner.group(1)
    return models


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[Any]
    log_handle: Any
    root_owned: bool = False

    @staticmethod
    def _group_members(pgid: int) -> list[int]:
        result = subprocess.run(
            ["ps", "-eo", "pid=,pgid=,stat="], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        members = []
        for line in result.stdout.splitlines():
            try:
                pid_text, pgid_text, status = line.split()
                pid, observed_pgid = int(pid_text), int(pgid_text)
            except ValueError:
                continue
            if observed_pgid == pgid and pid != os.getpid() and "Z" not in status:
                members.append(pid)
        return members

    def _signal_group_members(self, pgid: int, sig: signal.Signals) -> None:
        members = self._group_members(pgid)
        if not members:
            return
        if self.root_owned:
            subprocess.run(
                ["sudo", "-n", "kill", f"-{sig.name.removeprefix('SIG')}", "--", *map(str, members)],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            for pid in members:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    def stop(self) -> None:
        try:
            pgid = os.getpgid(self.process.pid)
        except ProcessLookupError:
            pgid = self.process.pid
        try:
            for sig, timeout_s in (
                (signal.SIGINT, 4.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0),
            ):
                members = self._group_members(pgid)
                if not members:
                    break
                self._signal_group_members(pgid, sig)
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline and self._group_members(pgid):
                    time.sleep(0.1)
            remaining = self._group_members(pgid)
            if remaining:
                raise SmokeFailure(f"{self.name} process group {pgid} survived cleanup: {remaining}")
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            self.log_handle.close()


class TelnetSession:
    def __init__(self, host: str, port: int, timeout_s: float, max_bytes: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.sock.sendall(b"\n")
        self._recv_prompt()

    def close(self) -> None:
        self.sock.close()

    def _recv_prompt(self) -> str:
        self.sock.settimeout(self.timeout_s)
        chunks: list[bytes] = []
        total = 0
        while total < self.max_bytes:
            chunk = self.sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if PROMPT in b"".join(chunks[-2:]):
                break
        payload = b"".join(chunks)
        if PROMPT not in payload:
            raise SmokeFailure(f"Telnet response lacks prompt: {payload[-500:]!r}")
        return payload.decode("utf-8", errors="replace")

    def command(self, text: str) -> tuple[int, int, int, int, str]:
        send_mono = time.monotonic_ns()
        send_wall = time.time_ns()
        self.sock.sendall(text.encode("ascii") + b"\n")
        response = self._recv_prompt()
        response_mono = time.monotonic_ns()
        response_wall = time.time_ns()
        if "ERROR" in response.upper():
            raise SmokeFailure(f"Telnet rejected {text!r}: {response[-1000:]}")
        return send_mono, send_wall, response_mono, response_wall, response


class LiveCsv:
    def __init__(self, command: Sequence[str], path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            list(command), cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1, start_new_session=True,
        )
        self.rows: list[tuple[int, int, str]] = []
        self.lock = threading.Lock()
        self.handle = path.open("w", encoding="utf-8")
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            wall_ns = time.time_ns()
            mono_ns = time.monotonic_ns()
            self.handle.write(f"{wall_ns},{mono_ns},{line.rstrip()}\n")
            self.handle.flush()
            if line and line[0].isdigit():
                with self.lock:
                    self.rows.append((wall_ns, mono_ns, line.rstrip()))

    def count(self) -> int:
        with self.lock:
            return len(self.rows)

    def snapshot(self) -> list[tuple[int, int, str]]:
        with self.lock:
            return list(self.rows)

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.kill(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.kill(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        self.handle.close()


class Runner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise SmokeFailure(f"create-only output already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        self.processes: list[ManagedProcess] = []
        self.live_csv: LiveCsv | None = None
        self.telnet: TelnetSession | None = None
        self.command_rows: list[dict[str, Any]] = []
        self.control_session_id = str(uuid.uuid4())
        self.ran_epoch_id = str(uuid.uuid4())
        self.restored = False
        self.current_rnti: int | None = None
        self.ue_ip: str | None = None
        self.started_at = utc_now()

    def path(self, relative: str) -> Path:
        return (ROOT / relative).resolve()

    def spawn(self, name: str, argv: Sequence[str], log_name: str, *, cwd: Path = ROOT, root_owned: bool = False) -> ManagedProcess:
        log_path = self.output_dir / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            list(argv), cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        managed = ManagedProcess(name, process, log, root_owned=root_owned)
        self.processes.append(managed)
        return managed

    def preflight(self) -> None:
        predecessor = self.config["predecessor"]
        bundle = self.path(predecessor["bundle_dir"])
        run_checked([
            str(self.path(self.config["paths"]["python"])), "-m",
            "rl_agent.ue_n1_freeze_oai_ul_actuator_v2", "--validate", str(bundle),
        ], timeout=30)
        run_checked(["sudo", "-n", "true"])
        for container in self.config["radio"]["core_containers"]:
            state = run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}", container,
            ]).stdout.strip()
            if not state.startswith("true") or "unhealthy" in state:
                raise SmokeFailure(f"core container is not ready: {container}={state!r}")
        for process_name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(
                ["sudo", "-n", "pgrep", "-a", "-x", process_name],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if found.returncode == 0 and found.stdout.strip():
                raise SmokeFailure(f"cold-RAN gate failed: {found.stdout.strip()}")
        stale_tunnels = oai_tunnel_interfaces()
        if stale_tunnels:
            raise SmokeFailure(f"cold-RAN gate failed: OAI UE tunnel(s) already exist: {stale_tunnels}")
        ports = [
            4043, self.config["actuator"]["telnet_port"],
            self.config["telemetry"]["gnb_port"], self.config["telemetry"]["ue_port"],
            self.config["telemetry"]["gnb_relay_port"], self.config["telemetry"]["ue_relay_port"],
        ]
        busy = [port for port in ports if not port_is_free(int(port))]
        if busy:
            raise SmokeFailure(f"required TCP ports are not free: {busy}")

    def materialize_configs(self) -> tuple[Path, Path]:
        conf_root = self.path(self.config["paths"]["oai_ran_conf"])
        gnb_base = (conf_root / self.config["paths"]["gnb_base_config"]).read_text(encoding="utf-8")
        ue_base = (conf_root / self.config["paths"]["ue_base_config"]).read_text(encoding="utf-8")
        channel = (conf_root / self.config["paths"]["channel_config"]).read_text(encoding="utf-8")
        expected_imsi = str(self.config["radio"]["expected_imsi"])
        uicc_blocks = re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue_base)
        imsis = re.findall(r'(?m)^\s*imsi\s*=\s*"([0-9]+)"\s*;', ue_base)
        if int(self.config["radio"]["ue_count"]) != 1 or len(uicc_blocks) != 1:
            raise SmokeFailure(
                f"effective UE config is not single-UE: configured={self.config['radio']['ue_count']} "
                f"uicc_blocks={len(uicc_blocks)}"
            )
        if imsis != [expected_imsi]:
            raise SmokeFailure(f"effective UE IMSI mismatch: expected={expected_imsi} observed={imsis}")
        channel, replacements = re.subn(
            r"noise_power_dB\s*=\s*[-+0-9.eE]+;", "noise_power_dB = -50;", channel,
        )
        if replacements != 3:
            raise SmokeFailure(f"expected exactly three single-UE channel noise values, found {replacements}")
        if "noise_power_dBFS" in channel:
            raise SmokeFailure("global noise_power_dBFS must remain unset")
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        if marker not in ue_base:
            raise SmokeFailure("UE base config lacks expected channel include")
        runtime = self.output_dir / "runtime"
        runtime.mkdir()
        gnb_path = runtime / "effective_gnb_clean_minus50.conf"
        ue_path = runtime / "effective_ue_clean_minus50.conf"
        atomic_text(gnb_path, gnb_base + "\n\n" + channel + "\n")
        atomic_text(ue_path, ue_base.replace(marker, channel))
        atomic_json(runtime / "config_hashes.json", {
            "gnb_sha256": sha256(gnb_path), "ue_sha256": sha256(ue_path),
            "source_channel_sha256": sha256(conf_root / self.config["paths"]["channel_config"]),
        })
        return gnb_path, ue_path

    def start_ran(self, gnb_config: Path, ue_config: Path) -> None:
        radio = self.config["radio"]
        build = self.path(self.config["paths"]["oai_ran_build"])
        gnb = [
            "sudo", "-n", "env", "-u", "SCENESENSE_FORCE_UL_MCS",
            f"SCENESENSE_MCS_POLICY={radio['mcs_policy']}", "./nr-softmodem",
            "-O", str(gnb_config), "--gNBs.[0].min_rxtxtime", "6", "--rfsim",
            "--rfsimulator.[0].options", "chanmod", "--telnetsrv",
            "--telnetsrv.listenaddr", self.config["actuator"]["telnet_host"],
            "--telnetsrv.listenport", str(self.config["actuator"]["telnet_port"]),
            "--T_stdout", "2", "--T_nowait", "--T_port", str(self.config["telemetry"]["gnb_port"]),
        ]
        self.spawn("gnb", gnb, "logs/gnb.log", cwd=build, root_owned=True)
        time.sleep(float(radio["gnb_start_lead_s"]))
        ue = [
            "sudo", "-n", "./nr-uesoftmodem", "--rfsim",
            "--rfsimulator.[0].serveraddr", "127.0.0.1",
            "--rfsimulator.[0].options", "chanmod", "-r", str(radio["prb"]),
            "--numerology", str(radio["numerology"]), "--band", str(radio["band"]),
            "-C", str(radio["downlink_frequency_hz"]), "-O", str(ue_config),
            "--T_stdout", "2", "--T_nowait", "--T_port", str(self.config["telemetry"]["ue_port"]),
        ]
        self.spawn("ue", ue, "logs/ue.log", cwd=build, root_owned=True)

    def wait_attach(self) -> None:
        radio = self.config["radio"]
        deadline = time.monotonic() + float(radio["attach_timeout_s"])
        interface = radio["ue_interface"]
        while time.monotonic() < deadline:
            if any(proc.process.poll() is not None for proc in self.processes[:2]):
                raise SmokeFailure("gNB or UE exited before attachment")
            address = subprocess.run(
                ["ip", "-j", "-4", "addr", "show", "dev", interface], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            observed_ips: list[str] = []
            if address.returncode == 0:
                try:
                    rows = json.loads(address.stdout)
                    observed_ips = [
                        str(info["local"])
                        for row in rows
                        for info in row.get("addr_info", [])
                        if info.get("family") == "inet" and info.get("local")
                    ]
                except (json.JSONDecodeError, KeyError, TypeError):
                    observed_ips = []
            if len(observed_ips) == 1:
                observed_ip = observed_ips[0]
                ping = subprocess.run(
                    ["ping", "-I", interface, "-c", "3", "-W", "2", radio["ext_dn_ip"]],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                if ping.returncode == 0:
                    self.ue_ip = observed_ip
                    atomic_text(self.output_dir / "logs/attach_ping.log", ping.stdout)
                    atomic_json(self.output_dir / "ue_network_identity.json", {
                        "ue_count": 1,
                        "imsi": radio["expected_imsi"],
                        "interface": interface,
                        "discovered_ipv4": observed_ip,
                        "ext_dn_ip": radio["ext_dn_ip"],
                        "ping_pass": True,
                    })
                    return
            time.sleep(1)
        raise SmokeFailure("single UE did not attach and reach ext-DN before timeout")

    def start_telemetry(self) -> None:
        telemetry = self.config["telemetry"]
        troot = self.path("OAI/openairinterface5g/common/utils/T/tracer")
        messages = self.path(self.config["paths"]["t_messages"])
        for source, port, relay in (
            ("gnb", telemetry["gnb_port"], telemetry["gnb_relay_port"]),
            ("ue", telemetry["ue_port"], telemetry["ue_relay_port"]),
        ):
            self.spawn(
                f"{source}_relay",
                [str(troot / "multi"), "-d", str(messages), "-ip", "127.0.0.1", "-p", str(port), "-lp", str(relay)],
                f"logs/{source}_relay.log",
            )
        wait_tcp(int(telemetry["gnb_relay_port"]), 10)
        wait_tcp(int(telemetry["ue_relay_port"]), 10)
        for source, relay in (("gnb", telemetry["gnb_relay_port"]), ("ue", telemetry["ue_relay_port"])):
            raw = self.output_dir / "ttracer" / source / f"{source}.raw"
            raw.parent.mkdir(parents=True, exist_ok=True)
            events = telemetry["events"][source]
            argv = [str(troot / "record"), "-d", str(messages), "-o", str(raw), "-OFF"]
            for event in events:
                argv += ["-on", event]
            argv += ["-ip", "127.0.0.1", "-p", str(relay)]
            self.spawn(f"{source}_record", argv, f"logs/{source}_record.log")
        live_fields = (
            "time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
            "txpower_calc", "rbSize", "mcs", "rssi",
        )
        command = [
            str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1", "-p",
            str(telemetry["gnb_relay_port"]), "-f", "-s", ",", "-t", "time",
            "GNB_MAC_PUSCH_POWER_CONTROL", *live_fields,
        ]
        self.live_csv = LiveCsv(command, self.output_dir / "ttracer/gnb/live_pusch_with_ingest.csv")

    def start_traffic(self) -> None:
        if self.ue_ip is None:
            raise SmokeFailure("cannot start traffic before discovering the UE tunnel IPv4")
        traffic = self.config["traffic"]
        capture = self.config["capture"]
        pid = int(run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", "oai-ext-dn",
        ]).stdout.strip())
        sink_out = self.output_dir / "traffic/sink_packets.csv"
        sink_out.parent.mkdir(parents=True, exist_ok=True)
        sink = [
            "sudo", "-n", "nsenter", "-t", str(pid), "-n", "/usr/bin/python3",
            str(self.path(traffic["sink_path"])), "--bind", self.config["radio"]["ext_dn_ip"],
            "--port", str(traffic["remote_port"]), "--out", str(sink_out),
            "--timeout-s", str(capture["sink_timeout_s"]),
        ]
        self.spawn("sink", sink, "logs/sink.log", root_owned=True)
        time.sleep(0.5)
        sender = [
            str(self.path(self.config["paths"]["python"])), str(self.path(traffic["sender_path"])),
            "--bind-host", self.ue_ip, "--remote-host", self.config["radio"]["ext_dn_ip"],
            "--remote-port", str(traffic["remote_port"]), "--fps", str(traffic["fps"]),
            "--frames", str(capture["sender_frames"]), "--frame-bytes", str(traffic["frame_bytes"]),
            "--chunk-bytes", str(traffic["chunk_bytes"]), "--idle-before-s", str(capture["sender_idle_before_s"]),
            "--cooldown-s", str(capture["sender_cooldown_s"]),
            "--log-csv", str(self.output_dir / "traffic/sender.csv"),
        ]
        self.spawn("sender", sender, "logs/sender.log")

    def wait_clean_pusch(self, baseline_count: int) -> None:
        assert self.live_csv is not None
        minimum = int(self.config["capture"]["minimum_clean_pusch_samples"])
        deadline = time.monotonic() + float(self.config["traffic"]["fresh_pusch_timeout_s"])
        sink_log = self.output_dir / "logs/sink.log"
        while time.monotonic() < deadline:
            fresh = self.live_csv.snapshot()[baseline_count:]
            rntis: set[int] = set()
            for _wall_ns, _mono_ns, line in fresh:
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    rntis.add(int(parts[1]))
                except ValueError:
                    continue
            sender = next((proc for proc in self.processes if proc.name == "sender"), None)
            sink = next((proc for proc in self.processes if proc.name == "sink"), None)
            sink_has_traffic = (
                sink_log.exists()
                and "rate=" in sink_log.read_text(encoding="utf-8", errors="replace")
            )
            if (
                len(fresh) >= minimum
                and len(rntis) == 1
                and sender is not None
                and sender.process.poll() is None
                and sink is not None
                and sink.process.poll() is None
                and sink_has_traffic
            ):
                self.current_rnti = next(iter(rntis))
                atomic_json(self.output_dir / "fresh_traffic_gate.json", {
                    "baseline_pusch_rows": baseline_count,
                    "fresh_pusch_rows": len(fresh),
                    "current_rnti": self.current_rnti,
                    "sender_alive": True,
                    "sink_alive": True,
                    "sink_receive_rate_observed": True,
                })
                return
            time.sleep(0.1)
        raise SmokeFailure(
            f"fresh post-traffic PUSCH/sink gate failed: "
            f"fresh={self.live_csv.count() - baseline_count} minimum={minimum}"
        )

    def open_and_validate_telnet(self) -> int:
        actuator = self.config["actuator"]
        self.telnet = TelnetSession(
            actuator["telnet_host"], int(actuator["telnet_port"]),
            float(actuator["response_timeout_s"]), int(actuator["max_response_bytes"]),
        )
        _, _, _, _, response = self.telnet.command("channelmod show current")
        atomic_text(self.output_dir / "channel_state_before.txt", response)
        models = parse_channel_models(response)
        name = actuator["channel_model_name"]
        if name not in models:
            raise SmokeFailure(f"active channel object missing: {name}; observed={sorted(models)}")
        row = models[name]
        if row.get("model_type") != actuator["channel_model_type"]:
            raise SmokeFailure(f"unexpected channel type: {row}")
        if row.get("owner") != actuator["channel_model_owner"]:
            raise SmokeFailure(f"unexpected channel owner: {row}")
        if abs(float(row.get("path_loss_db", math.nan)) - float(actuator["path_loss_db"])) > 1e-6:
            raise SmokeFailure(f"unexpected path loss: {row}")
        if abs(float(row.get("noise_power_db", math.nan)) + 50.0) > 1e-6:
            raise SmokeFailure(f"channel did not start clean at -50: {row}")
        return int(row["model_index"])

    @staticmethod
    def validate_modify_response(response: str, target: str) -> None:
        owner = re.search(r"model owner:\s*(\S+)", response)
        values = re.search(r"path loss:\s*([-+0-9.eE]+)\s+noise:\s*([-+0-9.eE]+)", response)
        if not owner or owner.group(1) != "rfsimulator" or not values:
            raise SmokeFailure(f"modify response lacks owner/noise evidence: {response[-1200:]}")
        if abs(float(values.group(1))) > 1e-6 or abs(float(values.group(2)) - float(target)) > 1e-6:
            raise SmokeFailure(f"modify response mismatch target={target}: {values.groups()}")

    def run_trace(self, model_index: int) -> None:
        assert self.telnet is not None
        schedule = self.config["schedule"]
        period = int(schedule["period_ns"])
        values = [str(value) for value in schedule["commanded_noise_plateaus_db"]]
        repeats = int(schedule["commands_per_plateau"])
        origin_mono = time.monotonic_ns()
        origin_wall = time.time_ns()
        anchor = origin_mono + 500_000_000
        trace_index = 0
        for plateau_index, target in enumerate(values):
            for repeat_index in range(repeats):
                scheduled = anchor + trace_index * period
                scheduled_wall = origin_wall + (scheduled - origin_mono)
                while True:
                    remaining = scheduled - time.monotonic_ns()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining / 1e9, 0.01))
                before_send = time.monotonic_ns()
                if before_send >= scheduled + period:
                    self.command_rows.append({
                        "trace_index": trace_index, "plateau_index": plateau_index,
                        "value_transition_index": plateau_index, "repeat_index": repeat_index,
                        "is_value_transition": repeat_index == 0,
                        "commanded_noise_power_db": target, "scheduled_monotonic_ns": scheduled,
                        "scheduled_wall_time_ns": scheduled_wall,
                        "status": "SKIPPED_OBSOLETE", "control_session_id": self.control_session_id,
                        "ran_epoch_id": self.ran_epoch_id,
                    })
                    trace_index += 1
                    continue
                command = f"channelmod modify {model_index} noise_power_dB {target}"
                send_mono, send_wall, response_mono, response_wall, response = self.telnet.command(command)
                self.validate_modify_response(response, target)
                self.command_rows.append({
                    "trace_index": trace_index, "plateau_index": plateau_index,
                    "value_transition_index": plateau_index, "repeat_index": repeat_index,
                    "is_value_transition": repeat_index == 0,
                    "commanded_noise_power_db": target, "scheduled_monotonic_ns": scheduled,
                    "scheduled_wall_time_ns": scheduled_wall,
                    "send_monotonic_ns": send_mono, "send_wall_time_ns": send_wall,
                    "response_received_monotonic_ns": response_mono,
                    "response_received_wall_time_ns": response_wall,
                    "handler_bracket_ms": (response_mono - send_mono) / 1e6,
                    "response_before_next_boundary": response_mono < scheduled + period,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "status": "ACK_VALIDATED", "control_session_id": self.control_session_id,
                    "ran_epoch_id": self.ran_epoch_id,
                })
                trace_index += 1
        sent = [row for row in self.command_rows if row.get("status") == "ACK_VALIDATED"]
        planned = len(values) * repeats
        if len(self.command_rows) != planned or len(sent) != planned:
            raise SmokeFailure(
                f"100-ms trace incomplete: planned={planned} rows={len(self.command_rows)} "
                f"sent={len(sent)}"
            )
        late = [row for row in sent if not bool(row.get("response_before_next_boundary"))]
        if late:
            raise SmokeFailure(
                f"{len(late)} command response(s) crossed the next 100-ms boundary"
            )
        time.sleep(float(self.config["telemetry"]["collector_tail_s"]))

    def restore(self, model_index: int) -> None:
        assert self.telnet is not None
        clean = self.config["actuator"]["clean_and_restore_commanded_noise_power_db"]
        _, _, _, _, response = self.telnet.command(
            f"channelmod modify {model_index} noise_power_dB {clean}"
        )
        self.validate_modify_response(response, clean)
        _, _, _, _, state = self.telnet.command("channelmod show current")
        models = parse_channel_models(state)
        row = models.get(self.config["actuator"]["channel_model_name"], {})
        if abs(float(row.get("noise_power_db", math.nan)) - float(clean)) > 1e-6:
            raise SmokeFailure(f"clean restore verification failed: {row}")
        atomic_text(self.output_dir / "channel_state_restored.txt", state)
        self.restored = True

    def write_command_log(self) -> None:
        path = self.output_dir / self.config["output"]["command_log"]
        fields: list[str] = []
        for row in self.command_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["status"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.command_rows)

    def extract_ttracer(self) -> None:
        script = self.path("scripts/ttracer_extract_csv_smoke.sh")
        for source, profile in (("gnb", "latency"), ("ue", "queue")):
            raw = self.output_dir / "ttracer" / source / f"{source}.raw"
            if not raw.exists() or raw.stat().st_size == 0:
                raise SmokeFailure(f"missing raw T file: {raw}")
            run_checked([
                str(script), "--raw", str(raw), "--source", source,
                "--output-root", str(self.output_dir),
                "--profile", profile, "--clean-output",
            ], timeout=300)

    @staticmethod
    def approximate_event_wall_ns(time_text: str, reference_ns: int) -> int:
        reference = datetime.fromtimestamp(reference_ns / 1e9).astimezone()
        parsed = datetime.strptime(time_text, "%H:%M:%S.%f").time()
        candidate = datetime.combine(reference.date(), parsed, tzinfo=reference.tzinfo)
        options = [candidate - timedelta(days=1), candidate, candidate + timedelta(days=1)]
        best = min(options, key=lambda value: abs(value.timestamp() * 1e9 - reference_ns))
        return int(best.timestamp() * 1e9)

    def analyze(self) -> dict[str, Any]:
        if self.current_rnti is None:
            raise SmokeFailure("current-session RNTI was not established")
        sent = [row for row in self.command_rows if row.get("status") == "ACK_VALIDATED"]
        transitions = [row for row in sent if row.get("is_value_transition")]
        if len(transitions) != len(self.config["schedule"]["commanded_noise_plateaus_db"]):
            raise SmokeFailure(f"missing value transitions: {len(transitions)}")
        base = self.output_dir / "ttracer" / "gnb" / "csv"
        power_path = base / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
        mcs_path = base / "GNB_MAC_UL_MCS_DECISION.csv"
        if not power_path.exists() or not mcs_path.exists():
            raise SmokeFailure("required extracted PUSCH/MCS CSV is absent")
        with power_path.open(newline="", encoding="utf-8") as handle:
            power_rows = list(csv.DictReader(handle))
        with mcs_path.open(newline="", encoding="utf-8") as handle:
            mcs_rows = list(csv.DictReader(handle))
        power_rntis = {int(row["rnti"]) for row in power_rows if row.get("rnti")}
        mcs_rntis = {int(row["rnti"]) for row in mcs_rows if row.get("rnti")}
        expected_rntis = {self.current_rnti}
        if power_rntis != expected_rntis or mcs_rntis != expected_rntis:
            raise SmokeFailure(
                f"single-current-RNTI gate failed: expected={expected_rntis} "
                f"power={power_rntis} scheduler={mcs_rntis}"
            )
        power_rows = [row for row in power_rows if int(row["rnti"]) == self.current_rnti]
        mcs_rows = [row for row in mcs_rows if int(row["rnti"]) == self.current_rnti]
        invalid_scheduler_seals = [
            row for row in mcs_rows
            if int(row.get("mcs_table", -999)) != 0
            or int(row.get("force_ul_mcs", -999)) != -1
        ]
        if invalid_scheduler_seals:
            raise SmokeFailure(
                "scheduler seal failed: expected mcs_table=0 and force_ul_mcs=-1; "
                f"invalid_rows={len(invalid_scheduler_seals)}"
            )
        reference_ns = int(transitions[0]["send_wall_time_ns"])
        for row in power_rows:
            row["event_wall_ns_approx"] = self.approximate_event_wall_ns(row["time"], reference_ns)
        for row in mcs_rows:
            row["event_wall_ns_approx"] = self.approximate_event_wall_ns(row["time"], reference_ns)
        joined_path = self.output_dir / self.config["output"]["joined_observations"]
        plateau_rows: list[dict[str, Any]] = []
        joined_rows: list[dict[str, Any]] = []
        tail_ns = int(float(self.config["analysis"]["settled_tail_s"]) * 1e9)
        for index, transition in enumerate(transitions):
            # The response time is only an upper bracket for handler
            # completion, so start after that bracket. Stop before the next
            # transition is sent to avoid assigning prior/next-state samples.
            start = int(transition["response_received_wall_time_ns"])
            end = (
                int(transitions[index + 1]["send_wall_time_ns"])
                if index + 1 < len(transitions)
                else int(transition["send_wall_time_ns"])
                + int(self.config["schedule"]["plateau_duration_s"] * 1e9)
            )
            tail_start = max(start, end - tail_ns)
            p_rows = [row for row in power_rows if start <= int(row["event_wall_ns_approx"]) < end]
            p_tail = [row for row in p_rows if int(row["event_wall_ns_approx"]) >= tail_start]
            s_rows = [row for row in mcs_rows if start <= int(row["event_wall_ns_approx"]) < end]
            s_tail = [row for row in s_rows if int(row["event_wall_ns_approx"]) >= tail_start]
            snr = [float(row["snrx10"]) / 10.0 for row in p_tail]
            pusch_mcs = [float(row["mcs"]) for row in p_tail]
            ema = [float(row["avg_snr_x10"]) / 10.0 for row in s_tail]
            final_mcs = [float(row["final_mcs"]) for row in s_tail]
            if not snr:
                raise SmokeFailure(
                    f"plateau {index} command={transition['commanded_noise_power_db']} "
                    "has no finite tail PUSCH observation"
                )
            if not ema or not final_mcs or not all(
                math.isfinite(value) for value in (*ema, *final_mcs)
            ):
                raise SmokeFailure(
                    f"plateau {index} command={transition['commanded_noise_power_db']} "
                    "has no finite tail scheduler SNR/MCS observation"
                )
            plateau_rows.append({
                "plateau_index": index,
                "commanded_noise_power_db": transition["commanded_noise_power_db"],
                "pusch_observation_count": len(p_rows),
                "tail_pusch_count": len(snr),
                "instantaneous_pusch_snr_db_median": statistics.median(snr) if snr else None,
                "instantaneous_pusch_snr_db_p25": percentile(snr, 0.25),
                "instantaneous_pusch_snr_db_p75": percentile(snr, 0.75),
                "pusch_reported_mcs_median": statistics.median(pusch_mcs) if pusch_mcs else None,
                "scheduler_ema_snr_db_median": statistics.median(ema) if ema else None,
                "scheduler_final_mcs_median": statistics.median(final_mcs) if final_mcs else None,
                "ema_settling_status": "SETTLED_COUNT_GATE" if len(p_rows) >= int(self.config["analysis"]["minimum_accepted_pusch_observations_for_settled_ema"]) else "NOT_SETTLED_INSUFFICIENT_COUNT",
                "timestamp_join_status": "APPROXIMATE_LOCAL_TIME_OF_DAY_MICROSECOND_NO_INGEST_CLOCK",
                "assignment_window": "AFTER_TRANSITION_ACK_UPPER_BRACKET_TO_BEFORE_NEXT_TRANSITION_SEND",
                "current_rnti": self.current_rnti,
            })
            for row in p_rows:
                joined_rows.append({
                    "plateau_index": index,
                    "commanded_noise_power_db": transition["commanded_noise_power_db"],
                    "event_time_text": row["time"],
                    "event_wall_ns_approx": row["event_wall_ns_approx"],
                    "rnti": row["rnti"], "frame": row["frame"], "slot": row["slot"],
                    "instantaneous_pusch_snr_db": float(row["snrx10"]) / 10.0,
                    "mcs": row["mcs"],
                    "join_status": "DESCRIPTIVE_APPROXIMATE_NOT_CAUSAL",
                })
        fields = list(plateau_rows[0])
        with (self.output_dir / self.config["output"]["plateau_summary"]).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(plateau_rows)
        with joined_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(joined_rows[0]) if joined_rows else ["join_status"])
            writer.writeheader(); writer.writerows(joined_rows)
        latencies = [float(row["handler_bracket_ms"]) for row in sent]
        sender_path = self.output_dir / "traffic/sender.csv"
        sink_path = self.output_dir / "traffic/sink_packets.csv"
        if not sender_path.exists() or not sink_path.exists():
            raise SmokeFailure(
                f"traffic evidence missing: sender={sender_path.exists()} sink={sink_path.exists()}"
            )
        with sender_path.open(newline="", encoding="utf-8") as handle:
            sender_rows = list(csv.DictReader(handle))
        with sink_path.open(newline="", encoding="utf-8") as handle:
            sink_rows = list(csv.DictReader(handle))
        sender_frames = len({row.get("frame_index") for row in sender_rows})
        sink_packets = len(sink_rows)
        if sender_frames < 120 or sink_packets < int(sender_frames * 0.90):
            raise SmokeFailure(
                f"traffic evidence insufficient: sender_frames={sender_frames} "
                f"sink_packets={sink_packets}"
            )
        achieved = [row["instantaneous_pusch_snr_db_median"] for row in plateau_rows]
        monotone = all(
            achieved[i] is not None and achieved[i + 1] is not None and float(achieved[i]) > float(achieved[i + 1])
            for i in range(len(achieved) - 1)
        )
        return {
            "status": SUCCESS_STATUS,
            "evidence_class": "PARTIAL_EVIDENCE",
            "full_raw_event_envelope_satisfied": False,
            "causal_first_effect_status": "UNAVAILABLE_STOCK_TRACER_LIMITATION",
            "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED",
            "planned_commands": sum(int(self.config["schedule"]["commands_per_plateau"]) for _ in self.config["schedule"]["commanded_noise_plateaus_db"]),
            "sent_commands": len(sent),
            "skipped_obsolete_commands": len(self.command_rows) - len(sent),
            "handler_bracket_ms_p50": percentile(latencies, 0.50),
            "handler_bracket_ms_p95": percentile(latencies, 0.95),
            "handler_bracket_ms_max": max(latencies) if latencies else None,
            "responses_before_next_boundary": sum(bool(row["response_before_next_boundary"]) for row in sent),
            "all_responses_before_next_boundary": all(bool(row["response_before_next_boundary"]) for row in sent),
            "descriptive_monotone_response": monotone,
            "traffic_sender_frames": sender_frames,
            "traffic_sink_packets": sink_packets,
            "traffic_delivery_ratio": sink_packets / sender_frames,
            "plateaus": plateau_rows,
            "restored_to_clean_minus50": self.restored,
            "next": "UE-N3",
        }

    def write_raw_limit_record(self) -> None:
        records = []
        for source in ("gnb", "ue"):
            raw = self.output_dir / "ttracer" / source / f"{source}.raw"
            records.append({
                "source": source, "raw_file": str(raw.relative_to(self.output_dir)),
                "raw_file_sha256": sha256(raw) if raw.exists() else None,
                "raw_event_envelope_status": "UNAVAILABLE_WITH_STOCK_EXPORTER",
                "missing_reason_code": "STOCK_CSV_NO_FULL_EPOCH_OR_INGEST_CLOCK",
            })
        atomic_text(
            self.output_dir / self.config["output"]["raw_radio_events"],
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        )

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        atomic_json(self.output_dir / self.config["output"]["meeting_summary_json"], summary)
        lines = [
            "# UE-N2 bounded OAI calibration smoke", "",
            f"**Status:** `{summary['status']}`", "",
            "This is partial evidence: command timing and descriptive per-plateau radio response are usable, but the stock tracer cannot prove a causal first-effect timestamp.", "",
            f"- Commands sent: {summary['sent_commands']} / {summary['planned_commands']}",
            f"- Handler bracket p50/p95/max: {summary['handler_bracket_ms_p50']:.3f} / {summary['handler_bracket_ms_p95']:.3f} / {summary['handler_bracket_ms_max']:.3f} ms",
            f"- All responses before the next 100-ms boundary: {summary['all_responses_before_next_boundary']}",
            f"- Descriptive monotone command-to-SNR response: {summary['descriptive_monotone_response']}",
            f"- Restored to clean -50: {summary['restored_to_clean_minus50']}", "",
            "| RFsim noise command (dB) | median PUSCH SNR (dB) | median scheduler EMA SNR (dB) | median final MCS | EMA status |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in summary["plateaus"]:
            def fmt(value: Any) -> str:
                return "n/a" if value is None else f"{float(value):.2f}"
            lines.append(
                f"| {row['commanded_noise_power_db']} | {fmt(row['instantaneous_pusch_snr_db_median'])} | {fmt(row['scheduler_ema_snr_db_median'])} | {fmt(row['scheduler_final_mcs_median'])} | {row['ema_settling_status']} |"
            )
        lines += ["", "Direct UL BLER remains unavailable; no value was zero-filled.", ""]
        atomic_text(self.output_dir / self.config["output"]["meeting_summary_md"], "\n".join(lines))

    def manifest(self, status: str) -> None:
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "FAILED.json", "UE_N2_SMOKE_CAPTURED.json"}:
                files.append({
                    "path": str(path.relative_to(self.output_dir)),
                    "bytes": path.stat().st_size, "sha256": sha256(path),
                })
        atomic_json(self.output_dir / "manifest.json", {
            "schema": "scenesense.ue_n2_oai_ul_calibration_smoke_manifest.v1",
            "status": status, "created_at": self.started_at, "completed_at": utc_now(),
            "config_path": str(self.config_path), "config_sha256": sha256(self.config_path),
            "ran_epoch_id": self.ran_epoch_id, "control_session_id": self.control_session_id,
            "outputs": files,
        })

    def cleanup(self, *, strict: bool = False) -> list[str]:
        errors: list[str] = []
        if self.telnet is not None:
            try:
                self.telnet.close()
            except Exception as exc:
                errors.append(f"telnet close: {exc}")
            self.telnet = None
        if self.live_csv is not None:
            try:
                self.live_csv.stop()
            except Exception as exc:
                errors.append(f"live csv stop: {exc}")
            self.live_csv = None
        for managed in reversed(self.processes):
            try:
                managed.stop()
            except Exception as exc:
                errors.append(f"{managed.name} stop: {exc}")

        ports = [
            4043, int(self.config["actuator"]["telnet_port"]),
            int(self.config["telemetry"]["gnb_port"]),
            int(self.config["telemetry"]["ue_port"]),
            int(self.config["telemetry"]["gnb_relay_port"]),
            int(self.config["telemetry"]["ue_relay_port"]),
        ]
        deadline = time.monotonic() + 10.0
        state: dict[str, Any] = {}
        while True:
            alive = [proc.name for proc in self.processes if proc.process.poll() is None]
            tunnel_interfaces = oai_tunnel_interfaces()
            busy_ports = [port for port in ports if not port_is_free(port)]
            state = {
                "owned_processes_alive": alive,
                "oai_tunnel_interfaces_present": tunnel_interfaces,
                "busy_owned_ports": busy_ports,
            }
            if not alive and not tunnel_interfaces and not busy_ports:
                break
            if time.monotonic() >= deadline:
                errors.append(f"post-cleanup cold-state gate failed: {state}")
                break
            time.sleep(0.25)
        report = {
            "clean": not errors,
            "errors": errors,
            **state,
            "checked_at": utc_now(),
        }
        atomic_json(self.output_dir / "cleanup_report.json", report)
        if strict and errors:
            raise SmokeFailure("cleanup verification failed: " + "; ".join(errors))
        return errors

    def best_effort_restore(self) -> None:
        if self.restored:
            return
        actuator = self.config["actuator"]

        def attempt(session: TelnetSession) -> bool:
            response = session.command("channelmod show current")[-1]
            row = parse_channel_models(response).get(actuator["channel_model_name"])
            if not row:
                return False
            clean = actuator["clean_and_restore_commanded_noise_power_db"]
            modified = session.command(
                f"channelmod modify {int(row['model_index'])} noise_power_dB {clean}"
            )[-1]
            self.validate_modify_response(modified, clean)
            verify = session.command("channelmod show current")[-1]
            final = parse_channel_models(verify).get(actuator["channel_model_name"], {})
            self.restored = (
                abs(float(final.get("noise_power_db", math.nan)) - float(clean)) <= 1e-6
            )
            if self.restored:
                atomic_text(
                    self.output_dir / "channel_state_restored_failure_cleanup.txt", verify
                )
            return self.restored

        # First preserve continuity by trying the owned control session. If it
        # is stale or broken, retry once on a separate cleanup-only session.
        if self.telnet is not None:
            try:
                if attempt(self.telnet):
                    return
            except Exception:
                pass
        cleanup_session: TelnetSession | None = None
        try:
            cleanup_session = TelnetSession(
                actuator["telnet_host"], int(actuator["telnet_port"]),
                float(actuator["response_timeout_s"]),
                int(actuator["max_response_bytes"]),
            )
            attempt(cleanup_session)
        except Exception:
            return
        finally:
            if cleanup_session is not None:
                cleanup_session.close()

    def run(self) -> int:
        atomic_json(self.output_dir / "resolved_config.json", self.config)
        previous_handlers: dict[signal.Signals, Any] = {}

        def raise_termination(signum: int, _frame: Any) -> None:
            raise SmokeFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught_signal in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught_signal] = signal.getsignal(caught_signal)
            signal.signal(caught_signal, raise_termination)
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            self.wait_attach()
            wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            self.start_telemetry()
            time.sleep(float(self.config["capture"]["recorder_lead_s"]))
            assert self.live_csv is not None
            pretraffic_pusch_count = self.live_csv.count()
            self.start_traffic()
            self.wait_clean_pusch(pretraffic_pusch_count)
            model_index = self.open_and_validate_telnet()
            self.run_trace(model_index)
            self.restore(model_index)
            self.write_command_log()
            self.cleanup(strict=True)
            self.extract_ttracer()
            self.write_raw_limit_record()
            summary = self.analyze()
            self.write_summary(summary)
            self.manifest(SUCCESS_STATUS)
            terminal = {
                "status": SUCCESS_STATUS, "evidence_class": "PARTIAL_EVIDENCE",
                "full_raw_event_envelope_satisfied": False,
                "causal_first_effect_status": "UNAVAILABLE_STOCK_TRACER_LIMITATION",
                "clean_restore_verified": self.restored,
                "meeting_summary": self.config["output"]["meeting_summary_md"],
                "manifest_sha256": sha256(self.output_dir / "manifest.json"),
                "next": "UE-N3",
            }
            atomic_json(self.output_dir / self.config["output"]["terminal_success"], terminal)
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
                "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                "clean_restore_verified": self.restored, "failed_at": utc_now(),
                "cleanup_errors": cleanup_errors,
            }
            atomic_json(self.output_dir / self.config["output"]["terminal_failure"], failure)
            self.manifest("FAILED")
            print(json.dumps({"output_dir": str(self.output_dir), **failure}, sort_keys=True), file=sys.stderr)
            return 1
        finally:
            for caught_signal, previous in previous_handlers.items():
                signal.signal(caught_signal, previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return Runner(Path(args.config), Path(args.output_dir)).run()


if __name__ == "__main__":
    raise SystemExit(main())
