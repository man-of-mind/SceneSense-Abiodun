#!/usr/bin/env python3
"""Short bidirectional-iperf qualification of the UE-side downlink SNR signal.

Question: is ``UE_PHY_MEAS.snr`` -- a UE *receive-side downlink* measurement --
available, fresh and informative enough to serve as the physical-channel input
to the SplitFusion policy?

This is explicitly **not** an attempt to show that the UE downlink SNR equals
the gNB's received uplink PUSCH SNR. They measure opposite link directions. The
two are compared for *ordering*, *association* and *tracking* only, and are
never given the same label.

Network only. No CARLA, no CUDA, no model inference, no spatial map, and no
edge/inference container. The OAI CN5G core is expected to be already running.

Reuses the known-safe launch/teardown mechanism from
``rl_agent.ue_n2_oai_ul_calibration_smoke`` rather than re-implementing it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_agent.ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_v1.json"
STATUS_OK = "UE_SNR_BRIDGE_EVIDENCE_CAPTURED"


class BridgeFailure(RuntimeError):
    """Any refusal that must abort the live run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeFailure(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def round_to_granularity(value: float, granularity: float) -> float:
    """Round a commanded dB value onto the actuator's command grid."""
    steps = round(float(value) / float(granularity))
    return round(steps * float(granularity), 6)


def inverse_interpolate_noise_command(
    target_snr_db: float, anchors: Sequence[Mapping[str, float]]
) -> tuple[float, bool]:
    """Map a target achieved SNR to a commanded ``noise_power_dB``.

    Anchors are (commanded noise, measured achieved median PUSCH SNR) pairs.
    Returns ``(command_db, clamped)``; a target outside the measured span is
    clamped to the nearest anchor and flagged rather than extrapolated.
    """
    ordered = sorted(anchors, key=lambda row: float(row["achieved_median_pusch_snr_db"]))
    lows = [float(row["achieved_median_pusch_snr_db"]) for row in ordered]
    cmds = [float(row["noise_power_db"]) for row in ordered]
    target = float(target_snr_db)
    if target <= lows[0]:
        return cmds[0], target < lows[0]
    if target >= lows[-1]:
        return cmds[-1], target > lows[-1]
    for index in range(len(ordered) - 1):
        left, right = lows[index], lows[index + 1]
        if left <= target <= right:
            span = right - left
            fraction = 0.0 if span == 0 else (target - left) / span
            return cmds[index] + fraction * (cmds[index + 1] - cmds[index]), False
    raise BridgeFailure(f"target {target} could not be mapped")


def load_trace_prefix(
    trace_csv: Path, profile_id: str, count: int
) -> list[dict[str, Any]]:
    """First ``count`` registered samples of one profile, in step order.

    The registered trace values are used verbatim; only the prefix length is
    bounded, so this remains the registered profile rather than a replacement.
    """
    rows: list[dict[str, Any]] = []
    with trace_csv.open("r", newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["profile_id"] != profile_id:
                continue
            rows.append(
                {
                    "step_index": int(record["step_index"]),
                    "state": record["state"],
                    "target_snr_db": float(record["target_snr_db"]),
                    "trace_id": record["trace_id"],
                }
            )
    rows.sort(key=lambda row: row["step_index"])
    require(len(rows) >= count, f"{profile_id}: trace has {len(rows)} < {count} samples")
    return rows[:count]


@dataclass
class IperfSession:
    """One one-way UDP iperf3 session and the JSON summary it produced."""

    label: str
    direction: str
    argv: list[str]
    process: subprocess.Popen[str] | None = None
    stdout_path: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)


class Runner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path
        self.config = json.loads(config_path.read_text())
        self.output_dir = output_dir
        self.processes: list[n2.ManagedProcess] = []
        self.telnet: n2.TelnetSession | None = None
        self.model_index: int | None = None
        self.ue_ip: str | None = None
        self.restored = False
        self.command_rows: list[dict[str, Any]] = []
        self.profile_rows: list[dict[str, Any]] = []
        self.traffic_rows: list[dict[str, Any]] = []
        self.anchors: list[dict[str, float]] = [
            dict(row) for row in self.config["actuator"]["existing_measured_anchors"]
        ]
        self.clock_anchors: list[dict[str, Any]] = []
        self.live_pusch: n2.LiveCsv | None = None
        self.teardown_notes: list[str] = []

    # -- small helpers -------------------------------------------------
    def path(self, relative: str) -> Path:
        return ROOT / relative

    def out(self, relative: str) -> Path:
        target = self.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def spawn(self, name: str, argv: Sequence[str], log_name: str, *, cwd: Path = ROOT,
              root_owned: bool = False) -> n2.ManagedProcess:
        log_path = self.out(log_name)
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            list(argv), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        managed = n2.ManagedProcess(name=name, process=process, log_handle=handle,
                                    root_owned=root_owned)
        self.processes.append(managed)
        return managed

    def take_clock_anchor(self, label: str) -> dict[str, Any]:
        """Same-host wall/monotonic anchor, used to date the tracer CSVs.

        The T-tracer renders CLOCK_REALTIME as a date-less local ``HH:MM:SS``,
        so the run date and UTC offset must be recovered from a recorded
        anchor rather than assumed.
        """
        wall_ns = time.time_ns()
        anchor = {
            "label": label,
            "wall_ns": wall_ns,
            "monotonic_ns": time.monotonic_ns(),
            "local_iso": datetime.fromtimestamp(wall_ns / 1e9).astimezone().isoformat(),
            "utc_offset_s": int(
                datetime.fromtimestamp(wall_ns / 1e9).astimezone().utcoffset().total_seconds()
            ),
        }
        self.clock_anchors.append(anchor)
        return anchor

    # -- B1 preflight --------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        radio = self.config["radio"]
        n2.run_checked(["sudo", "-n", "true"])
        container_state: dict[str, str] = {}
        for container in radio["core_containers"]:
            state = n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                container,
            ]).stdout.strip()
            container_state[container] = state
            require(state.startswith("true") and "unhealthy" not in state,
                    f"core container is not ready: {container}={state!r}")

        for process_name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-a", "-x", process_name],
                                   text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            require(not (found.returncode == 0 and found.stdout.strip()),
                    f"cold-RAN gate failed: {found.stdout.strip()}")
        stale = n2.oai_tunnel_interfaces()
        require(not stale, f"cold-RAN gate failed: stale UE tunnel(s) {stale}")

        ports = [4043, self.config["actuator"]["telnet_port"],
                 self.config["telemetry"]["gnb_port"], self.config["telemetry"]["ue_port"],
                 self.config["telemetry"]["gnb_relay_port"],
                 self.config["telemetry"]["ue_relay_port"],
                 self.config["traffic"]["downlink_port"]]
        busy = [port for port in ports if not n2.port_is_free(int(port))]
        require(not busy, f"required TCP ports are not free: {busy}")

        # The tracer refuses to attach unless its message database is
        # byte-identical to the copy compiled into the softmodem, so verify the
        # event we depend on is actually in the database we will pass it.
        messages = self.path(self.config["paths"]["t_messages"]).read_text()
        require("ID = UE_PHY_MEAS" in messages,
                "UE_PHY_MEAS is absent from T_messages.txt; it cannot be activated")

        carla = subprocess.run(["pgrep", "-af", "CarlaUE4|CarlaUnreal"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        require(not carla.stdout.strip(), "a CARLA process is running; this is network-only")

        snapshot = {
            "utc": utc_now(),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
            "core_containers": container_state,
            "cold_softmodem": True,
            "cold_tunnels": True,
            "free_ports": ports,
            "iperf3_host_version": n2.run_checked(["iperf3", "--version"]).stdout.splitlines()[0],
            "iperf3_ext_dn_version": n2.run_checked([
                "sudo", "-n", "docker", "exec", radio["ext_dn_container"],
                "iperf3", "--version"]).stdout.splitlines()[0],
            "ue_phy_meas_in_database": True,
        }
        self.out("preflight.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        return snapshot

    # -- RAN lifecycle (reused shape from the known-safe launcher) ------
    def materialize_configs(self) -> tuple[Path, Path]:
        paths, radio = self.config["paths"], self.config["radio"]
        conf_root = self.path(paths["oai_ran_conf"])
        gnb_base = (conf_root / paths["gnb_base_config"]).read_text(encoding="utf-8")
        ue_base = (conf_root / paths["ue_base_config"]).read_text(encoding="utf-8")
        channel = (conf_root / paths["channel_config"]).read_text(encoding="utf-8")

        uicc_blocks = re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue_base)
        imsis = re.findall(r'(?m)^\s*imsi\s*=\s*"([0-9]+)"\s*;', ue_base)
        require(int(radio["ue_count"]) == 1 and len(uicc_blocks) == 1,
                f"effective UE config is not single-UE: uicc_blocks={len(uicc_blocks)}")
        require(imsis == [str(radio["expected_imsi"])],
                f"effective UE IMSI mismatch: {imsis}")

        clean = self.config["actuator"]["clean_and_restore_commanded_noise_power_db"]
        channel, replacements = re.subn(
            r"noise_power_dB\s*=\s*[-+0-9.eE]+;", f"noise_power_dB = {clean};", channel)
        require(replacements == 3,
                f"expected exactly three single-UE channel noise values, found {replacements}")
        require("noise_power_dBFS" not in channel, "global noise_power_dBFS must remain unset")
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        require(marker in ue_base, "UE base config lacks expected channel include")

        gnb_path = self.out("runtime/effective_gnb.conf")
        ue_path = self.out("runtime/effective_ue.conf")
        gnb_path.write_text(gnb_base + "\n\n" + channel + "\n")
        ue_path.write_text(ue_base.replace(marker, channel))
        self.out("runtime/config_hashes.json").write_text(json.dumps({
            "gnb_sha256": n2.sha256(gnb_path), "ue_sha256": n2.sha256(ue_path),
            "source_channel_sha256": n2.sha256(conf_root / paths["channel_config"]),
            "commanded_clean_noise_power_db": clean,
        }, indent=2) + "\n")
        return gnb_path, ue_path

    def start_ran(self, gnb_config: Path, ue_config: Path) -> None:
        radio = self.config["radio"]
        build = self.path(self.config["paths"]["oai_ran_build"])
        telemetry, actuator = self.config["telemetry"], self.config["actuator"]
        gnb = ["sudo", "-n", "env", "-u", "SCENESENSE_FORCE_UL_MCS",
               f"SCENESENSE_MCS_POLICY={radio['mcs_policy']}", "./nr-softmodem",
               "-O", str(gnb_config), "--gNBs.[0].min_rxtxtime", "6", "--rfsim",
               "--rfsimulator.[0].options", "chanmod", "--telnetsrv",
               "--telnetsrv.listenaddr", actuator["telnet_host"],
               "--telnetsrv.listenport", str(actuator["telnet_port"]),
               "--T_stdout", "2", "--T_nowait", "--T_port", str(telemetry["gnb_port"])]
        self.spawn("gnb", gnb, "logs/gnb.log", cwd=build, root_owned=True)
        time.sleep(float(radio["gnb_start_lead_s"]))
        ue = ["sudo", "-n", "./nr-uesoftmodem", "--rfsim",
              "--rfsimulator.[0].serveraddr", "127.0.0.1",
              "--rfsimulator.[0].options", "chanmod", "-r", str(radio["prb"]),
              "--numerology", str(radio["numerology"]), "--band", str(radio["band"]),
              "-C", str(radio["downlink_frequency_hz"]), "-O", str(ue_config),
              "--T_stdout", "2", "--T_nowait", "--T_port", str(telemetry["ue_port"])]
        self.spawn("ue", ue, "logs/ue.log", cwd=build, root_owned=True)

    def wait_attach(self) -> None:
        radio = self.config["radio"]
        deadline = time.monotonic() + float(radio["attach_timeout_s"])
        interface = radio["ue_interface"]
        while time.monotonic() < deadline:
            require(all(proc.process.poll() is None for proc in self.processes[:2]),
                    "gNB or UE exited before attachment")
            address = subprocess.run(["ip", "-j", "-4", "addr", "show", "dev", interface],
                                     text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
            ips: list[str] = []
            if address.returncode == 0:
                try:
                    ips = [str(info["local"]) for row in json.loads(address.stdout)
                           for info in row.get("addr_info", [])
                           if info.get("family") == "inet" and info.get("local")]
                except (json.JSONDecodeError, KeyError, TypeError):
                    ips = []
            if len(ips) == 1:
                ping = subprocess.run(
                    ["ping", "-I", interface, "-c", "3", "-W", "2", radio["ext_dn_ip"]],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if ping.returncode == 0:
                    self.ue_ip = ips[0]
                    self.out("logs/attach_ping.log").write_text(ping.stdout)
                    self.out("ue_network_identity.json").write_text(json.dumps({
                        "ue_count": 1, "imsi": radio["expected_imsi"],
                        "interface": interface, "discovered_ipv4": self.ue_ip,
                        "ext_dn_ip": radio["ext_dn_ip"], "ping_pass": True,
                    }, indent=2) + "\n")
                    return
            time.sleep(1)
        raise BridgeFailure("single UE did not attach and reach ext-DN before timeout")

    def start_telemetry(self) -> None:
        telemetry = self.config["telemetry"]
        troot = self.path(self.config["paths"]["t_tracer_dir"])
        messages = self.path(self.config["paths"]["t_messages"])
        for source, port, relay in (("gnb", telemetry["gnb_port"], telemetry["gnb_relay_port"]),
                                    ("ue", telemetry["ue_port"], telemetry["ue_relay_port"])):
            self.spawn(f"{source}_relay",
                       [str(troot / "multi"), "-d", str(messages), "-ip", "127.0.0.1",
                        "-p", str(port), "-lp", str(relay)], f"logs/{source}_relay.log")
        n2.wait_tcp(int(telemetry["gnb_relay_port"]), 15)
        n2.wait_tcp(int(telemetry["ue_relay_port"]), 15)
        for source, relay in (("gnb", telemetry["gnb_relay_port"]),
                              ("ue", telemetry["ue_relay_port"])):
            raw = self.out(f"ttracer/{source}/{source}.raw")
            argv = [str(troot / "record"), "-d", str(messages), "-o", str(raw), "-OFF"]
            for event in telemetry["events"][source]:
                argv += ["-on", event]
            argv += ["-ip", "127.0.0.1", "-p", str(relay)]
            self.spawn(f"{source}_record", argv, f"logs/{source}_record.log")
        time.sleep(2.0)
        for source in ("gnb", "ue"):
            log = (self.output_dir / f"logs/{source}_record.log").read_text()
            require("ERROR" not in log, f"{source} tracer refused its event set:\n{log[-800:]}")

    # -- actuation -----------------------------------------------------
    def open_telnet(self) -> int:
        actuator = self.config["actuator"]
        self.telnet = n2.TelnetSession(actuator["telnet_host"], int(actuator["telnet_port"]),
                                       float(actuator["response_timeout_s"]),
                                       int(actuator["max_response_bytes"]))
        _, _, _, _, state = self.telnet.command("channelmod show current")
        self.out("channel_state_initial.txt").write_text(state)
        models = n2.parse_channel_models(state)
        name = actuator["channel_model_name"]
        require(name in models, f"channel model {name!r} not present: {sorted(models)}")
        index = int(models[name]["model_index"])
        self.model_index = index
        return index

    def send_noise_command(self, command_db: float, *, reason: str,
                           step_index: int | None = None,
                           profile_id: str | None = None,
                           target_snr_db: float | None = None,
                           clamped: bool = False) -> dict[str, Any]:
        assert self.telnet is not None and self.model_index is not None
        text = f"{command_db:g}"
        send_mono, send_wall, ack_mono, ack_wall, response = self.telnet.command(
            f"channelmod modify {self.model_index} noise_power_dB {text}")
        row = {
            "reason": reason, "profile_id": profile_id, "step_index": step_index,
            "target_snr_db": target_snr_db, "commanded_noise_power_db": float(command_db),
            "clamped": bool(clamped), "send_monotonic_ns": send_mono,
            "send_wall_ns": send_wall, "ack_monotonic_ns": ack_mono,
            "ack_wall_ns": ack_wall, "ack_latency_ms": (ack_mono - send_mono) / 1e6,
            "status": "ACK" if "ERROR" not in response.upper() else "ERROR",
        }
        self.command_rows.append(row)
        return row

    def start_live_pusch_feed(self) -> None:
        """Live gNB PUSCH-SNR stream, needed to close the mapping in-run."""
        telemetry = self.config["telemetry"]
        troot = self.path(self.config["paths"]["t_tracer_dir"])
        messages = self.path(self.config["paths"]["t_messages"])
        fields = ("time", "rnti", "frame", "slot", "snrx10", "phr", "tpc",
                  "tb_size", "txpower_calc", "rbSize", "mcs", "rssi")
        command = [str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                   "-p", str(telemetry["gnb_relay_port"]), "-f", "-s", ",",
                   "-t", "time", "GNB_MAC_PUSCH_POWER_CONTROL", *fields]
        self.live_pusch = n2.LiveCsv(
            command, self.out("ttracer/gnb/live_pusch_with_ingest.csv"))

    def live_pusch_snr_db(self, since_index: int) -> list[float]:
        """Decoded PUSCH SNR in dB from the live feed, from ``since_index`` on.

        The gNB reports ``snrx10``; the conversion to dB is recorded here and
        nowhere else, so the raw integer stays the retained quantity.
        """
        out: list[float] = []
        for _, _, line in self.live_pusch.snapshot()[since_index:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                out.append(int(parts[4]) / 10.0)
            except ValueError:
                continue
        return out

    def measure_upper_anchor(self) -> dict[str, Any]:
        """Close the registered mapping at its upper end, by live measurement.

        The eight registered anchors stop at 19.5 dB achieved, well below the
        FAVORABLE_STABLE targets, so replaying on them alone would clamp most
        of that profile and flatten exactly the contrast the qualification is
        testing. The upper anchor is therefore measured in-run with the
        registered first command, and appended to the mapping before any
        profile is replayed. Nothing is extrapolated.
        """
        upper = self.config["actuator"]["upper_anchor"]
        traffic = self.config["traffic"]
        command = float(upper["first_command_db"])
        duration = float(upper["measurement_duration_s"])

        self.send_noise_command(command, reason="UPPER_ANCHOR_CALIBRATION")
        sessions = self.run_bidirectional_traffic("CALIBRATION", duration + 3.0)
        time.sleep(1.5)                      # let the grant settle
        baseline = self.live_pusch.count()
        start_ns = time.time_ns()
        time.sleep(duration)
        samples = self.live_pusch_snr_db(baseline)
        end_ns = time.time_ns()
        self.collect_traffic(sessions, "CALIBRATION")

        resolved = len(samples) >= int(upper["minimum_pusch_samples"])
        median = (sorted(samples)[len(samples) // 2] if samples else None)
        outcome = {
            "commanded_noise_power_db": command,
            "window_start_wall_ns": start_ns,
            "window_end_wall_ns": end_ns,
            "pusch_sample_count": len(samples),
            "minimum_required": int(upper["minimum_pusch_samples"]),
            "achieved_median_pusch_snr_db": median,
            "resolved": bool(resolved),
            "snrx10_to_db": "gNB reports snrx10; dB = snrx10 / 10",
        }
        if resolved and median is not None:
            self.anchors.append({
                "noise_power_db": command,
                "achieved_median_pusch_snr_db": float(median),
            })
            outcome["mapping_upper_bound_db"] = float(median)
        else:
            outcome["note"] = (
                "upper anchor did not resolve; targets above the registered "
                "19.5 dB anchor will be clamped and counted, never extrapolated"
            )
        outcome["anchors_in_use"] = list(self.anchors)
        self.out("upper_anchor.json").write_text(json.dumps(outcome, indent=2) + "\n")

        clean = self.config["actuator"]["clean_and_restore_commanded_noise_power_db"]
        self.send_noise_command(float(clean), reason="POST_CALIBRATION_CLEAN")
        time.sleep(float(self.config["replay"]["clean_recovery_s"]))
        return outcome

    # -- traffic -------------------------------------------------------
    def start_iperf_servers(self) -> None:
        traffic, radio = self.config["traffic"], self.config["radio"]
        assert self.ue_ip is not None
        # Downlink server lives on the host, bound to the UE tunnel address.
        self.spawn("dl_server",
                   ["iperf3", "-s", "-p", str(traffic["downlink_port"]), "-B", self.ue_ip],
                   "logs/iperf_dl_server.log")
        # Uplink server lives inside the external data network container.
        self.spawn("ul_server",
                   ["sudo", "-n", "docker", "exec", radio["ext_dn_container"],
                    "iperf3", "-s", "-p", str(traffic["uplink_port"])],
                   "logs/iperf_ul_server.log")
        time.sleep(float(traffic["server_settle_s"]))

    def run_bidirectional_traffic(self, profile_id: str, duration_s: float) -> list[IperfSession]:
        """Two concurrent one-way UDP sessions at one constant offered load.

        ``--bidir`` is deliberately not used: it would force the same offered
        rate on both directions, and the design needs a sustained uplink with a
        smaller sustained downlink so that PDSCH receptions - and therefore UE
        measurements - keep occurring without the downlink competing for the
        uplink grant.
        """
        traffic, radio = self.config["traffic"], self.config["radio"]
        assert self.ue_ip is not None
        seconds = str(int(math.ceil(duration_s)))
        uplink = IperfSession(
            label=f"{profile_id}:UL", direction="UPLINK_UE_TO_NETWORK",
            argv=["iperf3", "-c", radio["ext_dn_ip"], "-p", str(traffic["uplink_port"]),
                  "-u", "-b", f"{traffic['uplink_mbps']}M", "-l", str(traffic["datagram_bytes"]),
                  "-t", seconds, "-B", self.ue_ip, "--forceflush", "-J"])
        downlink = IperfSession(
            label=f"{profile_id}:DL", direction="DOWNLINK_NETWORK_TO_UE",
            argv=["sudo", "-n", "docker", "exec", radio["ext_dn_container"],
                  "iperf3", "-c", self.ue_ip, "-p", str(traffic["downlink_port"]),
                  "-u", "-b", f"{traffic['downlink_mbps']}M",
                  "-l", str(traffic["datagram_bytes"]), "-t", seconds, "--forceflush", "-J"])
        for session in (uplink, downlink):
            suffix = "ul" if "UPLINK" in session.direction else "dl"
            session.stdout_path = self.out(f"traffic/{profile_id}_{suffix}.json")
            handle = session.stdout_path.open("w", encoding="utf-8")
            session.process = subprocess.Popen(
                session.argv, stdout=handle, stderr=subprocess.STDOUT, text=True,
                start_new_session=True)
            session.summary["_handle"] = handle
        return [uplink, downlink]

    def collect_traffic(self, sessions: Sequence[IperfSession], profile_id: str) -> None:
        for session in sessions:
            assert session.process is not None
            try:
                session.process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=10)
            session.summary.pop("_handle").close()
            parsed: dict[str, Any] = {}
            try:
                payload = json.loads(session.stdout_path.read_text())
                end = payload.get("end", {})
                summary = end.get("sum") or end.get("sum_sent") or {}
                parsed = {
                    "offered_bytes": summary.get("bytes"),
                    "achieved_mbps": (summary.get("bits_per_second") or 0) / 1e6,
                    "jitter_ms": summary.get("jitter_ms"),
                    "lost_packets": summary.get("lost_packets"),
                    "total_packets": summary.get("packets"),
                    "lost_percent": summary.get("lost_percent"),
                    "seconds": summary.get("seconds"),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                parsed = {"parse_error": str(exc)}
            self.traffic_rows.append({
                "profile_id": profile_id, "direction": session.direction,
                "argv": " ".join(session.argv), **parsed,
            })

    # -- one profile ---------------------------------------------------
    def run_profile(self, profile_id: str) -> None:
        replay = self.config["replay"]
        actuator = self.config["actuator"]
        total = int(replay["warmup_samples"]) + int(replay["measured_samples"])
        period = float(replay["sample_period_s"])
        samples = load_trace_prefix(
            self.path(self.config["paths"]["trace_csv"]), profile_id, total)

        duration = total * period
        self.take_clock_anchor(f"{profile_id}:before_traffic")
        sessions = self.run_bidirectional_traffic(profile_id, duration + 1.0)
        profile_start_mono = time.monotonic_ns()
        profile_start_wall = time.time_ns()

        # Actuate the registered trace on its own 100 ms grid. Obsolete samples
        # are skipped rather than burst, so a slow ACK never compresses the
        # remaining schedule.
        last_command: float | None = None
        sent = skipped = clamped_count = 0
        for sample in samples:
            due_mono = profile_start_mono + int(sample["step_index"] * period * 1e9)
            now = time.monotonic_ns()
            if now > due_mono + int(period * 1e9):
                skipped += 1
                continue
            if now < due_mono:
                time.sleep((due_mono - now) / 1e9)
            command, was_clamped = inverse_interpolate_noise_command(
                sample["target_snr_db"], self.anchors)
            command = round_to_granularity(command, float(actuator["command_granularity_db"]))
            clamped_count += int(was_clamped)
            if last_command is None or command != last_command:
                self.send_noise_command(
                    command, reason="PROFILE_REPLAY", step_index=sample["step_index"],
                    profile_id=profile_id, target_snr_db=sample["target_snr_db"],
                    clamped=was_clamped)
                last_command = command
                sent += 1

        measured_start_wall = profile_start_wall + int(
            int(replay["warmup_samples"]) * period * 1e9)
        profile_end_wall = time.time_ns()
        self.collect_traffic(sessions, profile_id)

        self.profile_rows.append({
            "profile_id": profile_id,
            "trace_id": samples[0]["trace_id"],
            "samples_scheduled": len(samples),
            "warmup_samples": int(replay["warmup_samples"]),
            "measured_samples": int(replay["measured_samples"]),
            "commands_sent": sent,
            "commands_skipped_obsolete": skipped,
            "targets_clamped_to_mapping": clamped_count,
            "profile_start_wall_ns": profile_start_wall,
            "measured_window_start_wall_ns": measured_start_wall,
            "profile_end_wall_ns": profile_end_wall,
            "target_snr_db_min": min(s["target_snr_db"] for s in samples),
            "target_snr_db_max": max(s["target_snr_db"] for s in samples),
        })

        # Return to the clean channel between profiles so a profile never
        # inherits the previous profile's fade.
        clean = actuator["clean_and_restore_commanded_noise_power_db"]
        self.send_noise_command(float(clean), reason="INTER_PROFILE_CLEAN_RECOVERY")
        time.sleep(float(replay["clean_recovery_s"]))

    # -- B9 restore / teardown ----------------------------------------
    def restore(self) -> None:
        assert self.telnet is not None and self.model_index is not None
        actuator = self.config["actuator"]
        clean = actuator["clean_and_restore_commanded_noise_power_db"]
        self.send_noise_command(float(clean), reason="FINAL_RESTORE")
        _, _, _, _, state = self.telnet.command("channelmod show current")
        models = n2.parse_channel_models(state)
        row = models.get(actuator["channel_model_name"], {})
        observed = float(row.get("noise_power_db", math.nan))
        require(abs(observed - float(clean)) <= 1e-6,
                f"clean restore read-back failed: observed={observed} expected={clean}")
        self.out("channel_state_restored.txt").write_text(state)
        self.out("restore_readback.json").write_text(json.dumps({
            "commanded_clean_noise_power_db": float(clean),
            "read_back_noise_power_db": observed,
            "verified": True, "utc": utc_now(),
        }, indent=2) + "\n")
        self.restored = True

    def extract_ttracer(self) -> None:
        script = self.path(self.config["paths"]["extract_script"])
        for source in ("gnb", "ue"):
            raw = self.output_dir / "ttracer" / source / f"{source}.raw"
            require(raw.exists() and raw.stat().st_size > 0, f"missing raw T file: {raw}")
            argv = [str(script), "--raw", str(raw), "--source", source,
                    "--output-root", str(self.output_dir), "--clean-output"]
            for event in self.config["telemetry"]["events"][source]:
                argv += ["--event", event]
            n2.run_checked(argv, timeout=600)

    def cleanup(self) -> list[str]:
        notes: list[str] = []
        if self.live_pusch is not None:
            try:
                self.live_pusch.stop()
            except Exception as exc:  # noqa: BLE001 - teardown must continue
                notes.append(f"live pusch feed: {exc}")
            self.live_pusch = None
        if self.telnet is not None:
            try:
                self.telnet.close()
            except OSError as exc:
                notes.append(f"telnet close: {exc}")
            self.telnet = None
        for managed in reversed(self.processes):
            try:
                managed.stop()
            except Exception as exc:  # noqa: BLE001 - teardown must continue
                notes.append(f"{managed.name}: {exc}")
        self.processes.clear()
        container = self.config["radio"]["ext_dn_container"]
        killed = subprocess.run(
            ["sudo", "-n", "docker", "exec", container, "pkill", "-f", "iperf3"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if killed.returncode not in (0, 1):
            notes.append(f"container iperf3 reap: rc={killed.returncode} {killed.stdout.strip()}")
        return notes

    def verify_cold(self) -> dict[str, Any]:
        """B9: prove nothing of ours survived."""
        orphans: dict[str, list[str]] = {}
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-a", "-x", name], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if found.returncode == 0 and found.stdout.strip():
                orphans[name] = found.stdout.strip().splitlines()
        for name, pattern in (("tracer", "T/tracer/(record|multi|csv)"),
                              ("iperf3", "iperf3")):
            found = subprocess.run(["pgrep", "-af", pattern], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            rows = [line for line in found.stdout.splitlines()
                    if str(self.output_dir) in line
                    or f"-p {self.config['traffic']['uplink_port']}" in line
                    or f"-p {self.config['traffic']['downlink_port']}" in line]
            if rows:
                orphans[name] = rows
        container = self.config["radio"]["ext_dn_container"]
        inside = subprocess.run(
            ["sudo", "-n", "docker", "exec", container, "pgrep", "-af", "iperf3"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if inside.returncode == 0 and inside.stdout.strip():
            orphans["ext_dn_iperf3"] = inside.stdout.strip().splitlines()
        tunnels = n2.oai_tunnel_interfaces()
        state = {
            "utc": utc_now(),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
            "orphan_processes": orphans,
            "residual_ue_tunnels": tunnels,
            "cold": not orphans and not tunnels,
        }
        self.out("final_cold_state.json").write_text(json.dumps(state, indent=2) + "\n")
        return state

    # -- artifacts -----------------------------------------------------
    def write_tables(self) -> None:
        def dump(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
            path = self.out(name)
            fields: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
                writer.writeheader()
                writer.writerows(rows)

        dump("command_log.csv", self.command_rows)
        dump("profiles.csv", self.profile_rows)
        dump("traffic_summary.csv", self.traffic_rows)
        self.out("clock_anchors.json").write_text(
            json.dumps(self.clock_anchors, indent=2) + "\n")

    def manifest(self, status: str, extra: Mapping[str, Any]) -> None:
        files: list[dict[str, Any]] = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({
                    "relative_path": str(path.relative_to(self.output_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": n2.sha256(path),
                })
        self.out("manifest.json").write_text(json.dumps({
            "schema": self.config["schema"],
            "experiment_id": self.config["experiment_id"],
            "claim_boundary": self.config["claim_boundary"],
            "status": status,
            "utc": utc_now(),
            "config_sha256": n2.sha256(self.config_path),
            "traffic_design": self.config["traffic"],
            "replay_design": self.config["replay"],
            **dict(extra),
            "files": files,
        }, indent=2) + "\n")

    # -- driver --------------------------------------------------------
    def run(self) -> int:
        status = "UE_SNR_BRIDGE_FAILED"
        failure: str | None = None
        interrupted = {"flag": False}

        def terminate(signum: int, _frame: Any) -> None:
            interrupted["flag"] = True
            raise BridgeFailure(f"received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, terminate)

        try:
            self.preflight()
            self.take_clock_anchor("run_start")
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            self.wait_attach()
            self.start_telemetry()
            self.open_telnet()
            self.start_iperf_servers()
            self.start_live_pusch_feed()
            self.measure_upper_anchor()
            for profile_id in self.config["replay"]["profile_order"]:
                self.run_profile(profile_id)
            self.restore()
            self.take_clock_anchor("run_end")
            status = STATUS_OK
        except BridgeFailure as exc:
            failure = str(exc)
        except Exception as exc:  # noqa: BLE001 - evidence must be preserved
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            # Restore the registered cold channel even on failure, while the
            # gNB is still up and the telnet session is still usable.
            if not self.restored and self.telnet is not None:
                try:
                    self.restore()
                except Exception as exc:  # noqa: BLE001
                    self.teardown_notes.append(f"best-effort restore failed: {exc}")
            self.teardown_notes.extend(self.cleanup())
            try:
                self.extract_ttracer()
            except Exception as exc:  # noqa: BLE001
                self.teardown_notes.append(f"extraction failed: {exc}")
            self.write_tables()
            cold = self.verify_cold()
            self.manifest(status, {
                "failure": failure,
                "interrupted": interrupted["flag"],
                "teardown_notes": self.teardown_notes,
                "final_cold_state": cold,
                "restored_clean_channel": self.restored,
                "profiles": self.profile_rows,
                "traffic": self.traffic_rows,
            })
        print(json.dumps({
            "status": status, "failure": failure,
            "output_dir": str(self.output_dir),
            "restored": self.restored,
            "cold": cold.get("cold"),
        }, indent=2))
        return 0 if status == STATUS_OK else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output_dir or (
        ROOT / config["paths"]["output_root"]
        / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    return Runner(args.config, output).run()


if __name__ == "__main__":
    raise SystemExit(main())
