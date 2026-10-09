#!/usr/bin/env python3
"""Bounded 4x3 live calibration of the UE-local [previous UL MCS, backlog] state.

Network only: no CARLA, no CUDA, no FCOS, no perception model. The OAI CN5G
core is expected to be already running; this runner owns only the gNB/UE
softmodems, the tracer, the traffic, and the RF actuation.

The RAN is torn down and rebuilt from cold **between every cell**. That is the
only way to make "no cross-cell queue contamination" a structural fact rather
than a measured hope: an RLC queue cannot survive a softmodem restart. It also
makes each cell's clean-cell marker unambiguous.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rl_agent.ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent.ue_mcs_backlog_calibration_v1 import contract as C  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_v1.json"
STATUS_OK = "UE_MCS_BACKLOG_CALIBRATION_CAPTURED"


class RunFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunFailure(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def round_to_granularity(value: float, granularity: float) -> float:
    return round(round(float(value) / granularity) * granularity, 6)


def inverse_interpolate(target_db: float, anchors: Sequence[Mapping[str, float]]
                        ) -> tuple[float, bool]:
    ordered = sorted(anchors, key=lambda r: float(r["achieved_median_pusch_snr_db"]))
    snrs = [float(r["achieved_median_pusch_snr_db"]) for r in ordered]
    cmds = [float(r["noise_power_db"]) for r in ordered]
    if target_db <= snrs[0]:
        return cmds[0], target_db < snrs[0]
    if target_db >= snrs[-1]:
        return cmds[-1], target_db > snrs[-1]
    for i in range(len(ordered) - 1):
        if snrs[i] <= target_db <= snrs[i + 1]:
            span = snrs[i + 1] - snrs[i]
            frac = 0.0 if span == 0 else (target_db - snrs[i]) / span
            return cmds[i] + frac * (cmds[i + 1] - cmds[i]), False
    raise RunFailure(f"target {target_db} unmapped")


class Runner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path
        self.config = json.loads(config_path.read_text())
        self.output_dir = output_dir
        self.processes: list[n2.ManagedProcess] = []
        self.telnet: n2.TelnetSession | None = None
        self.live_pusch: n2.LiveCsv | None = None
        self.model_index: int | None = None
        self.ue_ip: str | None = None
        self.anchors = [dict(r) for r in self.config["actuator"]["existing_measured_anchors"]]
        self.edge_host: str | None = None
        self.edge_pid: int | None = None
        self.aborted = False
        self.notes: list[str] = []

    # -- helpers -------------------------------------------------------
    def path(self, rel: str) -> Path:
        return ROOT / rel

    def out(self, rel: str) -> Path:
        target = self.output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def spawn(self, name: str, argv: Sequence[str], log: str, *, cwd: Path = ROOT,
              root_owned: bool = False, env: Mapping[str, str] | None = None
              ) -> n2.ManagedProcess:
        handle = self.out(log).open("w", encoding="utf-8")
        process = subprocess.Popen(
            list(argv), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        managed = n2.ManagedProcess(name=name, process=process, log_handle=handle,
                                    root_owned=root_owned)
        self.processes.append(managed)
        return managed

    # -- preflight -----------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        radio = self.config["radio"]
        n2.run_checked(["sudo", "-n", "true"])
        containers: dict[str, str] = {}
        for name in radio["core_containers"]:
            state = n2.run_checked([
                "sudo", "-n", "docker", "inspect", "-f",
                "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                name]).stdout.strip()
            containers[name] = state
            require(state.startswith("true") and "unhealthy" not in state,
                    f"core container not ready: {name}={state!r}")

        self.assert_cold_ran("preflight")

        carla = subprocess.run(["pgrep", "-af", "CarlaUE4|CarlaUnreal"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        require(not carla.stdout.strip(), "a CARLA process is running; this is network-only")

        # The edge reassembly endpoint is the ext-DN *container*. It must not be
        # an address local to this host: ip rule 0 ("from all lookup local")
        # matches a local destination first and delivers it without ever
        # entering oaitun_ue1, which would silently bypass the radio.
        container = radio["edge_container"]
        edge_host = n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container]).stdout.strip()
        require(bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", edge_host)),
                f"could not resolve {container} address: {edge_host!r}")
        local_table = n2.run_checked(
            ["ip", "route", "show", "table", "local"]).stdout
        require(f"local {edge_host} " not in local_table,
                f"edge host {edge_host} is a host-local address; traffic to it "
                f"would never reach the radio")
        self.edge_host = edge_host
        self.edge_pid = int(n2.run_checked([
            "sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
            container]).stdout.strip())

        snapshot = {
            "utc": utc_now(),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
            "core_containers": containers,
            "edge_host": edge_host,
            "edge_container": container,
            "edge_container_pid": self.edge_pid,
            "edge_host_is_not_local": True,
            "carla_running": False,
            "cuda_untouched": True,
            "nvidia_smi_absent_or_unused": not Path("/proc/driver/nvidia").exists()
            or "no running processes" in subprocess.run(
                ["bash", "-lc", "nvidia-smi 2>/dev/null | tail -5 | tr A-Z a-z || true"],
                text=True, stdout=subprocess.PIPE).stdout,
        }
        self.out("preflight.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        return snapshot

    def assert_cold_ran(self, stage: str) -> None:
        """No softmodem, tracer, tunnel or sender of ours may already exist."""
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-a", "-x", name], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            require(not (found.returncode == 0 and found.stdout.strip()),
                    f"{stage}: cold-RAN gate failed: {found.stdout.strip()}")
        stale = n2.oai_tunnel_interfaces()
        require(not stale, f"{stage}: stale UE tunnel(s) {stale}")
        ports = [4043, self.config["actuator"]["telnet_port"],
                 self.config["telemetry"]["gnb_port"], self.config["telemetry"]["ue_port"],
                 self.config["telemetry"]["gnb_relay_port"],
                 self.config["telemetry"]["ue_relay_port"]]
        busy = [p for p in ports if not n2.port_is_free(int(p))]
        require(not busy, f"{stage}: ports not free: {busy}")

    # -- RAN lifecycle -------------------------------------------------
    def materialize_configs(self, cell_dir: Path) -> tuple[Path, Path]:
        paths, radio = self.config["paths"], self.config["radio"]
        conf = self.path(paths["oai_ran_conf"])
        gnb_base = (conf / paths["gnb_base_config"]).read_text(encoding="utf-8")
        ue_base = (conf / paths["ue_base_config"]).read_text(encoding="utf-8")
        channel = (conf / paths["channel_config"]).read_text(encoding="utf-8")
        require(len(re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue_base)) == 1,
                "effective UE config is not single-UE")
        clean = self.config["actuator"]["clean_and_restore_commanded_noise_power_db"]
        channel, n = re.subn(r"noise_power_dB\s*=\s*[-+0-9.eE]+;",
                             f"noise_power_dB = {clean};", channel)
        require(n == 3, f"expected three channel noise values, found {n}")
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        require(marker in ue_base, "UE base config lacks expected channel include")
        gnb_path = cell_dir / "runtime/effective_gnb.conf"
        ue_path = cell_dir / "runtime/effective_ue.conf"
        gnb_path.parent.mkdir(parents=True, exist_ok=True)
        gnb_path.write_text(gnb_base + "\n\n" + channel + "\n")
        ue_path.write_text(ue_base.replace(marker, channel))
        return gnb_path, ue_path

    def start_ran(self, gnb_config: Path, ue_config: Path, cell_tag: str) -> list[str]:
        radio, tel, act = (self.config["radio"], self.config["telemetry"],
                           self.config["actuator"])
        build = self.path(self.config["paths"]["oai_ran_build"])
        gnb_argv = ["sudo", "-n", "env", "-u", "SCENESENSE_FORCE_UL_MCS",
                    f"SCENESENSE_MCS_POLICY={radio['mcs_policy']}", "./nr-softmodem",
                    "-O", str(gnb_config), "--gNBs.[0].min_rxtxtime", "6", "--rfsim",
                    "--rfsimulator.[0].options", "chanmod", "--telnetsrv",
                    "--telnetsrv.listenaddr", act["telnet_host"],
                    "--telnetsrv.listenport", str(act["telnet_port"]),
                    "--T_stdout", "2", "--T_nowait", "--T_port", str(tel["gnb_port"])]
        self.spawn("gnb", gnb_argv, f"cells/{cell_tag}/logs/gnb.log", cwd=build,
                   root_owned=True)
        time.sleep(float(radio["gnb_start_lead_s"]))
        ue_argv = ["sudo", "-n", "./nr-uesoftmodem", "--rfsim",
                   "--rfsimulator.[0].serveraddr", "127.0.0.1",
                   "--rfsimulator.[0].options", "chanmod", "-r", str(radio["prb"]),
                   "--numerology", str(radio["numerology"]), "--band", str(radio["band"]),
                   "-C", str(radio["downlink_frequency_hz"]), "-O", str(ue_config),
                   "--T_stdout", "2", "--T_nowait", "--T_port", str(tel["ue_port"])]
        self.spawn("ue", ue_argv, f"cells/{cell_tag}/logs/ue.log", cwd=build,
                   root_owned=True)
        return gnb_argv

    def wait_attach(self, cell_tag: str) -> None:
        radio = self.config["radio"]
        deadline = time.monotonic() + float(radio["attach_timeout_s"])
        iface = radio["ue_interface"]
        while time.monotonic() < deadline:
            require(all(p.process.poll() is None for p in self.processes[:2]),
                    "gNB or UE exited before attachment")
            res = subprocess.run(["ip", "-j", "-4", "addr", "show", "dev", iface],
                                 text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
            ips: list[str] = []
            if res.returncode == 0:
                try:
                    ips = [str(i["local"]) for r in json.loads(res.stdout)
                           for i in r.get("addr_info", [])
                           if i.get("family") == "inet" and i.get("local")]
                except (json.JSONDecodeError, KeyError, TypeError):
                    ips = []
            if len(ips) == 1:
                # Ping the ext-DN, not the edge host. The host's route to
                # 10.0.0.0/16 points at the corporate LAN, so an ICMP reply
                # from the edge host is misrouted even though the inbound data
                # path is fine. ext-DN is the one CN-side address with a
                # working return route, so it is what proves the PDU session.
                ping = subprocess.run(
                    ["ping", "-I", iface, "-c", "3", "-W", "2",
                     radio["attach_probe_ip"]],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if ping.returncode == 0:
                    self.ue_ip = ips[0]
                    self.out(f"cells/{cell_tag}/logs/attach_ping.log").write_text(ping.stdout)
                    return
            time.sleep(1)
        raise RunFailure(
            f"UE did not attach and reach {radio['attach_probe_ip']} before timeout")

    def verify_radio_path(self, cell_dir: Path) -> dict[str, Any]:
        """Prove the traffic will actually traverse the radio.

        Two independent checks, because "a datagram arrived" is not evidence
        that it went over the air. A host-local destination, or a destination
        whose route does not resolve through ``oaitun_ue1``, is delivered by
        the kernel without ever entering the UE stack. That failure is silent
        and would make every cell measure loopback instead of the radio, so it
        is checked before any traffic is trusted.
        """
        radio = self.config["radio"]
        assert self.ue_ip is not None and self.edge_host is not None
        iface = radio["ue_interface"]

        # `iif` turns this into a forwarding lookup and is rejected here, so the
        # source-selected form is used instead.
        route = subprocess.run(
            ["ip", "route", "get", self.edge_host, "from", self.ue_ip],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
        via_tunnel = f"dev {iface}" in route

        # Independent confirmation that does not rely on `ip route get`: the UE
        # policy rule must exist and its table must route through the tunnel.
        rules = subprocess.run(["ip", "rule", "show"], text=True,
                               stdout=subprocess.PIPE).stdout
        table = None
        for line in rules.splitlines():
            parts = line.split()
            if f"from" in parts and self.ue_ip in parts and "lookup" in parts:
                table = parts[parts.index("lookup") + 1]
                break
        table_routes = ""
        if table is not None:
            table_routes = subprocess.run(
                ["ip", "route", "show", "table", table], text=True,
                stdout=subprocess.PIPE).stdout
        rule_via_tunnel = iface in table_routes

        local_table = n2.run_checked(
            ["ip", "route", "show", "table", "local"]).stdout
        is_local = f"local {self.edge_host} " in local_table

        outcome = {
            "edge_host": self.edge_host, "ue_ip": self.ue_ip,
            "route_lookup": route, "routes_via_ue_tunnel": via_tunnel,
            "policy_rule_table": table,
            "policy_table_routes_via_tunnel": rule_via_tunnel,
            "policy_table_routes": table_routes.strip().splitlines(),
            "edge_host_is_host_local": is_local,
            "radio_path_verified": (via_tunnel or rule_via_tunnel) and not is_local,
        }
        (cell_dir / "radio_path_check.json").write_text(
            json.dumps(outcome, indent=2) + "\n")
        require(not is_local,
                f"edge host {self.edge_host} is host-local; traffic would bypass "
                f"the radio")
        require(via_tunnel or rule_via_tunnel,
                f"neither the route lookup nor the UE policy table sends "
                f"{self.ue_ip} -> {self.edge_host} through {iface}; traffic would "
                f"bypass the radio. Route: {route!r}; table {table}: "
                f"{table_routes.strip()!r}")
        return outcome

    def start_telemetry(self, cell_tag: str) -> None:
        tel = self.config["telemetry"]
        troot = self.path(self.config["paths"]["t_tracer_dir"])
        msgs = self.path(self.config["paths"]["t_messages"])
        for src, port, relay in (("gnb", tel["gnb_port"], tel["gnb_relay_port"]),
                                 ("ue", tel["ue_port"], tel["ue_relay_port"])):
            self.spawn(f"{src}_relay",
                       [str(troot / "multi"), "-d", str(msgs), "-ip", "127.0.0.1",
                        "-p", str(port), "-lp", str(relay)],
                       f"cells/{cell_tag}/logs/{src}_relay.log")
        n2.wait_tcp(int(tel["gnb_relay_port"]), 15)
        n2.wait_tcp(int(tel["ue_relay_port"]), 15)
        for src, relay in (("gnb", tel["gnb_relay_port"]), ("ue", tel["ue_relay_port"])):
            raw = self.out(f"cells/{cell_tag}/ttracer/{src}/{src}.raw")
            argv = [str(troot / "record"), "-d", str(msgs), "-o", str(raw), "-OFF"]
            for event in tel["events"][src]:
                argv += ["-on", event]
            argv += ["-ip", "127.0.0.1", "-p", str(relay)]
            self.spawn(f"{src}_record", argv, f"cells/{cell_tag}/logs/{src}_record.log")
        time.sleep(2.0)
        for src in ("gnb", "ue"):
            log = (self.output_dir / f"cells/{cell_tag}/logs/{src}_record.log").read_text()
            require("ERROR" not in log, f"{src} tracer refused its event set: {log[-500:]}")

    def open_telnet(self, cell_dir: Path) -> int:
        act = self.config["actuator"]
        self.telnet = n2.TelnetSession(act["telnet_host"], int(act["telnet_port"]),
                                       float(act["response_timeout_s"]),
                                       int(act["max_response_bytes"]))
        _, _, _, _, state = self.telnet.command("channelmod show current")
        (cell_dir / "channel_state_initial.txt").write_text(state)
        models = n2.parse_channel_models(state)
        name = act["channel_model_name"]
        require(name in models, f"channel model {name!r} absent: {sorted(models)}")
        self.model_index = int(models[name]["model_index"])
        return self.model_index

    def send_noise(self, command_db: float, *, reason: str, log: list[dict[str, Any]],
                   **extra: Any) -> None:
        assert self.telnet is not None and self.model_index is not None
        send_m, send_w, ack_m, ack_w, response = self.telnet.command(
            f"channelmod modify {self.model_index} noise_power_dB {command_db:g}")
        log.append({"reason": reason, "commanded_noise_power_db": float(command_db),
                    "send_monotonic_ns": send_m, "ack_monotonic_ns": ack_m,
                    "send_wall_ns": send_w, "ack_wall_ns": ack_w,
                    "ack_latency_ms": (ack_m - send_m) / 1e6,
                    "status": "ACK" if "ERROR" not in response.upper() else "ERROR",
                    **extra})

    def read_back_noise(self) -> float:
        assert self.telnet is not None
        _, _, _, _, state = self.telnet.command("channelmod show current")
        models = n2.parse_channel_models(state)
        row = models.get(self.config["actuator"]["channel_model_name"], {})
        return float(row.get("noise_power_db", math.nan))

    def restore_clean(self, cell_dir: Path, log: list[dict[str, Any]]) -> bool:
        clean = float(self.config["actuator"]["clean_and_restore_commanded_noise_power_db"])
        self.send_noise(clean, reason="RESTORE_CLEAN", log=log)
        observed = self.read_back_noise()
        (cell_dir / "restore_readback.json").write_text(json.dumps({
            "commanded_clean_noise_power_db": clean,
            "read_back_noise_power_db": observed,
            "verified": abs(observed - clean) <= 1e-6, "utc": utc_now(),
        }, indent=2) + "\n")
        return abs(observed - clean) <= 1e-6

    def teardown_ran(self) -> list[str]:
        notes: list[str] = []
        if self.live_pusch is not None:
            try:
                self.live_pusch.stop()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"live pusch: {exc}")
            self.live_pusch = None
        if self.telnet is not None:
            try:
                self.telnet.close()
            except OSError as exc:
                notes.append(f"telnet: {exc}")
            self.telnet = None
        for managed in reversed(self.processes):
            try:
                managed.stop()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{managed.name}: {exc}")
        self.processes.clear()
        self.model_index = None
        self.ue_ip = None
        return notes

    # -- calibration ---------------------------------------------------
    def start_live_pusch(self, cell_tag: str) -> None:
        tel = self.config["telemetry"]
        troot = self.path(self.config["paths"]["t_tracer_dir"])
        msgs = self.path(self.config["paths"]["t_messages"])
        fields = ("time", "rnti", "frame", "slot", "snrx10", "phr", "tpc",
                  "tb_size", "txpower_calc", "rbSize", "mcs", "rssi")
        self.live_pusch = n2.LiveCsv(
            [str(troot / "csv"), "-d", str(msgs), "-ip", "127.0.0.1",
             "-p", str(tel["gnb_relay_port"]), "-f", "-s", ",", "-t", "time",
             "GNB_MAC_PUSCH_POWER_CONTROL", *fields],
            self.out(f"cells/{cell_tag}/ttracer/gnb/live_pusch.csv"))

    def live_pusch_snr_db(self, since: int) -> list[float]:
        out: list[float] = []
        assert self.live_pusch is not None
        for _, _, line in self.live_pusch.snapshot()[since:]:
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    out.append(int(parts[4]) / 10.0)
                except ValueError:
                    continue
        return out

    def calibrate_upper_anchor(self, cell_dir: Path, log: list[dict[str, Any]]
                               ) -> dict[str, Any]:
        """Close the registered mapping at its upper end, once per campaign.

        The eight registered anchors stop at 19.5 dB achieved, below the
        FAVORABLE/MID/FADE targets, so replaying on them alone would clamp most
        of those profiles. The upper anchor is measured, never extrapolated.
        """
        upper = self.config["actuator"]["upper_anchor"]
        command = float(upper["first_command_db"])
        self.send_noise(command, reason="UPPER_ANCHOR_CALIBRATION", log=log)
        tier = C.resolve_load_tiers(ROOT)[0]      # low tier: enough PUSCH, no flood
        frames = int(upper["calibration_frames"])
        block = {"block_index": 0, "tier": tier.tier, "action_id": tier.action_id,
                 "payload_bytes": tier.payload_bytes,
                 "chunks_per_frame": tier.chunks_per_frame, "frames": frames,
                 "first_frame_index": 0,
                 "port": int(self.config["traffic"]["ports"][tier.tier])}
        sessions = self.launch_traffic(
            cell_tag="calibration", cell_id="calibration", blocks=[block],
            cell_dir=cell_dir, total_frames=frames)
        time.sleep(1.5)
        baseline = self.live_pusch.count() if self.live_pusch else 0
        time.sleep(float(upper["measurement_duration_s"]))
        samples = self.live_pusch_snr_db(baseline)
        self.finish_traffic(sessions)
        resolved = len(samples) >= int(upper["minimum_pusch_samples"])
        median = sorted(samples)[len(samples) // 2] if samples else None
        outcome = {"commanded_noise_power_db": command,
                   "pusch_sample_count": len(samples),
                   "achieved_median_pusch_snr_db": median,
                   "resolved": bool(resolved),
                   "snrx10_to_db": "gNB reports snrx10; dB = snrx10 / 10"}
        if resolved and median is not None:
            self.anchors.append({"noise_power_db": command,
                                 "achieved_median_pusch_snr_db": float(median)})
            outcome["mapping_upper_bound_db"] = float(median)
        else:
            outcome["note"] = ("upper anchor unresolved; targets above the "
                               "registered 19.5 dB anchor are clamped and counted")
        outcome["anchors_in_use"] = list(self.anchors)
        (cell_dir / "upper_anchor.json").write_text(json.dumps(outcome, indent=2) + "\n")
        self.restore_clean(cell_dir, log)
        return outcome

    # -- traffic -------------------------------------------------------
    def launch_traffic(self, *, cell_tag: str, cell_id: str,
                       blocks: Sequence[Mapping[str, Any]], cell_dir: Path,
                       total_frames: int) -> dict[str, Any]:
        """Start one receiver per block, then one continuous sender.

        Every receiver is up *before* the first datagram, so a block transition
        never waits on process startup. The production receiver is instantiated
        with a single ``expected_chunks_per_frame``, which is the only reason
        each block needs its own port.
        """
        traffic = self.config["traffic"]
        assert self.ue_ip is not None and self.edge_host is not None
        duration = total_frames / C.FPS + float(traffic["receiver_tail_s"])

        receivers: list[dict[str, Any]] = []
        for block in blocks:
            tag = f"block{block['block_index']}_{block['tier']}"
            events = cell_dir / f"receiver_{tag}_events.jsonl"
            summary = cell_dir / f"receiver_{tag}_summary.json"
            ready = cell_dir / f"receiver_{tag}_ready.json"
            process = self.spawn(
                f"receiver_{cell_tag}_{tag}",
                ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
                 sys.executable, str(self.path(C.PRODUCTION_RECEIVER_RELPATH)),
                 "--bind-host", "0.0.0.0", "--port", str(block["port"]),
                 "--events-jsonl", str(events), "--summary-json", str(summary),
                 "--ready-json", str(ready), "--duration-s", f"{duration:.3f}",
                 "--expected-frames", str(block["frames"]),
                 "--expected-chunks-per-frame", str(block["chunks_per_frame"]),
                 "--max-chunks-per-frame",
                 str(max(16, int(block["chunks_per_frame"]) * 2)),
                 "--socket-receive-buffer-bytes",
                 str(traffic["receive_buffer_bytes"])],
                f"cells/{cell_tag}/logs/receiver_{tag}.log", root_owned=True)
            receivers.append({"process": process, "ready": ready,
                              "block_index": int(block["block_index"]),
                              "tier": block["tier"], "events": events,
                              "summary": summary})

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if all(item["ready"].is_file() for item in receivers):
                break
            for item in receivers:
                require(item["process"].process.poll() is None,
                        f"receiver {item['tier']} exited before READY")
            time.sleep(0.2)
        require(all(item["ready"].is_file() for item in receivers),
                "not every block receiver reported READY")

        plan_path = cell_dir / "block_plan.json"
        plan_path.write_text(json.dumps(list(blocks), indent=2) + "\n")
        sender_csv = cell_dir / "sender_decisions.csv"
        sender_summary = cell_dir / "sender_summary.json"
        sender = self.spawn(
            f"sender_{cell_tag}",
            [sys.executable, "-m",
             "rl_agent.ue_mcs_backlog_calibration_v1.tagged_sender",
             "--cell-id", cell_id, "--bind-host", self.ue_ip,
             "--remote-host", self.edge_host,
             "--block-plan", str(plan_path),
             "--payload-seed", str(self.config["campaign"]["payload_seed"]),
             "--chunk-bytes", str(C.CHUNK_BYTES), "--fps", str(C.FPS),
             "--socket-sendbuf", str(traffic["send_buffer_bytes"]),
             "--log-csv", str(sender_csv), "--summary-json", str(sender_summary)],
            f"cells/{cell_tag}/logs/sender.log")
        return {"receivers": receivers, "sender": sender,
                "frames": total_frames, "sender_csv": sender_csv}

    def finish_traffic(self, sessions: Mapping[str, Any]) -> None:
        budget = sessions["frames"] / C.FPS + 90.0
        try:
            sessions["sender"].process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            self.notes.append("sender overran its budget")
        for item in sessions["receivers"]:
            try:
                item["process"].process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.notes.append(f"receiver {item['tier']} overran its budget")

    # -- one cell ------------------------------------------------------
    def run_cell(self, cell: C.Cell, profile: C.NetworkProfile) -> dict[str, Any]:
        cell_tag = f"{cell.run_index:02d}__{cell.cell_id}"
        cell_dir = self.output_dir / "cells" / cell_tag
        cell_dir.mkdir(parents=True, exist_ok=True)
        command_log: list[dict[str, Any]] = []
        record: dict[str, Any] = {
            "cell_tag": cell_tag, **cell.to_json(),
            "trace_id": profile.trace_id,
            "registered_trace_sha256": profile.trace_sha256,
            "started_utc": utc_now(), "status": "FAILED",
        }
        try:
            self.assert_cold_ran(f"cell {cell_tag}")
            gnb_config, ue_config = self.materialize_configs(cell_dir)
            gnb_argv = self.start_ran(gnb_config, ue_config, cell_tag)
            record["gnb_argv"] = " ".join(gnb_argv)
            record["mcs_policy_env"] = self.config["radio"]["mcs_policy"]
            self.wait_attach(cell_tag)
            record["ue_ip"] = self.ue_ip
            record["radio_path_check"] = self.verify_radio_path(cell_dir)
            self.start_telemetry(cell_tag)
            self.open_telnet(cell_dir)

            marker = {"clean_cell_marker_utc": utc_now(),
                      "clean_cell_monotonic_ns": time.monotonic_ns(),
                      "ran_rebuilt_from_cold": True,
                      "queue_empty_by_construction": True,
                      "initial_noise_read_back_db": self.read_back_noise()}
            (cell_dir / "clean_cell_marker.json").write_text(
                json.dumps(marker, indent=2) + "\n")
            record["clean_cell_marker"] = marker

            gran = float(self.config["actuator"]["command_granularity_db"])
            first_cmd, clamped0 = inverse_interpolate(
                profile.samples[0]["target_snr_db"], self.anchors)
            first_cmd = round_to_granularity(first_cmd, gran)
            self.send_noise(first_cmd, reason="PROFILE_PRIME", log=command_log,
                            profile_id=profile.profile_id, step_index=0,
                            target_snr_db=profile.samples[0]["target_snr_db"],
                            clamped=clamped0)
            observed = self.read_back_noise()
            record["profile_prime_command_db"] = first_cmd
            record["profile_prime_read_back_db"] = observed
            record["profile_read_back_ok"] = abs(observed - first_cmd) <= 1e-6
            require(record["profile_read_back_ok"],
                    f"profile read-back mismatch: {observed} != {first_cmd}")

            time.sleep(float(self.config["campaign"]["warmup_s"]))

            blocks = [block.to_json() for block in cell.blocks]
            sessions = self.launch_traffic(
                cell_tag=cell_tag, cell_id=cell.cell_id, blocks=blocks,
                cell_dir=cell_dir, total_frames=C.FRAMES_PER_CELL)

            start = time.monotonic_ns()
            period_ns = int(float(self.config["campaign"]["sample_period_s"]) * 1e9)
            last_cmd: float | None = first_cmd
            sent = skipped = clamped = 0
            for sample in profile.samples:
                due = start + sample["step_index"] * period_ns
                now = time.monotonic_ns()
                if now > due + period_ns:
                    skipped += 1
                    continue
                if now < due:
                    time.sleep((due - now) / 1e9)
                command, was_clamped = inverse_interpolate(
                    sample["target_snr_db"], self.anchors)
                command = round_to_granularity(command, gran)
                clamped += int(was_clamped)
                if command != last_cmd:
                    self.send_noise(command, reason="PROFILE_REPLAY", log=command_log,
                                    profile_id=profile.profile_id,
                                    step_index=sample["step_index"],
                                    target_snr_db=sample["target_snr_db"],
                                    clamped=was_clamped)
                    last_cmd = command
                    sent += 1
            record.update({"commands_sent": sent, "commands_skipped": skipped,
                           "targets_clamped": clamped})

            self.finish_traffic(sessions)
            record["restored"] = self.restore_clean(cell_dir, command_log)
            record["status"] = "CAPTURED"
        except Exception as exc:  # noqa: BLE001 - evidence must be preserved
            record["failure"] = f"{type(exc).__name__}: {exc}"
            if self.telnet is not None:
                try:
                    record["restored"] = self.restore_clean(cell_dir, command_log)
                except Exception as inner:  # noqa: BLE001
                    self.notes.append(f"{cell_tag} restore failed: {inner}")
        finally:
            record["teardown_notes"] = self.teardown_ran()
            (cell_dir / "command_log.json").write_text(
                json.dumps(command_log, indent=2) + "\n")
            try:
                self.extract_ttracer(cell_tag, cell_dir)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"{cell_tag} extraction failed: {exc}")
            record["finished_utc"] = utc_now()
            (cell_dir / "cell_record.json").write_text(
                json.dumps(record, indent=2) + "\n")
        return record

    def extract_ttracer(self, cell_tag: str, cell_dir: Path) -> None:
        script = self.path(self.config["paths"]["extract_script"])
        for src in ("gnb", "ue"):
            raw = cell_dir / "ttracer" / src / f"{src}.raw"
            if not raw.exists() or raw.stat().st_size == 0:
                self.notes.append(f"{cell_tag}: empty raw for {src}")
                continue
            argv = [str(script), "--raw", str(raw), "--source", src,
                    "--output-root", str(cell_dir), "--clean-output"]
            for event in self.config["telemetry"]["events"][src]:
                argv += ["--event", event]
            n2.run_checked(argv, timeout=900)

    # -- driver --------------------------------------------------------
    def run(self) -> int:
        status = "UE_MCS_BACKLOG_CALIBRATION_FAILED"
        failure: str | None = None

        def terminate(signum: int, _frame: Any) -> None:
            # Mark the campaign aborted so the cell loop stops instead of
            # absorbing the signal into one cell's failure handler.
            self.aborted = True
            raise RunFailure(f"received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, terminate)

        tiers = {tier.tier: tier for tier in C.resolve_load_tiers(ROOT)}
        profiles = {p.profile_id: p for p in
                    C.resolve_profiles(ROOT, C.FRAMES_PER_CELL)}
        plan = C.build_cell_plan(
            list(tiers.values()), ports=self.config["traffic"]["ports"],
            seed=int(self.config["campaign"]["cell_order_seed"]))
        plan_audit = C.audit_cell_plan(plan)
        cell_records: list[dict[str, Any]] = []
        calibration: dict[str, Any] = {}

        try:
            self.preflight()
            require(plan_audit["cells"] == plan_audit["expected_cells"]
                    and plan_audit["load_is_within_cell"]
                    and plan_audit["repetition_1_reverses_repetition_0"]
                    and all(plan_audit["position_balanced_per_channel"].values())
                    and all(plan_audit[
                        "all_six_transitions_balanced_per_channel"].values()),
                    f"cell plan is not the registered balanced design: {plan_audit}")
            self.out("plan.json").write_text(json.dumps({
                "design": {
                    "block_orders": [list(o) for o in C.BLOCK_ORDERS],
                    "contrast_profiles": list(C.CONTRAST_PROFILE_IDS),
                    "repetitions": C.REPETITIONS,
                    "frames_per_block": C.FRAMES_PER_BLOCK,
                    "frames_per_cell": C.FRAMES_PER_CELL,
                    "transient_decisions": C.TRANSIENT_DECISIONS,
                    "steady_state_decisions": C.STEADY_STATE_DECISIONS,
                },
                "plan_audit": plan_audit,
                "cells": [c.to_json() for c in plan],
                "tiers": [t.to_json() for t in tiers.values()],
                "profiles": [p.to_json() for p in profiles.values()
                             if p.profile_id in C.CONTRAST_PROFILE_IDS],
                "cell_order_seed": self.config["campaign"]["cell_order_seed"],
                "source_hashes": C.resolved_source_hashes(ROOT),
                "oai_citations": dict(C.OAI_CITATIONS),
            }, indent=2) + "\n")

            # One calibration RAN cycle: the mapping is a property of the radio
            # configuration, identical across cells, so it is measured once.
            cal_dir = self.output_dir / "calibration"
            cal_dir.mkdir(parents=True, exist_ok=True)
            cal_log: list[dict[str, Any]] = []
            self.assert_cold_ran("calibration")
            gnb_cfg, ue_cfg = self.materialize_configs(cal_dir)
            self.start_ran(gnb_cfg, ue_cfg, "calibration")
            self.wait_attach("calibration")
            self.verify_radio_path(cal_dir)
            self.start_telemetry("calibration")
            self.open_telnet(cal_dir)
            self.start_live_pusch("calibration")
            calibration = self.calibrate_upper_anchor(cal_dir, cal_log)
            (cal_dir / "command_log.json").write_text(
                json.dumps(cal_log, indent=2) + "\n")
            self.teardown_ran()

            for cell in plan:
                if self.aborted:
                    self.notes.append("campaign aborted before "
                                      f"{cell.cell_id}")
                    break
                cell_records.append(self.run_cell(cell, profiles[cell.profile_id]))
            require(not self.aborted, "campaign was aborted by signal")

            status = (STATUS_OK
                      if all(r["status"] == "CAPTURED" for r in cell_records)
                      else "UE_MCS_BACKLOG_CALIBRATION_PARTIAL")
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.notes.extend(self.teardown_ran())
            cold = self.final_cold_state()
            self.manifest(status, {
                "failure": failure, "notes": self.notes,
                "calibration": calibration, "plan_audit": plan_audit,
                "cells": cell_records, "final_cold_state": cold,
            })
        print(json.dumps({"status": status, "failure": failure,
                          "cells_captured": sum(1 for r in cell_records
                                                if r["status"] == "CAPTURED"),
                          "cells_planned": len(plan),
                          "output_dir": str(self.output_dir),
                          "cold": cold.get("cold")}, indent=2))
        return 0 if status == STATUS_OK else 1

    def final_cold_state(self) -> dict[str, Any]:
        orphans: dict[str, list[str]] = {}
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = subprocess.run(["sudo", "-n", "pgrep", "-a", "-x", name], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if found.returncode == 0 and found.stdout.strip():
                orphans[name] = found.stdout.strip().splitlines()
        for label, pattern in (("tracer", "T/tracer/(record|multi|csv|replay)"),
                               ("sender", "ue_mcs_backlog_calibration_v1.tagged_sender"),
                               ("receiver", "ue_n3_structured_udp_receiver")):
            found = subprocess.run(["pgrep", "-af", pattern], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            rows = [r for r in found.stdout.splitlines() if "pgrep" not in r]
            if rows:
                orphans[label] = rows
        tunnels = n2.oai_tunnel_interfaces()
        carla = subprocess.run(["pgrep", "-af", "CarlaUE4|CarlaUnreal"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        state = {"utc": utc_now(),
                 "loadavg": Path("/proc/loadavg").read_text().strip(),
                 "orphan_processes": orphans, "residual_ue_tunnels": tunnels,
                 "carla_running": bool(carla.stdout.strip()),
                 "cold": not orphans and not tunnels and not carla.stdout.strip()}
        self.out("final_cold_state.json").write_text(json.dumps(state, indent=2) + "\n")
        return state

    def manifest(self, status: str, extra: Mapping[str, Any]) -> None:
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({"relative_path": str(path.relative_to(self.output_dir)),
                              "size_bytes": path.stat().st_size,
                              "sha256": n2.sha256(path)})
        self.out("manifest.json").write_text(json.dumps({
            "schema": self.config["schema"], "contract_id": C.CONTRACT_ID,
            "status": status, "utc": utc_now(),
            "config_sha256": n2.sha256(self.config_path),
            "source_hashes": C.resolved_source_hashes(ROOT),
            **dict(extra), "files": files,
        }, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output_dir or (ROOT / config["paths"]["output_root"]
                                 / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    return Runner(args.config, output).run()


if __name__ == "__main__":
    raise SystemExit(main())
