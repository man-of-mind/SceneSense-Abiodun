#!/usr/bin/env python3
"""Fail-fast, self-logging orchestrator for DG-A and DG-A.1 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .endpoint import (
    CONTROLLERS,
    SEND_KINDS,
    build_parser as build_endpoint_parser,
    build_ttracer_csv_command,
    frame_onwire_bytes,
    raw_ns,
    validate_send_args,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dg_a_v1.yaml"


class StageFailure(RuntimeError):
    """A fail-fast gate stopped the stage."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def failure_record(exc: BaseException, *, status: str) -> dict:
    """Return the common fail-closed sentinel/summary payload."""
    return {
        "schema_version": "scenesense.multiue_oai.dg_a.failure.v1",
        "status": status,
        "decision": "HOLD_REPAIR",
        "failed_at": utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "next_stage_launched": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    return float(statistics.median(clean))


def parse_trace_time(value: str) -> float:
    parsed = datetime.strptime(value.strip(), "%H:%M:%S.%f")
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1e6


def parse_interface_ipv4(payload: str, expected_iface: str) -> Optional[str]:
    """Extract the sole IPv4 address from `ip -j -4 addr show dev IFACE`."""
    rows = json.loads(payload or "[]")
    addresses = {
        str(address["local"])
        for row in rows
        if str(row.get("ifname", "")) == expected_iface
        for address in row.get("addr_info", [])
        if address.get("family") == "inet" and address.get("local")
    }
    if len(addresses) > 1:
        raise ValueError(f"{expected_iface} has multiple IPv4 addresses: {sorted(addresses)}")
    return next(iter(addresses), None)


def parse_uicc_profiles(payload: str) -> Dict[int, dict]:
    """Parse the identity-bearing fields from libconfig ``uiccN`` blocks."""
    profiles: Dict[int, dict] = {}
    current: Optional[int] = None
    depth = 0
    for line in payload.splitlines():
        start = re.match(r"\s*uicc(\d+)\s*=\s*\{", line)
        if start:
            current = int(start.group(1))
            if current in profiles:
                raise ValueError(f"duplicate uicc{current} block")
            profiles[current] = {"imsi": None, "dnns": []}
            depth = line.count("{") - line.count("}")
            continue
        if current is None:
            continue
        imsi = re.search(r'\bimsi\s*=\s*"([0-9]+)"', line)
        if imsi:
            if profiles[current]["imsi"] is not None:
                raise ValueError(f"uicc{current} has multiple IMSI declarations")
            profiles[current]["imsi"] = imsi.group(1)
        profiles[current]["dnns"].extend(
            match.group(1) for match in re.finditer(r'\bdnn\s*=\s*"([^"]+)"', line)
        )
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            profiles[current]["dnns"] = sorted(set(profiles[current]["dnns"]))
            current = None
            depth = 0
    if current is not None:
        raise ValueError(f"unterminated uicc{current} block")
    return profiles


def parse_runtime_uicc_imsis(payload: str) -> List[str]:
    """Return the softmodem's UICC identities in internal-UE initialization order."""
    return re.findall(r"UICC simulation:\s*IMSI=([0-9]+)", payload)


def ue_network_contract_report(
    mapping: Mapping[int, Mapping[str, object]],
    expected_tunnels: Iterable[Mapping[str, object]],
    expected_subnet: str,
) -> dict:
    """Validate fixed tunnel identities and dynamic DNN-subnet IPv4 leases."""
    reasons: List[str] = []
    try:
        subnet = ipaddress.ip_network(str(expected_subnet), strict=True)
    except ValueError as exc:
        return {"pass": False, "expected_subnet": str(expected_subnet), "reasons": [str(exc)]}
    if subnet.version != 4:
        reasons.append(f"expected UE subnet is not IPv4: {subnet}")

    expected = {
        int(row["ue_id"]): str(row["iface"])
        for row in expected_tunnels
    }
    if set(mapping) != set(expected):
        reasons.append(
            f"UE IDs differ: expected={sorted(expected)}, actual={sorted(mapping)}"
        )

    seen_ips: Dict[str, int] = {}
    resolved = []
    for ue_id, iface in sorted(expected.items()):
        row = mapping.get(ue_id)
        if row is None:
            continue
        actual_iface = str(row.get("iface", ""))
        raw_ip = str(row.get("ip", ""))
        if actual_iface != iface:
            reasons.append(
                f"UE{ue_id} tunnel differs: expected={iface}, actual={actual_iface}"
            )
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            reasons.append(f"UE{ue_id} has invalid IPv4 address: {raw_ip!r}")
            continue
        if address.version != 4 or address not in subnet:
            reasons.append(f"UE{ue_id} address {address} is outside {subnet}")
        elif address in {subnet.network_address, subnet.broadcast_address}:
            reasons.append(f"UE{ue_id} address {address} is not a usable host in {subnet}")
        if raw_ip in seen_ips:
            reasons.append(f"UE{ue_id} duplicates UE{seen_ips[raw_ip]} address {raw_ip}")
        else:
            seen_ips[raw_ip] = ue_id
        resolved.append({"ue_id": ue_id, "iface": actual_iface, "ip": raw_ip})

    return {
        "pass": not reasons,
        "expected_subnet": str(subnet),
        "identity_basis": "expected_imsi_to_internal_ue_id_to_fixed_tunnel_to_dynamic_dnn_lease",
        "resolved": resolved,
        "reasons": reasons,
    }


def receiver_identity_report(
    rows: Iterable[Mapping[str, object]],
    expected_ues: Iterable[int],
    expected_nat_sources: Iterable[str],
) -> dict:
    """Validate logical UE identity after the UPF has source-NATed UE packets.

    The ext-DN cannot recover the originating tunnel from the post-NAT source
    address.  Logical identity is therefore validated from the message-ID UE
    prefix, while the source address is checked against the registered UPF N6
    address rather than against a UE tunnel address.
    """
    expected_ue_set = {int(value) for value in expected_ues}
    expected_nat_set = {str(value) for value in expected_nat_sources}
    observed: Dict[int, set[str]] = defaultdict(set)
    invalid_message_ids = []
    unexpected_nat_sources = set()
    for row in rows:
        ue_id = int(row["ue_id"])
        source_ip = str(row["source_ip"])
        observed[ue_id].add(source_ip)
        if source_ip not in expected_nat_set:
            unexpected_nat_sources.add(source_ip)
        try:
            message_ue_id = (int(row["message_id"]) >> 28) - 1
        except (KeyError, TypeError, ValueError):
            message_ue_id = None
        if message_ue_id != ue_id:
            invalid_message_ids.append(
                {
                    "ue_id": ue_id,
                    "source_ip": source_ip,
                    "message_id": row.get("message_id"),
                    "message_ue_id": message_ue_id,
                }
            )
    missing_ues = sorted(expected_ue_set - set(observed))
    unexpected_ues = sorted(set(observed) - expected_ue_set)
    return {
        "identity_basis": "message_id_ue_prefix_after_upf_snat",
        "expected_ues": sorted(expected_ue_set),
        "expected_nat_sources": sorted(expected_nat_set),
        "observed_sources": {str(key): sorted(value) for key, value in observed.items()},
        "invalid_message_id_count": len(invalid_message_ids),
        "invalid_message_id_examples": invalid_message_ids[:10],
        "unexpected_nat_sources": sorted(unexpected_nat_sources),
        "missing_ues": missing_ues,
        "unexpected_ues": unexpected_ues,
    }


def sender_route_report(
    sender: Mapping[str, object],
    expected_network_map: Mapping[int, Mapping[str, object]],
    tunnel_health: Mapping[str, Mapping[str, object]],
    *,
    tx_ratio_min: float,
    tx_ratio_max: float,
) -> dict:
    """Prove per-UE sender routing before UPF NAT obscures source identity."""
    raw_bindings = sender.get("socket_bindings", {})
    raw_per_ue = sender.get("per_ue", {})
    bindings = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    per_ue = raw_per_ue if isinstance(raw_per_ue, Mapping) else {}
    rows: Dict[str, dict] = {}
    for ue_id, expected in sorted(expected_network_map.items()):
        key = str(ue_id)
        binding = bindings.get(key, {})
        sender_row = per_ue.get(key, {})
        tunnel = tunnel_health.get(f"ue{ue_id}", {})
        expected_ip = str(expected["ip"])
        requested_ip = str(binding.get("requested_bind_ip", ""))
        actual_ip = str(binding.get("actual_local_ip", ""))
        sent_bytes = int(sender_row.get("sent_onwire_bytes", 0) or 0)
        tunnel_tx_bytes = int(tunnel.get("tx_bytes_delta", 0) or 0)
        ratio = tunnel_tx_bytes / sent_bytes if sent_bytes > 0 else None
        binding_pass = requested_ip == expected_ip and actual_ip == expected_ip
        byte_pass = (
            sent_bytes == 0 and tunnel_tx_bytes == 0
        ) or (
            sent_bytes > 0
            and ratio is not None
            and tx_ratio_min <= ratio <= tx_ratio_max
        )
        rows[key] = {
            "iface": str(expected["iface"]),
            "expected_bind_ip": expected_ip,
            "requested_bind_ip": requested_ip,
            "actual_local_ip": actual_ip,
            "actual_local_port": binding.get("actual_local_port"),
            "sent_onwire_bytes": sent_bytes,
            "tunnel_tx_bytes_delta": tunnel_tx_bytes,
            "tunnel_tx_to_sender_ratio": ratio,
            "binding_pass": binding_pass,
            "byte_accounting_pass": byte_pass,
            "pass": binding_pass and byte_pass,
        }
    return {
        "identity_basis": "socket_bind_then_fixed_tunnel_tx_counters_before_upf_snat",
        "tx_ratio_bounds": [tx_ratio_min, tx_ratio_max],
        "per_ue": rows,
        "pass": len(rows) == len(expected_network_map)
        and all(row["pass"] for row in rows.values()),
    }


def parse_channel_models(payload: str) -> Dict[str, dict]:
    """Parse `channelmod show current` into model-name keyed evidence."""
    models: Dict[str, dict] = {}
    current: Optional[dict] = None
    for line in payload.splitlines():
        header = re.match(r"^model\s+(\d+)\s+(\S+)\s+type\s+(\S+):$", line.strip())
        if header:
            current = {
                "model_index": int(header.group(1)),
                "model_name": header.group(2),
                "model_type": header.group(3),
            }
            models[str(current["model_name"])] = current
            continue
        if current is None:
            continue
        parameters = re.search(
            r"path loss:\s*([-+0-9.eE]+)\s+noise:\s*([-+0-9.eE]+)", line
        )
        if parameters:
            current["path_loss_db"] = float(parameters.group(1))
            current["noise_power_db"] = float(parameters.group(2))
    return models


def per_ue_radio_summary(
    power_rows: Iterable[Mapping[str, object]],
    rlc_rows: Iterable[Mapping[str, object]],
) -> dict:
    """Map each internal UE to its RNTI and summarize its gNB PUSCH telemetry."""
    candidates: Dict[int, Counter[int]] = defaultdict(Counter)
    for row in rlc_rows:
        try:
            candidates[int(row["ue_id"])][int(row["rnti"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    rnti_by_ue = {
        ue_id: counts.most_common(1)[0][0]
        for ue_id, counts in candidates.items()
        if counts
    }
    ue_by_rnti = {rnti: ue_id for ue_id, rnti in rnti_by_ue.items()}
    values: Dict[int, dict] = defaultdict(lambda: {"snr_db": [], "mcs": []})
    for row in power_rows:
        try:
            rnti = int(row["rnti"])
            ue_id = ue_by_rnti[rnti]
            values[ue_id]["snr_db"].append(float(row["snrx10"]) / 10.0)
            values[ue_id]["mcs"].append(float(row["mcs"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {
        str(ue_id): {
            "rnti": rnti,
            "sample_count": len(values[ue_id]["snr_db"]),
            "median_pusch_snr_db": median(values[ue_id]["snr_db"]),
            "median_ul_mcs": median(values[ue_id]["mcs"]),
        }
        for ue_id, rnti in sorted(rnti_by_ue.items())
    }


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("DG-A config must be a mapping")
    if data.get("stage") != "DG-A+DG-A.1":
        raise ValueError("runner may execute only stage DG-A+DG-A.1")
    forbidden = set(data["authorization_boundary"]["forbidden"])
    if "DG-B" not in forbidden or "RL" not in forbidden:
        raise ValueError("authorization boundary must forbid DG-B and RL")
    trials = data.get("trials", [])
    if [row.get("id") for row in trials] != [f"A{index}" for index in range(1, 10)]:
        raise ValueError("DG-A must contain exactly A1..A9")
    if int(data["radio"]["ue_count"]) != 2:
        raise ValueError("DG-A must use the existing two-UE setup")
    if int(data["transport"]["payload_bytes"]) != 409600:
        raise ValueError("DG-A decision traffic must remain 400 KiB")
    if data["radio"]["mcs_policy"] != "sinr":
        raise ValueError("DG-A requires SCENESENSE_MCS_POLICY=sinr")
    radio = data["radio"]
    tunnels = radio["expected_tunnels"]
    expected_tunnel_keys = [
        (index, f"oaitun_ue{index + 1}") for index in range(2)
    ]
    actual_tunnel_keys = [
        (int(row.get("ue_id", -1)), str(row.get("iface", ""))) for row in tunnels
    ]
    if actual_tunnel_keys != expected_tunnel_keys:
        raise ValueError(
            "DG-A tunnel identities must be fixed by internal UE index; IPs are discovered"
        )
    imsis = [str(row.get("imsi", "")) for row in tunnels]
    if len(set(imsis)) != 2 or any(not re.fullmatch(r"[0-9]{15}", imsi) for imsi in imsis):
        raise ValueError("DG-A requires two distinct 15-digit expected subscriber IMSIs")
    expected_dnn = str(radio.get("expected_dnn", ""))
    if not expected_dnn:
        raise ValueError("DG-A requires an expected DNN")
    try:
        expected_subnet = ipaddress.ip_network(str(radio["expected_ue_subnet"]), strict=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("DG-A requires a canonical expected UE IPv4 subnet") from exc
    if expected_subnet.version != 4 or expected_subnet.num_addresses < 4:
        raise ValueError("DG-A expected UE subnet must contain at least two usable IPv4 hosts")
    nat_sources = data["radio"].get("expected_receiver_nat_sources", [])
    if not nat_sources or len(set(str(value) for value in nat_sources)) != len(nat_sources):
        raise ValueError("DG-A requires distinct expected UPF N6 source address(es)")
    instrumentation = data["instrumentation"]
    tx_ratio_min = float(instrumentation["tunnel_tx_to_sender_ratio_min"])
    tx_ratio_max = float(instrumentation["tunnel_tx_to_sender_ratio_max"])
    if not 0 < tx_ratio_min <= tx_ratio_max:
        raise ValueError("invalid tunnel TX/application byte-ratio bounds")
    relay_port = int(instrumentation["ue_trace_relay_port"])
    trace_ports = {
        int(data["radio"]["gnb_ttracer_port"]),
        int(data["radio"]["ue_ttracer_port"]),
    }
    if not 0 < relay_port < 65536 or relay_port in trace_ports:
        raise ValueError("UE trace relay port must be valid and distinct from tracee ports")
    if float(instrumentation["receiver_finalize_timeout_s"]) <= 0:
        raise ValueError("receiver finalize timeout must be positive")
    if float(instrumentation["network_sampler_finalize_timeout_s"]) <= 0:
        raise ValueError("network sampler finalize timeout must be positive")

    trial_map = {str(row["id"]): row for row in trials}
    for trial in trials:
        trial_id = str(trial["id"])
        kind = str(trial.get("kind", ""))
        controller = str(trial.get("controller", "open_loop"))
        if kind not in SEND_KINDS or controller not in CONTROLLERS:
            raise ValueError(f"{trial_id} has unsupported kind/controller: {kind}/{controller}")
        if float(trial.get("duration_s", 0)) <= 0:
            raise ValueError(f"{trial_id} duration must be positive")
        expected_block = "A" if int(trial_id[1:]) <= 7 else "B"
        if str(trial.get("block")) != expected_block:
            raise ValueError(f"{trial_id} must run in block {expected_block}")
        if kind == "controlled" and controller == "open_loop":
            raise ValueError(f"{trial_id} controlled traffic requires a controller")
        if kind != "controlled" and controller != "open_loop":
            raise ValueError(f"{trial_id} non-controlled traffic cannot select {controller}")
        fractions = trial.get("fractions")
        if fractions is not None and (
            len(fractions) != 2 or any(float(value) < 0 for value in fractions)
        ):
            raise ValueError(f"{trial_id} fractions must contain two non-negative values")
        if kind == "equal" and float(trial.get("rho", 0)) <= 0:
            raise ValueError(f"{trial_id} equal traffic requires positive rho")
        if kind in {"asymmetric", "controlled"} and fractions is None:
            raise ValueError(f"{trial_id} requires explicit per-UE fractions")
        if kind == "burst":
            phases = trial.get("phases", [])
            if not phases or any(len(phase) != 3 for phase in phases):
                raise ValueError(f"{trial_id} requires START/END/RHO burst phases")
            parsed = [tuple(float(value) for value in phase) for phase in phases]
            if (
                abs(parsed[0][0]) > 1e-9
                or abs(parsed[-1][1] - float(trial["duration_s"])) > 1e-9
                or any(left[1] != right[0] for left, right in zip(parsed, parsed[1:]))
            ):
                raise ValueError(f"{trial_id} burst phases must be contiguous and cover the trial")
    for left, right in (("A6", "A7"), ("A8", "A9")):
        lhs, rhs = trial_map[left], trial_map[right]
        for key in ("fractions", "duration_s", "demand_seed"):
            if lhs.get(key) != rhs.get(key):
                raise ValueError(f"paired trials {left}/{right} differ in {key}")
        if (lhs.get("controller"), rhs.get("controller")) != (
            "decentralized_c1",
            "centralized_observable",
        ):
            raise ValueError(f"paired trials {left}/{right} have the wrong controllers")
    switch = data.get("runtime_switch", {})
    required_switch_keys = {
        "telnet_host",
        "telnet_port",
        "initial_noise_power_db",
        "target_noise_power_db",
        "baseline_traffic_s",
        "strong_traffic_s",
        "baseline_fractions",
        "strong_fractions",
        "production_gate_traffic_s",
        "production_gate_fractions",
        "controlled_gate_traffic_s",
        "controlled_gate_fractions",
        "minimum_snr_movement_db",
        "minimum_mcs_movement",
    }
    if not required_switch_keys <= set(switch):
        raise ValueError(
            f"runtime_switch lacks keys: {sorted(required_switch_keys - set(switch))}"
        )
    if int(switch["telnet_port"]) <= 0:
        raise ValueError("runtime-switch telnet port must be positive")
    for key in (
        "baseline_fractions",
        "strong_fractions",
        "production_gate_fractions",
        "controlled_gate_fractions",
    ):
        if len(switch[key]) != 2 or any(float(value) <= 0 for value in switch[key]):
            raise ValueError(f"runtime_switch.{key} must contain two positive UE offers")
    for key in (
        "baseline_traffic_s",
        "strong_traffic_s",
        "production_gate_traffic_s",
        "controlled_gate_traffic_s",
    ):
        if float(switch[key]) <= 0:
            raise ValueError(f"runtime_switch.{key} must be positive")
    if data["radio"].get("attach_strategy") != "clean_then_runtime_strong":
        raise ValueError("DG-A must attach clean and switch both uplinks at runtime")
    registration = data["radio"].get("strong_rung_registration", {})
    if (
        registration.get("observable")
        != "GNB_MAC_PUSCH_POWER_CONTROL.snrx10_div_10"
        or int(registration.get("registered_ue_count", 0)) != 2
        or float(registration.get("channel_noise_power_db", float("nan")))
        != float(switch["target_noise_power_db"])
    ):
        raise ValueError("strong-rung registration must match the two-UE runtime channel contract")
    return data


class Runner:
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        *,
        dry_run: bool = False,
        preflight_only: bool = False,
        attach_smoke_repeats: int = 0,
        attach_channel_mode: str = "strong",
        runtime_switch_smoke: bool = False,
        runtime_switch_startup_smoke: bool = False,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.output_dir = output_dir.resolve()
        self.dry_run = dry_run
        self.preflight_only = preflight_only
        self.attach_smoke_repeats = int(attach_smoke_repeats)
        if self.attach_smoke_repeats < 0:
            raise ValueError("attach_smoke_repeats cannot be negative")
        self.attach_channel_mode = str(attach_channel_mode)
        self.runtime_switch_smoke = bool(runtime_switch_smoke)
        self.runtime_switch_startup_smoke = bool(runtime_switch_startup_smoke)
        if self.runtime_switch_smoke and self.runtime_switch_startup_smoke:
            raise ValueError("choose either runtime-switch startup smoke or full smoke")
        runtime_diagnostic = self.runtime_switch_smoke or self.runtime_switch_startup_smoke
        if runtime_diagnostic and self.attach_smoke_repeats:
            raise ValueError("runtime-switch diagnostics and attach-only smoke are mutually exclusive")
        full_dg_a_mode = not runtime_diagnostic and self.attach_smoke_repeats == 0
        self.runtime_channel_control = runtime_diagnostic or full_dg_a_mode
        if self.attach_channel_mode not in {"strong", "clean"}:
            raise ValueError("attach_channel_mode must be 'strong' or 'clean'")
        if self.attach_channel_mode != "strong" and self.attach_smoke_repeats <= 0:
            raise ValueError("clean channel mode is restricted to attach-only smoke runs")
        self.paths = {
            name: resolve(ROOT, value) if name != "python" else Path(value)
            for name, value in self.config["paths"].items()
        }
        self.progress_path = self.output_dir / "progress.jsonl"
        self.driver_log = self.output_dir / "driver.log"
        self.started_core = False
        self.ran_active = False
        self.gnb_log_handle = None
        self.ue_log_handle = None
        self.block_data: Dict[str, dict] = {}
        self.trial_records: List[dict] = []
        self.ue_network_map: Dict[int, dict] = {}
        self.stop_requested = False
        self.d0_started = False
        self.dg_a_trials_started = False
        self.ue_trace_relay_proc: Optional[subprocess.Popen[bytes]] = None
        self.ue_trace_relay_block: Optional[str] = None

    def event(self, event: str, **data: object) -> None:
        row = {"timestamp": utc_now(), "event": event, **data}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def command(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.event("command", argv=list(args), cwd=str(cwd or ROOT))
        if self.dry_run:
            return subprocess.CompletedProcess(args, 0, "", "")
        result = subprocess.run(
            list(args),
            cwd=str(cwd or ROOT),
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            timeout=timeout,
            check=False,
        )
        if capture and result.stdout:
            self.event("command_output", argv0=args[0], output=result.stdout[-8000:])
        if check and result.returncode != 0:
            raise StageFailure(f"command failed ({result.returncode}): {' '.join(args)}")
        return result

    def preflight(self) -> None:
        radio = self.config["radio"]
        active_gnb_config = (
            radio["clean_gnb_config"]
            if self.attach_channel_mode == "clean"
            else radio["gnb_config"]
        )
        required = [
            self.paths["python"],
            self.paths["oai_ran_build"] / "nr-softmodem",
            self.paths["oai_ran_build"] / "nr-uesoftmodem",
            self.paths["ttracer_dir"] / "record",
            self.paths["ttracer_dir"] / "csv",
            self.paths["ttracer_dir"] / "replay",
            self.paths["ttracer_dir"] / "multi",
            self.paths["t_messages"],
            ROOT / "scripts/sample_oai_network_metrics.py",
            self.paths["oai_cn_dir"] / "conf/config.yaml",
            self.paths["oai_ran_conf"] / active_gnb_config,
            self.paths["oai_ran_conf"] / radio["ue_base_config"],
        ]
        if self.attach_channel_mode == "strong" or self.runtime_channel_control:
            required.append(self.paths["oai_ran_conf"] / radio["channel_config"])
        if self.runtime_channel_control:
            required.append(self.paths["oai_ran_build"] / "libtelnetsrv.so")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise StageFailure(f"preflight missing paths: {missing}")
        for tool in ("sudo", "docker", "setsid", "nsenter", "timeout"):
            if shutil.which(tool) is None:
                raise StageFailure(f"preflight missing executable: {tool}")
        self.command(["sudo", "-n", "true"])
        running = self.command(
            ["bash", "-lc", "pgrep -x nr-softmodem || true; pgrep -x nr-uesoftmodem || true"],
            check=False,
        ).stdout.strip()
        if running and not self.dry_run:
            raise StageFailure(f"preflight found existing RAN processes; refusing takeover: {running}")
        self._materialize_ue_config()
        subscriber_contract = self._validate_subscriber_contract()
        self._validate_sender_contracts()
        if self.runtime_channel_control:
            active_gnb_config = str(self.runtime_gnb_config)
        config_copy = self.output_dir / "resolved_config.yaml"
        config_copy.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        git = self.command(["git", "rev-parse", "HEAD"]).stdout.strip()
        status = self.command(["git", "status", "--short"], check=False).stdout
        oai_git = self.command(
            ["git", "-C", str(ROOT / "OAI/openairinterface5g"), "rev-parse", "HEAD"]
        ).stdout.strip()
        oai_status = self.command(
            ["git", "-C", str(ROOT / "OAI/openairinterface5g"), "status", "--short"], check=False
        ).stdout
        manifest = {
            "schema_version": self.config["schema_version"],
            "stage": self.config["stage"],
            "created_at": utc_now(),
            "repo_commit": git,
            "repo_status": status.splitlines(),
            "oai_commit": oai_git,
            "oai_status": oai_status.splitlines(),
            "config_sha256": sha256(config_copy),
            "t_messages_sha256": sha256(self.paths["t_messages"]),
            "binaries": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in {
                    "gnb": self.paths["oai_ran_build"] / "nr-softmodem",
                    "ue": self.paths["oai_ran_build"] / "nr-uesoftmodem",
                    "ttracer_multi": self.paths["ttracer_dir"] / "multi",
                }.items()
            },
            "authorization_boundary": self.config["authorization_boundary"],
            "dry_run": self.dry_run,
            "preflight_only": self.preflight_only,
            "attach_smoke_repeats": self.attach_smoke_repeats,
            "attach_channel_mode": self.attach_channel_mode,
            "runtime_switch_smoke": self.runtime_switch_smoke,
            "runtime_switch_startup_smoke": self.runtime_switch_startup_smoke,
            "active_gnb_config": active_gnb_config,
            "active_ue_config": str(self.runtime_ue_config),
            "channel_config_sha256": (
                sha256(self.paths["oai_ran_conf"] / radio["channel_config"])
                if self.attach_channel_mode == "strong" or self.runtime_channel_control
                else None
            ),
            "ue_identity_contract": {
                "stable_key": "expected_imsi_and_internal_ue_id_and_tunnel_name",
                "tunnel_rule": "ue_id_0=oaitun_ue1;ue_id_1=oaitun_ue2",
                "ip_assignment": "dynamic_unique_usable_host_within_expected_dnn_subnet",
                "sender_binding": "discover_ipv4_from_fixed_tunnel_each_ran_start",
                "subscriber_contract": subscriber_contract,
            },
        }
        atomic_json(self.output_dir / "run_manifest.json", manifest)
        self._local_transport_control()
        self.event("preflight_pass")

    def _validate_subscriber_contract(self) -> dict:
        """Cross-check UE config identities and the CN's configured DNN subnet."""
        radio = self.config["radio"]
        expected = {
            int(row["ue_id"]): {
                "imsi": str(row["imsi"]),
                "iface": str(row["iface"]),
            }
            for row in radio["expected_tunnels"]
        }
        profiles = parse_uicc_profiles(
            self.runtime_ue_config.read_text(encoding="utf-8", errors="replace")
        )
        if set(profiles) != set(expected):
            raise StageFailure(
                f"UE config UICC indexes differ: expected={sorted(expected)}, "
                f"actual={sorted(profiles)}"
            )
        expected_dnn = str(radio["expected_dnn"])
        for ue_id, identity in sorted(expected.items()):
            profile = profiles[ue_id]
            if profile["imsi"] != identity["imsi"]:
                raise StageFailure(
                    f"uicc{ue_id} IMSI differs: expected={identity['imsi']}, "
                    f"actual={profile['imsi']}"
                )
            if expected_dnn not in profile["dnns"]:
                raise StageFailure(
                    f"uicc{ue_id} does not request expected DNN {expected_dnn!r}: "
                    f"{profile['dnns']}"
                )

        cn_config_path = self.paths["oai_cn_dir"] / "conf/config.yaml"
        cn_config = yaml.safe_load(cn_config_path.read_text(encoding="utf-8"))
        matches = [
            row for row in cn_config.get("dnns", [])
            if str(row.get("dnn", "")) == expected_dnn
        ]
        if len(matches) != 1:
            raise StageFailure(
                f"CN config must define expected DNN {expected_dnn!r} exactly once"
            )
        configured_subnet = str(ipaddress.ip_network(str(matches[0]["ipv4_subnet"]), strict=True))
        expected_subnet = str(ipaddress.ip_network(str(radio["expected_ue_subnet"]), strict=True))
        if configured_subnet != expected_subnet:
            raise StageFailure(
                f"CN DNN subnet differs: expected={expected_subnet}, actual={configured_subnet}"
            )
        return {
            "dnn": expected_dnn,
            "ipv4_subnet": expected_subnet,
            "ues": [
                {"ue_id": ue_id, **identity} for ue_id, identity in sorted(expected.items())
            ],
        }

    def _materialize_ue_config(self) -> None:
        base_path = self.paths["oai_ran_conf"] / self.config["radio"]["ue_base_config"]
        if self.attach_channel_mode == "clean":
            # Reuse the known-good multi-UE contract verbatim. Without the
            # chanmod command-line option, the included model list is not
            # activated by RFsim.
            self.runtime_ue_config = base_path
            self.runtime_gnb_config = (
                self.paths["oai_ran_conf"] / self.config["radio"]["clean_gnb_config"]
            )
            return
        channel_path = self.paths["oai_ran_conf"] / self.config["radio"]["channel_config"]
        base = base_path.read_text(encoding="utf-8")
        channel = channel_path.read_text(encoding="utf-8")
        expected_models = {
            *(f'rfsimu_channel_enB{index}' for index in range(int(self.config["radio"]["ue_count"]))),
            *(f'rfsimu_channel_ue{index}' for index in range(int(self.config["radio"]["ue_count"]))),
        }
        missing_models = sorted(
            model for model in expected_models if f'model_name     = "{model}"' not in channel
        )
        if missing_models:
            raise StageFailure(
                f"channel config lacks explicit per-UE RFsim models: {missing_models}"
            )
        runtime_suffix = "awgn_strong"
        if self.runtime_channel_control:
            initial_noise = float(self.config["runtime_switch"]["initial_noise_power_db"])
            channel, replacements = re.subn(
                r"noise_power_dB\s*=\s*[-+0-9.eE]+;",
                f"noise_power_dB = {initial_noise:g};",
                channel,
            )
            if replacements != len(expected_models):
                raise StageFailure(
                    "runtime-switch initial channel did not rewrite exactly one noise value "
                    f"per explicit model: expected={len(expected_models)}, actual={replacements}"
                )
            runtime_suffix = "runtime_switch_initial_clean"
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        if marker not in base:
            raise StageFailure(f"expected channel include missing from {base_path}")
        resolved = base.replace(marker, channel)
        runtime = self.output_dir / "runtime" / f"ue.multi2.{runtime_suffix}.conf"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_text(resolved, encoding="utf-8")
        self.runtime_ue_config = runtime
        if self.runtime_channel_control:
            gnb_base = (
                self.paths["oai_ran_conf"] / self.config["radio"]["clean_gnb_config"]
            )
            runtime_gnb = self.output_dir / "runtime" / "gnb.runtime_switch_initial_clean.conf"
            runtime_gnb.write_text(
                gnb_base.read_text(encoding="utf-8") + "\n\n" + channel,
                encoding="utf-8",
            )
            self.runtime_gnb_config = runtime_gnb
        else:
            self.runtime_gnb_config = (
                self.paths["oai_ran_conf"] / self.config["radio"]["gnb_config"]
            )

    def _build_sender_command(
        self,
        trial: Mapping[str, object],
        trial_dir: Path,
        *,
        mu_hat: float,
        rnti_map: Optional[Mapping[int, int]] = None,
        service_conversion: float = 1.0,
        payload_bytes: Optional[int] = None,
        network_map: Optional[Mapping[int, Mapping[str, object]]] = None,
    ) -> List[str]:
        """Build and parse-check the exact endpoint command before trial side effects."""
        transport = self.config["transport"]
        command = [
            str(self.paths["python"]),
            "-m",
            "rl_agent.multiue_oai.endpoint",
            "send",
            "--remote-host",
            str(self.config["radio"]["ext_dn_ip"]),
            "--remote-port",
            str(transport["udp_port"]),
            "--run-dir",
            str(trial_dir),
            "--kind",
            str(trial["kind"]),
            "--controller",
            str(trial.get("controller", "open_loop")),
            "--mu-hat-mbps",
            str(mu_hat),
            "--duration-s",
            str(float(trial["duration_s"])),
            "--payload-bytes",
            str(payload_bytes or transport["payload_bytes"]),
            "--chunk-bytes",
            str(transport["chunk_bytes"]),
            "--tick-s",
            str(transport["tick_s"]),
            "--socket-send-buffer-bytes",
            str(transport["socket_send_buffer_bytes"]),
            "--pessimism-factor",
            str(self.config["c1"]["pessimism_factor"]),
            "--estimator-window-s",
            str(self.config["c1"]["estimator_window_s"]),
            "--estimator-ewma-alpha",
            str(self.config["c1"]["estimator_ewma_alpha"]),
            "--service-conversion",
            str(service_conversion),
        ]
        fractions = trial.get("fractions")
        if fractions is None:
            rho = float(trial.get("rho", 0.0))
            fractions = [rho / 2.0, rho / 2.0]
        fractions = [float(value) for value in fractions]
        if len(fractions) != int(self.config["radio"]["ue_count"]):
            raise StageFailure(
                f"{trial['id']} has {len(fractions)} demand fractions for "
                f"{self.config['radio']['ue_count']} UEs"
            )
        active_network = network_map if network_map is not None else self.ue_network_map
        expected_ids = set(range(int(self.config["radio"]["ue_count"])))
        if set(active_network) != expected_ids:
            raise StageFailure(
                f"{trial['id']} lacks a complete discovered UE network map: {active_network}"
            )
        for ue, fraction in zip(self.config["radio"]["expected_tunnels"], fractions):
            ue_id = int(ue["ue_id"])
            bind_ip = str(active_network[ue_id]["ip"])
            command.extend(["--ue", f"{ue_id},{bind_ip},{fraction}"])
        for phase in trial.get("phases", []):
            command.extend(["--phase", ",".join(str(value) for value in phase)])
        if "demand_seed" in trial:
            command.extend(["--demand-seed", str(trial["demand_seed"])])
        if trial.get("controller", "open_loop") != "open_loop":
            if not rnti_map or set(rnti_map) != expected_ids:
                raise StageFailure(f"{trial['id']} lacks a complete frozen UE-to-RNTI map")
            for ue_id, rnti in sorted(rnti_map.items()):
                command.extend(["--rnti-map", f"{ue_id},{rnti}"])
            command.extend(
                [
                    "--ttracer-csv",
                    str(self.paths["ttracer_dir"] / "csv"),
                    "--t-messages",
                    str(self.paths["t_messages"]),
                    "--ttracer-port",
                    str(self.config["instrumentation"]["ue_trace_relay_port"]),
                ]
            )
        try:
            parsed = build_endpoint_parser().parse_args(command[3:])
            validate_send_args(parsed)
        except (SystemExit, ValueError) as exc:
            raise StageFailure(
                f"{trial['id']} generated an invalid endpoint sender command: {exc}"
            ) from exc
        return command

    def _sender_contract_trials(self) -> List[dict]:
        switch = self.config["runtime_switch"]
        multiplier = float(self.config["calibration"]["offer_multiplier"])
        generated = [
            {
                "id": f"STRONG_GATE_{block}",
                "block": block,
                "kind": "smoke",
                "fractions": list(switch["production_gate_fractions"]),
                "duration_s": float(switch["production_gate_traffic_s"]),
            }
            for block in ("A", "B")
        ]
        generated.extend(
            {
                "id": f"CONTROLLED_PATH_GATE_{block}",
                "block": block,
                "kind": "controlled",
                "controller": "decentralized_c1",
                "fractions": list(switch["controlled_gate_fractions"]),
                "duration_s": float(switch["controlled_gate_traffic_s"]),
                "demand_seed": 61290 + (0 if block == "A" else 1),
            }
            for block in ("A", "B")
        )
        generated.extend(
            {
                "id": f"CAL_{block}",
                "block": block,
                "kind": "calibration",
                "fractions": [multiplier / 2.0, multiplier / 2.0],
                "duration_s": float(self.config["calibration"]["traffic_s"]),
            }
            for block in ("A", "B")
        )
        generated.extend(
            [
                {
                    "id": "D0_SMOKE",
                    "block": "A",
                    "kind": "smoke",
                    "fractions": [0.10, 0.10],
                    "duration_s": float(self.config["instrumentation"]["smoke_s"]),
                },
                *[
                    {
                        "id": trial_id,
                        "block": "A",
                        "kind": "equal",
                        "rho": 1.0,
                        "duration_s": float(
                            self.config["instrumentation"]["perturbation_s"]
                        ),
                    }
                    for trial_id in ("D0_TRACE_MIN", "D0_TRACE_FULL")
                ],
            ]
        )
        return generated + [dict(row) for row in self.config["trials"]]

    def _validate_sender_contracts(self) -> None:
        placeholder_network = {
            int(row["ue_id"]): {
                "ue_id": int(row["ue_id"]),
                "iface": str(row["iface"]),
                "ip": f"127.0.0.{int(row['ue_id']) + 1}",
            }
            for row in self.config["radio"]["expected_tunnels"]
        }
        placeholder_rnti = {0: 0x1001, 1: 0x1002}
        records = []
        for trial in self._sender_contract_trials():
            controlled = trial.get("controller", "open_loop") != "open_loop"
            payload = 92160 if trial["id"] == "D0_SMOKE" else None
            command = self._build_sender_command(
                trial,
                self.output_dir / "preflight" / "sender_contracts" / str(trial["id"]),
                mu_hat=float(self.config["calibration"]["prior_ceiling_mbps"]),
                rnti_map=placeholder_rnti if controlled else None,
                payload_bytes=payload,
                network_map=placeholder_network,
            )
            records.append(
                {
                    "trial_id": str(trial["id"]),
                    "kind": str(trial["kind"]),
                    "controller": str(trial.get("controller", "open_loop")),
                    "sender_argv": command,
                }
            )
        atomic_json(
            self.output_dir / "preflight" / "sender_contracts.json",
            {"validated_count": len(records), "trials": records},
        )

    def _local_transport_control(self) -> None:
        if self.dry_run:
            return
        control = self.output_dir / "preflight" / "local_transport"
        control.mkdir(parents=True, exist_ok=True)
        stop_file = control / "receiver_stop.request.json"
        receiver = self._popen_log(
            [
                "sudo",
                "-n",
                "nsenter",
                "-t",
                str(os.getpid()),
                "-n",
                "--",
                str(self.paths["python"]),
                "-m",
                "rl_agent.multiue_oai.endpoint",
                "receive",
                "--bind-host",
                "127.0.0.1",
                "--port",
                "56099",
                "--run-dir",
                str(control),
                "--stop-file",
                str(stop_file),
                "--max-duration-s",
                "20",
            ],
            control / "receiver.log",
            cwd=ROOT,
        )
        receiver_failure: Optional[str] = None
        sampler: Optional[subprocess.Popen[bytes]] = None
        sampler_handle = None
        sampler_failure: Optional[str] = None
        try:
            sampler_handle = (control / "network_sampler_stdout.log").open("wb")
            sampler = subprocess.Popen(
                [
                    str(self.paths["python"]),
                    str(ROOT / "scripts/sample_oai_network_metrics.py"),
                    "--run-group",
                    "PREFLIGHT_LOCAL_TRANSPORT",
                    "--duration-s",
                    "20",
                    "--output-dir",
                    str(control / "network"),
                    "--stop-file",
                    str(control / "network_sampler_stop.request.json"),
                    "--interface",
                    "lo:ue0",
                    "--interface",
                    "lo:ue1",
                ],
                cwd=str(ROOT),
                stdout=sampler_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(0.5)
            prior = float(self.config["calibration"]["prior_ceiling_mbps"])
            fractions = float(self.config["calibration"]["offer_multiplier"]) / 2.0
            result = self.command(
                [
                    str(self.paths["python"]),
                    "-m",
                    "rl_agent.multiue_oai.endpoint",
                    "send",
                    "--remote-host",
                    "127.0.0.1",
                    "--remote-port",
                    "56099",
                    "--run-dir",
                    str(control),
                    "--kind",
                    "smoke",
                    "--controller",
                    "open_loop",
                    "--mu-hat-mbps",
                    str(prior),
                    "--duration-s",
                    "8",
                    "--payload-bytes",
                    str(self.config["transport"]["payload_bytes"]),
                    "--chunk-bytes",
                    str(self.config["transport"]["chunk_bytes"]),
                    "--ue",
                    f"0,127.0.0.1,{fractions}",
                    "--ue",
                    f"1,127.0.0.1,{fractions}",
                ],
                timeout=20,
            )
            if result.returncode != 0:
                raise StageFailure("local transport sender failed")
        finally:
            receiver_failure = self._finalize_receiver(receiver, control)
            if sampler is not None:
                sampler_failure = self._finalize_network_sampler(
                    sampler, control, sampler_handle
                )
            elif sampler_handle is not None and not sampler_handle.closed:
                sampler_handle.close()
            self.command(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(control)],
                check=False,
            )
        if receiver_failure:
            raise StageFailure(f"local transport receiver lifecycle failure: {receiver_failure}")
        if sampler_failure:
            raise StageFailure(f"local transport network sampler lifecycle failure: {sampler_failure}")
        if not (control / "sender_summary.json").is_file():
            raise StageFailure("local transport sender omitted sender_summary.json")
        sender = json.loads((control / "sender_summary.json").read_text())
        received = json.loads((control / "receiver_summary.json").read_text())
        if received["complete_frames"] != sender["demand_count"]:
            raise StageFailure(
                f"local transport control lost frames: sent={sender['demand_count']} complete={received['complete_frames']}"
            )
        if received["partial_frames"] or received["checksum_failures"]:
            raise StageFailure(f"local transport control invalid: {received}")

    def start_core(self) -> None:
        existing = self.command(
            ["sudo", "-n", "docker", "ps", "--format", "{{.Names}}"], check=False
        ).stdout.splitlines()
        required = {"oai-amf", "oai-smf", "oai-upf", self.config["radio"]["ext_dn_container"]}
        self.started_core = not required.issubset(set(existing))
        self.command(["sudo", "-n", "docker", "compose", "up", "-d"], cwd=self.paths["oai_cn_dir"])
        deadline = time.monotonic() + 180
        last_state: Dict[str, str] = {}
        while time.monotonic() < deadline:
            states = {}
            for container in sorted(required):
                result = self.command(
                    [
                        "sudo",
                        "-n",
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                        container,
                    ],
                    check=False,
                )
                if result.returncode == 0:
                    states[container] = result.stdout.strip()
            last_state = states
            ready = len(states) == len(required) and all(
                value == "running none" or value == "running healthy"
                for value in states.values()
            )
            if ready:
                self.event("core_ready", containers=states)
                return
            time.sleep(3)
        raise StageFailure(f"core containers did not become healthy within 180 s: {last_state}")

    def _popen_log(
        self, command: Sequence[str], log_path: Path, *, cwd: Optional[Path] = None
    ) -> subprocess.Popen[bytes]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("wb")
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd or self.paths["oai_ran_build"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc._scenesense_log_handle = handle  # type: ignore[attr-defined]
        return proc

    @staticmethod
    def _tcp_socket_rows() -> List[tuple[int, int, str]]:
        """Return local port, remote port, and state from Linux procfs."""
        rows: List[tuple[int, int, str]] = []
        for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
            try:
                lines = path.read_text(encoding="ascii").splitlines()[1:]
            except OSError:
                continue
            for line in lines:
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    local_port = int(fields[1].rsplit(":", 1)[1], 16)
                    remote_port = int(fields[2].rsplit(":", 1)[1], 16)
                except (IndexError, ValueError):
                    continue
                rows.append((local_port, remote_port, fields[3]))
        return rows

    def _ensure_ue_trace_relay(self, block: str) -> None:
        """Keep one upstream UE tracer connection and fan out local consumers."""
        existing = self.ue_trace_relay_proc
        if existing is not None:
            if existing.poll() is not None:
                raise StageFailure(
                    f"UE trace relay exited unexpectedly with {existing.returncode}"
                )
            if self.ue_trace_relay_block != block:
                raise StageFailure(
                    "UE trace relay survived across RAN blocks; refusing cross-block telemetry"
                )
            return

        relay_port = int(self.config["instrumentation"]["ue_trace_relay_port"])
        source_port = int(self.config["radio"]["ue_ttracer_port"])
        if any(
            local == relay_port and state == "0A"
            for local, _remote, state in self._tcp_socket_rows()
        ):
            raise StageFailure(f"UE trace relay port {relay_port} is already in use")
        command = [
            str(self.paths["ttracer_dir"] / "multi"),
            "-d",
            str(self.paths["t_messages"]),
            "-ip",
            "127.0.0.1",
            "-p",
            str(source_port),
            "-lp",
            str(relay_port),
        ]
        log_path = self.output_dir / "blocks" / block / "ue_trace_relay.log"
        self.event(
            "ue_trace_relay_start",
            block=block,
            source_port=source_port,
            relay_port=relay_port,
            argv=command,
        )
        proc = self._popen_log(command, log_path, cwd=self.paths["ttracer_dir"])
        self.ue_trace_relay_proc = proc
        self.ue_trace_relay_block = block
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                returncode = proc.returncode
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                self._stop_ue_trace_relay()
                raise StageFailure(
                    f"UE trace relay exited during startup with {returncode}: {tail}"
                )
            sockets = self._tcp_socket_rows()
            listener_ready = any(
                local == relay_port and state == "0A"
                for local, _remote, state in sockets
            )
            source_connected = any(
                state == "01" and source_port in {local, remote}
                for local, remote, state in sockets
            )
            if listener_ready and source_connected:
                artifact = {
                    "schema_version": "scenesense.multiue_oai.trace_relay.v1",
                    "block": block,
                    "started_at": utc_now(),
                    "pid": proc.pid,
                    "source_port": source_port,
                    "relay_port": relay_port,
                    "upstream_connected": True,
                    "binary_sha256": sha256(self.paths["ttracer_dir"] / "multi"),
                }
                atomic_json(
                    self.output_dir / "blocks" / block / "ue_trace_relay.json",
                    artifact,
                )
                self.event("ue_trace_relay_ready", **artifact)
                return
            time.sleep(0.1)
        self._stop_ue_trace_relay()
        raise StageFailure(
            f"UE trace relay did not establish {source_port}->{relay_port} within 10 s"
        )

    def _stop_ue_trace_relay(self) -> None:
        proc = self.ue_trace_relay_proc
        block = self.ue_trace_relay_block
        if proc is None:
            return
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
        finally:
            handle = getattr(proc, "_scenesense_log_handle", None)
            if handle is not None and not handle.closed:
                handle.close()
            self.ue_trace_relay_proc = None
            self.ue_trace_relay_block = None
            self.event("ue_trace_relay_stop", block=block, returncode=proc.returncode)

    def start_ran(self, block: str) -> None:
        self.stop_ran()
        block_dir = self.output_dir / "blocks" / block
        radio = self.config["radio"]
        gnb_conf = self.runtime_gnb_config
        gnb_cmd = [
            "sudo",
            "-n",
            "env",
            f"SCENESENSE_MCS_POLICY={radio['mcs_policy']}",
            "./nr-softmodem",
            "-O",
            str(gnb_conf),
            "--gNBs.[0].min_rxtxtime",
            "6",
            "--rfsim",
            "--T_stdout",
            str(radio["t_stdout"]),
            "--T_nowait",
            "--T_port",
            str(radio["gnb_ttracer_port"]),
        ]
        if self.attach_channel_mode == "strong" or self.runtime_channel_control:
            gnb_cmd[gnb_cmd.index("--T_stdout"):gnb_cmd.index("--T_stdout")] = [
                "--rfsimulator.[0].options",
                "chanmod",
            ]
        if self.runtime_channel_control:
            switch = self.config["runtime_switch"]
            gnb_cmd[gnb_cmd.index("--T_stdout"):gnb_cmd.index("--T_stdout")] = [
                "--telnetsrv",
                "--telnetsrv.listenaddr",
                str(switch["telnet_host"]),
                "--telnetsrv.listenport",
                str(switch["telnet_port"]),
            ]
        self.event("ran_start", block=block, component="gnb", argv=gnb_cmd)
        self.gnb_proc = self._popen_log(gnb_cmd, block_dir / "gnb_stdout.log")
        self.ran_active = True
        time.sleep(5)
        ue_cmd = [
            "sudo",
            "-n",
            "./nr-uesoftmodem",
            "--rfsim",
            "--num-ues",
            str(radio["ue_count"]),
            "--rfsimulator.[0].serveraddr",
            "127.0.0.1",
            "-r",
            str(radio["prb"]),
            "--numerology",
            str(radio["numerology"]),
            "--band",
            str(radio["band"]),
            "-C",
            str(radio["downlink_frequency_hz"]),
            "-O",
            str(self.runtime_ue_config),
            "--T_stdout",
            str(radio["t_stdout"]),
            "--T_nowait",
            "--T_port",
            str(radio["ue_ttracer_port"]),
        ]
        if self.attach_channel_mode == "strong" or self.runtime_channel_control:
            ue_cmd[ue_cmd.index("-r"):ue_cmd.index("-r")] = [
                "--rfsimulator.[0].options",
                "chanmod",
            ]
        self.event("ran_start", block=block, component="ue", argv=ue_cmd)
        self.ue_proc = self._popen_log(ue_cmd, block_dir / "ue_stdout.log")
        self.wait_tunnels(block)

    def wait_tunnels(self, block: str) -> None:
        deadline = time.monotonic() + float(self.config["radio"]["attach_timeout_s"])
        while time.monotonic() < deadline:
            mapping = self._discover_ue_network_map()
            good = len(mapping) == int(self.config["radio"]["ue_count"])
            if good:
                network_contract = ue_network_contract_report(
                    mapping,
                    self.config["radio"]["expected_tunnels"],
                    str(self.config["radio"]["expected_ue_subnet"]),
                )
                if not network_contract["pass"]:
                    raise StageFailure(
                        f"{block} UE network identity/address contract failed: "
                        f"{network_contract['reasons']}"
                    )
                log_path = self.output_dir / "blocks" / block / "ue_stdout.log"
                observed_imsis = parse_runtime_uicc_imsis(
                    log_path.read_text(encoding="utf-8", errors="replace")
                )
                expected_imsis = [
                    str(row["imsi"])
                    for row in sorted(
                        self.config["radio"]["expected_tunnels"],
                        key=lambda row: int(row["ue_id"]),
                    )
                ]
                if observed_imsis != expected_imsis:
                    raise StageFailure(
                        f"{block} runtime subscriber identities differ: "
                        f"expected={expected_imsis}, actual={observed_imsis}"
                    )
                for ue in mapping.values():
                    ping = self.command(
                        [
                            "ping",
                            "-I",
                            ue["iface"],
                            "-c",
                            "2",
                            "-W",
                            "2",
                            self.config["radio"]["ext_dn_ip"],
                        ],
                        check=False,
                    )
                    if ping.returncode:
                        good = False
                        break
            if good:
                self._record_ue_network_map(block, mapping)
                self.event(
                    "ran_attached",
                    block=block,
                    tunnels=[mapping[index] for index in sorted(mapping)],
                    expected_imsis=expected_imsis,
                    network_contract=network_contract,
                )
                return
            if self.gnb_proc.poll() is not None or self.ue_proc.poll() is not None:
                raise StageFailure(
                    "softmodem exited before both UEs attached: "
                    f"gnb_rc={self.gnb_proc.poll()}, ue_rc={self.ue_proc.poll()}"
                )
            time.sleep(3)
        raise StageFailure("both expected UE tunnels did not attach")

    def _discover_ue_network_map(self) -> Dict[int, dict]:
        """Map stable internal UE identities to the current CN-assigned IPv4 addresses."""
        mapping: Dict[int, dict] = {}
        for registered in self.config["radio"]["expected_tunnels"]:
            iface = str(registered["iface"])
            result = self.command(
                ["ip", "-j", "-4", "addr", "show", "dev", iface], check=False
            )
            if result.returncode:
                continue
            try:
                ip = parse_interface_ipv4(result.stdout, iface)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise StageFailure(f"could not parse IPv4 identity for {iface}: {exc}") from exc
            if ip is None:
                continue
            ue_id = int(registered["ue_id"])
            mapping[ue_id] = {"ue_id": ue_id, "iface": iface, "ip": ip}
        ips = [str(row["ip"]) for row in mapping.values()]
        if len(ips) != len(set(ips)):
            raise StageFailure(f"duplicate UE tunnel IPv4 assignment: {mapping}")
        return mapping

    def _record_ue_network_map(self, block: str, mapping: Mapping[int, Mapping[str, object]]) -> None:
        resolved = [dict(mapping[index]) for index in sorted(mapping)]
        self.ue_network_map = {int(row["ue_id"]): dict(row) for row in resolved}
        artifact = self.output_dir / "blocks" / block / "ue_network_map.json"
        atomic_json(
            artifact,
            {
                "schema_version": "scenesense.multiue_oai.ue_network_map.v1",
                "block": block,
                "discovered_at": utc_now(),
                "identity_chain": "ue_id_to_fixed_tunnel_to_dynamic_ip",
                "address_contract": {
                    "dnn": str(self.config["radio"]["expected_dnn"]),
                    "subnet": str(self.config["radio"]["expected_ue_subnet"]),
                    "exact_ip_pinning": False,
                },
                "subscriber_identities": [
                    {
                        "ue_id": int(row["ue_id"]),
                        "imsi": str(row["imsi"]),
                        "iface": str(row["iface"]),
                    }
                    for row in self.config["radio"]["expected_tunnels"]
                ],
                "ues": resolved,
            },
        )
        manifest_path = self.output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("ue_network_maps", {})[block] = {
            "artifact": str(artifact.relative_to(self.output_dir)),
            "ues": resolved,
        }
        atomic_json(manifest_path, manifest)

    def stop_ran(self) -> None:
        self._stop_ue_trace_relay()
        if not getattr(self, "ran_active", False):
            return
        self.event("ran_stop")
        self.command(["sudo", "-n", "pkill", "-INT", "-x", "nr-uesoftmodem"], check=False)
        self.command(["sudo", "-n", "pkill", "-INT", "-x", "nr-softmodem"], check=False)
        time.sleep(3)
        self.command(["sudo", "-n", "pkill", "-TERM", "-x", "nr-uesoftmodem"], check=False)
        self.command(["sudo", "-n", "pkill", "-TERM", "-x", "nr-softmodem"], check=False)
        for proc in (getattr(self, "ue_proc", None), getattr(self, "gnb_proc", None)):
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.command(
                        ["sudo", "-n", "kill", "-KILL", "--", f"-{proc.pid}"],
                        check=False,
                    )
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.event("ran_process_kill_timeout", pid=proc.pid)
                handle = getattr(proc, "_scenesense_log_handle", None)
                if handle is not None and not handle.closed:
                    handle.close()
        self.ran_active = False
        self.ue_network_map = {}

    def _wait_for_cold_ran(self, timeout_s: float = 20.0) -> None:
        """Verify a repetition starts without old softmodems or tunnel devices."""
        deadline = time.monotonic() + timeout_s
        expected_ifaces = [str(row["iface"]) for row in self.config["radio"]["expected_tunnels"]]
        last_state: dict = {}
        while time.monotonic() < deadline:
            processes = {
                name: self.command(["pgrep", "-x", name], check=False).stdout.splitlines()
                for name in ("nr-softmodem", "nr-uesoftmodem")
            }
            interfaces = {
                iface: self.command(["ip", "link", "show", iface], check=False).returncode == 0
                for iface in expected_ifaces
            }
            last_state = {"processes": processes, "interfaces_present": interfaces}
            if not any(processes.values()) and not any(interfaces.values()):
                return
            time.sleep(1)
        raise StageFailure(f"RAN did not return to a cold state: {last_state}")

    def _attach_stability_evidence(self, block: str) -> dict:
        registered = {ue_id: dict(row) for ue_id, row in self.ue_network_map.items()}
        if set(registered) != set(range(int(self.config["radio"]["ue_count"]))):
            raise StageFailure(f"{block} lacks a complete discovered UE network map: {registered}")
        samples = []
        for sample_index in range(3):
            if self.gnb_proc.poll() is not None or self.ue_proc.poll() is not None:
                raise StageFailure(f"{block} softmodem exited during attach stability hold")
            per_ue = []
            current = self._discover_ue_network_map()
            if current != registered:
                raise StageFailure(
                    f"{block} UE network mapping changed during stability hold: "
                    f"registered={registered}, current={current}"
                )
            for ue_id in sorted(registered):
                ue = registered[ue_id]
                ping = self.command(
                    [
                        "ping",
                        "-I",
                        str(ue["iface"]),
                        "-c",
                        "3",
                        "-W",
                        "2",
                        str(self.config["radio"]["ext_dn_ip"]),
                    ],
                    check=False,
                )
                if ping.returncode != 0:
                    raise StageFailure(
                        f"{block} lost UE{ue['ue_id']} during stability hold: "
                        f"iface={ue['iface']}, ip={ue['ip']}, ping_rc={ping.returncode}"
                    )
                per_ue.append(
                    {
                        "ue_id": int(ue["ue_id"]),
                        "iface": str(ue["iface"]),
                        "ip": str(ue["ip"]),
                        "ping_pass": True,
                    }
                )
            samples.append({"sample_index": sample_index, "ues": per_ue})
            if sample_index < 2:
                time.sleep(3)
        return {"block": block, "samples": samples}

    def run_attach_smoke(self) -> dict:
        """Run only repeated cold two-UE attachment; never launch D0 or DG-A."""
        if self.attach_smoke_repeats <= 0:
            raise StageFailure("attach-only mode requires a positive repeat count")
        repetitions = []
        for repeat in range(1, self.attach_smoke_repeats + 1):
            block = f"ATTACH_R{repeat}"
            self._wait_for_cold_ran()
            started = time.monotonic()
            self.start_ran(block)
            evidence = self._attach_stability_evidence(block)
            evidence["attach_and_hold_elapsed_s"] = time.monotonic() - started
            self.stop_ran()
            self._wait_for_cold_ran()
            gnb_log = self.output_dir / "blocks" / block / "gnb_stdout.log"
            gnb_text = gnb_log.read_text(encoding="utf-8", errors="replace")
            evidence["attach_channel_mode"] = self.attach_channel_mode
            activated_models = [
                line for line in gnb_text.splitlines() if "Random channel rfsimu_channel_" in line
            ]
            if self.attach_channel_mode == "strong":
                missing_model_lines = [
                    line
                    for line in gnb_text.splitlines()
                    if "Model rfsimu_channel_" in line and "not found" in line
                ]
                if missing_model_lines:
                    raise StageFailure(f"{block} used RFsim model fallback: {missing_model_lines}")
                evidence["explicit_uplink_models_seen"] = {
                    name: f"Random channel {name} in rfsimulator activated" in gnb_text
                    for name in ("rfsimu_channel_ue0", "rfsimu_channel_ue1")
                }
                if not all(evidence["explicit_uplink_models_seen"].values()):
                    raise StageFailure(
                        f"{block} did not activate both explicit uplink models: "
                        f"{evidence['explicit_uplink_models_seen']}"
                    )
            elif activated_models:
                raise StageFailure(
                    f"{block} was registered clean but activated channel models: {activated_models}"
                )
            else:
                evidence["chanmod_activation_absent"] = True
            repetitions.append(evidence)
            atomic_json(
                self.output_dir / "attach_smoke_partial.json",
                {"requested_repeats": self.attach_smoke_repeats, "passed": repetitions},
            )
            self.event("attach_smoke_repeat_pass", repeat=repeat, evidence=evidence)
        summary = {
            "schema_version": "scenesense.multiue_oai.attach_smoke.v1",
            "status": "ATTACH_SMOKE_PASS",
            "decision": "DG_A_NOT_RUN_AWAIT_HUMAN_REVIEW",
            "requested_repeats": self.attach_smoke_repeats,
            "attach_channel_mode": self.attach_channel_mode,
            "passed_repeats": len(repetitions),
            "all_passed": len(repetitions) == self.attach_smoke_repeats,
            "repetitions": repetitions,
            "d0_launched": False,
            "dg_a_launched": False,
            "next_stage_launched": False,
        }
        atomic_json(self.output_dir / "results_summary.json", summary)
        return summary

    @staticmethod
    def _recv_telnet_prompt(connection: socket.socket, timeout_s: float) -> str:
        connection.settimeout(timeout_s)
        chunks: List[bytes] = []
        total = 0
        while total < 1024 * 1024:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if b"> " in b"".join(chunks[-2:]):
                break
        payload = b"".join(chunks).decode("utf-8", errors="replace")
        if "> " not in payload:
            raise StageFailure(f"telnet response lacked a softmodem prompt: {payload[-500:]}")
        return payload

    def _telnet_command(self, command: str, *, connect_timeout_s: float = 15.0) -> str:
        switch = self.config["runtime_switch"]
        deadline = time.monotonic() + connect_timeout_s
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(
                    (str(switch["telnet_host"]), int(switch["telnet_port"])), timeout=2
                ) as connection:
                    connection.sendall(b"\n")
                    self._recv_telnet_prompt(connection, 3.0)
                    connection.sendall(command.encode("utf-8") + b"\n")
                    response = self._recv_telnet_prompt(connection, 5.0)
                    self.event("runtime_channel_command", command=command, response=response)
                    return response
            except (OSError, StageFailure) as exc:
                last_error = exc
                time.sleep(0.5)
        raise StageFailure(
            f"gNB telnet control unavailable after {connect_timeout_s:.1f}s: {last_error}"
        )

    def _assert_active_uplink_models(self, block: str) -> dict:
        log_path = self.output_dir / "blocks" / block / "gnb_stdout.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        missing_model_lines = [
            line
            for line in text.splitlines()
            if "Model rfsimu_channel_" in line and "not found" in line
        ]
        active = {
            name: f"Random channel {name} in rfsimulator activated" in text
            for name in ("rfsimu_channel_ue0", "rfsimu_channel_ue1")
        }
        if missing_model_lines or not all(active.values()):
            raise StageFailure(
                f"{block} did not activate both explicit uplink objects: "
                f"active={active}, fallback={missing_model_lines}"
            )
        return {"active": active, "fallback_lines": missing_model_lines}

    def _switch_both_uplinks(self, block: str) -> dict:
        """Modify both live gNB UL objects and verify the resulting model state."""
        switch = self.config["runtime_switch"]
        names = ("rfsimu_channel_ue0", "rfsimu_channel_ue1")
        initial = float(switch["initial_noise_power_db"])
        target = float(switch["target_noise_power_db"])
        active = self._assert_active_uplink_models(block)
        before_raw = self._telnet_command("channelmod show current")
        before = parse_channel_models(before_raw)
        missing = [name for name in names if name not in before]
        if missing:
            raise StageFailure(f"runtime control cannot resolve both uplink objects: {missing}")
        wrong_initial = {
            name: before[name]
            for name in names
            if abs(float(before[name].get("noise_power_db", float("nan"))) - initial) > 1e-6
        }
        if wrong_initial:
            raise StageFailure(
                f"uplink model(s) did not start on the registered clean setting: {wrong_initial}"
            )
        modify_responses = {}
        for name in names:
            model_index = int(before[name]["model_index"])
            command = f"channelmod modify {model_index} noise_power_dB {target:g}"
            response = self._telnet_command(command)
            if "ERROR" in response:
                raise StageFailure(f"runtime channel command rejected for {name}: {response}")
            modify_responses[name] = {"command": command, "response": response}
        after_raw = self._telnet_command("channelmod show current")
        after = parse_channel_models(after_raw)
        wrong_target = {
            name: after.get(name)
            for name in names
            if name not in after
            or abs(float(after[name].get("noise_power_db", float("nan"))) - target) > 1e-6
        }
        if wrong_target:
            raise StageFailure(f"runtime channel switch was partial or a no-op: {wrong_target}")
        evidence = {
            "block": block,
            "active_uplink_objects": active,
            "initial_noise_power_db": initial,
            "target_noise_power_db": target,
            "before": {name: before[name] for name in names},
            "modify_responses": modify_responses,
            "after": {name: after[name] for name in names},
            "both_uplinks_modified": True,
        }
        atomic_json(self.output_dir / "blocks" / block / "runtime_channel_switch.json", evidence)
        return evidence

    def run_runtime_switch_startup_smoke(self) -> dict:
        """Validate only runtime-control process startup and attachment."""
        if not self.runtime_switch_startup_smoke:
            raise StageFailure("runtime-switch startup smoke was not selected")
        block = "RUNTIME_SWITCH_STARTUP"
        self._wait_for_cold_ran()
        self.start_ran(block)
        stability = self._attach_stability_evidence(block)
        active = self._assert_active_uplink_models(block)
        raw_state = self._telnet_command("channelmod show current")
        state = parse_channel_models(raw_state)
        initial = float(self.config["runtime_switch"]["initial_noise_power_db"])
        names = ("rfsimu_channel_ue0", "rfsimu_channel_ue1")
        invalid = {
            name: state.get(name)
            for name in names
            if name not in state
            or abs(float(state[name].get("noise_power_db", float("nan"))) - initial) > 1e-6
        }
        if invalid:
            raise StageFailure(f"startup telnet state lacks both initial-clean UL objects: {invalid}")
        summary = {
            "schema_version": "scenesense.multiue_oai.runtime_switch_startup_smoke.v1",
            "status": "RUNTIME_SWITCH_STARTUP_PASS",
            "decision": "FULL_RUNTIME_SWITCH_NOT_RUN",
            "block": block,
            "ue_network_map": [
                dict(self.ue_network_map[index]) for index in sorted(self.ue_network_map)
            ],
            "tunnel_stability": stability,
            "active_uplink_objects": active,
            "initial_channel_state": {name: state[name] for name in names},
            "d0_launched": False,
            "dg_a_launched": False,
            "next_stage_launched": False,
        }
        atomic_json(self.output_dir / "results_summary.json", summary)
        return summary

    def run_runtime_switch_smoke(self) -> dict:
        """Run a bounded clean-to-strong switch smoke; never launch D0 or DG-A."""
        if not self.runtime_switch_smoke:
            raise StageFailure("runtime-switch smoke was not selected")
        block = "RUNTIME_SWITCH_R1"
        switch = self.config["runtime_switch"]
        self._wait_for_cold_ran()
        self.start_ran(block)
        pre_hold = self._attach_stability_evidence(block)
        prior = float(self.config["calibration"]["prior_ceiling_mbps"])
        baseline_metrics = self.run_trial(
            {
                "id": "SWITCH_BASELINE_CLEAN",
                "block": block,
                "kind": "equal",
                "fractions": [float(value) for value in switch["baseline_fractions"]],
                "duration_s": float(switch["baseline_traffic_s"]),
            },
            mu_hat=prior,
            enforce_strong_rung=False,
        )
        channel_evidence = self._switch_both_uplinks(block)
        strong_metrics = self.run_trial(
            {
                "id": "SWITCH_STRONG_ASYMMETRIC",
                "block": block,
                "kind": "asymmetric",
                "fractions": [float(value) for value in switch["strong_fractions"]],
                "duration_s": float(switch["strong_traffic_s"]),
            },
            mu_hat=prior,
            enforce_strong_rung=True,
        )
        post_hold = self._attach_stability_evidence(block)
        baseline_radio = baseline_metrics["radio_validity"]["per_ue"]
        strong_radio = strong_metrics["radio_validity"]["per_ue"]
        movement = {}
        for ue_id in ("0", "1"):
            snr_movement = float(baseline_radio[ue_id]["median_pusch_snr_db"]) - float(
                strong_radio[ue_id]["median_pusch_snr_db"]
            )
            mcs_movement = float(baseline_radio[ue_id]["median_ul_mcs"]) - float(
                strong_radio[ue_id]["median_ul_mcs"]
            )
            movement[ue_id] = {
                "baseline": baseline_radio[ue_id],
                "strong": strong_radio[ue_id],
                "snr_drop_db": snr_movement,
                "mcs_drop": mcs_movement,
                "pass": snr_movement >= float(switch["minimum_snr_movement_db"])
                and mcs_movement >= float(switch["minimum_mcs_movement"]),
            }
        if not all(row["pass"] for row in movement.values()):
            raise StageFailure(
                "runtime channel command changed model state but did not empirically move "
                f"both UEs from clean to the strong rung: {movement}"
            )
        summary = {
            "schema_version": "scenesense.multiue_oai.runtime_switch_smoke.v1",
            "status": "RUNTIME_SWITCH_SMOKE_PASS",
            "decision": "DG_A_NOT_RUN_AWAIT_HUMAN_REVIEW",
            "block": block,
            "ue_network_map": [
                dict(self.ue_network_map[index]) for index in sorted(self.ue_network_map)
            ],
            "pre_switch_tunnel_stability": pre_hold,
            "channel_switch": channel_evidence,
            "baseline_metrics": baseline_metrics,
            "strong_metrics": strong_metrics,
            "per_ue_empirical_movement": movement,
            "post_switch_tunnel_stability": post_hold,
            "sender_routing_validated": True,
            "both_uplinks_on_strong_rung": True,
            "d0_launched": False,
            "dg_a_launched": False,
            "next_stage_launched": False,
        }
        atomic_json(self.output_dir / "results_summary.json", summary)
        return summary

    def _enter_registered_strong_rung(self, block: str) -> dict:
        """Attach clean, switch both ULs, then prove the rung with real traffic."""
        if not self.runtime_channel_control:
            raise StageFailure("registered strong-rung entry requires runtime channel control")
        pre_switch = self._attach_stability_evidence(block)
        channel_switch = self._switch_both_uplinks(block)
        switch = self.config["runtime_switch"]
        gate_trial_id = f"STRONG_GATE_{block}"
        gate_metrics = self.run_trial(
            {
                "id": gate_trial_id,
                "block": block,
                "kind": "smoke",
                "fractions": [
                    float(value) for value in switch["production_gate_fractions"]
                ],
                "duration_s": float(switch["production_gate_traffic_s"]),
            },
            mu_hat=float(self.config["calibration"]["prior_ceiling_mbps"]),
            enforce_strong_rung=True,
        )
        rnti_map = {
            int(ue_id): int(row["rnti"])
            for ue_id, row in gate_metrics["radio_validity"]["per_ue"].items()
        }
        control_trial_id = f"CONTROLLED_PATH_GATE_{block}"
        control_metrics = self.run_trial(
            {
                "id": control_trial_id,
                "block": block,
                "kind": "controlled",
                "controller": "decentralized_c1",
                "fractions": [
                    float(value) for value in switch["controlled_gate_fractions"]
                ],
                "duration_s": float(switch["controlled_gate_traffic_s"]),
                "demand_seed": 61290 + (0 if block == "A" else 1),
            },
            mu_hat=float(self.config["calibration"]["prior_ceiling_mbps"]),
            rnti_map=rnti_map,
            service_conversion=1.0,
            enforce_strong_rung=True,
        )
        post_switch = self._attach_stability_evidence(block)
        evidence = {
            "block": block,
            "gate_trial_id": gate_trial_id,
            "pre_switch_tunnel_stability": pre_switch,
            "channel_switch": channel_switch,
            "per_ue_radio": gate_metrics["radio_validity"]["per_ue"],
            "sender_route_gate": gate_metrics["sender_route_gate"],
            "controlled_path_gate": {
                "trial_id": control_trial_id,
                "rnti_map": rnti_map,
                "observer": control_metrics["observer"],
                "sender_route_gate": control_metrics["sender_route_gate"],
                "per_ue_radio": control_metrics["radio_validity"]["per_ue"],
                "recorder_and_observer_concurrent": True,
            },
            "post_switch_tunnel_stability": post_switch,
            "both_uplinks_on_registered_strong_rung": True,
        }
        atomic_json(
            self.output_dir / "blocks" / block / "strong_rung_entry_gate.json",
            evidence,
        )
        self.event("strong_rung_entry_pass", block=block, gate_trial_id=gate_trial_id)
        return evidence

    def _ext_dn_pid(self) -> str:
        return self.command(
            [
                "sudo",
                "-n",
                "docker",
                "inspect",
                "-f",
                "{{.State.Pid}}",
                self.config["radio"]["ext_dn_container"],
            ]
        ).stdout.strip()

    def _start_receiver(self, trial_dir: Path, max_duration: float) -> subprocess.Popen[bytes]:
        pid = self._ext_dn_pid()
        stop_file = trial_dir / "receiver_stop.request.json"
        command = [
            "sudo",
            "-n",
            "nsenter",
            "-t",
            pid,
            "-n",
            "--",
            str(self.paths["python"]),
            "-m",
            "rl_agent.multiue_oai.endpoint",
            "receive",
            "--bind-host",
            self.config["radio"]["ext_dn_ip"],
            "--port",
            str(self.config["transport"]["udp_port"]),
            "--run-dir",
            str(trial_dir),
            "--stop-file",
            str(stop_file),
            "--socket-receive-buffer-bytes",
            str(self.config["transport"]["socket_receive_buffer_bytes"]),
            "--max-duration-s",
            str(max_duration),
        ]
        return self._popen_log(command, trial_dir / "receiver_stdout.log", cwd=ROOT)

    def _trace_processes(
        self, trial_id: str, trial_dir: Path, duration: float, ue_profile: str, gnb_profile: str
    ) -> List[subprocess.Popen[bytes]]:
        trace_root = self.output_dir / "ttracer"
        procs = []
        try:
            for source, profile in (("ue", ue_profile), ("gnb", gnb_profile)):
                log_path = trial_dir / f"ttracer_{source}_record_stdout.log"
                handle = log_path.open("wb")
                command = [
                    "bash",
                    str(ROOT / "scripts/ttracer_record_smoke.sh"),
                    "--run-group",
                    trial_id,
                    "--source",
                    source,
                    "--duration-s",
                    str(duration),
                    "--output-root",
                    str(trace_root),
                    "--profile",
                    profile,
                ]
                if source == "ue":
                    command.extend(
                        [
                            "--port",
                            str(self.config["instrumentation"]["ue_trace_relay_port"]),
                        ]
                    )
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(ROOT),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    handle.close()
                    raise
                process._scenesense_log_handle = handle  # type: ignore[attr-defined]
                procs.append(process)
        except BaseException:
            for process in procs:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                process_handle = getattr(process, "_scenesense_log_handle", None)
                if process_handle is not None and not process_handle.closed:
                    process_handle.close()
            raise
        return procs

    def _extract_traces(self, trial_id: str, ue_profile: str, gnb_profile: str) -> None:
        trace_root = self.output_dir / "ttracer"
        for source, profile in (("ue", ue_profile), ("gnb", gnb_profile)):
            self.command(
                [
                    "bash",
                    str(ROOT / "scripts/ttracer_extract_csv_smoke.sh"),
                    "--run-group",
                    trial_id,
                    "--source",
                    source,
                    "--output-root",
                    str(trace_root),
                    "--profile",
                    profile,
                    "--timeout-s",
                    str(self.config["instrumentation"]["extraction_timeout_s"]),
                    "--clean-output",
                ],
                timeout=600,
            )

    @staticmethod
    def _last_rlc_totals(path: Path) -> Dict[int, int]:
        latest: Dict[tuple[int, int], tuple[float, int]] = {}
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (int(row["ue_id"]), int(row["lcid"]))
                    stamp = parse_trace_time(row["time"])
                    value = int(row["bytes_in_buffer"])
                except (KeyError, ValueError):
                    continue
                if key not in latest or stamp >= latest[key][0]:
                    latest[key] = (stamp, value)
        totals: Dict[int, int] = defaultdict(int)
        for (ue_id, _lcid), (_stamp, value) in latest.items():
            totals[ue_id] += value
        return dict(totals)

    def _live_queue_probe(self, seconds: float = 6.0) -> Dict[int, int]:
        command = build_ttracer_csv_command(
            str(self.paths["ttracer_dir"] / "csv"),
            str(self.paths["t_messages"]),
            int(self.config["instrumentation"]["ue_trace_relay_port"]),
            "NRUE_MAC_RLC_BUFFER_STATUS",
            ("time", "rnti", "ue_id", "lcid", "bytes_in_buffer"),
        )
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            time.sleep(0.5)
            for ue in self.config["radio"]["expected_tunnels"]:
                self.command(
                    [
                        "ping",
                        "-I",
                        ue["iface"],
                        "-c",
                        "1",
                        "-W",
                        "1",
                        self.config["radio"]["ext_dn_ip"],
                    ],
                    check=False,
                )
            time.sleep(max(0.0, seconds - 0.5))
            os.killpg(process.pid, signal.SIGINT)
            stdout, _stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _stderr = process.communicate()
        latest: Dict[tuple[int, int], int] = {}
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                ue_id = int(parts[-3])
                lcid = int(parts[-2])
                value = int(parts[-1])
            except ValueError:
                continue
            latest[(ue_id, lcid)] = value
        totals: Dict[int, int] = defaultdict(int)
        for (ue_id, _lcid), value in latest.items():
            totals[ue_id] += value
        return dict(totals)

    def _ensure_drain(self, trial_id: str) -> Dict[int, int]:
        rlc_path = (
            self.output_dir
            / "ttracer"
            / trial_id
            / "ue"
            / "csv"
            / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
        )
        totals = self._last_rlc_totals(rlc_path)
        threshold = int(self.config["transport"]["drain_threshold_bytes"])
        expected = {int(row["ue_id"]) for row in self.config["radio"]["expected_tunnels"]}
        if set(totals) == expected and all(value <= threshold for value in totals.values()):
            return totals
        deadline = time.monotonic() + float(self.config["transport"]["drain_timeout_s"])
        while time.monotonic() < deadline:
            totals = self._live_queue_probe(float(self.config["transport"]["drain_stable_s"]))
            if set(totals) == expected and all(value <= threshold for value in totals.values()):
                return totals
        raise StageFailure(
            f"{trial_id} queue did not verify below {threshold} bytes per UE before timeout; last={totals}"
        )

    def _radio_and_grant_validity(
        self,
        trial_id: str,
        *,
        enforce_strong_rung: bool = True,
        require_per_ue_radio: bool = True,
    ) -> dict:
        root = self.output_dir / "ttracer" / trial_id
        power_path = root / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
        rlc_path = root / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
        if not power_path.exists() or (require_per_ue_radio and not rlc_path.exists()):
            raise StageFailure(
                f"{trial_id} lacks PUSCH/RLC telemetry: power={power_path.exists()}, "
                f"rlc={rlc_path.exists()}, require_per_ue_radio={require_per_ue_radio}"
            )
        with power_path.open(newline="", encoding="utf-8") as power_handle:
            power_rows = list(csv.DictReader(power_handle))
        rlc_rows: List[dict] = []
        if rlc_path.exists():
            with rlc_path.open(newline="", encoding="utf-8") as rlc_handle:
                rlc_rows = list(csv.DictReader(rlc_handle))
        per_ue = per_ue_radio_summary(power_rows, rlc_rows) if rlc_rows else {}
        expected_ues = {
            str(int(row["ue_id"])) for row in self.config["radio"]["expected_tunnels"]
        }
        if require_per_ue_radio and set(per_ue) != expected_ues:
            raise StageFailure(
                f"{trial_id} lacks distinct per-UE PUSCH telemetry: "
                f"expected={sorted(expected_ues)}, observed={per_ue}"
            )
        for ue_id, row in per_ue.items():
            if not math.isfinite(float(row["median_pusch_snr_db"])) or not math.isfinite(
                float(row["median_ul_mcs"])
            ):
                raise StageFailure(f"{trial_id} UE{ue_id} has empty PUSCH telemetry: {row}")
        radio = self.config["radio"]
        if enforce_strong_rung and not require_per_ue_radio:
            raise StageFailure(
                f"{trial_id} cannot enforce the strong rung without per-UE radio mapping"
            )
        if enforce_strong_rung:
            off_rung = {
                ue_id: row
                for ue_id, row in per_ue.items()
                if abs(float(row["median_pusch_snr_db"]) - float(radio["expected_snr_db"]))
                > float(radio["snr_tolerance_db"])
                or abs(float(row["median_ul_mcs"]) - float(radio["expected_mcs"]))
                > float(radio["mcs_tolerance"])
            }
            if off_rung:
                raise StageFailure(
                    f"{trial_id} has UE(s) off the registered strong SNR/MCS rung: {off_rung}"
                )

        self.command(
            [
                str(self.paths["python"]),
                str(ROOT / "scripts/compare_nrue_gnb_grants.py"),
                "--run-group",
                trial_id,
                "--root",
                str(self.output_dir / "ttracer"),
            ],
            timeout=120,
        )
        comparison = root / "analysis" / "ue_gnb_grant_validation.csv"
        if not comparison.is_file():
            raise StageFailure(f"{trial_id} grant reconciliation output is missing: {comparison}")
        ratios: Dict[str, float] = {}
        with comparison.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["direction"] != "ul" or not row["ue_vs_gnb_mac_tbs_ratio"]:
                    continue
                ratio = float(row["ue_vs_gnb_mac_tbs_ratio"])
                ratios[row["rnti"]] = ratio
        if len(ratios) != 2:
            raise StageFailure(f"{trial_id} did not reconcile two UL RNTIs: {ratios}")
        tolerance = float(self.config["instrumentation"]["ue_gnb_tbs_tolerance_fraction"])
        if any(abs(value - 1.0) > tolerance for value in ratios.values()):
            raise StageFailure(f"{trial_id} UE/gNB TBS reconciliation exceeds tolerance: {ratios}")
        return {
            "enforce_strong_rung": enforce_strong_rung,
            "require_per_ue_radio": require_per_ue_radio,
            "per_ue": per_ue,
            "median_pusch_snr_db": (
                median(float(row["median_pusch_snr_db"]) for row in per_ue.values())
                if per_ue
                else None
            ),
            "median_ul_mcs": (
                median(float(row["median_ul_mcs"]) for row in per_ue.values())
                if per_ue
                else None
            ),
            "ue_gnb_tbs_ratios": ratios,
        }

    def _stop_root_process(self, proc: subprocess.Popen[bytes]) -> None:
        try:
            if proc.poll() is None:
                # _popen_log starts a new session. Kill the whole sudo/nsenter/endpoint
                # process group so a detached root receiver cannot survive the trial.
                self.command(
                    ["sudo", "-n", "kill", "-TERM", "--", f"-{proc.pid}"], check=False
                )
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.command(
                        ["sudo", "-n", "kill", "-KILL", "--", f"-{proc.pid}"], check=False
                    )
                    proc.wait(timeout=3)
        finally:
            handle = getattr(proc, "_scenesense_log_handle", None)
            if handle is not None and not handle.closed:
                handle.close()

    def _finalize_receiver(
        self, proc: subprocess.Popen[bytes], trial_dir: Path
    ) -> Optional[str]:
        """Request a normal receiver flush and validate its artifact contract.

        The endpoint may run behind ``sudo nsenter``. A process-group signal is
        not a reliable graceful-stop API across that boundary, so signals are
        retained only as cleanup after an explicit stop-file request times out.
        """
        stop_file = trial_dir / "receiver_stop.request.json"
        try:
            atomic_json(stop_file, {"requested_at": utc_now(), "receiver_pid": proc.pid})
            self.event(
                "receiver_stop_requested",
                trial_id=trial_dir.name,
                stop_file=str(stop_file),
                receiver_pid=proc.pid,
            )
        except BaseException as exc:
            self._stop_root_process(proc)
            return f"could not write graceful-stop request: {type(exc).__name__}: {exc}"

        timeout_s = float(self.config["instrumentation"]["receiver_finalize_timeout_s"])
        timed_out = False
        cleanup_failure: Optional[str] = None
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.event(
                "receiver_graceful_stop_timeout",
                trial_id=trial_dir.name,
                timeout_s=timeout_s,
                receiver_pid=proc.pid,
            )
            try:
                self._stop_root_process(proc)
            except BaseException as exc:
                cleanup_failure = f"{type(exc).__name__}: {exc}"
        finally:
            handle = getattr(proc, "_scenesense_log_handle", None)
            if handle is not None and not handle.closed:
                handle.close()

        log_name = "receiver.log" if trial_dir.name == "local_transport" else "receiver_stdout.log"
        log_path = trial_dir / log_name
        log_tail = (
            log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            if log_path.exists()
            else "<receiver log missing>"
        )
        if timed_out:
            return (
                f"graceful stop timed out after {timeout_s:.1f}s; "
                f"cleanup_failure={cleanup_failure!r}; log_tail={log_tail!r}"
            )
        if proc.returncode != 0:
            return f"receiver exited with {proc.returncode}; log_tail={log_tail!r}"
        required = (
            "receiver_chunks.csv",
            "receiver_frames.csv",
            "receiver_summary.json",
        )
        missing = [name for name in required if not (trial_dir / name).is_file()]
        if missing:
            return (
                f"receiver exited cleanly but omitted artifacts {missing}; "
                f"log_tail={log_tail!r}"
            )
        self.event(
            "receiver_finalize_pass",
            trial_id=trial_dir.name,
            receiver_pid=proc.pid,
            artifacts=list(required),
        )
        return None

    def _finalize_network_sampler(
        self,
        proc: subprocess.Popen[bytes],
        trial_dir: Path,
        log_handle: Optional[object] = None,
    ) -> Optional[str]:
        """Stop the sampler without racing its final summary/manifest writes."""
        stop_file = trial_dir / "network_sampler_stop.request.json"
        timeout_s = float(
            self.config["instrumentation"]["network_sampler_finalize_timeout_s"]
        )
        timed_out = False
        cleanup_failure: Optional[str] = None
        request_failure: Optional[str] = None
        try:
            atomic_json(stop_file, {"requested_at": utc_now(), "sampler_pid": proc.pid})
            self.event(
                "network_sampler_stop_requested",
                trial_id=trial_dir.name,
                stop_file=str(stop_file),
                sampler_pid=proc.pid,
            )
        except BaseException as exc:
            request_failure = f"{type(exc).__name__}: {exc}"

        try:
            if request_failure is None:
                proc.wait(timeout=timeout_s)
            else:
                raise subprocess.TimeoutExpired("network sampler stop request", timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = request_failure is None
            if timed_out:
                self.event(
                    "network_sampler_graceful_stop_timeout",
                    trial_id=trial_dir.name,
                    timeout_s=timeout_s,
                    sampler_pid=proc.pid,
                )
            try:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=3)
            except BaseException as exc:
                cleanup_failure = f"{type(exc).__name__}: {exc}"
        finally:
            if log_handle is not None and not getattr(log_handle, "closed", True):
                log_handle.close()

        log_path = trial_dir / "network_sampler_stdout.log"
        log_tail = (
            log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            if log_path.exists()
            else "<network sampler log missing>"
        )
        if request_failure is not None:
            return (
                f"could not write graceful-stop request: {request_failure}; "
                f"cleanup_failure={cleanup_failure!r}; log_tail={log_tail!r}"
            )
        if timed_out:
            return (
                f"graceful stop timed out after {timeout_s:.1f}s; "
                f"cleanup_failure={cleanup_failure!r}; log_tail={log_tail!r}"
            )
        if proc.returncode != 0:
            return f"sampler exited with {proc.returncode}; log_tail={log_tail!r}"

        network_dir = trial_dir / "network"
        required = (
            network_dir / "network_timeseries.csv",
            network_dir / "network_summary.csv",
            network_dir / "network_manifest.json",
        )
        missing = [str(path.relative_to(trial_dir)) for path in required if not path.is_file()]
        if missing:
            return f"sampler omitted artifacts {missing}; log_tail={log_tail!r}"
        with (network_dir / "network_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        expected_labels = {
            f"ue{int(row['ue_id'])}" for row in self.config["radio"]["expected_tunnels"]
        }
        observed_labels = {str(row.get("iface_label", "")) for row in rows}
        if observed_labels != expected_labels or any(
            int(float(row.get("samples", 0) or 0)) < 2 for row in rows
        ):
            return (
                f"sampler summary contract failed: expected_labels={sorted(expected_labels)}, "
                f"rows={rows}"
            )
        self.event(
            "network_sampler_finalize_pass",
            trial_id=trial_dir.name,
            sampler_pid=proc.pid,
            artifacts=[str(path.relative_to(trial_dir)) for path in required],
        )
        return None

    def run_trial(
        self,
        trial: Mapping[str, object],
        *,
        mu_hat: float,
        rnti_map: Optional[Mapping[int, int]] = None,
        service_conversion: float = 1.0,
        payload_bytes: Optional[int] = None,
        ue_profile: str = "all",
        gnb_profile: str = "latency",
        enforce_strong_rung: bool = True,
        require_per_ue_radio: bool = True,
    ) -> dict:
        trial_id = str(trial["id"])
        trial_dir = self.output_dir / "runs" / trial_id
        trial_dir.mkdir(parents=True, exist_ok=False)
        duration = float(trial["duration_s"])
        pre_wait_s = max(
            float(self.config["transport"]["pre_idle_s"]),
            float(self.config["instrumentation"]["recorder_lead_s"]),
        )
        post_wait_s = max(
            float(self.config["transport"]["post_idle_s"]),
            float(self.config["instrumentation"]["recorder_tail_s"]),
        )
        total_trace_s = duration + pre_wait_s + post_wait_s + 3
        manifest = {
            "trial": dict(trial),
            "mu_hat_mbps": mu_hat,
            "rnti_map": dict(rnti_map or {}),
            "service_conversion": service_conversion,
            "created_at": utc_now(),
            "ue_trace_profile": ue_profile,
            "gnb_trace_profile": gnb_profile,
            "ue_network_map": [
                dict(self.ue_network_map[index]) for index in sorted(self.ue_network_map)
            ],
        }
        sender = self._build_sender_command(
            trial,
            trial_dir,
            mu_hat=mu_hat,
            rnti_map=rnti_map,
            service_conversion=service_conversion,
            payload_bytes=payload_bytes,
        )
        self._ensure_ue_trace_relay(str(trial["block"]))
        manifest["ue_trace_relay"] = {
            "source_port": int(self.config["radio"]["ue_ttracer_port"]),
            "relay_port": int(self.config["instrumentation"]["ue_trace_relay_port"]),
            "persistent_for_ran_block": True,
        }
        manifest["sender_argv"] = sender
        atomic_json(trial_dir / "trial_manifest.json", manifest)
        self.event("trial_start", trial_id=trial_id, trial=trial)
        receiver: Optional[subprocess.Popen[bytes]] = None
        recorders: List[subprocess.Popen[bytes]] = []
        sampler: Optional[subprocess.Popen[bytes]] = None
        sampler_handle = None
        sampler_command = [
            str(self.paths["python"]),
            str(ROOT / "scripts/sample_oai_network_metrics.py"),
            "--run-group",
            trial_id,
            "--duration-s",
            str(total_trace_s),
            "--output-dir",
            str(trial_dir / "network"),
            "--stop-file",
            str(trial_dir / "network_sampler_stop.request.json"),
        ]
        for ue in self.config["radio"]["expected_tunnels"]:
            sampler_command.extend(
                ["--interface", f"{ue['iface']}:ue{int(ue['ue_id'])}"]
            )
        recorder_failures: List[str] = []
        sampler_failure: Optional[str] = None
        receiver_failure: Optional[str] = None
        try:
            receiver = self._start_receiver(trial_dir, total_trace_s + 30)
            recorders = self._trace_processes(
                trial_id, trial_dir, total_trace_s, ue_profile, gnb_profile
            )
            sampler_handle = (trial_dir / "network_sampler_stdout.log").open("wb")
            sampler = subprocess.Popen(
                sampler_command,
                cwd=str(ROOT),
                stdout=sampler_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(pre_wait_s)
            sender_log_path = trial_dir / "sender_stdout.log"
            with sender_log_path.open("w", encoding="utf-8") as sender_log:
                result = subprocess.run(
                    sender,
                    cwd=str(ROOT),
                    text=True,
                    stdout=sender_log,
                    stderr=subprocess.STDOUT,
                    timeout=duration + 30,
                    check=False,
                )
            if result.returncode:
                tail = sender_log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise StageFailure(
                    f"sender failed for {trial_id} with {result.returncode}: {tail}"
                )
            time.sleep(post_wait_s)
        finally:
            if receiver is not None:
                receiver_failure = self._finalize_receiver(receiver, trial_dir)
            for process in recorders:
                try:
                    process.wait(timeout=total_trace_s + 15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
                if process.returncode:
                    recorder_failures.append(
                        f"pid={process.pid},returncode={process.returncode}"
                    )
                handle = getattr(process, "_scenesense_log_handle", None)
                if handle is not None and not handle.closed:
                    handle.close()
            if sampler is not None:
                sampler_failure = self._finalize_network_sampler(
                    sampler, trial_dir, sampler_handle
                )
            elif sampler_handle is not None and not sampler_handle.closed:
                sampler_handle.close()
            self.command(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(trial_dir)],
                check=False,
            )
        if receiver_failure:
            raise StageFailure(f"{trial_id} receiver lifecycle failure: {receiver_failure}")
        sender_required = ("sender_summary.json", "sender_demands.csv")
        sender_missing = [
            name for name in sender_required if not (trial_dir / name).is_file()
        ]
        if sender_missing:
            raise StageFailure(
                f"{trial_id} sender exited without required artifacts: {sender_missing}"
            )
        if recorder_failures:
            raise StageFailure(f"{trial_id} trace recorder failure: {recorder_failures}")
        if sampler_failure:
            raise StageFailure(f"{trial_id} network sampler failure: {sampler_failure}")
        if self.ue_trace_relay_proc is None or self.ue_trace_relay_proc.poll() is not None:
            raise StageFailure(f"{trial_id} UE trace relay died during the trial")
        self._extract_traces(trial_id, ue_profile, gnb_profile)
        metrics = self._trial_metrics(trial, trial_dir, mu_hat)
        metrics["radio_validity"] = self._radio_and_grant_validity(
            trial_id,
            enforce_strong_rung=enforce_strong_rung,
            require_per_ue_radio=require_per_ue_radio,
        )
        metrics["drained_rlc_bytes"] = self._ensure_drain(trial_id)
        manifest["completed_at"] = utc_now()
        manifest["metrics"] = metrics
        atomic_json(trial_dir / "trial_manifest.json", manifest)
        self.trial_records.append({"id": trial_id, "path": str(trial_dir), **dict(trial), **metrics})
        self.event("trial_complete", trial_id=trial_id, metrics=metrics)
        return metrics

    def _trial_metrics(
        self, trial: Mapping[str, object], trial_dir: Path, mu_hat: float
    ) -> dict:
        trial_id = str(trial["id"])
        sender = json.loads((trial_dir / "sender_summary.json").read_text())
        receiver = json.loads((trial_dir / "receiver_summary.json").read_text())
        duration = float(sender["duration_target_s"])
        sent_bytes = sum(int(row["sent_onwire_bytes"]) for row in sender["per_ue"].values())
        offered = sent_bytes * 8 / max(duration, 1e-9) / 1e6
        chunks_path = trial_dir / "receiver_chunks.csv"
        received_bytes = 0
        with chunks_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                received_bytes += int(row["onwire_bytes"])
        goodput = received_bytes * 8 / max(duration, 1e-9) / 1e6
        metrics = {
            "offered_mbps": offered,
            "achieved_rho": offered / mu_hat if mu_hat > 0 else None,
            "receiver_onwire_mbps": goodput,
            "complete_frames": int(receiver["complete_frames"]),
            "partial_frames": int(receiver["partial_frames"]),
            "checksum_failures": int(receiver["checksum_failures"]),
            "latency_p50_ms": receiver["latency_p50_ms"],
            "latency_p95_ms": receiver["latency_p95_ms"],
            "demand_trace_sha256": sender["demand_trace_sha256"],
            "local_errors": int(sender["local_errors"]),
        }
        if sender.get("observer") is not None:
            metrics["observer"] = sender["observer"]
        expected_ues = {int(row["ue_id"]) for row in self.config["radio"]["expected_tunnels"]}
        with chunks_path.open(newline="", encoding="utf-8") as handle:
            metrics["receiver_identity_gate"] = receiver_identity_report(
                csv.DictReader(handle),
                expected_ues,
                self.config["radio"]["expected_receiver_nat_sources"],
            )
        if (
            metrics["receiver_identity_gate"]["invalid_message_id_count"]
            or metrics["receiver_identity_gate"]["unexpected_nat_sources"]
            or metrics["receiver_identity_gate"]["missing_ues"]
            or metrics["receiver_identity_gate"]["unexpected_ues"]
        ):
            raise StageFailure(
                f"{trial_id} post-NAT receiver identity failure: "
                f"{metrics['receiver_identity_gate']}"
            )
        if (
            metrics["local_errors"]
            or metrics["checksum_failures"]
            or int(receiver.get("identity_failures", 0))
            or int(receiver.get("invalid", 0))
        ):
            raise StageFailure(f"{trial_id} application validity failure: {metrics}")
        network_summary = trial_dir / "network" / "network_summary.csv"
        if not network_summary.exists():
            raise StageFailure(f"{trial_id} missing tunnel/network health summary")
        tunnel_health: Dict[str, dict] = {}
        with network_summary.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = row.get("iface_label", row.get("iface", "unknown"))
                drops = sum(
                    int(float(row.get(field, 0) or 0))
                    for field in ("tx_drops_delta", "rx_drops_delta", "tx_errors_delta", "rx_errors_delta")
                )
                tunnel_health[str(label)] = {
                    "iface": str(row.get("iface", "")),
                    "samples": int(float(row.get("samples", 0) or 0)),
                    "duration_s": float(row.get("duration_s", 0) or 0),
                    "tx_bytes_delta": int(float(row.get("tx_bytes_delta", 0) or 0)),
                    "tx_packets_delta": int(float(row.get("tx_packets_delta", 0) or 0)),
                    "drops_or_errors": drops,
                }
        metrics["tunnel_health"] = tunnel_health
        if len(tunnel_health) != 2 or any(
            row["drops_or_errors"] for row in tunnel_health.values()
        ):
            raise StageFailure(f"{trial_id} tunnel validity failure: {tunnel_health}")
        route = sender_route_report(
            sender,
            self.ue_network_map,
            tunnel_health,
            tx_ratio_min=float(
                self.config["instrumentation"]["tunnel_tx_to_sender_ratio_min"]
            ),
            tx_ratio_max=float(
                self.config["instrumentation"]["tunnel_tx_to_sender_ratio_max"]
            ),
        )
        metrics["sender_route_gate"] = route
        if not route["pass"]:
            raise StageFailure(f"{trial_id} per-UE sender routing failure: {route}")
        if str(trial.get("controller", "open_loop")) == "open_loop" and not trial_id.startswith(
            "D0_"
        ):
            if trial.get("kind") == "burst":
                phases = [tuple(float(value) for value in phase) for phase in trial.get("phases", [])]
                weighted_rho = sum((end - start) * rho for start, end, rho in phases) / duration
                target_fractions = [weighted_rho / 2.0, weighted_rho / 2.0]
            else:
                fractions = trial.get("fractions")
                if fractions is None:
                    rho = float(trial.get("rho", 0.0))
                    fractions = [rho / 2.0, rho / 2.0]
                target_fractions = [float(value) for value in fractions]
            tolerance = float(self.config["instrumentation"]["sender_rate_tolerance_fraction"])
            per_ue_rate_gate = {}
            for ue, fraction in zip(self.config["radio"]["expected_tunnels"], target_fractions):
                ue_id = str(ue["ue_id"])
                actual = (
                    int(sender["per_ue"][ue_id]["sent_onwire_bytes"])
                    * 8
                    / max(duration, 1e-9)
                    / 1e6
                )
                target = fraction * mu_hat
                deviation = abs(actual - target) / max(target, 1e-12)
                frame_rate_quantum_mbps = (
                    int(sender["onwire_bytes_per_frame"])
                    * 8
                    / max(duration, 1e-9)
                    / 1e6
                )
                allowed_absolute_mbps = max(
                    tolerance * target,
                    frame_rate_quantum_mbps,
                )
                per_ue_rate_gate[ue_id] = {
                    "target_mbps": target,
                    "actual_mbps": actual,
                    "deviation_fraction": deviation,
                    "absolute_deviation_mbps": abs(actual - target),
                    "frame_rate_quantum_mbps": frame_rate_quantum_mbps,
                    "allowed_absolute_mbps": allowed_absolute_mbps,
                    "pass": abs(actual - target) <= allowed_absolute_mbps + 1e-12,
                }
            metrics["open_loop_rate_gate"] = per_ue_rate_gate
            if any(not row["pass"] for row in per_ue_rate_gate.values()):
                raise StageFailure(
                    f"{trial_id} open-loop sender rate exceeds the larger of "
                    f"{tolerance:.1%} or one-frame quantization: "
                    f"{per_ue_rate_gate}"
                )
        return metrics

    def calibrate(self, block: str) -> dict:
        prior = float(self.config["calibration"]["prior_ceiling_mbps"])
        multiplier = float(self.config["calibration"]["offer_multiplier"])
        trial = {
            "id": f"CAL_{block}",
            "block": block,
            "kind": "calibration",
            "fractions": [multiplier / 2.0, multiplier / 2.0],
            "duration_s": float(self.config["calibration"]["traffic_s"]),
        }
        self.run_trial(trial, mu_hat=prior)
        trial_dir = self.output_dir / "runs" / str(trial["id"])
        sender = json.loads((trial_dir / "sender_summary.json").read_text())
        end_ns = int(sender["end_raw_ns"])
        start_window_ns = end_ns - int(float(self.config["calibration"]["service_window_s"]) * 1e9)
        per_ue_bytes = defaultdict(int)
        one_second_bytes = defaultdict(int)
        with (trial_dir / "receiver_chunks.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                when = int(row["recv_raw_ns"])
                if start_window_ns <= when <= end_ns:
                    value = int(row["onwire_bytes"])
                    per_ue_bytes[int(row["ue_id"])] += value
                    bin_index = min(
                        int(self.config["calibration"]["service_window_s"]) - 1,
                        max(0, int((when - start_window_ns) / 1e9)),
                    )
                    one_second_bytes[bin_index] += value
        window_s = float(self.config["calibration"]["service_window_s"])
        service_bins = [
            one_second_bytes[index] * 8 / 1e6 for index in range(max(1, int(window_s)))
        ]
        mu_hat = median(service_bins)
        if not math.isfinite(mu_hat) or mu_hat <= 0:
            raise StageFailure(f"{trial['id']} could not derive a positive service ceiling")

        trace_root = self.output_dir / "ttracer" / str(trial["id"]) / "ue" / "csv"
        rlc_path = trace_root / "NRUE_MAC_RLC_BUFFER_STATUS.csv"
        grant_path = trace_root / "NRUE_MAC_DCI_GRANT.csv"
        mappings: Dict[int, Counter[int]] = defaultdict(Counter)
        missing_telemetry = [
            str(path) for path in (rlc_path, grant_path) if not path.is_file()
        ]
        if missing_telemetry:
            raise StageFailure(f"missing calibration telemetry: {missing_telemetry}")
        with rlc_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mappings[int(row["ue_id"])][int(row["rnti"])] += 1
        rnti_map = {ue_id: counts.most_common(1)[0][0] for ue_id, counts in mappings.items() if counts}
        if set(rnti_map) != {0, 1} or len(set(rnti_map.values())) != 2:
            raise StageFailure(f"incomplete or duplicate UE/RNTI map in {trial['id']}: {rnti_map}")

        backlog_bins: Dict[int, set[int]] = defaultdict(set)
        with rlc_path.open(newline="", encoding="utf-8") as handle:
            rows_by_sample: Dict[tuple[int, int], int] = defaultdict(int)
            for row in csv.DictReader(handle):
                try:
                    ue_id = int(row["ue_id"])
                    sample_bin = int(parse_trace_time(row["time"]) / 0.05)
                    rows_by_sample[(ue_id, sample_bin)] += int(row["bytes_in_buffer"])
                except (KeyError, ValueError):
                    continue
            for (ue_id, sample_bin), value in rows_by_sample.items():
                if value > int(self.config["transport"]["drain_threshold_bytes"]):
                    backlog_bins[ue_id].add(sample_bin)
        common_backlog = set.intersection(*(backlog_bins[ue] for ue in sorted(rnti_map)))
        longest_run = 0
        current_run = 0
        previous_bin: Optional[int] = None
        for sample_bin in sorted(common_backlog):
            current_run = current_run + 1 if previous_bin is not None and sample_bin == previous_bin + 1 else 1
            longest_run = max(longest_run, current_run)
            previous_bin = sample_bin
        backlogged_s = longest_run * 0.05
        if backlogged_s < float(self.config["calibration"]["minimum_backlogged_s"]):
            raise StageFailure(
                f"{trial['id']} did not keep both UEs backlogged: {backlogged_s:.3f} s"
            )

        tbs_bytes = 0
        times: List[float] = []
        rows: List[dict] = []
        with grant_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["direction"]) != 1 or int(row["rv"]) > 0 or int(row["round"]) > 0:
                    continue
                stamp = parse_trace_time(row["time"])
                times.append(stamp)
                rows.append({"time": stamp, "tbs": int(row["tbs"])})
        if not times:
            raise StageFailure(f"no first-transmission UL grants in {grant_path}")
        end_trace = max(times)
        for row in rows:
            if row["time"] >= end_trace - window_s:
                tbs_bytes += int(row["tbs"])
        tbs_mbps = tbs_bytes * 8 / window_s / 1e6
        conversion = mu_hat / tbs_mbps if tbs_mbps > 0 else float("nan")
        if not 0.25 <= conversion <= 1.25:
            raise StageFailure(f"implausible application/TBS byte-domain conversion: {conversion}")
        achieved_rho = float(self.trial_records[-1]["offered_mbps"]) / mu_hat
        if achieved_rho < float(self.config["calibration"]["minimum_achieved_rho"]):
            raise StageFailure(
                f"{trial['id']} did not load to contend: achieved rho={achieved_rho:.3f}"
            )
        result = {
            "block": block,
            "mu_hat_mbps": mu_hat,
            "per_ue_receiver_mbps": {
                str(ue): value * 8 / window_s / 1e6 for ue, value in per_ue_bytes.items()
            },
            "first_tx_tbs_mbps": tbs_mbps,
            "service_conversion": conversion,
            "rnti_map": rnti_map,
            "achieved_rho": achieved_rho,
            "aggregate_service_1s_mbps": service_bins,
            "both_ues_continuously_backlogged_s": backlogged_s,
        }
        atomic_json(self.output_dir / "blocks" / block / "calibration_summary.json", result)
        self.block_data[block] = result
        self.event("calibration_complete", **result)
        return result

    def d0(self) -> None:
        self.d0_started = True
        prior = float(self.config["calibration"]["prior_ceiling_mbps"])
        smoke = {
            "id": "D0_SMOKE",
            "block": "A",
            "kind": "smoke",
            "fractions": [0.10, 0.10],
            "duration_s": float(self.config["instrumentation"]["smoke_s"]),
        }
        self.run_trial(smoke, mu_hat=prior, payload_bytes=92160, ue_profile="all", gnb_profile="latency")
        perturbation = []
        for trial_id, ue_profile, gnb_profile, require_per_ue_radio in (
            ("D0_TRACE_MIN", "clean", "clean", False),
            ("D0_TRACE_FULL", "all", "latency", True),
        ):
            metrics = self.run_trial(
                {
                    "id": trial_id,
                    "block": "A",
                    "kind": "equal",
                    "rho": 1.0,
                    "duration_s": float(self.config["instrumentation"]["perturbation_s"]),
                },
                mu_hat=prior,
                ue_profile=ue_profile,
                gnb_profile=gnb_profile,
                enforce_strong_rung=require_per_ue_radio,
                require_per_ue_radio=require_per_ue_radio,
            )
            perturbation.append(metrics)
        minimum, full = perturbation
        demand_hash_match = (
            minimum["demand_trace_sha256"] == full["demand_trace_sha256"]
        )
        service_delta = abs(full["receiver_onwire_mbps"] - minimum["receiver_onwire_mbps"]) / max(
            minimum["receiver_onwire_mbps"], 1e-9
        )
        min_p95 = float(minimum["latency_p95_ms"] or 0.0)
        full_p95 = float(full["latency_p95_ms"] or 0.0)
        latency_delta = max(0.0, full_p95 - min_p95) / max(min_p95, 1e-9)
        verdict = {
            "demand_hash_match": demand_hash_match,
            "service_delta_fraction": service_delta,
            "latency_delta_fraction": latency_delta,
            "pass": demand_hash_match
            and service_delta <= float(self.config["instrumentation"]["trace_service_tolerance_fraction"])
            and latency_delta <= float(self.config["instrumentation"]["trace_latency_tolerance_fraction"]),
        }
        atomic_json(self.output_dir / "D0_instrumentation_verdict.json", verdict)
        if not verdict["pass"]:
            raise StageFailure(f"D0 instrumentation perturbation gate failed: {verdict}")

    def run_dg_a(self) -> None:
        trials = {str(row["id"]): row for row in self.config["trials"]}
        self._wait_for_cold_ran()
        self.start_ran("A")
        strong_a = self._enter_registered_strong_rung("A")
        self.d0()
        block_a = self.calibrate("A")
        block_a["strong_rung_entry_gate"] = strong_a
        self.block_data["A"] = block_a
        self.dg_a_trials_started = True
        open_loop_ids = ["A1", "A2", "A3", "A4"]
        random.Random(61300).shuffle(open_loop_ids)
        atomic_json(self.output_dir / "block_a_registered_order.json", open_loop_ids + ["A5", "A6", "A7"])
        for trial_id in open_loop_ids + ["A5", "A6", "A7"]:
            self.run_trial(
                trials[trial_id],
                mu_hat=block_a["mu_hat_mbps"],
                rnti_map=block_a["rnti_map"],
                service_conversion=block_a["service_conversion"],
            )
        self.stop_ran()
        self._wait_for_cold_ran()
        self.start_ran("B")
        strong_b = self._enter_registered_strong_rung("B")
        block_b = self.calibrate("B")
        block_b["strong_rung_entry_gate"] = strong_b
        self.block_data["B"] = block_b
        for trial_id in ("A8", "A9"):
            self.run_trial(
                trials[trial_id],
                mu_hat=block_b["mu_hat_mbps"],
                rnti_map=block_b["rnti_map"],
                service_conversion=block_b["service_conversion"],
            )
        stage_manifest = {
            "schema_version": self.config["schema_version"],
            "stage": self.config["stage"],
            "completed_measurement_at": utc_now(),
            "blocks": self.block_data,
            "trials": self.trial_records,
            "forbidden_next_stage": "DG-B",
        }
        atomic_json(self.output_dir / "stage_manifest.json", stage_manifest)
        self.command(
            [
                str(self.paths["python"]),
                "-m",
                "rl_agent.multiue_oai.analyze",
                "--run-dir",
                str(self.output_dir),
                "--config",
                str(self.config_path),
            ],
            timeout=600,
        )

    def cleanup(self) -> None:
        try:
            self.stop_ran()
        finally:
            if self.started_core and not self.dry_run:
                self.command(
                    ["sudo", "-n", "docker", "compose", "down"],
                    cwd=self.paths["oai_cn_dir"],
                    check=False,
                )

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        run_mode = (
            "runtime_switch_startup_smoke"
            if self.runtime_switch_startup_smoke
            else
            "runtime_switch_smoke"
            if self.runtime_switch_smoke
            else "attach_smoke"
            if self.attach_smoke_repeats
            else "dg_a"
        )
        self.event(
            "stage_start",
            stage=self.config["stage"],
            run_mode=run_mode,
            attach_smoke_repeats=self.attach_smoke_repeats,
            attach_channel_mode=self.attach_channel_mode,
            runtime_switch_smoke=self.runtime_switch_smoke,
            runtime_switch_startup_smoke=self.runtime_switch_startup_smoke,
            pid=os.getpid(),
        )
        def interrupted(signum: int, _frame: object) -> None:
            raise StageFailure(f"received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        completion: Optional[dict] = None
        completion_event: Optional[dict] = None
        failure: Optional[BaseException] = None
        try:
            self.preflight()
            if self.dry_run or self.preflight_only:
                mode_status = "DRY_RUN_PASS" if self.dry_run else "PREFLIGHT_ONLY_PASS"
                atomic_json(
                    self.output_dir / "results_summary.json",
                    {
                        "schema_version": "scenesense.multiue_oai.dg_a.preflight.v1",
                        "status": mode_status,
                        "decision": "NOT_RUN",
                        "oai_started": False,
                        "next_stage_launched": False,
                    },
                )
                completion = {
                    "status": mode_status,
                    "completed_at": utc_now(),
                    "oai_started": False,
                    "summary": str(self.output_dir / "results_summary.json"),
                    "next_stage_launched": False,
                }
                completion_event = {"decision": "NOT_RUN", "run_mode": run_mode}
            else:
                self.start_core()
                if self.runtime_switch_startup_smoke:
                    summary = self.run_runtime_switch_startup_smoke()
                    completion = {
                        "status": "RUNTIME_SWITCH_STARTUP_COMPLETE",
                        "decision": summary["decision"],
                        "completed_at": utc_now(),
                        "summary": str(self.output_dir / "results_summary.json"),
                        "d0_launched": False,
                        "dg_a_launched": False,
                        "next_stage_launched": False,
                    }
                    completion_event = {
                        "decision": summary["decision"],
                        "run_mode": "runtime_switch_startup_smoke",
                    }
                elif self.runtime_switch_smoke:
                    summary = self.run_runtime_switch_smoke()
                    completion = {
                        "status": "RUNTIME_SWITCH_SMOKE_COMPLETE_HUMAN_REVIEW_REQUIRED",
                        "decision": summary["decision"],
                        "completed_at": utc_now(),
                        "summary": str(self.output_dir / "results_summary.json"),
                        "d0_launched": False,
                        "dg_a_launched": False,
                        "next_stage_launched": False,
                    }
                    completion_event = {
                        "decision": summary["decision"],
                        "run_mode": "runtime_switch_smoke",
                    }
                elif self.attach_smoke_repeats:
                    summary = self.run_attach_smoke()
                    completion = {
                        "status": "ATTACH_SMOKE_COMPLETE_HUMAN_REVIEW_REQUIRED",
                        "decision": summary["decision"],
                        "completed_at": utc_now(),
                        "summary": str(self.output_dir / "results_summary.json"),
                        "d0_launched": False,
                        "dg_a_launched": False,
                        "next_stage_launched": False,
                    }
                    completion_event = {
                        "decision": summary["decision"],
                        "run_mode": "attach_smoke",
                        "attach_channel_mode": self.attach_channel_mode,
                    }
                else:
                    self.run_dg_a()
                    summary_path = self.output_dir / "results_summary.json"
                    if not summary_path.exists():
                        raise StageFailure("DG-A.1 analyzer did not write results_summary.json")
                    summary = json.loads(summary_path.read_text())
                    completion = {
                        "status": "DG_A_COMPLETE_HUMAN_REVIEW_REQUIRED",
                        "decision": summary.get("decision"),
                        "completed_at": utc_now(),
                        "summary": str(summary_path),
                        "next_stage_launched": False,
                    }
                    completion_event = {
                        "decision": summary.get("decision"),
                        "run_mode": "dg_a",
                    }
        except BaseException as exc:
            failure = exc
        finally:
            try:
                self.cleanup()
            except BaseException as cleanup_exc:
                if failure is None:
                    failure = cleanup_exc
                else:
                    failure = StageFailure(
                        f"{type(failure).__name__}: {failure}; cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
        if failure is not None:
            record = failure_record(failure, status="FAILED_HOLD")
            record["run_mode"] = run_mode
            record["attach_channel_mode"] = self.attach_channel_mode
            record["d0_launched"] = self.d0_started
            record["dg_a_launched"] = self.dg_a_trials_started
            atomic_json(self.output_dir / "results_summary.json", record)
            atomic_json(self.output_dir / "FAILED.json", record)
            self.event("stage_failed", error_type=type(failure).__name__, error=str(failure))
            return 1
        if completion is None or completion_event is None:
            raise RuntimeError("runner reached terminal state without completion metadata")
        atomic_json(self.output_dir / "COMPLETED.json", completion)
        self.event("stage_complete", **completion_event)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--attach-smoke-repeats",
        type=int,
        default=0,
        help="Run only this many cold two-UE attachment repetitions; never launch D0/DG-A.",
    )
    parser.add_argument(
        "--attach-channel-mode",
        choices=("strong", "clean"),
        default="strong",
        help="RFsim channel used by attach-only smoke; clean mode never enables chanmod.",
    )
    parser.add_argument(
        "--runtime-switch-smoke",
        action="store_true",
        help="Run one bounded clean-to-strong two-UE switch smoke; never launch D0/DG-A.",
    )
    parser.add_argument(
        "--runtime-switch-startup-smoke",
        action="store_true",
        help="Run only initial-clean runtime-control startup/attach; never send decision traffic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    try:
        return Runner(
            Path(args.config),
            output,
            dry_run=args.dry_run,
            preflight_only=args.preflight_only,
            attach_smoke_repeats=args.attach_smoke_repeats,
            attach_channel_mode=args.attach_channel_mode,
            runtime_switch_smoke=args.runtime_switch_smoke,
            runtime_switch_startup_smoke=args.runtime_switch_startup_smoke,
        ).run()
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = failure_record(exc, status="FAILED_BEFORE_STAGE_START")
        atomic_json(output / "results_summary.json", failure)
        atomic_json(output / "FAILED.json", failure)
        print(f"DG-A runner construction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
