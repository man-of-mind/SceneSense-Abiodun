#!/usr/bin/env python3
"""Fail-fast, self-logging orchestrator for DG-A and DG-A.1 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .endpoint import frame_onwire_bytes, raw_ns


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
        self.stop_requested = False

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
            self.paths["t_messages"],
            self.paths["oai_ran_conf"] / active_gnb_config,
            self.paths["oai_ran_conf"] / radio["ue_base_config"],
        ]
        if self.attach_channel_mode == "strong":
            required.append(self.paths["oai_ran_conf"] / radio["channel_config"])
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
                }.items()
            },
            "authorization_boundary": self.config["authorization_boundary"],
            "dry_run": self.dry_run,
            "preflight_only": self.preflight_only,
            "attach_smoke_repeats": self.attach_smoke_repeats,
            "attach_channel_mode": self.attach_channel_mode,
            "active_gnb_config": active_gnb_config,
            "active_ue_config": radio["ue_base_config"],
            "channel_config_sha256": (
                sha256(self.paths["oai_ran_conf"] / radio["channel_config"])
                if self.attach_channel_mode == "strong"
                else None
            ),
        }
        atomic_json(self.output_dir / "run_manifest.json", manifest)
        self._materialize_ue_config()
        self._local_transport_control()
        self.event("preflight_pass")

    def _materialize_ue_config(self) -> None:
        base_path = self.paths["oai_ran_conf"] / self.config["radio"]["ue_base_config"]
        if self.attach_channel_mode == "clean":
            # Reuse the known-good multi-UE contract verbatim. Without the
            # chanmod command-line option, the included model list is not
            # activated by RFsim.
            self.runtime_ue_config = base_path
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
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        if marker not in base:
            raise StageFailure(f"expected channel include missing from {base_path}")
        resolved = base.replace(marker, channel)
        runtime = self.output_dir / "runtime" / "ue.multi2.awgn_strong.conf"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_text(resolved, encoding="utf-8")
        self.runtime_ue_config = runtime

    def _local_transport_control(self) -> None:
        if self.dry_run:
            return
        control = self.output_dir / "preflight" / "local_transport"
        control.mkdir(parents=True, exist_ok=True)
        receiver = subprocess.Popen(
            [
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
                "--max-duration-s",
                "20",
            ],
            cwd=str(ROOT),
            stdout=(control / "receiver.log").open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
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
            receiver.send_signal(signal.SIGTERM)
            try:
                receiver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(receiver.pid, signal.SIGKILL)
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
        self.started_core = "oai-amf" not in existing
        self.command(["sudo", "-n", "docker", "compose", "up", "-d"], cwd=self.paths["oai_cn_dir"])
        deadline = time.monotonic() + 180
        required = {"oai-amf", "oai-smf", "oai-upf", self.config["radio"]["ext_dn_container"]}
        while time.monotonic() < deadline:
            output = self.command(
                ["sudo", "-n", "docker", "ps", "--format", "{{.Names}} {{.Status}}"], check=False
            ).stdout
            present = {line.split()[0] for line in output.splitlines() if line.strip()}
            if required.issubset(present):
                self.event("core_ready", containers=sorted(present))
                return
            time.sleep(3)
        raise StageFailure("core containers did not become ready within 180 s")

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

    def start_ran(self, block: str) -> None:
        self.stop_ran()
        block_dir = self.output_dir / "blocks" / block
        radio = self.config["radio"]
        gnb_config = (
            radio["clean_gnb_config"]
            if self.attach_channel_mode == "clean"
            else radio["gnb_config"]
        )
        gnb_conf = self.paths["oai_ran_conf"] / gnb_config
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
        if self.attach_channel_mode == "strong":
            gnb_cmd[gnb_cmd.index("--T_stdout"):gnb_cmd.index("--T_stdout")] = [
                "--rfsimulator.[0].options",
                "chanmod",
            ]
        self.event("ran_start", block=block, component="gnb", argv=gnb_cmd)
        self.gnb_proc = self._popen_log(gnb_cmd, block_dir / "gnb_stdout.log")
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
        if self.attach_channel_mode == "strong":
            ue_cmd[ue_cmd.index("-r"):ue_cmd.index("-r")] = [
                "--rfsimulator.[0].options",
                "chanmod",
            ]
        self.event("ran_start", block=block, component="ue", argv=ue_cmd)
        self.ue_proc = self._popen_log(ue_cmd, block_dir / "ue_stdout.log")
        self.ran_active = True
        self.wait_tunnels(block)

    def wait_tunnels(self, block: str) -> None:
        deadline = time.monotonic() + float(self.config["radio"]["attach_timeout_s"])
        expected = self.config["radio"]["expected_tunnels"]
        while time.monotonic() < deadline:
            good = True
            for ue in expected:
                result = self.command(
                    ["ip", "-4", "-br", "addr", "show", ue["iface"]], check=False
                )
                if result.returncode or ue["ip"] not in result.stdout:
                    good = False
                    break
            if good:
                for ue in expected:
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
                self.event("ran_attached", block=block, tunnels=expected)
                return
            if self.gnb_proc.poll() is not None or self.ue_proc.poll() is not None:
                raise StageFailure("softmodem exited before both UEs attached")
            time.sleep(3)
        raise StageFailure("both expected UE tunnels did not attach")

    def stop_ran(self) -> None:
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
                    pass
                handle = getattr(proc, "_scenesense_log_handle", None)
                if handle is not None:
                    handle.close()
        self.ran_active = False

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
        samples = []
        for sample_index in range(3):
            if self.gnb_proc.poll() is not None or self.ue_proc.poll() is not None:
                raise StageFailure(f"{block} softmodem exited during attach stability hold")
            per_ue = []
            for ue in self.config["radio"]["expected_tunnels"]:
                address = self.command(
                    ["ip", "-4", "-br", "addr", "show", str(ue["iface"])], check=False
                )
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
                valid_address = address.returncode == 0 and str(ue["ip"]) in address.stdout
                if not valid_address or ping.returncode != 0:
                    raise StageFailure(
                        f"{block} lost UE{ue['ue_id']} during stability hold: "
                        f"address_ok={valid_address}, ping_rc={ping.returncode}"
                    )
                per_ue.append(
                    {
                        "ue_id": int(ue["ue_id"]),
                        "iface": str(ue["iface"]),
                        "ip": str(ue["ip"]),
                        "address": address.stdout.strip(),
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
        for source, profile in (("ue", ue_profile), ("gnb", gnb_profile)):
            log_path = trial_dir / f"ttracer_{source}_record_stdout.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                [
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
                ],
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process._scenesense_log_handle = handle  # type: ignore[attr-defined]
            procs.append(process)
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
        command = [
            str(self.paths["ttracer_dir"] / "csv"),
            "-d",
            str(self.paths["t_messages"]),
            "-ip",
            "127.0.0.1",
            "-p",
            str(self.config["radio"]["ue_ttracer_port"]),
            "-f",
            "-s",
            ",",
            "-t",
            "time",
            "-OFF",
            "-on",
            "NRUE_MAC_RLC_BUFFER_STATUS",
            "NRUE_MAC_RLC_BUFFER_STATUS",
            "time",
            "rnti",
            "ue_id",
            "lcid",
            "bytes_in_buffer",
        ]
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

    def _radio_and_grant_validity(self, trial_id: str) -> dict:
        root = self.output_dir / "ttracer" / trial_id
        power_path = root / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv"
        snr_values: List[float] = []
        mcs_values: List[float] = []
        if power_path.exists():
            with power_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        snr_values.append(float(row["snrx10"]) / 10.0)
                        mcs_values.append(float(row["mcs"]))
                    except (KeyError, ValueError):
                        continue
        observed_snr = median(snr_values)
        observed_mcs = median(mcs_values)
        if not math.isfinite(observed_snr) or not math.isfinite(observed_mcs):
            raise StageFailure(f"{trial_id} lacks populated gNB PUSCH SNR/MCS telemetry")
        radio = self.config["radio"]
        if abs(observed_snr - float(radio["expected_snr_db"])) > float(radio["snr_tolerance_db"]):
            raise StageFailure(
                f"{trial_id} SNR off registered strong rung: observed={observed_snr:.3f} dB"
            )
        if abs(observed_mcs - float(radio["expected_mcs"])) > float(radio["mcs_tolerance"]):
            raise StageFailure(
                f"{trial_id} MCS off registered strong rung: observed={observed_mcs:.3f}"
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
            "median_pusch_snr_db": observed_snr,
            "median_ul_mcs": observed_mcs,
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
        }
        atomic_json(trial_dir / "trial_manifest.json", manifest)
        self.event("trial_start", trial_id=trial_id, trial=trial)
        receiver = self._start_receiver(trial_dir, total_trace_s + 30)
        recorders = self._trace_processes(trial_id, trial_dir, total_trace_s, ue_profile, gnb_profile)
        sampler_handle = (trial_dir / "network_sampler_stdout.log").open("wb")
        sampler = subprocess.Popen(
            [
                str(self.paths["python"]),
                str(ROOT / "scripts/sample_oai_network_metrics.py"),
                "--run-group",
                trial_id,
                "--duration-s",
                str(total_trace_s),
                "--output-dir",
                str(trial_dir / "network"),
                "--interface",
                "oaitun_ue1:ue0",
                "--interface",
                "oaitun_ue2:ue1",
            ],
            cwd=str(ROOT),
            stdout=sampler_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        recorder_failures: List[str] = []
        sampler_failure: Optional[str] = None
        try:
            time.sleep(pre_wait_s)
            transport = self.config["transport"]
            sender = [
                str(self.paths["python"]),
                "-m",
                "rl_agent.multiue_oai.endpoint",
                "send",
                "--remote-host",
                self.config["radio"]["ext_dn_ip"],
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
                str(duration),
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
            for ue, fraction in zip(self.config["radio"]["expected_tunnels"], fractions):
                sender.extend(["--ue", f"{ue['ue_id']},{ue['ip']},{fraction}"])
            for phase in trial.get("phases", []):
                sender.extend(["--phase", ",".join(str(value) for value in phase)])
            if "demand_seed" in trial:
                sender.extend(["--demand-seed", str(trial["demand_seed"])])
            if trial.get("controller", "open_loop") != "open_loop":
                if not rnti_map or set(rnti_map) != {0, 1}:
                    raise StageFailure(f"{trial_id} lacks a complete frozen UE-to-RNTI map")
                for ue_id, rnti in sorted(rnti_map.items()):
                    sender.extend(["--rnti-map", f"{ue_id},{rnti}"])
                sender.extend(
                    [
                        "--ttracer-csv",
                        str(self.paths["ttracer_dir"] / "csv"),
                        "--t-messages",
                        str(self.paths["t_messages"]),
                        "--ttracer-port",
                        str(self.config["radio"]["ue_ttracer_port"]),
                    ]
                )
            sender_log = (trial_dir / "sender_stdout.log").open("w", encoding="utf-8")
            result = subprocess.run(
                sender,
                cwd=str(ROOT),
                text=True,
                stdout=sender_log,
                stderr=subprocess.STDOUT,
                timeout=duration + 30,
                check=False,
            )
            sender_log.close()
            if result.returncode:
                raise StageFailure(f"sender failed for {trial_id} with {result.returncode}")
            time.sleep(post_wait_s)
        finally:
            self._stop_root_process(receiver)
            for process in recorders:
                try:
                    process.wait(timeout=total_trace_s + 15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=5)
                if process.returncode:
                    recorder_failures.append(
                        f"pid={process.pid},returncode={process.returncode}"
                    )
                handle = getattr(process, "_scenesense_log_handle", None)
                if handle is not None:
                    handle.close()
            if sampler.poll() is None:
                os.killpg(sampler.pid, signal.SIGINT)
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(sampler.pid, signal.SIGKILL)
                sampler.wait(timeout=3)
            if sampler.returncode:
                sampler_failure = f"returncode={sampler.returncode}"
            sampler_handle.close()
            self.command(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(trial_dir)],
                check=False,
            )
        if recorder_failures:
            raise StageFailure(f"{trial_id} trace recorder failure: {recorder_failures}")
        if sampler_failure:
            raise StageFailure(f"{trial_id} network sampler failure: {sampler_failure}")
        self._extract_traces(trial_id, ue_profile, gnb_profile)
        metrics = self._trial_metrics(trial, trial_dir, mu_hat)
        metrics["radio_validity"] = self._radio_and_grant_validity(trial_id)
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
        if metrics["local_errors"] or metrics["checksum_failures"]:
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
                tunnel_health[str(label)] = {"drops_or_errors": drops}
        metrics["tunnel_health"] = tunnel_health
        if len(tunnel_health) != 2 or any(
            row["drops_or_errors"] for row in tunnel_health.values()
        ):
            raise StageFailure(f"{trial_id} tunnel validity failure: {tunnel_health}")
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
                per_ue_rate_gate[ue_id] = {
                    "target_mbps": target,
                    "actual_mbps": actual,
                    "deviation_fraction": deviation,
                }
            metrics["open_loop_rate_gate"] = per_ue_rate_gate
            if any(row["deviation_fraction"] > tolerance for row in per_ue_rate_gate.values()):
                raise StageFailure(
                    f"{trial_id} open-loop sender rate exceeds {tolerance:.1%} tolerance: "
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
        if not rlc_path.exists():
            raise StageFailure(f"missing calibration RLC mapping: {rlc_path}")
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
        for trial_id, ue_profile, gnb_profile in (
            ("D0_TRACE_MIN", "clean", "clean"),
            ("D0_TRACE_FULL", "all", "latency"),
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
            )
            perturbation.append(metrics)
        minimum, full = perturbation
        service_delta = abs(full["receiver_onwire_mbps"] - minimum["receiver_onwire_mbps"]) / max(
            minimum["receiver_onwire_mbps"], 1e-9
        )
        min_p95 = float(minimum["latency_p95_ms"] or 0.0)
        full_p95 = float(full["latency_p95_ms"] or 0.0)
        latency_delta = max(0.0, full_p95 - min_p95) / max(min_p95, 1e-9)
        verdict = {
            "service_delta_fraction": service_delta,
            "latency_delta_fraction": latency_delta,
            "pass": service_delta <= float(self.config["instrumentation"]["trace_service_tolerance_fraction"])
            and latency_delta <= float(self.config["instrumentation"]["trace_latency_tolerance_fraction"]),
        }
        atomic_json(self.output_dir / "D0_instrumentation_verdict.json", verdict)
        if not verdict["pass"]:
            raise StageFailure(f"D0 instrumentation perturbation gate failed: {verdict}")

    def run_dg_a(self) -> None:
        trials = {str(row["id"]): row for row in self.config["trials"]}
        self.start_ran("A")
        self.d0()
        block_a = self.calibrate("A")
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
        self.start_ran("B")
        block_b = self.calibrate("B")
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
        run_mode = "attach_smoke" if self.attach_smoke_repeats else "dg_a"
        self.event(
            "stage_start",
            stage=self.config["stage"],
            run_mode=run_mode,
            attach_smoke_repeats=self.attach_smoke_repeats,
            attach_channel_mode=self.attach_channel_mode,
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
                if self.attach_smoke_repeats:
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
            record["d0_launched"] = False if self.attach_smoke_repeats else None
            record["dg_a_launched"] = False if self.attach_smoke_repeats else None
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
