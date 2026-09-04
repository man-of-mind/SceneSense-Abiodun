#!/usr/bin/env python3
"""Autonomous clean-stack OAI bandwidth/TDD iperf sweep.

Each rate/repetition is isolated by a complete CN, gNB and UE teardown.  The
runner is intentionally operational rather than a generic OAI framework: its
four radio recipes and its measurement ladder are frozen in configs/sweep_v1.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = PACKAGE_DIR / "configs" / "sweep_v1.json"
OAI_ROOT = REPO_ROOT / "OAI"
CN_DIR = OAI_ROOT / "oai-cn5g"
RAN_ROOT = OAI_ROOT / "openairinterface5g"
RAN_BUILD = RAN_ROOT / "cmake_targets" / "ran_build" / "build"
RAN_CONF = RAN_ROOT / "targets" / "PROJECTS" / "GENERIC-NR-5GC" / "CONF"
TTRACER_RECORD = REPO_ROOT / "scripts" / "ttracer_record_smoke.sh"
TTRACER_EXTRACT = REPO_ROOT / "scripts" / "ttracer_extract_csv_smoke.sh"
GRANT_ANALYZER = REPO_ROOT / "scripts" / "analyze_nrue_grant_metrics.py"
CLIENT_LABEL = "scenesense.oai_tdd_sweep=1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


class CommandError(RuntimeError):
    pass


class SweepRunner:
    def __init__(self, mode: str, experiment_dir: Optional[Path]) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.mode = mode
        self._sudo_stop = threading.Event()
        self._sudo_thread: Optional[threading.Thread] = None
        self.local_processes: List[subprocess.Popen[Any]] = []
        self.current_cell: Optional[Path] = None

        if experiment_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = REPO_ROOT / self.config["output_root"]
            self.experiment_dir = output_root / f"{stamp}_{mode}"
            if self.experiment_dir.exists():
                raise FileExistsError(self.experiment_dir)
            self.experiment_dir.mkdir(parents=True)
            self.cells_dir = self.experiment_dir / "cells"
            self.cells_dir.mkdir()
            self.log_path = self.experiment_dir / "run.log"
            atomic_json(self.experiment_dir / "CONTRACT_SNAPSHOT.json", self.config)
            self.snapshot_configs()
            self.plan = self.build_plan()
            atomic_json(self.experiment_dir / "PLAN.json", self.plan)
            atomic_json(self.experiment_dir / "MANIFEST.json", self.build_manifest())
            atomic_json(
                self.experiment_dir / "STATUS.json",
                {"phase": "created", "mode": mode, "created_at": utc_now()},
            )
        else:
            self.experiment_dir = experiment_dir.resolve()
            manifest = json.loads((self.experiment_dir / "MANIFEST.json").read_text())
            contract_snapshot = self.experiment_dir / "CONTRACT_SNAPSHOT.json"
            if sha256(contract_snapshot) != manifest["contract_snapshot_sha256"]:
                raise ValueError("experiment contract snapshot hash mismatch")
            self.config = json.loads(contract_snapshot.read_text(encoding="utf-8"))
            self.mode = str(manifest["mode"])
            self.cells_dir = self.experiment_dir / "cells"
            self.log_path = self.experiment_dir / "run.log"
            self.plan = json.loads((self.experiment_dir / "PLAN.json").read_text())
            self.verify_snapshots()

    def say(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def build_manifest(self) -> Dict[str, Any]:
        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
            )
            return result.stdout.strip()

        return {
            "schema_version": 1,
            "experiment": self.config["experiment_name"],
            "mode": self.mode,
            "created_at": utc_now(),
            "repo_root": str(REPO_ROOT),
            "git_head": git("rev-parse", "HEAD"),
            "git_status_porcelain": git("status", "--short"),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "contract": str(CONFIG_PATH),
            "contract_sha256": sha256(CONFIG_PATH),
            "contract_snapshot": str(self.experiment_dir / "CONTRACT_SNAPSHOT.json"),
            "contract_snapshot_sha256": sha256(
                self.experiment_dir / "CONTRACT_SNAPSHOT.json"
            ),
            "subscriber_sql_sha256": sha256(CN_DIR / "database" / "oai_db.sql"),
            "config_snapshots": self.config_snapshot_manifest(),
            "notes": [
                "No CARLA or model process is launched.",
                "The core is recreated without deleting volumes before every cell.",
                "QoS is not changed; the existing subscriber/core configuration is shared.",
                "iperf3 runs from the ext-DN image because the host has no iperf binary.",
            ],
        }

    def snapshot_configs(self) -> None:
        snapshots = self.experiment_dir / "config_snapshots"
        snapshots.mkdir()
        configs = {item["id"]: item for item in self.config["radio"]["configurations"]}
        for item in self.config["radio"]["configurations"]:
            target = snapshots / f"{item['id']}.conf"
            if "gnb_source" in item:
                source = RAN_CONF / item["gnb_source"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copyfile(source, target)
            else:
                source_id = item["derive_from"]
                source = snapshots / f"{source_id}.conf"
                text = source.read_text(encoding="utf-8")
                text, dl_count = re.subn(
                    r"(nrofDownlinkSlots\s*=\s*)7(\s*;)", r"\g<1>4\2", text
                )
                text, ul_count = re.subn(
                    r"(nrofUplinkSlots\s*=\s*)2(\s*;)", r"\g<1>5\2", text
                )
                if (dl_count, ul_count) != (1, 1):
                    raise ValueError(
                        f"could not derive {item['id']}: replacements={dl_count}/{ul_count}"
                    )
                target.write_text(text, encoding="utf-8")

        for item in configs.values():
            self.validate_radio_config(snapshots / f"{item['id']}.conf", item)
        base = (snapshots / "bw100_tdd7d2u.conf").read_text().splitlines()
        derived = (snapshots / "bw100_tdd4d5u.conf").read_text().splitlines()
        differences = [(a, b) for a, b in zip(base, derived) if a != b]
        if len(base) != len(derived) or len(differences) != 2:
            raise ValueError("273-PRB UL-heavy config must differ in exactly two lines")
        if not all("nrofDownlinkSlots" in a or "nrofUplinkSlots" in a for a, _ in differences):
            raise ValueError("unexpected non-TDD change in derived 273-PRB config")

    def config_snapshot_manifest(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in self.config["radio"]["configurations"]:
            path = self.experiment_dir / "config_snapshots" / f"{item['id']}.conf"
            result[item["id"]] = {
                "path": str(path),
                "sha256": sha256(path),
                "source": item.get("gnb_source"),
                "derived_from": item.get("derive_from"),
            }
        return result

    def verify_snapshots(self) -> None:
        manifest = json.loads((self.experiment_dir / "MANIFEST.json").read_text())
        configs = {item["id"]: item for item in self.config["radio"]["configurations"]}
        for config_id, record in manifest["config_snapshots"].items():
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise ValueError(f"config snapshot hash mismatch: {path}")
            self.validate_radio_config(path, configs[config_id])

    @staticmethod
    def config_scalar(text: str, name: str) -> int:
        matches = re.findall(rf"\b{re.escape(name)}\s*=\s*([0-9]+)\s*;", text)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {name}, found {len(matches)}")
        return int(matches[0])

    def validate_radio_config(self, path: Path, item: Dict[str, Any]) -> None:
        text = path.read_text(encoding="utf-8")
        actual = {
            "dl_carrierBandwidth": self.config_scalar(text, "dl_carrierBandwidth"),
            "ul_carrierBandwidth": self.config_scalar(text, "ul_carrierBandwidth"),
            "nrofDownlinkSlots": self.config_scalar(text, "nrofDownlinkSlots"),
            "nrofUplinkSlots": self.config_scalar(text, "nrofUplinkSlots"),
        }
        expected = {
            "dl_carrierBandwidth": int(item["prb"]),
            "ul_carrierBandwidth": int(item["prb"]),
            "nrofDownlinkSlots": int(item["downlink_slots"]),
            "nrofUplinkSlots": int(item["uplink_slots"]),
        }
        if actual != expected:
            raise ValueError(f"radio config mismatch {path}: {actual} != {expected}")

    def build_plan(self) -> Dict[str, Any]:
        cells: List[Dict[str, Any]] = []
        configurations = list(self.config["radio"]["configurations"])
        mode_cfg = self.config[self.mode]
        repetitions = int(mode_cfg["repetitions"])

        for repetition in range(1, repetitions + 1):
            order = configurations if repetition % 2 else list(reversed(configurations))
            for config_item in order:
                rates = list(mode_cfg["rates_mbps"])
                if repetition % 2 == 0:
                    rates.reverse()
                for rate in rates:
                    cells.append(
                        self.make_cell(config_item, repetition, "udp", int(rate), None)
                    )

                if self.mode == "full":
                    parent_rate = int(mode_cfg["rates_mbps"][-1])
                    for rate in mode_cfg["extension_rates_mbps"]:
                        parent_id = self.cell_id(
                            config_item["id"], repetition, "udp", parent_rate
                        )
                        cells.append(
                            self.make_cell(
                                config_item, repetition, "udp", int(rate), parent_id
                            )
                        )
                        parent_rate = int(rate)
                    if mode_cfg.get("tcp_once_per_configuration_repetition", False):
                        cells.append(self.make_cell(config_item, repetition, "tcp", None, None))

        return {
            "schema_version": 1,
            "mode": self.mode,
            "created_at": utc_now(),
            "cells": cells,
            "cell_count_including_conditional": len(cells),
        }

    @staticmethod
    def cell_id(config_id: str, repetition: int, protocol: str, rate: Optional[int]) -> str:
        suffix = "max" if rate is None else f"{rate:03d}mbps"
        return f"r{repetition:02d}_{config_id}_{protocol}_{suffix}"

    def make_cell(
        self,
        config_item: Dict[str, Any],
        repetition: int,
        protocol: str,
        rate: Optional[int],
        conditional_parent: Optional[str],
    ) -> Dict[str, Any]:
        mode_cfg = self.config[self.mode]
        if protocol == "tcp":
            duration = int(mode_cfg["tcp_duration_s"])
            omit = int(mode_cfg["tcp_omit_s"])
        else:
            duration = int(mode_cfg["duration_s"])
            omit = int(mode_cfg["omit_s"])
        return {
            "id": self.cell_id(config_item["id"], repetition, protocol, rate),
            "configuration": config_item["id"],
            "bandwidth_mhz": int(config_item["bandwidth_mhz"]),
            "prb": int(config_item["prb"]),
            "downlink_slots": int(config_item["downlink_slots"]),
            "uplink_slots": int(config_item["uplink_slots"]),
            "repetition": repetition,
            "protocol": protocol,
            "offered_mbps": rate,
            "omit_s": omit,
            "duration_s": duration,
            "conditional_parent": conditional_parent,
        }

    def ensure_sudo(self) -> None:
        if subprocess.run(["sudo", "-n", "true"], check=False).returncode != 0:
            if not sys.stdin.isatty():
                raise CommandError("sudo credential unavailable; run `sudo -v` in a terminal first")
            self.say("sudo authentication is required once for the autonomous sweep")
            subprocess.run(["sudo", "-v"], check=True)

        def keepalive() -> None:
            while not self._sudo_stop.wait(45):
                subprocess.run(
                    ["sudo", "-n", "-v"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

        self._sudo_thread = threading.Thread(target=keepalive, daemon=True)
        self._sudo_thread.start()

    def stop_sudo_keepalive(self) -> None:
        self._sudo_stop.set()
        if self._sudo_thread is not None:
            self._sudo_thread.join(timeout=2)

    def command(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        check: bool = True,
        log: Optional[Path] = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            list(args),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log is not None:
            atomic_text(
                log,
                f"$ {' '.join(args)}\n\n[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}",
            )
        if check and result.returncode != 0:
            raise CommandError(
                f"command failed rc={result.returncode}: {' '.join(args)}; log={log}"
            )
        return result

    def sudo(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        check: bool = True,
        log: Optional[Path] = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.command(
            ["sudo", "-n", *args], cwd=cwd, timeout=timeout, check=check, log=log
        )

    def preflight(self) -> Dict[str, Any]:
        required = ["sudo", "docker", "ip", "ping", "ss", "git"]
        missing = [name for name in required if shutil.which(name) is None]
        paths = [
            CN_DIR / "docker-compose.yaml",
            CN_DIR / "database" / "oai_db.sql",
            RAN_BUILD / "nr-softmodem",
            RAN_BUILD / "nr-uesoftmodem",
            RAN_CONF / self.config["ue"]["config"],
            TTRACER_RECORD,
            TTRACER_EXTRACT,
            GRANT_ANALYZER,
        ]
        missing_paths = [str(path) for path in paths if not path.exists()]
        sql = (CN_DIR / "database" / "oai_db.sql").read_text(encoding="utf-8")
        subscriber_ok = (
            self.config["ue"]["imsi"] in sql and self.config["ue"]["static_ip"] in sql
        )
        subscriber_row = next(
            (
                line
                for line in sql.splitlines()
                if self.config["ue"]["imsi"] in line
                and self.config["ue"]["static_ip"] in line
            ),
            "",
        )
        five_qi_match = re.search(r'\\"5qi\\"\s*:\s*([0-9]+)', subscriber_row)
        result = {
            "checked_at": utc_now(),
            "commands": {name: shutil.which(name) for name in required},
            "missing_commands": missing,
            "missing_paths": missing_paths,
            "subscriber_static_ip_source_ok": subscriber_ok,
            "subscriber_imsi": self.config["ue"]["imsi"],
            "expected_ue_ip": self.config["ue"]["static_ip"],
            "configured_5qi_unchanged": (
                int(five_qi_match.group(1)) if five_qi_match else None
            ),
            "host_iperf3": shutil.which("iperf3"),
            "host_iperf": shutil.which("iperf"),
            "client_strategy": "oai-ext-dn image with Docker host networking",
        }
        result["pass"] = not missing and not missing_paths and subscriber_ok
        atomic_json(self.experiment_dir / "PREFLIGHT.json", result)
        if not result["pass"]:
            raise CommandError(f"preflight failed: {result}")
        return result

    def start_process(
        self, args: Sequence[str], stdout_path: Path, *, cwd: Optional[Path] = None
    ) -> subprocess.Popen[Any]:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        handle = stdout_path.open("ab", buffering=0)
        process = subprocess.Popen(
            list(args),
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        setattr(process, "_scenesense_log_handle", handle)
        self.local_processes.append(process)
        return process

    def stop_process(self, process: Optional[subprocess.Popen[Any]]) -> None:
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, 2)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, 15)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, 9)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)
        handle = getattr(process, "_scenesense_log_handle", None)
        if handle is not None:
            handle.close()
        if process in self.local_processes:
            self.local_processes.remove(process)

    def teardown(self, cell_dir: Path, label: str) -> Dict[str, Any]:
        logs = cell_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        for process in list(self.local_processes):
            self.stop_process(process)

        cleanup_commands: List[Dict[str, Any]] = []

        def best_effort(args: Sequence[str], name: str, cwd: Optional[Path] = None) -> None:
            result = self.sudo(args, cwd=cwd, check=False, timeout=60)
            cleanup_commands.append(
                {"name": name, "args": list(args), "returncode": result.returncode}
            )

        best_effort(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={CLIENT_LABEL}",
            ],
            "find_client_containers",
        )
        ids = self.sudo(
            ["docker", "ps", "-aq", "--filter", f"label={CLIENT_LABEL}"],
            check=False,
        ).stdout.split()
        if ids:
            best_effort(["docker", "rm", "-f", *ids], "remove_client_containers")

        best_effort(["pkill", "-INT", "-x", "nr-uesoftmodem"], "stop_ue_int")
        best_effort(["pkill", "-INT", "-x", "nr-softmodem"], "stop_gnb_int")
        time.sleep(float(self.config["cleanup"]["grace_s"]))
        best_effort(["pkill", "-TERM", "-x", "nr-uesoftmodem"], "stop_ue_term")
        best_effort(["pkill", "-TERM", "-x", "nr-softmodem"], "stop_gnb_term")
        time.sleep(1)
        best_effort(["pkill", "-KILL", "-x", "nr-uesoftmodem"], "stop_ue_kill")
        best_effort(["pkill", "-KILL", "-x", "nr-softmodem"], "stop_gnb_kill")

        iface = self.config["ue"]["interface"]
        if self.command(["ip", "link", "show", iface], check=False).returncode == 0:
            best_effort(["ip", "link", "delete", iface], "delete_stale_ue_tunnel")

        down_timeout = str(self.config["cleanup"]["docker_down_timeout_s"])
        best_effort(
            ["docker", "compose", "down", "--remove-orphans", "--timeout", down_timeout],
            "core_compose_down",
            CN_DIR,
        )

        ran_alive = any(
            self.command(["pgrep", "-x", name], check=False).returncode == 0
            for name in ("nr-softmodem", "nr-uesoftmodem")
        )
        tunnel_alive = self.command(["ip", "link", "show", iface], check=False).returncode == 0
        core_ids = self.sudo(
            ["docker", "compose", "ps", "-q"], cwd=CN_DIR, check=False
        ).stdout.split()
        client_ids = self.sudo(
            ["docker", "ps", "-aq", "--filter", f"label={CLIENT_LABEL}"], check=False
        ).stdout.split()
        result = {
            "label": label,
            "completed_at": utc_now(),
            "duration_s": time.monotonic() - started,
            "commands": cleanup_commands,
            "ran_processes_absent": not ran_alive,
            "ue_tunnel_absent": not tunnel_alive,
            "core_containers_absent": not core_ids,
            "client_containers_absent": not client_ids,
        }
        result["pass"] = all(
            result[key]
            for key in (
                "ran_processes_absent",
                "ue_tunnel_absent",
                "core_containers_absent",
                "client_containers_absent",
            )
        )
        atomic_json(logs / f"cleanup_{clean_id(label)}.json", result)
        return result

    def wait_core(self, cell_dir: Path) -> Dict[str, Any]:
        deadline = time.monotonic() + float(self.config["radio"]["core_timeout_s"])
        services_result = self.sudo(
            ["docker", "compose", "config", "--services"], cwd=CN_DIR
        )
        services = [line.strip() for line in services_result.stdout.splitlines() if line.strip()]
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            states: Dict[str, Any] = {}
            all_ready = True
            for service in services:
                cid = self.sudo(
                    ["docker", "compose", "ps", "-q", service], cwd=CN_DIR, check=False
                ).stdout.strip()
                if not cid:
                    states[service] = {"running": False, "health": None}
                    all_ready = False
                    continue
                inspect = self.sudo(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                        cid,
                    ],
                    check=False,
                ).stdout.strip().split()
                running = bool(inspect and inspect[0] == "true")
                health = inspect[1] if len(inspect) > 1 else None
                states[service] = {"running": running, "health": health, "container_id": cid}
                if not running or health not in (None, "none", "healthy"):
                    all_ready = False
            last = states
            if all_ready:
                result = {"pass": True, "services": states, "ready_at": utc_now()}
                atomic_json(cell_dir / "logs" / "core_ready.json", result)
                time.sleep(3)
                return result
            time.sleep(2)
        result = {"pass": False, "services": last, "failed_at": utc_now()}
        atomic_json(cell_dir / "logs" / "core_ready.json", result)
        self.sudo(
            ["docker", "compose", "logs", "--no-color", "--tail", "200"],
            cwd=CN_DIR,
            check=False,
            log=cell_dir / "logs" / "core_failure.log",
        )
        raise CommandError("core did not become ready")

    def start_core(self, cell_dir: Path) -> Dict[str, Any]:
        self.sudo(
            ["docker", "compose", "up", "-d", "--force-recreate", "--remove-orphans"],
            cwd=CN_DIR,
            timeout=180,
            log=cell_dir / "logs" / "core_up.log",
        )
        return self.wait_core(cell_dir)

    def start_gnb(self, cell: Dict[str, Any], cell_dir: Path, stack_attempt: int) -> subprocess.Popen[Any]:
        conf = self.experiment_dir / "config_snapshots" / f"{cell['configuration']}.conf"
        args = [
            "sudo",
            "-n",
            "env",
            "-u",
            "SCENESENSE_FORCE_UL_MCS",
            "-u",
            "SCENESENSE_HOLD_MCS_FEW_SAMPLES",
            "-u",
            "SCENESENSE_MCS_POLICY",
            "-u",
            "SCENESENSE_AIMD_MAX_DROP",
            str(RAN_BUILD / "nr-softmodem"),
            "-O",
            str(conf),
            "--gNBs.[0].min_rxtxtime",
            str(self.config["radio"]["min_rxtxtime"]),
            "--rfsim",
            "--T_stdout",
            "2",
            "--T_nowait",
            "--T_port",
            "2021",
        ]
        process = self.start_process(
            args,
            cell_dir / "logs" / f"gnb_stack{stack_attempt}.log",
            cwd=RAN_BUILD,
        )
        wait_s = float(self.config["radio"]["gnb_startup_wait_s"])
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise CommandError(f"gNB exited during startup rc={process.returncode}")
            time.sleep(1)
        return process

    def stop_ue_only(self, ue_process: Optional[subprocess.Popen[Any]]) -> None:
        self.stop_process(ue_process)
        self.sudo(["pkill", "-INT", "-x", "nr-uesoftmodem"], check=False)
        time.sleep(2)
        self.sudo(["pkill", "-KILL", "-x", "nr-uesoftmodem"], check=False)
        iface = self.config["ue"]["interface"]
        if self.command(["ip", "link", "show", iface], check=False).returncode == 0:
            self.sudo(["ip", "link", "delete", iface], check=False)

    def start_ue(
        self, cell: Dict[str, Any], cell_dir: Path, stack_attempt: int, ue_attempt: int
    ) -> subprocess.Popen[Any]:
        configs = {item["id"]: item for item in self.config["radio"]["configurations"]}
        radio = configs[cell["configuration"]]
        args = [
            "sudo",
            "-n",
            str(RAN_BUILD / "nr-uesoftmodem"),
            "--rfsim",
            "--rfsimulator.[0].serveraddr",
            self.config["ue"]["rfsim_server"],
            "-r",
            str(radio["prb"]),
            "--numerology",
            str(self.config["ue"]["numerology"]),
            "--band",
            str(self.config["ue"]["band"]),
            "-C",
            str(radio["ue_frequency_hz"]),
        ]
        if radio.get("ue_ssb") is not None:
            args.extend(["--ssb", str(radio["ue_ssb"])])
        args.extend(
            [
                "-O",
                str(RAN_CONF / self.config["ue"]["config"]),
                "--T_stdout",
                "2",
                "--T_nowait",
                "--T_port",
                "2023",
            ]
        )
        return self.start_process(
            args,
            cell_dir
            / "logs"
            / f"ue_stack{stack_attempt}_attempt{ue_attempt}.log",
            cwd=RAN_BUILD,
        )

    def wait_attach(self, ue_process: subprocess.Popen[Any], cell_dir: Path) -> Dict[str, Any]:
        iface = self.config["ue"]["interface"]
        ue_ip = self.config["ue"]["static_ip"]
        ext_ip = self.config["ue"]["ext_dn_ip"]
        deadline = time.monotonic() + float(self.config["radio"]["attach_timeout_s"])
        while time.monotonic() < deadline:
            if ue_process.poll() is not None:
                return {"pass": False, "reason": f"UE exited rc={ue_process.returncode}"}
            result = self.command(["ip", "-j", "-4", "addr", "show", "dev", iface], check=False)
            if result.returncode == 0:
                try:
                    records = json.loads(result.stdout)
                except json.JSONDecodeError:
                    records = []
                addresses = [
                    info.get("local")
                    for record in records
                    for info in record.get("addr_info", [])
                    if info.get("family") == "inet"
                ]
                if ue_ip in addresses:
                    route = self.command(
                        ["ip", "route", "get", ext_ip, "from", ue_ip], check=False
                    )
                    ping = self.command(
                        ["ping", "-I", iface, "-c", "3", "-W", "1", ext_ip], check=False
                    )
                    ext_route = self.sudo(
                        ["docker", "exec", "oai-ext-dn", "ip", "route"], check=False
                    )
                    ext_route_ok = bool(
                        re.search(
                            r"\b10\.0\.0\.0/16\s+via\s+192\.168\.70\.134\b",
                            ext_route.stdout,
                        )
                    )
                    attached = {
                        "pass": route.returncode == 0 and ping.returncode == 0 and ext_route_ok,
                        "interface": iface,
                        "addresses": addresses,
                        "expected_ip": ue_ip,
                        "route": route.stdout.strip(),
                        "ext_dn_route": ext_route.stdout.strip(),
                        "ext_dn_ue_subnet_route_ok": ext_route_ok,
                        "ping_returncode": ping.returncode,
                        "ping_stdout": ping.stdout,
                        "verified_at": utc_now(),
                    }
                    atomic_json(cell_dir / "logs" / "attach_verification.json", attached)
                    if attached["pass"]:
                        return attached
            time.sleep(2)
        result = {"pass": False, "reason": "attach timeout", "failed_at": utc_now()}
        atomic_json(cell_dir / "logs" / "attach_verification.json", result)
        return result

    def bring_up(self, cell: Dict[str, Any], cell_dir: Path) -> Dict[str, Any]:
        stack_attempts = int(self.config["radio"]["stack_attempts_per_cell"])
        ue_attempts = int(self.config["radio"]["ue_attempts_per_stack"])
        failures: List[Dict[str, Any]] = []
        for stack_attempt in range(1, stack_attempts + 1):
            cleanup = self.teardown(cell_dir, f"pre_stack_{stack_attempt}")
            if not cleanup["pass"]:
                failures.append({"stack_attempt": stack_attempt, "reason": "pre-cleanup failed"})
                continue
            try:
                self.start_core(cell_dir)
                self.start_gnb(cell, cell_dir, stack_attempt)
                for ue_attempt in range(1, ue_attempts + 1):
                    ue = self.start_ue(cell, cell_dir, stack_attempt, ue_attempt)
                    attached = self.wait_attach(ue, cell_dir)
                    if attached["pass"]:
                        return {
                            "pass": True,
                            "stack_attempt": stack_attempt,
                            "ue_attempt": ue_attempt,
                            "attach": attached,
                        }
                    failures.append(
                        {
                            "stack_attempt": stack_attempt,
                            "ue_attempt": ue_attempt,
                            "reason": attached.get("reason", "attach verification failed"),
                        }
                    )
                    self.stop_ue_only(ue)
                    time.sleep(4)
            except Exception as exc:  # Preserve evidence and advance to bounded retry.
                failures.append({"stack_attempt": stack_attempt, "reason": repr(exc)})
            self.teardown(cell_dir, f"failed_stack_{stack_attempt}")
        raise CommandError(f"bounded attach attempts exhausted: {failures}")

    def reset_iperf_server(self, cell_dir: Path) -> str:
        image = self.sudo(
            ["docker", "inspect", "-f", "{{.Config.Image}}", "oai-ext-dn"]
        ).stdout.strip()
        if not image:
            raise CommandError("could not resolve oai-ext-dn image")
        command = (
            "pkill -x iperf3 >/dev/null 2>&1 || true; "
            f"iperf3 -s -D -B {self.config['ue']['ext_dn_ip']} "
            f"-p {self.config['ue']['iperf_port']}"
        )
        self.sudo(
            ["docker", "exec", "oai-ext-dn", "sh", "-lc", command],
            log=cell_dir / "logs" / "iperf_server_start.log",
        )
        time.sleep(1)
        probe = self.sudo(
            ["docker", "exec", "oai-ext-dn", "pgrep", "-x", "iperf3"], check=False
        )
        if probe.returncode != 0:
            raise CommandError("fresh iperf3 server did not remain running")
        return image

    def start_tracers(
        self, cell: Dict[str, Any], cell_dir: Path
    ) -> Tuple[Optional[subprocess.Popen[Any]], Optional[subprocess.Popen[Any]]]:
        if not self.config[self.mode].get("enable_ttracer", True):
            return None, None
        trace_root = cell_dir / "ttracer"
        duration = int(cell["duration_s"] + cell["omit_s"] + 3)
        ue = self.start_process(
            [
                str(TTRACER_RECORD),
                "--run-group",
                cell["id"],
                "--source",
                "ue",
                "--duration-s",
                str(duration),
                "--output-root",
                str(trace_root),
                "--profile",
                "queue",
            ],
            cell_dir / "logs" / "ttracer_ue_record_stdout.log",
            cwd=REPO_ROOT,
        )
        gnb = self.start_process(
            [
                str(TTRACER_RECORD),
                "--run-group",
                cell["id"],
                "--source",
                "gnb",
                "--duration-s",
                str(duration),
                "--output-root",
                str(trace_root),
                "--profile",
                "latency",
            ],
            cell_dir / "logs" / "ttracer_gnb_record_stdout.log",
            cwd=REPO_ROOT,
        )
        time.sleep(1)
        return ue, gnb

    def finish_tracers(
        self,
        cell: Dict[str, Any],
        cell_dir: Path,
        processes: Tuple[Optional[subprocess.Popen[Any]], Optional[subprocess.Popen[Any]]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"enabled": any(processes), "raw": {}, "extraction": {}}
        if not any(processes):
            return result
        for process in processes:
            if process is None:
                continue
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.stop_process(process)
            else:
                self.stop_process(process)

        trace_root = cell_dir / "ttracer"
        for source, profile in (("ue", "queue"), ("gnb", "latency")):
            raw = trace_root / cell["id"] / source / f"{source}.raw"
            result["raw"][source] = {
                "path": str(raw),
                "exists": raw.is_file(),
                "bytes": raw.stat().st_size if raw.is_file() else 0,
                "sha256": sha256(raw) if raw.is_file() and raw.stat().st_size else None,
            }
            if not raw.is_file() or raw.stat().st_size == 0:
                result["extraction"][source] = {"pass": False, "reason": "raw trace missing"}
                continue
            extract_args = [
                str(TTRACER_EXTRACT),
                "--run-group",
                cell["id"],
                "--source",
                source,
                "--output-root",
                str(trace_root),
                "--profile",
                profile,
                "--clean-output",
            ]
            events = (
                (
                    "NRUE_MAC_DCI_GRANT",
                    "UE_PHY_UL_PAYLOAD_TX_BITS",
                    "NRUE_MAC_RLC_BUFFER_STATUS",
                    "NRUE_MAC_BSR_STATUS",
                )
                if source == "ue"
                else (
                    "GNB_MAC_UL_MCS_DECISION",
                    "GNB_MAC_BLER_MCS_DECISION",
                    "GNB_MAC_PUSCH_POWER_CONTROL",
                    "GNB_PHY_UL_PAYLOAD_RX_BITS",
                )
            )
            for event in events:
                extract_args.extend(["--event", event])
            extract = self.command(
                extract_args,
                cwd=REPO_ROOT,
                timeout=180,
                check=False,
                log=cell_dir / "logs" / f"ttracer_{source}_extract.log",
            )
            result["extraction"][source] = {"pass": extract.returncode == 0}

        grant_csv = trace_root / cell["id"] / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
        if grant_csv.is_file():
            analysis_dir = trace_root / cell["id"] / "ue" / "analysis"
            analyzed = self.command(
                [
                    sys.executable,
                    str(GRANT_ANALYZER),
                    "--csv",
                    str(grant_csv),
                    "--output-dir",
                    str(analysis_dir),
                    "--window-s",
                    "1.0",
                ],
                cwd=REPO_ROOT,
                timeout=120,
                check=False,
                log=cell_dir / "logs" / "grant_analysis.log",
            )
            result["grant_analysis_pass"] = analyzed.returncode == 0
            result["grant_summary"] = str(analysis_dir / "nrue_grant_summary.csv")
        else:
            result["grant_analysis_pass"] = False
        result["raw_capture_pass"] = all(item["bytes"] > 0 for item in result["raw"].values())
        return result

    @staticmethod
    def interface_counters(iface: str) -> Dict[str, Optional[int]]:
        result: Dict[str, Optional[int]] = {}
        for name in ("tx_bytes", "rx_bytes", "tx_packets", "rx_packets", "tx_dropped", "rx_dropped"):
            path = Path("/sys/class/net") / iface / "statistics" / name
            try:
                result[name] = int(path.read_text().strip())
            except (OSError, ValueError):
                result[name] = None
        return result

    def iperf_client_command(self, image: str, cell: Dict[str, Any]) -> List[str]:
        container_name = clean_id(f"scenesense-iperf-{self.experiment_dir.name}-{cell['id']}")[:120]
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--label",
            CLIENT_LABEL,
            "--name",
            container_name,
            "--entrypoint",
            "iperf3",
            image,
            "-c",
            self.config["ue"]["ext_dn_ip"],
            "-B",
            self.config["ue"]["static_ip"],
            "-p",
            str(self.config["ue"]["iperf_port"]),
            "-t",
            str(cell["duration_s"]),
            "-O",
            str(cell["omit_s"]),
            "-J",
            "--connect-timeout",
            "10000",
        ]
        if cell["protocol"] == "udp":
            command.extend(
                [
                    "-u",
                    "-b",
                    f"{cell['offered_mbps']}M",
                    "-l",
                    str(self.config["udp"]["datagram_bytes"]),
                ]
            )
        return command

    @staticmethod
    def parse_iperf(stdout: str, cell: Dict[str, Any]) -> Dict[str, Any]:
        start = stdout.find("{")
        if start < 0:
            raise ValueError("iperf3 output contains no JSON object")
        payload = json.loads(stdout[start:])
        if payload.get("error"):
            raise ValueError(f"iperf3 error: {payload['error']}")
        end = payload.get("end", {})
        if cell["protocol"] == "udp":
            received = end.get("sum_received") or end.get("sum") or {}
            # iperf3 3.9 emits one UDP `end.sum` object rather than the newer
            # separate sum_sent/sum_received pair. Its server-derived loss and
            # byte count are still the authoritative completed test summary.
            sent = end.get("sum_sent") or end.get("sum") or {}
            bits_per_second = float(received.get("bits_per_second", 0.0))
            loss_percent = float(received.get("lost_percent", 100.0))
            offered = float(cell["offered_mbps"])
            result = {
                "receiver_mbps": bits_per_second / 1_000_000.0,
                "sender_mbps": float(sent.get("bits_per_second", 0.0)) / 1_000_000.0,
                "receiver_bytes": received.get("bytes"),
                "sender_bytes": sent.get("bytes"),
                "goodput_ratio": bits_per_second / (offered * 1_000_000.0),
                "loss_percent": loss_percent,
                "delivery_ratio": max(0.0, 1.0 - loss_percent / 100.0),
                "jitter_ms": received.get("jitter_ms"),
                "packets": received.get("packets"),
                "lost_packets": received.get("lost_packets"),
                "out_of_order": received.get("out_of_order"),
            }
        else:
            received = end.get("sum_received") or end.get("sum") or {}
            sent = end.get("sum_sent") or {}
            result = {
                "receiver_mbps": float(received.get("bits_per_second", 0.0)) / 1_000_000.0,
                "sender_mbps": float(sent.get("bits_per_second", 0.0)) / 1_000_000.0,
                "receiver_bytes": received.get("bytes"),
                "sender_bytes": sent.get("bytes"),
                "retransmits": sent.get("retransmits"),
            }
        result["iperf_version"] = payload.get("start", {}).get("version")
        result["cpu_utilization_percent"] = end.get("cpu_utilization_percent")
        return result

    @staticmethod
    def parse_ping(text: str) -> Dict[str, Any]:
        rtts = [float(match) for match in re.findall(r"time[=<]([0-9.]+)\s*ms", text)]
        loss_match = re.search(r"([0-9.]+)% packet loss", text)
        return {
            "samples": len(rtts),
            "packet_loss_percent": float(loss_match.group(1)) if loss_match else None,
            "rtt_min_ms": min(rtts) if rtts else None,
            "rtt_median_ms": statistics.median(rtts) if rtts else None,
            "rtt_p95_ms": percentile(rtts, 95),
            "rtt_p99_ms": percentile(rtts, 99),
            "rtt_max_ms": max(rtts) if rtts else None,
        }

    def run_measurement(self, cell: Dict[str, Any], cell_dir: Path) -> Dict[str, Any]:
        image = self.reset_iperf_server(cell_dir)
        trace_processes = self.start_tracers(cell, cell_dir)
        total_duration = int(cell["duration_s"] + cell["omit_s"] + 2)
        ping_path = cell_dir / "loaded_ping.log"
        ping_handle = ping_path.open("wb", buffering=0)
        ping_process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "ping",
                "-D",
                "-I",
                self.config["ue"]["interface"],
                "-i",
                str(self.config[self.mode]["ping_interval_s"]),
                "-w",
                str(total_duration),
                self.config["ue"]["ext_dn_ip"],
            ],
            stdout=ping_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.local_processes.append(ping_process)
        counters_before = self.interface_counters(self.config["ue"]["interface"])
        started = time.monotonic()
        client = self.sudo(
            self.iperf_client_command(image, cell),
            timeout=float(cell["duration_s"] + cell["omit_s"] + 90),
            check=False,
        )
        elapsed = time.monotonic() - started
        atomic_text(cell_dir / "iperf.json", client.stdout)
        atomic_text(cell_dir / "iperf.stderr.log", client.stderr)
        try:
            ping_process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.stop_process(ping_process)
        else:
            self.stop_process(ping_process)
        ping_handle.close()
        trace_result = self.finish_tracers(cell, cell_dir, trace_processes)
        counters_after = self.interface_counters(self.config["ue"]["interface"])
        if client.returncode != 0:
            raise CommandError(f"iperf3 client failed rc={client.returncode}")
        iperf = self.parse_iperf(client.stdout, cell)
        ping = self.parse_ping(ping_path.read_text(encoding="utf-8", errors="replace"))
        tx_before = counters_before.get("tx_bytes")
        tx_after = counters_after.get("tx_bytes")
        tunnel_tx_delta = (
            int(tx_after) - int(tx_before)
            if tx_before is not None and tx_after is not None and tx_after >= tx_before
            else None
        )
        sender_bytes = iperf.get("sender_bytes")
        tunnel_sender_ratio = (
            float(tunnel_tx_delta) / float(sender_bytes)
            if tunnel_tx_delta is not None and sender_bytes not in (None, 0)
            else None
        )
        tunnel_path_pass = tunnel_sender_ratio is not None and tunnel_sender_ratio >= 0.80
        result = {
            "cell": cell,
            "measured_at": utc_now(),
            "wall_duration_s": elapsed,
            "client_image": image,
            "iperf": iperf,
            "loaded_ping": ping,
            "interface_counters_before": counters_before,
            "interface_counters_after": counters_after,
            "tunnel_tx_bytes_delta": tunnel_tx_delta,
            "tunnel_tx_to_iperf_sender_bytes_ratio": tunnel_sender_ratio,
            "tunnel_path_pass": tunnel_path_pass,
            "ttracer": trace_result,
        }
        if cell["protocol"] == "udp":
            result["capacity_pass"] = (
                iperf["goodput_ratio"] >= self.config["udp"]["capacity_goodput_ratio_min"]
                and iperf["loss_percent"] <= self.config["udp"]["capacity_loss_percent_max"]
            )
        else:
            result["capacity_pass"] = None
        if not tunnel_path_pass:
            atomic_json(cell_dir / "INVALID_TUNNEL_PATH.json", result)
            raise CommandError(
                "iperf traffic did not produce the required oaitun_ue1 TX-byte delta"
            )
        return result

    def conditional_allowed(self, cell: Dict[str, Any]) -> Tuple[bool, str]:
        parent = cell.get("conditional_parent")
        if not parent:
            return True, "unconditional"
        parent_path = self.cells_dir / parent / "COMPLETE.json"
        if not parent_path.is_file():
            return False, f"parent_not_complete:{parent}"
        result = json.loads(parent_path.read_text())
        if result.get("measurement", {}).get("capacity_pass") is True:
            return True, f"parent_capacity_pass:{parent}"
        return False, f"parent_capacity_failed:{parent}"

    def run_cell(self, cell: Dict[str, Any]) -> str:
        cell_dir = self.cells_dir / cell["id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        complete = cell_dir / "COMPLETE.json"
        failure = cell_dir / "FAILURE.json"
        skipped = cell_dir / "SKIPPED.json"
        measurement_path = cell_dir / "MEASUREMENT_COMPLETE.json"
        if complete.is_file():
            return "complete"
        if failure.is_file() or skipped.is_file():
            return "terminal"
        allowed, reason = self.conditional_allowed(cell)
        if not allowed:
            atomic_json(skipped, {"cell": cell, "reason": reason, "skipped_at": utc_now()})
            self.say(f"SKIP {cell['id']}: {reason}")
            return "skipped"

        self.current_cell = cell_dir
        atomic_json(cell_dir / "STARTED.json", {"cell": cell, "started_at": utc_now()})
        self.say(f"START {cell['id']} (fresh full stack)")
        try:
            if measurement_path.is_file():
                measurement = json.loads(measurement_path.read_text())
                self.say(f"RECOVER {cell['id']}: measurement exists; cleanup only")
            else:
                bringup = self.bring_up(cell, cell_dir)
                atomic_json(cell_dir / "BRINGUP_COMPLETE.json", bringup)
                measurement = self.run_measurement(cell, cell_dir)
                atomic_json(measurement_path, measurement)
            cleanup = self.teardown(cell_dir, "post_measurement")
            record = {
                "cell": cell,
                "completed_at": utc_now(),
                "measurement": measurement,
                "cleanup": cleanup,
            }
            if not cleanup["pass"]:
                raise CommandError("post-measurement cleanup failed")
            atomic_json(complete, record)
            self.say(
                f"DONE {cell['id']}: receiver={measurement['iperf']['receiver_mbps']:.3f} Mbps "
                f"loss={measurement['iperf'].get('loss_percent')}%"
            )
            return "complete"
        except KeyboardInterrupt:
            self.teardown(cell_dir, "interrupted")
            raise
        except Exception as exc:
            cleanup = self.teardown(cell_dir, "failure")
            atomic_json(
                failure,
                {
                    "cell": cell,
                    "failed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "cleanup": cleanup,
                },
            )
            self.say(f"FAIL {cell['id']}: {exc}")
            return "failed"
        finally:
            self.current_cell = None

    def aggregate(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        failures: List[str] = []
        skipped: List[str] = []
        pending: List[str] = []
        for cell in self.plan["cells"]:
            cell_dir = self.cells_dir / cell["id"]
            complete = cell_dir / "COMPLETE.json"
            if complete.is_file():
                record = json.loads(complete.read_text())
                measurement = record["measurement"]
                iperf = measurement["iperf"]
                ping = measurement["loaded_ping"]
                rows.append(
                    {
                        **cell,
                        "receiver_mbps": iperf.get("receiver_mbps"),
                        "sender_mbps": iperf.get("sender_mbps"),
                        "goodput_ratio": iperf.get("goodput_ratio"),
                        "loss_percent": iperf.get("loss_percent"),
                        "delivery_ratio": iperf.get("delivery_ratio"),
                        "jitter_ms": iperf.get("jitter_ms"),
                        "retransmits": iperf.get("retransmits"),
                        "capacity_pass": measurement.get("capacity_pass"),
                        "ping_samples": ping.get("samples"),
                        "ping_loss_percent": ping.get("packet_loss_percent"),
                        "rtt_median_ms": ping.get("rtt_median_ms"),
                        "rtt_p95_ms": ping.get("rtt_p95_ms"),
                        "rtt_p99_ms": ping.get("rtt_p99_ms"),
                        "ttracer_raw_pass": measurement.get("ttracer", {}).get(
                            "raw_capture_pass"
                        ),
                        "cleanup_pass": record.get("cleanup", {}).get("pass"),
                    }
                )
            elif (cell_dir / "FAILURE.json").is_file():
                failures.append(cell["id"])
            elif (cell_dir / "SKIPPED.json").is_file():
                skipped.append(cell["id"])
            else:
                pending.append(cell["id"])

        csv_path = self.experiment_dir / "results.csv"
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        result = {
            "generated_at": utc_now(),
            "mode": self.mode,
            "completed_cells": len(rows),
            "failed_cells": failures,
            "pending_cells": pending,
            "failed_or_missing_cells": failures + pending,
            "conditional_skips": skipped,
            "rows": rows,
        }
        atomic_json(self.experiment_dir / "results.json", result)
        self.write_report(result)
        return result

    def write_report(self, result: Dict[str, Any]) -> None:
        lines = [
            f"# OAI TDD/bandwidth iperf {self.mode} result",
            "",
            f"- Generated: `{result['generated_at']}`",
            f"- Completed cells: **{result['completed_cells']}**",
            f"- Failed or missing: **{len(result['failed_or_missing_cells'])}**",
            f"- Conditional high-rate skips: **{len(result['conditional_skips'])}**",
            "",
            "| Cell | Protocol | Offered Mbps | Received Mbps | Loss % | RTT p95 ms | Capacity pass |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in result["rows"]:
            def fmt(value: Any, digits: int = 3) -> str:
                return "" if value is None else f"{float(value):.{digits}f}"

            lines.append(
                "| {id} | {protocol} | {offered} | {received} | {loss} | {rtt} | {passed} |".format(
                    id=row["id"],
                    protocol=row["protocol"],
                    offered="" if row["offered_mbps"] is None else row["offered_mbps"],
                    received=fmt(row["receiver_mbps"]),
                    loss=fmt(row["loss_percent"]),
                    rtt=fmt(row["rtt_p95_ms"]),
                    passed=row["capacity_pass"],
                )
            )
        if result["failed_or_missing_cells"]:
            lines.extend(["", "## Failed or missing", ""])
            lines.extend(f"- `{cell}`" for cell in result["failed_or_missing_cells"])
        atomic_text(self.experiment_dir / "REPORT.md", "\n".join(lines) + "\n")

    def run(self) -> int:
        self.ensure_sudo()
        try:
            self.preflight()
            initial_cleanup = self.teardown(self.experiment_dir, "experiment_start")
            if not initial_cleanup["pass"]:
                raise CommandError("initial full-stack cleanup failed")
            atomic_json(
                self.experiment_dir / "STATUS.json",
                {"phase": "running", "mode": self.mode, "started_at": utc_now()},
            )
            for index, cell in enumerate(self.plan["cells"], 1):
                self.say(f"cell {index}/{len(self.plan['cells'])}")
                self.run_cell(cell)
                partial = self.aggregate()
                atomic_json(
                    self.experiment_dir / "STATUS.json",
                    {
                        "phase": "running",
                        "mode": self.mode,
                        "updated_at": utc_now(),
                        "completed_cells": partial["completed_cells"],
                        "failed_or_missing_cells": partial["failed_or_missing_cells"],
                    },
                )
            final_cleanup = self.teardown(self.experiment_dir, "experiment_end")
            result = self.aggregate()
            failures = result["failed_or_missing_cells"]
            if self.mode == "smoke":
                terminal = (
                    "OAI_TDD_BANDWIDTH_IPERF_SMOKE_PASSED"
                    if not failures and final_cleanup["pass"]
                    else "OAI_TDD_BANDWIDTH_IPERF_SMOKE_FAILED"
                )
            else:
                terminal = (
                    "OAI_TDD_BANDWIDTH_IPERF_SWEEP_COMPLETE"
                    if not failures and final_cleanup["pass"]
                    else "OAI_TDD_BANDWIDTH_IPERF_SWEEP_COMPLETE_WITH_INFRASTRUCTURE_GAPS"
                )
            terminal_record = {
                "terminal": terminal,
                "completed_at": utc_now(),
                "experiment_dir": str(self.experiment_dir),
                "completed_cells": result["completed_cells"],
                "failed_or_missing_cells": failures,
                "conditional_skips": result["conditional_skips"],
                "final_cleanup": final_cleanup,
            }
            atomic_json(self.experiment_dir / "TERMINAL.json", terminal_record)
            atomic_text(self.experiment_dir / "TERMINAL_VERDICT.txt", terminal + "\n")
            atomic_json(
                self.experiment_dir / "STATUS.json", {"phase": "terminal", **terminal_record}
            )
            self.say(f"TERMINAL {terminal}")
            self.say(f"artifacts: {self.experiment_dir}")
            return 0 if terminal.endswith(("PASSED", "COMPLETE")) else 1
        except KeyboardInterrupt:
            cleanup = self.teardown(self.current_cell or self.experiment_dir, "keyboard_interrupt")
            atomic_json(
                self.experiment_dir / "STATUS.json",
                {"phase": "interrupted", "at": utc_now(), "cleanup": cleanup},
            )
            self.say("interrupted safely; rerun with --resume")
            return 130
        finally:
            self.stop_sudo_keepalive()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="run four 5-Mbps smoke cells")
    group.add_argument("--full", action="store_true", help="run the registered full sweep")
    group.add_argument("--resume", type=Path, help="resume an existing experiment directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume is not None:
        runner = SweepRunner("resume", args.resume)
    else:
        runner = SweepRunner("smoke" if args.smoke else "full", None)
    print(f"experiment_dir={runner.experiment_dir}", flush=True)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
