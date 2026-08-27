#!/usr/bin/env python3
"""Supervise exactly one Route B v3 traffic_30_30 smoke and CARLA lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_COLLECTION = HERE.parent
REPO_ROOT = DATA_COLLECTION.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_collection.run_route_b_collection_supervised_v1 as base  # noqa: E402


V3_RUNNER = DATA_COLLECTION / "run_route_b_perception_collection_v3.py"
HEAVY_PAYLOAD_NAMES = (
    "rgb", "masks", "semantic_tags", "radar_tensors", "radar_points", "depth",
)


def _proc_rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def _host_ram_used_kib() -> int:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        return 0
    return max(0, values.get("MemTotal", 0) - values.get("MemAvailable", 0))


def _gpu_used_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return sum(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except ValueError:
        return None


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _delete_failed_heavy_payload(output_dir: Path) -> dict[str, Any]:
    """Delete only explicitly named children of the one failed create-only output."""
    resolved = output_dir.resolve()
    before = _tree_bytes(resolved)
    deleted: list[str] = []
    for name in HEAVY_PAYLOAD_NAMES:
        target = (resolved / name).resolve()
        if target.parent != resolved:
            raise RuntimeError(f"refusing unsafe failed-payload target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            deleted.append(str(target))
    after = _tree_bytes(resolved)
    return {
        "deleted_explicit_paths": deleted,
        "bytes_before": before,
        "bytes_after": after,
        "bytes_reclaimed": max(0, before - after),
        "recoverable": False,
    }


def _patch_summary_after_shutdown(
    output_dir: Path, shutdown: dict[str, Any], resources: dict[str, Any],
    client_returncode: int,
) -> dict[str, Any] | None:
    path = output_dir / "route_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["resource_usage"] = resources
    summary["carla_shutdown"] = shutdown
    summary["client_returncode"] = int(client_returncode)
    shutdown_ok = bool(shutdown.get("shutdown_verified"))
    client_ok = int(client_returncode) == 0
    summary.setdefault("v3_gates", {})["carla_shutdown_verified"] = shutdown_ok
    summary["v3_gates"]["client_process_exit_success"] = client_ok
    summary["gates"] = {**summary.get("v2_gates", {}), **summary["v3_gates"]}
    technical = all(bool(value) for value in summary["gates"].values())
    if not technical:
        summary["terminal"] = "ROUTE_B_V3_COLLECTION_FAILED"
        summary["status"] = "COLLECTION_EPISODE_FAILED"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_dir / "ROUTE_B_V3_30_30_SMOKE_REPORT.md"
    if report_path.is_file():
        with report_path.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## Supervised process lifecycle\n\n"
                f"- Final terminal: `{summary['terminal']}`.\n"
                f"- Client exit code: `{client_returncode}`.\n"
                f"- Peak client RSS: `{resources.get('peak_client_rss_kib')}` KiB.\n"
                f"- Peak host RAM used: `{resources.get('peak_host_ram_used_kib')}` KiB.\n"
                f"- Peak whole-device GPU memory used: `{resources.get('peak_gpu_memory_used_mib')}` MiB.\n"
                f"- CARLA shutdown verified: `{shutdown_ok}`; SIGKILL required: "
                f"`{shutdown.get('sigkill_required')}`.\n"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--carla-ready-timeout-s", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    report_path = args.report_json.resolve()
    if output_dir.exists() or report_path.exists():
        print("create-only output/report path already exists", file=sys.stderr)
        return 2
    if base.rpc_port_listening(args.port, args.host) or base.rpc_port_listening(args.tm_port, args.host):
        print("CARLA/TM port already in use; refusing to launch", file=sys.stderr)
        return 2

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    server_log = output_dir.parent / f"{output_dir.name}_carla_server.log"
    client_log = output_dir.parent / f"{output_dir.name}_client.log"
    server, pgid = base.start_carla(int(args.port), server_log)
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema": "route_b_perception_v3.supervised_smoke.v1",
        "output_dir": str(output_dir), "server_log": str(server_log),
        "client_log": str(client_log), "carla_pid": server.pid, "carla_pgid": pgid,
        "attempts": 1, "retry_permitted": False,
    }
    child: subprocess.Popen | None = None
    peak_client_rss = 0
    peak_host_used = _host_ram_used_kib()
    peak_gpu_used = _gpu_used_mib()
    shutdown: dict[str, Any] = {"shutdown_verified": False}
    client_returncode = 2
    try:
        version = base.wait_for_rpc(int(args.port), float(args.carla_ready_timeout_s))
        record["carla_rpc_ready"] = version is not None
        record["carla_server_version"] = version
        if version is None:
            record["failure"] = "fresh Epic CARLA did not become RPC-ready"
        else:
            command = [
                str(base.VENV_PYTHON), str(V3_RUNNER),
                "--density", "traffic_30_30", "--split", "smoke",
                "--output-dir", str(output_dir), "--host", str(args.host),
                "--port", str(args.port), "--tm-port", str(args.tm_port),
                "--scenario-seed", "101", "--tm-seed", "1101",
                "--target-speed-kph", "25.0", "--rasterizer", "fast",
                "--replenish-interval-s", "2.0", "--maximum-loop-sim-s", "600.0",
                "--no-hybrid-physics", "--allow-roadblock-clearing",
            ]
            record["client_argv"] = command
            with client_log.open("wb") as stream:
                child = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=base.child_env())
                while child.poll() is None:
                    peak_client_rss = max(peak_client_rss, _proc_rss_kib(child.pid))
                    peak_host_used = max(peak_host_used, _host_ram_used_kib())
                    gpu = _gpu_used_mib()
                    if gpu is not None:
                        peak_gpu_used = gpu if peak_gpu_used is None else max(peak_gpu_used, gpu)
                    time.sleep(5.0)
                client_returncode = int(child.returncode)
            record["client_exit"] = base.describe_exit(client_returncode)
    finally:
        shutdown = base.stop_carla(server, pgid, int(args.port))
        record["carla_shutdown"] = shutdown

    resources = {
        "sampling_interval_s": 5.0,
        "peak_client_rss_kib": peak_client_rss,
        "peak_host_ram_used_kib": peak_host_used,
        "peak_gpu_memory_used_mib": peak_gpu_used,
        "gpu_scope": "sum of whole-device memory.used across visible GPUs",
    }
    summary = _patch_summary_after_shutdown(
        output_dir, shutdown, resources, client_returncode)
    terminal = (summary or {}).get("terminal", "ROUTE_B_V3_COLLECTION_FAILED")
    accepted = terminal in {
        "ROUTE_B_V3_30_30_READY_FOR_MANUAL_REVIEW",
        "ROUTE_B_V3_VISIBILITY_NOT_EXERCISED",
    }
    record["wall_seconds"] = time.monotonic() - started
    record["resources"] = resources
    record["terminal"] = terminal
    record["accepted"] = accepted
    if not accepted:
        reclaim = _delete_failed_heavy_payload(output_dir) if output_dir.exists() else {
            "deleted_explicit_paths": [], "bytes_reclaimed": 0, "recoverable": False}
        record["failed_payload_reclaim"] = reclaim
        if summary is not None:
            summary["failed_payload_reclaim"] = reclaim
            (output_dir / "route_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
