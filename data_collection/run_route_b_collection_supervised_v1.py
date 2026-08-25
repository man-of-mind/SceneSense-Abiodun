#!/usr/bin/env python3
"""Parent launcher for one supervised Route B perception collection.

Exists so a native client fault - the previous 30/30 attempt segfaulted inside
``carla.cpython-310-x86_64-linux-gnu.so`` - is recorded rather than inferred.
It starts a fresh Epic CARLA server, waits for RPC readiness, runs exactly one
collection, records the child's exit status (including the terminating signal),
and then tears the server down. At most one retry, and only when explicitly
asked for.

It admits nothing: the collection's own gates decide the episode status. A
retried attempt writes to its own sibling output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CARLA_ROOT = HERE.parents[3]
CARLA_LAUNCHER = CARLA_ROOT / "CarlaUnreal.sh"
VENV_PYTHON = Path(
    "/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3")
COLLECTION_RUNNER = HERE / "run_route_b_perception_collection_v2.py"


def child_env() -> dict[str, str]:
    """A CARLA client environment with PYTHONPATH removed.

    Exporting PYTHONPATH shadows this tree with the stale ``neu_collab`` copy.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def start_carla(port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("wb")
    return subprocess.Popen(
        [
            str(CARLA_LAUNCHER), "-RenderOffScreen", "-nosound",
            "-quality-level=Epic", f"-carla-rpc-port={port}",
        ],
        cwd=str(CARLA_ROOT), stdout=stream, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, env=child_env(),
    )


def wait_for_rpc(port: int, timeout_s: float) -> str | None:
    probe = (
        "import carla,sys;"
        f"c=carla.Client('127.0.0.1',{port});c.set_timeout(5.0);"
        "sys.stdout.write(c.get_server_version())"
    )
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", probe], capture_output=True, text=True,
            env=child_env(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        time.sleep(3.0)
    return None


def stop_carla(process: subprocess.Popen, grace_s: float = 20.0) -> dict[str, Any]:
    if process.poll() is not None:
        return {"already_exited": True, "returncode": process.returncode}
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(1.0)
    forced = False
    if process.poll() is None:
        forced = True
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=30)
    return {"already_exited": False, "returncode": process.returncode, "forced": forced}


def describe_exit(returncode: int) -> dict[str, Any]:
    row: dict[str, Any] = {"returncode": int(returncode)}
    if returncode < 0:
        row["terminating_signal"] = int(-returncode)
        row["signal_name"] = signal.Signals(-returncode).name
        row["native_fault"] = True
    elif returncode > 128:
        row["terminating_signal"] = int(returncode - 128)
        try:
            row["signal_name"] = signal.Signals(returncode - 128).name
        except ValueError:
            row["signal_name"] = "unknown"
        row["native_fault"] = True
    else:
        row["native_fault"] = False
    return row


def run_attempt(args: argparse.Namespace, output_dir: Path, attempt: int) -> dict[str, Any]:
    server_log = output_dir.parent / f"{output_dir.name}_carla_server.log"
    client_log = output_dir.parent / f"{output_dir.name}_client.log"
    record: dict[str, Any] = {
        "attempt": attempt,
        "output_dir": str(output_dir),
        "carla_server_log": str(server_log),
        "client_log": str(client_log),
    }
    server = start_carla(int(args.port), server_log)
    record["carla_pid"] = server.pid
    try:
        version = wait_for_rpc(int(args.port), float(args.carla_ready_timeout_s))
        record["carla_rpc_ready"] = version is not None
        record["carla_server_version"] = version
        if version is None:
            record["status"] = "CARLA_NOT_READY"
            return record
        argv = [
            str(VENV_PYTHON), str(COLLECTION_RUNNER),
            "--density", str(args.density),
            "--split", str(args.split),
            "--output-dir", str(output_dir),
            "--host", str(args.host), "--port", str(args.port),
            "--tm-port", str(args.tm_port),
            "--scenario-seed", str(args.scenario_seed),
            "--tm-seed", str(args.tm_seed),
            "--target-speed-kph", str(args.target_speed_kph),
            "--rasterizer", str(args.rasterizer),
            "--maximum-loop-sim-s", str(args.maximum_loop_sim_s),
            "--no-hybrid-physics",
        ]
        record["client_argv"] = argv
        started = time.monotonic()
        with client_log.open("wb") as stream:
            child = subprocess.run(
                argv, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=child_env(),
            )
        record["wall_clock_s"] = round(time.monotonic() - started, 1)
        record["client_exit"] = describe_exit(child.returncode)
        record["carla_exited_during_run"] = server.poll() is not None
        summary_path = output_dir / "route_summary.json"
        events_path = output_dir.parent / f"{output_dir.name}_population_events.jsonl"
        record["route_summary_present"] = summary_path.is_file()
        record["population_events_present"] = events_path.is_file()
        if events_path.is_file():
            record["population_event_lines"] = sum(
                1 for _ in events_path.open("r", encoding="utf-8"))
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record["episode_status"] = summary.get("status")
            record["failed_gates"] = sorted(
                key for key, value in summary.get("gates", {}).items() if value is False
            )
            record["dropped_callback_frames"] = (
                summary.get("cadence", {}).get("dropped_callback_frames"))
        record["status"] = (
            "COLLECTION_ATTEMPT_PASSED"
            if record["client_exit"]["returncode"] == 0
            and record.get("episode_status") == "COLLECTION_EPISODE_PASSED"
            else "COLLECTION_ATTEMPT_FAILED"
        )
        return record
    finally:
        record["carla_shutdown"] = stop_carla(server)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", default="traffic_30_30")
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--scenario-seed", type=int, default=101)
    parser.add_argument("--tm-seed", type=int, default=1101)
    parser.add_argument("--target-speed-kph", type=float, default=25.0)
    parser.add_argument("--rasterizer", default="fast")
    parser.add_argument("--maximum-loop-sim-s", type=float, default=600.0)
    parser.add_argument("--carla-ready-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--allow-one-retry", action="store_true",
        help="on a native client/server fault only, retry once into <output-dir>_retry1",
    )
    parser.add_argument("--report-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": "route_b_perception_v2.supervised_collection.v1",
        "attempts": [],
    }
    output_dir = Path(args.output_dir).resolve()
    first = run_attempt(args, output_dir, 1)
    report["attempts"].append(first)
    final = first
    if (
        args.allow_one_retry
        and first["status"] != "COLLECTION_ATTEMPT_PASSED"
        and (first.get("client_exit", {}).get("native_fault")
             or first.get("carla_exited_during_run"))
    ):
        retry_dir = output_dir.with_name(output_dir.name + "_retry1")
        report["retry_reason"] = "native client or server fault on attempt 1"
        second = run_attempt(args, retry_dir, 2)
        report["attempts"].append(second)
        final = second

    report["status"] = (
        "TRAFFIC_30_30_READY_FOR_MANUAL_REVIEW"
        if final["status"] == "COLLECTION_ATTEMPT_PASSED"
        else "COLLECTION_SMOKE_FAILED"
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "TRAFFIC_30_30_READY_FOR_MANUAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
