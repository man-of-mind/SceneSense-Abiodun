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
import errno
import json
import os
import signal
import socket
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


def start_carla(port: int, log_path: Path) -> tuple[subprocess.Popen, int]:
    """Start CarlaUnreal.sh in its own session and return it with its PGID.

    The PGID is the only handle that covers the whole launch: the wrapper script
    exits well before the UE binary it spawned, so waiting on the wrapper alone
    reports a live server as a clean shutdown.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("wb")
    process = subprocess.Popen(
        [
            str(CARLA_LAUNCHER), "-RenderOffScreen", "-nosound",
            "-quality-level=Epic", f"-carla-rpc-port={port}",
        ],
        cwd=str(CARLA_ROOT), stdout=stream, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, env=child_env(),
    )
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        # Exited before we could read it; start_new_session makes pid == pgid.
        pgid = process.pid
    return process, pgid


def process_group_members(pgid: int) -> list[dict[str, Any]]:
    """Live PIDs in exactly this process group, read from /proc.

    Deliberately not a pkill/killall name match: only processes whose kernel
    process-group id equals the group this launcher created are ever considered,
    so another user's CARLA can never be seen or signalled.
    """
    members: list[dict[str, Any]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                fields = handle.read().rsplit(b")", 1)[1].split()
            # stat fields after comm: state, ppid, pgrp, ...
            if int(fields[2]) != int(pgid):
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\x00", b" ").decode(
                    "utf-8", "replace").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                IndexError, ValueError):
            continue
        members.append({"pid": pid, "cmdline": cmdline[:200]})
    return members


def rpc_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    """True when something still accepts connections on the RPC port."""
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError as error:
        if getattr(error, "errno", None) in (errno.ECONNREFUSED, errno.EHOSTUNREACH):
            return False
        return False


def signal_group(pgid: int, sig: int) -> str:
    """Signal exactly the launched group, never this launcher's own group."""
    if int(pgid) <= 0 or int(pgid) == os.getpgrp():
        return "refused_own_or_invalid_group"
    try:
        os.killpg(int(pgid), sig)
    except ProcessLookupError:
        return "no_such_group"
    except PermissionError:
        return "permission_denied"
    return "sent"


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


def stop_carla(
    process: subprocess.Popen,
    pgid: int,
    port: int,
    grace_s: float = 30.0,
    kill_grace_s: float = 20.0,
) -> dict[str, Any]:
    """Tear down the whole launched group and prove the RPC port is released.

    Shutdown is complete only when both are true: no PID remains in the group
    this launcher created, and nothing accepts connections on the RPC port. The
    wrapper's own return code is recorded but is never the completion test - the
    UE binary outlives it, absorbs SIGTERM, and previously survived as an
    orphan holding the port and its GPU allocation.
    """
    report: dict[str, Any] = {"pgid": int(pgid), "rpc_port": int(port)}

    def settled() -> tuple[list[dict[str, Any]], bool]:
        members = process_group_members(pgid)
        return members, rpc_port_listening(port)

    members, listening = settled()
    report["group_members_before_signal"] = members
    report["port_listening_before_signal"] = listening
    if not members and not listening:
        report["already_exited"] = True
        report["wrapper_returncode"] = process.poll()
        report["sigkill_required"] = False
        report["group_members_remaining"] = []
        report["port_listening_after"] = False
        report["shutdown_verified"] = True
        return report

    report["already_exited"] = False
    report["sigterm_result"] = signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + float(grace_s)
    while time.monotonic() < deadline:
        members, listening = settled()
        if not members and not listening:
            break
        time.sleep(1.0)

    report["group_members_after_sigterm"] = members
    report["port_listening_after_sigterm"] = listening
    sigkill_required = bool(members or listening)
    report["sigkill_required"] = sigkill_required
    if sigkill_required:
        # Exactly this group, by PGID. No name matching, no other CARLA touched.
        report["sigkill_result"] = signal_group(pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + float(kill_grace_s)
        while time.monotonic() < kill_deadline:
            members, listening = settled()
            if not members and not listening:
                break
            time.sleep(1.0)

    try:
        report["wrapper_returncode"] = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        report["wrapper_returncode"] = process.poll()

    members, listening = settled()
    report["group_members_remaining"] = members
    report["port_listening_after"] = listening
    report["shutdown_verified"] = not members and not listening
    return report


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
    server, server_pgid = start_carla(int(args.port), server_log)
    record["carla_pid"] = server.pid
    record["carla_pgid"] = int(server_pgid)
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
            "--replenish-interval-s", str(args.replenish_interval_s),
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
        # Group + RPC state, not the wrapper: the wrapper exits early on a
        # healthy server, so poll() alone reports a false death here.
        record["carla_wrapper_returncode_during_run"] = server.poll()
        record["carla_group_members_during_run"] = len(
            process_group_members(server_pgid))
        record["carla_rpc_listening_after_client"] = rpc_port_listening(
            int(args.port))
        record["carla_exited_during_run"] = (
            record["carla_group_members_during_run"] == 0
            and not record["carla_rpc_listening_after_client"])
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
        record["carla_shutdown"] = stop_carla(
            server, server_pgid, int(args.port))


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
    parser.add_argument(
        "--replenish-interval-s", type=float, default=2.0,
        help="population reconciliation interval in CARLA simulated seconds, "
             "forwarded unchanged to the collection runner (default: %(default)s)",
    )
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
        "SUPERVISED_COLLECTION_PASSED"
        if final["status"] == "COLLECTION_ATTEMPT_PASSED"
        else "COLLECTION_SMOKE_FAILED"
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "SUPERVISED_COLLECTION_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
