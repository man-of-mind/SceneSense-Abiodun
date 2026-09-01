#!/usr/bin/env python3
"""Supervise one fresh Epic CARLA process for the single v2 qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import data_collection.run_route_b_collection_supervised_v1 as supervisor

from .controlled_qualification import BLOCKED, IMPLEMENTATION_FAILED, QUALIFIED
from .core import write_json_x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        print(f"create-only output exists: {root}", file=sys.stderr)
        return 2
    if supervisor.rpc_port_listening(args.port, args.host):
        print(f"CARLA port already in use: {args.port}", file=sys.stderr)
        return 2
    root.mkdir(parents=True)
    server_log, client_log = root / "carla_server.log", root / "controlled_client.log"
    server, pgid = supervisor.start_carla(args.port, server_log)
    started = time.monotonic()
    record = {
        "schema": "publication_zbuffer_visibility_controlled_supervision_v2",
        "output_root": str(root),
        "server_log": str(server_log),
        "client_log": str(client_log),
        "carla_pid": int(server.pid),
        "carla_pgid": int(pgid),
        "qualification_attempts": 1,
    }
    code = 4
    try:
        version = supervisor.wait_for_rpc(args.port, args.ready_timeout_s)
        record["carla_server_version"] = version
        record["carla_rpc_ready"] = version is not None
        if version is None:
            record["terminal"] = IMPLEMENTATION_FAILED
            record["failure"] = "fresh Epic CARLA did not become RPC-ready"
        else:
            command = [
                str(supervisor.VENV_PYTHON),
                "-m",
                "data_collection.route_b_publication_zbuffer_visibility_v2.controlled_qualification",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--output-dir",
                str(root / "controlled_qualification"),
            ]
            record["client_argv"] = command
            with client_log.open("xb") as stream:
                child = subprocess.run(
                    command,
                    cwd=str(supervisor.HERE.parent),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=supervisor.child_env(),
                )
            code = int(child.returncode)
            record["client_exit"] = supervisor.describe_exit(code)
            result_path = root / "controlled_qualification/controlled_qualification_result.json"
            failure_path = root / "controlled_qualification/controlled_qualification_failure.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                record["terminal"] = result.get("terminal")
                record["qualified"] = result.get("qualified") is True
            elif failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                record["terminal"] = failure.get("terminal")
                record["qualified"] = False
                record["failure"] = failure.get("failure")
            else:
                record["terminal"] = IMPLEMENTATION_FAILED
                record["qualified"] = False
                record["failure"] = "controlled client emitted no result evidence"
    finally:
        record["carla_shutdown"] = supervisor.stop_carla(server, pgid, args.port)
    record["wall_seconds"] = time.monotonic() - started
    record["accepted"] = bool(
        code == 0
        and record.get("terminal") == QUALIFIED
        and record.get("qualified") is True
        and record["carla_shutdown"].get("shutdown_verified") is True
    )
    if record.get("terminal") not in {QUALIFIED, BLOCKED, IMPLEMENTATION_FAILED}:
        record["terminal"] = IMPLEMENTATION_FAILED
        record["accepted"] = False
    write_json_x(root / "controlled_supervision.json", record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record["accepted"] else (3 if record["terminal"] == BLOCKED else 4)


if __name__ == "__main__":
    raise SystemExit(main())
