#!/usr/bin/env python3
"""Launch one fresh Epic CARLA process for the single controlled qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import data_collection.run_route_b_collection_supervised_v1 as supervisor

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
    server_log = root / "carla_server.log"
    client_log = root / "controlled_client.log"
    server, pgid = supervisor.start_carla(args.port, server_log)
    started = time.monotonic()
    record = {
        "schema": "publication_visibility_controlled_supervision_v1",
        "output_root": str(root), "server_log": str(server_log),
        "client_log": str(client_log), "carla_pid": int(server.pid), "carla_pgid": int(pgid),
    }
    code = 3
    try:
        version = supervisor.wait_for_rpc(args.port, args.ready_timeout_s)
        record["carla_server_version"] = version
        record["carla_rpc_ready"] = version is not None
        if version is None:
            record["failure"] = "fresh Epic CARLA did not become RPC-ready"
        else:
            command = [
                str(supervisor.VENV_PYTHON), "-m",
                "data_collection.route_b_publication_instance_visibility_v1.controlled_scene",
                "--host", args.host, "--port", str(args.port),
                "--output-dir", str(root / "controlled_scene"),
            ]
            record["client_argv"] = command
            with client_log.open("xb") as stream:
                child = subprocess.run(
                    command, cwd=str(supervisor.HERE.parent), stdout=stream,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=supervisor.child_env(),
                )
            code = int(child.returncode)
            record["client_exit"] = supervisor.describe_exit(code)
    finally:
        record["carla_shutdown"] = supervisor.stop_carla(server, pgid, args.port)
    result_path = root / "controlled_scene/controlled_scene_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record["controlled_terminal"] = result.get("terminal")
        record["controlled_qualified"] = result.get("qualified") is True
    else:
        record["controlled_terminal"] = "PUBLICATION_VISIBILITY_GROUND_TRUTH_BLOCKED"
        record["controlled_qualified"] = False
    record["wall_seconds"] = time.monotonic() - started
    record["accepted"] = bool(
        code == 0 and record["controlled_qualified"]
        and record["carla_shutdown"].get("shutdown_verified") is True
    )
    write_json_x(root / "controlled_supervision.json", record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
