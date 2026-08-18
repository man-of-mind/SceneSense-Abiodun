#!/usr/bin/env python3
"""Validate and detach the reviewed two-trajectory Phase-2 pilot.

This launcher exits immediately after starting the self-logging runner. It
cannot authorize OAI, a full corpus, controller evaluation, or RL training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from data_collection.run_phase2_paired_causal_pilot import (
    REPO_ROOT,
    _load_config,
    _repo_path,
    build_plan,
)
from phase2_map_sharing.run_pilot_contract_preflight import preflight


DEFAULT_CONFIG = (
    REPO_ROOT
    / "data_collection"
    / "configs"
    / "phase2_paired_causal_pilot_reviewed_v1.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_launch_spec(
    config_path: Path,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> dict:
    config_path = Path(config_path).resolve()
    config, source, contract = _load_config(config_path)
    if config["implementation_status"] != "reviewed_pilot_only" or not bool(
        contract["live_run_authorized"]
    ):
        raise ValueError("detached launcher requires the reviewed pilot-only config")
    if bool(config["authorization"]["oai_launch"]) or bool(
        config["authorization"]["full_collection"]
    ):
        raise ValueError("detached pilot launcher forbids OAI and full collection")
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else _repo_path(config["output_root"])
    )
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not stamp.replace("_", "").isalnum():
        raise ValueError("launch timestamp contains unsupported characters")
    batch_root = root / f"{stamp}_pilot"
    plan = build_plan(config, source, batch_root)
    if not bool(plan["live_authorized"]) or len(plan["trajectories"]) != 2:
        raise ValueError("resolved plan is not the authorized two-trajectory pilot")
    contract_path = _repo_path(config["contract_config"])
    storage = preflight(contract_path, REPO_ROOT)
    if not bool(storage["live_pilot_authorized"]):
        raise ValueError("reviewed storage preflight did not retain pilot authorization")
    log_path = root / f"{stamp}_pilot.run.log"
    launch_manifest_path = root / f"{stamp}_pilot.launch.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.run_phase2_paired_causal_pilot",
        "--config",
        str(config_path),
        "--output-dir",
        str(batch_root),
        "--launch",
    ]
    return {
        "schema": "scenesense.phase2_detached_launch.v1",
        "status": "validated_not_started",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "contract_config": str(contract_path),
        "contract_config_sha256": _sha256(contract_path),
        "batch_root": str(batch_root),
        "run_log": str(log_path),
        "launch_manifest": str(launch_manifest_path),
        "command": command,
        "trajectory_count": 2,
        "population_mode": plan["population_mode"],
        "inference_timing_citable": False,
        "storage_preflight": storage["storage"],
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "progress_log": str(batch_root / "progress.jsonl"),
    }


def launch_detached(spec: Mapping[str, object]) -> dict:
    batch_root = Path(str(spec["batch_root"]))
    log_path = Path(str(spec["run_log"]))
    launch_manifest_path = Path(str(spec["launch_manifest"]))
    root = batch_root.parent
    root.mkdir(parents=True, exist_ok=True)
    for path in (batch_root, log_path, launch_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to reuse pilot launch artifact: {path}")
    log_stream = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(item) for item in spec["command"]],
            cwd=REPO_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        raise
    log_stream.close()
    launched = {
        **dict(spec),
        "status": "launched_detached",
        "pid": int(process.pid),
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    with launch_manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(launched, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-launch", action="store_true")
    mode.add_argument("--launch-detached", action="store_true")
    arguments = parser.parse_args()
    spec = build_launch_spec(arguments.config, output_root=arguments.output_root)
    result = launch_detached(spec) if arguments.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
