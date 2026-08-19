#!/usr/bin/env python3
"""Validate and detach only the frozen Phase-2 calibration audit stage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from data_collection.run_phase2_calibration_audit import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    _load_config,
    _repo_path,
    _select_trajectory_ids,
    _sha256,
    build_plan,
)


def build_launch_spec(
    config_path: Path,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
    operator_quality: str = "Epic",
    trajectory_ids: Optional[list[str]] = None,
) -> dict:
    config_path = Path(config_path).resolve()
    config, source, selected = _load_config(config_path)
    selected = _select_trajectory_ids(selected, trajectory_ids or [])
    if str(operator_quality) != str(config["carla"]["renderer_quality_level"]):
        raise ValueError("operator renderer declaration differs from frozen Epic contract")
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else _repo_path(config["output_root"])
    )
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not stamp.replace("_", "").isalnum():
        raise ValueError("launch timestamp contains unsupported characters")
    batch_root = root / f"{stamp}_audit"
    plan = build_plan(config, source, selected, batch_root)
    if not trajectory_ids and (
        int(plan["group_count"]) != 9 or int(plan["trajectory_count"]) != 15
    ):
        raise ValueError("detached plan is not the exact frozen audit stage")
    free_bytes = shutil.disk_usage(root.parent).free
    required = int(config["storage"]["preflight_required_free_bytes"])
    if free_bytes < required:
        raise RuntimeError(
            f"audit storage preflight failed: free={free_bytes}, required={required}"
        )
    log_path = root / f"{stamp}_audit.run.log"
    launch_manifest_path = root / f"{stamp}_audit.launch.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.run_phase2_calibration_audit",
        "--config",
        str(config_path),
        "--output-dir",
        str(batch_root),
        "--operator-quality",
        str(operator_quality),
        "--launch",
    ]
    for trajectory_id in trajectory_ids or []:
        command.extend(("--trajectory-id", str(trajectory_id)))
    return {
        "schema": "scenesense.phase2_calibration_audit_detached_launch.v1",
        "status": "validated_not_started",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "batch_root": str(batch_root),
        "run_log": str(log_path),
        "launch_manifest": str(launch_manifest_path),
        "command": command,
        "operator_quality": str(operator_quality),
        "required_server_launch_flag": config["carla"]["required_server_launch_flag"],
        "trajectory_count": int(plan["trajectory_count"]),
        "group_count": int(plan["group_count"]),
        "selection_scope": (
            "bounded_regression_subset" if trajectory_ids else "full_frozen_audit"
        ),
        "estimated_minutes": plan["estimated_minutes"],
        "storage_preflight": {
            "free_bytes": int(free_bytes),
            "required_free_bytes": required,
            "stage_hard_cap_bytes": int(config["storage"]["stage_hard_cap_bytes"]),
            "required_free_floor_bytes": int(config["storage"]["required_free_floor_bytes"]),
        },
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "progress_log": str(batch_root / "progress.jsonl"),
        "next_stage_chained": False,
        "oai_launched": False,
    }


def launch_detached(spec: Mapping[str, object]) -> dict:
    batch_root = Path(str(spec["batch_root"]))
    log_path = Path(str(spec["run_log"]))
    launch_manifest_path = Path(str(spec["launch_manifest"]))
    root = batch_root.parent
    root.mkdir(parents=True, exist_ok=True)
    for path in (batch_root, log_path, launch_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to reuse audit launch artifact: {path}")
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
        "status": "launched_detached_pending_startup_ack",
        "pid": int(process.pid),
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    with launch_manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(launched, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    # A PID is not evidence that a detached child survived the launcher
    # context. Require the child-created batch directory and plan/progress
    # artifact before reporting a successful launch.
    deadline = time.monotonic() + 15.0
    startup_evidence: Optional[Path] = None
    while time.monotonic() < deadline:
        for candidate in (
            batch_root / "progress.jsonl",
            batch_root / "plan.json",
            batch_root / "FAILED.json",
        ):
            if candidate.is_file():
                startup_evidence = candidate
                break
        if startup_evidence is not None:
            break
        return_code = process.poll()
        if return_code is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            failed = {
                **launched,
                "status": "startup_failed",
                "returncode": int(return_code),
                "run_log_tail": tail,
            }
            launch_manifest_path.write_text(
                json.dumps(failed, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "detached audit exited before startup acknowledgement: "
                f"returncode={return_code}; log_tail={tail!r}"
            )
        time.sleep(0.05)
    if startup_evidence is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        failed = {
            **launched,
            "status": "startup_ack_timeout_terminated",
            "run_log_tail": tail,
        }
        launch_manifest_path.write_text(
            json.dumps(failed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "detached audit produced no startup artifact within 15 s; child "
            f"was terminated; log_tail={tail!r}"
        )
    acknowledged = {
        **launched,
        "status": "launched_detached_startup_acknowledged",
        "startup_evidence": str(startup_evidence),
        "startup_ack_utc": datetime.now(timezone.utc).isoformat(),
    }
    launch_manifest_path.write_text(
        json.dumps(acknowledged, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return acknowledged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--operator-quality", choices=("Epic",), required=True)
    parser.add_argument("--trajectory-id", action="append", default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-launch", action="store_true")
    mode.add_argument("--launch-detached", action="store_true")
    args = parser.parse_args()
    spec = build_launch_spec(
        args.config,
        output_root=args.output_root,
        operator_quality=args.operator_quality,
        trajectory_ids=args.trajectory_id,
    )
    result = launch_detached(spec) if args.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
