#!/usr/bin/env python3
"""Run one declared CARLA renderer-quality stage of the paired perception gate.

The CARLA RPC does not expose a trustworthy engine-quality query.  Therefore
the operator must launch CARLA with the printed flag and declare the same level
here.  This runner records that declaration, reuses the reviewed paired causal
collector, and never authorizes OAI, a full corpus, controller evaluation, or
RL training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import yaml

from data_collection.run_phase2_paired_causal_pilot import (
    REPO_ROOT,
    _load_config as load_paired_config,
    _repo_path,
    build_plan,
    run_live,
)
from phase2_map_sharing.pilot_contract import load_and_validate_pilot_config
from phase2_map_sharing.run_pilot_contract_preflight import preflight


DEFAULT_CONFIG = (
    REPO_ROOT / "data_collection" / "configs" / "phase2_renderer_quality_gate_v1.yaml"
)
QUALITY_LEVELS = ("Low", "Epic")


def _write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_config(path: Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("renderer gate config root must be a mapping")
    if config.get("schema_version") != "scenesense.phase2_renderer_quality_gate.v1":
        raise ValueError("unexpected renderer gate schema")
    if config.get("implementation_status") != "reviewed_fail_fast_gate_only":
        raise ValueError("renderer gate must remain reviewed_fail_fast_gate_only")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "carla_launch",
        "oai_launch",
        "full_collection",
    }:
        raise ValueError("renderer gate authorization mapping is incomplete")
    if not bool(authorization["carla_launch"]) or bool(
        authorization["oai_launch"]
    ) or bool(authorization["full_collection"]):
        raise ValueError("renderer gate may authorize only its short CARLA capture")

    comparison = config.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("renderer comparison mapping is required")
    allowed = tuple(str(value) for value in comparison["allowed_quality_levels"])
    required = tuple(str(value) for value in comparison["required_quality_levels"])
    if set(allowed) != set(QUALITY_LEVELS) or set(required) != set(QUALITY_LEVELS):
        raise ValueError("renderer gate must compare exactly Low and Epic")
    flags = comparison.get("server_launch_flag_by_quality")
    if not isinstance(flags, Mapping) or {
        str(key): str(value) for key, value in flags.items()
    } != {"Low": "-quality-level=Low", "Epic": "-quality-level=Epic"}:
        raise ValueError("renderer server launch flags have drifted")
    if comparison.get("quality_verification") != (
        "operator_declared_server_launch_flag"
    ):
        raise ValueError("renderer quality must be explicitly operator-declared")

    capture = config.get("capture")
    metrics = config.get("metrics")
    if not isinstance(capture, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("renderer capture and metrics mappings are required")
    if int(capture["frames_per_trajectory"]) != 120 or not bool(
        capture["exact_sensor_contract"]
    ):
        raise ValueError("renderer gate must retain the reviewed 120-frame sensor contract")
    if bool(capture["inference_timing_is_citable"]):
        raise ValueError("shared-GPU renderer timing is diagnostic only")
    if float(metrics["primary_score_threshold"]) < 0.05:
        raise ValueError("postdecoder analysis cannot recover candidates below 0.05")
    thresholds = [float(value) for value in metrics["postdecoder_score_thresholds"]]
    if thresholds != sorted(set(thresholds)) or thresholds[0] != 0.05:
        raise ValueError("renderer score thresholds must be unique, sorted, and start at 0.05")
    if set(metrics["near_range_m_by_class"]) != {"pedestrian", "vehicle"}:
        raise ValueError("renderer gate requires pedestrian and vehicle range contracts")

    contract_path = _repo_path(config["contract_config"])
    contract_summary = load_and_validate_pilot_config(contract_path)
    if not bool(contract_summary["live_run_authorized"]):
        raise ValueError("renderer gate contract does not authorize its short CARLA run")
    with contract_path.open("r", encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    retention = contract["raw_retention"]
    if float(retention["maximum_window_seconds_per_trajectory"]) != float(
        capture["retained_raw_window_seconds"]
    ):
        raise ValueError("renderer integration and raw-retention window differ")
    if retention.get("enforcement_scope") != "per_collector_process_one_role":
        raise ValueError("renderer raw quota must declare its process-local scope")
    aggregate_cap = int(capture["aggregate_raw_bytes_cap"])
    if int(retention["maximum_raw_bytes_all_four_role_runs"]) != aggregate_cap:
        raise ValueError("renderer integration and aggregate raw-byte cap differ")
    if 4 * int(retention["maximum_raw_bytes_pilot_total"]) > aggregate_cap:
        raise ValueError("process-local raw quotas can exceed the aggregate gate cap")
    return config


def resolve_stage(
    config_path: Path, declared_quality: str
) -> tuple[dict, dict, dict, dict]:
    gate = load_gate_config(config_path)
    quality = str(declared_quality)
    if quality not in QUALITY_LEVELS:
        raise ValueError(f"renderer quality must be one of {QUALITY_LEVELS}")

    base_path = _repo_path(gate["base_integration_config"])
    effective, source, _base_contract = load_paired_config(base_path)
    effective = copy.deepcopy(effective)
    effective["contract_config"] = str(gate["contract_config"])
    effective["output_root"] = str(gate["output_root"])
    quality_slug = quality.lower()
    comparison = gate["comparison"]
    effective["renderer_quality"] = {
        "schema": "scenesense.renderer_quality_declaration.v1",
        "comparison_id": str(comparison["comparison_id"]),
        "declared_quality_level": quality,
        "required_server_launch_flag": str(
            comparison["server_launch_flag_by_quality"][quality]
        ),
        "verification_source": str(comparison["quality_verification"]),
        "rpc_introspection_available": False,
        "inference_timing_citable": False,
    }
    for trajectory in effective["trajectories"]:
        role = str(trajectory["scenario_role"])
        suffix = "positive" if role == "controlled_positive_occlusion" else "benign"
        trajectory["trajectory_id"] = f"renderer_{quality_slug}_{suffix}_001"
        trajectory["matched_pair_id"] = "renderer_quality_pair_001"
        trajectory["population_family"] = "frozen_renderer_quality_gate"
        trajectory["seed"] = int(gate["capture"]["seed"])

    contract_path = _repo_path(effective["contract_config"])
    contract_summary = load_and_validate_pilot_config(contract_path)
    if not bool(contract_summary["live_run_authorized"]):
        raise ValueError("resolved renderer contract is not live-gate authorized")
    return gate, effective, source, contract_summary


def build_launch_spec(
    config_path: Path,
    declared_quality: str,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> dict:
    config_path = Path(config_path).resolve()
    gate, effective, source, contract_summary = resolve_stage(
        config_path, declared_quality
    )
    quality = str(declared_quality)
    slug = quality.lower()
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else _repo_path(gate["output_root"])
    )
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not stamp.replace("_", "").isalnum():
        raise ValueError("launch timestamp contains unsupported characters")
    batch_root = root / f"{stamp}_{slug}"
    plan = build_plan(effective, source, batch_root)
    if not bool(plan["live_authorized"]) or len(plan["trajectories"]) != 2:
        raise ValueError("resolved renderer stage is not the two-trajectory gate")

    contract_path = _repo_path(effective["contract_config"])
    storage = preflight(contract_path, REPO_ROOT)
    if not bool(storage["live_pilot_authorized"]):
        raise ValueError("renderer storage preflight did not retain gate authorization")
    aggregate_raw_cap = int(gate["capture"]["aggregate_raw_bytes_cap"])
    free_bytes = int(shutil.disk_usage(root if root.exists() else REPO_ROOT).free)
    minimum_reserve = int(storage["storage"]["minimum_free_bytes_after_reservation"])
    if free_bytes < minimum_reserve + aggregate_raw_cap:
        raise ValueError("insufficient free space for renderer-stage aggregate raw quota")
    log_path = root / f"{stamp}_{slug}.run.log"
    manifest_path = root / f"{stamp}_{slug}.launch.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.run_phase2_renderer_quality_gate",
        "--config",
        str(config_path),
        "--declared-renderer-quality",
        quality,
        "--output-dir",
        str(batch_root),
        "--run-live-internal",
    ]
    declaration = effective["renderer_quality"]
    return {
        "schema": "scenesense.phase2_renderer_quality_launch.v1",
        "status": "validated_not_started",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "contract_config": str(contract_path),
        "contract_config_sha256": _sha256(contract_path),
        "base_integration_config": str(_repo_path(gate["base_integration_config"])),
        "declared_renderer_quality": quality,
        "required_server_launch_flag": declaration["required_server_launch_flag"],
        "quality_verification": declaration["verification_source"],
        "quality_empirically_introspected": False,
        "batch_root": str(batch_root),
        "run_log": str(log_path),
        "launch_manifest": str(manifest_path),
        "command": command,
        "trajectory_count": 2,
        "sensor_contract": contract_summary["sensor_contract"],
        "inference_timing_citable": False,
        "storage_preflight": storage["storage"],
        "aggregate_raw_bytes_cap": aggregate_raw_cap,
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "progress_log": str(batch_root / "progress.jsonl"),
    }


def launch_detached(spec: Mapping[str, object]) -> dict:
    batch_root = Path(str(spec["batch_root"]))
    log_path = Path(str(spec["run_log"]))
    launch_manifest = Path(str(spec["launch_manifest"]))
    batch_root.parent.mkdir(parents=True, exist_ok=True)
    for path in (batch_root, log_path, launch_manifest):
        if path.exists():
            raise FileExistsError(f"refusing to reuse renderer-gate artifact: {path}")
    log_stream = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(value) for value in spec["command"]],
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
    _write_json_create(launch_manifest, launched)
    return launched


def run_stage(config_path: Path, declared_quality: str, output_dir: Path) -> dict:
    gate, effective, source, contract_summary = resolve_stage(
        config_path, declared_quality
    )
    output_dir = Path(output_dir).resolve()
    plan = build_plan(effective, source, output_dir)
    try:
        run_live(effective, source, plan, output_dir)
    except BaseException as exc:
        if output_dir.is_dir() and not (output_dir / "FAILED.json").exists():
            _write_json_create(
                output_dir / "FAILED.json",
                {
                    "schema": "scenesense.phase2_renderer_quality_sentinel.v1",
                    "status": "failed",
                    "declared_renderer_quality": str(declared_quality),
                    "error": f"{type(exc).__name__}: {exc}",
                    "written_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise
    retained_raw_bytes = sum(
        path.stat().st_size
        for trajectory in output_dir.iterdir()
        if trajectory.is_dir()
        for role in ("helper", "recipient")
        for path in (trajectory / role / "retained_inputs").glob("*")
        if path.is_file()
    )
    aggregate_cap = int(gate["capture"]["aggregate_raw_bytes_cap"])
    if retained_raw_bytes > aggregate_cap:
        _write_json_create(
            output_dir / "FAILED.json",
            {
                "schema": "scenesense.phase2_renderer_quality_sentinel.v1",
                "status": "failed",
                "declared_renderer_quality": str(declared_quality),
                "error": "aggregate retained raw-byte cap exceeded",
                "retained_raw_bytes": retained_raw_bytes,
                "aggregate_raw_bytes_cap": aggregate_cap,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise RuntimeError("aggregate retained raw-byte cap exceeded")
    summary = {
        "schema": "scenesense.phase2_renderer_quality_sentinel.v1",
        "status": "capture_complete_pending_paired_renderer_analysis",
        "declared_renderer_quality": str(declared_quality),
        "required_server_launch_flag": effective["renderer_quality"][
            "required_server_launch_flag"
        ],
        "quality_empirically_introspected": False,
        "sensor_contract": contract_summary["sensor_contract"],
        "retained_raw_bytes": retained_raw_bytes,
        "aggregate_raw_bytes_cap": aggregate_cap,
        "batch_root": str(output_dir),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_create(output_dir / "RESULTS_SUMMARY.json", summary)
    _write_json_create(output_dir / "COMPLETED.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--declared-renderer-quality", required=True, choices=QUALITY_LEVELS
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timestamp", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-launch", action="store_true")
    mode.add_argument("--launch-detached", action="store_true")
    mode.add_argument("--run-live-internal", action="store_true")
    args = parser.parse_args()

    if args.run_live_internal:
        if args.output_dir is None:
            raise ValueError("--run-live-internal requires --output-dir")
        result = run_stage(
            args.config.resolve(), args.declared_renderer_quality, args.output_dir
        )
    else:
        if args.output_dir is not None:
            raise ValueError("--output-dir is reserved for the detached child")
        spec = build_launch_spec(
            args.config,
            args.declared_renderer_quality,
            output_root=args.output_root,
            timestamp=args.timestamp,
        )
        result = launch_detached(spec) if args.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
