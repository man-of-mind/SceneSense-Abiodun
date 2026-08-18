#!/usr/bin/env python3
"""Run one bounded training-density Low/Epic renderer confirmation stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from data_collection.run_policy_corpus import REPO_ROOT, _load_config, run_batch


CONFIG_BY_QUALITY = {
    "Low": REPO_ROOT
    / "data_collection/configs/phase2_renderer_dense_confirmation_low_v1.yaml",
    "Epic": REPO_ROOT
    / "data_collection/configs/phase2_renderer_dense_confirmation_epic_v1.yaml",
}
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data_collection/experiments/phase2_renderer_dense_confirmation_v1"
)
MINIMUM_FREE_BYTES = 30_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_stage_config(quality: str) -> tuple[Path, dict]:
    if quality not in CONFIG_BY_QUALITY:
        raise ValueError(f"renderer quality must be one of {tuple(CONFIG_BY_QUALITY)}")
    path = CONFIG_BY_QUALITY[quality].resolve()
    config = _load_config(path)
    renderer = config.get("renderer_quality")
    if not isinstance(renderer, Mapping):
        raise ValueError("renderer-quality declaration is missing")
    if renderer.get("declared_quality_level") != quality:
        raise ValueError("renderer overlay and requested quality differ")
    if renderer.get("required_server_launch_flag") != f"-quality-level={quality}":
        raise ValueError("renderer launch flag and declaration differ")
    authorization = config.get("authorization")
    expected_authorization = {
        "carla_launch": True,
        "oai_launch": False,
        "full_collection": False,
        "controller_evaluation": False,
        "rl_training": False,
    }
    if authorization != expected_authorization:
        raise ValueError("dense confirmation may authorize only its short CARLA stage")
    if not bool(config["carla"].get("reload_world_before_run")):
        raise ValueError("matched renderer trials require a world reset before every run")
    if len(config.get("runs", [])) != 2 or config.get("smoke_runs"):
        raise ValueError("dense confirmation must remain exactly two runs per quality")
    expected_populations = {
        "medium": (20, 25),
        "crowded": (28, 35),
    }
    for family, (vehicles, pedestrians) in expected_populations.items():
        args = [str(value) for value in config["family_args"][family]]
        values = {args[index]: args[index + 1] for index in range(0, len(args), 2)}
        if int(values["--npc-vehicles"]) != vehicles or int(
            values["--npc-pedestrians"]
        ) != pedestrians:
            raise ValueError(f"{family} population differs from training lineage")
    return path, config


def build_launch_spec(
    quality: str,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> dict:
    config_path, config = load_stage_config(quality)
    root = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(root).free)
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"dense confirmation requires at least {MINIMUM_FREE_BYTES} free bytes; "
            f"found {free_bytes}"
        )
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = quality.lower()
    batch_root = root / f"{stamp}_{slug}"
    log_path = root / f"{stamp}_{slug}.run.log"
    launch_path = root / f"{stamp}_{slug}.launch.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.run_phase2_renderer_dense_confirmation",
        "--declared-renderer-quality",
        quality,
        "--output-dir",
        str(batch_root),
        "--run-live-internal",
    ]
    return {
        "schema": "scenesense.phase2_renderer_dense_confirmation_launch.v1",
        "status": "validated_not_started",
        "declared_renderer_quality": quality,
        "required_server_launch_flag": f"-quality-level={quality}",
        "quality_empirically_introspected": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "comparison_id": config["renderer_quality"]["comparison_id"],
        "batch_root": str(batch_root),
        "run_log": str(log_path),
        "launch_manifest": str(launch_path),
        "command": command,
        "run_count": 2,
        "scenario_families": ["medium", "crowded"],
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "observed_free_bytes": free_bytes,
    }


def launch_detached(spec: Mapping[str, object]) -> dict:
    batch_root = Path(str(spec["batch_root"]))
    log_path = Path(str(spec["run_log"]))
    launch_path = Path(str(spec["launch_manifest"]))
    for path in (batch_root, log_path, launch_path):
        if path.exists():
            raise FileExistsError(f"refusing to reuse renderer artifact: {path}")
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
    _write_json_create(launch_path, launched)
    return launched


def run_stage(quality: str, output_dir: Path) -> dict:
    config_path, _config = load_stage_config(quality)
    output_dir = Path(output_dir).resolve()
    try:
        run_batch(config_path, "full", output_dir, False)
        manifest = json.loads(
            (output_dir / "batch_manifest.json").read_text(encoding="utf-8")
        )
        complete_statuses = {"complete", "complete_with_teardown_warning"}
        if manifest.get("status") != "collection_complete_pending_verification":
            raise RuntimeError("generic collection runner did not finish cleanly")
        if len(manifest.get("runs", [])) != 2 or any(
            item.get("status") not in complete_statuses for item in manifest["runs"]
        ):
            raise RuntimeError("one or more dense confirmation runs is incomplete")
    except BaseException as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (output_dir / "FAILED.json").exists():
            _write_json_create(
                output_dir / "FAILED.json",
                {
                    "schema": "scenesense.phase2_renderer_dense_confirmation_sentinel.v1",
                    "status": "failed",
                    "declared_renderer_quality": quality,
                    "error": f"{type(exc).__name__}: {exc}",
                    "written_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        raise
    summary = {
        "schema": "scenesense.phase2_renderer_dense_confirmation_sentinel.v1",
        "status": "capture_complete_pending_paired_renderer_analysis",
        "declared_renderer_quality": quality,
        "required_server_launch_flag": f"-quality-level={quality}",
        "quality_empirically_introspected": False,
        "run_count": 2,
        "scenario_families": ["medium", "crowded"],
        "batch_root": str(output_dir),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_create(output_dir / "RESULTS_SUMMARY.json", summary)
    _write_json_create(output_dir / "COMPLETED.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declared-renderer-quality", required=True, choices=tuple(CONFIG_BY_QUALITY)
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
        result = run_stage(args.declared_renderer_quality, args.output_dir)
    else:
        if args.output_dir is not None:
            raise ValueError("--output-dir is reserved for the detached child")
        spec = build_launch_spec(
            args.declared_renderer_quality,
            output_root=args.output_root,
            timestamp=args.timestamp,
        )
        result = launch_detached(spec) if args.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
