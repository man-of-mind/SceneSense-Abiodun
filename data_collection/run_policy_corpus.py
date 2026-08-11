#!/usr/bin/env python3
"""Run the pre-registered policy-corpus collection batch on L10319."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPING_ROOT = REPO_ROOT.parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "policy_corpus_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("collection config schema_version must be 1")
    run_ids = [str(item["episode_id"]) for item in config["runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("collection episode_id values must be unique")
    return config


def _gpu_inventory() -> Dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def _live_carla_preflight(config: Mapping[str, object]) -> Dict[str, object]:
    connection = config["carla"]
    probe = (
        "import carla,json,sys;"
        "c=carla.Client(sys.argv[1],int(sys.argv[2]));"
        "c.set_timeout(float(sys.argv[3]));"
        "w=c.get_world();"
        "a=w.get_actors();"
        "counts={p:len(a.filter(p)) for p in ['vehicle.*','walker.pedestrian.*','sensor.*','controller.ai.walker']};"
        "print(json.dumps({'town':w.get_map().name,'server_version':c.get_server_version(),'dynamic_actor_counts':counts}))"
    )
    payload: Dict[str, object] | None = None
    last_error = ""
    for _attempt in range(5):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(connection["host"]),
                str(connection["port"]),
                str(connection.get("timeout_s", 10.0)),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=float(connection.get("timeout_s", 10.0)) + 2.0,
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout.strip())
            break
        last_error = result.stderr.strip() or result.stdout.strip()
        time.sleep(0.5)
    if payload is None:
        raise RuntimeError(f"CARLA get_world failed after five attempts: {last_error}")
    town = str(payload["town"])
    server_version = str(payload["server_version"])
    expected_town = str(connection["expected_town"])
    expected_version = str(connection["expected_server_version"])
    if not town.endswith(expected_town):
        raise RuntimeError(f"expected loaded map {expected_town}, found {town}")
    if server_version != expected_version:
        raise RuntimeError(
            f"expected CARLA server {expected_version}, found {server_version}"
        )
    return {
        "town": town,
        "server_version": server_version,
        "dynamic_actor_counts": {
            str(name): int(count)
            for name, count in payload["dynamic_actor_counts"].items()
        },
    }


def _static_preflight(config: Mapping[str, object]) -> Dict[str, object]:
    provenance = config["provenance"]
    files = {
        "collector": _resolve_repo_path(str(config["collector"])),
        "checkpoint": _resolve_repo_path(str(provenance["checkpoint"])),
        "carla_version_file": SHIPPING_ROOT / str(provenance["carla_version_file"]),
        "town_umap": SHIPPING_ROOT / str(provenance["town_umap"]),
        "town_uexp": SHIPPING_ROOT / str(provenance["town_uexp"]),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prerequisites: " + ", ".join(missing))
    hashes = {name: _sha256(path) for name, path in files.items()}
    expected_checkpoint = str(provenance.get("checkpoint_sha256", ""))
    if expected_checkpoint and hashes["checkpoint"] != expected_checkpoint:
        raise RuntimeError("fusion checkpoint hash differs from pre-registered value")
    return {
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": hashes[name]}
            for name, path in files.items()
        },
        "python": sys.executable,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "loadavg": Path("/proc/loadavg").read_text(encoding="utf-8").strip(),
        "gpu": _gpu_inventory(),
    }


def _run_command(
    config: Mapping[str, object],
    run_spec: Mapping[str, object],
    run_dir: Path,
) -> List[str]:
    family = str(run_spec["scenario_family"])
    command = [sys.executable, str(_resolve_repo_path(str(config["collector"])))]
    command.extend(str(value) for value in config["common_args"])
    command.extend(str(value) for value in config["family_args"][family])
    command.extend(
        [
            "--seed",
            str(run_spec["seed"]),
            "--run-id",
            str(run_spec["episode_id"]),
            "--run-group",
            str(run_spec["run_group"]),
            "--metrics-run-dir",
            str(run_dir),
        ]
    )
    if str(run_spec.get("split", "")) == "smoke":
        command.extend(
            [
                "--overlay-save-dir",
                str(run_dir / "overlays"),
                "--overlay-save-every",
                "20",
            ]
        )
    for value in run_spec.get("extra_args", []):
        command.append(str(value))
    return command


def _single_csv(run_dir: Path, suffix: str) -> Path:
    matches = sorted((run_dir / "streams").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *{suffix} under {run_dir}, found {len(matches)}")
    return matches[0]


def _basic_run_gate(
    run_dir: Path,
    run_spec: Mapping[str, object],
    config: Mapping[str, object],
) -> Dict[str, object]:
    metrics_path = _single_csv(run_dir, "_metrics.csv")
    gt_path = _single_csv(run_dir, "_object_ground_truth.csv")
    prediction_path = _single_csv(run_dir, "_object_predictions.csv")
    metrics = pd.read_csv(metrics_path)
    gt = pd.read_csv(gt_path)
    predictions = pd.read_csv(prediction_path)
    wait = pd.to_numeric(metrics["camera_frame_wait_ms"], errors="coerce").dropna()
    requested = int(run_spec.get("requested_frames", config["requested_frames"]))
    timing = config["timing_gate"]
    summary = {
        "processed_frames": int(len(metrics)),
        "requested_frames": requested,
        "result_received_pct": 100.0 * float(metrics["result_received"].astype(bool).mean()),
        "gt_rows": int(len(gt)),
        "prediction_rows": int(len(predictions)),
        "pedestrian_gt_rows": int((gt["class_name"].astype(str) == "pedestrian").sum()),
        "camera_frame_wait_median_ms": float(wait.median()) if len(wait) else None,
        "camera_frame_wait_p95_ms": float(wait.quantile(0.95)) if len(wait) else None,
    }
    failures = []
    if len(metrics) < int(float(timing["minimum_processed_fraction"]) * requested):
        failures.append("too_few_processed_frames")
    if gt.empty:
        failures.append("empty_ground_truth")
    if predictions.empty:
        failures.append("empty_predictions")
    if summary["result_received_pct"] < float(timing["minimum_result_received_pct"]):
        failures.append("result_receive_collapse")
    if not len(wait):
        failures.append("missing_camera_frame_wait")
    else:
        if float(summary["camera_frame_wait_median_ms"]) > float(timing["median_max_ms"]):
            failures.append("camera_wait_median")
        if float(summary["camera_frame_wait_p95_ms"]) > float(timing["p95_max_ms"]):
            failures.append("camera_wait_p95")
    if str(run_spec["scenario_family"]) in {"ped_crossing", "mixed_urban"}:
        if int(summary["pedestrian_gt_rows"]) == 0:
            failures.append("missing_pedestrian_ground_truth")
    summary["pass"] = not failures
    summary["failures"] = failures
    return summary


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_batch(
    config_path: Path,
    mode: str,
    batch_dir: Path | None,
    dry_run: bool,
    only_episode_ids: Sequence[str] = (),
) -> Path:
    config = _load_config(config_path)
    selected_runs: Iterable[Mapping[str, object]]
    if mode == "smoke":
        selected_runs = config["smoke_runs"]
    else:
        selected_runs = config["runs"]
    selected_runs = list(selected_runs)
    if only_episode_ids:
        wanted = set(only_episode_ids)
        selected_runs = [item for item in selected_runs if str(item["episode_id"]) in wanted]
        found = {str(item["episode_id"]) for item in selected_runs}
        if found != wanted:
            raise ValueError("unknown --only-episode values: " + ", ".join(sorted(wanted - found)))
    preflight = _static_preflight(config)
    if not dry_run:
        preflight["live_carla"] = _live_carla_preflight(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if batch_dir is None:
        batch_dir = _resolve_repo_path(str(config["output_root"])) / f"{timestamp}_{mode}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    resolved_config_path = batch_dir / "resolved_collection_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest: MutableMapping[str, object] = {
        "schema": "policy_corpus_batch.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "dry_run" if dry_run else "running",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "batch_dir": str(batch_dir),
        "preflight": preflight,
        "runs": [],
    }
    manifest_path = batch_dir / "batch_manifest.json"
    _write_manifest(manifest_path, manifest)

    for run_spec in selected_runs:
        run_dir = batch_dir / "runs" / str(run_spec["episode_id"])
        command = _run_command(config, run_spec, run_dir)
        record: MutableMapping[str, object] = {
            **dict(run_spec),
            "command": command,
            "run_dir": str(run_dir),
            "status": "planned" if dry_run else "running",
        }
        manifest["runs"].append(record)
        _write_manifest(manifest_path, manifest)
        if dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log_stream:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = int(result.returncode)
        record["postflight_carla"] = _live_carla_preflight(config)
        baseline_actors = preflight["live_carla"]["dynamic_actor_counts"]
        postflight_actors = record["postflight_carla"]["dynamic_actor_counts"]
        actor_cleanup_pass = postflight_actors == baseline_actors
        record["actor_cleanup_pass"] = actor_cleanup_pass
        if not actor_cleanup_pass:
            record["status"] = "actor_cleanup_failed"
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"dynamic CARLA actors leaked after {run_spec['episode_id']}: "
                f"baseline={baseline_actors}, postflight={postflight_actors}"
            )
        try:
            record["basic_gate"] = _basic_run_gate(run_dir, run_spec, config)
        except Exception as exc:
            record["status"] = "basic_gate_error"
            record["basic_gate_error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        known_teardown_abort = (
            result.returncode == -6
            and "libc++abi" in log_tail
            and "std::exception" in log_tail
        )
        record["known_carla_teardown_abort"] = known_teardown_abort
        if result.returncode == 0:
            record["status"] = "complete" if record["basic_gate"]["pass"] else "gate_failed"
        elif record["basic_gate"]["pass"] and known_teardown_abort:
            # CARLA 0.10 can abort in the C++ client destructor after all files
            # are flushed and all dynamic actors are gone. Accept only that
            # narrowly verified teardown-only case and preserve the return code.
            record["status"] = "complete_with_teardown_warning"
            record["accepted_nonzero_returncode"] = True
        else:
            record["status"] = "collector_failed"
        _write_manifest(manifest_path, manifest)
        if not record["basic_gate"]["pass"]:
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"basic gate failed for {run_spec['episode_id']}: "
                + ", ".join(record["basic_gate"]["failures"])
            )
        if result.returncode != 0 and not known_teardown_abort:
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"collector exited {result.returncode} for {run_spec['episode_id']}; "
                f"see {log_path}"
            )

    if not dry_run:
        manifest["status"] = "collection_complete_pending_verification"
    _write_manifest(manifest_path, manifest)
    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-episode", action="append", default=[])
    args = parser.parse_args()
    output = run_batch(
        args.config.resolve(), args.mode, args.batch_dir, args.dry_run, args.only_episode
    )
    print(output)


if __name__ == "__main__":
    main()
