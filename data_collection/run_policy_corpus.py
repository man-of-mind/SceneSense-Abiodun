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


def _resolved_run_args(
    config: Mapping[str, object], run_spec: Mapping[str, object]
) -> List[str]:
    family = str(run_spec["scenario_family"])
    if family not in config["family_args"]:
        raise ValueError(
            f"unknown scenario_family {family!r} for {run_spec.get('episode_id', '<unknown>')}"
        )
    return [
        *(str(value) for value in config["common_args"]),
        *(str(value) for value in config["family_args"][family]),
        *(str(value) for value in run_spec.get("extra_args", [])),
    ]


def _effective_options(arguments: Sequence[str]) -> Dict[str, str]:
    """Return argparse-style last-option-wins values for config validation."""

    options: Dict[str, str] = {}
    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(arguments) and not str(arguments[index + 1]).startswith("--"):
            options[token] = str(arguments[index + 1])
            index += 2
        else:
            options[token] = "true"
            index += 1
    return options


def _validate_collection_contract(config: Mapping[str, object]) -> None:
    contract = config.get("collection_contract")
    if not isinstance(contract, Mapping):
        return

    all_runs = [*config.get("smoke_runs", []), *config.get("runs", [])]
    required = {
        str(option): str(value)
        for option, value in contract.get("required_effective_args", {}).items()
    }
    for run_spec in all_runs:
        episode_id = str(run_spec["episode_id"])
        options = _effective_options(_resolved_run_args(config, run_spec))
        for option, expected in required.items():
            actual = options.get(option)
            if actual != expected:
                raise ValueError(
                    f"{episode_id} must resolve {option}={expected!r}, found {actual!r}"
                )
        if str(contract.get("scope", "")) == "vehicle_only":
            if options.get("--npc-pedestrians") != "0":
                raise ValueError(f"{episode_id} vehicle-only run must request zero pedestrians")
            if options.get("--controlled-target") == "walker":
                raise ValueError(f"{episode_id} vehicle-only run cannot use a walker target")
        if bool(contract.get("require_requested_frames_match_max_frames", False)):
            requested = str(run_spec.get("requested_frames", config["requested_frames"]))
            if options.get("--max-frames") != requested:
                raise ValueError(
                    f"{episode_id} requested_frames={requested} but resolves "
                    f"--max-frames={options.get('--max-frames')!r}"
                )

    expected_counts = contract.get("required_family_split_counts", {})
    if expected_counts:
        observed: Dict[str, Dict[str, int]] = {}
        for run_spec in config.get("runs", []):
            family = str(run_spec["scenario_family"])
            split = str(run_spec["split"])
            family_counts = observed.setdefault(family, {})
            family_counts[split] = family_counts.get(split, 0) + 1
        normalized_expected = {
            str(family): {str(split): int(count) for split, count in counts.items()}
            for family, counts in expected_counts.items()
        }
        if observed != normalized_expected:
            raise ValueError(
                "full-run family/split counts differ from collection contract: "
                f"expected={normalized_expected}, observed={observed}"
            )
    if bool(contract.get("require_unique_full_run_seeds", False)):
        seeds = [int(run_spec["seed"]) for run_spec in config.get("runs", [])]
        if len(seeds) != len(set(seeds)):
            raise ValueError("full-run seeds must be unique across whole-trajectory splits")


def _deep_merge_config(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> Dict[str, object]:
    merged: Dict[str, object] = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _replace_config_strings(value: object, old: str, new: str) -> object:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_config_strings(item, old, new) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_config_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def _apply_common_argument_overrides(
    arguments: Sequence[object], overrides: Mapping[str, object]
) -> List[str]:
    tokens = [str(value) for value in arguments]
    for option, value in overrides.items():
        option = str(option)
        retained: List[str] = []
        index = 0
        while index < len(tokens):
            if tokens[index] != option:
                retained.append(tokens[index])
                index += 1
                continue
            index += 1
            if index < len(tokens) and not tokens[index].startswith("--"):
                index += 1
        tokens = retained
        if value is None or value is False:
            continue
        tokens.append(option)
        if value is not True:
            tokens.append(str(value))
    return tokens


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("collection config must be a YAML mapping")
    extends = raw.get("extends")
    if extends:
        base_path = Path(str(extends)).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = _load_config(base_path.resolve())
        revision = raw.get("revision_replace", {})
        if revision:
            if not isinstance(revision, Mapping) or set(revision) != {"from", "to"}:
                raise ValueError("revision_replace must contain exactly from/to")
            config = _replace_config_strings(
                config, str(revision["from"]), str(revision["to"])
            )
        overrides = raw.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("collection config overrides must be a mapping")
        config = _deep_merge_config(config, overrides)
        argument_overrides = raw.get("common_argument_overrides", {})
        if not isinstance(argument_overrides, Mapping):
            raise ValueError("common_argument_overrides must be a mapping")
        config["common_args"] = _apply_common_argument_overrides(
            config["common_args"], argument_overrides
        )
        config["config_lineage"] = {
            "extends": str(base_path.resolve()),
            "revision_replace": dict(revision),
            "source_config": str(path.resolve()),
        }
    else:
        config = dict(raw)
    run_overrides = raw.get("run_overrides_by_family", {})
    if not isinstance(run_overrides, Mapping):
        raise ValueError("run_overrides_by_family must be a mapping")
    if run_overrides:
        for run_list_name in ("smoke_runs", "runs"):
            revised_runs = []
            for original in config.get(run_list_name, []):
                item = dict(original)
                family_override = run_overrides.get(str(item["scenario_family"]), {})
                if not isinstance(family_override, Mapping):
                    raise ValueError(
                        "run_overrides_by_family entries must be mappings"
                    )
                unknown = set(family_override) - {
                    "requested_frames",
                    "argument_overrides",
                }
                if unknown:
                    raise ValueError(
                        "unsupported family run override fields: "
                        + ", ".join(sorted(unknown))
                    )
                if "requested_frames" in family_override:
                    item["requested_frames"] = int(
                        family_override["requested_frames"]
                    )
                argument_overrides = family_override.get("argument_overrides", {})
                if not isinstance(argument_overrides, Mapping):
                    raise ValueError("family argument_overrides must be a mapping")
                item["extra_args"] = _apply_common_argument_overrides(
                    item.get("extra_args", []), argument_overrides
                )
                revised_runs.append(item)
            config[run_list_name] = revised_runs
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("collection config schema_version must be 1")
    all_runs = [*config.get("smoke_runs", []), *config.get("runs", [])]
    run_ids = [str(item["episode_id"]) for item in all_runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("collection episode_id values must be unique")
    for item in all_runs:
        _resolved_run_args(config, item)
    _validate_collection_contract(config)
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
    # CARLA can transiently abort RPC handshakes for a few seconds while the
    # renderer is warming or while ``nvidia-smi`` releases its driver query.
    # Treat that as startup backpressure, not as evidence that the world is
    # unavailable; the bounded retry still fails closed before any run starts.
    attempts = 10
    for _attempt in range(attempts):
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
        if _attempt + 1 < attempts:
            time.sleep(1.0)
    if payload is None:
        raise RuntimeError(
            f"CARLA get_world failed after {attempts} attempts: {last_error}"
        )
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
    actor_counts = {
        str(name): int(count)
        for name, count in payload["dynamic_actor_counts"].items()
    }
    required_empty = [
        str(value) for value in connection.get("require_empty_dynamic_actor_patterns", [])
    ]
    occupied = {
        name: actor_counts.get(name, 0)
        for name in required_empty
        if actor_counts.get(name, 0)
    }
    if occupied:
        raise RuntimeError(
            "collection requires an uncontaminated CARLA world; dynamic actors already exist: "
            f"{occupied}"
        )
    return {
        "town": town,
        "server_version": server_version,
        "dynamic_actor_counts": actor_counts,
    }


def _reload_carla_world(config: Mapping[str, object]) -> Dict[str, object]:
    """Reset the declared Town before a matched, independently seeded run.

    ``load_world(..., True)`` resets world settings as well as actors.  That is
    important here: a previous synchronous collector must not leak either its
    actors *or* its clock settings into the next renderer-quality trial.
    """

    connection = config["carla"]
    try:
        import carla

        client = carla.Client(
            str(connection["host"]), int(connection["port"])
        )
        client.set_timeout(float(connection.get("timeout_s", 10.0)))
        world = client.load_world(str(connection["expected_town"]), True)
    except Exception as exc:
        raise RuntimeError(f"CARLA world reset failed: {type(exc).__name__}: {exc}") from exc
    reset = {
        "requested_map": str(connection["expected_town"]),
        "resolved_map": str(world.get_map().name),
        "reset_settings": True,
    }
    reset["post_reset_preflight"] = _live_carla_preflight(config)
    return reset


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
        run_baseline = preflight["live_carla"]
        if bool(config["carla"].get("reload_world_before_run", False)):
            record["world_reload"] = _reload_carla_world(config)
            run_baseline = record["world_reload"]["post_reset_preflight"]
            _write_manifest(manifest_path, manifest)
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
        baseline_actors = run_baseline["dynamic_actor_counts"]
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
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the config/collection contract and exit without preflight or filesystem output.",
    )
    parser.add_argument("--only-episode", action="append", default=[])
    args = parser.parse_args()
    if args.validate_config:
        config = _load_config(args.config.resolve())
        print(
            json.dumps(
                {
                    "experiment_name": str(config["experiment_name"]),
                    "output_root": str(config["output_root"]),
                    "smoke_runs": len(config.get("smoke_runs", [])),
                    "full_runs": len(config.get("runs", [])),
                    "status": "VALID",
                },
                sort_keys=True,
            )
        )
        return
    output = run_batch(
        args.config.resolve(), args.mode, args.batch_dir, args.dry_run, args.only_episode
    )
    print(output)


if __name__ == "__main__":
    main()
