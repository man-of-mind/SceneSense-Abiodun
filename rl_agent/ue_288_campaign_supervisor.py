#!/usr/bin/env python3
"""Fail-closed supervisor and offline validator for the fixed UE 288 campaign.

This is deliberately campaign-specific.  It enumerates the frozen action and
network registries, enforces the qualified Route B contract, owns create-only
cell attempts and the resume ledger, and starts a fresh Epic CARLA process for
each real attempt.  The actual cell adapter is a narrow dependency because the
qualified Route B collector and the certified split runtime do not currently
share ego/clock ownership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "rl_agent/configs/ue_288_campaign_v1.yaml"
DEFAULT_PILOT = ROOT / "rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED_PREFIX = "__REQUIRED_"
TERMINAL_NAMES = ("PASSED.json", "FAILED.json", "INTERRUPTED.json")
LEDGER_SCHEMA = "scenesense.ue_288_campaign_ledger.v1"


class CampaignError(RuntimeError):
    """A campaign contract or launch prerequisite is not satisfied."""


@dataclass(frozen=True)
class Cell:
    cell_id: str
    action_index: int
    action_id: str
    display_action_id: str
    model_family: str
    network_profile_id: str
    trace_id: str
    seed: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML root must be a mapping: {path}")
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_create_only(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def read_registry(config: Mapping[str, Any]) -> list[dict[str, str]]:
    actions = config["actions"]
    path = repo_path(str(actions["technical_registry_csv"]))
    require(path.is_file(), f"technical registry missing: {path}")
    require(
        sha256_file(path) == str(actions["technical_registry_sha256"]),
        "technical registry SHA-256 drift",
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 72, f"technical registry must contain exactly 72 actions, found {len(rows)}")
    require(len({row["profile_id"] for row in rows}) == 72, "technical registry profile IDs are not unique")
    require(
        [int(row["action_index"]) for row in rows] == list(range(72)),
        "technical registry action_index must be contiguous 0..71",
    )
    expected_status = str(actions["required_certification_status"])
    invalid = [row["profile_id"] for row in rows if row["certification_status"] != expected_status]
    require(not invalid, f"uncertified action rows: {invalid[:4]}")
    return rows


def selected_actions(config: Mapping[str, Any], registry: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    actions = config["actions"]
    selection = str(actions["selection"])
    if selection == "all_registered":
        chosen = list(registry)
    elif selection == "explicit_display_profile_ids":
        requested = list(actions.get("display_profile_ids", []))
        require(len(requested) == len(set(requested)), "pilot action selectors are duplicated")
        by_display = {row["display_profile_id"]: row for row in registry}
        missing = [value for value in requested if value not in by_display]
        require(not missing, f"pilot action selectors are absent from registry: {missing}")
        chosen = [by_display[value] for value in requested]
    else:
        raise CampaignError(f"unsupported fixed action selection: {selection}")
    expected = int(actions["expected_count"])
    require(len(chosen) == expected, f"expected {expected} selected actions, found {len(chosen)}")
    return chosen


def network_profiles(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = list(config["network"].get("profiles", []))
    require(len(profiles) == 4, f"campaign must contain exactly four network profiles, found {len(profiles)}")
    require(len({row["profile_id"] for row in profiles}) == 4, "network profile IDs are not unique")
    require(len({row["trace_id"] for row in profiles}) == 4, "network trace IDs are not unique")
    require(len({int(row["seed"]) for row in profiles}) == 4, "network profile seeds are not unique")
    return profiles


def enumerate_cells(config: Mapping[str, Any]) -> list[Cell]:
    registry = read_registry(config)
    actions = selected_actions(config, registry)
    profiles = network_profiles(config)
    cells: list[Cell] = []
    for profile in profiles:
        for row in actions:
            cells.append(
                Cell(
                    cell_id=f"a{int(row['action_index']):02d}__{profile['profile_id'].lower()}",
                    action_index=int(row["action_index"]),
                    action_id=row["profile_id"],
                    display_action_id=row["display_profile_id"],
                    model_family=row["model_family"],
                    network_profile_id=str(profile["profile_id"]),
                    trace_id=str(profile["trace_id"]),
                    seed=int(profile["seed"]),
                )
            )
    require(len(cells) == int(config["cell"]["count"]), "enumerated cell count differs from config")
    require(len({cell.cell_id for cell in cells}) == len(cells), "enumerated cell IDs are not unique")
    require(
        len({(cell.action_id, cell.network_profile_id) for cell in cells}) == len(cells),
        "action/network Cartesian product contains duplicates",
    )
    return cells


def verify_file_hashes(config: Mapping[str, Any]) -> None:
    network = config["network"]
    for path_key, hash_key in (
        ("traces_csv", "traces_sha256"),
        ("summary_csv", "summary_sha256"),
        ("mapping_csv", "mapping_sha256"),
    ):
        path = repo_path(str(network[path_key]))
        require(path.is_file(), f"missing frozen network input: {path}")
        require(sha256_file(path) == str(network[hash_key]), f"{path_key} SHA-256 drift")
    route = config["route_b"]
    for path_key, hash_key in (
        ("route_json", "route_json_sha256"),
        ("progress_csv", "progress_csv_sha256"),
        ("qualified_density_runner", "qualified_density_runner_sha256"),
    ):
        path = repo_path(str(route[path_key]))
        require(path.is_file(), f"missing Route B input: {path}")
        require(sha256_file(path) == str(route[hash_key]), f"{path_key} SHA-256 drift")


def verify_route_contract(config: Mapping[str, Any]) -> None:
    route = config["route_b"]
    require(route["density"] == "traffic_50_50", "agent campaign density is not hard-locked to traffic_50_50")
    require(route["fresh_carla_process_and_world_per_cell"] is True, "fresh CARLA per cell is not required")
    require(route["carla_quality"] == "Epic" and route["render_offscreen"] is True, "CARLA must be Epic off-screen")
    require(route["no_rendering_mode"] is False, "CARLA no-rendering mode is forbidden")
    require(route["hybrid_physics"] is False, "hybrid physics must be disabled")
    require(int(route["loops_per_process"]) == 1, "Route B must run one loop per fresh process")
    require(route["allow_roadblock_clearing"] is True, "accepted stationary-roadblock clearing is not enabled")
    require(
        route["roadblock_policy"] == "STATIONARY_BLOCKER_RELOCATED_OR_DESTROYED_ONLY",
        "roadblock clearing policy drift",
    )
    require(route["forced_overtaking"] is False and int(route["maximum_overtakes"]) == 0, "forced overtaking is forbidden")
    collection = load_yaml(repo_path(str(route["collection_config"])))
    require(collection["route"]["config_sha256"] == route["route_json_sha256"], "Route B collection route hash drift")
    require(collection["route"]["progress_csv_sha256"] == route["progress_csv_sha256"], "Route B progress hash drift")
    require(collection["scenario"]["hybrid_physics"] is False, "Route B source enables hybrid physics")
    require(collection["scenario"]["roadblock_clearing"] is True, "Route B source lacks accepted roadblock policy")


def verify_output_contract(config: Mapping[str, Any]) -> None:
    cell = config["cell"]
    required_outputs = {
        "per_frame_metrics.csv",
        "radio_trace.csv",
        "map_feedback.csv",
        "perception_metrics.csv",
        "resolved_config.yaml",
        "RESULTS_SUMMARY.json",
        "manifest.json",
    }
    require(set(cell["expected_outputs"]) == required_outputs, "per-cell output set drift")
    require(tuple(cell["terminal_files"]) == TERMINAL_NAMES, "terminal file names/statuses drift")
    require(cell["exactly_one_terminal"] is True, "exactly-one-terminal policy is disabled")
    require(cell["create_only"] is True, "cell attempt directories must be create-only")
    require(cell["skip_statuses"] == ["PASSED"], "resume may skip only PASSED cells")
    radio = set(cell["radio_trace_fields"])
    for field in (
        "profile_id", "trace_id", "seed", "step_index", "target_snr_db",
        "mapped_rfsim_command_db", "achieved_snr_db", "command_send_monotonic_ns",
        "command_ack_monotonic_ns", "command_timing_status",
    ):
        require(field in radio, f"radio_trace schema missing {field}")
    feedback = set(cell["map_feedback_fields"])
    for field in ("frame_id", "capture_at", "action_id", "install_timestamp", "result_status", "rejection_reason"):
        require(field in feedback, f"map_feedback schema missing {field}")


def verify_trace_prefixes(config: Mapping[str, Any]) -> dict[str, str]:
    design_path = repo_path(str(config["network"]["design_config"]))
    design = load_json(design_path)
    spec_by_id = {row["profile_id"]: row for row in design["profiles"]}
    count = int(config["network"]["prefix_samples"])
    require(count == 4200, "accepted trace prefix length must be 4200")

    module_path = ROOT / "rl_agent/generate_network_profile_meeting_figures.py"
    spec = importlib.util.spec_from_file_location("ue_network_profile_design_v2", module_path)
    require(spec is not None and spec.loader is not None, "cannot import deterministic SNR generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    generated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_hashes: dict[str, str] = {}
    for frozen in network_profiles(config):
        profile_id = str(frozen["profile_id"])
        spec_profile = spec_by_id.get(profile_id)
        require(spec_profile is not None, f"profile absent from design config: {profile_id}")
        require(int(spec_profile["seed"]) == int(frozen["seed"]), f"seed drift for {profile_id}")
        require(str(spec_profile["trace_id"]) == str(frozen["trace_id"]), f"trace ID drift for {profile_id}")
        sequence = module.DeterministicTargetSnrSequence(spec_profile, design)
        states = np.empty(count, dtype="<i4")
        targets = np.empty(count, dtype="<f8")
        for index in range(count):
            states[index], targets[index] = sequence.next_sample()
        digest = hashlib.sha256(states.tobytes() + targets.tobytes()).hexdigest()
        require(digest == str(frozen["trace_sha256"]), f"generated trace hash mismatch for {profile_id}")
        generated[profile_id] = (states, targets)
        observed_hashes[profile_id] = digest

    csv_rows: dict[str, list[dict[str, str]]] = {profile_id: [] for profile_id in generated}
    with repo_path(str(config["network"]["traces_csv"])).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["profile_id"] in csv_rows:
                csv_rows[row["profile_id"]].append(row)
    for profile_id, rows in csv_rows.items():
        rows.sort(key=lambda row: int(row["step_index"]))
        require(len(rows) == count, f"{profile_id}: traces.csv prefix has {len(rows)} rows")
        states, targets = generated[profile_id]
        require([int(row["step_index"]) for row in rows] == list(range(count)), f"{profile_id}: trace indices drift")
        require(np.array_equal(np.asarray([int(row["state_index"]) for row in rows]), states), f"{profile_id}: state prefix drift")
        rounded = np.asarray([float(row["target_snr_db"]) for row in rows])
        require(np.allclose(rounded, targets, rtol=0.0, atol=5.1e-7), f"{profile_id}: target prefix drift")
    return observed_hashes


def unresolved_models(config: Mapping[str, Any]) -> list[str]:
    unresolved: list[str] = []
    models = config["actions"]["final_model_registry_entries"]
    for family in ("noae", "ae32", "ae64", "ae128"):
        for field in ("checkpoint_path", "checkpoint_sha256"):
            value = str(models[family][field])
            if not value or value.startswith(UNRESOLVED_PREFIX):
                unresolved.append(f"{family}.{field}")
    return unresolved


def apply_model_overrides(config: dict[str, Any], overrides: Sequence[str]) -> None:
    for raw in overrides:
        family, separator, value = raw.partition("=")
        path, at, digest = value.rpartition("@")
        require(separator == "=" and at == "@", f"invalid --model {raw!r}; expected FAMILY=PATH@SHA256")
        require(family in {"noae", "ae32", "ae64", "ae128"}, f"unknown model family: {family}")
        require(path and SHA256_RE.fullmatch(digest) is not None, f"invalid path/hash in --model {raw!r}")
        config["actions"]["final_model_registry_entries"][family] = {
            "checkpoint_path": path,
            "checkpoint_sha256": digest,
        }


def verify_resolved_models(config: Mapping[str, Any], registry: Sequence[dict[str, str]]) -> None:
    missing = unresolved_models(config)
    require(not missing, "real launch refused: unresolved final models: " + ", ".join(missing))
    models = config["actions"]["final_model_registry_entries"]
    for family in ("noae", "ae32", "ae64", "ae128"):
        model_path = repo_path(str(models[family]["checkpoint_path"]))
        expected_hash = str(models[family]["checkpoint_sha256"])
        require(model_path.is_file(), f"final {family} model missing: {model_path}")
        require(SHA256_RE.fullmatch(expected_hash) is not None, f"invalid final {family} SHA-256")
        require(sha256_file(model_path) == expected_hash, f"final {family} model SHA-256 mismatch")
        rows = [row for row in registry if row["model_family"] == family]
        require(rows and {row["checkpoint_sha256"] for row in rows} == {expected_hash}, f"{family} is not bound to the final hash in the action registry")
        registry_paths = {repo_path(row["checkpoint_path"]) for row in rows}
        require(registry_paths == {model_path}, f"{family} final path is not bound in the action registry")


def adapter_value(config: Mapping[str, Any], override: str | None = None) -> str:
    return str(override or config["runtime"]["required_route_b_split_cell_adapter"])


def validate_static(config_path: Path) -> tuple[dict[str, Any], list[Cell], dict[str, str]]:
    config = load_yaml(config_path)
    require(config.get("schema") == "scenesense.ue_288_campaign.v1", "campaign schema drift")
    verify_file_hashes(config)
    verify_route_contract(config)
    verify_output_contract(config)
    cells = enumerate_cells(config)
    hashes = verify_trace_prefixes(config)
    network = config["network"]
    require(int(network["sample_period_ms"]) == 100, "target-SNR period must be 100 ms")
    require(network["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "catch-up policy drift")
    require(float(network["clean_restore_noise_power_db"]) == -50.0, "clean RFsim restore drift")
    require(network["continuation"] == "CONTINUE_SAME_RNG_AND_MARKOV_STATE_INDEFINITELY", "trace continuation drift")
    return config, cells, hashes


def resume_ledger_dry_run(cells: Sequence[Cell]) -> dict[str, Any]:
    require(len(cells) >= 3, "ledger dry run needs at least three cells")
    ledger = {
        cells[0].cell_id: [{"attempt": 1, "status": "PASSED"}],
        cells[1].cell_id: [{"attempt": 1, "status": "FAILED"}],
        cells[2].cell_id: [{"attempt": 1, "status": "INTERRUPTED"}],
    }
    skipped = [cell.cell_id for cell in cells if any(row["status"] == "PASSED" for row in ledger.get(cell.cell_id, []))]
    rerun = [cell.cell_id for cell in cells if cell.cell_id not in skipped]
    next_attempt = {
        cell.cell_id: len(ledger.get(cell.cell_id, [])) + 1
        for cell in cells[:3]
        if cell.cell_id not in skipped
    }
    require(skipped == [cells[0].cell_id], "ledger dry run skipped a non-PASSED cell")
    require(next_attempt == {cells[1].cell_id: 2, cells[2].cell_id: 2}, "failed/interrupted attempts would be overwritten")
    return {
        "status": "PASS",
        "external_processes_started": 0,
        "skipped_only_passed": skipped,
        "failed_and_interrupted_next_attempt": next_attempt,
        "scheduled_cells": len(rerun),
    }


def cell_to_dict(cell: Cell) -> dict[str, Any]:
    return dict(cell.__dict__)


def load_ledger(path: Path, campaign_id: str, config_sha256: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": LEDGER_SCHEMA,
            "campaign_id": campaign_id,
            "config_sha256": config_sha256,
            "created_at_unix_s": time.time(),
            "updated_at_unix_s": time.time(),
            "cells": {},
        }
    ledger = load_json(path)
    require(ledger.get("schema") == LEDGER_SCHEMA, "resume ledger schema drift")
    require(ledger.get("campaign_id") == campaign_id, "resume ledger campaign mismatch")
    require(ledger.get("config_sha256") == config_sha256, "resume ledger config hash mismatch")
    require(isinstance(ledger.get("cells"), dict), "resume ledger cells must be a mapping")
    return ledger


def terminal_files(attempt_dir: Path) -> list[Path]:
    return [attempt_dir / name for name in TERMINAL_NAMES if (attempt_dir / name).is_file()]


def passed_attempt_exists(campaign_root: Path, ledger_rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in ledger_rows:
        if row.get("status") != "PASSED":
            continue
        attempt_dir = campaign_root / str(row["attempt_dir"])
        terminals = terminal_files(attempt_dir)
        if len(terminals) == 1 and terminals[0].name == "PASSED.json":
            return True
    return False


def next_attempt_dir(campaign_root: Path, cell: Cell, rows: Sequence[Mapping[str, Any]]) -> tuple[int, Path]:
    attempt = max([int(row.get("attempt", 0)) for row in rows] or [0]) + 1
    attempt_dir = campaign_root / "cells" / cell.cell_id / "attempts" / f"attempt_{attempt:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt, attempt_dir


def import_lifecycle_helper(config: Mapping[str, Any]) -> Any:
    path = repo_path(str(config["runtime"]["carla_lifecycle_helper"]))
    spec = importlib.util.spec_from_file_location("route_b_carla_lifecycle", path)
    require(spec is not None and spec.loader is not None, "cannot import CARLA lifecycle helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_terminal(attempt_dir: Path, status: str, detail: Mapping[str, Any]) -> Path:
    require(status in {"PASSED", "FAILED", "INTERRUPTED"}, f"invalid cell terminal: {status}")
    require(not terminal_files(attempt_dir), f"attempt already has a terminal: {attempt_dir}")
    path = attempt_dir / f"{status}.json"
    write_create_only(path, json.dumps({"status": status, **dict(detail)}, indent=2, sort_keys=True) + "\n")
    require(len(terminal_files(attempt_dir)) == 1, "cell attempt does not have exactly one terminal")
    return path


def run_one_cell(
    *,
    config: Mapping[str, Any],
    cell: Cell,
    adapter: Path,
    campaign_root: Path,
    ledger_rows: list[dict[str, Any]],
    port: int,
) -> dict[str, Any]:
    attempt, attempt_dir = next_attempt_dir(campaign_root, cell, ledger_rows)
    resolved = {
        "schema": "scenesense.ue_288_cell_resolved.v1",
        "campaign": config,
        "cell": cell_to_dict(cell),
        "attempt": attempt,
        "attempt_dir": str(attempt_dir),
    }
    resolved_path = attempt_dir / "resolved_config.yaml"
    write_create_only(resolved_path, yaml.safe_dump(resolved, sort_keys=False))
    lifecycle = import_lifecycle_helper(config)
    carla_log = attempt_dir / "carla_server.log"
    server = None
    pgid = None
    status = "FAILED"
    child_rc: int | None = None
    cleanup: dict[str, Any] = {"shutdown_verified": False}
    started = time.time()
    try:
        server, pgid = lifecycle.start_carla(port, carla_log)
        version = lifecycle.wait_for_rpc(port, 180.0)
        require(version is not None, "fresh Epic CARLA did not become RPC-ready")
        argv = [
            sys.executable,
            str(adapter),
            "--resolved-config", str(resolved_path),
            "--attempt-dir", str(attempt_dir),
            "--carla-host", "127.0.0.1",
            "--carla-port", str(port),
        ]
        with (attempt_dir / "cell_adapter.log").open("xb") as stream:
            child = subprocess.run(
                argv,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=lifecycle.child_env(),
            )
        child_rc = int(child.returncode)
        missing = [name for name in config["cell"]["expected_outputs"] if not (attempt_dir / name).is_file()]
        status = "PASSED" if child_rc == 0 and not missing else "FAILED"
        detail = {"adapter_returncode": child_rc, "missing_outputs": missing}
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        detail = {"reason": "operator interrupt", "adapter_returncode": child_rc}
    except Exception as exc:
        detail = {"reason": f"{type(exc).__name__}: {exc}", "adapter_returncode": child_rc}
    finally:
        if server is not None and pgid is not None:
            cleanup = lifecycle.stop_carla(server, pgid, port)
        if not cleanup.get("shutdown_verified"):
            status = "FAILED" if status != "INTERRUPTED" else status
            detail["cleanup_error"] = "fresh CARLA process group or RPC port survived cleanup"
        terminal = write_terminal(
            attempt_dir,
            status,
            {
                **detail,
                "cell_id": cell.cell_id,
                "attempt": attempt,
                "started_at_unix_s": started,
                "finished_at_unix_s": time.time(),
                "carla_cleanup": cleanup,
            },
        )
    return {
        "attempt": attempt,
        "attempt_dir": str(attempt_dir.relative_to(campaign_root)),
        "status": status,
        "terminal": str(terminal.relative_to(campaign_root)),
        "terminal_sha256": sha256_file(terminal),
    }


def verify_pilot_gate(path: Path) -> None:
    require(path.is_file(), f"full sweep requires pilot ledger: {path}")
    ledger = load_json(path)
    require(ledger.get("schema") == LEDGER_SCHEMA, "pilot ledger schema drift")
    cells = ledger.get("cells", {})
    passed = sum(
        1
        for rows in cells.values()
        if isinstance(rows, list) and any(row.get("status") == "PASSED" for row in rows)
    )
    require(passed == 16 and len(cells) == 16, f"full sweep requires all 16 pilot cells PASSED; found {passed}/16")


def run_campaign(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config, cells, _hashes = validate_static(config_path)
    apply_model_overrides(config, args.model)
    registry = read_registry(config)
    verify_resolved_models(config, registry)
    adapter_raw = adapter_value(config, args.route_b_split_cell_adapter)
    require(not adapter_raw.startswith(UNRESOLVED_PREFIX), "real launch refused: qualified Route B split cell adapter is unresolved")
    adapter = repo_path(adapter_raw)
    require(adapter.is_file(), f"qualified Route B split cell adapter missing: {adapter}")

    if config["campaign_kind"] == "full_288":
        require(args.authorize_full_sweep, "full 288 sweep requires --authorize-full-sweep")
        require(args.pilot_ledger is not None, "full 288 sweep requires --pilot-ledger")
        verify_pilot_gate(args.pilot_ledger.resolve())
    else:
        require(not args.authorize_full_sweep, "--authorize-full-sweep is invalid for the 16-cell pilot")

    campaign_root = args.output_root.resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    config_digest = sha256_file(config_path)
    ledger_path = campaign_root / str(config["cell"]["resume_ledger"])
    ledger = load_ledger(ledger_path, str(config["campaign_id"]), config_digest)
    for cell in cells:
        rows = ledger["cells"].setdefault(cell.cell_id, [])
        require(isinstance(rows, list), f"ledger rows are not a list for {cell.cell_id}")
        if passed_attempt_exists(campaign_root, rows):
            continue
        result = run_one_cell(
            config=config,
            cell=cell,
            adapter=adapter,
            campaign_root=campaign_root,
            ledger_rows=rows,
            port=int(args.carla_port),
        )
        rows.append(result)
        ledger["updated_at_unix_s"] = time.time()
        atomic_json(ledger_path, ledger)
        if result["status"] == "INTERRUPTED":
            return 130
    return 0


def validate_command(args: argparse.Namespace) -> int:
    campaign, campaign_cells, campaign_hashes = validate_static(args.campaign.resolve())
    pilot, pilot_cells, pilot_hashes = validate_static(args.pilot.resolve())
    require(len(campaign_cells) == 288, "full campaign did not enumerate 288 cells")
    require(len(pilot_cells) == 16, "integration pilot did not enumerate 16 cells")
    require(campaign_hashes == pilot_hashes, "pilot/full trace hashes differ")
    require(campaign["route_b"] == pilot["route_b"], "pilot/full Route B contract differs")
    report = {
        "status": "OFFLINE_VALIDATION_PASS_WITH_LAUNCH_BLOCKERS",
        "yaml_parse": "PASS",
        "campaign_cells": len(campaign_cells),
        "campaign_unique_cells": len({cell.cell_id for cell in campaign_cells}),
        "pilot_cells": len(pilot_cells),
        "pilot_unique_cells": len({cell.cell_id for cell in pilot_cells}),
        "registered_actions": len({cell.action_id for cell in campaign_cells}),
        "network_profiles": len({cell.network_profile_id for cell in campaign_cells}),
        "density": campaign["route_b"]["density"],
        "trace_prefix_hashes": campaign_hashes,
        "resume_ledger_dry_run": resume_ledger_dry_run(campaign_cells),
        "real_launch_blockers": {
            "campaign_models": unresolved_models(campaign),
            "pilot_models": unresolved_models(pilot),
            "qualified_route_b_split_cell_adapter": adapter_value(campaign),
        },
        "external_processes_started": 0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="offline-only campaign/pilot validation")
    validate.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    validate.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    validate.set_defaults(func=validate_command)

    run = subparsers.add_parser("run", help="run or resume a guarded campaign")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="FAMILY=PATH@SHA256",
        help="supply each final model without mutating the unresolved source YAML",
    )
    run.add_argument("--route-b-split-cell-adapter", default=None)
    run.add_argument("--carla-port", type=int, default=2000)
    run.add_argument("--authorize-full-sweep", action="store_true")
    run.add_argument("--pilot-ledger", type=Path)
    run.set_defaults(func=run_campaign)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CampaignError as exc:
        print(f"campaign contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
