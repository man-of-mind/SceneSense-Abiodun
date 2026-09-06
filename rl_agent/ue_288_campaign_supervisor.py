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
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "rl_agent/configs/ue_288_campaign_v1.yaml"
DEFAULT_PILOT = ROOT / "rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml"
DEFAULT_LIVE_PILOT = ROOT / "rl_agent/configs/splitfusion_16_cell_live_carla_oai_pilot_v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED_PREFIX = "__REQUIRED_"
TERMINAL_NAMES = ("PASSED.json", "FAILED.json", "INTERRUPTED.json")
LEDGER_SCHEMA = "scenesense.ue_288_campaign_ledger.v1"
RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
RADIO_MAPPING_QUALIFIED = "QUALIFIED_ON_OAI_N78_100MHZ_273PRB_4D5U_V1"
RADIO_RUNTIME_BOUND = "BOUND_SPLITFUSION_100MHZ_4D5U"
RADIO_LAUNCHER_QUALIFIED = "QUALIFIED_SPLITFUSION_100MHZ_4D5U"
ARCHITECTURE_BOUND = "SPLITFUSION_72_ACTION_CATALOG_BOUND"
ACTION_CATALOG_COMMIT = "4e237d719df91e83cfdc568e5f20b542d9482ad7"
ACTION_CATALOG_SCHEMA = "splitfusion_72_action_catalog_v1"
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
Q_E4 = (0, 3000, 5000, 7000, 9000, 9800)
LIVE_PILOT_TOKEN = "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT"
PHASE15_QUALIFICATION_SCHEMA = "scenesense.splitfusion_phase15_live_deployment_qualification.v1"
PHASE15_QUALIFICATION_TERMINAL = "SPLITFUSION_PHASE15_LIVE_DEPLOYMENT_QUALIFIED"
LIVE_PILOT_EXPECTED_DIRTY_PATHS = {
    "OAI/openairinterface5g",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/lraspp_to_splitfusion_fcos_report_v1/FULL_TECHNICAL_REPORT_AVO_V2.md",
    "oaitelnet.history",
}


class CampaignError(RuntimeError):
    """A campaign contract or launch prerequisite is not satisfied."""


@dataclass(frozen=True)
class Cell:
    cell_id: str
    action_index: int
    action_id: int
    profile_id: str
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


def _compact_log_diagnostics(root: Path, *, include_tails: bool) -> list[dict[str, Any]]:
    """Summarize ephemeral service logs before their unconditional removal."""

    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return records
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        raw = path.read_bytes()
        record: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if include_tails:
            record["tail"] = raw[-8192:].decode("utf-8", errors="replace").splitlines()[-40:]
        records.append(record)
    return records


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


def verify_live_pilot_worktree(
    *, owned_resume_root: Path | None = None,
) -> dict[str, Any]:
    """Accept declared user paths plus untracked files in an exact resume root."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=str(ROOT),
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    records = completed.stdout.splitlines()
    paths: list[str] = []
    resume_paths: list[str] = []
    resume_prefix = ""
    if owned_resume_root is not None:
        experiments = (ROOT / "experiments").resolve(strict=True)
        resume_root = owned_resume_root.resolve(strict=True)
        try:
            resume_relative = resume_root.relative_to(ROOT.resolve(strict=True))
            resume_root.relative_to(experiments)
        except ValueError as exc:
            raise CampaignError("live-pilot resume root escapes experiments/repository") from exc
        resume_prefix = str(resume_relative).rstrip("/") + "/"
    for line in records:
        require(len(line) >= 4, f"unparseable porcelain record: {line!r}")
        require(" -> " not in line[3:], "renamed user-owned paths are not authorized")
        path = line[3:]
        if resume_prefix and path.startswith(resume_prefix):
            require(line[:2] == "??", "resume evidence may only be untracked")
            resume_paths.append(path)
        else:
            paths.append(path)
    require(
        set(paths) == LIVE_PILOT_EXPECTED_DIRTY_PATHS and len(paths) == len(LIVE_PILOT_EXPECTED_DIRTY_PATHS),
        f"unexpected worktree paths: {sorted(paths)}",
    )
    return {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip(),
        "dirty_paths": sorted(paths),
        "owned_resume_untracked_paths": len(resume_paths),
    }


def write_create_only(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def read_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = config["actions"]
    path = repo_path(str(actions["catalog_json"]))
    require(path.is_file(), f"locked action catalog missing: {path}")
    require(
        sha256_file(path) == str(actions["catalog_sha256"]),
        "locked action catalog SHA-256 drift",
    )
    require(actions.get("catalog_commit") == ACTION_CATALOG_COMMIT, "locked action catalog commit drift")
    require(actions.get("catalog_schema") == ACTION_CATALOG_SCHEMA, "configured action catalog schema drift")
    catalog = load_json(path)
    require(catalog.get("schema") == ACTION_CATALOG_SCHEMA, "action catalog schema drift")
    rows = catalog.get("profiles")
    require(isinstance(rows, list), "action catalog profiles must be a list")
    require(len(rows) == 72, f"action catalog must contain exactly 72 actions, found {len(rows)}")
    require(len({row["profile_id"] for row in rows}) == 72, "action catalog profile IDs are not unique")
    require(
        [int(row["action_id"]) for row in rows] == list(range(72)),
        "action catalog IDs must be contiguous 0..71",
    )
    expected_product = {
        (family, quantizer, q_e4)
        for family in FAMILIES
        for quantizer in QUANTIZERS
        for q_e4 in Q_E4
    }
    actual_product = {
        (str(row["family"]), str(row["quantizer"]), int(row["q_e4"]))
        for row in rows
    }
    require(actual_product == expected_product, "action catalog Cartesian coverage drift")
    invalid = [
        row["profile_id"]
        for row in rows
        if row.get("execution_mode") != "SPLIT"
        or row.get("capabilities", {}).get("transport_valid") is not True
        or row.get("capabilities", {}).get("agent_action_enabled") is not True
    ]
    require(not invalid, f"disabled or transport-invalid catalog actions: {invalid[:4]}")
    forbidden_copies = {"profiles", "metrics", "perception", "payload"} & set(actions)
    require(not forbidden_copies, f"campaign-local action semantics are forbidden: {sorted(forbidden_copies)}")
    return rows


def selected_actions(config: Mapping[str, Any], catalog: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = config["actions"]
    selection = str(actions["selection"])
    if selection == "all_catalog_actions":
        chosen = list(catalog)
    elif selection == "explicit_profile_ids":
        requested = list(actions.get("profile_ids", []))
        require(len(requested) == len(set(requested)), "pilot action selectors are duplicated")
        by_profile = {row["profile_id"]: row for row in catalog}
        missing = [value for value in requested if value not in by_profile]
        require(not missing, f"pilot action selectors are absent from catalog: {missing}")
        chosen = [by_profile[value] for value in requested]
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


def cell_mapping_sha256(cells: Sequence[Cell]) -> str:
    mapping = [
        {
            "action_id": cell.action_id,
            "cell_id": cell.cell_id,
            "network_profile_id": cell.network_profile_id,
            "profile_id": cell.profile_id,
        }
        for cell in cells
    ]
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def enumerate_cells(config: Mapping[str, Any]) -> list[Cell]:
    catalog = read_catalog(config)
    actions = selected_actions(config, catalog)
    profiles = network_profiles(config)
    cells: list[Cell] = []
    for profile in profiles:
        for row in actions:
            cells.append(
                Cell(
                    cell_id=f"a{int(row['action_id']):02d}__{profile['profile_id'].lower()}",
                    action_index=int(row["action_id"]),
                    action_id=int(row["action_id"]),
                    profile_id=str(row["profile_id"]),
                    model_family=str(row["family"]).lower(),
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
    require(
        cell_mapping_sha256(cells) == str(config["actions"]["cell_mapping_sha256"]),
        "action/network mapping digest drift",
    )
    return cells


def verify_file_hashes(config: Mapping[str, Any]) -> None:
    network = config["network"]
    design_path = repo_path(str(network["design_config"]))
    require(design_path.is_file(), f"missing network design input: {design_path}")
    require(
        sha256_file(design_path) == str(network["design_config_sha256"]),
        "network design SHA-256 drift",
    )
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
    runtime = config["runtime"]
    adapter = repo_path(str(runtime["required_route_b_split_cell_adapter"]))
    require(adapter.is_file(), f"missing SplitFusion cell adapter: {adapter}")
    require(
        sha256_file(adapter) == str(runtime["required_route_b_split_cell_adapter_sha256"]),
        "SplitFusion cell adapter SHA-256 drift",
    )
    split_runtime = repo_path(str(runtime["split_inference_runtime"]))
    require(split_runtime.is_file(), f"missing split inference runtime: {split_runtime}")
    require(
        sha256_file(split_runtime) == str(runtime["split_inference_runtime_sha256"]),
        "split inference runtime SHA-256 drift",
    )
    if config.get("campaign_kind") == "live_pilot_16":
        bridge = repo_path(str(runtime["live_dispatch_bridge"]))
        require(
            bridge.is_file() and sha256_file(bridge) == str(runtime["live_dispatch_bridge_sha256"]),
            "qualified SFD1-v2 live dispatcher bridge SHA-256 drift",
        )
        target_runtime = repo_path(str(runtime["target_snr_runtime"]))
        require(
            target_runtime.is_file() and sha256_file(target_runtime) == str(runtime["target_snr_runtime_sha256"]),
            "qualified continuous target-SNR runtime SHA-256 drift",
        )
        deployment = config.get("deployment")
        require(isinstance(deployment, dict), "live-pilot deployment bindings are missing")
        for name, item in deployment.items():
            require(isinstance(item, dict), f"deployment binding is invalid: {name}")
            path = repo_path(str(item.get("path", "")))
            require(path.is_file(), f"deployment dependency is missing: {name}")
            require(
                sha256_file(path) == str(item.get("sha256", "")),
                f"deployment dependency SHA-256 drift: {name}",
            )


def verify_live_pilot_provenance(config: Mapping[str, Any]) -> None:
    """Bind the later-qualified evidence without rewriting historical contracts."""

    if config.get("campaign_kind") != "live_pilot_16":
        return
    provenance = config.get("provenance")
    require(isinstance(provenance, dict), "live pilot provenance is missing")
    for name in ("phase12b_campaign_binding", "phase13a_runtime_binding", "phase14a_mapping_json"):
        item = provenance.get(name)
        require(isinstance(item, dict), f"live pilot provenance missing {name}")
        path = repo_path(str(item.get("path", "")))
        require(path.is_file() and sha256_file(path) == str(item.get("sha256", "")), f"live pilot provenance drift: {name}")
    corrected = provenance.get("phase14b_corrected")
    require(isinstance(corrected, dict), "corrected Phase-14B provenance is missing")
    for name in ("qualification", "report", "artifact_manifest", "terminal"):
        item = corrected.get(name)
        require(isinstance(item, dict), f"corrected Phase-14B {name} binding missing")
        path = repo_path(str(item.get("path", "")))
        require(path.is_file() and sha256_file(path) == str(item.get("sha256", "")), f"corrected Phase-14B {name} hash drift")
    qualification = load_json(repo_path(str(corrected["qualification"]["path"])))
    require(
        qualification.get("status") == "FOUR_PROFILE_REPLAY_COMPLETE"
        and qualification.get("all_profiles_passed") is True
        and qualification.get("mapping_qualified_for_16_cell_review") is True
        and qualification.get("campaign_288_authorized") is False,
        "corrected Phase-14B is not a qualified 16-cell-only mapping evidence record",
    )
    terminal = load_json(repo_path(str(corrected["terminal"]["path"])))
    require(
        terminal.get("status") == "SPLITFUSION_PHASE14B_CORRECTED_FOUR_PROFILE_REPLAY_COMPLETE"
        and terminal.get("qualification_sha256") == str(corrected["qualification"]["sha256"])
        and terminal.get("artifact_manifest_sha256") == str(corrected["artifact_manifest"]["sha256"]),
        "corrected Phase-14B terminal does not bind the qualification",
    )
    require(corrected.get("accepted_mapping_evidence") == "COMMAND_VALIDITY_COVERAGE", "wrong Phase-14B mapping evidence")
    require(corrected.get("probe_forbidden_during_pilot") is True, "Phase-14B probe is not forbidden for live pilot")


def verify_radio_baseline(config: Mapping[str, Any]) -> dict[str, Any]:
    network = config["network"]
    binding = network.get("radio_baseline")
    require(isinstance(binding, dict), "selected OAI radio baseline is missing")
    profile_path = repo_path(str(binding["profile_config"]))
    require(profile_path.is_file(), f"selected OAI radio baseline is missing: {profile_path}")
    profile_hash = str(binding["profile_config_sha256"])
    require(SHA256_RE.fullmatch(profile_hash) is not None, "selected OAI radio baseline hash is invalid")
    require(sha256_file(profile_path) == profile_hash, "selected OAI radio baseline SHA-256 drift")
    profile = load_json(profile_path)
    require(profile.get("schema") == "scenesense.oai_radio_baseline.v1", "OAI radio baseline schema drift")
    require(profile.get("profile_id") == RADIO_PROFILE_ID, "OAI radio baseline profile ID drift")
    require(binding.get("profile_id") == RADIO_PROFILE_ID, "campaign OAI radio profile ID drift")
    require(binding.get("selection_status") == "LOCKED", "OAI radio baseline is not locked")
    radio = profile.get("radio", {})
    require(int(radio.get("band", -1)) == 78, "OAI radio band must be n78")
    require(int(radio.get("bandwidth_mhz", -1)) == 100, "OAI radio bandwidth must be 100 MHz")
    require(int(radio.get("prb", -1)) == 273, "OAI radio allocation must be 273 PRBs")
    require(int(radio.get("numerology", -1)) == 1, "OAI radio numerology must be 1")
    tdd = radio.get("tdd", {})
    require(int(tdd.get("downlink_slots", -1)) == 4, "OAI TDD downlink-slot count must be 4")
    require(int(tdd.get("uplink_slots", -1)) == 5, "OAI TDD uplink-slot count must be 5")
    require(int(tdd.get("downlink_symbols", -1)) == 6, "OAI TDD downlink-symbol count must be 6")
    require(int(tdd.get("uplink_symbols", -1)) == 4, "OAI TDD uplink-symbol count must be 4")
    require(int(radio.get("pdu_session_5qi", -1)) == 6, "campaign PDU session must retain 5QI 6")
    mapping_status = str(binding.get("target_snr_mapping_status", ""))
    require(
        mapping_status in {"PENDING_100MHZ_4D5U_REQUALIFICATION", RADIO_MAPPING_QUALIFIED},
        "unsupported target-SNR mapping qualification state",
    )
    calibration_profile = str(network.get("mapping_calibration_radio_profile_id", ""))
    if mapping_status == RADIO_MAPPING_QUALIFIED:
        require(calibration_profile == RADIO_PROFILE_ID, "qualified target-SNR mapping is not bound to the selected radio")
    else:
        require(
            calibration_profile == "OAI_N78_40MHZ_106PRB_7D2U_LEGACY",
            "pending mapping must identify the retained 106-PRB calibration as legacy",
        )
    return profile


def real_launch_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(config["actions"].get("architecture_binding_status", "")) != ARCHITECTURE_BOUND:
        blockers.append("splitfusion_final_validation_and_action_registry")
    baseline = config["network"]["radio_baseline"]
    if str(baseline.get("target_snr_mapping_status", "")) != RADIO_MAPPING_QUALIFIED:
        blockers.append("target_snr_mapping_requalification_on_100mhz_4d5u")
    if str(config["network"].get("mapping_calibration_radio_profile_id", "")) != RADIO_PROFILE_ID:
        blockers.append("target_snr_mapping_radio_binding")
    runtime = config["runtime"]
    if str(runtime.get("oai_radio_runtime_binding_status", "")) != RADIO_RUNTIME_BOUND:
        blockers.append("splitfusion_100mhz_4d5u_runtime_binding")
    if str(runtime.get("oai_registered_profile_launcher_status", "")) != RADIO_LAUNCHER_QUALIFIED:
        blockers.append("splitfusion_100mhz_4d5u_launcher_qualification")
    launcher_hash = str(runtime.get("oai_registered_profile_launcher_sha256", ""))
    if SHA256_RE.fullmatch(launcher_hash) is None:
        blockers.append("splitfusion_100mhz_4d5u_launcher_hash")
    return blockers


def verify_real_launch_readiness(config: Mapping[str, Any]) -> None:
    blockers = real_launch_blockers(config)
    require(not blockers, "real launch refused: unresolved campaign bindings: " + ", ".join(blockers))
    runtime = config["runtime"]
    launcher = repo_path(str(runtime["oai_registered_profile_launcher"]))
    require(launcher.is_file(), f"qualified OAI launcher missing: {launcher}")
    require(
        sha256_file(launcher) == str(runtime["oai_registered_profile_launcher_sha256"]),
        "qualified OAI launcher SHA-256 drift",
    )


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
        "radio_profile_id", "radio_profile_sha256", "bandwidth_mhz", "prb",
        "downlink_slots", "uplink_slots",
        "profile_id", "trace_id", "seed", "step_index", "target_snr_db",
        "mapped_rfsim_command_db", "achieved_snr_db", "command_send_monotonic_ns",
        "command_ack_monotonic_ns", "command_timing_status",
    ):
        require(field in radio, f"radio_trace schema missing {field}")
    feedback = set(cell["map_feedback_fields"])
    for field in (
        "frame_id", "capture_at", "action_id", "install_timestamp",
        "result_status", "status", "terminal", "rejection_reason",
    ):
        require(field in feedback, f"map_feedback schema missing {field}")
    perception = set(cell["perception_metric_fields"])
    for field in (
        "exact_frame_prediction_available", "object_gt_evidence_available",
        "segmentation_evidence_available", "footprint_iou", "segmentation_iou",
    ):
        require(field in perception, f"perception_metrics schema missing {field}")


def verify_measurement_contract(config: Mapping[str, Any]) -> None:
    contract = config.get("measurement_contract")
    require(isinstance(contract, dict), "campaign measurement_contract is missing")
    require(float(contract["match_distance_m"]) == 3.0, "primary match distance must be 3.0 m")
    require(float(contract["max_gt_distance_m"]) == 40.0, "GT range gate must be 40.0 m")
    require(float(contract["min_gt_area_px"]) == 12.0, "GT area gate must be 12.0 px")
    require(float(contract["expected_prepared_hz"]) == 10.0, "prepared-input schedule must be 10 Hz")
    coverage = float(contract["minimum_sensor_preparation_coverage"])
    require(0.0 < coverage <= 1.0, "sensor/preparation coverage gate must be in (0, 1]")
    require(int(contract["installed_frame_history_size"]) > 0, "installed-frame history must be bounded and nonempty")
    require(float(contract["segmentation_evidence_retention_s"]) > 0.0, "segmentation evidence retention must be positive")


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


def apply_model_overrides(config: dict[str, Any], overrides: Sequence[str]) -> None:
    require(not overrides, "--model overrides are forbidden by the immutable SplitFusion action catalog")


def verify_resolved_models(config: Mapping[str, Any], catalog: Sequence[dict[str, Any]]) -> None:
    bindings: dict[str, str] = {}
    for row in catalog:
        candidates = [row.get("perception_checkpoint"), row.get("ae_checkpoint")]
        ranker = row.get("ranker", {})
        if isinstance(ranker, dict):
            candidates.append(ranker.get("checkpoint"))
        for binding in candidates:
            if binding is None:
                continue
            require(isinstance(binding, dict), "catalog checkpoint binding is not a mapping")
            path = str(binding.get("path", ""))
            digest = str(binding.get("sha256", ""))
            require(path and SHA256_RE.fullmatch(digest) is not None, "catalog checkpoint path/hash is invalid")
            require(path not in bindings or bindings[path] == digest, f"conflicting catalog checkpoint hashes: {path}")
            bindings[path] = digest
    require(bindings, "action catalog contains no checkpoint bindings")
    for path, digest in bindings.items():
        checkpoint = repo_path(path)
        require(checkpoint.is_file(), f"catalog checkpoint missing: {checkpoint}")
        require(sha256_file(checkpoint) == digest, f"catalog checkpoint SHA-256 mismatch: {path}")


def adapter_value(config: Mapping[str, Any], override: str | None = None) -> str:
    return str(override or config["runtime"]["required_route_b_split_cell_adapter"])


def validate_static(config_path: Path) -> tuple[dict[str, Any], list[Cell], dict[str, str]]:
    config = load_yaml(config_path)
    require(
        config.get("schema") in {"scenesense.ue_288_campaign.v1", "scenesense.splitfusion_16_cell_live_carla_oai_pilot.v1"},
        "campaign schema drift",
    )
    require(config.get("stop_on_first_failure") is True, "campaign must stop on the first failed/interrupted cell")
    verify_file_hashes(config)
    verify_radio_baseline(config)
    verify_route_contract(config)
    verify_measurement_contract(config)
    verify_output_contract(config)
    verify_live_pilot_provenance(config)
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


def passed_attempt_exists(
    campaign_root: Path,
    ledger_rows: Sequence[Mapping[str, Any]],
    expected_outputs: Sequence[str],
) -> bool:
    for row in ledger_rows:
        if row.get("status") != "PASSED":
            continue
        attempt_dir = campaign_root / str(row["attempt_dir"])
        terminals = terminal_files(attempt_dir)
        if len(terminals) != 1 or terminals[0].name != "PASSED.json":
            continue
        if row.get("terminal_sha256") != sha256_file(terminals[0]):
            continue
        try:
            terminal = load_json(terminals[0])
            summary = load_json(attempt_dir / "RESULTS_SUMMARY.json")
            manifest = load_json(attempt_dir / "manifest.json")
        except (OSError, json.JSONDecodeError):
            continue
        outputs_present = all((attempt_dir / name).is_file() for name in expected_outputs)
        manifest_rows = manifest.get("files", []) if isinstance(manifest, dict) else []
        manifest_hashes_valid = bool(manifest_rows) and all(
            isinstance(item, dict)
            and (attempt_dir / str(item.get("path", ""))).is_file()
            and item.get("sha256") == sha256_file(attempt_dir / str(item["path"]))
            for item in manifest_rows
        )
        if (
            outputs_present
            and manifest.get("registered_outputs") == list(expected_outputs)
            and manifest_hashes_valid
            and terminal.get("status") == "PASSED"
            and summary.get("status") == "PASSED"
            and summary.get("terminal_status") == "PASSED"
        ):
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


def _live_radio_state(config: Mapping[str, Any], cell: Cell, attempt: int) -> tuple[Path, Path]:
    """Return a fresh Phase-14A-compatible scratch leaf for one cell only."""

    profile_order = [str(row["profile_id"]) for row in config["network"]["profiles"]]
    index = profile_order.index(cell.network_profile_id)
    namespace = (
        ROOT / "experiments/splitfusion_oai_100mhz_4d5u_v1"
        / "splitfusion_16_cell_live_pilot_radio_scratch"
        / cell.cell_id / f"attempt_{attempt:04d}"
    )
    state = namespace / f"{index:02d}_{cell.network_profile_id}"
    require(not state.exists(), f"live-pilot radio scratch already exists: {state}")
    return namespace, state


def _start_live_radio(
    config: Mapping[str, Any], cell: Cell, attempt: int, service_log_dir: Path
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Use the qualified launcher and record its sealed three-process topology."""

    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

    runtime = config["runtime"]
    phase14a_config_path = repo_path(str(runtime["phase14a_config"]))
    phase14a_binding_path = repo_path(str(runtime["phase14a_binding"]))
    require(sha256_file(phase14a_config_path) == str(runtime["phase14a_config_sha256"]), "Phase-14A config hash drift")
    require(sha256_file(phase14a_binding_path) == str(runtime["phase14a_binding_sha256"]), "Phase-14A binding hash drift")
    base = phase14a.load_json(phase14a_config_path)
    namespace, radio_state = _live_radio_state(config, cell, attempt)
    phase14b.require_cold_profile_runtime(base, radio_state)
    launcher = repo_path(str(runtime["oai_registered_profile_launcher"]))
    try:
        with (service_log_dir / "oai_launcher.log").open("xb") as stream:
            launched = subprocess.run(
                [str(launcher), "--execute", "SPLITFUSION_OAI_100MHZ_4D5U_ATTACH", "--output", str(radio_state)],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                check=False, timeout=240.0,
            )
        require(
            launched.returncode == 0,
            f"qualified OAI launcher failed rc={launched.returncode}",
        )
        attached_path = radio_state / "ATTACHED_RADIO_STATE.json"
        require(attached_path.is_file(), "qualified OAI launcher omitted attached-radio snapshot")
        attached = load_json(attached_path)
        require(attached.get("status") == "ATTACHED_STABLE_100MHZ_4D5U_ONE_UE", "radio attachment status drift")
        for name in ("gnb", "ue"):
            topology = attached.get(f"{name}_process_topology", {})
            require(
                topology.get("endpoint_roles_verified") is True
                and int(topology.get("same_executable_process_count", -1)) == 3
                and len(topology.get("worker_pids", ())) == 2,
                f"qualified {name} three-process topology drift",
            )
        clean = phase14b.restore_interrupted_radio(base)
        require(
            clean is not None
            and clean.get("verified") is True
            and float(clean.get("noise_power_db", float("nan"))) == -50.0,
            "attached RFsim channel is not verified at noise_power_dB=-50",
        )
        return namespace, radio_state, {**attached, "clean_noise_preflight": clean}
    except BaseException as exc:
        restored = None
        restore_error = ""
        try:
            restored = phase14b.restore_interrupted_radio(base)
        except BaseException as restore_exc:
            restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        cleanup = phase14b.teardown_profile_runtime(
            base, radio_state, namespace, None,
            restore_verified=bool(
                not restore_error
                and (restored is None or restored.get("verified"))
            ),
        )
        raise CampaignError(
            f"{type(exc).__name__}: {exc}; partial radio restore_error={restore_error!r} "
            f"cleanup={cleanup}"
        ) from exc


def _stop_live_radio(
    config: Mapping[str, Any], namespace: Path, radio_state: Path, attached: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

    base = phase14a.load_json(repo_path(str(config["runtime"]["phase14a_config"])))
    restored = phase14b.restore_interrupted_radio(base)
    report = phase14b.teardown_profile_runtime(
        base, radio_state, namespace, attached, restore_verified=bool(restored and restored.get("verified")),
    )
    require(bool(report.get("all_lifecycle_gates_passed")), "live-pilot radio teardown/cold proof failed")
    return report


def _stop_phase15_application(config: Mapping[str, Any]) -> dict[str, Any]:
    """Stop only application resources owned after a cold Phase-15 startup."""

    edge = subprocess.run(
        [str(ROOT / "scripts/receiver_container_down.sh")], cwd=str(ROOT),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a

    owned_sources = {
        str(repo_path(str(config["runtime"][name])))
        for name in (
            "required_route_b_split_cell_adapter", "map_install_runtime",
            "target_snr_runtime",
        )
    }
    stopped: list[int] = []
    for row in phase14a.process_table():
        pid = int(row["pid"])
        command = str(row.get("command", ""))
        if any(source in command for source in owned_sources):
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append(pid)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        surviving = [
            pid for pid in stopped
            if (Path("/proc") / str(pid)).exists()
        ]
        if not surviving:
            break
        time.sleep(0.05)
    else:
        for pid in surviving:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    remaining = [pid for pid in stopped if (Path("/proc") / str(pid)).exists()]
    require(not remaining, f"owned Phase-15 application processes survived: {remaining}")
    edge_deadline = time.monotonic() + 30.0
    while True:
        edge_remaining = subprocess.run(
            ("sudo", "-n", "docker", "inspect", "oai-perception-rx"),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False,
        )
        if edge_remaining.returncode != 0 or time.monotonic() >= edge_deadline:
            break
        time.sleep(0.05)
    require(edge_remaining.returncode != 0, "owned Phase-15 edge container survived cleanup")
    return {
        "edge_container_down_returncode": int(edge.returncode),
        "owned_host_processes_stopped": stopped,
        "owned_host_processes_remaining": [],
    }


def _require_phase15_application_cold(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require absence of the edge/CARLA/adapter state not covered by Phase 14."""

    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a

    sources = {
        str(repo_path(str(config["runtime"][name])))
        for name in (
            "required_route_b_split_cell_adapter", "map_install_runtime",
            "target_snr_runtime", "live_dispatch_bridge",
        )
    }
    active = []
    for row in phase14a.process_table():
        command = str(row.get("command", ""))
        executable = Path(str(row.get("executable", ""))).name
        if executable.startswith("CarlaUnreal") or any(source in command for source in sources):
            active.append({"pid": int(row["pid"]), "executable": executable})
    require(not active, f"stale Phase-15 application processes exist: {active}")
    edge = subprocess.run(
        ("sudo", "-n", "docker", "inspect", "oai-perception-rx"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )
    require(edge.returncode != 0, "stale Phase-15 edge container exists")
    stale_tmp = sorted(
        path.name
        for prefix in (
            "ue_288_cell_runtime_*", "ue_288_seg_eval_*",
            "splitfusion_pilot_*", "splitfusion_live_edge_*",
        )
        for path in Path("/tmp").glob(prefix)
    )
    require(not stale_tmp, f"stale Phase-15 temporary runtime paths exist: {stale_tmp}")
    stale_shm = sorted(
        path.name
        for path in Path("/dev/shm").iterdir()
        if any(token in path.name.casefold() for token in ("carla", "splitfusion", "oai"))
    )
    require(not stale_shm, f"stale Phase-15 shared-memory objects exist: {stale_shm}")

    runtime = config["runtime"]
    requested = {
        "tcp": {2000, 8010, 35001},
        "udp": {
            int(runtime["edge_receive_port"]), int(runtime["edge_source_port"]),
            int(runtime["camera_result_port"]), int(runtime["map_ingest_port"]),
            39401,
        },
    }
    occupied: dict[str, list[int]] = {}
    for protocol, arguments in (
        ("tcp", ("ss", "-H", "-ltnp")),
        ("udp", ("ss", "-H", "-lunp")),
    ):
        completed = subprocess.run(
            arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, check=False,
        )
        require(completed.returncode == 0, f"cannot audit Phase-15 {protocol} listeners")
        used = {
            int(fields[3].rsplit(":", 1)[-1])
            for line in completed.stdout.splitlines()
            if len((fields := line.split())) >= 5
            and fields[3].rsplit(":", 1)[-1].isdigit()
        }
        occupied[protocol] = sorted(requested[protocol] & used)
        require(
            not occupied[protocol],
            f"stale/conflicting Phase-15 {protocol} listeners exist: {occupied[protocol]}",
        )
    return {
        "application_processes": [], "edge_container_absent": True,
        "runtime_temporary_paths": [], "shared_memory_objects": [],
        "conflicting_tcp_ports": [], "conflicting_udp_ports": [],
    }


def _phase15_gpu_audit() -> dict[str, Any]:
    """Run the approved infrastructure-aware GPU policy in a short-lived process."""

    code = (
        "import json; from rl_agent.splitfusion_live_dispatch_v1."
        "phase13b_qualification import _gpu_preflight; "
        "print(json.dumps(_gpu_preflight(), sort_keys=True))"
    )
    completed = subprocess.run(
        ("/usr/bin/python3", "-c", code), cwd=str(ROOT),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False, timeout=90.0,
    )
    require(
        completed.returncode == 0 and completed.stdout.splitlines(),
        "Phase-15 GPU/workload audit failed: "
        + (completed.stderr.strip() or completed.stdout.strip()),
    )
    report = json.loads(completed.stdout.splitlines()[-1])
    require(
        report.get("device_count") == 1
        and report.get("device_name") == "NVIDIA GeForce RTX 5090",
        "Phase-15 GPU identity drift",
    )
    return report


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
        "measurement_contract": dict(config["measurement_contract"]),
        "cell": cell_to_dict(cell),
        "attempt": attempt,
        "attempt_dir": str(attempt_dir),
    }
    resolved_path = attempt_dir / "resolved_config.yaml"
    write_create_only(resolved_path, yaml.safe_dump(resolved, sort_keys=False))
    lifecycle = import_lifecycle_helper(config)
    # Launcher and CARLA logs are bounded operational diagnostics, never pilot
    # evidence.  Keep them outside the create-only campaign and remove them
    # after the cell terminal is durable.
    service_log_dir = Path(tempfile.mkdtemp(prefix=f"splitfusion_pilot_{cell.cell_id}_"))
    carla_log = service_log_dir / "carla_server.log"
    server = None
    pgid = None
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    attached_radio: Mapping[str, Any] | None = None
    status = "FAILED"
    child_rc: int | None = None
    cleanup: dict[str, Any] = {"shutdown_verified": False, "radio_shutdown_verified": False}
    started = time.time()
    try:
        if config.get("campaign_kind") == "live_pilot_16":
            radio_namespace, radio_state, attached_radio = _start_live_radio(
                config, cell, attempt, service_log_dir
            )
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
        if args_maximum_loop_sim_s := config.get("_maximum_loop_sim_s_override"):
            argv += ["--maximum-loop-sim-s", str(float(args_maximum_loop_sim_s))]
        with (service_log_dir / "cell_adapter.log").open("xb") as stream:
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
        summary_status = ""
        summary_terminal_status = ""
        summary_error = ""
        if not missing and (attempt_dir / "RESULTS_SUMMARY.json").is_file():
            try:
                summary = load_json(attempt_dir / "RESULTS_SUMMARY.json")
                summary_status = str(summary.get("status", ""))
                summary_terminal_status = str(summary.get("terminal_status", ""))
            except (OSError, json.JSONDecodeError) as exc:
                summary_error = f"{type(exc).__name__}: {exc}"
        summary_passed = summary_status == "PASSED" and summary_terminal_status == "PASSED"
        status = "PASSED" if child_rc == 0 and not missing and summary_passed else "FAILED"
        detail = {
            "adapter_returncode": child_rc,
            "missing_outputs": missing,
            "results_summary_status": summary_status,
            "results_summary_terminal_status": summary_terminal_status,
            "results_summary_error": summary_error,
        }
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        detail = {"reason": "operator interrupt", "adapter_returncode": child_rc}
    except Exception as exc:
        detail = {"reason": f"{type(exc).__name__}: {exc}", "adapter_returncode": child_rc}
    finally:
        try:
            cleanup["application"] = _stop_phase15_application(config)
        except Exception as exc:
            cleanup["application_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None:
            cleanup["carla"] = lifecycle.stop_carla(server, pgid, port)
        if radio_namespace is not None and radio_state is not None:
            try:
                cleanup["radio"] = _stop_live_radio(
                    config, radio_namespace, radio_state, attached_radio
                )
                cleanup["radio_shutdown_verified"] = True
            except Exception as exc:
                cleanup["radio_error"] = f"{type(exc).__name__}: {exc}"
        try:
            cleanup["application_cold"] = _require_phase15_application_cold(config)
        except Exception as exc:
            cleanup["application_cold_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None and not cleanup.get("carla", {}).get("shutdown_verified"):
            status = "FAILED" if status != "INTERRUPTED" else status
            detail["cleanup_error"] = "fresh CARLA process group or RPC port survived cleanup"
        if cleanup.get("application_error"):
            status = "FAILED" if status != "INTERRUPTED" else status
            detail["application_cleanup_error"] = cleanup["application_error"]
        if cleanup.get("application_cold_error"):
            status = "FAILED" if status != "INTERRUPTED" else status
            detail["application_cold_error"] = cleanup["application_cold_error"]
        if radio_namespace is not None and not cleanup.get("radio_shutdown_verified"):
            status = "FAILED" if status != "INTERRUPTED" else status
            detail["radio_cleanup_error"] = "qualified radio lifecycle did not restore and prove cold cleanup"
        detail["service_diagnostics"] = _compact_log_diagnostics(
            service_log_dir, include_tails=status != "PASSED"
        )
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
        shutil.rmtree(service_log_dir, ignore_errors=True)
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
    expected_outputs = [
        "per_frame_metrics.csv", "radio_trace.csv", "map_feedback.csv",
        "perception_metrics.csv", "resolved_config.yaml", "RESULTS_SUMMARY.json",
        "manifest.json",
    ]
    passed = sum(
        1 for rows in cells.values()
        if isinstance(rows, list) and passed_attempt_exists(path.parent, rows, expected_outputs)
    )
    require(passed == 16 and len(cells) == 16, f"full sweep requires all 16 pilot cells PASSED; found {passed}/16")


def verify_phase15_qualification(path: Path) -> dict[str, Any]:
    """Require a hash-bound successful live qualification before the pilot."""

    root = path.resolve(strict=True)
    try:
        root.relative_to((ROOT / "experiments").resolve(strict=True))
    except ValueError as exc:
        raise CampaignError("Phase-15 qualification root escapes experiments") from exc
    qualification_path = root / "qualification.json"
    manifest_path = root / "artifact_manifest.json"
    report_path = root / "REPORT.md"
    terminal_path = root / PHASE15_QUALIFICATION_TERMINAL
    for artifact in (qualification_path, manifest_path, report_path, terminal_path):
        require(artifact.is_file(), f"Phase-15 qualification artifact missing: {artifact}")
    qualification = load_json(qualification_path)
    preflight_path = root / "preflight_inventory.json"
    preflight = load_json(preflight_path)
    require(
        qualification.get("preflight_inventory_sha256") == sha256_file(preflight_path)
        and preflight.get("status") == "PASS",
        "Phase-15 qualification/preflight binding drift",
    )
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema") == "scenesense.splitfusion_phase15_live_deployment_artifacts.v1",
        "Phase-15 qualification artifact manifest schema drift",
    )
    manifest_files = manifest.get("files", ())
    require(
        isinstance(manifest_files, list)
        and {str(item.get("path")) for item in manifest_files}
        == {
            "preflight_inventory.json", "qualification.json", "REPORT.md",
            "runtime/RESULTS_SUMMARY.json", "runtime/manifest.json",
            "runtime/per_frame_metrics.csv", "runtime/map_feedback.csv",
            "runtime/radio_trace.csv",
        },
        "Phase-15 qualification artifact inventory drift",
    )
    for item in manifest_files:
        raw_artifact = root / str(item.get("path", ""))
        require(raw_artifact.is_file(), f"Phase-15 qualification artifact missing: {raw_artifact}")
        artifact = raw_artifact.resolve(strict=True)
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise CampaignError("Phase-15 qualification manifest path escapes its root") from exc
        require(
            artifact.stat().st_size == int(item.get("bytes", -1))
            and sha256_file(artifact) == str(item.get("sha256", "")),
            f"Phase-15 qualification artifact drift: {item.get('path')}",
        )
    require(
        qualification.get("schema") == PHASE15_QUALIFICATION_SCHEMA
        and qualification.get("status") == PHASE15_QUALIFICATION_TERMINAL
        and int(qualification.get("captures", -1)) == 20
        and qualification.get("action_counts") == {"0": 5, "20": 5, "46": 5, "71": 5}
        and qualification.get("cold_cleanup_verified") is True,
        "Phase-15 live qualification gates are not complete",
    )
    terminal = load_json(terminal_path)
    require(
        terminal.get("status") == PHASE15_QUALIFICATION_TERMINAL
        and terminal.get("qualification_sha256") == sha256_file(qualification_path)
        and terminal.get("artifact_manifest_sha256") == sha256_file(manifest_path),
        "Phase-15 qualification terminal/hash binding drift",
    )
    image = subprocess.run(
        ("sudo", "-n", "docker", "image", "inspect", "oai-perception-rx:latest"),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False,
    )
    require(image.returncode == 0, "qualified Phase-15 edge image is absent")
    image_value = json.loads(image.stdout)
    require(
        isinstance(image_value, list) and len(image_value) == 1
        and image_value[0].get("Id") == preflight.get("edge_image", {}).get("image_id"),
        "qualified Phase-15 edge image identity drift",
    )
    return {
        "root": str(root.relative_to(ROOT)),
        "qualification_sha256": sha256_file(qualification_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "terminal_sha256": sha256_file(terminal_path),
        "edge_image_id": image_value[0]["Id"],
    }


def run_campaign(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config, cells, _hashes = validate_static(config_path)
    live_pilot = config.get("campaign_kind") == "live_pilot_16"
    if live_pilot:
        require(args.execute == LIVE_PILOT_TOKEN, "exact live-pilot execution token is required")
        worktree = verify_live_pilot_worktree(
            owned_resume_root=args.output_root if args.resume else None
        )
        require(args.qualification_root is not None, "live pilot requires --qualification-root")
        live_qualification = verify_phase15_qualification(args.qualification_root)
        _phase15_gpu_audit()
        _require_phase15_application_cold(config)
    else:
        worktree = {}
        live_qualification = {}
    apply_model_overrides(config, args.model)
    if args.maximum_loop_sim_s is not None:
        config["_maximum_loop_sim_s_override"] = float(args.maximum_loop_sim_s)
    catalog = read_catalog(config)
    verify_real_launch_readiness(config)
    verify_resolved_models(config, catalog)
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

    if args.cell_id is not None:
        selected = [cell for cell in cells if cell.cell_id == args.cell_id]
        require(len(selected) == 1, f"unknown --cell-id: {args.cell_id}")
        cells = selected

    campaign_root = args.output_root.resolve(strict=False)
    if live_pilot:
        experiments = (ROOT / "experiments").resolve(strict=True)
        try:
            campaign_root.relative_to(experiments)
        except ValueError as exc:
            raise CampaignError("live pilot output must remain beneath experiments") from exc
        if args.resume:
            require(campaign_root.is_dir(), "live-pilot resume output does not exist")
        else:
            require(not campaign_root.exists(), f"create-only live-pilot output exists: {campaign_root}")
            campaign_root.parent.mkdir(parents=True, exist_ok=True)
            campaign_root.mkdir(parents=False, exist_ok=False)
    else:
        campaign_root.mkdir(parents=True, exist_ok=True)
    config_digest = sha256_file(config_path)
    if live_pilot:
        manifest_path = campaign_root / "pilot_manifest.json"
        expected_manifest = {
            "schema": "scenesense.splitfusion_16_cell_live_carla_oai_pilot_manifest.v1",
            "campaign_id": config["campaign_id"], "config_sha256": config_digest,
            "git": worktree, "cell_mapping_sha256": cell_mapping_sha256(cells),
            "required_cells": 16, "full_288_campaign_authorized": False,
            "phase14b_mapping_evidence": "COMMAND_VALIDITY_COVERAGE",
            "phase14b_probe_started": False,
            "phase15_live_qualification": live_qualification,
        }
        if args.resume:
            require(manifest_path.is_file(), "live-pilot resume lacks immutable manifest")
            require(load_json(manifest_path) == expected_manifest, "live-pilot immutable manifest drift")
        else:
            write_create_only(manifest_path, json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n")
    ledger_path = campaign_root / str(config["cell"]["resume_ledger"])
    ledger = load_ledger(ledger_path, str(config["campaign_id"]), config_digest)
    for cell in cells:
        rows = ledger["cells"].setdefault(cell.cell_id, [])
        require(isinstance(rows, list), f"ledger rows are not a list for {cell.cell_id}")
        if passed_attempt_exists(campaign_root, rows, config["cell"]["expected_outputs"]):
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
        if result["status"] in {"FAILED", "INTERRUPTED"}:
            return 130 if result["status"] == "INTERRUPTED" else 1
    if live_pilot:
        finalize_live_pilot(campaign_root, config, cells, ledger)
    return 0


def live_preflight_command(args: argparse.Namespace) -> int:
    """Perform all guarded live-pilot checks without creating an output leaf."""

    config_path = args.config.resolve(strict=True)
    config, cells, trace_hashes = validate_static(config_path)
    require(config.get("campaign_kind") == "live_pilot_16", "preflight is only for the qualified live pilot")
    require(args.execute == LIVE_PILOT_TOKEN, "exact live-pilot execution token is required")
    require(len(cells) == 16, "live pilot must retain exactly 16 cells")
    worktree = verify_live_pilot_worktree()
    verify_real_launch_readiness(config)
    verify_resolved_models(config, read_catalog(config))
    output = args.output_root.resolve(strict=False)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        output.relative_to(experiments)
    except ValueError as exc:
        raise CampaignError("live pilot output must remain beneath experiments") from exc
    require(not output.exists(), f"create-only live-pilot output exists: {output}")
    require(args.qualification_root is not None, "live pilot requires --qualification-root")
    live_qualification = verify_phase15_qualification(args.qualification_root)
    gpu = _phase15_gpu_audit()
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

    radio = phase14a.load_json(repo_path(str(config["runtime"]["phase14a_config"])))
    phase14b.require_cold_profile_runtime(radio, output / "preflight_no_output")
    application_cold = _require_phase15_application_cold(config)
    print(json.dumps({
        "status": "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_PREFLIGHT_PASS",
        "head": worktree["head"], "dirty_paths": worktree["dirty_paths"],
        "cell_count": len(cells), "trace_prefix_hashes": trace_hashes,
        "cuda_device": gpu["device_name"], "gpu": gpu, "output_absent": True,
        "phase14b_mapping_evidence": "COMMAND_VALIDITY_COVERAGE",
        "probe_forbidden": True, "external_processes_started": 0,
        "phase15_live_qualification": live_qualification,
        "application_cold": application_cold,
    }, indent=2, sort_keys=True))
    return 0


def finalize_live_pilot(
    campaign_root: Path, config: Mapping[str, Any], cells: Sequence[Cell], ledger: Mapping[str, Any]
) -> None:
    """Create only compact, hash-bound evidence after all sixteen cells pass."""

    require(config.get("campaign_kind") == "live_pilot_16", "only the live pilot has this finalizer")
    expected_outputs = config["cell"]["expected_outputs"]
    rows: list[dict[str, Any]] = []
    for cell in cells:
        attempts = ledger["cells"].get(cell.cell_id, [])
        require(passed_attempt_exists(campaign_root, attempts, expected_outputs), f"cell lacks a valid passed attempt: {cell.cell_id}")
        passed = next(row for row in attempts if row.get("status") == "PASSED")
        attempt_dir = campaign_root / str(passed["attempt_dir"])
        summary = load_json(attempt_dir / "RESULTS_SUMMARY.json")
        per_frame = attempt_dir / "per_frame_metrics.csv"
        with per_frame.open(newline="", encoding="utf-8") as handle:
            frame_rows = list(csv.DictReader(handle))
        sent = [row for row in frame_rows if row.get("prepare_status") == "SENT"]
        structural = summary.get("structural_acceptance", {})
        installed = int(structural.get("ack_installed_frames", 0))
        with (attempt_dir / "map_feedback.csv").open(newline="", encoding="utf-8") as handle:
            feedback_rows = list(csv.DictReader(handle))
        terminal_feedback = [
            row for row in feedback_rows
            if str(row.get("terminal", "")).casefold() in {"1", "true"}
        ]
        installed_feedback = [row for row in feedback_rows if row.get("status") == "ACK_INSTALLED"]
        aoi_ms = [
            (float(row["install_timestamp"]) - float(row["capture_at"])) * 1000.0
            for row in installed_feedback
            if row.get("install_timestamp") not in (None, "")
            and row.get("capture_at") not in (None, "")
        ]
        latencies = [
            (float(row["edge_result_received_ns"]) - float(row["capture_started_ns"])) / 1e6
            for row in sent
            if row.get("edge_result_received_ns") not in (None, "") and row.get("capture_started_ns") not in (None, "")
        ]
        payloads = [int(row["sfd1_bytes"]) for row in sent if row.get("sfd1_bytes") not in (None, "")]
        with (attempt_dir / "radio_trace.csv").open(newline="", encoding="utf-8") as handle:
            radio_rows = list(csv.DictReader(handle))
        targets = [float(row["target_snr_db"]) for row in radio_rows if row.get("target_snr_db") not in (None, "")]
        commands = [float(row["mapped_rfsim_command_db"]) for row in radio_rows if row.get("mapped_rfsim_command_db") not in (None, "")]
        timing_counts: dict[str, int] = {}
        for radio_row in radio_rows:
            status = str(radio_row.get("command_timing_status") or "")
            timing_counts[status] = timing_counts.get(status, 0) + 1
        feedback_counts: dict[str, int] = {}
        for feedback_row in terminal_feedback:
            status = str(feedback_row.get("status") or "")
            feedback_counts[status] = feedback_counts.get(status, 0) + 1
        duplicate_feature_datagrams = sum(
            int(row["feature_duplicate_datagrams"])
            for row in sent if row.get("feature_duplicate_datagrams") not in (None, "")
        )
        feature_datagrams = sum(
            int(row["feature_received_datagrams"])
            for row in sent if row.get("feature_received_datagrams") not in (None, "")
        )
        passed_terminal = load_json(attempt_dir / "PASSED.json")
        passed_cleanup = passed_terminal.get("carla_cleanup", {})
        rows.append({
            "cell_id": cell.cell_id, "action_id": cell.action_id, "profile_id": cell.profile_id,
            "network_profile_id": cell.network_profile_id, "attempt": int(passed["attempt"]),
            "route_completed": bool(summary.get("route", {}).get("route_completed")),
            "captures_sent": len(sent), "ack_installed_frames": installed,
            "delivery_rate": (installed / len(sent)) if sent else 0.0,
            "mean_capture_to_edge_result_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "maximum_capture_to_edge_result_ms": max(latencies) if latencies else None,
            "mean_install_aoi_ms": (sum(aoi_ms) / len(aoi_ms)) if aoi_ms else None,
            "maximum_install_aoi_ms": max(aoi_ms) if aoi_ms else None,
            "mean_sfd1_bytes": (sum(payloads) / len(payloads)) if payloads else None,
            "total_sfd1_bytes": sum(payloads),
            "prepared_queue_drops": int(summary.get("split_frames_dropped", 0)),
            "feature_datagrams": feature_datagrams,
            "duplicate_feature_datagrams": duplicate_feature_datagrams,
            "terminal_feedback_outcomes": json.dumps(feedback_counts, sort_keys=True, separators=(",", ":")),
            "socket_buffers": json.dumps(summary.get("live_dispatch", {}).get("socket_buffers", {}), sort_keys=True, separators=(",", ":")),
            "radio_trace_rows": len(radio_rows),
            "radio_target_min_db": min(targets) if targets else None,
            "radio_target_mean_db": (sum(targets) / len(targets)) if targets else None,
            "radio_target_max_db": max(targets) if targets else None,
            "radio_command_min_db": min(commands) if commands else None,
            "radio_command_mean_db": (sum(commands) / len(commands)) if commands else None,
            "radio_command_max_db": max(commands) if commands else None,
            "radio_timing_outcomes": json.dumps(timing_counts, sort_keys=True, separators=(",", ":")),
            "route_density": config["route_b"]["density"],
            "route_scenario_seed": int(config["route_b"]["scenario_seed"]),
            "route_traffic_manager_seed": int(config["route_b"]["traffic_manager_seed"]),
            "route_ticks": int(structural.get("route_ticks", 0)),
            "sensor_preparation_coverage": structural.get("sensor_preparation_coverage"),
            "cold_cleanup_verified": all((
                bool(passed_cleanup.get("radio_shutdown_verified")),
                bool(passed_cleanup.get("carla", {}).get("shutdown_verified")),
                not passed_cleanup.get("application_cold_error"),
            )),
            "attempt_manifest_sha256": sha256_file(attempt_dir / "manifest.json"),
        })
    require(len(rows) == 16 and len({row["cell_id"] for row in rows}) == 16, "final pilot cell inventory drift")
    summary_path = campaign_root / "cell_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qualification = {
        "schema": "splitfusion_16_cell_live_carla_oai_pilot_qualification.v1",
        "status": "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_COMPLETE",
        "campaign_id": config["campaign_id"], "required_cells": 16, "valid_completed_cells": 16,
        "cell_mapping_sha256": config["actions"]["cell_mapping_sha256"],
        "phase14b_mapping_evidence": "COMMAND_VALIDITY_COVERAGE", "probe_executed": False,
        "full_288_campaign_authorized": False, "cells": rows,
    }
    qualification_path = campaign_root / "qualification.json"
    write_create_only(qualification_path, json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    report_path = campaign_root / "REPORT.md"
    report = ["# SplitFusion 16-cell live CARLA/OAI pilot", "", "All sixteen registered cells completed with fresh radio/CARLA lifecycles.", "", "| Cell | Action | Network profile | Captures | ACK installed | Delivery |", "|---|---:|---|---:|---:|---:|"]
    report.extend(
        f"| {row['cell_id']} | {row['action_id']} | {row['network_profile_id']} | {row['captures_sent']} | {row['ack_installed_frames']} | {row['delivery_rate']:.6f} |"
        for row in rows
    )
    write_create_only(report_path, "\n".join(report) + "\n")
    artifact = campaign_root / "artifact_manifest.json"
    artifact_value = {"schema": "splitfusion_16_cell_live_carla_oai_pilot_artifacts.v1", "files": [
        {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (summary_path, qualification_path, report_path, campaign_root / "pilot_manifest.json", campaign_root / str(config["cell"]["resume_ledger"]))
    ]}
    write_create_only(artifact, json.dumps(artifact_value, indent=2, sort_keys=True) + "\n")
    terminal = campaign_root / "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_COMPLETE"
    write_create_only(terminal, "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_COMPLETE\n")


def validate_command(args: argparse.Namespace) -> int:
    campaign, campaign_cells, campaign_hashes = validate_static(args.campaign.resolve())
    pilot, pilot_cells, pilot_hashes = validate_static(args.pilot.resolve())
    if pilot.get("campaign_kind") == "live_pilot_16":
        require(len(pilot_cells) == 16, "qualified live pilot did not enumerate 16 cells")
        require([cell.network_profile_id for cell in pilot_cells[::4]] == [
            "FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY",
        ], "live-pilot profile order drift")
        require([cell.action_id for cell in pilot_cells[:4]] == [0, 71, 46, 20], "registered live-pilot action order drift")
        print(json.dumps({
            "status": "LIVE_16_CELL_OFFLINE_VALIDATION_PASS",
            "external_processes_started": 0,
            "pilot_cells": len(pilot_cells),
            "pilot_mapping_sha256": cell_mapping_sha256(pilot_cells),
            "trace_prefix_hashes": pilot_hashes,
            "phase14b_corrected_mapping_evidence": "COMMAND_VALIDITY_COVERAGE",
            "phase14b_probe_forbidden_during_pilot": True,
            "full_288_campaign_unauthorized": True,
            "resume_ledger_dry_run": resume_ledger_dry_run(pilot_cells),
        }, indent=2, sort_keys=True))
        return 0
    require(len(campaign_cells) == 288, "full campaign did not enumerate 288 cells")
    require(len(pilot_cells) == 16, "integration pilot did not enumerate 16 cells")
    require(campaign_hashes == pilot_hashes, "pilot/full trace hashes differ")
    for key in ("catalog_json", "catalog_sha256", "catalog_commit", "catalog_schema"):
        require(
            campaign["actions"][key] == pilot["actions"][key],
            f"pilot/full action catalog {key} bindings differ",
        )
    require(
        campaign["network"]["profiles"] == pilot["network"]["profiles"],
        "pilot/full network profile bindings differ",
    )
    require(
        campaign["network"]["radio_baseline"] == pilot["network"]["radio_baseline"],
        "pilot/full selected OAI radio baselines differ",
    )
    require(campaign["route_b"] == pilot["route_b"], "pilot/full Route B contract differs")
    require(
        campaign["measurement_contract"] == pilot["measurement_contract"],
        "pilot/full measurement contracts differ",
    )
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
            "catalog_models": "hash-bound; files are checked only by the guarded real-launch preflight",
            "campaign_bindings": real_launch_blockers(campaign),
            "pilot_bindings": real_launch_blockers(pilot),
        },
        "campaign_mapping_sha256": cell_mapping_sha256(campaign_cells),
        "pilot_mapping_sha256": cell_mapping_sha256(pilot_cells),
        "pilot_actions": [
            {"action_id": cell.action_id, "profile_id": cell.profile_id}
            for cell in pilot_cells[:4]
        ],
        "selected_oai_radio_profile": campaign["network"]["radio_baseline"],
        "qualified_route_b_split_cell_adapter": adapter_value(campaign),
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
        help="legacy compatibility argument; any override is refused by the locked catalog",
    )
    run.add_argument("--route-b-split-cell-adapter", default=None)
    run.add_argument("--carla-port", type=int, default=2000)
    run.add_argument("--authorize-full-sweep", action="store_true")
    run.add_argument("--pilot-ledger", type=Path)
    run.add_argument("--cell-id", help="run exactly one registered cell (bounded integration smoke)")
    run.add_argument("--maximum-loop-sim-s", type=float, help="bounded Route B smoke override")
    run.add_argument("--execute")
    run.add_argument("--qualification-root", type=Path)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=run_campaign)

    preflight = subparsers.add_parser("preflight", help="read-only live-pilot gate")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--execute", required=True)
    preflight.add_argument("--qualification-root", type=Path)
    preflight.set_defaults(func=live_preflight_command)
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
