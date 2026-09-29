#!/usr/bin/env python3
"""Config loading, runtime merge and source inventory for the capture package.

The live runtime is the reviewed 273PRB/4D5U near-capacity runtime, overridden
only by this package's traffic ports, buffers and campaign settings.  The base
runtime is pinned by digest so a silent radio re-binding cannot happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import contract as C


DEFAULT_CONFIG = C.ROOT / C.PACKAGE_RELPATH / "config_v1.json"
CONFIG_SCHEMA = "scenesense.ue_production_queue_capture_config.v1"

# Every source file whose behaviour this capture depends on.  The inventory is
# re-verified before preflight, before every cell and at final sealing.
INVENTORY_FILES: tuple[str, ...] = (
    f"{C.PACKAGE_RELPATH}/contract.py",
    f"{C.PACKAGE_RELPATH}/config.py",
    f"{C.PACKAGE_RELPATH}/config_v1.json",
    f"{C.PACKAGE_RELPATH}/payload_schedule.py",
    f"{C.PACKAGE_RELPATH}/production_sender.py",
    f"{C.PACKAGE_RELPATH}/production_receiver.py",
    f"{C.PACKAGE_RELPATH}/authorization.py",
    f"{C.PACKAGE_RELPATH}/runner.py",
)

# Inherited lifecycle sources, imported unchanged.  They are committed and
# clean; a drift here invalidates the capture.
INHERITED_SOURCES: tuple[str, ...] = (
    "rl_agent/ue_mcs_backlog_calibration_v1/runner.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/runner.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_runner.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/radio_binding.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/protected_evidence.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    C.require(value.get("schema") == CONFIG_SCHEMA, "capture config schema drifted")
    C.require(value.get("package_id") == C.PACKAGE_ID, "capture package id drifted")
    packet = value["packetization"]
    C.require(
        packet["udp_chunk_bytes_including_header"]
        == C.UDP_CHUNK_BYTES_INCLUDING_HEADER
        and packet["chunk_header_bytes"] == C.UDP_CHUNK_HEADER_BYTES
        and packet["payload_bytes_per_datagram"]
        == C.UDP_PAYLOAD_BYTES_PER_DATAGRAM
        and packet["retransmission"] is False
        and packet["historical_60kb_binding_rejected"] is True,
        "capture packetization drifted from the live production binding",
    )
    design = value["design"]
    C.require(
        design["fps"] == C.FPS
        and design["frames_per_cell"] == C.FRAMES_PER_CELL
        and design["expected_cells"] == C.EXPECTED_CELLS
        and design["expected_raw_frames"] == C.EXPECTED_RAW_FRAMES
        and design["expected_primary_cycles"] == C.EXPECTED_PRIMARY_CYCLES,
        "capture design drifted from the frozen contract",
    )
    for tier, spec in (("floor", C.tier_by_name("floor")),
                       ("knee", C.tier_by_name("knee")),
                       ("guard", C.tier_by_name("guard"))):
        row = value["byte_roles"][tier]
        C.require(
            row["action_id"] == spec.action_id and row["mode_id"] == spec.mode_id
            and row["q_e4"] == spec.q_e4 and row["family"] == spec.family
            and row["quantizer"] == spec.quantizer,
            f"byte role {tier} drifted from the frozen contract",
        )
    C.require(set(value["traffic"]["ports"]) == set(C.TIER_NAMES),
             "traffic ports must cover exactly the three byte roles")
    return value


def base_runtime_config(
    config: Mapping[str, Any], repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    binding = config["base_runtime"]
    path = repo_root / binding["config_relative_path"]
    C.require(path.is_file(), "base runtime config missing")
    C.require(C.sha256_file(path) == binding["config_sha256"],
              "base runtime config drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    C.require(value["radio"]["profile_id"] == binding["radio_profile_id"],
              "base runtime radio profile drifted")
    return value


def effective_runtime_config(
    path: Path = DEFAULT_CONFIG, repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    """Base 273PRB runtime, overridden only by this package's own sections."""
    registration = load_config(path)
    runtime = base_runtime_config(registration, repo_root)
    runtime["traffic"] = registration["traffic"]
    runtime["campaign"] = registration["campaign"]
    runtime["paths"] = dict(runtime["paths"])
    runtime["paths"]["output_root"] = registration["paths"]["output_root"]
    runtime["production_queue_capture"] = {
        "package_id": registration["package_id"],
        "claim_boundary": registration["claim_boundary"],
        "design": registration["design"],
        "byte_roles": registration["byte_roles"],
        "packetization": registration["packetization"],
        "payload_authority": registration["payload_authority"],
        "failure_policy": registration["failure_policy"],
        "authorization": registration["authorization"],
    }
    return runtime


def source_inventory(repo_root: Path = C.ROOT) -> dict[str, Any]:
    entries = {}
    for relative in INVENTORY_FILES + INHERITED_SOURCES:
        path = repo_root / relative
        C.require(path.is_file(), f"inventory source missing: {relative}")
        entries[relative] = C.sha256_file(path)
    return {"files": entries, "inventory_sha256": C.canonical_sha256(entries)}
