"""Fail-closed design contract for the Run-4 physical queue calibration.

This package does not reinterpret the refused capacity qualification.  It
opens the separately sealed robust amendment, then uses its exact AE64/UINT8
actions as byte loads in a preregistered 12-cell queue experiment.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as NC
from rl_agent.ue_mcs_backlog_robust_bracket_v1 import amendment as AM


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RELPATH = "rl_agent/ue_mcs_backlog_run4_calibration_v1"
CONFIG_RELPATH = f"{PACKAGE_RELPATH}/config_v2.json"
DEFAULT_CONFIG = ROOT / CONFIG_RELPATH

CONFIG_SCHEMA = "scenesense.ue_mcs_backlog_run4_calibration_config.v2"
PACKAGE_ID = "ue_mcs_backlog_run4_calibration_v2"
CONTRACT_ID = PACKAGE_ID
CONTRACT_VERSION = 2
CLAIM_BOUNDARY = (
    "BOUNDED_PHYSICAL_QUEUE_CALIBRATION_FOR_OFFLINE_RUN4_MODELING_"
    "NOT_POLICY_VALIDATION_NOT_PERCEPTION_ENDORSEMENT"
)
AMENDMENT_STATUS = AM.STATUS
LEGACY_AMENDMENT_ACTION_IDS = {"low": 40, "medium": 39, "high": 38}
LEGACY_AMENDMENT_PAYLOAD_BYTES = {
    "low": 126_237, "medium": 374_264, "high": 619_563,
}
LEGACY_AMENDMENT_OFFERED_MBPS = {
    "low": 10.09896, "medium": 29.94112, "high": 49.56504,
}
AMENDMENT_RELPATH = (
    "rl_agent/ue_mcs_backlog_robust_bracket_v1/sealed/"
    "PROSPECTIVE_CAPACITY_AMENDMENT.json"
)
AMENDMENT_SHA256 = "fa6ef35816c61442aa7c287c31652d06f09fb1c0f8f914d467200629114c6bf0"
AMENDMENT_MANIFEST_SHA256 = (
    "bc739a5c4350680738121371df8751e1f30b18ba502170dae1a1c4a8c8ea8ec5"
)
AMENDMENT_TERMINAL_SHA256 = (
    "ef9c36aacf6cef310e793828265939ef0a2a2a4208e4c8278b04bf7a67a77dd1"
)
BASE_CONFIG_RELPATH = "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json"
BASE_CONFIG_SHA256 = "65fb3d1d3a92c53a6db40c3ba7dc6b4f74b3447428d28d5c6ada0aeb69773659"

TIER_DESIGN_RELPATH = f"{PACKAGE_RELPATH}/TIER_DESIGN_V2.json"
TIER_DESIGN_SHA256 = "abb908e54224d9305980eeef0b22581fe6b7b7ac0533b4858e8611dec4f72476"
PROFILE_BINDING_RELPATH = NC.PROFILE_BINDING_RELPATH
PROFILE_BINDING_SHA256 = "0dafbe9150859111956618f9573b94c48ad83b9edd43d7b06bad9da04f577f69"
PROFILE_TRACE_RELPATH = NC.PROFILE_TRACE_RELPATH
PROFILE_TRACE_SHA256 = "32f1be66e976cba322803c128eefdeb81a8d31ed64bd253ff85ea1ab0583303d"
TRACE_FIELDS = (
    "profile_id", "trace_id", "seed", "value_semantics", "step_index",
    "interval_start_s", "interval_end_s", "state_index", "state",
    "target_snr_db",
)
TRACE_VALUE_SEMANTICS = "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR"
TRACE_STATE_ORDER = ("ADVERSE", "INTERMEDIATE", "FAVORABLE")

TIER_ORDER = ("low", "medium", "high")
EXPECTED_ACTION_IDS = {"low": 71, "medium": 22, "high": 38}
EXPECTED_MODE_IDS = {"low": 11, "medium": 3, "high": 6}
EXPECTED_FAMILIES = {"low": "AE32", "medium": "AE128", "high": "AE64"}
EXPECTED_QUANTIZERS = {"low": "UINT4", "medium": "UINT8", "high": "UINT8"}
EXPECTED_Q_E4 = {"low": 9800, "medium": 9000, "high": 5000}
EXPECTED_PROFILE_IDS = {
    "low": "split_ae32_uint4_q9800",
    "medium": "split_ae128_uint8_q9000",
    "high": "split_ae64_uint8_q5000",
}
EXPECTED_PAYLOAD_BYTES = {"low": 6_398, "medium": 248_703, "high": 619_858}
EXPECTED_OFFERED_MBPS = {"low": 0.51184, "medium": 19.89624, "high": 49.58864}
PRODUCTION_PAYLOAD_SUPPORT_BYTES = (6_423, 427_605)
TENSORS_PER_POLICY_DECISION = 2
REWARD_THEN_HELD_ROLES = ("REWARD_REQUESTED", "HELD_NO_REWARD")
CHUNK_BYTES = 1_200
SSBURST_HEADER_BYTES = 24
UDP_HEADER_BYTES = 8
IPV4_HEADER_BYTES = 20
FULL_IPV4_PACKET_BYTES = 1_252
PATH_MTU_BYTES = 1_500
FPS = 10.0
FRAMES_PER_BLOCK = 150
BLOCKS_PER_CELL = 3
FRAMES_PER_CELL = FRAMES_PER_BLOCK * BLOCKS_PER_CELL
CONTRAST_PROFILE_IDS = ("FAVORABLE_STABLE", "ADVERSE_STABLE")
FIT = "FIT"
VALIDATION = "VALIDATION"
FIT_PERMUTATIONS = (
    ("low", "medium", "high"),
    ("medium", "high", "low"),
    ("high", "low", "medium"),
)
VALIDATION_PERMUTATIONS = tuple(tuple(reversed(row)) for row in FIT_PERMUTATIONS)
PERMUTATIONS = FIT_PERMUTATIONS + VALIDATION_PERMUTATIONS
PARTITIONS = (FIT,) * 3 + (VALIDATION,) * 3
CELL_ORDER_SEED = 2026092401
PAYLOAD_SEED = 2026092401
SAMPLE_PERIOD_S = 0.1
OUTPUT_ROOT_RELPATH = "rl_agent/experiments/ue_mcs_backlog_run4_calibration_v2"
MAX_GRANT_LIFETIME_SECONDS = 21_600
EXPECTED_CELLS = 12
EXPECTED_TENSOR_FRAMES = 5_400
EXPECTED_POLICY_DECISIONS = EXPECTED_TENSOR_FRAMES // TENSORS_PER_POLICY_DECISION
FUTURE_EPOCH_LEAD_S = 0.02
SENDER_ARM_TIMEOUT_S = 10.0

INHERITED_SOURCE_PINS: Mapping[str, str] = {
    "rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_runner.py":
        "c7cc27f71758aab09b2149149763b6e10f51572c94a73f470f279f01e9637427",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/runner.py":
        "8989f8f42db80fe23af7109a00aca3185760ace04f813d5c57253d864012238b",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/contract.py":
        "99dd192e7df07c1aa7e4d8fcc1d2dc3d5d68b5e23fcedd0109ab2265b962e36e",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/radio_binding.py":
        "7585f5925cb03ae5d6c0aca68fb311b84ef0af7154a850bbdfc0f46460694356",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/protected_evidence.py":
        "8c3da8e75d904dcdef721dc5c9ef901f4106d95717ee501666dd787face1859e",
    "rl_agent/ue_mcs_backlog_calibration_v1/runner.py":
        "6c2a74d79e1c2415cd11eb4593314d3782ebcd0536a9f5ab7d9d4a393538741d",
    "rl_agent/ue_mcs_backlog_calibration_v1/contract.py":
        "d87b4bbddfa116ecd0a98f57bd92ef187537efc92a37b5e4e31dab0e4a991f11",
    "rl_agent/ue_n3_structured_udp_receiver.py":
        "3e92ba8f756d2c028dcfe51c2a35f18d4f268a041516ea1e054f349f1b916446",
    "rl_agent/ue_mcs_backlog_robust_bracket_v1/amendment.py":
        "e4070eaec1a7110b013dc635aae98df5f2a6a53c22e372f99ecada2706b53ea8",
    "rl_agent/ue_mcs_backlog_robust_bracket_v1/selector.py":
        "5f160872d570fed160533b9bc6a52c4745d4ede938c6910ac438976a469e6db6",
}

PACKAGE_SOURCE_RELPATHS = (
    f"{PACKAGE_RELPATH}/__init__.py",
    f"{PACKAGE_RELPATH}/__main__.py",
    f"{PACKAGE_RELPATH}/contract.py",
    f"{PACKAGE_RELPATH}/authorization.py",
    f"{PACKAGE_RELPATH}/tagged_sender.py",
    f"{PACKAGE_RELPATH}/runner.py",
    f"{PACKAGE_RELPATH}/config_v1.json",
    CONFIG_RELPATH,
    TIER_DESIGN_RELPATH,
    f"{PACKAGE_RELPATH}/PREREGISTRATION.md",
    f"{PACKAGE_RELPATH}/PREREGISTRATION_V2.md",
    f"{PACKAGE_RELPATH}/INHERITED_LIFECYCLE_AUDIT.md",
    f"{PACKAGE_RELPATH}/test_run4_calibration_v1.py",
)


class ContractError(RuntimeError):
    """A registered source, amendment, packetization or plan gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(type(config) is dict and config.get("schema") == CONFIG_SCHEMA,
            "Run-4 calibration config schema mismatch")
    require(set(config) == {
        "schema", "package_id", "claim_boundary", "base_runtime",
        "robust_amendment", "tier_design", "actions", "packetization",
        "design", "paths", "authorization", "failure_policy",
    }, "Run-4 calibration config fields drifted")
    require(config.get("package_id") == PACKAGE_ID, "package identity mismatch")
    require(config.get("claim_boundary") == CLAIM_BOUNDARY,
            "claim boundary drifted")
    base = config.get("base_runtime")
    require(type(base) is dict
            and base.get("config_relative_path") == BASE_CONFIG_RELPATH
            and base.get("config_sha256") == BASE_CONFIG_SHA256
            and base.get("radio_profile_id") ==
                "OAI_N78_100MHZ_273PRB_4D5U_V1",
            "base runtime binding drifted")
    amendment = config.get("robust_amendment")
    require(type(amendment) is dict
            and amendment.get("relative_path") == AMENDMENT_RELPATH
            and amendment.get("amendment_sha256") == AMENDMENT_SHA256
            and amendment.get("manifest_sha256") == AMENDMENT_MANIFEST_SHA256
            and amendment.get("terminal_sha256") == AMENDMENT_TERMINAL_SHA256
            and amendment.get("status") == AMENDMENT_STATUS,
            "sealed amendment binding drifted")
    require(amendment.get("role") ==
            "historical_capacity_lineage_not_v2_tier_authority",
            "v1 amendment must not be relabelled as v2 tier authority")
    tier_design = config.get("tier_design")
    require(type(tier_design) is dict
            and tier_design.get("relative_path") == TIER_DESIGN_RELPATH
            and tier_design.get("sha256") == TIER_DESIGN_SHA256
            and tier_design.get("status") == "PROSPECTIVE_REVIEWED_NOT_EXECUTED",
            "prospective v2 tier-design binding drifted")
    actions = config.get("actions")
    require(type(actions) is dict
            and actions.get("perception_endorsement") is False
            and actions.get("byte_only_queue_design") is True,
            "action identity or caveat drifted")
    require(actions.get("production_payload_coordinate") ==
            "total_transmitted_bytes"
            and tuple(actions.get("production_payload_support_bytes", ())) ==
                PRODUCTION_PAYLOAD_SUPPORT_BYTES,
            "production payload coordinate/support drifted")
    for tier in TIER_ORDER:
        row = actions.get(tier)
        require(type(row) is dict
                and row.get("action_id") == EXPECTED_ACTION_IDS[tier]
                and row.get("mode_id") == EXPECTED_MODE_IDS[tier]
                and row.get("family") == EXPECTED_FAMILIES[tier]
                and row.get("quantizer") == EXPECTED_QUANTIZERS[tier]
                and row.get("q_e4") == EXPECTED_Q_E4[tier]
                and row.get("profile_id") == EXPECTED_PROFILE_IDS[tier]
                and row.get("payload_bytes") == EXPECTED_PAYLOAD_BYTES[tier]
                and row.get("offered_mbps") == EXPECTED_OFFERED_MBPS[tier],
                f"{tier} action binding drifted")
    packet = config.get("packetization")
    require(type(packet) is dict
            and packet.get("application_chunk_bytes") == CHUNK_BYTES
            and packet.get("ssburst_header_bytes") == SSBURST_HEADER_BYTES
            and packet.get("full_ipv4_packet_bytes") == FULL_IPV4_PACKET_BYTES
            and packet.get("udp_header_bytes") == UDP_HEADER_BYTES
            and packet.get("ipv4_header_bytes") == IPV4_HEADER_BYTES
            and packet.get("path_mtu_bytes") == PATH_MTU_BYTES
            and packet.get("mtu_safe_without_ipv4_fragmentation") is True,
            "MTU-safe packetization drifted")
    require(packet.get("identity") ==
            "CALIBRATION_ONLY_NOT_PRODUCTION_PACKETIZATION",
            "calibration packetization was confused with production")
    require(FULL_IPV4_PACKET_BYTES <= PATH_MTU_BYTES,
            "registered packetization exceeds path MTU")
    require(CHUNK_BYTES + SSBURST_HEADER_BYTES + UDP_HEADER_BYTES
            + IPV4_HEADER_BYTES == FULL_IPV4_PACKET_BYTES,
            "registered full IPv4 packet arithmetic drifted")
    design = config.get("design")
    require(type(design) is dict and design.get("fps") == FPS
            and design.get("frames_per_block") == FRAMES_PER_BLOCK
            and design.get("blocks_per_cell") == BLOCKS_PER_CELL
            and design.get("frames_per_cell") == FRAMES_PER_CELL
            and design.get("tensors_per_policy_decision") ==
                TENSORS_PER_POLICY_DECISION
            and tuple(design.get("reward_then_held_role_order", ())) ==
                REWARD_THEN_HELD_ROLES
            and tuple(design.get("profiles", ())) == CONTRAST_PROFILE_IDS
            and tuple(tuple(v) for v in design.get("fit_permutations", ())) ==
                FIT_PERMUTATIONS
            and tuple(tuple(v) for v in design.get("validation_permutations", ())) ==
                VALIDATION_PERMUTATIONS
            and design.get("cell_order_seed") == CELL_ORDER_SEED
            and design.get("future_epoch_lead_s") == FUTURE_EPOCH_LEAD_S
            and design.get("payload_seed") == PAYLOAD_SEED
            and design.get("sample_period_s") == SAMPLE_PERIOD_S
            and design.get("sender_arm_timeout_s") == SENDER_ARM_TIMEOUT_S
            and design.get("expected_cells") == EXPECTED_CELLS
            and design.get("expected_tensor_frames") == EXPECTED_TENSOR_FRAMES
            and design.get("expected_policy_decisions") ==
                EXPECTED_POLICY_DECISIONS,
            "12-cell FIT/VALIDATION design drifted")

    require(config.get("paths") == {"output_root": OUTPUT_ROOT_RELPATH},
            "create-only output root drifted")
    require(config.get("authorization") == {
        "schema": "scenesense.ue_mcs_backlog_run4_calibration_authorization.v2",
        "stage": "run4_physical_queue_calibration_v2",
        "token": "AUTHORIZE_RUN4_PHYSICAL_QUEUE_CALIBRATION_V2",
        "maximum_grant_lifetime_seconds": MAX_GRANT_LIFETIME_SECONDS,
        "consumption_directory": ".authorization_consumed",
    }, "authorization contract drifted")
    require(config.get("failure_policy") == {
        "empty_ttracer_is_failure": True,
        "sender_socket_drop_is_failure": True,
        "sender_schema_mismatch_is_failure": True,
        "sender_schedule_lag_over_one_period_is_failure": True,
        "receiver_schema_mismatch_is_failure": True,
        "actuator_skip_or_clamp_is_failure": True,
        "actuator_error_is_failure": True,
        "restore_or_teardown_note_is_failure": True,
        "non_cold_final_state_is_failure": True,
        "cell_retry_in_place_allowed": False,
        "live_launch_in_unit_tests_allowed": False,
    }, "failure policy drifted")
    return config



def verify_inherited_sources(repo_root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in INHERITED_SOURCE_PINS.items():
        path = repo_root / relative
        require(path.is_file(), f"inherited source missing: {relative}")
        digest = sha256_file(path)
        require(digest == expected, f"inherited source drifted: {relative}")
        observed[relative] = digest
    base = repo_root / BASE_CONFIG_RELPATH
    require(base.is_file() and sha256_file(base) == BASE_CONFIG_SHA256,
            "base runtime config drifted")
    return {"verified": True, "files": observed,
            "base_config_sha256": BASE_CONFIG_SHA256}


def _add_inventory_file(files: dict[str, dict[str, Any]], repo_root: Path,
                        relative: str, *, expected_sha256: str | None = None) -> None:
    path = repo_root / relative
    require(path.is_file(), f"execution source missing: {relative}")
    digest = sha256_file(path)
    if expected_sha256:
        require(digest == expected_sha256,
                f"execution source digest drifted: {relative}")
    files[relative] = {"size_bytes": path.stat().st_size, "sha256": digest}


def _runtime_python_identity(executable: str) -> dict[str, Any]:
    path = Path(executable).resolve(strict=True)
    completed = subprocess.run(
        [str(path), "-c", "import platform,sys; print(sys.version); "
         "print(platform.python_implementation())"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    require(completed.returncode == 0,
            f"cannot identify Python runtime {path}: {completed.stderr.strip()}")
    lines = completed.stdout.splitlines()
    require(len(lines) == 2, "Python runtime identity response drifted")
    return {"resolved_path": str(path), "sha256": sha256_file(path),
            "version": lines[0], "implementation": lines[1]}


def source_inventory(repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind every transitive file that can affect this live execution.

    This is intentionally broader than the import graph: launcher scripts,
    container mounts, radio binaries, profile authority, payload authority and
    the exact Python binaries are execution inputs even when Python never
    imports them.
    """
    files: dict[str, dict[str, Any]] = {}
    for relative in (*PACKAGE_SOURCE_RELPATHS, *INHERITED_SOURCE_PINS,
                     *getattr(__import__(
                         "rl_agent.ue_mcs_backlog_near_capacity_v1.capacity_runner",
                         fromlist=["TRANSITIVE_REPO_SOURCES"]),
                       "TRANSITIVE_REPO_SOURCES")):
        _add_inventory_file(files, repo_root, relative)
    _add_inventory_file(files, repo_root, BASE_CONFIG_RELPATH,
                        expected_sha256=BASE_CONFIG_SHA256)
    _add_inventory_file(files, repo_root, PROFILE_BINDING_RELPATH,
                        expected_sha256=PROFILE_BINDING_SHA256)
    _add_inventory_file(files, repo_root, PROFILE_TRACE_RELPATH,
                        expected_sha256=PROFILE_TRACE_SHA256)

    # Exact radio/runtime pins, including built binaries.
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB
    for pin in RB.PINS.values():
        expected = str(pin.get("sha256", "")) or None
        _add_inventory_file(files, repo_root, str(pin["path"]),
                            expected_sha256=expected)

    # Files mounted by the CN5G compose file are runtime inputs too.
    for relative in (
        "scripts/config.env",
        "OAI/oai-cn5g/database/oai_db.sql",
        "OAI/oai-cn5g/healthscripts/mysql-healthcheck.sh",
        "OAI/oai-cn5g/conf/sip.conf",
        "OAI/oai-cn5g/conf/users.conf",
        "OAI/oai-cn5g/conf/config.yaml",
        "rl_agent/splitfusion_phase14b_corrected_four_profile_replay_v1.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/production_scene_evidence.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/fit_scene_provider.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/held_payload.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/quality_adapter.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/scientific_basis.py",
        "rl_agent/splitfusion_hybrid_sac_v1/modeled_smoke_support.py",
    ):
        _add_inventory_file(files, repo_root, relative)

    # The authoritative profile verifier declares further provenance paths.
    profile_config = json.loads(
        (repo_root / PROFILE_BINDING_RELPATH).read_text(encoding="utf-8"))
    for value in profile_config.get("provenance", {}).values():
        if type(value) is dict and type(value.get("path")) is str:
            _add_inventory_file(files, repo_root, value["path"],
                                expected_sha256=value.get("sha256"))

    body = {"repo_files": dict(sorted(files.items())),
            "python_runtimes": {
                "current": _runtime_python_identity(sys.executable),
                "launcher": _runtime_python_identity("/usr/bin/python3"),
            }}
    return {**body, "inventory_sha256": canonical_sha256(body)}


def verify_amendment(repo_root: Path = ROOT) -> dict[str, Any]:
    amendment_path = repo_root / AMENDMENT_RELPATH
    manifest = amendment_path.parent / AM.MANIFEST_FILENAME
    terminal = amendment_path.parent / AM.TERMINAL_FILENAME
    require(sha256_file(amendment_path) == AMENDMENT_SHA256,
            "sealed amendment digest drifted")
    require(sha256_file(manifest) == AMENDMENT_MANIFEST_SHA256,
            "sealed amendment manifest digest drifted")
    require(sha256_file(terminal) == AMENDMENT_TERMINAL_SHA256,
            "sealed amendment terminal digest drifted")
    verified = AM.verify_amendment(amendment_path, repo_root=repo_root)
    require(verified.get("binding_verified") is True
            and verified.get("status") == AMENDMENT_STATUS
            and verified.get("original_capacity_qualification_overturned") is False
            and verified.get("original_tier_selection_qualified") is False,
            "robust amendment semantic verification failed")
    adoption = verified.get("adoption", {})
    require(adoption.get("selected_action_ids") == LEGACY_AMENDMENT_ACTION_IDS
            and adoption.get("selected_payload_bytes") ==
                LEGACY_AMENDMENT_PAYLOAD_BYTES
            and adoption.get("selected_offered_mbps") ==
                LEGACY_AMENDMENT_OFFERED_MBPS
            and adoption.get("selected_family") == "AE64"
            and adoption.get("selected_quantizer") == "UINT8"
            and adoption.get("perception_endorsement") is False,
            "robust amendment selected-action identity drifted")
    return verified


def effective_runtime_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Open the pinned live config and apply only registered additive values."""
    config = load_config(path)
    base_path = ROOT / config["base_runtime"]["config_relative_path"]
    require(sha256_file(base_path) == BASE_CONFIG_SHA256,
            "base runtime config changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    effective = copy.deepcopy(base)
    effective["schema"] = CONFIG_SCHEMA
    effective["experiment_id"] = PACKAGE_ID
    effective["claim_boundary"] = CLAIM_BOUNDARY
    effective["paths"]["output_root"] = config["paths"]["output_root"]
    effective["campaign"]["cell_order_seed"] = CELL_ORDER_SEED
    effective["campaign"]["payload_seed"] = int(
        config["design"]["payload_seed"])
    effective["campaign"]["sample_period_s"] = float(
        config["design"]["sample_period_s"])
    effective["authorization"] = copy.deepcopy(config["authorization"])
    effective["run4_calibration"] = copy.deepcopy(config)
    return effective


@dataclass(frozen=True)
class Tier:
    tier: str
    action_id: int
    mode_id: int
    family: str
    quantizer: str
    q_e4: int
    profile_id: str
    payload_bytes: int
    offered_mbps: float

    @property
    def chunks_per_frame(self) -> int:
        return math.ceil(self.payload_bytes / CHUNK_BYTES)

    @property
    def last_chunk_payload_bytes(self) -> int:
        return self.payload_bytes - (self.chunks_per_frame - 1) * CHUNK_BYTES

    def to_json(self) -> dict[str, Any]:
        return {"tier": self.tier, "action_id": self.action_id,
                "mode_id": self.mode_id, "family": self.family,
                "quantizer": self.quantizer, "q_e4": self.q_e4,
                "profile_id": self.profile_id,
                "payload_bytes": self.payload_bytes,
                "offered_mbps": self.offered_mbps,
                "chunks_per_frame": self.chunks_per_frame,
                "last_chunk_payload_bytes": self.last_chunk_payload_bytes}


def registered_tiers() -> tuple[Tier, ...]:
    return tuple(Tier(tier, EXPECTED_ACTION_IDS[tier], EXPECTED_MODE_IDS[tier],
                      EXPECTED_FAMILIES[tier], EXPECTED_QUANTIZERS[tier],
                      EXPECTED_Q_E4[tier], EXPECTED_PROFILE_IDS[tier],
                      EXPECTED_PAYLOAD_BYTES[tier], EXPECTED_OFFERED_MBPS[tier])
                 for tier in TIER_ORDER)


def _weighted_payload_quantile(rows: Sequence[tuple[int, float, str, str]],
                               quantile: Decimal
                               ) -> tuple[int, str, str]:
    require(rows and Decimal(0) <= quantile <= Decimal(1),
            "invalid payload quantile request")
    ordered = sorted(rows, key=lambda row: (row[0], row[2]))
    total = sum((Decimal(str(row[1])) for row in ordered), Decimal(0))
    threshold = quantile * total
    cumulative = Decimal(0)
    for payload, weight, sample_id, row_sha in ordered:
        require(payload > 0 and math.isfinite(weight) and weight > 0.0,
                "invalid production payload observation")
        cumulative += Decimal(str(weight))
        if cumulative >= threshold:
            return payload, sample_id, row_sha
    return ordered[-1][0], ordered[-1][2], ordered[-1][3]


def verify_tier_design(repo_root: Path = ROOT, *, rederive: bool = True
                       ) -> dict[str, Any]:
    """Verify the prospective byte tiers against train-only production evidence.

    The values are production ``total_transmitted_bytes`` coordinates used only
    to select constant calibration load magnitudes.  They are not claims that
    1200-byte calibration UDP packetization has production PDCP/RLC overhead.
    """
    path = repo_root / TIER_DESIGN_RELPATH
    require(path.is_file() and sha256_file(path) == TIER_DESIGN_SHA256,
            "prospective tier-design digest drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("status") == "PROSPECTIVE_REVIEWED_NOT_EXECUTED"
            and value.get("payload_coordinate") ==
                "production_total_transmitted_bytes_replayed_as_calibration_application_bytes"
            and tuple(value.get("production_support_bytes", ())) ==
                PRODUCTION_PAYLOAD_SUPPORT_BYTES,
            "tier-design scope or coordinate drifted")
    require(value.get("train_inventory_sha256") ==
        "1d0d102fafb8797fd4b69e8524dd301229ddbc6f9bc76072048545c1d022ebbd"
        and value.get("production_verification_sha256") ==
        "f1635d5dca7a65eaf0ef6732b1249d35ad04e9c6374bb8d623d2b658d7e1ac47",
        "production-scene authority binding drifted")
    tiers = value.get("tiers")
    require(type(tiers) is dict and set(tiers) == set(TIER_ORDER),
            "tier-design rows drifted")
    for tier in registered_tiers():
        row = tiers[tier.tier]
        require(row.get("action_id") == tier.action_id
                and row.get("mode_id") == tier.mode_id
                and row.get("family") == tier.family
                and row.get("quantizer") == tier.quantizer
                and row.get("q_e4") == tier.q_e4
                and row.get("profile_id") == tier.profile_id
                and row.get("payload_bytes") ==
                    tier.payload_bytes,
                f"{tier.tier} prospective tier identity drifted")
    require(EXPECTED_PAYLOAD_BYTES["low"] <= PRODUCTION_PAYLOAD_SUPPORT_BYTES[0]
            and EXPECTED_PAYLOAD_BYTES["high"] >=
                PRODUCTION_PAYLOAD_SUPPORT_BYTES[1]
            and EXPECTED_PAYLOAD_BYTES["low"] <
                EXPECTED_PAYLOAD_BYTES["medium"] <
                EXPECTED_PAYLOAD_BYTES["high"],
            "prospective tiers do not bracket production support")

    derivation: dict[str, Any] | None = None
    if rederive:
        from rl_agent.splitfusion_hybrid_sac_run4_v1.production_scene_evidence import (
            load_production_scene_evidence,
        )
        evidence = load_production_scene_evidence(repository_root=repo_root)
        try:
            require(evidence.inventory.canonical_sha256 ==
                    value["train_inventory_sha256"]
                    and evidence.verification.canonical_sha256 ==
                    value["production_verification_sha256"],
                    "loaded production-scene authority drifted")
            observed: dict[str, Any] = {}
            for tier in registered_tiers():
                rows: list[tuple[int, float, str, str]] = []
                for record in evidence.inventory.records:
                    selection = record.selection
                    result = evidence.surface.query_fit_q_e4(
                        selection.sample_id, tier.mode_id, tier.q_e4)
                    endpoint = result.hidden.endpoint_evidence[0]
                    rows.append((
                        int(result.policy.payload.total_transmitted_bytes),
                        float(selection.sampling_weight), selection.sample_id,
                        endpoint.row_sha256,
                    ))
                q = Decimal(str(tiers[tier.tier]["source_quantile"]))
                selected = _weighted_payload_quantile(rows, q)
                row = tiers[tier.tier]
                require(len(rows) == value["train_scene_count"] == 391
                        and min(v[0] for v in rows) ==
                            row["observed_min_bytes"]
                        and max(v[0] for v in rows) ==
                            row["observed_max_bytes"]
                        and selected == (
                            row["payload_bytes"],
                            row["selected_sample_id"], row["selected_row_sha256"]),
                        f"{tier.tier} production payload derivation drifted")
                observed[tier.tier] = {
                    "selected_total_transmitted_bytes": selected[0],
                    "selected_sample_id": selected[1],
                    "selected_row_sha256": selected[2],
                }
            derivation = observed
        finally:
            evidence.close()
    return {"verified": True, "tier_design_sha256": TIER_DESIGN_SHA256,
            "production_support_bracketed": True,
            "production_byte_coordinate": "total_transmitted_bytes",
            "calibration_byte_domain":
                "CALIBRATION_UDP_1200B_CHUNKS_REQUIRES_TYPED_CONVERTER",
            "rederived": derivation}


@dataclass(frozen=True)
class Block:
    block_index: int
    tier: str
    action_id: int
    payload_bytes: int
    chunks_per_frame: int
    frames: int
    first_frame_index: int
    port: int

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Cell:
    cell_id: str
    profile_id: str
    permutation_index: int
    partition: str
    sequence: tuple[str, ...]
    blocks: tuple[Block, ...]
    run_index: int

    @property
    def transitions(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.sequence, self.sequence[1:]))

    def to_json(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "profile_id": self.profile_id,
                "permutation_index": self.permutation_index,
                "partition": self.partition, "sequence": list(self.sequence),
                "transitions": [f"{a}->{b}" for a, b in self.transitions],
                "blocks": [block.to_json() for block in self.blocks],
                "run_index": self.run_index,
                "frames_per_block": FRAMES_PER_BLOCK,
                "frames_total": FRAMES_PER_CELL}


def build_cell_plan(*, ports: Mapping[str, int],
                    seed: int = CELL_ORDER_SEED) -> tuple[Cell, ...]:
    tiers = {row.tier: row for row in registered_tiers()}
    cells: list[Cell] = []
    for profile in CONTRAST_PROFILE_IDS:
        for index, (sequence, partition) in enumerate(zip(PERMUTATIONS, PARTITIONS)):
            blocks = tuple(Block(
                block_index=position, tier=tier,
                action_id=tiers[tier].action_id,
                payload_bytes=tiers[tier].payload_bytes,
                chunks_per_frame=tiers[tier].chunks_per_frame,
                frames=FRAMES_PER_BLOCK,
                first_frame_index=position * FRAMES_PER_BLOCK,
                port=int(ports[tier]),
            ) for position, tier in enumerate(sequence))
            cells.append(Cell(
                cell_id=f"{profile.lower()}__perm{index}__{partition.lower()}",
                profile_id=profile, permutation_index=index,
                partition=partition, sequence=sequence, blocks=blocks,
                run_index=-1))
    random.Random(seed).shuffle(cells)
    return tuple(Cell(cell.cell_id, cell.profile_id, cell.permutation_index,
                      cell.partition, cell.sequence, cell.blocks, run_index)
                 for run_index, cell in enumerate(cells))


def audit_cell_plan(cells: Sequence[Cell]) -> dict[str, Any]:
    partition_counts: dict[str, Counter[str]] = {
        profile: Counter() for profile in CONTRAST_PROFILE_IDS}
    position_counts: dict[str, Counter[tuple[str, int]]] = {
        profile: Counter() for profile in CONTRAST_PROFILE_IDS}
    transition_counts: dict[str, Counter[str]] = {
        profile: Counter() for profile in CONTRAST_PROFILE_IDS}
    for cell in cells:
        partition_counts[cell.profile_id][cell.partition] += 1
        for position, tier in enumerate(cell.sequence):
            position_counts[cell.profile_id][(tier, position)] += 1
        for before, after in cell.transitions:
            transition_counts[cell.profile_id][f"{before}->{after}"] += 1
    fit_transitions = {"low->medium", "medium->high", "high->low"}
    validation_transitions = {"high->medium", "medium->low", "low->high"}
    result = {
        "cells": len(cells),
        "tensor_frames": len(cells) * FRAMES_PER_CELL,
        "policy_decisions": (len(cells) * FRAMES_PER_CELL
                             // TENSORS_PER_POLICY_DECISION),
        "whole_cell_partition": True,
        "partition_counts_per_profile": {
            key: dict(value) for key, value in partition_counts.items()},
        "position_counts_per_profile": {
            key: {f"{tier}@{pos}": count for (tier, pos), count in value.items()}
            for key, value in position_counts.items()},
        "transition_counts_per_profile": {
            key: dict(value) for key, value in transition_counts.items()},
        "fit_transition_set": sorted(fit_transitions),
        "validation_transition_set": sorted(validation_transitions),
        "transition_sets_disjoint": fit_transitions.isdisjoint(validation_transitions),
        "all_packetization_mtu_safe": all(
            CHUNK_BYTES + SSBURST_HEADER_BYTES + UDP_HEADER_BYTES + IPV4_HEADER_BYTES <= PATH_MTU_BYTES
            and block.chunks_per_frame <= 1_024
            for cell in cells for block in cell.blocks),
        "d2_pairing_exact": (
            FRAMES_PER_BLOCK % TENSORS_PER_POLICY_DECISION == 0
            and FRAMES_PER_CELL % TENSORS_PER_POLICY_DECISION == 0
            and EXPECTED_TENSOR_FRAMES == len(cells) * FRAMES_PER_CELL
            and EXPECTED_POLICY_DECISIONS ==
                len(cells) * FRAMES_PER_CELL // TENSORS_PER_POLICY_DECISION
        ),
        "production_payload_support_bracketed": (
            EXPECTED_PAYLOAD_BYTES["low"] <=
                PRODUCTION_PAYLOAD_SUPPORT_BYTES[0]
            and EXPECTED_PAYLOAD_BYTES["high"] >=
                PRODUCTION_PAYLOAD_SUPPORT_BYTES[1]
        ),
    }
    result["registered_design"] = (
        result["cells"] == EXPECTED_CELLS
        and result["tensor_frames"] == EXPECTED_TENSOR_FRAMES
        and result["policy_decisions"] == EXPECTED_POLICY_DECISIONS
        and result["transition_sets_disjoint"]
        and result["all_packetization_mtu_safe"]
        and result["d2_pairing_exact"]
        and result["production_payload_support_bracketed"]
        and all(value == {FIT: 3, VALIDATION: 3}
                for value in result["partition_counts_per_profile"].values())
        and all(len(value) == 9 and set(value.values()) == {2}
                for value in result["position_counts_per_profile"].values())
    )
    return result


@dataclass(frozen=True)
class VerifiedNetworkProfile:
    profile_id: str
    trace_id: str
    seed: int
    trace_sha256: str
    binding_sha256: str
    used_prefix_sha256: str
    samples: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        targets = [float(row["target_snr_db"]) for row in self.samples]
        return {"profile_id": self.profile_id, "trace_id": self.trace_id,
                "seed": self.seed, "trace_sha256": self.trace_sha256,
                "binding_sha256": self.binding_sha256,
                "used_prefix_sha256": self.used_prefix_sha256,
                "samples": len(self.samples),
                "target_snr_db_min": min(targets),
                "target_snr_db_max": max(targets)}


def resolve_profiles(sample_count: int = FRAMES_PER_CELL,
                     repo_root: Path = ROOT) -> tuple[VerifiedNetworkProfile, ...]:
    """Open and verify the exact two 450-row RF prefixes used by this run.

    The whole binding/trace files are hash-pinned.  The authoritative Phase-14B
    generator reconstruction verifies the accepted trace, then this function
    independently enforces every retained CSV field and records a digest over
    exactly the rows the calibration will consume.
    """
    require(sample_count == FRAMES_PER_CELL == 450,
            "Run-4 calibration requires the exact 450-row profile prefix")
    binding_path = repo_root / PROFILE_BINDING_RELPATH
    trace_path = repo_root / PROFILE_TRACE_RELPATH
    require(sha256_file(binding_path) == PROFILE_BINDING_SHA256,
            "authoritative profile binding drifted")
    require(sha256_file(trace_path) == PROFILE_TRACE_SHA256,
            "authoritative trace table drifted")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(binding.get("schema") ==
            "scenesense.splitfusion_phase14b_corrected_four_profile_replay_config.v1",
            "authoritative profile binding schema drifted")
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as authority
    prepared = {row["profile_id"]: row
                for row in authority.prepare_frozen_profiles(binding)}
    registered = {row["profile_id"]: row for row in binding["profiles"]}
    csv_rows: dict[str, list[dict[str, str]]] = {
        profile: [] for profile in CONTRAST_PROFILE_IDS}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS,
                "authoritative trace CSV schema drifted")
        for row in reader:
            if row["profile_id"] in csv_rows:
                csv_rows[row["profile_id"]].append(dict(row))

    result: list[VerifiedNetworkProfile] = []
    for profile_id in CONTRAST_PROFILE_IDS:
        frozen = registered[profile_id]
        generated = prepared[profile_id]
        rows = csv_rows[profile_id]
        require(len(rows) >= sample_count,
                f"{profile_id}: trace prefix is too short")
        prefix = rows[:sample_count]
        normalized: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        for step, (row, authoritative) in enumerate(
                zip(prefix, generated["prefix"][:sample_count])):
            state_index = int(row["state_index"])
            target = float(row["target_snr_db"])
            require(row["profile_id"] == profile_id
                    and row["trace_id"] == frozen["trace_id"]
                    and int(row["seed"]) == int(frozen["seed"])
                    and row["value_semantics"] == TRACE_VALUE_SEMANTICS
                    and int(row["step_index"]) == step
                    and Decimal(row["interval_start_s"]) ==
                        Decimal(step) / Decimal(10)
                    and Decimal(row["interval_end_s"]) ==
                        Decimal(step + 1) / Decimal(10)
                    and state_index == int(authoritative[0])
                    and row["state"] == TRACE_STATE_ORDER[state_index]
                    and abs(target - float(authoritative[1])) <= 5.1e-7,
                    f"{profile_id}: trace semantics drifted at step {step}")
            normalized.append({field: row[field] for field in TRACE_FIELDS})
            samples.append({
                "step_index": step, "state_index": state_index,
                "state": row["state"], "target_snr_db": target,
                "interval_start_s": row["interval_start_s"],
                "interval_end_s": row["interval_end_s"],
            })
        result.append(VerifiedNetworkProfile(
            profile_id=profile_id, trace_id=str(frozen["trace_id"]),
            seed=int(frozen["seed"]), trace_sha256=str(frozen["trace_sha256"]),
            binding_sha256=PROFILE_BINDING_SHA256,
            used_prefix_sha256=canonical_sha256(normalized),
            samples=tuple(samples),
        ))
    require({row.profile_id for row in result} == set(CONTRAST_PROFILE_IDS),
            "contrast profile resolution drifted")
    return tuple(result)
