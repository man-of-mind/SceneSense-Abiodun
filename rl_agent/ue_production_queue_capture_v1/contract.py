#!/usr/bin/env python3
"""Frozen prospective contract for the production-domain UE queue/transport capture.

This package is additive.  It never imports, mutates or supersedes
``ue_mcs_backlog_run4_calibration_v1``; that package and its preserved
1,200-byte calibration design remain exactly as they are.

Claim boundary: this capture qualifies the **UE queue and production-domain
transport** process only.  It is not perception evidence, not policy
validation, and not deployment authorization.  The three byte-load roles are
``EMERGENCY_ONLY`` catalogue entries chosen for their byte loads; their
perception behaviour is explicitly not endorsed.

Importing this module performs no I/O beyond reading pinned files when
``verify_authorities`` is called explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RELPATH = "rl_agent/ue_production_queue_capture_v1"
PACKAGE_ID = "ue_production_queue_capture_v1"
SCHEMA_VERSION = 1

CLAIM_BOUNDARY = (
    "PRODUCTION_DOMAIN_UE_QUEUE_AND_TRANSPORT_CALIBRATION_ONLY__"
    "NOT_PERCEPTION_ENDORSEMENT__NOT_POLICY_VALIDATION__"
    "NOT_DEPLOYMENT_AUTHORIZATION"
)

# Frozen before collection.  Reinterpreting any threshold below after seeing
# results is a contract violation.
PROSPECTIVE_FREEZE = "FROZEN_BEFORE_COLLECTION"


class ContractError(RuntimeError):
    """A frozen identity or invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1.  Production packetization identity (verified, never assumed)
# ---------------------------------------------------------------------------
# The deployed SplitFusion uplink uses the `!IHH` chunk header over 12,500-byte
# UDP application datagrams with no retransmission.  The historical 60,000-byte
# default in ``phase2_map_sharing/transport.py`` is NOT the live binding and is
# explicitly rejected here.
UDP_CHUNK_BYTES_INCLUDING_HEADER = 12_500
UDP_CHUNK_HEADER_STRUCT = "!IHH"
UDP_CHUNK_HEADER_BYTES = 8
UDP_PAYLOAD_BYTES_PER_DATAGRAM = 12_492
RETRANSMISSION = False
HISTORICAL_CHUNK_BYTES_REJECTED = 60_000

# A full 12,500-byte UDP application datagram plus the 8-byte UDP header and
# the 20-byte IPv4 header is 12,528 bytes, far above the 1,500-byte path MTU.
# IPv4 fragmentation therefore HAPPENS on this path.  It is observed per cell
# from the kernel counters; it is never assumed away.
UDP_HEADER_BYTES = 8
IPV4_HEADER_BYTES = 20
PATH_MTU_BYTES = 1_500
FULL_IPV4_PACKET_BYTES = (
    UDP_CHUNK_BYTES_INCLUDING_HEADER + UDP_HEADER_BYTES + IPV4_HEADER_BYTES
)
IPV4_FRAGMENTATION_EXPECTED = True
FRAGMENTS_PER_FULL_DATAGRAM = math.ceil(
    (UDP_CHUNK_BYTES_INCLUDING_HEADER + UDP_HEADER_BYTES)
    / (PATH_MTU_BYTES - IPV4_HEADER_BYTES)
)

PACKETIZATION_IDENTITY = "PRODUCTION_SFD1_IHH_12500_NO_RETRANSMISSION"


def datagram_count(total_transmitted_bytes: int) -> int:
    require(
        type(total_transmitted_bytes) is int and total_transmitted_bytes > 0,
        "total_transmitted_bytes must be a positive exact integer",
    )
    return math.ceil(total_transmitted_bytes / UDP_PAYLOAD_BYTES_PER_DATAGRAM)


def udp_application_bytes(total_transmitted_bytes: int) -> int:
    return total_transmitted_bytes + UDP_CHUNK_HEADER_BYTES * datagram_count(
        total_transmitted_bytes
    )


# ---------------------------------------------------------------------------
# 2.  Byte-load roles, bound to the registered payload authority
# ---------------------------------------------------------------------------
# The payload coordinate is `total_transmitted_bytes` (SFD1 inner payload plus
# the SFD1 outer envelope).  That is exactly what the `!IHH` chunker chunks.
PAYLOAD_COORDINATE = "total_transmitted_bytes"

PAYLOAD_AUTHORITY_RELPATH = (
    "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
    "20260918_exact_continuous_q_grid_a1b_full/quality_rows.sqlite3"
)
PAYLOAD_AUTHORITY_MANIFEST_RELPATH = (
    "experiments/splitfusion_hybrid_sac_quality_grid_v1/"
    "20260918_exact_continuous_q_grid_a1b_full/run_manifest.json"
)
PAYLOAD_AUTHORITY_MANIFEST_SHA256 = (
    "8869d085e585bd4cb4d8231abdfea4b358095578e85a3cbfb811bf073bfc9b36"
)

MODE_COUNT = 12
CALIBRATION_MODE_IDS = (11, 6, 6)


@dataclass(frozen=True, slots=True)
class TierSpec:
    """One byte-load role.  Identity is the registered anchor, not a label."""

    tier: str
    action_id: int
    mode_id: int
    q_e4: int
    family: str
    quantizer: str
    profile_id: str
    # Distributional anchors observed in the registered authority; the live
    # schedule draws exact per-frame rows rather than using these medians.
    median_total_transmitted_bytes: int
    role: str
    inside_run4_modeled_payload_support: bool


# Run-4 modeled payload support, carried forward from the frozen Run-4
# calibration design (`production_payload_support_bytes`).
RUN4_MODELED_PAYLOAD_SUPPORT_BYTES = (6_423, 427_605)

# Retained empirical bootstrap uncertainty set for adverse capacity.  This is
# explicitly NOT a population confidence interval and the original tier
# selection it came from was REFUSED.  It is used only to place byte roles.
RETAINED_ADVERSE_CAPACITY_MBPS = {
    "lower": 28.512,
    "point": 30.576,
    "upper": 31.584,
    "interpretation":
        "RETAINED_EMPIRICAL_BOOTSTRAP_UNCERTAINTY_SET_NOT_POPULATION_"
        "CONFIDENCE_INTERVAL",
    "original_capacity_qualification": "REFUSED",
    "original_tier_selection_qualified": False,
}

TIERS: tuple[TierSpec, ...] = (
    TierSpec(
        tier="floor", action_id=71, mode_id=11, q_e4=9800,
        family="AE32", quantizer="UINT4", profile_id="split_ae32_uint4_q9800",
        median_total_transmitted_bytes=6_459,
        role="LOWER_PAYLOAD_BOUNDARY_OF_RUN4_MODELED_SUPPORT",
        inside_run4_modeled_payload_support=True,
    ),
    TierSpec(
        tier="knee", action_id=39, mode_id=6, q_e4=7000,
        family="AE64", quantizer="UINT8", profile_id="split_ae64_uint8_q7000",
        median_total_transmitted_bytes=374_531,
        role="INSIDE_RETAINED_ADVERSE_CAPACITY_UNCERTAINTY_SET",
        inside_run4_modeled_payload_support=True,
    ),
    TierSpec(
        tier="guard", action_id=38, mode_id=6, q_e4=5000,
        family="AE64", quantizer="UINT8", profile_id="split_ae64_uint8_q5000",
        median_total_transmitted_bytes=619_825,
        role="ABOVE_CAPACITY_AND_ABOVE_RUN4_EXECUTION_CEILING",
        inside_run4_modeled_payload_support=False,
    ),
)
TIER_NAMES = tuple(spec.tier for spec in TIERS)

# Every byte role is an EMERGENCY_ONLY catalogue entry.  This capture is a
# byte-only queue/transport design and endorses none of their perception
# behaviour.
CATALOGUE_CONTRACT_TIER = "EMERGENCY_ONLY"
PERCEPTION_ENDORSEMENT = False
BYTE_ONLY_QUEUE_DESIGN = True

# The guard role sits outside the Run-4 modeled payload support on purpose.
# It brackets the failure regime.  A fitted model must refuse rather than
# extrapolate a production prediction beyond the measured support.
GUARD_IS_BRACKET_NOT_OPERATING_POINT = True


def tier_by_name(name: str) -> TierSpec:
    for spec in TIERS:
        if spec.tier == name:
            return spec
    raise ContractError(f"unknown tier {name!r}")


def action_id_for(mode_id: int, q_e4: int) -> int:
    """Reproduce the registered SFD1 anchor action-id arithmetic."""
    catalog_q = (0, 3000, 5000, 7000, 9000, 9800)
    require(0 <= mode_id < MODE_COUNT, f"invalid mode_id {mode_id!r}")
    require(q_e4 in catalog_q, f"q_e4 {q_e4!r} is not a catalog anchor")
    return mode_id * len(catalog_q) + catalog_q.index(q_e4)


# ---------------------------------------------------------------------------
# 3.  Capture design
# ---------------------------------------------------------------------------
FPS = 10.0
STEP_PERIOD_NS = 100_000_000
FRAMES_PER_BLOCK = 150
BLOCKS_PER_CELL = 3
FRAMES_PER_CELL = FRAMES_PER_BLOCK * BLOCKS_PER_CELL          # 450
EXPECTED_CELLS = 12
EXPECTED_RAW_FRAMES = FRAMES_PER_CELL * EXPECTED_CELLS        # 5,400

# 10-Hz tensor frames, 5-Hz controller.  One cycle is
#   decision frame t (even)  ->  held frame t+1 (same mode and q)  ->  successor t+2
DURATION_STEPS = 2
DURATION_NS = DURATION_STEPS * STEP_PERIOD_NS
PRIMARY_CYCLES_PER_CELL = 224          # last even index has no in-cell successor
EXPECTED_PRIMARY_CYCLES = PRIMARY_CYCLES_PER_CELL * EXPECTED_CELLS   # 2,688
UNCLOSED_PER_CELL = 1

PROFILES = ("FAVORABLE_STABLE", "ADVERSE_STABLE")
FIT = "FIT"
VALIDATION = "VALIDATION"

# Counterbalanced tier orders.  Whole cells are FIT or VALIDATION; the two
# partitions use disjoint permutations AND disjoint scene splits.
FIT_PERMUTATIONS = (
    ("floor", "knee", "guard"),
    ("knee", "guard", "floor"),
    ("guard", "floor", "knee"),
)
VALIDATION_PERMUTATIONS = (
    ("guard", "knee", "floor"),
    ("floor", "guard", "knee"),
    ("knee", "floor", "guard"),
)
FIT_SCENE_SPLIT = "fit"
VALIDATION_SCENE_SPLIT = "held_scene"

CELL_ORDER_SEED = 2026092801
PAYLOAD_SCHEDULE_SEED = 2026092801

# Profile identity is audit-only.  It is never a feature, fit key or back-off
# key, and never reaches the actor.
PROFILE_IDENTITY_ROLE = "AUDIT_ONLY_NEVER_A_POLICY_FEATURE"


@dataclass(frozen=True, slots=True)
class Block:
    block_index: int
    tier: str
    action_id: int
    mode_id: int
    q_e4: int
    frames: int

    def to_json(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index, "tier": self.tier,
            "action_id": self.action_id, "mode_id": self.mode_id,
            "q_e4": self.q_e4, "frames": self.frames,
        }


@dataclass(frozen=True, slots=True)
class Cell:
    run_index: int
    cell_id: str
    profile_id: str
    partition: str
    permutation_index: int
    scene_split: str
    blocks: tuple[Block, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index, "cell_id": self.cell_id,
            "profile_id": self.profile_id, "partition": self.partition,
            "permutation_index": self.permutation_index,
            "scene_split": self.scene_split,
            "blocks": [block.to_json() for block in self.blocks],
        }


def _permutation_table() -> tuple[tuple[str, int, tuple[str, ...], str], ...]:
    rows: list[tuple[str, int, tuple[str, ...], str]] = []
    for index, order in enumerate(FIT_PERMUTATIONS):
        rows.append((FIT, index, order, FIT_SCENE_SPLIT))
    for index, order in enumerate(VALIDATION_PERMUTATIONS):
        rows.append((VALIDATION, len(FIT_PERMUTATIONS) + index, order,
                     VALIDATION_SCENE_SPLIT))
    return tuple(rows)


def planned_cells() -> tuple[Cell, ...]:
    """The frozen 12-cell plan: 2 profiles x 6 counterbalanced permutations."""
    cells: list[Cell] = []
    run_index = 0
    for profile_id in PROFILES:
        for partition, permutation_index, order, scene_split in _permutation_table():
            blocks = tuple(
                Block(
                    block_index=block_index, tier=tier,
                    action_id=tier_by_name(tier).action_id,
                    mode_id=tier_by_name(tier).mode_id,
                    q_e4=tier_by_name(tier).q_e4,
                    frames=FRAMES_PER_BLOCK,
                )
                for block_index, tier in enumerate(order)
            )
            cells.append(Cell(
                run_index=run_index,
                cell_id=f"{profile_id.lower()}__perm{permutation_index}",
                profile_id=profile_id, partition=partition,
                permutation_index=permutation_index,
                scene_split=scene_split, blocks=blocks,
            ))
            run_index += 1
    require(len(cells) == EXPECTED_CELLS, "planned cell count drifted")
    return tuple(cells)


def primary_cycle_indices(
    frames: int = FRAMES_PER_CELL,
) -> tuple[tuple[int, int, int], ...]:
    """Closed (decision, held, successor) frame triples inside one cell."""
    return tuple(
        (start, start + 1, start + DURATION_STEPS)
        for start in range(0, frames, DURATION_STEPS)
        if start + DURATION_STEPS < frames
    )


def unclosed_decision_indices(frames: int = FRAMES_PER_CELL) -> tuple[int, ...]:
    closed = {triple[0] for triple in primary_cycle_indices(frames)}
    return tuple(index for index in range(0, frames, DURATION_STEPS)
                 if index not in closed)


# ---------------------------------------------------------------------------
# 4.  Same-domain closure seal
# ---------------------------------------------------------------------------
# Collection happens directly in the production byte domain, so the old
# cross-domain ByteDomainProofV1 is replaced by a same-domain closure seal.
# Every one of these must be present and exact for a cell to be accepted.
CLOSURE_SEAL_REQUIREMENTS: tuple[str, ...] = (
    "EXACT_FRAME_ACTION_PAYLOAD_IDENTITY",
    "SENDER_FIRST_AND_LAST_SOCKET_HANDOFF_TIMESTAMPS",
    "UE_PDCP_INGRESS",
    "UE_RLC_INGRESS_AND_DEQUEUE",
    "CAUSAL_PRE_ACTION_RLC_BACKLOG",
    "CAUSAL_PREVIOUS_NEW_DATA_ROUND0_TABLE0_UL_MCS",
    "RECEIVER_FIRST_AND_LAST_DATAGRAM_TIMESTAMPS",
    "RECEIVER_COMPLETE_REASSEMBLY_TIMESTAMP",
    "OBSERVED_IP_FRAGMENT_ACCOUNTING",
    "ZERO_UNEXPLAINED_SENDER_TO_PDCP_TO_RLC_RESIDUAL",
    "EXACTLY_ONE_TERMINAL_OUTCOME_PER_SENT_FRAME",
)
CLOSURE_SEAL_ID = "PRODUCTION_SAME_DOMAIN_CLOSURE_SEAL_V1"
REPLACES_CROSS_DOMAIN_PROOF = "ByteDomainProofV1"

# Causal freshness: one tensor period.  Missing or stale means the registered
# external fallback, never a fabricated zero.
FRESHNESS_MAX_AGE_NS = STEP_PERIOD_NS
FRESHNESS_FALLBACK = "EXTERNAL_FALLBACK_NO_ZERO_FILL"
UE_MCS_TABLE = 0
UE_MCS_ROUND = 0
UE_MCS_MIN = 0
UE_MCS_MAX = 28

# Model inputs.  Anything else about the radio is audit-only.
PRIMARY_MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "pre_action_rlc_backlog_bytes",
    "prior_ul_mcs",
    "decision_frame_total_transmitted_bytes",
    "held_frame_total_transmitted_bytes",
)
AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "profile_id", "target_snr_db", "commanded_noise_power_db",
    "gnb_mcs", "achieved_pusch_snr_db", "post_decision_ul_mcs",
    "tier", "family", "quantizer",
)


# ---------------------------------------------------------------------------
# 5.  Latency boundary and the disjoint replacement of the 288 uplink term
# ---------------------------------------------------------------------------
# Both segments meet at exactly one endpoint: the LAST UDP socket handoff.
UE_ACTION_PATH_BOUNDARY = (
    "SEVEN_CHANNEL_CONSTRUCTION_START__TO__LAST_UDP_SOCKET_HANDOFF"
)
PRODUCTION_TRANSPORT_BOUNDARY = (
    "LAST_UDP_SOCKET_HANDOFF__TO__COMPLETE_RECEIVER_REASSEMBLY"
)
SHARED_ENDPOINT = "LAST_UDP_SOCKET_HANDOFF"

# The old 288 component is `application_feature_uplink_ms`, defined in
# `splitfusion_timing_diagnostic_v1` as
#     edge complete reassembly  -  UE FIRST send
# It is REPLACED, never added.  Disjointness is provable because the retained
# live evidence satisfies, exactly and per row,
#     application_feature_uplink_ms
#         == ue_send_loop_ms + post_send_to_reassembly_ms
# so moving `ue_send_loop_ms` into the UE action path and replacing the
# remainder with the new conditional transport model double-counts nothing.
REPLACED_288_COMPONENT = "application_feature_uplink_ms"
REPLACED_288_DEFINITION = "EDGE_COMPLETE_REASSEMBLY_MINUS_UE_FIRST_SEND"
DISJOINTNESS_IDENTITY = (
    "application_feature_uplink_ms == ue_send_loop_ms "
    "+ post_send_to_reassembly_ms"
)
DISJOINTNESS_EVIDENCE_RELPATH = (
    "experiments/splitfusion_timing_diagnostic_v1/"
    "20260909_live_carla_actions30_15_50_71_retry3/per_frame"
)
ADDING_BOTH_COMPONENTS_IS_FORBIDDEN = True

# Sender and receiver run on this one host, so both stamp the same
# ``time.monotonic_ns`` timeline.  That is what makes the interval between the
# last socket handoff and complete reassembly directly subtractable.
SINGLE_MONOTONIC_CLOCK = True

# A non-positive transport interval is physically impossible and means the two
# stamps raced inside one kernel delivery (observed on loopback for
# single-datagram frames, magnitude <= 0.19 ms).  Such a row is an
# instrumentation fault: it is EXCLUDED and counted, never clamped to zero and
# never silently dropped.
BOUNDARY_INVERSION_POLICY = "EXCLUDED_NOT_CLAMPED"
MAX_BOUNDARY_INVERSION_FRACTION = 0.01

REWARD_DEADLINE_MS = 170.0
REWARD_DEADLINE_NS = 170_000_000
REWARD_LATENCY_WEIGHT = 0.25
REGISTERED_FAILURE_REWARD = -1.0

# Anti-bias rule, frozen before collection.
INCOMPLETE_BY_DEADLINE_IS_FAILURE = True
SURVIVOR_ONLY_LATENCY_FITTING_IS_FORBIDDEN = True
INFRASTRUCTURE_FAULT_IS_EXCLUDED_NOT_CHARGED = True

TERMINAL_OUTCOMES: tuple[str, ...] = (
    "COMPLETE_WITHIN_DEADLINE",
    "COMPLETE_AFTER_DEADLINE",
    "INCOMPLETE_AT_DEADLINE",
    "NEVER_COMPLETED",
    "EXCLUDED_INFRASTRUCTURE_FAULT",
)
POLICY_CHARGEABLE_OUTCOMES: tuple[str, ...] = (
    "COMPLETE_WITHIN_DEADLINE",
    "COMPLETE_AFTER_DEADLINE",
    "INCOMPLETE_AT_DEADLINE",
    "NEVER_COMPLETED",
)
SUCCESS_OUTCOMES: tuple[str, ...] = ("COMPLETE_WITHIN_DEADLINE",)


def queue_next_backlog(
    current_bytes: int, measured_ingress_bytes: int, measured_service_bytes: int,
) -> int:
    """The registered conservation recurrence, exact integers only."""
    for name, value in (
        ("current", current_bytes), ("ingress", measured_ingress_bytes),
        ("service", measured_service_bytes),
    ):
        require(type(value) is int and value >= 0,
                f"{name} bytes must be a non-negative exact integer")
    return max(0, current_bytes + measured_ingress_bytes - measured_service_bytes)


# ---------------------------------------------------------------------------
# 6.  Frozen validation gates
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GateSpec:
    number: int
    key: str
    target: str
    thresholds: Mapping[str, Any]


GATES: tuple[GateSpec, ...] = (
    GateSpec(1, "COMPLETE_AND_SEALED_CAPTURE",
             "12 sealed cells, 5,400 exact raw frames, 2,688 closed d=2 cycles",
             {"cells": EXPECTED_CELLS, "raw_frames": EXPECTED_RAW_FRAMES,
              "primary_cycles": EXPECTED_PRIMARY_CYCLES}),
    GateSpec(2, "SAME_DOMAIN_CLOSURE_SEAL",
             "every closure-seal requirement present, zero unexplained residual",
             {"requirements": len(CLOSURE_SEAL_REQUIREMENTS),
              "max_unexplained_byte_residual": 0,
              "terminal_outcomes_per_sent_frame": 1}),
    GateSpec(3, "CAUSAL_INPUT_COVERAGE",
             "strictly-prior round-0/table-0 UE MCS and pre-enqueue backlog",
             {"ue_mcs_coverage": 1.0, "backlog_coverage": 1.0,
              "max_age_ms": 100.0, "ambiguity": 0}),
    GateSpec(4, "VALIDATION_NEXT_BACKLOG_ERROR",
             "held-out whole-cell d=2 next-backlog prediction",
             {"max_nmae": 0.10, "min_improvement_over_persistence": 0.20}),
    GateSpec(5, "VALIDATION_TRANSPORT_LATENCY_ERROR",
             "held-out last-handoff to complete-reassembly latency",
             {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0}),
    GateSpec(6, "VALIDATION_DEADLINE_OUTCOME_CALIBRATION",
             "held-out deadline success on the FULL sent population",
             {"max_false_success_rate": 0.05, "max_brier": 0.15,
              "prediction_threshold": 0.5,
              "population": "ALL_SENT_FRAMES_NOT_SURVIVORS_ONLY"}),
    GateSpec(7, "MCS_NONHARM_AND_DIRECTION",
             "MCS model versus backlog-only comparator on identical rows",
             {"max_brier_degradation": 0.01,
              "direction": "higher MCS does not worsen delivery"}),
    GateSpec(8, "MONOTONICITY",
             "inside measured support only",
             {"max_violations": 0,
              "axes": "more bytes/backlog not better; higher MCS not worse"}),
    GateSpec(9, "BOUNDARY_INTEGRITY",
             "non-positive transport intervals excluded, never clamped",
             {"policy": BOUNDARY_INVERSION_POLICY,
              "max_inversion_fraction": MAX_BOUNDARY_INVERSION_FRACTION}),
    GateSpec(10, "NO_HIDDEN_INPUT",
             "no profile label, target SNR, gNB-only or future measurement",
             {"forbidden": list(AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS)}),
)

# Per-tier censoring is REPORTED, not gated, because the guard role is designed
# to saturate the queue.  A saturated guard block is the intended bracket, not
# a capture defect.  Gates 4-8 are evaluated on the non-censored measured
# support and the fitted model refuses predictions outside it.
CENSORING_IS_REPORTED_PER_TIER_NOT_GATED = True
SATURATION_EXPECTED_FOR = ("guard",)


# ---------------------------------------------------------------------------
# 7.  Authority pins
# ---------------------------------------------------------------------------
AUTHORITY_PINS: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/contract.py":
        "2f501833a07a6bda2b126c6da95cfc432b64aca4c1234a0498ba4c3c3523a520",
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/schema.py":
        "a9d133ce98d5f29b6830d1a5f4034128d0d685c995d1e932917e684622e92d3b",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py":
        "c3aaf4450deb80e3427972d745bd171cacda517864860f1106ce9d140dc4c18b",
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.csv":
        "0512cb39982178e8c7c96a65ed26e272b3aa3a5aec0020a8dd9cf1cdb6696fbb",
    "rl_agent/ue_mcs_backlog_robust_bracket_v1/sealed/"
    "PROSPECTIVE_CAPACITY_AMENDMENT.json":
        "fa6ef35816c61442aa7c287c31652d06f09fb1c0f8f914d467200629114c6bf0",
    "phase2_map_sharing/transport.py":
        "c8d8d0b253356c11776e9c35b7d6b1bef009bfc980e4cecfcbba84ae95734a6e",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json":
        "65fb3d1d3a92c53a6db40c3ba7dc6b4f74b3447428d28d5c6ada0aeb69773659",
    PAYLOAD_AUTHORITY_MANIFEST_RELPATH: PAYLOAD_AUTHORITY_MANIFEST_SHA256,
}


def verify_authorities(repo_root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in AUTHORITY_PINS.items():
        path = repo_root / relative
        require(path.is_file(), f"authority missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"authority drifted: {relative}")
        observed[relative] = actual
    return {
        "verified": True, "source_sha256": observed,
        "authority_sha256": canonical_sha256(observed),
    }


def contract_document() -> dict[str, Any]:
    return {
        "package_id": PACKAGE_ID, "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY, "freeze": PROSPECTIVE_FREEZE,
        "packetization": {
            "identity": PACKETIZATION_IDENTITY,
            "chunk_bytes_including_header": UDP_CHUNK_BYTES_INCLUDING_HEADER,
            "chunk_header_struct": UDP_CHUNK_HEADER_STRUCT,
            "chunk_header_bytes": UDP_CHUNK_HEADER_BYTES,
            "payload_bytes_per_datagram": UDP_PAYLOAD_BYTES_PER_DATAGRAM,
            "retransmission": RETRANSMISSION,
            "historical_chunk_bytes_rejected": HISTORICAL_CHUNK_BYTES_REJECTED,
            "ipv4_fragmentation_expected": IPV4_FRAGMENTATION_EXPECTED,
            "fragments_per_full_datagram": FRAGMENTS_PER_FULL_DATAGRAM,
            "full_ipv4_packet_bytes": FULL_IPV4_PACKET_BYTES,
            "path_mtu_bytes": PATH_MTU_BYTES,
        },
        "payload_coordinate": PAYLOAD_COORDINATE,
        "payload_authority": {
            "relpath": PAYLOAD_AUTHORITY_RELPATH,
            "manifest_relpath": PAYLOAD_AUTHORITY_MANIFEST_RELPATH,
            "manifest_sha256": PAYLOAD_AUTHORITY_MANIFEST_SHA256,
        },
        "tiers": [
            {
                "tier": spec.tier, "action_id": spec.action_id,
                "mode_id": spec.mode_id, "q_e4": spec.q_e4,
                "family": spec.family, "quantizer": spec.quantizer,
                "profile_id": spec.profile_id, "role": spec.role,
                "median_total_transmitted_bytes":
                    spec.median_total_transmitted_bytes,
                "inside_run4_modeled_payload_support":
                    spec.inside_run4_modeled_payload_support,
            }
            for spec in TIERS
        ],
        "catalogue_contract_tier": CATALOGUE_CONTRACT_TIER,
        "perception_endorsement": PERCEPTION_ENDORSEMENT,
        "byte_only_queue_design": BYTE_ONLY_QUEUE_DESIGN,
        "run4_modeled_payload_support_bytes":
            list(RUN4_MODELED_PAYLOAD_SUPPORT_BYTES),
        "retained_adverse_capacity_mbps": dict(RETAINED_ADVERSE_CAPACITY_MBPS),
        "design": {
            "fps": FPS, "frames_per_block": FRAMES_PER_BLOCK,
            "blocks_per_cell": BLOCKS_PER_CELL,
            "frames_per_cell": FRAMES_PER_CELL,
            "expected_cells": EXPECTED_CELLS,
            "expected_raw_frames": EXPECTED_RAW_FRAMES,
            "duration_steps": DURATION_STEPS,
            "primary_cycles_per_cell": PRIMARY_CYCLES_PER_CELL,
            "expected_primary_cycles": EXPECTED_PRIMARY_CYCLES,
            "profiles": list(PROFILES),
            "fit_permutations": [list(value) for value in FIT_PERMUTATIONS],
            "validation_permutations":
                [list(value) for value in VALIDATION_PERMUTATIONS],
            "fit_scene_split": FIT_SCENE_SPLIT,
            "validation_scene_split": VALIDATION_SCENE_SPLIT,
            "cell_order_seed": CELL_ORDER_SEED,
            "payload_schedule_seed": PAYLOAD_SCHEDULE_SEED,
            "profile_identity_role": PROFILE_IDENTITY_ROLE,
        },
        "closure_seal": {
            "id": CLOSURE_SEAL_ID,
            "replaces": REPLACES_CROSS_DOMAIN_PROOF,
            "requirements": list(CLOSURE_SEAL_REQUIREMENTS),
        },
        "model_inputs": {
            "primary": list(PRIMARY_MODEL_INPUT_FIELDS),
            "audit_only": list(AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS),
            "freshness_max_age_ns": FRESHNESS_MAX_AGE_NS,
            "freshness_fallback": FRESHNESS_FALLBACK,
        },
        "latency": {
            "ue_action_path_boundary": UE_ACTION_PATH_BOUNDARY,
            "production_transport_boundary": PRODUCTION_TRANSPORT_BOUNDARY,
            "shared_endpoint": SHARED_ENDPOINT,
            "replaced_288_component": REPLACED_288_COMPONENT,
            "replaced_288_definition": REPLACED_288_DEFINITION,
            "disjointness_identity": DISJOINTNESS_IDENTITY,
            "adding_both_is_forbidden": ADDING_BOTH_COMPONENTS_IS_FORBIDDEN,
            "deadline_ms": REWARD_DEADLINE_MS,
            "latency_weight": REWARD_LATENCY_WEIGHT,
            "failure_reward": REGISTERED_FAILURE_REWARD,
            "incomplete_by_deadline_is_failure":
                INCOMPLETE_BY_DEADLINE_IS_FAILURE,
            "survivor_only_fitting_forbidden":
                SURVIVOR_ONLY_LATENCY_FITTING_IS_FORBIDDEN,
            "terminal_outcomes": list(TERMINAL_OUTCOMES),
            "single_monotonic_clock": SINGLE_MONOTONIC_CLOCK,
            "boundary_inversion_policy": BOUNDARY_INVERSION_POLICY,
            "max_boundary_inversion_fraction":
                MAX_BOUNDARY_INVERSION_FRACTION,
        },
        "gates": [
            {"number": gate.number, "key": gate.key, "target": gate.target,
             "thresholds": dict(gate.thresholds)}
            for gate in GATES
        ],
        "censoring_reported_per_tier_not_gated":
            CENSORING_IS_REPORTED_PER_TIER_NOT_GATED,
        "saturation_expected_for": list(SATURATION_EXPECTED_FOR),
        "authority_pins": dict(AUTHORITY_PINS),
    }


CONTRACT_SHA256 = canonical_sha256(contract_document())
