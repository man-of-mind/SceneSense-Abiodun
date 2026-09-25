"""Frozen design contract for the Run-4 near-capacity UE MCS/backlog sweep.

Run 3 (``ue_mcs_backlog_calibration_v1``) returned ``INCONCLUSIVE`` and its own
report asked, in §10, for exactly this follow-up: offered loads that *bracket*
the channel-dependent capacity, instead of one tier far below it and two far
above.  This module pins that follow-up.

Everything whose semantics Run 3 already qualified -- tracer schemas, the
round-0 grant rule, the chunking constant, the profile/trace binding, the
clock-bridge contract -- is **imported** from the Run-3 contract rather than
restated, so the two runs cannot silently diverge.  Only the *design* is new.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_mcs_backlog_calibration_v1 import contract as V3

# --------------------------------------------------------------------------
# Imported, unchanged, from the qualified Run-3 contract.
# --------------------------------------------------------------------------

ACTION_CATALOG_RELPATH = V3.ACTION_CATALOG_RELPATH
PROFILE_BINDING_RELPATH = V3.PROFILE_BINDING_RELPATH
PROFILE_TRACE_RELPATH = V3.PROFILE_TRACE_RELPATH
PRODUCTION_RECEIVER_RELPATH = V3.PRODUCTION_RECEIVER_RELPATH
OAI_CITATIONS = V3.OAI_CITATIONS

CHUNK_BYTES = V3.CHUNK_BYTES            # 60_000, production sender default
FPS = V3.FPS                            # 10.0
FRAMES_PER_BLOCK = V3.FRAMES_PER_BLOCK  # 150 decisions = 15 s
BLOCKS_PER_CELL = V3.BLOCKS_PER_CELL    # 3
FRAMES_PER_CELL = V3.FRAMES_PER_CELL    # 450
TIER_ORDER = V3.TIER_ORDER              # ("low", "medium", "high")
CONTRAST_PROFILE_IDS = V3.CONTRAST_PROFILE_IDS
TRANSIENT_DECISIONS = V3.TRANSIENT_DECISIONS      # 30
STEADY_STATE_DECISIONS = V3.STEADY_STATE_DECISIONS  # 80
AGENT_PATH_BUDGET_MS = V3.AGENT_PATH_BUDGET_MS      # 170.0
MCS_MISSING = V3.MCS_MISSING
NEW_DATA_HARQ_ROUND = V3.NEW_DATA_HARQ_ROUND
sha256_file = V3.sha256_file
resolve_profiles = V3.resolve_profiles
NetworkProfile = V3.NetworkProfile

ContractError = V3.ContractError

CONTRACT_ID = "ue_mcs_backlog_near_capacity_v1"
CONTRACT_VERSION = 1
CLAIM_BOUNDARY = (
    "BOUNDED_NEAR_CAPACITY_QUEUE_TRANSITION_CHARACTERIZATION_WITH_BLOCKED_"
    "VALIDATION_NOT_KERNEL_ACCEPTANCE_AND_NOT_PUBLICATION_EVIDENCE"
)

# --------------------------------------------------------------------------
# Immutable action authority.
#
# Bound by *action id* against the frozen catalogue, then cross-checked against
# independently recorded profile ids, payload sizes, chunk counts and the shared
# decoder checkpoint.  Any single mismatch refuses the run rather than guessing
# which side is right.
# --------------------------------------------------------------------------

ACTION_CATALOG_JSON_SHA256 = (
    "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3")
ACTION_CATALOG_CSV_RELPATH = (
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.csv")
ACTION_CATALOG_CSV_SHA256 = (
    "0512cb39982178e8c7c96a65ed26e272b3aa3a5aec0020a8dd9cf1cdb6696fbb")
SHARED_CHECKPOINT_SHA256 = (
    "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271")

#: tier -> (action_id, profile_id, payload_bytes, chunks, offered Mbps @10 fps)
EXPECTED_TIERS: Mapping[str, dict[str, Any]] = {
    "low":    {"action_id": 70, "profile_id": "split_ae32_uint4_q9000",
               "payload_bytes": 28_109,  "chunks": 1, "offered_mbps": 2.25},
    "medium": {"action_id": 69, "profile_id": "split_ae32_uint4_q7000",
               "payload_bytes": 81_087,  "chunks": 2, "offered_mbps": 6.49},
    "high":   {"action_id": 68, "profile_id": "split_ae32_uint4_q5000",
               "payload_bytes": 129_707, "chunks": 3, "offered_mbps": 10.38},
}

# --------------------------------------------------------------------------
# Measured capacity anchors.
#
# Derived read-only from the Run-3 evidence (``decisions_v2.csv``) by taking
# service = (payload enqueued - backlog delta) / dt over intervals where the
# queue was already deep (>2 MB) and below 85% of the 49,984,583 B ceiling, so
# the UE was continuously backlogged and the measured rate is the *capacity*,
# not the offered rate.
#
# These are pre-registered EXPECTATIONS, not gates.  They are stated in advance
# precisely so the outcome cannot be rationalised afterwards.
# --------------------------------------------------------------------------

MEASURED_CAPACITY_MBPS: Mapping[str, dict[str, float]] = {
    "ADVERSE_STABLE":   {"p10": 10.49, "p50": 12.05, "p90": 22.57, "n": 1200},
    "FAVORABLE_STABLE": {"p10": 18.11, "p50": 42.13, "p90": 46.50, "n": 998},
}
CAPACITY_SOURCE = (
    "rl_agent/experiments/ue_mcs_backlog_calibration_v1/20260924_131015/"
    "decisions_v2.csv (read-only re-derivation; the protected file is unchanged)")

#: The "~6 Mbps uplink" figure quoted by the Run-3 preregistration is
#: contradicted by Run-3's own data: FAVORABLE_STABLE sustained a 21.08 Mbps
#: offered load with median backlog 0 and 0.960 complete reassembly, which a
#: 6 Mbps link cannot do.  Recorded so no later reader re-derives tier
#: placement from the stale number.
SUPERSEDED_CAPACITY_ASSUMPTION_MBPS = 6.0


def expected_load_ratios() -> dict[str, dict[str, float]]:
    """Offered load as a fraction of each channel's measured P50 capacity."""
    return {
        profile: {tier: spec["offered_mbps"] / MEASURED_CAPACITY_MBPS[profile]["p50"]
                  for tier, spec in EXPECTED_TIERS.items()}
        for profile in CONTRAST_PROFILE_IDS
    }


# --------------------------------------------------------------------------
# Experimental design: all six orderings, blocked fit/validation.
# --------------------------------------------------------------------------

#: The three cyclic orders, predeclared as FIT.
FIT_PERMUTATIONS: tuple[tuple[str, ...], ...] = (
    ("low", "medium", "high"),
    ("medium", "high", "low"),
    ("high", "low", "medium"),
)
#: Their exact reverses, predeclared as held-out VALIDATION.
VALIDATION_PERMUTATIONS: tuple[tuple[str, ...], ...] = tuple(
    tuple(reversed(order)) for order in FIT_PERMUTATIONS)

PERMUTATIONS: tuple[tuple[str, ...], ...] = FIT_PERMUTATIONS + VALIDATION_PERMUTATIONS

FIT = "FIT"
VALIDATION = "VALIDATION"

PARTITIONS: tuple[str, ...] = (
    (FIT,) * len(FIT_PERMUTATIONS) + (VALIDATION,) * len(VALIDATION_PERMUTATIONS))

#: Pinned once, recorded, never re-drawn.  Shuffles only the *execution* order
#: of the 12 cells so monotone host/radio drift cannot masquerade as a design
#: effect.  Cells never move between partitions.
CELL_ORDER_SEED = 2026092401

EXPECTED_CELLS = len(PERMUTATIONS) * len(CONTRAST_PROFILE_IDS)          # 12
EXPECTED_DECISIONS = EXPECTED_CELLS * FRAMES_PER_CELL                   # 5400


def permutation_label(order: Sequence[str]) -> str:
    return "-".join(name[0].upper() for name in order)


@dataclass(frozen=True)
class LoadTier:
    """One offered-load tier, pinned to a catalogue action."""

    tier: str
    action_id: int
    profile_id: str
    payload_bytes: int
    family: str
    q_e4: int
    checkpoint_sha256: str

    @property
    def chunks_per_frame(self) -> int:
        return max(1, -(-self.payload_bytes // CHUNK_BYTES))

    @property
    def offered_mbps(self) -> float:
        return self.payload_bytes * 8 * FPS / 1e6

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier, "action_id": self.action_id,
            "profile_id": self.profile_id, "payload_bytes": self.payload_bytes,
            "family": self.family, "q_e4": self.q_e4,
            "checkpoint_sha256": self.checkpoint_sha256,
            "chunks_per_frame": self.chunks_per_frame,
            "offered_mbps": self.offered_mbps,
        }


def resolve_load_tiers(repo_root: Path) -> tuple[LoadTier, ...]:
    """Resolve the three near-capacity tiers, refusing on any mismatch.

    Checks, in order: catalogue file digests, action id presence, profile id,
    payload bytes, derived chunk count, offered rate, shared checkpoint, and
    strictly increasing payload.  Every one of these is a stop condition.
    """
    catalog_path = repo_root / ACTION_CATALOG_RELPATH
    csv_path = repo_root / ACTION_CATALOG_CSV_RELPATH
    for path, expected, label in (
            (catalog_path, ACTION_CATALOG_JSON_SHA256, "catalogue JSON"),
            (csv_path, ACTION_CATALOG_CSV_SHA256, "catalogue CSV")):
        if not path.is_file():
            raise ContractError(f"{label} missing at {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ContractError(
                f"{label} sha256 {observed} != pinned {expected}; refusing")

    data = json.loads(catalog_path.read_text())
    by_action = {int(entry["action_id"]): entry for entry in data["profiles"]}

    tiers: list[LoadTier] = []
    for tier, spec in EXPECTED_TIERS.items():
        action_id = int(spec["action_id"])
        entry = by_action.get(action_id)
        if entry is None:
            raise ContractError(
                f"tier {tier!r}: action {action_id} is not in the catalogue")
        if entry["profile_id"] != spec["profile_id"]:
            raise ContractError(
                f"tier {tier!r}: action {action_id} is {entry['profile_id']!r} in "
                f"the catalogue but was registered as {spec['profile_id']!r}; "
                f"refusing to guess which is intended")
        payload_bytes = int(entry["payload"]["zstd_median_bytes"])
        if payload_bytes != int(spec["payload_bytes"]):
            raise ContractError(
                f"tier {tier!r}: catalogue payload {payload_bytes} B != registered "
                f"{spec['payload_bytes']} B")
        checkpoint = str(entry["checkpoint_sha256"])
        if checkpoint != SHARED_CHECKPOINT_SHA256:
            raise ContractError(
                f"tier {tier!r}: checkpoint {checkpoint} != pinned shared "
                f"checkpoint {SHARED_CHECKPOINT_SHA256}")
        resolved = LoadTier(
            tier=tier, action_id=action_id, profile_id=entry["profile_id"],
            payload_bytes=payload_bytes, family=str(entry["family"]),
            q_e4=int(entry["q_e4"]), checkpoint_sha256=checkpoint)
        if resolved.chunks_per_frame != int(spec["chunks"]):
            raise ContractError(
                f"tier {tier!r}: {resolved.chunks_per_frame} chunks at "
                f"{CHUNK_BYTES} B != registered {spec['chunks']}")
        if abs(resolved.offered_mbps - float(spec["offered_mbps"])) > 0.01:
            raise ContractError(
                f"tier {tier!r}: offered {resolved.offered_mbps:.4f} Mbps != "
                f"registered {spec['offered_mbps']} Mbps")
        tiers.append(resolved)

    ordered = sorted(tiers, key=lambda item: item.payload_bytes)
    if [item.tier for item in ordered] != list(TIER_ORDER):
        raise ContractError(
            "tiers are not strictly increasing in payload: "
            + ", ".join(f"{i.tier}={i.payload_bytes}" for i in ordered))
    return tuple(ordered)


@dataclass(frozen=True)
class Block:
    """One constant-load block inside a cell."""

    block_index: int
    tier: str
    action_id: int
    payload_bytes: int
    chunks_per_frame: int
    frames: int
    first_frame_index: int
    port: int

    def to_json(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index, "tier": self.tier,
            "action_id": self.action_id, "payload_bytes": self.payload_bytes,
            "chunks_per_frame": self.chunks_per_frame, "frames": self.frames,
            "first_frame_index": self.first_frame_index, "port": self.port,
        }


@dataclass(frozen=True)
class Cell:
    """One (payload permutation, channel) cell, with its frozen partition."""

    cell_id: str
    profile_id: str
    permutation_index: int
    permutation_label: str
    partition: str
    sequence: tuple[str, ...]
    blocks: tuple[Block, ...]
    run_index: int

    @property
    def transitions(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.sequence, self.sequence[1:]))

    def to_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id, "profile_id": self.profile_id,
            "permutation_index": self.permutation_index,
            "permutation_label": self.permutation_label,
            "partition": self.partition,
            "sequence": list(self.sequence),
            "transitions": ["->".join(pair) for pair in self.transitions],
            "blocks": [block.to_json() for block in self.blocks],
            "run_index": self.run_index,
            "frames_per_block": FRAMES_PER_BLOCK,
            "frames_total": FRAMES_PER_CELL,
        }


def build_cell_plan(
    tiers: Sequence[LoadTier], *, ports: Mapping[str, int],
    seed: int = CELL_ORDER_SEED,
) -> tuple[Cell, ...]:
    """The 6-permutation x 2-channel plan in a seeded execution order."""
    by_tier = {tier.tier: tier for tier in tiers}
    missing = [name for name in TIER_ORDER if name not in by_tier]
    if missing:
        raise ContractError(f"cell plan needs tiers {missing}")

    cells: list[Cell] = []
    for profile_id in CONTRAST_PROFILE_IDS:
        for index, (sequence, partition) in enumerate(
                zip(PERMUTATIONS, PARTITIONS)):
            blocks = []
            for position, tier_name in enumerate(sequence):
                tier = by_tier[tier_name]
                blocks.append(Block(
                    block_index=position, tier=tier_name,
                    action_id=tier.action_id, payload_bytes=tier.payload_bytes,
                    chunks_per_frame=tier.chunks_per_frame,
                    frames=FRAMES_PER_BLOCK,
                    first_frame_index=position * FRAMES_PER_BLOCK,
                    port=int(ports[tier_name])))
            label = permutation_label(sequence)
            cells.append(Cell(
                cell_id=f"{profile_id.lower()}__perm{index}_{label.lower()}"
                        f"__{partition.lower()}",
                profile_id=profile_id, permutation_index=index,
                permutation_label=label, partition=partition,
                sequence=tuple(sequence), blocks=tuple(blocks), run_index=-1))

    rng = random.Random(seed)
    rng.shuffle(cells)
    return tuple(
        Cell(cell_id=c.cell_id, profile_id=c.profile_id,
             permutation_index=c.permutation_index,
             permutation_label=c.permutation_label, partition=c.partition,
             sequence=c.sequence, blocks=c.blocks, run_index=i)
        for i, c in enumerate(cells))


def audit_cell_plan(cells: Sequence[Cell]) -> dict[str, Any]:
    """Prove the plan is the balanced, partitioned design it claims to be."""
    from collections import Counter

    profiles = list(CONTRAST_PROFILE_IDS)
    position_counts = {p: Counter() for p in profiles}
    transition_counts = {p: Counter() for p in profiles}
    partition_counts = {p: Counter() for p in profiles}
    fit_transitions = {p: Counter() for p in profiles}
    val_transitions = {p: Counter() for p in profiles}
    fit_positions = {p: Counter() for p in profiles}
    val_positions = {p: Counter() for p in profiles}
    tier_counts: Counter = Counter()

    for cell in cells:
        for position, tier in enumerate(cell.sequence):
            position_counts[cell.profile_id][(tier, position)] += 1
            tier_counts[tier] += 1
            target = (fit_positions if cell.partition == FIT else val_positions)
            target[cell.profile_id][(tier, position)] += 1
        for pair in cell.transitions:
            key = "->".join(pair)
            transition_counts[cell.profile_id][key] += 1
            target = (fit_transitions if cell.partition == FIT else val_transitions)
            target[cell.profile_id][key] += 1
        partition_counts[cell.profile_id][cell.partition] += 1

    all_transitions = {f"{a}->{b}" for a in TIER_ORDER for b in TIER_ORDER if a != b}

    by_key = {(c.profile_id, c.permutation_index): c for c in cells}
    reverse_ok = True
    for profile in profiles:
        for index in range(len(FIT_PERMUTATIONS)):
            fit_cell = by_key.get((profile, index))
            val_cell = by_key.get((profile, index + len(FIT_PERMUTATIONS)))
            if (fit_cell is None or val_cell is None
                    or val_cell.sequence != tuple(reversed(fit_cell.sequence))
                    or fit_cell.partition != FIT or val_cell.partition != VALIDATION):
                reverse_ok = False

    def latin(counter: Counter) -> bool:
        return len(counter) == 9 and set(counter.values()) == {1}

    return {
        "cells": len(cells),
        "expected_cells": EXPECTED_CELLS,
        "decisions": len(cells) * FRAMES_PER_CELL,
        "expected_decisions": EXPECTED_DECISIONS,
        "tier_block_counts": dict(tier_counts),
        "position_counts_per_channel": {
            p: {f"{t}@{i}": n for (t, i), n in sorted(c.items())}
            for p, c in position_counts.items()},
        "position_balanced_per_channel": {
            p: len(c) == 9 and set(c.values()) == {2}
            for p, c in position_counts.items()},
        "transition_counts_per_channel": {
            p: dict(c) for p, c in transition_counts.items()},
        "all_six_transitions_balanced_per_channel": {
            p: set(c) == all_transitions and set(c.values()) == {2}
            for p, c in transition_counts.items()},
        "partition_counts_per_channel": {
            p: dict(c) for p, c in partition_counts.items()},
        "partition_balanced_per_channel": {
            p: c.get(FIT) == 3 and c.get(VALIDATION) == 3
            for p, c in partition_counts.items()},
        "validation_reverses_fit": reverse_ok,
        "fit_is_latin_square_per_channel": {
            p: latin(c) for p, c in fit_positions.items()},
        "validation_is_latin_square_per_channel": {
            p: latin(c) for p, c in val_positions.items()},
        "fit_transition_counts_per_channel": {
            p: dict(c) for p, c in fit_transitions.items()},
        "validation_transition_counts_per_channel": {
            p: dict(c) for p, c in val_transitions.items()},
        # Declared in advance, not discovered later: the cyclic orders and their
        # reverses partition the six ordered transitions into two disjoint sets,
        # so validation is an *extrapolation across transition direction*.
        "fit_validation_transitions_disjoint_per_channel": {
            p: not (set(fit_transitions[p]) & set(val_transitions[p]))
            for p in profiles},
        "load_is_within_cell": all(
            len(set(c.sequence)) == len(TIER_ORDER) for c in cells),
        "every_cell_has_a_partition": all(
            c.partition in (FIT, VALIDATION) for c in cells),
    }


def plan_is_registered_design(audit: Mapping[str, Any]) -> bool:
    """Single predicate the runner requires before any radio is started."""
    return bool(
        audit["cells"] == audit["expected_cells"]
        and audit["decisions"] == audit["expected_decisions"]
        and audit["load_is_within_cell"]
        and audit["every_cell_has_a_partition"]
        and audit["validation_reverses_fit"]
        and all(audit["position_balanced_per_channel"].values())
        and all(audit["all_six_transitions_balanced_per_channel"].values())
        and all(audit["partition_balanced_per_channel"].values())
        and all(audit["fit_is_latin_square_per_channel"].values())
        and all(audit["validation_is_latin_square_per_channel"].values())
    )


def resolved_source_hashes(repo_root: Path) -> dict[str, str]:
    """Content hashes of every authoritative source this contract resolved."""
    return {
        relpath: sha256_file(repo_root / relpath)
        for relpath in (
            ACTION_CATALOG_RELPATH, ACTION_CATALOG_CSV_RELPATH,
            PROFILE_BINDING_RELPATH, PROFILE_TRACE_RELPATH,
            PRODUCTION_RECEIVER_RELPATH,
            "rl_agent/ue_mcs_backlog_calibration_v1/contract.py",
            "rl_agent/ue_mcs_backlog_calibration_v1/runner.py",
            "rl_agent/ue_mcs_backlog_calibration_v1/tagged_sender.py",
            "rl_agent/ue_mcs_backlog_calibration_v1/decision_join.py",
        )
    }
