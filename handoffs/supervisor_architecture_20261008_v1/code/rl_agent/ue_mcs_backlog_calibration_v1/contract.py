#!/usr/bin/env python3
"""Frozen contract for the UE-local [previous UL MCS, pre-enqueue backlog] study.

The proposed runtime state is exactly two numbers, both observable at the UE
without any gNB telemetry:

``previous_ul_mcs``
    The most recent *strictly prior* new-data uplink MCS the UE decoded from
    its own UL DCI. In this build the gNB selects it from averaged measured
    PUSCH SNR through ``get_mcs_from_SINRx10``
    (``gNB_scheduler_ulsch.c:2028``, gated by ``scenesense_use_sinr_mcs_policy``
    at ``:2027``), and it reaches the UE through standard DCI with no added
    controller. It is therefore a *delayed, quantized scheduler decision
    derived from gNB-measured uplink SNR* -- not a UE measurement of the
    channel.

``pre_enqueue_backlog_bytes``
    Raw UE RLC transmit-buffer occupancy in bytes, from the last
    ``NRUE_MAC_RLC_BUFFER_STATUS`` tick strictly before the tagged payload is
    handed to PDCP. Because the sample precedes the enqueue, the tagged
    payload's own bytes are structurally excluded. This is demand/queue
    pressure, not a physical-channel quantity.

Observation *age* is carried as external validity evidence only. It is
deliberately **not** a policy feature.

Importing this module performs no I/O beyond reading the files it is explicitly
asked to resolve, samples no RNG, and touches no accelerator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT_ID = "ue_mcs_backlog_calibration_v1"
CONTRACT_VERSION = 1

# --------------------------------------------------------------------------
# Authoritative sources. Everything numeric is resolved from these, never
# retyped, and each is hashed into the run manifest.
# --------------------------------------------------------------------------

ACTION_CATALOG_RELPATH = (
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
)
PROFILE_BINDING_RELPATH = (
    "rl_agent/configs/splitfusion_phase14b_corrected_four_profile_replay_v1.json"
)
PROFILE_TRACE_RELPATH = (
    "rl_agent/experiments/network_profile_design_v2/20260822_route_b_v2/traces.csv"
)
#: The frozen SSBURST wire contract is imported from the production receiver
#: rather than restated, so the sender cannot drift from it.
PRODUCTION_RECEIVER_RELPATH = "rl_agent/ue_n3_structured_udp_receiver.py"

#: OAI provenance, cited so the report can be re-checked against the tree.
OAI_CITATIONS: Mapping[str, str] = {
    "sinr_policy_gate": (
        "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c:2027 "
        "(scenesense_use_sinr_mcs_policy, reads SCENESENSE_MCS_POLICY)"
    ),
    "sinr_mcs_selection": (
        "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c:2028 "
        "(get_mcs_from_SINRx10(mcs_table, pusch_pc.avg_snr * 10, nrOfLayers))"
    ),
    "sinr_mcs_table_support": (
        "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c:238-241 "
        "(get_mcs_from_SINRx10 supports MCS table 0 only)"
    ),
    "pdcp_enqueue_t0": (
        "OAI/openairinterface5g/openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.c:941-944 "
        "(NR_PDCP_TX_SDU, CLOCK_MONOTONIC, SDU entering PDCP TX on UE UL)"
    ),
    "rlc_buffer_status": (
        "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c:1462 "
        "(NRUE_MAC_RLC_BUFFER_STATUS, pre-multiplex raw bytes per LCID)"
    ),
}

# --------------------------------------------------------------------------
# Tracer schemas. Headers are compared exactly.
# --------------------------------------------------------------------------

DCI_GRANT_HEADER = (
    "time", "direction", "dci_format", "rnti_type", "rnti", "dci_frame",
    "dci_slot", "sched_frame", "sched_slot", "mcs", "mcs_table", "rb_start",
    "rb_size", "start_symbol", "nr_symbols", "tbs", "harq_pid", "ndi", "rv",
    "round", "qam_mod_order", "target_code_rate", "tpc", "n_cce", "N_cce",
)
RLC_BUFFER_HEADER = (
    "time", "rnti", "ue_id", "frame", "slot", "lcid", "lcgid",
    "bytes_in_buffer", "bj", "pbr", "priority",
)
PDCP_TX_SDU_HEADER = (
    "time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes",
)
GNB_MCS_DECISION_HEADER = (
    "time", "rnti", "frame", "slot", "sched_frame", "sched_slot",
    "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
    "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer",
    "sched_ul_bytes", "B", "min_rb", "available_rb_before",
    "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
    "force_ul_mcs",
)

#: ``NRUE_MAC_DCI_GRANT.direction`` value meaning uplink.
UL_DIRECTION = "1"

#: A new-data grant is HARQ round 0. Retransmission grants carry the MCS of the
#: original transmission and must never enter the policy feature.
NEW_DATA_HARQ_ROUND = 0

#: Candidate external validity bounds.  The causal join itself is deliberately
#: threshold-free and retains the raw prior grant and its age.  Analysis may
#: project any of these bounds without destroying evidence.  No member is an
#: actor feature or an empirically established optimum.
MCS_VALIDITY_CANDIDATES_MS = (100.0, 150.0, 200.0, 250.0)

#: Backwards-compatible name used by the v1 preregistration and plots.  It is a
#: hypothesis to test, not a rule applied by the v2 join.
MCS_MAX_AGE_MS = 200.0

#: Sentinel for "no prior grant". Must stay distinguishable from MCS 0.
MCS_MISSING = None

#: Production chunking default, from carla_shaped_udp_burst_sender.py:47.
CHUNK_BYTES = 60_000

FPS = 10.0

# --------------------------------------------------------------------------
# Experimental design (amended 2026-09-24).
#
# The first plan held one payload constant for a whole cell, which made load a
# purely *between-cell* factor: any backlog difference was confounded with
# everything else that differed between two RAN instances, so backlog could not
# be causally attributed to load. That plan is withdrawn.
#
# Load is now a **within-cell** factor. Every cell runs all three tiers as
# back-to-back blocks on one continuous monotonic timeline, so each cell
# contains its own load transitions and each decision's backlog can be read
# against a load change that happened inside the same radio instance.
#
#   3 block orders x 2 contrasting channels x 2 repetitions = 12 cells
#
# The three orders are the cyclic Latin square of order 3, so within each
# (channel, repetition) stratum every tier appears exactly once in every block
# position. Repetition 1 runs the exact reverse of repetition 0's order, which
# counterbalances transition *direction*: across the two repetitions all six
# ordered tier transitions occur, equally often, in each channel.
# --------------------------------------------------------------------------

#: Decisions per block. 150 at 10 fps = 15 s, long enough to hold a transient
#: window and a disjoint steady-state window.
FRAMES_PER_BLOCK = 150
BLOCKS_PER_CELL = 3
FRAMES_PER_CELL = FRAMES_PER_BLOCK * BLOCKS_PER_CELL

TIER_ORDER = ("low", "medium", "high")

#: Cyclic Latin square of order 3. Position-balanced by construction.
BLOCK_ORDERS: tuple[tuple[str, ...], ...] = (
    ("low", "medium", "high"),
    ("medium", "high", "low"),
    ("high", "low", "medium"),
)

REPETITIONS = 2

#: The two contrasting registered channels. Both are *stable*, so the channel
#: is held roughly steady while load moves; that is what makes a within-cell
#: load transition interpretable.
CONTRAST_PROFILE_IDS = ("FAVORABLE_STABLE", "ADVERSE_STABLE")

#: All four registered profiles remain resolvable; only two are exercised.
PROFILE_IDS = ("FAVORABLE_STABLE", "MID_VARIABLE", "FADE_RECOVERY", "ADVERSE_STABLE")

#: Pre-registered analysis windows, in decisions from the block boundary.
#: Disjoint by construction: 30 + 80 <= 150.
TRANSIENT_DECISIONS = 30
STEADY_STATE_DECISIONS = 80


def block_sequence(order_index: int, repetition: int) -> tuple[str, ...]:
    """The tier sequence for one cell.

    Repetition 0 runs the Latin-square order; repetition 1 runs its exact
    reverse, so the direction of every transition is counterbalanced.
    """
    if not 0 <= order_index < len(BLOCK_ORDERS):
        raise ContractError(f"order_index {order_index} out of range")
    if not 0 <= repetition < REPETITIONS:
        raise ContractError(f"repetition {repetition} out of range")
    order = BLOCK_ORDERS[order_index]
    return order if repetition == 0 else tuple(reversed(order))

#: Transport portion of the 170 ms agent-path budget. Only the uplink transport
#: segment is charged here; perception time is not added, because no perception
#: model runs in this qualification and folding in an unrelated constant would
#: make the outcome a statement about that constant instead of about transport.
AGENT_PATH_BUDGET_MS = 170.0


class ContractError(ValueError):
    """A resolved value disagrees with its authoritative source."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Load tiers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadTier:
    """One offered-load tier, pinned to a catalogue action."""

    tier: str
    action_id: int
    profile_id: str
    payload_bytes: int
    family: str
    q_e4: int

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
            "chunks_per_frame": self.chunks_per_frame,
            "offered_mbps": self.offered_mbps,
        }


#: The remembered tier -> action identities. Reconciled against the catalogue by
#: :func:`resolve_load_tiers`, which refuses rather than guesses on mismatch.
EXPECTED_TIER_ACTIONS: Mapping[str, int] = {"low": 71, "medium": 50, "high": 30}

#: Independently remembered identities, asserted against the catalogue so a
#: silent catalogue re-freeze cannot quietly change what "low" means.
EXPECTED_TIER_PROFILE_IDS: Mapping[str, str] = {
    "low": "split_ae32_uint4_q9800",
    "medium": "split_ae64_uint4_q5000",
    "high": "split_ae128_uint4_q0000",
}


def resolve_load_tiers(repo_root: Path) -> tuple[LoadTier, ...]:
    """Resolve the three tiers from the frozen catalogue.

    Refuses if the remembered action ids do not reconcile exactly with the
    catalogue, or if the tiers are not strictly increasing in payload.
    """
    catalog_path = repo_root / ACTION_CATALOG_RELPATH
    if not catalog_path.is_file():
        raise ContractError(f"action catalogue missing at {catalog_path}")
    data = json.loads(catalog_path.read_text())
    by_action = {int(entry["action_id"]): entry for entry in data["profiles"]}

    tiers: list[LoadTier] = []
    for tier, action_id in EXPECTED_TIER_ACTIONS.items():
        entry = by_action.get(action_id)
        if entry is None:
            raise ContractError(
                f"tier {tier!r}: action {action_id} is not in the catalogue")
        expected_profile = EXPECTED_TIER_PROFILE_IDS[tier]
        if entry["profile_id"] != expected_profile:
            raise ContractError(
                f"tier {tier!r}: action {action_id} is {entry['profile_id']!r} in "
                f"the catalogue but was recorded as {expected_profile!r}; refusing "
                f"to guess which is intended")
        tiers.append(LoadTier(
            tier=tier, action_id=action_id, profile_id=entry["profile_id"],
            payload_bytes=int(entry["payload"]["zstd_median_bytes"]),
            family=str(entry["family"]), q_e4=int(entry["q_e4"]),
        ))
    ordered = sorted(tiers, key=lambda item: item.payload_bytes)
    if [item.tier for item in ordered] != ["low", "medium", "high"]:
        raise ContractError(
            "tiers are not strictly increasing in payload: "
            + ", ".join(f"{item.tier}={item.payload_bytes}" for item in ordered))
    return tuple(ordered)


# --------------------------------------------------------------------------
# Network profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkProfile:
    """One registered RF profile and the trace prefix that will be replayed."""

    profile_id: str
    trace_id: str
    seed: int
    trace_sha256: str
    samples: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        targets = [row["target_snr_db"] for row in self.samples]
        return {
            "profile_id": self.profile_id, "trace_id": self.trace_id,
            "seed": self.seed, "registered_trace_sha256": self.trace_sha256,
            "samples": len(self.samples),
            "target_snr_db_min": min(targets), "target_snr_db_max": max(targets),
        }


def resolve_profiles(repo_root: Path, sample_count: int) -> tuple[NetworkProfile, ...]:
    """Resolve the four registered profiles and their trace prefixes.

    Both the binding and the trace table are authoritative sources; neither is
    retyped. The prefix length is bounded, but every value inside it is the
    registered value.
    """
    binding_path = repo_root / PROFILE_BINDING_RELPATH
    trace_path = repo_root / PROFILE_TRACE_RELPATH
    for path in (binding_path, trace_path):
        if not path.is_file():
            raise ContractError(f"authoritative profile source missing: {path}")
    binding = json.loads(binding_path.read_text())
    registered = {entry["profile_id"]: entry for entry in binding["profiles"]}

    missing = [name for name in PROFILE_IDS if name not in registered]
    if missing:
        raise ContractError(f"registered binding lacks profiles {missing}")

    rows_by_profile: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_IDS}
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            name = record["profile_id"]
            if name in rows_by_profile:
                rows_by_profile[name].append({
                    "step_index": int(record["step_index"]),
                    "state": record["state"],
                    "target_snr_db": float(record["target_snr_db"]),
                    "trace_id": record["trace_id"],
                })

    out: list[NetworkProfile] = []
    for name in PROFILE_IDS:
        rows = sorted(rows_by_profile[name], key=lambda item: item["step_index"])
        if len(rows) < sample_count:
            raise ContractError(
                f"{name}: trace holds {len(rows)} samples, need {sample_count}")
        entry = registered[name]
        trace_ids = {row["trace_id"] for row in rows[:sample_count]}
        if trace_ids != {entry["trace_id"]}:
            raise ContractError(
                f"{name}: trace table reports {trace_ids} but the binding "
                f"registers {entry['trace_id']!r}")
        out.append(NetworkProfile(
            profile_id=name, trace_id=entry["trace_id"], seed=int(entry["seed"]),
            trace_sha256=str(entry["trace_sha256"]),
            samples=tuple(rows[:sample_count]),
        ))
    return tuple(out)


# --------------------------------------------------------------------------
# Cell plan
# --------------------------------------------------------------------------


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
    """One (block order, channel, repetition) cell."""

    cell_id: str
    profile_id: str
    order_index: int
    repetition: int
    sequence: tuple[str, ...]
    blocks: tuple[Block, ...]
    run_index: int

    @property
    def transitions(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.sequence, self.sequence[1:]))

    def to_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id, "profile_id": self.profile_id,
            "order_index": self.order_index, "repetition": self.repetition,
            "sequence": list(self.sequence),
            "transitions": ["->".join(pair) for pair in self.transitions],
            "blocks": [block.to_json() for block in self.blocks],
            "run_index": self.run_index,
            "frames_per_block": FRAMES_PER_BLOCK,
            "frames_total": FRAMES_PER_CELL,
        }


def build_cell_plan(
    tiers: Sequence[LoadTier], *, ports: Mapping[str, int], seed: int
) -> tuple[Cell, ...]:
    """The amended 3 x 2 x 2 plan, in a seeded random run order.

    The *design* is fixed and balanced; only the order in which the 12 cells are
    executed is randomized, so a monotone drift in host or radio state cannot be
    mistaken for a channel, order or repetition effect. The seed is recorded.
    """
    import random

    by_tier = {tier.tier: tier for tier in tiers}
    missing = [name for name in TIER_ORDER if name not in by_tier]
    if missing:
        raise ContractError(f"cell plan needs tiers {missing}")

    cells: list[Cell] = []
    for profile_id in CONTRAST_PROFILE_IDS:
        for order_index in range(len(BLOCK_ORDERS)):
            for repetition in range(REPETITIONS):
                sequence = block_sequence(order_index, repetition)
                blocks = []
                for position, tier_name in enumerate(sequence):
                    tier = by_tier[tier_name]
                    blocks.append(Block(
                        block_index=position, tier=tier_name,
                        action_id=tier.action_id, payload_bytes=tier.payload_bytes,
                        chunks_per_frame=tier.chunks_per_frame,
                        frames=FRAMES_PER_BLOCK,
                        first_frame_index=position * FRAMES_PER_BLOCK,
                        port=int(ports[tier_name]),
                    ))
                cells.append(Cell(
                    cell_id=(f"{profile_id.lower()}__order{order_index}"
                             f"__rep{repetition}"),
                    profile_id=profile_id, order_index=order_index,
                    repetition=repetition, sequence=sequence,
                    blocks=tuple(blocks), run_index=-1,
                ))

    rng = random.Random(seed)
    rng.shuffle(cells)
    return tuple(
        Cell(cell_id=cell.cell_id, profile_id=cell.profile_id,
             order_index=cell.order_index, repetition=cell.repetition,
             sequence=cell.sequence, blocks=cell.blocks, run_index=index)
        for index, cell in enumerate(cells)
    )


def audit_cell_plan(cells: Sequence[Cell]) -> dict[str, Any]:
    """Prove the plan is the balanced design it claims to be.

    Returns the counts a reader needs to check balance without rerunning the
    construction, and a boolean per registered balance property.
    """
    from collections import Counter

    position_counts: dict[str, Counter] = {
        profile: Counter() for profile in CONTRAST_PROFILE_IDS}
    transition_counts: dict[str, Counter] = {
        profile: Counter() for profile in CONTRAST_PROFILE_IDS}
    tier_counts: Counter = Counter()
    reverse_ok = True

    by_key = {(cell.profile_id, cell.order_index, cell.repetition): cell
              for cell in cells}
    for cell in cells:
        for position, tier in enumerate(cell.sequence):
            position_counts[cell.profile_id][(tier, position)] += 1
            tier_counts[tier] += 1
        for pair in cell.transitions:
            transition_counts[cell.profile_id]["->".join(pair)] += 1
    for (profile, order_index, repetition), cell in by_key.items():
        if repetition == 1:
            base = by_key.get((profile, order_index, 0))
            if base is None or cell.sequence != tuple(reversed(base.sequence)):
                reverse_ok = False

    all_transitions = {f"{a}->{b}" for a in TIER_ORDER for b in TIER_ORDER if a != b}
    per_profile_transition_balance = {
        profile: (set(counter) == all_transitions
                  and len(set(counter.values())) == 1)
        for profile, counter in transition_counts.items()
    }
    per_profile_position_balance = {
        profile: len(set(counter.values())) == 1 and len(counter) == 9
        for profile, counter in position_counts.items()
    }
    return {
        "cells": len(cells),
        "expected_cells": len(BLOCK_ORDERS) * len(CONTRAST_PROFILE_IDS) * REPETITIONS,
        "tier_block_counts": dict(tier_counts),
        "position_balanced_per_channel": per_profile_position_balance,
        "transition_counts_per_channel": {
            profile: dict(counter) for profile, counter in transition_counts.items()},
        "all_six_transitions_balanced_per_channel": per_profile_transition_balance,
        "repetition_1_reverses_repetition_0": reverse_ok,
        "load_is_within_cell": all(len(set(cell.sequence)) == len(TIER_ORDER)
                                   for cell in cells),
    }


def resolved_source_hashes(repo_root: Path) -> dict[str, str]:
    """Content hashes of every authoritative source this contract resolved."""
    return {
        relpath: sha256_file(repo_root / relpath)
        for relpath in (
            ACTION_CATALOG_RELPATH, PROFILE_BINDING_RELPATH,
            PROFILE_TRACE_RELPATH, PRODUCTION_RECEIVER_RELPATH,
        )
    }
