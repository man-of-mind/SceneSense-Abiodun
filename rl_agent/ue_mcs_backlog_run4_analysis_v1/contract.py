"""Frozen scientific contract for the Run-4 queue/service analysis.

This module is deliberately outcome-free.  It was committed before the
12-cell calibration was launched.  It pins the capture authority, the causal
200-ms transition, the FIT-only transformations, the validation gates, and
the only two admissible ways a future composite environment may consume the
queue model.

The calibration packetization is useful for identifying UE RLC queue/service
dynamics.  It is *not* production transport-latency or delivery evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RELPATH = "rl_agent/ue_mcs_backlog_run4_analysis_v1"
PACKAGE_ID = "ue_mcs_backlog_run4_analysis_v1"
SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "OFFLINE_UE_QUEUE_SERVICE_CALIBRATION_ONLY__NOT_PRODUCTION_LATENCY__"
    "NOT_DELIVERY_EVIDENCE__NOT_POLICY_VALIDATION"
)

RUNNER_COMMIT = "d7d50b8df0cb723c1298eea5cce5addc85d280f2"
RUNNER_TREE = "7075257f3a6965c54be3a5ec54ec49154f25d3d9"
RUNNER_SOURCE_PINS: Mapping[str, str] = {
    "rl_agent/ue_mcs_backlog_run4_calibration_v1/config_v1.json":
        "6808c9aa29833db94036d4423b2aa171e13d74ddc4735261d31c36c69030e23f",
    "rl_agent/ue_mcs_backlog_run4_calibration_v1/contract.py":
        "db3300a6c999a5691db97c603712a91a1fe919b125f2112f88a009d088de226b",
    "rl_agent/ue_mcs_backlog_run4_calibration_v1/runner.py":
        "bb5499c884c07da3f9d0a4580a5b602920865df3c131192d39f360df92aa9886",
    "rl_agent/ue_mcs_backlog_run4_calibration_v1/tagged_sender.py":
        "0caf35720e816047e956fee4277e9c9a0db998b8849b48a5ef1fa520126e5b3e",
    "rl_agent/ue_mcs_backlog_run4_calibration_v1/PREREGISTRATION.md":
        "df4c75c6beba72afd0912070bae1f3ac12fa22017e37204b2e885353beed1fa5",
}
PRIOR_GATE_AUTHORITY_PINS: Mapping[str, str] = {
    "rl_agent/ue_mcs_backlog_near_capacity_v1/analysis_spec.py":
        "072742ef802f37c05e2fc68fcdd235956481d6f46db682b5074b93df0b0d6d99",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/contract.py":
        "99dd192e7df07c1aa7e4d8fcc1d2dc3d5d68b5e23fcedd0109ab2265b962e36e",
    "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py":
        "c3aaf4450deb80e3427972d745bd171cacda517864860f1106ce9d140dc4c18b",
}
PRODUCTION_PACKETIZATION_SOURCE_PINS: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/contract.py":
        "2f501833a07a6bda2b126c6da95cfc432b64aca4c1234a0498ba4c3c3523a520",
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/schema.py":
        "a9d133ce98d5f29b6830d1a5f4034128d0d685c995d1e932917e684622e92d3b",
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/manifest.py":
        "eb4f517388ecb8aa7279bf66bbe189d8a7435d3f3357bf05206a44b8b34f7b11",
}

# Capture design: 10-Hz physical observations, 5-Hz policy decisions.
STEP_PERIOD_NS = 100_000_000
DURATION_STEPS = 2
DURATION_NS = DURATION_STEPS * STEP_PERIOD_NS
FRAMES_PER_CELL = 450
EXPECTED_CELLS = 12
EXPECTED_RAW_DECISIONS = 5_400
PRIMARY_CYCLES_PER_CELL = 224
EXPECTED_PRIMARY_CYCLES = PRIMARY_CYCLES_PER_CELL * EXPECTED_CELLS
EXPECTED_FIT_CYCLES = EXPECTED_PRIMARY_CYCLES // 2
EXPECTED_VALIDATION_CYCLES = EXPECTED_PRIMARY_CYCLES // 2
FIT = "FIT"
VALIDATION = "VALIDATION"
PROFILES = ("FAVORABLE_STABLE", "ADVERSE_STABLE")
FIT_PERMUTATION_INDICES = (0, 1, 2)
VALIDATION_PERMUTATION_INDICES = (3, 4, 5)

# Calibration sender domain.
CALIBRATION_ACTION_IDS = (40, 39, 38)
CALIBRATION_MODE_ID = 6  # AE64/UINT8 in the frozen 12-mode catalogue.
CALIBRATION_TOTAL_TRANSMITTED_BYTES = (126_237, 374_264, 619_563)
CALIBRATION_CHUNK_PAYLOAD_BYTES = 1_200
CALIBRATION_APPLICATION_HEADER_BYTES = 24

# Deployed SFD1/UDP fragmentation domain.  ``chunk_bytes_including_header``
# is 12,500, therefore each !IHH datagram carries at most 12,492 bytes.
PRODUCTION_CHUNK_BYTES_INCLUDING_HEADER = 12_500
PRODUCTION_FRAGMENT_HEADER_BYTES = 8
PRODUCTION_PAYLOAD_BYTES_PER_DATAGRAM = 12_492
PRODUCTION_UDP_HEADER_BYTES = 8
PRODUCTION_IPV4_HEADER_BYTES = 20
PRODUCTION_FULL_UDP_APPLICATION_DATAGRAM_BYTES = 12_500
PRODUCTION_UNFRAGMENTED_IPV4_BYTES = 12_528
REGISTERED_PATH_MTU_BYTES = 1_500
PRODUCTION_PACKETIZATION_STATUS = "PACKETIZATION_TRANSFER_NOT_USED_FOR_LATENCY"
BYTE_DOMAIN_TRANSFER_UNRESOLVED = (
    "CALIBRATION_TO_PRODUCTION_PDCP_RLC_BYTE_DOMAIN_UNRESOLVED"
)
BYTE_DOMAIN_CONVERSION_PROVEN = "EXACT_PRODUCTION_PDCP_RLC_CONVERSION_PROVEN"

UE_RNTI_AND_BEARER_MUST_MATCH = True
UE_MCS_TABLE = 0
UE_MCS_ROUND = 0
UE_MCS_MIN = 0
UE_MCS_MAX = 28
CLOCK_BRIDGE_MAX_RESIDUAL_P95_US = 1.0
AGENT_PATH_BUDGET_MS = 170.0

REQUIRED_UE_EVENTS = (
    "NRUE_MAC_DCI_GRANT",
    "NRUE_MAC_RLC_BUFFER_STATUS",
    "NR_PDCP_TX_SDU",
    "NR_RLC_TX_SDU",
    "NR_RLC_TX_DEQUEUE",
)
REQUIRED_GNB_AUDIT_EVENTS = (
    "GNB_MAC_PUSCH_POWER_CONTROL",
    "GNB_MAC_UL_MCS_DECISION",
    "GNB_PDCP_RX_DELIVER",
)

# Fixed, outcome-independent model design inherited from the earlier
# preregistration.  Profile is intentionally absent.
BACKLOG_EDGES = (0.0, 1.0, 1e3, 1e4, 1e5, 1e6, 1e7, math.inf)
MCS_EDGES = (0, 4, 8, 12, 16, 20, 24, math.inf)
MIN_BIN_SUPPORT = 20
BACKOFF_ORDER_WITH_MCS = (
    ("pair_bytes", "backlog", "mcs"),
    ("pair_bytes", "backlog"),
    ("pair_bytes",),
    (),
)
BACKOFF_ORDER_WITHOUT_MCS = (
    ("pair_bytes", "backlog"),
    ("pair_bytes",),
    (),
)
PRIMARY_MODEL_INPUT_FIELDS = (
    "pre_action_backlog_bytes",
    "prior_new_data_round0_table0_ul_mcs",
    "decision_total_transmitted_bytes",
    "held_total_transmitted_bytes",
)
AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS = (
    "profile_id",
    "gnb_final_mcs",
    "held_step_ul_mcs",
    "target_snr_db",
)
assert "profile_id" not in PRIMARY_MODEL_INPUT_FIELDS

# One tensor period.  This is an external controller guard, never a learned
# actor input and never relaxed based on validation coverage.
FRESHNESS_MAX_AGE_NS = STEP_PERIOD_NS
FRESHNESS_POLICY_FIELDS = (
    "camera_si",
    "radar_p40",
    "prior_new_data_round0_table0_ul_mcs",
    "pre_action_rlc_backlog",
)
CALIBRATION_FRESHNESS_FIELDS = (
    "prior_new_data_round0_table0_ul_mcs",
    "pre_action_rlc_backlog",
)
FRESHNESS_SENSITIVITY_MS = (50, 75, 100, 125, 150, 200)
FRESHNESS_FALLBACK = "EXTERNAL_FALLBACK_NO_ZERO_FILL"

# Explicit transfer/support disclosures on every future prediction.
QUEUE_ONLY_DISCLOSURE = "CALIBRATION_QUEUE_SERVICE_DYNAMICS_ONLY"
SINGLE_UE_DISCLOSURE = "SINGLE_UE_RADIO_CONFIGURATION_ONLY"
PROFILE_TRANSFER_DISCLOSURE = "PROFILE_TRANSFER_UNVALIDATED"
MODE_TRANSFER_DISCLOSURE = "MODE_TRANSFER_UNVALIDATED"
PAYLOAD_INTERPOLATION_DISCLOSURE = "PAYLOAD_INTERPOLATION_UNVALIDATED"
PAYLOAD_OUT_OF_RANGE_REFUSAL = "OUT_OF_CALIBRATED_PAIR_BYTE_RANGE"


class ContractError(RuntimeError):
    """A frozen authority, causal ordering, support, or coupling gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    require(type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{label} must be a lowercase SHA-256 hex digest")


def verify_authorities(repo_root: Path = ROOT) -> dict[str, Any]:
    """Verify every source whose semantics are copied into this contract."""
    observed: dict[str, str] = {}
    for relative, expected in {
        **RUNNER_SOURCE_PINS,
        **PRIOR_GATE_AUTHORITY_PINS,
        **PRODUCTION_PACKETIZATION_SOURCE_PINS,
    }.items():
        path = repo_root / relative
        require(path.is_file(), f"authority missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"authority drifted: {relative}")
        observed[relative] = actual
    config = json.loads((repo_root / next(iter(RUNNER_SOURCE_PINS))).read_text(
        encoding="utf-8"))
    require(config.get("schema") ==
            "scenesense.ue_mcs_backlog_run4_calibration_config.v1",
            "capture config schema drifted")
    design = config.get("design", {})
    require(design.get("fps") == 10.0
            and design.get("frames_per_cell") == FRAMES_PER_CELL
            and design.get("expected_cells") == EXPECTED_CELLS
            and design.get("expected_decisions") == EXPECTED_RAW_DECISIONS,
            "capture design drifted")
    packet = config.get("packetization", {})
    require(packet.get("application_chunk_bytes") ==
            CALIBRATION_CHUNK_PAYLOAD_BYTES
            and packet.get("ssburst_header_bytes") ==
                CALIBRATION_APPLICATION_HEADER_BYTES,
            "calibration packetization drifted")
    return {
        "runner_commit": RUNNER_COMMIT,
        "runner_tree": RUNNER_TREE,
        "verified": True,
        "source_sha256": observed,
        "authority_sha256": canonical_sha256(observed),
    }


def production_datagram_count(total_transmitted_bytes: int) -> int:
    require(type(total_transmitted_bytes) is int
            and total_transmitted_bytes > 0,
            "total_transmitted_bytes must be a positive exact integer")
    return math.ceil(total_transmitted_bytes /
                     PRODUCTION_PAYLOAD_BYTES_PER_DATAGRAM)


def production_udp_application_bytes(total_transmitted_bytes: int) -> int:
    """SFD1 bytes plus the deployed 8-byte !IHH header per datagram."""
    return (total_transmitted_bytes
            + PRODUCTION_FRAGMENT_HEADER_BYTES
            * production_datagram_count(total_transmitted_bytes))


def production_unfragmented_ipv4_baseline_bytes(
    total_transmitted_bytes: int,
) -> int:
    """IPv4 bytes before the production packet crosses an MTU boundary.

    It is deliberately *not* named a PDCP/RLC conversion.  A full production
    datagram is 12,528 bytes at IPv4, larger than the registered 1,500-byte
    path MTU.  The exact fragmentation point/domain must be measured before
    this value can be translated into UE queue bytes.
    """
    count = production_datagram_count(total_transmitted_bytes)
    return (production_udp_application_bytes(total_transmitted_bytes)
            + (PRODUCTION_UDP_HEADER_BYTES + PRODUCTION_IPV4_HEADER_BYTES)
            * count)


def calibration_socket_bytes(total_transmitted_bytes: int) -> int:
    """Calibration payload plus its 24-byte SSBURST header per datagram."""
    require(type(total_transmitted_bytes) is int
            and total_transmitted_bytes > 0,
            "calibration payload must be a positive exact integer")
    chunks = math.ceil(total_transmitted_bytes /
                       CALIBRATION_CHUNK_PAYLOAD_BYTES)
    return total_transmitted_bytes + CALIBRATION_APPLICATION_HEADER_BYTES * chunks


def primary_cycle_indices(decisions: int = FRAMES_PER_CELL) -> tuple[tuple[int, int, int], ...]:
    """Return registered (decision, held, successor) indices for d=2.

    The final even decision has a held frame but no in-cell successor and is
    therefore reported as unclosed rather than silently zero-filled.
    """
    require(type(decisions) is int and decisions >= 3,
            "decision count must be an integer >=3")
    return tuple((start, start + 1, start + 2)
                 for start in range(0, decisions - 2, DURATION_STEPS))


def unclosed_decision_indices(decisions: int = FRAMES_PER_CELL) -> tuple[int, ...]:
    covered = {value for triple in primary_cycle_indices(decisions)
               for value in triple[:1]}
    return tuple(index for index in range(0, decisions, DURATION_STEPS)
                 if index not in covered)


@dataclass(frozen=True, slots=True)
class GateSpec:
    number: int
    key: str
    target: str
    thresholds: Mapping[str, Any]


GATES = (
    GateSpec(1, "COMPLETE_AND_PROVEN_CAPTURE",
             "12 sealed cells, 5,400 exact raw terminals, 2,688 registered d=2 cycles",
             {"cells": 12, "raw_decisions": 5_400,
              "primary_cycles": 2_688}),
    GateSpec(2, "CAUSAL_MCS_AND_PROVENANCE",
             "strict-prior UE round0/table0 MCS and verifier-only gNB provenance",
             {"ue_coverage": 1.0, "max_age_ms": 100.0,
              "gnb_unique_match_per_cell": 0.99, "ambiguity": 0,
              "mismatch": 0}),
    GateSpec(3, "CENSORING_BELOW_ONE_PERCENT",
             "queue-ceiling censoring in accepted decisions",
             {"max_fraction": 0.01, "ceiling_proximity_fraction": 0.95,
              "prior_observed_ceiling_bytes": 49_984_583}),
    GateSpec(4, "VALIDATION_NEXT_BACKLOG_ERROR",
             "whole-cell validation d=2 next-backlog prediction",
             {"max_nmae": 0.10,
              "min_improvement_over_persistence": 0.20}),
    GateSpec(5, "VALIDATION_QUEUE_CLEARANCE_LATENCY_ERROR",
             "per-cell sender-decision to UE RLC cohort-clearance timing; not total uplink",
             {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0}),
    GateSpec(6, "VALIDATION_QUEUE_CLEARANCE_OUTCOME",
             "one-sided queue clearance within 170 ms; not end-to-end success",
             {"budget_ms": 170.0, "max_false_success_rate": 0.05,
              "max_brier": 0.15, "prediction_threshold": 0.5}),
    GateSpec(7, "MCS_NONHARM_AND_DIRECTION",
             "MCS model versus backlog-only comparator on identical validation cycles",
             {"max_brier_degradation": 0.01,
              "direction": "higher MCS does not worsen queue clearance"}),
    GateSpec(8, "MONOTONICITY",
             "clearance probability/latency inside measured support",
             {"max_violations": 0,
              "axes": "pair bytes/backlog not better; MCS not worse"}),
)


@dataclass(frozen=True, slots=True)
class BacklogScaleObservation:
    cell_id: str
    cycle_start_index: int
    partition: str
    pre_action_backlog_bytes: int

    def validate(self) -> None:
        require(self.partition == FIT, "backlog scale accepts FIT cells only")
        require(type(self.pre_action_backlog_bytes) is int
                and self.pre_action_backlog_bytes >= 0,
                "backlog scale input must be a nonnegative integer")
        require(bool(self.cell_id), "scale observation needs a cell id")
        require(type(self.cycle_start_index) is int
                and self.cycle_start_index >= 0
                and self.cycle_start_index % DURATION_STEPS == 0,
                "scale observation is not a primary d=2 cycle")

    @property
    def identity(self) -> str:
        return f"{self.cell_id}:{self.cycle_start_index}"


@dataclass(frozen=True, slots=True)
class BacklogScaleDerivation:
    rule: str
    population_count: int
    nearest_rank_one_based: int
    population_sha256: str
    ordered_values_sha256: str
    backlog_log1p_scale: float
    clipping: bool = False


def derive_backlog_log1p_scale(
    observations: Sequence[BacklogScaleObservation],
) -> BacklogScaleDerivation:
    """Nearest-rank FIT P99 of log1p(raw backlog), with no clipping."""
    require(bool(observations), "FIT scale population is empty")
    checked = tuple(observations)
    for row in checked:
        require(type(row) is BacklogScaleObservation,
                "scale population contains a foreign record type")
        row.validate()
    identities = [row.identity for row in checked]
    require(len(identities) == len(set(identities)),
            "scale population contains duplicate cycle identities")
    population = sorted((row.identity, row.pre_action_backlog_bytes)
                        for row in checked)
    values = sorted(math.log1p(row.pre_action_backlog_bytes)
                    for row in checked)
    rank = math.ceil(0.99 * len(values))
    scale = values[rank - 1]
    require(math.isfinite(scale) and scale > 0.0,
            "FIT P99 log1p backlog scale must be finite and positive")
    return BacklogScaleDerivation(
        rule="nearest-rank P99(log1p(pre_action_backlog_bytes)); FIT only",
        population_count=len(values), nearest_rank_one_based=rank,
        population_sha256=canonical_sha256(population),
        ordered_values_sha256=canonical_sha256(values),
        backlog_log1p_scale=scale, clipping=False)


def normalize_backlog_unclipped(backlog_bytes: int, *, scale: float) -> float:
    require(type(backlog_bytes) is int and backlog_bytes >= 0,
            "backlog must be a nonnegative integer")
    require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
            "scale must be finite and positive")
    return math.log1p(backlog_bytes) / float(scale)


@dataclass(frozen=True, slots=True)
class FreshnessDecision:
    accepted: bool
    outcome: str
    missing: tuple[str, ...]
    stale: tuple[str, ...]


def evaluate_freshness(ages_ns: Mapping[str, int | None]) -> FreshnessDecision:
    require(set(ages_ns) == set(FRESHNESS_POLICY_FIELDS),
            "freshness input must name exactly SI, P40, MCS and RLC backlog")
    missing: list[str] = []
    stale: list[str] = []
    for field in FRESHNESS_POLICY_FIELDS:
        age = ages_ns[field]
        if age is None:
            missing.append(field)
        else:
            require(type(age) is int and age >= 0,
                    f"{field} age must be a nonnegative integer or missing")
            if age > FRESHNESS_MAX_AGE_NS:
                stale.append(field)
    accepted = not missing and not stale
    return FreshnessDecision(
        accepted=accepted,
        outcome="ACCEPT_CAUSAL_STATE" if accepted else FRESHNESS_FALLBACK,
        missing=tuple(missing), stale=tuple(stale))


@dataclass(frozen=True, slots=True)
class QueueServiceRequestV1:
    """Exact future provider request; intentionally has no profile field."""

    model_binding_sha256: str
    duration_steps: int
    step_period_ns: int
    pre_action_backlog_bytes: int
    backlog_age_ns: int
    prior_new_data_round0_table0_ul_mcs: int
    mcs_age_ns: int
    decision_mode_id: int
    decision_q_e4: int
    decision_total_transmitted_bytes: int
    held_mode_id: int
    held_q_e4: int
    held_total_transmitted_bytes: int

    def validate(self) -> None:
        _require_sha256(self.model_binding_sha256, "model binding")
        require(self.duration_steps == DURATION_STEPS
                and self.step_period_ns == STEP_PERIOD_NS,
                "queue request must be the registered d=2/100-ms process")
        require(type(self.pre_action_backlog_bytes) is int
                and self.pre_action_backlog_bytes >= 0,
                "pre-action backlog must be nonnegative")
        require(0 <= self.backlog_age_ns <= FRESHNESS_MAX_AGE_NS,
                "RLC backlog is stale")
        require(type(self.prior_new_data_round0_table0_ul_mcs) is int
                and UE_MCS_MIN <= self.prior_new_data_round0_table0_ul_mcs
                    <= UE_MCS_MAX,
                "prior UE UL MCS is outside table-0 support")
        require(0 <= self.mcs_age_ns <= FRESHNESS_MAX_AGE_NS,
                "prior UE UL MCS is stale")
        require(self.decision_mode_id == self.held_mode_id
                and self.decision_q_e4 == self.held_q_e4
                and self.decision_total_transmitted_bytes ==
                    self.held_total_transmitted_bytes,
                "held frame must reuse the exact decision action")
        require(0 <= self.decision_q_e4 <= 9_800,
                "q_e4 is outside the registered action range")
        production_datagram_count(self.decision_total_transmitted_bytes)

    @property
    def decision_udp_application_bytes(self) -> int:
        self.validate()
        return production_udp_application_bytes(
            self.decision_total_transmitted_bytes)

    @property
    def held_udp_application_bytes(self) -> int:
        self.validate()
        return production_udp_application_bytes(
            self.held_total_transmitted_bytes)

    @property
    def request_sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class JointServiceAtomV1:
    service_step0_bytes: int
    service_step1_bytes: int
    next_backlog_bytes: int
    decision_clearance_offset_ns: int | None
    weight: int

    def validate(self) -> None:
        require(all(type(value) is int and value >= 0 for value in (
            self.service_step0_bytes, self.service_step1_bytes,
            self.next_backlog_bytes)), "service atom byte values must be nonnegative")
        require(self.decision_clearance_offset_ns is None
                or (type(self.decision_clearance_offset_ns) is int
                    and self.decision_clearance_offset_ns >= 0),
                "clearance offset must be a nonnegative integer or missing")
        require(type(self.weight) is int and self.weight > 0,
                "service atom needs a positive integer weight")


@dataclass(frozen=True, slots=True)
class QueueServiceDistributionV1:
    model_binding_sha256: str
    request_sha256: str
    support_status: str
    disclosures: tuple[str, ...]
    atoms: tuple[JointServiceAtomV1, ...]

    def validate(self) -> None:
        _require_sha256(self.model_binding_sha256, "model binding")
        _require_sha256(self.request_sha256, "request binding")
        require(bool(self.support_status), "support status is empty")
        require(bool(self.atoms), "service distribution is empty")
        for atom in self.atoms:
            require(type(atom) is JointServiceAtomV1,
                    "distribution contains a foreign atom")
            atom.validate()
        required = {QUEUE_ONLY_DISCLOSURE, SINGLE_UE_DISCLOSURE,
                    PROFILE_TRANSFER_DISCLOSURE,
                    PRODUCTION_PACKETIZATION_STATUS,
                    BYTE_DOMAIN_CONVERSION_PROVEN}
        require(required.issubset(set(self.disclosures)),
                "service distribution omitted mandatory scope disclosures")

    @property
    def total_weight(self) -> int:
        self.validate()
        return sum(atom.weight for atom in self.atoms)

    def exact_draw(self, draw_below_total_weight: int) -> JointServiceAtomV1:
        """Select an atom from an externally supplied exact integer draw."""
        total = self.total_weight
        require(type(draw_below_total_weight) is int
                and 0 <= draw_below_total_weight < total,
                "draw must be an exact integer in [0,total_weight)")
        cumulative = 0
        for atom in self.atoms:
            cumulative += atom.weight
            if draw_below_total_weight < cumulative:
                return atom
        raise AssertionError("validated weighted draw did not resolve")

    def clearance_probability_fraction(self, budget_ns: int) -> tuple[int, int]:
        require(type(budget_ns) is int and budget_ns >= 0,
                "clearance budget must be a nonnegative integer")
        numerator = sum(
            atom.weight for atom in self.atoms
            if atom.decision_clearance_offset_ns is not None
            and atom.decision_clearance_offset_ns <= budget_ns)
        return numerator, self.total_weight


def support_disclosures(request: QueueServiceRequestV1) -> tuple[str, ...]:
    request.validate()
    disclosures = [QUEUE_ONLY_DISCLOSURE, SINGLE_UE_DISCLOSURE,
                   PROFILE_TRANSFER_DISCLOSURE,
                   PRODUCTION_PACKETIZATION_STATUS,
                   BYTE_DOMAIN_TRANSFER_UNRESOLVED]
    if request.decision_mode_id != CALIBRATION_MODE_ID:
        disclosures.append(MODE_TRANSFER_DISCLOSURE)
    if request.decision_total_transmitted_bytes not in \
            CALIBRATION_TOTAL_TRANSMITTED_BYTES:
        disclosures.append(PAYLOAD_INTERPOLATION_DISCLOSURE)
    return tuple(disclosures)


BYTE_DOMAIN_REQUIRED_PROOFS = (
    "source_evidence_manifest_sha256",
    "calibration_sender_to_pdcp_sha256",
    "calibration_pdcp_to_rlc_sha256",
    "production_sender_to_pdcp_sha256",
    "production_ip_fragment_inventory_sha256",
    "production_pdcp_to_rlc_sha256",
    "conversion_formula_sha256",
    "residual_support_sha256",
)


@dataclass(frozen=True, slots=True)
class ByteDomainProofV1:
    source_evidence_manifest_sha256: str
    calibration_sender_to_pdcp_sha256: str
    calibration_pdcp_to_rlc_sha256: str
    production_sender_to_pdcp_sha256: str
    production_ip_fragment_inventory_sha256: str
    production_pdcp_to_rlc_sha256: str
    conversion_formula_sha256: str
    residual_support_sha256: str
    exact_per_decision_accounting: bool
    no_calibration_chunk_effect_as_action_effect: bool
    production_fragmentation_observed_not_assumed: bool
    packetization_delivery_and_latency_excluded: bool

    def validate(self) -> None:
        for field in BYTE_DOMAIN_REQUIRED_PROOFS:
            _require_sha256(getattr(self, field), field)
        require(self.exact_per_decision_accounting is True,
                "byte-domain accounting is not exact per decision")
        require(self.no_calibration_chunk_effect_as_action_effect is True,
                "calibration chunk overhead leaked into the action effect")
        require(self.production_fragmentation_observed_not_assumed is True,
                "production IP fragmentation was assumed rather than observed")
        require(self.packetization_delivery_and_latency_excluded is True,
                "packetization-specific delivery/latency leaked into projection")

    @property
    def proof_sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        })


def qualified_support_disclosures(
    request: QueueServiceRequestV1,
    byte_domain_proof: ByteDomainProofV1,
) -> tuple[str, ...]:
    require(type(byte_domain_proof) is ByteDomainProofV1,
            "foreign byte-domain proof type")
    byte_domain_proof.validate()
    values = [value for value in support_disclosures(request)
              if value != BYTE_DOMAIN_TRANSFER_UNRESOLVED]
    values.append(BYTE_DOMAIN_CONVERSION_PROVEN)
    return tuple(values)


def require_payload_support(request: QueueServiceRequestV1) -> None:
    request.validate()
    minimum = min(CALIBRATION_TOTAL_TRANSMITTED_BYTES)
    maximum = max(CALIBRATION_TOTAL_TRANSMITTED_BYTES)
    require(minimum <= request.decision_total_transmitted_bytes <= maximum,
            PAYLOAD_OUT_OF_RANGE_REFUSAL)


@dataclass(frozen=True, slots=True)
class QueueServiceRefusalV1:
    model_binding_sha256: str
    request_sha256: str
    refusal_code: str
    disclosures: tuple[str, ...]

    def validate(self) -> None:
        _require_sha256(self.model_binding_sha256, "model binding")
        _require_sha256(self.request_sha256, "request binding")
        require(self.refusal_code in {
            PAYLOAD_OUT_OF_RANGE_REFUSAL,
            "MISSING_OR_STALE_PRE_ACTION_BACKLOG",
            "MISSING_OR_STALE_PRIOR_UL_MCS",
            "UNSUPPORTED_DURATION",
            "HELD_ACTION_IDENTITY_MISMATCH",
            "WRONG_RADIO_OR_UE_CONFIGURATION",
            "MODEL_OR_PACKETIZATION_BINDING_MISMATCH",
            "BYTE_DOMAIN_CONVERSION_UNRESOLVED",
        }, "unregistered queue-service refusal code")
        required = {QUEUE_ONLY_DISCLOSURE, SINGLE_UE_DISCLOSURE,
                    PROFILE_TRANSFER_DISCLOSURE,
                    PRODUCTION_PACKETIZATION_STATUS,
                    BYTE_DOMAIN_TRANSFER_UNRESOLVED}
        require(required.issubset(set(self.disclosures)),
                "refusal omitted mandatory scope disclosures")


@dataclass(frozen=True, slots=True)
class QueueServiceModelBindingV1:
    """Digests that make one fitted queue/service artifact self-auditing."""

    raw_manifest_sha256: str
    canonical_d2_cycles_sha256: str
    fit_cell_set_sha256: str
    fit_rows_sha256: str
    validation_cell_set_sha256: str
    validation_rows_sha256: str
    service_table_sha256: str
    byte_domain_proof_sha256: str
    backlog_scale_sha256: str
    freshness_policy_sha256: str
    gate_report_sha256: str
    verifier_report_sha256: str
    validation_influenced_fit: bool
    profile_is_model_input: bool
    packetization_transfer_used_for_latency: bool
    byte_domain_conversion_verified: bool

    def validate(self) -> None:
        for field in (
            "raw_manifest_sha256", "canonical_d2_cycles_sha256",
            "fit_cell_set_sha256", "fit_rows_sha256",
            "validation_cell_set_sha256", "validation_rows_sha256",
            "service_table_sha256", "byte_domain_proof_sha256",
            "backlog_scale_sha256",
            "freshness_policy_sha256", "gate_report_sha256",
            "verifier_report_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        require(self.validation_influenced_fit is False,
                "validation influenced the fitted model")
        require(self.profile_is_model_input is False,
                "hidden profile leaked into the queue model")
        require(self.packetization_transfer_used_for_latency is False,
                "calibration packetization was promoted to latency evidence")
        require(self.byte_domain_conversion_verified is True,
                "production PDCP/RLC byte-domain conversion is unresolved")

    @property
    def model_binding_sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        })


STATE_TRANSITION_ONLY = "STATE_TRANSITION_ONLY"
RESIDUALIZED_SERVICE_TO_LATENCY = "RESIDUALIZED_SERVICE_TO_LATENCY"
RESIDUALIZATION_REQUIRED_PROOFS = (
    "source_288_manifest_sha256",
    "endpoint_semantics_sha256",
    "row_identity_join_sha256",
    "overlap_replacement_sha256",
    "residual_distribution_sha256",
    "production_packetization_sha256",
    "delivery_invariance_sha256",
    "byte_domain_proof_sha256",
)


@dataclass(frozen=True, slots=True)
class ResidualizationProofV1:
    source_288_manifest_sha256: str
    endpoint_semantics_sha256: str
    row_identity_join_sha256: str
    overlap_replacement_sha256: str
    residual_distribution_sha256: str
    production_packetization_sha256: str
    delivery_invariance_sha256: str
    byte_domain_proof_sha256: str
    exact_no_double_count: bool
    service_replaces_overlapping_transport_segment: bool
    all_residual_intervals_nonnegative: bool
    receiver_delivery_population_unchanged: bool

    def validate(self) -> None:
        for field in RESIDUALIZATION_REQUIRED_PROOFS:
            value = getattr(self, field)
            _require_sha256(value, field)
        require(self.exact_no_double_count is True,
                "no-double-count proof did not pass")
        require(self.service_replaces_overlapping_transport_segment is True,
                "service was added without replacing its overlapping segment")
        require(self.all_residual_intervals_nonnegative is True,
                "transport residual contains a negative interval")
        require(self.receiver_delivery_population_unchanged is True,
                "residualization changed the production delivery population")


@dataclass(frozen=True, slots=True)
class CouplingDecisionV1:
    alternative: str
    queue_model_accepted: bool
    training_ready: bool
    reason: str
    proof_sha256: str | None


def resolve_composite_coupling(
    proof: ResidualizationProofV1 | None,
    *,
    byte_domain_proof: ByteDomainProofV1 | None = None,
) -> CouplingDecisionV1:
    """Choose the sole scientifically admissible composite coupling.

    With no residualization proof the queue model may update the next backlog,
    but the missing state-to-reward latency coupling blocks Run-4 training.
    A supplied proof is fail-closed; it is never silently downgraded.
    """
    if proof is None:
        return CouplingDecisionV1(
            alternative=STATE_TRANSITION_ONLY, queue_model_accepted=True,
            training_ready=False,
            reason=("queue dynamics may update next state, but service cannot "
                    "be added to the inclusive 288 uplink total"),
            proof_sha256=None)
    require(type(proof) is ResidualizationProofV1,
            "foreign residualization proof type")
    require(type(byte_domain_proof) is ByteDomainProofV1,
            "residualized coupling requires the typed byte-domain proof")
    byte_domain_proof.validate()
    proof.validate()
    require(proof.byte_domain_proof_sha256 ==
            byte_domain_proof.proof_sha256,
            "residualization proof is not bound to the byte-domain proof")
    body = {field: getattr(proof, field) for field in proof.__dataclass_fields__}
    return CouplingDecisionV1(
        alternative=RESIDUALIZED_SERVICE_TO_LATENCY,
        queue_model_accepted=True, training_ready=True,
        reason=("service-derived completion replaces the proven overlapping "
                "288 transport segment; only the disjoint residual is joined"),
        proof_sha256=canonical_sha256(body))


def contract_document() -> dict[str, Any]:
    """Canonical outcome-free document recorded in every future artifact."""
    return {
        "schema": "scenesense.ue_mcs_backlog_run4_analysis_contract.v1",
        "package_id": PACKAGE_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "runner_commit": RUNNER_COMMIT,
        "runner_tree": RUNNER_TREE,
        "raw_design": {
            "cells": EXPECTED_CELLS,
            "raw_decisions": EXPECTED_RAW_DECISIONS,
            "step_period_ns": STEP_PERIOD_NS,
            "duration_steps": DURATION_STEPS,
            "primary_cycles": EXPECTED_PRIMARY_CYCLES,
            "fit_cycles": EXPECTED_FIT_CYCLES,
            "validation_cycles": EXPECTED_VALIDATION_CYCLES,
            "primary_cycle_parity": "decision_index % 2 == 0",
            "final_even_cycle": "UNCLOSED_REPORTED_NOT_ZERO_FILLED",
        },
        "model_inputs": list(PRIMARY_MODEL_INPUT_FIELDS),
        "audit_only_not_model_inputs": list(AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS),
        "service_target": {
            "marginal": "RLC dequeue bytes in each actual 100-ms decision interval",
            "joint": "ordered two-step RLC service and next pre-action backlog",
            "clearance": "FIFO byte-cohort clearance from sender decision timestamp",
            "profile_is_hidden": False,
        },
        "freshness": {
            "max_age_ns": FRESHNESS_MAX_AGE_NS,
            "fields": list(FRESHNESS_POLICY_FIELDS),
            "fallback": FRESHNESS_FALLBACK,
            "sensitivity_ms": list(FRESHNESS_SENSITIVITY_MS),
            "validation_may_tune_bound": False,
        },
        "backlog_scaling": {
            "rule": "nearest-rank P99(log1p(pre_action_backlog_bytes))",
            "population": "accepted whole-cell FIT primary cycles only",
            "clipping": False,
            "validation_influence": False,
        },
        "packetization": {
            "production_chunk_bytes_including_header":
                PRODUCTION_CHUNK_BYTES_INCLUDING_HEADER,
            "production_payload_bytes_per_datagram":
                PRODUCTION_PAYLOAD_BYTES_PER_DATAGRAM,
            "production_fragment_header_bytes":
                PRODUCTION_FRAGMENT_HEADER_BYTES,
            "production_udp_header_bytes": PRODUCTION_UDP_HEADER_BYTES,
            "production_ipv4_header_bytes": PRODUCTION_IPV4_HEADER_BYTES,
            "production_full_udp_application_datagram_bytes":
                PRODUCTION_FULL_UDP_APPLICATION_DATAGRAM_BYTES,
            "production_unfragmented_ipv4_bytes":
                PRODUCTION_UNFRAGMENTED_IPV4_BYTES,
            "registered_path_mtu_bytes": REGISTERED_PATH_MTU_BYTES,
            "production_ip_fragmentation": "MUST_BE_OBSERVED_AND_RECONCILED",
            "byte_domain_status_before_proof": BYTE_DOMAIN_TRANSFER_UNRESOLVED,
            "byte_domain_required_proofs": list(BYTE_DOMAIN_REQUIRED_PROOFS),
            "boundary_examples": {
                "12492": production_udp_application_bytes(12_492),
                "12493": production_udp_application_bytes(12_493),
            },
            "status": PRODUCTION_PACKETIZATION_STATUS,
        },
        "gates": [
            {"number": gate.number, "key": gate.key,
             "target": gate.target, "thresholds": dict(gate.thresholds)}
            for gate in GATES
        ],
        "coupling": {
            "alternative_a": STATE_TRANSITION_ONLY,
            "alternative_a_training_ready": False,
            "alternative_b": RESIDUALIZED_SERVICE_TO_LATENCY,
            "alternative_b_requires": list(RESIDUALIZATION_REQUIRED_PROOFS),
            "blind_add_to_288_total": "FORBIDDEN",
            "silent_payload_byte_equality_across_packetizations": "BLOCKER",
        },
        "artifact_binding_fields": list(
            QueueServiceModelBindingV1.__dataclass_fields__),
    }


CONTRACT_SHA256 = canonical_sha256(contract_document())
