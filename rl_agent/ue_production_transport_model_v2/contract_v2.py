#!/usr/bin/env python3
"""Frozen v2 amendment contract: post-hoc development model, no new capture.

Evidence class
--------------
``POSTHOC_DEVELOPMENT_MODEL_FOR_EXPLORATORY_RUN4_TRAINING``.

This is **not** confirmatory evidence.  The v1 held-out validation population
has already been inspected twice (two full gate evaluations on 2026-09-28), so
it is no longer pristine.  Any number this package reports on that population
is a descriptive engineering audit, never an independent confirmation, and the
v1 Gate-5 result stays **FAILED** forever.

What this package may do
------------------------
Refit a small, monotone, reward-aligned transport model over the already
captured evidence, selected by grouped leave-one-whole-FIT-cell-out
cross-validation.

What it may not do
------------------
Re-capture, waive the v1 latency gate, add a policy-state variable, add a
reward term, use HARQ / retransmission / datagram count as a feature, or use
any future, held-frame, gNB-only or realized-outcome quantity as a predictor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_production_queue_capture_v1 import contract as V1


ROOT = V1.ROOT
PACKAGE_RELPATH = "rl_agent/ue_production_transport_model_v2"
PACKAGE_ID = "ue_production_transport_model_v2"
SCHEMA_VERSION = 2

EVIDENCE_CLASS = "POSTHOC_DEVELOPMENT_MODEL_FOR_EXPLORATORY_RUN4_TRAINING"
CONFIRMATORY = False
VALIDATION_POPULATION_STATUS = (
    "INSPECTED_TWICE_ON_20260928__NO_LONGER_PRISTINE__"
    "DESCRIPTIVE_ENGINEERING_AUDIT_ONLY"
)
V1_GATE5_STATUS = "FAILED__PRESERVED__NEVER_REWRITTEN_AS_PASSED"

CAPTURE_ROOT_RELPATH = (
    "rl_agent/experiments/ue_production_queue_capture_v1/20260928_214832_live"
)
PARSED_V1_RELPATH = (
    "rl_agent/experiments/ue_production_queue_capture_v1/20260928_214832_parsed"
)
PARSED_V2_RELPATH = (
    "rl_agent/experiments/ue_production_queue_capture_v1/20260928_parsed_v2"
)

# Preserved byte-for-byte.  A drift here invalidates this amendment.
PRESERVED_PINS: Mapping[str, str] = {
    f"{CAPTURE_ROOT_RELPATH}/TERMINAL.json":
        "6f1ce831fe035f7379c666eb68e7e8720612046739fc257c6f05f3281a2a4a3f",
    f"{CAPTURE_ROOT_RELPATH}/CAPTURE_RESULT.json":
        "a7dd167aef636534dec6b0a73fb418ffb15957dc1902432d23c5d98113afd8c5",
    f"{CAPTURE_ROOT_RELPATH}/manifest.json":
        "385be47305e10b347010091c9313e19870af4e43f933c94699b055c03de8af95",
    f"{PARSED_V1_RELPATH}/PARSE_REPORT.json":
        "6c5b5b9dea26099fe7bebdbbd3fd8e768a66858f36e54bac1f3833622dcfe822",
    f"{PARSED_V1_RELPATH}/ANALYSIS.json":
        "ea16c814e88dec00d346bd544b7659c85b5f39e10d64bd39d62f144245df1bab",
    f"{PARSED_V2_RELPATH}/PARSE_REPORT.json":
        "0a4264c16396c47ac1bfc1d94b12a6257da2c39b388b8ec7bb3e956265652271",
    f"{PARSED_V2_RELPATH}/ANALYSIS.json":
        "b0170979aadf8e939da555828e9b871c274c89580c5a969d4a4d355489d1c5a8",
    f"{PARSED_V2_RELPATH}/frames.csv":
        "340e0d394c37f9caa0a6420c93283b92f23a6f8c43784495768b5dbf56899bcf",
    f"{PARSED_V2_RELPATH}/cycles.csv":
        "4b70262cc112851e6b0d16ee4109cfdf9043b970e78838c9b944319dd6863f5b",
}

# Both historical gate-5 evaluations, retained as FAILED.
PRESERVED_GATE5_RESULTS: tuple[Mapping[str, Any], ...] = (
    {"evaluation": "v1_first", "passed": False,
     "p50_error_ms": 33.089929551971785, "p95_error_ms": 1780.4888943690967},
    {"evaluation": "v1_second", "passed": False,
     "p50_error_ms": 37.29388803650191, "p95_error_ms": 1056.64432504323},
)


class ContractV2Error(RuntimeError):
    """A v2 amendment invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractV2Error(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preserved(repo_root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in PRESERVED_PINS.items():
        path = repo_root / relative
        require(path.is_file(), f"preserved artifact missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected,
                f"preserved artifact was modified: {relative}")
        observed[relative] = actual
    return {"verified": True, "files": observed,
            "preserved_sha256": canonical_sha256(observed)}


# ---------------------------------------------------------------------------
# Causal state definition (Phase B)
# ---------------------------------------------------------------------------
# One scalar UE-local RLC queue snapshot taken strictly before the registered
# frame-open / action-release instant, before this frame is enqueued.  It is
# NOT a drain rate, NOT the current payload, NOT a future grant, NOT measured
# service and NOT a post-multiplex BSR.
PRE_ENQUEUE_BACKLOG_DEFINITION = (
    "UE_LOCAL_RLC_BYTES_WAITING_STRICTLY_BEFORE_FRAME_OPEN_AND_BEFORE_THIS_"
    "FRAME_IS_ENQUEUED__SINGLE_SCALAR_SNAPSHOT"
)
CAUSAL_CUTOFF_FIELD = "frame_open_monotonic_ns"
CAUSAL_CUTOFF_REJECTED_FIELD = "first_send_monotonic_ns"
MAX_CAUSAL_AGE_NS = 100_000_000
MIN_CAUSAL_AGE_NS = 0
DECISION_FRAME_STRIDE = 2          # 10-Hz frames, 5-Hz decisions

# The held frame reuses the decision taken at t.  It never contributes an
# independently observed successor backlog/MCS to the evaluation of action t.
HELD_FRAME_RULE = "HELD_FRAME_REUSES_DECISION_STATE__NO_FUTURE_OBSERVATION"


# ---------------------------------------------------------------------------
# Backlog normalization - exactly one backlog policy feature
# ---------------------------------------------------------------------------
# The single actor-visible queue feature is
#
#     backlog_scaled = min(1, log1p(pre_enqueue_backlog_bytes)
#                             / log1p(RLC_AM_TX_ADMISSION_CEILING_BYTES))
#
# Always call this **log-scaled backlog**.  Never call it "percentage full":
# 40 kB maps to 0.598 while physical occupancy is only 0.08%.
BACKLOG_FEATURE_NAME = "pre_action_rlc_backlog_log1p_scaled"
BACKLOG_FEATURE_FORM = (
    "min(1, log1p(pre_enqueue_backlog_bytes) / log1p(50000000))"
)
BACKLOG_FEATURE_SEMANTICS = "LOG_SCALED_BACKLOG__NEVER_PERCENTAGE_FULL"

# The reference is the *verified OAI AM transmit admission ceiling*, not a
# FIT-derived quantile.  It is a property of the deployed RLC entity:
#
#   common/platform_constants.h:60   #define RLC_TX_MAXSIZE 10000000
#   nr_rlc_entity.c:83               ret->tx_maxsize = tx_maxsize * 5;
#   nr_rlc_entity_am.c:1831          if (tx_size + size > tx_maxsize) reject
#
# so the admission ceiling is 10,000,000 * 5 = 50,000,000 bytes.
RLC_AM_TX_ADMISSION_CEILING_BYTES = 50_000_000
BACKLOG_REFERENCE_RULE = "VERIFIED_OAI_AM_TRANSMIT_ADMISSION_CEILING"
BACKLOG_REFERENCE_IS_FIT_DERIVED = False

OAI_CEILING_SOURCE_PINS: Mapping[str, str] = {
    "OAI/openairinterface5g/common/platform_constants.h":
        "3a2ebaa187881fbc53b9e3077a59dde4897362f05941b346ab71483681123d5a",
    "OAI/openairinterface5g/openair2/LAYER2/nr_rlc/nr_rlc_entity.c":
        "89620b2c0ea30593ce045839b4fd8897ba4e50f2f2cf72aa213d23ce8277ff98",
    "OAI/openairinterface5g/openair2/LAYER2/nr_rlc/nr_rlc_entity_am.c":
        "9283de37d39d7187f79cf07ba1118af0160778bb2067b7858df993e74b002a48",
}
OAI_CEILING_DERIVATION = {
    "rlc_tx_maxsize_define": 10_000_000,
    "am_multiplier": 5,
    "product_bytes": 50_000_000,
    "admission_rule": "reject when tx_size + size > tx_maxsize",
}

# The Run-3 era `bsr_log1p_scale = 1.0` is explicitly rejected: it leaves the
# feature an unnormalized log-byte magnitude rather than a bounded signal.
REJECTED_LEGACY_BACKLOG_SCALE = 1.0

# Exact zero backlog must normalize to exactly 0.0.
ZERO_BACKLOG_MAPS_TO = 0.0

# Normalization clips at 1.0, but the RAW backlog is retained and is what the
# model-support check uses.  Clipping must never be able to hide an
# out-of-support value: a raw backlog beyond the accepted training support
# invokes the registered external fallback, exactly like a stale sample.
BACKLOG_OUT_OF_SUPPORT_POLICY = (
    "RAW_BACKLOG_CHECKED_AGAINST_SUPPORT__CLIP_MUST_NOT_BYPASS_FALLBACK"
)
NORMALIZATION_CLIP_IS_NOT_A_SUPPORT_CHECK = True
BACKLOG_REPORTED_MAPPING_BYTES: tuple[int, ...] = (
    0, 40_000, 400_000, 1_000_000, 10_000_000,
    RLC_AM_TX_ADMISSION_CEILING_BYTES,
)

# No queue capacity and no drain rate are ever added as actor features.
FORBIDDEN_ADDITIONAL_QUEUE_FEATURES: tuple[str, ...] = (
    "queue_capacity_bytes", "drain_rate_bps", "service_rate_bps",
    "queue_occupancy_fraction",
)


def verify_oai_ceiling(repo_root: Path = ROOT) -> dict[str, Any]:
    """Confirm the ceiling's source identities before it may be used."""
    observed: dict[str, str] = {}
    for relative, expected in OAI_CEILING_SOURCE_PINS.items():
        path = repo_root / relative
        require(path.is_file(), f"OAI ceiling source missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected,
                f"OAI ceiling source drifted: {relative}")
        observed[relative] = actual
    require(OAI_CEILING_DERIVATION["rlc_tx_maxsize_define"]
            * OAI_CEILING_DERIVATION["am_multiplier"]
            == RLC_AM_TX_ADMISSION_CEILING_BYTES,
            "OAI ceiling derivation does not reproduce the bound constant")
    return {"verified": True, "ceiling_bytes":
            RLC_AM_TX_ADMISSION_CEILING_BYTES, "files": observed,
            "source_sha256": canonical_sha256(observed)}


def backlog_scaled(value: float) -> float:
    """The single registered backlog policy feature (log-scaled backlog).

    Clips at 1.0 by construction.  This is a *feature-range* clip only; the
    caller must separately check the RAW value against the accepted training
    support and invoke the external fallback when it is exceeded.
    """
    import math
    require(value >= 0, "pre-enqueue backlog cannot be negative")
    if value == 0:
        return ZERO_BACKLOG_MAPS_TO
    return min(1.0, math.log1p(value)
               / math.log1p(RLC_AM_TX_ADMISSION_CEILING_BYTES))


def backlog_mapping_report() -> list[dict[str, Any]]:
    return [
        {"bytes": value, "log_scaled_backlog": round(backlog_scaled(value), 3),
         "physical_occupancy_fraction":
             value / RLC_AM_TX_ADMISSION_CEILING_BYTES}
        for value in BACKLOG_REPORTED_MAPPING_BYTES
    ]


# ---------------------------------------------------------------------------
# Model family (Phase C) - fixed, not searched
# ---------------------------------------------------------------------------
MODEL_FAMILY = "MONOTONE_TWO_PART_DEADLINE_HEAD_PLUS_CONDITIONAL_LATENCY_HEAD"
MODEL_SEED = 17

ALLOWED_PREDICTORS: tuple[str, ...] = (
    "pre_enqueue_backlog_bytes",
    "prior_ul_mcs",
    "action_wire_bytes",
)
FORBIDDEN_PREDICTORS: tuple[str, ...] = (
    "harq_round", "retransmission_count", "datagram_count",
    "profile_id", "target_snr_db", "commanded_noise_power_db",
    "gnb_mcs", "achieved_pusch_snr_db",
    "measured_rlc_ingress_bytes", "measured_rlc_service_bytes",
    "held_frame_total_transmitted_bytes", "successor_rlc_backlog_bytes",
    "transport_latency_ns", "terminal_outcome", "complete",
    "datagrams_received", "post_decision_ul_mcs",
)

# Backlog and current-frame bytes get separate monotone effects.
SEPARATE_MONOTONE_EFFECTS = True
SHARED_BACKLOG_PLUS_BYTES_COEFFICIENT_FORBIDDEN = True

REWARD_DEADLINE_NS = V1.REWARD_DEADLINE_NS
REWARD_DEADLINE_MS = V1.REWARD_DEADLINE_MS
REWARD_LATENCY_WEIGHT = V1.REWARD_LATENCY_WEIGHT
REGISTERED_FAILURE_REWARD = V1.REGISTERED_FAILURE_REWARD


# ---------------------------------------------------------------------------
# Selection and acceptance (Phase D)
# ---------------------------------------------------------------------------
SELECTION_PROTOCOL = "GROUPED_LEAVE_ONE_WHOLE_FIT_CELL_OUT_CROSS_VALIDATION"
SELECTION_POPULATION = "FIT_CELLS_ONLY"
OPEN_ENDED_SEARCH_FORBIDDEN = True


@dataclass(frozen=True, slots=True)
class AcceptanceGate:
    number: int
    key: str
    target: str
    thresholds: Mapping[str, Any]


ACCEPTANCE_GATES: tuple[AcceptanceGate, ...] = (
    AcceptanceGate(1, "CAUSAL_JOIN_INTEGRITY",
                   "100% causal coverage at the frame-open cutoff",
                   {"coverage": 1.0, "max_age_ns": MAX_CAUSAL_AGE_NS,
                    "min_age_ns": MIN_CAUSAL_AGE_NS,
                    "cutoff_field": CAUSAL_CUTOFF_FIELD}),
    AcceptanceGate(2, "DEADLINE_HEAD_CALIBRATION",
                   "full sent population, grouped CV",
                   {"max_brier": 0.15, "max_false_success_rate": 0.05}),
    AcceptanceGate(3, "CONDITIONAL_ON_TIME_LATENCY_ERROR",
                   "conditional on completing within 170 ms, grouped CV",
                   {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0}),
    AcceptanceGate(4, "REWARD_ERROR",
                   "implied error in the registered reward",
                   {"max_p50": 0.025, "max_p95": 0.05}),
    AcceptanceGate(5, "MONOTONICITY",
                   "separate monotone effects inside measured support",
                   {"max_violations": 0}),
    AcceptanceGate(6, "FEATURE_PROVENANCE",
                   "every predictor is allowed, causal and timestamped",
                   {"forbidden": list(FORBIDDEN_PREDICTORS),
                    "allowed": list(ALLOWED_PREDICTORS)}),
)

# Reward-error correspondence, stated so the thresholds are auditable:
#   r = Q_perc - 0.25 * latency_ms / 170
#   |dr| = 0.25 * |d latency_ms| / 170
#   17 ms -> 0.025 ;  34 ms -> 0.050
REWARD_ERROR_PER_MS = REWARD_LATENCY_WEIGHT / REWARD_DEADLINE_MS


def reward_error_for_latency_error_ms(value: float) -> float:
    return REWARD_ERROR_PER_MS * float(value)


def contract_document() -> dict[str, Any]:
    return {
        "package_id": PACKAGE_ID, "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS, "confirmatory": CONFIRMATORY,
        "validation_population_status": VALIDATION_POPULATION_STATUS,
        "v1_gate5_status": V1_GATE5_STATUS,
        "v1_contract_sha256": V1.CONTRACT_SHA256,
        "preserved_pins": dict(PRESERVED_PINS),
        "preserved_gate5_results": [dict(r) for r in PRESERVED_GATE5_RESULTS],
        "causal": {
            "pre_enqueue_backlog_definition": PRE_ENQUEUE_BACKLOG_DEFINITION,
            "cutoff_field": CAUSAL_CUTOFF_FIELD,
            "rejected_cutoff_field": CAUSAL_CUTOFF_REJECTED_FIELD,
            "max_age_ns": MAX_CAUSAL_AGE_NS, "min_age_ns": MIN_CAUSAL_AGE_NS,
            "decision_frame_stride": DECISION_FRAME_STRIDE,
            "held_frame_rule": HELD_FRAME_RULE,
        },
        "model": {
            "family": MODEL_FAMILY, "seed": MODEL_SEED,
            "allowed_predictors": list(ALLOWED_PREDICTORS),
            "forbidden_predictors": list(FORBIDDEN_PREDICTORS),
            "separate_monotone_effects": SEPARATE_MONOTONE_EFFECTS,
            "shared_coefficient_forbidden":
                SHARED_BACKLOG_PLUS_BYTES_COEFFICIENT_FORBIDDEN,
        },
        "backlog_normalization": {
            "feature_name": BACKLOG_FEATURE_NAME,
            "form": BACKLOG_FEATURE_FORM,
            "semantics": BACKLOG_FEATURE_SEMANTICS,
            "reference_rule": BACKLOG_REFERENCE_RULE,
            "reference_is_fit_derived": BACKLOG_REFERENCE_IS_FIT_DERIVED,
            "rlc_am_tx_admission_ceiling_bytes":
                RLC_AM_TX_ADMISSION_CEILING_BYTES,
            "oai_ceiling_source_pins": dict(OAI_CEILING_SOURCE_PINS),
            "oai_ceiling_derivation": dict(OAI_CEILING_DERIVATION),
            "normalization_clip_is_not_a_support_check":
                NORMALIZATION_CLIP_IS_NOT_A_SUPPORT_CHECK,
            "mapping_report": backlog_mapping_report(),
            "rejected_legacy_scale": REJECTED_LEGACY_BACKLOG_SCALE,
            "zero_maps_to": ZERO_BACKLOG_MAPS_TO,
            "out_of_support_policy": BACKLOG_OUT_OF_SUPPORT_POLICY,
            "reported_mapping_bytes":
                list(BACKLOG_REPORTED_MAPPING_BYTES),
            "forbidden_additional_features":
                list(FORBIDDEN_ADDITIONAL_QUEUE_FEATURES),
        },
        "selection": {
            "protocol": SELECTION_PROTOCOL,
            "population": SELECTION_POPULATION,
            "open_ended_search_forbidden": OPEN_ENDED_SEARCH_FORBIDDEN,
        },
        "reward": {
            "deadline_ms": REWARD_DEADLINE_MS,
            "latency_weight": REWARD_LATENCY_WEIGHT,
            "failure_reward": REGISTERED_FAILURE_REWARD,
            "reward_error_per_ms": REWARD_ERROR_PER_MS,
        },
        "acceptance_gates": [
            {"number": g.number, "key": g.key, "target": g.target,
             "thresholds": dict(g.thresholds)} for g in ACCEPTANCE_GATES
        ],
    }


CONTRACT_V2_SHA256 = canonical_sha256(contract_document())
