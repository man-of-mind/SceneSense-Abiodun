"""Train-only exact-P95 penalty for the actual emitted float32 target.

V1 proved strict ordering in binary64, but its binding pair collapsed to the
same float32 replay value.  This superseding derivation pins the runtime
operation order and chooses the smallest non-negative finite binary64 penalty
whose *emitted* float32 value puts every positive-admission infeasible action
strictly below its context's emitted-float32 constrained winner.

This module enumerates registered training contexts only.  It does not query
fit-validation outcomes, train a policy, or make a live-SLA claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    FIXED_END_TO_FEEDBACK_STAGES_MS,
    PILOT_UTILITY_SPEC_SHA256,
    require_supported_action,
)
from .empirical_contextual_environment import EmpiricalOneStepEnvironmentV1
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_p95_hinge_selection import (
    BUDGET_MS,
    EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS,
    EXPECTED_TRAIN_CONTEXT_COUNT,
    EXPECTED_TRAIN_SCENE_COUNT,
    MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
    MEAN_QUALITY_MINIMUM_RETENTION,
    TrainSelectionContextV1,
    _training_contexts,
)
from .empirical_contextual_split_oracle import (
    CONDITIONAL_FEASIBILITY_SEMANTICS,
    EXACT_ACTION_COUNT_PER_SCENE,
    OracleAuditError,
    OracleOutcome,
    _best_index,
    _better,
    _network_vector,
    _outcome_from_vectors,
    _project_root,
    _surface_mode_vector,
    enumerate_supported_actions,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .transaction_identity import canonical_sha256

__all__ = [
    "Float32ExactPenaltyDerivationError",
    "REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256",
    "base_p95_expected_utility64_v2",
    "derive_float32_exact_deadline_penalty_v2",
    "emitted_float32_target_v2",
    "float32_exact_penalty_spec_document_v2",
    "run_train_exact_p95_deadline_penalty_v2",
    "shaped_p95_expected_utility64_v2",
]


SCHEMA = "splitfusion.train_exact_p95_deadline_penalty.v2"
SPEC_SCHEMA = "splitfusion.exact_p95_deadline_penalty_spec.v2"
QUALITY_WEIGHT = 1.0
LATENCY_WEIGHT = 0.25
NON_ADMISSION_UTILITY = -1.0
OLD_FLOAT64_V1_DEADLINE_PENALTY = 0.5742957604173842
SUPERSEDED_V1_COMMIT = "a35ddb12d0fdfd5a3f7ad273c8021a08ed160394"
REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256 = (
    "9ea4e4a3d2ffa791ae189b1ff478871f48ca6d84572b2830a0a6795f2cd254e3"
)


class Float32ExactPenaltyDerivationError(ValueError):
    """The emitted-float32 construction cannot satisfy its contract."""


def float32_exact_penalty_spec_document_v2() -> Dict[str, Any]:
    return {
        "acceptance": {
            "admission_max_absolute_drop": MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
            "mean_quality_minimum_retention": MEAN_QUALITY_MINIMUM_RETENTION,
            "predecessor_must_be_insufficient": True,
            "shaped_winner_must_equal_emitted_float32_constrained_winner": True,
            "train_p95_budget_miss_count": 0,
        },
        "action_domain": "ALL_52240_EXECUTABLE_INTEGER_MODE_Q_E4_PAIRS",
        "arithmetic": {
            "base64": "p*(Q-0.25*(L95/200.0))+(1.0-p)*(-1.0)",
            "emitted_target": "torch.tensor(shaped64,dtype=torch.float32).item()",
            "infeasible_shaped64": "base64-p*lambda",
            "runtime_intermediate_dtype": "PYTHON_BINARY64",
        },
        "deadline_ms": BUDGET_MS,
        "derivation": (
            "ORDINAL_BINARY_SEARCH_FOR_SMALLEST_NONNEGATIVE_FINITE_BINARY64_"
            "LAMBDA_USING_ACTUAL_EMITTED_FLOAT32_STRICT_ORDERING"
        ),
        "latency_coordinate": (
            "FIXED_113_MS_PLUS_CONDITIONAL_RETAINED_SURVIVOR_P95"
        ),
        "latency_weight": LATENCY_WEIGHT,
        "non_admission": {
            "p_eq_0_emitted_target": -1.0,
            "p_eq_0_role": "EXCLUDED_FROM_CONDITIONAL_SURVIVOR_COMPETITION",
        },
        "old_v1": {
            "commit": SUPERSEDED_V1_COMMIT,
            "deadline_penalty": OLD_FLOAT64_V1_DEADLINE_PENALTY,
            "status": "SUPERSEDED_NO_GO_DUE_TO_FLOAT32_DTYPE_COLLAPSE",
        },
        "penalty_placement": "INSIDE_ADMITTED_BRANCH",
        "quality_component": DIRECT_QUALITY_COMPONENT,
        "quality_weight": QUALITY_WEIGHT,
        "schema": SPEC_SCHEMA,
        "selection_contexts": (
            "REGISTERED_TRAIN_SCENES_CROSSED_WITH_FOUR_AUTHORED_PROFILES"
        ),
        "tie_rule": (
            "MAX_EMITTED_FLOAT32_REWARD_THEN_MIN_P95_THEN_MIN_MODE_ID_"
            "THEN_MIN_Q_E4"
        ),
    }


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def base_p95_expected_utility64_v2(
    *, p_admit: float, q_perc: float, latency_p95_ms: float
) -> float:
    """Execute the registered binary64 base formula in its pinned order."""

    p = _finite_scalar("p_admit", p_admit)
    quality = _finite_scalar("q_perc", q_perc)
    latency = _finite_scalar("latency_p95_ms", latency_p95_ms)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_admit must lie in [0,1]")
    if not 0.0 <= quality <= 1.0:
        raise ValueError("q_perc must lie in [0,1]")
    if latency < 0.0:
        raise ValueError("latency_p95_ms must be non-negative")
    return p * (quality - 0.25 * (latency / 200.0)) + (1.0 - p) * (-1.0)


def shaped_p95_expected_utility64_v2(
    *,
    p_admit: float,
    q_perc: float,
    latency_p95_ms: float,
    deadline_penalty: float,
) -> float:
    """Execute the pinned base formula and then its conditional subtraction."""

    penalty = _finite_scalar("deadline_penalty", deadline_penalty)
    if penalty < 0.0:
        raise ValueError("deadline_penalty must be non-negative")
    base64 = base_p95_expected_utility64_v2(
        p_admit=p_admit,
        q_perc=q_perc,
        latency_p95_ms=latency_p95_ms,
    )
    if float(latency_p95_ms) > 200.0:
        return base64 - float(p_admit) * penalty
    return base64


def emitted_float32_target_v2(value64: float) -> float:
    """Return exactly the scalar target emitted to the float32 replay path."""

    value = _finite_scalar("value64", value64)
    return float(torch.tensor(value, dtype=torch.float32).item())


def _float64_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _float64_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", int(bits)))[0]


def _float32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def _float32_ordered_ordinal(value: float) -> int:
    bits = _float32_bits(value)
    return ((~bits) & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def _float32_ulp_margin(higher: float, lower: float) -> int:
    margin = _float32_ordered_ordinal(higher) - _float32_ordered_ordinal(lower)
    if margin < 0:
        raise OracleAuditError("float32 ULP margin has reversed ordering")
    return margin


def _emitted_vector(values64: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values64, dtype=np.float64)
    result = torch.tensor(contiguous, dtype=torch.float32).numpy()
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("emitted float32 target vector is non-finite")
    return result


def _base64_vector(
    p_admit: np.ndarray, quality: np.ndarray, latency_p95: np.ndarray
) -> np.ndarray:
    # Keep these ufuncs separated to pin the same order as the scalar formula.
    normalized = latency_p95 / 200.0
    weighted_latency = 0.25 * normalized
    admitted = quality - weighted_latency
    admitted_term = p_admit * admitted
    failure_term = (1.0 - p_admit) * (-1.0)
    result = admitted_term + failure_term
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("binary64 base vector contains non-finite values")
    return result


def _shaped64_vector(
    base64: np.ndarray,
    p_admit: np.ndarray,
    latency_p95: np.ndarray,
    penalty: float,
) -> np.ndarray:
    result = base64.copy()
    infeasible = latency_p95 > 200.0
    result[infeasible] = base64[infeasible] - p_admit[infeasible] * penalty
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("binary64 shaped vector contains non-finite values")
    return result


def _materialize_requirements(
    requirements: Iterable[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = tuple(requirements)
    if not rows:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    targets = np.asarray([row[0] for row in rows], dtype=np.float32)
    bases = np.asarray([row[1] for row in rows], dtype=np.float64)
    probabilities = np.asarray([row[2] for row in rows], dtype=np.float64)
    if (
        not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(bases))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise Float32ExactPenaltyDerivationError(
            "requirements must contain finite targets/bases and p in (0,1]"
        )
    return targets, bases, probabilities


def _derive_from_arrays(
    targets: np.ndarray,
    bases: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, int, float, bool]:
    """Return penalty, predicate evaluations, predecessor, insufficiency."""

    if len(targets) == 0:
        return 0.0, 1, 0.0, False

    def sufficient(candidate: float) -> bool:
        emitted = _emitted_vector(bases - probabilities * candidate)
        return bool(np.all(emitted < targets))

    evaluations = 1
    if sufficient(0.0):
        return 0.0, evaluations, 0.0, False

    # This is only a bracket.  The authoritative decision is the cast-aware
    # predicate above, searched over the ordinal binary64 representation.
    ratios = np.maximum(0.0, (bases - targets.astype(np.float64)) / probabilities)
    upper = float(np.max(ratios))
    if upper == 0.0:
        upper = math.nextafter(0.0, math.inf)
    evaluations += 1
    while not sufficient(upper):
        upper *= 2.0
        if upper == 0.0:
            upper = math.nextafter(0.0, math.inf)
        if not math.isfinite(upper):
            raise Float32ExactPenaltyDerivationError(
                "no finite binary64 penalty establishes emitted-float32 ordering"
            )
        evaluations += 1

    lower_bits = _float64_bits(0.0)
    upper_bits = _float64_bits(upper)
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        middle = _float64_from_bits(middle_bits)
        evaluations += 1
        if sufficient(middle):
            upper_bits = middle_bits
        else:
            lower_bits = middle_bits
    penalty = _float64_from_bits(upper_bits)
    predecessor = _float64_from_bits(upper_bits - 1)
    evaluations += 2
    if not sufficient(penalty):
        raise Float32ExactPenaltyDerivationError(
            "ordinal search failed the emitted-float32 postcondition"
        )
    predecessor_insufficient = not sufficient(predecessor)
    if not predecessor_insufficient:
        raise Float32ExactPenaltyDerivationError(
            "ordinal search did not return the smallest binary64 penalty"
        )
    return penalty, evaluations, predecessor, predecessor_insufficient


def derive_float32_exact_deadline_penalty_v2(
    requirements: Iterable[Tuple[float, float, float]],
) -> Tuple[float, int, float, bool]:
    """Derive the smallest non-negative finite binary64 cast-aware penalty.

    Each tuple is ``(emitted_float32_constrained_reward, infeasible_base64,
    p_admit)``.  Zero-admission outcomes are deliberately not accepted here:
    they emit exactly -1 and are outside conditional-survivor competition.
    """

    targets, bases, probabilities = _materialize_requirements(requirements)
    return _derive_from_arrays(targets, bases, probabilities)


def _best_across_modes_v2(
    surfaces: Sequence[Mapping[str, np.ndarray]],
    networks: Sequence[Mapping[str, np.ndarray]],
    *,
    feasible_only: bool,
    penalty: Optional[float] = None,
    emitted_float32: bool = True,
) -> Optional[OracleOutcome]:
    best: Optional[OracleOutcome] = None
    for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
        base64 = _base64_vector(
            network["p_admit"], surface["quality"], network["p95"]
        )
        objective64 = (
            base64
            if penalty is None
            else _shaped64_vector(
                base64, network["p_admit"], network["p95"], penalty
            )
        )
        objective = _emitted_vector(objective64) if emitted_float32 else objective64
        admitted = network["p_admit"] > 0.0
        feasible = admitted & (network["p95"] <= 200.0)
        mask = feasible if feasible_only else admitted
        index = _best_index(objective, network["p95"], mask)
        if index is None:
            continue
        candidate = _outcome_from_vectors(
            mode_id, index, surface, network, objective
        )
        if _better(candidate, best, objective="reward", tie_percentile="p95"):
            best = candidate
    return best


def _context_vectors(
    environment: EmpiricalOneStepEnvironmentV1,
    context: TrainSelectionContextV1,
) -> Tuple[list[Mapping[str, np.ndarray]], list[Mapping[str, np.ndarray]]]:
    surfaces = [
        _surface_mode_vector(environment._surface, context.sample_id, mode_id)
        for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS))
    ]
    networks = [
        _network_vector(
            environment._network,
            context.network_profile,
            surface["payload"],
            surface["datagrams"],
        )
        for surface in surfaces
    ]
    return surfaces, networks


def _scalar_outcome_v2(
    environment: EmpiricalOneStepEnvironmentV1,
    context: TrainSelectionContextV1,
    mode_id: int,
    q_e4: int,
    penalty: float,
) -> Tuple[OracleOutcome, float, float]:
    require_supported_action(mode_id, q_e4)
    query = environment._surface.query_fit_q_e4(context.sample_id, mode_id, q_e4)
    component = query.policy.component(DIRECT_QUALITY_COMPONENT)
    if not component.valid or component.value is None:
        raise OracleAuditError("v2 scalar quality is undefined")
    payload = float(query.policy.payload.total_transmitted_bytes)
    datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
    prediction = environment._prediction_session.predict(
        network_profile=context.network_profile,
        payload_bytes=payload,
        datagram_count=datagrams,
    )
    latency = prediction.conditional_retained_survivor_latency_model()
    fixed = sum(value for _name, value in FIXED_END_TO_FEEDBACK_STAGES_MS)
    p_admit = prediction.p_edge_admission_given_sent
    latency_p95 = fixed + latency.p95_ms
    base64 = base_p95_expected_utility64_v2(
        p_admit=p_admit,
        q_perc=float(component.value),
        latency_p95_ms=latency_p95,
    )
    shaped64 = shaped_p95_expected_utility64_v2(
        p_admit=p_admit,
        q_perc=float(component.value),
        latency_p95_ms=latency_p95,
        deadline_penalty=penalty,
    )
    emitted = emitted_float32_target_v2(shaped64)
    return (
        OracleOutcome(
            mode_id=mode_id,
            q_e4=q_e4,
            q_perc=float(component.value),
            total_transmitted_bytes=payload,
            datagram_count=datagrams,
            p_edge_admission_given_sent=p_admit,
            latency_proxy_p50_ms=fixed + latency.p50_ms,
            latency_proxy_p95_ms=latency_p95,
            latency_proxy_p99_ms=fixed + latency.p99_ms,
            reward=emitted,
        ),
        base64,
        shaped64,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise OracleAuditError("CSV requires at least one row")
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _nearest_rank_distribution(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        raise OracleAuditError("ULP distribution requires at least one value")
    ordered = sorted(int(value) for value in values)

    def percentile(probability: float) -> int:
        if probability == 0.0:
            return ordered[0]
        rank = max(1, math.ceil(probability * len(ordered)))
        return ordered[rank - 1]

    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "nearest_rank": {
            "p0": percentile(0.0),
            "p25": percentile(0.25),
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "p100": percentile(1.0),
        },
        "one_ulp_count": sum(value == 1 for value in ordered),
        "at_most_2_ulp_count": sum(value <= 2 for value in ordered),
        "at_most_4_ulp_count": sum(value <= 4 for value in ordered),
    }


def _report(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    constrained = summary["emitted_float32_constrained_oracle"]
    baseline = summary["emitted_float32_unconstrained_oracle"]
    collision = summary["old_float64_v1_collision"]
    ulp = summary["float32_ulp_margin_distribution"]["nearest_rank"]
    return "\n".join(
        [
            "# Train-only emitted-float32 exact P95 deadline penalty v2",
            "",
            "V1 is superseded: its binary64 strict inequality collapsed to an",
            "equal float32 learning target. V2 derives the coefficient against",
            "the exact CPU float32 scalar emitted after the pinned binary64 formula.",
            "Only the 391 registered training scene IDs were queried.",
            "",
            f"- Decision: **{decision['status']}**",
            f"- Smallest binary64 lambda: `{decision['deadline_penalty']!r}`",
            f"- Lambda hex/bits: `{decision['deadline_penalty_float_hex']}` / `{decision['deadline_penalty_uint64_hex']}`",
            f"- Immediate predecessor insufficient: `{decision['predecessor_insufficient']}`",
            f"- Strict infeasible ordering violations: `{decision['strict_ordering_violation_count']}`",
            f"- Shaped/constrained identity matches: `{decision['winner_identity_match_count']}/{EXPECTED_TRAIN_CONTEXT_COUNT}`",
            f"- Train P95 misses: `{constrained['p95_miss_count']}/{EXPECTED_TRAIN_CONTEXT_COUNT}`",
            f"- Mean quality: `{baseline['mean_q_perc']:.6f}` -> `{constrained['mean_q_perc']:.6f}` ({constrained['quality_retention']:.3%})",
            f"- Mean admission: `{baseline['mean_p_admit']:.6f}` -> `{constrained['mean_p_admit']:.6f}` ({constrained['admission_change']:+.6f})",
            f"- Old-lambda emitted collisions: `{collision['collision_count']}` (binding float32 bits `{collision['constrained_emitted_float32_bits_hex']}`)",
            f"- Context-worst ULP margins p0/p50/p95/p100: `{ulp['p0']}/{ulp['p50']}/{ulp['p95']}/{ulp['p100']}`",
            "",
            "This is a modeled conditional-retained-survivor train-support proof,",
            "not a live 200-ms SLA. No validation outcome was queried and no policy",
            "training was run by this derivation.",
            "",
        ]
    )


def run_train_exact_p95_deadline_penalty_v2(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    observed_spec_hash = canonical_sha256(float32_exact_penalty_spec_document_v2())
    if observed_spec_hash != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256:
        raise OracleAuditError(
            f"float32 exact-P95 penalty specification hash drift: {observed_spec_hash}"
        )
    enumerate_supported_actions()
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    cuda_before = torch.cuda.is_initialized()
    partition = load_registered_empirical_fit_partition(project_root=root)
    contexts = _training_contexts(partition)
    train_scene_ids = tuple(
        sorted(
            row.sample_id
            for row in partition.scene_assignments
            if row.split == TRAIN_SPLIT
        )
    )
    validation_scene_ids = frozenset(
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    )
    if len(train_scene_ids) != EXPECTED_TRAIN_SCENE_COUNT:
        raise OracleAuditError("v2 train scene-ID count drift")
    validation_intersection = sorted(set(train_scene_ids) & validation_scene_ids)
    if validation_intersection:
        raise OracleAuditError("v2 train derivation intersects fit-validation IDs")

    environment = EmpiricalOneStepEnvironmentV1.load_registered(
        seed=0, project_root=root
    )
    first_pass: list[Dict[str, Any]] = []
    target_chunks: list[np.ndarray] = []
    base_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    context_chunks: list[np.ndarray] = []
    mode_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    first_pass_evaluations = 0
    try:
        # Pass 1: derive each emitted-float32 constrained winner, enumerate every
        # active infeasible requirement, then ordinal-search binary64 lambda.
        for context in contexts:
            surfaces, networks = _context_vectors(environment, context)
            first_pass_evaluations += sum(len(surface["q"]) for surface in surfaces)
            unconstrained = _best_across_modes_v2(
                surfaces, networks, feasible_only=False
            )
            constrained = _best_across_modes_v2(
                surfaces, networks, feasible_only=True
            )
            float64_v1_unconstrained = _best_across_modes_v2(
                surfaces,
                networks,
                feasible_only=False,
                emitted_float32=False,
            )
            float64_v1_constrained = _best_across_modes_v2(
                surfaces,
                networks,
                feasible_only=True,
                emitted_float32=False,
            )
            float64_v1_shaped = _best_across_modes_v2(
                surfaces,
                networks,
                feasible_only=False,
                penalty=OLD_FLOAT64_V1_DEADLINE_PENALTY,
                emitted_float32=False,
            )
            if any(
                outcome is None
                for outcome in (
                    unconstrained,
                    constrained,
                    float64_v1_unconstrained,
                    float64_v1_constrained,
                    float64_v1_shaped,
                )
            ):
                raise Float32ExactPenaltyDerivationError(
                    f"context {context.context_index} has no admitted feasible action"
                )
            assert unconstrained is not None
            assert constrained is not None
            assert float64_v1_unconstrained is not None
            assert float64_v1_constrained is not None
            assert float64_v1_shaped is not None
            for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
                base64 = _base64_vector(
                    network["p_admit"], surface["quality"], network["p95"]
                )
                base32 = _emitted_vector(base64)
                active = (
                    (network["p_admit"] > 0.0)
                    & (network["p95"] > 200.0)
                    & (base32 >= np.float32(constrained.reward))
                )
                indices = np.flatnonzero(active)
                if len(indices):
                    count = len(indices)
                    target_chunks.append(
                        np.full(count, constrained.reward, dtype=np.float32)
                    )
                    base_chunks.append(base64[indices].astype(np.float64, copy=True))
                    probability_chunks.append(
                        network["p_admit"][indices].astype(np.float64, copy=True)
                    )
                    context_chunks.append(
                        np.full(count, context.context_index, dtype=np.int32)
                    )
                    mode_chunks.append(np.full(count, mode_id, dtype=np.int16))
                    index_chunks.append(indices.astype(np.int32, copy=True))
            first_pass.append(
                {
                    "context": context,
                    "unconstrained": unconstrained,
                    "constrained": constrained,
                    "float64_v1_unconstrained": float64_v1_unconstrained,
                    "float64_v1_constrained": float64_v1_constrained,
                    "float64_v1_shaped": float64_v1_shaped,
                }
            )

        if first_pass_evaluations != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
            raise OracleAuditError("v2 first-pass action-context count drift")
        if not target_chunks:
            raise Float32ExactPenaltyDerivationError(
                "v2 expected at least one active infeasible requirement"
            )
        targets = np.concatenate(target_chunks)
        bases = np.concatenate(base_chunks)
        probabilities = np.concatenate(probability_chunks)
        requirement_contexts = np.concatenate(context_chunks)
        requirement_modes = np.concatenate(mode_chunks)
        requirement_indices = np.concatenate(index_chunks)
        penalty, search_evaluations, predecessor, predecessor_insufficient = (
            _derive_from_arrays(targets, bases, probabilities)
        )

        predecessor_emitted = _emitted_vector(bases - probabilities * predecessor)
        predecessor_failures = np.flatnonzero(predecessor_emitted >= targets)
        if len(predecessor_failures) == 0:
            raise OracleAuditError("v2 predecessor insufficiency witness disappeared")
        binding_requirement_index = int(predecessor_failures[0])
        binding_context_index = int(requirement_contexts[binding_requirement_index])
        binding_mode_id = int(requirement_modes[binding_requirement_index])
        binding_surface_index = int(requirement_indices[binding_requirement_index])
        binding_q_e4 = int(
            MODELED_SMOKE_MODE_Q_E4_BOUNDS[binding_mode_id][0]
            + binding_surface_index
        )

        rows: list[Dict[str, Any]] = []
        identity_matches = 0
        strict_ordering_violations = 0
        strict_infeasible_comparisons = 0
        predecessor_violation_count = 0
        old_collision_count = 0
        zero_admission_action_count = 0
        scalar_constrained_revalidated = 0
        scalar_shaped_revalidated = 0
        second_pass_evaluations = 0
        context_worst_ulp_margins: list[int] = []
        old_collision_witness: Optional[Dict[str, Any]] = None

        # Pass 2: independently rebuild all vectors, prove strict ordering for
        # every positive-admission infeasible action, and revalidate winners.
        for saved in first_pass:
            context = saved["context"]
            constrained = saved["constrained"]
            surfaces, networks = _context_vectors(environment, context)
            second_pass_evaluations += sum(len(surface["q"]) for surface in surfaces)
            shaped = _best_across_modes_v2(
                surfaces,
                networks,
                feasible_only=False,
                penalty=penalty,
            )
            if shaped is None:
                raise OracleAuditError("v2 shaped oracle found no admitted action")
            context_max_infeasible = -math.inf
            for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
                base64 = _base64_vector(
                    network["p_admit"], surface["quality"], network["p95"]
                )
                p_zero = network["p_admit"] == 0.0
                if np.any(p_zero):
                    zero_admission_action_count += int(np.count_nonzero(p_zero))
                    if not np.all(base64[p_zero] == -1.0):
                        raise OracleAuditError("p=0 base64 target is not exactly -1")
                    if not np.all(_emitted_vector(base64[p_zero]) == np.float32(-1.0)):
                        raise OracleAuditError("p=0 emitted target is not exactly -1")
                infeasible = (network["p_admit"] > 0.0) & (network["p95"] > 200.0)
                if not np.any(infeasible):
                    continue
                new64 = base64[infeasible] - network["p_admit"][infeasible] * penalty
                new32 = _emitted_vector(new64)
                strict_infeasible_comparisons += len(new32)
                strict_ordering_violations += int(
                    np.count_nonzero(new32 >= np.float32(constrained.reward))
                )
                context_max_infeasible = max(
                    context_max_infeasible, float(np.max(new32))
                )
                predecessor32 = _emitted_vector(
                    base64[infeasible]
                    - network["p_admit"][infeasible] * predecessor
                )
                predecessor_violation_count += int(
                    np.count_nonzero(
                        predecessor32 >= np.float32(constrained.reward)
                    )
                )
                old64 = (
                    base64[infeasible]
                    - network["p_admit"][infeasible]
                    * OLD_FLOAT64_V1_DEADLINE_PENALTY
                )
                old32 = _emitted_vector(old64)
                collisions = np.flatnonzero(
                    old32 == np.float32(constrained.reward)
                )
                old_collision_count += len(collisions)
                if len(collisions) and (
                    old_collision_witness is None
                    or (
                        context.context_index == 938
                        and mode_id == 2
                        and np.any(surface["q"][np.flatnonzero(infeasible)[collisions]] == 9000)
                    )
                ):
                    local_infeasible = np.flatnonzero(infeasible)
                    matching = [
                        int(value)
                        for value in collisions
                        if (
                            context.context_index == 938
                            and mode_id == 2
                            and int(surface["q"][local_infeasible[int(value)]]) == 9000
                        )
                    ]
                    chosen_collision = matching[0] if matching else int(collisions[0])
                    source_index = int(local_infeasible[chosen_collision])
                    old_collision_witness = {
                        "context_index": context.context_index,
                        "mode_id": mode_id,
                        "q_e4": int(surface["q"][source_index]),
                    }
            if not math.isfinite(context_max_infeasible):
                raise OracleAuditError("v2 context has no infeasible comparison")
            context_ulp_margin = _float32_ulp_margin(
                constrained.reward, context_max_infeasible
            )
            context_worst_ulp_margins.append(context_ulp_margin)

            constrained_scalar, constrained_base64, _ = _scalar_outcome_v2(
                environment,
                context,
                constrained.mode_id,
                constrained.q_e4,
                0.0,
            )
            if constrained_scalar.reward != constrained.reward:
                raise OracleAuditError(
                    "v2 constrained scalar/vector emitted target mismatch"
                )
            scalar_constrained_revalidated += 1
            shaped_scalar, shaped_base64, shaped64 = _scalar_outcome_v2(
                environment,
                context,
                shaped.mode_id,
                shaped.q_e4,
                penalty,
            )
            if shaped_scalar.reward != shaped.reward:
                raise OracleAuditError("v2 shaped scalar/vector target mismatch")
            scalar_shaped_revalidated += 1
            match = (shaped.mode_id, shaped.q_e4) == (
                constrained.mode_id,
                constrained.q_e4,
            )
            identity_matches += int(match)
            rows.append(
                {
                    **asdict(context),
                    "unconstrained_mode_id": saved["unconstrained"].mode_id,
                    "unconstrained_q_e4": saved["unconstrained"].q_e4,
                    "unconstrained_q_perc": saved["unconstrained"].q_perc,
                    "unconstrained_p_admit": saved["unconstrained"].p_edge_admission_given_sent,
                    "unconstrained_latency_p95_ms": saved["unconstrained"].latency_proxy_p95_ms,
                    "unconstrained_emitted_float32_base_reward": saved["unconstrained"].reward,
                    "constrained_mode_id": constrained.mode_id,
                    "constrained_q_e4": constrained.q_e4,
                    "constrained_q_perc": constrained.q_perc,
                    "constrained_p_admit": constrained.p_edge_admission_given_sent,
                    "constrained_latency_p95_ms": constrained.latency_proxy_p95_ms,
                    "constrained_base64_reward": constrained_base64,
                    "constrained_emitted_float32_reward": constrained.reward,
                    "shaped_mode_id": shaped.mode_id,
                    "shaped_q_e4": shaped.q_e4,
                    "shaped_base64_reward": shaped_base64,
                    "shaped64_reward": shaped64,
                    "shaped_emitted_float32_reward": shaped_scalar.reward,
                    "shaped_matches_constrained": match,
                    "context_worst_infeasible_ulp_margin": context_ulp_margin,
                    "float64_v1_constrained_mode_id": saved["float64_v1_constrained"].mode_id,
                    "float64_v1_constrained_q_e4": saved["float64_v1_constrained"].q_e4,
                }
            )

        if second_pass_evaluations != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
            raise OracleAuditError("v2 second-pass action-context count drift")
        if old_collision_witness is None:
            raise OracleAuditError("v2 failed to reproduce old-lambda collision")

        binding_context = contexts[binding_context_index]
        binding_constrained = first_pass[binding_context_index]["constrained"]
        binding_feasible_scalar, binding_feasible_base64, _ = _scalar_outcome_v2(
            environment,
            binding_context,
            binding_constrained.mode_id,
            binding_constrained.q_e4,
            0.0,
        )
        binding_infeasible_scalar, binding_infeasible_base64, binding_new64 = (
            _scalar_outcome_v2(
                environment,
                binding_context,
                binding_mode_id,
                binding_q_e4,
                penalty,
            )
        )
        _, _, binding_predecessor64 = _scalar_outcome_v2(
            environment,
            binding_context,
            binding_mode_id,
            binding_q_e4,
            predecessor,
        )

        old_context = contexts[int(old_collision_witness["context_index"])]
        old_constrained = first_pass[old_context.context_index]["constrained"]
        old_feasible_scalar, old_feasible_base64, _ = _scalar_outcome_v2(
            environment,
            old_context,
            old_constrained.mode_id,
            old_constrained.q_e4,
            0.0,
        )
        old_infeasible_scalar, old_infeasible_base64, old_shaped64 = (
            _scalar_outcome_v2(
                environment,
                old_context,
                int(old_collision_witness["mode_id"]),
                int(old_collision_witness["q_e4"]),
                OLD_FLOAT64_V1_DEADLINE_PENALTY,
            )
        )
    finally:
        environment.close()

    if not cuda_before and torch.cuda.is_initialized():
        raise OracleAuditError("CPU-only v2 screen initialized CUDA")

    unconstrained_quality = mean(row["unconstrained"].q_perc for row in first_pass)
    unconstrained_admission = mean(
        row["unconstrained"].p_edge_admission_given_sent for row in first_pass
    )
    unconstrained_misses = sum(
        row["unconstrained"].latency_proxy_p95_ms > 200.0 for row in first_pass
    )
    constrained_quality = mean(row["constrained"].q_perc for row in first_pass)
    constrained_admission = mean(
        row["constrained"].p_edge_admission_given_sent for row in first_pass
    )
    constrained_misses = sum(
        row["constrained"].latency_proxy_p95_ms > 200.0 for row in first_pass
    )
    quality_retention = constrained_quality / unconstrained_quality
    admission_change = constrained_admission - unconstrained_admission
    float64_v1_constrained_matches = sum(
        (row["constrained"].mode_id, row["constrained"].q_e4)
        == (
            row["float64_v1_constrained"].mode_id,
            row["float64_v1_constrained"].q_e4,
        )
        for row in first_pass
    )
    float64_v1_shaped_matches_its_constrained = sum(
        (row["float64_v1_shaped"].mode_id, row["float64_v1_shaped"].q_e4)
        == (
            row["float64_v1_constrained"].mode_id,
            row["float64_v1_constrained"].q_e4,
        )
        for row in first_pass
    )
    criteria = {
        "admission_drop_le_0_001": (
            admission_change >= -MEAN_ADMISSION_MAX_ABSOLUTE_DROP
        ),
        "mean_quality_retention_ge_0_95": (
            quality_retention >= MEAN_QUALITY_MINIMUM_RETENTION
        ),
        "predecessor_is_insufficient": (
            predecessor_insufficient and predecessor_violation_count > 0
        ),
        "shaped_matches_emitted_float32_constrained": (
            identity_matches == EXPECTED_TRAIN_CONTEXT_COUNT
        ),
        "strict_emitted_float32_ordering_for_every_infeasible_action": (
            strict_ordering_violations == 0
        ),
        "zero_train_p95_misses": constrained_misses == 0,
    }
    status = "GO" if all(criteria.values()) else "NO_GO"
    decision = {
        "criteria": criteria,
        "deadline_penalty": penalty,
        "deadline_penalty_float_hex": penalty.hex(),
        "deadline_penalty_uint64_hex": f"0x{_float64_bits(penalty):016x}",
        "predecessor": predecessor,
        "predecessor_float_hex": predecessor.hex(),
        "predecessor_insufficient": predecessor_insufficient,
        "predecessor_violation_count": predecessor_violation_count,
        "representable_search_evaluations": search_evaluations,
        "status": status,
        "strict_ordering_violation_count": strict_ordering_violations,
        "winner_identity_match_count": identity_matches,
    }
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_TRAIN_ONLY_EMITTED_FLOAT32_EXACT_P95_PENALTY_V2",
        "decision": decision,
        "runtime_arithmetic": {
            "base64": "p*(Q-0.25*(L95/200.0))+(1.0-p)*(-1.0)",
            "emitted_target": "torch.tensor(shaped64,dtype=torch.float32).item()",
            "infeasible_shaped64": "base64-p*lambda",
        },
        "binding_requirement": {
            "context_index": binding_context_index,
            "sample_id": binding_context.sample_id,
            "network_profile": binding_context.network_profile,
            "constrained_mode_id": binding_constrained.mode_id,
            "constrained_q_e4": binding_constrained.q_e4,
            "infeasible_mode_id": binding_mode_id,
            "infeasible_q_e4": binding_q_e4,
            "p_admit": binding_infeasible_scalar.p_edge_admission_given_sent,
            "latency_p95_ms": binding_infeasible_scalar.latency_proxy_p95_ms,
            "constrained_base64": binding_feasible_base64,
            "constrained_emitted_float32": binding_feasible_scalar.reward,
            "constrained_emitted_float32_bits_hex": f"0x{_float32_bits(binding_feasible_scalar.reward):08x}",
            "infeasible_base64": binding_infeasible_base64,
            "predecessor_shaped64": binding_predecessor64,
            "predecessor_emitted_float32": emitted_float32_target_v2(binding_predecessor64),
            "predecessor_emitted_float32_bits_hex": f"0x{_float32_bits(emitted_float32_target_v2(binding_predecessor64)):08x}",
            "selected_shaped64": binding_new64,
            "selected_emitted_float32": binding_infeasible_scalar.reward,
            "selected_emitted_float32_bits_hex": f"0x{_float32_bits(binding_infeasible_scalar.reward):08x}",
            "selected_ulp_margin": _float32_ulp_margin(
                binding_feasible_scalar.reward, binding_infeasible_scalar.reward
            ),
        },
        "old_float64_v1_collision": {
            "status": "SUPERSEDED_NO_GO_DUE_TO_FLOAT32_DTYPE_COLLAPSE",
            "superseded_commit": SUPERSEDED_V1_COMMIT,
            "deadline_penalty": OLD_FLOAT64_V1_DEADLINE_PENALTY,
            "deadline_penalty_float_hex": OLD_FLOAT64_V1_DEADLINE_PENALTY.hex(),
            "collision_count": old_collision_count,
            "context_index": old_context.context_index,
            "sample_id": old_context.sample_id,
            "network_profile": old_context.network_profile,
            "constrained_mode_id": old_constrained.mode_id,
            "constrained_q_e4": old_constrained.q_e4,
            "infeasible_mode_id": int(old_collision_witness["mode_id"]),
            "infeasible_q_e4": int(old_collision_witness["q_e4"]),
            "constrained_base64": old_feasible_base64,
            "infeasible_base64": old_infeasible_base64,
            "infeasible_shaped64": old_shaped64,
            "float64_strict_ordering": old_shaped64 < old_feasible_base64,
            "constrained_emitted_float32": old_feasible_scalar.reward,
            "infeasible_emitted_float32": old_infeasible_scalar.reward,
            "constrained_emitted_float32_bits_hex": f"0x{_float32_bits(old_feasible_scalar.reward):08x}",
            "infeasible_emitted_float32_bits_hex": f"0x{_float32_bits(old_infeasible_scalar.reward):08x}",
            "emitted_float32_strict_ordering": (
                old_infeasible_scalar.reward < old_feasible_scalar.reward
            ),
            "emitted_float32_collision": (
                old_infeasible_scalar.reward == old_feasible_scalar.reward
            ),
        },
        "emitted_float32_unconstrained_oracle": {
            "mean_q_perc": unconstrained_quality,
            "mean_p_admit": unconstrained_admission,
            "p95_miss_count": unconstrained_misses,
        },
        "emitted_float32_constrained_oracle": {
            "mean_q_perc": constrained_quality,
            "mean_p_admit": constrained_admission,
            "p95_miss_count": constrained_misses,
            "quality_retention": quality_retention,
            "admission_change": admission_change,
        },
        "float32_ulp_margin_distribution": {
            "population": "CONTEXT_WORST_INFEASIBLE_EMITTED_TARGET",
            "definition": (
                "ORDERED_FLOAT32_ORDINAL_OF_CONSTRAINED_WINNER_MINUS_"
                "ORDERED_FLOAT32_ORDINAL_OF_MAX_INFEASIBLE_TARGET"
            ),
            **_nearest_rank_distribution(context_worst_ulp_margins),
        },
        "float64_v1_winner_identity_comparison": {
            "emitted_float32_v2_constrained_equals_float64_v1_constrained_count": (
                float64_v1_constrained_matches
            ),
            "identity_difference_count": (
                EXPECTED_TRAIN_CONTEXT_COUNT - float64_v1_constrained_matches
            ),
            "float64_v1_shaped_equals_float64_v1_constrained_count": (
                float64_v1_shaped_matches_its_constrained
            ),
        },
        "exactness": {
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "first_pass_action_context_evaluations": first_pass_evaluations,
            "second_pass_action_context_evaluations": second_pass_evaluations,
            "active_derivation_requirement_count": len(targets),
            "strict_infeasible_comparison_count": strict_infeasible_comparisons,
            "scalar_constrained_revalidated_winner_count": scalar_constrained_revalidated,
            "scalar_shaped_revalidated_winner_count": scalar_shaped_revalidated,
            "train_context_count": EXPECTED_TRAIN_CONTEXT_COUNT,
            "train_scene_count": EXPECTED_TRAIN_SCENE_COUNT,
            "zero_admission_action_context_count": zero_admission_action_count,
        },
        "bindings": {
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "original_d1_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "float32_exact_penalty_spec_sha256": (
                REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
            ),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "scope": {
            "context_selection": "TRAIN_IDS_ONLY",
            "registered_bundle_load": (
                "WHOLE_REGISTERED_BUNDLES_INTEGRITY_AUDITED_AT_LOAD"
            ),
            "claims_excluded": [
                "POLICY_TRAINING",
                "VALIDATION_OR_TEST_SELECTION",
                "LIVE_200_MS_SLA",
                "EMPIRICAL_TIMEOUT_PROBABILITY",
            ],
            "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
            "p_eq_0_semantics": (
                "EMITS_EXACTLY_MINUS_ONE_AND_EXCLUDED_FROM_"
                "CONDITIONAL_SURVIVOR_COMPETITION"
            ),
            "queried_train_scene_id_count": len(train_scene_ids),
            "queried_train_scene_id_sha256": canonical_sha256(
                list(train_scene_ids)
            ),
            "fit_validation_scene_id_intersection_count": len(
                validation_intersection
            ),
            "fit_validation_outcome_query_count": 0,
        },
    }
    decision_payload = {
        "criteria": criteria,
        "deadline_penalty": penalty,
        "deadline_penalty_float_hex": penalty.hex(),
        "deadline_penalty_uint64_hex": f"0x{_float64_bits(penalty):016x}",
        "predecessor": predecessor,
        "predecessor_insufficient": predecessor_insufficient,
        "status": status,
    }
    payloads = {
        "train_context_oracles_v2.csv": _csv_bytes(rows),
        "selection_decision_v2.json": (
            json.dumps(decision_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    }
    payloads["REPORT_v2.md"] = _report(summary).encode()
    summary["files"] = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    payloads["summary_v2.json"] = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    for name, payload in payloads.items():
        _atomic_bytes(destination / name, payload)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_train_exact_p95_deadline_penalty_v2(output_dir=args.output)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
