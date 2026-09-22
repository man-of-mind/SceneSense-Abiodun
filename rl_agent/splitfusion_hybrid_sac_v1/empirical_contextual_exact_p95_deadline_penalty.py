"""Train-only exact P95 deadline-penalty preflight.

The penalty is derived analytically from the registered training support; it
is not selected from a coefficient grid.  Conditional P95 is a statistic of
admitted, retained survivors, so the deadline penalty belongs inside the
admitted outcome branch::

    R = p * (Q - 0.25 * L95 / 200 - lambda * 1[L95 > 200])
        + (1 - p) * (-1)

This module does not train a policy and makes no live-SLA claim.
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
    "ExactPenaltyDerivationError",
    "base_p95_expected_utility",
    "derive_exact_deadline_penalty",
    "exact_penalty_spec_document",
    "run_train_exact_p95_deadline_penalty",
    "shaped_p95_expected_utility",
]


SCHEMA = "splitfusion.train_exact_p95_deadline_penalty.v1"
QUALITY_WEIGHT = 1.0
LATENCY_WEIGHT = 0.25
NON_ADMISSION_UTILITY = -1.0
REGISTERED_EXACT_PENALTY_SPEC_SHA256 = (
    "5ea0c50dd384323004421fad076654dea2ab2cfeb54c6fd526237bf117a05c3c"
)


class ExactPenaltyDerivationError(ValueError):
    """The exact deadline penalty cannot satisfy the declared construction."""


def exact_penalty_spec_document() -> Dict[str, Any]:
    return {
        "acceptance": {
            "admission_max_absolute_drop": MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
            "mean_quality_minimum_retention": MEAN_QUALITY_MINIMUM_RETENTION,
            "shaped_winner_must_equal_constrained_base_winner": True,
            "train_p95_budget_miss_count": 0,
        },
        "deadline_ms": BUDGET_MS,
        "derivation": (
            "lambda=smallest_float64_strictly_above_max_positive_"
            "(infeasible_base-best_feasible_base)/p_admit; advance_by_"
            "nextafter_only_if_actual_float_subtraction_is_not_strict"
        ),
        "latency_coordinate": (
            "FIXED_113_MS_PLUS_CONDITIONAL_RETAINED_SURVIVOR_P95"
        ),
        "latency_weight": LATENCY_WEIGHT,
        "non_admission_utility": NON_ADMISSION_UTILITY,
        "penalty_placement": "INSIDE_ADMITTED_BRANCH",
        "pre_registration_status": (
            "ACCEPTANCE_CRITERIA_DECLARED_BEFORE_OFFICIAL_SCREEN;_"
            "CANONICAL_SPEC_HASH_PINNED_AFTER_IMPLEMENTATION"
        ),
        "quality_component": DIRECT_QUALITY_COMPONENT,
        "quality_weight": QUALITY_WEIGHT,
        "selection_contexts": (
            "REGISTERED_TRAIN_SCENES_CROSSED_WITH_FOUR_AUTHORED_PROFILES"
        ),
        "schema": "splitfusion.exact_p95_deadline_penalty_spec.v1",
    }


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def base_p95_expected_utility(
    *, p_admit: float, q_perc: float, latency_p95_ms: float
) -> float:
    p = _finite_scalar("p_admit", p_admit)
    quality = _finite_scalar("q_perc", q_perc)
    latency = _finite_scalar("latency_p95_ms", latency_p95_ms)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_admit must lie in [0,1]")
    if not 0.0 <= quality <= 1.0:
        raise ValueError("q_perc must lie in [0,1]")
    if latency < 0.0:
        raise ValueError("latency_p95_ms must be non-negative")
    admitted = QUALITY_WEIGHT * quality - LATENCY_WEIGHT * latency / BUDGET_MS
    return p * admitted + (1.0 - p) * NON_ADMISSION_UTILITY


def shaped_p95_expected_utility(
    *,
    p_admit: float,
    q_perc: float,
    latency_p95_ms: float,
    deadline_penalty: float,
) -> float:
    penalty = _finite_scalar("deadline_penalty", deadline_penalty)
    if penalty < 0.0:
        raise ValueError("deadline_penalty must be non-negative")
    base = base_p95_expected_utility(
        p_admit=p_admit, q_perc=q_perc, latency_p95_ms=latency_p95_ms
    )
    return base - float(p_admit) * penalty * float(latency_p95_ms > BUDGET_MS)


def derive_exact_deadline_penalty(
    requirements: Iterable[Tuple[float, float, float]],
) -> Tuple[float, float, int]:
    """Return ``(Delta, lambda, nextafter_steps)``.

    Each tuple is ``(best_feasible_base, infeasible_base, p_admit)``.  A
    zero-admission infeasible outcome cannot be shaped by an admitted-branch
    penalty; if it can beat the feasible optimum the construction fails.
    """

    materialized = tuple(requirements)
    if not materialized:
        return 0.0, math.nextafter(0.0, math.inf), 1
    ratios = []
    for best_feasible, infeasible, p_admit in materialized:
        feasible = _finite_scalar("best_feasible_base", best_feasible)
        candidate = _finite_scalar("infeasible_base", infeasible)
        p = _finite_scalar("p_admit", p_admit)
        if not 0.0 <= p <= 1.0:
            raise ExactPenaltyDerivationError("p_admit must lie in [0,1]")
        advantage = candidate - feasible
        if p == 0.0:
            # Conditional-survivor latency is not a meaningful outcome when
            # admission probability is zero.  Such actions are outside this
            # exact-penalty construction and retain their -1 failure utility.
            continue
        ratios.append(max(0.0, advantage / p))
    delta = max(ratios, default=0.0)
    active = tuple(
        row for row in materialized if row[2] > 0.0 and row[1] >= row[0]
    )

    def sufficient(candidate: float) -> bool:
        return all(
            infeasible - p * candidate < feasible
            for feasible, infeasible, p in active
        )

    evaluations = 0
    lower = delta
    upper = math.nextafter(delta, math.inf)
    evaluations += 1
    while not sufficient(upper):
        upper = upper * 2.0
        if upper == 0.0:
            upper = math.nextafter(0.0, math.inf)
        if not math.isfinite(upper):
            raise ExactPenaltyDerivationError(
                "no finite float64 penalty establishes strict preference"
            )
        evaluations += 1

    # Non-negative finite IEEE-754 doubles preserve numeric order in their
    # unsigned bit representation.  Binary-search that ordinal space for the
    # first value whose *executed* subtraction is strictly feasible-preferring.
    lower_bits = struct.unpack(">Q", struct.pack(">d", lower))[0]
    upper_bits = struct.unpack(">Q", struct.pack(">d", upper))[0]
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        middle = struct.unpack(">d", struct.pack(">Q", middle_bits))[0]
        evaluations += 1
        if sufficient(middle):
            upper_bits = middle_bits
        else:
            lower_bits = middle_bits
    penalty = struct.unpack(">d", struct.pack(">Q", upper_bits))[0]
    if not sufficient(penalty):
        raise ExactPenaltyDerivationError(
            "float64 exact-penalty search failed its postcondition"
        )
    return delta, penalty, evaluations


def _base_vector(
    p_admit: np.ndarray, quality: np.ndarray, latency_p95: np.ndarray
) -> np.ndarray:
    result = p_admit * (
        quality - LATENCY_WEIGHT * latency_p95 / BUDGET_MS
    ) + (1.0 - p_admit) * NON_ADMISSION_UTILITY
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("base P95 reward vector contains non-finite values")
    return result


def _best_across_modes(
    surfaces: Sequence[Mapping[str, np.ndarray]],
    networks: Sequence[Mapping[str, np.ndarray]],
    *,
    feasible_only: bool,
    penalty: Optional[float] = None,
) -> Optional[OracleOutcome]:
    best: Optional[OracleOutcome] = None
    best_feasible = False
    for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
        base = _base_vector(
            network["p_admit"], surface["quality"], network["p95"]
        )
        admitted_support = network["p_admit"] > 0.0
        feasible = admitted_support & (network["p95"] <= BUDGET_MS)
        objective = base
        if penalty is not None:
            objective = base - network["p_admit"] * penalty * (~feasible)
        mask = feasible if feasible_only else admitted_support
        index = _best_index(objective, network["p95"], mask)
        if index is None:
            continue
        candidate = _outcome_from_vectors(
            mode_id, index, surface, network, objective
        )
        candidate_feasible = bool(feasible[index])
        if best is None:
            best, best_feasible = candidate, candidate_feasible
            continue
        if candidate.reward != best.reward:
            choose = candidate.reward > best.reward
        elif candidate_feasible != best_feasible:
            choose = candidate_feasible  # deterministic feasible-first tie
        else:
            choose = _better(
                candidate, best, objective="reward", tie_percentile="p95"
            )
        if choose:
            best, best_feasible = candidate, candidate_feasible
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


def _scalar_outcome(
    environment: EmpiricalOneStepEnvironmentV1,
    context: TrainSelectionContextV1,
    mode_id: int,
    q_e4: int,
    penalty: float,
) -> Tuple[OracleOutcome, float]:
    require_supported_action(mode_id, q_e4)
    query = environment._surface.query_fit_q_e4(context.sample_id, mode_id, q_e4)
    component = query.policy.component(DIRECT_QUALITY_COMPONENT)
    if not component.valid or component.value is None:
        raise OracleAuditError("exact-penalty scalar quality is undefined")
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
    base = base_p95_expected_utility(
        p_admit=p_admit,
        q_perc=float(component.value),
        latency_p95_ms=latency_p95,
    )
    shaped = shaped_p95_expected_utility(
        p_admit=p_admit,
        q_perc=float(component.value),
        latency_p95_ms=latency_p95,
        deadline_penalty=penalty,
    )
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
            reward=shaped,
        ),
        base,
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


def _report(summary: Mapping[str, Any]) -> str:
    baseline = summary["unconstrained_oracle"]
    constrained = summary["constrained_oracle"]
    decision = summary["decision"]
    worst = summary["worst_penalty_requirement"]
    return "\n".join(
        [
            "# Train-only exact P95 deadline-penalty preflight",
            "",
            "The coefficient is derived analytically from registered training",
            "support; it was not selected from a post-hoc grid. The penalty is",
            "inside the admitted branch because P95 is conditional on retained",
            "survivors. Registered bundles are integrity-audited when loaded, but",
            "selection queries only the 391 registered training scene IDs.",
            "",
            f"- Decision: **{decision['status']}**",
            f"- Derived Delta: `{decision['delta']:.12g}`",
            f"- Derived lambda_D: `{decision['deadline_penalty']:.12g}`",
            f"- Float64 search evaluations: `{decision['representable_search_evaluations']}`",
            f"- Unconstrained P95 misses: `{baseline['p95_miss_count']}/{EXPECTED_TRAIN_CONTEXT_COUNT}`",
            f"- Constrained/shaped P95 misses: `{constrained['p95_miss_count']}/{EXPECTED_TRAIN_CONTEXT_COUNT}`",
            f"- Mean quality: `{baseline['mean_q_perc']:.6f}` -> `{constrained['mean_q_perc']:.6f}` ({constrained['quality_retention']:.3%})",
            f"- Mean admission: `{baseline['mean_p_admit']:.6f}` -> `{constrained['mean_p_admit']:.6f}` ({constrained['admission_change']:+.6f})",
            f"- Shaped/constrained identity matches: `{decision['winner_identity_match_count']}/{EXPECTED_TRAIN_CONTEXT_COUNT}`",
            f"- Binding worst case: context `{worst['context_index']}`, mode `{worst['mode_id']}`, q_e4 `{worst['q_e4']}`, ratio `{worst['required_penalty_ratio']:.12g}`",
            "",
            "This is modeled conditional-survivor compliance, not a live 200-ms SLA.",
            "No policy training or validation selection was performed.",
            "",
        ]
    )


def run_train_exact_p95_deadline_penalty(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    observed_spec_hash = canonical_sha256(exact_penalty_spec_document())
    if observed_spec_hash != REGISTERED_EXACT_PENALTY_SPEC_SHA256:
        raise OracleAuditError(
            f"exact P95 penalty specification hash drift: {observed_spec_hash}"
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
        raise OracleAuditError("train scene-ID proof count drift")
    validation_intersection = sorted(set(train_scene_ids) & validation_scene_ids)
    if validation_intersection:
        raise OracleAuditError("train selection intersects fit-validation scene IDs")
    environment = EmpiricalOneStepEnvironmentV1.load_registered(seed=0, project_root=root)
    first_pass: list[Dict[str, Any]] = []
    requirements: list[Tuple[float, float, float]] = []
    requirement_records: list[Dict[str, Any]] = []
    action_evaluations = 0
    try:
        for context in contexts:
            surfaces, networks = _context_vectors(environment, context)
            action_evaluations += sum(len(surface["q"]) for surface in surfaces)
            unconstrained = _best_across_modes(
                surfaces, networks, feasible_only=False
            )
            constrained = _best_across_modes(
                surfaces, networks, feasible_only=True
            )
            if unconstrained is None or constrained is None:
                raise ExactPenaltyDerivationError(
                    f"context {context.context_index} has no P95-feasible SPLIT action"
                )
            for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
                base = _base_vector(
                    network["p_admit"], surface["quality"], network["p95"]
                )
                infeasible = (network["p_admit"] > 0.0) & (
                    network["p95"] > BUDGET_MS
                )
                candidates = np.flatnonzero(
                    infeasible & (base >= constrained.reward)
                )
                for index in candidates:
                    base_value = float(base[index])
                    p_value = float(network["p_admit"][index])
                    requirement = (constrained.reward, base_value, p_value)
                    requirements.append(requirement)
                    requirement_records.append(
                        {
                            "context_index": context.context_index,
                            "sample_id": context.sample_id,
                            "network_profile": context.network_profile,
                            "mode_id": mode_id,
                            "q_e4": int(surface["q"][index]),
                            "q_perc": float(surface["quality"][index]),
                            "p_admit": p_value,
                            "latency_p95_ms": float(network["p95"][index]),
                            "infeasible_base_reward": base_value,
                            "best_feasible_base_reward": constrained.reward,
                            "required_penalty_ratio": (
                                (base_value - constrained.reward) / p_value
                            ),
                        }
                    )
            first_pass.append(
                {
                    "context": context,
                    "unconstrained": unconstrained,
                    "constrained": constrained,
                }
            )

        if action_evaluations != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
            raise OracleAuditError("exact action-context evaluation count drift")
        delta, penalty, search_evaluations = derive_exact_deadline_penalty(requirements)
        worst_requirement = max(
            requirement_records,
            key=lambda row: (
                row["required_penalty_ratio"],
                -row["context_index"],
                -row["mode_id"],
                -row["q_e4"],
            ),
        )
        worst_context = contexts[int(worst_requirement["context_index"])]
        worst_scalar, worst_scalar_base = _scalar_outcome(
            environment,
            worst_context,
            int(worst_requirement["mode_id"]),
            int(worst_requirement["q_e4"]),
            0.0,
        )
        for name, observed, expected in (
            ("base_reward", worst_scalar_base, worst_requirement["infeasible_base_reward"]),
            ("p_admit", worst_scalar.p_edge_admission_given_sent, worst_requirement["p_admit"]),
            ("latency_p95_ms", worst_scalar.latency_proxy_p95_ms, worst_requirement["latency_p95_ms"]),
        ):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-10):
                raise OracleAuditError(
                    f"worst exact-penalty requirement scalar mismatch for {name}"
                )

        rows: list[Dict[str, Any]] = []
        identity_matches = 0
        second_pass_evaluations = 0
        for saved in first_pass:
            context = saved["context"]
            constrained = saved["constrained"]
            surfaces, networks = _context_vectors(environment, context)
            second_pass_evaluations += sum(len(surface["q"]) for surface in surfaces)
            shaped = _best_across_modes(
                surfaces,
                networks,
                feasible_only=False,
                penalty=penalty,
            )
            if shaped is None:
                raise OracleAuditError("exact-penalty oracle found no action")
            scalar, scalar_base = _scalar_outcome(
                environment,
                context,
                shaped.mode_id,
                shaped.q_e4,
                penalty,
            )
            if not math.isclose(
                scalar.reward, shaped.reward, rel_tol=0.0, abs_tol=1e-10
            ):
                raise OracleAuditError("exact-penalty scalar/vector reward mismatch")
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
                    "unconstrained_base_reward": saved["unconstrained"].reward,
                    "constrained_mode_id": constrained.mode_id,
                    "constrained_q_e4": constrained.q_e4,
                    "constrained_q_perc": constrained.q_perc,
                    "constrained_p_admit": constrained.p_edge_admission_given_sent,
                    "constrained_latency_p95_ms": constrained.latency_proxy_p95_ms,
                    "constrained_base_reward": constrained.reward,
                    "shaped_mode_id": scalar.mode_id,
                    "shaped_q_e4": scalar.q_e4,
                    "shaped_latency_p95_ms": scalar.latency_proxy_p95_ms,
                    "shaped_base_reward": scalar_base,
                    "shaped_reward": scalar.reward,
                    "shaped_matches_constrained": match,
                }
            )
        if second_pass_evaluations != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
            raise OracleAuditError("second-pass action-context count drift")
    finally:
        environment.close()

    if not cuda_before and torch.cuda.is_initialized():
        raise OracleAuditError("CPU-only exact-penalty screen initialized CUDA")

    unconstrained_rows = [row for row in first_pass]
    unconstrained_quality = mean(row["unconstrained"].q_perc for row in unconstrained_rows)
    unconstrained_admission = mean(
        row["unconstrained"].p_edge_admission_given_sent for row in unconstrained_rows
    )
    unconstrained_misses = sum(
        row["unconstrained"].latency_proxy_p95_ms > BUDGET_MS
        for row in unconstrained_rows
    )
    constrained_quality = mean(row["constrained"].q_perc for row in unconstrained_rows)
    constrained_admission = mean(
        row["constrained"].p_edge_admission_given_sent for row in unconstrained_rows
    )
    constrained_misses = sum(
        row["constrained"].latency_proxy_p95_ms > BUDGET_MS
        for row in unconstrained_rows
    )
    quality_retention = constrained_quality / unconstrained_quality
    admission_change = constrained_admission - unconstrained_admission
    criteria = {
        "zero_p95_misses": constrained_misses == 0,
        "shaped_matches_constrained": identity_matches == EXPECTED_TRAIN_CONTEXT_COUNT,
        "quality_retention_ge_0_95": quality_retention >= MEAN_QUALITY_MINIMUM_RETENTION,
        "admission_drop_le_0_001": admission_change >= -MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
    }
    status = "GO" if all(criteria.values()) else "NO_GO"
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_TRAIN_ONLY_EXACT_P95_DEADLINE_PENALTY",
        "decision": {
            "status": status,
            "criteria": criteria,
            "deadline_penalty": penalty,
            "delta": delta,
            "representable_search_evaluations": search_evaluations,
            "winner_identity_match_count": identity_matches,
        },
        "worst_penalty_requirement": worst_requirement,
        "unconstrained_oracle": {
            "mean_q_perc": unconstrained_quality,
            "mean_p_admit": unconstrained_admission,
            "p95_miss_count": unconstrained_misses,
        },
        "constrained_oracle": {
            "mean_q_perc": constrained_quality,
            "mean_p_admit": constrained_admission,
            "p95_miss_count": constrained_misses,
            "quality_retention": quality_retention,
            "admission_change": admission_change,
        },
        "exactness": {
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "first_pass_action_context_evaluations": action_evaluations,
            "second_pass_action_context_evaluations": second_pass_evaluations,
            "requirement_count": len(requirements),
            "train_context_count": EXPECTED_TRAIN_CONTEXT_COUNT,
            "train_scene_count": EXPECTED_TRAIN_SCENE_COUNT,
            "scalar_revalidated_winner_count": len(rows),
        },
        "bindings": {
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "original_d1_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "exact_penalty_spec_sha256": REGISTERED_EXACT_PENALTY_SPEC_SHA256,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            "queried_train_scene_id_count": len(train_scene_ids),
            "queried_train_scene_id_sha256": canonical_sha256(
                list(train_scene_ids)
            ),
            "fit_validation_scene_id_intersection_count": len(
                validation_intersection
            ),
        },
    }
    decision_payload = {
        "status": status,
        "deadline_penalty": penalty,
        "criteria": criteria,
    }
    payloads = {
        "train_context_oracles.csv": _csv_bytes(rows),
        "selection_decision.json": (
            json.dumps(decision_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    }
    payloads["REPORT.md"] = _report(summary).encode()
    summary["files"] = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    payloads["summary.json"] = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    for name, payload in payloads.items():
        _atomic_bytes(destination / name, payload)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_train_exact_p95_deadline_penalty(output_dir=args.output)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
