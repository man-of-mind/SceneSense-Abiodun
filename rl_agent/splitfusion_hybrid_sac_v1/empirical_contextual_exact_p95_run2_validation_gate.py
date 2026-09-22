"""Exact pre-training validation gate for the frozen Run-2 P95 reward.

This module deliberately performs no policy training.  It first chooses one
fixed executable action using registered *training* contexts only.  Only after
that action is frozen does it open the registered fit-validation panel and
test whether the train-derived deadline penalty generalizes without retuning.

The reward is the registered admitted-branch formulation::

    R = p * (Q - 0.25 * L95 / 200 - lambda_D * 1[L95 > 200])
        + (1 - p) * (-1)

All continuous actor proposals execute through the finite integer ``q_e4``
quotient.  The exhaustive searches below therefore cover the complete
executable SPLIT action domain, not a sampled continuous approximation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_exact_p95_deadline_penalty import (
    BUDGET_MS,
    REGISTERED_EXACT_PENALTY_SPEC_SHA256,
    _base_vector,
    _best_across_modes,
    _context_vectors,
    _scalar_outcome,
    base_p95_expected_utility,
    shaped_p95_expected_utility,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_fit_validation_evaluator import (
    FitValidationActorEvaluatorV1,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
)
from .empirical_contextual_p95_hinge_selection import (
    EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS,
    EXPECTED_TRAIN_CONTEXT_COUNT,
    EXPECTED_TRAIN_SCENE_COUNT,
    _training_contexts,
)
from .empirical_contextual_split_oracle import (
    CONDITIONAL_FEASIBILITY_SEMANTICS,
    EXACT_ACTION_COUNT_PER_SCENE,
    OracleAuditError,
    OracleOutcome,
    _authoritative_outcome,
    _network_vector,
    _project_root,
    _surface_mode_vector,
    enumerate_supported_actions,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "RUN2_DEADLINE_PENALTY",
    "RUN2_PREREGISTRATION_SHA256",
    "Run2ValidationGateError",
    "run_exact_p95_run2_validation_gate",
    "select_fixed_action_from_train_sums",
    "validate_frozen_fixed_selection",
]


SCHEMA = "splitfusion.exact_p95_run2_validation_gate.v1"
RUN2_DEADLINE_PENALTY = 0.5742957604173842
RUN2_PREREGISTRATION_SHA256 = (
    "77e6e6b98c4fbf7e03994d73621bdec12ba9853872e1dfdcce17ce15905c1171"
)
PREREGISTRATION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_preregistration_v1/preregistration.json"
)
PENALTY_PREFLIGHT_SUMMARY_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v1/summary.json"
)
PENALTY_PREFLIGHT_DECISION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v1/selection_decision.json"
)
EXPECTED_VALIDATION_CONTEXT_COUNT = 340
EXPECTED_VALIDATION_SCENE_COUNT = 85
EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS = (
    EXPECTED_VALIDATION_CONTEXT_COUNT * EXACT_ACTION_COUNT_PER_SCENE
)
_SCALAR_ABSOLUTE_TOLERANCE = 2e-9


class Run2ValidationGateError(RuntimeError):
    """The frozen Run-2 validation gate failed an integrity invariant."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise Run2ValidationGateError(f"{path} is not a JSON object")
    return document


def _require_frozen_preregistration(root: Path) -> Tuple[Path, Dict[str, Any]]:
    path = root / PREREGISTRATION_RELATIVE_PATH
    observed = _sha256_file(path)
    if observed != RUN2_PREREGISTRATION_SHA256:
        raise Run2ValidationGateError(
            f"Run-2 preregistration hash drift: {observed}"
        )
    document = _read_json(path)
    reward = document.get("reward")
    gate = document.get("pretraining_go_no_go")
    if (
        document.get("status") != "FROZEN_BEFORE_RUN2_TRAINING"
        or not isinstance(reward, dict)
        or reward.get("deadline_ms") != BUDGET_MS
        or reward.get("deadline_penalty") != RUN2_DEADLINE_PENALTY
        or not isinstance(gate, dict)
        or gate.get("frozen_lambda_shaped_oracle_p95_miss_count") != 0
    ):
        raise Run2ValidationGateError("Run-2 preregistration content drift")
    bindings = document.get("bindings")
    if (
        not isinstance(bindings, dict)
        or bindings.get("exact_penalty_spec_sha256")
        != REGISTERED_EXACT_PENALTY_SPEC_SHA256
        or bindings.get("fit_partition_sha256")
        != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        or bindings.get("fit_validation_panel_sha256")
        != REGISTERED_FIT_VALIDATION_PANEL_SHA256
        or bindings.get("modeled_smoke_support_sha256")
        != MODELED_SMOKE_SUPPORT_SHA256
    ):
        raise Run2ValidationGateError("Run-2 preregistration binding drift")
    if (
        _sha256_file(root / PENALTY_PREFLIGHT_SUMMARY_RELATIVE_PATH)
        != bindings.get("exact_penalty_preflight_summary_sha256")
        or _sha256_file(root / PENALTY_PREFLIGHT_DECISION_RELATIVE_PATH)
        != bindings.get("exact_penalty_selection_decision_sha256")
        or _sha256_file(
            Path(__file__).with_name(
                "empirical_contextual_exact_p95_deadline_penalty.py"
            )
        )
        != bindings.get("exact_penalty_implementation_sha256")
    ):
        raise Run2ValidationGateError("exact-penalty source binding drift")
    return path, document


def select_fixed_action_from_train_sums(
    reward_sums: Sequence[np.ndarray], context_count: int
) -> Tuple[int, int, float]:
    """Choose max mean reward, then minimum mode and minimum q_e4.

    The arrays are ordered exactly like ``MODELED_SMOKE_MODE_Q_E4_BOUNDS``.
    Equality intentionally retains the earlier mode and the first (ascending)
    q coordinate.  This helper is public so the selection rule can be tested
    without running the exhaustive campaign.
    """

    if type(context_count) is not int or context_count < 1:
        raise ValueError("context_count must be a positive exact integer")
    if len(reward_sums) != len(MODELED_SMOKE_MODE_Q_E4_BOUNDS):
        raise ValueError("reward_sums must contain one array per mode")
    best: Optional[Tuple[int, int, float]] = None
    for mode_id, ((lower, upper), supplied) in enumerate(
        zip(MODELED_SMOKE_MODE_Q_E4_BOUNDS, reward_sums)
    ):
        values = np.asarray(supplied, dtype=np.float64)
        expected = upper - lower + 1
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(f"mode {mode_id} reward sums are invalid")
        maximum = float(np.max(values))
        indexes = np.flatnonzero(values == maximum)
        if len(indexes) < 1:
            raise ValueError(f"mode {mode_id} has no fixed-action candidate")
        q_e4 = lower + int(indexes[0])
        candidate = (mode_id, q_e4, maximum / context_count)
        if best is None or candidate[2] > best[2]:
            best = candidate
    if best is None:
        raise ValueError("fixed-action selection found no candidate")
    return best


def validate_frozen_fixed_selection(selection: Mapping[str, Any]) -> str:
    """Recompute the selection digest after removing only its digest field."""

    if not isinstance(selection, Mapping):
        raise Run2ValidationGateError("fixed selection must be a mapping")
    document = dict(selection)
    stated = document.pop("frozen_selection_sha256", None)
    observed = canonical_sha256(document)
    if stated != observed:
        raise Run2ValidationGateError("frozen fixed-action selection digest drift")
    return observed


def _validation_scalar(
    evaluator: FitValidationActorEvaluatorV1,
    entry: Any,
    mode_id: int,
    q_e4: int,
) -> Tuple[OracleOutcome, float, float]:
    raw = _authoritative_outcome(evaluator, entry, mode_id, q_e4)
    base = base_p95_expected_utility(
        p_admit=raw.p_edge_admission_given_sent,
        q_perc=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
    )
    shaped = shaped_p95_expected_utility(
        p_admit=raw.p_edge_admission_given_sent,
        q_perc=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
        deadline_penalty=RUN2_DEADLINE_PENALTY,
    )
    return raw, base, shaped


def _assert_scalar_vector(
    *,
    vector: OracleOutcome,
    scalar: OracleOutcome,
    scalar_reward: float,
    label: str,
) -> None:
    for name in (
        "q_perc",
        "total_transmitted_bytes",
        "datagram_count",
        "p_edge_admission_given_sent",
        "latency_proxy_p50_ms",
        "latency_proxy_p95_ms",
        "latency_proxy_p99_ms",
    ):
        observed = float(getattr(scalar, name))
        expected = float(getattr(vector, name))
        if not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=_SCALAR_ABSOLUTE_TOLERANCE
        ):
            raise Run2ValidationGateError(
                f"{label} scalar/vector mismatch for {name}: "
                f"{observed} != {expected}"
            )
    if (scalar.mode_id, scalar.q_e4) != (vector.mode_id, vector.q_e4):
        raise Run2ValidationGateError(f"{label} scalar/vector action mismatch")
    if not math.isclose(
        scalar_reward,
        vector.reward,
        rel_tol=0.0,
        abs_tol=_SCALAR_ABSOLUTE_TOLERANCE,
    ):
        raise Run2ValidationGateError(
            f"{label} scalar/vector reward mismatch: "
            f"{scalar_reward} != {vector.reward}"
        )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise Run2ValidationGateError("CSV requires at least one row")
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


def _train_fixed_action(
    *, root: Path
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...], frozenset[str]]:
    """Select and scalar-revalidate a fixed action without panel access."""

    partition = load_registered_empirical_fit_partition(project_root=root)
    contexts = _training_contexts(partition)
    train_ids = frozenset(
        row.sample_id
        for row in partition.scene_assignments
        if row.split == TRAIN_SPLIT
    )
    validation_ids = frozenset(
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    )
    if (
        len(contexts) != EXPECTED_TRAIN_CONTEXT_COUNT
        or len(train_ids) != EXPECTED_TRAIN_SCENE_COUNT
        or train_ids & validation_ids
    ):
        raise Run2ValidationGateError("registered train/validation partition drift")

    reward_sums = [
        np.zeros(upper - lower + 1, dtype=np.float64)
        for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS
    ]
    evaluation_count = 0
    # Selection must not construct the validation evaluator or panel.  Use the
    # same registered D1 environment behind the existing train-only oracle.
    from .empirical_contextual_environment import EmpiricalOneStepEnvironmentV1

    d1 = EmpiricalOneStepEnvironmentV1.load_registered(seed=0, project_root=root)
    try:
        for context in contexts:
            surfaces, networks = _context_vectors(d1, context)
            for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
                base = _base_vector(
                    network["p_admit"], surface["quality"], network["p95"]
                )
                feasible = (network["p_admit"] > 0.0) & (
                    network["p95"] <= BUDGET_MS
                )
                shaped = base - (
                    network["p_admit"]
                    * RUN2_DEADLINE_PENALTY
                    * (~feasible)
                )
                reward_sums[mode_id] += shaped
                evaluation_count += len(shaped)
        if evaluation_count != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
            raise Run2ValidationGateError(
                "train fixed-action exhaustive evaluation count drift"
            )
        mode_id, q_e4, vector_mean = select_fixed_action_from_train_sums(
            reward_sums, len(contexts)
        )
        scalar_rows = []
        for context in contexts:
            scalar, base = _scalar_outcome(
                d1, context, mode_id, q_e4, RUN2_DEADLINE_PENALTY
            )
            scalar_rows.append(
                {
                    "base_reward": base,
                    "latency_p95_ms": scalar.latency_proxy_p95_ms,
                    "p_admit": scalar.p_edge_admission_given_sent,
                    "q_perc": scalar.q_perc,
                    "shaped_reward": scalar.reward,
                }
            )
    finally:
        d1.close()
    scalar_mean = mean(row["shaped_reward"] for row in scalar_rows)
    if not math.isclose(
        scalar_mean,
        vector_mean,
        rel_tol=0.0,
        abs_tol=_SCALAR_ABSOLUTE_TOLERANCE,
    ):
        raise Run2ValidationGateError(
            "train-selected fixed action failed scalar mean revalidation"
        )
    selection_document = {
        "action": {"mode_id": mode_id, "q_e4": q_e4},
        "action_selection": (
            "MAXIMUM_MEAN_RUN2_REWARD_ON_TRAIN_THEN_MIN_MODE_ID_THEN_MIN_Q_E4"
        ),
        "context_count": len(contexts),
        "exhaustive_action_context_evaluations": evaluation_count,
        "fit_validation_outcome_access_before_freeze": False,
        "mean_base_p95_reward": mean(row["base_reward"] for row in scalar_rows),
        "mean_latency_p95_ms": mean(
            row["latency_p95_ms"] for row in scalar_rows
        ),
        "mean_p_admit": mean(row["p_admit"] for row in scalar_rows),
        "mean_q_perc": mean(row["q_perc"] for row in scalar_rows),
        "mean_shaped_reward": scalar_mean,
        "p95_miss_count": sum(
            row["latency_p95_ms"] > BUDGET_MS for row in scalar_rows
        ),
        "scalar_revalidation_count": len(scalar_rows),
        "selection_population": "REGISTERED_TRAIN_CONTEXTS_ONLY",
        "train_scene_count": len(train_ids),
        "validation_scene_id_intersection_count": len(train_ids & validation_ids),
    }
    selection_sha = canonical_sha256(selection_document)
    selection_document["frozen_selection_sha256"] = selection_sha
    validate_frozen_fixed_selection(selection_document)
    return selection_document, tuple(scalar_rows), train_ids


def _validation_gate(
    *,
    root: Path,
    fixed_selection: Mapping[str, Any],
    train_ids: frozenset[str],
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    action = fixed_selection["action"]
    fixed_mode = int(action["mode_id"])
    fixed_q = int(action["q_e4"])
    rows = []
    evaluation_count = 0
    feasible_contexts = 0
    identity_matches = 0
    shaped_misses = 0
    constrained_scalar_count = 0
    shaped_scalar_count = 0
    fixed_scalar_count = 0
    cuda_before = torch.cuda.is_initialized()

    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        panel = evaluator.panel
        if panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
            raise Run2ValidationGateError("fit-validation panel hash drift")
        validation_ids = frozenset(row.scene_sample_id for row in panel.entries)
        if len(validation_ids) != EXPECTED_VALIDATION_SCENE_COUNT:
            raise Run2ValidationGateError("validation scene count drift")
        if train_ids & validation_ids:
            raise Run2ValidationGateError("train/validation scene-ID overlap")

        for entry in panel.entries:
            surfaces = [
                _surface_mode_vector(
                    evaluator.environment._surface,
                    entry.scene_sample_id,
                    mode_id,
                )
                for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS))
            ]
            networks = [
                _network_vector(
                    evaluator.environment._network,
                    entry.network_profile,
                    surface["payload"],
                    surface["datagrams"],
                )
                for surface in surfaces
            ]
            evaluation_count += sum(len(surface["q"]) for surface in surfaces)
            constrained = _best_across_modes(
                surfaces, networks, feasible_only=True
            )
            shaped = _best_across_modes(
                surfaces,
                networks,
                feasible_only=False,
                penalty=RUN2_DEADLINE_PENALTY,
            )
            if shaped is None:
                raise Run2ValidationGateError(
                    f"validation context {entry.panel_index} has no admitted action"
                )
            shaped_raw, _shaped_base, shaped_reward = _validation_scalar(
                evaluator, entry, shaped.mode_id, shaped.q_e4
            )
            _assert_scalar_vector(
                vector=shaped,
                scalar=shaped_raw,
                scalar_reward=shaped_reward,
                label="shaped validation winner",
            )
            shaped_scalar_count += 1
            shaped_miss = shaped_raw.latency_proxy_p95_ms > BUDGET_MS
            shaped_misses += int(shaped_miss)
            fixed_raw, _fixed_base, fixed_reward = _validation_scalar(
                evaluator, entry, fixed_mode, fixed_q
            )
            fixed_scalar_count += 1
            if constrained is None:
                rows.append(
                    {
                        "panel_index": entry.panel_index,
                        "scene_rank": entry.scene_rank,
                        "sample_id": entry.scene_sample_id,
                        "network_profile": entry.network_profile,
                        "has_p95_feasible_action": False,
                        "constrained_mode_id": "",
                        "constrained_q_e4": "",
                        "constrained_base_reward": "",
                        "constrained_latency_p95_ms": "",
                        "shaped_mode_id": shaped.mode_id,
                        "shaped_q_e4": shaped.q_e4,
                        "constrained_q_perc": "",
                        "constrained_p_admit": "",
                        "shaped_reward": shaped_reward,
                        "shaped_latency_p95_ms": shaped_raw.latency_proxy_p95_ms,
                        "shaped_p95_miss": shaped_miss,
                        "shaped_q_perc": shaped_raw.q_perc,
                        "shaped_p_admit": shaped_raw.p_edge_admission_given_sent,
                        "shaped_matches_constrained": False,
                        "fixed_mode_id": fixed_mode,
                        "fixed_q_e4": fixed_q,
                        "fixed_shaped_reward": fixed_reward,
                        "fixed_latency_p95_ms": fixed_raw.latency_proxy_p95_ms,
                        "fixed_p95_miss": fixed_raw.latency_proxy_p95_ms > BUDGET_MS,
                        "fixed_q_perc": fixed_raw.q_perc,
                        "fixed_p_admit": fixed_raw.p_edge_admission_given_sent,
                    }
                )
                continue

            feasible_contexts += 1
            constrained_raw, constrained_base, _constrained_shaped = (
                _validation_scalar(
                    evaluator,
                    entry,
                    constrained.mode_id,
                    constrained.q_e4,
                )
            )
            _assert_scalar_vector(
                vector=constrained,
                scalar=constrained_raw,
                scalar_reward=constrained_base,
                label="constrained validation winner",
            )
            constrained_scalar_count += 1

            match = (shaped.mode_id, shaped.q_e4) == (
                constrained.mode_id,
                constrained.q_e4,
            )
            identity_matches += int(match)
            rows.append(
                {
                    "panel_index": entry.panel_index,
                    "scene_rank": entry.scene_rank,
                    "sample_id": entry.scene_sample_id,
                    "network_profile": entry.network_profile,
                    "has_p95_feasible_action": True,
                    "constrained_mode_id": constrained.mode_id,
                    "constrained_q_e4": constrained.q_e4,
                    "constrained_base_reward": constrained_base,
                    "constrained_latency_p95_ms": (
                        constrained_raw.latency_proxy_p95_ms
                    ),
                    "constrained_q_perc": constrained_raw.q_perc,
                    "constrained_p_admit": (
                        constrained_raw.p_edge_admission_given_sent
                    ),
                    "shaped_mode_id": shaped.mode_id,
                    "shaped_q_e4": shaped.q_e4,
                    "shaped_reward": shaped_reward,
                    "shaped_latency_p95_ms": shaped_raw.latency_proxy_p95_ms,
                    "shaped_p95_miss": shaped_miss,
                    "shaped_q_perc": shaped_raw.q_perc,
                    "shaped_p_admit": shaped_raw.p_edge_admission_given_sent,
                    "shaped_matches_constrained": match,
                    "fixed_mode_id": fixed_mode,
                    "fixed_q_e4": fixed_q,
                    "fixed_shaped_reward": fixed_reward,
                    "fixed_latency_p95_ms": fixed_raw.latency_proxy_p95_ms,
                    "fixed_p95_miss": fixed_raw.latency_proxy_p95_ms > BUDGET_MS,
                    "fixed_q_perc": fixed_raw.q_perc,
                    "fixed_p_admit": fixed_raw.p_edge_admission_given_sent,
                }
            )

    if not cuda_before and torch.cuda.is_initialized():
        raise Run2ValidationGateError("CPU-only gate initialized CUDA")
    if evaluation_count != EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS:
        raise Run2ValidationGateError(
            "validation exhaustive action-context evaluation count drift"
        )
    metrics = {
        "context_count": len(rows),
        "exhaustive_action_context_evaluations": evaluation_count,
        "feasible_context_count": feasible_contexts,
        "infeasible_context_count": len(rows) - feasible_contexts,
        "shaped_constrained_identity_match_count": identity_matches,
        "shaped_constrained_identity_mismatch_count": len(rows) - identity_matches,
        "shaped_oracle_p95_miss_count": shaped_misses,
        "constrained_scalar_revalidation_count": constrained_scalar_count,
        "shaped_scalar_revalidation_count": shaped_scalar_count,
        "fixed_scalar_evaluation_count": fixed_scalar_count,
    }
    return tuple(rows), metrics


def _profile_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    result = []
    observed_profiles = {str(row["network_profile"]) for row in rows}
    profiles = tuple(
        profile for profile in NETWORK_PROFILE_ORDER if profile in observed_profiles
    )
    for profile in (*profiles, "ALL_PROFILES"):
        selected = (
            list(rows)
            if profile == "ALL_PROFILES"
            else [row for row in rows if row["network_profile"] == profile]
        )
        valid = [row for row in selected if row["has_p95_feasible_action"]]
        result.append(
            {
                "network_profile": profile,
                "context_count": len(selected),
                "feasible_context_count": len(valid),
                "shaped_p95_miss_count": sum(
                    bool(row["shaped_p95_miss"]) for row in selected
                ),
                "shaped_identity_match_count": sum(
                    bool(row["shaped_matches_constrained"]) for row in valid
                ),
                "constrained_mean_q_perc": mean(
                    float(row["constrained_q_perc"])
                    for row in valid
                ) if valid else None,
                "constrained_mean_p_admit": mean(
                    float(row["constrained_p_admit"])
                    for row in valid
                ) if valid else None,
                "shaped_mean_q_perc": mean(
                    float(row["shaped_q_perc"]) for row in selected
                ),
                "shaped_mean_p_admit": mean(
                    float(row["shaped_p_admit"]) for row in selected
                ),
                "fixed_p95_miss_count": sum(
                    bool(row["fixed_p95_miss"]) for row in selected
                ),
                "fixed_mean_q_perc": mean(
                    float(row["fixed_q_perc"]) for row in selected
                ),
                "fixed_mean_p_admit": mean(
                    float(row["fixed_p_admit"]) for row in selected
                ),
                "fixed_mean_shaped_reward": mean(
                    float(row["fixed_shaped_reward"]) for row in selected
                ),
            }
        )
    return tuple(result)


def _decision(validation: Mapping[str, int]) -> Dict[str, Any]:
    criteria = {
        "every_context_has_p95_feasible_action": (
            validation["feasible_context_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
        "frozen_lambda_shaped_oracle_matches_constrained": (
            validation["shaped_constrained_identity_match_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
        "frozen_lambda_shaped_oracle_has_zero_p95_misses": (
            validation["shaped_oracle_p95_miss_count"] == 0
        ),
        "all_validation_winners_scalar_revalidated": (
            validation["constrained_scalar_revalidation_count"]
            == validation["feasible_context_count"]
            and validation["shaped_scalar_revalidation_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
    }
    return {
        "criteria": criteria,
        "failure_action": (
            "DO_NOT_RETUNE_ON_FIT_VALIDATION; REPORT_PENALTY_GENERALIZATION_"
            "FAILURE_AND_CREATE_SEPARATE_RUN3_DESIGN"
        ),
        "status": "GO" if all(criteria.values()) else "NO_GO",
    }


def _report(summary: Mapping[str, Any]) -> str:
    fixed = summary["train_selected_fixed_action"]
    validation = summary["validation_gate"]
    decision = summary["decision"]
    return "\n".join(
        [
            "# Exact P95 Run-2 pre-training validation gate",
            "",
            f"- Decision: **{decision['status']}**",
            f"- Frozen deadline penalty: `{RUN2_DEADLINE_PENALTY:.17g}`",
            f"- Validation contexts with a feasible action: `{validation['feasible_context_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Shaped/constrained oracle identity matches: `{validation['shaped_constrained_identity_match_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Shaped-oracle P95 misses: `{validation['shaped_oracle_p95_miss_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Train-selected fixed action: mode `{fixed['action']['mode_id']}`, q_e4 `{fixed['action']['q_e4']}`",
            f"- Train fixed-action search evaluations: `{fixed['exhaustive_action_context_evaluations']}`",
            f"- Validation oracle evaluations: `{validation['exhaustive_action_context_evaluations']}`",
            "",
            "The fixed action was selected before the validation panel was opened.",
            "No coefficient, action, or tie rule was tuned on validation outcomes.",
            "This is modeled conditional-survivor P95 compliance, not a live SLA.",
            "No policy training was performed.",
            "",
        ]
    )


def run_exact_p95_run2_validation_gate(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    prereg_path, prereg = _require_frozen_preregistration(root)
    actions = enumerate_supported_actions()
    if len(actions) != EXACT_ACTION_COUNT_PER_SCENE:
        raise Run2ValidationGateError("executable action count drift")

    # Deliberate ordering boundary: no panel/evaluator construction precedes
    # completion and hashing of this train-only fixed-action selection.
    fixed_selection, _train_rows, train_ids = _train_fixed_action(root=root)
    frozen_fixed_sha = validate_frozen_fixed_selection(fixed_selection)
    validation_rows, validation_metrics = _validation_gate(
        root=root, fixed_selection=fixed_selection, train_ids=train_ids
    )
    if validate_frozen_fixed_selection(fixed_selection) != frozen_fixed_sha:
        raise Run2ValidationGateError("fixed action changed after validation access")

    profile_rows = _profile_rows(validation_rows)
    decision = _decision(validation_metrics)
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_EXACT_P95_RUN2_PRETRAINING_VALIDATION_GATE",
        "decision": decision,
        "reward": {
            "deadline_ms": BUDGET_MS,
            "deadline_penalty": RUN2_DEADLINE_PENALTY,
            "formula": prereg["reward"]["formula"],
            "penalty_placement": "INSIDE_ADMITTED_BRANCH",
        },
        "train_selected_fixed_action": fixed_selection,
        "validation_gate": validation_metrics,
        "tie_rules": {
            "train_fixed": (
                "MAX_MEAN_RUN2_REWARD_THEN_MIN_MODE_ID_THEN_MIN_Q_E4"
            ),
            "constrained_oracle": (
                "MAX_FEASIBLE_BASE_REWARD_THEN_MIN_P95_THEN_MIN_MODE_ID_"
                "THEN_MIN_Q_E4"
            ),
            "shaped_oracle": (
                "MAX_SHAPED_REWARD_THEN_FEASIBLE_FIRST_THEN_MIN_P95_THEN_"
                "MIN_MODE_ID_THEN_MIN_Q_E4"
            ),
        },
        "exactness": {
            "action_domain": "ALL_EXECUTABLE_INTEGER_MODE_Q_E4_PAIRS",
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "train_context_count": EXPECTED_TRAIN_CONTEXT_COUNT,
            "validation_context_count": EXPECTED_VALIDATION_CONTEXT_COUNT,
            "validation_scene_count": EXPECTED_VALIDATION_SCENE_COUNT,
            "scalar_vector_absolute_tolerance": _SCALAR_ABSOLUTE_TOLERANCE,
        },
        "bindings": {
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "exact_penalty_spec_sha256": REGISTERED_EXACT_PENALTY_SPEC_SHA256,
            "run2_preregistration_sha256": RUN2_PREREGISTRATION_SHA256,
            "run2_preregistration_relative_path": PREREGISTRATION_RELATIVE_PATH,
            "run2_preregistration_observed_sha256": _sha256_file(prereg_path),
            "penalty_preflight_summary_sha256": _sha256_file(
                root / PENALTY_PREFLIGHT_SUMMARY_RELATIVE_PATH
            ),
            "penalty_preflight_decision_sha256": _sha256_file(
                root / PENALTY_PREFLIGHT_DECISION_RELATIVE_PATH
            ),
            "implementation_sha256": _sha256_file(Path(__file__)),
            "test_implementation_sha256": _sha256_file(
                Path(__file__).with_name(
                    "test_empirical_contextual_exact_p95_run2_validation_gate.py"
                )
            ),
            "source_file_sha256": {
                name: _sha256_file(Path(__file__).with_name(name))
                for name in (
                    "empirical_contextual_environment.py",
                    "empirical_contextual_exact_p95_deadline_penalty.py",
                    "empirical_contextual_fit_partition.py",
                    "empirical_contextual_fit_validation_evaluator.py",
                    "empirical_contextual_fit_validation_panel.py",
                    "empirical_contextual_split_oracle.py",
                    "modeled_smoke_support.py",
                )
            },
        },
        "scope": {
            "fixed_action_selection": "TRAIN_CONTEXTS_ONLY",
            "validation_access_before_fixed_action_freeze": False,
            "validation_role": "FROZEN_DEVELOPMENT_FIT_VALIDATION",
            "lambda_retuned_on_validation": False,
            "claims_excluded": [
                "POLICY_TRAINING",
                "LIVE_200_MS_SLA",
                "UNCONDITIONAL_SERVICE_GUARANTEE",
                "FINAL_TEST_GENERALIZATION",
            ],
            "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
        },
    }

    per_context_name = "validation_context_oracles.csv"
    profile_name = "profile_summary.csv"
    fixed_name = "train_selected_fixed_action.json"
    decision_name = "GO_NO_GO.json"
    report_name = "REPORT.md"
    payloads = {
        per_context_name: _csv_bytes(validation_rows),
        profile_name: _csv_bytes(profile_rows),
        fixed_name: (
            json.dumps(fixed_selection, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        decision_name: (
            json.dumps(decision, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        report_name: _report(summary).encode("utf-8"),
    }
    summary["files"] = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    for name, payload in payloads.items():
        _atomic_bytes(destination / name, payload)
    _atomic_bytes(
        destination / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact fixed-lambda Run-2 validation gate."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_exact_p95_run2_validation_gate(
        output_dir=args.output, project_root=args.project_root
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
