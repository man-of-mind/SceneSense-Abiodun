"""One-shot held-development gate for the frozen emitted-float32 P95 v2.

The gate refuses validation access until it has re-hashed and validated the
frozen v2 preregistration and train-only derivation.  It then opens exactly the
registered 85-scene fit-validation panel, crosses its four profiles, and
exhaustively evaluates all 52,240 executable actions per context once.

No coefficient selection, retuning, policy training, or replay mutation is
performed here.  V1 validation evidence remains immutable superseded history.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
    _base64_vector,
    _best_across_modes_v2,
    _emitted_vector,
    _shaped64_vector,
    base_p95_expected_utility64_v2,
    emitted_float32_target_v2,
    shaped_p95_expected_utility64_v2,
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
    "RUN2_V2_DEADLINE_PENALTY",
    "RUN2_V2_PREREGISTRATION_SHA256",
    "RUN2_V2_TRAIN_SUMMARY_SHA256",
    "FrozenV2ValidationContract",
    "Run2V2ValidationGateError",
    "run_exact_p95_run2_validation_gate_v2",
]


SCHEMA = "splitfusion.exact_p95_run2_validation_gate.v2"
RUN2_V2_DEADLINE_PENALTY = 0.5742957622788527
RUN2_V2_PREDECESSOR = 0.5742957622788526
RUN2_V2_PREREGISTRATION_SHA256 = (
    "b1b42a622ae514f469d7a4c49214abe2002739e4bedf5f1c94ef52c3d298b8ac"
)
RUN2_V2_TRAIN_SUMMARY_SHA256 = (
    "4e102b3309c1c2bf8c6e7868957285f440f091a194ff06756d1c5f49d33b8dcd"
)
RUN2_V2_TRAIN_DECISION_SHA256 = (
    "eeef505e81be93a01cc6d289315738191f9139ed13c387c7bbb54231d219fda7"
)
RUN2_V2_PENALTY_IMPLEMENTATION_SHA256 = (
    "40975469634225c651c78a8bedcdebd12399f875dd55680c1f83c91573bd20ff"
)
PREREGISTRATION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_exact_p95_run2_preregistration_v2/preregistration_v2.json"
)
TRAIN_SUMMARY_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v2/summary_v2.json"
)
TRAIN_DECISION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_train_exact_p95_deadline_penalty_v2/selection_decision_v2.json"
)
RUNTIME_ARITHMETIC = (
    "PYTHON_BINARY64_BASE_THEN_CONDITIONAL_BINARY64_SUBTRACTION_THEN_"
    "ONE_TORCH_FLOAT32_SCALAR_EMISSION"
)
EXPECTED_VALIDATION_CONTEXT_COUNT = 340
EXPECTED_VALIDATION_SCENE_COUNT = 85
EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS = (
    EXPECTED_VALIDATION_CONTEXT_COUNT * EXACT_ACTION_COUNT_PER_SCENE
)
_SCALAR_ABSOLUTE_TOLERANCE = 2e-9


class Run2V2ValidationGateError(RuntimeError):
    """The one-shot v2 gate failed a frozen integrity invariant."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file_sha256(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise Run2V2ValidationGateError(
            f"{label} hash drift: {observed} != {expected}"
        )
    return observed


def _read_json(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise Run2V2ValidationGateError(f"{path} is not a JSON object")
    return document


def _require_runtime_reward_contract(
    *, deadline_penalty: float, arithmetic: str
) -> None:
    if (
        type(deadline_penalty) is not float
        or deadline_penalty != RUN2_V2_DEADLINE_PENALTY
    ):
        raise Run2V2ValidationGateError("frozen v2 lambda drift")
    if arithmetic != RUNTIME_ARITHMETIC:
        raise Run2V2ValidationGateError(
            "v2 arithmetic must be binary64 followed by one float32 emission"
        )


def _require_disjoint_scene_ids(
    train_ids: frozenset[str], validation_ids: frozenset[str]
) -> None:
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise Run2V2ValidationGateError(
            f"train/validation scene-ID overlap: {overlap[:3]}"
        )


@dataclass(frozen=True, slots=True)
class FrozenV2ValidationContract:
    """Hash-closed capability required before evaluator construction."""

    root: Path
    preregistration_path: Path
    train_summary_path: Path
    train_decision_path: Path
    preregistration: Mapping[str, Any]
    train_summary: Mapping[str, Any]

    def require_current(self) -> None:
        _require_file_sha256(
            self.preregistration_path,
            RUN2_V2_PREREGISTRATION_SHA256,
            "v2 preregistration",
        )
        _require_file_sha256(
            self.train_summary_path,
            RUN2_V2_TRAIN_SUMMARY_SHA256,
            "v2 train summary",
        )
        _require_file_sha256(
            self.train_decision_path,
            RUN2_V2_TRAIN_DECISION_SHA256,
            "v2 train decision",
        )
        _require_runtime_reward_contract(
            deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
            arithmetic=RUNTIME_ARITHMETIC,
        )


def _require_frozen_contracts(root: Path) -> FrozenV2ValidationContract:
    prereg_path = root / PREREGISTRATION_RELATIVE_PATH
    train_summary_path = root / TRAIN_SUMMARY_RELATIVE_PATH
    train_decision_path = root / TRAIN_DECISION_RELATIVE_PATH
    _require_file_sha256(
        prereg_path, RUN2_V2_PREREGISTRATION_SHA256, "v2 preregistration"
    )
    _require_file_sha256(
        train_summary_path, RUN2_V2_TRAIN_SUMMARY_SHA256, "v2 train summary"
    )
    _require_file_sha256(
        train_decision_path, RUN2_V2_TRAIN_DECISION_SHA256, "v2 train decision"
    )
    prereg = _read_json(prereg_path)
    train_summary = _read_json(train_summary_path)
    reward = prereg.get("reward")
    gate = prereg.get("pretraining_go_no_go")
    bindings = prereg.get("bindings")
    if (
        prereg.get("status")
        != "FROZEN_AFTER_TRAIN_ONLY_DERIVATION_BEFORE_ANY_V2_VALIDATION_ACCESS"
        or prereg.get("training_authorization")
        != "NOT_AUTHORIZED_BY_THIS_ARTIFACT_PENDING_V2_VALIDATION_GATE_AND_SEPARATE_REPLAY_BINDING_UPDATE"
        or not isinstance(reward, dict)
        or reward.get("deadline_ms") != 200.0
        or reward.get("deadline_penalty") != RUN2_V2_DEADLINE_PENALTY
        or reward.get("deadline_penalty_float_hex")
        != RUN2_V2_DEADLINE_PENALTY.hex()
        or reward.get("immediate_binary64_predecessor") != RUN2_V2_PREDECESSOR
        or reward.get("emitted_target_formula")
        != "torch.tensor(shaped64,dtype=torch.float32).item()"
    ):
        raise Run2V2ValidationGateError("v2 preregistered reward contract drift")
    expected_gate = {
        "timing": "ONLY_AFTER_THIS_V2_PREREGISTRATION_IS_FROZEN",
        "population": "FROZEN_DEVELOPMENT_FIT_VALIDATION_PANEL",
        "lambda_retuning": "FORBIDDEN",
        "every_fit_validation_context_has_at_least_one_p95_feasible_action": True,
        "frozen_lambda_shaped_oracle_must_equal_emitted_float32_p95_constrained_oracle": True,
        "frozen_lambda_shaped_oracle_p95_miss_count": 0,
        "actual_emitted_float32_targets_must_be_used": True,
        "failure_action": (
            "DO_NOT_RETUNE_ON_FIT_VALIDATION_REPORT_AS_PENALTY_"
            "GENERALIZATION_FAILURE_AND_CREATE_SEPARATE_RUN3_DESIGN"
        ),
    }
    if gate != expected_gate:
        raise Run2V2ValidationGateError("v2 preregistered GO/NO-GO gate drift")
    if (
        not isinstance(bindings, dict)
        or bindings.get("float32_exact_penalty_spec_sha256")
        != REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256
        or bindings.get("exact_penalty_v2_summary_sha256")
        != RUN2_V2_TRAIN_SUMMARY_SHA256
        or bindings.get("exact_penalty_v2_selection_decision_sha256")
        != RUN2_V2_TRAIN_DECISION_SHA256
        or bindings.get("exact_penalty_v2_implementation_sha256")
        != RUN2_V2_PENALTY_IMPLEMENTATION_SHA256
        or bindings.get("fit_partition_sha256")
        != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        or bindings.get("modeled_smoke_support_sha256")
        != MODELED_SMOKE_SUPPORT_SHA256
    ):
        raise Run2V2ValidationGateError("v2 preregistration source binding drift")
    decision = train_summary.get("decision")
    scope = train_summary.get("scope")
    old_collision = train_summary.get("old_float64_v1_collision")
    if (
        train_summary.get("schema")
        != "splitfusion.train_exact_p95_deadline_penalty.v2"
        or not isinstance(decision, dict)
        or decision.get("status") != "GO"
        or decision.get("deadline_penalty") != RUN2_V2_DEADLINE_PENALTY
        or decision.get("predecessor") != RUN2_V2_PREDECESSOR
        or not decision.get("predecessor_insufficient")
        or decision.get("predecessor_violation_count") != 1
        or decision.get("strict_ordering_violation_count") != 0
        or not isinstance(scope, dict)
        or scope.get("context_selection") != "TRAIN_IDS_ONLY"
        or scope.get("fit_validation_outcome_query_count") != 0
        or not isinstance(old_collision, dict)
        or not old_collision.get("emitted_float32_collision")
    ):
        raise Run2V2ValidationGateError("v2 train-only derivation binding drift")
    penalty_path = Path(__file__).with_name(
        "empirical_contextual_exact_p95_deadline_penalty_v2.py"
    )
    _require_file_sha256(
        penalty_path,
        RUN2_V2_PENALTY_IMPLEMENTATION_SHA256,
        "v2 penalty implementation",
    )
    contract = FrozenV2ValidationContract(
        root=root,
        preregistration_path=prereg_path,
        train_summary_path=train_summary_path,
        train_decision_path=train_decision_path,
        preregistration=prereg,
        train_summary=train_summary,
    )
    contract.require_current()
    return contract


@contextmanager
def _open_validation_evaluator(
    *, root: Path, frozen: object
) -> Iterator[FitValidationActorEvaluatorV1]:
    # This guard intentionally precedes evaluator construction.  Unit tests
    # verify that an invalid/missing capability cannot touch validation.
    if type(frozen) is not FrozenV2ValidationContract:
        raise Run2V2ValidationGateError(
            "validation access forbidden before v2 preregistration freeze-check"
        )
    frozen.require_current()
    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        yield evaluator


def _validation_scalar_v2(
    evaluator: FitValidationActorEvaluatorV1,
    entry: Any,
    mode_id: int,
    q_e4: int,
    *,
    penalty: float,
) -> Tuple[OracleOutcome, float, float, float]:
    raw = _authoritative_outcome(evaluator, entry, mode_id, q_e4)
    base64 = base_p95_expected_utility64_v2(
        p_admit=raw.p_edge_admission_given_sent,
        q_perc=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
    )
    shaped64 = shaped_p95_expected_utility64_v2(
        p_admit=raw.p_edge_admission_given_sent,
        q_perc=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
        deadline_penalty=penalty,
    )
    emitted = emitted_float32_target_v2(shaped64)
    return raw, base64, shaped64, emitted


def _assert_scalar_vector_v2(
    *,
    vector: OracleOutcome,
    scalar: OracleOutcome,
    scalar_emitted_reward: float,
    label: str,
) -> None:
    if (scalar.mode_id, scalar.q_e4) != (vector.mode_id, vector.q_e4):
        raise Run2V2ValidationGateError(f"{label} scalar/vector action mismatch")
    for name in (
        "q_perc",
        "total_transmitted_bytes",
        "datagram_count",
        "p_edge_admission_given_sent",
        "latency_proxy_p50_ms",
        "latency_proxy_p95_ms",
        "latency_proxy_p99_ms",
    ):
        if not math.isclose(
            float(getattr(scalar, name)),
            float(getattr(vector, name)),
            rel_tol=0.0,
            abs_tol=_SCALAR_ABSOLUTE_TOLERANCE,
        ):
            raise Run2V2ValidationGateError(
                f"{label} scalar/vector mismatch for {name}"
            )
    if scalar_emitted_reward != vector.reward:
        raise Run2V2ValidationGateError(
            f"{label} emitted-float32 scalar/vector reward mismatch"
        )


def _validation_gate_one_shot(
    *, root: Path, frozen: FrozenV2ValidationContract
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    evaluation_count = 0
    infeasible_comparison_count = 0
    strict_ordering_violation_count = 0
    feasible_context_count = 0
    identity_match_count = 0
    shaped_miss_count = 0
    unconstrained_scalar_count = 0
    constrained_scalar_count = 0
    shaped_scalar_count = 0
    zero_admission_action_count = 0
    cuda_before = torch.cuda.is_initialized()

    partition = load_registered_empirical_fit_partition(project_root=root)
    train_ids = frozenset(
        row.sample_id
        for row in partition.scene_assignments
        if row.split == TRAIN_SPLIT
    )
    registered_validation_ids = frozenset(
        row.sample_id
        for row in partition.scene_assignments
        if row.split == FIT_VALIDATION_SPLIT
    )
    if len(train_ids) != 391 or len(registered_validation_ids) != 85:
        raise Run2V2ValidationGateError("registered scene split count drift")
    _require_disjoint_scene_ids(train_ids, registered_validation_ids)

    with _open_validation_evaluator(root=root, frozen=frozen) as evaluator:
        panel = evaluator.panel
        if panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
            raise Run2V2ValidationGateError("fit-validation panel hash drift")
        panel_validation_ids = frozenset(
            row.scene_sample_id for row in panel.entries
        )
        if (
            len(panel.entries) != EXPECTED_VALIDATION_CONTEXT_COUNT
            or len(panel_validation_ids) != EXPECTED_VALIDATION_SCENE_COUNT
            or panel_validation_ids != registered_validation_ids
        ):
            raise Run2V2ValidationGateError(
                "validation panel is not exactly the registered 85x4 population"
            )
        _require_disjoint_scene_ids(train_ids, panel_validation_ids)

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
            unconstrained = _best_across_modes_v2(
                surfaces, networks, feasible_only=False
            )
            constrained = _best_across_modes_v2(
                surfaces, networks, feasible_only=True
            )
            shaped = _best_across_modes_v2(
                surfaces,
                networks,
                feasible_only=False,
                penalty=RUN2_V2_DEADLINE_PENALTY,
            )
            if unconstrained is None or shaped is None:
                raise Run2V2ValidationGateError(
                    f"validation context {entry.panel_index} has no p>0 action"
                )

            unconstrained_raw, unconstrained_base64, _unused, unconstrained_emitted = (
                _validation_scalar_v2(
                    evaluator,
                    entry,
                    unconstrained.mode_id,
                    unconstrained.q_e4,
                    penalty=0.0,
                )
            )
            _assert_scalar_vector_v2(
                vector=unconstrained,
                scalar=unconstrained_raw,
                scalar_emitted_reward=unconstrained_emitted,
                label="unconstrained validation winner",
            )
            unconstrained_scalar_count += 1

            shaped_raw, shaped_base64, shaped64, shaped_emitted = (
                _validation_scalar_v2(
                    evaluator,
                    entry,
                    shaped.mode_id,
                    shaped.q_e4,
                    penalty=RUN2_V2_DEADLINE_PENALTY,
                )
            )
            _assert_scalar_vector_v2(
                vector=shaped,
                scalar=shaped_raw,
                scalar_emitted_reward=shaped_emitted,
                label="shaped validation winner",
            )
            shaped_scalar_count += 1
            shaped_miss = shaped_raw.latency_proxy_p95_ms > 200.0
            shaped_miss_count += int(shaped_miss)

            if constrained is None:
                rows.append(
                    {
                        "panel_index": entry.panel_index,
                        "scene_rank": entry.scene_rank,
                        "sample_id": entry.scene_sample_id,
                        "network_profile": entry.network_profile,
                        "has_positive_admission_p95_feasible_action": False,
                        "unconstrained_mode_id": unconstrained.mode_id,
                        "unconstrained_q_e4": unconstrained.q_e4,
                        "unconstrained_q_perc": unconstrained_raw.q_perc,
                        "unconstrained_p_admit": unconstrained_raw.p_edge_admission_given_sent,
                        "unconstrained_latency_p95_ms": unconstrained_raw.latency_proxy_p95_ms,
                        "unconstrained_base64": unconstrained_base64,
                        "unconstrained_emitted_float32": unconstrained_emitted,
                        "constrained_mode_id": "",
                        "constrained_q_e4": "",
                        "constrained_q_perc": "",
                        "constrained_p_admit": "",
                        "constrained_latency_p95_ms": "",
                        "constrained_base64": "",
                        "constrained_emitted_float32": "",
                        "shaped_mode_id": shaped.mode_id,
                        "shaped_q_e4": shaped.q_e4,
                        "shaped_q_perc": shaped_raw.q_perc,
                        "shaped_p_admit": shaped_raw.p_edge_admission_given_sent,
                        "shaped_latency_p95_ms": shaped_raw.latency_proxy_p95_ms,
                        "shaped_base64": shaped_base64,
                        "shaped64": shaped64,
                        "shaped_emitted_float32": shaped_emitted,
                        "shaped_p95_miss": shaped_miss,
                        "shaped_matches_constrained": False,
                    }
                )
                continue

            feasible_context_count += 1
            constrained_raw, constrained_base64, _same64, constrained_emitted = (
                _validation_scalar_v2(
                    evaluator,
                    entry,
                    constrained.mode_id,
                    constrained.q_e4,
                    penalty=0.0,
                )
            )
            _assert_scalar_vector_v2(
                vector=constrained,
                scalar=constrained_raw,
                scalar_emitted_reward=constrained_emitted,
                label="constrained validation winner",
            )
            constrained_scalar_count += 1
            match = (shaped.mode_id, shaped.q_e4) == (
                constrained.mode_id,
                constrained.q_e4,
            )
            identity_match_count += int(match)

            for surface, network in zip(surfaces, networks):
                base64 = _base64_vector(
                    network["p_admit"], surface["quality"], network["p95"]
                )
                p_zero = network["p_admit"] == 0.0
                zero_admission_action_count += int(np.count_nonzero(p_zero))
                if np.any(p_zero):
                    if not np.all(base64[p_zero] == -1.0):
                        raise Run2V2ValidationGateError(
                            "p=0 base64 target is not exactly -1"
                        )
                    if not np.all(
                        _emitted_vector(base64[p_zero]) == np.float32(-1.0)
                    ):
                        raise Run2V2ValidationGateError(
                            "p=0 emitted target is not exactly -1"
                        )
                infeasible = (
                    (network["p_admit"] > 0.0) & (network["p95"] > 200.0)
                )
                if np.any(infeasible):
                    emitted = _emitted_vector(
                        _shaped64_vector(
                            base64,
                            network["p_admit"],
                            network["p95"],
                            RUN2_V2_DEADLINE_PENALTY,
                        )[infeasible]
                    )
                    infeasible_comparison_count += len(emitted)
                    strict_ordering_violation_count += int(
                        np.count_nonzero(
                            emitted >= np.float32(constrained.reward)
                        )
                    )

            rows.append(
                {
                    "panel_index": entry.panel_index,
                    "scene_rank": entry.scene_rank,
                    "sample_id": entry.scene_sample_id,
                    "network_profile": entry.network_profile,
                    "has_positive_admission_p95_feasible_action": True,
                    "unconstrained_mode_id": unconstrained.mode_id,
                    "unconstrained_q_e4": unconstrained.q_e4,
                    "unconstrained_q_perc": unconstrained_raw.q_perc,
                    "unconstrained_p_admit": unconstrained_raw.p_edge_admission_given_sent,
                    "unconstrained_latency_p95_ms": unconstrained_raw.latency_proxy_p95_ms,
                    "unconstrained_base64": unconstrained_base64,
                    "unconstrained_emitted_float32": unconstrained_emitted,
                    "constrained_mode_id": constrained.mode_id,
                    "constrained_q_e4": constrained.q_e4,
                    "constrained_q_perc": constrained_raw.q_perc,
                    "constrained_p_admit": constrained_raw.p_edge_admission_given_sent,
                    "constrained_latency_p95_ms": constrained_raw.latency_proxy_p95_ms,
                    "constrained_base64": constrained_base64,
                    "constrained_emitted_float32": constrained_emitted,
                    "shaped_mode_id": shaped.mode_id,
                    "shaped_q_e4": shaped.q_e4,
                    "shaped_q_perc": shaped_raw.q_perc,
                    "shaped_p_admit": shaped_raw.p_edge_admission_given_sent,
                    "shaped_latency_p95_ms": shaped_raw.latency_proxy_p95_ms,
                    "shaped_base64": shaped_base64,
                    "shaped64": shaped64,
                    "shaped_emitted_float32": shaped_emitted,
                    "shaped_p95_miss": shaped_miss,
                    "shaped_matches_constrained": match,
                }
            )

    if not cuda_before and torch.cuda.is_initialized():
        raise Run2V2ValidationGateError("CPU-only v2 validation initialized CUDA")
    if evaluation_count != EXPECTED_VALIDATION_ACTION_CONTEXT_EVALUATIONS:
        raise Run2V2ValidationGateError(
            "v2 validation exhaustive action-context count drift"
        )
    frozen.require_current()
    metrics = {
        "context_count": len(rows),
        "validation_scene_count": len(registered_validation_ids),
        "exhaustive_action_context_evaluations": evaluation_count,
        "positive_admission_feasible_context_count": feasible_context_count,
        "no_positive_admission_feasible_context_count": (
            len(rows) - feasible_context_count
        ),
        "shaped_constrained_identity_match_count": identity_match_count,
        "shaped_constrained_identity_mismatch_count": (
            len(rows) - identity_match_count
        ),
        "shaped_oracle_p95_miss_count": shaped_miss_count,
        "unconstrained_scalar_revalidation_count": unconstrained_scalar_count,
        "constrained_scalar_revalidation_count": constrained_scalar_count,
        "shaped_scalar_revalidation_count": shaped_scalar_count,
        "positive_admission_infeasible_comparison_count": (
            infeasible_comparison_count
        ),
        "strict_infeasible_ordering_violation_count": (
            strict_ordering_violation_count
        ),
        "zero_admission_action_context_count": zero_admission_action_count,
        "train_validation_scene_id_intersection_count": 0,
    }
    return tuple(rows), metrics


def _quality_admission_summary(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    unconstrained_quality = mean(float(row["unconstrained_q_perc"]) for row in rows)
    unconstrained_admission = mean(float(row["unconstrained_p_admit"]) for row in rows)
    constrained_rows = [
        row
        for row in rows
        if row["has_positive_admission_p95_feasible_action"]
    ]
    constrained_quality = (
        mean(float(row["constrained_q_perc"]) for row in constrained_rows)
        if constrained_rows
        else None
    )
    constrained_admission = (
        mean(float(row["constrained_p_admit"]) for row in constrained_rows)
        if constrained_rows
        else None
    )
    return {
        "unconstrained": {
            "mean_q_perc": unconstrained_quality,
            "mean_p_admit": unconstrained_admission,
            "p95_miss_count": sum(
                float(row["unconstrained_latency_p95_ms"]) > 200.0
                for row in rows
            ),
        },
        "constrained": {
            "population_count": len(constrained_rows),
            "mean_q_perc": constrained_quality,
            "mean_p_admit": constrained_admission,
            "p95_miss_count": sum(
                float(row["constrained_latency_p95_ms"]) > 200.0
                for row in constrained_rows
            ),
            "quality_retention_vs_unconstrained": (
                None
                if constrained_quality is None
                else constrained_quality / unconstrained_quality
            ),
            "admission_change_vs_unconstrained": (
                None
                if constrained_admission is None
                else constrained_admission - unconstrained_admission
            ),
        },
        "shaped": {
            "mean_q_perc": mean(float(row["shaped_q_perc"]) for row in rows),
            "mean_p_admit": mean(float(row["shaped_p_admit"]) for row in rows),
            "p95_miss_count": sum(bool(row["shaped_p95_miss"]) for row in rows),
        },
    }


def _profile_rows(
    rows: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], ...]:
    result: list[Dict[str, Any]] = []
    for profile in (*NETWORK_PROFILE_ORDER, "ALL_PROFILES"):
        selected = (
            list(rows)
            if profile == "ALL_PROFILES"
            else [row for row in rows if row["network_profile"] == profile]
        )
        summary = _quality_admission_summary(selected)
        result.append(
            {
                "network_profile": profile,
                "context_count": len(selected),
                "positive_admission_feasible_context_count": sum(
                    bool(row["has_positive_admission_p95_feasible_action"])
                    for row in selected
                ),
                "shaped_identity_match_count": sum(
                    bool(row["shaped_matches_constrained"])
                    for row in selected
                ),
                "shaped_p95_miss_count": summary["shaped"]["p95_miss_count"],
                "unconstrained_p95_miss_count": summary["unconstrained"]["p95_miss_count"],
                "unconstrained_mean_q_perc": summary["unconstrained"]["mean_q_perc"],
                "unconstrained_mean_p_admit": summary["unconstrained"]["mean_p_admit"],
                "constrained_mean_q_perc": summary["constrained"]["mean_q_perc"],
                "constrained_mean_p_admit": summary["constrained"]["mean_p_admit"],
                "quality_retention_vs_unconstrained": summary["constrained"]["quality_retention_vs_unconstrained"],
                "admission_change_vs_unconstrained": summary["constrained"]["admission_change_vs_unconstrained"],
            }
        )
    return tuple(result)


def _decision_verbatim(
    metrics: Mapping[str, int], preregistration: Mapping[str, Any]
) -> Dict[str, Any]:
    gate = preregistration["pretraining_go_no_go"]
    criteria = {
        "every_fit_validation_context_has_at_least_one_p95_feasible_action": (
            metrics["positive_admission_feasible_context_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
        "frozen_lambda_shaped_oracle_must_equal_emitted_float32_p95_constrained_oracle": (
            metrics["shaped_constrained_identity_match_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
        "frozen_lambda_shaped_oracle_p95_miss_count": (
            metrics["shaped_oracle_p95_miss_count"]
            == gate["frozen_lambda_shaped_oracle_p95_miss_count"]
        ),
        "actual_emitted_float32_targets_must_be_used": (
            metrics["unconstrained_scalar_revalidation_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
            and metrics["constrained_scalar_revalidation_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
            and metrics["shaped_scalar_revalidation_count"]
            == EXPECTED_VALIDATION_CONTEXT_COUNT
        ),
    }
    return {
        "criteria": criteria,
        "failure_action": gate["failure_action"],
        "status": "GO" if all(criteria.values()) else "NO_GO",
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise Run2V2ValidationGateError("CSV requires at least one row")
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
    decision = summary["decision"]
    metrics = summary["validation_gate"]
    comparison = summary["quality_admission_vs_unconstrained"]
    quality_retention = comparison["constrained"][
        "quality_retention_vs_unconstrained"
    ]
    admission_change = comparison["constrained"][
        "admission_change_vs_unconstrained"
    ]
    quality_text = (
        "UNDEFINED_NO_FEASIBLE_CONTEXT"
        if quality_retention is None
        else f"{quality_retention:.6%}"
    )
    admission_text = (
        "UNDEFINED_NO_FEASIBLE_CONTEXT"
        if admission_change is None
        else f"{admission_change:+.9f}"
    )
    return "\n".join(
        [
            "# Emitted-float32 exact P95 Run-2 validation gate v2",
            "",
            f"- Decision: **{decision['status']}**",
            f"- Frozen lambda: `{RUN2_V2_DEADLINE_PENALTY!r}` (`{RUN2_V2_DEADLINE_PENALTY.hex()}`)",
            f"- Positive-admission feasible support: `{metrics['positive_admission_feasible_context_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Shaped/constrained identity matches: `{metrics['shaped_constrained_identity_match_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Shaped P95 misses: `{metrics['shaped_oracle_p95_miss_count']}/{EXPECTED_VALIDATION_CONTEXT_COUNT}`",
            f"- Exhaustive evaluations: `{metrics['exhaustive_action_context_evaluations']}`",
            f"- Strict infeasible comparisons/violations: `{metrics['positive_admission_infeasible_comparison_count']}/{metrics['strict_infeasible_ordering_violation_count']}`",
            f"- Quality retention vs unconstrained: `{quality_text}`",
            f"- Admission change vs unconstrained: `{admission_text}`",
            "",
            "The preregistration and train-only evidence were hash-verified before",
            "the validation evaluator was constructed. This was the one-shot frozen",
            "85-scene x 4-profile development validation; no lambda, arithmetic,",
            "action, or tie rule was tuned and no policy training was performed.",
            "V1 remains superseded history. This is modeled conditional-survivor",
            "P95 evidence, not a live service-level guarantee.",
            "",
        ]
    )


def run_exact_p95_run2_validation_gate_v2(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)

    # Frozen evidence checks must complete before the output campaign begins or
    # any validation identity/outcome is opened.
    frozen = _require_frozen_contracts(root)
    actions = enumerate_supported_actions()
    if len(actions) != EXACT_ACTION_COUNT_PER_SCENE:
        raise Run2V2ValidationGateError("executable action count drift")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)

    validation_rows, validation_metrics = _validation_gate_one_shot(
        root=root, frozen=frozen
    )
    quality_admission = _quality_admission_summary(validation_rows)
    profile_rows = _profile_rows(validation_rows)
    decision = _decision_verbatim(
        validation_metrics, frozen.preregistration
    )
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_ONE_SHOT_EXACT_P95_RUN2_VALIDATION_GATE_V2",
        "decision": decision,
        "reward": {
            "deadline_ms": 200.0,
            "deadline_penalty": RUN2_V2_DEADLINE_PENALTY,
            "deadline_penalty_float_hex": RUN2_V2_DEADLINE_PENALTY.hex(),
            "base64_formula": "p*(Q-0.25*(L95/200.0))+(1.0-p)*(-1.0)",
            "infeasible_shaped64_formula": "base64-p*lambda",
            "emitted_target": "torch.tensor(shaped64,dtype=torch.float32).item()",
            "arithmetic_contract": RUNTIME_ARITHMETIC,
        },
        "validation_gate": validation_metrics,
        "quality_admission_vs_unconstrained": quality_admission,
        "tie_rules": {
            "unconstrained_oracle": (
                "MAX_EMITTED_FLOAT32_BASE_REWARD_THEN_MIN_P95_THEN_"
                "MIN_MODE_ID_THEN_MIN_Q_E4_OVER_P_GT_0"
            ),
            "constrained_oracle": (
                "MAX_EMITTED_FLOAT32_BASE_REWARD_THEN_MIN_P95_THEN_"
                "MIN_MODE_ID_THEN_MIN_Q_E4_OVER_P_GT_0_AND_P95_LE_200"
            ),
            "shaped_oracle": (
                "MAX_EMITTED_FLOAT32_SHAPED_REWARD_THEN_MIN_P95_THEN_"
                "MIN_MODE_ID_THEN_MIN_Q_E4_OVER_P_GT_0"
            ),
        },
        "exactness": {
            "action_domain": "ALL_EXECUTABLE_INTEGER_MODE_Q_E4_PAIRS",
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "validation_context_count": EXPECTED_VALIDATION_CONTEXT_COUNT,
            "validation_scene_count": EXPECTED_VALIDATION_SCENE_COUNT,
            "profile_count": len(NETWORK_PROFILE_ORDER),
            "scalar_vector_metric_absolute_tolerance": _SCALAR_ABSOLUTE_TOLERANCE,
            "emitted_reward_comparison": "EXACT_FLOAT32_SCALAR_EQUALITY",
        },
        "bindings": {
            "run2_v2_preregistration_sha256": RUN2_V2_PREREGISTRATION_SHA256,
            "run2_v2_preregistration_observed_sha256": _sha256_file(
                frozen.preregistration_path
            ),
            "run2_v2_preregistration_relative_path": PREREGISTRATION_RELATIVE_PATH,
            "train_exact_penalty_v2_summary_sha256": RUN2_V2_TRAIN_SUMMARY_SHA256,
            "train_exact_penalty_v2_summary_observed_sha256": _sha256_file(
                frozen.train_summary_path
            ),
            "train_exact_penalty_v2_summary_relative_path": TRAIN_SUMMARY_RELATIVE_PATH,
            "train_exact_penalty_v2_decision_sha256": RUN2_V2_TRAIN_DECISION_SHA256,
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "float32_exact_penalty_spec_sha256": REGISTERED_FLOAT32_EXACT_PENALTY_SPEC_SHA256,
            "validation_implementation_sha256": _sha256_file(Path(__file__)),
            "validation_test_sha256": _sha256_file(
                Path(__file__).with_name(
                    "test_empirical_contextual_exact_p95_run2_validation_gate_v2.py"
                )
            ),
            "source_file_sha256": {
                name: _sha256_file(Path(__file__).with_name(name))
                for name in (
                    "empirical_contextual_environment.py",
                    "empirical_contextual_exact_p95_deadline_penalty_v2.py",
                    "empirical_contextual_fit_partition.py",
                    "empirical_contextual_fit_validation_evaluator.py",
                    "empirical_contextual_fit_validation_panel.py",
                    "empirical_contextual_split_oracle.py",
                    "modeled_smoke_support.py",
                )
            },
        },
        "scope": {
            "validation_role": "ONE_SHOT_FROZEN_DEVELOPMENT_FIT_VALIDATION",
            "queried_scene_population": "REGISTERED_FIT_VALIDATION_IDS_ONLY",
            "queried_validation_scene_count": EXPECTED_VALIDATION_SCENE_COUNT,
            "queried_train_scene_count": 0,
            "train_validation_scene_id_intersection_count": 0,
            "preregistration_verified_before_validation_access": True,
            "train_summary_verified_before_validation_access": True,
            "lambda_retuned_on_validation": False,
            "coefficient_change_count": 0,
            "validation_retry_count": 0,
            "policy_training_count": 0,
            "replay_change_count": 0,
            "v1_validation_evidence_status": "PRESERVED_SUPERSEDED_HISTORY",
            "claims_excluded": [
                "POLICY_TRAINING",
                "LIVE_200_MS_SLA",
                "UNCONDITIONAL_SERVICE_GUARANTEE",
                "FINAL_TEST_GENERALIZATION",
            ],
            "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
        },
    }
    payloads = {
        "validation_context_oracles_v2.csv": _csv_bytes(validation_rows),
        "profile_summary_v2.csv": _csv_bytes(profile_rows),
        "GO_NO_GO_v2.json": (
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    }
    payloads["REPORT_v2.md"] = _report(summary).encode("utf-8")
    summary["files"] = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    payloads["summary_v2.json"] = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    for name, payload in payloads.items():
        _atomic_bytes(destination / name, payload)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-shot frozen emitted-float32 v2 validation gate."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_exact_p95_run2_validation_gate_v2(
        output_dir=args.output, project_root=args.project_root
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["decision"]["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
