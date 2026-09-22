"""Train-only selection of a P95 deadline-hinge coefficient.

This is an additive, CPU-only pre-training screen.  It leaves the registered
D1/P50 utility and every completed checkpoint/artifact untouched.  The screen
crosses only the registered *training* scenes with the four registered network
profiles and enumerates every executable SPLIT action.  A coefficient qualifies
only if its exact reward oracle has zero 200-ms modeled-P95 misses, retains at
least 95 percent of the lambda-zero mean perception quality, and reduces mean
admission probability by no more than 0.001 absolute.  The smallest qualifying
coefficient is selected; if none qualifies the result is explicitly NO-GO.

This is not a live guarantee and it does not turn a conditional latency
percentile into a timeout probability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    FIXED_END_TO_FEEDBACK_STAGES_MS,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
    require_supported_action,
)
from .empirical_contextual_environment import EmpiricalOneStepEnvironmentV1
from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    EmpiricalFitPartitionV1,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_split_oracle import (
    CONDITIONAL_FEASIBILITY_SEMANTICS,
    EXACT_ACTION_COUNT_PER_SCENE,
    SCALAR_VECTOR_ABS_TOLERANCE,
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
    "HINGE_LAMBDA_GRID",
    "P95_HINGE_SPEC_SHA256",
    "MEAN_ADMISSION_MAX_ABSOLUTE_DROP",
    "MEAN_QUALITY_MINIMUM_RETENTION",
    "P95DeadlineHingeSpecV1",
    "select_smallest_eligible_lambda",
    "run_train_p95_hinge_selection",
]


SCHEMA = "splitfusion.train_p95_deadline_hinge_selection.v1"
SPEC_SCHEMA = "splitfusion.p95_deadline_hinge_suite.v1"
BUDGET_MS = 200.0
MEAN_QUALITY_MINIMUM_RETENTION = 0.95
MEAN_ADMISSION_MAX_ABSOLUTE_DROP = 0.001
HINGE_LAMBDA_GRID: Tuple[float, ...] = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
EXPECTED_TRAIN_SCENE_COUNT = 391
EXPECTED_TRAIN_CONTEXT_COUNT = EXPECTED_TRAIN_SCENE_COUNT * len(
    NETWORK_PROFILE_ORDER
)
EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS = (
    EXPECTED_TRAIN_CONTEXT_COUNT * EXACT_ACTION_COUNT_PER_SCENE
)
P95_HINGE_SPEC_SHA256 = (
    "12b471d4b20d6641bb10a8b903108265dba18f90717333870b4869555148d429"
)


@dataclass(frozen=True, slots=True)
class P95DeadlineHingeSpecV1:
    """One finite P95 excess-latency penalty hypothesis."""

    hinge_lambda: float
    deadline_ms: float = BUDGET_MS
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    service_non_admission_utility: float = -1.0

    def __post_init__(self) -> None:
        if type(self.hinge_lambda) is not float or self.hinge_lambda not in HINGE_LAMBDA_GRID:
            raise ValueError("hinge_lambda is outside the preregistered grid")
        expected = (BUDGET_MS, 1.0, 0.25, -1.0)
        observed = (
            self.deadline_ms,
            self.quality_weight,
            self.latency_weight,
            self.service_non_admission_utility,
        )
        if observed != expected or any(type(value) is not float for value in observed):
            raise ValueError("P95 hinge scalar hypothesis drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "deadline_ms": self.deadline_ms,
            "hinge_lambda": self.hinge_lambda,
            "hinge_term": "max(0,(latency_proxy_p95_ms-deadline_ms)/deadline_ms)",
            "latency_coordinate": (
                "FIXED_113_MS_PLUS_CONDITIONAL_RETAINED_SURVIVOR_P95"
            ),
            "latency_weight": self.latency_weight,
            "quality_component": "q_perc",
            "quality_weight": self.quality_weight,
            "service_non_admission_utility": self.service_non_admission_utility,
        }

    def expected_utility(
        self,
        *,
        p_edge_admission_given_sent: float,
        q_perc: float,
        latency_p95_ms: float,
    ) -> float:
        values = (
            p_edge_admission_given_sent,
            q_perc,
            latency_p95_ms,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("P95 hinge utility inputs must be finite scalars")
        if not 0.0 <= p_edge_admission_given_sent <= 1.0:
            raise ValueError("p_edge_admission_given_sent must lie in [0, 1]")
        if not 0.0 <= q_perc <= 1.0:
            raise ValueError("q_perc must lie in [0, 1]")
        if latency_p95_ms < 0.0:
            raise ValueError("latency_p95_ms must be non-negative")
        normalized_latency = latency_p95_ms / self.deadline_ms
        excess = max(0.0, normalized_latency - 1.0)
        admitted = (
            self.quality_weight * q_perc
            - self.latency_weight * normalized_latency
            - self.hinge_lambda * excess
        )
        return p_edge_admission_given_sent * admitted + (
            1.0 - p_edge_admission_given_sent
        ) * self.service_non_admission_utility


SPECS = tuple(P95DeadlineHingeSpecV1(value) for value in HINGE_LAMBDA_GRID)


@dataclass(frozen=True, slots=True)
class TrainSelectionContextV1:
    context_index: int
    scene_rank: int
    sample_id: str
    episode_id: str
    frame_id: int
    network_profile: str
    profile_rank: int


def _suite_document() -> Dict[str, Any]:
    return {
        "criterion": {
            "admission_rule": (
                "mean_p_admit >= lambda_zero_mean_p_admit - max_absolute_drop"
            ),
            "deadline_ms": BUDGET_MS,
            "deadline_rule": "train_p95_budget_miss_count == 0",
            "max_absolute_drop": MEAN_ADMISSION_MAX_ABSOLUTE_DROP,
            "minimum_quality_retention": MEAN_QUALITY_MINIMUM_RETENTION,
            "quality_rule": (
                "mean_q_perc >= minimum_retention * lambda_zero_mean_q_perc"
            ),
            "selection": (
                "SMALLEST_HINGE_LAMBDA_MEETING_CRITERION; NO_VALIDATION_ACCESS"
            ),
        },
        "d1_control_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "fixed_latency_stages_ms": [
            [name, value] for name, value in FIXED_END_TO_FEEDBACK_STAGES_MS
        ],
        "lambda_grid": list(HINGE_LAMBDA_GRID),
        "pre_registration_disclosure": (
            "AN_UNCOMMITTED_EXPLORATORY_PROBE_OF_THE_SAME_GRID_PRECEDED_THE_"
            "STRICT_ZERO_MISS_QUALITY_AND_ADMISSION_ACCEPTANCE_RULE"
        ),
        "probability_semantics": PILOT_UTILITY_SPEC.probability_semantics,
        "schema": SPEC_SCHEMA,
        "training_contexts": (
            "ALL_REGISTERED_TRAIN_SCENES_CROSSED_WITH_ALL_FOUR_NETWORK_PROFILES"
        ),
        "utility_specs": [spec.to_canonical_dict() for spec in SPECS],
    }


def _require_registered_suite() -> None:
    observed = canonical_sha256(_suite_document())
    if observed != P95_HINGE_SPEC_SHA256:
        raise OracleAuditError(f"P95 hinge utility-suite hash drift: {observed}")


def _training_contexts(
    partition: EmpiricalFitPartitionV1,
) -> Tuple[TrainSelectionContextV1, ...]:
    if type(partition) is not EmpiricalFitPartitionV1:
        raise OracleAuditError("training context builder requires exact fit partition")
    if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
        raise OracleAuditError("registered fit-partition hash drift")
    scenes = sorted(
        (row for row in partition.scene_assignments if row.split == TRAIN_SPLIT),
        key=lambda row: (row.sample_id, row.episode_id, row.frame_id),
    )
    if len(scenes) != EXPECTED_TRAIN_SCENE_COUNT:
        raise OracleAuditError("registered training scene count drift")
    contexts = tuple(
        TrainSelectionContextV1(
            context_index=scene_rank * len(NETWORK_PROFILE_ORDER) + profile_rank,
            scene_rank=scene_rank,
            sample_id=scene.sample_id,
            episode_id=scene.episode_id,
            frame_id=scene.frame_id,
            network_profile=profile,
            profile_rank=profile_rank,
        )
        for scene_rank, scene in enumerate(scenes)
        for profile_rank, profile in enumerate(NETWORK_PROFILE_ORDER)
    )
    if (
        len(contexts) != EXPECTED_TRAIN_CONTEXT_COUNT
        or tuple(row.context_index for row in contexts)
        != tuple(range(EXPECTED_TRAIN_CONTEXT_COUNT))
    ):
        raise OracleAuditError("training context cross-product drift")
    return contexts


def _reward_vector(
    spec: P95DeadlineHingeSpecV1,
    p_admit: np.ndarray,
    quality: np.ndarray,
    latency_p95: np.ndarray,
) -> np.ndarray:
    normalized_latency = latency_p95 / spec.deadline_ms
    excess = np.maximum(0.0, normalized_latency - 1.0)
    admitted = (
        spec.quality_weight * quality
        - spec.latency_weight * normalized_latency
        - spec.hinge_lambda * excess
    )
    result = p_admit * admitted + (
        1.0 - p_admit
    ) * spec.service_non_admission_utility
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("P95 hinge reward vector contains non-finite values")
    return result


def _authoritative_outcome(
    environment: EmpiricalOneStepEnvironmentV1,
    context: TrainSelectionContextV1,
    spec: P95DeadlineHingeSpecV1,
    mode_id: int,
    q_e4: int,
) -> OracleOutcome:
    require_supported_action(mode_id, q_e4)
    query = environment._surface.query_fit_q_e4(context.sample_id, mode_id, q_e4)
    component = query.policy.component(DIRECT_QUALITY_COMPONENT)
    if not component.valid or component.value is None:
        raise OracleAuditError("authoritative P95 hinge q_perc is undefined")
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
    reward = spec.expected_utility(
        p_edge_admission_given_sent=p_admit,
        q_perc=float(component.value),
        latency_p95_ms=latency_p95,
    )
    return OracleOutcome(
        mode_id=mode_id,
        q_e4=q_e4,
        q_perc=float(component.value),
        total_transmitted_bytes=payload,
        datagram_count=datagrams,
        p_edge_admission_given_sent=p_admit,
        latency_proxy_p50_ms=fixed + latency.p50_ms,
        latency_proxy_p95_ms=latency_p95,
        latency_proxy_p99_ms=fixed + latency.p99_ms,
        reward=reward,
    )


def _assert_same_outcome(vector: OracleOutcome, scalar: OracleOutcome) -> None:
    if (vector.mode_id, vector.q_e4, vector.datagram_count) != (
        scalar.mode_id,
        scalar.q_e4,
        scalar.datagram_count,
    ):
        raise OracleAuditError("P95 hinge scalar/vector identity mismatch")
    for name in (
        "q_perc",
        "total_transmitted_bytes",
        "p_edge_admission_given_sent",
        "latency_proxy_p50_ms",
        "latency_proxy_p95_ms",
        "latency_proxy_p99_ms",
        "reward",
    ):
        if not math.isclose(
            float(getattr(vector, name)),
            float(getattr(scalar, name)),
            rel_tol=0.0,
            abs_tol=SCALAR_VECTOR_ABS_TOLERANCE,
        ):
            raise OracleAuditError(f"P95 hinge scalar/vector mismatch for {name}")


def select_smallest_eligible_lambda(
    summaries: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Select only by preregistered train miss criterion and ascending lambda."""

    if len(summaries) != len(HINGE_LAMBDA_GRID):
        raise OracleAuditError("lambda-summary cardinality drift")
    by_lambda = {float(row["hinge_lambda"]): row for row in summaries}
    if tuple(sorted(by_lambda)) != HINGE_LAMBDA_GRID:
        raise OracleAuditError("lambda-summary grid drift")
    baseline = by_lambda[0.0]
    quality_floor = (
        MEAN_QUALITY_MINIMUM_RETENTION * float(baseline["mean_q_perc"])
    )
    admission_floor = (
        float(baseline["mean_p_edge_admission_given_sent"])
        - MEAN_ADMISSION_MAX_ABSOLUTE_DROP
    )
    for value in HINGE_LAMBDA_GRID:
        row = by_lambda[value]
        if (
            int(row["train_p95_budget_miss_count"]) == 0
            and float(row["mean_q_perc"]) >= quality_floor
            and float(row["mean_p_edge_admission_given_sent"])
            >= admission_floor
        ):
            return row
    return None


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise OracleAuditError("CSV output requires at least one row")
    from io import StringIO

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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _report_markdown(
    summaries: Sequence[Mapping[str, Any]], selected: Optional[Mapping[str, Any]]
) -> str:
    lines = [
        "# Train-only P95 deadline-hinge selection",
        "",
        "This CPU-only screen uses only the registered training partition. The",
        "frozen fit-validation panel, completed D1/P50 runs and checkpoints were",
        "not read for selection or modified.",
        "",
        "Disclosure: an uncommitted exploratory probe of this same lambda grid",
        "preceded the stricter zero-miss/quality/admission acceptance rule. The",
        "screen is therefore transparent train-only design evidence, not a claim",
        "that the final acceptance rule was registered before all inspection.",
        "",
        "| lambda | P95 misses | miss rate | mean quality | mean P95 | mean utility |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {hinge_lambda:g} | {train_p95_budget_miss_count}/{train_context_count} "
            "| {train_p95_budget_miss_rate:.3%} | {mean_q_perc:.4f} | "
            "{mean_latency_p95_ms:.1f} ms | {mean_reward:.6f} |".format(**row)
        )
    decision = (
        f"Selected coefficient: **lambda={float(selected['hinge_lambda']):g}**."
        if selected is not None
        else "Decision: **NO-GO**; no preregistered coefficient met every criterion."
    )
    lines.extend(
        [
            "",
            decision,
            "",
            "Selection rule: choose the smallest preregistered coefficient with",
            "zero modeled P95 deadline misses across all registered training",
            "scene/profile contexts, at least 95% of lambda-zero mean perception",
            "quality, and no more than 0.001 absolute loss in mean admission.",
            "A finite hinge is not claimed to provide a live hard guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def run_train_p95_hinge_selection(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Run exact train-only selection and materialize create-only evidence."""

    _require_registered_suite()
    enumerate_supported_actions()
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    cuda_before = torch.cuda.is_initialized()
    partition = load_registered_empirical_fit_partition(project_root=root)
    contexts = _training_contexts(partition)
    environment = EmpiricalOneStepEnvironmentV1.load_registered(
        seed=0, project_root=root
    )
    rows: list[Dict[str, Any]] = []
    action_context_evaluations = 0
    try:
        for context in contexts:
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
            action_context_evaluations += sum(len(surface["q"]) for surface in surfaces)
            for spec in SPECS:
                best: Optional[OracleOutcome] = None
                for mode_id, (surface, network) in enumerate(zip(surfaces, networks)):
                    rewards = _reward_vector(
                        spec,
                        network["p_admit"],
                        surface["quality"],
                        network["p95"],
                    )
                    index = _best_index(rewards, network["p95"])
                    if index is None:
                        raise OracleAuditError("P95 hinge oracle found no action")
                    candidate = _outcome_from_vectors(
                        mode_id, index, surface, network, rewards
                    )
                    if _better(
                        candidate,
                        best,
                        objective="reward",
                        tie_percentile="p95",
                    ):
                        best = candidate
                if best is None:
                    raise OracleAuditError("P95 hinge oracle has no winner")
                scalar = _authoritative_outcome(
                    environment, context, spec, best.mode_id, best.q_e4
                )
                _assert_same_outcome(best, scalar)
                rows.append(
                    {
                        "hinge_lambda": spec.hinge_lambda,
                        **asdict(context),
                        "mode_id": scalar.mode_id,
                        "q_e4": scalar.q_e4,
                        "q_perc": scalar.q_perc,
                        "total_transmitted_bytes": scalar.total_transmitted_bytes,
                        "datagram_count": scalar.datagram_count,
                        "p_edge_admission_given_sent": (
                            scalar.p_edge_admission_given_sent
                        ),
                        "latency_proxy_p50_ms": scalar.latency_proxy_p50_ms,
                        "latency_proxy_p95_ms": scalar.latency_proxy_p95_ms,
                        "latency_proxy_p99_ms": scalar.latency_proxy_p99_ms,
                        "p95_budget_miss": scalar.latency_proxy_p95_ms > BUDGET_MS,
                        "reward": scalar.reward,
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                    }
                )
    finally:
        environment.close()

    if not cuda_before and torch.cuda.is_initialized():
        raise OracleAuditError("CPU-only P95 hinge screen initialized CUDA")
    if action_context_evaluations != EXPECTED_EXACT_ACTION_CONTEXT_EVALUATIONS:
        raise OracleAuditError("exact action-context evaluation count drift")
    if len(rows) != EXPECTED_TRAIN_CONTEXT_COUNT * len(SPECS):
        raise OracleAuditError("P95 hinge per-context row-count drift")

    summaries: list[Dict[str, Any]] = []
    for spec in SPECS:
        selected_rows = [row for row in rows if row["hinge_lambda"] == spec.hinge_lambda]
        misses = sum(bool(row["p95_budget_miss"]) for row in selected_rows)
        summaries.append(
            {
                "hinge_lambda": spec.hinge_lambda,
                "train_context_count": len(selected_rows),
                "train_p95_budget_hit_count": len(selected_rows) - misses,
                "train_p95_budget_miss_count": misses,
                "train_p95_budget_miss_rate": misses / len(selected_rows),
                "mean_reward": mean(float(row["reward"]) for row in selected_rows),
                "mean_q_perc": mean(float(row["q_perc"]) for row in selected_rows),
                "mean_p_edge_admission_given_sent": mean(
                    float(row["p_edge_admission_given_sent"])
                    for row in selected_rows
                ),
                "mean_latency_p95_ms": mean(
                    float(row["latency_proxy_p95_ms"]) for row in selected_rows
                ),
            }
        )
    baseline = summaries[0]
    quality_floor = (
        MEAN_QUALITY_MINIMUM_RETENTION * float(baseline["mean_q_perc"])
    )
    admission_floor = (
        float(baseline["mean_p_edge_admission_given_sent"])
        - MEAN_ADMISSION_MAX_ABSOLUTE_DROP
    )
    for row in summaries:
        row["zero_p95_budget_misses"] = (
            int(row["train_p95_budget_miss_count"]) == 0
        )
        row["quality_retention_vs_lambda_zero"] = (
            float(row["mean_q_perc"]) / float(baseline["mean_q_perc"])
        )
        row["admission_absolute_change_vs_lambda_zero"] = (
            float(row["mean_p_edge_admission_given_sent"])
            - float(baseline["mean_p_edge_admission_given_sent"])
        )
        row["criterion_met"] = (
            bool(row["zero_p95_budget_misses"])
            and float(row["mean_q_perc"]) >= quality_floor
            and float(row["mean_p_edge_admission_given_sent"])
            >= admission_floor
        )
    selected_row = select_smallest_eligible_lambda(summaries)
    selected = None if selected_row is None else dict(selected_row)
    decision = {
        "decision": "NO_GO" if selected is None else "GO",
        "selected_hinge_lambda": (
            None if selected is None else float(selected["hinge_lambda"])
        ),
        "selection_rule": (
            "SMALLEST_PREREGISTERED_LAMBDA_WITH_ZERO_TRAIN_P95_MISSES_"
            "QUALITY_RETENTION_GE_0.95_ADMISSION_DROP_LE_0.001"
        ),
    }

    artifact_payloads = {
        "train_context_winners.csv": _csv_bytes(rows),
        "lambda_summary.csv": _csv_bytes(summaries),
        "selection_decision.json": (
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        "REPORT.md": _report_markdown(summaries, selected).encode("utf-8"),
    }
    for name, payload in artifact_payloads.items():
        _atomic_bytes(destination / name, payload)

    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "COMPLETE_TRAIN_ONLY_P95_HINGE_NO_GO"
            if selected is None
            else "COMPLETE_TRAIN_ONLY_P95_HINGE_GO"
        ),
        "scope": {
            "claims_excluded": [
                "TRAINING",
                "FIT_VALIDATION_PERFORMANCE",
                "LIVE_OR_ONLINE_PERFORMANCE",
                "TIMEOUT_PROBABILITY",
                "HARD_DEADLINE_GUARANTEE",
                "LOCAL_ACTION_NECESSITY",
            ],
            "original_d1_p50_code_and_artifacts": "UNCHANGED",
        },
        "bindings": {
            "d1_pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "p95_hinge_spec_sha256": P95_HINGE_SPEC_SHA256,
            "selection_implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "exactness": {
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "exact_action_context_evaluations": action_context_evaluations,
            "train_context_count": EXPECTED_TRAIN_CONTEXT_COUNT,
            "train_scene_count": EXPECTED_TRAIN_SCENE_COUNT,
            "utility_evaluations": action_context_evaluations * len(SPECS),
            "winner_validation": (
                "EVERY_LAMBDA_CONTEXT_WINNER_REEVALUATED_THROUGH_SCALAR_SURFACES"
            ),
        },
        "hypothesis_suite": _suite_document(),
        "selection_decision": decision,
        "selected": selected,
        "lambda_summary": summaries,
        "row_counts": {
            "lambda_summary": len(summaries),
            "train_context_winners": len(rows),
        },
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifact_payloads.items()
        },
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    _atomic_bytes(
        destination / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact train-only P95 deadline-hinge selection."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_train_p95_hinge_selection(output_dir=args.output)
    print(json.dumps(summary["selection_decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
