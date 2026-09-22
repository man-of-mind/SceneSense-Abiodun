"""Exact pre-training screen for smooth P50/P95/P99 latency utilities.

This module is an additive analysis.  It does not modify the frozen D1 P50
utility or any completed training artifact.  Instead, it registers two smooth
counterfactual hypotheses that replace only the conditional retained-survivor
latency coordinate used by the D1 linear latency cost.  A latency quantile is
never interpreted as a timeout probability.

Every executable ``(mode_id, q_e4)`` pair is enumerated on every context in
the frozen fit-validation panel.  Vectorized formulas accelerate the search,
but both the unconstrained reward winner and the best reward winner satisfying
its own ``L_percentile <= 200 ms`` constraint are re-evaluated through the
authoritative scalar quality and network surfaces.
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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .empirical_contextual_contract import (
    FIXED_END_TO_FEEDBACK_STAGES_MS,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
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
    SCALAR_VECTOR_ABS_TOLERANCE,
    OracleAuditError,
    OracleOutcome,
    _authoritative_outcome,
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
from .transaction_identity import canonical_sha256

__all__ = [
    "RISK_VARIANTS",
    "SMOOTH_RISK_SPEC_SHA256",
    "SmoothLatencyRiskSpecV1",
    "run_smooth_latency_risk_oracle",
]


SCHEMA = "splitfusion.smooth_latency_percentile_oracle.v1"
SPEC_SCHEMA = "splitfusion.smooth_latency_percentile_utility_suite.v1"
BUDGET_MS = 200.0
RISK_VARIANTS: Tuple[str, ...] = ("p50", "p95", "p99")
VARIANT_LABELS = {
    "p50": "D1_SMOOTH_P50_CONTROL",
    "p95": "COUNTERFACTUAL_SMOOTH_P95",
    "p99": "COUNTERFACTUAL_SMOOTH_P99",
}
SMOOTH_RISK_SPEC_SHA256 = (
    "4c57f2aa734a0dded9915a84c97a0bdfe9102477e6af3f58db9c5e9d7dcaafb3"
)


@dataclass(frozen=True, slots=True)
class SmoothLatencyRiskSpecV1:
    """A smooth tail-aware hypothesis with no deadline hinge or hard mask."""

    percentile: str
    label: str
    deadline_ms: float = BUDGET_MS
    quality_weight: float = 1.0
    latency_weight: float = 0.25
    service_non_admission_utility: float = -1.0

    def __post_init__(self) -> None:
        if self.percentile not in RISK_VARIANTS:
            raise ValueError("latency percentile must be p50, p95, or p99")
        if self.label != VARIANT_LABELS[self.percentile]:
            raise ValueError("smooth-risk variant label drift")
        expected = (BUDGET_MS, 1.0, 0.25, -1.0)
        observed = (
            self.deadline_ms,
            self.quality_weight,
            self.latency_weight,
            self.service_non_admission_utility,
        )
        if observed != expected or any(type(value) is not float for value in observed):
            raise ValueError("smooth-risk scalar hypothesis drift")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "deadline_ms": self.deadline_ms,
            "label": self.label,
            "latency_coordinate": (
                "FIXED_113_MS_PLUS_CONDITIONAL_RETAINED_SURVIVOR_"
                f"{self.percentile.upper()}"
            ),
            "latency_weight": self.latency_weight,
            "percentile": self.percentile,
            "quality_component": "q_perc",
            "quality_weight": self.quality_weight,
            "service_non_admission_utility": self.service_non_admission_utility,
        }

    def expected_utility(
        self, *, p_edge_admission_given_sent: float, q_perc: float, latency_ms: float
    ) -> float:
        values = (p_edge_admission_given_sent, q_perc, latency_ms)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("smooth-risk utility inputs must be finite scalars")
        if not 0.0 <= p_edge_admission_given_sent <= 1.0:
            raise ValueError("p_edge_admission_given_sent must lie in [0, 1]")
        if not 0.0 <= q_perc <= 1.0:
            raise ValueError("q_perc must lie in [0, 1]")
        if latency_ms < 0.0:
            raise ValueError("latency_ms must be non-negative")
        admitted = self.quality_weight * q_perc - self.latency_weight * (
            latency_ms / self.deadline_ms
        )
        return p_edge_admission_given_sent * admitted + (
            1.0 - p_edge_admission_given_sent
        ) * self.service_non_admission_utility


SPECS = tuple(
    SmoothLatencyRiskSpecV1(percentile=name, label=VARIANT_LABELS[name])
    for name in RISK_VARIANTS
)


def _suite_document() -> Dict[str, Any]:
    return {
        "deadline_semantics": (
            "REPORT_OWN_QUANTILE_LTE_200_MS; NO_HINGE_OR_ACTION_MASK"
        ),
        "d1_control_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
        "fixed_latency_stages_ms": [
            [name, value] for name, value in FIXED_END_TO_FEEDBACK_STAGES_MS
        ],
        "probability_semantics": PILOT_UTILITY_SPEC.probability_semantics,
        "schema": SPEC_SCHEMA,
        "timeout_probability_status": (
            "NOT_INFERRED_FROM_ANY_CONDITIONAL_LATENCY_QUANTILE"
        ),
        "variants": [spec.to_canonical_dict() for spec in SPECS],
    }


def _require_registered_suite() -> None:
    observed = canonical_sha256(_suite_document())
    if observed != SMOOTH_RISK_SPEC_SHA256:
        raise OracleAuditError(
            f"smooth-risk utility-suite hash drift: {observed}"
        )


def _reward_vector(
    spec: SmoothLatencyRiskSpecV1,
    p_admit: np.ndarray,
    quality: np.ndarray,
    latency: np.ndarray,
) -> np.ndarray:
    admitted = spec.quality_weight * quality - spec.latency_weight * (
        latency / spec.deadline_ms
    )
    result = p_admit * admitted + (
        1.0 - p_admit
    ) * spec.service_non_admission_utility
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("smooth-risk reward vector contains non-finite values")
    return result


def _scalar_variant_outcome(
    evaluator: FitValidationActorEvaluatorV1,
    entry: Any,
    spec: SmoothLatencyRiskSpecV1,
    mode_id: int,
    q_e4: int,
) -> OracleOutcome:
    base = _authoritative_outcome(evaluator, entry, mode_id, q_e4)
    reward = spec.expected_utility(
        p_edge_admission_given_sent=base.p_edge_admission_given_sent,
        q_perc=base.q_perc,
        latency_ms=base.latency(spec.percentile),
    )
    return OracleOutcome(
        mode_id=base.mode_id,
        q_e4=base.q_e4,
        q_perc=base.q_perc,
        total_transmitted_bytes=base.total_transmitted_bytes,
        datagram_count=base.datagram_count,
        p_edge_admission_given_sent=base.p_edge_admission_given_sent,
        latency_proxy_p50_ms=base.latency_proxy_p50_ms,
        latency_proxy_p95_ms=base.latency_proxy_p95_ms,
        latency_proxy_p99_ms=base.latency_proxy_p99_ms,
        reward=reward,
    )


def _assert_same_variant_outcome(
    vector: OracleOutcome, scalar: OracleOutcome
) -> None:
    if (vector.mode_id, vector.q_e4, vector.datagram_count) != (
        scalar.mode_id,
        scalar.q_e4,
        scalar.datagram_count,
    ):
        raise OracleAuditError("smooth-risk scalar/vector winner identity mismatch")
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
            raise OracleAuditError(
                f"smooth-risk scalar/vector winner mismatch for {name}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise OracleAuditError("CSV output requires at least one row")
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), lineterminator="\n"
    )
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


def _summary_rows(
    rows: Sequence[Mapping[str, Any]], profile_order: Sequence[str]
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for spec in SPECS:
        variant_rows = [row for row in rows if row["variant"] == spec.label]
        for profile in (*profile_order, "ALL_PROFILES"):
            selected = (
                variant_rows
                if profile == "ALL_PROFILES"
                else [row for row in variant_rows if row["network_profile"] == profile]
            )
            expected = 340 if profile == "ALL_PROFILES" else 85
            if len(selected) != expected:
                raise OracleAuditError("smooth-risk profile coverage drift")
            misses = sum(bool(row["own_quantile_budget_miss"]) for row in selected)
            constrained_differences = sum(
                not bool(row["same_action_as_constrained_oracle"])
                for row in selected
            )
            output.append(
                {
                    "variant": spec.label,
                    "latency_percentile": spec.percentile,
                    "network_profile": profile,
                    "context_count": len(selected),
                    "own_quantile_budget_hit_count": len(selected) - misses,
                    "own_quantile_budget_miss_count": misses,
                    "own_quantile_budget_miss_rate": misses / len(selected),
                    "mean_reward": mean(float(row["reward"]) for row in selected),
                    "mean_q_perc": mean(float(row["q_perc"]) for row in selected),
                    "mean_p_edge_admission_given_sent": mean(
                        float(row["p_edge_admission_given_sent"])
                        for row in selected
                    ),
                    "mean_own_quantile_latency_ms": mean(
                        float(row["own_quantile_latency_ms"]) for row in selected
                    ),
                    "constrained_oracle_different_action_count": (
                        constrained_differences
                    ),
                    "mean_reward_cost_to_obey_200ms": mean(
                        float(row["reward_cost_to_obey_200ms"])
                        for row in selected
                    ),
                    "mean_constrained_q_perc": mean(
                        float(row["constrained_q_perc"]) for row in selected
                    ),
                    "mean_constrained_reward": mean(
                        float(row["constrained_reward"]) for row in selected
                    ),
                    "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                }
            )
    return output


def _action_distribution_rows(
    rows: Sequence[Mapping[str, Any]], profile_order: Sequence[str]
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for spec in SPECS:
        variant_rows = [row for row in rows if row["variant"] == spec.label]
        for profile in (*profile_order, "ALL_PROFILES"):
            selected = (
                variant_rows
                if profile == "ALL_PROFILES"
                else [row for row in variant_rows if row["network_profile"] == profile]
            )
            counts: Dict[Tuple[int, int], int] = {}
            for row in selected:
                action = (int(row["mode_id"]), int(row["q_e4"]))
                counts[action] = counts.get(action, 0) + 1
            for (mode_id, q_e4), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            ):
                output.append(
                    {
                        "variant": spec.label,
                        "latency_percentile": spec.percentile,
                        "network_profile": profile,
                        "mode_id": mode_id,
                        "q_e4": q_e4,
                        "selection_count": count,
                        "selection_fraction": count / len(selected),
                    }
                )
    return output


def _report_markdown(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    overall = [row for row in summary_rows if row["network_profile"] == "ALL_PROFILES"]
    lines = [
        "# Smooth latency-percentile oracle screen",
        "",
        "This is a CPU-only, exact-support, pre-training counterfactual on the frozen",
        "fit-validation panel. It is not live evidence, a timeout-probability model,",
        "or a new D1 result.",
        "",
        "| utility | budget misses | mean quality | mean admission | mean latency | constrained reward cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {variant} | {miss}/{count} ({rate:.1%}) | {quality:.4f} | "
            "{admission:.4f} | {latency:.1f} ms | {cost:.6f} |".format(
                variant=row["variant"],
                miss=row["own_quantile_budget_miss_count"],
                count=row["context_count"],
                rate=row["own_quantile_budget_miss_rate"],
                quality=row["mean_q_perc"],
                admission=row["mean_p_edge_admission_given_sent"],
                latency=row["mean_own_quantile_latency_ms"],
                cost=row["mean_reward_cost_to_obey_200ms"],
            )
        )
    lines.extend(
        [
            "",
            "The smooth objective has no discontinuity at 200 ms. A miss means only",
            "that the exact reward maximizer preferred a quality/admission trade-off",
            "above the modeled conditional-quantile budget. The constrained comparator",
            "shows what the best same-reward action would be if that modeled budget were",
            "enforced. No failure or timeout probability is inferred.",
            "",
        ]
    )
    return "\n".join(lines)


def run_smooth_latency_risk_oracle(
    *, output_dir: Path, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute the exact screen and materialize create-only artifacts."""

    _require_registered_suite()
    enumerate_supported_actions()
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    cuda_before = torch.cuda.is_initialized()
    rows: list[Dict[str, Any]] = []
    action_evaluations = 0

    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        panel = evaluator.panel
        if panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
            raise OracleAuditError("frozen fit-validation panel identity drift")
        for entry in panel.entries:
            best: Dict[str, Optional[OracleOutcome]] = {
                spec.percentile: None for spec in SPECS
            }
            constrained: Dict[str, Optional[OracleOutcome]] = {
                spec.percentile: None for spec in SPECS
            }
            for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)):
                surface = _surface_mode_vector(
                    evaluator.environment._surface, entry.scene_sample_id, mode_id
                )
                network = _network_vector(
                    evaluator.environment._network,
                    entry.network_profile,
                    surface["payload"],
                    surface["datagrams"],
                )
                action_evaluations += len(surface["q"])
                for spec in SPECS:
                    latency = network[spec.percentile]
                    rewards = _reward_vector(
                        spec, network["p_admit"], surface["quality"], latency
                    )
                    index = _best_index(rewards, latency)
                    if index is None:
                        raise OracleAuditError("smooth-risk oracle found no action")
                    candidate = _outcome_from_vectors(
                        mode_id, index, surface, network, rewards
                    )
                    if _better(
                        candidate,
                        best[spec.percentile],
                        objective="reward",
                        tie_percentile=spec.percentile,
                    ):
                        best[spec.percentile] = candidate

                    constrained_index = _best_index(
                        rewards, latency, latency <= spec.deadline_ms
                    )
                    if constrained_index is not None:
                        constrained_candidate = _outcome_from_vectors(
                            mode_id,
                            constrained_index,
                            surface,
                            network,
                            rewards,
                        )
                        if _better(
                            constrained_candidate,
                            constrained[spec.percentile],
                            objective="reward",
                            tie_percentile=spec.percentile,
                        ):
                            constrained[spec.percentile] = constrained_candidate

            for spec in SPECS:
                vector = best[spec.percentile]
                feasible_vector = constrained[spec.percentile]
                if vector is None or feasible_vector is None:
                    raise OracleAuditError(
                        "SPLIT support has no own-quantile-feasible action at 200 ms"
                    )
                scalar = _scalar_variant_outcome(
                    evaluator, entry, spec, vector.mode_id, vector.q_e4
                )
                constrained_scalar = _scalar_variant_outcome(
                    evaluator,
                    entry,
                    spec,
                    feasible_vector.mode_id,
                    feasible_vector.q_e4,
                )
                _assert_same_variant_outcome(vector, scalar)
                _assert_same_variant_outcome(feasible_vector, constrained_scalar)
                if constrained_scalar.latency(spec.percentile) > spec.deadline_ms:
                    raise OracleAuditError("scalar constrained winner misses its budget")
                reward_cost = scalar.reward - constrained_scalar.reward
                if reward_cost < -SCALAR_VECTOR_ABS_TOLERANCE:
                    raise OracleAuditError("constrained oracle exceeds unconstrained reward")
                rows.append(
                    {
                        "variant": spec.label,
                        "latency_percentile": spec.percentile,
                        "panel_index": entry.panel_index,
                        "scene_rank": entry.scene_rank,
                        "sample_id": entry.scene_sample_id,
                        "network_profile": entry.network_profile,
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
                        "own_quantile_latency_ms": scalar.latency(spec.percentile),
                        "own_quantile_budget_miss": (
                            scalar.latency(spec.percentile) > spec.deadline_ms
                        ),
                        "reward": scalar.reward,
                        "constrained_mode_id": constrained_scalar.mode_id,
                        "constrained_q_e4": constrained_scalar.q_e4,
                        "constrained_q_perc": constrained_scalar.q_perc,
                        "constrained_p_edge_admission_given_sent": (
                            constrained_scalar.p_edge_admission_given_sent
                        ),
                        "constrained_own_quantile_latency_ms": (
                            constrained_scalar.latency(spec.percentile)
                        ),
                        "constrained_reward": constrained_scalar.reward,
                        "reward_cost_to_obey_200ms": max(0.0, reward_cost),
                        "same_action_as_constrained_oracle": (
                            (scalar.mode_id, scalar.q_e4)
                            == (constrained_scalar.mode_id, constrained_scalar.q_e4)
                        ),
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                    }
                )

        expected_evaluations = (
            len(panel.entries) * EXACT_ACTION_COUNT_PER_SCENE
        )
        if action_evaluations != expected_evaluations:
            raise OracleAuditError("exact action-context evaluation count drift")
        profile_order = tuple(panel.profile_order)

    if not cuda_before and torch.cuda.is_initialized():
        raise OracleAuditError("CPU-only oracle initialized CUDA")
    if len(rows) != 340 * len(SPECS):
        raise OracleAuditError("smooth-risk context row-count drift")

    summary_rows = _summary_rows(rows, profile_order)
    distribution_rows = _action_distribution_rows(rows, profile_order)
    artifact_payloads = {
        "per_context.csv": _csv_bytes(rows),
        "profile_summary.csv": _csv_bytes(summary_rows),
        "action_distribution.csv": _csv_bytes(distribution_rows),
        "REPORT.md": _report_markdown(summary_rows).encode("utf-8"),
    }
    for name, payload in artifact_payloads.items():
        _atomic_bytes(destination / name, payload)

    overall = {
        row["variant"]: dict(row)
        for row in summary_rows
        if row["network_profile"] == "ALL_PROFILES"
    }
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_CPU_ONLY_PRETRAINING_ORACLE_SCREEN",
        "scope": {
            "claims_excluded": [
                "TRAINING",
                "LIVE_OR_ONLINE_PERFORMANCE",
                "TIMEOUT_PROBABILITY",
                "HARD_DEADLINE_GUARANTEE",
                "LOCAL_ACTION_NECESSITY",
            ],
            "reward_change": (
                "P50_CONTROL_IS_D1_EXACT; P95_P99_REPLACE_ONLY_THE_SMOOTH_"
                "LATENCY_COORDINATE"
            ),
        },
        "bindings": {
            "d1_pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "fit_validation_panel_sha256": (
                REGISTERED_FIT_VALIDATION_PANEL_SHA256
            ),
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "oracle_implementation_sha256": _sha256_file(Path(__file__)),
            "smooth_risk_spec_sha256": SMOOTH_RISK_SPEC_SHA256,
        },
        "exactness": {
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "exhaustive_action_context_evaluations": action_evaluations,
            "panel_contexts": 340,
            "scalar_vector_absolute_tolerance": SCALAR_VECTOR_ABS_TOLERANCE,
            "winner_validation": (
                "UNCONSTRAINED_AND_CONSTRAINED_WINNERS_REEVALUATED_THROUGH_"
                "AUTHORITATIVE_SCALAR_SURFACES"
            ),
        },
        "hypothesis_suite": _suite_document(),
        "overall_results": overall,
        "row_counts": {
            "action_distribution": len(distribution_rows),
            "per_context": len(rows),
            "profile_summary": len(summary_rows),
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
        description="Run the exact smooth P50/P95/P99 pre-training oracle screen."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_smooth_latency_risk_oracle(output_dir=args.output)
    print(json.dumps(summary["overall_results"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
